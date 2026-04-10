import torch
import torch.multiprocessing as mp
import time
import numpy as np
import math
import os
import sys

from config import PinkIndexerConfig
from io_handler import GeometryHandler, CellHandler, DataLoader, PinkIndexerDataset
from io_handler import preload_h5_to_ram
from indexer import PinkIndexer
from stream_handler import StreamWriter
from integration import Integrator



def check_solution(fitted_indices, all_intensities, refined_B, input_B, 
                   peak_errors, tolerance, args):
    """
    Tiered rejection logic for indexing solutions.
    """
    if fitted_indices is None or fitted_indices.sum() < 3:
        return False

    # 1. Intensity Weighted Recall (IWR)
    # Rejects solutions that only fit noise peaks
    total_intensity = torch.sum(all_intensities)
    if total_intensity <= 1e-9: return False
    
    fitted_intensity = torch.sum(all_intensities[fitted_indices])
    iwr = (fitted_intensity / total_intensity).item()
    
    if iwr < args.rejection_iwr_threshold: 
        return False

    # 2. Cell Deformation Check
    # Ensures the unit cell hasn't been stretched beyond physical possibility
    norm_ref = torch.norm(input_B, dim=1)
    norm_refined = torch.norm(refined_B, dim=1)
    
    deviations = torch.abs((norm_refined / norm_ref) - 1.0)
    max_deformation = torch.max(deviations).item()
    
    if max_deformation > args.deformation_limit_percent:
        return False

    # 3. RMSD Check
    # Ensures the fit is tight, not just barely passing tolerance
    fitted_errors = peak_errors[fitted_indices]
    rmsd = torch.sqrt(torch.mean(fitted_errors**2)).item()
    
    if rmsd > (tolerance * args.rejection_rmsd_factor):
        return False

    return True

def check_solution_batch(fitted_mask_batch, intensities_batch, refined_B_batch, input_B_batch, 
                         min_dists_batch, masks_batch, args): # <--- Added masks_batch
    """
    Vectorized rejection logic. Returns boolean tensor (B).
    """
    # 0. STRICT MASKING: Ensure we never count padding (Ghost Peaks) as fitted
    # fitted_mask_batch might be True for ghosts if they are near (0,0,0), 
    # but masks_batch is False for ghosts.
    actual_fitted_mask = fitted_mask_batch & masks_batch

    # 1. Minimum Peak Count
    fitted_counts = actual_fitted_mask.sum(dim=1)
    mask_count = fitted_counts >= 3
    
    # 2. Intensity Weighted Recall (IWR)
    # Denominator: Sum of REAL peaks only (masks_batch ensures this, though padding intensity is 0 anyway)
    total_intensity = intensities_batch.sum(dim=1)
    total_intensity = total_intensity.clamp(min=1e-9)
    
    # Numerator: Sum of FITTED REAL peaks
    fitted_intensity = (intensities_batch * actual_fitted_mask.float()).sum(dim=1)
    iwr = fitted_intensity / total_intensity
    mask_iwr = iwr >= args.rejection_iwr_threshold
    
    # 3. Cell Deformation
    if input_B_batch.dim() == 2:
        norm_ref = torch.norm(input_B_batch, dim=1).unsqueeze(0) # (1, 3)
    else:
        norm_ref = torch.norm(input_B_batch, dim=2) # (B, 3)
        
    norm_refined = torch.norm(refined_B_batch, dim=2) # (B, 3)
    
    deviations = torch.abs((norm_refined / norm_ref) - 1.0)
    max_deformation = deviations.max(dim=1)[0]
    mask_def = max_deformation <= args.deformation_limit_percent
    
    # 4. RMSD
    # Only average error of FITTED REAL peaks
    sq_err = min_dists_batch ** 2
    masked_sq_err = sq_err * actual_fitted_mask.float()
    mean_sq_err = masked_sq_err.sum(dim=1) / fitted_counts.clamp(min=1)
    rmsd = torch.sqrt(mean_sq_err)
    
    mask_rmsd = rmsd <= (args.pinkIndexer_tolerance * args.rejection_rmsd_factor)
    
    return mask_count & mask_iwr & mask_def & mask_rmsd

def prepare_batch_for_gpu(batch, device, args):
    """Pads the CPU batch and transfers it to the GPU non-blocking."""
    valid_entries = []
    for idx, e in enumerate(batch):
        if len(e['peaks_m']) >= args.min_n_peaks:
            valid_entries.append((idx, e))

    if not valid_entries:
        return None

    valid_indices = [v[0] for v in valid_entries]
    valid_events = [v[1] for v in valid_entries]
    
    peaks_list_cpu = [e['peaks_m'] for e in valid_events]
    intens_list_cpu = [e['intensities'] for e in valid_events]
    shifts_list_cpu = [e['shift'] for e in valid_events]
    
    raw_max = max([p.shape[0] for p in peaks_list_cpu])
    limit = [30, 70, 127, 190, 255][min(max(args.pinkIndexer_considered_peaks_count, 0), 4)]
    max_peaks = min(raw_max, limit)
    
    final_peaks, final_intens, final_images = [], [], []
    for i, p in enumerate(peaks_list_cpu):
        intens = intens_list_cpu[i]
        if p.shape[0] > max_peaks:
            vals, sort_idx = torch.sort(intens, descending=True)
            sort_idx = sort_idx[:max_peaks]
            p = p[sort_idx]
            intens = intens[sort_idx]
        
        final_peaks.append(p)
        final_intens.append(intens)
        img = batch[valid_indices[i]]['image']
        final_images.append(torch.zeros(10, 10) if img is None else img)

    t_peaks_m = torch.nn.utils.rnn.pad_sequence(final_peaks, batch_first=True, padding_value=0)
    t_intensities = torch.nn.utils.rnn.pad_sequence(final_intens, batch_first=True, padding_value=0)
    t_shifts = torch.stack(shifts_list_cpu)
    t_images = torch.stack(final_images)
    
    lengths = torch.tensor([p.shape[0] for p in final_peaks])
    t_masks = torch.arange(t_peaks_m.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    
    # THE CRITICAL PART: Transfer to GPU asynchronously
    gpu_data = {
        'peaks': t_peaks_m.to(device, non_blocking=True),
        'intensities': t_intensities.to(device, non_blocking=True),
        'shifts': t_shifts.to(device, non_blocking=True),
        'images': t_images.to(device, non_blocking=True),
        'masks': t_masks.to(device, non_blocking=True),
        'indices': valid_indices,
        'events': valid_events
    }
    return gpu_data

def gpu_worker(rank, gpu_id, task_queue, args, cell_params, cell_centering, geom_params_dict, result_queue):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    indexer = PinkIndexer(
        geom_params_dict, cell_params, centering=cell_centering, device=device,
        reflection_radius=args.pinkIndexer_reflection_radius,
        max_res=args.pinkIndexer_max_resolution_for_indexing
    )
    integrator = Integrator(geom_params_dict, radii=tuple(args.int_radii), device=device, pixel_convention=args.pixel_convention)
    
    fs_v = geom_params_dict.get('fs_vec', np.array([1,0]))
    ss_v = geom_params_dict.get('ss_vec', np.array([0,1]))
    basis_M = np.array([[fs_v[0], ss_v[0]], [fs_v[1], ss_v[1]]])
    try: 
        inv_basis_M = np.linalg.inv(basis_M)
    except: 
        inv_basis_M = np.eye(2)
    
    inv_basis_tensor = torch.from_numpy(inv_basis_M).float().to(device)
    corner_x, corner_y, res = geom_params_dict['corner_x'], geom_params_dict['corner_y'], geom_params_dict['res']

    from io_handler import preload_h5_to_ram
    geom_handler = GeometryHandler(args.geometry_full_path)
    transfer_stream = torch.cuda.Stream(device=device)

    # --- DYNAMIC QUEUE CONSUMER ---
    def continuous_batch_generator(t_queue, geom, gpu_batch_size, pixel_convention):
        buffer = []
        while True:
            try:
                task = t_queue.get(timeout=10)
            except Exception:
                break
            
            if task is None: break # Poison pill
                
            h5_file, start_idx, end_idx = task
            chunk_size = end_idx - start_idx
            if chunk_size <= 0: continue
            
            ram_blocks = preload_h5_to_ram(h5_file, start_idx, end_idx, geom, gpu_batch_size, pixel_convention)
            
            for block in ram_blocks:
                buffer.extend(block)
                while len(buffer) >= gpu_batch_size:
                    yield buffer[:gpu_batch_size]
                    buffer = buffer[gpu_batch_size:]
                    
        if buffer: yield buffer

    batch_gen = continuous_batch_generator(task_queue, geom_handler, args.gpu_batch_size, args.pixel_convention)
    
    current_cpu_batch = next(batch_gen, None)
    next_gpu_data = None
    
    if current_cpu_batch is not None:
        with torch.cuda.stream(transfer_stream):
            next_gpu_data = prepare_batch_for_gpu(current_cpu_batch, device, args)

    while current_cpu_batch is not None:
        batch_start_time = time.time()
        
        torch.cuda.current_stream().wait_stream(transfer_stream)
        current_gpu_data = next_gpu_data
        
        next_cpu_batch = next(batch_gen, None)
        if next_cpu_batch is not None:
            with torch.cuda.stream(transfer_stream):
                next_gpu_data = prepare_batch_for_gpu(next_cpu_batch, device, args)
        
        batch_output_list = []
        
        if current_gpu_data is None:
            for e in current_cpu_batch:
                batch_output_list.append({
                    'accepted': False, 'filename': e['filename'], 'event': e['event_id'], 'reason': 'insufficient_peaks'
                })
        else:
            t_peaks_m, t_masks, t_shifts = current_gpu_data['peaks'], current_gpu_data['masks'], current_gpu_data['shifts']
            t_intensities, t_images = current_gpu_data['intensities'], current_gpu_data['images']
            valid_events = current_gpu_data['events']

            batch_results = indexer.process_batch_debug(
                t_peaks_m, t_masks, t_shifts, 
                angle_resolution_level=args.pinkIndexer_angle_resolution,
                steps=args.pinkIndexer_refinement_steps, deformation_limit=args.deformation_limit_percent,
                deformation_penalty=args.deformation_penalty, 
                max_rotogram_peaks=args.rotogram_peaks,
                lr_rot=args.lr_rot,  
                lr_shift=args.lr_shift,      
                lr_cell=args.lr_cell,        
                huber_delta=args.huber_delta,   
                radius_start=args.radius_start,       
                radius_end=args.radius_end, 
                max_shift_limit=args.max_shift_limit,
                shift_penalty=args.shift_penalty_weight,
                shell_multiplier=args.rotogram_shell_multiplier,
                expanded_radius_mult=args.expanded_radius_multiplier,
                rotogram_spin_steps=args.rotogram_spin_steps
            )
            
            R_final_stack = torch.stack([r['R_final'] for r in batch_results])
            B_final_stack = torch.stack([r['B_final'] for r in batch_results])

            S_final_stack = torch.stack([r['shift_final_tensor'] for r in batch_results])
            
            _, u_obs = indexer.backproject_batched(t_peaks_m, S_final_stack)
            # _, u_obs = indexer.backproject_batched(t_peaks_m, t_shifts) 
            q_obs = indexer.k0 * u_obs - indexer.k_in.unsqueeze(0).unsqueeze(0)
            q_obs_crys = torch.bmm(q_obs, R_final_stack)
            
            rlps_batch = torch.bmm(indexer.indices_ref.unsqueeze(0).expand(len(batch_results), -1, -1), B_final_stack.transpose(1, 2))
            
            dists = torch.cdist(q_obs_crys, rlps_batch)
            min_dists_batch, _ = torch.min(dists, dim=2)
            fitted_mask_batch = min_dists_batch < args.pinkIndexer_tolerance
            
            accepted_batch = check_solution_batch(fitted_mask_batch, t_intensities, B_final_stack, indexer.B_ref, min_dists_batch, t_masks, args)
            fit_counts_cpu = (fitted_mask_batch & t_masks).sum(dim=1).cpu().numpy()
            accepted_cpu = accepted_batch.cpu().numpy()
            
            for j, res_dict in enumerate(batch_results):
                e_meta = valid_events[j]
                if not accepted_cpu[j]:
                    batch_output_list.append({'accepted': False, 'filename': e_meta['filename'], 'event': e_meta['event_id'], 'reason': 'rejection'})
                    continue

                R_final, shift_tensor = R_final_stack[j], res_dict['shift_final_tensor']
                curr_rlps = rlps_batch[j]

                # --- MODIFIED: FRACTIONAL RESOLUTION LIMIT USING ALL PEAKS ---
                # 1. Fetch ALL peaks for this event and move to GPU
                all_peaks_m = e_meta['peaks_m'].to(device)
                
                # 2. Backproject all peaks into 3D reciprocal space using the refined shift
                _, u_obs_all = indexer.backproject(all_peaks_m, shift_tensor)
                q_obs_all = indexer.k0 * u_obs_all - indexer.k_in.unsqueeze(0)
                
                # 3. Rotate observed vectors into the crystal frame
                q_obs_crys_all = q_obs_all @ R_final
                
                # 4. Convert to fractional Miller indices using the inverse of the refined B matrix
                # Mathematical mapping: q = h @ B.T  =>  h_frac = q @ inv(B.T)
                B_inv_T = torch.inverse(B_final_stack[j].t())
                h_frac_all = q_obs_crys_all @ B_inv_T
                
                # 5. Calculate the fractional defect (distance to the nearest integer index)
                h_int_all = torch.round(h_frac_all)
                frac_defects = torch.abs(h_frac_all - h_int_all)
                
                # 6. Filter valid peaks: the defect in all 3 dimensions (h, k, l) must be < tolerance
                max_defects, _ = torch.max(frac_defects, dim=1)
                valid_matches_mask_all = max_defects < args.res_limit_tolerance
                
                if valid_matches_mask_all.sum() >= 3:
                    # 7. Calculate the resolution (1/d) of successfully matched peaks
                    matched_q_norms = torch.norm(q_obs_crys_all[valid_matches_mask_all], dim=1)
                    sorted_res, _ = torch.sort(matched_q_norms)
                    
                    # 8. Replicate CrystFEL's outlier rejection: n = n_acc / 50 (min 2)
                    n_acc = len(sorted_res)
                    n_drop = max(2, n_acc // 50)
                    limit_idx = max(0, n_acc - 1 - n_drop)
                    
                    diff_res_limit = sorted_res[limit_idx].item()
                else:
                    diff_res_limit = 0.0
                # -------------------------------------------------------------

                rlps_rot = curr_rlps @ R_final.t()
                k_out = indexer.k_in + rlps_rot

                # Mask 1: Ewald sphere intersection
                ewald_mask = torch.abs(torch.norm(k_out, dim=1) - indexer.k0) < (args.pinkIndexer_reflection_radius * 4.0)
                
                # Mask 2: Resolution limit (calculated limit + push_res)
                rlp_res = torch.norm(curr_rlps, dim=1)
                res_mask = rlp_res <= (diff_res_limit + args.push_res)
                
                # Combine masks
                pred_mask = ewald_mask & res_mask
                
                pred_hkl = indexer.indices_ref[pred_mask]
                u_pred = torch.nn.functional.normalize(k_out[pred_mask], dim=1)
                
                scale = geom_params_dict['detector_distance_m'] / (u_pred[:, 2] + 1e-9)
                target_x = ((u_pred[:, 0] * scale) - shift_tensor[0]) * res - corner_x
                target_y = ((u_pred[:, 1] * scale) - shift_tensor[1]) * res - corner_y
                
                pixels = inv_basis_tensor @ torch.stack([target_x, target_y], dim=0)
                # If outputting corner, subtract 0. If outputting center, subtract 0.5.
                offset = 0.0 if args.pixel_convention == 'corner' else 0.5
                pred_pix_tensor = torch.stack([pixels[0,:]-offset, pixels[1,:]-offset], dim=1)

                # --- FILTERING TO THE PANEL SIZE ---
                # 1. Determine exact panel limits
                if t_images[j].shape[0] > 10:
                    # If image is loaded, bounds are 0 to (Shape - 1)
                    H, W = t_images[j].shape
                    min_fs, max_fs = 0, W - 1
                    min_ss, max_ss = 0, H - 1
                else:
                    # Fallback to geometry boundaries
                    min_fs = geom_params_dict['min_fs']
                    max_fs = geom_params_dict['max_fs']
                    min_ss = geom_params_dict['min_ss']
                    max_ss = geom_params_dict['max_ss']

                # 2. Create the bounding box mask (Inclusive bounds using <= and >=)
                margin = args.int_radii[2]
                panel_mask = (pred_pix_tensor[:, 0] >= min_fs + margin) & (pred_pix_tensor[:, 0] <= max_fs - margin) & \
                             (pred_pix_tensor[:, 1] >= min_ss + margin) & (pred_pix_tensor[:, 1] <= max_ss - margin)

                # 3. Apply the mask
                pred_pix_tensor = pred_pix_tensor[panel_mask]
                pred_hkl = pred_hkl[panel_mask]
                # ----------------------------------
                
                integ_res = integrator.integrate(t_images[j], pred_pix_tensor)
                
                batch_output_list.append({
                    'accepted': True, 'filename': e_meta['filename'], 'event': e_meta['event_id'],
                    'input_peaks_m': t_peaks_m[j][t_masks[j]].cpu().numpy(), 'input_peaks_raw': e_meta['peaks_raw'], 
                    'input_intensities': e_meta['intensities'].cpu().numpy(),
                    'initial_shift_x': e_meta['shift'][0].item(), 'initial_shift_y': e_meta['shift'][1].item(),
                    'final_R': R_final.detach().cpu().numpy(), 'final_B': B_final_stack[j].detach().cpu().numpy(),
                    'refined_dx': res_dict['shift_final'][0], 
                    'refined_dy': res_dict['shift_final'][1],
                    'refl_hkl': pred_hkl.cpu().numpy(), 'refl_pred_pix': pred_pix_tensor.cpu().numpy(),
                    'refl_I': integ_res[0].cpu().numpy(), 'refl_sigma': integ_res[1].cpu().numpy(),
                    'refl_bg': integ_res[3].cpu().numpy(), 'refl_peak': integ_res[2].cpu().numpy(),
                    'beam_energy_eV': geom_params_dict['beam_energy_eV'], 'detector_distance_m': geom_params_dict['detector_distance_m'],
                    'pixel_size_m': geom_params_dict['pixel_size_m'], 'k0_Ainv': indexer.k0, 'peaks_fitted': fit_counts_cpu[j],
                    'diffraction_resolution_limit_Ainv': diff_res_limit
                })

        batch_output_list.sort(key=lambda x: x['event'])
        
        result_queue.put({
            'type': 'batch_result', 'data': batch_output_list,
            'compute_time': time.time() - batch_start_time, 'worker_rank': rank
        })
        current_cpu_batch = next_cpu_batch
            
    result_queue.put({'type': 'done'})
    



def main():
    conf = PinkIndexerConfig()
    args = conf.parse()
    
    wdir = args.wdir
    print(f"--- PinkIndexer (Batched + Stats) ---")
    
    geom_path = os.path.join(wdir, args.geometry)
    cell_path = os.path.join(wdir, args.cell)
    list_path = os.path.join(wdir, args.input)
    args.geometry_full_path = geom_path
    
    geom_handler = GeometryHandler(geom_path)
    cell_handler = CellHandler(cell_path)
    geom_params = geom_handler.get_experiment_params()
    
    loader = DataLoader(list_path, base_dir=wdir)
    print(f"Loaded {len(loader.files)} files.")
    
    writer = StreamWriter(os.path.join(wdir, args.output), args, geom_path, cell_path, cell_handler.lattice_type, cell_handler.centering)
    
    num_gpus = min(args.gpus, torch.cuda.device_count())
    if num_gpus == 0: 
        print("No GPU found. Exiting.")
        return

    workers_per_gpu = args.workers_per_gpu
    total_workers = num_gpus * workers_per_gpu
    print(f"Spawning {total_workers} workers ({workers_per_gpu} per GPU). Reporting every {args.gpu_batch_size} patterns.")
    
    # --- SMART CHUNK-BASED TASK DISTRIBUTION WITH CACHING ---
    #import os
    import json
    import h5py
    import math

    MAX_EVENTS_PER_CHUNK = args.chunk_size_events
    list_path = os.path.join(args.wdir, args.input)
    cache_file = list_path + ".cache.json"
    
    all_tasks = []
    total_events = 0
    use_cache = False
    
    # 1. Check Cache
    if os.path.exists(cache_file) and os.path.exists(list_path):
        if os.path.getmtime(cache_file) > os.path.getmtime(list_path):
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('chunk_size') == MAX_EVENTS_PER_CHUNK and cached_data.get('workers') == total_workers:
                        all_tasks = cached_data['tasks']
                        total_events = cached_data['total_events']
                        use_cache = True
                        print(f"Fast boot: Loaded task list from cache. ({total_events} events across {len(all_tasks)} chunks)")
            except Exception as e:
                pass

    # 2. Optimal Global Chunking
    if not use_cache:
        print("Inspecting HDF5 files to determine optimal distribution...")
        peak_path = geom_params.get('peak_path', '/peaks')
        file_info = []
        
        # Pass 1: Gather raw sizes
        for f_name in loader.files:
            try:
                with h5py.File(f_name, 'r') as f:
                    if peak_path in f and "nPeaks" in f[peak_path]:
                        n = f[f"{peak_path}/nPeaks"].shape[0]
                        file_info.append((f_name, n))
                        total_events += n
            except Exception as e:
                print(f"Warning: Could not read {f_name}: {e}")
                
        # Calculate the absolute optimal chunk size globally
        global_target_chunk = math.ceil(total_events / total_workers) if total_workers > 0 else 1
        actual_chunk_size = min(MAX_EVENTS_PER_CHUNK, max(1, global_target_chunk))

        # Pass 2: Slice files using the optimized size
        for f_name, n_events in file_info:
            for i in range(0, n_events, actual_chunk_size):
                end_idx = min(i + actual_chunk_size, n_events)
                all_tasks.append((f_name, i, end_idx))
                
        print(f"Total events found: {total_events}. Sliced into {len(all_tasks)} optimized chunks.")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump({'chunk_size': MAX_EVENTS_PER_CHUNK, 'workers': total_workers, 'total_events': total_events, 'tasks': all_tasks}, f)
        except Exception:
            pass

    if total_events == 0:
        print("No valid events found. Exiting.")
        return

    # 3. Dynamic Task Queue Setup
    
    mp.set_start_method('spawn', force=True)
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    
    for task in all_tasks: task_queue.put(task)
    for _ in range(total_workers): task_queue.put(None)
    
    
    processes = []
    global_start_time = time.time()
    
    for i in range(total_workers):
        target_gpu_index = (i // workers_per_gpu) % num_gpus
        p = mp.Process(target=gpu_worker, 
                       args=(i, target_gpu_index, task_queue, args, cell_handler.cell, cell_handler.centering, geom_params, result_queue))
        p.start()
        processes.append(p)
    
    active_workers = len(processes)
    idx_count = 0
    fail_count = 0
    
    # Timing variables for main loop
    last_batch_time = time.time()
    
    while active_workers > 0:
        msg = result_queue.get()
        if msg['type'] == 'done':
            active_workers -= 1
        elif msg['type'] == 'batch_result':
            batch_data = msg['data']
            n_batch = len(batch_data)
            
            if n_batch == 0: continue

            # Read the true compute time sent by the worker
            batch_duration = msg.get('compute_time', 0.0)
            worker_id = msg.get('worker_rank', '?')
            avg_time = batch_duration / n_batch if n_batch > 0 else 0.0
            
            # Calculate stats for the batch block
            n_indexed = sum(1 for r in batch_data if r['accepted'])
            n_rejected = n_batch - n_indexed
            
            # PRINT ACCURATE SUMMARY
            print(f"Worker {worker_id} | Batch ({n_batch}): {n_indexed} Indexed, {n_rejected} Rejected | Compute Time: {batch_duration:.3f}s ({avg_time:.4f}s/pat)")
            
            for res in batch_data:
                res['serial'] = idx_count + fail_count + 1
                
                # Write to stream (Buffered by OS usually, but grouped calls help)
                writer.write_chunk(res)
                
                if res['accepted']:
                    idx_count += 1
                   # print(f"  [Indexed] {res['filename']} // {res['event']} (Fits: {res.get('peaks_fitted', '?')})")
                else:
                    fail_count += 1
                   # print(f"  [Reject ] {res['filename']} // {res['event']} ({res.get('reason', 'unknown')})")

    for p in processes: p.join()
    
    total_duration = time.time() - global_start_time
    print(f"\n==================================================")
    print(f"Done. Indexed: {idx_count}, Failed/Rejected: {fail_count}")
    print(f"Total processing time: {total_duration:.2f} seconds")
    print(f"Overall Rate: {(idx_count+fail_count)/total_duration:.1f} Hz")
    print(f"==================================================")

if __name__ == "__main__":
    main()
