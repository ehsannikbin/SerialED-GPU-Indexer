import torch
import torch.nn as nn
import torch.optim as optim
import math
import re
import numpy as np
import pinkindexer_cuda

class LatticeGenerator:
    @staticmethod
    def apply_centering_rules(indices, centering):
        h, k, l = indices[:, 0], indices[:, 1], indices[:, 2]
        if centering == 'P': return indices
        elif centering == 'F':
            mods = (indices % 2).sum(dim=1)
            mask = (mods == 0) | (mods == 3)
            return indices[mask]
        elif centering == 'I': 
            return indices[((h + k + l) % 2) == 0]
        elif centering == 'C': return indices[((h + k) % 2) == 0]
        elif centering == 'A': return indices[((k + l) % 2) == 0]
        elif centering == 'B': return indices[((h + l) % 2) == 0]
        return indices

    @staticmethod
    def generate_rlps(unit_cell, resolution_limit_A, centering, device):
        a, b, c, alpha, beta, gamma = unit_cell
        ar, br, gr = map(math.radians, [alpha, beta, gamma])
        
        vol = a*b*c*math.sqrt(1 - math.cos(ar)**2 - math.cos(br)**2 - math.cos(gr)**2 + 2*math.cos(ar)*math.cos(br)*math.cos(gr))
        
        a_star = b*c*math.sin(ar)/vol
        b_star = a*c*math.sin(br)/vol
        c_star = a*b*math.sin(gr)/vol
        
        cos_al_star = (math.cos(br)*math.cos(gr) - math.cos(ar)) / (math.sin(br)*math.sin(gr))
        cos_be_star = (math.cos(ar)*math.cos(gr) - math.cos(br)) / (math.sin(ar)*math.sin(gr))
        cos_ga_star = (math.cos(ar)*math.cos(br) - math.cos(gr)) / (math.sin(ar)*math.sin(br))
        
        B = torch.zeros((3,3), device=device)
        B[0,0] = a_star
        B[0,1] = b_star * cos_ga_star
        B[1,1] = b_star * math.sqrt(1 - cos_ga_star**2)
        B[0,2] = c_star * cos_be_star
        B[1,2] = c_star * (cos_al_star - cos_be_star*cos_ga_star) / math.sqrt(1 - cos_ga_star**2)
        B[2,2] = math.sqrt(c_star**2 - B[0,2]**2 - B[1,2]**2)

        min_dim = min(a,b,c)
        limit = resolution_limit_A if resolution_limit_A > 0 else 0.9
        max_h = int(math.ceil(1.5 / limit * min_dim))
        
        r = torch.arange(-max_h, max_h+1, device=device)
        h, k, l = torch.meshgrid(r, r, r, indexing='ij')
        indices = torch.stack([h.flatten(), k.flatten(), l.flatten()], dim=1).float()
        indices = indices[torch.norm(indices, dim=1) > 0] 
        
        indices = LatticeGenerator.apply_centering_rules(indices, centering)
        
        rlps = indices @ B.T
        norms = torch.norm(rlps, dim=1)
        mask = (norms <= (1.0/resolution_limit_A)) & (norms > 0.0)
        
        return indices[mask], rlps[mask], B

class PinkIndexer:
    def __init__(self, geom_params, unit_cell, centering='P', device='cuda', reflection_radius=0.002, max_res=0.9):
        self.device = device
        self.reflection_radius = reflection_radius
        self.max_res = max_res
        
        def parse_val(val):
            if isinstance(val, (int, float)): return float(val)
            s = str(val).strip()
            m = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", s)
            if not m: return 0.0
            num = float(m.group(1))
            if 'kV' in s or 'kv' in s: return num * 1000.0
            return num 

        voltage = None
        for k in ['electron_voltage', 'voltage', 'beam_energy_eV', 'beam_energy']:
            if k in geom_params:
                voltage = parse_val(geom_params[k])
                break
        
        if voltage is None:
            for k, v in geom_params.items():
                if 'energy' in k.lower() or 'volt' in k.lower():
                    voltage = parse_val(v)
                    break
        
        if voltage is None:
             raise KeyError(f"Could not find beam energy/voltage in geometry parameters.\n")

        # Relativistic Electron Wavelength
        self.lambd = 12.2643 / math.sqrt(voltage * (1.0 + 0.97845e-6 * voltage))
        self.k0 = 1.0 / self.lambd

        if 'clen' in geom_params:
            self.clen = parse_val(geom_params['clen'])
        elif 'detector_distance_m' in geom_params:
            self.clen = parse_val(geom_params['detector_distance_m'])
        elif 'camera_length' in geom_params:
             self.clen = parse_val(geom_params['camera_length'])
        else:
            self.clen = 0.0

        self.res = 1.0
        if 'res' in geom_params:
            self.res = parse_val(geom_params['res'])
        elif 'pixel_size_m' in geom_params:
            px_size = parse_val(geom_params['pixel_size_m'])
            if px_size > 0: self.res = 1.0 / px_size

        self.k_in = torch.tensor([0.0, 0.0, 1.0], device=device) * self.k0 
        
        self.indices_ref, self.rlps_ref, self.B_ref = LatticeGenerator.generate_rlps(unit_cell, max_res, centering, device)
        self.rlp_norms = torch.norm(self.rlps_ref, dim=1)

    def backproject(self, peaks_m, shift_m=None):
        if shift_m is None: shift_m = np.zeros(2)
        if peaks_m.dim() == 1: peaks_m = peaks_m.view(-1, 2)
        if peaks_m.shape[0] == 0:
            return torch.zeros((0,3), device=self.device), torch.zeros((0,3), device=self.device)

        shift_t = torch.as_tensor(shift_m, device=self.device, dtype=torch.float32)
        
        px = peaks_m[:, 0] + shift_t[0]
        py = peaks_m[:, 1] + shift_t[1]
        pz = torch.full_like(px, self.clen)
        
        xyz_lab = torch.stack([px, py, pz], dim=1)
        u_obs = torch.nn.functional.normalize(xyz_lab, dim=1)
        
        return xyz_lab, u_obs

    def compute_rotogram(self, peaks_m, shift_m=None, angle_resolution_level=2, max_rotogram_peaks=40, **kwargs):
        if shift_m is None: shift_m = np.zeros(2)
        if peaks_m.dim() == 1: peaks_m = peaks_m.view(-1, 2)
        
        levels = {0: 96, 1: 128, 2: 192, 3: 256, 4: 320} #
        self.rot_size = levels.get(angle_resolution_level, 128)
        self.rot_center = self.rot_size // 2

        shell_mult = kwargs.get('rotogram_shell_multiplier', 5.0)
        spin_steps = kwargs.get('rotogram_spin_steps', 180)
        
        if peaks_m.shape[0] > max_rotogram_peaks:   # This changes the number of peaks used to calculate the rotogram
            peaks_active = peaks_m[:max_rotogram_peaks]
        else:
            peaks_active = peaks_m

        if peaks_active.shape[0] == 0:
            return torch.zeros((self.rot_size, self.rot_size, self.rot_size), device=self.device)

        ATAN_PI_4 = 0.6657732
        self.rot_scale = (self.rot_size / 2.0) / ATAN_PI_4
        self.rot_offset = float(self.rot_center)

        # 1. Backprojection (Keep in Python, it's fast)
        _, u_obs = self.backproject(peaks_active, shift_m)
        q_vecs = self.k0 * u_obs - self.k_in.unsqueeze(0)
        q_norms = torch.norm(q_vecs, dim=1) 
        
        #length_tol = float(max(0.02, self.reflection_radius * shell_mult))
        length_tol = float(self.reflection_radius * shell_mult)
        
        # Ensure contiguous memory for C++
        q_vecs_c = q_vecs.contiguous()
        q_norms_c = q_norms.contiguous()
        rlps_ref_c = self.rlps_ref.contiguous()
        rlp_norms_c = self.rlp_norms.contiguous()

        # 2. Dispatch to CUDA kernel!
        rotogram = pinkindexer_cuda.compute_rotogram_cpp(
            q_vecs_c, 
            q_norms_c, 
            rlps_ref_c, 
            rlp_norms_c, 
            self.rot_size, 
            length_tol, 
            self.rot_scale, 
            self.rot_offset,
            spin_steps
        )

        return rotogram

    def get_rotation_from_rotogram(self, rotogram):
        idx = torch.argmax(rotogram)
        z = idx % self.rot_size
        y = (idx // self.rot_size) % self.rot_size
        x = idx // (self.rot_size**2)
        
        if x > 0 and x < self.rot_size-1 and y > 0 and y < self.rot_size-1 and z > 0 and z < self.rot_size-1:
            cube = rotogram[x-1:x+2, y-1:y+2, z-1:z+2]
            cube_sum = cube.sum()
            
            if cube_sum > 1e-6:
                offsets = torch.arange(-1, 2, device=self.device)
                w_x = cube.sum(dim=(1,2))
                delta_x = (w_x * offsets).sum() / cube_sum
                w_y = cube.sum(dim=(0,2))
                delta_y = (w_y * offsets).sum() / cube_sum
                w_z = cube.sum(dim=(0,1))
                delta_z = (w_z * offsets).sum() / cube_sum
                x = x.float() + delta_x
                y = y.float() + delta_y
                z = z.float() + delta_z

        v_map = (torch.stack([x,y,z]).float().to(self.device) - self.rot_offset) / self.rot_scale
        v_map_len = torch.norm(v_map)
        
        if v_map_len < 1e-8: return torch.eye(3, device=self.device)
        
        angle = 4.0 * torch.tan(v_map_len)
        axis = v_map / v_map_len
        
        K = torch.tensor([[0, -axis[2], axis[1]], 
                          [axis[2], 0, -axis[0]], 
                          [-axis[1], axis[0], 0]], device=self.device)
        
        R = torch.eye(3, device=self.device) + torch.sin(angle)*K + (1-torch.cos(angle))*(K@K)
        return R

    def refine(self, R_init, peaks_m, shift_m=None, B_init=None, **kwargs):
        """
        Continuous Hierarchical Refinement
        """
        if B_init is None: B_init = self.B_ref
        if shift_m is None: shift_m = np.zeros(2)
        
        if peaks_m.dim() == 1: peaks_m = peaks_m.view(-1, 2)
        if peaks_m.shape[0] == 0: return R_init, B_init, shift_m

        steps = kwargs.get('steps', 100)
        def_penalty = kwargs.get('deformation_penalty', 100.0)
        def_limit = kwargs.get('deformation_limit', 0.05) # NEW parameter
        
        start_rad = 0.08
        end_rad = 0.004
        
        R_curr = R_init.clone()
        shift_t = torch.as_tensor(shift_m, device=self.device, dtype=torch.float32)
        shift_curr = shift_t.clone().detach().requires_grad_(True)
        
        rot_delta = torch.zeros(3, device=self.device, requires_grad=True)
        B_delta = torch.zeros(3, 3, device=self.device, requires_grad=True)
        
        optimizer = optim.Adam([
            {'params': [rot_delta], 'lr': 0.005},
            {'params': [shift_curr], 'lr': 0.005},
            {'params': [B_delta],   'lr': 0.002} 
        ])
        
        max_shift = 4.0 / self.res 

        for i in range(steps):
            optimizer.zero_grad()
            progress = i / steps
            current_radius = start_rad * (1.0 - progress) + end_rad * progress
            
            if progress < 0.3:
                basis_weight = 0.0
            else:
                basis_weight = 1.0

            angle = torch.norm(rot_delta)
            if angle < 1e-9:
                R_delta_mat = torch.eye(3, device=self.device)
            else:
                axis = rot_delta / angle
                K = torch.tensor([[0, -axis[2], axis[1]], 
                                  [axis[2], 0, -axis[0]], 
                                  [-axis[1], axis[0], 0]], device=self.device)
                R_delta_mat = torch.eye(3, device=self.device) + torch.sin(angle)*K + (1-torch.cos(angle))*(K@K)
            
            R_total = R_delta_mat @ R_curr
            
            # Apply variable deformation limit
            I = torch.eye(3, device=self.device)
            effective_B_delta = B_delta * basis_weight * def_limit 
            B_new = B_init @ (I + effective_B_delta)
            
            _, u_obs = self.backproject(peaks_m, shift_curr)
            q_obs = self.k0 * u_obs - self.k_in.unsqueeze(0)
            h_obs = q_obs @ R_total 
            
            rlps_new = self.indices_ref @ B_new.T
            
            dists = torch.cdist(h_obs, rlps_new)
            min_dists, idxs = torch.min(dists, dim=1)
            
            mask = min_dists < current_radius
            if mask.sum() < 3:
                mask = min_dists < (current_radius * 2.0)
            
            if mask.sum() >= 3:
                h_matched = rlps_new[idxs[mask]]
                q_pred = h_matched @ R_total.T
                u_valid = u_obs[mask]
                
                dot = torch.sum(q_pred * u_valid, dim=1, keepdim=True)
                projection = dot * u_valid
                rejection = q_pred - projection
                err = torch.norm(rejection, dim=1)
                
                loss_fit = torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=0.01)
                loss_shift = torch.sum((shift_curr - shift_t)**2) * 10.0
                reg_strength = 1.0 / max(def_penalty, 1.0)
                loss_reg = torch.sum(B_delta**2) * reg_strength
                
                loss = loss_fit + loss_shift + loss_reg
                loss.backward()
                optimizer.step()
            
            with torch.no_grad():
                shift_curr.data.copy_(torch.clamp(shift_curr.data, min=shift_t - max_shift, max=shift_t + max_shift))
                B_delta.clamp_(-1.0, 1.0) 

        with torch.no_grad():
            angle = torch.norm(rot_delta)
            if angle > 1e-9:
                axis = rot_delta / angle
                K = torch.tensor([[0, -axis[2], axis[1]], 
                                  [axis[2], 0, -axis[0]], 
                                  [-axis[1], axis[0], 0]], device=self.device)
                R_final = R_delta_mat @ R_curr
            else:
                R_final = R_curr
            
            I = torch.eye(3, device=self.device)
            # Use configured limit for final B
            B_final = B_init @ (I + B_delta * def_limit)
                
        return R_final, B_final, shift_curr.detach().cpu().numpy()
    
    def refine_batch(self, R_init_batch, peaks_batch, shifts_batch, masks_batch, **kwargs):
        """
        Fully vectorized continuous hierarchical refinement.
        Processes an entire batch of patterns simultaneously via C++ Backend.
        """
        steps = kwargs.get('steps', 100)
        def_penalty = kwargs.get('deformation_penalty', 2000.0)
        def_limit = kwargs.get('deformation_limit', 0.05)
        max_shift_limit = kwargs.get('max_shift_limit', 4.0)
        shift_penalty_weight = kwargs.get('shift_penalty_weight', 10.0)
        expanded_radius_multiplier = kwargs.get('expanded_radius_multiplier', 2.0)

        lr_rot = kwargs.get('lr_rot', 0.005)
        lr_shift = kwargs.get('lr_shift', 0.005)
        lr_cell = kwargs.get('lr_cell', 0.002)
        huber_delta = kwargs.get('huber_delta', 0.01)
        radius_start = kwargs.get('radius_start', 0.08)
        radius_end = kwargs.get('radius_end', 0.004)

        # Ensure inputs are contiguous to avoid C++ memory alignment issues
        R_init_batch = R_init_batch.contiguous()
        peaks_batch = peaks_batch.contiguous()
        shifts_batch = shifts_batch.contiguous()
        masks_batch = masks_batch.contiguous()
        
        # Execute bare-metal C++ refinement loop
        R_fin, B_fin, S_fin = pinkindexer_cuda.refine_batch_cpp(
            R_init_batch, peaks_batch, shifts_batch, masks_batch, 
            self.B_ref.contiguous(), self.indices_ref.contiguous(), 
            self.k0, self.k_in.contiguous(), self.clen, self.res,
            steps, def_penalty, def_limit, max_shift_limit, shift_penalty_weight, expanded_radius_multiplier,
            lr_rot, lr_shift, lr_cell,        
            huber_delta, radius_start, radius_end
        )
        
        return R_fin, B_fin, S_fin
    

    def index_pattern(self, peaks_m, shift_m=None, **kwargs):
        if shift_m is None: shift_m = np.zeros(2)
        ar_level = kwargs.get('angle_resolution_level', 2)
        
        if peaks_m.dim() == 1: peaks_m = peaks_m.view(-1, 2)

        with torch.no_grad():
            rotogram = self.compute_rotogram(peaks_m, shift_m, angle_resolution_level=ar_level)
            R_init = self.get_rotation_from_rotogram(rotogram)
        
        R_refined, B_refined, shift_refined = self.refine(R_init, peaks_m, shift_m, **kwargs)
        return R_refined, B_refined, shift_refined
    
    def backproject_batched(self, peaks_batch, shifts_batch):
        """
        Fully vectorized backprojection.
        peaks_batch: (Batch, N, 2)
        shifts_batch: (Batch, 2)
        Returns: xyz_lab (Batch, N, 3), u_obs (Batch, N, 3)
        """
        # Broadcast shifts to match peaks dimension
        px = peaks_batch[:, :, 0] + shifts_batch[:, 0].unsqueeze(1)
        py = peaks_batch[:, :, 1] + shifts_batch[:, 1].unsqueeze(1)
        pz = torch.full_like(px, self.clen)
        
        xyz_lab = torch.stack([px, py, pz], dim=-1)
        u_obs = torch.nn.functional.normalize(xyz_lab, dim=-1)
        
        return xyz_lab, u_obs

    def get_initial_rotations_batched(self, peaks_batch, shifts_batch, masks_batch, angle_resolution_level=2, max_rotogram_peaks=40, **kwargs):
        """
        Generates the batched initial rotation matrices (Batch, 3, 3).
        Safely isolates the ragged rotogram logic while preparing dense tensors for refinement.
        """
        B = peaks_batch.shape[0]
        R_init_list = []

        for b in range(B):
            # Extract only the valid peaks for this specific pattern using the mask
            valid_mask = masks_batch[b]
            curr_peaks = peaks_batch[b][valid_mask]
            curr_shift = shifts_batch[b]

            # Generate rotogram using the original, mathematically proven logic
            rotogram = self.compute_rotogram(curr_peaks, curr_shift, angle_resolution_level, max_rotogram_peaks, **kwargs)
            R_init = self.get_rotation_from_rotogram(rotogram)
            R_init_list.append(R_init)

        # Stack into a single contiguous batch tensor
        return torch.stack(R_init_list, dim=0)
    
    def process_batch_debug(self, t_peaks_padded, t_masks, t_shifts, **kwargs):
        """
        STEP 3 DEBUG: Fully Vectorized Refinement
        """
        batch_size = t_peaks_padded.shape[0]
        ar_level = kwargs.get('angle_resolution_level', 2)
        rot_peaks = kwargs.get('max_rotogram_peaks', 40)
        
        # 1. Batched Initial Rotations
        R_init_batched = self.get_initial_rotations_batched(
            t_peaks_padded, t_shifts, t_masks, **kwargs
        )
        
        # 2. MASSIVE BATCHED REFINEMENT
        R_fin_batch, B_fin_batch, S_fin_batch = self.refine_batch(
            R_init_batched, t_peaks_padded, t_shifts, t_masks, **kwargs
        )
        
        # 3. Unpack tensors
        results = []
        for i in range(batch_size):
            results.append({
                'R_final': R_fin_batch[i],          
                'B_final': B_fin_batch[i],          
                'shift_final_tensor': S_fin_batch[i],     # <--- NEW: Keep tensor on GPU
                'shift_final': S_fin_batch[i].cpu().numpy() # Keep numpy for final writing
            })
            
        return results
    