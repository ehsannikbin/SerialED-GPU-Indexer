import h5py
import numpy as np
import re
import os
import threading
import queue
import torch.utils.data
import math


def preload_h5_to_ram(filename, start_idx, end_idx, geom_handler, batch_size, pixel_convention='corner'):
    """
    Optimized: Reads the entire [start_idx:end_idx] block in ONE disk I/O call.
    Slices into PyTorch batches entirely in memory to eliminate HDF5 overhead.
    """
    params = geom_handler.get_experiment_params()
    cx, cy, fs_vec, ss_vec, res = geom_handler.get_pixel_to_lab_transform('p0')
    
    ram_batches = []
    buffer = []
    # If input is already corner, add 0. If input is center, add 0.5 to find the distance from the corner.
    offset = 0.0 if pixel_convention == 'corner' else 0.5
    
    with h5py.File(filename, 'r') as f:
        peak_root = params['peak_path']
        
        # --- 1. ONE-SHOT CONTIGUOUS DISK READ ---
        n_batch = f[f"{peak_root}/nPeaks"][start_idx:end_idx]
        
        ds_x = f.get(f"{peak_root}/peakXPosRaw")
        x_batch = ds_x[start_idx:end_idx] if ds_x is not None else None
        
        ds_y = f.get(f"{peak_root}/peakYPosRaw")
        y_batch = ds_y[start_idx:end_idx] if ds_y is not None else None
        
        ds_I = f.get(f"{peak_root}/peakTotalIntensity")
        I_batch = ds_I[start_idx:end_idx] if ds_I is not None else None
        
        sx_path, sy_path = params['shift_x_path'], params['shift_y_path']
        sx_slice = f[sx_path][start_idx:end_idx] if sx_path in f else np.zeros(end_idx - start_idx)
        sy_slice = f[sy_path][start_idx:end_idx] if sy_path in f else np.zeros(end_idx - start_idx)
        
        d_set = f.get(params['data_path'])
        img_slice = None
        if d_set is not None:
            if d_set.ndim == 3: img_slice = d_set[start_idx:end_idx]
            elif d_set.ndim == 2: img_slice = [d_set[()]] * (end_idx - start_idx)

        # --- 2. IN-MEMORY BATCHING & PINNING ---
        for k in range(end_idx - start_idx):
            n = int(n_batch[k])
            if n < 3: continue 

            x_raw = x_batch[k, :n]
            y_raw = y_batch[k, :n]
            intensities = I_batch[k, :n] if I_batch is not None else np.zeros(n)

            x_m = ((x_raw + offset) * fs_vec[0] + (y_raw + offset) * ss_vec[0] + cx) / res
            y_m = ((x_raw + offset) * fs_vec[1] + (y_raw + offset) * ss_vec[1] + cy) / res

            sx_val = float(sx_slice[k]) if hasattr(sx_slice, 'ndim') and sx_slice.ndim > 0 else float(sx_slice)
            sy_val = float(sy_slice[k]) if hasattr(sy_slice, 'ndim') and sy_slice.ndim > 0 else float(sy_slice)

            event = {
                'filename': os.path.basename(filename),
                'event_id': start_idx + k,
                'peaks_m': torch.from_numpy(np.stack([x_m, y_m], axis=1)).float().pin_memory(),
                'peaks_raw': np.stack([x_raw, y_raw], axis=1), 
                'shift': torch.from_numpy(np.array([sx_val*1e-3, sy_val*1e-3])).float().pin_memory(),
                'intensities': torch.from_numpy(intensities).float().pin_memory(),
                'image': torch.from_numpy(img_slice[k]).float().pin_memory() if img_slice is not None else None
            }
            
            buffer.append(event)
            if len(buffer) >= batch_size:
                ram_batches.append(buffer)
                buffer = []

    if buffer:
        ram_batches.append(buffer)
        
    return ram_batches
# ==================================================================================
# 1. ORIGINAL GEOMETRY HANDLER
# ==================================================================================
class GeometryHandler:
    def __init__(self, geom_file):
        self.params = {'res': 1.0, 'clen': 0.0, 'data': '/data', 'peak_list': '/peaks'}
        self.panels = {}
        if os.path.exists(geom_file):
            self._parse(geom_file)
        else:
            print(f"Warning: Geometry file {geom_file} not found.")

    def _parse_value_with_unit(self, val_str):
        if not val_str: return 0.0
        val_str = str(val_str).strip()
        multipliers = {'kV': 1000.0, 'eV': 1.0, 'mm': 1e-3, 'm': 1.0}
        match = re.match(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([a-zA-Z]*)", val_str)
        if not match: return float(val_str)
        num, unit = float(match.group(1)), match.group(2)
        return num * multipliers.get(unit, 1.0)

    def _parse(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                line = line.split(';')[0].strip()
                if not line or '=' not in line: continue
                key, val = [x.strip() for x in line.split('=', 1)]
                if '/' in key and not key.startswith('/'): 
                    panel, prop = key.split('/', 1)
                    if panel not in self.panels: self.panels[panel] = {}
                    self.panels[panel][prop] = val
                else: 
                    self.params[key] = val

    def get_pixel_to_lab_transform(self, panel_name='p0'):
        p = self.panels.get(panel_name)
        if not p: return 0.0, 0.0, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 1.0
        
        cx = float(p.get('corner_x', 0))
        cy = float(p.get('corner_y', 0))
        res = float(self.params.get('res', 1.0))
        
        def parse_vec(s):
            x_val, y_val = 0.0, 0.0
            if not s: return np.array([1.0, 0.0])
            pattern = r"([+-]?(?:(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)?)\s*(x|y)"
            matches = re.findall(pattern, s.replace(" ", ""))
            if not matches: return np.array([1.0, 0.0])
            
            for num_str, axis in matches:
                val = 1.0 if num_str in ['', '+'] else -1.0 if num_str == '-' else float(num_str)
                if axis == 'x': x_val += val
                elif axis == 'y': y_val += val
            return np.array([x_val, y_val])
            
        fs_vec = parse_vec(p.get('fs','1x'))
        ss_vec = parse_vec(p.get('ss','1y'))
        
        return cx, cy, fs_vec, ss_vec, res

    def get_experiment_params(self):
        res = float(self.params.get('res', 1.0))
        eV = 0.0
        if 'electron_voltage' in self.params:
            eV = self._parse_value_with_unit(self.params['electron_voltage'])
        elif 'photon_energy' in self.params:
            eV = self._parse_value_with_unit(self.params['photon_energy'])
        
        sx_path = self.params.get('detector_shift_x', '').split()[0]
        sy_path = self.params.get('detector_shift_y', '').split()[0]
        
        cx, cy, fs_vec, ss_vec, _ = self.get_pixel_to_lab_transform('p0')
        p0 = self.panels.get('p0', {})
        
        return {
            'beam_energy_eV': eV,
            'detector_distance_m': self._parse_value_with_unit(self.params.get('clen', 0)),
            'pixel_size_m': 1.0 / res if res!=0 else 1.0,
            'res': res,
            'corner_x': cx,
            'corner_y': cy,
            'fs_vec': fs_vec,
            'ss_vec': ss_vec,
            'data_path': self.params.get('data', '/data'),
            'peak_path': self.params.get('peak_list', '/peaks'),
            'shift_x_path': sx_path,
            'shift_y_path': sy_path,
            'min_fs': float(p0.get('min_fs', 0)),
            'max_fs': float(p0.get('max_fs', 2047)),
            'min_ss': float(p0.get('min_ss', 0)),
            'max_ss': float(p0.get('max_ss', 2047))
        }

# ==================================================================================
# 2. ORIGINAL CELL HANDLER
# ==================================================================================
class CellHandler:
    def __init__(self, filename):
        self.cell = None
        self.centering = 'P'
        self.lattice_type = 'unknown'
        if os.path.exists(filename):
            self._parse(filename)
        else:
            print(f"Warning: Cell file {filename} not found.")

    def _parse(self, filename):
        params = {}
        with open(filename, 'r') as f:
            for line in f:
                line = line.split(';')[0].strip().split('#')[0].strip()
                if '=' in line:
                    k, v = [x.strip() for x in line.split('=', 1)]
                    try:
                        val_str = v.split()[0]
                        params[k] = float(val_str)
                    except:
                        params[k] = v 
        
        if 'lattice_type' in params: self.lattice_type = str(params['lattice_type'])
        if 'centering' in params: self.centering = str(params['centering'])
        
        if 'a' in params:
            self.cell = (params['a'], params.get('b', params['a']), params.get('c', params['a']),
                         params.get('al', 90), params.get('be', 90), params.get('ga', 90))

# ==================================================================================
# 3. BACKGROUND GENERATOR (Threading)
# ==================================================================================
class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        threading.Thread.__init__(self)
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item
    
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

# ==================================================================================
# 4. OPTIMIZED DATA LOADER (Batch Slicing)
# ==================================================================================
class DataLoader:
    def __init__(self, lst_file, base_dir="."):
        self.files = []
        if os.path.exists(lst_file):
            with open(lst_file, 'r') as f:
                for line in f:
                    path = line.strip()
                    if path: self.files.append(os.path.join(base_dir, path) if not os.path.isabs(path) else path)

    def batch_generator(self, file_list, geom_handler, batch_size=32, pixel_convention='corner'):
        """
        Efficiently reads H5 files in chunks corresponding to batch_size.
        Optimized for both sparse files (aggregates them) and large files (streams them).
        """
        # 1. Pre-fetch geometry params (Avoid calling this 1000s of times)
        params = geom_handler.get_experiment_params()
        transform = geom_handler.get_pixel_to_lab_transform('p0')
        cx, cy, fs_vec, ss_vec, res = transform

        # If input is already corner, add 0. If input is center, add 0.5 to find the distance from the corner.
        offset = 0.0 if pixel_convention == 'corner' else 0.5
        
        # Buffer for aggregating small files
        buffer = []

        for filename in file_list:
            if not os.path.exists(filename): continue
            
            try:
                # Use context manager to ensure file closes
                with h5py.File(filename, 'r') as f:
                    
                    # --- CHECK METADATA ---
                    peak_root = params['peak_path']
                    if peak_root not in f: continue
                    
                    # Try to get number of events quickly
                    if f"{peak_root}/nPeaks" in f:
                        ds_n = f[f"{peak_root}/nPeaks"]
                        n_events = ds_n.shape[0]
                    else:
                        continue
                        
                    if n_events == 0: continue

                    # --- SETUP HANDLES (Lazy Pointers) ---
                    # We do NOT read the data yet. We just get pointers.
                    ds_x = f.get(f"{peak_root}/peakXPosRaw")
                    ds_y = f.get(f"{peak_root}/peakYPosRaw")
                    ds_I = f.get(f"{peak_root}/peakTotalIntensity")
                    
                    # Shift handles
                    ds_sx, ds_sy = None, None
                    if params['shift_x_path'] in f: ds_sx = f[params['shift_x_path']]
                    elif '/center/shift_x_mm' in f: ds_sx = f['/center/shift_x_mm']
                    elif '/entry/data/shift_x_mm' in f: ds_sx = f['/entry/data/shift_x_mm']
                    
                    if params['shift_y_path'] in f: ds_sy = f[params['shift_y_path']]
                    elif '/center/shift_y_mm' in f: ds_sy = f['/center/shift_y_mm']
                    elif '/entry/data/shift_y_mm' in f: ds_sy = f['/entry/data/shift_y_mm']

                    # Image handle
                    d_set = None
                    if params['data_path'] in f: d_set = f[params['data_path']]

                    # --- CHUNK LOOP ---
                    # Read in chunks of 'batch_size' to match GPU cadence
                    # This hides the I/O cost for large files.
                    for i in range(0, n_events, batch_size):
                        end = min(i + batch_size, n_events)
                        
                        # 1. Sliced Read (Fast C-based slicing)
                        n_batch = ds_n[i:end]
                        x_batch = ds_x[i:end] if ds_x is not None else None
                        y_batch = ds_y[i:end] if ds_y is not None else None
                        I_batch = ds_I[i:end] if ds_I is not None else None
                        
                        # Handle Image Slice (Only read what we need)
                        img_slice = None
                        if d_set is not None:
                            if d_set.ndim == 3: img_slice = d_set[i:end]
                            elif d_set.ndim == 2: img_slice = [d_set[()]] * (end-i)

                        # Handle Shift Slice
                        sx_slice = ds_sx[i:end] if ds_sx is not None else np.zeros(end-i)
                        sy_slice = ds_sy[i:end] if ds_sy is not None else np.zeros(end-i)

                        # 2. Process Slice into Events
                        for k in range(end - i):
                            # Global index in file
                            idx_in_file = i + k
                            
                            n = int(n_batch[k])
                            if n < 3: continue # Skip empty early

                            if x_batch is not None:
                                # Ragged slicing from the block we just read
                                # Note: CrystFEL H5 peaks are usually fixed-width arrays padded with 0
                                # OR 1D arrays indexed by another dataset. 
                                # Assuming standard CrystFEL 2D array [n_events, max_peaks] based on user snippet
                                x_raw = x_batch[k, :n]
                                y_raw = y_batch[k, :n]
                                intensities = I_batch[k, :n] if I_batch is not None else np.zeros(n)
                            else:
                                continue

                            # Vectorized geometry (numpy is fast for small arrays)
                            x_m = ((x_raw + offset) * fs_vec[0] + (y_raw + offset) * ss_vec[0] + cx) / res
                            y_m = ((x_raw + offset) * fs_vec[1] + (y_raw + offset) * ss_vec[1] + cy) / res

                            # Shift
                            sx_val = float(sx_slice[k]) if hasattr(sx_slice, 'ndim') and sx_slice.ndim > 0 else float(sx_slice)
                            sy_val = float(sy_slice[k]) if hasattr(sy_slice, 'ndim') and sy_slice.ndim > 0 else float(sy_slice)

                            event = {
                                'filename': os.path.basename(filename),
                                'event_id': idx_in_file,
                                'peaks_m': np.stack([x_m, y_m], axis=1),
                                'peaks_raw': np.stack([x_raw, y_raw], axis=1),
                                'shift': np.array([sx_val*1e-3, sy_val*1e-3]),
                                'intensities': intensities,
                                'image': img_slice[k] if img_slice is not None else None
                            }
                            
                            buffer.append(event)
                            
                            # 3. Yield if Buffer Full
                            # This handles aggregation of small files automatically
                            if len(buffer) >= batch_size:
                                yield buffer
                                buffer = []

            except Exception as e:
                # print(f"Skipping {filename}: {e}")
                continue

        # Yield any remaining
        if buffer:
            yield buffer

    def get_prefetch_generator(self, file_list, geom_handler, batch_size=32):
        """
        Wraps the efficient batch generator in a background thread.
        This allows the next H5 slice to be read while GPU processes current batch.
        """
        gen = self.batch_generator(file_list, geom_handler, batch_size)
        return BackgroundGenerator(gen, max_prefetch=3)
    
class PinkIndexerDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_list, geom_handler, batch_size=32, pixel_convention='corner'):
        self.files = file_list
        self.geom_handler = geom_handler
        self.batch_size = batch_size
        self.pixel_convention = pixel_convention
        # We instantiate DataLoader just to access its batch_generator method
        # We pass a dummy file since we won't use the internal file list of this instance
        self.loader_helper = DataLoader("dummy_non_existent.lst") 

    def __iter__(self):
        # 1. Handle Multiprocessing Split
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process loading
            my_files = self.files
        else:
            # Split files among workers
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            my_files = self.files[iter_start:iter_end]

        # 2. Generate Batches
        # We reuse your optimized H5 slicing logic
        gen = self.loader_helper.batch_generator(my_files, self.geom_handler, self.batch_size, self.pixel_convention)
        
        for batch in gen:
            # 3. Convert to Tensors HERE (Crucial for pin_memory=True to work)
            for event in batch:
                event['peaks_m'] = torch.from_numpy(event['peaks_m']).float()
                event['intensities'] = torch.from_numpy(event['intensities']).float()
                event['shift'] = torch.from_numpy(event['shift']).float()
                # Handle image if present
                if event['image'] is not None:
                    event['image'] = torch.from_numpy(event['image']).float()
            
            yield batch

    