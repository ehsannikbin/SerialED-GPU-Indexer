import h5py
import numpy as np
import os
import sys

# ==============================================================================
# USER CONFIGURATION
# ==============================================================================
# Path to the text file containing the list of HDF5 files to process
LIST_FILE = "files3.lst" 

# Base directory where the files in the list are located (if paths are relative)
WORKING_DIR = "Z:/Ehsan/Ehsan/2025_03_08_HKUST1/Indexer_test/"

# Name of the output HDF5 file
OUTPUT_FILE = "master.h5"

# Compression settings (optional, set to None for faster speed but larger file)
COMPRESSION = None #"gzip" 
COMPRESSION_OPTS = None #4

# ==============================================================================
# REPACKER SCRIPT
# ==============================================================================

def get_file_paths(list_file, base_dir):
    """Reads the list file and returns full paths."""
    paths = []
    if not os.path.exists(list_file):
        print(f"Error: List file '{list_file}' not found.")
        sys.exit(1)
        
    with open(list_file, 'r') as f:
        for line in f:
            path = line.strip()
            if path:
                full_path = os.path.join(base_dir, path) if not os.path.isabs(path) else path
                paths.append(full_path)
    return paths

def scan_files(file_paths):
    """
    Pass 1: Scans files to determine total event count, max peaks, and image shape.
    """
    total_events = 0
    global_max_peaks = 0
    image_shape = None
    image_dtype = None
    
    print(f"Scanning {len(file_paths)} files to determine dimensions...")

    for fpath in file_paths:
        if not os.path.exists(fpath):
            print(f"Warning: Skipping missing file {fpath}")
            continue
            
        try:
            with h5py.File(fpath, 'r') as f:
                # Check if file has necessary keys
                if '/peaks/nPeaks' not in f:
                    continue
                
                n_peaks_dset = f['/peaks/nPeaks']
                n_events_in_file = n_peaks_dset.shape[0]
                
                if n_events_in_file == 0:
                    continue

                # 1. Accumulate total events
                total_events += n_events_in_file

                # 2. Find max peaks in this file
                # We look at the actual peak data shape or the max value in nPeaks
                # Usually peak arrays are [n_events, max_peaks_local]
                # We trust the dataset shape if available, otherwise max(nPeaks)
                if '/peaks/peakXPosRaw' in f:
                    local_max_peaks = f['/peaks/peakXPosRaw'].shape[1]
                else:
                    # Fallback if raw data isn't shaped normally
                    local_max_peaks = np.max(n_peaks_dset[()])
                
                if local_max_peaks > global_max_peaks:
                    global_max_peaks = local_max_peaks

                # 3. Get Image Shape (only need to do this once)
                if image_shape is None and '/data' in f:
                    data_dset = f['/data']
                    # data is usually [n_events, H, W] or [H, W] if single event
                    if data_dset.ndim == 3:
                        image_shape = data_dset.shape[1:] # (H, W)
                    elif data_dset.ndim == 2:
                        image_shape = data_dset.shape # (H, W)
                    image_dtype = data_dset.dtype

        except Exception as e:
            print(f"Error scanning {fpath}: {e}")

    return total_events, global_max_peaks, image_shape, image_dtype

def create_datasets(h5_out, total_events, max_peaks, img_shape, img_dtype):
    """Creates the datasets in the output file with the correct size."""
    
    # Create groups
    if '/center' not in h5_out: h5_out.create_group('/center')
    if '/peaks' not in h5_out: h5_out.create_group('/peaks')

    # 1. /data (Images)
    # Shape: (Total_Events, Height, Width)
    dshape = (total_events,) + img_shape
    h5_out.create_dataset('/data', shape=dshape, dtype=img_dtype, 
                          chunks=(1,) + img_shape, # Chunk by 1 image
                          compression=COMPRESSION, compression_opts=COMPRESSION_OPTS)

    # 2. /peaks/nPeaks
    h5_out.create_dataset('/peaks/nPeaks', shape=(total_events,), dtype='i4')

    # 3. Peak Arrays (Padded to max_peaks)
    # Shape: (Total_Events, Global_Max_Peaks)
    for name in ['peakTotalIntensity', 'peakXPosRaw', 'peakYPosRaw']:
        h5_out.create_dataset(f'/peaks/{name}', shape=(total_events, max_peaks), dtype='f4')

    # 4. Center/Shift Arrays
    # These are usually 1D scalars per event
    for name in ['center_x', 'center_y', 'shift_x_mm', 'shift_y_mm']:
        h5_out.create_dataset(f'/center/{name}', shape=(total_events,), dtype='f4')

def repack_files(file_paths, output_file, total_events, global_max_peaks, img_shape, img_dtype):
    """
    Pass 2: Reads data, pads it, and writes to output.
    """
    current_idx = 0
    
    print(f"Repacking into {output_file}...")
    print(f"Total Events: {total_events}, Global Max Peaks: {global_max_peaks}")

    with h5py.File(output_file, 'w') as f_out:
        # Initialize datasets
        create_datasets(f_out, total_events, global_max_peaks, img_shape, img_dtype)
        
        # Get Handles
        ds_data = f_out['/data']
        ds_nPeaks = f_out['/peaks/nPeaks']
        ds_I = f_out['/peaks/peakTotalIntensity']
        ds_X = f_out['/peaks/peakXPosRaw']
        ds_Y = f_out['/peaks/peakYPosRaw']
        
        ds_cx = f_out['/center/center_x']
        ds_cy = f_out['/center/center_y']
        ds_sx = f_out['/center/shift_x_mm']
        ds_sy = f_out['/center/shift_y_mm']

        # Iterate and Fill
        for i, fpath in enumerate(file_paths):
            if not os.path.exists(fpath): continue
            
            try:
                with h5py.File(fpath, 'r') as f_in:
                    if '/peaks/nPeaks' not in f_in: continue
                    
                    # Determine number of events in this file
                    n_local = f_in['/peaks/nPeaks'].shape[0]
                    if n_local == 0: continue

                    # Define the slice range in the OUTPUT file
                    start = current_idx
                    end = current_idx + n_local
                    
                    # --- COPY 1D DATA ---
                    ds_nPeaks[start:end] = f_in['/peaks/nPeaks'][:]
                    
                    # Handle centers (sometimes they are single scalars, sometimes arrays)
                    # We broadcast scalar to array if necessary
                    for key, ds_out in [('center_x', ds_cx), ('center_y', ds_cy), 
                                        ('shift_x_mm', ds_sx), ('shift_y_mm', ds_sy)]:
                        path = f'/center/{key}'
                        if path in f_in:
                            data = f_in[path][()] # read all
                            if np.ndim(data) == 0: # Scalar
                                ds_out[start:end] = np.full(n_local, data)
                            elif len(data) == n_local: # Array
                                ds_out[start:end] = data
                            else: # Mismatch (rare, but handle safely)
                                ds_out[start:end] = np.zeros(n_local)
                        else:
                            ds_out[start:end] = np.zeros(n_local)

                    # --- COPY IMAGE DATA ---
                    if '/data' in f_in:
                        # Direct copy if dimensions match, implies read-then-write
                        # For very large files, might want to chunk this loop, 
                        # but usually fits in RAM for typical batch sizes.
                        ds_data[start:end] = f_in['/data'][:]

                    # --- COPY AND PAD PEAK DATA ---
                    # Key requirement: Pad from local_max -> global_max
                    for key, ds_out in [('peakTotalIntensity', ds_I), 
                                        ('peakXPosRaw', ds_X), 
                                        ('peakYPosRaw', ds_Y)]:
                        path = f'/peaks/{key}'
                        if path in f_in:
                            local_data = f_in[path][:] # Shape (n_local, local_max)
                            local_width = local_data.shape[1]
                            
                            if local_width == global_max_peaks:
                                # Widths match, direct copy
                                ds_out[start:end, :] = local_data
                            elif local_width < global_max_peaks:
                                # Pad with zeros
                                padded = np.zeros((n_local, global_max_peaks), dtype='f4')
                                padded[:, :local_width] = local_data
                                ds_out[start:end, :] = padded
                            else:
                                # This shouldn't happen if scan_files worked, but truncate just in case
                                ds_out[start:end, :] = local_data[:, :global_max_peaks]
                        else:
                             ds_out[start:end, :] = np.zeros((n_local, global_max_peaks))

                    current_idx += n_local
                    
                    # Progress indicator
                    if i % 10 == 0:
                        print(f"Processed file {i+1}/{len(file_paths)}: {os.path.basename(fpath)} (+{n_local} events)")

            except Exception as e:
                print(f"Error processing {fpath}: {e}")

    print("Repacking complete.")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # FIX: Construct the full path to the list file
    # This tells Python that files3.lst is INSIDE the WORKING_DIR
    full_list_path = os.path.join(WORKING_DIR, LIST_FILE)
    
    # Get list of files
    # We pass 'full_list_path' to find the .lst file
    # We pass 'WORKING_DIR' to find the h5 files listed INSIDE the .lst file
    files = get_file_paths(full_list_path, WORKING_DIR)
    
    if not files:
        print("No files found to process.")
        sys.exit()

    # Pass 1: Scan
    t_events, max_p, img_shape, img_dtype = scan_files(files)
    
    if t_events == 0:
        print("No events found in files.")
        sys.exit()

    # Pass 2: Write
    # We also put the output file inside the WORKING_DIR so it doesn't get lost
    full_output_path = os.path.join(WORKING_DIR, OUTPUT_FILE)
    repack_files(files, full_output_path, t_events, max_p, img_shape, img_dtype)
