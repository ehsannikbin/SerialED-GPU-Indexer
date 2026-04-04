import h5py
import sys

def print_structure(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name} | Shape: {obj.shape} | Type: {obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"Group:   {name}")

if len(sys.argv) < 2:
    print("Usage: python inspect_h5.py your_file.h5")
else:
    filename = sys.argv[1]
    print(f"--- Structure of {filename} ---")
    try:
        with h5py.File(filename, 'r') as f:
            f.visititems(print_structure)
    except Exception as e:
        print(f"Error reading file: {e}")
        