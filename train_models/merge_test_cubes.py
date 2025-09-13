import numpy as np
import os
import re
# from train_models.samples_parameters import test_og
from samples_parameters import test_og

def get_cube_info(fname):
    # Example: PA000150_vessel_cube_0_0_0_output.npy
    m = re.match(r'(.*)_cube_(\d+)_(\d+)_(\d+)_output\.npy', fname)
    if m:
        base = m.group(1)
        ix, iy, iz = map(int, [m.group(2), m.group(3), m.group(4)])
        return base, ix, iy, iz
    return None

def merge_cubes_for_volume(base_name, cube_dir, output_dir, original_shape, cube_size=128):
    # Find all cubes for this base_name
    cube_files = [f for f in os.listdir(cube_dir) if f.startswith(base_name + '_cube_') and f.endswith('_output.npy')]
    # Determine padded shape
    shape = [((s + cube_size - 1) // cube_size) * cube_size for s in original_shape]
    merged = np.zeros(shape, dtype=np.uint8)
    for fname in cube_files:
        info = get_cube_info(fname)
        if info:
            _, ix, iy, iz = info
            cube = np.load(os.path.join(cube_dir, fname))
            # Check if the corresponding input cube is all zeros
            input_fname = fname.replace('_output.npy', '_input.npy')
            input_path = os.path.join(cube_dir, input_fname)
            if os.path.exists(input_path):
                input_cube = np.load(input_path)
                if np.all(input_cube == 0):
                    # Skip merging this output cube
                    continue
            merged[
                ix*cube_size:(ix+1)*cube_size,
                iy*cube_size:(iy+1)*cube_size,
                iz*cube_size:(iz+1)*cube_size
            ] = cube
    # Remove padding
    merged = merged[:original_shape[0], :original_shape[1], :original_shape[2]]
    out_path = os.path.join(output_dir, base_name + '_merged.npy')
    np.save(out_path, merged)
    print(f'Saved merged volume: {out_path}')
    return out_path

def main():
    cube_dir = 'DeepCA/output_test'  # predicted cubes
    output_dir = 'DeepCA/output_merged'  # where to save merged
    os.makedirs(output_dir, exist_ok=True)
    orig_dir = 'datasets/CCTA_BP_OG'
    for og_name in test_og:
        orig_file = og_name + '_vessel.npy'
        orig_path = os.path.join(orig_dir, orig_file)
        if not os.path.exists(orig_path):
            print(f'Original file not found: {orig_path}')
            continue
        orig = np.load(orig_path)
        shape = orig.shape
        merge_cubes_for_volume(og_name + '_vessel', cube_dir, output_dir, shape)

if __name__ == '__main__':
    main()
