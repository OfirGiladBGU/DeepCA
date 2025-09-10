
import os
import numpy as np

def pad_to_cube_size(data, cube_size=128):
    shape = data.shape
    pad_width = []
    for dim in shape:
        remainder = dim % cube_size
        if remainder == 0:
            pad_width.append((0,0))
        else:
            pad = cube_size - remainder
            pad_width.append((0, pad))
    return np.pad(data, pad_width, mode='constant', constant_values=0)

def crop_and_save_cubes(src_folder, dst_folder, cube_size=128):
    os.makedirs(dst_folder, exist_ok=True)
    for fname in os.listdir(src_folder):
        if not fname.endswith('.npy'):
            continue
        fpath = os.path.join(src_folder, fname)
        data = np.load(fpath)
        # Pad data so each dimension is divisible by cube_size
        data = pad_to_cube_size(data, cube_size)
        shape = data.shape
        nx = shape[0] // cube_size
        ny = shape[1] // cube_size
        nz = shape[2] // cube_size
        count = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    cube = data[
                        ix*cube_size:(ix+1)*cube_size,
                        iy*cube_size:(iy+1)*cube_size,
                        iz*cube_size:(iz+1)*cube_size
                    ]
                    out_fname = f"{os.path.splitext(fname)[0]}_cube_{ix}_{iy}_{iz}.npy"
                    out_fpath = os.path.join(dst_folder, out_fname)
                    np.save(out_fpath, cube)
                    count += 1
        print(f"Processed {fname}: {count} cubes saved.")

if __name__ == "__main__":
    import shutil
    # Clean non-OG folders before cropping
    bp_folder = "/home/ofirgila/PycharmProjects/DeepCA/datasets/CCTA_BP"
    gt_folder = "/home/ofirgila/PycharmProjects/DeepCA/datasets/CCTA_GT"
    for folder in [bp_folder, gt_folder]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
    crop_and_save_cubes(
        "/home/ofirgila/PycharmProjects/DeepCA/datasets/CCTA_BP_OG",
        bp_folder
    )
    crop_and_save_cubes(
        "/home/ofirgila/PycharmProjects/DeepCA/datasets/CCTA_GT_OG",
        gt_folder
    )