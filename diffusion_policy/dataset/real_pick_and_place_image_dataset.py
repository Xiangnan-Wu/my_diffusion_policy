import numpy as np
import zarr
import os
import shutil

my_episode_ends = [99, 214, 320, 450]

array = np.array(my_episode_ends)

output_zarr_path = "my_replay_buffer.zarr"

if os.path.exists(output_zarr_path):
    shutil.rmtree(output_zarr_path)

root = zarr.open(output_zarr_path, mode='w')

meta_group = root.create_group('meta')
meta_group.create_dataset('episode_ends', data=array)
print(f"成功创建 Zarr 存储在 {output_zarr_path}")

verify_root = zarr.open(output_zarr_path, mode='r')

print(verify_root.tree())