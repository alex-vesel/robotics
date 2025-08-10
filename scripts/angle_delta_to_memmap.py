import sys
import os
import numpy as np
sys.path.append(".")
from pathlib import Path

from shared_utils.path_utils import recurse_dir_for_clips
from behavior_cloning.configs.path_config import DATA_DIR


data_path = Path(DATA_DIR)
clip_paths = recurse_dir_for_clips(data_path, match='rgb_frames')

# convert all angle deltas to memmap
for clip_path in clip_paths:
    angle_delta_path = os.path.join(clip_path, 'angle_delta')
    angle_delta_memmap_path = os.path.join(clip_path, 'angle_delta_memmap')
    os.makedirs(angle_delta_memmap_path, exist_ok=True)

    for angle_delta_file in os.listdir(angle_delta_path):
        angle_delta = np.loadtxt(os.path.join(angle_delta_path, angle_delta_file), delimiter=',')
        angle_delta_memmap = np.memmap(os.path.join(angle_delta_memmap_path, angle_delta_file.replace(".csv", ".bin")), dtype=np.float32, mode='w+', shape=angle_delta.shape)
        angle_delta_memmap[:] = angle_delta
        del angle_delta_memmap
        del angle_delta
