import os
import cv2
import json
import hashlib
import multiprocessing
import numpy as np
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from shared_utils.path_utils import recurse_dir_for_clips
from camera.depth_camera import DepthFrame
from behavior_cloning.configs.path_config import TASK_DESCRIPTION_CACHE_PATH
from behavior_cloning.configs.nn_config import LANGUAGE_HIDDEN_DIM

class DataFile(Dataset):
    def __init__(self,
                 clip_path: str,
                 depth_frame_transform=None,
                 wrist_frame_transform=None,
                 angle_transform=None,
                 augmentations=None,
                 angle_augmentations=None,
                 chunk_size=1,
                ):

        self.clip_path = clip_path
        self.clip_name = os.path.join(*clip_path.split('/')[-2:])
        self.angle_path = os.path.join(clip_path, 'angles')
        self.angle_delta_path = os.path.join(clip_path, 'angle_delta')
        self.meta_path = os.path.join(clip_path, 'meta')
        self.depth_frames_path = os.path.join(clip_path, 'depth_frames')
        self.rgb_frames_path = os.path.join(clip_path, 'rgb_frames')
        self.wrist_frames_path = os.path.join(clip_path, 'wrist_frames')
        self.depth_frame_transform = depth_frame_transform
        self.wrist_frame_transform = wrist_frame_transform
        self.augmentations = augmentations
        self.angle_augmentations = angle_augmentations
        self.chunk_size = chunk_size

        self.idx_to_frame_name = OrderedDict()
        # path names are float, so we need to sort them as floats
        sorted_frames = sorted(os.listdir(self.angle_path), key=lambda x: float(".".join(x.split('.')[:-1])))
        for idx, frame in enumerate(sorted_frames):
            self.idx_to_frame_name[idx] = ".".join(frame.split('.')[:-1])
        

    def __len__(self):
        return max(len(os.listdir(self.angle_path)) - self.chunk_size, 0)

    def __getitem__(self, idx):
        # check if idx is within bounds
        if idx >= len(self):
            raise IndexError

        # import IPython; IPython.embed(); exit(0)
        cur_frame_name = self.idx_to_frame_name[idx]
        next_frame_name = self.idx_to_frame_name[idx + 1]

        # get frame and angle
        depth_frame = self.get_depth_frame(cur_frame_name)
        wrist_frame = self.get_wrist_frame(cur_frame_name)
        angle = self.get_angle(cur_frame_name)
        meta = self.get_meta(cur_frame_name)

        delta_angle = []
        for i in range(self.chunk_size):
            delta_angle.append(self.get_angle_delta(self.idx_to_frame_name[idx + i]))
        delta_angle = np.stack(delta_angle)

        if 'task_description' in meta and meta['task_description'] is not None:
            task_description = meta['task_description'][np.random.randint(len(meta['task_description']))]
            task_description_embedding = self.get_task_description_embedding(task_description)
        else:
            task_description_embedding = np.zeros(LANGUAGE_HIDDEN_DIM, dtype=np.float32)

        weight = np.float32(1.0)

        return {
            'depth_frame': depth_frame,
            'wrist_frame': wrist_frame,
            'angle': (angle / 100),
            'delta_angle': delta_angle,
            'task_description_embedding': task_description_embedding,
            'weight': weight,
            'gripper_has_object': int(meta['gripper_has_object']) if 'gripper_has_object' in meta and meta['gripper_has_object'] is not None else 0,
            'gripper_has_object_mask': 1 if 'gripper_has_object' in meta and meta['gripper_has_object'] is not None else 0
        }


    def get_depth_frame(self, frame_name):
        bgr_frame = cv2.imread(os.path.join(self.rgb_frames_path, f'{frame_name}.png'))
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        depth_frame = cv2.imread(os.path.join(self.depth_frames_path, f'{frame_name}.png'), cv2.IMREAD_ANYDEPTH)

        frame = DepthFrame.create_depth_frame(rgb_frame, depth_frame).get_frame()

        if self.augmentations:
            frame = self.augmentations(frame)

        if self.depth_frame_transform:
            frame = self.depth_frame_transform(frame)

        return frame
    

    def get_wrist_frame(self, frame_name):
        wrist_frame = cv2.imread(os.path.join(self.wrist_frames_path, f'{frame_name}.png'))

        if self.augmentations:
            wrist_frame = self.augmentations(wrist_frame)

        if self.wrist_frame_transform:
            wrist_frame = self.wrist_frame_transform(wrist_frame)

        return wrist_frame
    

    def get_angle(self, frame_name):
        # angle = np.loadtxt(os.path.join(self.angle_path, f'{frame_name}.csv')).astype(np.float32)
        with open(os.path.join(self.angle_path, f'{frame_name}.csv'), 'r') as f:
            angle = np.array([float(line) for line in f.readlines()]).astype(np.float32)

        if self.angle_augmentations:
            angle = self.angle_augmentations(angle)

        return angle
    

    def get_angle_delta(self, frame_name):
        # angle_delta = np.loadtxt(os.path.join(self.angle_delta_path, f'{frame_name}.csv')).astype(np.float32)
        # same as above but faster
        with open(os.path.join(self.angle_delta_path, f'{frame_name}.csv'), 'r') as f:
            angle_delta = np.array([float(line) for line in f.readlines()]).astype(np.float32)

        return angle_delta
    

    def get_meta(self, frame_name):
        meta_path = os.path.join(self.meta_path, f'{frame_name}.json')
        if not os.path.exists(meta_path):
            return {}

        with open(meta_path, 'r') as f:
            meta = json.load(f)

            return meta


    def get_task_description_embedding(self, task_description):
        description_hash = hashlib.md5(task_description.encode()).hexdigest()
        cache_path = os.path.join(TASK_DESCRIPTION_CACHE_PATH, f'{description_hash}.npy')
        return np.load(cache_path).astype(np.float32, copy=False)



def aggregate_data(clip_paths, depth_frame_transform=None, wrist_frame_transform=None, angle_transform=None, augmentations=None, angle_augmentations=None, chunk_size=1, num_workers=1):
    data = []
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            pool.starmap(DataFile, [(clip_path, depth_frame_transform, wrist_frame_transform, angle_transform, augmentations, angle_augmentations, chunk_size) for clip_path in clip_paths])
    else:
        for clip_path in clip_paths:
            datafile = DataFile(clip_path, depth_frame_transform, wrist_frame_transform, angle_transform, augmentations, angle_augmentations, chunk_size)
            data.append(datafile)

    return data


def split_data(datafiles, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0):
    np.random.shuffle(datafiles)
    n = len(datafiles)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = datafiles[:train_end]
    val_data = datafiles[train_end:val_end]
    test_data = datafiles[val_end:]

    return train_data, val_data, test_data


if __name__ == '__main__':
    data_path = Path('./data')
    clip_paths = recurse_dir_for_clips(data_path)

    datafiles = aggregate_data(
        clip_paths=clip_paths,
    )

    datafiles[0][0]