import os
import cv2
import torch
import json
import multiprocessing
import numpy as np
from pathlib import Path
from collections import OrderedDict
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from shared_utils.path_utils import recurse_dir_for_clips
from camera.depth_camera import DepthFrame


class DataFile(Dataset):
    def __init__(self,
                 clip_path: str,
                 depth_frame_transform=None,
                 wrist_frame_transform=None,
                 angle_transform=None,
                 augmentations=None,
                 angle_augmentations=None,
                ):

        self.clip_path = clip_path
        self.clip_name = os.path.join(*clip_path.split('/')[-2:])
        self.angle_path = os.path.join(clip_path, 'angles')
        self.depth_frames_path = os.path.join(clip_path, 'depth_frames')
        self.rgb_frames_path = os.path.join(clip_path, 'rgb_frames')
        self.wrist_frames_path = os.path.join(clip_path, 'wrist_frames')
        self.depth_frame_transform = depth_frame_transform
        self.wrist_frame_transform = wrist_frame_transform
        self.augmentations = augmentations
        self.angle_augmentations = angle_augmentations

        self.idx_to_frame_name = OrderedDict()
        # path names are float, so we need to sort them as floats
        sorted_frames = sorted(os.listdir(self.angle_path), key=lambda x: float(".".join(x.split('.')[:-1])))
        for idx, frame in enumerate(sorted_frames):
            self.idx_to_frame_name[idx] = ".".join(frame.split('.')[:-1])
        

    def __len__(self):
        return len(os.listdir(self.angle_path)) - 1


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
        next_angle = self.get_angle(next_frame_name)

        delta_angle = next_angle - angle

        weight = 1.0
        # upweight when gripper is closing
        if delta_angle[-1] < 0:
            weight = 3.0

        if np.all(delta_angle == 0):
            weight = 0.0

        return {
            'depth_frame': depth_frame,
            'wrist_frame': wrist_frame,
            'angle': (angle / 100),
            'delta_angle': delta_angle,
            'weight': weight,
        }


    def get_depth_frame(self, frame_name):
        rgb_frame = cv2.imread(os.path.join(self.rgb_frames_path, f'{frame_name}.png'))
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
        angle = np.loadtxt(os.path.join(self.angle_path, f'{frame_name}.csv')).astype(np.float32)

        if self.angle_augmentations:
            angle = self.angle_augmentations(angle)

        return angle


def aggregate_data(clip_paths, depth_frame_transform=None, wrist_frame_transform=None, angle_transform=None, augmentations=None, angle_augmentations=None, num_workers=1):
    data = []
    if num_workers > 1:
        with multiprocessing.Pool(num_workers) as pool:
            pool.starmap(DataFile, [(clip_path, depth_frame_transform, wrist_frame_transform, angle_transform, augmentations, angle_augmentations) for clip_path in clip_paths])
    else:
        for clip_path in clip_paths:
            datafile = DataFile(clip_path, depth_frame_transform, wrist_frame_transform, angle_transform, augmentations, angle_augmentations)
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