import os
import numpy as np
import cv2
import datetime

from camera.depth_camera import DepthCamera

class DataLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_dir += datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        self.angles_dir = os.path.join(self.log_dir, 'angles')
        self.rgb_frames_dir = os.path.join(self.log_dir, 'rgb_frames')
        self.depth_frames_dir = os.path.join(self.log_dir, 'depth_frames')
        self.wrist_frames_dir = os.path.join(self.log_dir, 'wrist_frames')

        os.makedirs(self.angles_dir, exist_ok=True)
        os.makedirs(self.rgb_frames_dir, exist_ok=True)
        os.makedirs(self.depth_frames_dir, exist_ok=True)
        os.makedirs(self.wrist_frames_dir, exist_ok=True)

    def log(self, time, angles, depth_frame, wrist_frame):
        self.log_angles(time, angles)
        self.log_depth_frame(time, depth_frame)
        self.log_wrist_frame(time, wrist_frame)

    def log_angles(self, time, angles):
        np.savetxt(os.path.join(self.angles_dir, f'{time}.csv'), angles, delimiter=',')

    def log_depth_frame(self, time, depth_frame):
        cv2.imwrite(os.path.join(self.rgb_frames_dir, f'{time}.png'), depth_frame.extract_bgr_frame())
        cv2.imwrite(os.path.join(self.depth_frames_dir, f'{time}.png'), depth_frame.extract_depth_frame())

    def log_wrist_frame(self, time, wrist_frame):
        cv2.imwrite(os.path.join(self.wrist_frames_dir, f'{time}.png'), wrist_frame.extract_bgr_frame())