import os
import json
import numpy as np
import cv2
import datetime

from camera.depth_camera import DepthCamera

class DataLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.log_dir += datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S")
        self.angles_dir = os.path.join(self.log_dir, 'angles')
        self.angle_delta_dir = os.path.join(self.log_dir, "angle_delta")
        self.meta_dir = os.path.join(self.log_dir, "meta")
        self.rgb_frames_dir = os.path.join(self.log_dir, 'rgb_frames')
        self.depth_frames_dir = os.path.join(self.log_dir, 'depth_frames')
        self.wrist_frames_dir = os.path.join(self.log_dir, 'wrist_frames')

        os.makedirs(self.angles_dir, exist_ok=True)
        os.makedirs(self.angle_delta_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.rgb_frames_dir, exist_ok=True)
        os.makedirs(self.depth_frames_dir, exist_ok=True)
        os.makedirs(self.wrist_frames_dir, exist_ok=True)

    def log(self, time, angles, angle_delta, depth_frame, wrist_frame, meta={}):
        self.log_angles(time, angles)
        self.log_angle_delta(time, angle_delta)
        self.log_meta(time, meta)
        self.log_depth_frame(time, depth_frame)
        self.log_wrist_frame(time, wrist_frame)

    def log_angles(self, time, angles):
        np.savetxt(os.path.join(self.angles_dir, f'{time}.csv'), angles, delimiter=',')

    def log_angle_delta(self, time, angle_delta):
        np.savetxt(os.path.join(self.angle_delta_dir, f'{time}.csv'), angle_delta, delimiter=',')

    def log_meta(self, time, meta):
        with open(os.path.join(self.meta_dir, f'{time}.json'), 'w') as f:
            json.dump(meta, f)

    def log_depth_frame(self, time, depth_frame):
        cv2.imwrite(os.path.join(self.rgb_frames_dir, f'{time}.png'), depth_frame.extract_bgr_frame())
        cv2.imwrite(os.path.join(self.depth_frames_dir, f'{time}.png'), depth_frame.extract_depth_frame())

    def log_wrist_frame(self, time, wrist_frame):
        cv2.imwrite(os.path.join(self.wrist_frames_dir, f'{time}.png'), wrist_frame.extract_bgr_frame())