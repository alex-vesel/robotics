import os
import json
import numpy as np
import cv2
import datetime
import threading
import queue

from camera.depth_camera import DepthCamera

class DataLogger:
    def __init__(self, log_dir, num_worker_threads=4):
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

        # Create a queue for image logging tasks
        self.image_queue = queue.Queue()

        # Start worker threads
        self.threads = []
        for _ in range(num_worker_threads):
            thread = threading.Thread(target=self._image_writer_worker)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)

    def log(self, time, angles, angle_delta, depth_frame, wrist_frame, meta={}):
        self.log_angles(time, angles)
        self.log_angle_delta(time, angle_delta)
        self.log_meta(time, meta)

        # Add image save tasks to the queue
        self._enqueue_image_save_task(time, depth_frame, wrist_frame)

    def log_angles(self, time, angles):
        np.savetxt(os.path.join(self.angles_dir, f'{time}.csv'), angles, delimiter=',')

    def log_angle_delta(self, time, angle_delta):
        np.savetxt(os.path.join(self.angle_delta_dir, f'{time}.csv'), angle_delta, delimiter=',')

    def log_meta(self, time, meta):
        with open(os.path.join(self.meta_dir, f'{time}.json'), 'w') as f:
            json.dump(meta, f)

    def _enqueue_image_save_task(self, time, depth_frame, wrist_frame):
        # Put the image saving tasks onto the queue for workers to process
        self.image_queue.put((time, depth_frame, wrist_frame))

    def _image_writer_worker(self):
        while True:
            time, depth_frame, wrist_frame = self.image_queue.get()

            # Process image saving tasks
            self.log_depth_frame(time, depth_frame)
            self.log_wrist_frame(time, wrist_frame)

            # Signal that the task has been processed
            self.image_queue.task_done()

    def log_depth_frame(self, time, depth_frame):
        rgb_frame = depth_frame.extract_rgb_frame()
        depth_frame = depth_frame.extract_depth_frame()
        rgb_frame = cv2.resize(rgb_frame, (rgb_frame.shape[1] // 2, rgb_frame.shape[0] // 2))
        depth_frame = cv2.resize(depth_frame, (depth_frame.shape[1] // 2, depth_frame.shape[0] // 2))
        cv2.imwrite(os.path.join(self.rgb_frames_dir, f'{time}.png'), rgb_frame)
        cv2.imwrite(os.path.join(self.depth_frames_dir, f'{time}.png'), depth_frame)

    def log_wrist_frame(self, time, wrist_frame):
        wrist_frame = wrist_frame.extract_rgb_frame()
        wrist_frame = cv2.resize(wrist_frame, (wrist_frame.shape[1] // 2, wrist_frame.shape[0] // 2))
        cv2.imwrite(os.path.join(self.wrist_frames_dir, f'{time}.png'), wrist_frame)
