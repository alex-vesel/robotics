import sys
import cv2
import os
import pygame
import threading
import datetime
import argparse
import numpy as np
from time import sleep
sys.path.append(".")
import math
from time import monotonic
from pymycobot.mycobot280 import MyCobot280

from camera.depth_camera import DepthCamera
from camera.camera import Camera
from camera.camera_thread import CameraThread
from shared_utils.loggers.data_logger import DataLogger
from teleop.utils.xbox_teleop import XboxTeleop
from teleop.teleop_constants import INITIAL_ANGLES, TELEOP_DATA_DIR, SAVED_ANGLES_DIR, ZERO_ANGLES

parser = argparse.ArgumentParser()
parser.add_argument('--record', type=bool, default=False)
parser.add_argument('--experiment_name', type=str, default='default')
parser.add_argument('--num_recordings', type=int, default=1, help='Number of recordings to make, allows to capture camera noise')
parser.add_argument('--gripper_has_object', type=str, default=None)

STEP_COUNT = 0


def update_teleop(teleop, depth_camera, wrist_camera, experiment_loggers, meta):
    global STEP_COUNT
    cur_time = monotonic()

    angles, angle_delta, was_change, reset = teleop.get_command()

    if reset:
        raise RuntimeError("Resetting")

    if args.record and was_change:
        depth_frames = []
        wrist_frames = []
        d_frames = []
        w_frames = []
        for i, experiment_logger in enumerate(experiment_loggers):
            depth_frames.append(depth_camera.get_frame())
            wrist_frames.append(wrist_camera.get_frame())
            d_frames.append(depth_frames[-1].get_frame())
            w_frames.append(wrist_frames[-1].get_frame())
            if i < len(experiment_loggers) - 1:
                sleep(0.02)
        # shape is (num_cameras, height, width, channels)
        # get unique
        # if STEP_COUNT % 10 == 0:
        #     print("Unique depth frames: ", np.unique(np.array(d_frames), axis=0).shape[0])
        #     print("Unique wrist frames: ", np.unique(np.array(w_frames), axis=0).shape[0])

    if was_change:
        teleop.send_command()

    if args.record and was_change:
        # don't log first 5 frames of data if gripper has object to allow for closing of gripper
        STEP_COUNT += 1
        if (meta["gripper_has_object"] is not None) and (STEP_COUNT < 20 or STEP_COUNT % 2 != 0):
            return
        for i, experiment_logger in enumerate(experiment_loggers):
            experiment_logger.log(cur_time - start_time, angles, angle_delta, depth_frames[i], wrist_frames[i], meta)
    # print(monotonic() - cur_time)
    # sleep(0.2)


def run_teleop(teleop, depth_camera, wrist_camera, experiment_logger, meta):
    while True:
        update_teleop(teleop, depth_camera, wrist_camera, experiment_logger, meta)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.gripper_has_object is not None:
        gripper_has_object = args.gripper_has_object == "True"
    else:
        gripper_has_object = None
    meta = {"gripper_has_object": gripper_has_object}
    print(meta)

    pygame.init()

    # randomly select initial angles from SAVED_ANGLES_DIR
    if args.record:
        depth_camera = CameraThread(DepthCamera())
        wrist_camera = CameraThread(Camera())
        depth_camera.start()
        wrist_camera.start()
    else:
        depth_camera = None
        wrist_camera = None
        experiment_logger = None

    mc = MyCobot280('/dev/tty.usbserial-588D0018121',115200)

    # main data recording loop
    while True:
        # saved_angles_paths = os.listdir(SAVED_ANGLES_DIR)
        # INITIAL_ANGLES = np.load(os.path.join(SAVED_ANGLES_DIR, np.random.choice(saved_angles_paths))).tolist()
        # for i in range(6):
        #     INITIAL_ANGLES[i] += np.random.randint(-5, 5)

        # INITIAL_ANGLES = [angle + np.random.randint(-30, 30) for angle in ZERO_ANGLES]
        # INITIAL_ANGLES[0] = np.random.randint(-40, 40)
        # INITIAL_ANGLES[1] = np.random.randint(-80, -50)
        # INITIAL_ANGLES[-1] = -45 + np.random.randint(-5, 5)
        # INITIAL_ANGLES.append(np.random.randint(0, 90))

        INITIAL_ANGLES = ZERO_ANGLES.copy()
        # INITIAL_ANGLES[0] = np.random.randint(-40, 40)
        # INITIAL_ANGLES[1] = np.random.randint(5, 20)
        # INITIAL_ANGLES[2] = np.random.randint(-10, 10)
        # INITIAL_ANGLES[3] = np.random.randint(-10, 10)
        # INITIAL_ANGLES[4] = np.random.randint(-10, 10)
        # INITIAL_ANGLES[5] = -45 + np.random.randint(-5, 5)
        INITIAL_ANGLES.append(0)

        teleop = XboxTeleop(mc, initial_angles=INITIAL_ANGLES)

        experiment_loggers = []
        if args.record:
            data_dir = os.path.join(TELEOP_DATA_DIR, args.experiment_name + datetime.datetime.now().strftime("_%Y-%m-%d_%H-%M-%S"))
            for i in range(args.num_recordings):
                experiment_loggers.append(DataLogger(os.path.join(data_dir, f'{args.experiment_name}_{i}')))

        mc.set_fresh_mode(0)
        mc.sync_send_angles(INITIAL_ANGLES[:6], 40, timeout=3)
        mc.set_gripper_value(int(INITIAL_ANGLES[6]), 40)
        mc.set_fresh_mode(1)

        start_time = monotonic()
        r_pressed = False
        try:
            run_teleop(teleop, depth_camera, wrist_camera, experiment_loggers, meta)
        except KeyboardInterrupt:
            mc._serial_port.close()
            exit()
        except RuntimeError:
            sleep(0.5)
