import sys
import cv2
import os
import pygame
import threading
import argparse
import numpy as np
from time import sleep
sys.path.append(".")
import math
from time import monotonic
from pymycobot.mycobot280 import MyCobot280

from camera.depth_camera import DepthCamera
from camera.camera import Camera
from shared_utils.loggers.data_logger import DataLogger
from teleop.utils.xbox_teleop import XboxTeleop
from teleop.teleop_constants import INITIAL_ANGLES, TELEOP_DATA_DIR, SAVED_ANGLES_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--record', type=bool, default=False)
parser.add_argument('--experiment_name', type=str, default='default')
parser.add_argument('--gripper_has_object', type=str, default=None)

STEP_COUNT = 0

def update_teleop(teleop, depth_camera, wrist_camera, experiment_logger, meta):
    global STEP_COUNT
    angles, angle_delta, was_change = teleop.update()
    if args.record and was_change:
        depth_frame = depth_camera.get_frame()
        wrist_frame = wrist_camera.get_frame()
        # don't log first 5 frames of data if gripper has object to allow for closing of gripper
        STEP_COUNT += 1
        if meta["gripper_has_object"] is not None and STEP_COUNT < 5:
            return
        experiment_logger.log(monotonic() - start_time, angles, angle_delta, depth_frame, wrist_frame, meta)


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
        depth_camera = DepthCamera()
        wrist_camera = Camera()
    else:
        depth_camera = None
        wrist_camera = None
        experiment_logger = None

    # main data recording loop
    while True:
        # saved_angles_paths = os.listdir(SAVED_ANGLES_DIR)
        # INITIAL_ANGLES = np.load(os.path.join(SAVED_ANGLES_DIR, np.random.choice(saved_angles_paths))).tolist()

        mc = MyCobot280('/dev/tty.usbmodem588D0018121',115200)
        teleop = XboxTeleop(mc, initial_angles=INITIAL_ANGLES)

        if args.record:
            experiment_logger = DataLogger(os.path.join(TELEOP_DATA_DIR, args.experiment_name))

        mc.set_fresh_mode(0)
        mc.sync_send_angles(INITIAL_ANGLES[:6], 40, timeout=3)
        mc.set_gripper_value(int(INITIAL_ANGLES[6]), 40)
        mc.set_fresh_mode(1)

        start_time = monotonic()
        r_pressed = False
        try:
            run_teleop(teleop, depth_camera, wrist_camera, experiment_logger, meta)
        except KeyboardInterrupt:
            sleep(1)
            pass
