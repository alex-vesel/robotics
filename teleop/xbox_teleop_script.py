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

import cProfile
def profileit(func):
    def wrapper(*args, **kwargs):
        datafn = func.__name__ + ".profile" # Name the data file sensibly
        prof = cProfile.Profile()
        retval = prof.runcall(func, *args, **kwargs)
        prof.dump_stats(datafn)
        return retval

    return wrapper

# @profileit
def update_teleop(teleop, depth_camera, wrist_camera, experiment_logger):
    angles, angle_delta, was_change = teleop.update()
    if args.record and was_change:
        depth_frame = depth_camera.get_frame()
        wrist_frame = wrist_camera.get_frame()
        experiment_logger.log(monotonic() - start_time, angles, angle_delta, depth_frame, wrist_frame)


def run_teleop(teleop, depth_camera, wrist_camera, experiment_logger):
    while True:
        update_teleop(teleop, depth_camera, wrist_camera, experiment_logger)


if __name__ == '__main__':
    args = parser.parse_args()

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
            run_teleop(teleop, depth_camera, wrist_camera, experiment_logger)
        except KeyboardInterrupt:
            sleep(1)
            pass
