import time
import cv2
import sys
import json
import os
import tty
import termios
import threading

from pymycobot.mycobot280 import MyCobot280


if __name__ == "__main__":
    mc = MyCobot280('/dev/tty.usbmodem588D0018121',115200)
    # mc.send_angles([0, 0, 0, 0, 0, 0], 90)

    # mc.solve_inv_kinematics([52.0, -64., 409.7], [0., 0., 0., 0., 0., 0.])
    # import IPython; IPython.embed(); exit(0)
    # mc.solve_inv_kinematics([0., 0., 0.], [0., 0., 0., 0., 0., 0.])

    # mc.solve_inv_kinematics([-82.9, -52.2, 237.7, -145.01, 3.64, -56.21], mc.get_angles_coords()[:6])

    # mc.sync_send_coords([-82.9, -52.2, 237.7, -145.01, 3.64, -56.21], 40)
    for _ in range(5):
        coords = mc.get_angles_coords()[6:]
        coords[2] -= 10
        mc.sync_send_coords(coords, 40, timeout=0.1)

    import IPython; IPython.embed(); exit(0)

    mc.angles_to_coords([0, 0, 0, 0, 0, 0])