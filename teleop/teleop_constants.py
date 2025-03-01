import numpy as np
import os

from teleop.utils.xbox_to_mycobot_mapping import XboxToMyCobotMapping

TELEOP_DATA_DIR = './teleop/data_wrist_presave'
SAVED_ANGLES_DIR = './teleop/saved_angles'


ZERO_ANGLES = [0, 0, 0, 0, 0, -45]

# randomize intial angles plus or minus 10 degrees
INITIAL_ANGLES = [angle + np.random.randint(-10, 10) for angle in ZERO_ANGLES]
INITIAL_ANGLES[-1] = -45
# add a random angle for the gripper
INITIAL_ANGLES.append(np.random.randint(0, 90))

HIGH_PRECISION_SCALE_FACTOR = 3

XBOX_MYCOBOT_MAPPINGS = [
    XboxToMyCobotMapping('Base', 'LeftJoystickX', 3),
    XboxToMyCobotMapping('Shoulder', 'LeftJoystickY', 3),
    XboxToMyCobotMapping('Elbow', 'RightJoystickY', 3),
    XboxToMyCobotMapping('Wrist1', 'RightJoystickX', -3),
    XboxToMyCobotMapping('Wrist2', 'X', -3, negative_key='B'),
    XboxToMyCobotMapping('Wrist3', 'RightTrigger', -3, negative_key='LeftTrigger'),
]

GRIPPER_MAPPING = XboxToMyCobotMapping('Gripper', 'LeftBumper', 6, negative_key='RightBumper')