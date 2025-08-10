import math
import time

from teleop.utils.xbox_controller import XboxController
from teleop.teleop_constants import *


class XboxTeleop:
    def __init__(self, mc, initial_angles=INITIAL_ANGLES):
        self.controller = XboxController()
        self.mc = mc
        self.high_precision_mode = False
        self.prev_state_y = False
        self.global_angle = initial_angles[:6]
        self.global_gripper = int(initial_angles[6])

    def get_command(self):
        prev_global_angle = self.global_angle.copy()
        prev_global_angle.append(self.global_gripper)
        # process
        angle_deltas, gripper_delta, return_init, reset = self._get_controller_command()
        if reset:
            return None, None, None, True
        if return_init:
            self.angle_change = self.global_angle != ZERO_ANGLES
            for i in range(6):
                delta_angle = self.global_angle[i] - ZERO_ANGLES[i]
                self.global_angle[i] = self.global_angle[i] - math.copysign(1, delta_angle) * min(4, abs(delta_angle))
        else:
            # if abs(gripper_delta) > 0:
            #     for i in range(6):
            #         angle_deltas[i] /= 2
            self.global_angle = [current_angle + delta for current_angle, delta in zip(self.global_angle, angle_deltas)]
            self.angle_change = angle_deltas != [0, 0, 0, 0, 0, 0]

        self.global_gripper += gripper_delta
        self.global_gripper = min(max(0, self.global_gripper), 90)
        gripper_change = self.global_gripper != prev_global_angle[6]

        if self.angle_change or gripper_change:
            cur_global_angle = self.global_angle.copy()
            cur_global_angle.append(self.global_gripper)
            angle_delta = np.array(cur_global_angle) - np.array(prev_global_angle)
        else:
            angle_delta = None

        # randomly perturb gripper
        if self.angle_change:
            self.global_gripper += int(np.random.randint(-2, 2))
            self.global_gripper = min(max(0, self.global_gripper), 90)

        self.perturb_gripper_change = self.global_gripper != prev_global_angle[6]

        # send commands
        if self.angle_change:
            # perturb angles randomly similar to DART
            for i in range(6):
                self.global_angle[i] += np.random.normal(0.0, 0.6)
            self.global_angle[1] = min(max(-130, self.global_angle[1]), 100)

        # if gripper_change:
        # if perturb_gripper_change:
        #     self.mc.set_gripper_value(self.global_gripper, 50)

        out = self.global_angle.copy()
        out.append(self.global_gripper)

        return out, angle_delta, self.angle_change or gripper_change, False


    def send_command(self):
        if self.angle_change:
            self.mc.send_angles(self.global_angle, 60)
            time.sleep(0.02)
        # print(self.global_angle)
        # print(self.global_gripper)
        if self.perturb_gripper_change:
            self.mc.set_gripper_value(self.global_gripper, 40)
            time.sleep(0.02)


    def _get_controller_command(self):
        controller_state = self.controller.read()
        
        angle_deltas = [mapping.parse(controller_state) for mapping in XBOX_MYCOBOT_MAPPINGS]
        gripper_delta = GRIPPER_MAPPING.parse(controller_state)

        if controller_state['UpDPad']:
            return None, None, None, True

        if controller_state['Y'] and not self.prev_state_y:
            self.high_precision_mode = not self.high_precision_mode
            print("toggle high precision: ", self.high_precision_mode)
        self.prev_state_y = controller_state['Y']

        if self.high_precision_mode:
            angle_deltas = [angle / HIGH_PRECISION_SCALE_FACTOR for angle in angle_deltas]

        return angle_deltas, gripper_delta, controller_state['A'], False