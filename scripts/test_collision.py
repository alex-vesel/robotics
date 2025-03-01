import threading
import numpy as np
from scipy import stats
import time
import cv2
import sys
import json
import os
import tty
import termios
import threading

from pymycobot.mycobot280 import MyCobot280


NOMINAL_VOLTAGES = [12.2, 12.2, 12.2, 7.4, 7.4, 7.4]

class CircularBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []

    def add(self, item):
        if len(self.buffer) >= self.size:
            self.buffer.pop(0)
        self.buffer.append(item)

# class get angles starts a thread which gets the angles of the robot and prints them
class get_angles(threading.Thread):
    def __init__(self, mc):
        threading.Thread.__init__(self)
        self.mc = mc
        self.buffer = CircularBuffer(10)
        self.voltage_buffer = CircularBuffer(10)
        self.is_moving_buffer = CircularBuffer(10)
        self.counter = 0

    def run(self):
        next_values = self.mc.get_angles_coords()[:6]
        self.r_values = [0, 0, 0, 0, 0, 0]
        slopes = [20, 20, 20, 20, 20, 20]
        while True:
            # print(self.mc.get_servo_status())
            # new_angles = self.mc.get_angles_coords()[:6]
            # print(new_angles)
            # if new_angles == self.prev_angles:
            #     print('no change')
            # self.prev_angles = new_angles

            # print(self.mc.get_servo_speeds()[1])
            new_angles = self.mc.get_angles_coords()[:6]
            self.buffer.add(new_angles)

            voltages = self.mc.get_servo_voltages()
            voltage_deviation = np.mean(np.abs(np.array(voltages) - np.array(NOMINAL_VOLTAGES)))


            self.voltage_buffer.add(voltage_deviation)
            self.is_moving_buffer.add(self.mc.is_moving())
            if len(self.voltage_buffer.buffer) == 10:
                print(np.mean(self.voltage_buffer.buffer))
                if np.mean(self.voltage_buffer.buffer) > 0.15 and np.mean(self.is_moving_buffer.buffer) == 0:
                    print('voltage deviation detected')
                    self.mc.stop()
                    self.mc.set_color(255, 0, 0)
                    exit()

            # print error

            # fit a line to the buffer along each axis
            if len(self.buffer.buffer) == 10:
                # print("r: ", r_value)
                # print(f'error: {new_angles[0] - next_value}')

                score = 0
                for i in range(6):
                    # if i == 4:
                    #     print(abs(new_angles[i] - next_values[i]) )
                    if abs(self.r_values[i]) > 0.95 and abs(new_angles[i] - next_values[i]) > abs(slopes[i]) * 2:
                        # print(i, self.r_values[i], new_angles[i], next_values[i], slopes[i])
                        # print(np.array(self.buffer.buffer)[:, i])
                        score += 1
                        # print('collision detected')
                        # self.mc.stop()
                        # self.mc.set_color(255, 0, 0)
                        # exit()
                    # print(score)
                    if score > 1:
                        print('collision detected')
                        self.mc.stop()
                        self.mc.set_color(255, 0, 0)
                        exit()
                # if abs(r_value) > 0.95 and abs(new_angles[0] - next_value) > 1.5:
                #     print('collision detected')
                #     self.mc.stop()
                #     exit()

                next_values = []
                slopes = []
                self.r_values = []
                for i in range(6):
                    slope, intercept, r_value, p_value, std_err = stats.linregress(range(10), [x[i] for x in self.buffer.buffer])
                    # print(f'slope: {slope}, intercept: {intercept}, r_value: {r_value}, p_value: {p_value}, std_err: {std_err}')
                    # predict the next value
                    # next_value = slope * 10 + intercept

                    slopes.append(slope)
                    next_values.append(slope * 10 + intercept)
                    self.r_values.append(r_value)


            #     print(np.mean(np.max(self.buffer.buffer, axis=0) - np.min(self.buffer.buffer, axis=0)))
            #     # print(np.mean(np.std(self.buffer.buffer, axis=0)))
            #     print(np.std(self.buffer.buffer, axis=0))
            #     if np.mean(np.std(self.buffer.buffer, axis=0)) > 100:
            #         print('collision detected')
            #         self.mc.stop()
            #         exit()

    def clear_buffer(self):
        self.buffer = CircularBuffer(10)
        self.r_values = [0, 0, 0, 0, 0, 0]


class Raw(object):
    """Set raw input mode for device"""

    def __init__(self, stream):
        self.stream = stream
        self.fd = self.stream.fileno()

    def __enter__(self):
        self.original_stty = termios.tcgetattr(self.stream)
        tty.setcbreak(self.stream)

    def __exit__(self, type, value, traceback):
        termios.tcsetattr(self.stream, termios.TCSANOW, self.original_stty)


class Helper(object):
    def __init__(self) -> None:
        self.w, self.h = os.get_terminal_size()

    def echo(self, msg):
        print("\r{}".format(" " * self.w), end="")
        print("\r{}".format(msg), end="")


class TeachingTest(Helper):
    def __init__(self, mycobot) -> None:
        super().__init__()
        self.mc = mycobot
        self.recording = False
        self.playing = False
        self.record_list = []
        self.record_t = None
        self.play_t = None

    def print_menu(self):
        print(
            """\
        \r q: quit
        \r r: start record
        \r c: stop record
        \r p: play once
        \r P: loop play / stop loop play
        \r s: save to local
        \r l: load from local
        \r f: release mycobot
        \r e: stand erect
        \r----------------------------------
            """
        )

    def start(self):
        self.print_menu()

        while not False:
            with Raw(sys.stdin):
                key = sys.stdin.read(1)
                current_angles = self.mc.get_angles_coords()[:6]
                gripper_value = self.mc.get_gripper_value()
                # if key == "q":
                #     break
                if key == "a":
                    current_angles[0] -= 10
                    self.mc.send_angles(current_angles, 20)
                elif key == "d":
                    current_angles[0] += 10
                    self.mc.send_angles(current_angles, 20)
                elif key == "w":
                    current_angles[1] -= 10
                    self.mc.send_angles(current_angles, 20)
                elif key == "s":
                    current_angles[1] += 10
                    self.mc.send_angles(current_angles, 20)
                elif key == "q":
                    current_angles[2] -= 10
                    self.mc.send_angles(current_angles, 20)
                elif key == "e":
                    current_angles[2] += 10
                    self.mc.send_angles(current_angles, 20)
                elif key == "r":
                    gripper_value -= 20
                    gripper_value = max(0, gripper_value)
                    self.mc.set_gripper_value(gripper_value, 20)
                elif key == "f":
                    gripper_value += 20
                    gripper_value = min(100, gripper_value)
                    self.mc.set_gripper_value(gripper_value, 20)
                elif key == "p":  # stand erect
                    self.mc.send_angles([0, 0, 0, 0, 0, 0], 90)
                else:
                    print(key)
                    continue


if __name__ == "__main__":
    mc = MyCobot280('/dev/tty.usbserial-588D0018121',115200)

    # start the get_angles thread
    ga_class = get_angles(mc)
    ga_class.start()

    mc.set_color(0, 255, 0)

    mc.set_fresh_mode(1)
    print(mc.get_fresh_mode())

    # print('moving 1')
    mc.sync_send_angles([0, 0, 0, 0, 0, 0], 20, timeout=3)
    ga_class.clear_buffer()
    # print('moving 2')
    # mc.sync_send_angles([0, 0, 0, 0, 50, 0], 20, timeout=3)
    mc.sync_send_angles([30, 10, 50, 20, 30, 40], 20)

    # mc.send_angles([0, 0, 0, 0, 0, 0], 90)
    # mc.set_fresh_mode(0)
    # recorder = TeachingTest(mc)
    # recorder.start()

    # while True:
    #     mc.send_angles([0, 0, 0, 0, 0, 0], 50)
    #     mc.send_angles([0, 0, 0, 0, 50, 0], 50)
