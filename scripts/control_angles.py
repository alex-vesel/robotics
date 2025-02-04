import time
import cv2
import sys
import json
import os
import tty
import termios
import threading

from pymycobot.mycobot280 import MyCobot280



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
        self.path = os.path.dirname(os.path.abspath(__file__)) + "/record.txt"

    def record(self):
        self.record_list = []
        self.recording = True
        #self.mc.set_fresh_mode(0)
        def _record():
            main_start_t = time.time()
            while self.recording:
                start_t = time.time()
                angles = self.mc.get_encoders()
                speeds = self.mc.get_servo_speeds()
                gripper_value = self.mc.get_encoder(7)
                interval_time = time.time() - start_t
                main_time = time.time() - main_start_t
                if angles and speeds:
                    record = [angles, speeds, gripper_value, interval_time, main_time]
                    self.record_list.append(record)
                    # time.sleep(0.1)
                    print("\r {}".format(time.time() - start_t), end="")

        self.echo("Start recording.")
        self.record_t = threading.Thread(target=_record, daemon=True)
        self.record_t.start()

    def stop_record(self):
        if self.recording:
            self.recording = False
            self.record_t.join()
            self.echo("Stop record")

    def play(self):
        self.echo("Start play")
        i = 0
        main_start_t = time.time()
        for record in self.record_list:
            angles, speeds, gripper_value, interval_time, main_time = record
            print(time.time() - main_start_t, main_time)
            #print(angles)
            # import IPython; IPython.embed(); exit(0)
            # start = time.time()
            print(speeds)
            self.mc.set_encoders(angles, 80)
            self.mc.set_encoder(7, gripper_value, 80)
            # self.mc.send_angles([0, 0, 0, 0, 0, 0], 90)
            # print("set_encoders", time.time() - start)
            print(main_time - (time.time() - main_start_t))
            time.sleep(max(main_time - (time.time() - main_start_t), 0))
            # self.mc.set_encoder(7, gripper_value, 80)
            # if i == 0:
            #     time.sleep(3)
            # i+=1
            # time.sleep(interval_time)
        self.echo("Finish play")

    def loop_play(self):
        self.playing = True

        def _loop():
            while self.playing:
                self.play()

        self.echo("Start loop play.")
        self.play_t = threading.Thread(target=_loop, daemon=True)
        self.play_t.start()

    def stop_loop_play(self):
        if self.playing:
            self.playing = False
            self.play_t.join()
            self.echo("Stop loop play.")

    def save_to_local(self):
        if not self.record_list:
            self.echo("No data should save.")
            return
        with open(self.path, "w") as f:
            json.dump(self.record_list, f, indent=2)
            self.echo("save dir:  {}\n".format(self.path))

    def load_from_local(self):
        with open(self.path, "r") as f:
            try:
                data = json.load(f)
                self.record_list = data
                self.echo("Load data success.")
            except Exception:
                self.echo("Error: invalid data.")

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
# def send_angle_test(mc):
#     mc.send_angles([0, 0, 0, 0, 50, 0], 20)

if __name__ == "__main__":
    mc = MyCobot280('/dev/tty.usbmodem588D0018121',115200)
    # send_angle_test(mc)
    # exit()
    # import IPython; IPython.embed(); exit(0)
    # mc.send_angles([0, 0, 0, 0, 140, 0], 90)
    mc.send_angles([0, 0, 0, 0, 0, 0], 90)
    mc.set_fresh_mode(0)
    recorder = TeachingTest(mc)
    recorder.start()