import pygame
import math


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.init_controller_values()

        pygame.init()
        self.controller = pygame.joystick.Joystick(0)
        self.controller.init()

    def read(self):
        self._monitor_controller()
        return {
            'LeftJoystickX': self.LeftJoystickX,
            'LeftJoystickY': self.LeftJoystickY,
            'RightJoystickX': self.RightJoystickX,
            'RightJoystickY': self.RightJoystickY,
            'LeftTrigger': self.LeftTrigger,
            'RightTrigger': self.RightTrigger,
            'LeftBumper': self.LeftBumper,
            'RightBumper': self.RightBumper,
            'A': self.A,
            'X': self.X,
            'Y': self.Y,
            'B': self.B,
            'LeftThumb': self.LeftThumb,
            'RightThumb': self.RightThumb,
            'Back': self.Back,
            'Start': self.Start,
            'LeftDPad': self.LeftDPad,
            'RightDPad': self.RightDPad,
            'UpDPad': self.UpDPad,
            'DownDPad': self.DownDPad
        }

    def init_controller_values(self):
        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

    def _monitor_controller(self):
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == 0:
                    self.LeftJoystickX = event.value
                elif event.axis == 1:
                    self.LeftJoystickY = event.value
                elif event.axis == 2:
                    self.RightJoystickX = event.value
                elif event.axis == 3:
                    self.RightJoystickY = event.value
                elif event.axis == 4:
                    self.LeftTrigger = (event.value + 1) / 2
                elif event.axis == 5:
                    self.RightTrigger = (event.value + 1) / 2
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    self.A = 1
                elif event.button == 1:
                    self.B = 1
                elif event.button == 2:
                    self.X = 1
                elif event.button == 3:
                    self.Y = 1
                elif event.button == 9:
                    self.LeftBumper = 1
                elif event.button == 10:
                    self.RightBumper = 1
                elif event.button == 11:
                    self.UpDPad = 1
                elif event.button == 12:
                    self.DownDPad = 1
                elif event.button == 13:
                    self.LeftDPad = 1
                elif event.button == 14:
                    self.RightDPad = 1
            elif event.type == pygame.JOYBUTTONUP:
                if event.button == 0:
                    self.A = 0
                elif event.button == 1:
                    self.B = 0
                elif event.button == 2:
                    self.X = 0
                elif event.button == 3:
                    self.Y = 0
                elif event.button == 9:
                    self.LeftBumper = 0
                elif event.button == 10:
                    self.RightBumper = 0
                elif event.button == 11:
                    self.UpDPad = 0
                elif event.button == 12:
                    self.DownDPad = 0
                elif event.button == 13:
                    self.LeftDPad = 0
                elif event.button == 14:
                    self.RightDPad = 0


if __name__ == '__main__':
    # test the controller
    xbox = XboxController()
    while True:
        print(xbox.read())
        pygame.time.wait(100)