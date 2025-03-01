import time
from threading import Thread
import cv2

# This class is used to create a thread for the camera and stores the frame
class CameraThread(Thread):
    def __init__(self, camera):
        Thread.__init__(self)
        self.camera = camera
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            self.frame = self.camera.get_frame()
            time.sleep(0.01)

    def get_frame(self):
        return self.frame

    def stop(self):
        self.running = False
        self.join()


if __name__ == '__main__':
    camera = Camera()
    camera_thread = CameraThread(camera)
    camera_thread.start()
    time.sleep(0.5)

    while True:
        frame = camera_thread.frame.extract_rgb_frame()
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera_thread.stop()
    cv2.destroyAllWindows()