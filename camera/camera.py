import cv2
import numpy as np
import matplotlib.pyplot as plt


class CameraFrame():
    def __init__(self, frame):
        self.frame = cv2.resize(frame, (640, 480))
    
    def extract_bgr_frame(self):
        return self.frame[:, :, ::-1].astype(np.uint8)
    
    def extract_rgb_frame(self):
        return self.frame.astype(np.uint8)
    
    def render_frame(self):
        # show both rgb and depth images
        bgr_frame = self.extract_rgb_frame()

        plt.imshow(bgr_frame)
        plt.show()
    
    def get_frame(self):
        return self.frame

    @staticmethod
    def create_camera_frame(bgr_frame):
        return CameraFrame(bgr_frame)


class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
 
    def get_frame(self):
        ret, frame = self.cap.read()
        return CameraFrame(frame)
 
    def __del__(self):
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    camera = Camera()
    while True:
        frame = camera.get_frame().extract_rgb_frame()
        frame = cv2.resize(frame, (640, 480))
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()