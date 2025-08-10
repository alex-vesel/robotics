import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt
from time import monotonic
import cv2


class DepthFrame():
    def __init__(self, frame):
        self.frame = frame
    
    def extract_bgr_frame(self):
        return self.frame[:, :, :3][:, :, ::-1].astype(np.uint8)

    def extract_rgb_frame(self):
        return self.frame[:, :, :3].astype(np.uint8)
    
    def extract_depth_frame(self):
        return self.frame[:, :, -1].astype(np.uint16)

    def swap_rgb(self):
        self.frame[:, :, :3] = self.frame[:, :, :3][:, :, ::-1]

    def render_frame(self):
        # show both rgb and depth images
        bgr_frame = self.extract_rgb_frame()
        depth_frame = self.extract_depth_frame()

        plt.subplot(1, 2, 1)
        plt.imshow(bgr_frame)
        plt.subplot(1, 2, 2)
        plt.imshow(depth_frame)
        plt.show()

    def get_frame(self):
        return self.frame

    @staticmethod
    def create_depth_frame(bgr_frame, depth_frame):
        return DepthFrame(np.dstack((bgr_frame, np.expand_dims(depth_frame, axis=-1))))


class DepthCamera():
    def __init__(self):
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
        self.profile = self.pipeline.start(config)

        self.depth_sensor = self.profile.get_device().query_sensors()[1]
        self.depth_sensor.set_option(rs.option.enable_auto_exposure, False)
        # self.autoexpose_counter = 0

        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
        # if self.autoexpose_counter == 15:
        #     self.depth_sensor.set_option(rs.option.enable_auto_exposure, False)
        # self.autoexpose_counter += 1

        frame = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frame)

        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        output = DepthFrame(np.dstack((color_image, np.expand_dims(depth_image, axis=-1))))
        return output


    def __del__(self):
        self.pipeline.stop()


if __name__ == "__main__":
    depth_camera = DepthCamera()
    prev_frame = None
    while True:
        frame = depth_camera.get_frame()
        # if prev_frame is not None:
        #     # cv2 render difference
        #     # diff = cv2.absdiff(frame.extract_rgb_frame(), prev_frame.extract_rgb_frame())
        #     diff = cv2.absdiff(frame.extract_depth_frame(), prev_frame.extract_depth_frame())
        #     cv2.imshow("diff", diff / 1000)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        # save image
        # cv2.imwrite("rgb_frame.png", frame.extract_rgb_frame())
        # depth_frame = frame.extract_depth_frame()
        # # save depth frame as heatmap
        # depth_frame_normalized = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX)
        # depth_frame_colored = cv2.applyColorMap(depth_frame_normalized.astype(np.uint8), cv2.COLORMAP_JET)
        # cv2.imwrite("depth_frame_colored.png", depth_frame_colored)
        # cv2.imwrite("depth_frame.png", frame.extract_depth_frame())


        # render rgb frame
        cv2.imshow("rgb", frame.extract_rgb_frame())
        # cv2.imshow("depth", frame.extract_depth_frame() / 1000)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_frame = frame
        # frame.render_frame()