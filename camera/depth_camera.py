import pyrealsense2 as rs
import numpy as np
import matplotlib.pyplot as plt


class DepthFrame():
    def __init__(self, frame):
        self.frame = frame
    
    def extract_bgr_frame(self):
        return self.frame[:, :, :3].astype(np.uint8)
    
    def extract_rgb_frame(self):
        return self.frame[:, :, :3][:, :, ::-1].astype(np.uint8)
    
    def extract_depth_frame(self):
        return self.frame[:, :, -1].astype(np.uint16)

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
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

        align_to = rs.stream.color
        self.align = rs.align(align_to)

        # ctx = rs.context()
        # devices = ctx.query_devices()
        # for dev in devices:
        #     dev.hardware_reset()


    def get_frame(self):
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
    while True:
        frame = depth_camera.get_frame()
        frame.render_frame()