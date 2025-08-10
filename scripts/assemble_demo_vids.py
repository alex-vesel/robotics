import os
import cv2
import numpy as np

VIDEO_DIR = "./run_data"

# for every folder in VIDEO DIR 
# assemble video of 
# VIDEO_DIR/folder/rgb_frames
# VIDEO_DIR/folder/wrist_frames
# VIDEO_DIR/folder/depth_frames

for folder in os.listdir(VIDEO_DIR):
    folder_path = os.path.join(VIDEO_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    rgb_frames_path = os.path.join(folder_path, "rgb_frames")
    wrist_frames_path = os.path.join(folder_path, "wrist_frames")
    depth_frames_path = os.path.join(folder_path, "depth_frames")

    if not os.path.exists(rgb_frames_path) or not os.path.exists(wrist_frames_path) or not os.path.exists(depth_frames_path):
        print(f"Missing frames in {folder}")
        continue

    # assemble video

    frame_names = sorted(os.listdir(rgb_frames_path), key=lambda x: float(".".join(x.split('.')[:-1])))
    rgb_frame_names = [os.path.join(rgb_frames_path, frame) for frame in frame_names]
    wrist_frame_names = [os.path.join(wrist_frames_path, frame) for frame in frame_names]
    depth_frame_names = [os.path.join(depth_frames_path, frame) for frame in frame_names]
    # make sure all three lists are the same length
    assert len(rgb_frame_names) == len(wrist_frame_names) == len(depth_frame_names), f"Frame lists are not the same length in {folder}"
    
    # create video
    os.makedirs(os.path.join(folder_path, "videos"), exist_ok=True)
    rgb_video_path = os.path.join(folder_path, "videos", "rgb.mp4")
    wrist_video_path = os.path.join(folder_path, "videos", "wrist.mp4")
    depth_video_path = os.path.join(folder_path, "videos", "depth.mp4")
    # create video writer
    rgb_video = cv2.VideoWriter(rgb_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    wrist_video = cv2.VideoWriter(wrist_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    depth_video = cv2.VideoWriter(depth_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    # loop through frames
    
    for frame_name, wrist_frame_name, depth_frame_name in zip(rgb_frame_names, wrist_frame_names, depth_frame_names):
        # read frames
        rgb_frame = cv2.imread(frame_name)
        wrist_frame = cv2.imread(wrist_frame_name)
        depth_frame = cv2.imread(depth_frame_name, cv2.IMREAD_ANYDEPTH)
        # resize frames
        rgb_frame = cv2.resize(rgb_frame, (640, 480))
        wrist_frame = cv2.resize(wrist_frame, (640, 480))
        wrist_frame = cv2.rotate(wrist_frame, cv2.ROTATE_180)
        depth_frame = cv2.resize(depth_frame, (640, 480))
        # convert depth frame to 8 bit

        depth_frame = np.clip(depth_frame, 0, 1536)
        depth_frame = (depth_frame / 1536 * 255).astype(np.uint8)
        # apply color map
        depth_frame = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)

        
        # write frames to video
        rgb_video.write(rgb_frame)
        wrist_video.write(wrist_frame)
        depth_video.write(depth_frame)
    # release video writers
    rgb_video.release()
    wrist_video.release()
    depth_video.release()
    print(f"Created videos in {folder}")