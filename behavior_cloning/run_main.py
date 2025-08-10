from pathlib import Path
import sys
import time
import copy
import hashlib
import json
from time import monotonic
from time import sleep
import os
import pygame
sys.path.append(".")

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset
from pymycobot.mycobot280 import MyCobot280
from sentence_transformers import SentenceTransformer

from behavior_cloning.utils.custom_transforms import *
from behavior_cloning.utils.train_configs import TrainConfigModule, process_delta_angle
from behavior_cloning.utils.train import train_model
from behavior_cloning.models.resnet import resnet18, resnet10, simplecnn
from behavior_cloning.models.image_angle_net import ImageAngleNet
from shared_utils.loggers.data_logger import DataLogger
from behavior_cloning.models.fcn import FullyConnectedNet
from behavior_cloning.models.heads import BinaryClassificationHead, TanhRegressionHead, GaussianHead
from behavior_cloning.configs.path_config import *
from behavior_cloning.configs.nn_config import *

from camera.depth_camera import DepthCamera
from camera.camera import Camera
from camera.camera_thread import CameraThread
from teleop.teleop_constants import INITIAL_ANGLES, ZERO_ANGLES, SAVED_ANGLES_DIR


np.random.seed(0)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else device)

# open task_description_embedding_dict.json
with open('task_description_embedding_dict.json', 'r') as f:
    task_description_embedding_dict = json.load(f)

# device = 'cpu'

# TASK_DESCRIPTION = "Place the earplug on the container and then return home."
TASK_DESCRIPTION = "Pick up the earplug and return home."
# TASK_DESCRIPTION = "Place the smiski in the bowl and then return home."
# TASK_DESCRIPTION = "Place the gold chocolate in the bowl and then return home."
# TASK_DESCRIPTION = "Place the earplug in the bowl and then return home."
# TASK_DESCRIPTION = "Pick up the object from the white bowl and return home."
# TASK_DESCRIPTION = "Place the smiski on the container and then return home."
# TASK_DESCRIPTION = "Pick up the gold wrapped chocolate and place it in the container and then return home."
# TASK_DESCRIPTION = "Pick up the white wrapped chocolate and return home."

def main(mc, depth_camera, wrist_camera):
    ## Get text embedding
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    task_description_embedding = model.encode(TASK_DESCRIPTION).astype(np.float32, copy=False)
    task_description_embedding = torch.tensor(task_description_embedding).to(device).unsqueeze(0)

    # task hash
    # task_hash = hashlib.md5(TASK_DESCRIPTION.encode()).hexdigest()
    # if task_hash in task_description_embedding_dict:
    #     task_description_embedding = torch.tensor(task_description_embedding_dict[task_hash]).to(device).unsqueeze(0)

    del model

    ## Define transforms
    frame_transform_list = [
        ResizeImage(mode='half'),
        ToFloat(),
        ResizeImage((224, 224)),
        MoveAxis(2, 0),
        DivideByScalar(255., axis=0, channels=[0, 1, 2]),
        DivideByScalar(4096., axis=0, channels=[3]),
        NormalizeToRange(0, 1, -1, 1),
        ClampToMax(1),
    ]
    composed_depth_frame_transform = Compose(frame_transform_list)

    frame_transform_list = [
        ResizeImage(mode='half'),
        ToFloat(),
        ResizeImage((224, 224)),
        MoveAxis(2, 0),
        DivideByScalar(255., axis=0, channels=[0, 1, 2]),
        NormalizeToRange(0, 1, -1, 1),
        ClampToMax(1),
    ]
    composed_wrist_frame_transform = Compose(frame_transform_list)

    ## Define model
    backbone_depth, latent_dim_depth = resnet10(num_channels=4, avg_pool_shape=(2, 2), last_activation='none')
    backbone_wrist, latent_dim_wrist = resnet10(num_channels=3, avg_pool_shape=(1, 1), last_activation='none')
    latent_dim_wrist = 3136*2
    latent_dim_depth = 3136*2
    backbone_angle = FullyConnectedNet(
        input_dim=ANGLE_FC_INPUT_DIM,
        hidden_dim=ANGLE_FC_HIDDEN_DIM,
        num_layers=ANGLE_FC_NUM_LAYERS,
        output_dim=ANGLE_FC_OUTPUT_DIM,
        final_batchnorm=True,
    )
    print("Neck input dim:", latent_dim_depth+latent_dim_wrist+ANGLE_FC_OUTPUT_DIM)
    state_neck = FullyConnectedNet(
        input_dim=latent_dim_wrist,
        hidden_dim=OBJECT_NECK_HIDDEN_DIM,
        num_layers=OBJECT_NUM_LAYERS,
        output_dim=OBJECT_OUTPUT_DIM,
    )
    language_neck = FullyConnectedNet(
        input_dim=latent_dim_depth+latent_dim_wrist+ANGLE_FC_OUTPUT_DIM+LANGUAGE_HIDDEN_DIM,
        hidden_dim=NECK_HIDDEN_DIM,
        num_layers=NECK_NUM_LAYERS,
        output_dim=NECK_OUTPUT_DIM,
    )
    language_stem = FullyConnectedNet(
        input_dim=LANGUAGE_INPUT_DIM,
        hidden_dim=LANGUAGE_HIDDEN_DIM,
        num_layers=LANGUAGE_STEM_NUM_LAYERS,
        output_dim=LANGUAGE_HIDDEN_DIM,
        final_batchnorm=True,
    )

    configs = [
        TrainConfigModule(
            name='delta_angle_regression',
            loss_fn=torch.nn.MSELoss(reduction='none'),
            process_gnd_truth_fn=process_delta_angle,
            head=TanhRegressionHead(NECK_OUTPUT_DIM, 7, chunk_size=CHUNK_SIZE, use_task_description=True),
            type='regression',
            gt_key='delta_angle',
            mask=[
                lambda x: x['gripper_has_object_mask'] == 0,
            ],
            group_by='task_name',
        ),
        TrainConfigModule(
            name='gripper_has_object_classification',
            loss_fn=torch.nn.BCEWithLogitsLoss(reduction='none'),
            process_gnd_truth_fn=lambda x: x.float(),
            head=BinaryClassificationHead(OBJECT_OUTPUT_DIM, 1, use_task_description=False),
            type='classification',
            gt_key='gripper_has_object',
            mask=[
                lambda x: x['gripper_has_object_mask'] == 1,
            ],
        ),
    ]

    model = ImageAngleNet(backbone_depth, backbone_wrist, backbone_angle, state_neck, language_neck, language_stem, configs).to(device)

    # Reload model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'], strict=True)

    model = model.to(device)
    model.print_num_params()
    model.eval()

    angles = ZERO_ANGLES.copy()
    angles.append(0)
    angles = np.array(angles)

    experiment_logger = DataLogger(os.path.join("run_data", "object_high_loss"))
    start_time = monotonic()

    index = 0
    prev_angle_deltas = []
    alpha = 0.4  # Exponential averaging factor

    while True:
        start = time.time()
        depth_frame_raw = depth_camera.get_frame()
        depth_frame_swap = copy.deepcopy(depth_frame_raw)
        depth_frame_swap.swap_rgb()
        depth_frame = composed_depth_frame_transform(depth_frame_swap.get_frame())

        wrist_frame_raw = wrist_camera.get_frame()
        wrist_frame = composed_wrist_frame_transform(wrist_frame_raw.extract_rgb_frame())
        # print(time.time() - start)

        # depth_frame = depth_frame[:3]
        # start = time.time()
        with torch.no_grad():
            output = model(
                torch.tensor([depth_frame]).to(device), 
                torch.tensor([wrist_frame]).to(device), 
                torch.tensor([angles]).float().to(device) / 100, 
                task_description_embedding
            )

        # print(time.time() - start)

        angle_deltas = output['delta_angle_regression'][0].detach().cpu().numpy()
        angle_deltas[:, :6] *= 4
        angle_deltas[:, 6] *= 6

        # RUN ALL ANGLE DELTA
        # for i in range(2):
        #     old_angles = angles.copy()
        #     angles[:6] = angles[:6] + angle_deltas[i, :6]
        #     angles[6] = int(np.clip(angles[6] + angle_deltas[i, 6], 0, 90))
        #     mc.send_angles(list(angles)[:6], 60)
        #     sleep(0.1)
        #     if abs(angles[6] - old_angles[6]) < 1:
        #         pass
        #     else:
        #         sleep(0.03)
        #         mc.set_gripper_value(int(angles[6]), 40)

        # AGGREGATE ANGLE DELTA
        prev_angle_deltas.append(angle_deltas.copy())
        if len(prev_angle_deltas) > 5:
            prev_angle_deltas.pop(0)

        weight = 1.0
        total_weight = 0.0
        angle_delta = np.zeros(7)
        for i, delta in enumerate(reversed(prev_angle_deltas)):
            angle_delta += delta[i] * weight
            total_weight += weight
            weight *= alpha
        angle_delta /= total_weight

        old_angles = angles.copy()
        angles = angles + angle_delta

        angles[6] = int(np.clip(angles[6], 0, 90))

        mc.send_angles(list(angles)[:6], 60)
        if abs(angles[6] - old_angles[6]) < 1:
            pass
        else:
            sleep(0.03)
            mc.set_gripper_value(int(angles[6]), 40)

        # print(angle_delta)
        print("Gripper has object p: ", float(F.sigmoid(output['gripper_has_object_classification']).detach().cpu().numpy()))

        # if float(F.sigmoid(output['gripper_has_object_classification']).detach().cpu().numpy()) > 0.2:
        # experiment_logger.log(monotonic() - start_time, old_angles, angle_delta, depth_frame_raw, wrist_frame_raw, {'gripper_has_object': False, "task_name": "object_classification"})

        # wait for key press
        # while True:
        #     done = False
        #     for event in pygame.event.get():
        #         if event.type == pygame.KEYDOWN:
        #             if event.key == pygame.K_q:
        #                 return
        #             elif event.key == pygame.K_s:
        #                 print("Saving angles")
        # if index % 3 == 0:
        #     np.save(os.path.join(SAVED_ANGLES_DIR, f'{int(time.time())}_{index}.npy'), angles)

        index += 1
        #             done = True
        #     if done:
        #         break


if __name__ == '__main__':
    pygame.init()

    mc = MyCobot280('/dev/tty.usbserial-588D0018121',115200)
    depth_camera = CameraThread(DepthCamera())
    wrist_camera = CameraThread(Camera())
    depth_camera.start()
    wrist_camera.start()

    mc.sync_send_angles(ZERO_ANGLES, 40, timeout=3)
    mc.set_gripper_value(0, 40)
    mc.set_fresh_mode(1)

    os.makedirs(SAVED_ANGLES_DIR, exist_ok=True)

    main(mc, depth_camera, wrist_camera)