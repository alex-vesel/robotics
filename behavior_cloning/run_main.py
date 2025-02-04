from pathlib import Path
import sys
import time
import os
import pygame
sys.path.append(".")

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset
from pymycobot.mycobot280 import MyCobot280

from behavior_cloning.utils.custom_transforms import *
from behavior_cloning.utils.train_configs import TrainConfigModule, process_delta_angle
from behavior_cloning.utils.train import train_model
from behavior_cloning.models.resnet import resnet18, resnet10, simplecnn
from behavior_cloning.models.image_angle_net import ImageAngleNet
from behavior_cloning.models.fcn import FullyConnectedNet
from behavior_cloning.models.heads import BinaryClassificationHead, TanhRegressionHead, GaussianHead
from behavior_cloning.configs.path_config import *
from behavior_cloning.configs.nn_config import *

from camera.depth_camera import DepthCamera
from camera.camera import Camera
from teleop.teleop_constants import INITIAL_ANGLES, ZERO_ANGLES, SAVED_ANGLES_DIR


np.random.seed(0)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else device)


def main(mc, depth_camera, wrist_camera):
    ## Define transforms
    frame_transform_list = [
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
        ToFloat(),
        ResizeImage((224, 224)),
        MoveAxis(2, 0),
        DivideByScalar(255., axis=0, channels=[0, 1, 2]),
        NormalizeToRange(0, 1, -1, 1),
        ClampToMax(1),
    ]
    composed_wrist_frame_transform = Compose(frame_transform_list)

    ## Define model
    backbone_depth, latent_dim_depth = resnet10(num_channels=4)
    backbone_wrist, latent_dim_wrist = resnet10(num_channels=3)
    backbone_angle = FullyConnectedNet(
        input_dim=ANGLE_FC_INPUT_DIM,
        hidden_dim=ANGLE_FC_HIDDEN_DIM,
        num_layers=ANGLE_FC_NUM_LAYERS,
        output_dim=ANGLE_FC_OUTPUT_DIM,
    )
    print("Neck input dim:", latent_dim_depth+latent_dim_wrist+ANGLE_FC_OUTPUT_DIM)
    neck = FullyConnectedNet(
        input_dim=latent_dim_depth+latent_dim_wrist+ANGLE_FC_OUTPUT_DIM,
        hidden_dim=NECK_HIDDEN_DIM,
        num_layers=NECK_NUM_LAYERS,
        output_dim=NECK_OUTPUT_DIM,
    )

    configs = [
        TrainConfigModule(
            name='delta_angle_regression',
            loss_fn=torch.nn.MSELoss(reduction='none'),
            process_gnd_truth_fn=process_delta_angle,
            head=TanhRegressionHead(NECK_OUTPUT_DIM, 7),
            type='regression',
            mask=None,
        ),
        # TrainConfigModule(
        #     name='motor_activation',
        #     loss_fn=torch.nn.BCEWithLogitsLoss(reduction='none'),
        #     process_gnd_truth_fn=lambda x: (x!=0).float(),
        #     head=BinaryClassificationHead(NECK_OUTPUT_DIM, 7),
        #     type='classification',
        #     mask=None,
        # ),
    ]

    model = ImageAngleNet(backbone_depth, backbone_wrist, backbone_angle, neck, configs).to(device)

    # Reload model
    model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'], strict=True)

    model = model.to(device)
    model.print_num_params()
    model.eval()

    angles = ZERO_ANGLES.copy()
    angles.append(0)

    while True:
        depth_frame = depth_camera.get_frame().get_frame()
        depth_frame = composed_depth_frame_transform(depth_frame)

        wrist_frame = wrist_camera.get_frame().get_frame()
        wrist_frame = composed_wrist_frame_transform(wrist_frame)

        output = model(torch.tensor([depth_frame]).to(device), torch.tensor([wrist_frame]).to(device), torch.tensor([angles]).to(device).float() / 100)

        angle_delta = output['delta_angle_regression'][0].detach().cpu().numpy() * 30
    
        # motor_activation_prob = F.sigmoid(output['motor_activation'][0]).detach().cpu().numpy()
        # motor_activation_mask = motor_activation_prob > 0.5
        # angle_delta = angle_delta * motor_activation_mask

        # print(motor_activation_prob)

        angles = np.array(angles) + angle_delta

        angles[6] = int(np.clip(angles[6], 0, 90))

        mc.sync_send_angles(list(angles)[:6], 40, timeout=3)
        mc.set_gripper_value(int(angles[6]), 40)

        print(angle_delta)

        # wait for key press
        while True:
            done = False
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return
                    elif event.key == pygame.K_s:
                        print("Saving angles")
                        np.save(os.path.join(SAVED_ANGLES_DIR, f'{int(time.time())}.npy'), angles)
                    done = True
            if done:
                break


if __name__ == '__main__':
    pygame.init()

    mc = MyCobot280('/dev/tty.usbmodem588D0018121',115200)
    depth_camera = DepthCamera()
    wrist_camera = Camera()

    mc.sync_send_angles(ZERO_ANGLES, 40, timeout=3)
    mc.set_fresh_mode(1)

    os.makedirs(SAVED_ANGLES_DIR, exist_ok=True)

    main(mc, depth_camera, wrist_camera)