from pathlib import Path
import sys
sys.path.append(".")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset
from sentence_transformers import SentenceTransformer

from behavior_cloning.utils.custom_transforms import *
from behavior_cloning.utils.datafile import aggregate_data, split_data
from shared_utils.path_utils import recurse_dir_for_clips
from behavior_cloning.utils.train_configs import TrainConfigModule, process_delta_angle
from behavior_cloning.utils.train import train_model
from shared_utils.loggers.experiment_logger import ExperimentLogger
from behavior_cloning.models.resnet import resnet18, resnet10, simplecnn
from behavior_cloning.models.image_angle_net import ImageAngleNet
from behavior_cloning.models.fcn import FullyConnectedNet
from behavior_cloning.models.heads import BinaryClassificationHead, TanhRegressionHead, GaussianHead
from behavior_cloning.configs.path_config import *
from behavior_cloning.configs.nn_config import *


np.random.seed(0)

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else device)

DATA_DIR = "./run_data"

# TASK_DESCRIPTION = "Place the earplug on the container and then return home."
TASK_DESCRIPTION = "Pick up the earplug and return home."

## Get clip list
def main():
    model = SentenceTransformer('BAAI/bge-base-en-v1.5')
    task_description_embedding = model.encode(TASK_DESCRIPTION).astype(np.float32, copy=False)
    task_description_embedding = torch.tensor(task_description_embedding).to(device).unsqueeze(0)
    del model

    data_path = Path(DATA_DIR)
    clip_paths = recurse_dir_for_clips(data_path, match='rgb_frames')
    
    dedicated_train_clips = [clip for clip in clip_paths if 'train' in clip]
    dedicated_val_clips = [clip for clip in clip_paths if 'val' in clip]
    clip_paths = [clip for clip in clip_paths if 'train' not in clip]
    clip_paths = [clip for clip in clip_paths if 'val' not in clip]

    train_clips, val_clips, test_clips = split_data(clip_paths)
    
    train_clips.extend(dedicated_train_clips)
    val_clips.extend(dedicated_val_clips)
    # remove blacklist
    # train_clips = [clip for clip in train_clips if not any([blacklist_file in clip for blacklist_file in BLACKLIST_FILES])]
    val_clips = train_clips

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

    composed_angle_transform = {
        'target_x_percent': Compose([ToFloat(pytorch=False), NormalizeToRange(0, 1, -1, 1)]),
        'target_y_percent': Compose([ToFloat(pytorch=False), NormalizeToRange(0, 1, -1, 1)]),
    }

    ## Create augmentations
    train_augmentations = Compose([
        # VerticalFlip(p=TRAIN_AUGMENTATION_PROB["prob_vertical_flip"]),
        # HorizontalFlip(p=TRAIN_AUGMENTATION_PROB["prob_horizontal_flip"]),
        # ColorJitter(p=0.7),
        # RandomCrop(),
    ])

    angle_augmentations = Compose([
        # RandomJitter(p=1.0, jitter=0.1),
    ])


    ## Aggregate data
    val_datafiles = aggregate_data(
        clip_paths=val_clips,
        depth_frame_transform=composed_depth_frame_transform,
        wrist_frame_transform=composed_wrist_frame_transform,
        angle_transform=composed_angle_transform,
        augmentations=None,
        num_workers=1,
    )

    ## Create dataloaders
    val_loader = DataLoader(
        ConcatDataset(val_datafiles),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Total val clips: {len(val_clips)}, total val frames: {len(val_loader.dataset)}")

    ## Define model
    backbone_depth, latent_dim_depth = resnet10(num_channels=4, avg_pool_shape=(1, 1))
    backbone_wrist, latent_dim_wrist = resnet10(num_channels=3, avg_pool_shape=(1, 1))
    backbone_angle = FullyConnectedNet(
        input_dim=ANGLE_FC_INPUT_DIM,
        hidden_dim=ANGLE_FC_HIDDEN_DIM,
        num_layers=ANGLE_FC_NUM_LAYERS,
        output_dim=ANGLE_FC_OUTPUT_DIM,
    )
    print("Neck input dim:", latent_dim_depth+latent_dim_wrist+ANGLE_FC_OUTPUT_DIM)
    state_neck = FullyConnectedNet(
        input_dim=latent_dim_depth+latent_dim_wrist,
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

    model = ImageAngleNet(backbone_depth, backbone_wrist, backbone_angle, state_neck, language_neck, configs).to(device)

    # Reload model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'], strict=True)
                                    
    model = model.to(device)
    model.print_num_params()

    model.eval()

    for datafile in val_datafiles:
        outputs = []
        print(datafile.clip_name)
        for i in range(len(datafile)):
            batch = datafile[i]
            for key in batch:
                try:
                    batch[key] = torch.tensor(batch[key]).to(device)
                except:
                    continue

            output = model(batch['depth_frame'].unsqueeze(0), batch['wrist_frame'].unsqueeze(0), batch['angle'].unsqueeze(0), task_description_embedding)
            # print(batch['angle']*100)
            # print(output['delta_angle_regression'][0]*6)
            # outputs.append(output['delta_angle_regression'][0]*6)
            # import IPython; IPython.embed(); exit(0)
            # angle_delta = output['delta_angle_regression'][0]
            # angle_delta[:6] = angle_delta[:6] * 4
            # angle_delta[6] = angle_delta[6] * 6
            # outputs.append(angle_delta)
            print("Gripper has object p: ", float(F.sigmoid(output['gripper_has_object_classification']).detach().cpu().numpy()))

            # mse = torch.mean((output['delta_angle_regression'][0] - batch['delta_angle'] / 30)**2)
            # print(mse)

        # exit()
        # get std
        outputs = torch.stack(outputs)
        print(outputs.mean(dim=0))
        print(outputs.std(dim=0))

    # 55794 mean: -0.0115, -0.2965 std: tensor([0.0884, 0.0944, 0.0664, 0.0647, 0.0490, 0.0434, 0.0998]
    # 55794_1 mean: 0.0062, -1.1597 std:tensor([0.0917, 0.1457, 0.0696, 0.0646, 0.0423, 0.0469, 0.1055]
    # 75326 mean: -0.6711, -0.3503 std: tensor([0.1454, 0.2085, 0.0753, 0.0654, 0.0698, 0.0954, 0.1589]
    # 113679 mean: tensor([-0.0044, -1.7394, -0.0913, -0.0759, -0.1218,  0.0166, -0.0844],
    #        std:    tensor([0.1520, 0.1807, 0.0392, 0.0361, 0.0756, 0.0735, 0.0814]
    # 113679_aug mean: tensor([-0.0778, -1.5708, -0.0282, -0.0139,  0.1323, -0.1585,  0.1313]
    #                std tensor([0.1036, 0.1293, 0.0308, 0.0411, 0.0613, 0.0570, 0.1226]
        # no autoexpose tensor([ 0.4023, -1.6118,  0.0245, -0.0289,  0.0271,  0.0497, -0.0495],
        # tensor([0.0609, 0.1000, 0.0254, 0.0226, 0.0239, 0.0334, 0.0679],

if __name__ == '__main__':
    main()