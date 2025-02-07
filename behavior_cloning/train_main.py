from pathlib import Path
import sys
sys.path.append(".")

import torch
from torchvision.transforms import Compose
from torch.utils.data import DataLoader, ConcatDataset

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

## Get clip list
def main():
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
        ColorJitter(p=0.7),
        # ImageShift(p=1.0, padding=10),
        # RandomCrop(),
    ])

    angle_augmentations = Compose([
        # RandomJitter(p=1.0, jitter=0.2),
    ])


    ## Aggregate data
    train_datafiles = aggregate_data(
        clip_paths=train_clips,
        depth_frame_transform=composed_depth_frame_transform,
        wrist_frame_transform=composed_wrist_frame_transform,
        angle_transform=composed_angle_transform,
        augmentations=train_augmentations,
        angle_augmentations=angle_augmentations,
        num_workers=1,
    )

    val_datafiles = aggregate_data(
        clip_paths=val_clips,
        depth_frame_transform=composed_depth_frame_transform,
        wrist_frame_transform=composed_wrist_frame_transform,
        angle_transform=composed_angle_transform,
        augmentations=None,
        num_workers=1,
    )

    ## Create dataloaders
    train_loader = DataLoader(
        ConcatDataset(train_datafiles),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True,
    )

    val_loader = DataLoader(
        ConcatDataset(val_datafiles),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    print(f"Total train clips: {len(train_clips)}, total train frames: {len(train_loader.dataset)}")
    print(f"Total val clips: {len(val_clips)}, total val frames: {len(val_loader.dataset)}")

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
    angle_neck = FullyConnectedNet(
        input_dim=latent_dim_depth+latent_dim_wrist,
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
            gt_key='delta_angle',
            mask=[
                lambda x: x['gripper_has_object_mask'] == 0,
            ],
        ),
        # TrainConfigModule(
        #     name='angle_regression',
        #     loss_fn=torch.nn.MSELoss(reduction='none'),
        #     process_gnd_truth_fn=lambda x: x,
        #     head=TanhRegressionHead(NECK_OUTPUT_DIM, 7),
        #     type='regression',
        #     gt_key='angle',
        #     mask=None,
        # ),
        TrainConfigModule(
            name='gripper_has_object_classification',
            loss_fn=torch.nn.BCEWithLogitsLoss(reduction='none'),
            process_gnd_truth_fn=lambda x: x.float(),
            head=BinaryClassificationHead(NECK_OUTPUT_DIM, 1),
            type='classification',
            gt_key='gripper_has_object',
            mask=[
                lambda x: x['gripper_has_object_mask'] == 1,
            ]
        ),
    ]

    model = ImageAngleNet(backbone_depth, backbone_wrist, backbone_angle, neck, angle_neck, configs).to(device)

    # Reload model
    if RELOAD_MODEL:
        print(f"Reloading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'], strict=False)
                   
    model = model.to(device)
    model.print_num_params()

    ## Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if RELOAD_MODEL:
        optimizer.load_state_dict(torch.load(MODEL_PATH)['optimizer_state_dict'])

    ## Train model
    logger = ExperimentLogger(logdir=EXPERIMENT_RESULTS_DIR)

    train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        configs=configs,
        optimizer=optimizer,
        logger=logger,
        num_epochs=TRAIN_EPOCHS,
        log_steps=LOG_STEPS,
        device=device,
)

if __name__ == '__main__':
    main()