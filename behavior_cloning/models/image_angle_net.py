import torch
import torch.nn as nn

import os
from behavior_cloning.configs.path_config import TASK_DESCRIPTION_CACHE_PATH
from behavior_cloning.configs.nn_config import LANGUAGE_INPUT_DIM


class ImageAngleNet(nn.Module):
    def __init__(self, backbone_depth, backbone_wrist, backbone_angle, state_neck, language_neck, language_stem, configs):
        super(ImageAngleNet, self).__init__()
        self.backbone_depth = backbone_depth
        self.backbone_wrist = backbone_wrist
        self.backbone_angle = backbone_angle
        self.state_neck = state_neck
        self.language_neck = language_neck
        self.language_stem = language_stem
        
        # self.language_embeddings = nn.Embedding(len(os.listdir(TASK_DESCRIPTION_CACHE_PATH)), LANGUAGE_INPUT_DIM)

        self.heads = nn.ModuleDict()
        for config in configs:
            self.heads[config.name] = config.head

    def forward(self, depth_frames, wrist_frames, angles, task_description_embedding):
        depth_features = self.backbone_depth(depth_frames)
        wrist_features = self.backbone_wrist(wrist_frames)
        angle_features = self.backbone_angle(angles)

        # task_description_embedding = self.language_embeddings(task_description_embedding)
        task_features = self.language_stem(task_description_embedding)

        state_features = wrist_features
        state_features = self.state_neck(state_features)

        features_with_task_description = torch.cat((depth_features, wrist_features, angle_features, task_features), dim=1)
        features_with_task_description = self.language_neck(features_with_task_description)
        
        output = {}
        for head_name, head in self.heads.items():
            if head.use_task_description:
                output[head_name] = head(features_with_task_description)
            else:
                output[head_name] = head(state_features)

        return output

    def print_num_params(self):
        num_params_depth = sum(p.numel() for p in self.backbone_depth.parameters())
        num_params_wrist = sum(p.numel() for p in self.backbone_wrist.parameters())
        num_params_angle = sum(p.numel() for p in self.backbone_angle.parameters())
        num_params_neck = sum(p.numel() for p in self.state_neck.parameters())
        num_head_params = sum(p.numel() for p in self.heads.parameters())
        print(f"Number of parameters in depth backbone: {num_params_depth}")
        print(f"Number of parameters in wrist backbone: {num_params_wrist}")
        print(f"Number of parameters in angle backbone: {num_params_angle}")
        print(f"Number of parameters in neck: {num_params_neck}")
        print(f"Number of parameters in heads: {num_head_params}")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
