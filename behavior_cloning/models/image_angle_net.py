import torch
import torch.nn as nn


class ImageAngleNet(nn.Module):
    def __init__(self, backbone_depth, backbone_wrist, backbone_angle, neck, configs):
        super(ImageAngleNet, self).__init__()
        self.backbone_depth = backbone_depth
        self.backbone_wrist = backbone_wrist
        self.backbone_angle = backbone_angle
        self.neck = neck

        self.heads = nn.ModuleDict()
        for config in configs:
            self.heads[config.name] = config.head

    def forward(self, depth_frames, wrist_frames, angles, task_description_embedding):
        depth_features = self.backbone_depth(depth_frames)
        wrist_features = self.backbone_wrist(wrist_frames)
        angle_features = self.backbone_angle(angles)

        features = torch.cat((depth_features, wrist_features, angle_features), dim=1)
        features = self.neck(features)

        features_with_task_description = torch.cat((features, task_description_embedding), dim=1)

        output = {}
        for head_name, head in self.heads.items():
            if head.use_task_description:
                output[head_name] = head(features_with_task_description)
            else:
                output[head_name] = head(features)

        return output
    
    def print_num_params(self):
        num_params_depth = sum(p.numel() for p in self.backbone_depth.parameters())
        num_params_wrist = sum(p.numel() for p in self.backbone_wrist.parameters())
        num_params_angle = sum(p.numel() for p in self.backbone_angle.parameters())
        num_params_neck = sum(p.numel() for p in self.neck.parameters())
        num_head_params = sum(p.numel() for p in self.heads.parameters())
        print(f"Number of parameters in depth backbone: {num_params_depth}")
        print(f"Number of parameters in wrist backbone: {num_params_wrist}")
        print(f"Number of parameters in angle backbone: {num_params_angle}")
        print(f"Number of parameters in neck: {num_params_neck}")
        print(f"Number of parameters in heads: {num_head_params}")
        print(f"Total number of parameters: {sum(p.numel() for p in self.parameters())}")
