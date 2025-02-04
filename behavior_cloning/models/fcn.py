import torch
import torch.nn as nn


class FullyConnectedNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(FullyConnectedNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers - 1)],
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)