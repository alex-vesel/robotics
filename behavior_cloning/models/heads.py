import torch
import torch.nn as nn

class BinaryClassificationHead(nn.Module):
    def __init__(self, in_features, out_features=1, use_task_description=True):
        super(BinaryClassificationHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_task_description = use_task_description
        self.fc = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, inference=False):
        x = self.fc(x)
        if inference:
            x = self.sigmoid(x)
        return x


class TanhRegressionHead(nn.Module):
    def __init__(self, in_features, out_features=1, chunk_size=1, use_task_description=True):
        super(TanhRegressionHead, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.chunk_size = chunk_size
        self.use_task_description = use_task_description
        self.fc = nn.Linear(in_features, out_features*chunk_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.tanh(x)
        x = x.view(-1, self.chunk_size, self.out_features)
        return x
    

class LinearRegressionHead(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(LinearRegressionHead, self).__init__()
        self.in_features = in_features
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.fc(x)
        return x
    
class GaussianHead(nn.Module):
    def __init__(self, in_features, out_features=3):
        super(GaussianHead, self).__init__()
        self.in_features = in_features
        self.fc = nn.Linear(in_features, out_features)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.fc(x)
        means = self.tanh(x[:, :2])
        variances = self.softplus(x[:, [2]])
        return torch.cat((means, variances), dim=1)