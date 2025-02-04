import torch
import torch.nn as nn

class BinaryClassificationHead(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(BinaryClassificationHead, self).__init__()
        self.in_features = in_features
        self.fc = nn.Linear(in_features, out_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, inference=False):
        x = self.fc(x)
        if inference:
            x = self.sigmoid(x)
        return x


class TanhRegressionHead(nn.Module):
    def __init__(self, in_features, out_features=1):
        super(TanhRegressionHead, self).__init__()
        self.in_features = in_features
        self.fc = nn.Linear(in_features, out_features)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc(x)
        x = self.tanh(x)
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