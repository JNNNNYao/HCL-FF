import torch.nn as nn
import torch.nn.functional as F


class ProjHead(nn.Module):
    def __init__(self, in_dim, out_dim=128, use_bn=True):
        super().__init__()
        layers = [nn.Linear(in_dim, out_dim, bias=False)]
        if use_bn: layers += [nn.BatchNorm1d(out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, feat):
        z = F.normalize(self.mlp(feat), dim=1)
        return z