import torch
import torch.nn as nn
import torch.nn.functional as F


class CWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, num_class=10, dropout=0):
        super(CWConv, self).__init__()
        assert out_channels % num_class == 0
        self.num_class = num_class

        if out_channels <= 100:
            self.pool_size = 4
        elif out_channels <= 400:
            self.pool_size = 2
        else:
            self.pool_size = 1

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.relu = nn.ReLU(inplace=True)

        torch.nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, no_norm=False):
        x = x.detach()

        y = self.conv(x)
        y = self.relu(y)
        y = self.dropout(y)

        # classified score
        g = y.view(y.size(0), self.num_class, -1)
        g = g.mean(dim=2)
        
        if not no_norm:
            # normalize feature y
            y = F.group_norm(y, self.num_class)
        
        feat = F.adaptive_avg_pool2d(y, (self.pool_size, self.pool_size)).view(x.size(0), -1)
        return y, g, feat