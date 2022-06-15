import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import kornia as K


class kBrightness(nn.Module):
    def __init__(self, init_param):
        super(kBrightness, self).__init__()
        self.trans_param = Parameter(torch.Tensor([init_param]))

    def forward(self, x):
        clamped_param = torch.clamp(self.trans_param, min=0., max=1.)
        transform = K.enhance.AdjustBrightness(brightness_factor=clamped_param)
        out = transform(x)
        return out
    
    def get_param_val(self):
        return self.trans_param.item()