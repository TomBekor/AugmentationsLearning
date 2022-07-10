import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import kornia as K

class kContrast(nn.Module):
    def __init__(self, init_param):
        super(kContrast, self).__init__()
        self.trans_param = Parameter(torch.Tensor([init_param]))

    def forward(self, x):
        clamped_param = torch.clamp(self.trans_param, min=0.)
        transform = K.enhance.AdjustContrast(contrast_factor=clamped_param)
        out = transform(x)
        return out
    
    def get_param_val(self):
        return self.trans_param.item()