import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import kornia as K


class kRotation(nn.Module):
    def __init__(self, init_param):
        super(kRotation, self).__init__()
        self.trans_param = Parameter(torch.Tensor([init_param]))

    def forward(self, x):
        clamped_param = torch.clamp(self.trans_param, min=-360., max=360.)
        transform = K.geometry.transform.Rotate(angle=clamped_param)
        out = transform(x)
        return out
    
    def get_param_val(self):
        return self.trans_param.item()