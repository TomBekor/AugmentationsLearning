import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import kornia as K

class kTranslateX(nn.Module):
    def __init__(self, init_param):
        super(kTranslateX, self).__init__()
        self.trans_param = Parameter(torch.Tensor([init_param]))

    def forward(self, x):
        param_device = self.trans_param.device
        translationY = torch.tensor([0.]).to(param_device)
        translation = torch.stack([self.trans_param, translationY], dim=1)
        transform = K.geometry.transform.Translate(translation=translation)
        out = transform(x)
        return out
    
    def get_param_val(self):
        return self.trans_param.item()