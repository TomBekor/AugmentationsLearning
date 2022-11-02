from turtle import forward
import torch
from torch import logit, nn
from torch.nn.functional import gumbel_softmax
from torch.nn.parameter import Parameter

import kornia
import kornia.augmentation as K


class DUniform(nn.Module):
    def __init__(self, d_transfrom_class, bounds):
        super(DUniform, self).__init__()
        self.d_transfrom_class = d_transfrom_class
        self.learned_bounds = Parameter(torch.tensor(bounds))
        # self.entropy = None
        # self.range_scale = d_transfrom_class.range_scale if hasattr(d_transfrom_class, 'range_scale') else 1


    def forward(self, input):
        l, u  = self.learned_bounds
        magnitude_u_dist = torch.distributions.uniform.Uniform(l, u)
        rsampled_magnitude = magnitude_u_dist.rsample()
        d_transform = self.d_transfrom_class(rsampled_magnitude)
        return d_transform(input)

    def get_param_val(self):
        return self.learned_bounds.detach().cpu().numpy()

    def get_entropy(self):
        l, u  = self.learned_bounds
        magnitude_u_dist = torch.distributions.uniform.Uniform(l, u)
        return magnitude_u_dist.entropy()



class DGumbel(nn.Module):
    def __init__(self, d_transfrom_class_list, logits_init=None):
        super(DGumbel, self).__init__()

        self.d_transfrom_class_list = nn.ModuleList(d_transfrom_class_list)
        
        if not logits_init:
            transforms_len = len(d_transfrom_class_list)
            uniform_prob = 1/transforms_len
            self.logits = Parameter(torch.tensor([[uniform_prob]*transforms_len])) # uniform distribtion initialization
        else:
            self.logits = Parameter(torch.tensor(logits_init))


    def forward(self, input):

        # logits = self.logits.repeat(input.size[0], 1)

        one_hot_sample = gumbel_softmax(logits=self.logits, tau=1, hard=True)[0]

        out = torch.zeros_like(input)

        for d_transfrom_class, b in zip(self.d_transfrom_class_list, one_hot_sample):

            out += b * d_transfrom_class(input)

        return out

    def get_param_val(self):
        return torch.sigmoid(self.logits.detach()).cpu().numpy()



