# Author: Yifei Ning (Couson)
# Last Update: 3/18/2022
# Reference: https://discuss.pytorch.org/t/hinge-loss-in-pytorch/86220
# Reference: https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html
# Reference: https://en.wikipedia.org/wiki/Hinge_loss

import torch
import torch.nn as nn
from torch.nn import HingeEmbeddingLoss

class HingeLossRegularizer(torch.nn.Module):

    def __init__(self, constant, hinge_loss_fn_name):
        super(HingeLossRegularizer, self).__init__()
        if hinge_loss_fn_name == 'relu':
            self.hinge_loss_fn = nn.ReLU()
        elif hinge_loss_fn_name == 'exp':
            self.hinge_loss_fn= torch.exp
        else:
            raise Exception("%s not handled" % (hinge_loss_fn_name))
        self.lam = constant

    def forward(self, f_x1, f_x2):
        diff = torch.sum(self.hinge_loss_fn(f_x1[:,1] - f_x2[:,1])) + torch.sum(self.hinge_loss_fn(f_x2[:,0] - f_x1[:,0]))
        # diff = torch.sum(self.relu(f_x1[:,1] - f_x2[:,1])) + torch.sum(self.relu(f_x2[:,0] - f_x1[:,0]))
        out = diff * self.lam
        return (out)
