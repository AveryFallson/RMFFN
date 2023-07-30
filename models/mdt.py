import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.rvs import rvs            #没用到
import pdb
import models.resnet as resnet
from torch.autograd import Variable
# from torchstat import stat
# from Models import Transformer
from models.Layers import EncoderLayer, DecoderLayer #没用到

from torch import Tensor


def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class TA34(nn.Module):
    def __init__(self, n_class=40, n_view=24, pretrained=False):
        super(TA34, self).__init__()
        self.view = n_view
        # -------------------------------------------------------------------------
        self.net1 = resnet.__dict__['resnet18']()
        self.av1pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            # nn.Linear(128, n_class)
        )
        self.fc3 = nn.Linear(128, n_class)
        self.fc3over = nn.Linear(128, 100)

    def forward(self, x):
        b, v, _, _, _ = x.shape
        fts = self.av1pool(self.net1(x.reshape(-1, 3, 224, 224))).squeeze(-1).squeeze(-1)  # multi-view
        fts = fts.reshape(b, v, -1)
        x = torch.max(fts,dim=1)[0]
        glo_ft = x
        x = self.classifier(x)
        out = self.fc3(x)
        # print(self.fc3.weight[:3,:5])
        out_over = self.fc3over(x)
        return [out, out_over], fts, glo_ft
        # -------------------------------------------------------------------------


if __name__ == '__main__':
    model = TA34()
    # stat(model, (20, 3, 224, 224))