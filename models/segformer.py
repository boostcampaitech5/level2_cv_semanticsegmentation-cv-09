import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch 

class segformer(nn.Module):
    def __init__(self, num_classes=29):
        self.model = eval('SegFormer')(
            backbone='MiT-B3',
            num_classes=150
        )
        self.model.load_state_dict(torch.load('pretrain/segformer.b3.ade.pth', map_location='cpu'))
        self.model.linear_pred = nn.Conv2d(768, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        return self.model(x)