import torch
import torch.nn as nn
from torchvision import models

class FcnResnet50(nn.Module):
    def __init__(self,pretrained=True,num_classes=29):
        super().__init__()
        self.model =  models.segmentation.fcn_resnet50(pretrained=pretrained)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self,x):
        x = self.model(x)
        return x['out']