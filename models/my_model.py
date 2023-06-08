import torch
import torch.nn as nn
from torchvision import models
from .hrnet_ocr import get_seg_model
import torch.nn.functional as F

class FcnResnet50(nn.Module):
    def __init__(self,pretrained=True,num_classes=29):
        super().__init__()
        self.model =  models.segmentation.fcn_resnet50(pretrained=pretrained)
        self.model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
        
    def forward(self,x):
        x = self.model(x)['out']
        return x
    
class HRNet48OCR(nn.Module):
    def __init__(self,pretrained="",num_classes=29):
        super().__init__()
        self.model = get_seg_model(name='hrnet48',pretrained=pretrained)
    
    def forward(self,x):
        x = self.model(x)
        return x