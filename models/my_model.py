import torch
import torch.nn as nn
from torchvision import models
from .hrnet_ocr import get_seg_model
import torch.nn.functional as F
import segmentation_models_pytorch as smp


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
    

class Unet(nn.Module):
    def __init__(self, num_classes=29):
        super(Unet, self).__init__()
        self.model = smp.Unet(
            encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )
    def forward(self, x):
        return self.model(x)


