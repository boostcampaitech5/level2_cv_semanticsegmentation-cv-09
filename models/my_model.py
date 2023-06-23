import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from .hrnet_ocr import get_ocr_model
from .unet_family import unet_3plus, nested_unet


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
        self.model = get_ocr_model(name='hrnet48',pretrained=pretrained)
    
    def forward(self,x):
        x = self.model(x)
        return x
    
    
class HRNet32OCR(nn.Module):
    def __init__(self,pretrained="",num_classes=29):
        super().__init__()
        self.model = get_ocr_model(name='hrnet32',pretrained=pretrained)
    
    def forward(self,x):
        x = self.model(x)
        return x
    
"""
    UNet++
"""
class NestedUNet(nn.Module):
    def __init__(self, num_classes=29):
        super(NestedUNet,self).__init__()
        self.model = nested_unet.get_nestedu_model(num_classes=num_classes)
    
    def forward(self,x):
        x = self.model(x)
        return x
        
"""
    UNet3+
"""
class UNet3plus(nn.Module):
    def __init__(self, num_classes=29):
        super(UNet3plus, self).__init__()
        self.model = unet_3plus.get_unet3plus_model(num_classes = num_classes)
        
    def forward(self,x):
        x = self.model(x)
        return x
    
"""
    UNet
"""
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


class PSPNet(nn.Module):
    def __init__(self, num_classes=29):
        super(PSPNet, self).__init__()
        self.model = smp.PSPNet(
            encoder_name="resnet101",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,                      # model output channels (number of classes in your dataset)
        )
    def forward(self, x):
        return self.model(x)

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=29):
        super(DeepLabV3Plus, self).__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet101",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    def forward(self, x):
        return self.model(x)