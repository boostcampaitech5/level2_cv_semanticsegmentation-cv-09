import torch
import torch.nn as nn
import utils
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
    
"""
    UNet++
"""
class Unet_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out

class Nested_UNet(nn.Module):
    def __init__(self, num_classes=29, input_channels=3):
        super().__init__()

        num_filter = [32, 64, 128, 256, 512]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # DownSampling
        self.conv0_0 = Unet_block(input_channels, num_filter[0], num_filter[0])
        self.conv1_0 = Unet_block(num_filter[0], num_filter[1], num_filter[1])
        self.conv2_0 = Unet_block(num_filter[1], num_filter[2], num_filter[2])
        self.conv3_0 = Unet_block(num_filter[2], num_filter[3], num_filter[3])
        self.conv4_0 = Unet_block(num_filter[3], num_filter[4], num_filter[4])

        # Upsampling & Dense skip
        # N to 1 skip
        self.conv0_1 = Unet_block(num_filter[0] + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_1 = Unet_block(num_filter[1] + num_filter[2], num_filter[1], num_filter[1])
        self.conv2_1 = Unet_block(num_filter[2] + num_filter[3], num_filter[2], num_filter[2])
        self.conv3_1 = Unet_block(num_filter[3] + num_filter[4], num_filter[3], num_filter[3])
       
        # N to 2 skip
        self.conv0_2 = Unet_block(num_filter[0]*2 + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_2 = Unet_block(num_filter[1]*2 + num_filter[2], num_filter[1], num_filter[1])
        self.conv2_2 = Unet_block(num_filter[2]*2 + num_filter[3], num_filter[2], num_filter[2])

        # N to 3 skip
        self.conv0_3 = Unet_block(num_filter[0]*3 + num_filter[1], num_filter[0], num_filter[0])
        self.conv1_3 = Unet_block(num_filter[1]*3 + num_filter[2], num_filter[1], num_filter[1])

        # N to 4 skip
        self.conv0_4 = Unet_block(num_filter[0]*4 + num_filter[1], num_filter[0], num_filter[0])

        self.output = nn.Conv2d(num_filter[0], num_classes, kernel_size=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                utils.init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                utils.init_weights(m, init_type='kaiming')

    def forward(self, x):                    # (Batch, 3, 256, 256)

        x0_0 = self.conv0_0(x)               
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))

        output = self.output(x0_4)

        return output
    
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


