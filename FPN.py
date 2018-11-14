import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.resnet import resnet101, resnet152, resnet50


class resnet_backbone(nn.Module):
    def __init__(self, num_class = 1, input_channel = 3, output_stride=32, layer=101):
        super(resnet_backbone, self).__init__()
        if layer == 101:
            self.resnet = resnet101(pretrained=True, output_stride=output_stride)
        elif layer == 152:
            self.resnet = resnet152(pretrained=True, output_stride=output_stride)
        elif layer == 50:
            self.resnet = resnet50(pretrained=True, output_stride=output_stride)
        else:
            raise ValueError("only support ResNet101 and ResNet152 now")

        if input_channel == 1:
            self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding = 3, bias=False)
        elif input_channel == 3:
            self.conv1 = self.resnet.conv1
        else:
            raise ValueError("input channel should be 3 or 1")

    def forward(self, x):

        c1 = self.conv1(x) #1, 320*320
        c1 = self.resnet.bn1(c1)
        c1 = self.resnet.relu(c1)
        c1 = self.resnet.maxpool(c1) #4, 80*80

        c2 = self.resnet.layer1(c1)
        c3 = self.resnet.layer2(c2) #8, 40*40
        c4 = self.resnet.layer3(c3) #16, 20*20
        c5 = self.resnet.layer4(c4) #32, 10*10

        return c1, c2, c3, c4, c5

class lateral_connect(nn.Module):
    '''
    according to paper, there is no nonlinear function in these extra layers
    '''
    def __init__(self, input_channel, output_channel):
        super(lateral_connect, self).__init__()
        self.lateral = nn.Conv2d(input_channel, output_channel, kernel_size=1)
        self.append_conv = nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
    def upsample_and_add(self, target, small):
        n, c, h, w = target.size()
        return F.interpolate(small, size=(h, w), mode='bilinear', align_corners=True) + target
    def forward(self, bottom_up, top_down):
        return self.append_conv(self.upsample_and_add(self.lateral(bottom_up), top_down))


class fpn_base(nn.Module):
    def __init__(self, num_class = 1, input_channel = 3):
        super(fpn_base, self).__init__()
        
        self.backbone = resnet_backbone(num_class, input_channel)

        self.top = nn.Conv2d(2048, 256, kernel_size=1)
        self.lateral1 = lateral_connect(1024, 256)
        self.lateral2 = lateral_connect( 512, 256)
        self.lateral3 = lateral_connect( 256, 256)
    
    def forward(self, x):
        c1, c2, c3, c4, c5 = self.backbone(x)
        p5 = self.top(c5)
        p4 = self.lateral1(c4, p5)
        p3 = self.lateral2(c3, p4)
        p2 = self.lateral3(c2, p3)

        return p2, p3, p4, p5

class fpn_segmentation(nn.Module):
    def __init__(self, num_class = 1, input_channel = 3):
        super(fpn_segmentation, self).__init__()
        self.fpn_base = fpn_base(num_class, input_channel)
        self.predict = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                       nn.BatchNorm2d(256),
                                       nn.Conv2d(256, num_class, kernel_size=1, stride=1))


    def forward(self, img):
        p2, p3, p4, p5 = self.fpn_base(img)

        p2 = F.interpolate(p2, scale_factor= 4, mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, scale_factor= 8, mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, scale_factor=16, mode='bilinear', align_corners=True)
        p5 = F.interpolate(p5, scale_factor=32, mode='bilinear', align_corners=True)

        p2 = self.predict(p2)
        p3 = self.predict(p3)
        p4 = self.predict(p4)
        p5 = self.predict(p5)

        return p2, p3, p4, p5


if __name__ == "__main__":
    model = fpn_segmentation(input_channel=3, num_class=1).cuda()
    image = torch.randn(1, 3, 320, 320 ).cuda()
    with torch.no_grad():
        p2, p3, p4, p5 = model.forward(image)
    print(p2.shape, p3.shape, p4.shape, p5.shape)



