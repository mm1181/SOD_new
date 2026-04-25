import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

#2. 编码器设计
#2.1 CNN分支(ResNet34)

#2.1 移除全连接层，保留四个阶段的输出特征图
#2.1 输出分辨率分别为输⼊图像的1/4、1/8、1/16、1/32; 通道数分别为64、128、256、512
#2.1 使⽤ImageNet预训练权重初始化


class ResNet34(nn.Module):
    def __init__(self, pretrained=True, bn_momentum=0.05):
        super(ResNet34, self).__init__()
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
            backbone = resnet34(weights=weights)
        else:
            backbone = resnet34(weights=None)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        
        self.out_channels = [64, 128, 256, 512]
        self.out_strides = [4, 8, 16, 32]
        
        # 调整所有BatchNorm层的动量
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
    
    def forward(self, x):
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        features.append(x)
        
        x = self.layer2(x)
        features.append(x)
        
        x = self.layer3(x)
        features.append(x)
        
        x = self.layer4(x)
        features.append(x)
        
        return features
