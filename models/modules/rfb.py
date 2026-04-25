import torch
import torch.nn as nn
import torch.nn.functional as F

#3. 特征融合模块
#3.2 感受野模块（RFB）
#3.2 并行扩张率1、3、5空洞卷积 → 拼接 + 1×1降维 → 扩大感受野

#9 为何选用BiFPN与RFB
#9 BiFPN：双向加权特征融合
#9 RFB：多尺度感受野增强
#9 二者联合提升检测精度与效率

class RFB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFB, self).__init__()
        
        branch_channels = out_channels // 3
        remainder = out_channels % 3
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels + (remainder if remainder > 0 else 0), kernel_size=1),
            nn.BatchNorm2d(branch_channels + (remainder if remainder > 0 else 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels + (remainder if remainder > 0 else 0), branch_channels + (remainder if remainder > 0 else 0), kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(branch_channels + (remainder if remainder > 0 else 0)),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=3, dilation=3),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=5, dilation=5),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.conv_out(out)
        
        return out


class RFBModule(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(RFBModule, self).__init__()
        
        self.rfb1 = RFB(in_channels_list[0], out_channels)
        self.rfb2 = RFB(in_channels_list[1], out_channels)
        self.rfb3 = RFB(in_channels_list[2], out_channels)
        self.rfb4 = RFB(in_channels_list[3], out_channels)
    
    def forward(self, features):
        out1 = self.rfb1(features[0])
        out2 = self.rfb2(features[1])
        out3 = self.rfb3(features[2])
        out4 = self.rfb4(features[3])
        
        return [out1, out2, out3, out4]
