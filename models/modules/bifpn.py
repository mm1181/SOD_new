import torch
import torch.nn as nn
import torch.nn.functional as F

#3. 特征融合模块
#3.1 双向特征金字塔（BiFPN）

#3.1 双向跨尺度融合 + 自适应加权聚合
#3.1 输出多尺度增强特征用于预测


class BiFPNBlock(nn.Module):
    def __init__(self, channels):
        super(BiFPNBlock, self).__init__()
        
        self.w1 = nn.Parameter(torch.ones(2))
        self.w2 = nn.Parameter(torch.ones(3))
        
        self.conv_p5_up = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_p4_up = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_p3_up = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_p2_up = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
        self.conv_p3_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_p4_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.conv_p5_down = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features):
        p2, p3, p4, p5 = features
        
        w1 = F.relu(self.w1)
        w1 = w1 / (w1.sum() + 1e-7)
        w2 = F.relu(self.w2)
        w2 = w2 / (w2.sum() + 1e-7)
        
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='bilinear', align_corners=False)
        p4_td = self.conv_p4_up(w1[0] * p4 + w1[1] * p5_up)
        
        p4_up = F.interpolate(p4_td, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3_td = self.conv_p3_up(w1[0] * p3 + w1[1] * p4_up)
        
        p3_up = F.interpolate(p3_td, size=p2.shape[2:], mode='bilinear', align_corners=False)
        p2_out = self.conv_p2_up(w1[0] * p2 + w1[1] * p3_up)
        
        p2_down = F.interpolate(p2_out, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p3_out = self.conv_p3_down(w2[0] * p3 + w2[1] * p3_td + w2[2] * p2_down)
        
        p3_down = F.interpolate(p3_out, size=p4.shape[2:], mode='bilinear', align_corners=False)
        p4_out = self.conv_p4_down(w2[0] * p4 + w2[1] * p4_td + w2[2] * p3_down)
        
        p4_down = F.interpolate(p4_out, size=p5.shape[2:], mode='bilinear', align_corners=False)
        p5_out = self.conv_p5_down(w2[0] * p5 + w2[1] * p4_down)
        
        return [p2_out, p3_out, p4_out, p5_out]


class BiFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(BiFPN, self).__init__()
        
        self.p2_conv = nn.Sequential(
            nn.Conv2d(in_channels_list[0], out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.p3_conv = nn.Sequential(
            nn.Conv2d(in_channels_list[1], out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.p4_conv = nn.Sequential(
            nn.Conv2d(in_channels_list[2], out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.p5_conv = nn.Sequential(
            nn.Conv2d(in_channels_list[3], out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.bifpn_block = BiFPNBlock(out_channels)
    
    def forward(self, features):
        p2 = self.p2_conv(features[0])
        p3 = self.p3_conv(features[1])
        p4 = self.p4_conv(features[2])
        p5 = self.p5_conv(features[3])
        
        out = self.bifpn_block([p2, p3, p4, p5])
        
        return out
