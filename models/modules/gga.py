import torch
import torch.nn as nn
import torch.nn.functional as F

#2. 编码器设计
#2.4 全局空间-通道双维度门控注意力（GGA）
#2.4 通道与空间门控联合生成注意力权重，残差连接保留原始特征，仅作用于高层语义层

#9 GGA的作用
#9 增强通道与空间全局上下文建模，提升高层特征定位能力
#9 低层特征不经GGA，以保持局部细节信息

class GGA(nn.Module):
    def __init__(self, channels, reduction=16):
        super(GGA, self).__init__()
        
        self.channel_gap = nn.AdaptiveAvgPool2d(1)
        self.channel_gmp = nn.AdaptiveMaxPool2d(1)
        
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        
        self.spatial_conv1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.spatial_conv2 = nn.Conv2d(channels // reduction, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        gap = self.channel_gap(x).view(B, C)
        gmp = self.channel_gmp(x).view(B, C)
        
        channel_weight = self.channel_mlp(gap) + self.channel_mlp(gmp)
        g_c = torch.sigmoid(channel_weight).view(B, C, 1, 1)
        
        spatial = self.spatial_conv1(x)
        spatial = F.relu(spatial, inplace=True)
        spatial = self.spatial_conv2(spatial)
        g_s = torch.sigmoid(spatial)
        
        out = x * g_c * g_s
        
        out = x + out
        
        return out
