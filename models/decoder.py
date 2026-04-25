import torch
import torch.nn as nn
import torch.nn.functional as F

#4. 解码器与边缘监督

#4.1 解码器结构
#4.1 U-Net式上采样融合BiFPN特征，引入CBAM注意力

#4.2 多分辨率边缘监督分支
#4.2 1/4与1/8分辨率双路预测，上采样平均融合为最终边缘
#4.2 Sobel边缘图监督，双路BCE损失求和

#9 设计动机
#9 浅层细节与中层语义互补，提升边缘质量


class ChannelAttention(nn.Module):
    """CBAM 通道注意力"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return torch.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 空间注意力"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return torch.sigmoid(self.conv(x))


class CBAM(nn.Module):
    """CBAM: Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attn = ChannelAttention(channels, reduction)
        self.spatial_attn = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attn(x)
        x = x * self.spatial_attn(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        # CBAM 注意力在 concat 后应用
        self.cbam = CBAM(in_channels + skip_channels, reduction=16)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([x, skip], dim=1)
        x = self.cbam(x)  # 应用 CBAM 注意力
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x


class EdgeBranch(nn.Module):
    def __init__(self, in_channels):
        super(EdgeBranch, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(in_channels // 2, 1, kernel_size=1)
        )
    
    def forward(self, x, target_size):
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super(Decoder, self).__init__()
        
        self.decoder4 = DecoderBlock(in_channels_list[3], in_channels_list[2], out_channels)
        self.decoder3 = DecoderBlock(out_channels, in_channels_list[1], out_channels // 2)
        self.decoder2 = DecoderBlock(out_channels // 2, in_channels_list[0], out_channels // 4)
        
        self.edge_branch_1_4 = EdgeBranch(out_channels // 4)
        self.edge_branch_1_8 = EdgeBranch(out_channels // 2)
        
        # 深度监督：每层解码器输出一个显著图预测
        self.sal_head_1_4 = nn.Conv2d(out_channels // 4, 1, kernel_size=1)   # d2: 1/4分辨率
        self.sal_head_1_8 = nn.Conv2d(out_channels // 2, 1, kernel_size=1)   # d3: 1/8分辨率
        self.sal_head_1_16 = nn.Conv2d(out_channels, 1, kernel_size=1)       # d4: 1/16分辨率
    
    def forward(self, features, input_size):
        p2, p3, p4, p5 = features
        
        d4 = self.decoder4(p5, p4)       # 1/16 分辨率
        d3 = self.decoder3(d4, p3)       # 1/8 分辨率
        d2 = self.decoder2(d3, p2)       # 1/4 分辨率
        
        edge_1_4 = self.edge_branch_1_4(d2, input_size)
        edge_1_8 = self.edge_branch_1_8(d3, input_size)
        
        # 深度监督：多尺度显著图预测，全部上采样到输入尺寸
        sal_1_4 = self.sal_head_1_4(d2)
        sal_1_4 = F.interpolate(sal_1_4, size=input_size, mode='bilinear', align_corners=False)
        
        sal_1_8 = self.sal_head_1_8(d3)
        sal_1_8 = F.interpolate(sal_1_8, size=input_size, mode='bilinear', align_corners=False)
        
        sal_1_16 = self.sal_head_1_16(d4)
        sal_1_16 = F.interpolate(sal_1_16, size=input_size, mode='bilinear', align_corners=False)
        
        # 返回显著图列表 [1/4, 1/8, 1/16] 和边缘预测
        saliency_list = [sal_1_4, sal_1_8, sal_1_16]
        
        return saliency_list, edge_1_4, edge_1_8
