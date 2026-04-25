import torch
import torch.nn as nn
import torch.nn.functional as F

#2. 编码器设计
#2.3 空间自适应门控融合模块（SAGF）
#2.3 层级自适应：高分辨率层用加权求和控计算量；中低层用轻量交叉注意力强化对齐
#2.3 模态权重：L2范数+Softmax生成动态像素级权重图

#9 SAGF工作原理
#9 功能：编码器各层像素级跨模态动态对齐
#9 策略：高层效率优先（加权求和），中低层精度优先（交叉注意力）
#9 优势：应对多目标、遮挡与复杂背景


class ChannelAttention(nn.Module):
    """轻量级SE通道注意力模块"""
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        B, C, _, _ = x.shape
        y = self.avg_pool(x).view(B, C)
        y = self.fc(y).view(B, C, 1, 1)
        return x * y


class SAGFHighResolution(nn.Module):
    def __init__(self, cnn_channels, transformer_channels, out_channels):
        super(SAGFHighResolution, self).__init__()
        self.cnn_proj = nn.Conv2d(cnn_channels, out_channels, kernel_size=1)
        self.trans_proj = nn.Conv2d(transformer_channels, out_channels, kernel_size=1)
        self.channel_attn = ChannelAttention(out_channels)
    
    def forward(self, f_cnn, f_trans):
        f_c = self.cnn_proj(f_cnn)
        f_t = self.trans_proj(f_trans)
        
        a_c = torch.norm(f_c, p=2, dim=1, keepdim=True)
        a_t = torch.norm(f_t, p=2, dim=1, keepdim=True)
        
        weights = torch.cat([a_c, a_t], dim=1)
        weights = F.softmax(weights, dim=1)
        alpha, beta = weights[:, 0:1, :, :], weights[:, 1:2, :, :]
        
        f_fusion = alpha * f_c + beta * f_t
        f_fusion = self.channel_attn(f_fusion)
        
        return f_fusion


class SAGFLowResolution(nn.Module):
    def __init__(self, cnn_channels, transformer_channels, out_channels):
        super(SAGFLowResolution, self).__init__()
        self.cnn_proj = nn.Conv2d(cnn_channels, out_channels, kernel_size=1)
        self.trans_proj = nn.Conv2d(transformer_channels, out_channels, kernel_size=1)
        
        self.query_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        self.depthwise_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        
        self.norm = nn.LayerNorm(out_channels)
        self.channel_attn = ChannelAttention(out_channels)
        
        self.scale = out_channels ** -0.5
    
    def forward(self, f_cnn, f_trans):
        f_c = self.cnn_proj(f_cnn)
        f_t = self.trans_proj(f_trans)
        
        a_c = torch.norm(f_c, p=2, dim=1, keepdim=True)
        a_t = torch.norm(f_t, p=2, dim=1, keepdim=True)
        
        weights = torch.cat([a_c, a_t], dim=1)
        weights = F.softmax(weights, dim=1)
        alpha, beta = weights[:, 0:1, :, :], weights[:, 1:2, :, :]
        
        q = alpha * f_c + beta * f_t
        k = beta * f_c + alpha * f_t
        v = k
        
        q_proj = self.query_conv(q)
        k_proj = self.key_conv(k)
        v_proj = self.value_conv(v)
        
        B, C, H, W = q_proj.shape
        
        q_flat = q_proj.view(B, C, -1)
        k_flat = k_proj.view(B, C, -1)
        v_flat = v_proj.view(B, C, -1)
        
        attn = torch.bmm(q_flat.transpose(1, 2), k_flat) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(v_flat, attn.transpose(1, 2))
        out = out.view(B, C, H, W)
        
        out = self.depthwise_conv(out)
        out = self.out_conv(out)
        
        f_fusion = f_c + out
        f_fusion = f_fusion.permute(0, 2, 3, 1)
        f_fusion = self.norm(f_fusion)
        f_fusion = f_fusion.permute(0, 3, 1, 2)
        
        f_fusion = self.channel_attn(f_fusion)
        
        return f_fusion


class SAGF(nn.Module):
    def __init__(self, cnn_channels_list, trans_channels_list, out_channels_list):
        super(SAGF, self).__init__()
        
        self.sagf_1_4 = SAGFHighResolution(cnn_channels_list[0], trans_channels_list[0], out_channels_list[0])
        self.sagf_1_8 = SAGFLowResolution(cnn_channels_list[1], trans_channels_list[1], out_channels_list[1])
        self.sagf_1_16 = SAGFLowResolution(cnn_channels_list[2], trans_channels_list[2], out_channels_list[2])
        self.sagf_1_32 = SAGFLowResolution(cnn_channels_list[3], trans_channels_list[3], out_channels_list[3])
    
    def forward(self, cnn_features, trans_features):
        f1 = self.sagf_1_4(cnn_features[0], trans_features[0])
        f2 = self.sagf_1_8(cnn_features[1], trans_features[1])
        f3 = self.sagf_1_16(cnn_features[2], trans_features[2])
        f4 = self.sagf_1_32(cnn_features[3], trans_features[3])
        
        return [f1, f2, f3, f4]
