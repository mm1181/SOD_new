import torch
import torch.nn as nn
from .backbones import ResNet34, PVTV2B2
from .modules import SAGF, GGA, BiFPN, RFBModule
from .decoder import Decoder

#1. 总体架构
#1. 并行双流特征提取：ResNet34（局部细节）+ PVTv2-B2（全局语义），输出多尺度特征。
#1. 自适应融合：SAGF模块逐层动态融合双流特征。
#1. 全局增强：顶层GGA模块强化通道与空间全局依赖。
#1. 多级增强：BiFPN跨层连接+RFB扩大感受野。
#1. 解码与监督：U-Net式上采样+边缘监督分支优化边界。

#9 ResNet34与PVTv2-B2优点
#9 二者优势互补，分别提供局部细节与全局语义信息，并支持预训练加速收敛。

class SODModel(nn.Module):
    def __init__(self, pretrained_resnet=True, pretrained_pvt_path=None, out_channels=256):
        super(SODModel, self).__init__()
        
        self.cnn_backbone = ResNet34(pretrained=pretrained_resnet)
        cnn_channels = self.cnn_backbone.out_channels
        
        self.transformer_backbone = PVTV2B2(pretrained_path=pretrained_pvt_path)
        trans_channels = self.transformer_backbone.out_channels
        
        sagf_channels = [64, 128, 256, 256]
        self.sagf = SAGF(cnn_channels, trans_channels, sagf_channels)
        
        # GGA 模块：1/32 和 1/16 层都使用
        self.gga_32 = GGA(sagf_channels[3])
        self.gga_16 = GGA(sagf_channels[2])
        
        bifpn_in_channels = sagf_channels
        self.bifpn = BiFPN(bifpn_in_channels, out_channels)
        
        self.rfb = RFBModule([out_channels] * 4, out_channels)
        
        self.decoder = Decoder([out_channels] * 4, out_channels)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        cnn_features = self.cnn_backbone(x)
        trans_features = self.transformer_backbone(x)
        
        fused_features = self.sagf(cnn_features, trans_features)
        
        # GGA 应用于 1/32 和 1/16 层
        fused_features[3] = self.gga_32(fused_features[3])
        fused_features[2] = self.gga_16(fused_features[2])
        
        bifpn_features = self.bifpn(fused_features)
        
        rfb_features = self.rfb(bifpn_features)
        
        # 返回 saliency_list, edge_1_4, edge_1_8
        saliency_list, edge_1_4, edge_1_8 = self.decoder(rfb_features, input_size)
        
        return saliency_list, edge_1_4, edge_1_8
