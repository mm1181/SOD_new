import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import sobel
from scipy import ndimage
import numpy as np

#5. 损失函数

#5. 二值交叉熵损失（BCE Loss）：像素级二分类损失
#5. IoU损失：1 - IoU，直接优化交并比
#5. 边缘损失（Edge Loss）：边缘预测与Sobel真值的BCE之和，权重0.8
#5. SSIM损失：结构相似性损失，权重0.3
#5. 深度监督：多尺度预测加权损失求和

#9 损失函数的设计考虑

#9 BCE Loss：保证像素分类精度
#9 IoU Loss：提升区域重叠度
#9 Edge Loss：增强边界清晰度
#9 SSIM Loss：改善结构一致性
#9 深度监督：强化多尺度学习
#9 组合效果：兼顾边界与整体质量

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        iou = intersection / (union + 1e-7)
        
        return 1 - iou


class BoundaryWeightedEdgeLoss(nn.Module):
    """边界加权损失：对边界附近像素给予更高权重"""
    def __init__(self, alpha=2.0, sigma=3.0):
        super(BoundaryWeightedEdgeLoss, self).__init__()
        self.alpha = alpha
        self.sigma = sigma
    
    def _compute_boundary_weight(self, target):
        """计算边界权重图"""
        B, C, H, W = target.shape
        weight_maps = []
        
        for b in range(B):
            mask = target[b, 0].cpu().numpy()
            # 计算边界
            edge = sobel(target[b:b+1]).squeeze().cpu().numpy()
            edge_binary = (edge > 0.1).astype(np.float32)
            
            # 距离变换
            if edge_binary.sum() > 0:
                dist = ndimage.distance_transform_edt(1 - edge_binary)
            else:
                dist = np.ones_like(edge_binary) * 10
            
            # 权重 = 1 + α * exp(-distance/σ)
            weight = 1 + self.alpha * np.exp(-dist / self.sigma)
            weight_maps.append(torch.from_numpy(weight).unsqueeze(0))
        
        weight_tensor = torch.stack(weight_maps, dim=0).to(target.device)
        return weight_tensor
    
    def forward(self, edge_pred_1, edge_pred_2, target):
        # 计算边界真值
        target_edge = sobel(target)
        target_edge = (target_edge > 0.1).float()
        
        # 计算边界权重
        weight = self._compute_boundary_weight(target)
        
        # 加权 BCE 损失
        loss1 = F.binary_cross_entropy_with_logits(edge_pred_1, target_edge, weight=weight, reduction='mean')
        loss2 = F.binary_cross_entropy_with_logits(edge_pred_2, target_edge, weight=weight, reduction='mean')
        
        return loss1 + loss2


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, channel=1):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.window = self._create_window(window_size, channel)
    
    def _create_window(self, window_size, channel):
        def gaussian(window_size, sigma):
            gauss = torch.Tensor([
                torch.exp(torch.tensor(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)))
                for x in range(window_size)
            ])
            return gauss / gauss.sum()
        
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(self, img1, img2, window, window_size, channel):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim_map.mean()
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        window = self.window.to(pred.device).type_as(pred)
        ssim_val = self._ssim(pred, target, window, self.window_size, self.channel)
        return 1 - ssim_val


class CombinedLoss(nn.Module):
    def __init__(self, focal_weight=0.5, iou_weight=1.5, edge_weight=0.8, ssim_weight=0.3,
                 deep_sup_weights=None):
        super(CombinedLoss, self).__init__()
        
        self.focal_loss = FocalLoss()
        self.iou_loss = IoULoss()
        self.edge_loss = BoundaryWeightedEdgeLoss(alpha=2.0, sigma=3.0)
        self.ssim_loss = SSIMLoss()
        
        self.focal_weight = focal_weight
        self.iou_weight = iou_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight
        
        # 深度监督权重：[1/4分辨率, 1/8分辨率, 1/16分辨率]
        self.deep_sup_weights = deep_sup_weights or [1.0, 0.5, 0.25]
    
    def _compute_saliency_loss(self, pred, target):
        """计算单个显著图预测的损失"""
        focal = self.focal_loss(pred, target)
        iou = self.iou_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        
        focal = torch.clamp(focal, min=0.0, max=100.0)
        iou = torch.clamp(iou, min=0.0, max=100.0)
        ssim = torch.clamp(ssim, min=0.0, max=100.0)
        
        loss = self.focal_weight * focal + self.iou_weight * iou + self.ssim_weight * ssim
        return loss, focal, iou, ssim
    
    def forward(self, saliency_list, edge_1_4, edge_1_8, target):
        # 深度监督：对每个尺度的显著图预测计算加权损失
        total_sal_loss = 0
        focal_val = 0
        iou_val = 0
        ssim_val = 0
        
        for i, (sal_pred, w) in enumerate(zip(saliency_list, self.deep_sup_weights)):
            sal_loss, f, io, ss = self._compute_saliency_loss(sal_pred, target)
            total_sal_loss = total_sal_loss + w * sal_loss
            if i == 0:  # 记录主输出的损失值用于日志
                focal_val = f
                iou_val = io
                ssim_val = ss
        
        # 边界加权损失
        edge = self.edge_loss(edge_1_4, edge_1_8, target)
        edge = torch.clamp(edge, min=0.0, max=100.0)
        
        total_loss = total_sal_loss + self.edge_weight * edge
        
        return total_loss, {
            'focal': focal_val.item(),
            'iou': iou_val.item(),
            'edge': edge.item(),
            'ssim': ssim_val.item(),
            'total': total_loss.item()
        }
