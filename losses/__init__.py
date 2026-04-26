import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import sobel


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, pred, target):
        if self.smoothing > 0:
            target = target * (1 - self.smoothing) + self.smoothing * 0.5
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


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        pred = torch.sigmoid(pred).view(-1)
        target = target.view(-1)
        tp = (pred * target).sum()
        fn = ((1 - pred) * target).sum()
        fp = (pred * (1 - target)).sum()
        tversky = tp / (tp + self.alpha * fn + self.beta * fp + 1e-7)
        return 1 - tversky


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, edge_pred_1, edge_pred_2, target):
        target_edge = sobel(target)
        target_edge = (target_edge > 0.1).float()

        loss1 = F.binary_cross_entropy_with_logits(edge_pred_1, target_edge)
        loss2 = F.binary_cross_entropy_with_logits(edge_pred_2, target_edge)

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
    def __init__(self, focal_weight=0.5, iou_weight=1.5, edge_weight=0.8, ssim_weight=0.5,
                 deep_sup_weights=None, tversky_weight=2.5, label_smoothing=0.02,
                 weight_warmup_epochs=5):
        super(CombinedLoss, self).__init__()

        self.focal_loss = FocalLoss(alpha=0.25, smoothing=label_smoothing)
        self.iou_loss = IoULoss()
        self.tversky_loss = TverskyLoss(alpha=0.3, beta=0.7)
        self.edge_loss = EdgeLoss()
        self.ssim_loss = SSIMLoss()

        self.focal_weight = focal_weight
        self.edge_weight = edge_weight
        self.ssim_weight = ssim_weight

        self.weight_warmup_epochs = weight_warmup_epochs
        self.current_epoch = 0

        self.target_weights = deep_sup_weights or [1.0, 0.8, 0.6]
        self.base_weights = [1.0, 0.5, 0.25]

        self.tversky_weight_target = tversky_weight
        self.tversky_weight_start = 0.0
        self.iou_weight_target = 0.5
        self.iou_weight_start = iou_weight
        self.alpha_target = 0.30
        self.alpha_start = 0.25

    def set_epoch(self, epoch):
        self.current_epoch = epoch

        if epoch < self.weight_warmup_epochs:
            ratio = (epoch + 1) / self.weight_warmup_epochs
            alpha = self.alpha_start + (self.alpha_target - self.alpha_start) * ratio
            self.focal_loss.alpha = alpha
        else:
            self.focal_loss.alpha = self.alpha_target

    def _get_deep_sup_weights(self):
        if self.current_epoch >= self.weight_warmup_epochs:
            return self.target_weights
        ratio = (self.current_epoch + 1) / self.weight_warmup_epochs
        return [b + (t - b) * ratio for b, t in zip(self.base_weights, self.target_weights)]

    def _get_tversky_iou_weights(self):
        if self.current_epoch >= self.weight_warmup_epochs:
            return self.tversky_weight_target, self.iou_weight_target
        ratio = (self.current_epoch + 1) / self.weight_warmup_epochs
        tv = self.tversky_weight_start + (self.tversky_weight_target - self.tversky_weight_start) * ratio
        io = self.iou_weight_start + (self.iou_weight_target - self.iou_weight_start) * ratio
        return tv, io

    def _compute_saliency_loss(self, pred, target, tversky_w, iou_w, use_ssim=True):
        focal = self.focal_loss(pred, target)
        iou = self.iou_loss(pred, target)
        tversky = self.tversky_loss(pred, target)

        focal = torch.clamp(focal, min=0.0, max=100.0)
        iou = torch.clamp(iou, min=0.0, max=100.0)
        tversky = torch.clamp(tversky, min=0.0, max=100.0)

        loss = self.focal_weight * focal + iou_w * iou + tversky_w * tversky

        if use_ssim:
            ssim = self.ssim_loss(pred, target)
            ssim = torch.clamp(ssim, min=0.0, max=100.0)
            loss = loss + self.ssim_weight * ssim
            return loss, focal, iou, tversky, ssim
        else:
            return loss, focal, iou, tversky, torch.tensor(0.0, device=pred.device)

    def forward(self, saliency_list, edge_1_4, edge_1_8, target):
        device = saliency_list[0].device

        deep_weights_list = self._get_deep_sup_weights()
        deep_weights = [torch.tensor(w, device=device) for w in deep_weights_list]

        tversky_w_raw, iou_w_raw = self._get_tversky_iou_weights()
        tversky_w = torch.tensor(tversky_w_raw, device=device)
        iou_w = torch.tensor(iou_w_raw, device=device)

        total_sal_loss = torch.tensor(0.0, device=device)
        focal_val = None
        iou_val = None
        tversky_val = None
        ssim_val = None

        for i, (sal_pred, w) in enumerate(zip(saliency_list, deep_weights)):
            use_ssim = (i == 0)
            sal_loss, f, io, tv, ss = self._compute_saliency_loss(
                sal_pred, target, tversky_w, iou_w, use_ssim
            )
            total_sal_loss = total_sal_loss + w * sal_loss
            if i == 0:
                focal_val = f
                iou_val = io
                tversky_val = tv
                ssim_val = ss

        edge = self.edge_loss(edge_1_4, edge_1_8, target)
        edge = torch.clamp(edge, min=0.0, max=100.0)

        total_loss = total_sal_loss + self.edge_weight * edge

        return total_loss, {
            'focal': focal_val.item(),
            'iou': iou_val.item(),
            'tversky': tversky_val.item(),
            'edge': edge.item(),
            'ssim': ssim_val.item(),
            'total': total_loss.item()
        }
