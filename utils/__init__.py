import torch
import numpy as np
from scipy.ndimage import uniform_filter


def compute_mae(pred, target):
    pred = pred.view(-1)
    target = target.view(-1)
    return torch.abs(pred - target).mean().item()


def compute_fmeasure(pred, target, beta2=0.3):
    pred = pred.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()
    target = (target >= 0.5).astype(np.float32)

    if target.sum() == 0:
        return 0.0

    best_f = 0.0
    for th in range(256):
        binary_pred = (pred * 255 >= th).astype(np.float32)
        tp = (binary_pred * target).sum()
        fp = (binary_pred * (1 - target)).sum()
        fn = ((1 - binary_pred) * target).sum()

        pre = tp / (tp + fp + 1e-7)
        rec = tp / (tp + fn + 1e-7)
        f = (1 + beta2) * pre * rec / (beta2 * pre + rec + 1e-7)
        if f > best_f:
            best_f = f

    return best_f


def compute_smeasure(pred, target, alpha=0.5):
    """
    计算 S-measure (Structure-measure)
    参考论文: Structure-measure: A New Way to Evaluate Foreground Maps (ICCV 2017)
    """
    pred = pred.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()
    
    # 确保是2D数组
    if pred.ndim != 2:
        pred = pred.reshape(pred.shape[-2], pred.shape[-1])
    if target.ndim != 2:
        target = target.reshape(target.shape[-2], target.shape[-1])
    
    y = target.mean()
    
    if y == 0:
        # 全背景情况
        score = 1.0 - pred.mean()
        return score
    elif y == 1:
        # 全前景情况
        score = pred.mean()
        return score
    else:
        # 计算目标感知的结构相似性 S_o
        s_object = _s_object(pred, target)
        # 计算区域感知的结构相似性 S_r
        s_region = _s_region(pred, target)
        # 组合
        s_measure = alpha * s_object + (1 - alpha) * s_region
        return s_measure


def _s_object(pred, target):
    """计算目标感知的结构相似性"""
    # 前景区域
    fg = target.copy()
    # 背景区域
    bg = 1 - target
    
    # 前景区域的结构相似性
    o_fg = _object_score(pred, fg)
    # 背景区域的结构相似性
    o_bg = _object_score(1 - pred, bg)
    
    # 前景和背景的面积比例
    u = target.mean()
    
    return u * o_fg + (1 - u) * o_bg


def _object_score(pred, target):
    """计算单个区域的目标分数"""
    # 计算预测图和目标图的均值
    x = pred[target > 0.5] if (target > 0.5).sum() > 0 else np.array([0])
    
    if len(x) == 0:
        return 0
    
    mu_x = x.mean()
    sigma_x = x.std()
    
    # 目标分数：2 * mu * sigma / (mu^2 + sigma^2)
    score = 2 * mu_x / (mu_x ** 2 + 1 + 1e-7)
    
    return score


def _s_region(pred, target):
    """计算区域感知的结构相似性"""
    h, w = target.shape
    
    # 将图像分成4个区域
    h2, w2 = h // 2, w // 2
    
    if h2 == 0 or w2 == 0:
        return _ssim(pred, target)
    
    # 计算目标图的质心
    total = target.sum()
    if total == 0:
        x_center, y_center = w2, h2
    else:
        y_indices, x_indices = np.where(target > 0.5)
        if len(x_indices) > 0:
            x_center = int(x_indices.mean())
            y_center = int(y_indices.mean())
        else:
            x_center, y_center = w2, h2
    
    # 确保质心在有效范围内
    x_center = max(1, min(w - 1, x_center))
    y_center = max(1, min(h - 1, y_center))
    
    # 分割成4个区域
    gt1 = target[0:y_center, 0:x_center]
    gt2 = target[0:y_center, x_center:w]
    gt3 = target[y_center:h, 0:x_center]
    gt4 = target[y_center:h, x_center:w]
    
    pred1 = pred[0:y_center, 0:x_center]
    pred2 = pred[0:y_center, x_center:w]
    pred3 = pred[y_center:h, 0:x_center]
    pred4 = pred[y_center:h, x_center:w]
    
    # 计算每个区域的权重
    w1 = gt1.size / target.size
    w2 = gt2.size / target.size
    w3 = gt3.size / target.size
    w4 = gt4.size / target.size
    
    # 计算每个区域的SSIM
    s1 = _ssim(pred1, gt1)
    s2 = _ssim(pred2, gt2)
    s3 = _ssim(pred3, gt3)
    s4 = _ssim(pred4, gt4)
    
    return w1 * s1 + w2 * s2 + w3 * s3 + w4 * s4


def _ssim(pred, target):
    """计算结构相似性 (SSIM)"""
    if pred.size == 0 or target.size == 0:
        return 0
    
    x = pred.flatten()
    y = target.flatten()
    
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x = x.std()
    sigma_y = y.std()
    sigma_xy = ((x - mu_x) * (y - mu_y)).mean()
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / \
           ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x ** 2 + sigma_y ** 2 + C2) + 1e-7)
    
    return max(0, ssim)


def compute_emeasure(pred, target):
    """
    计算 max E-measure (Enhanced-alignment measure)
    参考论文: Enhanced-alignment Measure for Binary Foreground Map Evaluation (IJCAI 2018)
    扫描 256 个阈值取最大对齐值
    """
    pred = pred.cpu().numpy().squeeze()
    target = target.cpu().numpy().squeeze()

    if pred.ndim != 2:
        pred = pred.reshape(pred.shape[-2], pred.shape[-1])
    if target.ndim != 2:
        target = target.reshape(target.shape[-2], target.shape[-1])

    target = (target >= 0.5).astype(np.float32)
    gt_mean = target.mean()

    if gt_mean == 0:
        best_e = 0.0
        for th in range(256):
            binary_pred = (pred * 255 >= th).astype(np.float32)
            e = (1 - binary_pred).mean()
            if e > best_e:
                best_e = e
        return best_e
    elif gt_mean == 1:
        best_e = 0.0
        for th in range(256):
            binary_pred = (pred * 255 >= th).astype(np.float32)
            e = binary_pred.mean()
            if e > best_e:
                best_e = e
        return best_e

    best_e = 0.0
    for th in range(256):
        binary_pred = (pred * 255 >= th).astype(np.float32)
        pred_mean = binary_pred.mean()

        align_pred = binary_pred - pred_mean
        align_gt = target - gt_mean

        align_matrix = 2 * align_pred * align_gt / (align_pred ** 2 + align_gt ** 2 + 1e-7)
        enhanced_matrix = (align_matrix + 1) ** 2 / 4
        e = enhanced_matrix.mean()
        if e > best_e:
            best_e = e

    return best_e


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename):
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['best_mae']


def align_to_multiple(h, w, multiple=32):
    import math
    return math.ceil(h / multiple) * multiple, math.ceil(w / multiple) * multiple


def multi_scale_inference(model, image, scales=(0.75, 1.0, 1.25)):
    orig_h, orig_w = image.shape[2:]
    total_pred = None
    for scale in scales:
        h, w = align_to_multiple(int(orig_h * scale), int(orig_w * scale))
        scaled = torch.nn.functional.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
        with torch.no_grad():
            sal_list, _, _ = model(scaled)
            pred = torch.sigmoid(sal_list[0])
        pred = torch.nn.functional.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        total_pred = pred if total_pred is None else total_pred + pred
    return total_pred / len(scales)
