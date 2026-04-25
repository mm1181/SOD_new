import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from models import SODModel
from utils import compute_mae, compute_fmeasure, compute_smeasure, compute_emeasure


#9 数据集：DUTS-TR训练；DUTS-TE、ECSSD、HKU-IS、PASCAL-S测试
#9 评估指标：MAE、F-measure、S-measure、E-measure
#9 核心模块：空间自适应门控融合CNN局部与Transformer全局；全局双维度门控增强高层定位；多分辨率边缘监督细化边界
#12345

def parse_args():
    parser = argparse.ArgumentParser(description='SOD Testing')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--img_size', type=int, default=320, help='Input image size')
    parser.add_argument('--no_tta', action='store_true', help='Disable Test Time Augmentation')
    parser.add_argument('--tta_scales', type=str, default='0.75,1.0,1.25', help='TTA scales (comma separated)')
    return parser.parse_args()


def load_model(checkpoint_path, device):
    model = SODModel(pretrained_resnet=False, pretrained_pvt_path=None)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def get_dataset_paths(data_root, dataset_name):
    if dataset_name == 'DUTS-TE':
        image_dir = os.path.join(data_root, 'DUTS-TE', 'DUTS-TE-Image')
        mask_dir = os.path.join(data_root, 'DUTS-TE', 'DUTS-TE-Mask')
    elif dataset_name == 'ECSSD':
        image_dir = os.path.join(data_root, 'ECSSD', 'images')
        mask_dir = os.path.join(data_root, 'ECSSD', 'masks')
    elif dataset_name == 'HKU-IS':
        image_dir = os.path.join(data_root, 'HKU-IS', 'images')
        mask_dir = os.path.join(data_root, 'HKU-IS', 'masks')
    elif dataset_name == 'PASCAL-S':
        image_dir = os.path.join(data_root, 'PASCAL-S', 'images')
        mask_dir = os.path.join(data_root, 'PASCAL-S', 'masks')
    elif dataset_name == 'FT':
        image_dir = os.path.join(data_root, 'FT', 'images')
        mask_dir = os.path.join(data_root, 'FT', 'masks')
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    return image_dir, mask_dir


def get_mask_name(image_name, dataset_name):
    base_name = os.path.splitext(image_name)[0]
    return f"{base_name}.png"


def predict_single(model, image, img_size, device):
    """单次预测"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # 调整大小并转换为张量
    image_resized = TF.resize(image, (img_size, img_size))
    image_tensor = TF.to_tensor(image_resized)
    image_tensor = normalize(image_tensor).unsqueeze(0).to(device)
    
    saliency_list, _, _ = model(image_tensor)
    saliency = torch.sigmoid(saliency_list[0])
    
    return saliency


def predict_with_tta(model, image, img_size, device, scales):
    """使用 TTA 进行预测：多尺度 + 水平翻转"""
    original_size = image.size[::-1]  # (H, W)
    predictions = []
    
    for scale in scales:
        scaled_size = int(img_size * scale)
        
        # 原图预测
        pred = predict_single(model, image, scaled_size, device)
        # 上采样到原始尺寸
        pred = F.interpolate(pred, size=original_size, mode='bilinear', align_corners=False)
        predictions.append(pred)
        
        # 水平翻转预测
        image_flip = TF.hflip(image)
        pred_flip = predict_single(model, image_flip, scaled_size, device)
        pred_flip = F.interpolate(pred_flip, size=original_size, mode='bilinear', align_corners=False)
        # 翻转回来
        pred_flip = torch.flip(pred_flip, dims=[3])
        predictions.append(pred_flip)
    
    # 平均所有预测
    final_pred = torch.stack(predictions, dim=0).mean(dim=0)
    
    return final_pred


def test(model, image_dir, mask_dir, device, img_size, dataset_name, use_tta=False, tta_scales=[0.75, 1.0, 1.25]):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    mae_list = []
    f_list = []
    s_list = []
    e_list = []
    
    desc = 'Testing (TTA)' if use_tta else 'Testing'
    
    with torch.no_grad():
        for image_name in tqdm(image_files, desc=desc):
            image_path = os.path.join(image_dir, image_name)
            image = Image.open(image_path).convert('RGB')
            original_size = image.size[::-1]  # (H, W)
            
            if use_tta:
                # TTA 预测
                saliency = predict_with_tta(model, image, img_size, device, tta_scales)
                saliency = saliency.squeeze().cpu().numpy()
            else:
                # 普通预测
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                transform = transforms.Compose([
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                    normalize
                ])
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                saliency_list, _, _ = model(input_tensor)
                saliency = torch.sigmoid(saliency_list[0]).squeeze().cpu().numpy()
                
                # 调整到原始尺寸
                saliency_pil = Image.fromarray((saliency * 255).astype(np.uint8))
                saliency_pil = saliency_pil.resize((original_size[1], original_size[0]), Image.BILINEAR)
                saliency = np.array(saliency_pil) / 255.0
            
            mask_name = get_mask_name(image_name, dataset_name)
            mask_path = os.path.join(mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert('L')
                mask = mask.resize((original_size[1], original_size[0]), Image.NEAREST)
                mask_np = np.array(mask) / 255.0
                
                pred_tensor = torch.from_numpy(saliency).unsqueeze(0).unsqueeze(0)
                mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
                
                mae = compute_mae(pred_tensor, mask_tensor)
                f_measure = compute_fmeasure(pred_tensor, mask_tensor)
                s_measure = compute_smeasure(pred_tensor, mask_tensor)
                e_measure = compute_emeasure(pred_tensor, mask_tensor)
                
                mae_list.append(mae)
                f_list.append(f_measure)
                s_list.append(s_measure)
                e_list.append(e_measure)
    
    results = {
        'MAE': np.mean(mae_list) if mae_list else 0,
        'F-measure': np.mean(f_list) if f_list else 0,
        'S-measure': np.mean(s_list) if s_list else 0,
        'E-measure': np.mean(e_list) if e_list else 0
    }
    
    return results


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 固定模型地址
    checkpoint_path = 'outputs/best_model.pth'
    print(f'Loading model from {checkpoint_path}...')
    model = load_model(checkpoint_path, device)
    
    # 解析 TTA 尺度
    tta_scales = [float(s) for s in args.tta_scales.split(',')]
    
    # 要求用户输入数据集名称
    valid_datasets = ['DUTS-TE', 'ECSSD', 'HKU-IS', 'PASCAL-S', 'FT']
    dataset_name = input(f'请输入要测试的数据集名称 ({", ".join(valid_datasets)}): ').strip()
    while dataset_name not in valid_datasets:
        print('无效的数据集名称，请重新输入！')
        dataset_name = input(f'请输入要测试的数据集名称 ({", ".join(valid_datasets)}): ').strip()
    
    image_dir, mask_dir = get_dataset_paths(args.data_root, dataset_name)
    print(f'Testing on {dataset_name} dataset')
    print(f'Image directory: {image_dir}')
    print(f'Mask directory: {mask_dir}')
    
    use_tta = not args.no_tta
    if use_tta:
        print(f'TTA enabled with scales: {tta_scales}')
    else:
        print('TTA disabled')
    
    # 计算性能指标
    results = test(model, image_dir, mask_dir, device, args.img_size, dataset_name, 
                   use_tta=use_tta, tta_scales=tta_scales)
    
    print('\n' + '=' * 50)
    print(f'Results on {dataset_name}:')
    if use_tta:
        print(f'(with TTA, scales={tta_scales})')
    print('=' * 50)
    print(f'MAE:       {results["MAE"]:.4f}')
    print(f'F-measure: {results["F-measure"]:.4f}')
    print(f'S-measure: {results["S-measure"]:.4f}')
    print(f'E-measure: {results["E-measure"]:.4f}')
    print('=' * 50)


if __name__ == '__main__':
    main()
