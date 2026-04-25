import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class DUTSDataset(Dataset):
    def __init__(self, image_dir, mask_dir, size_list=[256, 288, 320, 352, 384], is_train=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.is_train = is_train
        self.size_list = size_list
        
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.current_size = random.choice(self.size_list) if is_train else 320

    def _get_mask_name(self, image_name):
        base_name = os.path.splitext(image_name)[0]
        possible_names = [
            f"{base_name}.png",
            f"{base_name}.jpg",
        ]
        
        for name in possible_names:
            mask_path = os.path.join(self.mask_dir, name)
            if os.path.exists(mask_path):
                return name
        
        mask_files = os.listdir(self.mask_dir)
        for mask_file in mask_files:
            mask_base = os.path.splitext(mask_file)[0]
            if mask_base in base_name or base_name in mask_base:
                return mask_file
        
        return None
    
    def __len__(self):
        return len(self.image_files)
    
    def _load_image_mask(self, idx):
        """加载图像和掩码"""
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('RGB')
        
        mask_name = self._get_mask_name(image_name)
        if mask_name is None:
            raise FileNotFoundError(f"Mask not found for image: {image_name}")
        
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path).convert('L')
        
        return image, mask
    
    def __getitem__(self, idx):
        image, mask = self._load_image_mask(idx)
        
        target_size = self.current_size
        
        # Copy-Paste 增强（概率 0.2）
        if self.is_train and random.random() < 0.2:
            # 随机选择另一张图像
            other_idx = random.randint(0, len(self.image_files) - 1)
            if other_idx != idx:
                try:
                    other_image, other_mask = self._load_image_mask(other_idx)
                    image, mask = self._copy_paste(image, mask, other_image, other_mask, target_size)
                except:
                    pass  # 如果加载失败，跳过 Copy-Paste
        
        image, mask = self._transform(image, mask, target_size)
        
        return image, mask
    
    def _copy_paste(self, image1, mask1, image2, mask2, target_size):
        """Copy-Paste 增强：将 image2 的前景复制到 image1 上"""
        # 调整到相同尺寸
        image1 = TF.resize(image1, (target_size, target_size))
        mask1 = TF.resize(mask1, (target_size, target_size), interpolation=Image.NEAREST)
        image2 = TF.resize(image2, (target_size, target_size))
        mask2 = TF.resize(mask2, (target_size, target_size), interpolation=Image.NEAREST)
        
        # 转换为 numpy
        img1_np = np.array(image1)
        mask1_np = np.array(mask1)
        img2_np = np.array(image2)
        mask2_np = np.array(mask2)
        
        # 获取 image2 的前景区域
        fg_mask = mask2_np > 127
        
        if fg_mask.sum() > 100:  # 确保有足够的前景像素
            # 随机缩放前景
            scale = random.uniform(0.5, 1.0)
            if scale < 1.0:
                new_size = int(target_size * scale)
                img2_resized = TF.resize(Image.fromarray(img2_np), (new_size, new_size))
                mask2_resized = TF.resize(Image.fromarray(mask2_np), (new_size, new_size), interpolation=Image.NEAREST)
                img2_np = np.array(img2_resized)
                mask2_np = np.array(mask2_resized)
                fg_mask = mask2_np > 127
            
            # 随机位置粘贴
            h2, w2 = img2_np.shape[:2]
            max_y = max(0, target_size - h2)
            max_x = max(0, target_size - w2)
            
            if max_y > 0 and max_x > 0:
                paste_y = random.randint(0, max_y)
                paste_x = random.randint(0, max_x)
                
                # 创建粘贴区域的掩码
                paste_region = np.zeros((target_size, target_size), dtype=bool)
                paste_region[paste_y:paste_y+h2, paste_x:paste_x+w2] = fg_mask
                
                # 粘贴前景
                for c in range(3):
                    img1_np[paste_y:paste_y+h2, paste_x:paste_x+w2, c] = np.where(
                        fg_mask, img2_np[:, :, c], img1_np[paste_y:paste_y+h2, paste_x:paste_x+w2, c]
                    )
                
                # 更新掩码
                mask1_np[paste_y:paste_y+h2, paste_x:paste_x+w2] = np.where(
                    fg_mask, mask2_np, mask1_np[paste_y:paste_y+h2, paste_x:paste_x+w2]
                )
        
        return Image.fromarray(img1_np), Image.fromarray(mask1_np)
    
    def _transform(self, image, mask, target_size):
        image = TF.resize(image, (target_size, target_size))
        mask = TF.resize(mask, (target_size, target_size), interpolation=Image.NEAREST)
        
        if self.is_train:
            # 水平翻转
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # 垂直翻转
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # 旋转
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)
            
            # 缩放
            if random.random() > 0.5:
                scale = random.uniform(0.5, 1.5)
                w, h = image.size
                new_w, new_h = int(w * scale), int(h * scale)
                image = TF.resize(image, (new_h, new_w))
                mask = TF.resize(mask, (new_h, new_w), interpolation=Image.NEAREST)
                
                if scale > 1.0:
                    i = random.randint(0, new_h - target_size)
                    j = random.randint(0, new_w - target_size)
                    image = TF.crop(image, i, j, target_size, target_size)
                    mask = TF.crop(mask, i, j, target_size, target_size)
                else:
                    image = TF.resize(image, (target_size, target_size))
                    mask = TF.resize(mask, (target_size, target_size), interpolation=Image.NEAREST)
            
            # 颜色抖动
            if random.random() > 0.5:
                brightness = random.uniform(0.5, 1.5)
                contrast = random.uniform(0.5, 1.5)
                saturation = random.uniform(0.5, 1.5)
                hue = random.uniform(-0.2, 0.2)
                
                image = TF.adjust_brightness(image, brightness)
                image = TF.adjust_contrast(image, contrast)
                image = TF.adjust_saturation(image, saturation)
                image = TF.adjust_hue(image, hue)
            
            # 高斯模糊
            if random.random() > 0.3:
                sigma = random.uniform(0.1, 2.0)
                image = TF.gaussian_blur(image, kernel_size=(5, 5), sigma=sigma)
            
            # 随机擦除（降低概率）
            if random.random() > 0.9:
                erase_area = random.uniform(0.05, 0.3)
                h, w = image.size[1], image.size[0]
                erase_h = int(h * erase_area)
                erase_w = int(w * erase_area)
                i = random.randint(0, h - erase_h)
                j = random.randint(0, w - erase_w)
                
                # 创建擦除区域
                erase_color = tuple(random.randint(0, 255) for _ in range(3))
                erase_image = Image.new('RGB', (erase_w, erase_h), erase_color)
                image.paste(erase_image, (j, i))
                
                # 同时擦除掩码
                erase_mask = Image.new('L', (erase_w, erase_h), 0)
                mask.paste(erase_mask, (j, i))
            
            # 随机灰度
            if random.random() > 0.7:
                image = TF.rgb_to_grayscale(image, num_output_channels=3)
            
            # 随机对比度增强
            if random.random() > 0.5:
                factor = random.uniform(0.5, 2.0)
                image = TF.adjust_contrast(image, factor)
        
        image = self.to_tensor(image)
        image = self.normalize(image)
        
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()
        
        return image, mask


class CutMix:
    def __init__(self, prob=0.3):
        self.prob = prob
    
    def __call__(self, images, masks):
        if random.random() > self.prob:
            return images, masks
        
        batch_size = images.shape[0]
        indices = torch.randperm(batch_size)
        
        lam = np.random.beta(1.0, 1.0)
        
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.shape[2:], lam)
        
        images[:, :, bbx1:bbx2, bby1:bby2] = images[indices, :, bbx1:bbx2, bby1:bby2]
        masks[:, :, bbx1:bbx2, bby1:bby2] = masks[indices, :, bbx1:bbx2, bby1:bby2]
        
        return images, masks
    
    def _rand_bbox(self, size, lam):
        H, W = size
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


def get_dataloader(image_dir, mask_dir, batch_size=16, num_workers=4, is_train=True, size_list=[256, 288, 320, 352, 384]):
    dataset = DUTSDataset(image_dir, mask_dir, size_list=size_list, is_train=is_train)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    
    return dataloader, dataset
