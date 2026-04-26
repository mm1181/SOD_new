import os
import math
import random
import argparse
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import SODModel
from losses import CombinedLoss
from datasets import get_dataloader, CutMix, split_dataset
from utils import AverageMeter, compute_mae, compute_fmeasure, save_checkpoint, multi_scale_inference


class EMA:
    """指数移动平均"""
    def __init__(self, model, decay=0.999, warmup_steps=100):
        self.model = model
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.shadow = {}
        self.backup = {}
        self._register()
    
    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        self.step_count += 1
        # 在 warmup 期间使用较低的 decay，让 EMA 更快跟上模型
        if self.step_count <= self.warmup_steps:
            decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        else:
            decay = self.decay
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用 EMA 权重（用于验证）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """恢复原始权重（验证后）"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name].clone()
        self.backup = {}
    
    def state_dict(self):
        return {'shadow': self.shadow, 'step_count': self.step_count}
    
    def load_state_dict(self, state_dict):
        if isinstance(state_dict, dict) and 'shadow' in state_dict:
            self.shadow = state_dict['shadow']
            self.step_count = state_dict.get('step_count', 0)
        else:
            self.shadow = state_dict


def parse_args():
    parser = argparse.ArgumentParser(description='SOD Training')
    parser.add_argument('--data_root', type=str, default='./data', help='Data root directory')
    parser.add_argument('--pvt_weights', type=str, default='./weights/model.safetensors', help='PVTv2 pretrained weights')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--accum_steps', type=int, default=2, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers')
    parser.add_argument('--early_stop', type=int, default=25, help='Early stopping patience')
    parser.add_argument('--cutmix_prob', type=float, default=0.3, help='CutMix probability')
    parser.add_argument('--ema_decay', type=float, default=0.999, help='EMA decay rate')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--multi_gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--ms_test', action='store_true', help='Enable multi-scale testing')
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, scheduler, device, epoch, writer, args, cutmix, ema):
    model.train()
     
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    optimizer.zero_grad()
    
    for i, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)
        
        if cutmix and torch.rand(1).item() < args.cutmix_prob:
            images, masks = cutmix(images, masks)
        
        with autocast():
            saliency_list, edge_1_4, edge_1_8 = model(images)
            loss, loss_dict = criterion(saliency_list, edge_1_4, edge_1_8, masks)
            loss = loss / args.accum_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % args.accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            
            if ema is not None:
                ema.update()
        
        saliency = saliency_list[0]
        pred = torch.sigmoid(saliency.detach())
        mae = compute_mae(pred, masks.detach())
        
        loss_meter.update(loss.item() * args.accum_steps, images.size(0))
        mae_meter.update(mae, images.size(0))
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'mae': f'{mae_meter.avg:.4f}'
        })
    
    # flush remaining gradients at end of epoch
    remaining = len(dataloader) % args.accum_steps
    if remaining != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
    
    assert scheduler.last_epoch + 1 <= scheduler.total_steps, \
        f"scheduler overflow: last_epoch={scheduler.last_epoch}, total_steps={scheduler.total_steps}"
    
    global_step = epoch * len(dataloader)
    writer.add_scalar('Train/Loss', loss_meter.avg, global_step)
    writer.add_scalar('Train/MAE', mae_meter.avg, global_step)
    
    return loss_meter.avg, mae_meter.avg


def validate(model, dataloader, criterion, device, epoch, writer, ema=None, ms_test=False):
    use_ema = ema is not None and epoch >= 2
    if use_ema:
        ema.apply_shadow()
    
    model.eval()
    
    loss_meter = AverageMeter()
    mae_meter = AverageMeter()
    fmeasure_meter = AverageMeter()
    mid_ratio_sum = 0.0
    mid_ratio_count = 0
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Validation'):
            images = images.to(device)
            masks = masks.to(device)
            
            if ms_test:
                pred = multi_scale_inference(model, images)
                with autocast():
                    saliency_list, edge_1_4, edge_1_8 = model(images)
                    loss, loss_dict = criterion(saliency_list, edge_1_4, edge_1_8, masks)
                loss_meter.update(loss.item(), images.size(0))
            else:
                with autocast():
                    saliency_list, edge_1_4, edge_1_8 = model(images)
                    loss, loss_dict = criterion(saliency_list, edge_1_4, edge_1_8, masks)
                loss_meter.update(loss.item(), images.size(0))
                saliency = saliency_list[0]
                pred = torch.sigmoid(saliency)
            
            mae = compute_mae(pred, masks)
            fmeasure = compute_fmeasure(pred, masks)
            
            mid = ((pred > 0.1) & (pred < 0.9)).float().mean().item()
            mid_ratio_sum += mid * images.size(0)
            mid_ratio_count += images.size(0)
            
            mae_meter.update(mae, images.size(0))
            fmeasure_meter.update(fmeasure, images.size(0))
    
    mid_ratio = mid_ratio_sum / max(mid_ratio_count, 1)
    
    if use_ema:
        ema.restore()
    
    writer.add_scalar('Val/Loss', loss_meter.avg, epoch)
    writer.add_scalar('Val/MAE', mae_meter.avg, epoch)
    writer.add_scalar('Val/F-measure', fmeasure_meter.avg, epoch)
    writer.add_scalar('Val/Mid_Prob_Ratio', mid_ratio, epoch)
    writer.add_histogram('Val/Pred_Prob', pred, epoch)
    
    return loss_meter.avg, mae_meter.avg, fmeasure_meter.avg, mid_ratio


def main():
    args = parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        print(f'Using {torch.cuda.device_count()} GPUs')
    
    model = SODModel(
        pretrained_resnet=True,
        pretrained_pvt_path=args.pvt_weights
    )
    
    if torch.cuda.device_count() > 1 and args.multi_gpu:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    ema = EMA(model, decay=args.ema_decay)
    print(f'EMA enabled with decay={args.ema_decay}')
    
    criterion = CombinedLoss(focal_weight=0.5, iou_weight=1.5, edge_weight=0.8, ssim_weight=0.5,
                             tversky_weight=2.5, label_smoothing=0.02)
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scaler = GradScaler()
    
    cutmix = CutMix(prob=args.cutmix_prob)
    
    train_image_dir = os.path.join(args.data_root, 'DUTS-TR', 'DUTS-TR-Image')
    train_mask_dir = os.path.join(args.data_root, 'DUTS-TR', 'DUTS-TR-Mask')

    train_files, val_files = split_dataset(train_image_dir, train_mask_dir, val_ratio=0.1)

    train_loader, train_dataset = get_dataloader(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=True,
        file_list=train_files
    )

    val_loader, _ = get_dataloader(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
        file_list=val_files
    )
    
    steps_per_epoch = math.ceil(len(train_loader) / args.accum_steps)
    total_steps = steps_per_epoch * args.epochs
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=5000.0,
        three_phase=False,
    )
    
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    start_epoch = 0
    best_mae = float('inf')
    best_fmeasure = 0.0
    patience_counter = 0
    smoothing_disabled = False
    
    if args.resume:
        if os.path.exists(args.resume):
            checkpoint = torch.load(args.resume)
            if args.multi_gpu:
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler'])
            if 'ema' in checkpoint:
                ema.load_state_dict(checkpoint['ema'])
            start_epoch = checkpoint['epoch']
            best_mae = checkpoint.get('best_mae', float('inf'))
            best_fmeasure = checkpoint.get('best_fmeasure', 0.0)
            smoothing_disabled = checkpoint.get('smoothing_disabled', False)
            if smoothing_disabled:
                criterion.focal_loss.smoothing = 0.0
            print(f"Loaded checkpoint from epoch {start_epoch}")
    
    print(f'Starting training for {args.epochs} epochs...')
    print(f'Training dataset: DUTS-TR ({len(train_loader.dataset)} samples)')
    print(f'Validation dataset: DUTS-TR-split ({len(val_loader.dataset)} samples)')
    print(f'Batch size: {args.batch_size}, Accum steps: {args.accum_steps}, Effective batch: {args.batch_size * args.accum_steps}')
    print(f'OneCycleLR: total_steps={total_steps}, steps_per_epoch={steps_per_epoch}, div_factor=10')
    
    for epoch in range(start_epoch, args.epochs):
        train_dataset.current_size = random.choice([256, 288, 320, 352, 384])
        print(f'Epoch {epoch}: Using image size {train_dataset.current_size}')
        
        criterion.set_epoch(epoch)
        
        train_loss, train_mae = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler, device, epoch, writer, args, cutmix, ema
        )
        
        val_loss, val_mae, val_fmeasure, mid_ratio = validate(
            model, val_loader, criterion, device, epoch, writer, ema, ms_test=True
        )
        
        # Auto-disable Label Smoothing if over-smoothed
        if not smoothing_disabled and epoch >= 2 and mid_ratio > 0.9:
            criterion.focal_loss.smoothing = 0.0
            smoothing_disabled = True
            print(f'  Mid-prob ratio {mid_ratio:.3f} > 0.9, Label Smoothing disabled for future epochs')
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Train/LR', current_lr, epoch)
        
        print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Train MAE={train_mae:.4f}, '
              f'Val Loss={val_loss:.4f}, Val MAE={val_mae:.4f}, Val F-measure={val_fmeasure:.4f}, LR={current_lr:.6f}')
        
        improved = False
        if val_mae < best_mae:
            if epoch <= 10:
                improved = True
            else:
                if val_fmeasure >= best_fmeasure * 0.985:
                    improved = True
                else:
                    print(f'  MAE improved ({best_mae:.4f} -> {val_mae:.4f}) but F-measure dropped below 98.5% threshold '
                          f'({val_fmeasure:.4f} < {best_fmeasure * 0.985:.4f}), skipping save')
        
        if improved:
            best_mae = val_mae
            if val_fmeasure > best_fmeasure:
                best_fmeasure = val_fmeasure
            patience_counter = 0
            
            use_ema_save = epoch >= 2
            if use_ema_save:
                ema.apply_shadow()
            state = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema': ema.state_dict(),
                'best_mae': best_mae,
                'best_fmeasure': best_fmeasure,
                'smoothing_disabled': smoothing_disabled,
            }
            save_checkpoint(state, os.path.join(args.output_dir, 'best_model.pth'))
            if use_ema_save:
                ema.restore()
            print(f'  New best MAE: {best_mae:.4f}, F-measure: {best_fmeasure:.4f}')
        else:
            if val_fmeasure > best_fmeasure:
                best_fmeasure = val_fmeasure
            patience_counter += 1
            if patience_counter >= args.early_stop:
                print(f'Early stopping at epoch {epoch}')
                break
        
        if (epoch + 1) % 10 == 0:
            use_ema_save = epoch >= 2
            if use_ema_save:
                ema.apply_shadow()
            state = {
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict() if args.multi_gpu else model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema': ema.state_dict(),
                'best_mae': best_mae,
                'best_fmeasure': best_fmeasure,
                'smoothing_disabled': smoothing_disabled,
            }
            save_checkpoint(state, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            if use_ema_save:
                ema.restore()
    
    writer.close()
    print(f'Training completed. Best MAE: {best_mae:.4f}, Best F-measure: {best_fmeasure:.4f}')


if __name__ == '__main__':
    main()
