import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
import glob
import numpy as np

# 数据集与模型
from geoprop.data.s3dis.s3dis_dataset import S3DISDataset 
from geoprop.models.point_jafar import DecoupledPointJAFAR
from geoprop.utils.metrics import IoUCalculator

# [核心修复] 避免循环引用！
# 不要写: from geoprop.core import ...
# 要写具体的子模块路径:
from geoprop.core.inferencer import validate_full_scene_logic 

def compute_gradient_boundary(xyz, rgb, k=16):
    B, C, N = xyz.shape
    xyz_t, rgb_t = xyz.transpose(1, 2), rgb.transpose(1, 2)
    idx = torch.randperm(N, device=xyz.device)[:1024]
    xyz_ctx, rgb_ctx = xyz_t[:, idx, :], rgb_t[:, idx, :]
    chunk = 2000; bdy_list = []
    for i in range(0, N, chunk):
        end = min(i+chunk, N)
        dist = torch.cdist(xyz_t[:, i:end, :], xyz_ctx)
        _, idx_k = torch.topk(dist, k, dim=-1, largest=False)
        batch_idx = torch.arange(B, device=xyz.device).view(B,1,1).expand(-1, end-i, k)
        xyz_n = xyz_ctx[batch_idx, idx_k, :]
        rgb_n = rgb_ctx[batch_idx, idx_k, :]
        geo_std = torch.std(xyz_n, dim=2).mean(dim=-1)
        col_std = torch.std(rgb_n, dim=2).mean(dim=-1)
        geo_std = (geo_std - geo_std.min()) / (geo_std.max() - geo_std.min() + 1e-6)
        col_std = (col_std - col_std.min()) / (col_std.max() - col_std.min() + 1e-6)
        bdy_list.append(torch.max(geo_std, col_std))
    return (torch.cat(bdy_list, dim=1) > 0.15).float()

def validate_block_proxy(model, cfg, val_loader):
    model.eval()
    evaluator = IoUCalculator()
    limit = cfg['train'].get('val_sample_batches', 100)
    
    with torch.no_grad():
        for i, (xyz, sft, _, lbl) in enumerate(val_loader):
            if i >= limit: break
            xyz, sft, lbl = xyz.cuda().transpose(1, 2), sft.cuda().transpose(1, 2), lbl.cuda()
            g_cpu = torch.Generator(); g_cpu.manual_seed(i)
            M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
            perm = torch.randperm(xyz.shape[2], generator=g_cpu)[:M].to(xyz.device)
            xyz_lr, sft_lr, lbl_lr = xyz[:,:,perm], sft[:,:,perm], lbl[:,perm]
            val_lr = torch.zeros(xyz.shape[0], 13, M).cuda().scatter_(1, lbl_lr.unsqueeze(1), 1.0)
            probs, _ = model(xyz, xyz_lr, sft_lr, val_lr)
            pred = torch.argmax(probs, dim=1)
            for b in range(xyz.shape[0]):
                evaluator.update(pred[b].cpu().numpy(), lbl[b].cpu().numpy())
    _, miou, _ = evaluator.compute()
    return miou

def run_training(cfg, save_path):
    logger = logging.getLogger("geoprop")
    logger.info(">>> [Phase 1] Training Started...")
    
    train_ds = S3DISDataset(cfg, split='train')
    val_ds = S3DISDataset(cfg, split='val')
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=4)
    
    lr = float(cfg['train']['learning_rate'])
    wd = float(cfg['train']['weight_decay'])
    
    model = DecoupledPointJAFAR(cfg['model']['qk_dim'], cfg['model']['k_neighbors']).cuda()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['train']['epochs'])
    crit_mse = nn.MSELoss(); crit_bce = nn.BCELoss()
    
    best_miou = 0.0
    val_mode = cfg['train'].get('val_mode', 'block_proxy')

    # 预加载全量文件列表 (仅当需要全图验证时)
    all_files = []
    if val_mode == 'full_scene':
        # 这里需要 Dataset 里的文件列表逻辑，或者重新 glob
        # 简单起见，我们重新初始化一个 inference split 的 dataset 拿文件
        inf_ds = S3DISDataset(cfg, split='inference')
        all_files = inf_ds.files

    for ep in range(cfg['train']['epochs']):
        model.train(); loss_acc = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg['train']['epochs']}", leave=False)
        
        for xyz, sft, rgb, _ in pbar:
            xyz, sft, rgb = xyz.cuda().transpose(1,2), sft.cuda().transpose(1,2), rgb.cuda().transpose(1,2)
            M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
            perm = torch.randperm(xyz.shape[2], device=xyz.device)[:M]
            opt.zero_grad()
            rec, bdy = model(xyz, xyz[:,:,perm], sft[:,:,perm], rgb[:,:,perm])
            loss = crit_mse(rec, rgb) + cfg['train']['loss_weights']['boundary'] * crit_bce(bdy, compute_gradient_boundary(xyz, rgb).unsqueeze(1))
            loss.backward(); opt.step(); loss_acc += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        sched.step()
        epoch_loss = loss_acc / len(train_loader)
        
        if (ep+1) % cfg['train']['val_interval'] == 0:
            if val_mode == 'full_scene':
                logger.info("  > Running Full Scene Validation (Slow)...")
                val_miou = validate_full_scene_logic(model, cfg, all_files)
            else:
                val_miou = validate_block_proxy(model, cfg, val_loader)
                
            logger.info(f"Epoch {ep+1:02d} | Loss: {epoch_loss:.4f} | Val ({val_mode}) mIoU: {val_miou*100:.2f}%")
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), save_path)
                logger.info(f"  --> Best Model Saved to {save_path}")
        else:
            logger.info(f"Epoch {ep+1:02d} | Loss: {epoch_loss:.4f}")
            
    return model