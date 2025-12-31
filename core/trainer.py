import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import logging
import numpy as np

from geoprop.data.s3dis.s3dis_dataset import S3DISDataset 
from geoprop.models.point_jafar import DecoupledPointJAFAR
from geoprop.utils.metrics import IoUCalculator
from geoprop.core.inferencer import validate_full_scene_logic 
from geoprop.core.modules.gblobs import compute_dual_gblobs 

def compute_gradient_boundary(xyz, rgb, k=16):
    """ Unsupervised boundary extraction """
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
    num_classes = cfg['dataset'].get('num_classes', 13)
    evaluator = IoUCalculator(num_classes)
    limit = cfg['train'].get('val_sample_batches', 100)
    
    with torch.no_grad():
        for i, (xyz, sft, rgb, lbl) in enumerate(val_loader):
            if i >= limit: break
            
            xyz = xyz.cuda().transpose(1, 2)
            rgb = rgb.cuda().transpose(1, 2)
            lbl = lbl.cuda()
            
            # Compute Dual Blobs for validation
            geo_blobs, rgb_blobs = compute_dual_gblobs(xyz, rgb, k=32)
            
            g_cpu = torch.Generator(); g_cpu.manual_seed(i)
            M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
            perm = torch.randperm(xyz.shape[2], generator=g_cpu)[:M].to(xyz.device)
            
            xyz_lr = xyz[:,:,perm]
            geo_blobs_lr = geo_blobs[:,:,perm]
            rgb_blobs_lr = rgb_blobs[:,:,perm]
            lbl_lr = lbl[:,perm]
            
            # Val Value: One-Hot Labels
            val_lr = torch.zeros(xyz.shape[0], num_classes, M).cuda().scatter_(1, lbl_lr.unsqueeze(1), 1.0)
            
            probs, _ = model(xyz, xyz_lr, val_lr, geo_blobs, geo_blobs_lr, rgb_blobs, rgb_blobs_lr)
            pred = torch.argmax(probs, dim=1)
            
            for b in range(xyz.shape[0]):
                evaluator.update(pred[b].cpu().numpy(), lbl[b].cpu().numpy())
                
    oa, miou, _ = evaluator.compute()
    return oa, miou

def run_training(cfg, save_path):
    logger = logging.getLogger("geoprop")
    logger.info(">>> [Phase 1] Training V31: Masked Loss + Dual GBlobs...")
    
    train_ds = S3DISDataset(cfg, split='train')
    val_ds = S3DISDataset(cfg, split='val')
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=4)
    
    lr = float(cfg['train']['learning_rate'])
    wd = float(cfg['train']['weight_decay'])
    
    # Input 18 dim
    model = DecoupledPointJAFAR(
        qk_dim=cfg['model']['qk_dim'], 
        k=cfg['model']['k_neighbors'],
        input_geo_dim=18
    ).cuda()
    
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['train']['epochs'])
    
    crit_mse = nn.MSELoss()
    crit_bce = nn.BCELoss()
    best_miou = 0.0
    val_mode = cfg['train'].get('val_mode', 'block_proxy')
    
    all_files = []
    if val_mode == 'full_scene':
        inf_ds = S3DISDataset(cfg, split='inference')
        all_files = inf_ds.files

    for ep in range(cfg['train']['epochs']):
        model.train(); loss_acc = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg['train']['epochs']}", leave=False)
        
        for xyz, sft, rgb, _ in pbar:
            xyz, sft, rgb = xyz.cuda().transpose(1,2), sft.cuda().transpose(1,2), rgb.cuda().transpose(1,2)
            
            # 1. Compute Features (Inputs: 18 dim relative)
            geo_blobs, rgb_blobs = compute_dual_gblobs(xyz, rgb, k=32)
            
            # 2. Masked Selection
            N = xyz.shape[2]
            M = max(int(N * cfg['dataset']['label_ratio']), 1)
            
            perm = torch.randperm(N, device=xyz.device)
            seed_indices = perm[:M]      # Known Seeds
            target_indices = perm[M:]    # Unknown Targets (Masked)
            
            # Key/Value (Seeds)
            xyz_seeds = xyz[:, :, seed_indices]
            geo_blobs_seeds = geo_blobs[:, :, seed_indices]
            rgb_blobs_seeds = rgb_blobs[:, :, seed_indices]
            
            # Value to Propagate (Input to Net): Absolute XYZ + RGB (6 dim)
            val_seeds_abs = torch.cat([xyz_seeds, rgb[:,:,seed_indices]], dim=1) 
            
            opt.zero_grad()
            
            # Forward
            rec_val, bdy = model(xyz, xyz_seeds, val_seeds_abs, geo_blobs, geo_blobs_seeds, rgb_blobs, rgb_blobs_seeds)
            
            # 3. Masked Loss Calculation
            # Only calculate MSE loss on points that were NOT seeds.
            # Target: Absolute XYZ + RGB
            gt_full = torch.cat([xyz, rgb], dim=1)
            
            rec_masked = rec_val[:, :, target_indices]
            gt_masked = gt_full[:, :, target_indices]
            
            loss_rec = crit_mse(rec_masked, gt_masked)
            
            # Boundary loss uses full scene (needs continuity)
            gt_bdy = compute_gradient_boundary(xyz, rgb).unsqueeze(1)
            loss_bdy = crit_bce(bdy, gt_bdy)
            
            loss = loss_rec + cfg['train']['loss_weights']['boundary'] * loss_bdy
            
            loss.backward(); opt.step(); loss_acc += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        sched.step()
        epoch_loss = loss_acc / len(train_loader)
        
        if (ep+1) % cfg['train']['val_interval'] == 0:
            if val_mode == 'full_scene':
                val_miou = validate_full_scene_logic(model, cfg, all_files)
                val_oa = 0.0
            else:
                val_oa, val_miou = validate_block_proxy(model, cfg, val_loader)
                
            logger.info(f"Epoch {ep+1:02d} | Loss: {epoch_loss:.4f} | Val mIoU: {val_miou*100:.2f}% | Val OA: {val_oa*100:.2f}%")
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), save_path)
                logger.info(f"  --> Best Model Saved")
        else:
            logger.info(f"Epoch {ep+1:02d} | Loss: {epoch_loss:.4f}")
            
    return model