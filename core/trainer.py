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

def compute_class_weights(labels, num_classes=13):
    """
    Compute dynamic weights based on inverse class frequency in the SEEDS.
    If a class is rare in seeds, its reconstruction loss weight goes UP.
    """
    # labels: [B, M] (Sparse seed labels)
    weights = torch.ones(num_classes, device=labels.device)
    
    # Flatten and count
    flat_lbl = labels.view(-1)
    counts = torch.bincount(flat_lbl, minlength=num_classes).float()
    
    # Avoid division by zero. If count is 0, weight is 0 (ignore class? or 1?)
    # Usually we want high weight for rare.
    # Logic: Weight = Total / (Count + epsilon)
    total = counts.sum()
    valid_mask = counts > 0
    
    if valid_mask.sum() > 0:
        weights[valid_mask] = total / (counts[valid_mask] * valid_mask.sum())
        # Clip extreme weights to avoid explosion
        weights = torch.clamp(weights, 0.1, 10.0)
    
    return weights

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
            
            # [V3.0 Input] Absolute XYZ + RGB
            feat = torch.cat([xyz, rgb], dim=1) # [B, 6, N]
            
            # Fixed seed for validation
            g_cpu = torch.Generator(); g_cpu.manual_seed(i)
            M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
            perm = torch.randperm(xyz.shape[2], generator=g_cpu)[:M].to(xyz.device)
            
            feat_lr = feat[:, :, perm]
            
            # Validation Value: Labels
            val_lr = torch.zeros(xyz.shape[0], num_classes, M).cuda().scatter_(1, lbl[:,perm].unsqueeze(1), 1.0)
            
            probs, _ = model(xyz, xyz[:,:,perm], val_lr, feat, feat_lr)
            pred = torch.argmax(probs, dim=1)
            
            for b in range(xyz.shape[0]):
                evaluator.update(pred[b].cpu().numpy(), lbl[b].cpu().numpy())
                
    oa, miou, _ = evaluator.compute()
    return oa, miou

def run_training(cfg, save_path):
    logger = logging.getLogger("geoprop")
    logger.info(">>> [Phase 1] Training V3.0: Fixed Seeds + Dynamic Weights...")
    
    train_ds = S3DISDataset(cfg, split='train')
    val_ds = S3DISDataset(cfg, split='val')
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=4)
    
    model = DecoupledPointJAFAR(cfg['model']['qk_dim'], cfg['model']['k_neighbors'], input_geo_dim=6).cuda()
    opt = optim.Adam(model.parameters(), lr=float(cfg['train']['learning_rate']), weight_decay=float(cfg['train']['weight_decay']))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['train']['epochs'])
    
    # Loss functions
    crit_mse = nn.MSELoss(reduction='none') # Need 'none' to apply weights
    crit_bce = nn.BCELoss()
    best_miou = 0.0
    val_mode = cfg['train'].get('val_mode', 'block_proxy')
    
    all_files = []
    if val_mode == 'full_scene':
        inf_ds = S3DISDataset(cfg, split='inference')
        all_files = inf_ds.files

    # Fixed seed generator for training (if enabled)
    fix_seeds = cfg['train'].get('fix_train_seeds', False)
    use_dyn_w = cfg['train'].get('use_dynamic_weights', False)
    
    for ep in range(cfg['train']['epochs']):
        model.train(); loss_acc = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}", leave=False)
        
        for i, (xyz, sft, rgb, lbl) in enumerate(pbar):
            xyz, sft, rgb, lbl = xyz.cuda().transpose(1,2), sft.cuda().transpose(1,2), rgb.cuda().transpose(1,2), lbl.cuda()
            
            # [V3.0] Absolute Features
            feat = torch.cat([xyz, rgb], dim=1) # [B, 6, N]
            
            N_pts = xyz.shape[2]
            M = max(int(N_pts * cfg['dataset']['label_ratio']), 1)
            
            # 1. Seed Selection Strategy
            if fix_seeds:
                # Deterministic seeds per batch/sample index
                # We use 'i' (batch idx) to seed the generator. 
                # This ensures every epoch, batch 'i' uses the SAME seeds.
                g_cuda = torch.Generator(device=xyz.device)
                g_cuda.manual_seed(i) 
                perm = torch.randperm(N_pts, generator=g_cuda, device=xyz.device)
            else:
                perm = torch.randperm(N_pts, device=xyz.device)
                
            seed_indices = perm[:M]
            target_indices = perm[M:]
            
            xyz_seeds = xyz[:, :, seed_indices]
            feat_seeds = feat[:, :, seed_indices]
            # Value: Absolute XYZ+RGB
            val_seeds_abs = torch.cat([xyz_seeds, rgb[:,:,seed_indices]], dim=1) 
            
            # 2. Dynamic Weights Calculation
            loss_weights_per_point = 1.0
            if use_dyn_w:
                # Get labels of seeds
                lbl_seeds = lbl[:, seed_indices] # [B, M]
                
                # Calculate class weights based on seed distribution
                # We calculate one set of weights per batch for stability
                class_weights = compute_class_weights(lbl_seeds, cfg['dataset'].get('num_classes', 13))
                
                # Assign weights to TARGET points based on THEIR gt class
                # "I am a chair point, and chairs are rare in seeds, so reconstruct me well!"
                lbl_target = lbl[:, target_indices] # [B, N-M]
                # Gather weights: [B, N-M]
                loss_weights_per_point = class_weights[lbl_target]
                # Unsqueeze for broadcasting over channels [B, 1, N-M]
                loss_weights_per_point = loss_weights_per_point.unsqueeze(1)

            opt.zero_grad()
            
            # Forward
            rec_val, bdy = model(xyz, xyz_seeds, val_seeds_abs, feat, feat_seeds)
            
            # 3. Weighted Reconstruction Loss
            rec_target = rec_val[:, :, target_indices]
            gt_target = torch.cat([xyz, rgb], dim=1)[:, :, target_indices]
            
            # MSE (unreduced) -> Apply Weights -> Mean
            mse_raw = crit_mse(rec_target, gt_target) # [B, 6, N-M]
            loss_rec = (mse_raw * loss_weights_per_point).mean()
            
            # Boundary Loss
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