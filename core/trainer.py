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
    """ Pre-compute boundary based on color/geo gradients for supervision """
    B, C, N = xyz.shape
    xyz_t, rgb_t = xyz.transpose(1, 2), rgb.transpose(1, 2)
    # Subsample context points for efficiency
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
        
        # Normalize to 0-1
        geo_std = (geo_std - geo_std.min()) / (geo_std.max() - geo_std.min() + 1e-6)
        col_std = (col_std - col_std.min()) / (col_std.max() - col_std.min() + 1e-6)
        
        bdy_list.append(torch.max(geo_std, col_std))
        
    return (torch.cat(bdy_list, dim=1) > 0.15).float()

def validate_block_proxy(model, cfg, val_loader):
    """
    Validation on blocks. 
    Crucially, we evaluate on DENSE labels here to ensure reconstruction quality.
    """
    model.eval()
    # [FIX] Initialize evaluator with correct number of classes
    num_classes = cfg['dataset'].get('num_classes', 13)
    evaluator = IoUCalculator(num_classes)
    limit = cfg['train'].get('val_sample_batches', 100)
    
    with torch.no_grad():
        for i, (xyz, sft, _, lbl) in enumerate(val_loader):
            if i >= limit: break
            
            xyz, sft, lbl = xyz.cuda().transpose(1, 2), sft.cuda().transpose(1, 2), lbl.cuda()
            
            # Use deterministic seed for validation to be fair
            g_cpu = torch.Generator(); g_cpu.manual_seed(i)
            M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
            
            # Select Sparse Seeds (Simulating the input condition)
            perm = torch.randperm(xyz.shape[2], generator=g_cpu)[:M].to(xyz.device)
            
            xyz_lr, sft_lr, lbl_lr = xyz[:,:,perm], sft[:,:,perm], lbl[:,perm]
            
            # Encode sparse labels
            val_lr = torch.zeros(xyz.shape[0], num_classes, M).cuda().scatter_(1, lbl_lr.unsqueeze(1), 1.0)
            
            # Predict Dense
            probs, _ = model(xyz, xyz_lr, sft_lr, val_lr)
            pred = torch.argmax(probs, dim=1)
            
            # Update Evaluator (Dense Prediction vs Dense GT)
            for b in range(xyz.shape[0]):
                evaluator.update(pred[b].cpu().numpy(), lbl[b].cpu().numpy())
    
    # [FIX] Unpack 3 values: OA, mIoU, IoUs
    oa, miou, _ = evaluator.compute()
    
    # Return both for logging, but trainer usually needs one scaler for comparison
    return oa, miou

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
    
    crit_mse = nn.MSELoss()
    crit_bce = nn.BCELoss()
    
    best_miou = 0.0
    val_mode = cfg['train'].get('val_mode', 'block_proxy')

    # Full Scene Validation Setup (if enabled)
    all_files = []
    if val_mode == 'full_scene':
        inf_ds = S3DISDataset(cfg, split='inference')
        all_files = inf_ds.files

    for ep in range(cfg['train']['epochs']):
        model.train()
        loss_acc = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{cfg['train']['epochs']}", leave=False)
        
        for xyz, sft, rgb, _ in pbar:
            xyz, sft, rgb = xyz.cuda().transpose(1,2), sft.cuda().transpose(1,2), rgb.cuda().transpose(1,2)
            
            # Random sparse seeds for training
            M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
            perm = torch.randperm(xyz.shape[2], device=xyz.device)[:M]
            
            opt.zero_grad()
            rec, bdy = model(xyz, xyz[:,:,perm], sft[:,:,perm], rgb[:,:,perm])
            
            # Loss: Reconstruction + Boundary
            loss = crit_mse(rec, rgb) + cfg['train']['loss_weights']['boundary'] * crit_bce(bdy, compute_gradient_boundary(xyz, rgb).unsqueeze(1))
            
            loss.backward()
            opt.step()
            loss_acc += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        sched.step()
        epoch_loss = loss_acc / len(train_loader)
        
        # Validation Logic
        if (ep+1) % cfg['train']['val_interval'] == 0:
            if val_mode == 'full_scene':
                logger.info("  > Running Full Scene Validation (Slow)...")
                # Assume full_scene validation returns mIoU
                val_miou = validate_full_scene_logic(model, cfg, all_files)
                val_oa = 0.0 # Placeholder
            else:
                val_oa, val_miou = validate_block_proxy(model, cfg, val_loader)
                
            # Log both metrics
            logger.info(f"Epoch {ep+1:02d} | Loss: {epoch_loss:.4f} | Val mIoU: {val_miou*100:.2f}% | Val OA: {val_oa*100:.2f}%")
            
            # [CRITICAL] Selection logic based on mIoU (Balanced Performance)
            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(model.state_dict(), save_path)
                logger.info(f"  --> New Best Model Saved (mIoU: {best_miou:.4f})")
        else:
            logger.info(f"Epoch {ep+1:02d} | Loss: {epoch_loss:.4f}")
            
    return model