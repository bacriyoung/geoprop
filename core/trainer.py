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
    weights = torch.ones(num_classes, device=labels.device)
    flat_lbl = labels.view(-1)
    counts = torch.bincount(flat_lbl, minlength=num_classes).float()
    total = counts.sum()
    valid_mask = counts > 0
    if valid_mask.sum() > 0:
        weights[valid_mask] = total / (counts[valid_mask] * valid_mask.sum())
        weights = torch.clamp(weights, 0.1, 10.0)
    return weights

def prepare_features(xyz, rgb, input_mode):
    if input_mode == "absolute":
        # xyz is already centered from dataset, but that's fine for features
        return torch.cat([xyz, rgb], dim=1) 
    elif input_mode == "gblobs":
        geo, col = compute_dual_gblobs(xyz, rgb, k=32)
        return torch.cat([geo, col], dim=1) 
    else:
        raise ValueError("Unknown input_mode")

def gather_points(tensor, indices):
    C = tensor.shape[1]
    return torch.gather(tensor, 2, indices.unsqueeze(1).expand(-1, C, -1))

def validate_block_proxy(model, cfg, val_loader):
    model.eval()
    num_classes = cfg['dataset'].get('num_classes', 13)
    evaluator = IoUCalculator(num_classes)
    limit = cfg['train'].get('val_sample_batches', 100)
    
    input_mode = cfg['model']['input_mode']
    fixed_val = cfg['train']['seed_mode']['val']
    
    with torch.no_grad():
        for i, (xyz, sft, rgb, lbl, seed_mask) in enumerate(val_loader):
            if i >= limit: break
            # xyz is CENTERED from dataset
            xyz = xyz.cuda().transpose(1, 2)
            rgb = rgb.cuda().transpose(1, 2)
            lbl = lbl.cuda()
            seed_mask = seed_mask.cuda()
            
            feat = prepare_features(xyz, rgb, input_mode)
            
            # [V2.0 LOGIC RESTORED]
            if not fixed_val:
                # Deterministic random seeds per block (Matches V2.0 exactly)
                g_cpu = torch.Generator(); g_cpu.manual_seed(i)
                M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
                # Generate same perm as V2.0
                perm = torch.randperm(xyz.shape[2], generator=g_cpu)[:M].to(xyz.device)
                seed_idx = perm.unsqueeze(0).expand(xyz.shape[0], -1) # Broad-cast or per batch?
                # Actually V2.0 did: perm = ...; xyz_lr = xyz[:,:,perm]
                # So it used same seeds for all batches in that step?
                # V2.0 code: perm = torch.randperm(...)[:M]; xyz_lr = xyz[:,:,perm]
                # Yes, same perm for the batch. Let's replicate.
                seed_idx = perm.unsqueeze(0).repeat(xyz.shape[0], 1)
            else:
                # [V3.0 Logic]
                M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
                idx_list = []
                for b in range(xyz.shape[0]):
                    idxs = torch.where(seed_mask[b])[0]
                    if len(idxs)==0: idxs = torch.tensor([0], device=xyz.device)
                    if len(idxs)>=M: idxs = idxs[:M]
                    else: idxs = torch.cat([idxs, idxs[0].repeat(M-len(idxs))])
                    idx_list.append(idxs)
                seed_idx = torch.stack(idx_list)

            B, M = seed_idx.shape
            
            # Gather
            xyz_source = feat[:, :3, :] if input_mode == "absolute" else xyz
            xyz_lr = gather_points(xyz_source, seed_idx)
            feat_lr = gather_points(feat, seed_idx)
            
            lbl_lr = torch.gather(lbl, 1, seed_idx)
            val_lr = torch.zeros(B, num_classes, M).cuda().scatter_(1, lbl_lr.unsqueeze(1), 1.0)
            
            probs, _ = model(xyz, xyz_lr, val_lr, feat, feat_lr)
            pred = torch.argmax(probs, dim=1)
            
            for b in range(B):
                evaluator.update(pred[b].cpu().numpy(), lbl[b].cpu().numpy())
                
    oa, miou, _ = evaluator.compute()
    return oa, miou

def run_training(cfg, save_path):
    logger = logging.getLogger("geoprop")
    
    train_ds = S3DISDataset(cfg, split='train')
    val_ds = S3DISDataset(cfg, split='val')
    train_loader = DataLoader(train_ds, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=8, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['train']['batch_size'], shuffle=False, num_workers=4)
    
    model = DecoupledPointJAFAR(cfg['model']['qk_dim'], cfg['model']['k_neighbors'], input_mode=cfg['model']['input_mode']).cuda()
    opt = optim.Adam(model.parameters(), lr=float(cfg['train']['learning_rate']), weight_decay=float(cfg['train']['weight_decay']))
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg['train']['epochs'])
    
    crit_mse = nn.MSELoss(reduction='none') 
    crit_bce = nn.BCELoss()
    best_miou = 0.0
    val_mode = cfg['train'].get('val_mode', 'block_proxy')
    
    input_mode = cfg['model']['input_mode']
    fixed_train = cfg['train']['seed_mode']['train']
    use_dyn_w = cfg['train']['use_dynamic_weights']

    all_files = []
    if val_mode == 'full_scene':
        inf_ds = S3DISDataset(cfg, split='inference')
        all_files = inf_ds.files

    for ep in range(cfg['train']['epochs']):
        model.train(); loss_acc = 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}", leave=False)
        
        for i, (xyz, sft, rgb, lbl, seed_mask) in enumerate(pbar):
            # xyz is CENTERED. rgb is normalized.
            xyz, sft, rgb, lbl = xyz.cuda().transpose(1,2), sft.cuda().transpose(1,2), rgb.cuda().transpose(1,2), lbl.cuda()
            seed_mask = seed_mask.cuda()
            
            feat = prepare_features(xyz, rgb, input_mode)
            
            # Seeds
            if fixed_train:
                # [V3.0] Mask based
                M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
                idx_list = []
                for b in range(xyz.shape[0]):
                    idxs = torch.where(seed_mask[b])[0]
                    if len(idxs)==0: idxs = torch.tensor([0], device=xyz.device)
                    if len(idxs)>=M: idxs = idxs[:M]
                    else: idxs = torch.cat([idxs, idxs[0].repeat(M-len(idxs))])
                    idx_list.append(idxs)
                seed_idx = torch.stack(idx_list)
            else:
                # [V2.0] Random Permutation per batch
                M = max(int(xyz.shape[2] * cfg['dataset']['label_ratio']), 1)
                seed_indices_list = [torch.randperm(xyz.shape[2], device=xyz.device)[:M] for _ in range(xyz.shape[0])]
                seed_idx = torch.stack(seed_indices_list)
            
            # Gather
            xyz_seeds = gather_points(xyz, seed_idx)
            rgb_seeds = gather_points(rgb, seed_idx)
            # V2.0 val_seeds_abs is simple cat
            val_seeds_abs = torch.cat([xyz_seeds, rgb_seeds], dim=1)
            
            feat_seeds = gather_points(feat, seed_idx)
            
            loss_weights_per_point = 1.0
            if use_dyn_w:
                lbl_seeds = torch.gather(lbl, 1, seed_idx)
                class_weights = compute_class_weights(lbl_seeds, cfg['dataset'].get('num_classes', 13))
                loss_weights_per_point = class_weights[lbl].unsqueeze(1)
            
            opt.zero_grad()
            rec_val, bdy = model(xyz, xyz_seeds, val_seeds_abs, feat, feat_seeds)
            
            gt_target = torch.cat([xyz, rgb], dim=1)
            mse_raw = crit_mse(rec_val, gt_target)
            loss_rec = (mse_raw * loss_weights_per_point).mean()
            
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