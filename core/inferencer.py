import numpy as np
import torch
import os
from tqdm import tqdm
import logging
from collections import OrderedDict

from geoprop.data.s3dis.s3dis_dataset import S3DISDataset
from geoprop.core.modules import confidence_filter, geometric_gating, graph_refine, spatial_smooth
from geoprop.core.modules.gblobs import compute_dual_gblobs  
from geoprop.utils.metrics import IoUCalculator, calc_local_metrics
from geoprop.utils.visualizer import generate_viz, CLASS_NAMES

def process_room_full_pipeline(cfg, model, data, return_all=False):
    N = len(data)
    xyz_full = data[:, :3]
    rgb_full = data[:, 3:6]
    lbl = data[:, 6].astype(int)
    
    # [FIX] Read num_classes from config
    num_classes = cfg['dataset'].get('num_classes', 13)
    input_mode = cfg['model']['input_mode']
    
    rgb_norm = rgb_full/255.0 if rgb_full.max()>1.1 else rgb_full.copy()
    
    full_xyz_tensor = torch.from_numpy(xyz_full).float().cuda().unsqueeze(0).transpose(1, 2)
    full_rgb_tensor = torch.from_numpy(rgb_norm).float().cuda().unsqueeze(0).transpose(1, 2)

    # --- Seed Hash ---
    h1 = np.abs(xyz_full[:, 0] * 73856093).astype(np.int64)
    h2 = np.abs(xyz_full[:, 1] * 19349663).astype(np.int64)
    h3 = np.abs(xyz_full[:, 2] * 83492791).astype(np.int64)
    seed_hash = h1 ^ h2 ^ h3
    
    label_ratio = cfg['dataset'].get('label_ratio', 0.001)
    threshold = int(label_ratio * 100000)
    is_seed = (seed_hash % 100000) < threshold
    
    global_seed_map = np.full(N, -1, dtype=int)
    global_seed_map[is_seed] = lbl[is_seed]
    valid_global_seeds = np.where(is_seed)[0]
    
    accum_probs = np.zeros((N, num_classes), dtype=np.float32)
    accum_counts = np.zeros(N, dtype=np.float32)
    lbl_direct = np.full(N, -100, dtype=int)
    
    min_c, max_c = xyz_full.min(0), xyz_full.max(0)
    rounds = cfg['inference']['tta']['rounds'] if cfg['inference']['tta']['enabled'] else 1
    
    for t in range(rounds):
        if t == 0: aug_xyz = xyz_full
        else: aug_xyz = xyz_full + np.random.normal(0, cfg['inference']['tta']['jitter'], xyz_full.shape)
        
        for x in np.arange(min_c[0], max_c[0], 1.5):
            for y in np.arange(min_c[1], max_c[1], 1.5):
                mask_indices = np.where((aug_xyz[:,0]>=x) & (aug_xyz[:,0]<x+2.0) & 
                                        (aug_xyz[:,1]>=y) & (aug_xyz[:,1]<y+2.0))[0]

                if len(mask_indices) > 50:
                    b_xyz = torch.from_numpy(aug_xyz[mask_indices]).float().cuda().unsqueeze(0).transpose(1, 2)
                    b_rgb = full_rgb_tensor[:, :, mask_indices]
                    
                    # [V3.1 Fix] Robust Seed Filtering for ScanNet Compatibility
                    # Filter out seeds with invalid labels (e.g., 255 or -100)
                    block_seed_labels = global_seed_map[mask_indices]
                    temp_rel_idx = np.where(block_seed_labels != -1)[0]
                    temp_seed_lbls = block_seed_labels[temp_rel_idx]
                    
                    # Keep only labels within [0, num_classes-1] to prevent CUDA errors
                    valid_range_mask = (temp_seed_lbls >= 0) & (temp_seed_lbls < num_classes)
                    rel_seed_idx = temp_rel_idx[valid_range_mask]
                    current_seed_lbls = temp_seed_lbls[valid_range_mask]
                    
                    if len(rel_seed_idx) == 0: continue
                    
                    # Cap max seeds to prevent OOM
                    if len(rel_seed_idx) > 512:
                        rng = np.random.RandomState(42 + len(mask_indices))
                        sub_choice = rng.choice(len(rel_seed_idx), 512, replace=False)
                        rel_seed_idx = rel_seed_idx[sub_choice]
                        current_seed_lbls = current_seed_lbls[sub_choice]
                    
                    # Prepare relative coordinates (Centering)
                    block_center = b_xyz.mean(2, keepdim=True)
                    loc = ls_xyz = b_xyz[:, :, rel_seed_idx] - block_center
                    cur = b_xyz - block_center
                    
                    with torch.no_grad():
                        # Feature Preparation
                        if input_mode == "absolute":
                            feat = torch.cat([cur, b_rgb], dim=1)
                        elif input_mode == "gblobs":
                            geo, col = compute_dual_gblobs(b_xyz, b_rgb, k=32)
                            feat = torch.cat([geo, col], dim=1)
                        elif input_mode == "mix":
                            # Use Geometric GBlobs (9 dims) + Raw RGB (3 dims)
                            geo, _ = compute_dual_gblobs(b_xyz, b_rgb, k=32)
                            feat = torch.cat([geo, b_rgb], dim=1)
                        
                        feat_seeds = feat[:, :, rel_seed_idx]
                        
                        ls_val_sem = torch.zeros(1, num_classes, len(rel_seed_idx)).cuda()
                        ls_val_sem.scatter_(1, torch.from_numpy(current_seed_lbls).long().cuda().unsqueeze(0).unsqueeze(1), 1.0)
                        
                        # [V3.1 Feature] Aggressive TTA: Rotation Voting
                        # Check config for aggressive mode
                        use_aggressive = cfg['inference'].get('tta', {}).get('aggressive', False)
                        # Only rotate on the first jitter round for efficiency
                        rot_angles = [0, 1, 2, 3] if (use_aggressive and t == 0) else [0]
                        
                        avg_probs = None
                        
                        for rot_idx in rot_angles:
                            cur_aug = cur.clone()
                            loc_aug = loc.clone()
                            
                            if rot_idx > 0:
                                # Apply 90-degree rotations around Z-axis
                                theta = rot_idx * np.pi / 2
                                c, s = np.cos(theta), np.sin(theta)
                                # [Fix] Ensure rot_mat has the same dtype (float32) as input tensor
                                rot_mat = torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]], 
                                                       device=cur.device, dtype=cur.dtype)
                                cur_aug = torch.matmul(cur_aug.transpose(1,2), rot_mat).transpose(1,2)
                                loc_aug = torch.matmul(loc_aug.transpose(1,2), rot_mat).transpose(1,2)
                            
                            p, _ = model(cur_aug, loc_aug, ls_val_sem, feat, feat_seeds)
                            
                            if avg_probs is None: avg_probs = p
                            else: avg_probs += p
                        
                        probs = avg_probs / len(rot_angles)
                    
                    prob_np = probs.squeeze(0).transpose(0, 1).cpu().numpy()
                    accum_probs[mask_indices] += prob_np
                    accum_counts[mask_indices] += 1
                    if t == 0: lbl_direct[mask_indices] = np.argmax(prob_np, axis=1)

    del full_xyz_tensor
    del full_rgb_tensor
    torch.cuda.empty_cache()

    valid_mask = accum_counts > 0
    lbl_tta = np.full(N, -100, dtype=int)
    if valid_mask.sum() > 0:
        lbl_tta[valid_mask] = np.argmax(accum_probs[valid_mask] / accum_counts[valid_mask, None], axis=1)
    lbl_tta = confidence_filter.knn_fill(xyz_full, lbl_tta)
    
    confidence_map = np.zeros(N, dtype=np.float32)
    if valid_mask.sum() > 0:
        confidence_map[valid_mask] = np.max(accum_probs[valid_mask] / np.maximum(accum_counts[valid_mask, None], 1.0), axis=1)

    result_stages = OrderedDict()
    result_stages["Direct Inference"] = lbl_direct
    if cfg['inference']['tta']['enabled']: result_stages["TTA Integration"] = lbl_tta
    current_lbl = lbl_tta
    
    # [Restored] Confidence Filter Logic
    if cfg['inference']['confidence_filter']['enabled']:
        # Assuming confidence_filter has a run method compatible with other modules
        # If strict reconstruction error filtering is needed
        current_lbl = confidence_filter.run(cfg['inference']['confidence_filter'], xyz_full, rgb_full, current_lbl, confidence=confidence_map)
        result_stages["Confidence Filter"] = current_lbl

    if cfg['inference']['geometric_gating']['enabled']:
        current_lbl = geometric_gating.run(cfg['inference']['geometric_gating'], xyz_full, rgb_full, current_lbl, confidence=confidence_map)
        result_stages["Geometric Gating"] = current_lbl
        
    if cfg['inference']['graph_refine']['enabled']:
        seed_mask_bool = np.zeros(N, dtype=bool); seed_mask_bool[valid_global_seeds] = True
        current_lbl = graph_refine.run(cfg['inference']['graph_refine'], xyz_full, rgb_full, current_lbl, current_lbl, seed_mask=seed_mask_bool, gt_lbl=lbl, confidence=confidence_map)
        result_stages["Graph Refine"] = current_lbl
        
    if cfg['inference']['spatial_smooth']['enabled']:
        current_lbl = spatial_smooth.run(cfg['inference']['spatial_smooth'], xyz_full, current_lbl)
        result_stages["Spatial Smooth"] = current_lbl

    if return_all:
        result_stages['GT'] = lbl
        return result_stages
    else:
        return current_lbl

def validate_full_scene_logic(model, cfg, all_files):
    model.eval()
    num_classes = 13
    evaluator = IoUCalculator(num_classes)
    for f in all_files:
        data = np.load(f)
        res_dict = process_room_full_pipeline(cfg, model, data, return_all=True)
        pred = list(res_dict.values())[-2]
        evaluator.update(pred, res_dict['GT'])
    _, miou, _ = evaluator.compute()
    return miou

def run_inference(cfg, model):
    logger = logging.getLogger("geoprop")
    logger.info(">>> [Phase 2] Final Inference (V3.0 Compatible)...")
    model.eval()
    dataset = S3DISDataset(cfg, split='inference')
    
    # [FIX] Read num_classes from config
    num_classes = cfg['dataset'].get('num_classes', 13)
    
    if len(dataset.files) > 0:
        dummy_res = process_room_full_pipeline(cfg, model, np.load(dataset.files[0]), return_all=True)
        all_stages = [k for k in dummy_res.keys() if k != 'GT']
    else: all_stages = []
    
    ablation_mode = cfg['inference'].get('ablation_mode', False)
    eval_stages = all_stages if ablation_mode else ([all_stages[-1]] if all_stages else [])
    
    # Define all possible stages to ensure they appear in the log
    full_stage_map = OrderedDict([
        ("Direct Inference", "direct_inference"),
        ("TTA Integration", "tta"),
        ("Confidence Filter", "confidence_filter"), # [Added]
        ("Geometric Gating", "geometric_gating"),
        ("Graph Refine", "graph_refine"),
        ("Spatial Smooth", "spatial_smooth")
    ])

    logger.info("-" * 80)
    logger.info("Inference Pipeline Configuration Summary:")
    
    # Iterate over ALL known stages, not just the enabled ones
    for stage_name, config_key in full_stage_map.items():
        is_enabled = False
        
        # Check 'enabled' status in config
        if config_key == "direct_inference":
            is_enabled = True # Always enabled
        else:
            is_enabled = cfg['inference'].get(config_key, {}).get('enabled', False)
            
        status_info = "Enabled" if is_enabled else "Disabled"
        
        # Check Save Status
        save_info = ""
        if is_enabled:
            stage_cfg = cfg['inference'].get(config_key, {})
            if cfg['inference']['save_npy'] and stage_cfg.get('save_output', False):
                save_info = "| [NPY SAVED]"
            else:
                save_info = "| [No Save]"
        else:
            save_info = "" # Don't show save info if disabled

        logger.info(f"  > {stage_name:<20} : {status_info} {save_info}")
    logger.info("-" * 80)
    
    # [Crucial] Update stage_key_map for later usage in saving/metrics
    stage_key_map = full_stage_map

    evals = {stage: IoUCalculator(num_classes) for stage in eval_stages}
    
    if cfg['inference']['save_npy'] and ablation_mode:
        for stage in eval_stages:
            key = stage_key_map.get(stage)
            if key and cfg['inference'].get(key, {}).get('save_output', False):
                os.makedirs(os.path.join(cfg['paths']['npy'], stage.replace(" ", "_")), exist_ok=True)
            
    # [Log Header Logic]
    # Check if we are in ablation mode or production mode
    if ablation_mode:
        # Ablation mode: Show mIoU for each stage column
        header = f"{'Room Name':<20} | " + " | ".join([f"{s:<18}" for s in eval_stages])
    else:
        # Production mode: Show per-class IoU (Horizontal) + mIoU + OA
        # Truncate class names to 6 chars for compactness to fit in one line
        cls_headers = " | ".join([f"{c[:6]:<6}" for c in CLASS_NAMES]) 
        header = f"{'Room Name':<20} | {cls_headers} | {'mIoU':<6} | {'OA':<6}"
    
    logger.info("-" * len(header))
    logger.info(header)
    logger.info("-" * len(header))
    
    for f in tqdm(dataset.files):
        room_name = os.path.basename(f)
        data = np.load(f)
        res = process_room_full_pipeline(cfg, model, data, return_all=True)
        
        # [Log Row Logic]
        if ablation_mode:
            # Ablation mode: Original logic (calculate mIoU for each stage)
            ious = []
            for stage in eval_stages:
                evals[stage].update(res[stage], res['GT'])
                _, miou = calc_local_metrics(res[stage], res['GT'], num_classes)
                ious.append(miou)
            
            row_str = f"{room_name[:20]:<20} | " + " | ".join([f"{v:.4f}".center(18) for v in ious])
            logger.info(row_str)
            
        else:
            # Production mode: Calculate and print detailed class metrics for the single stage
            if len(eval_stages) > 0:
                stage = eval_stages[-1] # Get the last (and only) stage
                
                # Update global evaluator
                evals[stage].update(res[stage], res['GT'])
                
                # Calculate temporary metrics for the current room
                room_eval = IoUCalculator(num_classes)
                room_eval.update(res[stage], res['GT'])
                oa, miou, cls_ious = room_eval.compute()
                
                # Format: Room Name | Class IoUs | mIoU | OA
                cls_str = " | ".join([f"{v*100:.1f}".center(6) for v in cls_ious])
                row_str = f"{room_name[:20]:<20} | {cls_str} | {miou*100:.1f}% | {oa*100:.1f}%"
                logger.info(row_str)
        
        if cfg['inference']['save_img']:
            generate_viz(data[:,:3], {k:res[k] for k in eval_stages}, res['GT'], room_name, cfg['paths']['viz'])
            
        if cfg['inference']['save_npy']:
            for stage in eval_stages:
                key = stage_key_map.get(stage)
                if key and cfg['inference'].get(key, {}).get('save_output', False):
                    save_data = data.copy(); save_data[:, 6] = res[stage]
                    path = os.path.join(cfg['paths']['npy'], stage.replace(" ", "_"), room_name) if ablation_mode else os.path.join(cfg['paths']['npy'], room_name)
                    np.save(path, save_data)
                
    logger.info("\n" + "="*100)
    logger.info("FINAL PER-CLASS EVALUATION")
    
    ious_dict = {}
    miou_dict = {}
    oa_dict = {}
    
    for stage in eval_stages:
        oa, miou, ious = evals[stage].compute()
        ious_dict[stage] = ious
        miou_dict[stage] = miou
        oa_dict[stage] = oa
    
    header_row = f"{'Class':<12} | " + " | ".join([f"{s:<18}" for s in eval_stages])
    logger.info(header_row)
    logger.info("-" * len(header_row))
    
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:<12} | " + " | ".join([f"{ious_dict[s][i]*100:.2f}".center(18) for s in eval_stages])
        logger.info(row)
        
    logger.info("-" * len(header_row))
    logger.info(f"{'mIoU':<12} | " + " | ".join([f"{miou_dict[s]*100:.2f}".center(18) for s in eval_stages]))
    logger.info(f"{'OA':<12}   | " + " | ".join([f"{oa_dict[s]*100:.2f}".center(18) for s in eval_stages]))
    logger.info("="*100)