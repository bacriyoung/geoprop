import numpy as np
import torch
import os
from tqdm import tqdm
import logging
from collections import OrderedDict

from geoprop.data.s3dis.s3dis_dataset import S3DISDataset
from geoprop.data.seeds import get_fixed_seeds
from geoprop.core.modules import confidence_filter, geometric_gating, graph_refine, spatial_smooth
from geoprop.core.modules.gblobs import compute_dual_gblobs  
from geoprop.utils.metrics import IoUCalculator, calc_local_metrics
from geoprop.utils.visualizer import generate_viz, CLASS_NAMES

def process_room_full_pipeline(cfg, model, data, return_all=False):
    N = len(data)
    xyz_full = data[:, :3]
    rgb_full = data[:, 3:6]
    lbl = data[:, 6].astype(int)
    num_classes = cfg['dataset'].get('num_classes', 13)
    
    # Normalize RGB
    rgb_norm = rgb_full/255.0 if rgb_full.max()>1.1 else rgb_full.copy()
    
    # [MEMORY FIX] Just prepare raw tensors. DO NOT compute global blobs here.
    full_xyz_tensor = torch.from_numpy(xyz_full).float().cuda().unsqueeze(0).transpose(1, 2)
    full_rgb_tensor = torch.from_numpy(rgb_norm).float().cuda().unsqueeze(0).transpose(1, 2)

    # 1. Handle Seeds
    seeds = get_fixed_seeds(N, cfg['dataset']['label_ratio'], lbl, cfg['project']['seed'])
    seed_lbls = lbl[seeds]
    valid_seed_mask = (seed_lbls >= 0) & (seed_lbls < num_classes)
    seeds = seeds[valid_seed_mask]
    
    # Sparse Seeds Data
    s_xyz = full_xyz_tensor[:, :, seeds]
    s_rgb = full_rgb_tensor[:, :, seeds]
    s_lbl = torch.from_numpy(lbl[seeds]).long().cuda().unsqueeze(0)
    
    with torch.no_grad():
        # Compute sparse blobs for seeds only
        s_geo_blobs, s_rgb_blobs = compute_dual_gblobs(s_xyz, s_rgb, k=min(32, len(seeds)))
    
    accum_probs = np.zeros((N, num_classes), dtype=np.float32)
    accum_counts = np.zeros(N, dtype=np.float32)
    lbl_direct = np.full(N, -100, dtype=int)
    lbl_filtered_sparse = np.full(N, -100, dtype=int)
    
    min_c, max_c = xyz_full.min(0), xyz_full.max(0)
    rounds = cfg['inference']['tta']['rounds'] if cfg['inference']['tta']['enabled'] else 1
    
    for t in range(rounds):
        if t == 0: aug_xyz = xyz_full
        else: aug_xyz = xyz_full + np.random.normal(0, cfg['inference']['tta']['jitter'], xyz_full.shape)
        
        # Sliding Window
        for x in np.arange(min_c[0], max_c[0], 1.5):
            for y in np.arange(min_c[1], max_c[1], 1.5):
                mask = np.where((aug_xyz[:,0]>=x) & (aug_xyz[:,0]<x+2.0) & 
                                (aug_xyz[:,1]>=y) & (aug_xyz[:,1]<y+2.0))[0]
                
                if len(mask) > 50:
                    # 2. Block Data
                    b_xyz = torch.from_numpy(aug_xyz[mask]).float().cuda().unsqueeze(0).transpose(1, 2)
                    b_rgb = full_rgb_tensor[:, :, mask]
                    
                    with torch.no_grad():
                        # [CRITICAL] Compute Blobs LOCALLY
                        b_geo_blobs, b_rgb_blobs = compute_dual_gblobs(b_xyz, b_rgb, k=32)
                        
                        # 3. Dynamic Seed Matching
                        dist = torch.cdist(b_xyz.mean(2, keepdim=True).transpose(1, 2), s_xyz.transpose(1, 2))
                        _, s_idx = torch.topk(dist, min(256, len(seeds)), dim=-1, largest=False)
                        s_idx = s_idx.view(-1)
                        
                        # Gather seeds
                        ls_xyz = s_xyz[:, :, s_idx]
                        ls_geo_blobs = s_geo_blobs[:, :, s_idx]
                        ls_rgb_blobs = s_rgb_blobs[:, :, s_idx]
                        
                        ls_val_sem = torch.zeros(1, num_classes, len(s_idx)).cuda()
                        ls_val_sem.scatter_(1, s_lbl[:, s_idx].unsqueeze(1), 1.0)
                        
                        # 4. Relative Pos
                        loc = ls_xyz - b_xyz.mean(2, keepdim=True)
                        cur = b_xyz - b_xyz.mean(2, keepdim=True)
                        
                        # 5. Forward
                        probs, _ = model(cur, loc, ls_val_sem, b_geo_blobs, ls_geo_blobs, b_rgb_blobs, ls_rgb_blobs)
                    
                    prob_np = probs.squeeze(0).transpose(0, 1).cpu().numpy()
                    accum_probs[mask] += prob_np
                    accum_counts[mask] += 1
                    
                    if t == 0:
                        pred_block = np.argmax(prob_np, axis=1)
                        lbl_direct[mask] = pred_block
                        lbl_filtered_sparse[mask] = pred_block

    # --- Post-processing ---
    
    valid_mask = accum_counts > 0
    
    lbl_tta = np.full(N, -100, dtype=int)
    if valid_mask.sum() > 0:
        lbl_tta[valid_mask] = np.argmax(accum_probs[valid_mask] / accum_counts[valid_mask, None], axis=1)
    
    lbl_tta = confidence_filter.knn_fill(xyz_full, lbl_tta)
    
    confidence_map = np.zeros(N, dtype=np.float32)
    if valid_mask.sum() > 0:
        safe_counts = accum_counts.copy()
        safe_counts[safe_counts == 0] = 1.0
        confidence_map = np.max(accum_probs / safe_counts[:, None], axis=1)
        confidence_map[~valid_mask] = 0.0

    result_stages = OrderedDict()
    result_stages["Direct Inference"] = lbl_direct
    if cfg['inference']['tta']['enabled']:
        result_stages["TTA Integration"] = lbl_tta
    
    current_lbl = lbl_tta if cfg['inference']['tta']['enabled'] else lbl_direct
    
    # Geometric Gating
    geom_cfg = cfg['inference']['geometric_gating']
    geom_cfg['num_classes'] = num_classes
    if geom_cfg['enabled']:
        lbl_geom = geometric_gating.run(geom_cfg, xyz_full, rgb_full, current_lbl, confidence=confidence_map)
        result_stages["Geometric Gating"] = lbl_geom
        current_lbl = lbl_geom

    # Graph Refine
    graph_cfg = cfg['inference']['graph_refine']
    graph_cfg['num_classes'] = num_classes
    if graph_cfg['enabled']:
        seed_mask_bool = np.zeros(N, dtype=bool); seed_mask_bool[seeds] = True
        # [FIX] Added gt_lbl=lbl argument correctly
        lbl_graph = graph_refine.run(graph_cfg, xyz_full, rgb_full, current_lbl, current_lbl, seed_mask=seed_mask_bool, gt_lbl=lbl, confidence=confidence_map)
        result_stages["Graph Refine"] = lbl_graph
        current_lbl = lbl_graph

    # Spatial Smooth
    if cfg['inference']['spatial_smooth']['enabled']:
        lbl_final = spatial_smooth.run(cfg['inference']['spatial_smooth'], xyz_full, current_lbl)
        result_stages["Spatial Smooth"] = lbl_final
        current_lbl = lbl_final

    if return_all:
        result_stages['GT'] = lbl
        return result_stages
    else:
        return current_lbl

def validate_full_scene_logic(model, cfg, all_files):
    model.eval()
    num_classes = cfg['dataset'].get('num_classes', 13)
    evaluator = IoUCalculator(num_classes)
    for f in all_files:
        data = np.load(f)
        res_dict = process_room_full_pipeline(cfg, model, data, return_all=True)
        stages = list(res_dict.keys())
        if 'GT' in stages: stages.remove('GT')
        pred = res_dict[stages[-1]]
        evaluator.update(pred, res_dict['GT'])
    _, miou, _ = evaluator.compute()
    return miou

def run_inference(cfg, model):
    logger = logging.getLogger("geoprop")
    logger.info(">>> [Phase 2] Final Inference & Generation...")
    model.eval()
    dataset = S3DISDataset(cfg, split='inference')
    files = dataset.files
    
    # [FIX] Define num_classes here!
    num_classes = cfg['dataset'].get('num_classes', 13)
    
    # Dummy run to get active stages
    dummy_res = process_room_full_pipeline(cfg, model, np.load(files[0]), return_all=True)
    all_active_stages = [k for k in dummy_res.keys() if k != 'GT']
    
    ablation_mode = cfg['inference'].get('ablation_mode', False)
    
    eval_stages = []  
    save_stages = []  
    
    stage_to_config_key = {
        "Direct Inference": "direct_inference",
        "TTA Integration": "tta",
        "Geometric Gating": "geometric_gating",
        "Graph Refine": "graph_refine",
        "Spatial Smooth": "spatial_smooth"
    }

    if not ablation_mode:
        if all_active_stages:
            final_stage = all_active_stages[-1]
            eval_stages = [final_stage]
            save_stages = [final_stage]
        logger.info(f"Mode: PRODUCTION. Processing final stage: {eval_stages}")
        cls_header = " | ".join([f"{c[:4]:<5}" for c in CLASS_NAMES])
        header = f"{'Room Name':<20} | {cls_header} | {'mIoU':<6}"
    else:
        eval_stages = all_active_stages
        for stage in all_active_stages:
            config_key = stage_to_config_key.get(stage)
            if config_key and cfg['inference'].get(config_key, {}).get('save_output', False):
                save_stages.append(stage)
            elif stage in ["TTA Integration", "Spatial Smooth"]: 
                save_stages.append(stage)
                
        logger.info(f"Mode: ABLATION.")
        logger.info(f"  > Logging metrics for: {eval_stages}")
        logger.info(f"  > Saving .npy for:     {save_stages}")
        
        for stage in save_stages:
            safe_dirname = stage.replace(" ", "_")
            stage_dir = os.path.join(cfg['paths']['npy'], safe_dirname)
            os.makedirs(stage_dir, exist_ok=True)
            
        header = f"{'Room Name':<20} | " + " | ".join([f"{s:<18}" for s in eval_stages])

    # Now num_classes is defined
    evals = {stage: IoUCalculator(num_classes) for stage in eval_stages}
    
    logger.info(header)
    logger.info("-" * len(header))
    
    for f in tqdm(files):
        room_name = os.path.basename(f)
        data = np.load(f)
        
        res = process_room_full_pipeline(cfg, model, data, return_all=True)
        gt = res['GT']
        
        if not ablation_mode:
            if eval_stages:
                stage = eval_stages[0]
                pred = res[stage]
                local_eval = IoUCalculator(num_classes)
                local_eval.update(pred, gt)
                _, miou, cls_ious = local_eval.compute()
                row_str = f"{room_name[:20]:<20} | " + " | ".join([f"{v*100:.1f}".center(5) for v in cls_ious]) + f" | {miou*100:.2f}".center(6)
                logger.info(row_str)
                evals[stage].update(pred, gt)
        else:
            ms_vals = []
            for stage in eval_stages:
                evals[stage].update(res[stage], gt)
                _, miou = calc_local_metrics(res[stage], gt, num_classes)
                ms_vals.append(miou)
            row_str = f"{room_name[:20]:<20} | " + " | ".join([f"{v:.4f}".center(18) for v in ms_vals])
            logger.info(row_str)
        
        if cfg['inference']['save_img']:
            viz_path = cfg['paths']['viz']
            viz_dict = {k: res[k] for k in eval_stages}
            generate_viz(data[:,:3], viz_dict, gt, room_name, viz_path)
        
        if cfg['inference']['save_npy']:
            npy_root = cfg['paths']['npy']
            for stage in save_stages:
                save_data = data.copy()
                save_data[:, 6] = res[stage]
                if ablation_mode:
                    safe_dirname = stage.replace(" ", "_")
                    save_path = os.path.join(npy_root, safe_dirname, room_name)
                else:
                    save_path = os.path.join(npy_root, room_name)
                np.save(save_path, save_data)

    logger.info("\n" + "="*100)
    logger.info("FINAL PER-CLASS EVALUATION (GLOBAL)")
    
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