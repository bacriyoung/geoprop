import numpy as np
import torch
import os
import glob
from tqdm import tqdm
import logging
from collections import OrderedDict

from geoprop.data.s3dis.s3dis_dataset import S3DISDataset
from geoprop.data.seeds import get_fixed_seeds
from geoprop.core.modules import confidence_filter, geometric_gating, graph_refine, spatial_smooth
from geoprop.utils.metrics import IoUCalculator, calc_local_miou
from geoprop.utils.visualizer import generate_viz, CLASS_NAMES

def process_room_full_pipeline(cfg, model, data, return_all=False):
    """
    Complete Inference Pipeline.
    Sequential processing: Inference -> TTA -> Confidence Filter -> Geo Gating -> Graph Refine -> Spatial Smooth.
    Only enabled stages are added to result_stages.
    """
    N = len(data)
    xyz_full = data[:, :3]
    rgb_full = data[:, 3:6]
    lbl = data[:, 6].astype(int)
    
    num_classes = cfg['dataset'].get('num_classes', 13)
    
    # 1. Get Fixed Seeds
    seeds = get_fixed_seeds(N, cfg['dataset']['label_ratio'], lbl, cfg['project']['seed'])
    seed_mask = np.zeros(N, dtype=bool); seed_mask[seeds] = True
    
    # Prepare Tensor Data
    rgb_norm = rgb_full/255.0 if rgb_full.max()>1.1 else rgb_full.copy()
    s_xyz = torch.from_numpy(data[seeds, :3]).float().cuda().unsqueeze(0)
    
    rgb_seeds = data[seeds, 3:6].copy()
    rgb_seeds = rgb_seeds/255.0 if rgb_seeds.max()>1.1 else rgb_seeds
    s_rgb = torch.from_numpy(rgb_seeds).float().cuda().unsqueeze(0)
    s_lbl = torch.from_numpy(data[seeds, 6].astype(int)).long().cuda().unsqueeze(0)
    
    # 2. Base Inference Loop (Direct & TTA)
    accum_probs = np.zeros((N, num_classes), dtype=np.float32)
    accum_counts = np.zeros(N, dtype=np.float32)
    lbl_direct = np.full(N, -100, dtype=int)
    lbl_filtered_sparse = np.full(N, -100, dtype=int)
    
    min_c, max_c = xyz_full.min(0), xyz_full.max(0)
    rounds = cfg['inference']['tta']['rounds'] if cfg['inference']['tta']['enabled'] else 1
    
    for t in range(rounds):
        # Apply jitter only if rounds > 1 and t > 0
        if t == 0: aug_xyz = xyz_full
        else: aug_xyz = xyz_full + np.random.normal(0, cfg['inference']['tta']['jitter'], xyz_full.shape)
        
        for x in np.arange(min_c[0], max_c[0], 1.0):
            for y in np.arange(min_c[1], max_c[1], 1.0):
                mask = np.where((aug_xyz[:,0]>=x) & (aug_xyz[:,0]<x+2) & 
                                (aug_xyz[:,1]>=y) & (aug_xyz[:,1]<y+2))[0]
                if len(mask) > 50:
                    b_xyz = torch.from_numpy(aug_xyz[mask]).float().cuda().unsqueeze(0)
                    b_rgb = torch.from_numpy(rgb_norm[mask]).float().cuda().unsqueeze(0)
                    dist = torch.cdist(b_xyz.mean(1, keepdim=True), s_xyz)
                    _, idx = torch.topk(dist, min(256, len(seeds)), dim=-1, largest=False)
                    idx = idx.view(-1)
                    
                    loc = (s_xyz[:,idx,:] - b_xyz.mean(1, keepdim=True)).transpose(1,2)
                    cur = (b_xyz - b_xyz.mean(1, keepdim=True)).transpose(1,2)
                    norm = (loc - cur.min(2,keepdim=True)[0]) / (cur.max(2,keepdim=True)[0] - cur.min(2,keepdim=True)[0] + 1e-6)
                    val = torch.zeros(1, num_classes, len(idx)).cuda().scatter_(1, s_lbl[:,idx].unsqueeze(1), 1.0)
                    
                    with torch.no_grad():
                        rec, bdy = model(cur, loc, torch.cat([s_rgb[:,idx].transpose(1,2), norm],1), s_rgb[:,idx].transpose(1,2))
                        probs, _ = model(cur, loc, torch.cat([s_rgb[:,idx].transpose(1,2), norm],1), val)
                    
                    prob_np = probs.squeeze().transpose(0,1).cpu().numpy()
                    accum_probs[mask] += prob_np
                    accum_counts[mask] += 1
                    
                    if t == 0:
                        pred_block = np.argmax(prob_np, axis=1)
                        # Confidence Filter S1 (Sparse)
                        lbl_direct[mask] = pred_block
                        lbl_filtered_sparse = confidence_filter.run(cfg['inference']['confidence_filter'], 
                                                                    rec, bdy, b_rgb, 
                                                                    pred_block, mask, lbl_filtered_sparse)

    # Fill sparse filtered labels to dense
    lbl_filtered_filled = confidence_filter.knn_fill(xyz_full, lbl_filtered_sparse.copy())

    # TTA Aggregation
    valid_tta = accum_counts > 0
    lbl_tta = np.full(N, -100, dtype=int)
    lbl_tta[valid_tta] = np.argmax(accum_probs[valid_tta]/accum_counts[valid_tta, None], axis=1)
    lbl_tta = confidence_filter.knn_fill(xyz_full, lbl_tta)

    # Compute Global Confidence Map
    confidence_map = np.zeros(N, dtype=np.float32)
    confidence_map[valid_tta] = np.max(accum_probs[valid_tta] / accum_counts[valid_tta, None], axis=1)

    # 3. Dynamic Result Construction
    # Only enabled stages are added to the dictionary.
    result_stages = OrderedDict()
    
    # Direct Inference is always present
    result_stages["Direct Inference"] = lbl_direct
    lbl_current = lbl_direct

    if cfg['inference']['confidence_filter']['enabled']:
        result_stages["Confidence Filter"] = lbl_filtered_filled
        lbl_current = lbl_filtered_filled

    if cfg['inference']['tta']['enabled']:
        result_stages["TTA Integration"] = lbl_tta
        lbl_current = lbl_tta

    # --- Geometric Gating ---
    geom_cfg = cfg['inference']['geometric_gating']
    geom_cfg['num_classes'] = num_classes
    
    lbl_geom = geometric_gating.run(geom_cfg, xyz_full, rgb_full, lbl_current, confidence=confidence_map)
    
    if geom_cfg['enabled']:
        result_stages["Geometric Gating"] = lbl_geom
        shape_guide = lbl_geom
        lbl_current = lbl_geom
    else:
        # If disabled, current labels guide the next step
        shape_guide = lbl_current

    # --- Graph Refine ---
    graph_cfg = cfg['inference']['graph_refine']
    graph_cfg['num_classes'] = num_classes
    
    lbl_graph = graph_refine.run(graph_cfg, xyz_full, rgb_full, lbl_current, shape_guide, seed_mask, lbl, confidence=confidence_map)
    
    if graph_cfg['enabled']:
        result_stages["Graph Refine"] = lbl_graph
        lbl_current = lbl_graph

    # --- Spatial Smooth ---
    lbl_final = spatial_smooth.run(cfg['inference']['spatial_smooth'], xyz_full, lbl_current)
    if cfg['inference']['spatial_smooth']['enabled']:
        result_stages["Spatial Smooth"] = lbl_final
        lbl_current = lbl_final

    if return_all:
        result_stages['GT'] = lbl
        return result_stages
    else:
        return lbl_current

def validate_full_scene_logic(model, cfg, all_files):
    model.eval()
    evaluator = IoUCalculator(num_classes=cfg['dataset'].get('num_classes', 13))
    for f in all_files:
        data = np.load(f)
        # Validation logic: prioritize Confidence Filter if available
        res_dict = process_room_full_pipeline(cfg, model, data, return_all=True)
        if "Confidence Filter" in res_dict:
            pred = res_dict["Confidence Filter"]
        else:
            pred = res_dict["Direct Inference"]
        evaluator.update(pred, res_dict['GT'])
    _, miou = evaluator.compute()
    return miou

def run_inference(cfg, model):
    logger = logging.getLogger("geoprop")
    logger.info(">>> [Phase 2] Final Inference & Generation...")
    model.eval()
    
    dataset = S3DISDataset(cfg, split='inference')
    files = dataset.files
    num_classes = cfg['dataset'].get('num_classes', 13)
    
    # 1. Determine active stages by running a dummy pass
    dummy_res = process_room_full_pipeline(cfg, model, np.load(files[0]), return_all=True)
    all_active_stages = [k for k in dummy_res.keys() if k != 'GT']
    
    ablation_mode = cfg['inference'].get('ablation_mode', False)
    
    # Separate lists for Logging (Eval) and Saving (IO)
    eval_stages = []  
    save_stages = []  
    
    # [Fix] Mapped Direct Inference to its new config key
    stage_to_config_key = {
        "Direct Inference": "direct_inference",  # <--- FIXED HERE
        "Confidence Filter": "confidence_filter",
        "TTA Integration": "tta",
        "Geometric Gating": "geometric_gating",
        "Graph Refine": "graph_refine",
        "Spatial Smooth": "spatial_smooth"
    }

    if not ablation_mode:
        # --- Production Mode ---
        # Eval & Save: Only final active stage
        if all_active_stages:
            final_stage = all_active_stages[-1]
            eval_stages = [final_stage]
            save_stages = [final_stage]
        
        logger.info(f"Mode: PRODUCTION. Processing final stage: {eval_stages}")
        
        # Build header for per-class metrics
        cls_header = " | ".join([f"{c[:4]:<5}" for c in CLASS_NAMES])
        header = f"{'Room Name':<20} | {cls_header} | {'mIoU':<6}"
        
    else:
        # --- Ablation Mode ---
        # Eval: ALL active stages
        eval_stages = all_active_stages
        
        # Save: Only if 'save_output' is True in config
        for stage in all_active_stages:
            config_key = stage_to_config_key.get(stage)
            
            # Now Direct Inference will hit this logic correctly
            if config_key and cfg['inference'].get(config_key):
                if cfg['inference'][config_key].get('save_output', False):
                    save_stages.append(stage)
            
        logger.info(f"Mode: ABLATION.")
        logger.info(f"  > Logging metrics for: {eval_stages}")
        logger.info(f"  > Saving .npy for:     {save_stages}")
        
        # Prepare subdirectories for SAVE stages only
        for stage in save_stages:
            safe_dirname = stage.replace(" ", "_")
            stage_dir = os.path.join(cfg['paths']['npy'], safe_dirname)
            os.makedirs(stage_dir, exist_ok=True)
            
        # Build header for stage comparison
        header = f"{'Room Name':<20} | " + " | ".join([f"{s:<18}" for s in eval_stages])

    # Initialize Global Evaluators for EVAL stages
    evals = {stage: IoUCalculator(num_classes) for stage in eval_stages}
    
    logger.info(header)
    logger.info("-" * len(header))
    
    for f in tqdm(files):
        room_name = os.path.basename(f)
        data = np.load(f)
        
        # Run Pipeline
        res = process_room_full_pipeline(cfg, model, data, return_all=True)
        gt = res['GT']
        
        # --- Logging Logic (Based on eval_stages) ---
        if not ablation_mode:
            # Production: Per-Class IoU
            if eval_stages:
                stage = eval_stages[0]
                pred = res[stage]
                
                local_eval = IoUCalculator(num_classes)
                local_eval.update(pred, gt)
                cls_ious, miou = local_eval.compute()
                
                row_str = f"{room_name[:20]:<20} | " + " | ".join([f"{v*100:.1f}".center(5) for v in cls_ious]) + f" | {miou*100:.2f}".center(6)
                logger.info(row_str)
                
                evals[stage].update(pred, gt)
            
        else:
            # Ablation: Stage Comparison
            ms_vals = []
            for stage in eval_stages:
                evals[stage].update(res[stage], gt)
                ms_vals.append(calc_local_miou(res[stage], gt, num_classes))
            
            row_str = f"{room_name[:20]:<20} | " + " | ".join([f"{v:.4f}".center(18) for v in ms_vals])
            logger.info(row_str)
        
        # --- Save Logic (Based on save_stages) ---
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
    
    # Final Summary (Global)
    ious = {k: evals[k].compute()[0] for k in eval_stages}
    mious = {k: evals[k].compute()[1] for k in eval_stages}
    
    header_row = f"{'Class':<12} | " + " | ".join([f"{s:<18}" for s in eval_stages])
    logger.info(header_row)
    logger.info("-" * len(header_row))
    
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:<12} | " + " | ".join([f"{ious[s][i]*100:.2f}".center(18) for s in eval_stages])
        logger.info(row)
        
    logger.info("-" * len(header_row))
    logger.info(f"{'mIoU':<12} | " + " | ".join([f"{mious[s]*100:.2f}".center(18) for s in eval_stages]))
    logger.info("="*100)