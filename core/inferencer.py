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
from geoprop.utils.visualizer import generate_viz

def process_room_full_pipeline(cfg, model, data, return_all=False):
    """
    完整的 S1 -> S4 推理流水线 (Fully Optimized with Confidence Awareness)
    """
    N = len(data)
    xyz_full = data[:, :3]
    rgb_full = data[:, 3:6]
    lbl = data[:, 6].astype(int)
    
    # 1. 获取固定种子
    seeds = get_fixed_seeds(N, cfg['dataset']['label_ratio'], lbl, cfg['project']['seed'])
    seed_mask = np.zeros(N, dtype=bool); seed_mask[seeds] = True
    
    # 准备 Tensor 数据
    rgb_norm = rgb_full/255.0 if rgb_full.max()>1.1 else rgb_full.copy()
    s_xyz = torch.from_numpy(data[seeds, :3]).float().cuda().unsqueeze(0)
    
    rgb_seeds = data[seeds, 3:6].copy()
    rgb_seeds = rgb_seeds/255.0 if rgb_seeds.max()>1.1 else rgb_seeds
    s_rgb = torch.from_numpy(rgb_seeds).float().cuda().unsqueeze(0)
    s_lbl = torch.from_numpy(data[seeds, 6].astype(int)).long().cuda().unsqueeze(0)
    
    # 2. 基础推理循环 (S1 & S2)
    accum_probs = np.zeros((N, cfg['dataset']['num_classes']), dtype=np.float32)
    accum_counts = np.zeros(N, dtype=np.float32)
    lbl_direct = np.full(N, -100, dtype=int)
    lbl_filtered_sparse = np.full(N, -100, dtype=int)
    
    min_c, max_c = xyz_full.min(0), xyz_full.max(0)
    rounds = cfg['inference']['tta']['rounds'] if cfg['inference']['tta']['enabled'] else 1
    
    for t in range(rounds):
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
                    val = torch.zeros(1, cfg['dataset']['num_classes'], len(idx)).cuda().scatter_(1, s_lbl[:,idx].unsqueeze(1), 1.0)
                    
                    with torch.no_grad():
                        rec, bdy = model(cur, loc, torch.cat([s_rgb[:,idx].transpose(1,2), norm],1), s_rgb[:,idx].transpose(1,2))
                        probs, _ = model(cur, loc, torch.cat([s_rgb[:,idx].transpose(1,2), norm],1), val)
                    
                    prob_np = probs.squeeze().transpose(0,1).cpu().numpy()
                    accum_probs[mask] += prob_np
                    accum_counts[mask] += 1
                    
                    if t == 0:
                        pred_block = np.argmax(prob_np, axis=1)
                        lbl_direct[mask] = pred_block
                        lbl_filtered_sparse = confidence_filter.run(cfg['inference']['confidence_filter'], 
                                                                    rec, bdy, b_rgb, 
                                                                    pred_block, mask, lbl_filtered_sparse)

    lbl_s1_1 = confidence_filter.knn_fill(xyz_full, lbl_filtered_sparse.copy())

    valid_tta = accum_counts > 0
    lbl_s2 = np.full(N, -100, dtype=int)
    lbl_s2[valid_tta] = np.argmax(accum_probs[valid_tta]/accum_counts[valid_tta, None], axis=1)
    lbl_s2 = confidence_filter.knn_fill(xyz_full, lbl_s2)

    # 计算全局置信度图
    confidence_map = np.zeros(N, dtype=np.float32)
    confidence_map[valid_tta] = np.max(accum_probs[valid_tta] / accum_counts[valid_tta, None], axis=1)

    # 3. 构建结果字典 (Confidence Injection)
    result_stages = OrderedDict()
    result_stages["Direct Inference"] = lbl_direct
    lbl_current = lbl_direct

    if cfg['inference']['confidence_filter']['enabled']:
        result_stages["Confidence Filter"] = lbl_s1_1
        lbl_current = lbl_s1_1

    if cfg['inference']['tta']['enabled']:
        result_stages["TTA Integration"] = lbl_s2
        lbl_current = lbl_s2

    # --- Geometric Gating (S2.1) ---
    lbl_geom = geometric_gating.run(
        cfg['inference']['geometric_gating'], 
        xyz_full, 
        rgb_full, 
        lbl_current,
        confidence=confidence_map
    )
    if cfg['inference']['geometric_gating']['enabled']:
        result_stages["Geometric Gating"] = lbl_geom
        shape_guide = lbl_geom
        lbl_current = lbl_geom # S2.1 的结果作为 S3 的 Base
    else:
        shape_guide = lbl_current

    # --- Graph Refine (S3) [优化点] ---
    # 传入 confidence_map 实现保护机制
    lbl_graph = graph_refine.run(
        cfg['inference']['graph_refine'], 
        xyz_full, 
        rgb_full, 
        lbl_current, # Base Input (S2.1 or S2)
        shape_guide, # Shape Guide
        seed_mask, 
        lbl,
        confidence=confidence_map # <--- 关键修改
    )
    if cfg['inference']['graph_refine']['enabled']:
        result_stages["Graph Refine"] = lbl_graph
        lbl_current = lbl_graph

    # --- Spatial Smooth (S4) ---
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
    evaluator = IoUCalculator()
    for f in all_files:
        data = np.load(f)
        res_dict = process_room_full_pipeline(cfg, model, data, return_all=True)
        # 验证指标：优先使用 Confidence Filter
        if "Confidence Filter" in res_dict:
            pred = res_dict["Confidence Filter"]
        else:
            pred = res_dict["Direct Inference"]
        evaluator.update(pred, res_dict['GT'])
    _, miou, _ = evaluator.compute()
    return miou

def run_inference(cfg, model):
    logger = logging.getLogger("geoprop")
    logger.info(">>> [Phase 2] Final Inference & Generation...")
    model.eval()
    
    dataset = S3DISDataset(cfg, split='inference')
    files = dataset.files
    
    logger.info(f"Target Areas: {cfg['dataset']['split']['inference_areas']}")
    logger.info(f"Target Files: {len(files)} rooms")
    
    dummy_res = process_room_full_pipeline(cfg, model, np.load(files[0]), return_all=True)
    active_stages = [k for k in dummy_res.keys() if k != 'GT']
    
    evals = [IoUCalculator() for _ in range(len(active_stages))]
    
    header = f"{'Room Name':<20} | " + " | ".join([f"{s:<18}" for s in active_stages])
    logger.info(header)
    logger.info("-" * len(header))
    
    for f in tqdm(files):
        room_name = os.path.basename(f)
        data = np.load(f)
        
        res = process_room_full_pipeline(cfg, model, data, return_all=True)
        gt = res['GT']
        
        ms_vals = []
        for i, stage in enumerate(active_stages):
            evals[i].update(res[stage], gt)
            ms_vals.append(calc_local_miou(res[stage], gt))
            
        logger.info(f"{room_name[:20]:<20} | " + " | ".join([f"{v:.4f}".center(18) for v in ms_vals]))
        
        if cfg['inference']['save_img']:
            viz_path = cfg['paths']['viz']
            generate_viz(data[:,:3], {k: res[k] for k in active_stages}, gt, room_name, viz_path)
        
        if cfg['inference']['save_npy']:
            npy_path = cfg['paths']['npy']
            final_stage = active_stages[-1]
            data[:, 6] = res[final_stage]
            np.save(os.path.join(npy_path, room_name), data)

    logger.info("\n" + "="*100)
    logger.info("FINAL PER-CLASS EVALUATION")
    
    ious = [e.compute()[0] for e in evals]
    mious = [e.compute()[1] for e in evals]
    
    header_row = f"{'Class':<12} | " + " | ".join([f"{s:<18}" for s in active_stages])
    logger.info(header_row)
    logger.info("-" * len(header_row))
    
    from geoprop.utils.visualizer import CLASS_NAMES
    for i, name in enumerate(CLASS_NAMES):
        row = f"{name:<12} | " + " | ".join([f"{ious[s][i]*100:.2f}".center(18) for s in range(len(active_stages))])
        logger.info(row)
        
    logger.info("-" * len(header_row))
    logger.info(f"{'mIoU':<12} | " + " | ".join([f"{m*100:.2f}".center(18) for m in mious]))
    logger.info("="*100)