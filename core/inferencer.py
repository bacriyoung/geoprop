import numpy as np
import torch
import os
from tqdm import tqdm
import logging
from collections import OrderedDict

# Project Imports
from geoprop.data.s3dis.s3dis_dataset import S3DISDataset
from geoprop.data.seeds import get_fixed_seeds
from geoprop.core.modules import confidence_filter, geometric_gating, graph_refine, spatial_smooth
from geoprop.core.modules.gblobs import compute_dual_gblobs  
from geoprop.utils.metrics import IoUCalculator, calc_local_metrics
from geoprop.utils.visualizer import generate_viz, CLASS_NAMES

def process_room_full_pipeline(cfg, model, data, return_all=False):
    """
    Main inference pipeline with Block-wise Feature Computation to ensure
    memory safety and feature alignment with training.
    """
    # 1. Unpack Data
    N = len(data)
    xyz_full = data[:, :3]
    rgb_full = data[:, 3:6]
    lbl = data[:, 6].astype(int)
    num_classes = cfg['dataset'].get('num_classes', 13)
    
    # 2. Prepare Base Tensors
    # Normalize RGB to [0, 1]
    rgb_norm = rgb_full/255.0 if rgb_full.max() > 1.1 else rgb_full.copy()
    
    # [MEMORY NOTE] We only create base tensors here. 
    # We DO NOT compute global features (GBlobs) to avoid OOM on large scenes.
    full_xyz_tensor = torch.from_numpy(xyz_full).float().cuda().unsqueeze(0).transpose(1, 2)
    full_rgb_tensor = torch.from_numpy(rgb_norm).float().cuda().unsqueeze(0).transpose(1, 2)

    # 3. Setup Global Seeds
    # We fix seeds deterministically for inference
    seeds = get_fixed_seeds(N, cfg['dataset']['label_ratio'], lbl, cfg['project']['seed'])
    seed_lbls = lbl[seeds]
    
    # Filter valid seeds (sanity check)
    valid_seed_mask = (seed_lbls >= 0) & (seed_lbls < num_classes)
    valid_global_seeds = seeds[valid_seed_mask]
    valid_global_labels = seed_lbls[valid_seed_mask]

    # Create a fast lookup map: Global Index -> Seed Label
    # Initialize with -1 (indicating not a seed)
    # This allows O(1) checking if a point in a local block is a seed
    global_seed_map = np.full(N, -1, dtype=int)
    global_seed_map[valid_global_seeds] = valid_global_labels
    
    # 4. Inference Initialization
    accum_probs = np.zeros((N, num_classes), dtype=np.float32)
    accum_counts = np.zeros(N, dtype=np.float32)
    lbl_direct = np.full(N, -100, dtype=int)
    
    # TTA Settings
    min_c, max_c = xyz_full.min(0), xyz_full.max(0)
    rounds = cfg['inference']['tta']['rounds'] if cfg['inference']['tta']['enabled'] else 1
    
    # -------------------------------------------------------------------------
    # Sliding Window Inference Loop
    # -------------------------------------------------------------------------
    for t in range(rounds):
        # Apply Jitter for TTA (except first round if you prefer, but usually all rounds jitter)
        if t == 0: 
            aug_xyz = xyz_full
        else: 
            aug_xyz = xyz_full + np.random.normal(0, cfg['inference']['tta']['jitter'], xyz_full.shape)
        
        # Stride = 1.5, Block Size = 2.0 (Overlapping blocks)
        stride = 1.5
        block_size = 2.0
        
        for x in np.arange(min_c[0], max_c[0], stride):
            for y in np.arange(min_c[1], max_c[1], stride):
                # 1. Get Block Indices (Global indices of points in this block)
                mask_indices = np.where(
                    (aug_xyz[:, 0] >= x) & (aug_xyz[:, 0] < x + block_size) & 
                    (aug_xyz[:, 1] >= y) & (aug_xyz[:, 1] < y + block_size)
                )[0]
                
                # Skip small blocks
                if len(mask_indices) < 50:
                    continue
                    
                # 2. Extract Block Data (Dense)
                # [1, 3, N_block]
                b_xyz = torch.from_numpy(aug_xyz[mask_indices]).float().cuda().unsqueeze(0).transpose(1, 2)
                b_rgb = full_rgb_tensor[:, :, mask_indices]
                
                # 3. Identify Seeds INSIDE this block
                # Check lookup map: global_seed_map[mask_indices] gives label or -1
                block_seed_labels = global_seed_map[mask_indices]
                is_seed_in_block = block_seed_labels != -1
                
                # Get indices relative to the block (0 to BlockSize-1)
                rel_seed_idx = np.where(is_seed_in_block)[0]
                
                # [CRITICAL ALIGNMENT LOGIC]
                # Training logic relies on using seeds within the context.
                # If block has no seeds, we cannot propagate information reliably.
                if len(rel_seed_idx) == 0:
                    continue
                    
                # Limit max seeds to avoid OOM in Attention matrix (e.g., 512)
                if len(rel_seed_idx) > 512:
                    rng = np.random.RandomState(42 + len(mask_indices))
                    rel_seed_idx = rng.choice(rel_seed_idx, 512, replace=False)
                
                # Get actual labels for these internal seeds
                current_seed_lbls = block_seed_labels[rel_seed_idx]
                
                with torch.no_grad():
                    # A. [ALIGNMENT FIX] Compute Dense Blobs for the Block FIRST
                    # This ensures features capture the dense surface geometry (walls, floors)
                    # exactly like they do in training.
                    b_geo_blobs, b_rgb_blobs = compute_dual_gblobs(b_xyz, b_rgb, k=32)
                    
                    # B. [ALIGNMENT FIX] Slice Seed Features from Dense Blobs
                    # Now ls_geo_blobs contains context-aware features, NOT sparse point features.
                    ls_geo_blobs = b_geo_blobs[:, :, rel_seed_idx]
                    ls_rgb_blobs = b_rgb_blobs[:, :, rel_seed_idx]
                    
                    # C. Prepare Network Inputs
                    # Seed Coordinates (Relative slice from block)
                    ls_xyz = b_xyz[:, :, rel_seed_idx]
                    
                    # Seed Labels (Value to propagate) - One Hot Encoded
                    ls_val_sem = torch.zeros(1, num_classes, len(rel_seed_idx)).cuda()
                    # unsqueeze(1) is needed to match dim for scatter_
                    ls_val_sem.scatter_(1, torch.from_numpy(current_seed_lbls).long().cuda().unsqueeze(0).unsqueeze(1), 1.0)
                    
                    # Relative Position Encoding (Centering)
                    block_center = b_xyz.mean(2, keepdim=True)
                    loc = ls_xyz - block_center
                    cur = b_xyz - block_center
                    
                    # D. Forward Pass
                    # Network uses dense block features (b_*) to query sparse seed features (ls_*)
                    probs, _ = model(cur, loc, ls_val_sem, b_geo_blobs, ls_geo_blobs, b_rgb_blobs, ls_rgb_blobs)
                
                # Accumulate Predictions
                prob_np = probs.squeeze(0).transpose(0, 1).cpu().numpy()
                accum_probs[mask_indices] += prob_np
                accum_counts[mask_indices] += 1
                
                if t == 0:
                    lbl_direct[mask_indices] = np.argmax(prob_np, axis=1)

    # -------------------------------------------------------------------------
    # Post-Processing
    # -------------------------------------------------------------------------
    
    valid_mask = accum_counts > 0
    lbl_tta = np.full(N, -100, dtype=int)
    
    # 1. TTA Aggregation
    if valid_mask.sum() > 0:
        lbl_tta[valid_mask] = np.argmax(accum_probs[valid_mask] / accum_counts[valid_mask, None], axis=1)
    
    # 2. KNN Fill (Handle points missed by sliding windows)
    lbl_tta = confidence_filter.knn_fill(xyz_full, lbl_tta)
    
    # 3. Compute Confidence Map (Needed for Gating / Refinement)
    confidence_map = np.zeros(N, dtype=np.float32)
    if valid_mask.sum() > 0:
        safe_counts = accum_counts.copy()
        safe_counts[safe_counts == 0] = 1.0
        confidence_map = np.max(accum_probs / safe_counts[:, None], axis=1)
        confidence_map[~valid_mask] = 0.0

    # Prepare Result Dictionary
    result_stages = OrderedDict()
    result_stages["Direct Inference"] = lbl_direct
    if cfg['inference']['tta']['enabled']:
        result_stages["TTA Integration"] = lbl_tta
    
    current_lbl = lbl_tta if cfg['inference']['tta']['enabled'] else lbl_direct
    
    # 4. Geometric Gating
    if cfg['inference']['geometric_gating']['enabled']:
        current_lbl = geometric_gating.run(cfg['inference']['geometric_gating'], xyz_full, rgb_full, current_lbl, confidence=confidence_map)
        result_stages["Geometric Gating"] = current_lbl

    # 5. Graph Refinement
    if cfg['inference']['graph_refine']['enabled']:
        seed_mask_bool = np.zeros(N, dtype=bool)
        seed_mask_bool[valid_global_seeds] = True
        
        current_lbl = graph_refine.run(
            cfg['inference']['graph_refine'], 
            xyz_full, rgb_full, 
            current_lbl, current_lbl, 
            seed_mask=seed_mask_bool, 
            gt_lbl=lbl,  # Enforce seed correctness
            confidence=confidence_map
        )
        result_stages["Graph Refine"] = current_lbl

    # 6. Spatial Smoothing
    if cfg['inference']['spatial_smooth']['enabled']:
        current_lbl = spatial_smooth.run(cfg['inference']['spatial_smooth'], xyz_full, current_lbl)
        result_stages["Spatial Smooth"] = current_lbl

    # [CRITICAL FIX] Ensure GT is included for evaluation
    if return_all:
        result_stages['GT'] = lbl
        return result_stages
    else:
        return current_lbl

def validate_full_scene_logic(model, cfg, all_files):
    """
    Validation loop for full scenes using the inference pipeline.
    """
    model.eval()
    num_classes = cfg['dataset'].get('num_classes', 13)
    evaluator = IoUCalculator(num_classes)
    
    for f in all_files:
        data = np.load(f)
        res_dict = process_room_full_pipeline(cfg, model, data, return_all=True)
        # Use the final stage prediction
        stages = list(res_dict.keys())
        if 'GT' in stages: stages.remove('GT')
        pred = res_dict[stages[-1]]
        
        evaluator.update(pred, res_dict['GT'])
        
    _, miou, _ = evaluator.compute()
    return miou

def run_inference(cfg, model):
    """
    Main entry point for inference.
    """
    logger = logging.getLogger("geoprop")
    logger.info(">>> [Phase 2] Final Inference & Generation...")
    model.eval()
    
    dataset = S3DISDataset(cfg, split='inference')
    files = dataset.files
    
    # [FIX] Define num_classes here
    num_classes = cfg['dataset'].get('num_classes', 13)
    
    # Dummy run to identify active stages
    if len(files) > 0:
        dummy_res = process_room_full_pipeline(cfg, model, np.load(files[0]), return_all=True)
        all_stages = [k for k in dummy_res.keys() if k != 'GT']
    else:
        all_stages = []
        logger.warning("No files found for inference!")

    ablation_mode = cfg['inference'].get('ablation_mode', False)
    
    # Decide which stages to evaluate/save
    if not ablation_mode:
        eval_stages = [all_stages[-1]] if all_stages else []
    else:
        eval_stages = all_stages
    
    evals = {stage: IoUCalculator(num_classes) for stage in eval_stages}
    
    # Pre-create directories if saving NPY
    if cfg['inference']['save_npy']:
        for stage in eval_stages:
            if ablation_mode:
                os.makedirs(os.path.join(cfg['paths']['npy'], stage.replace(" ", "_")), exist_ok=True)
            else:
                os.makedirs(cfg['paths']['npy'], exist_ok=True)
    
    # Header Logging
    header_str = f"{'Room Name':<20} | " + " | ".join([f"{s:<18}" for s in eval_stages])
    logger.info(header_str)
    logger.info("-" * len(header_str))
    
    # Process Files
    for f in tqdm(files):
        room_name = os.path.basename(f)
        data = np.load(f)
        
        # Run Pipeline
        res = process_room_full_pipeline(cfg, model, data, return_all=True)
        gt = res['GT']
        
        # Update Metrics & Log
        ious = []
        for stage in eval_stages:
            evals[stage].update(res[stage], gt)
            _, miou = calc_local_metrics(res[stage], gt, num_classes)
            ious.append(miou)
            
        row_str = f"{room_name[:20]:<20} | " + " | ".join([f"{v:.4f}".center(18) for v in ious])
        logger.info(row_str)
        
        # Save Visualizations
        if cfg['inference']['save_img']:
            # Viz dict contains predictions + GT
            viz_dict = {k: res[k] for k in eval_stages}
            generate_viz(data[:,:3], viz_dict, gt, room_name, cfg['paths']['viz'])
            
        # Save NPY
        if cfg['inference']['save_npy']:
            for stage in eval_stages:
                save_data = data.copy()
                save_data[:, 6] = res[stage] # Overwrite label column
                
                if ablation_mode:
                    path = os.path.join(cfg['paths']['npy'], stage.replace(" ", "_"), room_name)
                else:
                    path = os.path.join(cfg['paths']['npy'], room_name)
                np.save(path, save_data)
            
    # Final Summary
    logger.info("\n" + "="*80)
    for stage in eval_stages:
        oa, miou, _ = evals[stage].compute()
        logger.info(f"{stage:<20} | mIoU: {miou*100:.2f}% | OA: {oa*100:.2f}%")
    logger.info("="*80)