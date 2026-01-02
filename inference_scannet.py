import os
import sys
import argparse
import yaml
import torch
import numpy as np
import logging
import glob
from tqdm import tqdm
from collections import OrderedDict

# Ensure import paths are correct
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from geoprop.models.point_jafar import DecoupledPointJAFAR
from geoprop.core.inferencer import process_room_full_pipeline
from geoprop.utils.logger import setup_logger
from geoprop.utils.visualizer import generate_viz
from geoprop.utils.metrics import IoUCalculator

# --- Constants for ScanNet ---
S3DIS_MEAN_RGB = np.array([120.0, 120.0, 110.0])
S3DIS_STD_RGB  = np.array([70.0, 70.0, 70.0])

SCANNET_COLORS = np.array([
    [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120], [188, 189, 34],
    [140, 86, 75], [255, 152, 150], [214, 39, 40], [197, 176, 213], [148, 103, 189],
    [196, 156, 148], [23, 190, 207], [247, 182, 210], [219, 219, 141], [255, 127, 14],
    [158, 218, 229], [44, 160, 44], [112, 128, 144], [227, 119, 194], [82, 84, 163]
], dtype=np.float32) / 255.0

SCANNET_CLASS_NAMES = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 
    'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 
    'refridg', 'shower', 'toilet', 'sink', 'bathtub', 'other'
]

# [Config Mapping] Map display names to YAML config keys
STAGE_TO_CONFIG_KEY = {
    "Direct Inference": "direct_inference",
    "TTA Integration": "tta",
    "Confidence Filter": "confidence_filter",  # Explicitly added
    "Geometric Gating": "geometric_gating",
    "Graph Refine": "graph_refine",
    "Spatial Smooth": "spatial_smooth"
}

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def align_domain(xyz, rgb):
    z_min = xyz[:, 2].min()
    xyz[:, 2] -= z_min
    rgb_norm = (rgb - rgb.mean(axis=0)) / (rgb.std(axis=0) + 1e-6)
    rgb_aligned = rgb_norm * S3DIS_STD_RGB + S3DIS_MEAN_RGB
    rgb_aligned = np.clip(rgb_aligned, 0, 255)
    return xyz, rgb_aligned

def load_scannet_data(npz_path, align=True):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Data not found: {npz_path}")
    
    try:
        data = np.load(npz_path)
    except Exception as e:
        raise ValueError(f"Failed to load numpy file: {e}")

    if 'coords' in data: xyz = data['coords']
    elif 'xyz' in data: xyz = data['xyz']
    elif 'points' in data: xyz = data['points']
    else: raise ValueError("Missing coordinates in npz")

    if 'colors' in data: rgb = data['colors']
    elif 'rgb' in data: rgb = data['rgb']
    else: raise ValueError("Missing colors in npz")

    if 'semantic_gt' in data: lbl = data['semantic_gt']
    elif 'labels' in data: lbl = data['labels']
    else: lbl = np.full(xyz.shape[0], -100)

    if rgb.max() <= 1.1: rgb = rgb * 255.0

    if align:
        xyz, rgb = align_domain(xyz, rgb)

    valid_mask = (lbl >= 0) & (lbl < 20)
    lbl[~valid_mask] = -100
    
    full_data = np.concatenate([xyz, rgb, lbl.reshape(-1, 1)], axis=1)
    return full_data, xyz.shape[0]

def inject_scannet_config(cfg, args):
    target_classes = 20
    if 'dataset' not in cfg: cfg['dataset'] = {}
    cfg['dataset']['name'] = 'scannet'
    cfg['dataset']['num_classes'] = target_classes
    cfg['dataset']['label_ratio'] = 0.001
    
    if 'model' not in cfg: cfg['model'] = {}
    cfg['model']['num_classes'] = target_classes
    cfg['model']['input_mode'] = args.input_mode
    
    if 'inference' in cfg:
        # [Fix] Ensure confidence_filter gets updated too
        for mod in ['geometric_gating', 'graph_refine', 'confidence_filter', 'spatial_smooth']:
            if mod in cfg['inference']:
                cfg['inference'][mod]['num_classes'] = target_classes
        
        if args.aggressive_tta:
            if 'tta' not in cfg['inference']: cfg['inference']['tta'] = {}
            cfg['inference']['tta']['aggressive'] = True
            
    return cfg

def main():
    parser = argparse.ArgumentParser(description="GeoProp V4.0 - ScanNet Batch Inference")
    parser.add_argument('--input', type=str, default='/home/work/research/geoprop/datasets/scannet/train', help="Path or Directory")
    parser.add_argument('--checkpoint', type=str, default='/home/work/research/geoprop/outputs/s3dis/20260102_175345_GBL_TR-Rnd-Rnd_INF-Abl/last_model.pth')
    parser.add_argument('--config', type=str, default='config/global.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/scannet_full_npy')
    parser.add_argument('--input_mode', type=str, default='gblobs', choices=['absolute', 'gblobs'])
    parser.add_argument('--aggressive_tta', action='store_true', help="Force enable rotation voting")
    parser.add_argument('--no_viz', action='store_true', help="Disable visualization")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, log_filename="inference_scannet.log")
    
    cfg = load_config(args.config)
    cfg = inject_scannet_config(cfg, args)
    
    # File Discovery
    if os.path.isdir(args.input):
        search_path = os.path.join(args.input, "*.npz")
        files = sorted(glob.glob(search_path))
        if not files:
            search_path = os.path.join(args.input, "**", "*.npz")
            files = sorted(glob.glob(search_path, recursive=True))
    else:
        files = [args.input]
        
    if len(files) == 0:
        logger.error(f"No .npz files found in {args.input}")
        return

    logger.info(f"="*80)
    logger.info(f"ScanNet Batch Inference (V4.0 Final)")
    logger.info(f"Input Source : {args.input}")
    logger.info(f"Found Files  : {len(files)}")
    logger.info(f"Input Mode   : {cfg['model']['input_mode']}")
    logger.info(f"Aggressive TTA: {cfg['inference']['tta'].get('aggressive', False)}")
    logger.info(f"="*80)

    model = DecoupledPointJAFAR(
        qk_dim=cfg['model']['qk_dim'], 
        k=cfg['model']['k_neighbors'],
        input_mode=cfg['model']['input_mode'] 
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        try:
            model.load_state_dict(torch.load(args.checkpoint))
            logger.info("Weights loaded successfully.")
        except Exception as e:
            logger.error(f"Weights load failed: {e}")
            return
    else:
        logger.error("No CUDA detected.")
        return

    global_evaluators = {}
    
    # Header
    cls_headers = " | ".join([f"{c[:5]:<5}" for c in SCANNET_CLASS_NAMES])
    header = f"{'Scene ID':<15} | {cls_headers} | {'mIoU':<6} | {'OA':<6}"
    
    logger.info("-" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    # Hot-patch Visualizer
    import geoprop.utils.visualizer as vis_module
    vis_module.S3DIS_COLORS = SCANNET_COLORS 
    vis_module.CLASS_NAMES = SCANNET_CLASS_NAMES

    model.eval()
    torch.cuda.empty_cache()
    
    for file_path in tqdm(files, desc="Processing"):
        scene_id = os.path.basename(file_path).split('.')[0]
        
        try:
            data, _ = load_scannet_data(file_path, align=True)
            results = process_room_full_pipeline(cfg, model, data, return_all=True)
            gt_lbl = results['GT'].astype(int)
            
            # Initialize evaluators on first run
            if not global_evaluators:
                for stage_name in results.keys():
                    if stage_name != 'GT':
                        global_evaluators[stage_name] = IoUCalculator(num_classes=20)

            # Identify Final Stage
            stage_keys = [k for k in results.keys() if k != 'GT']
            final_stage = stage_keys[-1]
            
            # --- Selective Saving Logic (Strict YAML compliance) ---
            global_save_enabled = cfg['inference'].get('save_npy', False)
            
            for stage_name in stage_keys:
                pred_lbl = results[stage_name]
                
                # 1. Update Metrics
                mask = (gt_lbl >= 0) & (gt_lbl < 20) & (pred_lbl >= 0)
                if mask.sum() > 0:
                    global_evaluators[stage_name].update(pred_lbl[mask], gt_lbl[mask])
                
                # 2. Strict YAML Check for Saving
                # Check 1: Is global saving enabled?
                # Check 2: Does this stage have 'save_output: true' in config?
                config_key = STAGE_TO_CONFIG_KEY.get(stage_name)
                
                if global_save_enabled and config_key:
                    stage_specific_cfg = cfg['inference'].get(config_key, {})
                    if stage_specific_cfg.get('save_output', False):
                        safe_name = stage_name.replace(" ", "_")
                        save_npy_path = os.path.join(args.output_dir, f"{scene_id}_{safe_name}.npy")
                        np.save(save_npy_path, pred_lbl)
                        # No log print to keep terminal clean, just save it.

            # --- Log Final Stage ---
            pred_final = results[final_stage]
            mask_final = (gt_lbl >= 0) & (gt_lbl < 20) & (pred_final >= 0)
            
            if mask_final.sum() > 0:
                room_eval = IoUCalculator(num_classes=20)
                room_eval.update(pred_final[mask_final], gt_lbl[mask_final])
                oa, miou, cls_ious = room_eval.compute()
                cls_str = " | ".join([f"{v*100:.1f}".center(5) for v in cls_ious])
                logger.info(f"{scene_id[:15]:<15} | {cls_str} | {miou*100:.1f}% | {oa*100:.1f}%")
            else:
                logger.warning(f"{scene_id}: No valid points for evaluation.")

            # --- Visualization ---
            if not args.no_viz:
                save_path_viz = os.path.join(args.output_dir, 'viz')
                os.makedirs(save_path_viz, exist_ok=True)
                
                xyz_viz = data[:, :3]
                if len(xyz_viz) > 50000:
                    idx = np.random.choice(len(xyz_viz), 50000, replace=False)
                    final_viz_dict = {k: v[idx] for k, v in results.items() if k != 'GT'}
                    generate_viz(xyz_viz[idx], final_viz_dict, gt_lbl[idx], scene_id, save_path_viz)
                else:
                    final_viz_dict = {k: v for k, v in results.items() if k != 'GT'}
                    generate_viz(xyz_viz, final_viz_dict, gt_lbl, scene_id, save_path_viz)

        except Exception as e:
            logger.error(f"Error processing {scene_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Final Global Summary
    logger.info("="*80)
    logger.info("FINAL DATASET SUMMARY - STAGE PROGRESSION")
    logger.info("="*80)
    logger.info(f"{'Stage Name':<20} | {'mIoU':<8} | {'OA':<8}")
    logger.info("-" * 40)
    
    for stage_name, evaluator in global_evaluators.items():
        g_oa, g_miou, _ = evaluator.compute()
        logger.info(f"{stage_name:<20} | {g_miou*100:.2f}% | {g_oa*100:.2f}%")
        
    logger.info("-" * 40)
    last_stage = list(global_evaluators.keys())[-1]
    g_oa, g_miou, g_ious = global_evaluators[last_stage].compute()
    
    logger.info(f"\n>>> Detailed Report for Final Stage: {last_stage}")
    logger.info(f"{'Class':<15} | {'IoU':<10}")
    logger.info("-" * 30)
    for i, iou in enumerate(g_ious):
        logger.info(f"{SCANNET_CLASS_NAMES[i]:<15} | {iou*100:.2f}%")
    logger.info("="*80)

if __name__ == "__main__":
    main()