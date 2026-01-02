import os
import sys
import argparse
import yaml
import torch
import numpy as np
import logging
from tqdm import tqdm

# Ensure import paths are correct
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from geoprop.models.point_jafar import DecoupledPointJAFAR
from geoprop.core.inferencer import process_room_full_pipeline
from geoprop.utils.logger import setup_logger
from geoprop.utils.visualizer import generate_viz
from geoprop.utils.metrics import calc_local_metrics, IoUCalculator

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
    data = np.load(npz_path)
    if 'coords' in data: xyz = data['coords']
    elif 'xyz' in data: xyz = data['xyz']
    else: xyz = data['points']
    if 'colors' in data: rgb = data['colors']
    elif 'rgb' in data: rgb = data['rgb']
    else: raise ValueError("No color data found.")
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
    cfg['model']['num_classes'] = target_classes
    cfg['model']['input_mode'] = args.input_mode
    if 'inference' in cfg:
        for mod in ['geometric_gating', 'graph_refine', 'confidence_filter', 'spatial_smooth']:
            if mod in cfg['inference']:
                cfg['inference'][mod]['num_classes'] = target_classes
        if args.aggressive_tta:
            cfg['inference']['tta']['aggressive'] = True
    return cfg

def main():
    parser = argparse.ArgumentParser()
    
    # [Updated] Added default paths so you don't need to type them every time
    parser.add_argument('--input', type=str, 
                        default='/home/work/research/geoprop/datasets/scannet/train/scene0001_01.npz',
                        help="Path to .npz file (Default: scene0000_00.npz)")
                        
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/work/research/geoprop/outputs/s3dis/20260102_113104_GBL_TR-Rnd-Fix_INF-Prod/last_model.pth',
                        help="Path to .pth file (Default: outputs/s3dis/best_model.pth)")
                        
    parser.add_argument('--config', type=str, default='config/global.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/scannet')
    parser.add_argument('--input_mode', type=str, default='gblobs', choices=['absolute', 'gblobs'])
    parser.add_argument('--aggressive_tta', action='store_true', help="Enable rotation voting")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, log_filename="inference_scannet.log")
    
    cfg = load_config(args.config)
    cfg = inject_scannet_config(cfg, args)
    
    logger.info(f"="*60)
    logger.info(f"ScanNet Inference (V3.1 Adapted)")
    logger.info(f"Input File: {os.path.basename(args.input)}")
    logger.info(f"Checkpoint: {os.path.basename(args.checkpoint)}") # Show ckpt used
    logger.info(f"="*60)

    try:
        data, N = load_scannet_data(args.input, align=True)
        logger.info(f"Loaded {N} points (Domain Aligned).")
    except Exception as e:
        logger.error(f"Data load error: {e}")
        return

    model = DecoupledPointJAFAR(
        qk_dim=cfg['model']['qk_dim'], 
        k=cfg['model']['k_neighbors'],
        input_mode=cfg['model']['input_mode'] 
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        try:
            model.load_state_dict(torch.load(args.checkpoint))
            logger.info("Weights loaded.")
        except Exception as e:
            logger.error(f"Weights load failed: {e}")
            return
    else:
        logger.error("No CUDA.")
        return

    logger.info("Running Pipeline...")
    torch.cuda.empty_cache()
    model.eval()
    
    results = process_room_full_pipeline(cfg, model, data, return_all=True)
    
    gt_lbl = results['GT'].astype(int)
    final_viz_dict = {}
    
    print("\n" + "-"*60)
    print(f"{'Stage':<20} | {'mIoU':<8} | {'OA':<8}")
    print("-" * 60)
    
    for stage_name, pred_lbl in results.items():
        if stage_name == 'GT': continue
        final_viz_dict[stage_name] = pred_lbl
        
        metric_mask = (gt_lbl >= 0) & (gt_lbl < 20) & (pred_lbl >= 0)
        
        if metric_mask.sum() > 0:
            evaluator = IoUCalculator(num_classes=20)
            evaluator.update(pred_lbl[metric_mask], gt_lbl[metric_mask])
            oa, miou, ious = evaluator.compute()
            
            logger.info(f"{stage_name:<20} | {miou*100:.2f}% | {oa*100:.2f}%")
            print(f"{stage_name:<20} | {miou*100:.2f}%   | {oa*100:.2f}%")
            
            print(f"  > Details for {stage_name}:")
            for i, iou in enumerate(ious):
                print(f"    {SCANNET_CLASS_NAMES[i]:<12}: {iou*100:.2f}%")
        else:
            logger.warning(f"{stage_name}: No valid points for metrics.")
            
    print("-" * 60)

    import geoprop.utils.visualizer as vis_module
    vis_module.S3DIS_COLORS = SCANNET_COLORS 
    vis_module.CLASS_NAMES = SCANNET_CLASS_NAMES
    
    scene_id = os.path.basename(args.input).split('.')[0]
    save_path_viz = os.path.join(args.output_dir, 'viz')
    os.makedirs(save_path_viz, exist_ok=True)
    
    xyz_viz = data[:, :3]
    if len(xyz_viz) > 50000:
        idx = np.random.choice(len(xyz_viz), 50000, replace=False)
        final_viz_dict_sub = {k: v[idx] for k, v in final_viz_dict.items()}
        try:
            generate_viz(xyz_viz[idx], final_viz_dict_sub, gt_lbl[idx], scene_id, save_path_viz)
            logger.info(f"Viz saved to {save_path_viz}")
        except Exception as e:
            logger.error(f"Viz failed: {e}")
    else:
        try:
            generate_viz(xyz_viz, final_viz_dict, gt_lbl, scene_id, save_path_viz)
        except Exception as e:
            logger.error(f"Viz failed: {e}")

if __name__ == "__main__":
    main()