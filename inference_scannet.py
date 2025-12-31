import os
import sys
import argparse
import yaml
import torch
import numpy as np
import logging
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from geoprop.models.point_jafar import DecoupledPointJAFAR
from geoprop.core.inferencer import process_room_full_pipeline
from geoprop.utils.logger import setup_logger
from geoprop.utils.visualizer import generate_viz

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_scannet_data(npz_path):
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Data not found: {npz_path}")
    data = np.load(npz_path)
    if 'coords' in data: xyz = data['coords']
    elif 'xyz' in data: xyz = data['xyz']
    else: xyz = data['points']
    if 'colors' in data: rgb = data['colors']
    elif 'rgb' in data: rgb = data['rgb']
    else: raise ValueError("Cannot find colors in npz")
    if 'semantic_gt' in data: lbl = data['semantic_gt']
    elif 'labels' in data: lbl = data['labels']
    else: lbl = np.full(xyz.shape[0], -100)
    if rgb.max() <= 1.1: rgb = rgb * 255.0
    full_data = np.concatenate([xyz, rgb, lbl.reshape(-1, 1)], axis=1)
    return full_data, xyz.shape[0]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/global.yaml')
    parser.add_argument('--output_dir', type=str, default='outputs/scannet_inference')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, log_filename="inference.log")
    
    cfg = load_config(args.config)
    if 'dataset' not in cfg: cfg['dataset'] = {}
    cfg['dataset']['name'] = 'scannet'
    cfg['dataset']['num_classes'] = 20
    if 'label_ratio' not in cfg['dataset']: cfg['dataset']['label_ratio'] = 0.001

    logger.info(f"Inference Mode: Direct ScanNet Propagation (20 Classes) + Dual GBlobs")
    
    try:
        data, N = load_scannet_data(args.input)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # [V31] Initialize with 18 dim
    model = DecoupledPointJAFAR(
        qk_dim=cfg['model']['qk_dim'], 
        k=cfg['model']['k_neighbors'],
        input_geo_dim=18
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        try:
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda'))
            model.load_state_dict(checkpoint)
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Weight loading failed: {e}")
            return
    else:
        return

    logger.info("Running GeoProp Pipeline...")
    results = process_room_full_pipeline(cfg, model, data, return_all=True)
    
    # Visualization & Metrics
    gt_lbl = results['GT'].astype(int)
    from geoprop.utils.metrics import calc_local_metrics
    final_viz_dict = {}
    for stage_name, pred_lbl in results.items():
        if stage_name == 'GT': continue
        final_viz_dict[stage_name] = pred_lbl
        valid_mask = (gt_lbl >= 0) & (gt_lbl < 20)
        if valid_mask.sum() > 0:
            oa, miou = calc_local_metrics(pred_lbl[valid_mask], gt_lbl[valid_mask], 20)
            logger.info(f"{stage_name:<20} | {miou*100:.2f}% | {oa*100:.2f}%")
            
    scene_id = os.path.basename(args.input).split('.')[0]
    save_path_viz = os.path.join(args.output_dir, 'viz')
    generate_viz(data[:, :3], final_viz_dict, gt_lbl, scene_id, save_path_viz, dataset_name='scannet')

if __name__ == "__main__":
    main()