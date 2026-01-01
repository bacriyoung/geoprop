import sys
import os
import yaml
import argparse
import torch
import datetime

# Fix path to ensure imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoprop.utils.logger import setup_logger
from geoprop.core.trainer import run_training
from geoprop.core.inferencer import run_inference
from geoprop.models.point_jafar import DecoupledPointJAFAR

def load_config(args):
    """
    [V2.0 Logic Restored]
    Load Global Config and AUTOMATICALLY merge with Dataset Specific Config.
    This eliminates redundancy.
    """
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Global config not found at: {args.config}")

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    target_dataset = cfg['project']['target_dataset']
    
    # Locate dataset config dynamically (Standard V2.0 behavior)
    # Assumes config/dataset_name/dataset_name.yaml
    base_conf_dir = os.path.dirname(os.path.abspath(args.config))
    dataset_config_path = os.path.join(
        base_conf_dir, 
        target_dataset, 
        f"{target_dataset}.yaml"
    )
    
    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(f"Dataset config not found at: {dataset_config_path}")
        
    with open(dataset_config_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f)
        
    # Merge: Dataset config overrides Global placeholders
    # This ensures 'root_dir' and 'label_ratio' are only managed in s3dis.yaml
    if 'dataset' in dataset_cfg:
        # Update existing keys, preserve 'num_workers' if in global
        for k, v in dataset_cfg['dataset'].items():
            cfg['dataset'][k] = v
            
    return cfg

def prepare_output_dirs(cfg, timestamp):
    """
    [V2.0 Logic Restored]
    Generate output directory structure relative to the script location.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_output = os.path.join(root_dir, "outputs", cfg['dataset']['name'], timestamp)
    
    dirs = {
        'root': base_output,
        'logs': os.path.join(base_output, 'logs'),
        'viz': os.path.join(base_output, 'viz'),
        'npy': os.path.join(base_output, 'npy') # V2.0 used 'pseudo_labels' or 'npy'
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    return dirs

def log_key_config(cfg, logger):
    """
    [Hybrid Log] V2.0 Detailed Modules + V3.0 New Features
    """
    logger.info("="*80)
    logger.info(f"GEOPROP PIPELINE CONFIGURATION SUMMARY (V3.0)")
    logger.info("="*80)
    
    # 1. Project & Data
    ds = cfg['dataset']
    logger.info(f"Project Name   : {cfg['project']['name']}")
    logger.info(f"Dataset        : {ds['name']} (Points: {ds.get('num_points', 'Auto')})")
    logger.info(f"Root Dir       : {ds.get('root_dir', 'MISSING')}")
    logger.info(f"Label Ratio    : {ds.get('label_ratio', 'MISSING')}")
    logger.info("-" * 80)
    
    # 2. V3.0 Strategies (The New Stuff)
    tr = cfg['train']
    md = cfg['model']
    logger.info(f"V3.0 STRATEGIES")
    logger.info(f"  > Input Mode   : {md.get('input_mode', 'UNKNOWN').upper()}")
    logger.info(f"  > Dyn. Weights : {tr.get('use_dynamic_weights')}")
    logger.info(f"  > Seed Mode    : Train={'FIXED' if tr['seed_mode']['train'] else 'RANDOM'} | Val={'FIXED' if tr['seed_mode']['val'] else 'RANDOM'}")
    logger.info("-" * 80)
    
    # 3. Training Details
    logger.info(f"TRAIN Enabled  : {tr['enable']}")
    if tr['enable']:
        logger.info(f"  Epochs       : {tr['epochs']}")
        logger.info(f"  Batch Size   : {tr['batch_size']}")
        logger.info(f"  Val Interval : {tr['val_interval']}")
    logger.info("-" * 80)
    
    # 4. Inference Details (V2.0 Detailed Style)
    inf = cfg['inference']
    logger.info(f"INFER Enabled  : {inf['enable']}")
    if inf['enable']:
        logger.info(f"  Ablation Mode: {inf.get('ablation_mode', False)}")
        logger.info(f"  TTA          : {inf['tta']['enabled']} (Rounds: {inf['tta'].get('rounds')})")
        logger.info(f"  Conf. Filter : {inf['confidence_filter']['enabled']} (Strict: {inf['confidence_filter'].get('rec_err_strict')})")
        
        geo = inf['geometric_gating']
        logger.info(f"  Geo. Gating  : {geo['enabled']} (Str: {geo.get('gate_strength')}, Conf: {geo.get('confidence_threshold')})")
        
        graph = inf['graph_refine']
        logger.info(f"  Graph Refine : {graph['enabled']} (Voxel: {graph.get('fine_voxel_n')}, K: {graph.get('k_neighbors')})")
        
        spatial = inf['spatial_smooth']
        logger.info(f"  Spatial Sm.  : {spatial['enabled']} (K: {spatial.get('k_neighbors')})")
    logger.info("="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/global.yaml')
    # output_root is now used as base if needed, but we default to relative
    parser.add_argument('--output_root', type=str, default='outputs') 
    args = parser.parse_args()
    
    # 0. Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load & Merge Config (The Fix)
    cfg = load_config(args)
    
    # 2. Prepare directories (The Fix)
    out_dirs = prepare_output_dirs(cfg, timestamp)
    cfg['paths'] = out_dirs 
    
    # 3. Setup Logger
    logger = setup_logger(out_dirs['logs'], log_filename=f"pipeline_{timestamp}.log")
    logger.info(f"Task Started at: {timestamp}")
    
    # 4. Log Detailed Configuration
    log_key_config(cfg, logger)
    
    trained_model = None
    
    # Phase 1: Training
    if cfg['train']['enable']:
        save_path = os.path.join(out_dirs['root'], "best_model.pth")
        trained_model = run_training(cfg, save_path)
    
    # Phase 2: Inference
    if cfg['inference']['enable']:
        if trained_model is None:
            # Logic to find checkpoint
            cfg_ckpt = cfg['inference'].get('checkpoint_path')
            local_ckpt = os.path.join(out_dirs['root'], "best_model.pth")
            
            ckpt_path = None
            if cfg_ckpt and os.path.exists(cfg_ckpt):
                ckpt_path = cfg_ckpt
            elif os.path.exists(local_ckpt):
                ckpt_path = local_ckpt
                
            if not ckpt_path:
                logger.error("No checkpoint found for inference.")
                return
            
            logger.info(f"Loading weights from: {ckpt_path}")
            model_args = {
                'qk_dim': cfg['model']['qk_dim'],
                'k_neighbors': cfg['model']['k_neighbors'],
                'input_mode': cfg['model']['input_mode']
            }
            trained_model = DecoupledPointJAFAR(**model_args).cuda()
            trained_model.load_state_dict(torch.load(ckpt_path))
            
        run_inference(cfg, trained_model)

if __name__ == "__main__":
    main()