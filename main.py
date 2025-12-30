import sys
import os
import yaml
import argparse
import torch
import datetime

# ==============================================================================
# Path Injection
# Ensure 'geoprop' package is importable when running from the root directory.
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../geoprop
parent_dir = os.path.dirname(current_dir)                # .../Project_Root
sys.path.append(parent_dir)

from geoprop.utils.logger import setup_logger
from geoprop.core.trainer import run_training
from geoprop.core.inferencer import run_inference
from geoprop.models.point_jafar import DecoupledPointJAFAR

def load_config(args):
    """
    Load Global Config and merge with Dataset Specific Config.
    """
    if not os.path.exists(args.global_config):
        raise FileNotFoundError(f"Global config not found at: {args.global_config}")

    with open(args.global_config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    target_dataset = cfg['project']['target_dataset']
    
    # Locate dataset config dynamically
    base_conf_dir = os.path.dirname(os.path.abspath(args.global_config))
    dataset_config_path = os.path.join(
        base_conf_dir, 
        target_dataset, 
        f"{target_dataset}.yaml"
    )
    
    if not os.path.exists(dataset_config_path):
        raise FileNotFoundError(f"Dataset config not found at: {dataset_config_path}")
        
    with open(dataset_config_path, 'r') as f:
        dataset_cfg = yaml.safe_load(f)
        
    # Merge configurations
    cfg['dataset'] = dataset_cfg['dataset']
    return cfg

def prepare_output_dirs(cfg, timestamp):
    """
    Generate output directory structure with timestamp.
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_output = os.path.join(root_dir, "outputs", cfg['dataset']['name'], timestamp)
    
    dirs = {
        'root': base_output,
        'logs': os.path.join(base_output, 'logs'),
        'viz': os.path.join(base_output, 'viz'),
        'npy': os.path.join(base_output, 'pseudo_labels')
    }
    
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
        
    return dirs

def log_key_config(cfg, logger):
    """
    Log essential configuration parameters for quick reference.
    """
    logger.info("="*80)
    logger.info(f"GEOPROP PIPELINE CONFIGURATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Project Name   : {cfg['project']['name']}")
    logger.info(f"Dataset        : {cfg['dataset']['name']} (Classes: {cfg['dataset'].get('num_classes', 'Unknown')})")
    logger.info(f"Root Dir       : {cfg['dataset']['root_dir']}")
    logger.info(f"Label Ratio    : {cfg['dataset']['label_ratio']}")
    logger.info("-" * 80)
    logger.info(f"TRAIN Enabled  : {cfg['train']['enable']}")
    if cfg['train']['enable']:
        logger.info(f"  Epochs       : {cfg['train']['epochs']}")
        logger.info(f"  Batch Size   : {cfg['train']['batch_size']}")
    logger.info("-" * 80)
    logger.info(f"INFER Enabled  : {cfg['inference']['enable']}")
    if cfg['inference']['enable']:
        inf = cfg['inference']
        logger.info(f"  Ablation Mode: {inf.get('ablation_mode', False)}")
        logger.info(f"  TTA          : {inf['tta']['enabled']} (Rounds: {inf['tta']['rounds']})")
        logger.info(f"  Conf. Filter : {inf['confidence_filter']['enabled']} (Strict: {inf['confidence_filter']['rec_err_strict']}, Loose: {inf['confidence_filter']['rec_err_loose']})")
        logger.info(f"  Geo. Gating  : {inf['geometric_gating']['enabled']} (Str: {inf['geometric_gating']['gate_strength']}, Conf: {inf['geometric_gating'].get('confidence_threshold')})")
        logger.info(f"  Graph Refine : {inf['graph_refine']['enabled']} (Voxel: {inf['graph_refine']['fine_voxel_n']})")
        logger.info(f"  Spatial Sm.  : {inf['spatial_smooth']['enabled']}")
    logger.info("="*80)

def main():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(os.path.dirname(__file__), 'config/global.yaml')
    parser.add_argument('--global_config', type=str, default=default_config)
    args = parser.parse_args()
    
    # 0. Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Load config
    cfg = load_config(args)
    
    # 2. Prepare directories
    out_dirs = prepare_output_dirs(cfg, timestamp)
    cfg['paths'] = out_dirs 
    
    # Update project root output for trainer
    cfg['project']['root_output'] = out_dirs['root']
    
    # 3. Setup Logger
    log_filename = f"pipeline_{timestamp}.log"
    logger = setup_logger(out_dirs['logs'], log_filename=log_filename)
    
    logger.info(f"Task Started at: {timestamp}")
    
    # 4. Log Key Configuration [New Feature]
    log_key_config(cfg, logger)
    
    trained_model = None
    
    # Phase 1: Training
    if cfg['train']['enable']:
        save_path = os.path.join(out_dirs['root'], "best_model.pth")
        trained_model = run_training(cfg, save_path)
    
    # Phase 2: Inference
    if cfg['inference']['enable']:
        if trained_model is None:
            local_ckpt = os.path.join(out_dirs['root'], "best_model.pth")
            cfg_ckpt = cfg['inference'].get('checkpoint_path')
            
            if os.path.exists(local_ckpt):
                ckpt_path = local_ckpt
            elif cfg_ckpt and os.path.exists(cfg_ckpt):
                ckpt_path = cfg_ckpt
            else:
                ckpt_path = None
            
            if not ckpt_path:
                logger.error(f"Training skipped and No Checkpoint found at local: {local_ckpt} or config: {cfg_ckpt}")
                return
            
            logger.info(f"Loading weights from: {ckpt_path}")
            model = DecoupledPointJAFAR(cfg['model']['qk_dim'], cfg['model']['k_neighbors']).to(cfg['project']['device'])
            model.load_state_dict(torch.load(ckpt_path))
            trained_model = model
            
        run_inference(cfg, trained_model)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()