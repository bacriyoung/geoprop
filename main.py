import sys
import os
import yaml
import argparse
import torch
import datetime  

current_dir = os.path.dirname(os.path.abspath(__file__)) # .../geoprop
parent_dir = os.path.dirname(current_dir)                # .../Project_Root
sys.path.append(parent_dir)

from geoprop.utils.logger import setup_logger
from geoprop.core.trainer import run_training
from geoprop.core.inferencer import run_inference
from geoprop.models.point_jafar import DecoupledPointJAFAR

def load_config(args):
    """
    加载 Global Config 并合并 Dataset Specific Config
    """
    if not os.path.exists(args.global_config):
        raise FileNotFoundError(f"Global config not found at: {args.global_config}")

    with open(args.global_config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    target_dataset = cfg['project']['target_dataset']
    
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
        
    cfg['dataset'] = dataset_cfg['dataset']
    return cfg

def prepare_output_dirs(cfg, timestamp):
    """

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

def main():
    parser = argparse.ArgumentParser()
    default_config = os.path.join(os.path.dirname(__file__), 'config/global.yaml')
    parser.add_argument('--global_config', type=str, default=default_config)
    args = parser.parse_args()
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    cfg = load_config(args)

    out_dirs = prepare_output_dirs(cfg, timestamp)
    cfg['paths'] = out_dirs 
    
    cfg['project']['root_output'] = out_dirs['root']
    
    log_filename = f"pipeline_{timestamp}.log"
    logger = setup_logger(out_dirs['logs'], log_filename=log_filename)
    
    logger.info(f"Task Started at: {timestamp}")
    logger.info(f"Loaded Config: {args.global_config}")
    logger.info(f"Output Directory: {out_dirs['root']}")
    logger.info(f"Log File: {os.path.join(out_dirs['logs'], log_filename)}")
    
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