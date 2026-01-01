import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import torch
import logging
from geoprop.utils.logger import setup_logger
from geoprop.core.trainer import run_training
from geoprop.core.inferencer import run_inference
from geoprop.models.point_jafar import DecoupledPointJAFAR

def print_config_summary(logger, cfg):
    logger.info("="*80)
    logger.info("GEOPROP PIPELINE CONFIGURATION SUMMARY (V3.0 Compatible)")
    logger.info("="*80)
    logger.info(f"Project Name    : {cfg['project']['name']}")
    logger.info(f"Target Dataset  : {cfg['project']['target_dataset']}")
    logger.info("-" * 80)
    
    tr = cfg['train']
    md = cfg['model']
    logger.info(f"STRATEGY SETTINGS")
    logger.info(f"  > Input Mode    : {md.get('input_mode', 'UNKNOWN').upper()}")
    logger.info(f"  > Dynamic Weight: {tr.get('use_dynamic_weights')}")
    logger.info(f"  > Seed Mode     : Train={'FIXED' if tr['seed_mode']['train'] else 'RANDOM'} | Val={'FIXED' if tr['seed_mode']['val'] else 'RANDOM'}")
    logger.info("-" * 80)
    
    if tr['enable']:
        logger.info(f"TRAIN Enabled   : True (Epochs: {tr['epochs']}, Batch: {tr['batch_size']})")
    else:
        logger.info(f"TRAIN Enabled   : False")
    logger.info("="*80)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/global.yaml')
    parser.add_argument('--output_root', type=str, default='outputs')
    args = parser.parse_args()

    # Load Config
    with open(args.config, 'r') as f: 
        cfg = yaml.safe_load(f)
    
    # Setup Paths
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    dataset_name = cfg['project'].get('target_dataset', 's3dis')
    output_dir = os.path.join(args.output_root, dataset_name, timestamp)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'viz'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'npy'), exist_ok=True)
    
    cfg['paths'] = {
        'output': output_dir, 
        'viz': os.path.join(output_dir, 'viz'), 
        'npy': os.path.join(output_dir, 'npy')
    }
    
    # Setup Logger
    logger = setup_logger(output_dir)
    logger.info(f"Task Started at: {timestamp}")
    
    # Print Summary
    print_config_summary(logger, cfg)

    # Train
    if cfg['train']['enable']:
        trained_model = run_training(cfg, os.path.join(output_dir, 'best_model.pth'))
    else:
        logger.info(f"Loading checkpoint from: {cfg['inference']['checkpoint_path']}")
        model_args = {
            'qk_dim': cfg['model']['qk_dim'],
            'k_neighbors': cfg['model']['k_neighbors'],
            'input_mode': cfg['model']['input_mode']
        }
        trained_model = DecoupledPointJAFAR(**model_args).cuda()
        trained_model.load_state_dict(torch.load(cfg['inference']['checkpoint_path']))

    # Inference
    if cfg['inference']['enable']:
        run_inference(cfg, trained_model)

if __name__ == "__main__":
    main()