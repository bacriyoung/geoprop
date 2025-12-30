import sys
import os
import yaml
import argparse
import torch
import datetime  # [新增]

# ==============================================================================
# [核心修复] 路径注入
# 目的：确保在 geoprop 目录下运行 main.py 时，仍能识别 'from geoprop...' 包导入
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../geoprop
parent_dir = os.path.dirname(current_dir)                # .../Project_Root
sys.path.append(parent_dir)

# 必须在 sys.path 修正后导入
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
    
    # 动态定位 dataset config: geoprop/config/{name}/{name}.yaml
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
        
    # 合并配置
    cfg['dataset'] = dataset_cfg['dataset']
    return cfg

def prepare_output_dirs(cfg, timestamp):
    """
    在 geoprop/outputs/{dataset_name}/{timestamp} 下生成目录结构
    这样每次运行都会有独立的文件夹，不会覆盖。
    """
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # [核心修改] 目录结构增加时间戳层级
    # 例如: outputs/s3dis/20251230_153000/
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
    # 默认路径改为相对于 main.py
    default_config = os.path.join(os.path.dirname(__file__), 'config/global.yaml')
    parser.add_argument('--global_config', type=str, default=default_config)
    args = parser.parse_args()
    
    # 0. 生成任务开始时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 加载配置
    cfg = load_config(args)
    
    # 2. 准备输出目录 (带时间戳)
    out_dirs = prepare_output_dirs(cfg, timestamp)
    cfg['paths'] = out_dirs 
    
    # 更新 project.root_output 以便 trainer 使用
    cfg['project']['root_output'] = out_dirs['root']
    
    # 3. 初始化日志 (文件名带时间戳)
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
            # 优先查找当前 output 目录下的模型 (本次训练生成的)
            local_ckpt = os.path.join(out_dirs['root'], "best_model.pth")
            
            # 其次查找 config 中指定的 checkpoint_path (外部指定的)
            cfg_ckpt = cfg['inference'].get('checkpoint_path')
            
            # 逻辑优化：如果本次没训练(local不存在)，且配置了外部路径，则用外部路径
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