import os
import sys
import argparse
import yaml
import torch
import numpy as np
import logging
from tqdm import tqdm

# 获取项目根目录，确保 import 路径正确
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from geoprop.models.point_jafar import DecoupledPointJAFAR
from geoprop.core.inferencer import process_room_full_pipeline
from geoprop.utils.logger import setup_logger
from geoprop.utils.visualizer import generate_viz

SCANNET_COLORS = np.array([
    [174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120], [188, 189, 34],
    [140, 86, 75], [255, 152, 150], [214, 39, 40], [197, 176, 213], [148, 103, 189],
    [196, 156, 148], [23, 190, 207], [247, 182, 210], [219, 219, 141], [255, 127, 14],
    [158, 218, 229], [44, 160, 44], [112, 128, 144], [227, 119, 194], [82, 84, 163]
])

def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def load_scannet_data(npz_path):
    """
    Robust ScanNet .npz 数据加载器 (修复标签越界问题)
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"找不到数据文件: {npz_path}")
    
    data = np.load(npz_path)
    
    # 获取坐标
    if 'coords' in data: xyz = data['coords']
    elif 'xyz' in data: xyz = data['xyz']
    else: xyz = data['points']
    
    # 获取颜色
    if 'colors' in data: rgb = data['colors']
    elif 'rgb' in data: rgb = data['rgb']
    else: raise ValueError("npz 文件中找不到颜色信息 (rgb/colors)")
    
    # 获取标签
    if 'semantic_gt' in data: lbl = data['semantic_gt']
    elif 'labels' in data: lbl = data['labels']
    else: lbl = np.full(xyz.shape[0], -100)
    
    # [关键修复] 强制处理标签，防止越界
    # ScanNet 通常有 20 个有效类 (0-19)。
    # 如果遇到 255 或其他乱七八糟的标签，统一设为 -100 (Ignored)
    # 这样后续代码在使用 label 时需要做有效性检查
    
    valid_mask = (lbl >= 0) & (lbl < 20)
    lbl[~valid_mask] = -100 # 将所有非法标签设为 -100
    
    # 归一化颜色
    if rgb.max() <= 1.1: 
        rgb = rgb * 255.0
        
    full_data = np.concatenate([xyz, rgb, lbl.reshape(-1, 1)], axis=1)
    return full_data, xyz.shape[0]

def main():
    parser = argparse.ArgumentParser()
    
    # --- 这里为你设置了默认路径 ---
    # 1. 默认数据集路径 (请根据你的实际文件名修改此处的 default)
    parser.add_argument('--input', type=str, 
                        default='/home/work/research/geoprop/datasets/scannet/train/scene0001_01.npz', 
                        help="ScanNet .npz 文件路径")
    
    # 2. 默认权重文件路径
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/work/research/geoprop/outputs/s3dis/20260102_113104_GBL_TR-Rnd-Fix_INF-Prod/last_model.pth', 
                        help="训练好的 .pth 权重路径")
    
    # 3. 默认配置文件
    parser.add_argument('--config', type=str, default='config/global.yaml')
    
    # 4. 默认输出目录
    parser.add_argument('--output_dir', type=str, default='outputs/scannet_inference')
    
    # 5. 默认特征模式 (gblobs 或 absolute)
    parser.add_argument('--input_mode', type=str, default='gblobs', choices=['absolute', 'gblobs'])
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, log_filename="inference_scannet.log")

    cfg = load_config(args.config)
    
    # --- 核心改进：全局配置自动映射逻辑 ---
    target_num_classes = 20 # ScanNet 目标类别
    
    # 1. 注入到 dataset 节点 (影响数据加载和 metrics)
    if 'dataset' not in cfg: cfg['dataset'] = {}
    cfg['dataset']['name'] = 'scannet'
    cfg['dataset']['num_classes'] = target_num_classes
    
    # 2. 注入到 inference 的子模块 (解决你现在的报错)
    # 强制让 geometric_gating, graph_refine 等模块知道现在是 20 类
    if 'inference' in cfg:
        for module in ['geometric_gating', 'graph_refine', 'confidence_filter', 'spatial_smooth']:
            if module in cfg['inference']:
                cfg['inference'][module]['num_classes'] = target_num_classes
    
    # 3. 注入到 model 节点 (影响 forward 内部判断)
    if 'model' not in cfg: cfg['model'] = {}
    cfg['model']['num_classes'] = target_num_classes

    # 设置 label_ratio
    if 'label_ratio' not in cfg['dataset']: cfg['dataset']['label_ratio'] = 0.001
    
    # 覆盖输入模式
    cfg['model']['input_mode'] = args.input_mode
    input_mode = cfg['model']['input_mode']
    # ---------------------------------------

    logger.info(f"="*50)
    logger.info(f"ScanNet 推理模式 (V3.0 自动映射版)")
    # ... 其余逻辑不变
    logger.info(f"输入模式  : {input_mode.upper()}")
    logger.info(f"权重文件  : {args.checkpoint}")
    logger.info(f"输入数据  : {args.input}")
    logger.info(f"="*50)
    
    try:
        data, N = load_scannet_data(args.input)
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        return

    # 初始化模型
    model = DecoupledPointJAFAR(
        qk_dim=cfg['model']['qk_dim'], 
        k=cfg['model']['k_neighbors'],
        input_mode=input_mode 
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        try:
            checkpoint = torch.load(args.checkpoint, map_location=torch.device('cuda'))
            model.load_state_dict(checkpoint)
            logger.info("权重加载成功。")
        except Exception as e:
            logger.error(f"权重加载失败: {e}")
            return
    else:
        logger.error("未检测到 CUDA。")
        return

    logger.info("开始执行推理流水线...")
    model.eval()
    results = process_room_full_pipeline(cfg, model, data, return_all=True)
    
    # 指标计算
    gt_lbl = results['GT'].astype(int)
    from geoprop.utils.metrics import calc_local_metrics
    final_viz_dict = {}
    
    print("\n" + "-"*40)
    print(f"{'阶段 (Stage)':<20} | {'mIoU':<8} | {'OA':<8}")
    print("-"*40)
    
    for stage_name, pred_lbl in results.items():
        if stage_name == 'GT': continue
        final_viz_dict[stage_name] = pred_lbl
        metric_mask = (gt_lbl >= 0) & (gt_lbl < 20) & (pred_lbl >= 0)
        
        if metric_mask.sum() > 0:
            oa, miou = calc_local_metrics(pred_lbl[metric_mask], gt_lbl[metric_mask], 20)
            logger.info(f"{stage_name:<20} | {miou*100:.2f}% | {oa*100:.2f}%")
            print(f"{stage_name:<20} | {miou*100:.2f}%   | {oa*100:.2f}%")
        else:
            logger.warning(f"阶段 {stage_name}: 没有有效的预测点参与指标计算。")
            
    scene_id = os.path.basename(args.input).split('.')[0]
    save_path_viz = os.path.join(args.output_dir, 'viz')

    # ... 指标计算逻辑 ...

    # [核心修复] 热补丁：在不改动源码文件的情况下，运行时替换颜色表
    import geoprop.utils.visualizer as vis_module
    # 强制进行归一化处理
    vis_module.S3DIS_COLORS = SCANNET_COLORS.astype(np.float32) / 255.0
    
    # 顺便更新图例名称
    vis_module.CLASS_NAMES = [f"Class_{i}" for i in range(20)]
    
    # 可选：如果你的 generate_viz 里面还在用 CLASS_NAMES 打印图例，也顺便换掉
    vis_module.CLASS_NAMES = [f"Class_{i}" for i in range(20)] 

    # 运行原有的可视化逻辑
    scene_id = os.path.basename(args.input).split('.')[0]
    save_path_viz = os.path.join(args.output_dir, 'viz')
    
    os.makedirs(save_path_viz, exist_ok=True)

    # 建议带上下采样，ScanNet 点太多了，html会卡死
    xyz_viz = data[:, :3]
    if len(xyz_viz) > 100000:
        idx = np.random.choice(len(xyz_viz), 100000, replace=False)
        generate_viz(xyz_viz[idx], {k: v[idx] for k, v in final_viz_dict.items()}, gt_lbl[idx], scene_id, save_path_viz)
    else:
        generate_viz(xyz_viz, final_viz_dict, gt_lbl, scene_id, save_path_viz)

    try:
        generate_viz(data[:, :3], final_viz_dict, gt_lbl, scene_id, save_path_viz)
        logger.info(f"可视化结果已保存至: {save_path_viz}")
    except Exception as e:
        logger.error(f"可视化生成失败: {e}")
    
    # 保存结果
    save_file = os.path.join(args.output_dir, f"{scene_id}_pred.npy")
    np.save(save_file, list(results.values())[-2])
    logger.info(f"结果已保存至: {save_file}")

if __name__ == "__main__":
    main()