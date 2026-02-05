import os
import numpy as np
import glob
import shutil
import argparse
from tqdm import tqdm

def sparsify_raw_data(data_root, backup_root, ratio=0.001, seed=0, ignore_index=255):
    """
    修改 S3DIS raw 数据为稀疏标注，并将原始 GT 备份到独立文件夹。
    data_root: data/S3DIS/s3disfull/raw
    backup_root: data/S3DIS/s3disfull/raw_gt_backup
    """
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root {data_root} does not exist.")

    # 1. 创建备份目录
    if not os.path.exists(backup_root):
        os.makedirs(backup_root)
        print(f"Created backup directory: {backup_root}")

    # 设置随机种子
    np.random.seed(seed)

    # 2. 扫描 raw 目录下的文件
    all_files = glob.glob(os.path.join(data_root, '*.npy'))
    
    train_files = []
    skipped_count = 0
    
    for fpath in all_files:
        fname = os.path.basename(fpath)
        
        # 跳过 Area 5 (验证集)
        if 'Area_5' in fname:
            skipped_count += 1
            continue
            
        train_files.append(fpath)

    if len(train_files) == 0:
        print("No training files found.")
        return

    print(f"Found {len(train_files)} training files. (Skipped {skipped_count} val files)")
    print(f"Target Ratio: {ratio * 100}%")
    print(f"Backup Dir: {backup_root}")

    for fpath in tqdm(train_files, desc="Sparsifying"):
        fname = os.path.basename(fpath)
        backup_path = os.path.join(backup_root, fname)

        # --- A. 备份逻辑 ---
        # 如果备份不存在，则当前的 fpath 是原始全量数据，复制过去
        if not os.path.exists(backup_path):
            shutil.copy2(fpath, backup_path)
        
        # --- B. 读取逻辑 (始终从备份读取 GT) ---
        # 这样即使你运行多次脚本，也是基于全量数据采样
        data = np.load(backup_path)
        
        # --- C. 稀疏化逻辑 ---
        # S3DIS PointNeXt npy: (N, 7) -> XYZ(3), RGB(3), Label(1)
        labels = data[:, -1].astype(np.int32)
        num_points = labels.shape[0]
        
        num_labeled = int(num_points * ratio)
        if num_labeled < 1: num_labeled = 1

        # 随机采样
        selected_indices = np.random.choice(num_points, num_labeled, replace=False)
        
        # 构建稀疏标签
        sparse_labels = np.full_like(labels, ignore_index)
        sparse_labels[selected_indices] = labels[selected_indices]
        
        # 替换最后一列
        data[:, -1] = sparse_labels
        
        # --- D. 写入逻辑 ---
        # 覆盖 raw 里的文件
        np.save(fpath, data)

    print("\n[Success] Raw data sparsified.")
    print(f"Original full data is safely kept in: {backup_root}")
    print("IMPORTANT: Run 'rm -rf data/S3DIS/s3disfull/processed' to apply changes!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/S3DIS/s3disfull/raw', 
                        help='Path to raw data folder')
    parser.add_argument('--backup', type=str, default='data/S3DIS/s3disfull/raw_gt_backup', 
                        help='Path to backup folder')
    parser.add_argument('--ratio', type=float, default=0.001, 
                        help='Label ratio (0.01 = 1%)')
    
    args = parser.parse_args()
    sparsify_raw_data(args.root, args.backup, ratio=args.ratio)