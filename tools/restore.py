import os
import glob
import shutil
import argparse
from tqdm import tqdm

def restore_raw_data(data_root, backup_root):
    if not os.path.exists(backup_root):
        print(f"[Error] Backup directory not found: {backup_root}")
        return

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    # 扫描备份目录
    backup_files = glob.glob(os.path.join(backup_root, '*.npy'))
    
    if len(backup_files) == 0:
        print("No backup files found.")
        return

    print(f"Restoring {len(backup_files)} files from backup...")
    print(f"Source: {backup_root}")
    print(f"Target: {data_root}")

    for src_path in tqdm(backup_files, desc="Restoring"):
        fname = os.path.basename(src_path)
        dst_path = os.path.join(data_root, fname)
        
        # 强制覆盖
        shutil.copy2(src_path, dst_path)

    print("\n[Success] Dataset restored to full annotations.")
    print("IMPORTANT: Run 'rm -rf data/S3DIS/s3disfull/processed' to apply changes!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data/S3DIS/s3disfull/raw', 
                        help='Path to raw data folder')
    parser.add_argument('--backup', type=str, default='data/S3DIS/s3disfull/raw_gt_backup', 
                        help='Path to backup folder')
    
    args = parser.parse_args()
    restore_raw_data(args.root, args.backup)