import os
import numpy as np
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ================= Configuration =================
DATA_ROOT = 'data/semantic_kitti'
NUM_TRIALS = 4000        # 增加尝试次数，19类全覆盖难度大
LABELED_RATIO = 0.001   # 0.1% Labels
NUM_CLASSES = 19
IGNORE_INDEX = 255
MAX_WORKERS = 8         # 建议根据CPU核心数调整

CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
    "other-ground", "building", "fence", "vegetation", "trunk",
    "terrain", "pole", "traffic-sign",
]

TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
CLIP_RANGE = [-35.2, -35.2, -4, 35.2, 35.2, 2]
# =================================================

GLOBAL_DATA_CACHE = None

def get_learning_map(ignore_index):
    map_dict = {
        0 : ignore_index, 1 : ignore_index, 10: 0, 11: 1, 13: 5, 15: 3, 16: 5, 18: 4, 
        20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 
        52: ignore_index, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 18, 99: ignore_index,
        252: 0, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5
    }
    max_key = max(map_dict.keys())
    map_array = np.full(max_key + 1, ignore_index, dtype=np.int64)
    for k, v in map_dict.items():
        map_array[k] = v
    return map_array

def load_all_data_optimized(data_root):
    """
    全量扫描数据。
    为了防止OOM，我们过滤掉没有标签的点，并且只保留在CLIP_RANGE内的有效点。
    """
    print(f"📂 Scanning ALL training frames from {data_root}...")
    mapping = get_learning_map(IGNORE_INDEX)
    loaded_frames = []
    
    min_xyz = np.array(CLIP_RANGE[:3])
    max_xyz = np.array(CLIP_RANGE[3:])

    for seq in TRAIN_SEQUENCES:
        velodyne_dir = os.path.join(data_root, 'dataset', 'sequences', seq, 'velodyne')
        label_dir = os.path.join(data_root, 'dataset', 'sequences', seq, 'labels')
        if not os.path.exists(velodyne_dir): continue
        
        files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
        
        for f in tqdm(files, desc=f"Loading Seq {seq}"):
            bin_path = os.path.join(velodyne_dir, f)
            label_path = os.path.join(label_dir, f.replace('.bin', '.label'))
            
            try:
                points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                coord = points[:, :3]
                label_raw = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
                sem_label = label_raw & 0xFFFF 
                
                # 1. 范围裁剪
                mask = np.all((coord >= min_xyz) & (coord <= max_xyz), axis=1)
                coord, sem_label = coord[mask], sem_label[mask]
                
                # 2. 标签映射
                segment = np.full_like(sem_label, IGNORE_INDEX, dtype=np.int64)
                valid_indices = sem_label < len(mapping)
                segment[valid_indices] = mapping[sem_label[valid_indices]]
                
                # 3. 极简缓存：只保留有有效语义标签的点，减小内存占用
                mask_valid = segment != IGNORE_INDEX
                if np.any(mask_valid):
                    loaded_frames.append((coord[mask_valid].astype(np.float32), segment[mask_valid].astype(np.uint8)))
                
            except Exception as e:
                continue

    print(f"✅ Total valid frames loaded: {len(loaded_frames)}")
    return loaded_frames

def evaluate_seed(args):
    global GLOBAL_DATA_CACHE
    seed_tuple, ratio = args
    h1_k, h2_k, h3_k = seed_tuple
    threshold = int(ratio * 100000)
    
    total_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    for coord, segment in GLOBAL_DATA_CACHE:
        # 这里的计算逻辑必须与 Dataset 里的 Hash 逻辑完全一致
        h1 = np.abs(coord[:, 0] * h1_k).astype(np.int64)
        h2 = np.abs(coord[:, 1] * h2_k).astype(np.int64)
        h3 = np.abs(coord[:, 2] * h3_k).astype(np.int64)
        
        seed_hash = h1 ^ h2 ^ h3
        label_mask = (seed_hash % 100000) < threshold
        
        valid_labels = segment[label_mask]
        if valid_labels.size > 0:
            # 使用 bincount 统计，注意 segment 是 uint8
            counts = np.bincount(valid_labels, minlength=NUM_CLASSES)
            total_class_counts += counts
            
    min_count = np.min(total_class_counts)
    covered_classes = np.sum(total_class_counts > 0)
    return seed_tuple, min_count, covered_classes, total_class_counts

def main():
    global GLOBAL_DATA_CACHE
    GLOBAL_DATA_CACHE = load_all_data_optimized(DATA_ROOT)
    
    # 随机生成候选种子
    candidates = [np.random.randint(1e7, 1e8, 3).tolist() for _ in range(NUM_TRIALS)]
    tasks = [(c, LABELED_RATIO) for c in candidates]
    
    print(f"🚀 Searching GOLDEN SEED among {NUM_TRIALS} candidates on FULL dataset...")
    
    best_min = -1
    best_res = None
    best_counts = None
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 chunksize 提高并行效率
        results = list(tqdm(executor.map(evaluate_seed, tasks, chunksize=10), total=len(tasks)))

    for seed, min_c, covered, counts in results:
        if covered == NUM_CLASSES:
            if min_c > best_min:
                best_min = min_c
                best_res = seed
                best_counts = counts

    if best_res is None:
        print("\n❌ CRITICAL: Still failed to cover all 19 classes even with full data.")
        print("💡 Suggestion: The labeled_ratio (0.1%) might be too low for the rarest class in SemanticKITTI.")
        print("   Try increasing NUM_TRIALS to 2000 or slightly increase LABELED_RATIO to 0.0012.")
        return

    # 计算权重
    total = np.sum(best_counts)
    weights = total / (best_counts * NUM_CLASSES + 1e-6)
    weights = np.clip(weights, 1.0, 20.0) # 室外场景不平衡严重，上限放宽到20

    print("\n" + "="*80)
    print("🏆 FULL DATASET GOLDEN SEED FOUND 🏆")
    print("="*80)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<15} | Count: {best_counts[i]:<10} | Weight: {weights[i]:.4f}")
    print("-" * 40)
    print(f"🎯 Min Class Points: {best_min} | Seeds: {best_res}")
    print("="*80)
    
    print(f"\n📋 [COPY TO semantic_kitti_geo.py]:\nhash_seed_1={best_res[0]}, hash_seed_2={best_res[1]}, hash_seed_3={best_res[2]}")
    print(f"\n📋 [COPY TO geo_ptv3.py]:\nclass_weights={[round(w, 4) for w in weights]}")

if __name__ == "__main__":
    main()