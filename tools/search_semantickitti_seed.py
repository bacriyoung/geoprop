import os
import glob
import numpy as np
import gc
import yaml
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ================= Configuration =================
DATA_ROOT = 'data/semantic_kitti'
NUM_TRIALS = 500        # Number of random seeds to try
LABELED_RATIO = 0.001   # 0.1% Labels
NUM_CLASSES = 19
IGNORE_INDEX = 255

# [CRITICAL] Memory Management
# SemanticKITTI is huge. We cannot load 43k frames into RAM.
# We skip frames to approximate the distribution.
# Stride 10 means we use 10% of data for searching (approx 2000 frames), which is statistically sufficient.
FRAME_STRIDE = 10 

# [CRITICAL] Parallel Workers
# Reading .bin files is CPU heavy. Adjust based on your CPU cores/RAM.
MAX_WORKERS = 8

CLASS_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
    "bicyclist", "motorcyclist", "road", "parking", "sidewalk",
    "other-ground", "building", "fence", "vegetation", "trunk",
    "terrain", "pole", "traffic-sign",
]

# Standard Training Splits (00-07, 09-10) - 08 is Val
TRAIN_SEQUENCES = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']

# Point Clip Range (Must match config!)
CLIP_RANGE = [-35.2, -35.2, -4, 35.2, 35.2, 2]
# =================================================

GLOBAL_DATA_CACHE = None

def get_learning_map(ignore_index):
    """ Standard SemanticKITTI 19-class mapping """
    map_dict = {
        0 : ignore_index, 1 : ignore_index, 10: 0, 11: 1, 13: 5, 15: 3, 16: 5, 18: 4, 
        20: 5, 30: 6, 31: 7, 32: 8, 40: 9, 44: 10, 48: 11, 49: 12, 50: 13, 51: 14, 
        52: ignore_index, 60: 9, 70: 15, 71: 16, 72: 17, 80: 18, 81: 18, 99: ignore_index,
        252: 1, 253: 7, 254: 6, 255: 8, 256: 5, 257: 5, 258: 4, 259: 5
    }
    # Fix moving-car mapping standard
    map_dict[252] = 0 
    
    max_key = max(map_dict.keys())
    map_array = np.full(max_key + 1, ignore_index, dtype=np.int64)
    for k, v in map_dict.items():
        map_array[k] = v
    return map_array

def load_downsampled_data(data_root):
    print(f"📂 Loading training data from {data_root} (Sequences: {TRAIN_SEQUENCES})")
    print(f"⚠️  Sampling 1 frame every {FRAME_STRIDE} frames to save RAM...")
    
    mapping = get_learning_map(IGNORE_INDEX)
    loaded_frames = []
    
    min_xyz = np.array(CLIP_RANGE[:3])
    max_xyz = np.array(CLIP_RANGE[3:])

    for seq in TRAIN_SEQUENCES:
        velodyne_dir = os.path.join(data_root, 'dataset', 'sequences', seq, 'velodyne')
        label_dir = os.path.join(data_root, 'dataset', 'sequences', seq, 'labels')
        
        if not os.path.exists(velodyne_dir): continue
        
        # Get all frames and sort
        files = sorted([f for f in os.listdir(velodyne_dir) if f.endswith('.bin')])
        
        # Apply Stride
        files = files[::FRAME_STRIDE]
        
        for f in tqdm(files, desc=f"Seq {seq}", leave=False):
            bin_path = os.path.join(velodyne_dir, f)
            label_path = os.path.join(label_dir, f.replace('.bin', '.label'))
            
            if not os.path.exists(label_path): continue
            
            try:
                # Load Coord
                points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
                coord = points[:, :3]
                
                # Load Label
                label_raw = np.fromfile(label_path, dtype=np.uint32).reshape(-1)
                sem_label = label_raw & 0xFFFF 
                
                # Clip Points (Crucial: Filter out points that won't be used in training)
                mask_clip = np.all((coord >= min_xyz) & (coord <= max_xyz), axis=1)
                coord = coord[mask_clip]
                sem_label = sem_label[mask_clip]
                
                # Map Labels
                valid_mask = sem_label < len(mapping)
                segment = np.full_like(sem_label, IGNORE_INDEX, dtype=np.int64)
                segment[valid_mask] = mapping[sem_label[valid_mask]]
                
                # Filter ignore index
                mask_valid = segment != IGNORE_INDEX
                loaded_frames.append((coord[mask_valid], segment[mask_valid]))
                
            except Exception as e:
                print(f"Error loading {f}: {e}")

    print(f"✅ Loaded {len(loaded_frames)} representative frames into RAM.")
    return loaded_frames

def evaluate_seed(args):
    global GLOBAL_DATA_CACHE
    seed_tuple, ratio = args
    h1_k, h2_k, h3_k = seed_tuple
    
    total_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    # Iterate all loaded frames
    for coord, segment in GLOBAL_DATA_CACHE:
        # Hash Calculation (Aligned with GeoDatasetMixin)
        h1 = np.abs(coord[:, 0] * h1_k).astype(np.int64)
        h2 = np.abs(coord[:, 1] * h2_k).astype(np.int64)
        h3 = np.abs(coord[:, 2] * h3_k).astype(np.int64)
        
        seed_hash = h1 ^ h2 ^ h3
        threshold = int(ratio * 100000)
        label_mask = (seed_hash % 100000) < threshold
        
        valid_labels = segment[label_mask]
        
        if valid_labels.size > 0:
            counts = np.bincount(valid_labels, minlength=NUM_CLASSES)
            total_class_counts += counts
            
    min_count = np.min(total_class_counts)
    covered_classes = np.sum(total_class_counts > 0)
    
    # Force GC
    del h1, h2, h3, seed_hash, label_mask
    
    return seed_tuple, min_count, covered_classes, total_class_counts

def calculate_weights(counts):
    # Standard inverse frequency (capped at 10.0 like Pointcept usually does)
    total = np.sum(counts)
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        if counts[c] > 0:
            weights[c] = total / (counts[c] * NUM_CLASSES)
    
    # Normalize/Clip to reasonable range for loss stability
    # In S3DIS we clipped to 10.0, here maybe 50.0 due to extreme imbalance? 
    # Let's keep it safe at 10.0 first, or rely on provided weights.
    weights = np.clip(weights, 1.0, 10.0)
    return weights

def main():
    global GLOBAL_DATA_CACHE
    
    # 1. Load Data
    GLOBAL_DATA_CACHE = load_downsampled_data(DATA_ROOT)
    if not GLOBAL_DATA_CACHE:
        print("❌ No data loaded. Check paths.")
        return

    # 2. Generate Random Seeds
    candidates = [np.random.randint(1e7, 1e8, 3).tolist() for _ in range(NUM_TRIALS)]
    tasks = [(c, LABELED_RATIO) for c in candidates]
    
    print(f"🚀 Searching best seeds among {NUM_TRIALS} candidates...")
    
    best_min = -1
    best_res = None
    best_counts = None
    
    # 3. Execution Loop
    if MAX_WORKERS <= 1:
        iterator = tqdm(tasks, total=len(tasks))
        results = (evaluate_seed(task) for task in iterator)
    else:
        executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        iterator = tqdm(executor.map(evaluate_seed, tasks), total=len(tasks))
        results = iterator

    # 4. Find Best
    try:
        for seed, min_c, covered, counts in results:
            # We prioritize covering ALL classes first
            if covered == NUM_CLASSES:
                # Then maximize the minimum class count
                if min_c > best_min:
                    best_min = min_c
                    best_res = seed
                    best_counts = counts
    except Exception as e:
        print(f"Error: {e}")
        if MAX_WORKERS > 1: executor.shutdown(wait=False)
        return

    if best_res is None:
        print("\n❌ Failed to find a seed that covers all 19 classes!")
        print("   Consider increasing LABELED_RATIO or NUM_TRIALS.")
        return

    # 5. Report
    weights = calculate_weights(best_counts)
    weight_str = ", ".join([f"{w:.4f}" for w in weights])

    print("\n" + "="*80)
    print("🏆 SEMANTIC KITTI GOLDEN SEED FOUND 🏆")
    print("="*80)
    print(f"\n📊 Class Distribution (0.1% Subset, Stride={FRAME_STRIDE}):")
    print(f"{'Class':<15} | {'Count':<10} | {'Rec. Weight':<8}")
    print("-" * 40)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<15} | {best_counts[i]:<10} | {weights[i]:.4f}")
    print("-" * 40)
    print(f"🎯 Min Class Points: {best_min}")
    print("="*80)
    
    print("\n📋 [COPY TO DATASET CONFIG] (semantic_kitti_geo.py):")
    print(f"hash_seed_1={best_res[0]},")
    print(f"hash_seed_2={best_res[1]},")
    print(f"hash_seed_3={best_res[2]}")
    
    print("\n📋 [OPTIONAL: UPDATE MODEL CONFIG] (configs/semantic_kitti/geo_ptv3.py):")
    print(f"# Replaces 'class_weights' if you want dynamic balancing:")
    print(f"class_weights=[{weight_str}]")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()