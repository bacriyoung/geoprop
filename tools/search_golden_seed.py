import os
import glob
import numpy as np
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ================= Configuration =================
DATA_ROOT = 'data/s3dis'
NUM_TRIALS = 800        
LABELED_RATIO = 0.001   
NUM_CLASSES = 13
CLASS_NAMES = [
    "ceiling", "floor", "wall", "beam", "column", "window", "door",
    "table", "chair", "sofa", "bookcase", "board", "clutter"
]

# [CRITICAL SETTINGS]
# Reduce this if it crashes! (8 -> 4 -> 2 -> 0)
# Set to 0 or 1 to run sequentially (slow but safe)
MAX_WORKERS = 4  
# =================================================

GLOBAL_ROOM_DATA = None

def load_all_data(data_root):
    print(f"📂 Loading ALL training data from {data_root} (excluding Area 5)...")
    files = glob.glob(os.path.join(data_root, "**", "coord.npy"), recursive=True)
    room_list = []
    
    for f in tqdm(files):
        if "Area_5" in f: continue
        try:
            coord = np.load(f).astype(np.float32)
            label_file = f.replace("coord.npy", "segment.npy")
            if not os.path.exists(label_file): continue
            segment = np.load(label_file).astype(np.int64).reshape(-1)
            room_list.append((coord, segment))
        except Exception: 
            pass
            
    print(f"✅ Total rooms loaded: {len(room_list)}")
    return room_list

def evaluate_seed(args):
    global GLOBAL_ROOM_DATA
    seed_tuple, ratio = args
    h1_k, h2_k, h3_k = seed_tuple
    
    total_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    # Iterate rooms
    for coord, segment in GLOBAL_ROOM_DATA:
        # Optimization: Compute mask in chunks or efficiently
        # Here we rely on numpy's speed
        h1 = np.abs(coord[:, 0] * h1_k).astype(np.int64)
        h2 = np.abs(coord[:, 1] * h2_k).astype(np.int64)
        h3 = np.abs(coord[:, 2] * h3_k).astype(np.int64)
        
        seed_hash = h1 ^ h2 ^ h3
        threshold = int(ratio * 100000)
        label_mask = (seed_hash % 100000) < threshold
        
        valid_labels = segment[label_mask]
        valid_labels = valid_labels[valid_labels != 255]
        
        if valid_labels.size > 0:
            counts = np.bincount(valid_labels, minlength=NUM_CLASSES)
            total_class_counts += counts
    
    min_count = np.min(total_class_counts)
    covered_classes = np.sum(total_class_counts > 0)
    
    # Explicit garbage collection to release temp arrays immediately
    del h1, h2, h3, seed_hash, label_mask
    gc.collect()
    
    return seed_tuple, min_count, covered_classes, total_class_counts

def calculate_weights(counts):
    total = np.sum(counts)
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        if counts[c] > 0:
            weights[c] = total / (counts[c] * NUM_CLASSES)
    weights = np.clip(weights, 1.0, 10.0)
    return weights

def main():
    global GLOBAL_ROOM_DATA
    GLOBAL_ROOM_DATA = load_all_data(DATA_ROOT)
    
    candidates = [np.random.randint(1e7, 1e8, 3).tolist() for _ in range(NUM_TRIALS)]
    tasks = [(c, LABELED_RATIO) for c in candidates]
    
    print(f"🚀 Searching best seeds among {NUM_TRIALS} candidates...")
    print(f"⚙️  Concurrency Mode: {'Sequential' if MAX_WORKERS <= 1 else f'Parallel ({MAX_WORKERS} workers)'}")
    
    best_min = -1
    best_res = None
    best_counts = None
    
    # Logic Switch: Sequential vs Parallel
    if MAX_WORKERS <= 1:
        # Sequential Loop (Crash Proof)
        iterator = tqdm(tasks, total=len(tasks))
        results = (evaluate_seed(task) for task in iterator)
    else:
        # Parallel Loop
        executor = ProcessPoolExecutor(max_workers=MAX_WORKERS)
        iterator = tqdm(executor.map(evaluate_seed, tasks), total=len(tasks))
        results = iterator

    try:
        for seed, min_c, covered, counts in results:
            if covered == NUM_CLASSES:
                if min_c > best_min:
                    best_min = min_c
                    best_res = seed
                    best_counts = counts
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        print("💡 Hint: Try setting MAX_WORKERS = 1 to avoid memory crashes.")
        if MAX_WORKERS > 1: executor.shutdown(wait=False)
        return

    if best_res is None:
        print("\n❌ Failed to find a seed that covers all classes!")
        return

    weights = calculate_weights(best_counts)
    weight_str = ", ".join([f"{w:.2f}" for w in weights])

    print("\n" + "="*80)
    print("🏆 GOLDEN SEED FOUND & WEIGHTS CALCULATED 🏆")
    print("="*80)
    print(f"\n📊 Class Distribution (0.1% Subset):")
    print(f"{'Class':<12} | {'Count':<8} | {'Weight':<6}")
    print("-" * 34)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<12} | {best_counts[i]:<8} | {weights[i]:.2f}")
    print("-" * 34)
    print(f"🎯 Min Class Points: {best_min}")
    print("="*80)
    
    print("\n📋 [COPY TO DATASET CONFIG] (s3dis_co_train.py):")
    print(f"hash_seed_1={best_res[0]},")
    print(f"hash_seed_2={best_res[1]},")
    print(f"hash_seed_3={best_res[2]}")
    
    print("\n📋 [COPY TO MODEL CONFIG] (configs/s3dis/geo_ptv3.py -> class_weights):")
    print(f"class_weights=[{weight_str}]")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()