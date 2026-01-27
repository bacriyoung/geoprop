import os
import glob
import numpy as np
import gc
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ================= Configuration =================
# [Path] Point usually separates train/val folders for ScanNet
DATA_ROOT = 'data/scannet/train' 
NUM_TRIALS = 800        
LABELED_RATIO = 0.001   
NUM_CLASSES = 20
IGNORE_INDEX = -1

CLASS_NAMES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door",
    "window", "bookshelf", "picture", "counter", "desk", "curtain",
    "refridgerator", "shower curtain", "toilet", "sink", "bathtub",
    "otherfurniture"
]

# [CRITICAL SETTINGS]
# Reduce this if it crashes! (8 -> 4 -> 2 -> 0)
# Set to 0 or 1 to run sequentially (slow but safe)
MAX_WORKERS = 8
# =================================================

GLOBAL_SCENE_DATA = None

def load_all_data(data_root):
    """
    Load all ScanNet training scenes into memory.
    Structure: data_root/sceneXXXX_XX/coord.npy
    """
    print(f"📂 Loading ALL training data from {data_root} ...")
    
    # Search for all coord files inside subdirectories
    files = glob.glob(os.path.join(data_root, "*", "coord.npy"))
    files = sorted(files)
    
    scene_list = []
    
    for f in tqdm(files):
        try:
            # Load Coordinates
            coord = np.load(f).astype(np.float32)
            
            # Load Segments: Check segment20.npy first, then segment.npy
            scene_dir = os.path.dirname(f)
            seg_path_20 = os.path.join(scene_dir, "segment20.npy")
            seg_path_raw = os.path.join(scene_dir, "segment.npy")
            
            if os.path.exists(seg_path_20):
                segment = np.load(seg_path_20).astype(np.int64).reshape(-1)
            elif os.path.exists(seg_path_raw):
                segment = np.load(seg_path_raw).astype(np.int64).reshape(-1)
            else:
                continue # Skip if no label found
                
            scene_list.append((coord, segment))
        except Exception as e: 
            print(f"Error loading {f}: {e}")
            pass
            
    print(f"✅ Total scenes loaded: {len(scene_list)}")
    return scene_list

def evaluate_seed(args):
    global GLOBAL_SCENE_DATA
    seed_tuple, ratio = args
    h1_k, h2_k, h3_k = seed_tuple
    
    total_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    # Iterate scenes
    for coord, segment in GLOBAL_SCENE_DATA:
        # 1. Hashing Logic (Must match Dataset code exactly)
        h1 = np.abs(coord[:, 0] * h1_k).astype(np.int64)
        h2 = np.abs(coord[:, 1] * h2_k).astype(np.int64)
        h3 = np.abs(coord[:, 2] * h3_k).astype(np.int64)
        
        seed_hash = h1 ^ h2 ^ h3
        threshold = int(ratio * 100000)
        label_mask = (seed_hash % 100000) < threshold
        
        # 2. Filter Valid Points
        valid_labels = segment[label_mask]
        valid_labels = valid_labels[valid_labels != IGNORE_INDEX]
        
        # 3. Count
        if valid_labels.size > 0:
            counts = np.bincount(valid_labels, minlength=NUM_CLASSES)
            # Ensure we don't count out-of-bounds classes if any exist
            if len(counts) > NUM_CLASSES:
                counts = counts[:NUM_CLASSES]
            total_class_counts += counts
    
    min_count = np.min(total_class_counts)
    covered_classes = np.sum(total_class_counts > 0)
    
    # Explicit garbage collection
    del h1, h2, h3, seed_hash, label_mask, valid_labels
    # gc.collect() # Optional: too frequent GC slows down parallel processing
    
    return seed_tuple, min_count, covered_classes, total_class_counts

def calculate_weights(counts):
    """
    Standard Inverse Class Frequency Weighting
    """
    total = np.sum(counts)
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    for c in range(NUM_CLASSES):
        if counts[c] > 0:
            weights[c] = total / (counts[c] * NUM_CLASSES)
    
    # Clip weights to prevent explosion on rare classes
    weights = np.clip(weights, 1.0, 10.0)
    return weights

def main():
    global GLOBAL_SCENE_DATA
    
    if not os.path.exists(DATA_ROOT):
        print(f"❌ Error: Data root {DATA_ROOT} does not exist.")
        return

    GLOBAL_SCENE_DATA = load_all_data(DATA_ROOT)
    
    if len(GLOBAL_SCENE_DATA) == 0:
        print("❌ No data loaded. Check path structure.")
        return

    # Generate random seeds
    candidates = [np.random.randint(1e7, 1e8, 3).tolist() for _ in range(NUM_TRIALS)]
    tasks = [(c, LABELED_RATIO) for c in candidates]
    
    print(f"🚀 Searching best seeds among {NUM_TRIALS} candidates for ScanNet...")
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
            # Check if all classes are covered
            if covered == NUM_CLASSES:
                # We want to maximize the minimum points per class (balance)
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
        print("\n❌ Failed to find a seed that covers all 20 classes! Try increasing NUM_TRIALS.")
        return

    weights = calculate_weights(best_counts)
    weight_str = ", ".join([f"{w:.2f}" for w in weights])

    print("\n" + "="*80)
    print("🏆 GOLDEN SEED FOUND & WEIGHTS CALCULATED (ScanNet) 🏆")
    print("="*80)
    print(f"\n📊 Class Distribution ({LABELED_RATIO*100}% Subset):")
    print(f"{'Class':<20} | {'Count':<8} | {'Weight':<6}")
    print("-" * 42)
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<20} | {best_counts[i]:<8} | {weights[i]:.2f}")
    print("-" * 42)
    print(f"🎯 Min Class Points: {best_min}")
    print("="*80)
    
    print("\n📋 [COPY TO DATASET CONFIG] (pointcept/datasets/scannet_geo.py):")
    print(f"hash_seed_1={best_res[0]},")
    print(f"hash_seed_2={best_res[1]},")
    print(f"hash_seed_3={best_res[2]}")
    
    print("\n📋 [COPY TO MODEL CONFIG] (configs/scannet/geo_ptv3.py -> class_weights):")
    print(f"class_weights=[{weight_str}]")
    print("\n" + "="*80)

if __name__ == "__main__":
    main()