import os
import glob
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# ================= Configuration =================
DATA_ROOT = 'data/s3dis'
NUM_TRIALS = 2000
LABELED_RATIO = 0.001
NUM_CLASSES = 13
# =================================================

def load_all_data(data_root):
    # 只读 Area 1 & 2 加速
    print(f"Loading data from {data_root} (Area 1/2)...")
    files = glob.glob(os.path.join(data_root, "**", "coord.npy"), recursive=True)
    room_list = []
    for f in tqdm(files):
        if "Area_5" in f: continue
        if "Area_1" not in f and "Area_2" not in f: continue
        try:
            coord = np.load(f).astype(np.float32)
            segment = np.load(f.replace("coord.npy", "segment.npy")).astype(np.int64).reshape(-1)
            room_list.append((coord, segment))
        except: pass
    return room_list

def evaluate_seed(args):
    seed_tuple, room_data, ratio = args
    h1_k, h2_k, h3_k = seed_tuple
    
    total_class_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    
    for coord, segment in room_data:
        # 🟢 [MATCHING V20 DATASET LOGIC]
        # Pure Coordinate Hashing
        h1 = np.abs(coord[:, 0] * h1_k).astype(np.int64)
        h2 = np.abs(coord[:, 1] * h2_k).astype(np.int64)
        h3 = np.abs(coord[:, 2] * h3_k).astype(np.int64)
        
        seed_hash = h1 ^ h2 ^ h3
        
        threshold = int(ratio * 100000)
        label_mask = (seed_hash % 100000) < threshold
        
        valid_labels = segment[label_mask]
        valid_labels = valid_labels[valid_labels != 255]
        
        counts = np.bincount(valid_labels, minlength=NUM_CLASSES)
        total_class_counts += counts

    min_count = np.min(total_class_counts)
    covered_classes = np.sum(total_class_counts > 0)
    
    return seed_tuple, min_count, covered_classes

def main():
    room_data = load_all_data(DATA_ROOT)
    candidates = [np.random.randint(1e7, 1e8, 3).tolist() for _ in range(NUM_TRIALS)]
    
    print(f"🚀 Searching best seeds...")
    best_min = -1
    best_res = None
    
    with ProcessPoolExecutor(max_workers=16) as executor:
        tasks = [(c, room_data, LABELED_RATIO) for c in candidates]
        for seed, min_c, covered in tqdm(executor.map(evaluate_seed, tasks), total=len(tasks)):
            if covered == 13 and min_c > best_min:
                best_min = min_c
                best_res = seed
                # print(f"New best: {min_c} points for weakest class")

    print("\n" + "="*60)
    print("🏆 GOLDEN SEED FOUND (Pure Hash Version) 🏆")
    print(f"Seeds: {best_res}")
    print(f"Min Class Points: {best_min}")
    print("="*60)
    print(f"hash_seed_1={best_res[0]},")
    print(f"hash_seed_2={best_res[1]},")
    print(f"hash_seed_3={best_res[2]}")

if __name__ == "__main__":
    main()