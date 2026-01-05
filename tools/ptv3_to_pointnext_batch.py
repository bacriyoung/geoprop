import numpy as np
import os
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from pathlib import Path

def process_single_scene(scene_info):
    """
    Worker function to process a single scene.
    scene_info: tuple (scene_path, output_dir)
    """
    scene_path, output_dir = scene_info
    
    # 1. Parse Area and Room name from path
    path_obj = Path(scene_path)
    
    # Extract names: assuming structure .../Area_N/Room_Name/coord.npy
    room_name = path_obj.name
    area_name = path_obj.parent.name
    
    output_filename = f"{area_name}_{room_name}.npy"
    save_path = os.path.join(output_dir, output_filename)

    try:
        # 2. Load data
        # PTV3 data loading
        coord = np.load(os.path.join(scene_path, "coord.npy")).astype(np.float32)
        color = np.load(os.path.join(scene_path, "color.npy")).astype(np.float32)
        normal = np.load(os.path.join(scene_path, "normal.npy")).astype(np.float32)
        sem = np.load(os.path.join(scene_path, "segment.npy")).astype(np.int64)
        ins = np.load(os.path.join(scene_path, "instance.npy")).astype(np.int64)

        # --- Label Safety Check ---
        # S3DIS usually has labels 0-12. PTV3 might use -1 for ignore.
        # Your trainer expects valid labels >= 0.
        # Map -1 to 255 (or a safe ignore index defined in your config, usually 255)
        sem[sem == -1] = 255

        # 3. Concatenate in the EXACT order your Dataset expects:
        # Index 0-2: Coord (XYZ)
        # Index 3-5: Color (RGB)
        # Index 6  : Semantic Label (lbl) <--- CRITICAL FIX
        # Index 7  : Instance Label (ins)
        # Index 8+ : Normal (nx, ny, nz)
        
        data = np.concatenate([
            coord,                          # [:, 0:3]
            color,                          # [:, 3:6]
            sem.reshape(-1, 1).astype(np.float32),  # [:, 6]  <- Target
            ins.reshape(-1, 1).astype(np.float32),  # [:, 7]
            normal                          # [:, 8:11]
        ], axis=1)

        # 4. Save
        np.save(save_path, data)
        return True
    except Exception as e:
        print(f"\nError processing {scene_path}: {e}")
        return False

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch convert PTV3 format to PointNeXt format")
    parser.add_argument("--src", type=str, required=True, help="Path to ptv3 s3dis root (containing Area_1, etc.)")
    parser.add_argument("--dest", type=str, required=True, help="Path to save processed .npy files")
    parser.add_argument("--num_workers", type=int, default=cpu_count() // 2, help="Number of parallel processes")
    args = parser.parse_args()

    src_root = Path(args.src)
    os.makedirs(args.dest, exist_ok=True)

    # 1. Find all scene directories (directories containing coord.npy)
    print(f"Scanning {src_root} for scenes...")
    # Use glob to find all coord.npy files and get their parent directories
    scene_dirs = [os.path.dirname(p) for p in glob.glob(str(src_root / "Area_*" / "*" / "coord.npy"))]
    
    if not scene_dirs:
        print("No scenes found! Please check if --src points to the directory containing Area_1, Area_2, etc.")
        return

    print(f"Found {len(scene_dirs)} scenes. Starting conversion using {args.num_workers} workers...")

    # 2. Prepare task list
    tasks = [(sd, args.dest) for sd in scene_dirs]

    # 3. Multi-processing execution
    # Using 'spawn' context if needed, but default 'fork' works fine on Linux
    with Pool(args.num_workers) as p:
        results = list(tqdm(p.imap(process_single_scene, tasks), total=len(tasks)))

    success_count = sum(results)
    print(f"\nDone! Successfully processed {success_count}/{len(tasks)} scenes.")
    print(f"Data saved to: {args.dest}")

if __name__ == "__main__":
    main()