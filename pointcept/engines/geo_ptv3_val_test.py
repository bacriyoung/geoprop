import torch
import numpy as np

# =========================================================================
# Sliding Window Generator (Strictly replicates training distribution)
# =========================================================================
def chunk_scene_data(coord, feat, block_size=2.0, stride=1.0, min_points=1000):
    """
    Splits a large scene into overlapping chunks (3D blocks).
    Each chunk is centered and normalized, mimicking the training phase.
    
    Args:
        coord: [N, 3] Full scene coordinates (float).
        feat: [N, C] Full scene features (RGB, etc.).
        block_size: Size of the window in meters (keep consistent with train crop size).
        stride: Stride of the window (overlap is block_size - stride).
        min_points: Minimum points required to process a chunk.
    
    Returns:
        List of dictionaries containing chunk data and global indices.
    """
    # Calculate grid range
    coord_min = np.min(coord, 0)
    coord_max = np.max(coord, 0)
    
    sx = np.arange(coord_min[0], coord_max[0], stride)
    sy = np.arange(coord_min[1], coord_max[1], stride)
    
    chunks = []
    
    for x in sx:
        for y in sy:
            # Select points within the current block [x, x+block_size]
            x_cond = (coord[:, 0] >= x) & (coord[:, 0] < x + block_size)
            y_cond = (coord[:, 1] >= y) & (coord[:, 1] < y + block_size)
            mask = x_cond & y_cond
            
            # Skip empty or sparse blocks
            if np.sum(mask) < min_points:
                continue
            
            # Extract Global Indices (Crucial for Voting)
            indices = np.where(mask)[0]
            
            # Extract Data
            chunk_coord = coord[indices]
            chunk_feat = feat[indices]
            
            # [CRITICAL] Center the block coordinates! 
            # This ensures the coordinate values are in the same range as training crops.
            # Without this, coordinates would be too large and destabilize the network.
            chunk_coord_centered = chunk_coord - chunk_coord.mean(0)
            
            chunks.append({
                "coord": chunk_coord_centered,  # Normalized coords for network
                "raw_coord": chunk_coord,       # Original coords (if needed)
                "feat": chunk_feat,
                "index": indices                # Global indices for scatter add
            })
            
    return chunks

# =========================================================================
# Inference Loop with Voting (CPU Memory Efficient)
# =========================================================================
def infer_scene_sliding_window(model, input_dict, num_classes=13, block_size=2.0, stride=1.0):
    full_coord = input_dict["coord"].cpu().numpy()
    full_feat = input_dict["feat"].cpu().numpy()
    total_points = full_coord.shape[0]
    
    global_logits = torch.zeros((total_points, num_classes), dtype=torch.float16)
    global_count = torch.zeros((total_points), dtype=torch.float16)
    
    # Generate Chunks
    chunks = chunk_scene_data(full_coord, full_feat, block_size=block_size, stride=stride)
    
    model.eval()
    
    with torch.no_grad():
        for i, chunk in enumerate(chunks):
            chunk_coord_tensor = torch.from_numpy(chunk["coord"]).float().cuda()
            chunk_feat_tensor = torch.from_numpy(chunk["feat"]).float().cuda()
            
            # [Fix] Construct Offset for PTV3
            offset = torch.tensor([chunk_coord_tensor.shape[0]], device="cuda", dtype=torch.int32)
            
            chunk_input = {
                "coord": chunk_coord_tensor,
                "feat": chunk_feat_tensor,
                "batch": torch.zeros(chunk_coord_tensor.shape[0], dtype=torch.long).cuda(),
                "offset": offset, 
                "grid_size": 0.04 # 显式传入 grid_size 有时能避免某些 voxelization bug，视你 config 而定
            }
            
            # Inference
            ret_dict = model(chunk_input)
            pred_logits = ret_dict["seg_logits"]
            
            # Probability
            pred_prob = torch.nn.functional.softmax(pred_logits, dim=-1).cpu().float()
            
            # Voting
            global_idx = torch.from_numpy(chunk["index"]).long()
            if pred_prob.dim() == 3: pred_prob = pred_prob.squeeze(0)
            
            global_logits[global_idx] += pred_prob
            global_count[global_idx] += 1
            
            # Aggressive Memory Cleanup
            del chunk_input, ret_dict, pred_logits, pred_prob, chunk_coord_tensor, chunk_feat_tensor
            
    # Finalize
    global_count = global_count.clamp(min=1.0).unsqueeze(-1).float()
    final_prob = global_logits.float() / global_count
    final_pred = final_prob.max(1)[1].numpy()
    
    return final_pred