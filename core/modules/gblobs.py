import torch

def get_knn_memory_efficient(xyz, k=32, chunk_size=1000):
    """
    Computes KNN indices using chunking.
    XYZ: [B, 3, N]
    """
    B, C, N = xyz.shape
    xyz_t = xyz.transpose(1, 2).contiguous() # [B, N, 3]
    all_indices = []
    
    # Chunk size can be slightly larger since we are not doing global anymore
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        query_chunk = xyz_t[:, i:end, :] 
        
        dist = torch.cdist(query_chunk, xyz_t)
        _, idx_chunk = torch.topk(dist, k=k, dim=-1, largest=False)
        all_indices.append(idx_chunk)
        
    return torch.cat(all_indices, dim=1) # [B, N, k]

def compute_covariance_features(features, knn_indices, k=32):
    """
    Generic Covariance (GBlobs) calculation.
    """
    B, C, N = features.shape
    feat_t = features.transpose(1, 2).contiguous() # [B, N, C]
    
    # Advanced indexing protection
    if N > 50000:
        chunk_gather = 20000
        gathered_list = []
        batch_idx_base = torch.arange(B, device=features.device).view(B, 1, 1)
        
        for i in range(0, N, chunk_gather):
            end = min(i+chunk_gather, N)
            sub_idx = knn_indices[:, i:end, :]
            # Proper expansion for the chunk
            sub_batch = batch_idx_base.expand(-1, end-i, k)
            gathered_list.append(feat_t[sub_batch, sub_idx, :])
        neighbors = torch.cat(gathered_list, dim=1)
    else:
        batch_idx = torch.arange(B, device=features.device).view(B, 1, 1).expand(-1, N, k)
        neighbors = feat_t[batch_idx, knn_indices, :]
    
    # 1. Centering
    local_mean = neighbors.mean(dim=2, keepdim=True) 
    centered = neighbors - local_mean 
    
    # 2. Covariance
    centered_t = centered.permute(0, 1, 3, 2) 
    cov = torch.matmul(centered_t, centered) / (k - 1 + 1e-6)
    
    # 3. Flatten
    cov_flat = cov.view(B, N, C*C).transpose(1, 2).contiguous() 
    
    return cov_flat

def compute_dual_gblobs(xyz, rgb, k=32):
    """
    Computes both Geometric GBlobs and Color GBlobs.
    """
    N = xyz.shape[2]
    
    # Adaptive strategy
    if N > 5000:
        idx = get_knn_memory_efficient(xyz, k, chunk_size=1000)
    else:
        xyz_t = xyz.transpose(1, 2).contiguous()
        dist = torch.cdist(xyz_t, xyz_t)
        _, idx = torch.topk(dist, k=k, dim=-1, largest=False)
        
    geo_blobs = compute_covariance_features(xyz, idx, k) * 1000.0
    rgb_blobs = compute_covariance_features(rgb, idx, k) * 100.0 
    
    return geo_blobs, rgb_blobs