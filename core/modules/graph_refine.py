import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.cluster import MiniBatchKMeans
from scipy import sparse
from scipy import stats

@torch.no_grad()
def fast_gpu_kmeans(x_tensor, n_clusters, max_iter=30, tol=1e-4):
    """

    """
    N, D = x_tensor.shape
    device = x_tensor.device
    
    safe_chunk = int(5 * 10**8 / (n_clusters * 4 + 1))
    chunk_size = min(50000, safe_chunk) 
    chunk_size = max(1000, chunk_size)

    indices = torch.randperm(N, device=device)[:n_clusters]
    centroids = x_tensor[indices].clone()
    
    for i in range(max_iter):
        old_centroids = centroids.clone()
        
        cluster_sum = torch.zeros((n_clusters, D), device=device)
        cluster_count = torch.zeros(n_clusters, device=device)
        
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            x_chunk = x_tensor[start:end]
            
            dists = torch.cdist(x_chunk, centroids)
            labels = torch.argmin(dists, dim=1)
            
            cluster_sum.index_add_(0, labels, x_chunk)

            counts = torch.bincount(labels, minlength=n_clusters).float()
            cluster_count += counts

        mask = cluster_count > 0
        
        centroids[mask] = cluster_sum[mask] / cluster_count[mask].unsqueeze(1)
        
        if (~mask).any():
            empty_indices = torch.where(~mask)[0]
            valid_indices = torch.where(mask)[0]
            new_seeds = x_tensor[torch.randint(0, N, (len(empty_indices),), device=device)]
            centroids[empty_indices] = new_seeds

        shift = torch.norm(centroids - old_centroids, dim=1).mean()
        if shift < tol:
            break
            
    final_labels = []
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        x_chunk = x_tensor[start:end]
        dists = torch.cdist(x_chunk, centroids)
        final_labels.append(torch.argmin(dists, dim=1))
    
    del cluster_sum, cluster_count, old_centroids
    
    return torch.cat(final_labels).cpu().numpy()


def fast_mode_voting(sv_ids, labels, n_classes):
    """
    """
    if len(sv_ids) == 0: return np.array([])
    
    mask = labels >= 0
    sv = sv_ids[mask]
    lbl = labels[mask]
    
    if len(sv) == 0: return np.full(sv_ids.max() + 1, -100)

    n_sv = sv.max() + 1
    data = np.ones(len(sv), dtype=int)
    mat = sparse.coo_matrix((data, (sv, lbl)), shape=(n_sv, n_classes)).tocsr()
    
    votes = np.argmax(mat.toarray(), axis=1)
    
    row_sums = mat.getnnz(axis=1)
    votes[row_sums == 0] = -100
    
    return votes

def run(cfg, xyz, rgb, pred_lbl, shape_guide, seed_mask, gt_lbl=None, confidence=None):
    """
    Graph Refinement Module.
    [FIX] Made gt_lbl optional to prevent crashing if not provided.
    """
    if not cfg['enabled']:
        return pred_lbl

    num_classes = cfg.get('num_classes', 13)

    if rgb.max() > 1.1: rgb = rgb / 255.0
    N = len(xyz)
    
    # Supervoxel Clustering (GPU Accelerated)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xyz_t = torch.from_numpy(xyz).float().to(device)
    rgb_t = torch.from_numpy(rgb).float().to(device)
    
    feats_t = torch.cat([xyz_t * cfg['fine_weight_xyz'], rgb_t * 20.0], 1)
    
    n_clus_f = max(50, N // cfg['fine_voxel_n'])
    
    sv_f = fast_gpu_kmeans(feats_t, n_clus_f, max_iter=20)
    
    # Smooth SV assignments
    tree_pts = KDTree(xyz)
    _, idx_pts = tree_pts.query(xyz, k=4)
    neighbor_sv = sv_f[idx_pts]
    sv_f = stats.mode(neighbor_sv, axis=1, keepdims=True)[0].flatten()
    
    # Initial Vote
    sv_labels_arr = fast_mode_voting(sv_f, shape_guide, n_classes=num_classes)
    
    # Graph Construction
    n_sv = sv_f.max() + 1
    count = np.bincount(sv_f, minlength=n_sv)
    count[count==0] = 1
    
    centroids = np.zeros((n_sv, 3))
    for i in range(3):
        centroids[:, i] = np.bincount(sv_f, weights=xyz[:, i], minlength=n_sv) / count
        
    present_mask = np.bincount(sv_f, minlength=n_sv) > 0
    present_ids = np.where(present_mask)[0]
    active_centroids = centroids[present_ids]
    
    # Graph Smoothing
    if len(active_centroids) > cfg['k_neighbors']:
        knn_graph = NearestNeighbors(n_neighbors=cfg['k_neighbors'], n_jobs=-1).fit(active_centroids)
        _, neighbor_indices = knn_graph.kneighbors(active_centroids)
        
        neighbor_real_sv_ids = present_ids[neighbor_indices]
        neighbor_labels = sv_labels_arr[neighbor_real_sv_ids]
        
        refined_labels_active, _ = stats.mode(neighbor_labels, axis=1, keepdims=True)
        refined_labels_active = refined_labels_active.flatten()
        
        final_lookup = sv_labels_arr.copy()
        final_lookup[present_ids] = refined_labels_active
    else:
        final_lookup = sv_labels_arr
        
    final_lbl = final_lookup[sv_f]
    
    # Restore -100 (Unlabeled)
    final_lbl[pred_lbl == -100] = -100
    
    # [FIX] Enforce seeds only if gt_lbl is provided
    if seed_mask is not None and gt_lbl is not None:
        final_lbl[seed_mask] = gt_lbl[seed_mask]

    # Confidence Protection
    if confidence is not None:
        thresh = cfg.get('confidence_threshold', 0.65)
        protect_mask = confidence > thresh
        final_lbl[protect_mask] = pred_lbl[protect_mask]
        
    return final_lbl