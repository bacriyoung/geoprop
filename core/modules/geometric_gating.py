import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree
from scipy import sparse

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



def compute_geometry_features(xyz, k=20):
    if len(xyz) > 20000:
        idx = np.random.choice(len(xyz), 20000, replace=False)
        tree = KDTree(xyz[idx])
    else:
        tree = KDTree(xyz)
    _, indices = tree.query(xyz, k=k)
    diffs = xyz[indices] - xyz[indices].mean(axis=1, keepdims=True)
    vals, vecs = np.linalg.eigh(np.einsum('nij,nik->njk', diffs, diffs) / k)
    return 1.0 - (vals[:, 0] / (vals[:, 2] + 1e-6)), np.abs(vecs[:, :, 0][:, 2])

def run(cfg, xyz, rgb, labels, confidence=None):
    if not cfg['enabled']:
        return labels.copy()

    num_classes = cfg.get('num_classes', 13)

    pla, ver = compute_geometry_features(xyz)
    
    valid_mask = labels != -100
    if not valid_mask.any(): return labels
    
    l_valid = labels[valid_mask]
    p_valid = pla[valid_mask]
    v_valid = ver[valid_mask]
    
    counts = np.bincount(l_valid, minlength=num_classes)
    p_sums = np.bincount(l_valid, weights=p_valid, minlength=num_classes)
    v_sums = np.bincount(l_valid, weights=v_valid, minlength=num_classes)
    
    safe_counts = counts.copy()
    safe_counts[safe_counts==0] = 1
    proto_pla = p_sums / safe_counts
    proto_ver = v_sums / safe_counts
    
    protos = {c: {'pla': proto_pla[c], 'ver': proto_ver[c]} for c in range(num_classes)}
    
    rgb_n = rgb/255.0 if rgb.max()>1.1 else rgb

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    xyz_t = torch.from_numpy(xyz).float().to(device)
    rgb_t = torch.from_numpy(rgb_n).float().to(device)

    feats_t = torch.cat([xyz_t * cfg['weights']['xyz'], rgb_t * cfg['weights']['rgb']], 1)
    
    n_clusters = max(10, len(xyz)//cfg['voxel_n'])
    sv = fast_gpu_kmeans(feats_t, n_clusters, max_iter=10)
    
    n_sv = sv.max() + 1
    sv_counts = np.bincount(sv, minlength=n_sv) + 1e-6
    avg_p = np.bincount(sv, weights=pla, minlength=n_sv) / sv_counts
    avg_v = np.bincount(sv, weights=ver, minlength=n_sv) / sv_counts
    
    safe_labels = labels.copy()
    safe_labels[labels == -100] = 0 
    pp_p = proto_pla[safe_labels]
    pp_v = proto_ver[safe_labels]
    
    w = np.exp(-cfg['gate_strength'] * (np.abs(pp_p - avg_p[sv]) + np.abs(pp_v - avg_v[sv])))
    w[labels == -100] = 0
    
    mask = labels != -100
    sv_valid = sv[mask]
    lbl_valid = labels[mask]
    w_valid = w[mask]
    
    if len(sv_valid) == 0: return labels

    mat = sparse.coo_matrix((w_valid, (sv_valid, lbl_valid)), shape=(n_sv, num_classes)).tocsr()
    
    dense_sums = mat.toarray() 
    best_labels = np.argmax(dense_sums, axis=1)
    
    has_votes = dense_sums.sum(axis=1) > 0
    
    refined_candidate = labels.copy()
    update_mask = has_votes[sv]
    refined_candidate[update_mask] = best_labels[sv[update_mask]]

    if confidence is not None:
        threshold = cfg.get('confidence_threshold', 0.65)
        protect_mask = confidence > threshold
        refined_candidate[protect_mask] = labels[protect_mask]

    return refined_candidate