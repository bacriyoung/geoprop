import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.cluster import MiniBatchKMeans
from scipy import sparse

def fast_mode_voting(sv_ids, labels, n_classes=13):
    """
    Args:
        sv_ids: (N,) 
        labels: (N,) 
    Returns:
        votes: (max_sv_id+1,) 
    """
    if len(sv_ids) == 0: return np.array([])
    
    # filter
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

def run(cfg, xyz, rgb, pred_lbl, shape_guide, seed_mask, gt_lbl, confidence=None):
    if not cfg['enabled']:
        return pred_lbl

    if rgb.max() > 1.1: rgb = rgb / 255.0
    N = len(xyz)
    
    # 1. Fine Discretization
    feats_f = np.concatenate([xyz * cfg['fine_weight_xyz'], rgb * 20.0], axis=1)
    n_clus_f = max(50, N // cfg['fine_voxel_n'])
    km_f = MiniBatchKMeans(n_clusters=n_clus_f, batch_size=8192, 
                           n_init=1, max_iter=20, random_state=42) # 优化参数
    sv_f = km_f.fit_predict(feats_f)
    
    tree_pts = KDTree(xyz)
    _, idx_pts = tree_pts.query(xyz, k=4)

    neighbor_sv = sv_f[idx_pts] # (N, 4)

    from scipy import stats
    sv_f = stats.mode(neighbor_sv, axis=1, keepdims=True)[0].flatten()
    
    # 2. Map Shape Guide Labels to Fine SVs 
    sv_labels_arr = fast_mode_voting(sv_f, shape_guide)
    
    # 3. Graph Cleaning
    n_sv = sv_f.max() + 1
    count = np.bincount(sv_f, minlength=n_sv)
    count[count==0] = 1
    
    # row=sv_f, col=0 (dummy), data=coord
    centroids = np.zeros((n_sv, 3))
    for i in range(3):
        centroids[:, i] = np.bincount(sv_f, weights=xyz[:, i], minlength=n_sv) / count
        
    present_mask = np.bincount(sv_f, minlength=n_sv) > 0
    present_ids = np.where(present_mask)[0]
    active_centroids = centroids[present_ids]
    
    if len(active_centroids) > cfg['k_neighbors']:
        knn_graph = NearestNeighbors(n_neighbors=cfg['k_neighbors'], n_jobs=-1).fit(active_centroids)
        _, neighbor_indices = knn_graph.kneighbors(active_centroids)
        
        neighbor_real_sv_ids = present_ids[neighbor_indices]
        neighbor_labels = sv_labels_arr[neighbor_real_sv_ids] # (M, k)
        
        refined_labels_active, _ = stats.mode(neighbor_labels, axis=1, keepdims=True)
        refined_labels_active = refined_labels_active.flatten()
        
        final_lookup = sv_labels_arr.copy()
        final_lookup[present_ids] = refined_labels_active
    else:
        final_lookup = sv_labels_arr
        
    final_lbl = final_lookup[sv_f]
    
    # Restore holes and seeds
    final_lbl[pred_lbl == -100] = -100
    if seed_mask is not None:
        final_lbl[seed_mask] = gt_lbl[seed_mask]

    # Confidence Protection
    if confidence is not None:
        thresh = cfg.get('confidence_threshold', 0.65)
        protect_mask = confidence > thresh
        final_lbl[protect_mask] = pred_lbl[protect_mask]
        
    return final_lbl