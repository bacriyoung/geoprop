import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.cluster import MiniBatchKMeans
from scipy import sparse

def fast_mode_voting(sv_ids, labels, n_classes=13):
    """
    使用稀疏矩阵加速众数投票 (替代 pandas groupby + mode)
    Args:
        sv_ids: (N,) 体素ID
        labels: (N,) 标签
    Returns:
        votes: (max_sv_id+1,) 每个体素的众数标签
    """
    if len(sv_ids) == 0: return np.array([])
    
    # 过滤无效标签
    mask = labels >= 0
    sv = sv_ids[mask]
    lbl = labels[mask]
    
    if len(sv) == 0: return np.full(sv_ids.max() + 1, -100)

    n_sv = sv.max() + 1
    # 构建稀疏矩阵: rows=体素, cols=类别, values=计数
    # coo_matrix 会自动累加重复坐标的值
    data = np.ones(len(sv), dtype=int)
    mat = sparse.coo_matrix((data, (sv, lbl)), shape=(n_sv, n_classes)).tocsr()
    
    # 对每一行求最大值的索引 (argmax)
    # toarray() 将稀疏转稠密，因为 n_classes 很小(13)，这很快
    votes = np.argmax(mat.toarray(), axis=1)
    
    # 处理没有投票的空体素 (全0行)，设为 -100
    row_sums = mat.getnnz(axis=1)
    votes[row_sums == 0] = -100
    
    return votes

def run(cfg, xyz, rgb, pred_lbl, shape_guide, seed_mask, gt_lbl, confidence=None):
    if not cfg['enabled']:
        return pred_lbl

    if rgb.max() > 1.1: rgb = rgb / 255.0
    N = len(xyz)
    
    # 1. Fine Discretization
    # 加速点：减少聚类迭代次数 (n_init=1, max_iter=10 足够了)
    feats_f = np.concatenate([xyz * cfg['fine_weight_xyz'], rgb * 20.0], axis=1)
    n_clus_f = max(50, N // cfg['fine_voxel_n'])
    km_f = MiniBatchKMeans(n_clusters=n_clus_f, batch_size=8192, 
                           n_init=1, max_iter=20, random_state=42) # 优化参数
    sv_f = km_f.fit_predict(feats_f)
    
    # 平滑体素ID (Voting)
    tree_pts = KDTree(xyz)
    _, idx_pts = tree_pts.query(xyz, k=4)
    # 这里用 mode 比较慢，但 KNN k=4 还可以接受，也可以用 fast_mode_voting 优化
    # 为了保持逻辑简单，这里暂保留 stats.mode 或手写逻辑
    # 简单的众数逻辑：
    neighbor_sv = sv_f[idx_pts] # (N, 4)
    # 快速行众数
    from scipy import stats
    sv_f = stats.mode(neighbor_sv, axis=1, keepdims=True)[0].flatten()
    
    # 2. Map Shape Guide Labels to Fine SVs [核心加速]
    # 使用稀疏矩阵替代 pandas groupby
    sv_labels_arr = fast_mode_voting(sv_f, shape_guide)
    
    # 3. Graph Cleaning
    # 计算质心 (这一步用 bincount 加速)
    n_sv = sv_f.max() + 1
    count = np.bincount(sv_f, minlength=n_sv)
    # 防止除以0
    count[count==0] = 1
    
    # 快速计算 sum
    # 利用 coo_matrix 进行 reduce sum
    # row=sv_f, col=0 (dummy), data=coord
    centroids = np.zeros((n_sv, 3))
    for i in range(3):
        centroids[:, i] = np.bincount(sv_f, weights=xyz[:, i], minlength=n_sv) / count
        
    # 只保留存在的体素
    present_mask = np.bincount(sv_f, minlength=n_sv) > 0
    present_ids = np.where(present_mask)[0]
    active_centroids = centroids[present_ids]
    
    if len(active_centroids) > cfg['k_neighbors']:
        knn_graph = NearestNeighbors(n_neighbors=cfg['k_neighbors'], n_jobs=-1).fit(active_centroids)
        _, neighbor_indices = knn_graph.kneighbors(active_centroids)
        
        # 获取邻居的标签
        # neighbor_indices 是在 active_centroids 中的索引，需要映射回 real_sv_ids
        neighbor_real_sv_ids = present_ids[neighbor_indices]
        neighbor_labels = sv_labels_arr[neighbor_real_sv_ids] # (M, k)
        
        # 对邻居标签求众数
        refined_labels_active, _ = stats.mode(neighbor_labels, axis=1, keepdims=True)
        refined_labels_active = refined_labels_active.flatten()
        
        # 更新查找表
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