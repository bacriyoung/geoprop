import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.cluster import MiniBatchKMeans
import scipy.stats as stats

def run(cfg, xyz, rgb, pred_lbl, lbl_aggressive, seed_mask, gt_lbl):
    """ S3: Graph Cut Refinement """
    if not cfg['enabled']:
        return pred_lbl

    if rgb.max() > 1.1: rgb = rgb / 255.0
    N = len(xyz)
    
    feats_f = np.concatenate([xyz * cfg['fine_weight_xyz'], rgb * 20.0], axis=1)
    n_clus_f = max(50, N // cfg['fine_voxel_n'])
    km_f = MiniBatchKMeans(n_clusters=n_clus_f, batch_size=8192, n_init='auto', random_state=42)
    sv_f = km_f.fit_predict(feats_f)
    
    tree_pts = KDTree(xyz)
    _, idx_pts = tree_pts.query(xyz, k=4)
    sv_f = stats.mode(sv_f[idx_pts], axis=1, keepdims=True)[0].flatten()
    
    df = pd.DataFrame({'sv': sv_f, 'lbl': lbl_aggressive})
    sv_labels_series = df.groupby('sv')['lbl'].agg(lambda x: stats.mode(x, keepdims=True)[0][0])
    
    max_sv_id = sv_f.max()
    sv_labels_arr = np.full(max_sv_id + 1, -100, dtype=int)
    sv_labels_arr[sv_labels_series.index] = sv_labels_series.values
    
    df_xyz = pd.DataFrame(xyz, columns=['x','y','z'])
    df_xyz['sv'] = sv_f
    grouped = df_xyz.groupby('sv')
    centroids = grouped.mean().values
    present_sv_ids = grouped.mean().index.values
    
    if len(centroids) > cfg['k_neighbors']:
        knn_graph = NearestNeighbors(n_neighbors=cfg['k_neighbors'], n_jobs=-1).fit(centroids)
        _, neighbor_indices = knn_graph.kneighbors(centroids)
        
        neighbor_real_sv_ids = present_sv_ids[neighbor_indices]
        neighbor_labels = sv_labels_arr[neighbor_real_sv_ids]
        
        refined_labels, _ = stats.mode(neighbor_labels, axis=1, keepdims=True)
        refined_labels = refined_labels.flatten()
        
        final_lookup = sv_labels_arr.copy()
        final_lookup[present_sv_ids] = refined_labels
    else:
        final_lookup = sv_labels_arr
        
    final_lbl = final_lookup[sv_f]
    
    final_lbl[pred_lbl == -100] = -100
    if seed_mask is not None:
        final_lbl[seed_mask] = gt_lbl[seed_mask]
        
    return final_lbl