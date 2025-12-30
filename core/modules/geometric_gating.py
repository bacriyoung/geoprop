import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree
from scipy import sparse

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
    feats = np.concatenate([xyz * cfg['weights']['xyz'], rgb_n * cfg['weights']['rgb']], 1)
    
    sv = MiniBatchKMeans(max(10, len(xyz)//cfg['voxel_n']), batch_size=8192, 
                         n_init=1, max_iter=10, random_state=42).fit_predict(feats)
    
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