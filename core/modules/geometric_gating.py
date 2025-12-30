import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KDTree
import pandas as pd

def compute_geometry_features(xyz, k=20):
    if len(xyz) > 50000: tree = KDTree(xyz[np.random.choice(len(xyz), 50000, replace=False)])
    else: tree = KDTree(xyz)
    _, indices = tree.query(xyz, k=k)
    diffs = xyz[indices] - xyz[indices].mean(axis=1, keepdims=True)
    vals, vecs = np.linalg.eigh(np.einsum('nij,nik->njk', diffs, diffs) / k)
    return 1.0 - (vals[:, 0] / (vals[:, 2] + 1e-6)), np.abs(vecs[:, :, 0][:, 2])

def run(cfg, xyz, rgb, labels):
    """ S2.1: V8 Geometric Gating """
    if not cfg['enabled']:
        return labels.copy()

    pla, ver = compute_geometry_features(xyz)
    
    if len(xyz) > 50000: 
        idx = np.random.choice(len(xyz), 50000, replace=False)
        s_l, s_p, s_v = labels[idx], pla[idx], ver[idx]
    else: 
        s_l, s_p, s_v = labels, pla, ver
    
    protos = {c: {'pla': s_p[s_l==c].mean() if (s_l==c).sum()>20 else 0.5, 
                  'ver': s_v[s_l==c].mean() if (s_l==c).sum()>20 else 0.5} 
              for c in range(13)}
    
    rgb_n = rgb/255.0 if rgb.max()>1.1 else rgb
    feats = np.concatenate([xyz * cfg['weights']['xyz'], rgb_n * cfg['weights']['rgb']], 1)
    
    sv = MiniBatchKMeans(max(10, len(xyz)//cfg['voxel_n']), batch_size=8192, 
                         n_init='auto', random_state=42).fit_predict(feats)
    
    cnt = np.bincount(sv) + 1e-6
    avg_p = np.bincount(sv, weights=pla)/cnt
    avg_v = np.bincount(sv, weights=ver)/cnt
    
    pp_p = np.array([protos[l]['pla'] if l!=-100 else 0.5 for l in labels])
    pp_v = np.array([protos[l]['ver'] if l!=-100 else 0.5 for l in labels])
    
    w = np.exp(-cfg['gate_strength'] * (np.abs(pp_p - avg_p[sv]) + np.abs(pp_v - avg_v[sv])))
    w[labels == -100] = 0
    
    refined = labels.copy()
    df = pd.DataFrame({'sv': sv, 'lbl': labels, 'w': w})
    df = df[df['lbl'] != -100]
    
    if len(df) > 0:
        g = df.groupby(['sv', 'lbl'])['w'].sum().reset_index().sort_values('w', ascending=False).drop_duplicates('sv')
        lookup = np.full(sv.max()+1, -1)
        lookup[g['sv'].values] = g['lbl'].values
        new = lookup[sv]
        refined[new != -1] = new[new != -1]
        
    return refined