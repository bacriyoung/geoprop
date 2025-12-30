import scipy.stats as stats
from sklearn.neighbors import KDTree

def run(cfg, xyz, labels):
    """ Spatial Smoothing """
    if not cfg['enabled']:
        return labels
        
    tree = KDTree(xyz)
    _, idx = tree.query(xyz, k=cfg['k_neighbors'])
    neighbor_lbls = labels[idx]
    modes, _ = stats.mode(neighbor_lbls, axis=1, keepdims=True)
    return modes.flatten()