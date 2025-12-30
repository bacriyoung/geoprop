import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def run(cfg, rec, bdy_pred, b_rgb, pred_block, mask, lbl_container):
    """ S1.1: Confidence Filter """
    if not cfg['enabled']:
        return lbl_container

    err = torch.abs(rec - b_rgb.transpose(1,2)).mean(1).squeeze().cpu().numpy()
    bdy_val = bdy_pred.squeeze().cpu().numpy()
    
    thr = cfg['rec_err_loose'] - (cfg['rec_err_loose'] - cfg['rec_err_strict']) * bdy_val
    trust = err < thr
    
    lbl_container[mask[trust]] = pred_block[trust]
    return lbl_container

def knn_fill(xyz, labels):
    if (labels == -100).sum() > 0:
        valid = labels != -100
        if valid.sum() > 0:
            knn = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(xyz[valid])
            labels[~valid] = labels[valid][knn.kneighbors(xyz[~valid])[1].flatten()]
    return labels