import numpy as np

def get_fixed_seeds(N, ratio, labels, seed):
    rng = np.random.default_rng(seed) 
    M = max(int(N * ratio), 20)
    
    seeds = []
    for c in np.unique(labels):
        idxs = np.where(labels==c)[0]
        if len(idxs) > 0:
            seeds.append(rng.choice(idxs))
    seeds = np.array(seeds)
    
    if len(seeds) < M:
        extra_count = M - len(seeds)
        mask = np.ones(N, dtype=bool)
        mask[seeds] = False
        candidates = np.where(mask)[0]
        if len(candidates) > 0:
            if len(candidates) < extra_count:
                extra_seeds = candidates
            else:
                extra_seeds = rng.choice(candidates, extra_count, replace=False)
            seeds = np.concatenate([seeds, extra_seeds])
            
    return seeds.astype(int)