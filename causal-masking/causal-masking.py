import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    scores = np.array(scores, dtype=float)
    T = scores.shape[-1]
    # Upper triangle excluding diagonal → future positions
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    result = scores.copy()
    result[..., mask] = mask_value
    return result