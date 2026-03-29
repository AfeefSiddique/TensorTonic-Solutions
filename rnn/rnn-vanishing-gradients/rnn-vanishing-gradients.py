import numpy as np

def compute_gradient_norm_decay(T: int, W_hh: np.ndarray) -> list:
    spectral_norm = np.linalg.norm(W_hh, ord=2)
    return [spectral_norm ** t for t in range(T)]