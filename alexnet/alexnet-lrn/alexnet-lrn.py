import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                  alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    B, H, W, C = x.shape
    out = np.zeros_like(x)
    half = n // 2

    for i in range(C):
        j_start = max(0, i - half)
        j_end   = min(C, i + half + 1)
        norm    = k + alpha * np.sum(x[:, :, :, j_start:j_end] ** 2, axis=3, keepdims=True)
        out[:, :, :, i:i+1] = x[:, :, :, i:i+1] / (norm ** beta)

    return out