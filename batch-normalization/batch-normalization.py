import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x     = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta  = np.array(beta, dtype=float)

    if x.ndim == 2:
        # (N, D) — normalize over axis 0
        mean = x.mean(axis=0)
        var  = x.var(axis=0)
        x_hat = (x - mean) / np.sqrt(var + eps)
        return gamma * x_hat + beta

    else:
        # (N, C, H, W) — normalize over axes (0, 2, 3), keep C dim
        mean = x.mean(axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)
        var  = x.var(axis=(0, 2, 3),  keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        # Reshape gamma/beta to (1, C, 1, 1) for broadcasting
        g = gamma.reshape(1, -1, 1, 1)
        b = beta.reshape(1, -1, 1, 1)
        return g * x_hat + b