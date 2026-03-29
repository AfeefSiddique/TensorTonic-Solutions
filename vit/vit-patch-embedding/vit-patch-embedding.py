import numpy as np

def patch_embed(image: np.ndarray, patch_size: int, embed_dim: int) -> np.ndarray:
    B, H, W, C = image.shape
    P          = patch_size
    N          = (H // P) * (W // P)   # number of patches
    patch_dim  = P * P * C

    # Reshape into patches: (B, H/P, P, W/P, P, C)
    x = image.reshape(B, H//P, P, W//P, P, C)
    # Reorder to (B, N, patch_dim)
    x = x.transpose(0, 1, 3, 2, 4, 5).reshape(B, N, patch_dim)

    # Linear projection: patch_dim → embed_dim
    W_proj = np.random.randn(patch_dim, embed_dim) * 0.02
    return x @ W_proj   # (B, N, embed_dim)