import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    # x: (batch, H, W, C)
    batch, H, W, C = x.shape
    H_out, W_out   = H // 2, W // 2

    # Reshape to group 2×2 non-overlapping windows, then take max
    x = x[:, :H_out*2, :W_out*2, :]                          # trim if odd
    x = x.reshape(batch, H_out, 2, W_out, 2, C)
    return x.max(axis=(2, 4))                                 # (batch, H_out, W_out, C)