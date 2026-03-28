import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    B, H, W, C = x.shape

    # Two 3×3 valid convs: each reduces H and W by 2
    H2, W2 = H - 4, W - 4

    skip_out = np.zeros((B, H2, W2, out_channels))
    pool_out = np.zeros((B, H2 // 2, W2 // 2, out_channels))

    return pool_out, skip_out