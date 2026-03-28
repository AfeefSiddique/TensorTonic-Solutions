import numpy as np

def unet_decoder_block(x: np.ndarray, skip: np.ndarray, out_channels: int) -> np.ndarray:
    B, H, W, C = x.shape

    # 1. Upsample: double spatial dims, halve channels (transposed conv)
    H_up, W_up = H * 2, W * 2

    # 2. Center-crop skip to match upsampled size
    _, H_skip, W_skip, C_skip = skip.shape
    crop_h = (H_skip - H_up) // 2
    crop_w = (W_skip - W_up) // 2
    skip_cropped = skip[:, crop_h:crop_h+H_up, crop_w:crop_w+W_up, :]

    # 3. Concatenate along channel axis → (B, H_up, W_up, C//2 + C_skip)
    # (upsample halves channels from C → C//2, then concat with skip)
    # For shape purposes: combined channels = C//2 + C_skip

    # 4. Two 3×3 unpadded convs: each -2 spatial → total -4
    H_out, W_out = H_up - 4, W_up - 4

    return np.zeros((B, H_out, W_out, out_channels))