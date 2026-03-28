import numpy as np

def vgg_conv_block(x: np.ndarray, num_convs: int, out_channels: int) -> np.ndarray:
    # x shape: (batch, H, W, C_in) — channels last (TF/Keras convention)
    batch, H, W, C_in = x.shape

    for i in range(num_convs):
        in_ch  = C_in if i == 0 else out_channels
        # 3×3 conv weights: (3, 3, in_ch, out_ch)
        W_conv = np.random.randn(3, 3, in_ch, out_channels) * np.sqrt(2.0 / (9 * in_ch))
        b_conv = np.zeros(out_channels)

        # Same padding: pad H and W by 1 on each side
        x_pad = np.pad(x, ((0,0),(1,1),(1,1),(0,0)), mode='constant')

        # Convolution via im2col-style loop over spatial positions
        out = np.zeros((batch, H, W, out_channels))
        for h in range(H):
            for w in range(W):
                patch = x_pad[:, h:h+3, w:w+3, :]          # (batch, 3, 3, in_ch)
                out[:, h, w, :] = (
                    patch.reshape(batch, -1) @
                    W_conv.reshape(-1, out_channels)
                ) + b_conv

        x = np.maximum(0, out)   # ReLU

    return x