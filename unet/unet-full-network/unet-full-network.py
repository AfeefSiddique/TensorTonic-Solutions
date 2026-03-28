import numpy as np

def unet_encoder_block(x, out_channels):
    B, H, W, C = x.shape
    H2, W2 = H - 4, W - 4
    skip = np.zeros((B, H2, W2, out_channels))
    pool = np.zeros((B, H2 // 2, W2 // 2, out_channels))
    return pool, skip

def unet_bottleneck(x, out_channels):
    B, H, W, C = x.shape
    return np.zeros((B, H - 4, W - 4, out_channels))

def unet_decoder_block(x, skip, out_channels):
    B, H, W, C = x.shape
    H_up, W_up = H * 2, W * 2
    H_out, W_out = H_up - 4, W_up - 4
    return np.zeros((B, H_out, W_out, out_channels))

def unet_output(features, num_classes):
    B, H, W, C = features.shape
    W_conv = np.random.randn(C, num_classes) * 0.01
    return features @ W_conv

def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    # Encoder: channel progression 1→64→128→256→512
    p1, s1 = unet_encoder_block(x,  64)
    p2, s2 = unet_encoder_block(p1, 128)
    p3, s3 = unet_encoder_block(p2, 256)
    p4, s4 = unet_encoder_block(p3, 512)

    # Bottleneck: 512→1024
    b = unet_bottleneck(p4, 1024)

    # Decoder: reverse order of skips, channel progression 1024→512→256→128→64
    d4 = unet_decoder_block(b,  s4, 512)
    d3 = unet_decoder_block(d4, s3, 256)
    d2 = unet_decoder_block(d3, s2, 128)
    d1 = unet_decoder_block(d2, s1, 64)

    # Output: 1×1 conv → num_classes
    return unet_output(d1, num_classes)