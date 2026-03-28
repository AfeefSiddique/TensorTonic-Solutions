import numpy as np

def unet_output(features: np.ndarray, num_classes: int) -> np.ndarray:
    B, H, W, C = features.shape
    W_conv = np.random.randn(C, num_classes) * 0.01
    return features @ W_conv    # (B, H, W, num_classes)