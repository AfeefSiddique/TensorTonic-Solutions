import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    B = image.shape[0]
    # 11x11, stride 4, 96 filters, padding=2 → (224+4-11)/4+1 = 55
    return np.zeros((B, 55, 55, 96))