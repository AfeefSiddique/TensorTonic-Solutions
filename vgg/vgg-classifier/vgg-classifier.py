import numpy as np

def vgg_classifier(features: np.ndarray, num_classes: int = 1000) -> np.ndarray:
    batch   = features.shape[0]
    x       = features.reshape(batch, -1)   # (batch, 7*7*512 = 25088)
    in_dim  = x.shape[1]

    # FC1: in_dim → 4096
    W1 = np.random.randn(in_dim, 4096) * np.sqrt(2.0 / in_dim)
    b1 = np.zeros(4096)
    x  = np.maximum(0, x @ W1 + b1)        # ReLU

    # FC2: 4096 → 4096
    W2 = np.random.randn(4096, 4096) * np.sqrt(2.0 / 4096)
    b2 = np.zeros(4096)
    x  = np.maximum(0, x @ W2 + b2)        # ReLU

    # FC3: 4096 → num_classes (no activation)
    W3 = np.random.randn(4096, num_classes) * np.sqrt(2.0 / 4096)
    b3 = np.zeros(num_classes)
    return x @ W3 + b3                      # (batch, num_classes)