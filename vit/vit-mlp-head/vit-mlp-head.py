import numpy as np

def classification_head(encoder_output: np.ndarray, num_classes: int) -> np.ndarray:
    cls = encoder_output[:, 0, :]                          # (B, D)
    
    # LayerNorm
    mean = cls.mean(axis=-1, keepdims=True)
    var  = cls.var(axis=-1,  keepdims=True)
    cls  = (cls - mean) / np.sqrt(var + 1e-6)             # (B, D)

    # Linear projection
    D    = cls.shape[1]
    W    = np.random.randn(D, num_classes) * 0.02
    b    = np.zeros(num_classes)
    return cls @ W + b                                     # (B, num_classes)