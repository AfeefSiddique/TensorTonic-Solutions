import numpy as np

def discriminator(x: np.ndarray) -> np.ndarray:
    input_dim  = x.shape[1]
    hidden_dim = 256

    # Layer 1: input → hidden
    W1 = np.random.randn(input_dim, hidden_dim) * 0.02
    b1 = np.zeros(hidden_dim)
    h  = np.maximum(0, x @ W1 + b1)   # ReLU

    # Layer 2: hidden → 1 logit
    W2 = np.random.randn(hidden_dim, 1) * 0.02
    b2 = np.zeros(1)
    logit = h @ W2 + b2               # (batch, 1)

    return 1 / (1 + np.exp(-logit))   # Sigmoid → [0, 1]