import numpy as np

def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    input_dim = z.shape[1]
    hidden_dim = 256

    # Layer 1: noise → hidden
    W1 = np.random.randn(input_dim, hidden_dim) * 0.02
    b1 = np.zeros(hidden_dim)
    h  = np.maximum(0, z @ W1 + b1)   # ReLU

    # Layer 2: hidden → output
    W2 = np.random.randn(hidden_dim, output_dim) * 0.02
    b2 = np.zeros(output_dim)
    return np.tanh(h @ W2 + b2)       # Tanh → [-1, 1]