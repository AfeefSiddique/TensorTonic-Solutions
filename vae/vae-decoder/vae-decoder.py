import numpy as np

def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    latent_dim = z.shape[1]
    hidden_dim = 256

    # Hidden layer
    W_h = np.random.randn(latent_dim, hidden_dim) * 0.02
    b_h = np.zeros(hidden_dim)
    h   = np.maximum(0, z @ W_h + b_h)      # ReLU

    # Output layer → sigmoid for [0, 1] range
    W_o = np.random.randn(hidden_dim, output_dim) * 0.02
    b_o = np.zeros(output_dim)
    return 1 / (1 + np.exp(-(h @ W_o + b_o)))  # sigmoid