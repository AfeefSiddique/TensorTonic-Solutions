import numpy as np

def vae_encoder(x: np.ndarray, latent_dim: int) -> tuple:
    input_dim  = x.shape[1]
    hidden_dim = 256

    # Hidden layer
    W_h = np.random.randn(input_dim,  hidden_dim) * 0.02
    b_h = np.zeros(hidden_dim)
    h   = np.maximum(0, x @ W_h + b_h)   # ReLU

    # Mean and log-variance heads (no activation)
    W_mu  = np.random.randn(hidden_dim, latent_dim) * 0.02
    b_mu  = np.zeros(latent_dim)
    W_lv  = np.random.randn(hidden_dim, latent_dim) * 0.02
    b_lv  = np.zeros(latent_dim)

    mu      = h @ W_mu + b_mu   # (batch, latent_dim)
    log_var = h @ W_lv + b_lv   # (batch, latent_dim)

    return mu, log_var