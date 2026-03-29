import numpy as np

def reparameterize(mu: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    eps   = np.random.randn(*mu.shape)
    sigma = np.exp(0.5 * log_var)
    return mu + sigma * eps