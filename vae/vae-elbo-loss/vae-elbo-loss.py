import numpy as np

def vae_loss(x: np.ndarray, x_recon: np.ndarray,
             mu: np.ndarray, log_var: np.ndarray) -> dict:

    # Reconstruction: MSE sum over features, mean over batch
    recon = float(np.mean(np.sum((x - x_recon) ** 2, axis=1)))

    # KL divergence: -½ Σ(1 + log σ² - μ² - exp(log σ²))
    kl = float(np.mean(-0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=1)))

    return {"total": recon + kl, "recon": recon, "kl": kl}