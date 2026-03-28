import numpy as np

eps = 1e-8

def discriminator_loss(real_probs: np.ndarray, fake_probs: np.ndarray) -> float:
    # LD = -E[log D(x) + log(1 - D(G(z)))]
    return float(-np.mean(np.log(real_probs + eps) + np.log(1 - fake_probs + eps)))

def generator_loss(fake_probs: np.ndarray) -> float:
    # LG = -E[log D(G(z))]  (non-saturating)
    return float(-np.mean(np.log(fake_probs + eps)))