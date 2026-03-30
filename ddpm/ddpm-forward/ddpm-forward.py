import numpy as np

def get_alpha_bar(betas: np.ndarray) -> np.ndarray:
    return np.cumprod(1 - betas)

def forward_diffusion(x_0: np.ndarray, t: int, betas: np.ndarray) -> tuple:
    alpha_bar = get_alpha_bar(betas)
    a_bar     = alpha_bar[t - 1]          # t is 1-indexed

    eps = np.random.randn(*x_0.shape)
    x_t = np.sqrt(a_bar) * x_0 + np.sqrt(1 - a_bar) * eps

    return x_t, eps