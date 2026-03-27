import numpy as np

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float).flatten()
    grad = x.copy()
    for dF in reversed(gradients_F):
        dF = np.array(dF, dtype=float)
        if dF.ndim == 2:
            grad = dF.T @ grad
        else:
            grad = dF * grad
    return grad

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float).flatten()
    grad = x.copy()
    for dF in reversed(gradients_F):
        dF = np.array(dF, dtype=float)
        if dF.ndim == 2:
            I = np.eye(dF.shape[0])
            grad = (I + dF).T @ grad
        else:
            grad = (1 + dF) * grad
    return grad