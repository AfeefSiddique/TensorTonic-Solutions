import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def reset_gate(h_prev, x_t, W_r, b_r):
    concat = np.concatenate([h_prev, x_t], axis=-1)
    return sigmoid(concat @ W_r.T + b_r)