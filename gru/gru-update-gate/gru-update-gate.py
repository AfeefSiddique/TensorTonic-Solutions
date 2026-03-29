import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def update_gate(h_prev, x_t, W_z, b_z):
    concat = np.concatenate([h_prev, x_t], axis=-1)
    return sigmoid(concat @ W_z.T + b_z)