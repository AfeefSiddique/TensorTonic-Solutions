import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def gru_cell(x_t, h_prev, W_r, W_z, W_h, b_r, b_z, b_h):
    concat  = np.concatenate([h_prev, x_t], axis=-1)

    r_t     = sigmoid(concat @ W_r.T + b_r)
    z_t     = sigmoid(concat @ W_z.T + b_z)

    concat_r = np.concatenate([r_t * h_prev, x_t], axis=-1)
    h_tilde  = np.tanh(concat_r @ W_h.T + b_h)

    return z_t * h_prev + (1 - z_t) * h_tilde