import numpy as np

def candidate_hidden(h_prev, x_t, r_t, W_h, b_h):
    concat = np.concatenate([r_t * h_prev, x_t], axis=-1)
    return np.tanh(concat @ W_h.T + b_h)