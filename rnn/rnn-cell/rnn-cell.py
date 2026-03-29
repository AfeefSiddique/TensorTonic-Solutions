import numpy as np

def rnn_cell(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(x_t @ W_xh.T + h_prev @ W_hh.T + b_h)