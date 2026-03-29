import numpy as np

def bptt_single_step(dh_next, h_t, h_prev, x_t, W_hh):
    dtanh  = (1 - h_t ** 2) * dh_next   # tanh derivative
    dW_hh  = dtanh.T @ h_prev            # (hidden, hidden)
    dh_prev = dtanh @ W_hh               # (batch, hidden)
    return dh_prev, dW_hh