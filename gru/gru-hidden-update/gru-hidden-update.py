import numpy as np

def hidden_update(h_prev, h_tilde, z_t):
    return z_t * h_prev + (1 - z_t) * h_tilde