import numpy as np

def rnn_step_backward(dh, cache):
    x_t, h_prev, h_t, W, U, b = cache
    x_t    = np.array(x_t,    dtype=float)
    h_prev = np.array(h_prev, dtype=float)
    h_t    = np.array(h_t,    dtype=float)
    W      = np.array(W,      dtype=float)
    U      = np.array(U,      dtype=float)
    dh     = np.array(dh,     dtype=float)

    # 1. Backprop through tanh: dz = dh * (1 - h_t²)
    dz = dh * (1.0 - h_t ** 2)   # (H,)

    # 2. Gradients w.r.t. parameters
    dW = np.outer(dz, x_t)        # (H, D)
    dU = np.outer(dz, h_prev)     # (H, H)
    db = dz                        # (H,)

    # 3. Gradients w.r.t. inputs
    dx_t   = W.T @ dz             # (D,)
    dh_prev = U.T @ dz            # (H,)

    return dx_t, dh_prev, dW, dU, db