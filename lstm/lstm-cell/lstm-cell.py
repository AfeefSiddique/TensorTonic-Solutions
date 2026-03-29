import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def lstm_cell(x_t, h_prev, C_prev,
              W_f, W_i, W_c, W_o,
              b_f, b_i, b_c, b_o):
    concat  = np.concatenate([h_prev, x_t], axis=-1)  # (N, H+D)

    f_t     = sigmoid(concat @ W_f.T + b_f)            # forget gate
    i_t     = sigmoid(concat @ W_i.T + b_i)            # input gate
    c_tilde = np.tanh(concat @ W_c.T + b_c)            # candidate
    o_t     = sigmoid(concat @ W_o.T + b_o)            # output gate

    C_t     = f_t * C_prev + i_t * c_tilde             # cell state
    h_t     = o_t * np.tanh(C_t)                       # hidden state

    return h_t, C_t