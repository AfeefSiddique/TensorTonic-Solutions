import numpy as np

def rnn_forward(X, h_0, W_xh, W_hh, b_h):
    h = h_0
    hidden_states = []

    for t in range(X.shape[1]):
        h = np.tanh(X[:, t, :] @ W_xh.T + h @ W_hh.T + b_h)
        hidden_states.append(h)

    h_all = np.stack(hidden_states, axis=1)   # (batch, seq_len, hidden)
    return h_all, h