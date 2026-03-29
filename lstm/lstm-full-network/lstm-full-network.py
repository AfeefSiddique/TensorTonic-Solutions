import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class LSTM:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.W_f = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_i = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_c = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.W_o = np.random.randn(hidden_dim, hidden_dim + input_dim) * scale
        self.b_f = np.zeros(hidden_dim)
        self.b_i = np.zeros(hidden_dim)
        self.b_c = np.zeros(hidden_dim)
        self.b_o = np.zeros(hidden_dim)
        self.W_y = np.random.randn(output_dim, hidden_dim) * np.sqrt(2.0 / (hidden_dim + output_dim))
        self.b_y = np.zeros(output_dim)

    def forward(self, X: np.ndarray) -> tuple:
        # X: (batch, seq_len, input_dim)
        batch, seq_len, _ = X.shape
        h = np.zeros((batch, self.hidden_dim))
        C = np.zeros((batch, self.hidden_dim))
        outputs = []

        for t in range(seq_len):
            x_t    = X[:, t, :]
            concat = np.concatenate([h, x_t], axis=-1)   # (batch, H+D)

            f_t     = sigmoid(concat @ self.W_f.T + self.b_f)
            i_t     = sigmoid(concat @ self.W_i.T + self.b_i)
            c_tilde = np.tanh(concat @ self.W_c.T + self.b_c)
            o_t     = sigmoid(concat @ self.W_o.T + self.b_o)

            C = f_t * C + i_t * c_tilde
            h = o_t * np.tanh(C)

            y_t = h @ self.W_y.T + self.b_y              # (batch, output_dim)
            outputs.append(y_t)

        Y = np.stack(outputs, axis=1)                     # (batch, seq_len, output_dim)
        return Y, h, C