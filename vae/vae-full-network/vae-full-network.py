import numpy as np

class VAE:
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256):
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Encoder weights: input → hidden → (mu, log_var)
        self.E_W1   = np.random.randn(input_dim,  hidden_dim)  * 0.02
        self.E_b1   = np.zeros(hidden_dim)
        self.W_mu   = np.random.randn(hidden_dim, latent_dim)  * 0.02
        self.b_mu   = np.zeros(latent_dim)
        self.W_lv   = np.random.randn(hidden_dim, latent_dim)  * 0.02
        self.b_lv   = np.zeros(latent_dim)

        # Decoder weights: latent → hidden → output
        self.D_W1   = np.random.randn(latent_dim, hidden_dim)  * 0.02
        self.D_b1   = np.zeros(hidden_dim)
        self.D_W2   = np.random.randn(hidden_dim, input_dim)   * 0.02
        self.D_b2   = np.zeros(input_dim)

    def _encode(self, x: np.ndarray):
        h       = np.maximum(0, x @ self.E_W1 + self.E_b1)
        mu      = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_lv + self.b_lv
        return mu, log_var

    def _reparameterize(self, mu, log_var):
        eps = np.random.randn(*mu.shape)
        return mu + np.exp(0.5 * log_var) * eps

    def _decode(self, z: np.ndarray):
        h = np.maximum(0, z @ self.D_W1 + self.D_b1)
        return 1 / (1 + np.exp(-(h @ self.D_W2 + self.D_b2)))  # sigmoid

    def forward(self, x: np.ndarray) -> tuple:
        mu, log_var = self._encode(x)
        z           = self._reparameterize(mu, log_var)
        x_recon     = self._decode(z)
        return x_recon, mu, log_var

    def generate(self, n_samples: int) -> np.ndarray:
        z = np.random.randn(n_samples, self.latent_dim)
        return self._decode(z)