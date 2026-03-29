import numpy as np

eps = 1e-8

class GAN:
    def __init__(self, data_dim: int, noise_dim: int, hidden_dim: int = 256):
        self.data_dim  = data_dim
        self.noise_dim = noise_dim

        # Generator weights: noise → hidden → data
        self.G_W1 = np.random.randn(noise_dim,  hidden_dim) * 0.02
        self.G_b1 = np.zeros(hidden_dim)
        self.G_W2 = np.random.randn(hidden_dim, data_dim)   * 0.02
        self.G_b2 = np.zeros(data_dim)

        # Discriminator weights: data → hidden → 1
        self.D_W1 = np.random.randn(data_dim,   hidden_dim) * 0.02
        self.D_b1 = np.zeros(hidden_dim)
        self.D_W2 = np.random.randn(hidden_dim, 1)          * 0.02
        self.D_b2 = np.zeros(1)

    def _generator(self, z: np.ndarray) -> np.ndarray:
        h = np.maximum(0, z @ self.G_W1 + self.G_b1)   # ReLU
        return np.tanh(h @ self.G_W2 + self.G_b2)       # tanh → [-1,1]

    def _discriminator(self, x: np.ndarray) -> np.ndarray:
        h     = np.maximum(0, x @ self.D_W1 + self.D_b1)  # ReLU
        logit = h @ self.D_W2 + self.D_b2
        return 1 / (1 + np.exp(-logit))                    # sigmoid → [0,1]

    def generate(self, n_samples: int) -> np.ndarray:
        z = np.random.randn(n_samples, self.noise_dim)
        return self._generator(z)

    def discriminate(self, x: np.ndarray) -> np.ndarray:
        return self._discriminator(x)

    def train_step(self, real_data: np.ndarray) -> dict:
        batch_size = real_data.shape[0]

        # Step 1: Discriminator loss
        z_d        = np.random.randn(batch_size, self.noise_dim)
        fake_d     = self._generator(z_d)
        real_probs = self._discriminator(real_data)
        fake_probs = self._discriminator(fake_d)
        d_loss     = float(-np.mean(
            np.log(real_probs + eps) + np.log(1 - fake_probs + eps)
        ))

        # Step 2: Generator loss (non-saturating)
        z_g        = np.random.randn(batch_size, self.noise_dim)
        fake_g     = self._generator(z_g)
        fool_probs = self._discriminator(fake_g)
        g_loss     = float(-np.mean(np.log(fool_probs + eps)))

        return {"d_loss": d_loss, "g_loss": g_loss}