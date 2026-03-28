import numpy as np

def tanh(x):
    return np.tanh(x)

class BertPooler:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size
        self.W = np.random.randn(hidden_size, hidden_size) * 0.02
        self.b = np.zeros(hidden_size)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        # Extract [CLS] token (position 0) → (batch, hidden_size)
        cls_hidden = hidden_states[:, 0, :]
        return tanh(cls_hidden @ self.W + self.b)


class SequenceClassifier:
    def __init__(self, hidden_size: int, num_classes: int, dropout_prob: float = 0.1):
        self.pooler       = BertPooler(hidden_size)
        self.dropout_prob = dropout_prob
        self.classifier   = np.random.randn(hidden_size, num_classes) * 0.02

    def forward(self, hidden_states: np.ndarray, training: bool = True) -> np.ndarray:
        pooled = self.pooler.forward(hidden_states)   # (batch, hidden_size)

        # Dropout during training
        if training and self.dropout_prob > 0:
            mask   = np.random.binomial(1, 1 - self.dropout_prob, pooled.shape)
            pooled = pooled * mask / (1 - self.dropout_prob)

        return pooled @ self.classifier                # (batch, num_classes)