import numpy as np
from typing import List, Optional


class MockBertEncoder:
    def __init__(self, hidden_size: int = 768, num_layers: int = 12):
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.layers       = [np.random.randn(hidden_size, hidden_size) * 0.01 for _ in range(num_layers)]
        self.layer_frozen = [False] * num_layers

    def freeze_layers(self, layer_indices: List[int]):
        for i in layer_indices:
            self.layer_frozen[i] = True

    def unfreeze_all(self):
        self.layer_frozen = [False] * self.num_layers

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        x = embeddings
        for i, layer in enumerate(self.layers):
            x = x @ layer + x   # frozen or not, forward pass is identical
        return x


class BertForSequenceClassification:
    def __init__(self, hidden_size: int, num_labels: int, freeze_bert: bool = False):
        self.encoder    = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02
        self.freeze_bert = freeze_bert

        if freeze_bert:
            self.encoder.freeze_layers(list(range(12)))

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        # (batch, seq_len, hidden) → encoder → extract [CLS] → classify
        hidden  = self.encoder.forward(embeddings)   # (batch, seq_len, hidden)
        cls_out = hidden[:, 0, :]                    # (batch, hidden)
        return cls_out @ self.classifier             # (batch, num_labels)


class BertForTokenClassification:
    def __init__(self, hidden_size: int, num_labels: int):
        self.encoder    = MockBertEncoder(hidden_size)
        self.classifier = np.random.randn(hidden_size, num_labels) * 0.02

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        # (batch, seq_len, hidden) → encoder → classify each token
        hidden = self.encoder.forward(embeddings)    # (batch, seq_len, hidden)
        return hidden @ self.classifier              # (batch, seq_len, num_labels)