import numpy as np

class BertEmbeddings:
    def __init__(self, vocab_size: int, hidden_size: int, max_position: int, num_segments: int = 2):
        self.token_embeddings    = np.random.randn(vocab_size,   hidden_size) * 0.02
        self.position_embeddings = np.random.randn(max_position, hidden_size) * 0.02
        self.segment_embeddings  = np.random.randn(num_segments, hidden_size) * 0.02

    def forward(self, token_ids: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
        seq_len   = token_ids.shape[1]
        positions = np.arange(seq_len)

        tok_emb = self.token_embeddings[token_ids]     # (batch, seq_len, hidden)
        pos_emb = self.position_embeddings[positions]  # (seq_len, hidden) → broadcasts
        seg_emb = self.segment_embeddings[segment_ids] # (batch, seq_len, hidden)

        return tok_emb + pos_emb + seg_emb