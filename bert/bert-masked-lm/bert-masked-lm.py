import numpy as np
from typing import Tuple

def apply_mlm_mask(
    token_ids: np.ndarray,
    vocab_size: int,
    mask_token_id: int = 103,
    mask_prob: float = 0.15,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    SPECIAL = {101, 102, 0}  # [CLS], [SEP], [PAD]

    token_ids  = np.array(token_ids)
    flat_ids   = token_ids.flatten()
    masked_ids = flat_ids.copy()
    labels     = np.full_like(flat_ids, -100)
    mask_positions = []

    candidates = [i for i in range(len(flat_ids)) if flat_ids[i].item() not in SPECIAL]

    num_to_mask = max(1, int(round(len(candidates) * mask_prob)))
    num_to_mask = min(num_to_mask, len(candidates))
    selected    = np.random.choice(candidates, size=num_to_mask, replace=False)

    for pos in selected:
        labels[pos]     = flat_ids[pos]
        mask_positions.append(int(pos))

        r = np.random.random()
        if r < 0.80:
            masked_ids[pos] = mask_token_id
        elif r < 0.90:
            masked_ids[pos] = np.random.randint(0, vocab_size)
        # else keep original

    # Reshape back to original shape
    masked_ids = masked_ids.reshape(token_ids.shape)
    labels     = labels.reshape(token_ids.shape)

    return masked_ids, labels, np.array(sorted(mask_positions))


class MLMHead:
    def __init__(self, hidden_size: int, vocab_size: int):
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.W = np.random.randn(hidden_size, vocab_size) * 0.02
        self.b = np.zeros(vocab_size)

    def forward(self, hidden_states: np.ndarray) -> np.ndarray:
        logits = hidden_states @ self.W + self.b
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)