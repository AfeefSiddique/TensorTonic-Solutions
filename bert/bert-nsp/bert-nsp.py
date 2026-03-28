import numpy as np
from typing import List, Tuple
import random

def create_nsp_examples(
    documents: List[List[str]],
    num_examples: int,
    seed: int = None
) -> List[Tuple[str, str, int]]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    examples = []
    num_positive = num_examples // 2
    num_negative = num_examples - num_positive

    # Positive examples: consecutive sentences from same document
    pos_count = 0
    attempts  = 0
    while pos_count < num_positive and attempts < num_examples * 10:
        attempts += 1
        doc = random.choice(documents)
        if len(doc) < 2:
            continue
        idx = random.randint(0, len(doc) - 2)
        examples.append((doc[idx], doc[idx + 1], 1))
        pos_count += 1

    # Negative examples: sentence A from one doc, sentence B from different doc
    neg_count = 0
    attempts  = 0
    while neg_count < num_negative and attempts < num_examples * 10:
        attempts += 1
        if len(documents) < 2:
            # Only one document — pick non-consecutive sentences
            doc = documents[0]
            if len(doc) < 3:
                break
            idx_a = random.randint(0, len(doc) - 1)
            idx_b = random.randint(0, len(doc) - 1)
            if abs(idx_a - idx_b) > 1:
                examples.append((doc[idx_a], doc[idx_b], 0))
                neg_count += 1
        else:
            doc_a, doc_b = random.sample(documents, 2)
            if not doc_a or not doc_b:
                continue
            sent_a = random.choice(doc_a)
            sent_b = random.choice(doc_b)
            examples.append((sent_a, sent_b, 0))
            neg_count += 1

    random.shuffle(examples)
    return examples


class NSPHead:
    def __init__(self, hidden_size: int):
        self.W = np.random.randn(hidden_size, 2) * 0.02
        self.b = np.zeros(2)

    def forward(self, cls_hidden: np.ndarray) -> np.ndarray:
        # cls_hidden: (hidden_size,) or (batch, hidden_size)
        logits = cls_hidden @ self.W + self.b   # (..., 2)
        return softmax(logits)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)