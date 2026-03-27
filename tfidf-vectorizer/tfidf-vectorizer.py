import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    if not documents:
        return np.zeros((0, 0)), []

    # Tokenize
    tokenized = [doc.lower().split() for doc in documents]

    # Build sorted vocabulary
    vocab = sorted(set(t for tokens in tokenized for t in tokens))
    vocab_index = {t: i for i, t in enumerate(vocab)}
    N = len(documents)
    V = len(vocab)

    # TF matrix
    tf = np.zeros((N, V), dtype=float)
    for d, tokens in enumerate(tokenized):
        if not tokens:
            continue
        counts = Counter(tokens)
        for term, cnt in counts.items():
            tf[d, vocab_index[term]] = cnt / len(tokens)

    # IDF vector: log(N / df(t)), df from column sums of binary presence
    presence = (tf > 0).astype(float)          # (N, V)
    df = presence.sum(axis=0)                  # (V,)
    idf = np.where(df > 0, np.log(N / df), 0.0)

    tfidf_matrix = tf * idf

    return tfidf_matrix, vocab