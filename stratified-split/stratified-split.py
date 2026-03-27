import numpy as np

def stratified_split(X, y, test_size=0.2, rng=None):
    X = np.array(X)
    y = np.array(y)

    if rng is None:
        rng = np.random.default_rng()

    train_idx = []
    test_idx  = []

    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)

        n_test = round(len(idx) * test_size)
        n_test = min(n_test, len(idx) - 1)
        n_test = max(n_test, 0)

        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.sort(np.array(train_idx))
    test_idx  = np.sort(np.array(test_idx))

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]