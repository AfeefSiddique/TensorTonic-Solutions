import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    X = np.array(X)
    y = np.array(y)
    N = len(X)
    
    if rng is None:
        rng = np.random.default_rng()
    
    perm = rng.permutation(N)
    X_shuf = X[perm]
    y_shuf = y[perm]
    
    num_full = N // batch_size
    num_batches = num_full if drop_last else (N + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        yield X_shuf[start:end], y_shuf[start:end]