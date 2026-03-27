import numpy as np

def impute_missing(X, strategy='mean'):
    X = np.array(X, dtype=float)
    single = X.ndim == 1
    if single:
        X = X.reshape(-1, 1)
    
    result = X.copy()
    
    for col in range(X.shape[1]):
        column = X[:, col]
        mask = np.isnan(column)
        if not mask.any():
            continue
        observed = column[~mask]
        if len(observed) == 0:
            fill = 0.0
        elif strategy == 'mean':
            fill = np.mean(observed)
        else:
            fill = np.median(observed)
        result[mask, col] = fill
    
    return result.reshape(-1) if single else result