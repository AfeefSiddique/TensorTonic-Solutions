import numpy as np

def gaussian_naive_bayes(X_train, y_train, X_test):
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train)
    X_test  = np.array(X_test, dtype=float)
    eps     = 1e-9
    n       = len(y_train)
    classes = np.unique(y_train)

    log_priors = []
    means      = []
    variances  = []

    for c in classes:
        mask  = y_train == c
        Xc    = X_train[mask]
        nc    = mask.sum()
        log_priors.append(np.log(nc / n))
        means.append(Xc.mean(axis=0))
        variances.append(Xc.var(axis=0) + eps)   # population variance + eps

    log_priors = np.array(log_priors)   # (C,)
    means      = np.array(means)        # (C, D)
    variances  = np.array(variances)    # (C, D)

    # Log-likelihood: (N_test, C)
    # Broadcast X_test (N,1,D) against class stats (C,D)
    X_test_exp = X_test[:, np.newaxis, :]          # (N, 1, D)
    log_like   = (
        -0.5 * np.log(2 * np.pi * variances)       # (C, D)
        - 0.5 * (X_test_exp - means) ** 2 / variances
    ).sum(axis=2)                                  # (N, C)

    log_post = log_like + log_priors               # (N, C)
    preds    = classes[np.argmax(log_post, axis=1)]

    return preds.tolist()