import numpy as np

def decision_tree_split(X, y):
    X = np.array(X, dtype=float)
    y = np.array(y)
    N, D = X.shape

    def gini(labels):
        if len(labels) == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return 1.0 - np.sum(p ** 2)

    parent_gini = gini(y)
    best_gain = -np.inf
    best_feat = 0
    best_thresh = np.inf

    for f in range(D):
        vals = X[:, f]
        unique_vals = np.unique(vals)
        if len(unique_vals) < 2:
            continue
        thresholds = (unique_vals[:-1] + unique_vals[1:]) / 2.0

        for t in thresholds:
            left_mask  = vals <= t
            right_mask = ~left_mask
            nL, nR = left_mask.sum(), right_mask.sum()
            if nL == 0 or nR == 0:
                continue

            weighted = (nL * gini(y[left_mask]) + nR * gini(y[right_mask])) / N
            gain = parent_gini - weighted

            if (gain > best_gain or
               (gain == best_gain and f < best_feat) or
               (gain == best_gain and f == best_feat and t < best_thresh)):
                best_gain   = gain
                best_feat   = f
                best_thresh = t

    return [best_feat, best_thresh]