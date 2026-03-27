import numpy as np

def roc_curve(y_true, y_score):
    y_true  = np.array(y_true)
    y_score = np.array(y_score)

    P = y_true.sum()
    N = len(y_true) - P

    # Sort by descending score
    desc = np.argsort(y_score)[::-1]
    y_true_sorted  = y_true[desc]
    y_score_sorted = y_score[desc]

    # Cumulative TP and FP as threshold sweeps down
    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)

    # Group ties: keep last index of each unique score block
    # (all samples with same score get classified together)
    unique_mask = np.concatenate([
        np.diff(y_score_sorted) != 0,
        [True]   # always keep last point
    ])

    tp_pts = tp_cum[unique_mask]
    fp_pts = fp_cum[unique_mask]
    thresh = y_score_sorted[unique_mask]

    tpr = tp_pts / P
    fpr = fp_pts / N

    # Prepend (0, 0, inf) starting point
    fpr        = np.concatenate([[0.0], fpr])
    tpr        = np.concatenate([[0.0], tpr])
    thresholds = np.concatenate([[np.inf], thresh])

    return fpr.tolist(), tpr.tolist(), thresholds.tolist()