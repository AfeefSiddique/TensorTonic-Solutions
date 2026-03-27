import numpy as np

def compute_monitoring_metrics(system_type, y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    def safe_div(a, b):
        return float(a / b) if b != 0 else 0.0

    if system_type == "classification":
        TP = ((y_pred == 1) & (y_true == 1)).sum()
        TN = ((y_pred == 0) & (y_true == 0)).sum()
        FP = ((y_pred == 1) & (y_true == 0)).sum()
        FN = ((y_pred == 0) & (y_true == 1)).sum()

        accuracy  = safe_div(TP + TN, len(y_true))
        precision = safe_div(TP, TP + FP)
        recall    = safe_div(TP, TP + FN)
        f1        = safe_div(2 * precision * recall, precision + recall)

        metrics = {"accuracy": accuracy, "f1": f1,
                   "precision": precision, "recall": recall}

    elif system_type == "regression":
        errors = y_true - y_pred
        mae    = float(np.mean(np.abs(errors)))
        rmse   = float(np.sqrt(np.mean(errors ** 2)))
        metrics = {"mae": mae, "rmse": rmse}

    elif system_type == "ranking":
        top3_idx      = np.argsort(y_pred)[::-1][:3]
        top3_relevant = y_true[top3_idx].sum()
        total_relevant = y_true.sum()

        p_at_3 = safe_div(top3_relevant, 3)
        r_at_3 = safe_div(top3_relevant, total_relevant)
        metrics = {"precision_at_3": p_at_3, "recall_at_3": r_at_3}

    return sorted(metrics.items())