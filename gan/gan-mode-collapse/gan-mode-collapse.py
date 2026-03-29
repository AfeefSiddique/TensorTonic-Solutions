import numpy as np

def detect_mode_collapse(generated_samples: np.ndarray, threshold: float = 0.1) -> dict:
    diversity_score = float(np.std(generated_samples, axis=0).mean())
    return {
        "diversity_score": diversity_score,
        "is_collapsed":    diversity_score < threshold
    }