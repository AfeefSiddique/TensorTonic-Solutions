import numpy as np

def prepend_class_token(patches: np.ndarray, embed_dim: int) -> np.ndarray:
    B = patches.shape[0]
    cls_token = np.zeros((1, 1, embed_dim))                    # learnable [CLS]
    cls_batch = np.tile(cls_token, (B, 1, 1))                  # (B, 1, D)
    return np.concatenate([cls_batch, patches], axis=1)        # (B, N+1, D)