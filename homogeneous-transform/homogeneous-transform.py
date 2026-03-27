import numpy as np

def apply_homogeneous_transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.array(points)
    single = points.ndim == 1
    if single:
        points = points.reshape(1, 3)
    
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack([points, ones])          # (N, 4)
    transformed = (T @ points_h.T).T              # (N, 4)
    result = transformed[:, :3]                   # (N, 3)
    
    return result.reshape(3) if single else result