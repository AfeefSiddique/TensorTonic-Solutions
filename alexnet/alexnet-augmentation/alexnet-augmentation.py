import numpy as np

def random_crop(image: np.ndarray, crop_size: int = 224) -> np.ndarray:
    H, W, C = image.shape
    top  = np.random.randint(0, H - crop_size + 1)
    left = np.random.randint(0, W - crop_size + 1)
    return image[top:top+crop_size, left:left+crop_size, :]

def random_horizontal_flip(image: np.ndarray, p: float = 0.5) -> np.ndarray:
    if np.random.random() < p:
        return image[:, ::-1, :]
    return image