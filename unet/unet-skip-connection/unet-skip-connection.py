import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    _, H_enc, W_enc, _ = encoder_features.shape
    _, H_dec, W_dec, _ = decoder_features.shape

    # Center crop encoder to match decoder spatial size
    crop_h = (H_enc - H_dec) // 2
    crop_w = (W_enc - W_dec) // 2
    cropped = encoder_features[:, crop_h:crop_h+H_dec, crop_w:crop_w+W_dec, :]

    return np.concatenate([cropped, decoder_features], axis=-1)