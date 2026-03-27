import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Position indices: (seq_length, 1)
    pos = np.arange(seq_length).reshape(-1, 1)
    
    # Dimension indices for the formula: 2i for i in range(d_model/2)
    # This creates: [0, 2, 4, ...] or up to d_model
    i = np.arange(0, d_model, 2)
    
    # Compute the angle rates: 1 / (10000^(2i/d_model))
    # Using exp(log) for numerical stability
    angle_rates = 1 / np.power(10000, (2 * i) / np.float32(d_model))
    
    # Initialize PE matrix
    pe = np.zeros((seq_length, d_model))
    
    # Even indices: sin(pos * angle_rates)
    pe[:, 0::2] = np.sin(pos * angle_rates)
    
    # Odd indices: cos(pos * angle_rates)
    # Need to handle if d_model is odd
    if d_model % 2 == 0:
        pe[:, 1::2] = np.cos(pos * angle_rates)
    else:
        pe[:, 1::2] = np.cos(pos * angle_rates[:-1])
    
    return pe