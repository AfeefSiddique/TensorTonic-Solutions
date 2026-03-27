import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Get dimensions
    batch_size, seq_len_q, d_model = Q.shape
    seq_len_k = K.shape[1]
    
    # d_k per head
    d_k = d_model // num_heads
    
    # Step 1: Linear projections
    # Q, K, V projections: (batch, seq_len, d_model)
    Q_proj = Q @ W_q  # (batch, seq_len_q, d_model)
    K_proj = K @ W_k  # (batch, seq_len_k, d_model)
    V_proj = V @ W_v  # (batch, seq_len_k, d_model)
    
    # Step 2: Reshape to separate heads
    # From (batch, seq_len, d_model) to (batch, num_heads, seq_len, d_k)
    Q_heads = Q_proj.reshape(batch_size, seq_len_q, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len_k, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # Step 3: Scaled dot-product attention for each head
    # Q_heads: (batch, num_heads, seq_len_q, d_k)
    # K_heads: (batch, num_heads, seq_len_k, d_k)
    # Transpose K for matmul: (batch, num_heads, d_k, seq_len_k)
    scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)  # (batch, num_heads, seq_len_q, seq_len_k)
    
    # Scale by sqrt(d_k)
    scores = scores / np.sqrt(d_k)
    
    # Apply softmax
    attn_weights = softmax(scores, axis=-1)  # (batch, num_heads, seq_len_q, seq_len_k)
    
    # Apply attention to values
    # V_heads: (batch, num_heads, seq_len_k, d_k)
    attn_output = attn_weights @ V_heads  # (batch, num_heads, seq_len_q, d_k)
    
    # Step 4: Concatenate heads
    # Transpose back: (batch, seq_len_q, num_heads, d_k)
    attn_output = attn_output.transpose(0, 2, 1, 3)
    # Reshape to (batch, seq_len_q, d_model)
    concat = attn_output.reshape(batch_size, seq_len_q, d_model)
    
    # Final linear projection
    output = concat @ W_o  # (batch, seq_len_q, d_model)
    
    return output