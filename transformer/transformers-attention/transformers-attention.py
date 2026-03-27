import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Get dimensions
    # Q: (batch, seq_len_q, d_k)
    # K: (batch, seq_len_k, d_k)
    # V: (batch, seq_len_k, d_v)
    
    d_k = Q.size(-1)  # key dimension
    
    # Step 1: Compute attention scores (Q @ K^T)
    # K.transpose(-2, -1) swaps the last two dimensions: (batch, d_k, seq_len_k)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len_q, seq_len_k)
    
    # Step 2: Scale by sqrt(d_k) for numerical stability
    # This prevents dot products from becoming too large
    scaled_scores = scores / math.sqrt(d_k)
    
    # Step 3: Apply softmax to get attention weights
    # Softmax is applied over the last dimension (keys)
    attention_weights = F.softmax(scaled_scores, dim=-1)  # (batch, seq_len_q, seq_len_k)
    
    # Step 4: Multiply attention weights by values
    output = torch.matmul(attention_weights, V)  # (batch, seq_len_q, d_v)
    
    return output