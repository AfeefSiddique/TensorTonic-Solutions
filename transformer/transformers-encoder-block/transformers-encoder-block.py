import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x, gamma, beta, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta

def multi_head_attention(Q, K, V, W_q, W_k, W_v, W_o, num_heads):
    B, T, d_model = Q.shape
    d_k = d_model // num_heads

    # Project
    Q_ = Q @ W_q   # (B, T, d_model)
    K_ = K @ W_k
    V_ = V @ W_v

    # Split into heads → (B, num_heads, T, d_k)
    def split_heads(x):
        return x.reshape(B, T, num_heads, d_k).transpose(0, 2, 1, 3)

    Q_ = split_heads(Q_)
    K_ = split_heads(K_)
    V_ = split_heads(V_)

    # Scaled dot-product attention
    scale  = np.sqrt(d_k)
    scores = Q_ @ K_.transpose(0, 1, 3, 2) / scale   # (B, H, T, T)
    attn   = softmax(scores, axis=-1)
    out    = attn @ V_                                 # (B, H, T, d_k)

    # Merge heads → (B, T, d_model)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, d_model)
    return out @ W_o

def feed_forward(x, W1, b1, W2, b2):
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def encoder_block(x, W_q, W_k, W_v, W_o, W1, b1, W2, b2,
                  gamma1, beta1, gamma2, beta2, num_heads):
    # Sub-layer 1: MHA + residual + LN
    x = layer_norm(x + multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads),
                   gamma1, beta1)
    # Sub-layer 2: FFN + residual + LN
    x = layer_norm(x + feed_forward(x, W1, b1, W2, b2), gamma2, beta2)
    return x