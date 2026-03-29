import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def layer_norm(x, eps=1e-6):
    mean = x.mean(axis=-1, keepdims=True)
    var  = x.var(axis=-1,  keepdims=True)
    return (x - mean) / np.sqrt(var + eps)

def msa(x, embed_dim, num_heads):
    B, N, D = x.shape
    head_dim = D // num_heads
    # Single projection for Q, K, V
    W_qkv = np.random.randn(D, 3 * D) * 0.02
    W_o   = np.random.randn(D, D)     * 0.02
    qkv   = x @ W_qkv                          # (B, N, 3D)
    Q, K, V = np.split(qkv, 3, axis=-1)        # each (B, N, D)
    # Reshape for multi-head: (B, heads, N, head_dim)
    Q = Q.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)
    K = K.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)
    V = V.reshape(B, N, num_heads, head_dim).transpose(0,2,1,3)
    scale  = np.sqrt(head_dim)
    scores = Q @ K.transpose(0,1,3,2) / scale  # (B, heads, N, N)
    e      = np.exp(scores - scores.max(axis=-1, keepdims=True))
    attn   = e / e.sum(axis=-1, keepdims=True)
    out    = attn @ V                           # (B, heads, N, head_dim)
    out    = out.transpose(0,2,1,3).reshape(B, N, D)
    return out @ W_o

def mlp(x, embed_dim, mlp_ratio):
    hidden_dim = int(embed_dim * mlp_ratio)
    W1 = np.random.randn(embed_dim, hidden_dim) * 0.02
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, embed_dim) * 0.02
    b2 = np.zeros(embed_dim)
    return gelu(x @ W1 + b1) @ W2 + b2

def vit_encoder_block(x: np.ndarray, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0) -> np.ndarray:
    # x' = x + MSA(LN(x))
    x = x + msa(layer_norm(x), embed_dim, num_heads)
    # x'' = x' + MLP(LN(x'))
    x = x + mlp(layer_norm(x), embed_dim, mlp_ratio)
    return x