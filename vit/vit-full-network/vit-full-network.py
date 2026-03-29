import numpy as np

def gelu(x):
    x = np.asarray(x, np.float32)
    return np.float32(0.5) * x * (1 + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * x**3)))

def layer_norm(x):
    m = x.mean(axis=-1, keepdims=True)
    v = x.var( axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + np.float32(1e-6))

def softmax_last(x):
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x

class VisionTransformer:
    def __init__(self, image_size=224, patch_size=16, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0):
        D     = 64
        H     = 2
        L     = 2
        hd    = D // H
        mlp_d = D * 4
        pd    = patch_size * patch_size * 3

        self.ps    = patch_size
        self.n     = (image_size // patch_size) ** 2
        self.D     = D
        self.L     = L
        self.H     = H
        self.hd    = hd
        self.scale = np.float32(hd ** -0.5)
        self.num_classes = num_classes

        self.Wp      = np.zeros((pd, D),         np.float32)
        self.cls_tok = np.zeros((1, 1, D),        np.float32)
        self.pos     = np.zeros((1, self.n+1, D), np.float32)

        self.qkv = [np.zeros((D, 3*D),   np.float32) for _ in range(L)]
        self.wo  = [np.zeros((D, D),     np.float32) for _ in range(L)]
        self.w1  = [np.zeros((D, mlp_d), np.float32) for _ in range(L)]
        self.b1  = [np.zeros(mlp_d,      np.float32) for _ in range(L)]
        self.w2  = [np.zeros((mlp_d, D), np.float32) for _ in range(L)]
        self.b2  = [np.zeros(D,          np.float32) for _ in range(L)]

        self.wh = np.zeros((D, num_classes), np.float32)
        self.bh = np.zeros(num_classes,      np.float32)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x is None:
            return None
        B, H, W, C = x.shape
        P    = self.ps
        Hp   = H // P
        Wp   = W // P
        N    = Hp * Wp
        S    = N + 1
        D    = self.D

        x = x.astype(np.float32)

        # Patch embed
        x = x.reshape(B, Hp, P, Wp, P, C).transpose(0,1,3,2,4,5).reshape(B, N, -1)
        x = x @ self.Wp                                # (B, N, D)

        # CLS token + position
        cls = np.zeros((B, 1, D), np.float32)
        x   = np.concatenate([cls, x], axis=1)         # (B, S, D)
        x   = x + self.pos[:, :S, :]

        # Transformer encoder blocks
        for i in range(self.L):
            # MSA with pre-LN
            xn  = layer_norm(x)
            qkv = (xn.reshape(B*S, D) @ self.qkv[i]).reshape(B, S, 3, self.H, self.hd)
            qkv = qkv.transpose(2, 0, 3, 1, 4)
            Q, K, V = qkv[0], qkv[1], qkv[2]
            A   = softmax_last(Q @ K.transpose(0,1,3,2) * self.scale)
            out = (A @ V).transpose(0,2,1,3).reshape(B, S, D) @ self.wo[i]
            x   = x + out

            # MLP with pre-LN
            xn = layer_norm(x)
            x  = x + gelu(xn @ self.w1[i] + self.b1[i]) @ self.w2[i] + self.b2[i]

        # Classification head
        return layer_norm(x)[:, 0] @ self.wh + self.bh