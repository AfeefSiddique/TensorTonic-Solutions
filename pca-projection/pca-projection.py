import numpy as np

def pca_projection(X, k):
    X = np.array(X, dtype=float)
    
    # 1. Center
    Xc = X - X.mean(axis=0)
    
    # 2. Sample covariance (divide by n-1)
    C = np.cov(Xc, rowvar=False)
    
    # 3. Eigen-decomposition, sort by descending eigenvalue
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]
    W = eigenvectors[:, order[:k]]   # (d, k)
    
    # 4. Project
    return (Xc @ W).tolist()