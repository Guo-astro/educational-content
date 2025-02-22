import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans  # (if needed for a more advanced method)


# ---------------------------
# Helper: Full Transformer Attention
# ---------------------------
def compute_full_attention(Q, K, V):
    """
    Standard transformer attention.
    Q, K, V: arrays of shape (n, d)
    """
    d = Q.shape[1]
    scores = np.dot(Q, K.T) / np.sqrt(d)
    A = np.exp(scores)
    D = np.sum(A, axis=1, keepdims=True)
    T = A @ V / D
    return T


# ---------------------------
# 1. Uniform Subsampling
# ---------------------------
def uniform_subsampling(X, n_out):
    """
    Randomly sample n_out rows from X without replacement.
    Returns the thinned set and indices.
    """
    indices = np.random.choice(X.shape[0], n_out, replace=False)
    return X[indices], indices


# ---------------------------
# 2. Kernel Halving (KH(δ))
# ---------------------------
def kh_thinning(X, delta=0.1):
    """
    A simplified version of KH(δ).
    Process consecutive pairs and choose the element with larger norm.
    Returns the thinned set and indices.
    """
    n, d = X.shape
    assert n % 2 == 0, "n must be even."
    selected = []
    indices = []
    for i in range(n // 2):
        idx1, idx2 = 2 * i, 2 * i + 1
        x1, x2 = X[idx1], X[idx2]
        # Choose the one with larger norm (as a toy rule)
        if np.linalg.norm(x1) >= np.linalg.norm(x2):
            selected.append(x1)
            indices.append(idx1)
        else:
            selected.append(x2)
            indices.append(idx2)
    return np.array(selected), np.array(indices)


# ---------------------------
# 3. KH-COMPRESS(δ)
# ---------------------------
def kh_compress(X, delta=0.1, target=None):
    """
    Recursively applies KH-thinning until the number of points equals target.
    If target is None, set target = n/2.
    """
    n = X.shape[0]
    if target is None:
        target = n // 2
    X_current = X.copy()
    while X_current.shape[0] > target:
        X_current, _ = kh_thinning(X_current, delta)
    return X_current, None  # We do not track global indices in this toy version


# ---------------------------
# 4. Gram-Schmidt Thinning (GS-THIN)
# ---------------------------
def gs_thin(X, delta=0.1):
    """
    A simplified GS-THIN implementation.
    Process consecutive pairs and select the element that is more "orthogonal"
    to an accumulated basis.
    Returns the thinned set and indices.
    """
    n, d = X.shape
    assert n % 2 == 0, "n must be even."
    selected = []
    indices = []
    B = []  # Accumulated basis
    for i in range(n // 2):
        idx1, idx2 = 2 * i, 2 * i + 1
        x1, x2 = X[idx1], X[idx2]

        def orth_dist(x, B):
            if len(B) == 0:
                return np.linalg.norm(x)
            P = np.column_stack(B)
            proj = P @ np.linalg.pinv(P) @ x
            return np.linalg.norm(x - proj)

        d1 = orth_dist(x1, B)
        d2 = orth_dist(x2, B)
        if d1 >= d2:
            selected.append(x1)
            indices.append(idx1)
            B.append(x1)
        else:
            selected.append(x2)
            indices.append(idx2)
            B.append(x2)
    return np.array(selected), np.array(indices)


# ---------------------------
# 5. GS-COMPRESS
# ---------------------------
def gs_compress(X, delta=0.1, target=None):
    """
    Recursively applies GS-THIN until the number of points equals target.
    """
    n = X.shape[0]
    if target is None:
        target = n // 2
    X_current = X.copy()
    while X_current.shape[0] > target:
        X_current, _ = gs_thin(X_current, delta)
    return X_current, None  # Global indices are not tracked in this toy version


# ---------------------------
# Thinformer: Transformer Attention with Thinning
# ---------------------------
def thinformer_transformer_attention(Q, K, V, thinning_method, delta=0.1, target=None):
    """
    Approximate transformer attention using a thinning algorithm.

    thinning_method: a function that takes a matrix X and returns either
      (thinned_X, indices) or just thinned_X.

    Returns:
      T_approx: approximated attention output of shape (n, d)
      chosen_indices: the indices (if available) of the keys selected.
    """
    n = K.shape[0]
    if n % 2 != 0:
        K = K[:-1]
        V = V[:-1]
        Q = Q[:-1]
        n = K.shape[0]

    # Try calling the thinning method with (X, delta, target).
    try:
        result = thinning_method(K, delta=delta, target=target)
    except TypeError:
        result = thinning_method(K, delta=delta)

    if isinstance(result, tuple):
        thinned_keys, chosen_indices = result
    else:
        thinned_keys = result
        chosen_indices = None

    # If chosen_indices is available, use them to select corresponding values.
    if chosen_indices is not None:
        thinned_values = V[chosen_indices]
    else:
        # Otherwise, assume the thinning method preserves order and simply take the first n_out rows.
        n_out = thinned_keys.shape[0]
        thinned_values = V[:n_out]

    d = Q.shape[1]
    scores = np.dot(Q, thinned_keys.T) / np.sqrt(d)
    A = np.exp(scores)
    D = np.sum(A, axis=1, keepdims=True)
    T_approx = A @ thinned_values / D
    return T_approx, chosen_indices


# ---------------------------
# Main Experiment and Plotting
# ---------------------------
if __name__ == '__main__':
    np.random.seed(42)

    # Parameters for transformer attention:
    n = 1000  # number of tokens (ensure even)
    d = 64  # model dimension
    delta = 0.1  # failure parameter for thinning

    # Generate random queries, keys, and values.
    Q = np.random.randn(n, d)
    K_full = np.random.randn(n, d)
    V_full = np.random.randn(n, d)

    # Compute full attention output (gold standard)
    T_full = compute_full_attention(Q, K_full, V_full)

    # Define list of thinning methods (names and function handles)
    methods = [
        ("Uniform Subsampling", lambda X, delta, target: uniform_subsampling(X, target)),
        ("KH", kh_thinning),
        ("KH-COMPRESS", kh_compress),
        ("GS-THIN", gs_thin),
        ("GS-COMPRESS", gs_compress)
    ]

    target = n // 2  # For methods that support target, we set target = n/2.

    errors = []
    method_names = []

    # For each method, compute approximated attention and measure max absolute error.
    for name, method in methods:
        try:
            T_approx, idx = thinformer_transformer_attention(Q, K_full, V_full,
                                                             thinning_method=method,
                                                             delta=delta, target=target)
        except Exception as e:
            print(f"Error with method {name}: {e}")
            continue
        err = np.max(np.abs(T_full - T_approx))
        errors.append(err)
        method_names.append(name)
        print(f"{name}: max error = {err:.4f}")

    # Plot the errors in a bar chart.
    plt.figure(figsize=(8, 5))
    bars = plt.bar(method_names, errors, color='skyblue')
    plt.xlabel("Thinning Method")
    plt.ylabel("Max Absolute Error in Attention Output")
    plt.title("Comparison of Transformer Attention Approximations")
    for bar, err in zip(bars, errors):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01, f"{err:.4f}", ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
