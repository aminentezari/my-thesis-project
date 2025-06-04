import numpy as np
from scipy import linalg


class LDA:
    def __init__(self, n_components, tau=1e-10):
        self.n_components = n_components
        self.linear_discriminants = None
        self.eigenvalues = None
        self.tau = tau

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)
        n_classes = len(class_labels)

        # Ensure n_components is valid (max is n_classes-1)
        self.n_components = min(self.n_components, n_classes - 1)

        # Calculate cluster means and global mean
        mean_overall = np.mean(X, axis=0)

        # Initialize scatter matrices
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))

        # Calculate scatter matrices for each cluster
        for c in class_labels:
            X_c = X[y == c]
            n_c = X_c.shape[0]  # Number of samples in this cluster
            mean_c = np.mean(X_c, axis=0)

            # Within-class scatter for this cluster
            S_c = (X_c - mean_c).T.dot(X_c - mean_c)
            S_W += S_c

            # Between-class scatter
            mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += n_c * mean_diff.dot(mean_diff.T)

        # Calculate and print determinant of S_W before regularization
        try:
            det_S_W = np.linalg.det(S_W)
            print(f"Determinant of S_W before regularization: {det_S_W}")
            # Check if the matrix is singular or near-singular
            if abs(det_S_W) < 1e-10:
                print("Warning: S_W is nearly singular, regularization is needed.")
        except:
            print("Determinant calculation failed - S_W might be singular or ill-conditioned.")

        # Step 5: Compute largest eigenvalue of S_W and apply regularization
        S_W_eigenvalues = np.linalg.eigvalsh(S_W)
        d12 = np.max(S_W_eigenvalues)  # Largest eigenvalue
        epsilon = self.tau * d12

        print(f"Largest eigenvalue of S_W: {d12}")
        print(f"Epsilon value for regularization: {epsilon}")

        # Regularize S_W
        S_W_reg = S_W + epsilon * np.eye(n_features)

        # Calculate determinant after regularization
        sign, logdet = np.linalg.slogdet(S_W_reg)
        print("Log-determinant of S_W after regularization:", logdet)

        # Step 6: Compute Cholesky factor K
        # S_W_reg = K.T * K, where K is upper triangular
        try:
            K = linalg.cholesky(S_W_reg, lower=False)  # Upper triangular Cholesky factor #Sw reg= K.T . K
        except np.linalg.LinAlgError:
            # If Cholesky fails, fall back to eigendecomposition
            print("Warning: Cholesky decomposition failed. Using eigendecomposition instead.")
            eigenvalues, eigenvectors = np.linalg.eigh(S_W_reg)
            # Ensure all eigenvalues are positive
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            K = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues)))

        # Step 7: Compute eigenvalues and eigenvectors of K^(-T) * S_B * K^(-1)
        # First compute K^(-1)
        try:
            # Try triangular solver for Cholesky
            K_inv = linalg.solve_triangular(K, np.eye(n_features), lower=False)
        except:
            # Fallback if K is not triangular (e.g., from eigendecomposition)
            K_inv = np.linalg.pinv(K)

        # Compute A = K^(-T) * S_B * K^(-1)
        A = K_inv.T @ S_B @ K_inv

        # Clip to avoid overflow
        A = np.clip(A, -1e100, 1e100)

        # Ensure symmetry
        A = (A + A.T) / 2

        # Check for NaN or Inf
        if not np.all(np.isfinite(A)):
            raise ValueError(
                "Matrix A contains inf or NaN values. This usually indicates instability in regularization (tau) or data.")

        # Compute eigenvalues and eigenvectors of A
        eigenvalues, eigenvectors_w = np.linalg.eigh(A)

        # Sort eigenvalues in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors_w = eigenvectors_w[:, idx]

        # Store top eigenvalues
        self.eigenvalues = eigenvalues[:self.n_components]

        # Step 8: Compute eigenvectors q of (S_W + epsilon*I)^(-1) * S_B
        # by solving K * q = w
        linear_discriminants = np.zeros((self.n_components, n_features))

        for j in range(self.n_components):
            try:
                # Try solving with triangular solver (fast)
                q = linalg.solve_triangular(K, eigenvectors_w[:, j], lower=False)
            except:
                # Use general solver if K is not triangular
                q = np.linalg.solve(K, eigenvectors_w[:, j])
            linear_discriminants[j] = q

        # Store linear discriminants
        self.linear_discriminants = linear_discriminants

    def transform(self, X):
        # Project data onto linear discriminants
        return np.dot(X, self.linear_discriminants.T)