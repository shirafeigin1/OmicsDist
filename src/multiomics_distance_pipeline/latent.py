"""Fit once on training observations, then transform held-out rows."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ModalityPCA:
    def __init__(self, n_components=15, random_state=0):
        self.n_components = n_components
        self.seed = random_state

    def fit(self, x):
        self.constant_mask = np.all(x == x[0], axis=0)
        self.constant_values = np.asarray(x[0, self.constant_mask]).copy()
        self.scaler = StandardScaler().fit(x)
        self.pca = PCA(
            n_components=min(self.n_components, len(x) - 1, x.shape[1]),
            svd_solver="full",
            random_state=self.seed,
        )
        self.pca.fit(self.scaler.transform(x))
        return self

    def transform(self, x):
        return self.pca.transform(self.scaler.transform(x))

    def inverse(self, z):
        values = self.scaler.inverse_transform(self.pca.inverse_transform(z))
        # Preserve training constants exactly after inverse projection.
        values[:, self.constant_mask] = self.constant_values
        return values

    @property
    def explained_variance_sum(self):
        return float(self.pca.explained_variance_ratio_.sum())


class Design:
    def __init__(self, whiten=False):
        self.whiten = whiten

    def fit(self, z):
        self.scaler = StandardScaler().fit(z)
        z = self.scaler.transform(z)
        self.chol = (
            np.linalg.cholesky(z.T @ z / len(z) + 1e-6 * np.eye(z.shape[1]))
            if self.whiten
            else np.eye(z.shape[1])
        )
        return self

    def transform(self, z):
        z = np.linalg.solve(self.chol, self.scaler.transform(z).T).T
        return np.column_stack([np.ones(len(z)), z])
