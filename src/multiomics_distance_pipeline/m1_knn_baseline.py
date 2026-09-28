"""Cross-omics KNN regression in training PCA space.

Neighbors are selected using the observed modality only. Their target PCs are
averaged with distance weights, then reconstructed. No metadata enters KNN.
"""

from sklearn.neighbors import KNeighborsRegressor


class PCAKNN:
    def fit(self, z_micro, z_metab, k):
        self.to_micro = KNeighborsRegressor(
            n_neighbors=min(k, len(z_micro)), weights="distance"
        ).fit(z_metab, z_micro)
        self.to_metab = KNeighborsRegressor(
            n_neighbors=min(k, len(z_micro)), weights="distance"
        ).fit(z_micro, z_metab)
        return self
