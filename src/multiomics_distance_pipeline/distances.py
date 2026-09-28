import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform


class DistanceSpace:
    """Common PC1/PC2 distance geometry, fitted only on the training partition."""

    def fit(self, micro, metab):
        x = np.column_stack([micro, metab])
        self.scaler = StandardScaler().fit(x)
        self.pca = PCA(n_components=2, svd_solver="full").fit(self.scaler.transform(x))
        return self

    def coordinates(self, micro, metab):
        return self.pca.transform(
            self.scaler.transform(np.column_stack([micro, metab]))
        )

    def distance(self, micro, metab):
        return squareform(pdist(self.coordinates(micro, metab)))


def mantel(reference, predicted, permutations=999, seed=0):
    """Two-sided Mantel permutation test by permuting sample labels, never pairs."""
    from scipy.stats import rankdata

    n = len(reference)
    tri = np.triu_indices(n, 1)
    a, b = reference[tri], predicted[tri]

    def corr(x, y):
        xc = x - x.mean()
        yc = y - y.mean()
        den = np.linalg.norm(xc) * np.linalg.norm(yc)
        return float(xc @ yc / den) if den > 0 else float("nan")

    r = corr(a, b)
    rs = corr(rankdata(a), rankdata(b))
    if not np.isfinite(r):
        return r, rs, float("nan")
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        p = rng.permutation(n)
        count += abs(corr(a, predicted[np.ix_(p, p)][tri])) >= abs(r)
    return r, rs, (count + 1) / (permutations + 1) if permutations else float("nan")
