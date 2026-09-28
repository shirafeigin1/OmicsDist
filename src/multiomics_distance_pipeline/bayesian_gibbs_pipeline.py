"""Training and prediction for cross-omics imputation models.

The two regressions are separate conditional predictive models trained on
complete pairs. Each direction defines a conditional predictive distribution.
"""

import json
import time
from dataclasses import dataclass
import numpy as np
from .latent import ModalityPCA, Design
from .composition import Composition
from .gibbs_sampler import GaussianChain
from .arviz_diagnostics import run_chains
from .m1_knn_baseline import PCAKNN
from .distances import DistanceSpace

METHODS = ("knn_pca", "gaussian")


@dataclass
class Prediction:
    micro: np.ndarray
    metab: np.ndarray
    micro_draws: np.ndarray | None
    metab_draws: np.ndarray | None
    runtime: float


class FittedModels:
    def fit(self, micro, metab, cfg, out, seed):
        out.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.out = out
        t = time.perf_counter()
        self.micro_pca = ModalityPCA(cfg.n_components).fit(micro)
        self.metab_pca = ModalityPCA(cfg.n_components).fit(metab)
        self.space = DistanceSpace().fit(micro, metab)
        self.composition = Composition(cfg.micro_scale).fit(micro)
        zm, zt = self.micro_pca.transform(micro), self.metab_pca.transform(metab)
        self.micro_design = Design().fit(zm)
        self.metab_design = Design().fit(zt)
        xm, xt = self.micro_design.transform(zm), self.metab_design.transform(zt)
        self.preprocessing_seconds = time.perf_counter() - t
        t = time.perf_counter()
        self.knn = PCAKNN().fit(zm, zt, cfg.knn_neighbors)
        self.fit_times = {"knn_pca": time.perf_counter() - t}
        self.posteriors = {}
        self.status = {}
        self.direction_times = {}
        for key, x, y, cls, offset in [
            ("to_metab", xm, zt, GaussianChain, 11),
            ("to_micro", xt, zm, GaussianChain, 22),
        ]:
            print(f" Fitting {out.name}/{key}", flush=True)
            t = time.perf_counter()
            draws, status = run_chains(
                lambda s: cls(x, y, cfg, s), cfg, seed + offset, out / key
            )
            self.posteriors[key] = draws
            self.status[key] = status
            self.direction_times[key] = time.perf_counter() - t
        self.fit_times["gaussian"] = (
            self.direction_times["to_metab"] + self.direction_times["to_micro"]
        )
        details = {
            "microbiome_scale": self.composition.scale,
            "microbiome_pca_variance": self.micro_pca.explained_variance_sum,
            "metabolome_pca_variance": self.metab_pca.explained_variance_sum,
            "microbiome_components": self.micro_pca.pca.n_components_,
            "metabolome_components": self.metab_pca.pca.n_components_,
            "training_pairs": len(micro),
            "microbiome_constant_features": int(self.micro_pca.constant_mask.sum()),
            "metabolome_constant_features": int(self.metab_pca.constant_mask.sum()),
            "zero_fraction": float(np.mean(self.composition.probabilities(micro) == 0)),
            "preprocessing_seconds": self.preprocessing_seconds,
            "method_fit_seconds": self.fit_times,
            "direction_seconds": self.direction_times,
        }
        (out / "fit_details.json").write_text(json.dumps(details, indent=2))
        return self

    def converged(self, method):
        if method == "knn_pca":
            return True
        keys = ["to_metab", "to_micro"]
        return all(self.status[k]["converged"] for k in keys)

    def _predict_draws(self, key, x, pca, seed):
        """Average original-feature draws over every retained state of every chain.

        Retain in memory a balanced evenly spaced subset of predictive draws for plots and
        approximate predictive intervals. The mean does not use this subset.
        """
        post = self.posteriors[key]
        c, d = post["beta"].shape[:2]
        per_chain = min(d, max(1, self.cfg.predictive_draws // c))
        indices = np.linspace(0, d - 1, per_chain, dtype=int)
        selected = set(indices.tolist())
        retained = []
        total = None
        n = 0
        rng = np.random.default_rng(seed)
        for chain in range(c):
            for start in range(0, d, 32):
                stop = min(start + 32, d)
                eta = np.einsum("np,dpq->dnq", x, post["beta"][chain, start:stop])
                scores = eta + rng.normal(size=eta.shape) * np.sqrt(
                    post["sigma2"][chain, start:stop, None, :]
                )
                values = pca.inverse(scores.reshape(-1, scores.shape[-1])).reshape(
                    len(scores), len(x), -1
                )
                if total is None:
                    total = np.zeros(values.shape[1:])
                total += values.sum(0)
                n += len(values)
                retained.extend(
                    values[j - start].copy()
                    for j in range(start, stop)
                    if j in selected
                )
        mean = total / n
        # Restore exact constants after floating-point accumulation as well.
        mean[:, pca.constant_mask] = pca.constant_values
        return mean, np.asarray(retained)

    def predict(self, micro, metab, method, seed=0):
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        t = time.perf_counter()
        micro = micro.copy()
        metab = metab.copy()
        missing_m = np.isnan(micro).all(1)
        missing_t = np.isnan(metab).all(1)
        if np.any(missing_m & missing_t):
            raise ValueError("A sample cannot miss both blocks.")
        for x in (micro, metab):
            if np.any(np.isnan(x).any(1) & ~np.isnan(x).all(1)):
                raise ValueError("Input contains a partially missing omics block.")
        md = td = None
        if missing_m.any():
            z = self.metab_pca.transform(metab[missing_m])
            if method == "knn_pca":
                micro[missing_m] = self.micro_pca.inverse(self.knn.to_micro.predict(z))
            else:
                micro[missing_m], md = self._predict_draws(
                    "to_micro",
                    self.metab_design.transform(z),
                    self.micro_pca,
                    seed + 17,
                )
        if missing_t.any():
            z = self.micro_pca.transform(micro[missing_t])
            if method == "knn_pca":
                metab[missing_t] = self.metab_pca.inverse(self.knn.to_metab.predict(z))
            else:
                metab[missing_t], td = self._predict_draws(
                    "to_metab",
                    self.micro_design.transform(z),
                    self.metab_pca,
                    seed + 29,
                )
        if not np.isfinite(micro).all() or not np.isfinite(metab).all():
            raise FloatingPointError("Nonfinite prediction.")
        return Prediction(micro, metab, md, td, time.perf_counter() - t)
