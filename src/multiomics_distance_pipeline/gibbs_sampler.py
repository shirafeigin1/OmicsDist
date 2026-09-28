"""Normal-inverse-gamma regression trained on observed pairs.

Missing responses integrate out of the training likelihood. A collapsed Gibbs
chain samples beta and sigma2 on training pairs only. Posterior prediction is
then independent of the set of held-out/missing rows requested.
"""

import numpy as np


class GaussianChain:
    def __init__(self, x, y, cfg, seed):
        self.rng = np.random.default_rng(seed)
        self.x, self.y, self.cfg = x, y, cfg
        n, p = x.shape
        q = y.shape[1]
        self.precision = np.eye(p) / cfg.tau2
        self.precision[0, 0] /= 100
        self.v = np.linalg.inv(x.T @ x + self.precision)
        self.chol = np.linalg.cholesky(self.v)
        self.mean = self.v @ x.T @ y
        # Overdispersed, independent initialization across chains.
        self.sigma = np.maximum(np.var(y, axis=0), 0.1) * np.exp(
            self.rng.normal(0, 1, q)
        )
        self.beta = (
            self.mean
            + (self.chol @ self.rng.normal(size=(p, q))) * np.sqrt(self.sigma) * 3
        )
        self.n, self.p = n, p

    def step(self, iteration):
        self.beta = self.mean + (
            self.chol @ self.rng.normal(size=self.beta.shape)
        ) * np.sqrt(self.sigma)
        resid = self.y - self.x @ self.beta
        quad = np.sum(self.beta * (self.precision @ self.beta), axis=0)
        # p/2 is required because beta|sigma2 has sigma2-dependent normalization.
        shape = self.cfg.a0 + (self.n + self.p) / 2
        scale = self.cfg.b0 + (np.sum(resid * resid, axis=0) + quad) / 2
        self.sigma = scale / self.rng.gamma(shape, size=len(scale))

    def values(self):
        return {"beta": self.beta.copy(), "sigma2": self.sigma.copy()}

    def status(self):
        return {}
