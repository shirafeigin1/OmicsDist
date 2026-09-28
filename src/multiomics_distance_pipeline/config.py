from dataclasses import dataclass, asdict


@dataclass
class Config:
    n_components: int = 15
    knn_neighbors: int = 5
    n_chains: int = 4
    burn_in: int = 2000
    min_iter: int = 6000
    max_iter: int = 20000
    check_every: int = 2000
    thin: int = 5
    rhat_limit: float = 1.01
    ess_min: int = 400
    predictive_draws: int = 200
    tau2: float = 10.0
    a0: float = 2.0
    b0: float = 1.0
    micro_scale: str = "auto"
    folds: int = 5
    repeats: int = 2
    missing_rates: tuple = (0.1, 0.2, 0.3, 0.4, 0.5)
    permutations: int = 999
    seed: int = 42

    def validate(self):
        if not (
            self.n_chains >= 2 and 0 <= self.burn_in < self.min_iter <= self.max_iter
        ):
            raise ValueError("Require chains >= 2 and burn_in < min_iter <= max_iter.")
        if (
            min(
                self.thin,
                self.check_every,
                self.n_components,
                self.knn_neighbors,
                self.predictive_draws,
            )
            < 1
        ):
            raise ValueError("Counts must be positive.")
        if (self.min_iter - self.burn_in) // self.thin < 8:
            raise ValueError("Need at least 8 retained draws per chain.")
        if (
            self.folds < 2
            or self.repeats < 1
            or any(not 0 < r <= 1 for r in self.missing_rates)
        ):
            raise ValueError("Invalid cross-validation settings.")
        if self.permutations < 0:
            raise ValueError("Permutation count must be nonnegative.")
        if min(self.tau2, self.a0, self.b0, self.ess_min) <= 0 or self.rhat_limit <= 1:
            raise ValueError("Invalid priors or convergence thresholds.")
        if self.micro_scale not in ("auto", "relative", "clr"):
            raise ValueError("micro_scale must be auto, relative or clr.")
        return self

    def dict(self):
        return asdict(self)
