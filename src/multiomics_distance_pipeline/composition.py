"""Microbiome input scale validation."""

import numpy as np
from scipy.special import softmax


class Composition:
    def __init__(self, scale="auto"):
        self.scale = scale

    def fit(self, x):
        x = np.asarray(x, float)
        relative = np.all(x >= 0) and np.allclose(x.sum(1), 1, atol=1e-6)
        clr = np.any(x < 0) and np.allclose(x.mean(1), 0, atol=1e-6)
        if self.scale == "auto":
            if relative:
                self.scale = "relative"
            elif clr:
                self.scale = "clr"
            else:
                raise ValueError(
                    "Unknown microbiome scale: supply relative abundances or centered CLR."
                )
        if self.scale == "relative" and not relative:
            raise ValueError("Relative abundances must be nonnegative and sum to one.")
        if self.scale == "clr" and not np.allclose(x.mean(1), 0, atol=1e-6):
            raise ValueError("CLR rows must have zero mean.")
        return self

    def probabilities(self, x):
        return softmax(x, axis=-1) if self.scale == "clr" else np.asarray(x, float)
