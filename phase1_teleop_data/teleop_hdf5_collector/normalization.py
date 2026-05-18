"""Numerically stable running statistics for online normalization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass
class RunningStats:
    """Track per-dimension mean and variance with Welford updates."""

    dim: int

    def __post_init__(self) -> None:
        self.count: int = 0
        self.mean: FloatArray = np.zeros((self.dim,), dtype=np.float64)
        self.m2: FloatArray = np.zeros((self.dim,), dtype=np.float64)

    def update(self, value: NDArray[np.floating]) -> None:
        """Update statistics with one observation.

        Args:
            value: Shape `[dim]` vector.
        """
        x = np.asarray(value, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(f"Expected shape {(self.dim,)}, got {x.shape}")

        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def std(self, eps: float = 1e-6) -> FloatArray:
        """Return per-dimension standard deviation."""
        if self.count < 2:
            return np.ones((self.dim,), dtype=np.float64)
        variance = self.m2 / max(self.count - 1, 1)
        return np.sqrt(np.maximum(variance, eps))

    def normalize(self, value: NDArray[np.floating], eps: float = 1e-6) -> FloatArray:
        """Z-score normalize a vector using current stats."""
        x = np.asarray(value, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(f"Expected shape {(self.dim,)}, got {x.shape}")
        return (x - self.mean) / self.std(eps=eps)
