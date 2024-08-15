"""Maths & probability utilities."""

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ProbabilityMappingConfig:
    """Configuration for the ProbabilityMapper.

    Args:
        target_p: The target mean probability.
        target_std: The target standard deviation. If None, the original standard deviation is used.
        max_history: The maximum number of past probabilities to store for unknown mean and \
std estimation. If None, every probability is stored."""

    target_p: float
    target_std: float | None = None
    max_history: int | None = None


class ProbabilityMapper:
    """Map a series of random probabilities of unknown distribution to a series of random values
    of known mean and standard deviation using z-score normalization."""

    def __init__(self, config: ProbabilityMappingConfig):
        """
        Args:
            target_p: The target mean probability.
            target_std: The target standard deviation.
            max_history: The maximum number of past probabilities to store for unknown mean and \
std estimation. If None, every probability is stored.
        """
        self.target_p = config.target_p
        self.target_std = config.target_std
        self.max_history = config.max_history

        if self.max_history is not None and self.max_history < 2:
            raise ValueError("max_history must be at least 2.")

        self.history: deque[float] = deque(maxlen=self.max_history)

    def map(self, p: float) -> float:
        """Map a probability to a value with the target mean and standard deviation.

        Args:
            p: The probability to map.

        Returns:
            The mapped value.
        """

        self.history.append(p)

        if len(self.history) == 0:
            return p

        mean = np.mean(self.history)
        std = np.std(self.history)

        if self.target_std is None:
            return np.clip(p - mean + self.target_p, 0, 1)  # type: ignore

        if std == 0:
            return p

        return np.clip((p - mean) * self.target_std / std + self.target_p, 0, 1)  # type: ignore
