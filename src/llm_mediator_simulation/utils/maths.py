"""Maths & probability utilities."""

from collections import deque

import numpy as np


class ProbabilityMapper:
    """Map a series of random probabilities of unknown distribution to a series of random values
    of known mean and standard deviation using z-score normalization."""

    def __init__(
        self, target_p: float, target_std: float, max_history: int | None = None
    ):
        """
        Args:
            target_p: The target mean probability.
            target_std: The target standard deviation.
            max_history: The maximum number of past probabilities to store for unknown mean and \
std estimation. If None, every probability is stored.
        """

        if max_history is not None and max_history < 2:
            raise ValueError("max_history must be at least 2.")

        self.target_p = target_p
        self.target_std = target_std
        self.max_history = max_history
        self.history: deque[float] = deque(maxlen=max_history)

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

        if std == 0:
            return p

        return (p - mean) * self.target_std / std + self.target_p  # type: ignore
