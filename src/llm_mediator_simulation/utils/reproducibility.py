"""Random seeding for reproducibility."""

from transformers import set_seed


def set_transformers_seed(seed: int) -> None:
    """Set the transformer's random seed for reproducibility.

    Args:
        seed: The seed to set.
    """
    set_seed(seed)
