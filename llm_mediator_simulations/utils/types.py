"""Type hinting utilities"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class Message:
    """Basic message type for the debate simulation

    Attributes:
        text (str): The text content of the message.
        timestamp (datetime): The timestamp of the message.
        authorId (int): The author's index in the debater config list.
        metrics (Metrics | None): The metrics associated with the message.
    """

    authorId: int
    text: str
    timestamp: datetime
    metrics: "Metrics | None" = None


@dataclass
class Metrics:
    """Message metrics.

    Attributes:
        perspective (float | None): Toxicity rating ranging from 0 (not toxic) to 1 (definitely toxic).
    """

    perspective: float | None = None
    # TODO: add more metrics (our custom ones, for instance)
