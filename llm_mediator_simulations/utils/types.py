"""Type hinting utilities"""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

from llm_mediator_simulations.metrics.criteria import ArgumentQuality
from llm_mediator_simulations.utils.model_utils import Agreement


@dataclass
class Message:
    """Basic message type for the debate simulation

    Attributes:
        authorId (int): The author's index in the debater config list. If None, the message is from a mediator.
        text (str): The text content of the message.
        timestamp (datetime): The timestamp of the message.
        metrics (Metrics | None): The metrics associated with the message.
    """

    authorId: int | None
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
    argument_qualities: dict[ArgumentQuality, Agreement] | None = None


class LLMMessage(TypedDict):
    """LLM message intervention response format."""

    do_intervene: bool
    intervention_justification: str
    text: str
