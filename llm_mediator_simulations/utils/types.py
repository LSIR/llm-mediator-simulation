"""Type hinting utilities"""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

from llm_mediator_simulations.metrics.criteria import ArgumentQuality
from llm_mediator_simulations.utils.model_utils import Agreement


@dataclass
class Intervention:
    """Basic intervention data type for the debate simulation

    Attributes:
        authorId (int): The author's index in the debater config list. If None, the intervention is from a mediator.
        text (str | None): The text content of the intervention. If None, the author decided not to intervene.
        justification (str): The justification for the intervention.
        timestamp (datetime): The timestamp of the intervention.
        metrics (Metrics | None): The metrics associated with the intervention.
    """

    authorId: int | None
    text: str | None
    justification: str
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
