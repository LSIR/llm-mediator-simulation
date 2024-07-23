"""Type hinting utilities"""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.utils.model_utils import Agreement


@dataclass
class Intervention:
    """Basic intervention data type for the debate simulation

    Attributes:
        authorId: The author's index in the debater config list. If None, the intervention is from a mediator.
        text: The text content of the intervention. If None, the author decided not to intervene.
        prompt: The prompt for the intervention.
        justification: The justification for the intervention.
        timestamp: The timestamp of the intervention.
        metrics: The metrics associated with the intervention.
    """

    authorId: int | None
    text: str | None
    prompt: str
    justification: str
    timestamp: datetime
    metrics: "Metrics | None" = None


@dataclass
class Metrics:
    """Message metrics.

    Attributes:
        perspective: Toxicity rating ranging from 0 (not toxic) to 1 (definitely toxic).
    """

    perspective: float | None = None
    argument_qualities: dict[ArgumentQuality, Agreement] | None = None


class LLMMessage(TypedDict):
    """LLM message intervention response format."""

    do_intervene: bool
    intervention_justification: str
    text: str
