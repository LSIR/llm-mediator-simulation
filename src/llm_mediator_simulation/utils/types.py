"""Type hinting utilities"""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.simulation.configuration import Debater
from llm_mediator_simulation.utils.model_utils import Agreement


@dataclass
class Intervention:
    """Basic intervention data type for the debate simulation

    Attributes:
        debater: The configuration of the debater who intervened. If None, the author is a mediator.
        text: The text content of the intervention. If None, the author decided not to intervene.
        prompt: The prompt for the intervention.
        justification: The justification for the intervention.
        timestamp: The timestamp of the intervention.
        metrics: The metrics associated with the intervention.
    """

    debater: Debater | None
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


class LLMProbaMessage(TypedDict):
    """LLM message intervention response format.
    Same as LLMMessage, but with a probability of intervention instead of a decision."""

    do_intervene: float
    intervention_justification: str
    text: str
