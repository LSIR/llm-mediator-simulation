"""Type hinting utilities"""

from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
from llm_mediator_simulation.utils.model_utils import Agreement


@dataclass
class PrintableIntervention:
    """Simpler / printable version of the Intervention dataclass."""

    debater: str
    # debater_update: Actually not necessary since we already have CLI to plot evolution of personalities.
    text: str | None
    prompt: list[str]
    justification: str
    timestamp: datetime
    metrics: "Metrics | None" = None


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

    debater: DebaterConfig | None
    text: str | None
    prompt: str
    justification: str
    timestamp: datetime
    metrics: "Metrics | None" = None

    def to_printable(self) -> PrintableIntervention:
        return PrintableIntervention(
            debater=self.debater.name if self.debater else "Mediator",
            text=self.text,
            prompt=self.prompt.splitlines(),
            justification=self.justification,
            timestamp=self.timestamp,
            metrics=self.metrics,
        )


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

    do_write: bool
    justification: str
    text: str


class LLMProbaMessage(TypedDict):
    """LLM message intervention response format.
    Same as LLMMessage, but with a probability of intervention instead of a decision."""

    do_intervene: float
    justification: str
    text: str
