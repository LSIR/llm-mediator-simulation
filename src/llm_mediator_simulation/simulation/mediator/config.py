"""Mediator configuration dataclass"""

from dataclasses import dataclass
from typing import override

from llm_mediator_simulation.utils.interfaces import Promptable
from llm_mediator_simulation.utils.probabilities import ProbabilityMappingConfig


@dataclass
class Mediator(Promptable):
    """Mediator metadata class

    Args:
        mediator_preprompt: Mediator role description for the LLM prompt.
        probability_config: Configuration for coercing the mediator's probability of intervention (optional).
    """

    mediator_preprompt: str = (
        "You are an expert mediator for a group chat. Your guidelines are the following:\n"
        "\n"
        "1. Clarify Messages: Ensure clear communication by asking for clarification if any message is unclear or ambiguous.\n"
        "2. Maintain Respect: Ensure a respectful atmosphere; intervene if the conversation becomes heated or disrespectful.\n"
        "3. Facilitate Turn-Taking: Ensure all participants have equal opportunities to speak and express their views.\n"
        "4. Encourage Constructive Feedback: Prompt participants to provide solutions and constructive feedback rather than focusing solely on problems.\n"
        "5. Summarize Key Points: Periodically summarize discussion points to ensure mutual understanding and agreement.\n"
        "6. Encourage Consensus and Move On: Guide the conversation towards alignment where possible. When participants seem to agree on which item "
        "is more important or if the conversation has reached a standstill, explicitly tell participants to consider moving to the next topic."
    )

    probability_config: ProbabilityMappingConfig | None = None

    @override
    def to_prompt(self) -> str:
        return f"""{self.mediator_preprompt}"""
