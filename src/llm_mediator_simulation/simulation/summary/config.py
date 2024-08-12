"""Summary configuration dataclass"""

from dataclasses import dataclass

from llm_mediator_simulation.simulation.debater.config import DebaterConfig


@dataclass
class SummaryConfig:
    """Configuration for debate summaries provided to debaters prompts.

    Args:
        latest_messages_limit (int, optional): The number of latest messages to keep track of. Defaults to 3.
        debaters (list[DebaterConfig], optional): The list of debaters in the conversation. Only their names are used, as a mean of identification.
    """

    latest_messages_limit: int = 3
    debaters: list[DebaterConfig] | None = None
