"""Summary configuration dataclass"""

from dataclasses import dataclass

from llm_mediator_simulation.simulation.debater.config import DebaterConfig


@dataclass
class PrintableSummaryConfig:
    """Simpler/printable version of the SummaryConfig dataclass."""

    latest_messages_limit: int
    ignore: bool


@dataclass
class SummaryConfig:
    """Configuration for debate summaries provided to debaters prompts.

    Args:
        latest_messages_limit (int, optional): The number of latest messages to keep track of. Defaults to 3.
        debaters (list[DebaterConfig], optional): The list of debaters in the conversation. Only their names are used, as a mean of identification.
        ignore (bool, optional): If True, the summary will be ignored. Defaults to False.
    """

    latest_messages_limit: int = 3
    debaters: list[DebaterConfig] | None = None
    ignore: bool = False

    def to_printable(self):
        """Convert the SummaryConfig to a simpler PrintableSummaryConfig version for printing with pprint without overwheling informations."""
        return PrintableSummaryConfig(
            latest_messages_limit=self.latest_messages_limit,
            ignore=self.ignore,
        )
