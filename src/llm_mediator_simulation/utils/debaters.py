"""Debaters utilities."""

from llm_mediator_simulation.simulation.debater.async_handler import AsyncDebaterHandler
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
from llm_mediator_simulation.simulation.debater.handler import DebaterHandler


def remove_statement_from_personalities(
    debaters: list[DebaterHandler] | list[AsyncDebaterHandler], statement: str
) -> None:
    """remove the statement from the list of statements if it is in the debater's agreement_with_statements"""
    for debater in debaters:
        if isinstance(debater, AsyncDebaterHandler):
            for config in debater.configs:
                _remove_statement_from_personality(config, statement)
        else:
            _remove_statement_from_personality(debater.config, statement)


def _remove_statement_from_personality(config: DebaterConfig, statement: str) -> None:
    if config.personality is not None and config.personality.agreement_with_statements:
        if statement in config.personality.agreement_with_statements:
            if isinstance(config.personality.agreement_with_statements, list):
                config.personality.agreement_with_statements.remove(statement)
            elif isinstance(config.personality.agreement_with_statements, dict):  # type: ignore
                config.personality.agreement_with_statements.pop(statement)
            else:
                raise ValueError("agreement_with_statements should be a list or a dict")
