"""Prompt utilities for the debate simulation."""

from llm_mediator_simulation.models.language_model import LanguageModel
from llm_mediator_simulation.simulation.configuration import (
    DebateConfig,
    Debater,
    Mediator,
)
from llm_mediator_simulation.simulation.summary_handler import SummaryHandler
from llm_mediator_simulation.utils.decorators import retry
from llm_mediator_simulation.utils.json import json_prompt, parse_llm_json
from llm_mediator_simulation.utils.types import LLMMessage

LLM_RESPONSE_FORMAT: dict[str, str] = {
    "do_intervene": "bool",
    "intervention_justification": "a string justification of why you want to intervene or not",
    "text": "the text message for your intervention. Leave empty if you decide not to intervene",
}


@retry(attempts=5, verbose=True)
def debater_intervention(
    model: LanguageModel,
    config: DebateConfig,
    summary: SummaryHandler,
    debater: Debater,
) -> LLMMessage:
    """Debater intervention: decision, motivation for the intervention, and intervention content."""

    prompt = f"""{config.to_prompt()}. {debater.to_prompt()} {summary.to_prompt()}

    Do you want to add a comment to the online debate right now?
    You should often add a comment when the previous context is empty or not in the favor of your \
    position. However, you should almost never add a comment when the previous context already \
    supports your position. Use short chat messages, no more than 3 sentences.

    {json_prompt(LLM_RESPONSE_FORMAT)}
    """

    response = model.sample(prompt)
    return parse_llm_json(response, LLMMessage)


@retry(attempts=5, verbose=True)
def mediator_intervention(
    model: LanguageModel,
    config: DebateConfig,
    mediator: Mediator,
    summary: SummaryHandler,
) -> LLMMessage:
    """Mediator intervention: decision, motivation for the intervention, and intervention content."""
    prompt = f"""{config.to_prompt()}. 

{summary.debaters_prompt()}

CONVERSATION HISTORY WITH TIMESTAMPS:
{summary.raw_history_prompt()} 

{mediator.to_prompt()}

{json_prompt(LLM_RESPONSE_FORMAT)}
    """

    response = model.sample(prompt)
    return parse_llm_json(response, LLMMessage)
