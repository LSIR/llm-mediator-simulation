"""Prompt utilities for the debate simulation."""

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)
from llm_mediator_simulation.simulation.configuration import (
    DebateConfig,
    Debater,
    Mediator,
)
from llm_mediator_simulation.simulation.summary_handler import (
    AsyncSummaryHandler,
    SummaryHandler,
)
from llm_mediator_simulation.utils.decorators import retry
from llm_mediator_simulation.utils.json import (
    json_prompt,
    parse_llm_json,
    parse_llm_jsons,
)
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


async def async_debater_interventions(
    model: AsyncLanguageModel,
    config: DebateConfig,
    summary: AsyncSummaryHandler,
    debaters: list[Debater],
    retry_attempts: int = 5,
) -> list[LLMMessage]:
    """Debater intervention: decision, motivation for the intervention, and intervention content. Asynchonous / batched."""

    prompts: list[str] = []
    summary_prompts = await summary.to_prompts()

    for debater, debate_summary in zip(debaters, summary_prompts):

        prompts.append(
            f"""{config.to_prompt()}. {debater.to_prompt()} {debate_summary}

        Do you want to add a comment to the online debate right now?
        You should often add a comment when the previous context is empty or not in the favor of your \
        position. However, you should almost never add a comment when the previous context already \
        supports your position. Use short chat messages, no more than 3 sentences.

        {json_prompt(LLM_RESPONSE_FORMAT)}
        """
        )

    responses = await model.sample(prompts)
    coerced, failed = parse_llm_jsons(responses, LLMMessage)

    attempts = 1
    while len(failed) > 0 and attempts < retry_attempts:
        prompts = [prompts[i] for i in failed]
        responses = await model.sample(prompts)
        new_coerced, new_failed = parse_llm_jsons(responses, LLMMessage)
        failed = new_failed
        coerced.extend(new_coerced)
        attempts += 1

    if len(failed) > 0:
        raise ValueError(
            f"Failed to parse {len(failed)} LLM responses after {retry_attempts} attempts"
        )

    return coerced


async def async_mediator_interventions(
    model: AsyncLanguageModel,
    config: DebateConfig,
    mediator: Mediator,
    summary: AsyncSummaryHandler,
    retry_attempts: int = 5,
) -> list[LLMMessage]:

    prompts: list[str] = []
    summary_prompts = summary.raw_history_prompts()

    for debate_summary in summary_prompts:
        prompts.append(
            f"""{config.to_prompt()}. 

            {summary.debaters_prompt()}

            CONVERSATION HISTORY WITH TIMESTAMPS:
            {debate_summary} 

            {mediator.to_prompt()}

            {json_prompt(LLM_RESPONSE_FORMAT)}
            """
        )

    responses = await model.sample(prompts)
    coerced, failed = parse_llm_jsons(responses, LLMMessage)

    attempts = 1
    while len(failed) > 0 and attempts < retry_attempts:
        prompts = [prompts[i] for i in failed]
        responses = await model.sample(prompts)
        new_coerced, new_failed = parse_llm_jsons(responses, LLMMessage)
        failed = new_failed
        coerced.extend(new_coerced)
        attempts += 1

    if len(failed) > 0:
        raise ValueError(
            f"Failed to parse {len(failed)} LLM responses after {retry_attempts} attempts"
        )

    return coerced
