"""Prompt utilities for the debate simulation."""

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)
from llm_mediator_simulation.simulation.configuration import (
    AxisPosition,
    DebateConfig,
    Debater,
    Mediator,
    PersonalityAxis,
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
from llm_mediator_simulation.utils.model_utils import clip
from llm_mediator_simulation.utils.types import Intervention, LLMMessage

LLM_RESPONSE_FORMAT: dict[str, str] = {
    "do_intervene": "bool",
    "intervention_justification": "a string justification of why you want to intervene or not, which will not be visible by others.",
    "text": "the text message for your intervention, visible by others. Leave empty if you decide not to intervene",
}


@retry(attempts=5, verbose=True)
def debater_intervention(
    model: LanguageModel,
    config: DebateConfig,
    summary: SummaryHandler,
    debater: Debater,
) -> tuple[LLMMessage, str]:
    """Debater intervention: decision, motivation for the intervention, and intervention content."""

    prompt = f"""{config.to_prompt()}. {debater.to_prompt()} {summary.to_prompt()}

Do you want to add a comment to the online debate right now?
You should often add a comment when the previous context is empty or not in the favor of your \
position. However, you should almost never add a comment when the previous context already \
supports your position. Use short chat messages, no more than 3 sentences.

{json_prompt(LLM_RESPONSE_FORMAT)}
"""

    response = model.sample(prompt)
    return parse_llm_json(response, LLMMessage), prompt


@retry(attempts=5, verbose=True)
def debater_personality_update(
    model: LanguageModel,
    debater: Debater,
    interventions: list[Intervention],
) -> str:
    """Update a debater's personality based on the interventions passed as arguments.
    The debater configuration personality is updated in place."""

    if debater.personalities is None:
        return ""

    sep = "\n"

    positions: list[str] = []
    for axis, position in (debater.personalities or {}).items():
        positions.append(
            f"{axis.value.name}: from {axis.value.left} to {axis.value.right} on a scale from 0 to 4, you are currently at {position.value}"
        )

    answer_format: dict[str, str] = {
        axis.value.name: "an integer (-1, 0, 1) to update this axis"
        for axis in (debater.personalities or {}).keys()
    }

    prompt = f"""You have the opportunity to make your personality evolve based on the things people have said after your last intervention.

Here is your current personality:
{sep.join(positions)}

Here are the last messages:
{sep.join([intervention.text for intervention in interventions if intervention.text ])}

You can choose to evolve your personality on all axes by +1, -1 or 0.
{json_prompt(answer_format)}
"""

    response = model.sample(prompt)
    data: dict[str, str] = parse_llm_json(response)

    # Process the axis updates
    for axis, update in data.items():
        update = int(update)  # Fails on error
        axis = PersonalityAxis.from_string(axis)
        if update not in [-1, 0, 1]:
            raise ValueError("Personality update must be -1, 0 or 1.")
        if axis not in debater.personalities:
            raise ValueError(f"Unknown personality axis: {axis}")

        new_value = clip(debater.personalities[axis].value + update, 0, 4)
        new_position = AxisPosition(new_value)
        debater.personalities[axis] = new_position

    return prompt


@retry(attempts=5, verbose=True)
def mediator_intervention(
    model: LanguageModel,
    config: DebateConfig,
    mediator: Mediator,
    summary: SummaryHandler,
) -> tuple[LLMMessage, str]:
    """Mediator intervention: decision, motivation for the intervention, and intervention content."""
    prompt = f"""{config.to_prompt()}. 

{summary.debaters_prompt()}

CONVERSATION HISTORY WITH TIMESTAMPS:
{summary.raw_history_prompt()} 

{mediator.to_prompt()}

{json_prompt(LLM_RESPONSE_FORMAT)}
    """

    response = model.sample(prompt)
    return parse_llm_json(response, LLMMessage), prompt


async def async_debater_interventions(
    model: AsyncLanguageModel,
    config: DebateConfig,
    summary: AsyncSummaryHandler,
    debaters: list[Debater],
    retry_attempts: int = 5,
) -> tuple[list[LLMMessage], list[str]]:
    """Debater intervention: decision, motivation for the intervention, and intervention content. Asynchonous / batched.

    Args:
        model: The language model to use.
        config: The debate configuration.
        summary: The conversation summary handler for the parallel debates.
        debaters: The debaters participating the respective debates (1 per debate. They can be the same repeated).
        retry_attempts: The number of retry attempts in case of parsing failure. Defaults to 5.
    """

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
        # Print the prompt and response of one of the failed attempts
        prompt = prompts[failed[0]]
        response = responses[failed[0]]

        print("Prompt for last failed invocation:")
        print(prompt)
        print()

        print("Response for last failed invocation:")
        print(response)
        print()

        raise ValueError(
            f"Failed to parse {len(failed)} LLM responses after {retry_attempts} attempts"
        )

    return coerced, prompts


async def async_mediator_interventions(
    model: AsyncLanguageModel,
    config: DebateConfig,
    mediator: Mediator,
    summary: AsyncSummaryHandler,
    retry_attempts: int = 5,
) -> tuple[list[LLMMessage], list[str]]:

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
        # Print the prompt and response of one of the failed attempts
        prompt = prompts[failed[0]]
        response = responses[failed[0]]

        print("Prompt for last failed invocation:")
        print(prompt)
        print()

        print("Response for last failed invocation:")
        print(response)
        print()

        raise ValueError(
            f"Failed to parse {len(failed)} LLM responses after {retry_attempts} attempts"
        )

    return coerced, prompts
