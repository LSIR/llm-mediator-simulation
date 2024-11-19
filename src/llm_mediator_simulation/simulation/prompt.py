"""Prompt utilities for the debate simulation."""

from numpy import random

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import (
    AxisPosition,
    DebaterConfig,
    PersonalityAxis,
)

from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.summary.async_handler import AsyncSummaryHandler
from llm_mediator_simulation.simulation.summary.handler import SummaryHandler
from llm_mediator_simulation.utils.decorators import retry
from llm_mediator_simulation.utils.json import (
    json_prompt,
    parse_llm_json,
    parse_llm_jsons,
)
from llm_mediator_simulation.utils.model_utils import clip
from llm_mediator_simulation.utils.probabilities import ProbabilityMapper
from llm_mediator_simulation.utils.types import (
    Intervention,
    LLMMessage,
    LLMProbaMessage,
)

LLM_RESPONSE_FORMAT: dict[str, str] = {
    "do_intervene": "bool",
    "intervention_justification": "a string justification of why you want to intervene or not, which will not be visible by others.",
    "text": "the text message for your intervention, visible by others. Leave empty if you decide not to intervene",
}

LLM_PROBA_RESPONSE_FORMAT: dict[str, str] = {
    "do_intervene": "a float probability of intervention",
    "intervention_justification": "a string justification of why you want to intervene or not, which will not be visible by others.",
    "text": "the text message for your intervention, visible by others. Leave empty if you decide not to intervene",
}


    

# @retry(attempts=5, verbose=True)
# def debater_intervention(
#     time: str,
#     model: LanguageModel,
#     config: DebateConfig,
#     summary: SummaryHandler,
#     debater: DebaterConfig,
# ) -> tuple[LLMMessage, str]:
    
   
#     """Debater intervention: decision, motivation for the intervention, and intervention content."""
    
#     prompt = f"""{config.to_prompt()}. {debater.to_prompt()} {summary.to_prompt()}

#     Do you want to comment the online debate right now? Use short chat messages, no more than 3 sentences. 
    
#     {json_prompt(LLM_RESPONSE_FORMAT)}
    
#     """

#     initial_response = model.generate(prompt)
#     with open("initial_response.txt", "w") as file:
#         file.write(initial_response)
    
#     reflexion_prompt = f"""
#     Reflect on the following interaction and provide detailed constructive feedback. Evaluate the response based on the following criteria: 
#     1. Does it address the prompt adequately?
#     2. Is this response natural and similar to what a human would say?
#     3. Does the response stay coherent with the assigned role, maintaining appropriate tone, perspective, and alignment with the role's objectives?
#     4. Does the response accurately understand who it is speaking to and its own role, without mistakenly assuming there is another person with the same name?
#     5. Are there areas for improvement?

#     Initial Response:
#     {initial_response}
    
#     Provided Prompt:
#     {prompt}

#     Feedback:
    
#     """
#     reflexion_response = model.generate(reflexion_prompt)
    
        
#     final_prompt = f"""
#     You were asked to respond to the following instruction:
#     {prompt}

#     Your previous response was:
#     {initial_response}

#     Based on the reflection feedback provided below, please improve your response:
#     {reflexion_response}

#     Revised Response:
    
    
#     {json_prompt(LLM_RESPONSE_FORMAT)}
#     """
    
#     final_response = model.generate(final_prompt)

    
#     return  parse_llm_json(final_response, LLMMessage), final_prompt


@retry(attempts=5, verbose=True)
def debater_intervention(
    model: LanguageModel,
    config: DebateConfig,
    summary: SummaryHandler,
    debater: DebaterConfig,
    seed: int | None = None,
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
    debater: DebaterConfig,
    interventions: list[Intervention],
) -> str:
    """Update a debater's personality based on the interventions passed as arguments.
    The debater configuration personality is updated in place.

    Returns the prompt used for the personality update."""

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
    mediator: MediatorConfig,
    summary: SummaryHandler,
    probability_mapper: ProbabilityMapper | None = None,
) -> tuple[LLMProbaMessage, str, bool]:
    """Mediator intervention: decision, motivation for the intervention, and intervention content.

    Returns:
        - The parsed LLM response.
        - The prompt used.
        - A boolean that determines if the mediator intervenes or not.
    """

    prompt = f"""{config.to_prompt()}. 

{summary.debaters_prompt()}


{summary.raw_history_prompt()} 

{mediator.to_prompt()}

{json_prompt(LLM_PROBA_RESPONSE_FORMAT)}
    """

    response = model.sample(prompt)
    parsed_response = parse_llm_json(response, LLMProbaMessage)

    p = parsed_response["do_intervene"]
    if probability_mapper is not None:
        p = probability_mapper.map(p)
    do_intervene = random.rand() < p

    return parsed_response, prompt, do_intervene


async def async_debater_interventions(
    model: AsyncLanguageModel,
    config: DebateConfig,
    summary: AsyncSummaryHandler,
    debaters: list[DebaterConfig],
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
    mediator: MediatorConfig,
    summary: AsyncSummaryHandler,
    valid_indexes: list[int] | None = None,
    retry_attempts: int = 5,
) -> tuple[list[LLMMessage], list[str]]:

    prompts: list[str] = []

    summary_prompts = summary.raw_history_prompts()

    if valid_indexes is not None:
        summary_prompts = [summary_prompts[i] for i in valid_indexes]

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


@retry(attempts=5, verbose=True)
async def async_debater_personality_update(
    model: AsyncLanguageModel,
    debaters: list[DebaterConfig],
    interventions: list[list[Intervention]],
) -> list[str]:
    """Update multiple debater personalities based on the respective interventions passed as arguments, asynchronously.
    The debater configuration personality is updated in place.

    Returns the prompts used for the personality updates."""

    assert len(debaters) == len(
        interventions
    ), "Debaters and interventions must have the same length."

    if len(debaters) == 0 or debaters[0].personalities is None:
        return [""] * len(debaters)

    sep = "\n"

    # Build the prompts for every debater variant
    prompts: list[str] = []

    for debater, debater_interventions in zip(debaters, interventions):
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
{sep.join([intervention.text for intervention in debater_interventions if intervention.text ])}

You can choose to evolve your personality on all axes by +1, -1 or 0.
{json_prompt(answer_format)}
"""

        prompts.append(prompt)

    responses = await model.sample(prompts)

    datas = [parse_llm_json(response) for response in responses]

    for data, debater in zip(datas, debaters):
        if debater.personalities is None:
            continue

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

    return prompts
