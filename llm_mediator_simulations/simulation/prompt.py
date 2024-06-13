"""Prompt utilities for the debate simulation."""

from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.simulation.configuration import (
    DebateConfig,
    Debater,
    Mediator,
)
from llm_mediator_simulations.simulation.summary_handler import SummaryHandler
from llm_mediator_simulations.utils.model_utils import ask_closed_question

###################################################################################################
#                                     INTERVENTION DECISION                                       #
###################################################################################################


def should_participant_intervene(
    model: LanguageModel,
    config: DebateConfig,
    summary: SummaryHandler,
    debater: Debater,
) -> bool:
    """Decide whether to intervene in the debate.

    Args:
        model (LanguageModel): The language model to use.
        config (DebateConfig): The debate configuration.
        debater (Debater): The debater instance.
    """

    # Closed question prompt
    prompt = f"""{config.to_prompt()}. {debater.to_prompt()} 

    {summary.to_prompt()}

    Do you want to add a comment to the online debate right now?
    You should often add a comment when the previous context is empty or not in the favor of your \
    position. However, you should almost never add a comment when the previous context already \
    supports your position.
    """
    return ask_closed_question(model, prompt)


def should_mediator_intervene(
    model: LanguageModel,
    config: DebateConfig,
    mediator: Mediator,
    summary: SummaryHandler,
):
    """Decide whether to intervene in the debate.

    Args:
        model (LanguageModel): The language model to use.
        config (DebateConfig): The debate configuration.
    """

    # Closed question prompt
    prompt = f"""{config.to_prompt()}. 

    {summary.to_prompt()}

    {mediator.to_prompt()}
    {mediator.detection_instructions}
    """

    return ask_closed_question(model, prompt)


###################################################################################################
#                                         COMMENT PROMPT                                          #
###################################################################################################


def debater_comment(
    model: LanguageModel,
    config: DebateConfig,
    debater: Debater,
    summary: SummaryHandler,
) -> str:
    """Prompt a debater to add a comment to the debate."""

    # Prepare the prompt.
    prompt = f"""{config.to_prompt()}. {debater.to_prompt()} 
    {config.instructions}
    
    {summary.to_prompt()}
    """

    return model.sample(prompt)


def mediator_comment(
    model: LanguageModel,
    config: DebateConfig,
    mediator: Mediator,
    summary: SummaryHandler,
) -> str:
    """Prompt a mediator to add a comment to the debate."""

    # Prepare the prompt.
    prompt = f"""{config.to_prompt()}.
    {config.instructions}
    
    {summary.to_prompt()}
    
    {mediator.to_prompt()}
    {mediator.intervention_instructions}
    """

    return model.sample(prompt)
