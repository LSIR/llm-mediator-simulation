"""Prompt utilities for the debate simulation."""

from llm_mediator_simulations.models.language_model import LanguageModel
from llm_mediator_simulations.simulation.configuration import (
    DebateConfig,
    DebatePosition,
    Debater,
)
from llm_mediator_simulations.simulation.summary import Summary
from llm_mediator_simulations.utils.model_utils import ask_closed_question

###################################################################################################
#                                     INTERVENTION DECISION                                       #
###################################################################################################


def should_participant_intervene(
    model: LanguageModel,
    config: DebateConfig,
    summary: Summary,
    debater: Debater,
) -> bool:
    """Decide whether to intervene in the debate.

    Args:
        model (LanguageModel): The language model to use.
        config (DebateConfig): The debate configuration.
        debater (Debater): The debater instance.
    """

    msg_sep = "\n\n"

    # Closed question prompt
    prompt = f"""{config.context} {config.statement}. You are arguing 
    {'in favor of' if debater.position == DebatePosition.FOR else 'against'} the statement.
    
    Your personality is {', '.join(map(lambda x: x.value, debater.personality or []))}.
    Here is a summary of the last exchanges (if empty, the conversation just started):
    {summary.summary}

    Here are the last messages exchanged (you should focus your argumentation on them):
    {msg_sep.join(summary.latest_messages)}

    Do you want to add a comment to the online debate right now?
    You should often add a comment when the previous context is empty or not in the favor of your \
    position. However, you should almost never add a comment when the previous context already \
    supports your position.
    """
    return ask_closed_question(model, prompt)


def should_mediator_intervene(
    model: LanguageModel,
    config: DebateConfig,
):
    """Decide whether to intervene in the debate.

    Args:
        model (LanguageModel): The language model to use.
        config (DebateConfig): The debate configuration.
    """

    raise NotImplementedError("Mediator intervention decision is not yet implemented.")


###################################################################################################
#                                         COMMENT PROMPT                                          #
###################################################################################################


def debater_comment(
    model: LanguageModel,
    config: DebateConfig,
    debater: Debater,
    summary: Summary,
) -> str:
    """Prompt a debater to add a comment to the debate."""

    # Prepare the prompt.
    msg_sep = "\n\n"
    prompt = f"""{config.context} {config.statement}. You are arguing 
    {'in favor of' if debater.position == DebatePosition.FOR else 'against'} the statement.
    {config.instructions}
    
    Your personality is {', '.join(map(lambda x: x.value, debater.personality or []))}.

    Here is a summary of the last exchanges (if empty, the conversation just started):
    {summary.summary}

    Here are the last messages exchanged (you should focus your argumentation on them):
    {msg_sep.join(summary.latest_messages)}
    """

    return model.sample(prompt)