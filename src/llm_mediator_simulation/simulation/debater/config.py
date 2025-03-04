"""Debater configuration dataclasses"""

from dataclasses import dataclass
from typing import override

from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import Likert7AgreementLevel
from llm_mediator_simulation.utils.interfaces import Promptable

###################################################################################################
#                                   Debater Characteristics                                       #
###################################################################################################


@dataclass
class TopicOpinion:
    """Agent opinion on a topic."""

    agreement: Likert7AgreementLevel
    variable: bool = False


###################################################################################################
#                                  Debater Configuration Dataclass                                #
###################################################################################################


@dataclass
class DebaterConfig(Promptable):
    """Debater metadata class

    Args:
        name (str): The name of the debater.
        personality (Personality, optional): The debater's personality. Defaults to None.
        topic_opinion (TopicOpinion, optional): The debater's opinion on the debate topic. Defaults to None.
        variable_topic_opinion (bool, optional): If True, the debater's opinion on the topic will change during the debate. Defaults to False.
    """

    name: str
    personality: Personality | None = None
    topic_opinion: TopicOpinion | None = None
    variable_topic_opinion: bool = False

    @override
    def to_prompt(self) -> str:
        return f"""You are roleplaying this real person: 
        name: {self.name}; 
        {self.personality.to_prompt() if self.personality is not None else ""}
        
        You {self.topic_opinion.agreement.value if self.topic_opinion is not None else Likert7AgreementLevel.NEUTRAL.value} with the current debate statement.
        Remember that you are allowed to insult, diminish and curse the people you debate with."""  # TODO Remove the last sentence
