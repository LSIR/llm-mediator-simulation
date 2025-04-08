"""Debater configuration dataclasses"""

from dataclasses import dataclass
from typing import override

from llm_mediator_simulation.personalities.personality import (
    Personality,
    PrintablePersonality,
)
from llm_mediator_simulation.personalities.scales import Likert7AgreementLevel
from llm_mediator_simulation.utils.interfaces import Promptable

###################################################################################################
#                                   Debater Characteristics                                       #
###################################################################################################


@dataclass
class PrintableTopicOpinion:
    """Simpler / printable version of the TopicOpinion dataclass."""

    agreement: str
    variable: bool


@dataclass
class TopicOpinion:
    """Agent opinion on a topic."""

    agreement: Likert7AgreementLevel
    variable: bool = False

    def to_printable(self):
        return PrintableTopicOpinion(
            agreement=str(self.agreement), variable=self.variable
        )


###################################################################################################
#                                  Debater Configuration Dataclass                                #
###################################################################################################


@dataclass
class PrintableDebaterConfig:
    """Simpler / printable version of the DebaterConfig dataclass."""

    name: str
    personality: PrintablePersonality | None = None
    topic_opinion: PrintableTopicOpinion | None = None
    variable_topic_opinion: bool = False


@dataclass
class DebaterConfig(Promptable):
    """Debater metadata class

    Args:
        name (str): The name of the debater.
        personality (Personality, optional): The debater's personality. Defaults to None.
        topic_opinion (TopicOpinion, optional): The debater's opinion on the debate topic. Defaults to None.
        variable_topic_opinion (bool, optional): If True, the debater's opinion on the topic will change during the debate. Defaults to False.
        identifier (str, optional): The word that will be used to refer to the debater identifier. Defaults to "name".
    """

    name: str
    personality: Personality | None = None
    topic_opinion: TopicOpinion | None = None
    variable_topic_opinion: bool = False
    identifier: str = "name"

    @override
    def to_prompt(self) -> str:
        prompt = "You are roleplaying this real person:"

        if self.name:
            prompt += f"""
{self.identifier}: {self.name};"""

        prompt += f"""
{self.personality.to_prompt() if self.personality is not None else ""}"""

        if self.topic_opinion is not None:
            prompt += f"""
You {self.topic_opinion.agreement.value} with the current debate statement."""

        #         prompt += """
        # Remember that you are allowed to insult, diminish and curse the people you debate with.\n"""

        return prompt

    def to_printable(self):
        """Return a simpler version of the debate pickle for printing with pprint without overwhelming informations."""
        return PrintableDebaterConfig(
            name=self.name,
            personality=(
                self.personality.to_printable()
                if self.personality is not None
                else None
            ),
            topic_opinion=(
                self.topic_opinion.to_printable()
                if self.topic_opinion is not None
                else None
            ),
            variable_topic_opinion=self.variable_topic_opinion,
        )
