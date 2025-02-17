from collections import defaultdict
from dataclasses import dataclass
from typing import override

from llm_mediator_simulation.personalities.cognitive_biases import CognitiveBias
from llm_mediator_simulation.personalities.facets import PersonalityFacet
from llm_mediator_simulation.personalities.fallacies import Fallacy
from llm_mediator_simulation.personalities.human_values import BasicHumanValues
from llm_mediator_simulation.personalities.moral_foundations import MoralFoundation
from llm_mediator_simulation.personalities.ideologies import Ideology, Issues
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.scales import (
    KeyingDirection,
    Likert11LikelihoodLevel,
    Likert3Level,
    Likert5ImportanceLevel,
    Likert5Level,
    Likert7AgreementLevel,
)
from llm_mediator_simulation.utils.interfaces import Promptable


@dataclass
class Personality(Promptable):
    """Personality of an agent."""

    demographic_profile: dict[DemographicCharacteristic, str] | None = None
    traits: dict[PersonalityTrait, Likert3Level] | list[PersonalityTrait] | None = None
    facets: dict[PersonalityFacet, KeyingDirection] | list[PersonalityFacet] | None = (
        None
    )
    moral_foundations: (
        dict[MoralFoundation, Likert5Level] | list[MoralFoundation] | None
    ) = None
    basic_human_values: dict[BasicHumanValues, Likert5ImportanceLevel] | None = None
    cognitive_biases: list[CognitiveBias] | None = None
    fallacies: list[Fallacy] | None = None
    vote_last_presidential_election: str | None = None
    ideologies: dict[Issues, Ideology] | Ideology | None = None
    agreement_with_statements: dict[str, Likert7AgreementLevel] | None = None
    likelihood_of_beliefs: list[str, Likert11LikelihoodLevel] | None = None
    free_form_opinions: list[str] | None = None

    variable_traits: bool = False
    variable_facets: bool = False
    variable_moral_foundations: bool = False
    variable_basic_human_values: bool = False
    variable_cognitive_biases: bool = (
        False  # TODO: If True, randomly sample cognitive biases
    )
    variable_fallacies: bool = False  # TODO: If True, randomly sample fallacies
    variable_ideologies: bool = False
    variable_agreement_with_statements: bool = False
    variable_likelihood_of_beliefs: bool = False
    variable_free_form_opinions: bool = False

    @override
    def to_prompt(self) -> str:
        prompt = ""
        if self.demographic_profile:
            for characteristic, value in self.demographic_profile.items():
                prompt += f"{characteristic.value}: {value.lower()};\n"
            prompt += "\n"

        if self.traits:
            if isinstance(self.traits, list):
                for trait in self.traits:
                    prompt += f"{trait.value.level(level=Likert3Level.HIGH)}\n"
            elif isinstance(self.traits, dict):
                for trait, level in self.traits.items():
                    prompt += f"{trait.value.level(level)}\n"
            else:
                raise ValueError("Invalid traits type")
            prompt += "\n"

        if self.facets:
            if isinstance(self.facets, list):
                for facet in self.facets:
                    for item in facet.value.level(KeyingDirection.POSITIVE):
                        if prompt[-1] == ".":
                            prompt += " "
                        prompt += f"{item.description}"
                    prompt += "\n"
            elif isinstance(self.facets, dict):
                for facet, direction in self.facets.items():
                    for item in facet.value.level(direction):
                        if prompt[-1] == ".":
                            prompt += " "
                        prompt += f"{item.description}"
                    prompt += "\n"
            else:
                raise ValueError("Invalid facets type")
            prompt += "\n"

        if self.moral_foundations:
            prompt += "When you decide whether something is right or wrong, your moral matrix rests on the following foundations:\n"
            if isinstance(self.moral_foundations, list):
                descriptions = []
                conceptual_definitions = []
                for foundation in self.moral_foundations:
                    if foundation.value.description is not None:
                        descriptions.append(foundation.value.description)
                    if foundation.value.conceptual_definition is not None:
                        conceptual_definitions.append(
                            foundation.value.conceptual_definition
                        )
                if descriptions:
                    prompt += f"- You are {self.format_list(descriptions)}.\n"

                if conceptual_definitions:
                    prompt += f"Intuitions about {self.format_list(conceptual_definitions)} are relevant to your thinking.\n"

            elif isinstance(self.moral_foundations, dict):
                descriptions = defaultdict(list)
                conceptual_definitions = defaultdict(list)
                for foundation, level in self.moral_foundations.items():
                    if foundation.value.description is not None:
                        descriptions[level].append(foundation.value.description)
                    if foundation.value.conceptual_definition is not None:
                        conceptual_definitions[level].append(
                            foundation.value.conceptual_definition
                        )

                # group by level
                for level in Likert5Level:
                    if descriptions[level]:
                        prompt += f"You are {level.value.standard} {self.format_list(descriptions[level])}."

                    if conceptual_definitions[level]:
                        if prompt[-1] == ".":  # if there is a description
                            prompt += " "
                        prompt += f"""Intuitions about {self.format_list(conceptual_definitions[level])} 
are {level.value.get_alternative()} relevant to your thinking.\n"""

            else:
                raise ValueError("Invalid moral foundations type")
            prompt += "\n"

        if self.basic_human_values:
            prompt += "As a guiding principle in your life:\n"
            if isinstance(self.basic_human_values, list):
                prompt += f"""{self.format_list_and_conjugate_be([human_value.value.description for human_value in self.basic_human_values])} 
important to your values.\n"""

            elif isinstance(self.basic_human_values, dict):
                descriptions = defaultdict(list)
                # group by level
                for human_value, level in self.basic_human_values.items():
                    descriptions[level].append(human_value.value.description)

                for level, description_list in descriptions.items():
                    prompt += f"""- {self.format_list_and_conjugate_be(description_list).capitalize()} 
{level.value} to your values.\n"""

            else:
                raise ValueError("Invalid basic human values type")

            prompt += "\n"

        if self.cognitive_biases or self.fallacies:
            prompt += "Remember, you are role-playing as a real person. Like humans, you are susceptible to the following "
            if self.cognitive_biases:
                prompt += "cognitive biases:\n"
                for bias in self.cognitive_biases:
                    prompt += f"- {bias.name}, that is, {bias.description.lower()}\n"
                prompt += "\n"

                if self.fallacies:
                    prompt += "And you are also susceptible to the following "

            if self.fallacies:
                prompt += "fallacies:\n"
                for fallacy in self.fallacies:
                    prompt += (
                        f"- {fallacy.name}, that is, {fallacy.description.lower()}\n"
                    )
                prompt += "\n"

        if self.vote_last_presidential_election:
            # self.vote_last_presidential_election can be "voted for the Democratic candidate", "voted with an invalid ballot", "were an eligible voter but did not vote", "were disenfranchised".
            prompt += f"In the last presidential election, you {self.vote_last_presidential_election}.\n\n"

        if self.ideologies:
            prompt += "You identify as"
            if isinstance(self.ideologies, Ideology):
                prompt += f" {self.ideologies.value}.\n"
            elif isinstance(self.ideologies, dict):
                prompt += ":\n"
                for issue, ideology in self.ideologies.items():
                    prompt += f"- {ideology.value.capitalize()} on {issue.value.description}.\n"
            else:
                raise ValueError("Invalid ideologies type")
            prompt += "\n"

        return prompt
        # TODO Shuffle lists and dict...
        # TODO agreement_with_statements, likelihood_of_beliefs, free_form_opinions

    def conjugate_be(self, str_list: list[str]) -> str:
        if len(str_list) > 1:
            return "are"
        else:  # len(str_list) == 1
            return "is"

    def format_list(self, str_list: list[str]) -> str:
        """Format a list of strings into a sentence."""
        if len(str_list) > 2:
            return f"{', '.join(str_list[:-1])}, and {str_list[-1]}"
        elif len(str_list) == 2:
            return f"{str_list[0]} and {str_list[1]}"
        elif len(str_list) == 1:
            return f"{str_list[0]}"

    def format_list_and_conjugate_be(self, str_list: list[str]) -> str:
        return f"{self.format_list(str_list)} {self.conjugate_be(str_list)}"
