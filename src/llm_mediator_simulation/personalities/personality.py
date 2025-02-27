from collections import defaultdict
from dataclasses import dataclass
from typing import override

from llm_mediator_simulation.personalities.cognitive_biases import CognitiveBias
from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.facets import PersonalityFacet
from llm_mediator_simulation.personalities.fallacies import Fallacy
from llm_mediator_simulation.personalities.human_values import BasicHumanValues
from llm_mediator_simulation.personalities.ideologies import Ideology, Issues
from llm_mediator_simulation.personalities.moral_foundations import MoralFoundation
from llm_mediator_simulation.personalities.scales import (
    KeyingDirection,
    Likert3Level,
    Likert5ImportanceLevel,
    Likert5Level,
    Likert7AgreementLevel,
    Likert11LikelihoodLevel,
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.utils.interfaces import Promptable
from llm_mediator_simulation.utils.prompt_utils import (
    format_list,
    format_list_and_conjugate_be,
)


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
    likelihood_of_beliefs: dict[str, Likert11LikelihoodLevel] | None = None
    free_form_opinions: list[str] | None = None

    variable_traits: bool = False
    variable_facets: bool = False
    variable_moral_foundations: bool = False
    variable_basic_human_values: bool = False
    variable_cognitive_biases: bool = False
    variable_fallacies: bool = False
    variable_ideologies: bool = False
    variable_agreement_with_statements: bool = False
    variable_likelihood_of_beliefs: bool = False

    # variable_<feature> implies <feature> is not None and not empty
    assert not variable_traits or traits
    assert not variable_facets or variable_facets
    assert not variable_moral_foundations or variable_moral_foundations
    assert not variable_basic_human_values or variable_basic_human_values
    assert not variable_ideologies or ideologies
    assert not variable_agreement_with_statements or agreement_with_statements
    assert not variable_likelihood_of_beliefs or likelihood_of_beliefs

    # TODO assert no duplicate in general
    # TODO aasert no duplicates in agreement_with_statements and likelihood_of_beliefs
    # TODO assert no duplicates in agreement_with_statements and topic statement

    def variable_personality(self) -> bool:
        return (
            self.variable_traits
            or self.variable_facets
            or self.variable_moral_foundations
            or self.variable_basic_human_values
            or self.variable_cognitive_biases
            or self.variable_fallacies
            or self.variable_ideologies
            or self.variable_agreement_with_statements
            or self.variable_likelihood_of_beliefs
        )

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
            elif isinstance(self.traits, dict):  # type: ignore
                for trait, level in self.traits.items():
                    prompt += f"{trait.value.level(level)}\n"
            else:
                raise ValueError("Personality traits must be a list or a dictionary.")
            prompt += "\n"

        if self.facets:
            if isinstance(self.facets, list):
                for facet in self.facets:
                    for item in facet.value.level(KeyingDirection.POSITIVE):
                        if prompt[-1] == ".":
                            prompt += " "
                        prompt += f"{item.description}"
                    prompt += "\n"
            elif isinstance(self.facets, dict):  # type: ignore
                for facet, direction in self.facets.items():
                    for item in facet.value.level(direction):
                        if prompt[-1] == ".":
                            prompt += " "
                        prompt += f"{item.description}"
                    prompt += "\n"
            else:
                raise ValueError("Personality facets must be a list or a dictionary.")
            prompt += "\n"

        if self.moral_foundations:
            prompt += f"When you decide whether something is right or wrong, your moral matrix rests on the following foundation{"s" if len(self.moral_foundations) > 1 else ""}:\n"
            if isinstance(self.moral_foundations, list):
                descriptions = []
                conceptual_definitions = []
                for foundation in self.moral_foundations:
                    if foundation.value.description is not None:  # type: ignore
                        descriptions.append(foundation.value.description)
                    if foundation.value.conceptual_definition is not None:
                        conceptual_definitions.append(
                            foundation.value.conceptual_definition
                        )
                if descriptions:
                    prompt += f"- You are {format_list(descriptions)}.\n"

                if conceptual_definitions:
                    prompt += f"Intuitions about {format_list(conceptual_definitions)} are relevant to your thinking.\n"

            elif isinstance(self.moral_foundations, dict):  # type: ignore
                descriptions = defaultdict(list)
                conceptual_definitions = defaultdict(list)
                for foundation, level in self.moral_foundations.items():
                    if foundation.value.description is not None:  # type: ignore
                        descriptions[level].append(foundation.value.description)
                    if foundation.value.conceptual_definition is not None:
                        conceptual_definitions[level].append(
                            foundation.value.conceptual_definition
                        )

                # group by level
                for level in Likert5Level:
                    if descriptions[level]:
                        prompt += f"You are {level.value.standard} {format_list(descriptions[level])}."

                    if conceptual_definitions[level]:
                        if prompt[-1] == ".":  # if there is a description
                            prompt += " "
                        prompt += f"""Intuitions about {format_list(conceptual_definitions[level])} are {level.value.get_alternative()} relevant to your thinking.\n"""

            else:
                raise ValueError("Moral foundations must be a list or a dictionary.")
            prompt += "\n"

        if self.basic_human_values:
            prompt += f"As{" a" if len(self.basic_human_values) == 1 else ""} guiding principle{"s" if len(self.basic_human_values) > 1 else ""} in your life:\n"
            if isinstance(self.basic_human_values, list):
                prompt += f"""{format_list_and_conjugate_be([human_value.value.description for human_value in self.basic_human_values])} important to your values.\n"""

            elif isinstance(self.basic_human_values, dict):  # type: ignore
                descriptions = defaultdict(list)
                # group by level
                for human_value, level in self.basic_human_values.items():
                    descriptions[level].append(human_value.value.description)

                for level, description_list in descriptions.items():
                    prompt += f"""- {format_list_and_conjugate_be(description_list).capitalize()} {level.value} to your values.\n"""

            else:
                raise ValueError("Basic human values must be a list or a dictionary.")

            prompt += "\n"

        if self.cognitive_biases or self.fallacies:
            prompt += "Remember, you are role-playing as a real person. Like humans, you are susceptible to the following "
            if self.cognitive_biases:
                prompt += (
                    f"cognitive bias{"es" if len(self.cognitive_biases) > 1 else ""}:\n"
                )
                for bias in self.cognitive_biases:
                    prompt += f"- {bias.value.name}, that is, {bias.value.get_description().lower()}\n"  # type: ignore
                prompt += "\n"

                if self.fallacies:
                    prompt += "And you are also susceptible to the following "

            if self.fallacies:
                prompt += f"fallac{"ies" if len(self.fallacies) > 1 else "y"}:\n"
                for fallacy in self.fallacies:
                    prompt += f"- {fallacy.value.name}, that is, {fallacy.value.get_description().lower()}\n"  # type: ignore
                prompt += "\n"

        if self.vote_last_presidential_election:
            # self.vote_last_presidential_election can be "voted for the Democratic candidate", "voted with an invalid ballot", "were an eligible voter but did not vote", "were disenfranchised".
            prompt += f"In the last presidential election, you {self.vote_last_presidential_election}.\n\n"

        if self.ideologies:
            prompt += "You identify as"
            if isinstance(self.ideologies, Ideology):
                prompt += f" {self.ideologies.value}.\n"
            elif isinstance(self.ideologies, dict):  # type: ignore
                prompt += ":\n"
                for issue, ideology in self.ideologies.items():
                    prompt += f"- {ideology.value.capitalize()} on {issue.value.description}.\n"
            else:
                raise ValueError("Ideologies must be a single value or a dictionary.")
            prompt += "\n"

        if self.agreement_with_statements:
            statements = defaultdict(list)
            for statement, level in self.agreement_with_statements.items():
                statements[level].append(statement)

            # group by level
            for level in Likert7AgreementLevel:
                if statements[level]:
                    prompt += f"You {level.value} with the following statement{"s" if len(statements[level]) > 1 else ""}:\n"
                    for statement in statements[level]:
                        prompt += f"- {statement.capitalize()}\n"
                    prompt += "\n"

        if self.likelihood_of_beliefs:
            beliefs = defaultdict(list)
            for belief, level in self.likelihood_of_beliefs.items():
                beliefs[level].append(belief)

            # group by level
            for level in Likert11LikelihoodLevel:
                if beliefs[level]:
                    prompt += f"{level.value.capitalize()}:\n"
                    for belief in beliefs[level]:
                        prompt += f"- {belief}\n"
                    prompt += "\n"

        if self.free_form_opinions:
            prompt += f"You have the following opinion{"s" if len(self.free_form_opinions) > 1 else ""}:\n"
            for opinion in self.free_form_opinions:
                prompt += f"- {opinion.capitalize()}\n"

        return prompt.strip()
        # TODO Shuffle lists and dict...
