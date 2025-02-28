from collections import defaultdict
from dataclasses import dataclass
from random import shuffle
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

    # assert no duplicate in general
    assert len(set(traits)) == len(traits) if traits else True
    assert len(set(facets)) == len(facets) if facets else True
    assert (
        len(set(moral_foundations)) == len(moral_foundations)
        if moral_foundations
        else True
    )
    assert (
        len(set(basic_human_values)) == len(basic_human_values)
        if basic_human_values
        else True
    )
    assert (
        len(set(cognitive_biases)) == len(cognitive_biases)
        if cognitive_biases
        else True
    )
    assert len(set(fallacies)) == len(fallacies) if fallacies else True
    assert len(set(ideologies)) == len(ideologies) if ideologies else True
    assert (
        len(set(agreement_with_statements)) == len(agreement_with_statements)
        if agreement_with_statements
        else True
    )
    assert (
        len(set(likelihood_of_beliefs)) == len(likelihood_of_beliefs)
        if likelihood_of_beliefs
        else True
    )

    # assert no duplicates in agreement_with_statements and likelihood_of_beliefs
    if agreement_with_statements and likelihood_of_beliefs:
        assert len(set(agreement_with_statements) & set(likelihood_of_beliefs)) == 0

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
        # Not shuffled
        if self.demographic_profile:
            for characteristic, value in self.demographic_profile.items():
                prompt += f"{characteristic.value}: {value.lower()};\n"
            prompt += "\n"

        # Shuffled
        if self.traits:
            if isinstance(self.traits, list):
                traits = self.traits.copy()
                shuffle(traits)
                for trait in traits:
                    prompt += f"{trait.value.level(level=Likert3Level.HIGH)}\n"
            elif isinstance(self.traits, dict):  # type: ignore
                traits_and_levels = list(self.traits.items())
                shuffle(traits_and_levels)
                for trait, level in traits_and_levels:
                    prompt += f"{trait.value.level(level)}\n"
            else:
                raise ValueError("Personality traits must be a list or a dictionary.")
            prompt += "\n"

        # Shuffled
        if self.facets:
            if isinstance(self.facets, list):
                facets = self.facets.copy()
                shuffle(facets)
                for facet in facets:
                    for item in facet.value.level(KeyingDirection.POSITIVE):
                        if prompt[-1] == ".":
                            prompt += " "
                        prompt += f"{item.description}"
                    prompt += "\n"
            elif isinstance(self.facets, dict):  # type: ignore
                facets_and_directions = list(self.facets.items())
                shuffle(facets_and_directions)
                for facet, direction in facets_and_directions:
                    for item in facet.value.level(direction):
                        if prompt[-1] == ".":
                            prompt += " "
                        prompt += f"{item.description}"
                    prompt += "\n"
            else:
                raise ValueError("Personality facets must be a list or a dictionary.")
            prompt += "\n"

        # Shuffled
        if self.moral_foundations:
            prompt += f"When you decide whether something is right or wrong, your moral matrix rests on the following foundation{"s" if len(self.moral_foundations) > 1 else ""}:\n"
            if isinstance(self.moral_foundations, list):
                moral_foundations = self.moral_foundations.copy()
                shuffle(moral_foundations)
                descriptions = []
                conceptual_definitions = []
                for foundation in moral_foundations:
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
                moral_foundations_and_levels = list(self.moral_foundations.items())
                shuffle(moral_foundations_and_levels)
                descriptions = defaultdict(list)
                conceptual_definitions = defaultdict(list)
                for foundation, level in moral_foundations_and_levels:
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

        # Shuffled
        if self.basic_human_values:
            prompt += f"As{" a" if len(self.basic_human_values) == 1 else ""} guiding principle{"s" if len(self.basic_human_values) > 1 else ""} in your life:\n"
            if isinstance(self.basic_human_values, list):
                basic_human_values = self.basic_human_values.copy()
                assert isinstance(basic_human_values, list)
                shuffle(
                    basic_human_values
                )  # not sure to understand why ruff complains here without the assert...
                prompt += f"""{format_list_and_conjugate_be([human_value.value.description for human_value in basic_human_values])} important to your values.\n"""

            elif isinstance(self.basic_human_values, dict):  # type: ignore
                basic_human_values_and_levels = list(self.basic_human_values.items())
                shuffle(basic_human_values_and_levels)
                descriptions = defaultdict(list)
                # group by level
                for human_value, level in basic_human_values_and_levels:
                    descriptions[level].append(human_value.value.description)

                for level, description_list in descriptions.items():
                    prompt += f"""- {format_list_and_conjugate_be(description_list).capitalize()} {level.value} to your values.\n"""

            else:
                raise ValueError("Basic human values must be a list or a dictionary.")

            prompt += "\n"

        # Shuffled
        if self.cognitive_biases or self.fallacies:
            prompt += "Remember, you are role-playing as a real person. Like humans, you are susceptible to the following "
            if self.cognitive_biases:
                cognitive_biases = self.cognitive_biases.copy()
                shuffle(cognitive_biases)
                prompt += (
                    f"cognitive bias{"es" if len(cognitive_biases) > 1 else ""}:\n"
                )
                for bias in cognitive_biases:
                    prompt += f"- {bias.value.name}, that is, {bias.value.get_description().lower()}\n"  # type: ignore
                prompt += "\n"

                if self.fallacies:
                    prompt += "And you are also susceptible to the following "

            if self.fallacies:
                fallacies = self.fallacies.copy()
                shuffle(fallacies)
                prompt += f"fallac{"ies" if len(fallacies) > 1 else "y"}:\n"
                for fallacy in fallacies:
                    prompt += f"- {fallacy.value.name}, that is, {fallacy.value.get_description().lower()}\n"  # type: ignore
                prompt += "\n"

        # Not shuffled
        if self.vote_last_presidential_election:
            # self.vote_last_presidential_election can be "voted for the Democratic candidate", "voted with an invalid ballot", "were an eligible voter but did not vote", "were disenfranchised".
            prompt += f"In the last presidential election, you {self.vote_last_presidential_election}.\n\n"

        # Shuffled iif ideologies per issue
        if self.ideologies:
            prompt += "You identify as"
            if isinstance(self.ideologies, Ideology):
                prompt += f" {self.ideologies.value}.\n"
            elif isinstance(self.ideologies, dict):  # type: ignore
                issue_and_ideologies = list(self.ideologies.items())
                shuffle(issue_and_ideologies)
                prompt += ":\n"
                for issue, ideology in issue_and_ideologies:
                    prompt += f"- {ideology.value.capitalize()} on {issue.value.description}.\n"
            else:
                raise ValueError("Ideologies must be a single value or a dictionary.")
            prompt += "\n"

        # Shuffled
        if self.agreement_with_statements:
            agreement_with_statements = list(self.agreement_with_statements.items())
            shuffle(agreement_with_statements)
            statements = defaultdict(list)
            for statement, level in agreement_with_statements:
                statements[level].append(statement)

            # group by level
            for level in Likert7AgreementLevel:
                if statements[level]:
                    prompt += f"You {level.value} with the following statement{"s" if len(statements[level]) > 1 else ""}:\n"
                    for statement in statements[level]:
                        prompt += f"- {statement.capitalize()}\n"
                    prompt += "\n"

        # Shuffled
        if self.likelihood_of_beliefs:
            likelihood_of_beliefs = list(self.likelihood_of_beliefs.items())
            shuffle(likelihood_of_beliefs)
            beliefs = defaultdict(list)
            for belief, level in likelihood_of_beliefs:
                beliefs[level].append(belief)

            # group by level
            for level in Likert11LikelihoodLevel:
                if beliefs[level]:
                    prompt += f"{level.value.capitalize()}:\n"
                    for belief in beliefs[level]:
                        prompt += f"- {belief}\n"
                    prompt += "\n"

        if self.free_form_opinions:
            free_form_opinions = self.free_form_opinions.copy()
            shuffle(free_form_opinions)
            prompt += f"You have the following opinion{"s" if len(free_form_opinions) > 1 else ""}:\n"
            for opinion in free_form_opinions:
                prompt += f"- {opinion.capitalize()}\n"

        return prompt.strip()
