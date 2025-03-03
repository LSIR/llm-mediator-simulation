"""Prompt utilities for the debate simulation."""

from random import randint, sample, shuffle
from typing import Literal, Sequence, Type, TypeVar, cast

from numpy import random

from llm_mediator_simulation.models.language_model import (
    AsyncLanguageModel,
    LanguageModel,
)
from llm_mediator_simulation.personalities.cognitive_biases import (
    CognitiveBias,
    ReasoningError,
)
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
    Scale,
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import (
    DebaterConfig,
    TopicOpinion,
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
from llm_mediator_simulation.utils.probabilities import ProbabilityMapper
from llm_mediator_simulation.utils.prompt_utils import format_list
from llm_mediator_simulation.utils.types import (
    Intervention,
    LLMMessage,
    LLMProbaMessage,
)

T_scale = TypeVar("T_scale", bound=Scale)
T_resonning_error = TypeVar("T_resonning_error", bound=ReasoningError)

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
"""  # TODO review these instructions

    response = model.sample(prompt, seed=seed)
    return parse_llm_json(response, LLMMessage), prompt


@retry(attempts=5, verbose=True)
def debater_update(
    model: LanguageModel,
    debate_statement: str,
    debater: DebaterConfig,
    interventions: list[Intervention],
) -> str:
    """Update a debater's topic opinion and personality based on the interventions passed as arguments.
    The debater configuration topic opinion and personality are updated in place.

    Returns the prompt used for the personality update."""
    if not debater.variable_topic_opinion:
        if debater.personality is None or not (
            debater.personality.variable_personality()
        ):
            return ""

    ################################################
    # Build the prompt for the debater's current personality
    ################################################
    prompt = f"""You are taking part in an online debate about the following topic: {debate_statement}

You are roleplaying this real person:
name: {debater.name};\n"""
    personality = debater.personality
    if personality is not None:
        # Not shuffled
        if personality.demographic_profile is not None:
            for (
                characteristic,
                value,
            ) in personality.demographic_profile.items():
                prompt += f"{characteristic.value}: {value.lower()};\n"
            prompt += "\n"

        prompt += """Here is your current personality:\n"""

        # Not shuffled
        if personality.vote_last_presidential_election:
            prompt += f"In the last presidential election, you {personality.vote_last_presidential_election}.\n\n"

        # Shuffled
        if personality.traits:
            traits = personality.traits.copy()
            prompt += f"Trait{"s" if len(traits) > 1 else ""}:\n"
            if isinstance(traits, list):
                shuffle(traits)
                for trait in traits:
                    prompt += f"- {trait.value.name.capitalize()}: {Likert3Level.HIGH.value}\n"
            elif isinstance(traits, dict):  # type: ignore
                traits_and_values = list(traits.items())
                shuffle(traits_and_values)
                for trait, value in traits_and_values:
                    prompt += f"- {trait.value.name.capitalize()}: {value.value}\n"
            else:
                raise ValueError("Personality traits must be a list or a dictionary.")
            prompt += "\n"

        # Shuffled
        if personality.facets:
            facets = personality.facets.copy()
            prompt += f"Facet{"s" if len(facets) > 1 else ""}:\n"
            if isinstance(facets, list):
                shuffle(facets)
                for facet in facets:
                    prompt += f"- {facet.value.name.capitalize()}: {KeyingDirection.POSITIVE.value}\n"
            elif isinstance(facets, dict):  # type: ignore
                facets_and_values = list(facets.items())
                shuffle(facets_and_values)
                for facet, value in facets_and_values:
                    prompt += f"- {facet.value.name.capitalize()}: {value.value}\n"
            else:
                raise ValueError("Personality facets must be a list or a dictionary.")
            prompt += "\n"

        # Shuffled
        if personality.moral_foundations:
            moral_foundations = personality.moral_foundations.copy()
            prompt += f"Moral foundation{"s" if len(moral_foundations) > 1 else ""}:\n"
            if isinstance(moral_foundations, list):
                shuffle(moral_foundations)
                for foundation in moral_foundations:
                    prompt += f"- {foundation.value.name.capitalize()}: {Likert5Level.EXTREMELY.value.standard}\n"
            elif isinstance(personality.moral_foundations, dict):  # type: ignore
                moral_foundations_and_values = list(moral_foundations.items())
                shuffle(moral_foundations_and_values)
                for foundation, value in moral_foundations_and_values:
                    prompt += f"- {foundation.value.name.capitalize().split(" (v1) ")[0]}: {value.value.standard}\n"
            else:
                raise ValueError("Moral foundations must be a list or a dictionary.")
            prompt += "\n"

        # Shuffled
        if personality.basic_human_values:
            basic_human_values = personality.basic_human_values.copy()
            prompt += (
                f"Basic human value{"s" if len(basic_human_values) > 1 else ""}:\n"
            )
            if isinstance(basic_human_values, list):
                shuffle(basic_human_values)
                for value in basic_human_values:
                    prompt += f"- {value.value.name.capitalize()}: {Likert5ImportanceLevel.IMPORTANT.value}\n"
            elif isinstance(basic_human_values, dict):  # type: ignore
                basic_human_values_and_level = list(basic_human_values.items())
                shuffle(basic_human_values_and_level)
                for value, level in basic_human_values_and_level:
                    prompt += f"- {value.value.name.capitalize()}: {level.value}\n"
            else:
                raise ValueError("Basic human values must be a list or a dictionary.")
            prompt += "\n"

        # Shuffled
        if personality.cognitive_biases:
            cognitive_biases = personality.cognitive_biases.copy()
            shuffle(cognitive_biases)
            prompt += f"Cognitive bias{"es" if len(cognitive_biases) > 1 else ""}:\n"
            for bias in cognitive_biases:
                prompt += f"- {bias.value.name.capitalize()}\n"
            prompt += "\n"

        # Shuffled
        if personality.fallacies:
            fallacies = personality.fallacies.copy()
            shuffle(fallacies)
            prompt += f"Fallac{"y" if len(fallacies) > 1 else "ies"}:\n"
            for fallacy in fallacies:
                prompt += f"- {fallacy.value.name.capitalize()}\n"
            prompt += "\n"

        # Shuffled iif ideologies per issue
        if personality.ideologies:
            if isinstance(personality.ideologies, Ideology):
                prompt += f"Ideology: {personality.ideologies.value}\n"
            elif isinstance(personality.ideologies, dict):  # type: ignore
                issues_and_ideologies = list(personality.ideologies.items())
                shuffle(issues_and_ideologies)
                prompt += (
                    f"Ideolog{"y" if len(personality.ideologies) > 1 else "ies"}:\n"
                )
                for issue, ideology in issues_and_ideologies:
                    prompt += f"- {issue.value.name.capitalize()}: {ideology.value}\n"
            else:
                raise ValueError("Ideologies must be a single value or a dictionary.")
            prompt += "\n"

    if personality and personality.agreement_with_statements:
        prompt += "Statements:\n"
    else:
        prompt += "Statement:\n"
    prompt += f"- {debate_statement.capitalize().rstrip(".")} (current debate statement): {debater.topic_opinion.agreement.value if debater.topic_opinion is not None else Likert7AgreementLevel.NEUTRAL.value}\n"

    # Shuffled
    if personality is not None and personality.agreement_with_statements:
        agreement_with_statements = list(personality.agreement_with_statements.items())
        shuffle(agreement_with_statements)
        for (
            statement,
            agreement,
        ) in agreement_with_statements:
            prompt += f"- {statement.capitalize()}: {agreement.value}\n"
    prompt += "\n"

    if personality is not None:
        # Shuffled
        if personality.likelihood_of_beliefs:
            likelihood_of_beliefs = list(personality.likelihood_of_beliefs.items())
            shuffle(likelihood_of_beliefs)
            prompt += f"Belief{"s" if len(likelihood_of_beliefs) > 1 else ""}:\n"
            for belief, likelihood in likelihood_of_beliefs:
                prompt += (
                    f"- {belief.capitalize()}: {likelihood.name.replace("_", " ")}\n"
                )
            prompt += "\n"

        # Shuffled
        if personality.free_form_opinions:
            free_form_opinions = personality.free_form_opinions.copy()
            shuffle(free_form_opinions)
            prompt += f"You have the following opinion{"s" if len(free_form_opinions) else ""}:\n"
            for opinion in free_form_opinions:
                prompt += f"- {opinion.capitalize()}\n"
            prompt += "\n"

    ################################################
    # Build the prompt for the debate's last interventions
    ################################################

    prompt += """You have the opportunity to make your personality evolve based on the things people have said after your last intervention.\n\n"""

    prompt += """Here are the last messages:\n"""
    for intervention in interventions:
        if intervention.debater is None:
            name = "Mediator"
        else:
            name = intervention.debater.name
        prompt += f"""— {name}: {intervention.text}\n"""  # https://en.wikipedia.org/wiki/Quotation_mark#Quotation_dash
    prompt += "\n"

    ################################################
    # Build the prompt for the debater's personality update instructions and format
    ################################################

    answer_format: dict[str, str] = dict()
    more_less_feature_to_evolve_list = []
    more_less_text_in_answer_format = """a string ("{0}", "{1}", or "{2}") to update"""
    update_options_yes_no = [
        "more",
        "less",
        "same",
    ]  # The order is shuffle each time to avoid bias for one specific option on average...
    if personality is not None:
        if personality.variable_traits and personality.traits:
            more_less_feature_to_evolve_list.append("traits")
            update_options_trait = update_options_yes_no.copy()
            traits = list(personality.traits.copy())
            shuffle(traits)
            for trait in traits:
                shuffle(update_options_trait)
                answer_format["_".join(trait.value.name.split(" "))] = (
                    f"""{more_less_text_in_answer_format.format(*update_options_trait)} this trait"""
                )

        if personality.variable_moral_foundations and personality.moral_foundations:
            more_less_feature_to_evolve_list.append("moral foundations")
            update_options_moral_foundation = update_options_yes_no.copy()
            moral_foundations = list(personality.moral_foundations.copy())
            shuffle(moral_foundations)
            for foundation in moral_foundations:
                shuffle(update_options_moral_foundation)
                answer_format["_".join(foundation.value.name.split(" "))] = (
                    f"""{more_less_text_in_answer_format.format(*update_options_moral_foundation)} this moral foundation"""
                )

        if personality.variable_basic_human_values and personality.basic_human_values:
            more_less_feature_to_evolve_list.append("basic human values")
            update_options_moral_basic_human_values = update_options_yes_no.copy()
            basic_human_values = list(personality.basic_human_values.copy())
            shuffle(basic_human_values)
            for value in basic_human_values:
                shuffle(update_options_moral_basic_human_values)
                answer_format["_".join(value.value.name.split(" "))] = (
                    f"""{more_less_text_in_answer_format.format(*update_options_moral_basic_human_values)} this basic human value"""
                )

    if debater.variable_topic_opinion:
        more_less_feature_to_evolve_list.append("agreement with statements")
        update_options_topic_opinion = update_options_yes_no.copy()
        shuffle(update_options_topic_opinion)
        answer_format.update(
            {
                "current_dabate_statement": f"""{more_less_text_in_answer_format.format(*update_options_topic_opinion)} your agreement with the current debate statement"""
            }
        )

    elif personality is not None:
        if (
            personality.variable_agreement_with_statements
            and personality.agreement_with_statements
        ):
            more_less_feature_to_evolve_list.append("agreement with statements")
            update_options_agreement_with_statements = update_options_yes_no.copy()
            agreement_with_statements = list(
                personality.agreement_with_statements.copy()
            )
            shuffle(agreement_with_statements)
            for debate_statement in agreement_with_statements:
                shuffle(update_options_agreement_with_statements)
                answer_format["_".join(debate_statement.lower().split(" "))] = (
                    f"""{more_less_text_in_answer_format.format(*update_options_agreement_with_statements)} your agreement with this statement"""
                )

    if personality is not None:
        if (
            personality.variable_likelihood_of_beliefs
            and personality.likelihood_of_beliefs
        ):
            more_less_feature_to_evolve_list.append("likelihood of beliefs")
            update_options_likelihood_of_beliefs = update_options_yes_no.copy()
            likelihood_of_beliefs = list(personality.likelihood_of_beliefs.copy())
            shuffle(likelihood_of_beliefs)
            for belief in likelihood_of_beliefs:
                shuffle(update_options_likelihood_of_beliefs)
                answer_format["_".join(belief.lower().split(" "))] = (
                    f"""{more_less_text_in_answer_format.format(*update_options_likelihood_of_beliefs)} your assessment of this belief's likelihood"""
                )

    if more_less_feature_to_evolve_list:
        update_options_more_less_same = update_options_yes_no.copy()
        shuffle(update_options_more_less_same)
        prompt += "You can choose to evolve your "
        prompt += format_list(more_less_feature_to_evolve_list)
        prompt += """ with "{0}", "{1}", or "{2}".\n""".format(
            *update_options_more_less_same
        )

    if personality is not None:
        if personality.variable_facets and personality.facets:
            update_options_yes_no = ["yes", "no"]
            shuffle(update_options_yes_no)
            prompt += """You can choose to evolve your facets with "{0}" or "{1}".\n""".format(
                *update_options_yes_no
            )

            facets = list(personality.facets.copy())
            shuffle(facets)
            for facet in facets:
                shuffle(update_options_yes_no)
                answer_format["_".join(facet.value.name.split(" "))] = (
                    """a string ("{0}" or "{1}") to update this facet"""
                ).format(*update_options_yes_no)

        if personality.variable_ideologies and personality.ideologies:
            prompt += """You can choose to evolve your """
            update_options_liberal_conservative = [
                "more liberal",
                "more conservative",
                "same",
                "libertarian",
            ]
            shuffle(update_options_liberal_conservative)
            if isinstance(personality.ideologies, Ideology):
                prompt += """ideology """
                answer_format.update(
                    {
                        "ideology": """a string ("{0}", "{1}", "{2}", or "{3}") to update your ideology""".format(
                            *update_options_liberal_conservative
                        )
                    }
                )
            elif isinstance(personality.ideologies, dict):  # type: ignore
                prompt += """ideologies """
                ideologies = list(personality.ideologies.copy())
                shuffle(ideologies)
                for issue in ideologies:
                    shuffle(update_options_liberal_conservative)
                    answer_format["_".join(issue.value.name.split(" "))] = (
                        """a string ("{0}", "{1}", "{2}", or "{3}") to update your ideology on this issue""".format(
                            *update_options_liberal_conservative
                        )
                    )

            else:
                raise ValueError("Ideologies must be a single value or a dictionary.")
            shuffle(update_options_liberal_conservative)
            prompt += """with "{0}", "{1}", "{2}", or "{3}".\n""".format(
                *update_options_liberal_conservative
            )
    prompt += "\n"
    prompt += f"""{json_prompt(answer_format)}"""

    # If only cognitive bias or fallacies can evolve, then no need for an LLM call since it's purely based on random sampling
    if debater.variable_topic_opinion or (
        personality is not None
        and any(
            [
                personality.variable_traits,
                personality.variable_facets,
                personality.variable_moral_foundations,
                personality.variable_basic_human_values,
                personality.variable_ideologies,
                personality.variable_agreement_with_statements,
                personality.variable_likelihood_of_beliefs,
            ]
        )
    ):
        response = model.sample(prompt)
        data: dict[str, str] = parse_llm_json(response)

        ################################################
        # Process the personality updates
        ################################################
        feature_key_to_personality_field_name: dict[
            str, Literal["trait", "facet", "moral_foundation", "basic_human_value"]
        ] = {}
        feature_key_to_feature: dict[str, Scale] = {}
        for feature_type, personality_field_name in [
            (PersonalityTrait, "traits"),
            (PersonalityFacet, "facets"),
            (MoralFoundation, "moral_foundations"),
            (BasicHumanValues, "basic_human_values"),
        ]:
            for feature in list(feature_type):
                feature_key_to_personality_field_name[
                    "_".join(feature.value.name.split(" "))
                ] = personality_field_name
                feature_key_to_feature["_".join(feature.value.name.split(" "))] = (
                    feature
                )

        statement_name_to_statement: dict[str, str] = {}
        belief_name_to_belief: dict[str, str] = {}

        if personality is not None:
            if (
                personality.variable_agreement_with_statements
                and personality.agreement_with_statements
            ):
                statement_name_to_statement.update(
                    {
                        "_".join(statement.lower().split(" ")): statement
                        for statement in personality.agreement_with_statements
                    }
                )
            if (
                personality.variable_likelihood_of_beliefs
                and personality.likelihood_of_beliefs
            ):
                belief_name_to_belief.update(
                    {
                        "_".join(belief.lower().split(" ")): belief
                        for belief in personality.likelihood_of_beliefs
                    }
                )

        for feature_key, update in data.items():
            if feature_key == "current_debate_statement":
                if debater.variable_topic_opinion:
                    if debater.topic_opinion is not None:
                        previous_value = debater.topic_opinion.agreement
                        updated_value = update_to_scale_value(previous_value, update)
                        debater.topic_opinion.agreement = updated_value
                    else:
                        debater.topic_opinion = TopicOpinion(
                            agreement=update_to_scale_value(
                                Likert7AgreementLevel.NEUTRAL, update
                            )
                        )
                else:
                    raise ValueError(
                        "Agent wants to update its topic opinon but the debater's topic opinion is not variable."
                    )
            if (
                feature_key in feature_key_to_personality_field_name
            ):  # e.g. feature_key = "openness_to_experience" and update = "more"
                # update the personality field with the update
                personality_field_name = feature_key_to_personality_field_name[
                    feature_key
                ]  # e.g. personality_field = "traits"
                # Check that the personality field was set in the previous personality
                if personality is not None:
                    if getattr(
                        personality, f"variable_{personality_field_name}"
                    ):  # Check that e.g. personality.variable_traits is True
                        if (
                            getattr(personality, personality_field_name) is not None
                        ):  # Check that e.g. personality.traits is set
                            personality_field = getattr(
                                personality, personality_field_name
                            )
                            if (
                                feature_key_to_feature[feature_key] in personality_field
                            ):  # Check that e.g. PersonalityTrait.OPENNESS is in personality.traits
                                if isinstance(personality_field, list):
                                    # Transform the list (e.g. debater.personality.traits = [Personality.OPENNESS, Personality.CONSCIENTIOUSNESS])
                                    # into a dictionary with default values (e.g. {Personality.OPENNESS: Likert3Level.AVERAGE, Personality.CONSCIENTIOUSNESS: Likert3Level.AVERAGE])
                                    if personality_field_name == "trait":
                                        new_personality_field = {
                                            feature: update_to_scale_value(
                                                Likert3Level.HIGH, update
                                            )
                                            for feature in personality_field
                                        }
                                    elif personality_field_name == "facet":
                                        new_personality_field = {
                                            feature: update_to_scale_value(
                                                KeyingDirection.POSITIVE, update
                                            )
                                            for feature in personality_field
                                        }
                                    elif personality_field_name == "moral_foundation":
                                        new_personality_field = {
                                            feature: update_to_scale_value(
                                                Likert5Level.FAIRLY, update
                                            )
                                            for feature in personality_field
                                        }
                                    elif personality_field_name == "basic_human_value":
                                        new_personality_field = {
                                            feature: update_to_scale_value(
                                                Likert5ImportanceLevel.IMPORTANT, update
                                            )
                                            for feature in personality_field
                                        }
                                    else:
                                        raise ValueError(
                                            f"Invalid personality field {personality_field_name}."
                                        )
                                    setattr(
                                        personality,
                                        personality_field_name,
                                        new_personality_field,
                                    )

                                elif isinstance(personality_field, dict):
                                    previous_value = personality_field[
                                        feature_key_to_feature[feature_key]
                                    ]
                                    updated_value = update_to_scale_value(
                                        previous_value, update
                                    )
                                    personality_field[
                                        feature_key_to_feature[feature_key]
                                    ] = updated_value

                                else:
                                    raise ValueError(
                                        f"Personality field {personality_field_name} must be a list or a dictionary."
                                    )
                            else:
                                raise ValueError(
                                    f"Feature {feature_key} is not set in the previous personality's {personality_field_name}."
                                )
                        else:
                            raise ValueError(
                                f"Personality field {personality_field_name} is not set in the previous personality."
                            )
                    else:
                        raise ValueError(
                            f"Personality field {personality_field_name} is not variable in the previous personality."
                        )
                else:
                    raise ValueError(
                        "Personality is not set in the previous personality."
                    )

            elif feature_key == "ideology":
                if personality is not None:
                    if personality.variable_ideologies:
                        if isinstance(personality.ideologies, Ideology):
                            new_ideology = update_to_ideology_value(
                                personality.ideologies, update
                            )
                            personality.ideologies = new_ideology
                        else:
                            raise ValueError(
                                "Agent tries to update single ideology while the personality field ideologies is breakdown into ideologies related to specific issues."
                            )
                    else:
                        raise ValueError(
                            "Personality field ideologies is not variable in the previous personality"
                        )
                else:
                    raise ValueError(
                        "Personality is not set in the previous personality."
                    )

            elif feature_key in [
                "_".join(issue.value.name.split(" ")) for issue in Issues
            ]:
                if personality is not None:
                    if personality.variable_ideologies:
                        if isinstance(personality.ideologies, dict):
                            issue_name_to_issue = {
                                "_".join(issue.value.name.split(" ")): issue
                                for issue in Issues
                            }
                            issue = issue_name_to_issue[feature_key]
                            if issue in personality.ideologies:
                                new_ideology = update_to_ideology_value(
                                    personality.ideologies[issue], update
                                )
                                personality.ideologies[issue] = new_ideology
                            else:
                                raise ValueError(
                                    f"Issue {issue} is not set in the previous personality's ideologies."
                                )
                        else:
                            raise ValueError(
                                "Agent tries to update ideologies related to aspecific issues while the personality field ideology is summarized into a single ideology."
                            )
                    else:
                        raise ValueError(
                            "Personality field ideologies is not variable in the previous personality."
                        )
                else:
                    raise ValueError(
                        "Personality is not set in the previous personality."
                    )

            elif personality is not None:
                if (
                    personality.variable_agreement_with_statements
                    and personality.agreement_with_statements
                ):
                    if feature_key in statement_name_to_statement:
                        statement = statement_name_to_statement[feature_key]
                        assert statement in personality.agreement_with_statements
                        previous_value = personality.agreement_with_statements[
                            statement
                        ]
                        updated_value = update_to_scale_value(previous_value, update)
                        personality.agreement_with_statements[statement] = updated_value
                if (
                    personality.variable_likelihood_of_beliefs
                    and personality.likelihood_of_beliefs
                ):
                    if feature_key in belief_name_to_belief:
                        belief = belief_name_to_belief[feature_key]
                        assert belief in personality.likelihood_of_beliefs
                        previous_value = personality.likelihood_of_beliefs[belief]
                        updated_value = update_to_scale_value(previous_value, update)
                        personality.likelihood_of_beliefs[belief] = updated_value
            else:
                raise ValueError(f"Feature {feature_key} is not valid.")

    if personality is not None:
        if personality.variable_cognitive_biases:
            previous_biases = personality.cognitive_biases
            new_biases = update_feature_list_randomly(
                cast(Sequence[ReasoningError] | None, previous_biases), CognitiveBias
            )
            if previous_biases is not None and new_biases:
                personality.cognitive_biases = cast(list[CognitiveBias], new_biases)

        if personality.variable_fallacies:
            previous_fallacies = personality.fallacies
            new_fallacies = update_feature_list_randomly(
                cast(Sequence[ReasoningError], previous_fallacies), Fallacy
            )
            if previous_fallacies is not None and new_fallacies:
                personality.fallacies = cast(list[Fallacy], new_fallacies)

    return prompt


def update_to_scale_value(previous_value: T_scale, update: str) -> T_scale:
    """Update a Likert scale or a binary value based on the update string.
    e.g. update_to_value(Likert3Level.AVERAGE, "more") -> Likert3Level.HIGH"""
    members: list[Scale] = list(type(previous_value))
    if update == "more" or update == "yes":
        return cast(
            T_scale, members[min(members.index(previous_value) + 1, len(members) - 1)]
        )
    elif update == "less" or update == "no":
        return cast(T_scale, members[max(members.index(previous_value) - 1, 0)])
    elif update == "same":
        return previous_value
    else:
        raise ValueError(f"Invalid update {update}.")


def update_to_ideology_value(
    previous_value: Ideology,
    update: str,
) -> Ideology:
    """Update an ideology based on the update string.
    e.g. update_to_value(Ideology.MODERATE, "more conservative") -> Ideology.SLIGHTLY_CONSERVATIVE
    """
    if update == "libertarian":
        return Ideology.LIBERTARIAN
    elif update == "more liberal":
        if previous_value in (Ideology.LIBERTARIAN, Ideology.MODERATE):
            return Ideology.SLIGHTLY_LIBERAL
        elif previous_value == Ideology.LIBERAL:
            return Ideology.EXTREMELY_LIBERAL
        elif previous_value == Ideology.SLIGHTLY_LIBERAL:
            return Ideology.LIBERAL
        elif previous_value == Ideology.SLIGHTLY_CONSERVATIVE:
            return Ideology.CONSERVATIVE
        elif previous_value == Ideology.CONSERVATIVE:
            return Ideology.SLIGHTLY_CONSERVATIVE
        elif previous_value == Ideology.EXTREMELY_CONSERVATIVE:
            return Ideology.CONSERVATIVE
        else:
            raise ValueError("Invalid ideology.")
    elif update == "more conservative":
        if previous_value in (Ideology.LIBERTARIAN, Ideology.MODERATE):
            return Ideology.SLIGHTLY_CONSERVATIVE
        elif previous_value == Ideology.CONSERVATIVE:
            return Ideology.EXTREMELY_CONSERVATIVE
        elif previous_value == Ideology.SLIGHTLY_CONSERVATIVE:
            return Ideology.CONSERVATIVE
        elif previous_value == Ideology.SLIGHTLY_LIBERAL:
            return Ideology.MODERATE
        elif previous_value == Ideology.LIBERAL:
            return Ideology.SLIGHTLY_LIBERAL
        elif previous_value == Ideology.EXTREMELY_LIBERAL:
            return Ideology.LIBERAL
        else:
            raise ValueError("Invalid ideology.")
    elif update == "same":
        return previous_value
    else:
        raise ValueError(f"Invalid update {update} for ideology.")


def update_feature_list_randomly(
    previous_reasoning_error_list: Sequence[T_resonning_error] | None,
    resonning_error: Type[CognitiveBias] | Type[Fallacy],
) -> Sequence[T_resonning_error]:
    """Randomly update a list of ReasoningError, where the ResoningError list is a CognitiveBias list or a Fallacy list."""
    # Randomly remove a random number of existing elements in the list
    if previous_reasoning_error_list:
        new_feature_list = sample(
            previous_reasoning_error_list,
            randint(0, len(previous_reasoning_error_list)),
        )
    else:
        new_feature_list = []
    # Add a random number between 0 and min(number of existing elements, 1) new elements
    previous_feature_list_num = (
        len(previous_reasoning_error_list) if previous_reasoning_error_list else 0
    )

    new_feature_list += sample(
        list(resonning_error),
        randint(0, min(previous_feature_list_num, 1)),
    )

    return cast(Sequence[T_resonning_error], new_feature_list)


@retry(attempts=5, verbose=True)
def mediator_intervention(
    model: LanguageModel,
    config: DebateConfig,
    mediator: MediatorConfig,
    summary: SummaryHandler,
    probability_mapper: ProbabilityMapper | None = None,
    seed: int | None = None,
) -> tuple[LLMProbaMessage, str, bool]:
    """Mediator intervention: decision, motivation for the intervention, and intervention content.

    Returns:
        - The parsed LLM response.
        - The prompt used.
        - A boolean that determines if the mediator intervenes or not.
    """

    prompt = f"""{config.to_prompt()}. 

{summary.debaters_prompt()}

CONVERSATION HISTORY WITH TIMESTAMPS:
{summary.raw_history_prompt()} 

{mediator.to_prompt()}

{json_prompt(LLM_PROBA_RESPONSE_FORMAT)}
    """

    response = model.sample(prompt, seed=seed)
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


# TODO Adapt to async
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
{sep.join([intervention.text for intervention in debater_interventions if intervention.text])}

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
