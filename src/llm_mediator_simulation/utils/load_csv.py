"""Helper to load data from a CSV file"""

import json
import re
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import TypeVar

import polars as pd

from llm_mediator_simulation.personalities.cognitive_biases import CognitiveBias
from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.facets import PersonalityFacet
from llm_mediator_simulation.personalities.fallacies import Fallacy
from llm_mediator_simulation.personalities.human_values import BasicHumanValues
from llm_mediator_simulation.personalities.ideologies import Ideology, Issues
from llm_mediator_simulation.personalities.moral_foundations import MoralFoundation
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import (
    KeyingDirection,
    Likert3Level,
    Likert5ImportanceLevel,
    Likert5Level,
    Likert7AgreementLevel,
    Scale,
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debater.config import (
    DebaterConfig,
    TopicOpinion,
)
from llm_mediator_simulation.utils.types import Intervention

T_enum = TypeVar("T_enum", bound=Enum)


def get_feature_from_str(
    enum: type[T_enum] | type[str], value_str: str, strict: bool = False
) -> T_enum | str | None:
    if issubclass(enum, str):
        # remove anything between parenthesis since 03 often write rationale / justifications in parenthesis
        value_str = re.sub(r"\(.*?\)", "", value_str)
        if (
            not value_str
            or "unknown" in value_str.lower()
            or "not applicable" in value_str.lower()
        ):
            return None
        return value_str.strip()
    else:
        members: list[T_enum] = list(enum)
        # enum_values = [m.value for m in members]
        enum_values = []
        for member in members:
            value_name = member.value
            if enum == Likert5Level:
                value_name = value_name.standard
            elif not enum == DemographicCharacteristic and not issubclass(enum, Scale):
                value_name = value_name.name

            # 1. replace spaces with underscores
            v = value_name.replace(" ", "_")
            value_str = value_str.replace(" ", "_")
            # 2. lowercase
            v = v.lower()
            enum_values.append(v)
        if value_str in enum_values:
            # Get the index of the scale value
            index = enum_values.index(value_str)
            return members[index]
        else:
            if strict:
                raise ValueError(f"Feature {value_str} not found in {enum.__name__}")

            return None


def load_reddit_csv_conv(
    path: str,
    truncated_num: int | None = 2,
    force_truncated_order: bool = False,
    load_debater_profiles: bool = False,
    debater_profiles_path: str | None = None,
    statement: str | None = None,
    prune_debaters: bool = True,
) -> tuple[list[DebaterConfig], list[Intervention], list[str] | None]:
    """Extract a list of debater configs and interventions from a chat CSV file.
    Args:
        path (str): Path to the CSV file.
        truncated_num (int | None): Number of messages to truncate from the end of the chat. Defaults to 2.
        force_truncated_order (bool): If True, forces the order of debaters to be the same as in the CSV file. Defaults to False.
        load_debater_profiles (bool): If True, loads debater profiles from a JSON file. Defaults to False.
        debater_profiles_path (str | None): Path to the debater profiles JSON file. Required if load_debater_profiles is True.
        statement (str | None): Statement to use for the debate. Defaults to None.
        prune_debaters (bool): If True, prunes the debater profiles and opinion to only include the relevant features. Defaults to True. Indeed
                               automatic profiling with LLM may fill uncertain features with placeholder values e.g., "average" value for a trait.

    Returns:
        tuple[list[DebaterConfig], list[Intervention], list[str] | None]: A tuple containing a list of debater configs, a list of interventions, and a list of forced debater order.
    """
    assert (
        not (load_debater_profiles) or debater_profiles_path is not None
    ), "If load_debater_profiles is True, debater_profiles_path must be set."

    assert (
        not (force_truncated_order) or truncated_num
    ), "If force_truncated_order is True, truncated_num must be set to a value greater than 0."

    df = pd.read_csv(path)

    if force_truncated_order:
        assert (
            truncated_num
        ), "If force_truncated_order is True, truncated_num must be set to a value greater than 0."
        forced_debater_order = df[-truncated_num:]["User Name"].to_list()
    else:
        forced_debater_order = None

    expected_columns = ["User ID", "User Name", "Text", "Timestamp"]

    if not all(col in df.columns for col in expected_columns):
        raise ValueError(
            f"Expected columns: {expected_columns} to be present in a chat CSV file"
        )

    # Extract users
    users = df.select("User ID", "User Name").unique(
        subset=["User ID"], maintain_order=True
    )
    # maintain_order = True enforces reproducibility.
    # Otherwise, the order of the users may vary hence the random order in the queries to debaters will not be correctly set by the random seed.
    users_id_map = dict(zip(users["User ID"], users["User Name"]))

    if load_debater_profiles:
        # Load debater profiles
        with open(debater_profiles_path, "r", encoding="utf-8") as f:
            debater_profiles = json.load(f)
    else:
        debater_profiles = None

    debaters: dict[int, DebaterConfig] = {}
    # ID to debater config
    for id, name in users_id_map.items():
        # id == name
        if name in ["[deleted]" or ["removed"]]:
            name = ""  # We assume that all messages written by deleted or removed users are written by the same user, which might not be always true...
        if load_debater_profiles and id in debater_profiles:
            assert (
                debater_profiles is not None
            ), "Debater profiles must be loaded if load_debater_profiles is True."
            # assert (
            #     id in debater_profiles
            # ), f"Debater profile for {id} not found in {debater_profiles_path}."

            debater_profile_loaded = debater_profiles[id]
            confidence = debater_profile_loaded.get("confidence", 10)
            submission_num = debater_profile_loaded.get("submission_num", 100)

            if confidence <= 1 or submission_num == 0:
                debaters[id] = DebaterConfig(name, identifier="username")

            else:
                # 0. replace blank spaces with underscores
                assert isinstance(statement, str), "statement must be a string"
                s = statement.replace(" ", "_")
                # 1. remove any special characters from the string parenthesis, brackets, single/double quotes, punctuation, etc.
                s = re.sub("[^A-Za-z0-9_]+", "", s)
                # 2. lowercase the string
                s = s.lower()

                debater_agreement_with_statement_dict: dict[str, str] | None = (
                    debater_profile_loaded.get("agreement_with_statement", None)
                )
                agreement = None
                if debater_agreement_with_statement_dict is not None:
                    debater_agreement_with_statement = (
                        debater_agreement_with_statement_dict.get(s, None)
                    )
                    if debater_agreement_with_statement is not None:
                        # Search in enum Likert7ag
                        agreement = get_feature_from_str(
                            Likert7AgreementLevel,
                            debater_agreement_with_statement,
                            strict=False,
                        )
                        assert isinstance(agreement, Likert7AgreementLevel)
                if agreement is not None:
                    topic_opinion = TopicOpinion(agreement=agreement)
                else:
                    topic_opinion = None

                profile_dict = defaultdict()

                for (
                    str_personality_features,
                    personality_features,
                    personality_values,
                ) in zip(
                    [
                        "demographics",
                        "traits",
                        "facets",
                        "moral_foundations",
                        "basic_human_values",
                        "cognitive_biases",
                        "fallacies",
                        "ideologies",
                        "vote_last_presidential_election",
                    ],
                    [
                        DemographicCharacteristic,
                        PersonalityTrait,
                        PersonalityFacet,
                        MoralFoundation,
                        BasicHumanValues,
                        CognitiveBias,
                        Fallacy,
                        Issues,
                        str,
                    ],
                    [
                        str,
                        Likert3Level,
                        KeyingDirection,
                        Likert5Level,
                        Likert5ImportanceLevel,
                        None,
                        None,
                        Ideology,
                        None,
                    ],
                ):
                    str_features_collection: dict[str, str] | list[str] | str | None = (
                        debater_profile_loaded.get(str_personality_features, None)
                    )
                    if str_features_collection is not None:
                        if isinstance(
                            str_features_collection, dict
                        ):  # Case of demographics, traits, facets, moral_foundations, basic_human_values, ideologies
                            features_collection = {}
                            assert not features_collection

                            for (
                                feature_str,
                                value_str,
                            ) in str_features_collection.items():
                                feature = get_feature_from_str(
                                    personality_features,
                                    feature_str,
                                    strict=True,
                                )
                                assert (
                                    feature not in features_collection
                                ), f"{str_personality_features} dict already contains {feature_str}"
                                assert personality_values is not None
                                value = get_feature_from_str(
                                    personality_values,
                                    value_str,
                                    strict=True,
                                )
                                if value:
                                    features_collection[feature] = value

                        elif isinstance(
                            str_features_collection, list
                        ):  # Case of cognitive_biases, fallacies
                            features_collection = []
                            assert not features_collection
                            for feature_str in str_features_collection:
                                feature = get_feature_from_str(
                                    personality_features,
                                    feature_str,
                                    strict=False,
                                )
                                assert (
                                    feature not in features_collection
                                ), f"{str_personality_features} list already contains {feature_str}"
                                if feature:
                                    features_collection.append(feature)

                        elif isinstance(str_features_collection, str):  # type: ignore ; Case of vote_last_presidential_election or if Ideology not specifying the issue
                            if (
                                str_personality_features
                                == "vote_last_presidential_election"
                            ):
                                features_collection = get_feature_from_str(
                                    str, str_features_collection
                                )
                            elif (
                                str_personality_features == "ideologies"
                            ):  # Case of ideology not specifying the issue
                                features_collection = get_feature_from_str(
                                    Ideology, str_features_collection
                                )
                            else:
                                raise ValueError(
                                    f"If {str_personality_features} is a string, it should correspond to a vote_last_presidential_election or an ideology"
                                )

                        else:
                            raise ValueError(
                                f"Invalid type for {str_personality_features}: {type(str_features_collection)}"
                            )

                        profile_dict[str_personality_features] = features_collection

                personality = Personality(
                    demographic_profile=(
                        profile_dict["demographics"]
                        if profile_dict["demographics"]
                        else None
                    ),
                    traits=profile_dict["traits"] if profile_dict["traits"] else None,
                    # facets=profile_dict["facets"] if profile_dict["facets"] else None,
                    moral_foundations=(
                        profile_dict["moral_foundations"]
                        if profile_dict["moral_foundations"]
                        else None
                    ),
                    basic_human_values=(
                        profile_dict["basic_human_values"]
                        if profile_dict["basic_human_values"]
                        else None
                    ),
                    cognitive_biases=(
                        profile_dict["cognitive_biases"]
                        if profile_dict["cognitive_biases"]
                        else None
                    ),
                    fallacies=(
                        profile_dict["fallacies"] if profile_dict["fallacies"] else None
                    ),
                    vote_last_presidential_election=(
                        profile_dict["vote_last_presidential_election"]
                        if profile_dict["vote_last_presidential_election"]
                        else None
                    ),
                    ideologies=(
                        profile_dict["ideologies"]
                        if profile_dict["ideologies"]
                        else None
                    ),
                )

                debater_config = DebaterConfig(
                    name,
                    topic_opinion=topic_opinion,
                    personality=personality,
                    identifier="username",
                )

                if prune_debaters:
                    debater_config.prune()

                debaters[id] = debater_config

        else:
            debaters[id] = DebaterConfig(name, identifier="username")

    if truncated_num is not None:
        # Remove the last n rows of the dataframe
        df = df[:-truncated_num]

    # Extract interventions
    app_interventions = df.select("User ID", "Text", "Timestamp")

    date_format = "%a %b %d %Y (%H:%M)"
    interventions = [
        Intervention(
            debater=(
                debaters[app_interventions["User ID"][i]]
                if app_interventions["User ID"][i] != ""
                else None
            ),
            text=re.sub(
                r">\s*(.*?)\n", r"You said: '\1'\n", app_interventions["Text"][i]
            ),  # An LLM might not understand that the > symbol is used to quote a message. See sub 4lq5n0 for a good example
            prompt="",
            justification="",
            timestamp=datetime.strptime(
                datetime.fromtimestamp(app_interventions["Timestamp"][i]).strftime(
                    date_format
                ),
                date_format,
            ),
        )
        for i in range(len(app_interventions))
    ]

    return list(debaters.values()), interventions, forced_debater_order


def load_deliberate_lab_csv_chat(
    path: str,
) -> tuple[list[DebaterConfig], list[Intervention], None]:
    """Extract a list of debater configs and interventions from a chat CSV file.
    Note that no personalities and debate positions can be inferred for debaters.
    Positions default to FOR.
    """

    df = pd.read_csv(path)

    # Check for columns
    expected_columns = ["User ID", "User Name", "Message Type", "Text", "Timestamp"]

    if not all(col in df.columns for col in expected_columns):
        raise ValueError(
            f"Expected columns: {expected_columns} to be present in a chat CSV file"
        )

    # Extract users
    users = df.select("User ID", "User Name").unique(subset=["User ID"])
    users_id_map = dict(zip(users["User ID"], users["User Name"]))
    users_id_map[""] = "Mediator"

    # ID to debater config
    debaters: dict[str, DebaterConfig] = {
        id: DebaterConfig(
            name,
            topic_opinion=None,
        )
        for id, name in users_id_map.items()
        if name != "Mediator"
    }

    # Extract interventions
    app_interventions = df.select("Message Type", "User ID", "Text", "Timestamp")

    date_format = "%a %b %d %Y (%H:%M)"
    interventions = [
        Intervention(
            debater=(
                debaters[app_interventions["User ID"][i]]
                if app_interventions["User ID"][i] != ""
                else None
            ),
            text=app_interventions["Text"][i],
            prompt="",
            justification="",
            timestamp=datetime.strptime(app_interventions["Timestamp"][i], date_format),
        )
        for i in range(len(app_interventions))
    ]

    return list(debaters.values()), interventions, None
