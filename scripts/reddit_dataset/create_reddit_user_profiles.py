import asyncio
import json
import os
import textwrap
from collections import defaultdict
from enum import Enum
from typing import Literal, Type

import openai
import pandas as pd
import tiktoken
from BAScraper.BAScraper_async import ArcticShiftAsync
from pydantic import BaseModel, Field, create_model
from rich.pretty import pprint

from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.facets import PersonalityFacet
from llm_mediator_simulation.personalities.human_values import BasicHumanValues
from llm_mediator_simulation.personalities.ideologies import Ideology, Issues
from llm_mediator_simulation.personalities.moral_foundations import MoralFoundation
from llm_mediator_simulation.personalities.scales import (
    KeyingDirection,
    Likert3Level,
    Likert5ImportanceLevel,
    Likert7AgreementLevel,
    Scale,
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait

openai.api_key = os.getenv("GPT_API_KEY")
DATAPATH = "data/reddit/cmv"


def is_exploitable(text: str) -> bool:
    """
    Check if the text is exploitable or not.
    :param text: The text to check.
    :return: True if the text is exploitable, False otherwise.
    """
    return text not in {"", "[deleted]", "[removed]"}


# Build the BaseModel automatically
def _build_model(
    it: Type[Enum] | list[str], model_name: str, scale: Type[Scale] | None = None
) -> Type[BaseModel]:
    """
    Turn Enum members into fields of a pydantic model.

    • Field name  : enum.value   (or enum.name if you prefer)
    • Field type  : Optional[str] (change to anything you like)
    • Field title : enum.name    – useful for OpenAPI generation
    """
    fields = {}
    for member in it:
        if isinstance(it, list):  # Statements
            assert isinstance(member, str), "Statements must be a list of strings"
            m = member
            s = scale
            f = Field(title=member)
        elif model_name == "Demographics":
            assert isinstance(member, Enum)
            m = member.value.replace(" ", "_")
            s = str
            f = Field(title=member.name.replace("_", " ").title())
        else:
            assert isinstance(member, Enum)
            m = member.value.name.replace(" ", "_")
            s = scale
            f = Field(title=member.name.replace("_", " ").title())

        fields[m] = (s, f)

    # if isinstance(it, list):
    #     assert model_name == "Statements", "Only Statements can be a list"
    #     fields = {
    #         member: (  # field name
    #             scale,  # type
    #             Field(title=member.title()),
    #         )  # Field(...)
    #         for member in it
    #     }
    # else:
    #     if model_name == "Demographics":
    #         fields = {
    #             member.value.replace(" ", "_"): (  # field name
    #                 str,  # type
    #                 Field(title=member.name.replace("_", " ").title()),
    #             )  # Field(...)
    #             for member in it
    #         }
    #     else:
    #         fields = {
    #             member.value.name.replace(" ", "_"): (  # field name
    #                 scale,  # type
    #                 Field(title=member.name.replace("_", " ").title()),
    #             )  # Field(...)
    #             for member in it
    #         }

    # Same as writing: class Demographics(BaseModel): age: Optional[str] = None ...
    return create_model(model_name, **fields)  # <- subclass of BaseModel


class Likert5LevelBis(Scale):
    # If we use Likert5Level,
    # the schema itself is syntactically valid JSON, but it violates the (draft-07) JSON-Schema rules that OpenAI’s validator applies to the
    # json_schema field so we need to recreate the class without the alternative values...
    """Level on a 5-point likert scale axis. Based on the Moral Foundation Questionnaire-2 in yourmorals.org."""

    NOT_AT_ALL = "not at all"
    SLIGHTLY = "slightly"
    MODERATELY = "moderately"
    FAIRLY = "fairly"
    EXTREMELY = "extremely"


Demographics = _build_model(DemographicCharacteristic, model_name="Demographics")
Traits = _build_model(PersonalityTrait, model_name="Traits", scale=Likert3Level)
Facets = _build_model(
    PersonalityFacet, model_name="Facets", scale=KeyingDirection
)  # <- subclass of BaseModel
MoralFoundations = _build_model(
    MoralFoundation, model_name="MoralFoundations", scale=Likert5LevelBis
)  # <- subclass of BaseModel
BasicHumanValues = _build_model(
    BasicHumanValues, model_name="BasicHumanValues", scale=Likert5ImportanceLevel
)  # <- subclass of BaseModel
Issues = _build_model(
    Issues, model_name="Issues", scale=Ideology
)  # <- subclass of BaseModel


def call_openai(messages, Profile):
    # while True:

    # pprint(response_schema)

    resp = openai.beta.chat.completions.parse(
        model="o3", messages=messages, response_format=Profile
    )
    return resp.choices[0].message.content
    # except openai.
    #     print("Rate-limit hit; sleeping", delay, "s")
    #     time.sleep(delay)
    #     delay = min(delay * 2, 60)
    # except openai.error.OpenAIError as e:
    #     print("OpenAI error:", e)
    #     time.sleep(delay)


def count_tokens(text: str, model: str = "o3") -> int:
    """Return the number of tokens in a plain-text prompt."""
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def assert_fits_context(
    text: str, model: str = "o3", max_completion_tokens: int = 0
) -> None:
    """
    Assert that `text` + the number of tokens you still intend to generate
    (`max_completion_tokens`) fits into the model’s context window.
    """
    used = count_tokens(text, model)
    if used + max_completion_tokens > 200000:
        raise ValueError(
            f"Request would use {used + max_completion_tokens} tokens, "
            f"exceeding the {200000}-token window "
            f"of model {model}."
        )


async def main():
    # maintain a list of the number of submissions per user
    user_submissions_count = defaultdict(int)
    user_counter = 0
    # Iterate over all user_ids
    for user_id, statement_list in USERID_TO_STATEMENTS.items():
        if not is_exploitable(user_id):
            print(f"User {user_id} is not exploitable")
            continue
        # if user_counter > 2:
        #     os._exit(0)
        # get all submissions from user_id
        user_submissions = await asa.fetch(
            mode="submissions_search", author=user_id, limit=101
        )  # limit max 100 https://github.com/maxjo020418/BAScraper/blob/cd5c2ef24f45f66e7f0fb26570c2c1529706a93f/README.md?plain=1#L239

        user_flairs_aggregation = await asa.fetch(
            mode="user_flairs_aggregation", author=user_id
        )  # no max limit https://github.com/maxjo020418/BAScraper/blob/cd5c2ef24f45f66e7f0fb26570c2c1529706a93f/README.md?plain=1#L342

        submissions_txt = []
        for submission in user_submissions.values():
            # Append the string of the json with the subreddit, title, and selftext
            subreddit = submission["subreddit"]
            title = submission["title"]
            selftext = submission["selftext"]

            if is_exploitable(title) or is_exploitable(selftext):
                assert is_exploitable(subreddit), "Subreddit is deleted or removed"
                submission_txt = f"{{subreddit: {subreddit}, "
                if is_exploitable(title):
                    submission_txt += f"title: {title}, "
                if is_exploitable(selftext):
                    submission_txt += f"selftext: {selftext}}}"
                else:
                    submission_txt = submission_txt[:-2] + "}"

                submissions_txt.append(submission_txt)
                user_submissions_count[user_id] += 1

        corpus = "\n\n-----\n\n".join(submissions_txt)
        assert_fits_context(
            corpus,
            model="o3",
            max_completion_tokens=2000,
        )
        if user_flairs_aggregation:
            flairs = str(user_flairs_aggregation)
        else:
            flairs = "unknown"

        # statement_json = ", ".join(
        #     [f'"{statement}": 1-100' for statement in statement_list]
        # )

        # system_prompt = textwrap.dedent(
        #     f"""
        # You are an expert computational social scientist.
        # Using the Reddit flairs and posts provided, infer the author’s profile with the
        # following JSON schema; fill "unknown" if evidence is insufficient:

        # {{
        # "demographics": {{ "ethnicity": "", "biological_sex": "", "gender_identity": "", "nationality": "", "age": 0-100, "marital_status": "", "education": "", "occupation": "", "political_leaning": "", "religion": "", "sexual_orientation": "", "health_condition": "", "income": "", "household_size": 0-20, "number_of_dependent": 0-10, "living_quarters": "", "language_spoken": "", "city_of_residence": "", "primary_mode_of_transportation": "", "background": "" }},
        # "traits": {{ "openness": 1-3, "conscientiousness": 1-3, "extraversion": 1-3, "agreeableness": 1-3, "neuroticism": 1-3 }},
        # "facets": {{ "anxiety": 0-1, "anger": 0-1, "depression": 0-1, "self_consciousness": 0-1, "immoderation": 0-1, "vulnerability": 0-1,
        #             "friendliness": 0-1, "gregariousness": 0-1, "assertiveness": 0-1, "activity_level": 0-1, "excitement_seeking": 0-1, "cheerfulness": 0-1,
        #             "imagination": 0-1, "artistic_interests": 0-1, "emotionality": 0-1, "adventurousness": 0-1, "intellect": 0-1, "liberalism": 0-1,
        #             "trust": 0-1, "morality": 0-1, "altruism": 0-1, "cooperation": 0-1, "modesty": 0-1, "sympathy": 0-1,
        #             "self_efficacy": 0-1, "orderliness": 0-1, "dutifulness": 0-1, "achievement_striving": 0-1, "self_discipline": 0-1, "cautiousness": 0-1 }},
        # "moral_foundations": {{ "care": 1-5, "fairness": 1-5, "loyalty": 1-5, "authority": 1-5, "liberty": 1-5 }},
        # "basic_human_values": {{ "self_direction_thought": 1-5, "self_direction_action": 1-5, "stimulation": 1-5, "hedonism": 1-5, "achievement": 1-5, "power_dominance": 1-5, "power_resources": 1-5, "face": 1-5, "security_personal": 1-5, "security_societal": 1-5, "tradition": 1-5, "conformity_rules": 1-5, "conformity_interpersonal": 1-5, "humility": 1-5, "benevolenc_dependability": 1-5, "benevolenc_caring": 1-5, "universalism_concern": 1-5, "universalism_nature": 1-5, "universalism_tolerance": 1-5 }},
        # "cognitive_biases": [ "bias1", "bias2", ... ],
        # "fallacies": [ "fallacy1", "fallacy2", ... ],
        # "vote_last_presidential_election": "",
        # "ideologies": {{ "economic": "", "social": "", ... }},
        # "agreement_with_statement": {{ {statement_json} }},
        # "confidence": 0-100
        # }}

        # Ground your judgments only on text given; do not hallucinate private data.
        # Answer with JSON ONLY.
        # """
        # ).strip()

        system_prompt = textwrap.dedent(
            """
        You are an expert computational social scientist.
        Using the Reddit flairs and posts provided, infer the author’s profile into the given structure; fill "unknown" if evidence is insufficient:
        Ground your judgments only on text given; do not hallucinate private data.
        """
        ).strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Here are the flairs of user {user_id}:\n{flairs}\n\nHere are the Reddit posts of user {user_id}:\n\n{corpus}""",
            },
        ]

        Statements = _build_model(
            statement_list, model_name="Statements", scale=Likert7AgreementLevel
        )

        class Profile(BaseModel):
            """
            Class to represent a user profile.
            """

            demographics: Demographics
            traits: Traits
            facets: Facets
            moral_foundations: MoralFoundations
            basic_human_values: BasicHumanValues
            cognitive_biases: list[str]  # list[CognitiveBias]
            fallacies: list[str]  # list[Fallacy]
            vote_last_presidential_election: str
            ideologies: Issues
            agreement_with_statement: Statements
            confidence: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # schema = Profile.model_json_schema(by_alias=True)

        json_response = call_openai(messages, Profile)
        profile = json.loads(json_response)
        profile["submission_num"] = user_submissions_count[user_id]
        assert user_id not in profiles, f"User {user_id} already exists in profiles"
        profiles[user_id] = profile
        user_counter += 1
        print(f"Processed {user_counter} users")
        pprint(profile)
        # os._exit(0)


if __name__ == "__main__":
    # ppa = PullPushAsync(log_stream_level="DEBUG", task_num=1) # PullPush Scheduled Maintenance
    # The Pullpush API will be temporarily unavailable until mid-May due to essential hardware upgrades and a full reindexing process.

    profiles = {}

    with open(os.path.join(DATAPATH, "statements.json"), encoding="utf-8") as f:
        statements = json.load(f)

    # Load the user_ids that will be simulated
    datapath = DATAPATH
    USERID_TO_STATEMENTS = defaultdict(list[str])
    filenames = list(os.listdir(datapath))
    filenames.sort()

    for filename in filenames:
        if filename.endswith(".csv"):
            submission_id = filename.split("-")[0].split("_")[1]
            statement = statements[submission_id]
            # open the csv file
            df = pd.read_csv(os.path.join(datapath, filename))
            # get the last 2 user ids from the conversation
            for user_id in df["User ID"][-2:]:
                USERID_TO_STATEMENTS[user_id].append(statement)

    # Remove duplicates but keep the order
    # userid_to_statement = list(dict.fromkeys(userid_to_statement))
    print(f"Found {len(USERID_TO_STATEMENTS)} unique user ids")
    asa = ArcticShiftAsync(log_stream_level="DEBUG", task_num=16, pace_mode="auto-soft")
    asyncio.run(main())

    # Save the profiles to a json file at datapath
    with open(os.path.join(DATAPATH, "reddit_user_profiles.json"), "w") as f:
        json.dump(profiles, f, indent=4)

    # TODO Anonimize, Shuffle, and keep the mapping secret
