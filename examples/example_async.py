"""Example script to run parallel debate simulations on nuclear energy.

BUG: Google async models may raise an error when deallocated at the end due to this bug. It is harmless.
-> https://github.com/google-gemini/generative-ai-python/issues/207#issuecomment-2308952931
"""

import asyncio
import os

from dotenv import load_dotenv

from llm_mediator_simulation.metrics.async_metrics_handler import AsyncMetricsHandler
from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from llm_mediator_simulation.models.google_models import AsyncGoogleModel
from llm_mediator_simulation.personalities.cognitive_biases import CognitiveBias
from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.facets import PersonalityFacet
from llm_mediator_simulation.personalities.fallacies import Fallacy
from llm_mediator_simulation.personalities.human_values import BasicHumanValues
from llm_mediator_simulation.personalities.ideologies import Ideology
from llm_mediator_simulation.personalities.moral_foundations import MoralFoundation
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import (
    KeyingDirection,
    Likert3Level,
    Likert5ImportanceLevel,
    Likert5Level,
    Likert7AgreementLevel,
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debate.async_handler import AsyncDebateHandler
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debater.config import (
    DebaterConfig,
    TopicOpinion,
)
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.summary.config import SummaryConfig

load_dotenv()


gpt_key = os.getenv("GPT_API_KEY") or ""
google_key = os.getenv("VERTEX_AI_API_KEY") or ""
perspective_key = os.getenv("PERSPECTIVE_API_KEY") or ""

# mediator_model = AsyncGPTModel(api_key=gpt_key, model_name="gpt-4o")
# debater_model = BatchedMistralLocalModel(model_name="/mnt/datastore/models/mistralai/Mistral-7B-Instruct-v0.2" ,max_length=200, debug=True, json=True)
mediator_model = AsyncGoogleModel(api_key=google_key, model_name="gemini-1.5-pro")

PARALLEL_DEBATES = 2

# Debater participants
variable_traits = True
variable_facets = True
variable_ideologies = True

debaters = [
    DebaterConfig(
        name="Alice",
        topic_opinion=TopicOpinion(agreement=Likert7AgreementLevel.STRONGLY_DISAGREE),
        personality=Personality(
            demographic_profile={
                DemographicCharacteristic.ETHNICITY: "White",
                DemographicCharacteristic.BIOLOGICAL_SEX: "female",
            },
            traits={
                PersonalityTrait.AGREEABLENESS: Likert3Level.HIGH,
                PersonalityTrait.CONSCIENTIOUSNESS: Likert3Level.LOW,
                PersonalityTrait.EXTRAVERSION: Likert3Level.AVERAGE,
                PersonalityTrait.NEUROTICISM: Likert3Level.HIGH,
                PersonalityTrait.OPENNESS: Likert3Level.LOW,
            },
            facets={
                PersonalityFacet.ALTRUISM: KeyingDirection.POSITIVE,
                PersonalityFacet.ANGER: KeyingDirection.NEGATIVE,
                PersonalityFacet.ANXIETY: KeyingDirection.POSITIVE,
            },
            moral_foundations={
                MoralFoundation.CARE_HARM: Likert5Level.EXTREMELY,
                MoralFoundation.AUTHORITY_SUBVERSION: Likert5Level.SLIGHTLY,
                MoralFoundation.FAIRNESS_CHEATING_PROPORTIONALITY: Likert5Level.MODERATELY,
                MoralFoundation.LOYALTY_BETRAYAL: Likert5Level.NOT_AT_ALL,
                MoralFoundation.SANCTITY_DEGRADATION_PURITY: Likert5Level.SLIGHTLY,
            },
            basic_human_values={
                BasicHumanValues.SELF_DIRECTION_THOUGHT: Likert5ImportanceLevel.IMPORTANT,
                BasicHumanValues.STIMULATION: Likert5ImportanceLevel.VERY_IMPORTANT,
                BasicHumanValues.HEDONISM: Likert5ImportanceLevel.VERY_IMPORTANT,
                BasicHumanValues.ACHIEVEMENT: Likert5ImportanceLevel.IMPORTANT,
                BasicHumanValues.POWER_DOMINANCE: Likert5ImportanceLevel.NOT_IMPORTANT,
                BasicHumanValues.TRADITION: Likert5ImportanceLevel.OF_SUPREME_IMPORTANCE,
            },
            cognitive_biases=[
                CognitiveBias.CONFIRMATION_BIAS,  # type: ignore
                CognitiveBias.AGENT_DETECTION,  # type: ignore
            ],
            fallacies=[Fallacy.AD_HOMINEM, Fallacy.APPEAL_TO_AUTHORITY],  # type: ignore
            vote_last_presidential_election="voted for Donald Trump",
            ideologies=Ideology.MODERATE,
            variable_traits=variable_traits,
            variable_facets=variable_facets,
            variable_ideologies=variable_ideologies,
        ),
    ),
    DebaterConfig(
        name="Bob",
        topic_opinion=TopicOpinion(agreement=Likert7AgreementLevel.STRONGLY_AGREE),
        personality=Personality(
            demographic_profile={
                DemographicCharacteristic.ETHNICITY: "White",
                DemographicCharacteristic.BIOLOGICAL_SEX: "male",
            },
            traits={
                PersonalityTrait.AGREEABLENESS: Likert3Level.HIGH,
                PersonalityTrait.CONSCIENTIOUSNESS: Likert3Level.LOW,
                PersonalityTrait.EXTRAVERSION: Likert3Level.AVERAGE,
                PersonalityTrait.NEUROTICISM: Likert3Level.HIGH,
                PersonalityTrait.OPENNESS: Likert3Level.LOW,
            },
            facets={
                PersonalityFacet.ALTRUISM: KeyingDirection.POSITIVE,
                PersonalityFacet.ANGER: KeyingDirection.NEGATIVE,
                PersonalityFacet.ANXIETY: KeyingDirection.POSITIVE,
            },
            moral_foundations={
                MoralFoundation.CARE_HARM: Likert5Level.EXTREMELY,
                MoralFoundation.AUTHORITY_SUBVERSION: Likert5Level.SLIGHTLY,
                MoralFoundation.FAIRNESS_CHEATING_PROPORTIONALITY: Likert5Level.MODERATELY,
                MoralFoundation.LOYALTY_BETRAYAL: Likert5Level.NOT_AT_ALL,
                MoralFoundation.SANCTITY_DEGRADATION_PURITY: Likert5Level.SLIGHTLY,
            },
            basic_human_values={
                BasicHumanValues.SELF_DIRECTION_THOUGHT: Likert5ImportanceLevel.IMPORTANT,
                BasicHumanValues.STIMULATION: Likert5ImportanceLevel.VERY_IMPORTANT,
                BasicHumanValues.HEDONISM: Likert5ImportanceLevel.VERY_IMPORTANT,
                BasicHumanValues.ACHIEVEMENT: Likert5ImportanceLevel.IMPORTANT,
                BasicHumanValues.POWER_DOMINANCE: Likert5ImportanceLevel.NOT_IMPORTANT,
                BasicHumanValues.TRADITION: Likert5ImportanceLevel.OF_SUPREME_IMPORTANCE,
            },
            cognitive_biases=[
                CognitiveBias.ADDITIVE_BIAS,  # type: ignore
                CognitiveBias.AGENT_DETECTION,  # type: ignore
            ],
            fallacies=[Fallacy.AD_HOMINEM, Fallacy.APPEAL_TO_AUTHORITY],  # type: ignore
            vote_last_presidential_election="voted for Donald Trump",
            ideologies=Ideology.MODERATE,
            variable_traits=variable_traits,
            variable_facets=variable_facets,
            variable_ideologies=variable_ideologies,
        ),
    ),
]

metrics = AsyncMetricsHandler(
    model=mediator_model,
    argument_qualities=[
        ArgumentQuality.APPROPRIATENESS,
        ArgumentQuality.CLARITY,
        ArgumentQuality.LOCAL_ACCEPTABILITY,
        ArgumentQuality.EMOTIONAL_APPEAL,
    ],
)  # perspective=PerspectiveScorer(api_key=perspective_key))


# The conversation summary handler (keep track of the general history and of the n latest messages)
summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters)

# The debate configuration (which topic to discuss, and customisable instructions)
debate_config = DebateConfig(
    statement="We should use nuclear power.",
)


mediator_config = MediatorConfig()


debate = AsyncDebateHandler(
    debater_model=mediator_model,
    mediator_model=mediator_model,
    debaters=debaters,
    config=debate_config,
    summary_config=summary_config,
    metrics_handler=metrics,
    mediator_config=mediator_config,
    parallel_debates=PARALLEL_DEBATES,
)


asyncio.run(debate.run(rounds=3))


debate.pickle("async_debate")
