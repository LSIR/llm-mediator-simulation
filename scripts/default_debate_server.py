"""Example script to run a debate simulation on nuclear energy."""

import time
import os

from dotenv import load_dotenv

from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.models.mistral_local_server_model import MistralLocalServerModel
from llm_mediator_simulation.personalities.cognitive_biases import CognitiveBias
from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.facets import PersonalityFacet
from llm_mediator_simulation.personalities.fallacies import Fallacy
from llm_mediator_simulation.personalities.human_values import BasicHumanValues
from llm_mediator_simulation.personalities.ideologies import Ideology, Issues
from llm_mediator_simulation.personalities.moral_foundations import MoralFoundation
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import KeyingDirection, Likert3Level, Likert5ImportanceLevel, Likert5Level, Likert7AgreementLevel
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.simulation.debater.config import (
    AxisPosition,
    TopicOpinion,
    DebaterConfig,
)
from llm_mediator_simulation.simulation.summary.config import SummaryConfig
from llm_mediator_simulation.visualization.transcript import debate_transcript

load_dotenv()

SEED = 42


# Load the mediator model
# google_key = os.getenv("VERTEX_AI_API_KEY") or ""
# Seed not currently available through Google's GenerativeAI SDK
# mediator_model = GoogleModel(api_key=google_key, model_name="gemini-1.5-pro", seed=None)

gpt_key = os.getenv("GPT_API_KEY") or ""
mediator_model =  GPTModel(api_key=gpt_key, model_name="gpt-4o")

debater_model = MistralLocalServerModel(port=8000)

# Debater participants
debaters = [
    DebaterConfig(
        name="Alice",
        topic_opinion=TopicOpinion(agreement=Likert7AgreementLevel.STRONGLY_DISAGREE),
        personality=Personality(
            demographic_profile={DemographicCharacteristic.ETHNICITY: "White", DemographicCharacteristic.BIOLOGICAL_SEX: "male"},
            traits={
                PersonalityTrait.AGREEABLENESS: Likert3Level.HIGH,
                PersonalityTrait.CONSCIENTIOUSNESS: Likert3Level.LOW,
                PersonalityTrait.EXTRAVERSION: Likert3Level.AVERAGE,
                PersonalityTrait.NEUROTICISM: Likert3Level.HIGH,
                PersonalityTrait.OPENNESS: Likert3Level.LOW
            },
            facets={
                PersonalityFacet.ALTRUISM: KeyingDirection.POSITIVE,
                PersonalityFacet.ANGER: KeyingDirection.NEGATIVE,
                PersonalityFacet.ANXIETY: KeyingDirection.POSITIVE
            },
            moral_foundations=
            {
                MoralFoundation.CARE_HARM: Likert5Level.EXTREMELY,
                MoralFoundation.AUTHORITY_SUBVERSION: Likert5Level.SLIGHTLY,
                MoralFoundation.FAIRNESS_CHEATING_PROPORTIONALITY: Likert5Level.MODERATELY,
                MoralFoundation.LOYALTY_BETRAYAL: Likert5Level.NOT_AT_ALL,
                MoralFoundation.SANCTITY_DEGRADATION_PURITY: Likert5Level.SLIGHTLY
            },
            basic_human_values={
                BasicHumanValues.SELF_DIRECTION_THOUGHT: Likert5ImportanceLevel.IMPORTANT,
                BasicHumanValues.STIMULATION: Likert5ImportanceLevel.VERY_IMPORTANT,
                BasicHumanValues.HEDONISM: Likert5ImportanceLevel.VERY_IMPORTANT,
                BasicHumanValues.ACHIEVEMENT: Likert5ImportanceLevel.IMPORTANT,
                BasicHumanValues.POWER_DOMINANCE: Likert5ImportanceLevel.NOT_IMPORTANT,
                BasicHumanValues.TRADITION: Likert5ImportanceLevel.OF_SUPREME_IMPORTANCE
            },
            cognitive_biases=[CognitiveBias.ADDITIVE_BIAS, CognitiveBias.AGENT_DETECTION],
            fallacies=[Fallacy.AD_HOMINEM, Fallacy.APPEAL_TO_AUTHORITY],
            vote_last_presidential_election="voted for Donald Trump",
            ideologies={
                Issues.ECONOMIC: Ideology.MODERATE,
                Issues.SOCIAL: Ideology.CONSERVATIVE
            }
        )
    ),
    DebaterConfig(
        name="Bob",
        topic_opinion=TopicOpinion(agreement=Likert7AgreementLevel.STRONGLY_AGREE),
        personality=Personality(
            demographic_profile={DemographicCharacteristic.ETHNICITY: "White", DemographicCharacteristic.BIOLOGICAL_SEX: "male"},
            traits={
                PersonalityTrait.AGREEABLENESS: Likert3Level.HIGH,
                PersonalityTrait.CONSCIENTIOUSNESS: Likert3Level.LOW,
                PersonalityTrait.EXTRAVERSION: Likert3Level.AVERAGE,
                PersonalityTrait.NEUROTICISM: Likert3Level.HIGH,
                PersonalityTrait.OPENNESS: Likert3Level.LOW
            },
            facets={
                PersonalityFacet.ALTRUISM: KeyingDirection.POSITIVE,
                PersonalityFacet.ANGER: KeyingDirection.NEGATIVE,
                PersonalityFacet.ANXIETY: KeyingDirection.POSITIVE
            },
            moral_foundations=
            {
                MoralFoundation.CARE_HARM: Likert5Level.EXTREMELY,
                MoralFoundation.AUTHORITY_SUBVERSION: Likert5Level.SLIGHTLY,
                MoralFoundation.FAIRNESS_CHEATING_PROPORTIONALITY: Likert5Level.MODERATELY,
                MoralFoundation.LOYALTY_BETRAYAL: Likert5Level.NOT_AT_ALL,
                MoralFoundation.SANCTITY_DEGRADATION_PURITY: Likert5Level.SLIGHTLY
            },
            basic_human_values={
                BasicHumanValues.SELF_DIRECTION_THOUGHT: Likert5ImportanceLevel.IMPORTANT,
                BasicHumanValues.STIMULATION: Likert5ImportanceLevel.VERY_IMPORTANT,
                BasicHumanValues.HEDONISM: Likert5ImportanceLevel.VERY_IMPORTANT,
                BasicHumanValues.ACHIEVEMENT: Likert5ImportanceLevel.IMPORTANT,
                BasicHumanValues.POWER_DOMINANCE: Likert5ImportanceLevel.NOT_IMPORTANT,
                BasicHumanValues.TRADITION: Likert5ImportanceLevel.OF_SUPREME_IMPORTANCE
            },
            cognitive_biases=[CognitiveBias.ADDITIVE_BIAS, CognitiveBias.AGENT_DETECTION],
            fallacies=[Fallacy.AD_HOMINEM, Fallacy.APPEAL_TO_AUTHORITY],
            vote_last_presidential_election="voted for Donald Trump",
            ideologies={
                Issues.ECONOMIC: Ideology.MODERATE,
                Issues.SOCIAL: Ideology.CONSERVATIVE
            }
        )
    ),
]

# The conversation summary handler (keep track of the general history and of the n latest messages)
summary_config = SummaryConfig(latest_messages_limit=3, debaters=debaters)

# The debate configuration (which topic to discuss, and customisable instructions)
debate_config = DebateConfig(
    statement="We should use nuclear power.",
)

# mediator_config = MediatorConfig()

# The debate runner
debate = DebateHandler(
    debater_model=debater_model,
    mediator_model=mediator_model,
    debaters=debaters,
    config=debate_config,
    summary_config=summary_config,
    metrics_handler=None,
    mediator_config=None,
    seed=SEED,
    variable_personality=False,
)

debate.run(rounds=4)

name_timestamp = time.strftime('%Y%m%d-%H%M%S')
output_path = f"debates_sandbox"
data = debate.to_debate_pickle()
print(debate_transcript(data))
debate.pickle(os.path.join(output_path, f"debate_{name_timestamp}"))


