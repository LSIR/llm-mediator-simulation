# Debater participants
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
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debater.config import (
    DebaterConfig,
    TopicOpinion,
)

debaters = [
    DebaterConfig(
        name="Alice",
        topic_opinion=TopicOpinion(agreement=Likert7AgreementLevel.STRONGLY_DISAGREE),
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
                CognitiveBias.CONFIRMATION_BIAS,  # type: ignore
                CognitiveBias.AGENT_DETECTION,  # type: ignore
            ],
            fallacies=[Fallacy.AD_HOMINEM, Fallacy.APPEAL_TO_AUTHORITY],  # type: ignore
            vote_last_presidential_election="voted for Donald Trump",
            ideologies={
                Issues.ECONOMIC: Ideology.MODERATE,
                Issues.SOCIAL: Ideology.CONSERVATIVE,
            },
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
            ideologies={
                Issues.ECONOMIC: Ideology.MODERATE,
                Issues.SOCIAL: Ideology.CONSERVATIVE,
            },
        ),
    ),
]
