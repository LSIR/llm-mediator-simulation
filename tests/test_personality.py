import random
import unittest
from datetime import datetime

from llm_mediator_simulation.models.dummy_model import DummyModel
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
    Likert11LikelihoodLevel,
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debater.config import (
    DebaterConfig,
    TopicOpinion,
)
from llm_mediator_simulation.simulation.prompt import debater_update
from llm_mediator_simulation.utils.types import Intervention


class TestPersonality(unittest.TestCase):

    def exhaustive_profile(self) -> Personality:
        demo_profile = {
            DemographicCharacteristic.ETHNICITY: "White",
            DemographicCharacteristic.BIOLOGICAL_SEX: "male",
        }

        traits = {
            PersonalityTrait.AGREEABLENESS: Likert3Level.HIGH,
            PersonalityTrait.CONSCIENTIOUSNESS: Likert3Level.LOW,
            PersonalityTrait.EXTRAVERSION: Likert3Level.AVERAGE,
            PersonalityTrait.NEUROTICISM: Likert3Level.HIGH,
            PersonalityTrait.OPENNESS: Likert3Level.LOW,
        }

        facets = {
            PersonalityFacet.ALTRUISM: KeyingDirection.POSITIVE,
            PersonalityFacet.ANGER: KeyingDirection.NEGATIVE,
            PersonalityFacet.ANXIETY: KeyingDirection.POSITIVE,
        }

        moral_foundations = {
            MoralFoundation.CARE_HARM: Likert5Level.EXTREMELY,
            MoralFoundation.AUTHORITY_SUBVERSION: Likert5Level.SLIGHTLY,
            MoralFoundation.FAIRNESS_CHEATING_PROPORTIONALITY: Likert5Level.MODERATELY,
            MoralFoundation.LOYALTY_BETRAYAL: Likert5Level.NOT_AT_ALL,
            MoralFoundation.SANCTITY_DEGRADATION_PURITY: Likert5Level.SLIGHTLY,
        }

        basic_human_values = {
            BasicHumanValues.SELF_DIRECTION_THOUGHT: Likert5ImportanceLevel.IMPORTANT,
            BasicHumanValues.STIMULATION: Likert5ImportanceLevel.VERY_IMPORTANT,
            BasicHumanValues.HEDONISM: Likert5ImportanceLevel.VERY_IMPORTANT,
            BasicHumanValues.ACHIEVEMENT: Likert5ImportanceLevel.IMPORTANT,
            BasicHumanValues.POWER_DOMINANCE: Likert5ImportanceLevel.NOT_IMPORTANT,
            BasicHumanValues.TRADITION: Likert5ImportanceLevel.OF_SUPREME_IMPORTANCE,
        }

        cognitive_biases = [CognitiveBias.ADDITIVE_BIAS, CognitiveBias.AGENT_DETECTION]  # type: ignore

        fallacies = [Fallacy.AD_HOMINEM, Fallacy.APPEAL_TO_AUTHORITY]  # type: ignore

        vote_last_presidential_election = "voted for Donald Trump"

        ideologies = {
            Issues.ECONOMIC: Ideology.MODERATE,
            Issues.SOCIAL: Ideology.CONSERVATIVE,
        }

        agreement_with_statements = {
            "abortions should be illegal": Likert7AgreementLevel.NEUTRAL,
            "guns should be banned": Likert7AgreementLevel.NEUTRAL,
            "the death penalty should be abolished": Likert7AgreementLevel.SLIGHTLY_AGREE,
        }

        likelihood_of_beliefs = {
            "the Earth is flat": Likert11LikelihoodLevel.NEUTRAL,
            "vaccines cause autism": Likert11LikelihoodLevel.NEUTRAL,
            "climate change is a hoax": Likert11LikelihoodLevel.SOMEWHAT_UNLIKELY,
        }

        free_form_opinions = ["immigration is a problem", "the government is corrupt"]

        personality = Personality(
            demographic_profile=demo_profile,
            traits=traits,
            facets=facets,
            moral_foundations=moral_foundations,
            basic_human_values=basic_human_values,
            cognitive_biases=cognitive_biases,
            fallacies=fallacies,
            vote_last_presidential_election=vote_last_presidential_election,
            ideologies=ideologies,
            agreement_with_statements=agreement_with_statements,
            likelihood_of_beliefs=likelihood_of_beliefs,
            free_form_opinions=free_form_opinions,
        )
        return personality

    def test_to_prompt(self):
        """Test personality.to_prompt with an exhaustive profile."""
        personality = self.exhaustive_profile()
        prompt = personality.to_prompt()
        expected_prompt = """ethnicity: white;
biological sex: male;

You are compassionate, good-natured, and eager to cooperate and avoid conflict.
You are easy-going, not very well organized, and sometimes careless. You prefer not to make plans.
You are moderate in activity and enthusiasm. You enjoy the company of others but also values privacy.
You are sensitive, emotional, and prone to experience feelings that are upsetting.
You are down-to-earth, practical, traditional, and pretty much set in your ways.

You make people feel welcome. You anticipate the needs of others. You love to help others. You are concerned about others. You have a good word for everyone.
You rarely get irritated. You seldom get mad. You are not easily annoyed. You keep your cool. You rarely complain.
You worry about things. You fear for the worst. You are afraid of many things. You get stressed out easily. You get caught up in your problems.

When you decide whether something is right or wrong, your moral matrix rests on the following foundations:
You are not at all sensitive to signs that another person is (or is not) a team player. Intuitions about cooperating with ingroups and competing with outgroups are not at all relevant to your thinking.
You are slightly sensitive to signs of rank or status, and to signs that other people are (or are not) behaving properly, given their position and wary of a diverse array of symbolic objects and threats. Intuitions about deference toward legitimate authorities and the defense of traditions, all of which are seen as providing stability and fending off chaos and avoiding bodily and spiritual contamination and degradation are slightly relevant to your thinking.
You are moderately willing to shun or punish cheaters. Intuitions about individuals getting rewarded in proportion to their merit or contribution are somewhat relevant to your thinking.
You are extremely sensitive to signs of suffering and need. Intuitions about avoiding emotional and physical damage to another individual are extremely relevant to your thinking.

As guiding principles in your life:
- Freedom to cultivate one's own ideas and abilities and success according to social standards are important to your values.
- Excitement, novelty, and change and pleasure and sensuous gratification are very important to your values.
- Power through exercising control over people is not important to your values.
- Maintaining and preserving cultural, family, or religious traditions is of supreme importance to your values.

Remember, you are role-playing as a real person. Like humans, you are susceptible to the following cognitive biases:
- Additive bias, that is, the tendency to solve problems through addition, even when subtraction is a better approach.
- Agent detection, that is, the inclination to presume the purposeful intervention of a sentient or intelligent agent.

And you are also susceptible to the following fallacies:
- Ad Hominem, that is, attacking the arguer instead of the argument.
- Appeal to Authority, that is, an assertion is deemed true because of the position or authority of the person asserting it.

In the last presidential election, you voted for Donald Trump.

You identify as:
- Moderate on economic issues.
- Conservative on social issues.

You neither agree nor disagree with the following statements:
- Abortions should be illegal
- Guns should be banned

You slightly agree with the following statement:
- The death penalty should be abolished

You believe it is somewhat unlikely that:
- climate change is a hoax

You do not know whether:
- the Earth is flat
- vaccines cause autism

You have the following opinions:
- Immigration is a problem
- The government is corrupt"""

        assert prompt == expected_prompt

    def test_profile_update(self):
        personality = self.exhaustive_profile()
        personality.variable_traits = True
        personality.variable_facets = True
        personality.variable_moral_foundations = False
        personality.variable_basic_human_values = False
        personality.variable_cognitive_biases = False
        personality.variable_fallacies = True
        personality.variable_ideologies = True
        personality.variable_agreement_with_statements = False
        personality.variable_likelihood_of_beliefs = True
        debater = DebaterConfig(
            name="Bob",
            topic_opinion=TopicOpinion(
                agreement=Likert7AgreementLevel.STRONGLY_DISAGREE
            ),
            personality=personality,
            variable_topic_opinion=True,
        )
        random.seed(42)
        assert debater.personality is not None
        old_personality_prompt = debater.personality.to_prompt()
        update_prompt = debater_update(
            model=DummyModel(),
            debater=debater,
            debate_statement="We should use nuclear power.",
            interventions=[
                Intervention(
                    debater=DebaterConfig(name="Dummy name"),
                    text="Dummy text",
                    prompt="Dummy prompt",
                    justification="Dummy justification",
                    timestamp=datetime.now(),
                )
            ],
        )

        expected_update_prompt = """You are roleplaying this real person:
name: Bob;
ethnicity: white;
biological sex: male;

Here is your current personality:
In the last presidential election, you voted for Donald Trump.

Traits:
- Agreeableness: high
- Conscientiousness: low
- Extraversion: average
- Neuroticism: high
- Openness to experience: low

Facets:
- Altruism: yes
- Anger: no
- Anxiety: yes

Moral foundations:
- Care/harm: extremely
- Authority/subversion: slightly
- Fairness/cheating: moderately
- Loyalty/betrayal: not at all
- Sanctity/degradation: slightly

Basic human values:
- Self-direction thought: important
- Stimulation: very important
- Hedonism: very important
- Achievement: important
- Power dominance: not important
- Tradition: of supreme importance

Cognitive biases:
- Additive bias
- Agent detection

Fallacy:
- Ad hominem
- Appeal to authority

Ideology:
- Economic: moderate
- Social: conservative

Statements:
- We should use nuclear power (current debate statement): strongly disagree
- Abortions should be illegal: neither agree nor disagree
- Guns should be banned: neither agree nor disagree
- The death penalty should be abolished: slightly agree

Beliefs:
- The earth is flat: NEUTRAL
- Vaccines cause autism: NEUTRAL
- Climate change is a hoax: SOMEWHAT UNLIKELY

You have the following opinions:
- Immigration is a problem
- The government is corrupt

You have the opportunity to make your personality evolve based on the things people have said after your last intervention.

Here are the last messages:
— Dummy name: Dummy text

You can choose to evolve your traits, agreement with statements, and likelihood of beliefs with "more", "less", or "same".
You can choose to evolve your facets with "yes" or "no".
You can choose to evolve your ideologies with "more liberal", "more conservative", "same" or "libertarian".

Answer in valid JSON format with the following structure only:
```json
{
    "agreeableness": a string ("more", "less", or "same") to update this trait,
    "conscientiousness": a string ("more", "less", or "same") to update this trait,
    "extraversion": a string ("more", "less", or "same") to update this trait,
    "neuroticism": a string ("more", "less", or "same") to update this trait,
    "openness_to_experience": a string ("more", "less", or "same") to update this trait,
    "current_dabate_statement": a string ("more", "less", or "same") to update your agreement with the current debate statement,
    "the_earth_is_flat": a string ("more", "less", or "same") to update your assessment of this belief's likelihood,
    "vaccines_cause_autism": a string ("more", "less", or "same") to update your assessment of this belief's likelihood,
    "climate_change_is_a_hoax": a string ("more", "less", or "same") to update your assessment of this belief's likelihood,
    "altruism": a string ("yes" or "no") to update this facet,
    "anger": a string ("yes" or "no") to update this facet,
    "anxiety": a string ("yes" or "no") to update this facet,
    "economic": a string ("more liberal", "more conservative", "same", or "libertarian") to update your ideology on this issue,
    "social": a string ("more liberal", "more conservative", "same", or "libertarian") to update your ideology on this issue
}
```
"""

        assert update_prompt == expected_update_prompt

        personality = debater.personality
        assert personality is not None
        # Traits updated
        assert personality.traits is not None and isinstance(personality.traits, dict)
        assert personality.traits[PersonalityTrait.AGREEABLENESS] == Likert3Level.HIGH
        assert (
            personality.traits[PersonalityTrait.CONSCIENTIOUSNESS]
            == Likert3Level.AVERAGE
        )
        assert personality.traits[PersonalityTrait.EXTRAVERSION] == Likert3Level.HIGH
        assert personality.traits[PersonalityTrait.NEUROTICISM] == Likert3Level.HIGH
        assert personality.traits[PersonalityTrait.OPENNESS] == Likert3Level.AVERAGE

        # Facets updated
        assert personality.facets is not None and isinstance(personality.facets, dict)
        assert personality.facets[PersonalityFacet.ALTRUISM] == KeyingDirection.POSITIVE
        assert personality.facets[PersonalityFacet.ANGER] == KeyingDirection.POSITIVE
        assert personality.facets[PersonalityFacet.ANXIETY] == KeyingDirection.POSITIVE

        # Moral foundations not updated
        assert personality.moral_foundations is not None and isinstance(
            personality.moral_foundations, dict
        )
        assert (
            personality.moral_foundations[MoralFoundation.CARE_HARM]
            == Likert5Level.EXTREMELY
        )
        assert (
            personality.moral_foundations[MoralFoundation.AUTHORITY_SUBVERSION]
            == Likert5Level.SLIGHTLY
        )
        assert (
            personality.moral_foundations[
                MoralFoundation.FAIRNESS_CHEATING_PROPORTIONALITY
            ]
            == Likert5Level.MODERATELY
        )
        assert (
            personality.moral_foundations[MoralFoundation.LOYALTY_BETRAYAL]
            == Likert5Level.NOT_AT_ALL
        )
        assert (
            personality.moral_foundations[MoralFoundation.SANCTITY_DEGRADATION_PURITY]
            == Likert5Level.SLIGHTLY
        )

        # Basic human values not updated
        assert personality.basic_human_values is not None and isinstance(
            personality.basic_human_values, dict
        )
        assert (
            personality.basic_human_values[BasicHumanValues.SELF_DIRECTION_THOUGHT]
            == Likert5ImportanceLevel.IMPORTANT
        )
        assert (
            personality.basic_human_values[BasicHumanValues.STIMULATION]
            == Likert5ImportanceLevel.VERY_IMPORTANT
        )
        assert (
            personality.basic_human_values[BasicHumanValues.HEDONISM]
            == Likert5ImportanceLevel.VERY_IMPORTANT
        )
        assert (
            personality.basic_human_values[BasicHumanValues.ACHIEVEMENT]
            == Likert5ImportanceLevel.IMPORTANT
        )
        assert (
            personality.basic_human_values[BasicHumanValues.POWER_DOMINANCE]
            == Likert5ImportanceLevel.NOT_IMPORTANT
        )
        assert (
            personality.basic_human_values[BasicHumanValues.TRADITION]
            == Likert5ImportanceLevel.OF_SUPREME_IMPORTANCE
        )

        # Cognitive biases not updated
        assert personality.cognitive_biases is not None
        assert CognitiveBias.ADDITIVE_BIAS in personality.cognitive_biases  # type: ignore
        assert CognitiveBias.AGENT_DETECTION in personality.cognitive_biases  # type: ignore
        assert len(personality.cognitive_biases) == 2

        # Fallacies updated
        assert personality.fallacies is not None
        assert Fallacy.AD_HOMINEM in personality.fallacies
        assert Fallacy.APPEAL_TO_AUTHORITY in personality.fallacies
        assert Fallacy.BACONIAN_FALLACY in personality.fallacies
        assert len(personality.fallacies) == 3

        # Ideologies updated
        assert personality.ideologies is not None and isinstance(
            personality.ideologies, dict
        )
        assert personality.ideologies[Issues.ECONOMIC] == Ideology.SLIGHTLY_LIBERAL
        assert personality.ideologies[Issues.SOCIAL] == Ideology.SLIGHTLY_CONSERVATIVE

        # Agreement with statements not updated
        assert personality.agreement_with_statements is not None and isinstance(
            personality.agreement_with_statements, dict
        )
        assert (
            personality.agreement_with_statements["abortions should be illegal"]
            == Likert7AgreementLevel.NEUTRAL
        )
        assert (
            personality.agreement_with_statements["guns should be banned"]
            == Likert7AgreementLevel.NEUTRAL
        )
        assert (
            personality.agreement_with_statements[
                "the death penalty should be abolished"
            ]
            == Likert7AgreementLevel.SLIGHTLY_AGREE
        )

        # Likelihood of beliefs updated # TODO test does not pass because of uppercase Earth
        assert personality.likelihood_of_beliefs is not None and isinstance(
            personality.likelihood_of_beliefs, dict
        )
        print(personality.likelihood_of_beliefs)
        assert (
            personality.likelihood_of_beliefs["the Earth is flat"]
            == Likert11LikelihoodLevel.SOMEWHAT_LIKELY
        )
        assert (
            personality.likelihood_of_beliefs["vaccines cause autism"]
            == Likert11LikelihoodLevel.SOMEWHAT_LIKELY
        )
        assert (
            personality.likelihood_of_beliefs["climate change is a hoax"]
            == Likert11LikelihoodLevel.NEUTRAL
        )


if __name__ == "__main__":
    unittest.main()
