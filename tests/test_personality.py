import unittest

from llm_mediator_simulation.personalities.cognitive_biases import CognitiveBias
from llm_mediator_simulation.personalities.facets import PersonalityFacet
from llm_mediator_simulation.personalities.fallacies import Fallacy
from llm_mediator_simulation.personalities.human_values import BasicHumanValues
from llm_mediator_simulation.personalities.ideologies import Ideology, Issues
from llm_mediator_simulation.personalities.moral_foundations import MoralFoundation
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.scales import (
    KeyingDirection,
    Likert3Level,
    Likert5ImportanceLevel,
    Likert5Level,
)
from llm_mediator_simulation.personalities.traits import PersonalityTrait


class TestPersonality(unittest.TestCase):
    def test_to_prompt(self):
        """Test to_prompt with a full profile."""
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

        cognitive_biases = [CognitiveBias.ADDITIVE_BIAS, CognitiveBias.AGENT_DETECTION]

        fallacies = [Fallacy.AD_HOMINEM, Fallacy.APPEAL_TO_AUTHORITY]

        vote_last_presidential_election = "voted for Donald Trump"

        ideologies = {
            Issues.ECONOMIC: Ideology.MODERATE,
            Issues.SOCIAL: Ideology.CONSERVATIVE,
        }

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
        )
        prompt = personality.to_prompt()

        assert (
            prompt.strip()
            == """ethnicity: white;
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
You are not at all sensitive to signs that another person is (or is not) a team player. Intuitions about cooperating with ingroups and competing with outgroups 
                        are not at all relevant to your thinking.
You are slightly sensitive to signs of rank or status, and to signs that other people are (or are not) behaving properly, given their position and wary of a diverse array of symbolic objects and threats. Intuitions about deference toward legitimate authorities and the defense of traditions, all of which are seen as providing stability and fending off chaos and avoiding bodily and spiritual contamination and degradation 
                        are slightly relevant to your thinking.
You are moderately willing to shun or punish cheaters. Intuitions about individuals getting rewarded in proportion to their merit or contribution 
                        are somewhat relevant to your thinking.
You are extremely sensitive to signs of suffering and need. Intuitions about avoiding emotional and physical damage to another individual 
                        are extremely relevant to your thinking.

As a guiding principle in your life:
- Freedom to cultivate one's own ideas and abilities and success according to social standards are 
                    important to your values.
- Excitement, novelty, and change and pleasure and sensuous gratification are 
                    very important to your values.
- Power through exercising control over people is 
                    not important to your values.
- Maintaining and preserving cultural, family, or religious traditions is 
                    of supreme importance to your values.

Remember, you are role-playing as a real person. Like humans, you are susceptible to the following cognitive biases:
- Additive bias, that is, the tendency to solve problems through addition, even when subtraction is a better approach.
- Agent detection, that is, the inclination to presume the purposeful intervention of a sentient or intelligent agent.

And you are also susceptible to the following fallacies:
- Ad Hominem, that is, attacking the arguer instead of the argument.
- Appeal to Authority, that is, an assertion is deemed true because of the position or authority of the person asserting it.

In the last presidential election, you voted for Donald Trump.

You identify as:
- Moderate on economic issues.
- Conservative on social issues."""
        )


if __name__ == "__main__":
    unittest.main()
