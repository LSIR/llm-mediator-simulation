from dataclasses import dataclass
from enum import Enum


class DemographicCharacteristic(Enum):
    """Demographic characteristic for agents.
        Based on:
            - https://github.com/yunshiuan/llm-agent-opinion-dynamics/blob/main/prompts/opinion_dynamics/Flache_2017/list_agent_descriptions.csv
            - 10.48550/arXiv.2310.05984 - Fig. 1
            - 10.5555/3618408.3619652 - Table 2
            - https://electionstudies.org/wp-content/uploads/2024/05/anes_specialstudy_2024ets_qnnaire_20240403.pdf
            - https://www.pewresearch.org/wp-content/uploads/sites/20/2024/07/2024-NPORS-Paper-Questionnaire.pdf 
    """

    # NAME = "name" ; name is not a demographic characteristic but a required field for agents

    ETHNICITY = "ethnicity"

    GENDER = "gender"

    NATIONALITY = "nationality"

    AGE = "age"

    MARITAL_STATUS = "marital status"

    EDUCATION = "education"

    OCCUPATION = "occupation"

    POLITICAL_LEANING = "political leaning"

    RELIGION = "religion or spiritual beliefs"

    SEXUAL_ORIENTATION = "sexual orientation"

    HEALTH_CONDITION = "health condition"

    INCOME = "total income of your family in the past 12 months"

    HOUSEHOLD_SIZE = "household size"

    NUMBER_OF_DEPENDENT = "number of dependent children or elderly family members"

    LIVING_QUARTERS = "living quarters"

    LANGUAGE_SPOKEN = "language spoken"

    CITY_OF_RESIDENCE = "city of residence"

    PRIMARY_MODE_OF_TRANSPORTATION = "primary mode of transportation"

    BACKGROUND = "background"


@dataclass
class PersonalityTraitValue:
    """Typing for the values of a personality trait."""

    name: str
    low: str
    average: str
    high: str

    
class PersonalityTrait(Enum):
    """Big 5 Personality Trait for agents.
        Based on:
            - 10.18653/v1/2023.findings-emnlp.156 (Descriptions)
            - 10.1093/acrefore/9780190236557.013.560 - Table 1 (3-point Likert Scale and descriptions for each level for each trait)
            - https://en.wikipedia.org/wiki/Big_Five_personality_traits 
                "A FFM-associated test was used by Cambridge Analytica, and was part of the 'psychographic profiling' 
                controversy during the 2016 US presidential election."
    """

    OPENNESS = PersonalityTraitValue("openness to experience", 
                                     "You are down-to-earth, practical, traditional, and pretty much set in your ways.",
                                     "You are practical but willing to consider new ways of doing things. You seek a balance between the old and the new.",
                                     "You are open to new experiences. You have broad interests and are very imaginative.")

    CONSCIENTIOUSNESS = PersonalityTraitValue("conscientiousness",
                                              "You are easy-going, not very well organized, and sometimes careless. You prefer not to make plans.",
                                              "You are dependable and moderately well organized. You generally have clear goals but are able to set work aside.",
                                              "You are very conscientious and well organized. You have high standards and always strives to achieve goals.")

    EXTRAVERSION = PersonalityTraitValue("extraversion",
                                         "You are introverted, reserved, and serious. You prefer to be alone or with a few close friends.",
                                         "You are moderate in activity and enthusiasm. You enjoy the company of others but also values privacy.",
                                         "You are extraverted, outgoing, active, and high-spirited. You prefer to be around people most of the time.")

    AGREEABLENESS = PersonalityTraitValue("agreeableness",
                                          "You are hardheaded, skeptical, proud, and competitive. You tend to express anger directly.",
                                          "You are generally warm, trusting, and agreeable, but you can sometimes be stubborn and competitive.",
                                          "You are compassionate, good-natured, and eager to cooperate and avoid conflict.")

    NEUROTICISM = PersonalityTraitValue("neuroticism",
                                        "You are secure, hardy, and generally relaxed, even under stressful conditions.",
                                        "You are generally calm and able to deal with stress, but sometimes experiences feelings of guilt, anger, or sadness.",
                                        "You are sensitive, emotional, and prone to experience feelings that are upsetting.")
    

class Likert3Level(Enum):
    """Level on a 3-point likert scale axis."""

    LOW = 0
    AVERAGE = 1
    HIGH = 2


class KeyingDirection(Enum):
    """Binary value.
    Based on:
            - 10.1016/j.jrp.2014.05.003 
    """

    NEGATIVE = 0
    POSITIVE = 1


@dataclass
class item:
    """Personality trait item for agents.
        Based on:
            - https://ipip.ori.org/newNEOFacetsKey.htm
                
    """

    trait: PersonalityTrait
    key: KeyingDirection
    description: str


@dataclass
class PersonalityFacetValue:
    """Typing for the values of a personality trait."""

    name: str
    trait: PersonalityTrait
    positively_keyed_items: list[item]
    negatively_keyed_items: list[item]


class PersonalityFacet(Enum):
    """Big 5 Personality Facet for agents.
        Based on:
            - 10.1016/j.jrp.2014.05.003 - Table 1
            - https://ipip.ori.org/newNEOFacetsKey.htm (Descriptions)
            - https://en.wikipedia.org/wiki/Big_Five_personality_traits 
                
    """
    ###################################################################################################
    #                                     Neuroticism Facets                                          #
    ###################################################################################################

    ANXIEITY = PersonalityTraitValue("anxiety", 
        PersonalityTrait.NEUROTICISM,
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You worry about things."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You fear for the worst."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You are afraid of many things."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You get stressed out easily."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You get caught up in your problems.")]
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are not easilly bothered by things."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are relaxed most of the time.."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are not easily disturbed by events."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You don't worry about things that have already happened."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You adapt easily to new situations.")])
    

    ANGER = PersonalityTraitValue("anger",
        PersonalityTrait.NEUROTICISM,
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You get angry easily."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You get irritated easily."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You get upset easily."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You are often in a bad mood."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You lose your temper.")],
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You rarely get irritated."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You seldom get mad."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are not easily annoyed."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You keep your cool."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You rarely complain.")])


    DEPRESSION = PersonalityTraitValue("depression",
        PersonalityTrait.NEUROTICISM,
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You often feel blue."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You dislike yourself."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You are often down in the dumps."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You have a low opinion of yourself."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You have frequent mood swings."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You feel desperate."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You feel that your life lacks direction.")],
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You seldom feel blue."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You feel comfortable with yourself."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are very pleased with yourself.")])


    SELF_CONSCIOUSNESS = PersonalityTraitValue("self-consciousness",
        PersonalityTrait.NEUROTICISM,
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You are easily intimidated."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You are afraid that you will do the wrong thing."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You find it difficult to approach others."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You are afraid to draw attention to yourself."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You only feel comfortable with friends."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You stumble over your words.")],
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are not embarrassed easily."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are comfortable in unfamiliar situations."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are not bothered by difficult social situations."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are able to stand up for yourself.")])


    IMMODERATION = PersonalityTraitValue("immoderation",
        PersonalityTrait.NEUROTICISM,
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You often eat too much."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You don't know why you do some of the things you do."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You do things you later regret."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You go on binges."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You love to eat.")],
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You rarely overindulge."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You easily resist temptations."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are able to control your cravings."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You never spend more than you can afford."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You never splurge.")])


    VULNERABILITY = PersonalityTraitValue("vulnerability",
        PersonalityTrait.NEUROTICISM,
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You panic easily."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You become overwhelmed by events."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You feel that you're unable to deal with things."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You can't make up your mind."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.POSITIVE, "You get overwhelmed by emotions.")],
        [item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You remain calm under pressure."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You can handle complex problems."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You know how to cope."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You readily overcome setbacks."),
        item(PersonalityTrait.NEUROTICISM, KeyingDirection.NEGATIVE, "You are calm even in tense situations.")])
    

    ###################################################################################################
    #                                     Extraversion Facets                                         #
    ###################################################################################################

    FRIENDLINESS = PersonalityTraitValue("friendliness",
        PersonalityTrait.EXTRAVERSION,
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You make friends easily."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You warm up quickly to others."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You feel comfortable around people."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You act comfortably with others."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You cheer people up.")],
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You are hard to get to know."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You often feel uncomfortable around others."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You avoid contacts with others."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You are not really interested in others."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You keep others at a distance.")])


    GREGARIOUSNESS = PersonalityTraitValue("gregariousness",
        PersonalityTrait.EXTRAVERSION,
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You love large parties."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You talk to a lot of different people at parties."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You enjoy being part of a group."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You involve others in what you are doing."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You love surprise parties.")],
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You prefer to be alone."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You want to be left alone."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You don't like crowded events."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You avoid crowds."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You seek quiet.")])


    ASSERTIVENESS = PersonalityTraitValue("assertiveness",
        PersonalityTrait.EXTRAVERSION,
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You take charge."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You try to lead others."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You can talk others into doing things."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You seek to influence others."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You take control of things.")],
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You wait for others to lead the way."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You keep in the background."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You have little to say."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You don't like to draw attention to yourself."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You hold back your opinions.")])


    ACTIVITY_LEVEL = PersonalityTraitValue("activity_level",
        PersonalityTrait.EXTRAVERSION,
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You are always busy."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You are always on the go."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You do a lot in your spare time."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You can manage many things at the same time."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You react quickly.")],
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You like to take it easy."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You like to take your time."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You like a leisurely lifestyle."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You let things proceed at their own pace."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You react slowly.")])


    EXCITEMENT_SEEKING = PersonalityTraitValue("excitement_seeking",
        PersonalityTrait.EXTRAVERSION,
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You love excitement."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You seek adventure."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You love action."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You enjoy being part of a loud crowd."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You enjoy being reckless."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You act wild and crazy."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You are willing to try anything once."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You seek danger.")],
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You would never go hang gliding or bungee jumping."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You dislike loud music.")])


    CHEERFULNESS = PersonalityTraitValue("cheerfulness",
        PersonalityTrait.EXTRAVERSION,
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You radiate joy."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You have a lot of fun."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You express childlike joy."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You laugh your way through life."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You love life."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You look at the bright side of life."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You laugh aloud."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.POSITIVE, "You amuse your friends.")],
        [item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You are not easily amused."),
        item(PersonalityTrait.EXTRAVERSION, KeyingDirection.NEGATIVE, "You seldom joke around.")])

    
    ###################################################################################################
    #                                     Openness Facets                                             #
    ###################################################################################################

    IMAGINATION = PersonalityTraitValue("imagination",
        PersonalityTrait.OPENNESS,
        [item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You have a vivid imagination."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You enjoy wild flights of fantasy."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You love to daydream."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You like to get lost in thought."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You indulge in your fantasies."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You spend time reflecting on things.")],
        [item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You seldom daydream."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You do not have a good imagination."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You seldom get lost in thought."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You have difficulty imagining things.")])


    ARTISTIC_INTERESTS = PersonalityTraitValue("artistic_interests",
        PersonalityTrait.OPENNESS,
        [item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You believe in the importance of art."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You like music."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You see beauty in things that others might not notice."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You love flowers."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You enjoy the beauty of nature.")],
        [item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You do not like art."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You do not like poetry."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You do not enjoy going to art museums."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You do not like concerts."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You do not enjoy watching dance performances.")])


    EMOTIONALITY = PersonalityTraitValue("emotionality",
        PersonalityTrait.OPENNESS,
        [item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You experience your emotions intensely."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You feel others' emotions."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You are passionate about causes."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You enjoy examining yourself and your life."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You try to understand yourself.")],
        [item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You seldom get emotional."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You are not easily affected by your emotions."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You rarely notice your emotional reactions."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You experience very few emotional highs and lows."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You don't understand people who get emotional.")])


    ADVENTUROUSNESS = PersonalityTraitValue("adventurousness",
        PersonalityTrait.OPENNESS,
        [item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You prefer variety to routine."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You like to visit new places."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You are interested in many things."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You like to begin new things.")],
        [item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You prefer to stick with things that you know."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You dislike changes."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You don't like the idea of change."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You are a creature of habit."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You dislike new foods."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You are attached to conventional ways.")])


    INTELLECT = PersonalityTraitValue("intellect",
        PersonalityTrait.OPENNESS,
        [item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You like to solve complex problems."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You love to read challenging material."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You have a rich vocabulary."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You can handle a lot of information."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You enjoy thinking about things.")],
        [item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You are not interested in abstract ideas."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You avoid philosophical discussions."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You have difficulty understanding abstract ideas."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You are not interested in theoretical discussions."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You avoid difficult reading material.")])


    LIBERALISM = PersonalityTraitValue("liberalism",
        PersonalityTrait.OPENNESS,
        [item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You tend to vote for liberal political candidates."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You believe that there is no absolute right and wrong."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.POSITIVE, "You believe that criminals should receive help rather than punishment.")],
        [item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You believe in one true religion."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You tend to vote for conservative political candidates."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You believe that too much tax money goes to support artists."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You believe laws should be strictly enforced."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You believe that we coddle criminals too much."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You believe that we should be tough on crime."),
        item(PersonalityTrait.OPENNESS, KeyingDirection.NEGATIVE, "You like to stand during the national anthem.")])

    ###################################################################################################
    #                                     Agreeableness Facets                                        #
    ###################################################################################################

    TRUST = PersonalityTraitValue("trust",
        PersonalityTrait.AGREEABLENESS,
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You trust others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You believe that others have good intentions."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You trust what people say."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You believe that people are basically moral."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You believe in human goodness."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You think that all will be well.")],
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You distrust people."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You suspect hidden motives in others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You are wary of others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You believe that people are essentially evil.")])


    MORALITY = PersonalityTraitValue("morality",
        PersonalityTrait.AGREEABLENESS,
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You would never cheat on your taxes."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You stick to the rules.")],
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You use flattery to get ahead."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You use others for your own ends."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You know how to get around the rules."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You cheat to get ahead."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You put people under pressure."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You pretend to be concerned for others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You take advantage of others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You obstruct others' plans.")])


    ALTRUISM = PersonalityTraitValue("altruism",
        PersonalityTrait.AGREEABLENESS,
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You make people feel welcome."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You anticipate the needs of others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You love to help others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You are concerned about others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You have a good word for everyone.")],
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You look down on others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You are indifferent to the feelings of others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You make people feel uncomfortable."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You turn your back on others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You take no time for others.")])


    COOPERATION = PersonalityTraitValue("cooperation",
        PersonalityTrait.AGREEABLENESS,
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You are easy to satisfy."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You can't stand confrontations."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You hate to seem pushy.")],
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You have a sharp tongue."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You contradict others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You love a good fight."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You yell at people."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You insult people."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You get back at others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You hold a grudge.")])


    MODESTY = PersonalityTraitValue("modesty",
        PersonalityTrait.AGREEABLENESS,
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You dislike being the center of attention."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You dislike talking about yourself."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You consider yourself an average person."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You seldom toot your own horn.")],
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You believe that you are better than others."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You think highly of yourself."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You have a high opinion of yourself."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You know the answers to many questions."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You boast about your virtues."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You make yourself the center of attention.")])


    SYMPATHY = PersonalityTraitValue("sympathy",
        PersonalityTrait.AGREEABLENESS,
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You sympathize with the homeless."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You feel sympathy for those who are worse off than yourself."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You value cooperation over competition."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.POSITIVE, "You suffer from others' sorrows.")],
        [item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You are not interested in other people's problems."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You tend to dislike soft-hearted people."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You believe in an eye for an eye."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You try not to think about the needy."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You believe people should fend for themselves."),
        item(PersonalityTrait.AGREEABLENESS, KeyingDirection.NEGATIVE, "You can't stand weak people.")])

    ###################################################################################################
    #                                     Conscientiousness Facets                                    #
    ###################################################################################################

    SELF_EFFICACY = PersonalityTraitValue("self_efficacy",
        PersonalityTrait.CONSCIENTIOUSNESS,
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You complete tasks successfully."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You excel in what you do."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You handle tasks smoothly."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You are sure of your ground."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You come up with good solutions."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You know how to get things done.")],
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You misjudge situations."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You don't understand things."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You have little to contribute."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You don't see the consequences of things.")])


    ORDERLINESS = PersonalityTraitValue("orderliness",
        PersonalityTrait.CONSCIENTIOUSNESS,
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You like order."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You like to tidy up."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You want everything to be 'just right.'"),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You love order and regularity."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You do things according to a plan.")],
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You often forget to put things back in their proper place."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You leave a mess in your room."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You leave your belongings around."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You are not bothered by messy people."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You are not bothered by disorder.")])


    DUTIFULNESS = PersonalityTraitValue("dutifulness",
        PersonalityTrait.CONSCIENTIOUSNESS,
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You try to follow the rules."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You keep your promises."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You pay your bills on time."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You tell the truth."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You listen to your conscience.")],
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You break rules."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You break your promises."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You get others to do your duties."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You do the opposite of what is asked."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You misrepresent the facts.")])


    ACHIEVEMENT_STRIVING = PersonalityTraitValue("achievement_striving",
        PersonalityTrait.CONSCIENTIOUSNESS,
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You go straight for the goal."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You work hard."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You turn plans into actions."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You plunge into tasks with all your heart."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You do more than what's expected of you."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You set high standards for yourself and others."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You demand quality.")],
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You are not highly motivated to succeed."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You do just enough work to get by."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You put little time and effort into your work.")])


    SELF_DISCIPLINE = PersonalityTraitValue("self_discipline",
        PersonalityTrait.CONSCIENTIOUSNESS,
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You get chores done right away."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You are always prepared."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You start tasks right away."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You get to work at once."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You carry out your plans.")],
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You find it difficult to get down to work."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You waste your time."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You need a push to get started."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You have difficulty starting tasks."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You postpone decisions.")])


    CAUTIOUSNESS = PersonalityTraitValue("cautiousness",
        PersonalityTrait.CONSCIENTIOUSNESS,
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You avoid mistakes."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You choose your words with care."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.POSITIVE, "You stick to your chosen path.")],
        [item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You jump into things without thinking."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You make rash decisions."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You like to act on a whim."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You rush into things."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You do crazy things."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You act without thinking."),
        item(PersonalityTrait.CONSCIENTIOUSNESS, KeyingDirection.NEGATIVE, "You often make last-minute plans.")])



@dataclass
class Personality:
    """Personality of an agent."""
    demographic_profile: dict[DemographicCharacteristic, str] | None = None
    traits: dict[PersonalityTrait, Likert3Level] | None = None
    facets: dict[PersonalityTraitValue, KeyingDirection] | None = None