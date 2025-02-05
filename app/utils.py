from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import Likert3Level, Likert7AgreementLevel
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debater.config import DebaterConfig, TopicOpinion


import random

MODELS = ["deepseek-r1:8b", "deepseek-r1:1.5b", "llama3.2", "mistral", "mistral-nemo"]

UNISEXNAMES = ["Alex", "Riley", "Jordan", "Parker", "Sawyer", "Taylor", "Casey", "Avery", "Jamie", "Quinn"]


def get_debater_profile(agent_num):
    name = UNISEXNAMES[agent_num]
    # avatar = st.sidebar.text_input(f"Agent {agent_num} Avatar URL", value=f"https://via.placeholder.com/50?text=A{agent_num}")
    # return {"name": name} # , "avatar": avatar}

    personality = Personality(
        demographic_profile={DemographicCharacteristic.ETHNICITY: random.choice(["White", "Black", "Asian", "Hispanic"]),
                             DemographicCharacteristic.BIOLOGICAL_SEX: random.choice(["Male", "Female"]),
                             DemographicCharacteristic.NATIONALITY: random.choice(["American"]),
                             DemographicCharacteristic.AGE: str(random.randint(18, 80)),
                             DemographicCharacteristic.MARITAL_STATUS: random.choice(["Single", "Married", "Divorced", "Widowed"]),
                             DemographicCharacteristic.EDUCATION: random.choice(["High School", "College", "Graduate School"]),
                             DemographicCharacteristic.OCCUPATION: random.choice(["Engineer", "Doctor", "Teacher", "Artist", "Unemployed", "Student"]),
                             DemographicCharacteristic.POLITICAL_LEANING: random.choice(["Democrat", "Republican", "Independent"]),
                             DemographicCharacteristic.RELIGION: random.choice(["Christian", "Muslim", "Jewish", "Atheist"]),
                             DemographicCharacteristic.SEXUAL_ORIENTATION: random.choice(["Heterosexual", "Homosexual", "Bisexual"]),
                             DemographicCharacteristic.HEALTH_CONDITION: random.choice(["Disabled", "Non-disabled"]),
                             DemographicCharacteristic.INCOME: str(random.randint(30000, 100000)),
                             DemographicCharacteristic.HOUSEHOLD_SIZE: str(random.randint(1, 5)),
                             DemographicCharacteristic.NUMBER_OF_DEPENDENT: str(random.randint(0, 3)),
                             DemographicCharacteristic.LIVING_QUARTERS: random.choice(["House", "Apartment", "Condo"]),
                             DemographicCharacteristic.CITY_OF_RESIDENCE: random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]),
                             DemographicCharacteristic.PRIMARY_MODE_OF_TRANSPORTATION: random.choice(["Car", "Public Transit", "Bicycle", "Walking"]),
                             }
    )

    traits = {}
    random_traits = random.sample(list(PersonalityTrait), random.randint(1, 5))
    for trait in random_traits:
        traits[trait] = random.choice([Likert3Level.LOW, Likert3Level.AVERAGE, Likert3Level.HIGH])

    personality.traits = traits
    return DebaterConfig(name=name,
                         topic_opinion=TopicOpinion(agreement=random.choice(list(Likert7AgreementLevel))),
                         personality = personality)


SEED = 42