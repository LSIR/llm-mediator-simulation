import os
from dotenv import load_dotenv
from components.simulator import debate_simulator_page
from components.agents_profiles import agent_profiles_page
from components.settings import settings_page
import streamlit as st
import random

from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import Likert3Level, Likert7AgreementLevel
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.simulation.debater.config import DebaterConfig, TopicOpinion

from openai_key_manager import save_api_key

load_dotenv()
gpt_key = os.getenv("GPT_API_KEY") or ""

def init_session_state_vars():
    
    if "api_key" not in st.session_state:
        st.session_state.api_key = gpt_key

    if "key_verified" not in st.session_state:
        st.session_state.key_verified = False

    if "show_key_input" not in st.session_state:
        st.session_state.show_key_input = True

    if "debate_topic" not in st.session_state:
        st.session_state.debate_topic = "Abortion should me made illegal."

    if "num_debaters" not in st.session_state:
        st.session_state.num_debaters = 2

    if "debaters" not in st.session_state:
        st.session_state.debaters = [get_debater_profile(i) for i in range(st.session_state.num_debaters)]

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
                             DemographicCharacteristic.INCOME: str(random.randint(0, 100000)),
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
    
UNISEXNAMES = ["Alex", "Riley", "Jordan", "Parker", "Sawyer", "Taylor", "Casey", "Avery", "Jamie", "Quinn"]

init_session_state_vars()

if gpt_key:
    save_api_key()

# Streamlit app configuration
st.set_page_config(page_title="Debate Simulator", layout="wide")


# Page navigation
# st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to",
                        ["Debate Simulator", "Agent Profiles", "Settings"])

if page == "Agent Profiles":
    agent_profiles_page()
elif page == "Debate Simulator":
    debate_simulator_page()
elif page == "Settings":
    settings_page()