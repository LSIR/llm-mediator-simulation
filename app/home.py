import os
from dotenv import load_dotenv
from components.settings_page import settings_page
import streamlit as st
import random

from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.personalities.demographics import DemographicCharacteristic
from llm_mediator_simulation.personalities.personality import Personality
from llm_mediator_simulation.personalities.scales import Likert3Level, Likert7AgreementLevel
from llm_mediator_simulation.personalities.traits import PersonalityTrait
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.simulation.debater.config import DebaterConfig, TopicOpinion
from llm_mediator_simulation.simulation.summary.config import SummaryConfig

from openai_key_manager import save_api_key

load_dotenv()
gpt_key = os.getenv("GPT_API_KEY") or ""

SEED = 42
random.seed(SEED)
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
# st.title("Debate Simulator")

# Detailed agent profiles page
def agent_profiles_page():
    st.header("Debate Settings")
    st.subheader("Topic Configuration")
    debate_topic = st.text_input("Set Debate Topic", value=st.session_state.debate_topic)
    st.session_state.debate_topic = debate_topic
    
    st.subheader("Debaters Configuration")
    st.session_state.num_debaters = st.sidebar.slider("Number of Debaters", min_value=2, max_value=5, value=2)

    col1, col2, col3 = st.columns([1, 1, 1])
    for i, debater in enumerate(st.session_state.debaters):
        with col1:
            with st.expander(f"**Debater {i + 1}'s profile**", expanded=True):
                st.markdown("Name")
                debater.name = st.text_input(f"Name for Debater {i+1}", value=debater.name)
                if st.button(f"Show Debater {i + 1}'s personality"):
                    with col2:
                        st.markdown(f"Debater {i + 1}'s ({debater.name}) demographic profile")
                        for characteristic_name, characteristic_value in debater.personality.demographic_profile.items():
                            debater.personality.demographic_profile[characteristic_name] = st.text_input(f"Debater {i + 1}'s {characteristic_name.value}", 
                                                                                                        value=characteristic_value)
                    with col3:
                        st.markdown(f"Debater {i + 1}'s ({debater.name}) psychological traits")
                        for trait_name, trait_value in debater.personality.traits.items():
                            debater.personality.traits[trait_name] = st.selectbox(f"Debater {i + 1}'s {trait_name.value.name}", 
                                                                                options=[Likert3Level.LOW, Likert3Level.AVERAGE, Likert3Level.HIGH],
                                                                                format_func=lambda x: x.value.capitalize(),
                                                                                index=list(Likert3Level).index(trait_value))

                debater.topic_opinion.agreement = st.selectbox(f"Debater {i + 1}'s opinion on the debate topic",
                                                        options=list(Likert7AgreementLevel),
                                                        format_func=lambda x: x.value.capitalize(),
                                                        index=list(Likert7AgreementLevel).index(debater.topic_opinion.agreement))

# Settings page for API key
def debate_simulator_page(): 
    # Main chat simulation
    st.header("Debate Simulator")

    summary_config = SummaryConfig(latest_messages_limit=5, debaters=st.session_state.debaters)
    
    debate_config = DebateConfig(
        statement=st.session_state.debate_topic,
    )   

    mediator_model =  GPTModel(api_key=st.session_state.api_key, model_name="gpt-4o", seed=SEED)
    debater_model =  GPTModel(api_key=st.session_state.api_key, model_name="gpt-4o", seed=SEED)

    if "interventions" not in st.session_state:
        st.session_state.interventions = []
    
    debate = DebateHandler(
        debater_model=debater_model,
        mediator_model=mediator_model,
        debaters=st.session_state.debaters,
        config=debate_config,
        summary_config=summary_config,
        metrics_handler=None,
        mediator_config=None,
        seed=SEED,
        variable_personality=False,
    )

    chat_container = st.container()
    with chat_container:
        for intervention in st.session_state.interventions:
            col1, col2 = st.columns([1, 9])
            with col1:
                #st.image("https://i.redd.it/67xiprjzuzod1.png", width=50)
                pass
            with col2:
                st.markdown(f"**{intervention.debater.name}**: {intervention.text}")

    # Chat controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Simulate single round"):
            debate.run(rounds=1)
            st.session_state.interventions.append(debate.interventions[-1])

    with col2:
        if st.button("Reset Chat"):
            reset_chat(debate)

    with col3:
        rounds = st.number_input("Number of rounds", min_value=1, value=3)
        if st.button("Simulate multiple rounds"):
            debate.run(rounds=rounds)
            st.session_state.interventions.extend(debate.interventions[-rounds:])
            

def reset_chat(debate):
    debate = DebateHandler(
        debater_model=debate.debater_model,
        mediator_model=debate.mediator_model,
        debaters=st.session_state.debaters,
        config=debate.config,
        summary_config=debate.summary_config,
        metrics_handler=None,
        mediator_config=None,
        seed=SEED,
        variable_personality=False,
    )







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

# Real-time update loop (optional, uncomment to auto-update)
# while True:
#     simulate_message()
#     time.sleep(1)
#     st.experimental_rerun()