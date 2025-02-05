import os
import random
from dotenv import load_dotenv
from utils import SEED
from utils import get_debater_profile
from components.simulator import debate_simulator_page
from components.agents_profiles import agent_profiles_page
from components.settings import settings_page
import streamlit as st

from llm_mediator_simulation.simulation.debate.handler import DebateHandler

from openai_key_manager import save_api_key

load_dotenv()
gpt_key = os.getenv("GPT_API_KEY") or ""

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

    if "debater_model" not in st.session_state:
        st.session_state.debater_model = "mistral-nemo"

    if "remaining_rounds" not in st.session_state:
        st.session_state.remaining_rounds = 0

    if "unmediated" not in st.session_state:
        st.session_state.unmediated = True

    if "mediated" not in st.session_state:
        st.session_state.mediated = False
    


init_session_state_vars()

if gpt_key:
    save_api_key()

# Streamlit app configuration
st.set_page_config(page_title="Debate Simulator", layout="wide")


# Page navigation
# st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to",
                        ["Debate Simulator", "Debate Settings", "OpenAI API Settings"])

if page == "Debate Settings":
    agent_profiles_page()
elif page == "Debate Simulator":
    debate_simulator_page()
elif page == "OpenAI API Settings":
    settings_page()