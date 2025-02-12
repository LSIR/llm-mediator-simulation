import os
import random
from dotenv import load_dotenv
from llm_mediator_simulation.metrics.criteria import ArgumentQuality
from utils import SEED, get_predefined_debater_profile
from utils import get_random_debater_profile
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
        st.session_state.debate_topic = "Companies and governments should end their DEI programs."

    if "num_debaters" not in st.session_state:
        st.session_state.num_debaters = 2

    if "debaters" not in st.session_state:
        st.session_state.debaters = [get_predefined_debater_profile(i) for i in range(st.session_state.num_debaters)]

    if "debater_model" not in st.session_state:
        st.session_state.debater_model = "mistral-nemo"

    if "remaining_rounds" not in st.session_state:
        st.session_state.remaining_rounds = 0

    if "unmediated" not in st.session_state:
        st.session_state.unmediated = True

    if "mediated" not in st.session_state:
        st.session_state.mediated = False

    if "metrics" not in st.session_state:
        st.session_state.metrics = {ArgumentQuality.APPROPRIATENESS: False,
                                    ArgumentQuality.CLARITY: False,
                                    ArgumentQuality.EMOTIONAL_APPEAL: False}
    


init_session_state_vars()

if gpt_key:
    save_api_key()

# Streamlit app configuration
st.set_page_config(page_title="Debate Simulator Prototype", layout="wide")


# Add EPFL logo
left_co, cent_co, last_co = st.sidebar.columns([1, 3, 1])
cent_co.image("app/images/EPFL_Logo_184X53 2.svg", width=100)

# Add My Lab's QR Code at the bottom
perso_cols = st.sidebar.columns([1, 1])
perso_cols[1].markdown("Distributed Information Systems Laboratory")
perso_cols[0].image("app/images/qr-code-lsir.svg", width=100)


st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")


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

st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")
st.sidebar.text(" \n")
# Add my personal web page QR code at the bottom
perso_cols = st.sidebar.columns([1, 1])
perso_cols[1].text(" \n")
perso_cols[1].header("Leo Laugier")
perso_cols[0].image("app/images/qr-code-site-perso.svg", width=100)
