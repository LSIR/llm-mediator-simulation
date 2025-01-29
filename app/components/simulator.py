# Settings page for API key
import random
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
import streamlit as st

from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.simulation.summary.config import SummaryConfig

SEED = 42
random.seed(SEED)

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
        st.button("Simulate single round", on_click=run_debate, kwargs={"debate": debate, 
                                                                       "rounds": 1})

    with col2:
        st.button("Reset Chat", on_click=reset_chat, kwargs={"debate": debate})

    with col3:
        rounds = st.number_input("Number of rounds", min_value=1, value=3)
        st.button("Simulate multiple rounds", on_click=run_debate, kwargs={"debate": debate, 
                                                                       "rounds": rounds})

def run_debate(debate: DebateHandler, rounds: int=1):
    debate.run(rounds=rounds)
    st.session_state.interventions.extend(debate.interventions[:-rounds])

def reset_chat(debate: DebateHandler):
    st.session_state.interventions = []
    debate.preload_chat(debaters=st.session_state.debaters, interventions=[])