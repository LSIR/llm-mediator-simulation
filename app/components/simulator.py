import random
import time
from utils import SEED
from llm_mediator_simulation.models.dummy_model import DummyModel
from llm_mediator_simulation.models.ollama_local_server_model import OllamaLocalModel
from llm_mediator_simulation.personalities.scales import Likert7AgreementLevel
from llm_mediator_simulation.simulation.debater.config import DebaterConfig
import streamlit as st

from llm_mediator_simulation.models.gpt_models import GPTModel
from llm_mediator_simulation.simulation.debate.config import DebateConfig
from llm_mediator_simulation.simulation.debate.handler import DebateHandler
from llm_mediator_simulation.simulation.mediator.config import MediatorConfig
from llm_mediator_simulation.simulation.summary.config import SummaryConfig

def debate_simulator_page():
    # Main chat simulation
    st.header("Debate Simulator")
    st.markdown(f"**Debate Topic**: {st.session_state.debate_topic}")
    st.markdown(f"**Debater Model**: {st.session_state.debater_model}")
    st.markdown(f"**Mediator activated**: {'Yes' if st.session_state.activate_mediator else 'No'}")
    # st.markdown(f"Debaters:")
    cols = st.columns(len(st.session_state.debaters))
    for col, debater in zip(cols, st.session_state.debaters):
        with col:
            with st.expander(f"{likert7_to_emoji(debater.topic_opinion.agreement)} **{debater.name}** ({debater.topic_opinion.agreement.value.capitalize()})"):
                if debater.personality and debater.personality.demographic_profile:
                    # st.write("**Demographic Profile:**")
                    for characteristic, value in debater.personality.demographic_profile.items():
                        st.markdown(f"<li style='margin-bottom: 2px;'><b>{characteristic.value.capitalize()}:</b> {value}</li>", unsafe_allow_html=True)
                else:
                    st.write("*No demographic information available.*")


    if ("debate" not in st.session_state 
        or st.session_state.debate_topic != st.session_state.debate.config.statement
        or st.session_state.activate_mediator != (st.session_state.debate.mediator_config is not None)
        or st.session_state.num_debaters != len(st.session_state.debate.debaters)
        or st.session_state.debater_model != st.session_state.debate.debater_model.model_name
        or any(
            debater != st.session_state.debate.debaters[i].config
            for i, debater in enumerate(st.session_state.debaters)
        )   
    ): 
        initialize_debate()

    chat_container = st.container()
    with chat_container:
        for intervention in st.session_state.debate.interventions:
            if intervention.text:
                col1, col2 = st.columns([1, 9])
                with col1:
                    #st.image("https://i.redd.it/67xiprjzuzod1.png", width=50)
                    pass
                with col2:
                    if intervention.debater:
                        st.markdown(f"👤 **{intervention.debater.name}**: {intervention.text}")
                    else: # Mediator
                        st.markdown(f"🤝 **Mediator**: {intervention.text}")
    # Chat controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Simulate single round"):
            st.session_state.remaining_rounds = 1
            
    with col2:
        st.button("Reset Chat", on_click=reset_chat, kwargs={"debate": st.session_state})

    with col3:
        rounds = st.number_input("Number of rounds", min_value=1, value=3)
        if st.button("Simulate multiple rounds"):
            st.session_state.remaining_rounds = rounds

    
    if st.session_state.remaining_rounds:
        st.session_state.debate.run(rounds=1)
        # Refresh the interface
        st.session_state.remaining_rounds -= 1
        st.rerun()

def reset_chat(debate: DebateHandler):
    random.seed(SEED)
    st.session_state.debate.preload_chat(debaters=st.session_state.debaters, interventions=[])

# Convert Likert7AgreementLevel to emojis
def likert7_to_emoji(level: Likert7AgreementLevel) -> str:
    if level == Likert7AgreementLevel.STRONGLY_DISAGREE:
        return "❌❌❌"
    elif level == Likert7AgreementLevel.DISAGREE:
        return "❌❌"
    elif level == Likert7AgreementLevel.SLIGHTLY_DISAGREE:
        return "❌"
    elif level == Likert7AgreementLevel.NEUTRAL:
        return "🤷"
    elif level == Likert7AgreementLevel.SLIGHTLY_AGREE:
        return "✅"
    elif level == Likert7AgreementLevel.AGREE:
        return "✅✅"
    elif level == Likert7AgreementLevel.STRONGLY_AGREE:
        return "✅✅✅"
    

def initialize_debate():
    summary_config = SummaryConfig(latest_messages_limit=5, debaters=st.session_state.debaters)

    debate_config = DebateConfig(
        statement=st.session_state.debate_topic,
    )

    mediator_model =  GPTModel(api_key=st.session_state.api_key, model_name="gpt-4o", seed=SEED)
    #debater_model = DummyModel() #  # GPTModel(api_key=st.session_state.api_key, model_name="gpt-4o", seed=SEED)
    debater_model = OllamaLocalModel(model_name = st.session_state.debater_model)

    if st.session_state.activate_mediator:
        mediator_config = MediatorConfig()
    else:
        mediator_config = None

    st.session_state.debate = DebateHandler(
        debater_model=debater_model,
        mediator_model=mediator_model,
        debaters=st.session_state.debaters,
        config=debate_config,
        summary_config=summary_config,
        metrics_handler=None,
        mediator_config=mediator_config,
        seed=SEED,
        variable_personality=False,
    )
