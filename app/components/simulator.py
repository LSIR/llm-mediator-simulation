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
    
    st.markdown("**Debate Type**: Select the debate types you want to simulate, selecting both is valid.")
    #checkboxes for st.session_state.unmediated and st.session_state.mediated
    st.session_state.unmediated = st.checkbox("Unmediated", st.session_state.unmediated)
    st.session_state.mediated = st.checkbox("Mediated", st.session_state.mediated)
    
    if not st.session_state.unmediated and not st.session_state.mediated:
        st.warning("Please select at least one debate type.")


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


    if ("unmediated_debate" not in st.session_state 
        or st.session_state.debate_topic != st.session_state.unmediated_debate.config.statement
        or st.session_state.num_debaters != len(st.session_state.unmediated_debate.debaters)
        or st.session_state.debater_model != st.session_state.unmediated_debate.debater_model.model_name
        or any(
            debater != st.session_state.unmediated_debate.debaters[i].config
            for i, debater in enumerate(st.session_state.debaters)
        )   
    ): 
        initialize_debate()

    debate_cols = st.columns(2)
    
    for col, debate, type in zip(debate_cols, 
                           [st.session_state.unmediated_debate, 
                            st.session_state.mediated_debate],
                            ["Without mediator", "With mediator"]):
        with col:
            st.subheader(f"{type}")
            # with chat_container:
            for intervention in debate.interventions:
                if intervention.text:
                    if intervention.debater:
                        st.markdown(f"👤 **{intervention.debater.name}**: {intervention.text}")
                    else: # Mediator
                        st.markdown(f"🤝 **Mediator**: {intervention.text}")
    # Chat controls
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Simulate single round", 
                     disabled=not (st.session_state.unmediated or st.session_state.mediated)):
            st.session_state.remaining_rounds = 1
            
    with col2:
        st.button("Reset Chat", on_click=reset_chat,
                  disabled=not (st.session_state.unmediated or st.session_state.mediated))

    with col3:
        rounds = st.number_input("Number of rounds", min_value=1, value=3)
        if st.button("Simulate multiple rounds",
                     disabled=not (st.session_state.unmediated or st.session_state.mediated)):
            st.session_state.remaining_rounds = rounds

    
    if st.session_state.remaining_rounds:
        if st.session_state.unmediated:
            st.session_state.unmediated_debate.run(rounds=1)
        if st.session_state.mediated:
            st.session_state.mediated_debate.run(rounds=1)
        # Refresh the interface
        st.session_state.remaining_rounds -= 1
        st.rerun()

def reset_chat():
    random.seed(SEED)
    st.session_state.unmediated_debate.preload_chat(debaters=st.session_state.debaters, interventions=[])
    st.session_state.mediated_debate.preload_chat(debaters=st.session_state.debaters, interventions=[])

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
    summary_config = SummaryConfig(latest_messages_limit=5, 
                                   debaters=st.session_state.debaters,
                                   ignore=False)

    debate_config = DebateConfig(
        statement=st.session_state.debate_topic,
    )

    mediator_model =  GPTModel(api_key=st.session_state.api_key, model_name="gpt-4o")
    # debater_model = DummyModel(model_name = st.session_state.debater_model) #  # GPTModel(api_key=st.session_state.api_key, model_name="gpt-4o", seed=SEED)
    debater_model = OllamaLocalModel(model_name = st.session_state.debater_model)

    st.session_state.unmediated_debate = DebateHandler(
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

    st.session_state.mediated_debate = DebateHandler(
        debater_model=debater_model,
        mediator_model=mediator_model,
        debaters=st.session_state.debaters,
        config=debate_config,
        summary_config=summary_config,
        metrics_handler=None,
        mediator_config=MediatorConfig(),
        seed=SEED,
        variable_personality=False,
    )



    
