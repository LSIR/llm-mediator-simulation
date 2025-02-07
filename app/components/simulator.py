import random

from numpy import copy
from llm_mediator_simulation.metrics.metrics_handler import MetricsHandler
from utils import SEED, flip_debate_type, flip_metric, streamlit_plot_metrics
from llm_mediator_simulation.models.dummy_model import DummyModel
from llm_mediator_simulation.models.ollama_local_server_model import OllamaLocalModel
from llm_mediator_simulation.personalities.scales import Likert7AgreementLevel
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
    
    checks = st.columns([1, 5, 5])
    with checks[0]:
        st.markdown("**Debate Type**: Select the debate types you want to simulate, selecting both is valid.")
    with checks[1]:
        st.checkbox("Unmediated", value=st.session_state.unmediated, 
                        on_change=flip_debate_type,
                        kwargs={"debate_type": "unmediated"})
    with checks[2]:
        st.checkbox("Mediated", value=st.session_state.mediated, 
                        on_change=flip_debate_type,
                        kwargs={"debate_type": "mediated"})

    if not st.session_state.unmediated and not st.session_state.mediated:
        if "unmediated_debate" in st.session_state:
            del st.session_state.unmediated_debate
        if "mediated_debate" in st.session_state:
            del st.session_state.mediated_debate
        st.warning("Please select at least one debate type.")
    
    metrics_num = len(st.session_state.metrics)
    metrics_columns = st.columns([1] + [int(10/metrics_num)]*metrics_num)

    with metrics_columns[0]:
        st.markdown("**Metrics**")
    for i, k in enumerate(st.session_state.metrics):
        with metrics_columns[i+1]:
            st.checkbox(k.value[0], value=st.session_state.metrics[k], 
                        key=f"check_{k.value[0]}", 
                        on_change=flip_metric, 
                        kwargs={"metric": k})

    st.markdown("**Debater profiles**")
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

    # print(st.session_state.metrics)
    # print((st.session_state.unmediated_debate.metrics_handler is not None))
    # print(st.session_state.unmediated_debate.metrics_handler.argument_qualities)
    if ((
         st.session_state.unmediated or st.session_state.mediated
        ) 
        and
        (
            "unmediated_debate" not in st.session_state 
            or st.session_state.debate_topic != st.session_state.unmediated_debate.config.statement
            or st.session_state.num_debaters != len(st.session_state.unmediated_debate.debaters)
            or st.session_state.debater_model != st.session_state.unmediated_debate.debater_model.model_name
            or any(
                debater != st.session_state.unmediated_debate.debaters[i].config
                for i, debater in enumerate(st.session_state.debaters)
            )
            or (st.session_state.unmediated_debate.metrics_handler is None) and any(
                st.session_state.metrics[metric] for metric in st.session_state.metrics)
            or (st.session_state.unmediated_debate.metrics_handler is not None) and all(
                not(st.session_state.metrics[metric]) for metric in st.session_state.metrics)
            or (
                    st.session_state.unmediated_debate.metrics_handler is not None
                and
                    (
                    len([metric for metric in st.session_state.metrics if st.session_state.metrics[metric]]) 
                    != len(st.session_state.unmediated_debate.metrics_handler.argument_qualities)
                    )
                )
            or any(
                metric != st.session_state.unmediated_debate.metrics_handler.argument_qualities[i]
                for i, metric in enumerate([m for m in st.session_state.metrics if st.session_state.metrics[m]])
            )   
        )
    ): 
        print("Initializing debate")
        initialize_debate()

    if any(st.session_state.metrics.values()):
        metric_to_display = st.selectbox("**Display metric**", [metric for metric in st.session_state.metrics if st.session_state.metrics[metric]])

    if st.session_state.unmediated or st.session_state.mediated:
        debate_cols = st.columns([10, 1, 10, 1])

        debates = [st.session_state.unmediated_debate, 
                st.session_state.mediated_debate]
        
        headers = ["Without mediator", "With mediator"]
        counters = [1, 1]
        
        for i in range(2):
            debate = debates[i]
            with debate_cols[2*i]:
                # Make the subheader centered
                # st.subheader(f"{headers[i]}")
                st.markdown(f"<h2 style='text-align: center; color: grey;'>{headers[i]}</h2>", unsafe_allow_html=True)
                if any(value for value in st.session_state.metrics.values()):
                    streamlit_plot_metrics(debate, metric_to_display)
                for intervention in debate.interventions:
                    if intervention.text:
                        if intervention.debater:
                            st.markdown(f"👤 ({counters[i]}) **{intervention.debater.name}**: {intervention.text}")
                            counters[i] += 1
                        else: # Mediator
                            st.markdown(f"🤝 **Mediator**: {intervention.text}")
        
            # with debate_cols[2*i+1]:
            #     if st.session_state.metrics:
            #         st.subheader(f"")
            #         for intervention in debate.interventions:
            #             if intervention.metrics:
            #                 for argument_quality, value in intervention.metrics.argument_qualities.items():
            #                     st.markdown(f"**{argument_quality.value[0]}**: {value.value}")
        
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
                                   ignore=True)

    debate_config = DebateConfig(
        statement=st.session_state.debate_topic,
    )

    mediator_model =  GPTModel(api_key=st.session_state.api_key, model_name="gpt-4o")
    # debater_model = DummyModel(model_name = st.session_state.debater_model) #  # GPTModel(api_key=st.session_state.api_key, model_name="gpt-4o", seed=SEED)
    debater_model = OllamaLocalModel(model_name = st.session_state.debater_model)

    argument_qualities = []

    for metric, value in st.session_state.metrics.items():
        if value:
            argument_qualities.append(metric)

    if argument_qualities:
        metrics = MetricsHandler(
            model=mediator_model,
            argument_qualities=argument_qualities,)
    else:
        metrics = None

    st.session_state.unmediated_debate = DebateHandler(
        debater_model=debater_model,
        mediator_model=mediator_model,
        debaters=st.session_state.debaters,
        config=debate_config,
        summary_config=summary_config,
        metrics_handler=metrics,
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
        metrics_handler=metrics,
        mediator_config=MediatorConfig(),
        seed=SEED,
        variable_personality=False,
    )



    
