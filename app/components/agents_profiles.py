import streamlit as st

# Detailed agent profiles page
from utils import MODELS, get_debater_profile
from llm_mediator_simulation.personalities.scales import Likert3Level, Likert7AgreementLevel


def agent_profiles_page():
    st.header("Debate Settings")
    st.subheader("Topic Configuration")
    debate_topic = st.text_input("Set Debate Topic", value=st.session_state.debate_topic)
    st.session_state.debate_topic = debate_topic

    st.subheader("Debaters Configuration")
    st.session_state.num_debaters = st.slider("Number of Debaters", min_value=2, max_value=5, value=st.session_state.num_debaters)
    if len(st.session_state.debaters) != st.session_state.num_debaters:
        st.session_state.debaters = [get_debater_profile(i) for i in range(st.session_state.num_debaters)]

    st.session_state.debater_model = st.selectbox("Debater Model", options=MODELS, index=MODELS.index(st.session_state.debater_model))


    col1, col2, col3 = st.columns([1, 1, 1])
    for i, debater in enumerate(st.session_state.debaters):
        with col1:
            with st.expander(f"**Debater {i + 1}'s profile**", expanded=True):
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