import streamlit as st
import openai


def verify_api_key(api_key):
    """
    Verify the OpenAI API key by checking if the user can list the models.

    Args:
        - api_key (str): The API key to verify.
    Returns:
        - bool: True if the key is valid, False otherwise.
    """
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True


def save_api_key():
    """
    Save the OpenAI API key to the session state.
    """
    if "user_openai_key_input" in st.session_state:
        api_key = st.session_state.user_openai_key_input 
    else:
        api_key = st.session_state.api_key

    if verify_api_key(api_key):
        st.session_state.api_key = api_key
        st.session_state.show_key_input = False
        st.session_state.key_verified = True


def remove_api_key():
    """
    Remove the OpenAI API key from the session state.
    """
    st.session_state.api_key = ""
    st.session_state.show_key_input = True
    st.session_state.key_verified = False