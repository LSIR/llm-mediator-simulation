import streamlit as st
from openai_key_manager import remove_api_key, save_api_key


def settings_page():
    st.write("#### 🔑 OpenAI API Key")
    st.header("Settings")
    openai_key_prompt = "No API key? No problem! " "Create one easily [here](https://platform.openai.com/account/api-keys)"
    api_key_placeholder = "Paste your OpenAI API Key here (sk-...)"

    if st.session_state.show_key_input:
        st.info(openai_key_prompt, icon="ℹ️")

        cols = st.columns([0.8, 0.2], vertical_alignment="bottom")
        show_error = False

        with cols[0]:
            st.session_state.api_key =st.text_input(
                        "Enter your OpenAI API Key",
                        key="user_openai_key_input",
                        type="password",
                        autocomplete="current-password",
                        placeholder=api_key_placeholder,
                        label_visibility="collapsed",
                    )
        with cols[1]:
            if st.button("Save", on_click=save_api_key, use_container_width=True):
                show_error = True

        if show_error:
            st.error("Please provide a valid OpenAI key.", icon="🚨")

    if st.session_state.key_verified:
        st.success("API Key verified and saved.", icon="✅")
        api_key = st.session_state.api_key
        preview_api_key = api_key[:14] + "*" * (len(api_key) - 14)

        if len(preview_api_key) > 150:
            preview_api_key = preview_api_key[:150]

        st.write("**API Key:**", preview_api_key)
        st.button("Remove Key", on_click=remove_api_key, type="primary")