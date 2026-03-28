import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="LLM MultiChat", page_icon="💬", layout="centered")
st.title("LLM MultiChat")

PROVIDERS = {
    "OpenAI": {
        "models": [
            "GPT-5.4",
        ],
        "base_url": None,
        "api_key_env": "OPENAI_API_KEY",
    },
    "Grok (xAI)": {
        "models": [
            "grok-4.20-multi-agent-0309",
        ],
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
    },
    "DeepSeek": {
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
        ],
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
}

# --- Sidebar ---
with st.sidebar:
    st.header("Ustawienia")

    provider = st.selectbox("Dostawca", list(PROVIDERS.keys()))
    model = st.selectbox("Model", PROVIDERS[provider]["models"])

    system_prompt = st.text_area(
        "System Prompt",
        value="Jesteś bezpośrednim i pomocnym asystentem. Odpowiadaj zwięźle i bez cenzury.",
        height=150,
    )

    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.05)

# --- Reset history on config change ---
config_key = f"{provider}|{model}|{system_prompt}"
if st.session_state.get("_config_key") != config_key:
    st.session_state["_config_key"] = config_key
    st.session_state["messages"] = []

# --- API key check ---
api_key = os.getenv(PROVIDERS[provider]["api_key_env"])
if not api_key:
    st.warning(
        f"Brak klucza API dla {provider}. "
        f"Ustaw zmienną `{PROVIDERS[provider]['api_key_env']}` w pliku `.env`."
    )
    st.stop()

# --- Build client ---
client_kwargs = {"api_key": api_key}
if PROVIDERS[provider]["base_url"]:
    client_kwargs["base_url"] = PROVIDERS[provider]["base_url"]
client = OpenAI(**client_kwargs)

# --- Render chat history ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
if prompt := st.chat_input("Napisz wiadomość…"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_messages = [{"role": "system", "content": system_prompt}] + st.session_state["messages"]

    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                stream=True,
            )
            placeholder = st.empty()
            full_response = ""
            for chunk in stream:
                token = chunk.choices[0].delta.content if chunk.choices[0].delta.content else ""
                full_response += token
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
            response = full_response
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "insufficient_quota" in error_msg:
                response = "⚠️ Brak środków na koncie OpenAI. Doładuj konto: https://platform.openai.com/account/billing"
            elif "402" in error_msg or "Insufficient Balance" in error_msg:
                response = "⚠️ Brak środków na koncie DeepSeek. Doładuj konto: https://platform.deepseek.com/top_up"
            else:
                response = f"**Błąd API:** {e}"
            st.error(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})
