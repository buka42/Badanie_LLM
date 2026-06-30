import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
import os
import json
import html as html_lib

import image_core as core

load_dotenv()

st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="centered")
st.title("🎨 AI Image Generator")
st.caption("Generate images with OpenAI, Google Gemini, or Grok (xAI) — straight from their official APIs.")

PROVIDERS = core.PROVIDERS


def render_copyable(text: str):
    """Render a textarea with the reasoning summary plus a copy button."""
    safe_html = html_lib.escape(text)
    safe_js = json.dumps(text)
    component = f"""
<div style="font-family: 'Source Sans Pro', sans-serif;">
  <textarea readonly style="width:100%;height:180px;padding:10px;border-radius:8px;
    border:1px solid #ccc;font-size:13px;box-sizing:border-box;resize:vertical;">{safe_html}</textarea>
  <button id="copyBtn" style="margin-top:8px;padding:6px 14px;border:none;border-radius:8px;
    background:#FF4B4B;color:white;font-size:14px;cursor:pointer;">📋 Kopiuj</button>
</div>
<script>
  const btn = document.getElementById('copyBtn');
  btn.addEventListener('click', async () => {{
    try {{
      await navigator.clipboard.writeText({safe_js});
    }} catch (e) {{
      const ta = document.querySelector('textarea');
      ta.select();
      document.execCommand('copy');
    }}
    btn.innerText = '✅ Skopiowano';
    setTimeout(() => {{ btn.innerText = '📋 Kopiuj'; }}, 2000);
  }});
</script>
"""
    components.html(component, height=250)


# --- Sidebar ---
with st.sidebar:
    st.header("Settings")

    provider = st.selectbox("Provider", list(PROVIDERS.keys()))
    model = st.selectbox("Model", PROVIDERS[provider]["models"])

    custom_model = st.text_input(
        "Custom model (optional)",
        placeholder="override the model ID",
        help="If filled, this exact model ID is sent instead of the one selected above.",
    ).strip()

    # Provider/model specific options.
    size = "1024x1024"
    quality = "auto"
    aspect_ratio = "1:1"
    max_n = 1

    if provider == "OpenAI":
        if model == "dall-e-3":
            size = st.selectbox("Size", ["1024x1024", "1792x1024", "1024x1792"])
            quality = st.selectbox("Quality", ["standard", "hd"])
            max_n = 1  # dall-e-3 only supports a single image per request
        elif model == "dall-e-2":
            size = st.selectbox("Size", ["1024x1024", "512x512", "256x256"])
            max_n = 10
        else:  # gpt-image family
            size = st.selectbox("Size", ["1024x1024", "1536x1024", "1024x1536", "auto"])
            quality = st.selectbox("Quality", ["auto", "low", "medium", "high"])
            max_n = 10
    elif provider == "Gemini (Google)":
        if model.startswith("imagen"):
            aspect_ratio = st.selectbox("Aspect ratio", ["1:1", "3:4", "4:3", "9:16", "16:9"])
            max_n = 4
        else:
            max_n = 1  # native Gemini image gen returns a single image
    elif provider == "Grok (xAI)":
        max_n = 10

    if max_n > 1:
        n = st.slider("Number of images", 1, max_n, 1)
    else:
        n = 1
        st.caption("This model generates 1 image per request.")

    # --- Optional reasoning / thinking mode (off by default) ---
    st.divider()
    st.subheader("🧠 Reasoning (optional)")
    thinking_mode = st.checkbox(
        "OpenAI thinking / reasoning",
        value=False,
        help="Uses an OpenAI reasoning model to craft a more detailed prompt before "
             "generating the image. More accurate, but slower and more expensive. "
             "Requires OPENAI_API_KEY.",
    )
    default_effort = core.get_reasoning_effort()
    effort = default_effort
    if thinking_mode:
        st.caption("⚠️ Slower & more expensive — uses `OPENAI_API_KEY` to refine the prompt first.")
        effort = st.selectbox(
            "Reasoning effort",
            core.REASONING_EFFORTS,
            index=core.REASONING_EFFORTS.index(default_effort)
            if default_effort in core.REASONING_EFFORTS else 1,
        )

# Resolve the actual model id to send.
active_model = custom_model or model

# --- API key check (default flow, unchanged) ---
api_key = os.getenv(PROVIDERS[provider]["api_key_env"])
if not api_key:
    st.warning(
        f"No API key for {provider}. "
        f"Set `{PROVIDERS[provider]['api_key_env']}` in your `.env` file."
    )
    st.stop()

if provider == "Gemini (Google)" and not core.GENAI_AVAILABLE:
    st.error("The `google-genai` package is not installed. Run: `pip install google-genai`")
    st.stop()

# --- Main: prompt + generate ---
prompt = st.text_area(
    "Prompt",
    placeholder="A photorealistic red panda coding on a laptop, cinematic lighting",
    height=120,
)

if st.button("Generate", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        proceed = True
        image_prompt = prompt
        reasoning_summary = ""

        # Optional thinking step: refine the prompt with an OpenAI reasoning model.
        # Failures here are reported but never break the standard generation flow.
        if thinking_mode:
            openai_key = os.getenv("OPENAI_API_KEY")
            if not openai_key:
                st.error("🧠 OpenAI thinking mode requires `OPENAI_API_KEY` in your `.env` file. "
                         "Uncheck it to use the standard mode.")
                proceed = False
            else:
                with st.spinner("🧠 Thinking about the best prompt…"):
                    try:
                        refined, reasoning_summary = core.refine_image_prompt(
                            openai_key, prompt, effort=effort
                        )
                        image_prompt = refined or prompt
                    except Exception as e:
                        st.error("🧠 OpenAI thinking failed: " + core.friendly_error(e))
                        proceed = False

        if proceed:
            with st.spinner(f"Generating with {active_model}…"):
                try:
                    if provider == "OpenAI":
                        images = core.generate_openai(api_key, active_model, image_prompt, n, size, quality)
                    elif provider == "Gemini (Google)":
                        images = core.generate_gemini(api_key, active_model, image_prompt, n, aspect_ratio)
                    else:
                        images = core.generate_grok(api_key, active_model, image_prompt, n)

                    st.session_state["images"] = images
                    st.session_state["last_prompt"] = image_prompt
                    st.session_state["reasoning_summary"] = reasoning_summary
                    st.session_state["thinking_used"] = bool(thinking_mode)
                except Exception as e:
                    st.session_state["images"] = []
                    st.error(core.friendly_error(e))

# --- Render results (kept in session_state so downloads don't clear them) ---
images = st.session_state.get("images", [])
if images:
    st.subheader("Result")
    if st.session_state.get("last_prompt"):
        st.caption(st.session_state["last_prompt"])
    cols = st.columns(2) if len(images) > 1 else [st]
    for i, img in enumerate(images):
        target = cols[i % len(cols)]
        target.image(img, use_container_width=True)
        target.download_button(
            "Download",
            data=img,
            file_name=f"image_{i + 1}.png",
            mime="image/png",
            key=f"dl_{i}",
        )

    # Reasoning summary — only shown when thinking mode produced the image.
    if st.session_state.get("thinking_used"):
        st.subheader("🧠 Reasoning summary")
        summary = st.session_state.get("reasoning_summary") or ""
        if summary:
            render_copyable(summary)
        else:
            st.info("Brak dostępnego podsumowania rozumowania dla tej odpowiedzi.")
