import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64
import urllib.request

# Gemini (Google) uses its own SDK rather than the OpenAI-compatible layer.
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

load_dotenv()

st.set_page_config(page_title="AI Image Generator", page_icon="🎨", layout="centered")
st.title("🎨 AI Image Generator")
st.caption("Generate images with OpenAI, Google Gemini, or Grok (xAI) — straight from their official APIs.")

PROVIDERS = {
    "OpenAI": {
        "models": ["gpt-image-1", "dall-e-3", "dall-e-2"],
        "api_key_env": "OPENAI_API_KEY",
    },
    "Gemini (Google)": {
        "models": [
            "imagen-3.0-generate-002",
            "imagen-4.0-generate-001",
            "gemini-2.0-flash-preview-image-generation",
        ],
        "api_key_env": "GEMINI_API_KEY",
    },
    "Grok (xAI)": {
        "models": ["grok-2-image-1212"],
        "api_key_env": "XAI_API_KEY",
    },
}


# --- Helpers ---
def _fetch_url(url: str) -> bytes:
    """Fallback for providers that return a URL instead of base64."""
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def _decode(item) -> bytes:
    """Turn an OpenAI-style image data item into raw bytes."""
    b64 = getattr(item, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(item, "url", None)
    if url:
        return _fetch_url(url)
    raise ValueError("API response did not contain image data.")


def generate_openai(api_key, model, prompt, n, size, quality):
    client = OpenAI(api_key=api_key)
    kwargs = {"model": model, "prompt": prompt, "n": n, "size": size}
    if model == "gpt-image-1":
        # gpt-image-1 always returns base64 and rejects response_format.
        kwargs["quality"] = quality
    elif model == "dall-e-3":
        kwargs["quality"] = quality
        kwargs["response_format"] = "b64_json"
    else:  # dall-e-2 (no quality parameter)
        kwargs["response_format"] = "b64_json"
    result = client.images.generate(**kwargs)
    return [_decode(d) for d in result.data]


def generate_grok(api_key, model, prompt, n):
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    result = client.images.generate(
        model=model, prompt=prompt, n=n, response_format="b64_json"
    )
    return [_decode(d) for d in result.data]


def generate_gemini(api_key, model, prompt, n, aspect_ratio):
    client = genai.Client(api_key=api_key)
    images = []
    if model.startswith("imagen"):
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=n,
                aspect_ratio=aspect_ratio,
            ),
        )
        for generated in response.generated_images:
            images.append(generated.image.image_bytes)
    else:
        # Native Gemini image generation (e.g. gemini-2.0-flash-preview-image-generation).
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"]),
        )
        for part in response.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                images.append(part.inline_data.data)
    return images


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "401" in msg or "invalid_api_key" in low or "incorrect api key" in low or "api key not valid" in low:
        return "❌ Invalid API key. Double-check the key in your `.env` file."
    if "402" in msg or "insufficient balance" in low:
        return "❌ Insufficient balance. Please top up your account."
    if "429" in msg or "insufficient_quota" in low or "quota" in low or "rate limit" in low:
        return "❌ Rate limit or quota exceeded / out of credits. Check your billing."
    if "not found" in low or "does not exist" in low or "not supported" in low:
        return f"❌ Model not available for your account, or the model ID is wrong. Try the *custom model* field in the sidebar.\n\nDetails: {msg}"
    return f"❌ API error: {msg}"


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
        else:  # gpt-image-1
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

# Resolve the actual model id to send.
active_model = custom_model or model

# --- API key check ---
api_key = os.getenv(PROVIDERS[provider]["api_key_env"])
if not api_key:
    st.warning(
        f"No API key for {provider}. "
        f"Set `{PROVIDERS[provider]['api_key_env']}` in your `.env` file."
    )
    st.stop()

if provider == "Gemini (Google)" and not GENAI_AVAILABLE:
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
        with st.spinner(f"Generating with {active_model}…"):
            try:
                if provider == "OpenAI":
                    images = generate_openai(api_key, active_model, prompt, n, size, quality)
                elif provider == "Gemini (Google)":
                    images = generate_gemini(api_key, active_model, prompt, n, aspect_ratio)
                else:
                    images = generate_grok(api_key, active_model, prompt, n)

                st.session_state["images"] = images
                st.session_state["last_prompt"] = prompt
            except Exception as e:
                st.session_state["images"] = []
                st.error(friendly_error(e))

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
