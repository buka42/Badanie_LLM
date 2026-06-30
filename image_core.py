"""Backend logic for the image generator.

Kept free of Streamlit and free of top-level SDK imports so it can be unit
tested without those packages installed. The OpenAI SDK is imported lazily
inside the functions that need it. The API keys never leave this layer.
"""

import os
import base64
import urllib.request

# Gemini (Google) uses its own SDK rather than the OpenAI-compatible layer.
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on optional dependency
    GENAI_AVAILABLE = False


# --- Configuration ---
PROVIDERS = {
    "OpenAI": {
        "models": ["gpt-image-2"],
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

DEFAULT_THINKING_MODEL = "gpt-5.5"
DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORTS = ["low", "medium", "high"]

# Instruction for the reasoning model. It returns ONLY the final image prompt;
# the reasoning summary is read separately from the Responses API output.
_PROMPT_ENGINEER_INSTRUCTION = (
    "You are an expert image-generation prompt engineer. Take the user's idea "
    "and expand it into a single, vivid, detailed prompt for an image "
    "generation model. Respond with ONLY the final prompt text — no preamble, "
    "no explanations, no surrounding quotes."
)


def get_thinking_model() -> str:
    """Reasoning model id, configurable via OPENAI_THINKING_MODEL."""
    return os.getenv("OPENAI_THINKING_MODEL") or DEFAULT_THINKING_MODEL


def get_reasoning_effort() -> str:
    """Reasoning effort, configurable via OPENAI_REASONING_EFFORT."""
    return os.getenv("OPENAI_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT


# --- Image helpers ---
def _fetch_url(url: str) -> bytes:
    """Fallback for providers that return a URL instead of base64."""
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def decode_image_item(item) -> bytes:
    """Turn an OpenAI-style image data item into raw bytes."""
    b64 = getattr(item, "b64_json", None)
    if b64:
        return base64.b64decode(b64)
    url = getattr(item, "url", None)
    if url:
        return _fetch_url(url)
    raise ValueError("API response did not contain image data.")


def generate_openai(api_key, model, prompt, n, size, quality):
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    kwargs = {"model": model, "prompt": prompt, "n": n, "size": size}
    if model.startswith("gpt-image"):
        # gpt-image models always return base64 and reject response_format.
        kwargs["quality"] = quality
    elif model == "dall-e-3":
        kwargs["quality"] = quality
        kwargs["response_format"] = "b64_json"
    else:  # dall-e-2 (no quality parameter)
        kwargs["response_format"] = "b64_json"
    result = client.images.generate(**kwargs)
    return [decode_image_item(d) for d in result.data]


def generate_grok(api_key, model, prompt, n):
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    result = client.images.generate(
        model=model, prompt=prompt, n=n, response_format="b64_json"
    )
    return [decode_image_item(d) for d in result.data]


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


# --- Reasoning / thinking mode (OpenAI Responses API) ---
def extract_reasoning_summary(response) -> str:
    """Extract the official reasoning summary from a Responses API result.

    OpenAI does not expose raw chain-of-thought; only the ``summary`` parts of
    reasoning items are returned. Anything else is ignored. Returns an empty
    string when no summary is present.
    """
    output = getattr(response, "output", None)
    if output is None and isinstance(response, dict):
        output = response.get("output")
    parts = []
    for item in output or []:
        item_type = getattr(item, "type", None)
        if item_type is None and isinstance(item, dict):
            item_type = item.get("type")
        if item_type != "reasoning":
            continue
        summary = getattr(item, "summary", None)
        if summary is None and isinstance(item, dict):
            summary = item.get("summary")
        for entry in summary or []:
            text = getattr(entry, "text", None)
            if text is None and isinstance(entry, dict):
                text = entry.get("text")
            if text:
                parts.append(text)
    return "\n\n".join(parts).strip()


def refine_image_prompt(api_key, user_prompt, model=None, effort=None,
                        timeout=60.0, client=None):
    """Use an OpenAI reasoning model to craft a better image prompt.

    Returns ``(final_prompt, reasoning_summary)``. The reasoning summary is
    informational only and must NOT be used as the image prompt. ``client`` may
    be injected for testing; otherwise an OpenAI client is built lazily.
    """
    if client is None:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, timeout=timeout)
    response = client.responses.create(
        model=model or get_thinking_model(),
        input=[
            {"role": "developer", "content": _PROMPT_ENGINEER_INSTRUCTION},
            {"role": "user", "content": user_prompt},
        ],
        reasoning={"effort": effort or get_reasoning_effort(), "summary": "auto"},
    )
    final_text = (getattr(response, "output_text", None) or "").strip()
    summary = extract_reasoning_summary(response)
    return final_text, summary


def friendly_error(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "401" in msg or "invalid_api_key" in low or "incorrect api key" in low or "api key not valid" in low:
        return "❌ Invalid API key. Double-check the key in your `.env` file."
    if "402" in msg or "insufficient balance" in low:
        return "❌ Insufficient balance. Please top up your account."
    if "429" in msg or "insufficient_quota" in low or "quota" in low or "rate limit" in low:
        return "❌ Rate limit or quota exceeded / out of credits. Check your billing."
    if "timeout" in low or "timed out" in low:
        return "❌ The request timed out. Please try again."
    if "not found" in low or "does not exist" in low or "not supported" in low or "model_not_found" in low:
        return f"❌ Model not available for your account, or the model ID is wrong. Try the *custom model* field in the sidebar.\n\nDetails: {msg}"
    return f"❌ API error: {msg}"
