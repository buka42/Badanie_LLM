"""Tests for image_core.

These cover the default (non-thinking) helpers and the optional OpenAI
thinking mode, without requiring the openai/streamlit/google SDKs or any
network access. Fake response objects mimic the OpenAI Responses API shape.
"""

import base64
import struct
from types import SimpleNamespace

import image_core as core


def _make_png(width, height):
    return (b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x0dIHDR"
            + struct.pack(">II", width, height) + b"\x00" * 8)


# --- Config (env-driven, with fallbacks) ---
def test_thinking_model_default(monkeypatch):
    monkeypatch.delenv("OPENAI_THINKING_MODEL", raising=False)
    assert core.get_thinking_model() == "gpt-5.5"


def test_thinking_model_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_THINKING_MODEL", "o5-pro")
    assert core.get_thinking_model() == "o5-pro"


def test_reasoning_effort_default(monkeypatch):
    monkeypatch.delenv("OPENAI_REASONING_EFFORT", raising=False)
    assert core.get_reasoning_effort() == "medium"


def test_reasoning_effort_env_override(monkeypatch):
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "high")
    assert core.get_reasoning_effort() == "high"


# --- Image decoding (default flow) ---
def test_decode_b64():
    raw = b"hello-image-bytes"
    item = SimpleNamespace(b64_json=base64.b64encode(raw).decode())
    assert core.decode_image_item(item) == raw


def test_decode_raises_without_data():
    item = SimpleNamespace(b64_json=None, url=None)
    try:
        core.decode_image_item(item)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --- Reasoning summary parsing ---
def _reasoning_response(summary_texts, output_text="final prompt"):
    reasoning_item = SimpleNamespace(
        type="reasoning",
        summary=[SimpleNamespace(type="summary_text", text=t) for t in summary_texts],
    )
    message_item = SimpleNamespace(type="message", summary=None)
    return SimpleNamespace(output=[reasoning_item, message_item], output_text=output_text)


def test_extract_summary_joins_parts():
    resp = _reasoning_response(["First thought.", "Second thought."])
    assert core.extract_reasoning_summary(resp) == "First thought.\n\nSecond thought."


def test_extract_summary_empty_when_no_reasoning():
    resp = SimpleNamespace(output=[SimpleNamespace(type="message", summary=None)],
                           output_text="x")
    assert core.extract_reasoning_summary(resp) == ""


def test_extract_summary_handles_dict_shape():
    resp = {"output": [{"type": "reasoning", "summary": [{"text": "dict thought"}]}]}
    assert core.extract_reasoning_summary(resp) == "dict thought"


def test_extract_summary_no_output():
    assert core.extract_reasoning_summary(SimpleNamespace(output=None)) == ""


# --- Thinking mode end-to-end (with an injected fake client) ---
class _FakeResponses:
    def __init__(self, response, captured):
        self._response = response
        self._captured = captured

    def create(self, **kwargs):
        self._captured.update(kwargs)
        return self._response


class _FakeClient:
    def __init__(self, response, captured):
        self.responses = _FakeResponses(response, captured)


def test_refine_image_prompt_returns_prompt_and_summary():
    captured = {}
    resp = _reasoning_response(["I considered lighting and composition."],
                              output_text="  A detailed final prompt.  ")
    client = _FakeClient(resp, captured)

    final, summary = core.refine_image_prompt(
        "key", "a cat", model="gpt-5.5", effort="high", client=client
    )

    assert final == "A detailed final prompt."  # trimmed
    assert summary == "I considered lighting and composition."
    # Responses API called with reasoning effort + auto summary.
    assert captured["model"] == "gpt-5.5"
    assert captured["reasoning"] == {"effort": "high", "summary": "auto"}
    # The user's idea is forwarded as input.
    assert any(m.get("role") == "user" and m.get("content") == "a cat"
               for m in captured["input"])


def test_refine_image_prompt_without_summary():
    captured = {}
    resp = SimpleNamespace(output=[], output_text="just a prompt")
    client = _FakeClient(resp, captured)

    final, summary = core.refine_image_prompt("key", "a dog", client=client)
    assert final == "just a prompt"
    assert summary == ""


def test_refine_image_prompt_uses_env_defaults(monkeypatch):
    monkeypatch.setenv("OPENAI_THINKING_MODEL", "my-model")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    captured = {}
    client = _FakeClient(_reasoning_response([]), captured)

    core.refine_image_prompt("key", "x", client=client)
    assert captured["model"] == "my-model"
    assert captured["reasoning"]["effort"] == "low"


# --- Friendly error messages ---
def test_friendly_error_missing_model():
    msg = core.friendly_error(Exception("The model `gpt-5.5` does not exist"))
    assert "Model not available" in msg


def test_friendly_error_quota():
    assert "billing" in core.friendly_error(Exception("Error code: 429 insufficient_quota"))


def test_friendly_error_timeout():
    assert "timed out" in core.friendly_error(Exception("Request timeout"))


# --- Image info (content type / size / dimensions) ---
def test_image_info_png():
    png = _make_png(1448, 1086)
    info = core.image_info(png)
    assert info["content_type"] == "image/png"
    assert info["width"] == 1448
    assert info["height"] == 1086
    assert info["size_bytes"] == len(png)


def test_image_info_unknown_format():
    info = core.image_info(b"not-an-image")
    assert info["content_type"] == "application/octet-stream"
    assert info["width"] is None and info["height"] is None
    assert info["size_bytes"] == len(b"not-an-image")


# --- Run metadata assembly ---
def test_build_metadata_structure():
    png = _make_png(1024, 768)
    md = core.build_metadata(
        prompt={"user_prompt": "a cat", "final_prompt_used": "a fluffy cat"},
        provider="OpenAI",
        model="gpt-image-2",
        parameters={"number_of_images": 1, "size": "1024x1024", "quality": "high"},
        images=[png],
        run_started_at="2026-06-30T08:45:43+00:00",
        run_completed_at="2026-06-30T08:47:23+00:00",
        reasoning={"enabled": True, "effort": "high", "summary": "considered lighting"},
        provider_response={"usage": {"total_tokens": 42}, "revised_prompt": "a fluffy cat"},
    )

    assert md["schema_name"] == core.METADATA_SCHEMA_NAME
    assert md["source"] == "official_api"
    assert md["provider"] == "OpenAI" and md["model"] == "gpt-image-2"
    assert md["prompt"]["final_prompt_used"] == "a fluffy cat"
    assert md["images"][0]["width"] == 1024 and md["images"][0]["height"] == 768
    assert md["reasoning"]["effort"] == "high"
    assert md["provider_response"]["usage"]["total_tokens"] == 42
    # The browser-only gap is documented in the export.
    assert isinstance(md["not_available_from_api"], list) and md["not_available_from_api"]


def test_build_metadata_defaults_reasoning_disabled():
    md = core.build_metadata(
        prompt={"user_prompt": "x", "final_prompt_used": "x"},
        provider="Grok (xAI)",
        model="grok-2-image-1212",
        parameters={"number_of_images": 1},
        images=[],
        run_started_at="a",
        run_completed_at="b",
    )
    assert md["reasoning"] == {"enabled": False}
    assert md["images"] == []


def test_as_dict_handles_model_dump():
    obj = SimpleNamespace(model_dump=lambda: {"total_tokens": 7})
    assert core._as_dict(obj) == {"total_tokens": 7}


def test_as_dict_none():
    assert core._as_dict(None) is None
