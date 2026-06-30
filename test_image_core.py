"""Tests for image_core.

These cover the default (non-thinking) helpers and the optional OpenAI
thinking mode, without requiring the openai/streamlit/google SDKs or any
network access. Fake response objects mimic the OpenAI Responses API shape.
"""

import base64
from types import SimpleNamespace

import image_core as core


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
