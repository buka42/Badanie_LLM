"""Tests for image_core.

These cover the default (non-thinking) helpers and the optional OpenAI
thinking mode, without requiring the openai/streamlit/google SDKs or any
network access. Fake response objects mimic the OpenAI Responses API shape.
"""

import base64
import os
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


# --- Gemini reasoning ---
def test_gemini_thinking_model_default(monkeypatch):
    monkeypatch.delenv("GEMINI_THINKING_MODEL", raising=False)
    assert core.get_gemini_thinking_model() == "gemini-2.5-flash"


def test_gemini_thinking_model_env_override(monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_MODEL", "gemini-x")
    assert core.get_gemini_thinking_model() == "gemini-x"


def test_effort_to_thinking_budget():
    assert core._effort_to_thinking_budget("low") == 1024
    assert core._effort_to_thinking_budget("medium") == 4096
    assert core._effort_to_thinking_budget("high") == 12288
    assert core._effort_to_thinking_budget("unknown") == -1


def _gemini_response(thought_texts, answer_texts):
    parts = [SimpleNamespace(text=t, thought=True) for t in thought_texts]
    parts += [SimpleNamespace(text=t, thought=False) for t in answer_texts]
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))]
    )


def test_parse_gemini_thoughts():
    resp = _gemini_response(["planning the scene"], ["a vivid final prompt"])
    final, summary = core.parse_gemini_thoughts(resp)
    assert final == "a vivid final prompt"
    assert summary == "planning the scene"


def test_parse_gemini_thoughts_no_thoughts():
    resp = _gemini_response([], ["only answer"])
    final, summary = core.parse_gemini_thoughts(resp)
    assert final == "only answer"
    assert summary == ""


class _FakeGeminiModels:
    def __init__(self, response, captured):
        self._response = response
        self._captured = captured

    def generate_content(self, **kwargs):
        self._captured.update(kwargs)
        return self._response


class _FakeGeminiClient:
    def __init__(self, response, captured):
        self.models = _FakeGeminiModels(response, captured)


def test_refine_image_prompt_gemini():
    captured = {}
    resp = _gemini_response(["considered composition"], ["  a detailed prompt  "])
    client = _FakeGeminiClient(resp, captured)

    final, summary = core.refine_image_prompt_gemini(
        "key", "a cat", model="gemini-2.5-flash", effort="high", client=client
    )

    assert final == "a detailed prompt"  # trimmed
    assert summary == "considered composition"
    assert captured["model"] == "gemini-2.5-flash"
    tc = captured["config"]["thinking_config"]
    assert tc["include_thoughts"] is True
    assert tc["thinking_budget"] == 12288  # high
    assert "a cat" in captured["contents"]


def test_refine_image_prompt_gemini_uses_env_defaults(monkeypatch):
    monkeypatch.setenv("GEMINI_THINKING_MODEL", "gemini-custom")
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "low")
    captured = {}
    client = _FakeGeminiClient(_gemini_response([], ["x"]), captured)

    core.refine_image_prompt_gemini("key", "y", client=client)
    assert captured["model"] == "gemini-custom"
    assert captured["config"]["thinking_config"]["thinking_budget"] == 1024  # low


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


# --- Saving generations to disk ---
def test_provider_prefix():
    assert core.provider_prefix("OpenAI") == "chatgpt"
    assert core.provider_prefix("Gemini (Google)") == "gemini"
    assert core.provider_prefix("Grok (xAI)") == "grok"
    assert core.provider_prefix("Unknown") == "model"


def test_build_prefix_with_category():
    # Spaces become underscores; provider follows the category.
    assert core.build_prefix("OpenAI", "political event") == "political_event_chatgpt"
    assert core.build_prefix("Gemini (Google)", "nature") == "nature_gemini"


def test_build_prefix_without_category():
    assert core.build_prefix("OpenAI", "") == "chatgpt"
    assert core.build_prefix("OpenAI", "   ") == "chatgpt"
    assert core.build_prefix("Grok (xAI)", None) == "grok"


def test_sanitize():
    assert core._sanitize("a b/c:d") == "a_b_c_d"
    assert core._sanitize("") == "image"


def test_save_generation_with_reasoning_and_category(tmp_path):
    png = _make_png(10, 10)
    res = core.save_generation(
        results_dir=str(tmp_path), provider="OpenAI", base_name="img_x",
        images=[png], metadata_json='{"a": 1}', reasoning_text="thoughts here",
        category="political event",
    )
    assert os.path.basename(res["folder"]) == "political_event_chatgpt_img_x"
    names = sorted(os.path.basename(p) for p in res["files"])
    assert names == [
        "political_event_chatgpt_img_x.json",
        "political_event_chatgpt_img_x.png",
        "political_event_chatgpt_img_x.txt",
    ]

    folder = res["folder"]
    with open(os.path.join(folder, "political_event_chatgpt_img_x.txt"), encoding="utf-8") as f:
        assert f.read() == "thoughts here"
    with open(os.path.join(folder, "political_event_chatgpt_img_x.png"), "rb") as f:
        assert f.read() == png


def test_save_generation_without_category(tmp_path):
    png = _make_png(4, 4)
    res = core.save_generation(
        results_dir=str(tmp_path), provider="Grok (xAI)", base_name="g1",
        images=[png], metadata_json="{}",
    )
    names = sorted(os.path.basename(p) for p in res["files"])
    assert names == ["grok_g1.json", "grok_g1.png"]


def test_save_generation_multiple_images(tmp_path):
    res = core.save_generation(
        results_dir=str(tmp_path), provider="Gemini (Google)", base_name="m",
        images=[_make_png(2, 2), _make_png(3, 3)], metadata_json="{}",
    )
    names = sorted(os.path.basename(p) for p in res["files"])
    assert names == ["gemini_m.json", "gemini_m_1.png", "gemini_m_2.png"]


def test_get_results_dir_env(monkeypatch):
    monkeypatch.delenv("RESULTS_DIR", raising=False)
    assert core.get_results_dir() == "results"
    monkeypatch.setenv("RESULTS_DIR", "out")
    assert core.get_results_dir() == "out"
