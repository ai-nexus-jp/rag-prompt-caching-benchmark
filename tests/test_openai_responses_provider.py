"""Unit tests for OpenAIResponsesProvider."""

from unittest.mock import MagicMock

from benchmark.cli import create_provider
from benchmark.config import get_api_key, get_default_model
from benchmark.providers.base import PromptPayload
from benchmark.providers.openai_responses_provider import OpenAIResponsesProvider


def test_openai_responses_config_resolution(monkeypatch):
    monkeypatch.setenv("DEFAULT_OPENAI_MODEL", "gpt-6.1-sol")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    assert get_default_model("openai-responses") == "gpt-6.1-sol"
    assert get_api_key("openai-responses") == "test-openai-key"

    provider = create_provider("openai-responses", "gpt-6.1-sol", "test-openai-key")
    assert isinstance(provider, OpenAIResponsesProvider)
    assert provider.provider_name == "openai-responses"


def test_openai_responses_parse_response_usage():
    mock_resp = MagicMock()
    mock_resp.output_text = "Here is the answer."

    # Turn 1: Cache creation (write)
    mock_resp.usage.input_tokens = 4050
    mock_resp.usage.output_tokens = 80
    mock_resp.usage.input_tokens_details.cache_write_tokens = 4000
    mock_resp.usage.input_tokens_details.cached_tokens = 0

    usage1 = OpenAIResponsesProvider.parse_response_usage(mock_resp, latency_seconds=1.23)
    assert usage1.base_input_tokens == 50
    assert usage1.cache_write_tokens == 4000
    assert usage1.cache_read_tokens == 0
    assert usage1.output_tokens == 80
    assert usage1.response_text == "Here is the answer."
    assert usage1.latency_seconds == 1.23

    # Turn 2: Cache hit (read)
    mock_resp.usage.input_tokens = 4060
    mock_resp.usage.output_tokens = 75
    mock_resp.usage.input_tokens_details.cache_write_tokens = 10
    mock_resp.usage.input_tokens_details.cached_tokens = 4000

    usage2 = OpenAIResponsesProvider.parse_response_usage(mock_resp, latency_seconds=0.45)
    assert usage2.base_input_tokens == 50
    assert usage2.cache_write_tokens == 10
    assert usage2.cache_read_tokens == 4000
    assert usage2.output_tokens == 75
    assert usage2.latency_seconds == 0.45


def test_openai_responses_send_turn_parameter_routing(monkeypatch):
    provider = OpenAIResponsesProvider(model="gpt-6.1-sol", api_key="dummy-key")
    mock_call_api = MagicMock()
    monkeypatch.setattr(provider, "_call_api", mock_call_api)

    mock_resp = MagicMock()
    mock_resp.usage.input_tokens = 100
    mock_resp.usage.output_tokens = 10
    mock_resp.usage.input_tokens_details.cache_write_tokens = 0
    mock_resp.usage.input_tokens_details.cached_tokens = 0
    mock_call_api.return_value = mock_resp

    # Good prefix payload
    good_payload = PromptPayload(
        layout="good_prefix",
        static_context="STATIC MANUAL",
        dynamic_context="DYNAMIC CHUNK",
        user_question="QUESTION",
    )
    provider.send_turn(good_payload)
    mock_call_api.assert_called_with(
        instructions="STATIC MANUAL",
        input_text="DYNAMIC CHUNK\n\nQUESTION",
        cache_key="rag_bench_gpt-6.1-sol",
        retention="24h",
    )

    # Bad prefix payload
    bad_payload = PromptPayload(
        layout="bad_prefix",
        static_context="STATIC MANUAL",
        dynamic_context="DYNAMIC CHUNK",
        user_question="QUESTION",
    )
    provider.send_turn(bad_payload)
    last_call_kwargs = mock_call_api.call_args.kwargs
    assert last_call_kwargs["instructions"] is None
    assert "DYNAMIC CHUNK" in last_call_kwargs["input_text"]
    assert "STATIC MANUAL" in last_call_kwargs["input_text"]
    assert last_call_kwargs["cache_key"].startswith("bad_")
    assert last_call_kwargs["retention"] is None
