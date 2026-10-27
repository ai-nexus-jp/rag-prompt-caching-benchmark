"""Unit tests for token parsing, cost calculation, prompt layout, and reporting."""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from benchmark.cli import build_parser, main
from benchmark.config import (
    calculate_cost,
    get_api_key,
    get_default_model,
    get_model_pricing,
)
from benchmark.core import (
    BenchmarkRunner,
    build_dynamic_rag_chunk,
    generate_static_manual,
)
from benchmark.providers.anthropic_provider import AnthropicProvider
from benchmark.providers.base import BaseProvider, PromptPayload, TurnUsage
from benchmark.providers.gemini_provider import GeminiProvider
from benchmark.providers.openai_provider import OpenAIProvider
from benchmark.reporters.file_reporter import FileReporter


class DummyMockProvider(BaseProvider):
    """Deterministic mock provider simulating realistic prefix caching behavior."""

    def __init__(self, model: str) -> None:
        super().__init__(model=model, api_key="mock-key")
        self._turn_counter: dict[str, int] = {"bad_prefix": 0, "good_prefix": 0}

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def send_turn(self, payload: PromptPayload) -> TurnUsage:
        self._turn_counter[payload.layout] += 1
        turn_num = self._turn_counter[payload.layout]

        if payload.layout == "bad_prefix":
            # Anti-pattern: dynamic prefix invalidates static manual every turn
            return TurnUsage(
                base_input_tokens=150,
                cache_write_tokens=3600,
                cache_read_tokens=0,
                output_tokens=80,
                response_text="Mock bad_prefix response",
                latency_seconds=1.25,
            )

        # Optimized pattern: Turn 1 writes cache, Turns 2+ hit cache
        if turn_num == 1:
            return TurnUsage(
                base_input_tokens=150,
                cache_write_tokens=3600,
                cache_read_tokens=0,
                output_tokens=80,
                response_text="Mock good_prefix cold turn 1",
                latency_seconds=1.30,
            )
        return TurnUsage(
            base_input_tokens=150,
            cache_write_tokens=0,
            cache_read_tokens=3600,
            output_tokens=80,
            response_text=f"Mock good_prefix warm turn {turn_num}",
            latency_seconds=0.45,
        )


def test_pricing_table_lookup_and_prefix_matching() -> None:
    """Verify exact and snapshot-prefix model lookups in MODEL_PRICING_TABLE."""
    sonnet_45 = get_model_pricing("claude-sonnet-4-5")
    assert sonnet_45.input_per_1m == 3.00
    assert sonnet_45.cache_write_per_1m == 3.75
    assert sonnet_45.cache_read_per_1m == 0.30
    assert sonnet_45.output_per_1m == 15.00

    # Dated snapshot fallback
    sonnet_snapshot = get_model_pricing("claude-sonnet-4-5-20250929")
    assert sonnet_snapshot == sonnet_45

    gpt4o_mini_snap = get_model_pricing("gpt-4o-mini-2024-07-18")
    assert gpt4o_mini_snap.input_per_1m == 0.15
    assert gpt4o_mini_snap.cache_read_per_1m == 0.075

    with pytest.raises(KeyError):
        get_model_pricing("unknown-model-xyz")


def test_calculate_cost_across_providers() -> None:
    """Verify exact USD cost math for Anthropic, OpenAI, and Gemini models."""
    # Anthropic Claude Sonnet 4.5 / 3.5 Sonnet:
    # Base: $3.00/1M, Write: $3.75/1M, Read: $0.30/1M, Output: $15.00/1M
    cost_claude = calculate_cost(
        model="claude-sonnet-4-5",
        base_input_tokens=1_000,
        cache_write_tokens=2_000,
        cache_read_tokens=3_000,
        output_tokens=500,
    )
    # (1000*3.0 + 2000*3.75 + 3000*0.30 + 500*15.0) / 1,000,000
    # = (3000 + 7500 + 900 + 7500) / 1,000,000 = 18900 / 1,000,000 = 0.0189
    assert cost_claude == pytest.approx(0.0189)

    # OpenAI GPT-4o:
    # Base: $2.50/1M, Read: $1.25/1M, Output: $10.00/1M
    cost_gpt4o = calculate_cost(
        model="gpt-4o",
        base_input_tokens=1_000,
        cache_write_tokens=0,
        cache_read_tokens=4_000,
        output_tokens=200,
    )
    # (1000*2.50 + 4000*1.25 + 200*10.0) / 1,000,000 = (2500 + 5000 + 2000)/1e6 = 0.0095
    assert cost_gpt4o == pytest.approx(0.0095)

    # OpenAI GPT-4.1:
    # Base: $2.00/1M, Read: $0.50/1M, Output: $8.00/1M
    cost_gpt41 = calculate_cost(
        model="gpt-4.1",
        base_input_tokens=1_000,
        cache_write_tokens=0,
        cache_read_tokens=4_000,
        output_tokens=250,
    )
    # (1000*2.0 + 4000*0.50 + 250*8.0) / 1e6 = (2000 + 2000 + 2000)/1e6 = 0.006
    assert cost_gpt41 == pytest.approx(0.006)

    # Gemini 3.8 Flash:
    # Base: $0.75/1M, Read: $0.075/1M, Output: $3.75/1M
    cost_gemini38 = calculate_cost(
        model="gemini-3.8-flash",
        base_input_tokens=1_000,
        cache_write_tokens=0,
        cache_read_tokens=10_000,
        output_tokens=400,
    )
    # (1000*0.75 + 10000*0.075 + 400*3.75) / 1e6 = (750 + 750 + 1500)/1e6 = 0.003
    assert cost_gemini38 == pytest.approx(0.003)

    # Gemini 3.1 Pro:
    # Base: $2.00/1M, Read: $0.20/1M, Output: $12.00/1M
    cost_gemini31_pro = calculate_cost(
        model="gemini-3.1-pro",
        base_input_tokens=2_000,
        cache_write_tokens=0,
        cache_read_tokens=5_000,
        output_tokens=500,
    )
    # (2000*2.0 + 5000*0.20 + 500*12.0) / 1e6 = (4000 + 1000 + 6000)/1e6 = 0.011
    assert cost_gemini31_pro == pytest.approx(0.011)

    # Gemini 3.1 Flash-Lite:
    # Base: $0.25/1M, Read: $0.025/1M, Output: $1.50/1M
    cost_gemini31_lite = calculate_cost(
        model="gemini-3.1-flash-lite",
        base_input_tokens=2_000,
        cache_write_tokens=0,
        cache_read_tokens=10_000,
        output_tokens=500,
    )
    # (2000*0.25 + 10000*0.025 + 500*1.50) / 1e6 = (500 + 250 + 750)/1e6 = 0.0015
    assert cost_gemini31_lite == pytest.approx(0.0015)


def test_env_model_and_api_key_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure models and API keys are read strictly from environment / CLI overrides."""
    monkeypatch.setenv("DEFAULT_ANTHROPIC_MODEL", "claude-sonnet-4-5")
    monkeypatch.setenv("DEFAULT_OPENAI_MODEL", "gpt-4.1")
    monkeypatch.setenv("DEFAULT_GEMINI_MODEL", "gemini-3.8-flash")

    assert get_default_model("anthropic") == "claude-sonnet-4-5"
    assert get_default_model("openai") == "gpt-4.1"
    assert get_default_model("gemini") == "gemini-3.8-flash"

    # CLI override takes precedence over .env
    assert get_default_model("openai", cli_model="gpt-5") == "gpt-5"

    # Missing environment variable without CLI override raises ValueError
    monkeypatch.delenv("DEFAULT_OPENAI_MODEL", raising=False)
    with pytest.raises(ValueError, match="DEFAULT_OPENAI_MODEL"):
        get_default_model("openai")

    # Placeholder API key is rejected
    monkeypatch.setenv("ANTHROPIC_API_KEY", "your-anthropic-api-key-here")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        get_api_key("anthropic")

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-real-key-123")
    assert get_api_key("anthropic") == "sk-ant-real-key-123"


def test_anthropic_response_parsing_and_block_layout() -> None:
    """Test AnthropicProvider usage extraction and cache_control block ordering."""
    mock_response = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=180,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=3620,
            output_tokens=95,
        ),
        content=[SimpleNamespace(text="規定第3章によりMFAが必須です。")],
    )

    usage = AnthropicProvider.parse_response_usage(mock_response, latency_seconds=0.812)
    assert usage.base_input_tokens == 180
    assert usage.cache_write_tokens == 0
    assert usage.cache_read_tokens == 3620
    assert usage.output_tokens == 95
    assert usage.total_input_tokens == 3800
    assert usage.cache_hit_rate == pytest.approx((3620 / 3800) * 100, rel=1e-3)
    assert "MFA" in usage.response_text

    with patch("anthropic.Anthropic"):
        provider = AnthropicProvider(model="claude-sonnet-4-5", api_key="test-key")
        bad_payload = PromptPayload(
            layout="bad_prefix",
            static_context="STATIC_DOC",
            dynamic_context="DYNAMIC_RAG",
            user_question="QUESTION?",
        )
        bad_blocks = provider._build_content_blocks(bad_payload)
        assert bad_blocks[0]["text"] == "DYNAMIC_RAG"
        assert "cache_control" not in bad_blocks[0]
        assert bad_blocks[1]["text"] == "STATIC_DOC"
        assert bad_blocks[1]["cache_control"] == {"type": "ephemeral"}

        good_payload = PromptPayload(
            layout="good_prefix",
            static_context="STATIC_DOC",
            dynamic_context="DYNAMIC_RAG",
            user_question="QUESTION?",
        )
        good_blocks = provider._build_content_blocks(good_payload)
        assert good_blocks[0]["text"] == "STATIC_DOC"
        assert good_blocks[0]["cache_control"] == {"type": "ephemeral"}
        assert good_blocks[1]["text"] == "DYNAMIC_RAG"


def test_openai_response_parsing() -> None:
    """Test OpenAIProvider usage extraction including cached_tokens deduction."""
    mock_response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=3850,
            completion_tokens=110,
            prompt_tokens_details=SimpleNamespace(
                cached_tokens=3584,
                cache_write_tokens=0,
            ),
        ),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="バックアップは35日間保持されます。")
            )
        ],
    )

    usage = OpenAIProvider.parse_response_usage(mock_response, latency_seconds=0.654)
    # base_input_tokens = 3850 - 3584 = 266
    assert usage.base_input_tokens == 266
    assert usage.cache_write_tokens == 0
    assert usage.cache_read_tokens == 3584
    assert usage.output_tokens == 110
    assert usage.total_input_tokens == 3850
    assert usage.cache_hit_rate == pytest.approx((3584 / 3850) * 100, rel=1e-3)
    assert "35日間" in usage.response_text


def test_gemini_response_parsing_and_cache_lifecycle() -> None:
    """Test GeminiProvider usage extraction and explicit cache creation/deletion."""
    mock_response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=3900,
            cached_content_token_count=3650,
            candidates_token_count=85,
        ),
        text="P1インシデントは15分以内にエスカレーションが必要です。",
    )

    usage = GeminiProvider.parse_response_usage(mock_response, latency_seconds=0.52)
    assert usage.base_input_tokens == 250
    assert usage.cache_write_tokens == 0
    assert usage.cache_read_tokens == 3650
    assert usage.output_tokens == 85
    assert usage.total_input_tokens == 3900
    assert usage.cache_hit_rate == pytest.approx((3650 / 3900) * 100, rel=1e-3)

    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.caches.create.return_value = SimpleNamespace(name="cachedContents/test-123")

        cold_resp = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=3900,
                cached_content_token_count=0,
                candidates_token_count=70,
            ),
            text="Turn 1 response",
        )
        warm_resp = SimpleNamespace(
            usage_metadata=SimpleNamespace(
                prompt_token_count=3900,
                cached_content_token_count=3650,
                candidates_token_count=70,
            ),
            text="Turn 2 response",
        )
        mock_client.models.generate_content.side_effect = [cold_resp, warm_resp]

        provider = GeminiProvider(model="gemini-3.8-flash", api_key="test-gemini-key")
        provider.prepare_layout("good_prefix", "STATIC_MANUAL_CONTENT")

        payload = PromptPayload(
            layout="good_prefix",
            static_context="STATIC_MANUAL_CONTENT",
            dynamic_context="DYNAMIC_CHUNK",
            user_question="Q1",
        )
        u1 = provider.send_turn(payload)
        assert u1.cache_read_tokens == 0
        mock_client.caches.create.assert_called_once()

        u2 = provider.send_turn(payload)
        assert u2.cache_read_tokens == 3650

        provider.cleanup_layout("good_prefix")
        mock_client.caches.delete.assert_called_once_with(name="cachedContents/test-123")


def test_benchmark_runner_and_reporters(tmp_path: Path) -> None:
    """Verify end-to-end BenchmarkRunner metrics and CSV/Markdown file generation."""
    provider = DummyMockProvider(model="claude-sonnet-4-5")
    runner = BenchmarkRunner(
        provider=provider,
        turns=4,
        delay_seconds=0.0,
        static_context=generate_static_manual(target_sections=8),
    )

    result = runner.run()

    # bad_prefix: 0% hit rate across all 4 turns
    assert result.bad_summary.average_hit_rate == 0.0
    assert result.bad_summary.total_cache_read_tokens == 0

    # good_prefix: turn 1 is 0%, turns 2..4 hit 3600 / 3750 = 96.0%
    assert result.good_summary.warm_turns_hit_rate == pytest.approx(96.0)
    assert result.good_summary.average_hit_rate == pytest.approx(72.0)
    assert result.good_summary.total_cache_read_tokens == 3600 * 3
    assert result.cost_reduction_rate > 50.0
    assert result.cost_savings_usd > 0.0

    # Export CSV & Markdown
    reporter = FileReporter(output_dir=tmp_path)
    csv_file = reporter.export_csv(result)
    md_file = reporter.export_markdown(result)

    assert csv_file.exists()
    assert md_file.exists()

    with csv_file.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8  # 4 turns * 2 layouts
    assert rows[0]["layout"] == "bad_prefix"
    assert rows[4]["layout"] == "good_prefix"
    assert int(rows[5]["cache_read_tokens"]) == 3600

    md_content = md_file.read_text(encoding="utf-8")
    assert "bad_prefix" in md_content
    assert "good_prefix" in md_content
    assert "claude-sonnet-4-5" in md_content


def test_static_manual_and_dynamic_chunk_generation() -> None:
    """Ensure generated manual is substantial and dynamic chunks have unique UUIDs."""
    manual = generate_static_manual()
    assert len(manual) > 5000
    assert "第1章" in manual
    assert "第10章" in manual

    chunk1 = build_dynamic_rag_chunk(1)
    chunk2 = build_dynamic_rag_chunk(1)
    assert chunk1 != chunk2  # UUID guarantees uniqueness


def test_load_static_manual_and_questions(tmp_path: Path) -> None:
    """Verify loading static manual and questions from disk files."""
    from benchmark.core import load_questions, load_static_manual

    # Default file load
    loaded_manual = load_static_manual()
    assert "全社統合情報セキュリティ" in loaded_manual
    assert len(loaded_manual) > 3000

    loaded_q = load_questions()
    assert len(loaded_q) >= 10
    assert "VPN" in loaded_q[0] or "MFA" in loaded_q[0]

    # Custom file load
    custom_doc = tmp_path / "custom.md"
    custom_doc.write_text("# Custom Policy\nSection A", encoding="utf-8")
    assert load_static_manual(custom_doc) == "# Custom Policy\nSection A"

    custom_q = tmp_path / "custom_q.json"
    custom_q.write_text('[{"turn": 1, "question": "What is A?"}]', encoding="utf-8")
    assert load_questions(custom_q) == ["What is A?"]


def test_cli_parser_and_main_with_mock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test CLI argument parsing and full execution flow with mocked provider."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-mock-12345")
    monkeypatch.setenv("DEFAULT_ANTHROPIC_MODEL", "claude-sonnet-4-5")

    parser = build_parser()
    args = parser.parse_args(
        ["--provider", "anthropic", "--turns", "2", "--delay", "0.0", "--output-dir", str(tmp_path)]
    )
    assert args.provider == "anthropic"
    assert args.turns == 2

    with patch(
        "benchmark.cli.create_provider",
        return_value=DummyMockProvider(model="claude-sonnet-4-5"),
    ):
        exit_code = main(
            [
                "--provider",
                "anthropic",
                "--turns",
                "3",
                "--delay",
                "0.0",
                "--output-dir",
                str(tmp_path),
            ]
        )
    assert exit_code == 0
    assert len(list(tmp_path.glob("benchmark_*.csv"))) == 1
    assert len(list(tmp_path.glob("report_*.md"))) == 1
