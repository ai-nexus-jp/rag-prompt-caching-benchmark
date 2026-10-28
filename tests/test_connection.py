"""Pytest test suite for testing API connectivity and key validity.

To run:
    # Requires network (bypass sandbox if sandboxed):
    pytest tests/test_connection.py -v
"""

from __future__ import annotations

import os

import pytest

from benchmark.check_connection import check_anthropic, check_gemini, check_openai
from benchmark.config import get_api_key


def _should_run_live(provider: str) -> bool:
    if os.getenv("RUN_LIVE_TESTS", "0") != "1":
        return False
    try:
        key = get_api_key(provider)
        return bool(key and not key.startswith("your-"))
    except Exception:
        return False


@pytest.mark.skipif(
    not _should_run_live("anthropic"),
    reason="RUN_LIVE_TESTS != 1 or ANTHROPIC_API_KEY is not configured",
)
def test_anthropic_connection() -> None:
    result = check_anthropic()
    assert result.status == "SUCCESS", f"Anthropic check failed: {result.message}"


@pytest.mark.skipif(
    not _should_run_live("openai"),
    reason="RUN_LIVE_TESTS != 1 or OPENAI_API_KEY is not configured",
)
def test_openai_connection() -> None:
    result = check_openai()
    assert result.status == "SUCCESS", f"OpenAI check failed: {result.message}"


@pytest.mark.skipif(
    not _should_run_live("gemini"),
    reason="RUN_LIVE_TESTS != 1 or GEMINI_API_KEY is not configured",
)
def test_gemini_connection() -> None:
    result = check_gemini()
    assert result.status == "SUCCESS", f"Gemini check failed: {result.message}"
