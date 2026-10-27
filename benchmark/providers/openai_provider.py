"""OpenAI Chat Completions provider implementation with automatic prefix caching."""

from __future__ import annotations

import time
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from benchmark.providers.base import BaseProvider, PromptPayload, TurnUsage


class OpenAIProvider(BaseProvider):
    """Adapter for OpenAI Chat Completions API with automatic Prefix Caching."""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model=model, api_key=api_key)
        self.client = openai.OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    def _build_messages(self, payload: PromptPayload) -> list[dict[str, str]]:
        """Build chat messages according to the prompt layout."""
        return [
            {
                "role": "user",
                "content": payload.render_full_prompt(),
            }
        ]

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=16.0),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.InternalServerError,
            )
        ),
    )
    def _call_api(self, messages: list[dict[str, str]]) -> Any:
        """Invoke OpenAI Chat Completions API with exponential backoff retry."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            max_completion_tokens=256,
        )

    @staticmethod
    def parse_response_usage(response: Any, latency_seconds: float = 0.0) -> TurnUsage:
        """Extract token usage metrics from an OpenAI ChatCompletion response object.

        Note: OpenAI's `usage.prompt_tokens` includes `cached_tokens` (and optionally
        `cache_write_tokens`). To avoid double-charging cached tokens, `base_input_tokens`
        is computed as `max(0, prompt_tokens - cached_tokens - cache_write_tokens)`.
        """
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        cache_write_tokens = int(getattr(details, "cache_write_tokens", 0) or 0)

        base_input_tokens = max(0, prompt_tokens - cached_tokens - cache_write_tokens)

        response_text = ""
        choices = getattr(response, "choices", None)
        if choices and len(choices) > 0:
            message = getattr(choices[0], "message", None)
            content = getattr(message, "content", None)
            if content:
                response_text = str(content).strip()

        return TurnUsage(
            base_input_tokens=base_input_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cached_tokens,
            output_tokens=completion_tokens,
            response_text=response_text,
            latency_seconds=round(latency_seconds, 3),
        )

    def send_turn(self, payload: PromptPayload) -> TurnUsage:
        """Execute a single turn against OpenAI API and return normalized usage."""
        messages = self._build_messages(payload)
        start = time.perf_counter()
        response = self._call_api(messages)
        elapsed = time.perf_counter() - start
        return self.parse_response_usage(response, latency_seconds=elapsed)

