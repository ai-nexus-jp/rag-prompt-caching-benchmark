"""Anthropic Claude API provider implementation with explicit ephemeral cache breakpoints."""

from __future__ import annotations

import time
from typing import Any

import anthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from benchmark.providers.base import BaseProvider, PromptPayload, TurnUsage


class AnthropicProvider(BaseProvider):
    """Adapter for Anthropic Messages API with Prompt Caching support."""

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model=model, api_key=api_key)
        self.client = anthropic.Anthropic(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def _build_content_blocks(self, payload: PromptPayload) -> list[dict[str, Any]]:
        """Build content block list according to layout.

        In 'bad_prefix', dynamic context precedes the static manual, invalidating
        prefix matching on the static manual block across turns.
        In 'good_prefix', the static manual is placed first with an explicit
        ephemeral cache_control breakpoint so turns 2+ hit the cache.
        """
        if payload.layout == "bad_prefix":
            return [
                {
                    "type": "text",
                    "text": payload.dynamic_context,
                },
                {
                    "type": "text",
                    "text": payload.static_context,
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": f"### User Question\n{payload.user_question}",
                },
            ]

        return [
            {
                "type": "text",
                "text": payload.static_context,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": payload.dynamic_context,
            },
            {
                "type": "text",
                "text": f"### User Question\n{payload.user_question}",
            },
        ]

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=16.0),
        retry=retry_if_exception_type(
            (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            )
        ),
    )
    def _call_api(self, content_blocks: list[dict[str, Any]]) -> Any:
        """Invoke Anthropic Messages API with exponential backoff retry."""
        return self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": content_blocks,  # type: ignore[typeddict-item]
                }
            ],
        )

    @staticmethod
    def parse_response_usage(response: Any, latency_seconds: float = 0.0) -> TurnUsage:
        """Extract token usage metrics from an Anthropic Message response object."""
        usage = getattr(response, "usage", None)
        base_input = int(getattr(usage, "input_tokens", 0) or 0)
        cache_write = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        text_parts: list[str] = []
        for block in getattr(response, "content", []) or []:
            block_text = getattr(block, "text", None)
            if block_text:
                text_parts.append(str(block_text))

        return TurnUsage(
            base_input_tokens=base_input,
            cache_write_tokens=cache_write,
            cache_read_tokens=cache_read,
            output_tokens=output_tokens,
            response_text="\n".join(text_parts).strip(),
            latency_seconds=round(latency_seconds, 3),
        )

    def send_turn(self, payload: PromptPayload) -> TurnUsage:
        """Execute a single turn against Anthropic API and return normalized usage."""
        blocks = self._build_content_blocks(payload)
        start = time.perf_counter()
        response = self._call_api(blocks)
        elapsed = time.perf_counter() - start
        return self.parse_response_usage(response, latency_seconds=elapsed)

