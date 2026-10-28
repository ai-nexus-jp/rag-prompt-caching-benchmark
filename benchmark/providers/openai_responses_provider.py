"""OpenAI Responses API provider implementation with explicit caching parameters."""

from __future__ import annotations

import time
import uuid
from typing import Any

import openai
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from benchmark.providers.base import BaseProvider, PromptPayload, TurnUsage


class OpenAIResponsesProvider(BaseProvider):
    """Adapter for OpenAI Responses API with explicit instruction caching.

    Uses `client.responses.create` with:
    - `instructions`: Static reference manual (cached across turns in good_prefix).
    - `input`: Dynamic RAG metadata and user question.
    - `prompt_cache_key`: Explicit stable cache identifier.
    - `prompt_cache_retention`: Extended 24h retention.
    """

    def __init__(self, model: str, api_key: str) -> None:
        super().__init__(model=model, api_key=api_key)
        self.client = openai.OpenAI(api_key=api_key)

    @property
    def provider_name(self) -> str:
        return "openai-responses"

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
    def _call_api(
        self,
        instructions: str | None,
        input_text: str,
        cache_key: str | None,
        retention: str | None,
    ) -> Any:
        """Invoke OpenAI Responses API with retry handling."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_text,
            "max_output_tokens": 256,
        }
        if instructions:
            kwargs["instructions"] = instructions
        if cache_key:
            kwargs["prompt_cache_key"] = cache_key
        if retention:
            kwargs["prompt_cache_retention"] = retention

        return self.client.responses.create(**kwargs)

    @staticmethod
    def parse_response_usage(response: Any, latency_seconds: float = 0.0) -> TurnUsage:
        """Extract token metrics from an OpenAI Responses API response object."""
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        details = getattr(usage, "input_tokens_details", None)
        cached_tokens = int(getattr(details, "cached_tokens", 0) or 0)
        cache_write_tokens = int(getattr(details, "cache_write_tokens", 0) or 0)

        base_input_tokens = max(0, input_tokens - cached_tokens - cache_write_tokens)

        response_text = ""
        output_text = getattr(response, "output_text", None)
        if output_text:
            response_text = str(output_text).strip()

        return TurnUsage(
            base_input_tokens=base_input_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_read_tokens=cached_tokens,
            output_tokens=output_tokens,
            response_text=response_text,
            latency_seconds=round(latency_seconds, 3),
        )

    def send_turn(self, payload: PromptPayload) -> TurnUsage:
        """Execute a single turn with layout-aware parameter routing."""
        if payload.layout == "good_prefix":
            # Good Prefix (Responses API Recommended Pattern):
            # Static manual is placed into `instructions` with stable `prompt_cache_key`
            instructions = payload.static_context
            input_text = f"{payload.dynamic_context}\n\n{payload.user_question}"
            cache_key = f"rag_bench_{self.model}"
            retention = "24h"
        else:
            # Bad Prefix (Anti-pattern):
            # Dynamic parts precede static content in full prompt; dynamic changes break cache prefix
            instructions = None
            input_text = payload.render_full_prompt()
            cache_key = f"bad_{uuid.uuid4().hex}"
            retention = None

        start = time.perf_counter()
        response = self._call_api(
            instructions=instructions,
            input_text=input_text,
            cache_key=cache_key,
            retention=retention,
        )
        elapsed = time.perf_counter() - start
        return self.parse_response_usage(response, latency_seconds=elapsed)
