"""Google Gemini API provider implementation using google-genai SDK."""

from __future__ import annotations

import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from benchmark.providers.base import BaseProvider, LayoutType, PromptPayload, TurnUsage


class GeminiProvider(BaseProvider):
    """Adapter for Google Gemini API supporting Context Caching (Explicit & Implicit).

    For 'good_prefix':
      - Turn 1 sends the full prompt normally (cold turn, 0 cached tokens) and creates
        an explicit Context Cache (`caches.create` with a short 5-minute TTL = "300s")
        for the static document if supported by the model.
      - Turns 2+ reference the explicit `cached_content` (or rely on Gemini's native
        Implicit Caching if explicit cache creation is unsupported / below threshold),
        recording `usage_metadata.cached_content_token_count`.
      - When the layout completes, `cleanup_layout` explicitly deletes the created cache.
    For 'bad_prefix':
      - The dynamic RAG chunk is prepended before the static document on every turn,
        preventing prefix cache hits.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        use_explicit_cache: bool = True,
        cache_ttl_seconds: int = 300,
    ) -> None:
        super().__init__(model=model, api_key=api_key)
        self.client = genai.Client(api_key=api_key)
        self.use_explicit_cache = use_explicit_cache
        self.cache_ttl_seconds = cache_ttl_seconds
        self._active_cache_name: str | None = None
        self._good_prefix_turn_index: int = 0

    @property
    def provider_name(self) -> str:
        return "gemini"

    def prepare_layout(self, layout: LayoutType, static_context: str) -> None:
        """Reset turn counter and any previous cache state before starting a layout."""
        self._good_prefix_turn_index = 0
        self._delete_active_cache()

    def cleanup_layout(self, layout: LayoutType) -> None:
        """Ensure any explicit context cache created during the test is deleted."""
        self._delete_active_cache()

    def _create_explicit_cache(self, static_context: str) -> str | None:
        """Attempt to create an explicit Gemini CachedContent with short TTL (default 5m).

        If the target model only supports implicit caching or rejects explicit cache
        creation due to token threshold rules, gracefully falls back to None so
        implicit prefix caching continues seamlessly.
        """
        if not self.use_explicit_cache:
            return None
        try:
            cache_obj = self.client.caches.create(
                model=self.model,
                config=types.CreateCachedContentConfig(
                    display_name="rag_benchmark_static_prefix",
                    contents=[static_context],
                    ttl=f"{self.cache_ttl_seconds}s",
                ),
            )
            return getattr(cache_obj, "name", None)
        except Exception:
            # Fallback to implicit caching when explicit caching is unsupported on the model
            return None

    def _delete_active_cache(self) -> None:
        """Delete the active explicit cache if one exists."""
        if self._active_cache_name:
            try:
                self.client.caches.delete(name=self._active_cache_name)
            except Exception:
                pass
            finally:
                self._active_cache_name = None

    @retry(
        reraise=True,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1.0, min=1.0, max=16.0),
        retry=retry_if_exception_type((genai_errors.APIError, ConnectionError, TimeoutError)),
    )
    def _call_api(self, contents: str, cached_content_name: str | None = None) -> Any:
        """Invoke Gemini generate_content with exponential backoff retry."""
        config = types.GenerateContentConfig(max_output_tokens=256)
        if cached_content_name:
            config.cached_content = cached_content_name
        return self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

    @staticmethod
    def parse_response_usage(response: Any, latency_seconds: float = 0.0) -> TurnUsage:
        """Extract token usage metrics from a Gemini GenerateContentResponse object.

        `usage_metadata.prompt_token_count` includes `cached_content_token_count`.
        Therefore, `base_input_tokens` is `max(0, prompt_token_count - cached_content_token_count)`.
        """
        metadata = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(metadata, "prompt_token_count", 0) or 0)
        cached_tokens = int(getattr(metadata, "cached_content_token_count", 0) or 0)
        candidates_tokens = int(getattr(metadata, "candidates_token_count", 0) or 0)

        base_input_tokens = max(0, prompt_tokens - cached_tokens)

        response_text = ""
        try:
            text_attr = getattr(response, "text", None)
            if text_attr:
                response_text = str(text_attr).strip()
        except Exception:
            response_text = ""

        return TurnUsage(
            base_input_tokens=base_input_tokens,
            cache_write_tokens=0,
            cache_read_tokens=cached_tokens,
            output_tokens=candidates_tokens,
            response_text=response_text,
            latency_seconds=round(latency_seconds, 3),
        )

    def send_turn(self, payload: PromptPayload) -> TurnUsage:
        """Execute a single turn against Gemini API and return normalized usage."""
        start = time.perf_counter()

        if payload.layout == "good_prefix":
            self._good_prefix_turn_index += 1
            if self._good_prefix_turn_index == 1:
                # Turn 1: Initial cold request with full prompt, then warm explicit cache if supported
                response = self._call_api(payload.render_full_prompt(), cached_content_name=None)
                elapsed = time.perf_counter() - start
                self._active_cache_name = self._create_explicit_cache(payload.static_context)
                return self.parse_response_usage(response, latency_seconds=elapsed)

            if self._active_cache_name:
                # Subsequent turns with explicit cached_content: send only dynamic suffix + question
                dynamic_suffix = (
                    f"{payload.dynamic_context}\n\n### User Question\n{payload.user_question}"
                )
                response = self._call_api(
                    dynamic_suffix, cached_content_name=self._active_cache_name
                )
                elapsed = time.perf_counter() - start
                return self.parse_response_usage(response, latency_seconds=elapsed)

        # Default path (bad_prefix or implicit caching on good_prefix turns 2+)
        response = self._call_api(payload.render_full_prompt(), cached_content_name=None)
        elapsed = time.perf_counter() - start
        return self.parse_response_usage(response, latency_seconds=elapsed)

