"""Abstract base class and data models for LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

LayoutType = Literal["bad_prefix", "good_prefix"]


@dataclass(frozen=True)
class PromptPayload:
    """Structured prompt components for a single benchmark turn.

    Attributes:
        layout: Either 'bad_prefix' (dynamic first) or 'good_prefix' (static first).
        static_context: Long static document (~3,500-4,000 tokens).
        dynamic_context: Dynamic RAG chunk with timestamp and UUID.
        user_question: The user query for the current turn.
    """

    layout: LayoutType
    static_context: str
    dynamic_context: str
    user_question: str

    def render_full_prompt(self) -> str:
        """Assemble the full text prompt based on the selected layout order."""
        if self.layout == "bad_prefix":
            return (
                f"{self.dynamic_context}\n\n"
                f"{self.static_context}\n\n"
                f"### User Question\n{self.user_question}"
            )
        return (
            f"{self.static_context}\n\n"
            f"{self.dynamic_context}\n\n"
            f"### User Question\n{self.user_question}"
        )


@dataclass(frozen=True)
class TurnUsage:
    """Normalized token usage metrics returned by a provider for a single turn.

    Attributes:
        base_input_tokens: Uncached input tokens billed at standard rate.
        cache_write_tokens: Input tokens written to cache (e.g., Anthropic cache creation).
        cache_read_tokens: Input tokens read from cache (cache hits).
        output_tokens: Generated output/completion tokens.
        response_text: Text response from the model.
        latency_seconds: Wall-clock duration of the API request.
    """

    base_input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    response_text: str = ""
    latency_seconds: float = 0.0

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens processed across uncached, cache write, and cache read."""
        return self.base_input_tokens + self.cache_write_tokens + self.cache_read_tokens

    @property
    def cache_hit_rate(self) -> float:
        """Percentage of total input tokens served from cache (0.0 to 100.0)."""
        total = self.total_input_tokens
        if total <= 0:
            return 0.0
        return round((self.cache_read_tokens / total) * 100.0, 2)


class BaseProvider(ABC):
    """Common abstract interface for LLM prompt caching benchmark providers."""

    def __init__(self, model: str, api_key: str) -> None:
        self.model = model
        self.api_key = api_key

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the canonical provider identifier (e.g., 'anthropic', 'openai', 'gemini')."""

    def prepare_layout(self, layout: LayoutType, static_context: str) -> None:  # noqa: B027
        """Optional hook called before running a sequence of turns for a layout."""

    def cleanup_layout(self, layout: LayoutType) -> None:  # noqa: B027
        """Optional hook called after finishing a sequence of turns for a layout."""

    @abstractmethod
    def send_turn(self, payload: PromptPayload) -> TurnUsage:
        """Execute a single turn against the provider API and extract normalized token metrics.

        Args:
            payload: Structured prompt components and layout mode.

        Returns:
            Normalized TurnUsage instance.
        """

