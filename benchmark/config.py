"""Configuration, model pricing tables, and environment variable loaders."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

TOKENS_PER_MILLION: Final[float] = 1_000_000.0


@dataclass(frozen=True)
class ModelPricing:
    """Pricing structure per 1M tokens in USD.

    Attributes:
        input_per_1m: Standard uncached input token cost (USD / 1M tokens).
        cache_write_per_1m: Cache creation/write token cost (USD / 1M tokens).
        cache_read_per_1m: Cache hit/read token cost (USD / 1M tokens).
        output_per_1m: Output / completion token cost (USD / 1M tokens).
    """

    input_per_1m: float
    cache_write_per_1m: float
    cache_read_per_1m: float
    output_per_1m: float


# Pricing table (USD per 1M tokens) based on the official 2026-10-28 survey
MODEL_PRICING_TABLE: Final[dict[str, ModelPricing]] = {
    # --- Anthropic ---
    "claude-fable-5-1": ModelPricing(
        input_per_1m=10.00,
        cache_write_per_1m=12.50,
        cache_read_per_1m=0.25,
        output_per_1m=50.00,
    ),
    "claude-opus-5-5": ModelPricing(
        input_per_1m=4.00,
        cache_write_per_1m=5.00,
        cache_read_per_1m=0.20,
        output_per_1m=20.00,
    ),
    "claude-sonnet-5-5": ModelPricing(
        input_per_1m=2.00,
        cache_write_per_1m=2.50,
        cache_read_per_1m=0.10,
        output_per_1m=10.00,
    ),
    "claude-sonnet-5": ModelPricing(
        input_per_1m=2.00,
        cache_write_per_1m=2.50,
        cache_read_per_1m=0.20,
        output_per_1m=10.00,
    ),
    "claude-haiku-5-5": ModelPricing(
        input_per_1m=0.10,
        cache_write_per_1m=0.125,
        cache_read_per_1m=0.01,
        output_per_1m=0.50,
    ),
    "claude-sonnet-4-6": ModelPricing(
        input_per_1m=3.00,
        cache_write_per_1m=3.75,
        cache_read_per_1m=0.30,
        output_per_1m=15.00,
    ),
    "claude-haiku-4-5": ModelPricing(
        input_per_1m=1.00,
        cache_write_per_1m=1.25,
        cache_read_per_1m=0.10,
        output_per_1m=5.00,
    ),
    # Legacy / Deprecation scheduled Anthropic models (retained for backward compatibility)
    "claude-sonnet-4-5": ModelPricing(
        input_per_1m=3.00,
        cache_write_per_1m=3.75,
        cache_read_per_1m=0.30,
        output_per_1m=15.00,
    ),
    "claude-3-7-sonnet-latest": ModelPricing(
        input_per_1m=3.00,
        cache_write_per_1m=3.75,
        cache_read_per_1m=0.30,
        output_per_1m=15.00,
    ),
    "claude-3-5-sonnet-latest": ModelPricing(
        input_per_1m=3.00,
        cache_write_per_1m=3.75,
        cache_read_per_1m=0.30,
        output_per_1m=15.00,
    ),
    # --- OpenAI ---
    "gpt-6-astra": ModelPricing(
        input_per_1m=10.00,
        cache_write_per_1m=12.50,
        cache_read_per_1m=1.00,
        output_per_1m=50.00,
    ),
    "gpt-6.1-sol": ModelPricing(
        input_per_1m=2.00,
        cache_write_per_1m=2.50,
        cache_read_per_1m=0.10,
        output_per_1m=10.00,
    ),
    "gpt-6.1-terra": ModelPricing(
        input_per_1m=1.00,
        cache_write_per_1m=1.25,
        cache_read_per_1m=0.05,
        output_per_1m=5.00,
    ),
    "gpt-6-luna": ModelPricing(
        input_per_1m=0.10,
        cache_write_per_1m=0.125,
        cache_read_per_1m=0.01,
        output_per_1m=0.50,
    ),
    # Legacy / Deprecation scheduled OpenAI models (retained for backward compatibility)
    "gpt-5": ModelPricing(
        input_per_1m=1.25,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.125,
        output_per_1m=10.00,
    ),
    "gpt-5-mini": ModelPricing(
        input_per_1m=0.25,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.025,
        output_per_1m=1.00,
    ),
    "gpt-4.1": ModelPricing(
        input_per_1m=2.00,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.50,
        output_per_1m=8.00,
    ),
    "gpt-4.1-mini": ModelPricing(
        input_per_1m=0.40,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.10,
        output_per_1m=1.60,
    ),
    "gpt-4o": ModelPricing(
        input_per_1m=2.50,
        cache_write_per_1m=0.0,
        cache_read_per_1m=1.25,
        output_per_1m=10.00,
    ),
    "gpt-4o-mini": ModelPricing(
        input_per_1m=0.15,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.075,
        output_per_1m=0.60,
    ),
    # --- Google Gemini ---
    "gemini-4-argon": ModelPricing(
        input_per_1m=2.00,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.10,
        output_per_1m=10.00,
    ),
    "gemini-3.8-flash": ModelPricing(
        input_per_1m=0.75,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.075,
        output_per_1m=3.75,
    ),
    "gemini-3.6-flash": ModelPricing(
        input_per_1m=0.75,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.075,
        output_per_1m=3.75,
    ),
    "gemini-3.5-flash-lite": ModelPricing(
        input_per_1m=0.30,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.03,
        output_per_1m=2.50,
    ),
    "gemini-3.1-pro-preview": ModelPricing(
        input_per_1m=2.00,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.20,
        output_per_1m=12.00,
    ),
    "gemini-3.1-pro": ModelPricing(
        input_per_1m=2.00,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.20,
        output_per_1m=12.00,
    ),
    "gemini-3.1-flash-lite": ModelPricing(
        input_per_1m=0.25,
        cache_write_per_1m=0.0,
        cache_read_per_1m=0.025,
        output_per_1m=1.50,
    ),
}

PROVIDER_ENV_MODEL_KEYS: Final[dict[str, str]] = {
    "anthropic": "DEFAULT_ANTHROPIC_MODEL",
    "openai": "DEFAULT_OPENAI_MODEL",
    "openai-responses": "DEFAULT_OPENAI_MODEL",
    "openai_responses": "DEFAULT_OPENAI_MODEL",
    "gemini": "DEFAULT_GEMINI_MODEL",
}

PROVIDER_ENV_API_KEYS: Final[dict[str, str]] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-responses": "OPENAI_API_KEY",
    "openai_responses": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


def get_model_pricing(model: str) -> ModelPricing:
    """Resolve pricing for a given model name.

    Matches exact keys first, then falls back to prefix matching for dated snapshots.

    Args:
        model: Model identifier string.

    Returns:
        ModelPricing instance for the model.

    Raises:
        KeyError: If the model is not registered in MODEL_PRICING_TABLE.
    """
    normalized = model.strip()
    if normalized in MODEL_PRICING_TABLE:
        return MODEL_PRICING_TABLE[normalized]

    # Prefix match for snapshot versions (e.g. gpt-4.1-2025-04-14 -> gpt-4.1)
    # Sort keys by length descending so more specific names (e.g. gpt-4o-mini) match before shorter ones (gpt-4o)
    for key in sorted(MODEL_PRICING_TABLE.keys(), key=len, reverse=True):
        if normalized.startswith(key):
            return MODEL_PRICING_TABLE[key]

    raise KeyError(
        f"Pricing for model '{model}' is not defined in MODEL_PRICING_TABLE. "
        f"Supported models: {', '.join(sorted(MODEL_PRICING_TABLE.keys()))}"
    )


def calculate_cost(
    model: str,
    base_input_tokens: int,
    cache_write_tokens: int,
    cache_read_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate the USD cost for a single turn given token counts.

    Args:
        model: Model identifier string.
        base_input_tokens: Uncached standard input tokens.
        cache_write_tokens: Tokens written to cache (if billed separately, e.g., Anthropic).
        cache_read_tokens: Tokens read from cache (cache hit).
        output_tokens: Generated completion/output tokens.

    Returns:
        Total cost in USD.
    """
    pricing = get_model_pricing(model)
    cost = (
        (base_input_tokens / TOKENS_PER_MILLION) * pricing.input_per_1m
        + (cache_write_tokens / TOKENS_PER_MILLION) * pricing.cache_write_per_1m
        + (cache_read_tokens / TOKENS_PER_MILLION) * pricing.cache_read_per_1m
        + (output_tokens / TOKENS_PER_MILLION) * pricing.output_per_1m
    )
    return round(cost, 8)


def get_default_model(provider: str, cli_model: str | None = None) -> str:
    """Resolve target model from CLI override or environment variable (.env).

    Model names are never hardcoded as defaults in code; they must be provided
    via CLI option or environment variable (.env).

    Args:
        provider: Provider name ('anthropic', 'openai', or 'gemini').
        cli_model: Optional model override passed from CLI.

    Returns:
        Resolved model identifier string.

    Raises:
        ValueError: If provider is unknown or model is not configured in .env / CLI.
    """
    if cli_model and cli_model.strip():
        return cli_model.strip()

    provider_key = provider.lower().strip()
    env_var = PROVIDER_ENV_MODEL_KEYS.get(provider_key)
    if not env_var:
        raise ValueError(f"Unsupported provider: '{provider}'")

    env_val = os.getenv(env_var, "").strip()
    if not env_val:
        raise ValueError(
            f"Model for provider '{provider}' is not specified. "
            f"Please set {env_var} in your .env file or pass --model via CLI."
        )
    return env_val


def get_api_key(provider: str) -> str:
    """Retrieve the API key for the specified provider from environment variables.

    Args:
        provider: Provider name ('anthropic', 'openai', or 'gemini').

    Returns:
        The configured API key string.

    Raises:
        ValueError: If the API key is missing or still set to the placeholder value.
    """
    provider_key = provider.lower().strip()
    env_var = PROVIDER_ENV_API_KEYS.get(provider_key)
    if not env_var:
        raise ValueError(f"Unsupported provider: '{provider}'")

    api_key = os.getenv(env_var, "").strip()
    if not api_key or api_key.startswith("your-"):
        raise ValueError(
            f"Valid API key for '{provider}' not found. "
            f"Please set {env_var} in your .env file or environment."
        )
    return api_key


def get_default_turns() -> int:
    """Return default benchmark turns from environment or fallback to 5."""
    val = os.getenv("DEFAULT_TURNS", "5").strip()
    try:
        return max(1, int(val))
    except ValueError:
        return 5


def get_default_delay() -> float:
    """Return default delay in seconds between turns from environment or fallback to 1.5."""
    val = os.getenv("DEFAULT_DELAY_SECONDS", "1.5").strip()
    try:
        return max(0.0, float(val))
    except ValueError:
        return 1.5


def get_default_output_dir() -> Path:
    """Return default output directory from environment or fallback to ./results."""
    val = os.getenv("OUTPUT_DIR", "./results").strip()
    return Path(val if val else "./results")

