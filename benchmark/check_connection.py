"""Simple connection and API key validation test for Anthropic, OpenAI, and Gemini.

Sends a minimal (1-token / shortest prompt) request to each configured provider to verify:
1. The API key is set and recognized by the provider.
2. The network connection can reach the provider endpoint.
3. The configured default model accepts requests.

Usage:
    # Run via CLI script
    python3 -m benchmark.check_connection

    # Filter to a specific provider
    python3 -m benchmark.check_connection --provider anthropic
    python3 -m benchmark.check_connection --provider openai
    python3 -m benchmark.check_connection --provider gemini

    # Run via pytest (skips unconfigured keys)
    pytest tests/test_connection.py -v
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from rich import box
from rich.console import Console
from rich.table import Table

from benchmark.config import get_api_key, get_default_model


@dataclass
class CheckResult:
    provider: str
    model: str
    status: str  # "SUCCESS", "FAILED", "SKIPPED"
    latency_seconds: float = 0.0
    message: str = ""


def check_anthropic(model: str | None = None) -> CheckResult:
    """Send a minimal request to Anthropic Messages API."""
    try:
        api_key = get_api_key("anthropic")
        target_model = model or get_default_model("anthropic")
    except ValueError as exc:
        return CheckResult(
            provider="anthropic",
            model=model or "(not set)",
            status="SKIPPED",
            message=str(exc),
        )

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        start = time.perf_counter()
        resp = client.messages.create(
            model=target_model,
            max_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        elapsed = time.perf_counter() - start
        resp_id = getattr(resp, "id", "ok")
        return CheckResult(
            provider="anthropic",
            model=target_model,
            status="SUCCESS",
            latency_seconds=round(elapsed, 3),
            message=f"Connected successfully (response_id: {resp_id})",
        )
    except Exception as exc:
        return CheckResult(
            provider="anthropic",
            model=target_model,
            status="FAILED",
            message=f"{type(exc).__name__}: {exc}",
        )


def check_openai(model: str | None = None) -> CheckResult:
    """Send a minimal request to OpenAI Chat Completions API."""
    try:
        api_key = get_api_key("openai")
        target_model = model or get_default_model("openai")
    except ValueError as exc:
        return CheckResult(
            provider="openai",
            model=model or "(not set)",
            status="SKIPPED",
            message=str(exc),
        )

    try:
        import openai

        client = openai.OpenAI(api_key=api_key)
        start = time.perf_counter()
        resp = client.chat.completions.create(
            model=target_model,
            max_completion_tokens=1,
            messages=[{"role": "user", "content": "ping"}],
        )
        elapsed = time.perf_counter() - start
        resp_id = getattr(resp, "id", "ok")
        return CheckResult(
            provider="openai",
            model=target_model,
            status="SUCCESS",
            latency_seconds=round(elapsed, 3),
            message=f"Connected successfully (response_id: {resp_id})",
        )
    except Exception as exc:
        return CheckResult(
            provider="openai",
            model=target_model,
            status="FAILED",
            message=f"{type(exc).__name__}: {exc}",
        )


def check_gemini(model: str | None = None) -> CheckResult:
    """Send a minimal request to Google Gemini API via google-genai SDK."""
    try:
        api_key = get_api_key("gemini")
        target_model = model or get_default_model("gemini")
    except ValueError as exc:
        return CheckResult(
            provider="gemini",
            model=model or "(not set)",
            status="SKIPPED",
            message=str(exc),
        )

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        start = time.perf_counter()
        resp = client.models.generate_content(
            model=target_model,
            contents="ping",
            config=types.GenerateContentConfig(max_output_tokens=1),
        )
        elapsed = time.perf_counter() - start
        cand_count = len(getattr(resp, "candidates", []) or [])
        return CheckResult(
            provider="gemini",
            model=target_model,
            status="SUCCESS",
            latency_seconds=round(elapsed, 3),
            message=f"Connected successfully (candidates: {cand_count})",
        )
    except Exception as exc:
        return CheckResult(
            provider="gemini",
            model=target_model,
            status="FAILED",
            message=f"{type(exc).__name__}: {exc}",
        )


def run_checks(providers: list[str] | None = None) -> list[CheckResult]:
    """Execute connectivity checks for the specified or all providers."""
    targets = [p.lower().strip() for p in (providers or ["anthropic", "openai", "gemini"])]
    results: list[CheckResult] = []

    for p in targets:
        if p == "anthropic":
            results.append(check_anthropic())
        elif p == "openai":
            results.append(check_openai())
        elif p == "gemini":
            results.append(check_gemini())
        else:
            results.append(
                CheckResult(
                    provider=p,
                    model="(unknown)",
                    status="FAILED",
                    message=f"Unsupported provider: {p}",
                )
            )

    return results


def print_results(results: list[CheckResult], console: Console | None = None) -> None:
    """Render check results table using rich."""
    c = console or Console()
    table = Table(
        title="LLM API Connectivity & Key Validation Check",
        box=box.ROUNDED,
        header_style="bold cyan",
    )
    table.add_column("Provider", style="bold")
    table.add_column("Model")
    table.add_column("Status", justify="center")
    table.add_column("Latency", justify="right")
    table.add_column("Details / Message")

    for r in results:
        if r.status == "SUCCESS":
            status_text = "[bold green]✔ SUCCESS[/bold green]"
            latency_str = f"{r.latency_seconds:.2f}s"
            msg = f"[green]{r.message}[/green]"
        elif r.status == "SKIPPED":
            status_text = "[bold yellow]⚠ SKIPPED[/bold yellow]"
            latency_str = "-"
            msg = f"[yellow]{r.message}[/yellow]"
        else:
            status_text = "[bold red]✖ FAILED[/bold red]"
            latency_str = "-"
            msg = f"[red]{r.message}[/red]"

        table.add_row(r.provider, r.model, status_text, latency_str, msg)

    c.print()
    c.print(table)
    c.print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m benchmark.check_connection",
        description="Verify API key validity and network connectivity for Anthropic, OpenAI, and Gemini.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=["anthropic", "openai", "gemini"],
        help="Check only a specific provider (default: check all 3).",
    )
    args = parser.parse_args(argv)

    targets = [args.provider] if args.provider else ["anthropic", "openai", "gemini"]
    results = run_checks(targets)
    print_results(results)

    # Return exit code 1 if any checked provider failed
    has_failed = any(r.status == "FAILED" for r in results)
    return 1 if has_failed else 0


if __name__ == "__main__":
    sys.exit(main())

