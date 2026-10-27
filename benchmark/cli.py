"""CLI argument parser and execution entry point for the Prompt Caching Benchmark tool."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

from benchmark.config import (
    get_api_key,
    get_default_delay,
    get_default_model,
    get_default_output_dir,
    get_default_turns,
    get_model_pricing,
)
from benchmark.core import BenchmarkRunner, load_questions, load_static_manual
from benchmark.providers.anthropic_provider import AnthropicProvider
from benchmark.providers.base import BaseProvider
from benchmark.providers.gemini_provider import GeminiProvider
from benchmark.providers.openai_provider import OpenAIProvider
from benchmark.reporters.console_reporter import ConsoleReporter
from benchmark.reporters.file_reporter import FileReporter


def build_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python3 -m benchmark.cli",
        description=(
            "LLM Prompt Caching Benchmark CLI (v2): Quantitatively compare cache hit rates "
            "and token costs between bad_prefix (dynamic first) and good_prefix (static first)."
        ),
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        choices=["anthropic", "openai", "gemini"],
        help="Target LLM provider ('anthropic', 'openai', or 'gemini').",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target model identifier. Overrides DEFAULT_<PROVIDER>_MODEL in .env.",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=None,
        help="Number of turns to execute per layout pattern (overrides DEFAULT_TURNS in .env).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Delay in seconds between turns (overrides DEFAULT_DELAY_SECONDS in .env).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path for CSV and Markdown outputs (overrides OUTPUT_DIR in .env).",
    )
    parser.add_argument(
        "--manual-file",
        type=Path,
        default=None,
        help="Path to custom Markdown manual file for static context (default: data/static_manual.md).",
    )
    parser.add_argument(
        "--questions-file",
        type=Path,
        default=None,
        help="Path to custom JSON questions file (default: data/questions.json).",
    )
    return parser


def create_provider(provider_name: str, model: str, api_key: str) -> BaseProvider:
    """Instantiate the requested provider adapter."""
    normalized = provider_name.lower().strip()
    if normalized == "anthropic":
        return AnthropicProvider(model=model, api_key=api_key)
    if normalized == "openai":
        return OpenAIProvider(model=model, api_key=api_key)
    if normalized == "gemini":
        return GeminiProvider(model=model, api_key=api_key)
    raise ValueError(f"Unsupported provider: '{provider_name}'")


def main(argv: list[str] | None = None) -> int:
    """Main CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()

    try:
        model = get_default_model(args.provider, args.model)
        # Validate pricing existence early before making API calls
        get_model_pricing(model)
        api_key = get_api_key(args.provider)
    except (ValueError, KeyError) as exc:
        console.print(f"[bold red]Configuration Error:[/bold red] {exc}")
        return 1

    turns = args.turns if args.turns is not None else get_default_turns()
    delay = args.delay if args.delay is not None else get_default_delay()
    output_dir = args.output_dir if args.output_dir is not None else get_default_output_dir()

    if turns < 1:
        console.print("[bold red]Error:[/bold red] --turns must be at least 1.")
        return 1

    console_reporter = ConsoleReporter(console=console)
    console_reporter.print_header(
        provider=args.provider,
        model=model,
        turns=turns,
        delay=delay,
    )

    try:
        provider = create_provider(args.provider, model=model, api_key=api_key)
        static_context = load_static_manual(args.manual_file)
        questions = load_questions(args.questions_file)
        runner = BenchmarkRunner(
            provider=provider,
            turns=turns,
            delay_seconds=delay,
            static_context=static_context,
            questions=questions,
        )
        result = runner.run(
            on_layout_start=console_reporter.print_layout_start,
            on_turn_complete=console_reporter.print_turn_record,
        )
    except Exception as exc:
        console.print(f"\n[bold red]Benchmark Execution Error:[/bold red] {exc}")
        return 1

    console_reporter.print_summary(result)

    file_reporter = FileReporter(output_dir=output_dir)
    csv_path = file_reporter.export_csv(result)
    md_path = file_reporter.export_markdown(result)

    console.print(
        f"\n[bold cyan]Artifacts Saved:[/bold cyan]\n"
        f"  • CSV Log:         [underline]{csv_path}[/underline]\n"
        f"  • Markdown Report: [underline]{md_path}[/underline]"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

