"""Terminal UI reporter using the rich library."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from benchmark.core import BenchmarkResult, TurnRecord
from benchmark.providers.base import LayoutType


class ConsoleReporter:
    """Renders real-time turn progress and final comparison summary in the terminal."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def print_header(
        self,
        provider: str,
        model: str,
        turns: int,
        delay: float,
    ) -> None:
        """Display benchmark configuration banner."""
        title = Text("LLM Prompt Caching Benchmark CLI (v2)", style="bold cyan")
        body = (
            f"[bold]Provider:[/bold] {provider}\n"
            f"[bold]Model:[/bold]    {model}\n"
            f"[bold]Turns:[/bold]    {turns} turns × 2 layouts (bad_prefix vs. good_prefix)\n"
            f"[bold]Delay:[/bold]    {delay:.1f}s between turns"
        )
        self.console.print(Panel(body, title=title, border_style="cyan", expand=False))

    def print_layout_start(self, layout: LayoutType, total_turns: int) -> None:
        """Print section header when starting a layout run."""
        if layout == "bad_prefix":
            label = "[bold red]▶ Pattern 1: bad_prefix (Anti-Pattern: Dynamic RAG First)[/bold red]"
        else:
            label = (
                "[bold green]▶ Pattern 2: good_prefix (Optimized: Static Manual First)[/bold green]"
            )
        self.console.print(f"\n{label} — Running {total_turns} turns...")

    def print_turn_record(self, record: TurnRecord) -> None:
        """Print single turn metrics on completion."""
        hit_color = "green" if record.cache_hit_rate >= 50.0 else "yellow"
        if record.cache_hit_rate == 0.0:
            hit_color = "dim"

        self.console.print(
            f"  [bold]Turn {record.turn:02d}[/bold] | "
            f"Base Input: [cyan]{record.base_input_tokens:>6,}[/cyan] | "
            f"Cache Write: [magenta]{record.cache_write_tokens:>6,}[/magenta] | "
            f"Cache Hit: [{hit_color}]{record.cache_read_tokens:>6,} ({record.cache_hit_rate:>5.1f}%)[/{hit_color}] | "
            f"Output: {record.output_tokens:>4,} | "
            f"Turn Cost: [bold yellow]${record.turn_cost_usd:.6f}[/bold yellow] "
            f"[dim]({record.latency_seconds:.2f}s)[/dim]"
        )

    def print_summary(self, result: BenchmarkResult) -> None:
        """Render final comparison summary table and cost reduction highlight."""
        self.console.print()
        table = Table(
            title=f"Prompt Caching Benchmark Summary ({result.provider} / {result.model})",
            box=box.ROUNDED,
            header_style="bold cyan",
            show_lines=True,
        )
        table.add_column("Pattern (Layout)", style="bold")
        table.add_column("Turns", justify="right")
        table.add_column("Avg HIT Rate", justify="right")
        table.add_column("Turns 2+ HIT Rate", justify="right")
        table.add_column("Total Base Input", justify="right")
        table.add_column("Total Cache Write", justify="right")
        table.add_column("Total Cache Read", justify="right")
        table.add_column("Total Cost (USD)", justify="right")
        table.add_column("Cost Reduction", justify="right")

        bad = result.bad_summary
        good = result.good_summary

        table.add_row(
            "[red]bad_prefix (Anti-Pattern)[/red]",
            str(bad.turns),
            f"{bad.average_hit_rate:.1f}%",
            f"{bad.warm_turns_hit_rate:.1f}%",
            f"{bad.total_base_input_tokens:,}",
            f"{bad.total_cache_write_tokens:,}",
            f"{bad.total_cache_read_tokens:,}",
            f"${bad.total_cost_usd:.6f}",
            "Baseline (0.0%)",
        )

        reduction_str = f"[bold green]-{result.cost_reduction_rate:.1f}%[/bold green]"
        table.add_row(
            "[green]good_prefix (Optimized)[/green]",
            str(good.turns),
            f"[bold green]{good.average_hit_rate:.1f}%[/bold green]",
            f"[bold green]{good.warm_turns_hit_rate:.1f}%[/bold green]",
            f"{good.total_base_input_tokens:,}",
            f"{good.total_cache_write_tokens:,}",
            f"[bold green]{good.total_cache_read_tokens:,}[/bold green]",
            f"[bold green]${good.total_cost_usd:.6f}[/bold green]",
            reduction_str,
        )

        self.console.print(table)

        summary_panel = Panel(
            f"[bold]Cost Saved:[/bold] [green]${result.cost_savings_usd:.6f} USD[/green] "
            f"([bold green]{result.cost_reduction_rate:.2f}% reduction[/bold green])\n"
            f"[bold]Cache Read Tokens (good_prefix):[/bold] {good.total_cache_read_tokens:,} tokens "
            f"(vs. {bad.total_cache_read_tokens:,} tokens in bad_prefix)",
            title="[bold green]Optimization Impact[/bold green]",
            border_style="green",
            expand=False,
        )
        self.console.print(summary_panel)

    def print_multi_provider_summary(self, results: list[BenchmarkResult]) -> None:
        """Render a comparative table across multiple providers."""
        self.console.print()
        table = Table(
            title="Cross-Provider Prompt Caching Comparison Summary",
            box=box.ROUNDED,
            header_style="bold magenta",
            show_lines=True,
        )
        table.add_column("Provider", style="bold")
        table.add_column("Model")
        table.add_column("Turns", justify="right")
        table.add_column("Warm HIT Rate (Turns 2+)", justify="right")
        table.add_column("bad_prefix Cost", justify="right")
        table.add_column("good_prefix Cost", justify="right")
        table.add_column("Cost Reduction", justify="right")

        for res in results:
            bad = res.bad_summary
            good = res.good_summary
            table.add_row(
                res.provider.upper(),
                res.model,
                str(res.turns),
                f"[bold green]{good.warm_turns_hit_rate:.1f}%[/bold green]",
                f"${bad.total_cost_usd:.6f}",
                f"[bold green]${good.total_cost_usd:.6f}[/bold green]",
                f"[bold green]-{res.cost_reduction_rate:.1f}%[/bold green]",
            )

        self.console.print(table)

