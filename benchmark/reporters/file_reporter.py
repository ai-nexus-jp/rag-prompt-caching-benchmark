"""CSV and Markdown report exporters for benchmark results."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from benchmark.config import get_model_pricing
from benchmark.core import BenchmarkResult


class FileReporter:
    """Generates CSV raw logs and publication-ready Markdown reports."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_filename(value: str) -> str:
        """Sanitize model or provider strings for safe filesystem filenames."""
        return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in value)

    def export_csv(self, result: BenchmarkResult, filename: str | None = None) -> Path:
        """Export all turn records to a CSV file.

        Args:
            result: Completed BenchmarkResult instance.
            filename: Optional explicit filename.

        Returns:
            Path to the written CSV file.
        """
        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_model = self._sanitize_filename(result.model)
            filename = f"benchmark_{result.provider}_{safe_model}_{ts}.csv"

        csv_path = self.output_dir / filename
        fieldnames = [
            "timestamp",
            "provider",
            "model",
            "layout",
            "turn",
            "base_input_tokens",
            "cache_write_tokens",
            "cache_read_tokens",
            "output_tokens",
            "total_input_tokens",
            "cache_hit_rate",
            "turn_cost_usd",
            "latency_seconds",
        ]

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in result.all_records:
                writer.writerow(
                    {
                        "timestamp": r.timestamp,
                        "provider": r.provider,
                        "model": r.model,
                        "layout": r.layout,
                        "turn": r.turn,
                        "base_input_tokens": r.base_input_tokens,
                        "cache_write_tokens": r.cache_write_tokens,
                        "cache_read_tokens": r.cache_read_tokens,
                        "output_tokens": r.output_tokens,
                        "total_input_tokens": r.total_input_tokens,
                        "cache_hit_rate": f"{r.cache_hit_rate:.2f}",
                        "turn_cost_usd": f"{r.turn_cost_usd:.8f}",
                        "latency_seconds": f"{r.latency_seconds:.3f}",
                    }
                )

        return csv_path

    def export_markdown(self, result: BenchmarkResult, filename: str | None = None) -> Path:
        """Generate a Zenn/Qiita-ready Markdown report comparing bad_prefix and good_prefix.

        Args:
            result: Completed BenchmarkResult instance.
            filename: Optional explicit filename.

        Returns:
            Path to the written Markdown file.
        """
        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            safe_model = self._sanitize_filename(result.model)
            filename = f"report_{result.provider}_{safe_model}_{ts}.md"

        md_path = self.output_dir / filename
        bad = result.bad_summary
        good = result.good_summary
        pricing = get_model_pricing(result.model)

        lines: list[str] = [
            f"# LLMプロンプトキャッシュ実測ベンチマークレポート ({result.provider.upper()} / `{result.model}`)",
            "",
            f"- **計測日時 (UTC)**: `{result.started_at}`",
            f"- **プロバイダ**: `{result.provider}`",
            f"- **対象モデル**: `{result.model}`",
            f"- **検証ターン数**: 各レイアウト {result.turns} ターン",
            f"- **単価設定 (USD / 1M tokens)**: 通常入力 `${pricing.input_per_1m}` / キャッシュ書込 `${pricing.cache_write_per_1m}` / キャッシュ読取 `${pricing.cache_read_per_1m}` / 出力 `${pricing.output_per_1m}`",
            "",
            "---",
            "",
            "## 1. エグゼクティブサマリー（検証結果の結論）",
            "",
            f"> **プロンプト先頭に固定コンテキストを配置する最適化（`good_prefix`）により、全{result.turns}ターンの累計利用コストが `${bad.total_cost_usd:.6f}` から `${good.total_cost_usd:.6f}` へと `{result.cost_reduction_rate:.2f}%` 削減されました（2ターン目以降の平均キャッシュヒット率: `{good.warm_turns_hit_rate:.1f}%`）。**",
            "",
            "| 検証パターン (Layout) | 試行回数 | 全ターン平均HIT率 | 2ターン目以降HIT率 | 累計通常入力 (tok) | 累計キャッシュ書込 (tok) | 累計キャッシュ読取 (tok) | 累計コスト (USD) | コスト削減効果 |",
            "| :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | :---: |",
            f"| **`bad_prefix`** (動的RAG先頭・アンチパターン) | {bad.turns} | {bad.average_hit_rate:.1f}% | {bad.warm_turns_hit_rate:.1f}% | {bad.total_base_input_tokens:,} | {bad.total_cache_write_tokens:,} | {bad.total_cache_read_tokens:,} | `${bad.total_cost_usd:.6f}` | ベースライン (0.0%) |",
            f"| **`good_prefix`** (固定規定先頭・最適化パターン) | {good.turns} | **{good.average_hit_rate:.1f}%** | **{good.warm_turns_hit_rate:.1f}%** | {good.total_base_input_tokens:,} | {good.total_cache_write_tokens:,} | **{good.total_cache_read_tokens:,}** | **`${good.total_cost_usd:.6f}`** | **-{result.cost_reduction_rate:.2f}%** (`-${result.cost_savings_usd:.6f}`) |",
            "",
            "---",
            "",
            "## 2. ターン別計測ログ詳細",
            "",
            "### (A) アンチパターン (`bad_prefix`: `[動的RAGチャンク + 日時 + UUID]` → `[固定業務規定]` → `[質問]`)",
            "",
            "| Turn | Base Input (tok) | Cache Write (tok) | Cache Read / Hit (tok) | Hit Rate (%) | Output (tok) | Turn Cost (USD) | Latency (s) |",
            "| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]

        for r in bad.records:
            lines.append(
                f"| {r.turn} | {r.base_input_tokens:,} | {r.cache_write_tokens:,} | {r.cache_read_tokens:,} | {r.cache_hit_rate:.1f}% | {r.output_tokens:,} | `${r.turn_cost_usd:.6f}` | {r.latency_seconds:.2f}s |"
            )

        lines.extend(
            [
                "",
                "### (B) 最適化パターン (`good_prefix`: `[固定業務規定]` → `[動的RAGチャンク + 日時 + UUID]` → `[質問]`)",
                "",
                "| Turn | Base Input (tok) | Cache Write (tok) | Cache Read / Hit (tok) | Hit Rate (%) | Output (tok) | Turn Cost (USD) | Latency (s) |",
                "| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )

        for r in good.records:
            lines.append(
                f"| {r.turn} | {r.base_input_tokens:,} | {r.cache_write_tokens:,} | **{r.cache_read_tokens:,}** | **{r.cache_hit_rate:.1f}%** | {r.output_tokens:,} | **`${r.turn_cost_usd:.6f}`** | {r.latency_seconds:.2f}s |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 3. 検証インプットデータおよびプロンプト構造",
                "",
                "本ベンチマークで使用されたプロンプト構成要素は以下の通りです。",
                "",
                "1. **固定コンテキスト（約3,500〜4,000トークン）**:",
                "   - 参照ファイル: [`data/static_manual.md`](../data/static_manual.md)",
                "   - 内容: 全社統合情報セキュリティ・クラウド運用・AIガバナンス業務規定マニュアル（全10章）",
                "   - 役割: キャッシュ発動閾値（1,024〜2,048トークン）を超過し、毎ターン完全に一致する静的コンテキスト",
                "2. **動的コンテキスト（毎ターン変化するRAG検索チャンク）**:",
                "   - 内容: ターン毎に一意に発行されるセッションUUID、ISOタイムスタンプ（ミリ秒UTC）、検索スコア、監査メモ",
                "   - 生成例:",
                "     ```markdown",
                "     ### [Dynamic RAG Context & Session Metadata]",
                "     - session_request_id: e4b2d3c1-58a9-4cf7-9a82-123456789abc",
                "     - retrieved_timestamp_utc: 2026-10-28T08:30:00.123456+00:00",
                "     - rag_turn_sequence: 1",
                "     - retrieved_chunk_score: 0.9820",
                "     - dynamic_rag_note: 直近の監査連絡事項（リアルタイム検索チャンク）...",
                "     ```",
                "3. **ユーザー質問**:",
                "   - 参照ファイル: [`data/questions.json`](../data/questions.json)",
                "   - 内容: 規定マニュアルの各章（VPN/MFA、DBバックアップ、P1エスカレーション等）に関する質問",
                "",
                "---",
                "",
                "## 4. アーキテクチャ考察",
                "",
                "1. **前方一致（Prefix Match）の原則**:",
                "   主要LLMプロバイダのプロンプトキャッシュはすべてプロンプト先頭からの完全一致を条件としています。`bad_prefix` のように先頭へ現在時刻・セッションUUID・動的RAG検索結果を挿入すると、後続の固定マニュアル（数千トークン）が毎ターン「新規入力」または「新規キャッシュ書込」として課金されます。",
                "2. **RAGプロンプト設計のベストプラクティス**:",
                "   `good_prefix` のように、不変のシステム指示や共通業務規定をプロンプトの最前方に固定し、動的に変化するRAGチャンクやユーザー質問を末尾へ配置することで、2ターン目以降の入力トークンの大半がキャッシュヒットとなり、コストとレイテンシを劇的に圧縮できます。",
                "",
            ]
        )

        md_path.write_text("\n".join(lines), encoding="utf-8")
        return md_path

    def export_comparison_markdown(
        self, results: list[BenchmarkResult], filename: str | None = None
    ) -> Path:
        """Generate a cross-provider comparison Markdown report comparing multiple providers.

        Args:
            results: List of completed BenchmarkResult instances across providers.
            filename: Optional explicit filename.

        Returns:
            Path to the written comparison Markdown file.
        """
        if not filename:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_report_{ts}.md"

        md_path = self.output_dir / filename
        lines: list[str] = [
            "# LLMプロンプトキャッシュ 3社一括実測・比較ベンチマークレポート",
            "",
            f"- **計測日時 (UTC)**: `{datetime.now(timezone.utc).isoformat()}`",
            f"- **比較プロバイダ数**: {len(results)} プロバイダ",
            "",
            "---",
            "",
            "## 1. プロバイダ別サマリー比較（横並び）",
            "",
            "| プロバイダ | 対象モデル | 試行ターン数 | 最適化後HIT率 (Turns 2+) | bad_prefixコスト | good_prefixコスト | コスト削減率 (%) | コスト削減額 (USD) |",
            "| :--- | :--- | :---: | :---: | ---: | ---: | :---: | ---: |",
        ]

        for res in results:
            bad = res.bad_summary
            good = res.good_summary
            lines.append(
                f"| **{res.provider.upper()}** | `{res.model}` | 各{res.turns}T | **{good.warm_turns_hit_rate:.1f}%** | `${bad.total_cost_usd:.6f}` | **`${good.total_cost_usd:.6f}`** | **`-{res.cost_reduction_rate:.2f}%`** | `-${res.cost_savings_usd:.6f}` |"
            )

        lines.extend(
            [
                "",
                "---",
                "",
                "## 2. レイテンシ（応答時間）測定に関する技術的考察と前提",
                "",
                "> [!NOTE]",
                "> **レイテンシの計測境界**: 全プロバイダ共通で `time.perf_counter()` を使用し、**各社公式SDKがHTTPリクエストを送信開始してから、レスポンスの受信・デシリアライズを完了するまでの純粋なAPIターンアラウンド往復時間** を計測しています。",
                "",
                "1. **キャッシュヒット時（good_prefix）のレイテンシ短縮効果**:",
                "   - キャッシュヒット時は、LLMサーバー側で固定文脈（約3,500トークン）のPrefillフェーズ（プロンプト処理）のアテンション計算がスキップされるため、TTFT（Time-To-First-Token）および全体レイテンシが大幅に短縮されます。",
                "2. **プロバイダ間の絶対値レイテンシ比較における注意点**:",
                "   - プロバイダ間での絶対値（ミリ秒）比較は、物理的なAPIサーバー所在地（米東海岸 vs 米西海岸 vs 日本等）、モデルのパラメータ規模（大規模Dense vs 軽量MoE）、および各社インフラの混雑状況に依存します。",
                "   - したがって、ベンチマークにおける最も妥当な評価指標は「他社との絶対時間差」ではなく、**「同一プロバイダ内における bad_prefix（ミス）と good_prefix（ヒット）のレイテンシ改善比率」** です。",
                "",
                "---",
                "",
                "## 3. 各プロバイダ個別レポートへのリンク",
                "",
            ]
        )

        for res in results:
            lines.append(f"- **{res.provider.upper()} (`{res.model}`)**: 各種ログおよび個別詳細レポートを参照")

        md_path.write_text("\n".join(lines), encoding="utf-8")
        return md_path

