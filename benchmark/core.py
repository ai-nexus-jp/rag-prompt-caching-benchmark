"""Core benchmark engine: prompt generation, turn orchestration, and metric aggregation."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from benchmark.config import calculate_cost
from benchmark.providers.base import BaseProvider, LayoutType, PromptPayload, TurnUsage

DEFAULT_DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"
DEFAULT_STATIC_MANUAL_PATH: Path = DEFAULT_DATA_DIR / "static_manual.md"
DEFAULT_QUESTIONS_PATH: Path = DEFAULT_DATA_DIR / "questions.json"


def load_static_manual(file_path: Path | str | None = None) -> str:
    """Load the static corporate policy manual from a Markdown file.

    Falls back to generate_static_manual() if the file does not exist.
    """
    path = Path(file_path) if file_path else DEFAULT_STATIC_MANUAL_PATH
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return generate_static_manual()


def load_questions(file_path: Path | str | None = None) -> list[str]:
    """Load test questions from a JSON file.

    Falls back to SAMPLE_QUESTIONS if the file does not exist.
    """
    path = Path(file_path) if file_path else DEFAULT_QUESTIONS_PATH
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                questions = [
                    item["question"] if isinstance(item, dict) and "question" in item else str(item)
                    for item in data
                ]
                if questions:
                    return questions
        except Exception:
            pass
    return SAMPLE_QUESTIONS

SAMPLE_QUESTIONS: list[str] = [
    "第3章の規定に基づき、リモートワーク時のVPN接続と多要素認証（MFA）の要件を簡潔に教えてください。",
    "第6章のクラウドインフラ管理規定において、本番環境データベースのバックアップ保持期間と暗号化方式はどう定められていますか？",
    "第9章のセキュリティインシデント対応フローにて、重大度レベル1（P1）発生時のエスカレーション期限は何分以内ですか？",
    "第12章の生成AI・外部API利用ガイドラインにおいて、機密区分Class-Aデータの送信が許可される条件を要約してください。",
    "第15章のソフトウェアサプライチェーン管理において、サードパーティ製ライブラリの脆弱性スキャン頻度と承認権限者を教えてください。",
    "第4章のアクセス権限管理において、特権ID（管理者アカウント）の棚卸し周期と最大貸与時間はどう規定されていますか？",
    "第8章のログ監査規定に基づき、APIゲートウェイおよび認証サーバのアクセスログ保存期間を回答してください。",
    "第11章の出張および社外端末持ち出し規定において、海外渡航時に義務付けられている端末設定を教えてください。",
    "第14章の事業継続計画（BCP）における目標復旧時間（RTO）と目標復旧時点（RPO）の数値を教えてください。",
    "第18章の委託先・ベンダー管理規定において、年次セキュリティ監査の提出書類として必須の項目は何ですか？",
]


def generate_static_manual(target_sections: int = 10) -> str:
    """Generate a realistic corporate policy manual (~3,500-4,000 tokens).

    Designed to reliably exceed the 1,024-token (OpenAI/Anthropic) and 2,048-token
    (Gemini) prompt caching thresholds across tokenizers.

    Args:
        target_sections: Number of detailed policy sections to include.

    Returns:
        Markdown string representing the static corporate policy manual.
    """
    header = (
        "# 全社統合情報セキュリティ・クラウド運用・AIガバナンス業務規定マニュアル (v4.2)\n\n"
        "文書管理番号: SEC-POL-2026-GLOBAL-001\n"
        "制定日: 2022年4月1日 / 最終改定日: 2026年10月1日\n"
        "所管部門: コーポレートIT統括本部・最高情報セキュリティ責任者（CISO）室\n\n"
        "本規定は、当社およびグループ各社におけるすべての役員、正社員、契約社員、派遣社員、ならびに業務委託先パートナーが、"
        "社内情報資産・クラウドインフラストラクチャ・業務システム・生成AI基盤を利用および運用する上で遵守すべき"
        "標準ルールと技術的統制基準を定めたものである。すべてのシステム設計および運用オペレーションは本規定に準拠しなければならない。\n"
    )

    sections: list[str] = [header]

    domain_topics = [
        (
            "情報資産の機密区分と取扱基準",
            "Class-A（極秘・個人情報・認証鍵）、Class-B（社外秘・未公開財務情報・設計書）、Class-C（社内限・業務手順書）、Class-D（公開情報）の4段階に分類し、保存・転送・廃棄の各ライフサイクルにおける暗号化およびアクセス制御を義務付ける。",
        ),
        (
            "アイデンティティ管理とゼロトラスト認証基盤",
            "すべての業務アプリケーションはSAML 2.0またはOpenID Connect（OIDC）によるシングルサインオン（SSO）を必須とし、FIDO2/WebAuthn準拠のハードウェアセキュリティキーまたは生体認証による多要素認証（MFA）を強制する。",
        ),
        (
            "リモートワークおよびエンドポイントセキュリティ規定",
            "社外ネットワークからの業務システム接続時は、常時起動型ゼロトラストネットワークアクセス（ZTNA）または次世代VPNクライアントを経由し、EDR（Endpoint Detection and Response）エージェントが正常稼働している端末のみ接続を許可する。",
        ),
        (
            "特権ID管理（PAM）およびジャストインタイム権限昇格",
            "本番環境への変更権限を持つ特権アカウントは常時付与を禁止し、チケット承認番号と紐付いたジャストインタイム（JIT）申請により最大4時間を上限として一時貸与する。すべての操作セッションは証跡録画される。棚卸しは毎月第1営業日に実施する。",
        ),
        (
            "クラウドインフラストラクチャ構成管理（IaC）",
            "AWS、Google Cloud、Microsoft Azure上のすべてのリソースはTerraformまたはOpenTofuによるInfrastructure as Code（IaC）で管理し、手動コンソール変更（ClickOps）を原則禁止する。CI/CDパイプラインでのポリシー検証（OPA/Conftest）通過をデプロイ条件とする。",
        ),
        (
            "データベース運用・バックアップ・暗号化基準",
            "本番環境データベースはAES-256による保管時暗号化（Encryption at Rest）およびTLS 1.3による通信経路暗号化を必須とする。日次スナップショットおよび継続的ポイントインタイムリカバリ（PITR）を有効化し、バックアップデータは別リージョンへ35日間保持する。",
        ),
        (
            "コンテナおよびKubernetesクラスタセキュリティ",
            "本番稼働するすべてのコンテナイメージはDistrolessまたは最小構成ベースイメージを採用し、署名検証（Cosign/Sigstore）およびSBOM（Software Bill of Materials）生成を必須とする。特権コンテナ（privileged: true）の実行は全面禁止する。",
        ),
        (
            "ログ収集・SIEM統合・監査証跡保存期間",
            "APIゲートウェイ、認証基盤、データベース監査ログ、クラウドコントロールプレーン操作ログはリアルタイムで中央SIEM基盤へ転送し、改ざん防止ストレージ（WORM）にて最低3年間（1,095日間）保持する。時刻同期はNTPによりUTC基準でミリ秒単位に統一する。",
        ),
        (
            "セキュリティインシデント対応フロー（CSIRT規定）",
            "情報漏洩またはサービス停止を伴う重大度レベル1（P1）事案を検知した場合、発見者は15分以内にCSIRT専用ホットラインへ第1報をエスカレーションしなければならない。初動封じ込めは60分以内、経営陣および監督官庁への報告判断は4時間以内に実施する。",
        ),
        (
            "セキュアソフトウェア開発ライフサイクル（SSDLC）",
            "すべてのプルリクエストには静的コード解析（SAST）、シークレットスキャン、依存パッケージ脆弱性検査（SCA）の自動実行を義務付ける。CVSSスコア7.0以上の脆弱性が検出された場合、本番リリースゲートを自動的にブロックする。",
        ),
        (
            "社外端末持ち出しおよび海外渡航セキュリティ",
            "海外出張または社外へのPC持ち出し時は、フルディスク暗号化（BitLocker/FileVault）の有効化確認に加え、渡航先専用のクリーン端末貸出制度を利用すること。空港・ホテル等の公衆Wi-Fiへの直接接続は禁止し、法人契約eSIMテザリングまたは専用VPNを使用する。",
        ),
        (
            "生成AI・大規模言語モデル（LLM）および外部API利用ガイドライン",
            "業務におけるLLM API利用は、入力データがモデル学習に再利用されない法人契約（Zero Data Retention / Opt-out済）エンドポイントに限定する。機密区分Class-Aデータを送信する場合は、事前にPIIマスキングゲートウェイを通過させ、CISO室の書面承認を得ること。",
        ),
        (
            "RAGシステムおよびプロンプトエンジニアリング標準設計",
            "検索拡張生成（RAG）パイプラインを構築する際は、プロンプトインジェクション対策としてシステム指示と外部検索チャンクを明確に構造分離すること。また、Prefix Caching（プロンプトキャッシュ）のヒット率を最大化するため、不変のシステム指示・共通規定ドキュメントをプロンプト先頭に配置し、動的な検索結果やタイムスタンプは後方に配置するアーキテクチャを標準とする。",
        ),
        (
            "事業継続計画（BCP）およびディザスタリカバリ（DR）",
            "基幹業務システムの目標復旧時間（RTO）は2時間以内、目標復旧時点（RPO）は15分以内と定める。年2回（5月および11月）、本番トラフィックをセカンダリリージョンへ切り替える実動フェイルオーバー訓練を実施し、所要時間を計測・記録する。",
        ),
        (
            "ソフトウェアサプライチェーンおよびOSSガバナンス",
            "サードパーティ製ライブラリおよびOSSパッケージの導入時は、ライセンス互換性審査（GPL汚染防止等）および日次自動脆弱性スキャンを実施する。新規外部パッケージの採用承認権限者はテックリードまたはプリンシパルエンジニアとする。",
        ),
        (
            "API設計標準およびレートリミット・スロットリング",
            "外部公開および社内マイクロサービス間APIはOpenAPI 3.1仕様書で定義し、OAuth 2.0クライアントクレデンシャルまたはmTLSによる相互認証を行う。テナント単位のレートリミットおよびサーキットブレーカーを実装し、連鎖障害を防止する。",
        ),
        (
            "個人情報保護（GDPR・APPI準拠）およびデータ主権",
            "顧客の個人識別情報（PII）は収集目的を明示した上で最小限のフィールドのみ保持し、国内データセンター（東京・大阪リージョン）外への越境移転を原則禁止する。本人からの削除請求（忘れられる権利）には7営業日以内に対応する仕組みを実装する。",
        ),
        (
            "業務委託先・サードパーティベンダーセキュリティ管理",
            "重要情報システムまたはClass-B以上のデータを取り扱う外部委託先に対しては、契約締結前のセキュリティチェックシート審査および年1回の定期監査（SOC2 Type II報告書またはISO/IEC 27001認証の確認）を必須提出書類とする。",
        ),
    ]

    for idx in range(1, target_sections + 1):
        title, core_rule = domain_topics[(idx - 1) % len(domain_topics)]
        section_text = (
            f"## 第{idx}章：{title}（規定コード: POL-SEC-{idx:03d}）\n\n"
            f"### {idx}.1 基本方針および適用範囲\n"
            f"{core_rule}\n"
            f"本条項は、全社クラウド基盤（Production / Staging / Development 環境）およびオンプレミス連携ネットワークの"
            f"すべてのワークロードに適用される。各システムオーナーは四半期ごとに遵守状況を自己点検し、"
            f"証跡ログをコンプライアンス管理ポータルへ提出しなければならない。\n\n"
            f"### {idx}.2 技術的統制パラメータおよび運用基準値\n"
            f"- **統制管理ID**: CTRL-{idx:03d}-A / 監査優先度: HIGH / 自動コンプライアンスチェック: ENABLED\n"
            f"- **暗号化・通信要件**: TLS 1.3以上（暗号スイート TLS_AES_256_GCM_SHA384 推奨）、証明書有効期限90日以内の自動ローテーション。\n"
            f"- **監視・アラート閾値**: 異常検知メトリクスがベースラインの3シグマを超過した場合、5分以内にPagerDuty経由でオンコール担当者へ通知。\n"
            f"- **例外申請プロセス**: 技術的制約により本基準の即時適用が困難な場合は、代替統制（Compensating Controls）を明記したリスク受容申請書を提出し、"
            f"CISOおよび担当役員の承認を得ること。例外承認の有効期限は最大90日間とし、恒久的な適用除外は認めない。\n"
        )
        sections.append(section_text)

    return "\n".join(sections)


def build_dynamic_rag_chunk(turn: int) -> str:
    """Create a dynamic RAG context chunk with current timestamp and unique UUID.

    Because the UUID and timestamp change on every turn, placing this block at the
    very beginning of the prompt ('bad_prefix') forces a 100% prefix cache miss
    on all subsequent static context tokens.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    req_uuid = uuid.uuid4()
    return (
        "### [Dynamic RAG Context & Session Metadata]\n"
        f"- session_request_id: {req_uuid}\n"
        f"- retrieved_timestamp_utc: {now_iso}\n"
        f"- rag_turn_sequence: {turn}\n"
        f"- retrieved_chunk_score: {0.985 - (turn * 0.003):.4f}\n"
        "- dynamic_rag_note: 直近の監査連絡事項（リアルタイム検索チャンク）— "
        f"ターン#{turn}における一時セッショントークン `{req_uuid.hex[:12]}` が発行されました。"
        "回答時は全社統合業務規定マニュアルの該当章番号を明記して簡潔に答えてください。"
    )


@dataclass
class TurnRecord:
    """Complete record of a single benchmark turn including cost and metadata."""

    timestamp: str
    provider: str
    model: str
    layout: LayoutType
    turn: int
    base_input_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    output_tokens: int
    total_input_tokens: int
    cache_hit_rate: float
    turn_cost_usd: float
    latency_seconds: float


@dataclass
class LayoutSummary:
    """Aggregated metrics across all turns for a single layout pattern."""

    layout: LayoutType
    turns: int
    records: list[TurnRecord] = field(default_factory=list)

    @property
    def total_base_input_tokens(self) -> int:
        return sum(r.base_input_tokens for r in self.records)

    @property
    def total_cache_write_tokens(self) -> int:
        return sum(r.cache_write_tokens for r in self.records)

    @property
    def total_cache_read_tokens(self) -> int:
        return sum(r.cache_read_tokens for r in self.records)

    @property
    def total_output_tokens(self) -> int:
        return sum(r.output_tokens for r in self.records)

    @property
    def total_input_tokens(self) -> int:
        return sum(r.total_input_tokens for r in self.records)

    @property
    def average_hit_rate(self) -> float:
        """Overall cache hit rate across all turns in percentage (0.0 - 100.0)."""
        if not self.records:
            return 0.0
        return round(
            sum(r.cache_hit_rate for r in self.records) / len(self.records),
            2,
        )

    @property
    def warm_turns_hit_rate(self) -> float:
        """Average cache hit rate for turns 2+ (after cache is populated)."""
        warm = [r.cache_hit_rate for r in self.records if r.turn >= 2]
        if not warm:
            return self.average_hit_rate
        return round(sum(warm) / len(warm), 2)

    @property
    def total_cost_usd(self) -> float:
        return round(sum(r.turn_cost_usd for r in self.records), 8)


@dataclass
class BenchmarkResult:
    """Complete benchmark result comparing bad_prefix and good_prefix."""

    provider: str
    model: str
    turns: int
    started_at: str
    bad_summary: LayoutSummary
    good_summary: LayoutSummary

    @property
    def all_records(self) -> list[TurnRecord]:
        return [*self.bad_summary.records, *self.good_summary.records]

    @property
    def cost_savings_usd(self) -> float:
        """Absolute USD saved by good_prefix compared to bad_prefix."""
        return round(self.bad_summary.total_cost_usd - self.good_summary.total_cost_usd, 8)

    @property
    def cost_reduction_rate(self) -> float:
        """Percentage cost reduction achieved by good_prefix over bad_prefix."""
        bad_cost = self.bad_summary.total_cost_usd
        if bad_cost <= 0:
            return 0.0
        reduction = ((bad_cost - self.good_summary.total_cost_usd) / bad_cost) * 100.0
        return round(reduction, 2)


TurnCallback = Callable[[TurnRecord], None]
LayoutStartCallback = Callable[[LayoutType, int], None]


class BenchmarkRunner:
    """Orchestrates executing both bad_prefix and good_prefix layouts across N turns."""

    def __init__(
        self,
        provider: BaseProvider,
        turns: int = 5,
        delay_seconds: float = 1.5,
        static_context: str | None = None,
        questions: list[str] | None = None,
    ) -> None:
        self.provider = provider
        self.turns = max(1, turns)
        self.delay_seconds = max(0.0, delay_seconds)
        self.static_context = static_context or load_static_manual()
        self.questions = questions or load_questions()

    def run_layout(
        self,
        layout: LayoutType,
        on_layout_start: LayoutStartCallback | None = None,
        on_turn_complete: TurnCallback | None = None,
    ) -> LayoutSummary:
        """Execute a single layout ('bad_prefix' or 'good_prefix') for `self.turns` turns."""
        summary = LayoutSummary(layout=layout, turns=self.turns)
        if on_layout_start:
            on_layout_start(layout, self.turns)

        self.provider.prepare_layout(layout, self.static_context)
        try:
            for turn_idx in range(1, self.turns + 1):
                question = self.questions[(turn_idx - 1) % len(self.questions)]
                dynamic_chunk = build_dynamic_rag_chunk(turn_idx)
                payload = PromptPayload(
                    layout=layout,
                    static_context=self.static_context,
                    dynamic_context=dynamic_chunk,
                    user_question=question,
                )

                usage: TurnUsage = self.provider.send_turn(payload)
                turn_cost = calculate_cost(
                    model=self.provider.model,
                    base_input_tokens=usage.base_input_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    output_tokens=usage.output_tokens,
                )

                record = TurnRecord(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    provider=self.provider.provider_name,
                    model=self.provider.model,
                    layout=layout,
                    turn=turn_idx,
                    base_input_tokens=usage.base_input_tokens,
                    cache_write_tokens=usage.cache_write_tokens,
                    cache_read_tokens=usage.cache_read_tokens,
                    output_tokens=usage.output_tokens,
                    total_input_tokens=usage.total_input_tokens,
                    cache_hit_rate=usage.cache_hit_rate,
                    turn_cost_usd=turn_cost,
                    latency_seconds=usage.latency_seconds,
                )
                summary.records.append(record)

                if on_turn_complete:
                    on_turn_complete(record)

                if turn_idx < self.turns and self.delay_seconds > 0:
                    time.sleep(self.delay_seconds)
        finally:
            self.provider.cleanup_layout(layout)

        return summary

    def run(
        self,
        on_layout_start: LayoutStartCallback | None = None,
        on_turn_complete: TurnCallback | None = None,
    ) -> BenchmarkResult:
        """Run both 'bad_prefix' and 'good_prefix' sequentially and compile results."""
        started_at = datetime.now(timezone.utc).isoformat()
        bad_summary = self.run_layout(
            "bad_prefix",
            on_layout_start=on_layout_start,
            on_turn_complete=on_turn_complete,
        )
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        good_summary = self.run_layout(
            "good_prefix",
            on_layout_start=on_layout_start,
            on_turn_complete=on_turn_complete,
        )
        return BenchmarkResult(
            provider=self.provider.provider_name,
            model=self.provider.model,
            turns=self.turns,
            started_at=started_at,
            bad_summary=bad_summary,
            good_summary=good_summary,
        )
