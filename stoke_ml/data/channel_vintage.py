"""Channel → 3-dim vintage-governance declaration (v15 §六/§十, v16 §十二).

数据「版本时间」声明：宏观与基本面数据缺少 version time —— 即使有真实的
``disclose_date``，今天读到的值也可能是原始发布值的「后续修订」（财报更正
restatements、宏观统计修订 statistical revisions、历史行业分类重构 industry
reclassification、供应商回补 vendor backfills）。把今天读到的「最新修订版」值
映射回它的原始发布日期，就是 revision leakage。

完整的 ``vintage_time / revision_time / retrieved_at / version_id`` 系统是
理想高线（aspirational high bar，计划中明确 deferred）。§六/§十/§十二 的最低
要求是：

    正式报告要标明每个 Channel 的三维 vintage 分类（source_vintage /
    transform / pit_alignment），并在正式（formal）门禁下按 ``VintagePolicy``
    强制放行集合。

本模块就是那个最低要求的落地：一份策展式的 channel→三维分类 声明，供
``scripts/production/data_quality_gate_run.py`` 的正式报告消费，并由
``stoke_ml/data/vintage_policy.py`` 推导训练允许消费的 channel 集合。它只做
标注与分类，不做任何存储层改动（不新增 revision_time / version_id 列）。

三维分类（3-dim classification semantics）
-------------------------------------------
``source_vintage`` —— 值本身的来源是否「时点固定」：
``"immutable_snapshot"``
    Historical values are immutable point-in-time snapshots / events: a value
    read today for date T equals what was knowable at T.  No later revision
    rewrites the past.  历史值是不可变的时点快照/事件（不可变事件 / 交易所当日
    快照 / 原始价格输入）；今天读到的 T 日值 = T 日可知值。
``"latest_revised"``
    The value read today for date T may embed revisions made after T (external
    restatements / reclassifications) — the current (latest-revised) value
    mapped back to its original event / publication date → potential revision
    leakage.  今天读到的 T 日值可能嵌入了 T 之后才发生的修订。
``"unknown"``
    RESERVED default — the return value of the accessors for any channel with
    no curated declaration.  NEVER a declared ``source_vintage``; undeclared
    channels are denied by default under every policy.

``transform`` —— 存储值是否经过模型/公式变换。这是 RECORDING dimension，不是
deny axis：
``"raw"``
    Stored as-fetched; no model/formula transform.  原样存储，未经变换。
``"model_versioned"``
    Passes through a model (e.g. FinBERT sentiment) — regenerable if the model
    version changes, so the model version must be RECORDED, not the channel
    denied.
``"formula_versioned"``
    Vendor/derived formula (e.g. capital_flow is vendor-computed, qfq
    re-anchoring, market_env breadth) — the version must be RECORDED.
``"unknown"``
    RESERVED default of the accessors; NEVER a declared ``transform``.

``pit_alignment`` —— 决策日对齐的可信度：
``"verified"``
    Decision-date alignment is deterministic/verified at storage (immutable
    channels, incl. the post-close→next-trading-day PIT mapping).
``"proxy"``
    Value aligned to its original event date but may embed later revisions /
    re-anchoring, so the alignment to "what was knowable at T" is only a proxy.
``"unknown"``
    RESERVED default of the accessors; NEVER a declared ``pit_alignment``.

安全门禁在 revision-safe / allow-revised 下只看 source + transform 两层
（``vintage_policy.channel_allowed``）：``revision-safe``（原 ``safe-only``，
§T3 改名）放行 ``immutable_snapshot`` 来源且 transform 非 ``"unknown"`` 的
channel，拒绝 ``latest_revised`` 来源；``allow-revised`` 额外放行
``latest_revised`` 来源。``pit_alignment`` 在 revision-safe / allow-revised
下是记录维，不参与放行判定；新增的 ``headline-strict`` 档把它升级为
admission gate —— 要求 ``pit_alignment == "verified"``，proxy 对齐的 channel
除非在 scale-invariant waiver 白名单（``HEADLINE_STRICT_WAIVER_CHANNELS``，
daily_qfq / market_env）否则拒绝。

分类准则（honest, not comfortable）
-----------------------------------
- ``immutable_snapshot``（不可变事件 / 当日快照 / 原始价格输入）：
  sentiment / guba / comment（已发布的不可变文本/评分）、announcement（更正以
  「新公告」出现而非改写）、margin / northbound / dragon_tiger / capital_flow /
  etf_flow / block_trade（交易所当日快照，按当日记录不改写）、lockup /
  dividend（不可变事件日程）、board（已实现的当日价格行为）、limit_up（deferred）、
  topic（ablation-only，基于不可变标题文本）、daily_qfq（原始 OHLC + 复权因子
  不可变，versioned/derived 的一面落在 transform=formula_versioned）、
  market_env / industry（原始价格输入不变；排名/广度由公式派生）。
- ``latest_revised``（修订敏感 / 派生自修订敏感）：
  fundamental（财报更正）、macro（宏观统计修订）、earnings（盈利预测被修正）、
  sector / concept / index_membership（历史分类重构）、valuation（价格×基本面，
  继承基本面修订）、market_env_refine（派生自 macro，继承其修订风险）、
  pledge / shareholder（登记簿快照可能被更正/重述）。

策略（policy）
-------------
``stoke_ml/data/vintage_policy.py`` 的 ``VintagePolicy`` 决定训练可消费的
channel 集合：``revision-safe``（默认，§T3 由 ``safe-only`` 改名）只放行
immutable_snapshot 来源的 channel（含 daily_qfq，保证价格 channel 永远可用），
拒绝 latest_revised 来源；``allow-revised`` 额外放行 latest_revised 来源；
``headline-strict``（§T3 新增，最严）还要求 ``pit_alignment == "verified"``
（scale-invariant 的 daily_qfq / market_env 有显式 waiver）。正式门禁在
formal 模式下强制执行本声明 —— 不完整（missing documented channel 或三维
声明含 ``"unknown"``）或自相矛盾（拒绝 daily_qfq）的声明使 ``passed = False``。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelVintageStatus:
    """One data channel's 3-dim vintage classification.

    Attributes:
        channel:        the data channel name (matches the CLAUDE.md ``use_*``
                        dimension it feeds, plus the core daily/qfq price channel).
        source_vintage: one of ``"immutable_snapshot"`` / ``"latest_revised"``
                        (the 2 DECLARED source states).  ``"unknown"`` is the
                        reserved deny-by-default fallback of the accessors and
                        is NEVER a declared ``source_vintage``.
        transform:      one of ``"raw"`` / ``"model_versioned"`` /
                        ``"formula_versioned"`` (the 3 DECLARED transform
                        states).  ``"unknown"`` is the reserved accessor
                        fallback and is NEVER a declared ``transform``.  This is
                        a RECORDING dimension, not a deny axis.
        pit_alignment:  one of ``"verified"`` / ``"proxy"`` (the 2 DECLARED
                        alignment states).  ``"unknown"`` is the reserved
                        accessor fallback and is NEVER a declared
                        ``pit_alignment``.
        rationale:      why — concrete and honest, one or two sentences.
    """

    channel: str
    source_vintage: str
    transform: str
    pit_alignment: str
    rationale: str


# The curated, hand-reviewed declaration.  Order is stable (grouped by axis for
# readability, CLAUDE.md table order preserved within each group) so consumers
# can rely on deterministic serialization.  Every channel MUST carry a
# non-empty rationale and three DECLARED (non-``"unknown"``) labels.
CHANNEL_VINTAGE: tuple[ChannelVintageStatus, ...] = (
    # ── immutable_snapshot / model_versioned: immutable text + model score ────
    ChannelVintageStatus(
        channel="sentiment",
        source_vintage="immutable_snapshot",
        transform="model_versioned",
        pit_alignment="verified",
        rationale="Immutable published news text/headlines; a correction appears "
                  "as a new article, never a rewrite of a stored row.  The "
                  "sentiment score is FinBERT-derived (model_versioned): "
                  "regenerable if the model version changes, so the model "
                  "version must be RECORDED.",
    ),
    ChannelVintageStatus(
        channel="guba",
        source_vintage="immutable_snapshot",
        transform="model_versioned",
        pit_alignment="verified",
        rationale="Immutable published forum posts keyed by post_id; stored "
                  "sentiment is a snapshot of the post as fetched.  The "
                  "sentiment score is model-derived (model_versioned): "
                  "regenerable if the model version changes, so the model "
                  "version must be RECORDED.",
    ),
    ChannelVintageStatus(
        channel="comment",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="Immutable published rating/comment snapshots keyed by date; "
                  "a rating for date T is what was knowable at T.",
    ),
    # ── immutable_snapshot: disclosure events ────────────────────────────────
    ChannelVintageStatus(
        channel="announcement",
        source_vintage="immutable_snapshot",
        transform="model_versioned",
        pit_alignment="verified",
        rationale="Immutable disclosure events; a 更正/restatement appears as a "
                  "NEW announcement rather than overwriting the original row.  "
                  "The sentiment score is model-derived (model_versioned): "
                  "regenerable if the model version changes, so the model "
                  "version must be RECORDED.",
    ),
    # ── immutable_snapshot: daily exchange snapshots ─────────────────────────
    ChannelVintageStatus(
        channel="margin",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="Daily 融资融券 balance is an exchange snapshot recorded for "
                  "the day; not later rewritten.",
    ),
    ChannelVintageStatus(
        channel="northbound",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="Daily northbound flow/holding is an exchange snapshot "
                  "recorded for the day.",
    ),
    ChannelVintageStatus(
        channel="dragon_tiger",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="Daily 龙虎榜 seat/amount data is an exchange snapshot "
                  "recorded for the trading day it occurred.",
    ),
    ChannelVintageStatus(
        channel="capital_flow",
        source_vintage="immutable_snapshot",
        transform="formula_versioned",
        pit_alignment="verified",
        rationale="Daily 资金流向 is a VENDOR-COMPUTED series "
                  "(formula_versioned): the vendor's money-flow formula derives "
                  "the value, so a formula/definition change can regenerate "
                  "history — the version must be RECORDED, not the channel "
                  "denied.  Stored as an as-of daily snapshot, recorded for the "
                  "day.",
    ),
    ChannelVintageStatus(
        channel="etf_flow",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="Daily sector ETF flow is a snapshot recorded for the day.",
    ),
    ChannelVintageStatus(
        channel="block_trade",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="Daily 大宗交易 records are exchange-reported events recorded "
                  "for the trading day.",
    ),
    # ── immutable_snapshot: immutable event schedules ─────────────────────────
    ChannelVintageStatus(
        channel="lockup",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="Lockup 解禁 schedule is an event list; each unlock date is "
                  "recorded as an immutable event.",
    ),
    ChannelVintageStatus(
        channel="dividend",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="Dividend 分红 schedule is an event list (ex-date/amount); each "
                  "event is recorded as an immutable snapshot.",
    ),
    # ── immutable_snapshot: realized daily price action / event ecology ───────
    ChannelVintageStatus(
        channel="board",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="打板/limit-up ecology features are realized daily price "
                  "actions, not restated after the fact.",
    ),
    # ── immutable_snapshot: deferred / ablation-only channels (still declared) ─
    ChannelVintageStatus(
        channel="limit_up",
        source_vintage="immutable_snapshot",
        transform="raw",
        pit_alignment="verified",
        rationale="Deferred (NOT yet in the default feature set).  Realized "
                  "daily limit-up ecology, immutable once recorded.",
    ),
    ChannelVintageStatus(
        channel="topic",
        source_vintage="immutable_snapshot",
        transform="model_versioned",
        pit_alignment="verified",
        rationale="Ablation-only (OFF by default).  Derived from immutable news "
                  "titles with a frozen global_frozen model; note the §七 PIT "
                  "corpus concern, which is a leakage issue separate from "
                  "vintage.  The topic score is model-derived "
                  "(model_versioned): regenerable if the model version changes, "
                  "so the model version must be RECORDED.",
    ),
    # ── immutable_snapshot source: derived + versioned transform (formula) ────
    ChannelVintageStatus(
        channel="daily_qfq",
        source_vintage="immutable_snapshot",
        transform="formula_versioned",
        pit_alignment="proxy",
        rationale="Raw OHLC + adjustment-factor history are immutable snapshots; "
                  "the 前复权 (qfq) re-anchoring is a formula transform "
                  "(formula_versioned) that recomputes ALL historical adjusted "
                  "closes whenever a new dividend/split occurs, so today's qfq "
                  "history embeds future corporate actions.  Because of that, "
                  "its alignment to 'what was knowable at T' is only a PROXY, "
                  "not verified.  The repo chose the RESEARCH_QFQ_DAILY "
                  "contract knowing this; this label does not hide it.",
    ),
    ChannelVintageStatus(
        channel="market_env",
        source_vintage="immutable_snapshot",
        transform="formula_versioned",
        pit_alignment="proxy",
        rationale="Market-breadth features are computed from realized daily "
                  "prices via a formula (formula_versioned); the underlying qfq "
                  "price channel is re-anchored by a corporate action, so the "
                  "alignment to 'what was knowable at T' is only a PROXY.",
    ),
    ChannelVintageStatus(
        channel="industry",
        source_vintage="immutable_snapshot",
        transform="formula_versioned",
        pit_alignment="proxy",
        rationale="Daily industry ranking is computed from realized prices via a "
                  "formula (formula_versioned) — the qfq price series is "
                  "versioned (recomputed on every corporate action), so the "
                  "ranking inherits that; its membership also derives from the "
                  "sector classification, which is restructured historically "
                  "(a documented caveat).  Alignment to 'what was knowable at "
                  "T' is only a PROXY, so the label stays honest while the "
                  "channel remains admissible.",
    ),
    # ── latest_revised: registry snapshots / classification ───────────────────
    ChannelVintageStatus(
        channel="shareholder",
        source_vintage="latest_revised",
        transform="raw",
        pit_alignment="proxy",
        rationale="股东户数/holdings registry snapshots are published and can be "
                  "corrected or restated by the issuer; stored form is "
                  "latest-revised history aligned to its disclosure date.",
    ),
    ChannelVintageStatus(
        channel="pledge",
        source_vintage="latest_revised",
        transform="raw",
        pit_alignment="proxy",
        rationale="股权质押 registry records can be corrected/restated by the "
                  "exchange; stored form is latest-revised history aligned to "
                  "its event date.",
    ),
    ChannelVintageStatus(
        channel="sector",
        source_vintage="latest_revised",
        transform="raw",
        pit_alignment="proxy",
        rationale="行业分类 is restructured historically (历史行业分类重构); "
                  "today's sector for a stock may not equal its classification "
                  "at an earlier date.",
    ),
    ChannelVintageStatus(
        channel="concept",
        source_vintage="latest_revised",
        transform="raw",
        pit_alignment="proxy",
        rationale="概念分类 is restructured historically; today's concept "
                  "membership may embed later reclassification.",
    ),
    ChannelVintageStatus(
        channel="index_membership",
        source_vintage="latest_revised",
        transform="raw",
        pit_alignment="proxy",
        rationale="Index constituents are backfilled/restructured historically; "
                  "today's membership for date T may embed later changes.",
    ),
    # ── latest_revised: the documented revision-leakage source ────────────────
    ChannelVintageStatus(
        channel="fundamental",
        source_vintage="latest_revised",
        transform="raw",
        pit_alignment="proxy",
        rationale="Financial statements are restated/corrected (财报更正); a "
                  "value read today for report_date T may embed revisions made "
                  "after T.  Stored as latest-revised history aligned to "
                  "disclose_date.",
    ),
    ChannelVintageStatus(
        channel="macro",
        source_vintage="latest_revised",
        transform="raw",
        pit_alignment="proxy",
        rationale="Macro statistical series are revised (宏观统计修订, e.g. PMI/"
                  "M2/CPI/PPI); a value read today for date T may embed "
                  "statistical revisions made after T.",
    ),
    ChannelVintageStatus(
        channel="earnings",
        source_vintage="latest_revised",
        transform="raw",
        pit_alignment="proxy",
        rationale="Analyst earnings forecasts are repeatedly revised; today's "
                  "forecast mapped to an earlier target date is the latest "
                  "revision, not the original.",
    ),
    ChannelVintageStatus(
        channel="valuation",
        source_vintage="latest_revised",
        transform="formula_versioned",
        pit_alignment="proxy",
        rationale="Valuation is computed from price × fundamentals via a formula "
                  "(formula_versioned); it inherits the fundamental revision "
                  "risk (and the qfq price revision), so its source is "
                  "latest-revised and its alignment is only a PROXY.",
    ),
    ChannelVintageStatus(
        channel="market_env_refine",
        source_vintage="latest_revised",
        transform="formula_versioned",
        pit_alignment="proxy",
        rationale="Macro regime-refine features are derived from macro series via "
                  "a formula (formula_versioned); they inherit the macro "
                  "statistical-revision risk, so their source is latest-revised "
                  "and their alignment is only a PROXY.",
    ),
)

# Stable name → declaration index for programmatic lookup.
CHANNEL_VINTAGE_BY_NAME: dict[str, ChannelVintageStatus] = {
    e.channel: e for e in CHANNEL_VINTAGE
}

# The 27 documented ``use_*`` dimensions (CLAUDE.md Feature Layer table).
DOCUMENTED_USE_DIMS: frozenset[str] = frozenset({
    "sentiment", "guba", "comment", "announcement", "margin", "northbound",
    "dragon_tiger", "fundamental", "earnings", "valuation", "etf_flow",
    "capital_flow", "block_trade", "shareholder", "lockup", "dividend",
    "board", "sector", "concept", "industry", "macro", "pledge",
    "index_membership", "market_env", "market_env_refine", "limit_up",
    "topic",
})

# The three-axis vocabularies.  The ``"unknown"`` value on each axis is the
# RESERVED deny-by-default fallback of the accessors — NEVER a declared label.
KNOWN_SOURCE_VINTAGES: frozenset[str] = frozenset({
    "immutable_snapshot",
    "latest_revised",
    "unknown",
})
KNOWN_TRANSFORMS: frozenset[str] = frozenset({
    "raw",
    "model_versioned",
    "formula_versioned",
    "unknown",
})
KNOWN_PIT_ALIGNMENTS: frozenset[str] = frozenset({
    "verified",
    "proxy",
    "unknown",
})


def declaration_of(
    channel: str,
    *,
    vintage_by_name: dict | None = None,
) -> ChannelVintageStatus | None:
    """Return a channel's full 3-dim declaration, or None when undeclared."""
    vbn = vintage_by_name if vintage_by_name is not None else CHANNEL_VINTAGE_BY_NAME
    return vbn.get(channel)


def source_vintage_of(channel: str, *, vintage_by_name: dict | None = None) -> str:
    """Return a channel's declared ``source_vintage``, or ``"unknown"``.

    ``"unknown"`` is the deny-by-default fallback for any channel that has no
    curated declaration — it is never a declared ``ChannelVintageStatus``.
    """
    entry = declaration_of(channel, vintage_by_name=vintage_by_name)
    return entry.source_vintage if entry is not None else "unknown"


def transform_of(channel: str, *, vintage_by_name: dict | None = None) -> str:
    """Return a channel's declared ``transform``, or ``"unknown"``.

    ``"unknown"`` is the deny-by-default fallback for any channel that has no
    curated declaration — it is never a declared ``ChannelVintageStatus``.
    """
    entry = declaration_of(channel, vintage_by_name=vintage_by_name)
    return entry.transform if entry is not None else "unknown"


def pit_alignment_of(channel: str, *, vintage_by_name: dict | None = None) -> str:
    """Return a channel's declared ``pit_alignment``, or ``"unknown"``.

    ``"unknown"`` is the deny-by-default fallback for any channel that has no
    curated declaration — it is never a declared ``ChannelVintageStatus``.
    """
    entry = declaration_of(channel, vintage_by_name=vintage_by_name)
    return entry.pit_alignment if entry is not None else "unknown"
