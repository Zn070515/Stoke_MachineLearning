"""Channel → vintage-status governance declaration (v15 §六/§十).

数据「版本时间」声明：宏观与基本面数据缺少 version time —— 即使有真实的
``disclose_date``，今天读到的值也可能是原始发布值的「后续修订」（财报更正
restatements、宏观统计修订 statistical revisions、历史行业分类重构 industry
reclassification、供应商回补 vendor backfills）。把今天读到的「最新修订版」值
映射回它的原始发布日期，就是 revision leakage。

完整的 ``vintage_time / revision_time / retrieved_at / version_id`` 系统是
理想高线（aspirational high bar，计划中明确 deferred）。§六/§十 的最低要求是：

    正式报告要标明哪些 Channel 是真正 vintage-safe，哪些是「按发布日期对齐的
    最新修订历史」，并在正式（formal）门禁下按 ``VintagePolicy`` 强制放行集合。

本模块就是那个最低要求的落地：一份策展式的 channel→status 声明，供
``scripts/production/data_quality_gate_run.py`` 的正式报告消费，并由
``stoke_ml/data/vintage_policy.py`` 推导训练允许消费的 channel 集合。它只做
标注与状态分类，不做任何存储层改动（不新增 revision_time / version_id 列）。

四种状态（status semantics）
---------------------------
``"raw_vintage_safe"``
    Historical values are immutable point-in-time snapshots / events: a value
    read today for date T equals what was knowable at T. No later revision
    rewrites the past.  历史值是不可变的时点快照/事件；今天读到的 T 日值 = T 日
    可知值，后续修订不会改写过去。

``"derived_versioned"``
    The stored series is DERIVED from realized prices AND versioned: qfq
    re-anchoring recomputes ALL historical levels whenever a corporate action
    occurs.  Honest label: derived + versioned, NOT an immutable raw snapshot.
    该 channel 是「派生 + 版本化」：前复权 qfq 在每次分红/拆股时重算全部历史
    复权价 —— 不是不可变的原始快照。

``"latest_revised_aligned"``
    The channel stores the CURRENT (latest-revised) value mapped back to its
    original event / publication date. A value read today for date T may embed
    revisions made after T → potential revision leakage.  该 channel 存的是
    「当前最新修订版」值，按原始事件/发布日期对齐；今天读到的 T 日值可能嵌入了
    T 之后才发生的修订 → 潜在 revision leakage。

``"unknown_vintage"``
    RESERVED default — the return value of ``status_of()`` for any channel with
    no curated declaration.  NEVER used as a declared ``ChannelVintageStatus``;
    undeclared channels are denied by default under every policy.  保留默认态：
    ``status_of()`` 对未声明 channel 的返回值；绝不作声明的 status。

分类准则（honest, not comfortable）
----------------------------------
- ``raw_vintage_safe``（不可变事件 / 当日快照）：
  sentiment / guba / comment（已发布的不可变文本/评分）、announcement（更正以
  「新公告」出现而非改写）、margin / northbound / dragon_tiger / capital_flow /
  etf_flow / block_trade（交易所当日快照，按当日记录不改写）、lockup /
  dividend（不可变事件日程）、board（已实现的当日价格行为）、limit_up（deferred）、
  topic（ablation-only，基于不可变标题文本）。
- ``derived_versioned``（派生自已实现价格、版本化）：
  daily_qfq（前复权日线 —— 每次分红/拆股重算全部历史复权价，今天读到的历史
  已嵌入未来的公司行为）、market_env（广度派生自 qfq 价格）、industry（行业排名
  派生自 qfq 价格）。据实标注「派生 + 版本化」，不伪装成不可变原始快照。
- ``latest_revised_aligned``（修订敏感 / 派生自修订敏感）：
  fundamental（财报更正）、macro（宏观统计修订）、earnings（盈利预测被修正）、
  sector / concept / index_membership（历史分类重构）、valuation（价格×基本面，
  继承基本面修订）、market_env_refine（派生自 macro，继承其修订风险）、
  pledge / shareholder（登记簿快照可能被更正/重述）。

策略（policy）
-------------
``stoke_ml/data/vintage_policy.py`` 的 ``VintagePolicy`` 决定训练可消费的
channel 集合：``safe-only``（默认）只放行 raw_vintage_safe + derived_versioned
（derived_versioned 含 daily_qfq，保证价格 channel 永远可用）；
``allow-revised`` 额外放行 latest_revised_aligned。正式门禁在 formal 模式下
强制执行本声明 —— 不完整（missing documented channel）或自相矛盾（拒绝
daily_qfq）的声明使 ``passed = False``。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChannelVintageStatus:
    """One data channel's vintage classification.

    Attributes:
        channel:   the data channel name (matches the CLAUDE.md ``use_*``
                   dimension it feeds, plus the core daily/qfq price channel).
        status:    one of ``"raw_vintage_safe"`` / ``"derived_versioned"`` /
                   ``"latest_revised_aligned"`` (the 3 DECLARED states).
                   ``"unknown_vintage"`` is the reserved deny-by-default
                   fallback of ``status_of()`` and is NEVER a declared status.
        rationale: why — concrete and honest, one or two sentences.
    """

    channel: str
    status: str
    rationale: str


# The curated, hand-reviewed declaration.  Order is stable (grouped by status
# for readability, CLAUDE.md table order preserved within each group) so
# consumers can rely on deterministic serialization.  Every channel MUST carry
# a non-empty rationale.
CHANNEL_VINTAGE: tuple[ChannelVintageStatus, ...] = (
    # ── raw_vintage_safe: immutable published text / ratings ────────────────
    ChannelVintageStatus(
        channel="sentiment",
        status="raw_vintage_safe",
        rationale="Immutable published news text/headlines; a correction appears "
                  "as a new article, never a rewrite of a stored row.",
    ),
    ChannelVintageStatus(
        channel="guba",
        status="raw_vintage_safe",
        rationale="Immutable published forum posts keyed by post_id; stored "
                  "sentiment is a snapshot of the post as fetched.",
    ),
    ChannelVintageStatus(
        channel="comment",
        status="raw_vintage_safe",
        rationale="Immutable published rating/comment snapshots keyed by date; "
                  "a rating for date T is what was knowable at T.",
    ),
    # ── raw_vintage_safe: disclosure events ─────────────────────────────────
    ChannelVintageStatus(
        channel="announcement",
        status="raw_vintage_safe",
        rationale="Immutable disclosure events; a 更正/restatement appears as a "
                  "NEW announcement rather than overwriting the original row.",
    ),
    # ── raw_vintage_safe: daily exchange snapshots ──────────────────────────
    ChannelVintageStatus(
        channel="margin",
        status="raw_vintage_safe",
        rationale="Daily 融资融券 balance is an exchange snapshot recorded for "
                  "the day; not later rewritten.",
    ),
    ChannelVintageStatus(
        channel="northbound",
        status="raw_vintage_safe",
        rationale="Daily northbound flow/holding is an exchange snapshot "
                  "recorded for the day.",
    ),
    ChannelVintageStatus(
        channel="dragon_tiger",
        status="raw_vintage_safe",
        rationale="Daily 龙虎榜 seat/amount data is an exchange snapshot "
                  "recorded for the trading day it occurred.",
    ),
    ChannelVintageStatus(
        channel="capital_flow",
        status="raw_vintage_safe",
        rationale="Daily 资金流向 is a vendor-computed snapshot recorded for the "
                  "day; stored as an as-of value.",
    ),
    ChannelVintageStatus(
        channel="etf_flow",
        status="raw_vintage_safe",
        rationale="Daily sector ETF flow is a snapshot recorded for the day.",
    ),
    ChannelVintageStatus(
        channel="block_trade",
        status="raw_vintage_safe",
        rationale="Daily 大宗交易 records are exchange-reported events recorded "
                  "for the trading day.",
    ),
    # ── raw_vintage_safe: immutable event schedules ─────────────────────────
    ChannelVintageStatus(
        channel="lockup",
        status="raw_vintage_safe",
        rationale="Lockup 解禁 schedule is an event list; each unlock date is "
                  "recorded as an immutable event.",
    ),
    ChannelVintageStatus(
        channel="dividend",
        status="raw_vintage_safe",
        rationale="Dividend 分红 schedule is an event list (ex-date/amount); each "
                  "event is recorded as an immutable snapshot.",
    ),
    # ── raw_vintage_safe: realized daily price action / event ecology ───────
    ChannelVintageStatus(
        channel="board",
        status="raw_vintage_safe",
        rationale="打板/limit-up ecology features are realized daily price "
                  "actions, not restated after the fact.",
    ),
    # ── raw_vintage_safe: deferred / ablation-only channels (still declared) ─
    ChannelVintageStatus(
        channel="limit_up",
        status="raw_vintage_safe",
        rationale="Deferred (NOT yet in the default feature set).  Realized "
                  "daily limit-up ecology, immutable once recorded.",
    ),
    ChannelVintageStatus(
        channel="topic",
        status="raw_vintage_safe",
        rationale="Ablation-only (OFF by default).  Derived from immutable news "
                  "titles with a frozen global_frozen model; note the §七 PIT "
                  "corpus concern, which is a leakage issue separate from "
                  "vintage.",
    ),
    # ── derived_versioned: derived from realized prices AND versioned ────────
    ChannelVintageStatus(
        channel="daily_qfq",
        status="derived_versioned",
        rationale="前复权 (qfq) adjustment recomputes ALL historical adjusted "
                  "closes whenever a new dividend/split occurs, so today's qfq "
                  "history embeds future corporate actions.  The stored series "
                  "is DERIVED from realized prices AND versioned (re-anchored "
                  "on every corporate action) — not an immutable raw snapshot.  "
                  "The repo chose the RESEARCH_QFQ_DAILY contract knowing this; "
                  "this label does not hide it.",
    ),
    ChannelVintageStatus(
        channel="market_env",
        status="derived_versioned",
        rationale="Market-breadth features are DERIVED from realized daily "
                  "prices, which are qfq-adjusted and hence versioned: the "
                  "breadth recomputes whenever the underlying qfq price channel "
                  "is re-anchored by a corporate action.",
    ),
    ChannelVintageStatus(
        channel="industry",
        status="derived_versioned",
        rationale="Daily industry ranking is DERIVED from realized prices — the "
                  "qfq price series is versioned (recomputed on every corporate "
                  "action), so the ranking inherits that; its membership also "
                  "derives from the sector classification, which is "
                  "restructured historically.",
    ),
    # ── latest_revised_aligned: registry snapshots / classification ──────────
    ChannelVintageStatus(
        channel="shareholder",
        status="latest_revised_aligned",
        rationale="股东户数/holdings registry snapshots are published and can be "
                  "corrected or restated by the issuer; stored form is "
                  "latest-revised history aligned to its disclosure date.",
    ),
    ChannelVintageStatus(
        channel="pledge",
        status="latest_revised_aligned",
        rationale="股权质押 registry records can be corrected/restated by the "
                  "exchange; stored form is latest-revised history aligned to "
                  "its event date.",
    ),
    ChannelVintageStatus(
        channel="sector",
        status="latest_revised_aligned",
        rationale="行业分类 is restructured historically (历史行业分类重构); "
                  "today's sector for a stock may not equal its classification "
                  "at an earlier date.",
    ),
    ChannelVintageStatus(
        channel="concept",
        status="latest_revised_aligned",
        rationale="概念分类 is restructured historically; today's concept "
                  "membership may embed later reclassification.",
    ),
    ChannelVintageStatus(
        channel="index_membership",
        status="latest_revised_aligned",
        rationale="Index constituents are backfilled/restructured historically; "
                  "today's membership for date T may embed later changes.",
    ),
    # ── latest_revised_aligned: the documented revision-leakage source ───────
    ChannelVintageStatus(
        channel="fundamental",
        status="latest_revised_aligned",
        rationale="Financial statements are restated/corrected (财报更正); a "
                  "value read today for report_date T may embed revisions made "
                  "after T.  Stored as latest-revised history aligned to "
                  "disclose_date.",
    ),
    ChannelVintageStatus(
        channel="macro",
        status="latest_revised_aligned",
        rationale="Macro statistical series are revised (宏观统计修订, e.g. PMI/"
                  "M2/CPI/PPI); a value read today for date T may embed "
                  "statistical revisions made after T.",
    ),
    ChannelVintageStatus(
        channel="earnings",
        status="latest_revised_aligned",
        rationale="Analyst earnings forecasts are repeatedly revised; today's "
                  "forecast mapped to an earlier target date is the latest "
                  "revision, not the original.",
    ),
    ChannelVintageStatus(
        channel="valuation",
        status="latest_revised_aligned",
        rationale="Valuation is computed from price × fundamentals; it inherits "
                  "the fundamental revision risk (and the qfq price revision).",
    ),
    ChannelVintageStatus(
        channel="market_env_refine",
        status="latest_revised_aligned",
        rationale="Macro regime-refine features are derived from macro series; "
                  "they inherit the macro statistical-revision risk.",
    ),
)

# Stable name → status index for programmatic lookup.
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

# The full status vocabulary: the 3 DECLARED states plus the reserved
# ``unknown_vintage`` deny-by-default fallback of ``status_of()``.
KNOWN_STATUSES: frozenset[str] = frozenset({
    "raw_vintage_safe",
    "derived_versioned",
    "latest_revised_aligned",
    "unknown_vintage",
})


def status_of(channel: str, *, vintage_by_name: dict | None = None) -> str:
    """Return a channel's declared vintage status, or ``"unknown_vintage"``.

    ``unknown_vintage`` is the deny-by-default fallback for any channel that has
    no curated declaration — it is never a declared ``ChannelVintageStatus``.
    """
    vbn = vintage_by_name if vintage_by_name is not None else CHANNEL_VINTAGE_BY_NAME
    entry = vbn.get(channel)
    return entry.status if entry is not None else "unknown_vintage"
