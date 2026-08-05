"""Aux-channel column-name constants and state-staleness limits (§十七-1).

These constants drive ``AuxAligner``'s per-channel column selection and the
§P1-7 staleness tracking bounds in ``stoke_ml.features.aux_aligner``.  They live
in their own module so the aligner and any consumer that needs the feature
column names can import them without pulling in the full alignment code.
"""

SENTIMENT_COLS = [
    "sentiment_mean", "sentiment_std", "news_count",
    "positive_ratio", "negative_ratio", "has_news",
]

MARGIN_COLS = [
    "margin_balance", "margin_buy", "short_balance", "margin_net",
]

NORTHBOUND_COLS = [
    "north_hold_pct", "north_net_buy",
]

DRAGON_TIGER_COLS = [
    "lhb_net_amount", "lhb_buy_ratio", "lhb_present",
]

ETF_FLOW_COLS = [
    "sector_etf_flow", "sector_etf_amount",
]

GUBA_COLS = [
    "guba_sentiment_mean", "guba_sentiment_std", "guba_post_count",
    "guba_positive_ratio", "guba_negative_ratio", "has_guba_post",
]

COMMENT_COLS = [
    "comment_score", "comment_attention", "comment_institution",
    "comment_trend", "has_comment",
]

FUNDAMENTAL_COLS = [
    "roe", "roa", "eps", "revenue_yoy", "profit_yoy",
    "debt_ratio", "gross_margin", "net_margin",
]

EARNINGS_COLS = [
    "has_forecast", "net_profit_yoy_low", "net_profit_yoy_high",
    "net_profit_low", "net_profit_high", "forecast_age",
]

VALUATION_COLS = ["pe_ttm", "pb_mrq", "ps_ttm", "pcf_ttm"]

# NOTE: ind_matched_return / stock_vs_industry were REMOVED:
# they map a stock onto its industry via the current-snapshot sector_map.json,
# backfilling today's classification onto historical rows (present-backfill
# bias).  The industry-level columns below are PIT-safe — they are daily
# cross-sectional stats over all industry indexes, with no per-stock membership.
INDUSTRY_COLS = [
    "ind_pct_up", "ind_return_mean", "ind_return_std",
    "ind_return_max", "ind_return_min", "ind_return_skew",
    "ind_dispersion_20d",
]

MACRO_COLS = [
    "shibor_O_N", "shibor_1W", "shibor_2W", "shibor_1M",
    "shibor_3M", "shibor_6M", "shibor_9M", "shibor_1Y",
    "fx_usd_cny", "fx_eur_cny", "fx_jpy_cny", "fx_hkd_cny", "fx_gbp_cny",
    "bond_cn_2y", "bond_cn_5y", "bond_cn_10y", "bond_cn_30y",
    "bond_cn_10y2y_spread",
    "bond_us_2y", "bond_us_5y", "bond_us_10y", "bond_us_30y",
    "bond_us_10y2y_spread",
    "gdp_cn_yoy", "m2_yoy", "m1_yoy", "sf_total", "cpi_yoy",
]

# Must match scripts/_preprocess_market_env.py output exactly (7 cols, no
# limit-up temperature cols — that family is deferred).
MARKET_ENV_COLS = [
    "high_low_ratio", "mkt_cap_total_z", "avg_account_cap_z",
    "investor_new_num", "investor_new_z", "market_adv_ratio", "market_turnover_z",
]

# §P1-7: per state-channel maximum acceptable staleness in CALENDAR days.
# A forward-filled state value older than this is flagged `{prefix}_is_stale`
# (and `{prefix}_staleness_days` records the exact age) so a balance last
# updated 60 days ago is never silently consumed as the "latest true state".
# Defaults follow each channel's disclosure cadence: daily (margin / northbound
# / valuation / macro / market_env / industry), quarterly (fundamental /
# shareholder), monthly (pledge), and per-disclosure forecast bands (earnings).
# Overridable per call via ``max_staleness=`` on the merge.
STATE_MAX_STALENESS: dict[str, int] = {
    "margin": 5,
    "northbound": 5,
    "valuation": 5,
    "fundamental": 120,
    "earnings": 120,
    "shareholder": 120,
    "pledge": 40,
    "macro": 5,
    "market_env": 5,
    "industry": 5,
}
