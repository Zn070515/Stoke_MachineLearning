"""File-level asset contracts for the single-file broadcast channels (§十七).

``industry`` (``a_shares/industry/industry_returns.parquet``) and ``market_env``
(``a_shares/market_breadth/market_env_daily.parquet``) are MARKET-WIDE
single-file stores: one parquet covers every stock, with the trading dates in
the file's DatetimeIndex rather than a column.  They are written by scripts
(``download_industry.py`` / the archived ``_preprocess_market_env.py``) and read
by ``AuxAligner``'s broadcast merge paths, so the file-level contract lives here
in the data layer where both sides import it without a feature-layer dependency.

Both channels are ``immutable_snapshot``-sourced with a ``formula_versioned``
transform and PROXY pit alignment (their values derive from the qfq price
series, which is re-anchored on every corporate action) — see
``channel_vintage``.  ``contract_for_channel`` pulls those labels in, so the
manifest of each broadcast file carries the same vintage the training
admission is judged against.  Their ``extent_column="date"`` is DatetimeIndex
-backed: ``asset_contract._extent`` falls back to the index when the named
column is absent, so the manifest's ``start``/``end`` bound the trading-date
span of the file.
"""
from stoke_ml.data.asset_contract import (
    DataAssetContract,
    contract_for_channel,
)

INDUSTRY_ASSET: DataAssetContract = contract_for_channel(
    "industry",
    data_type="industry_returns",
    partition="single_file",
    extent_column="date",  # DatetimeIndex-backed
    effective_date_policy="index_date",
)

MARKET_ENV_ASSET: DataAssetContract = contract_for_channel(
    "market_env",
    data_type="market_env_daily",
    partition="single_file",
    extent_column="date",  # DatetimeIndex-backed
    effective_date_policy="index_date",
)
