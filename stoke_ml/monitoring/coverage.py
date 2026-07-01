"""Feature coverage monitor: per-source availability and quality stats.

Answers: which data sources have data, for how many stocks/days,
and what are the sentiment distributions?
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Known source → gold directory mapping
_SOURCE_DIRS = {
    "sentiment": "sentiment",
    "guba": "guba_sentiment",
    "xueqiu": "xueqiu_sentiment",
    "comment": "comment_sentiment",
    "announcement": "announcement_sentiment",
}

# Columns expected per source (standard + any extras auto-discovered)
_SOURCE_STD_COLS = {
    "sentiment": ["sentiment_mean", "sentiment_std", "news_count",
                  "positive_ratio", "negative_ratio", "has_news"],
    "guba": ["guba_sentiment_mean", "guba_sentiment_std", "guba_post_count",
             "guba_positive_ratio", "guba_negative_ratio", "has_guba_post"],
    "xueqiu": ["xueqiu_sentiment_mean", "xueqiu_sentiment_std", "xueqiu_post_count",
               "xueqiu_positive_ratio", "xueqiu_negative_ratio", "has_xueqiu_post"],
}


class CoverageMonitor:
    """Track per-source feature coverage and data quality across stocks."""

    def __init__(self, data_dir: str):
        self._root = data_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan_source(
        self, source: str, stock_codes: list[str] | None = None
    ) -> pd.DataFrame:
        """Scan one source's gold data and return per-stock coverage stats.

        Returns DataFrame with columns: stock_code, n_days, n_days_with_data,
        coverage_pct, n_feature_cols, extra_cols, sentiment_mean_avg,
        sentiment_std_avg.
        """
        gold_dir = self._gold_dir(source)
        if not os.path.isdir(gold_dir):
            logger.warning("Gold directory not found: %s", gold_dir)
            return pd.DataFrame()

        if stock_codes is None:
            stock_codes = self._discover_stocks(gold_dir)

        rows = []
        for code in stock_codes:
            df = self._load_gold(source, code)
            if df.empty:
                rows.append({"stock_code": code, "n_days": 0,
                             "n_days_with_data": 0, "coverage_pct": 0.0})
                continue

            std_cols = _SOURCE_STD_COLS.get(source, [])
            extra_cols = [c for c in df.columns
                          if c not in std_cols and c not in ("date", "stock_code")]

            has_data = df.get("has_news" if source == "sentiment"
                              else f"has_{source}_post",
                              pd.Series(True, index=df.index))
            n_with_data = int(has_data.sum()) if has_data.dtype == bool else len(df)

            sent_col = self._sentiment_col(source, df)
            sent_mean = float(df[sent_col].mean()) if sent_col and sent_col in df.columns else np.nan
            sent_std = float(df[sent_col].std()) if sent_col and sent_col in df.columns else np.nan

            rows.append({
                "stock_code": code,
                "n_days": len(df),
                "n_days_with_data": n_with_data,
                "coverage_pct": round(100.0 * n_with_data / max(len(df), 1), 1),
                "n_feature_cols": len(std_cols) + len(extra_cols),
                "n_extra_cols": len(extra_cols),
                "extra_cols": ",".join(extra_cols) if extra_cols else "",
                "sentiment_mean_avg": round(sent_mean, 4) if not np.isnan(sent_mean) else np.nan,
                "sentiment_std_avg": round(sent_std, 4) if not np.isnan(sent_std) else np.nan,
            })

        return pd.DataFrame(rows).sort_values("coverage_pct", ascending=False)

    def report(self, stock_codes: list[str] | None = None) -> str:
        """Generate a multi-source coverage report as formatted text."""
        lines = ["=" * 70, "FEATURE COVERAGE REPORT", "=" * 70]

        for source in _SOURCE_DIRS:
            stats = self.scan_source(source, stock_codes)
            if stats.empty:
                lines.append(f"\n  {source}: no data")
                continue

            n_stocks = len(stats)
            n_with_data = int((stats["n_days_with_data"] > 0).sum())
            avg_coverage = stats["coverage_pct"].mean()
            max_extra = stats["n_extra_cols"].max()

            lines.append(
                f"\n  {source}: {n_with_data}/{n_stocks} stocks, "
                f"avg {avg_coverage:.0f}% coverage, "
                f"up to {max_extra} extra cols"
            )

            if max_extra > 0:
                sample = stats[stats["n_extra_cols"] > 0]
                if len(sample) > 0:
                    lines.append(f"    extra cols: {sample.iloc[0]['extra_cols']}")

        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _gold_dir(self, source: str) -> str:
        subdir = _SOURCE_DIRS.get(source, source)
        return os.path.join(self._root, "a_shares", subdir)

    def _discover_stocks(self, gold_dir: str) -> list[str]:
        codes = set()
        for root, _dirs, files in os.walk(gold_dir):
            for f in files:
                if f.endswith(".parquet"):
                    codes.add(f.replace(".parquet", ""))
        return sorted(codes)

    def _load_gold(self, source: str, stock_code: str) -> pd.DataFrame:
        """Load gold data preferring flat file, falling back to partitions."""
        base = self._gold_dir(source)
        flat = os.path.join(base, f"{stock_code}.parquet")
        if os.path.isfile(flat):
            return pd.read_parquet(flat)

        frames = []
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f == f"{stock_code}.parquet":
                    frames.append(pd.read_parquet(os.path.join(root, f)))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _sentiment_col(source: str, df: pd.DataFrame) -> str | None:
        candidates = {
            "sentiment": "sentiment_mean",
            "guba": "guba_sentiment_mean",
            "xueqiu": "xueqiu_sentiment_mean",
        }
        col = candidates.get(source)
        return col if col and col in df.columns else None


def coverage_report(data_dir: str, stock_codes: list[str] | None = None) -> str:
    """Convenience: generate and print a multi-source coverage report."""
    monitor = CoverageMonitor(data_dir)
    return monitor.report(stock_codes)
