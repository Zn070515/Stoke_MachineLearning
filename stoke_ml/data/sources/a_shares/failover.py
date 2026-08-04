"""Failover orchestrator for A-share data sources.

Tries sources in priority order:
0. Efinance (preferred - fast, reliable)
1. AKShare (fallback - comprehensive)
2. Tushare (optional - requires token)
3. Baostock (last resort - free, limited)

All providers normalize to a common convention at the adapter boundary:
OHLC is 前复权 (qfq), ``volume`` is 股 (shares), ``amount`` is 元 (CNY).
Cross-source backfill rebases the older segment's OHLC onto the primary
source's 前复权 anchor to avoid a fake price jump at the seam (v6 §六).
"""
import time
import logging
import pandas as pd
from stoke_ml.data.sources.a_shares.base import AShareSourceBase
from stoke_ml.data.sources.a_shares.efinance_source import EfinanceSource
from stoke_ml.data.sources.a_shares.akshare_source import AKShareSource
from stoke_ml.data.sources.a_shares.tushare_source import TushareSource
from stoke_ml.data.sources.a_shares.baostock_source import BaostockSource

logger = logging.getLogger(__name__)


class AShareDownloader:
    """Multi-source A-share data downloader with automatic failover."""

    def __init__(self):
        self._sources: list[AShareSourceBase] = [
            EfinanceSource(),
            AKShareSource(),
            TushareSource(),
            BaostockSource(),
        ]
        self._failure_counts: dict[str, int] = {}
        self._circuit_open: dict[str, float] = {}
        self._cooldown_sec = 300
        self._failure_threshold = 15

    def fetch_daily(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        df = None
        source_used = None

        for source in self._sources:
            name = source.SOURCE_NAME
            if not source.is_available():
                logger.debug(f"Source {name} unavailable, skipping")
                continue
            if self._is_circuit_open(name):
                logger.debug(f"Circuit open for {name}, skipping")
                continue

            df = source.fetch_daily(stock_code, start_date, end_date)
            if len(df) > 0:
                self._record_success(name)
                source_used = name
                break
            else:
                self._record_failure(name)
                logger.warning(f"Source {name} returned empty for {stock_code}")

        if df is None or len(df) == 0:
            logger.error(f"All sources failed for {stock_code}")
            return pd.DataFrame()

        # ── Date-range stitching: backfill pre-2015 data via Baostock ──
        requested_start = pd.to_datetime(start_date).date()
        got_start = pd.to_datetime(df["date"]).min().date()
        if got_start > requested_start:
            # Fetch THROUGH got_start so there is an overlap day to calibrate
            # the 前复权 anchors (Baostock anchors to its returned window's
            # end, EastMoney/others to the latest date).  See _stitch_segments.
            backfill_end = got_start + pd.Timedelta(days=5)
            got_start_str = str(got_start)
            got_end_str = str(pd.to_datetime(df["date"]).max().date() if len(df) else "?")
            logger.info(
                "  %s: %s returned %s→%s, backfilling %s→%s via Baostock",
                stock_code, source_used, got_start_str, got_end_str,
                start_date, backfill_end,
            )
            try:
                bs_source = self._sources[-1]  # Baostock is last in priority list
                bs_df = bs_source.fetch_daily(
                    stock_code, start_date, str(backfill_end)
                )
                if len(bs_df) > 0:
                    rebased, ratio = self._stitch_segments(bs_df, df)
                    if ratio is not None:
                        df = pd.concat([rebased, df], ignore_index=True)
                        df = df.sort_values("date").reset_index(drop=True)
                        df = df.drop_duplicates(subset="date", keep="last")
                        df.attrs["backfilled_from"] = "baostock"
                        logger.info(
                            "  %s: stitched %d + %d = %d rows [%s → %s] "
                            "(seam rebase ratio=%.4f)",
                            stock_code, len(rebased), len(df) - len(rebased),
                            len(df),
                            str(pd.to_datetime(df["date"]).min().date()),
                            str(pd.to_datetime(df["date"]).max().date()),
                            ratio,
                        )
                        if not 0.5 <= ratio <= 2.0:
                            logger.warning(
                                "  %s: 前复权 anchor gap %.2f× at the %s/%s seam; "
                                "backfill OHLC was rebased to stay continuous "
                                "(v6 §六 seam guard)",
                                stock_code, ratio, start_date, got_start_str,
                            )
                    else:
                        logger.warning(
                            "  %s: Baostock backfill has no overlap with primary "
                            "(%s), cannot calibrate seam — appending raw",
                            stock_code, got_start_str,
                        )
                        df = pd.concat([bs_df, df], ignore_index=True)
                        df = df.sort_values("date").reset_index(drop=True)
                        df = df.drop_duplicates(subset="date", keep="last")
                else:
                    logger.info("  %s: Baostock backfill returned empty, keeping %d rows",
                                stock_code, len(df))
            except Exception as e:
                logger.warning("  %s: Baostock backfill failed: %s", stock_code, e)

        df.attrs["source"] = source_used
        df.attrs["adjustment_mode"] = "qfq"
        return self._repair_pct_change(df)

    @staticmethod
    def _stitch_segments(
        backfill: pd.DataFrame, primary: pd.DataFrame
    ) -> tuple[pd.DataFrame, float | None]:
        """Rebase `backfill` OHLC onto `primary`'s 前复权 anchor (v6 §六).

        Both segments are 前复权 but anchored to different reference dates
        (Baostock anchors to its returned window's end, EastMoney/Tushare/
        AKShare to the latest date), so a naive concat would inject a fake
        price jump at the seam that gets read as a real 涨跌.  Rebasing the
        backfill OHLC by the ratio observed on the shared overlap day(s)
        removes the anchor discontinuity while preserving each segment's
        internal daily returns.  volume/amount/pct_change are untouched
        (already unit-consistent / exchange-provided).

        Returns ``(backfill_without_overlap_rows, ratio)``.  ``ratio`` is
        ``None`` when there is no overlap to calibrate on.
        """
        if backfill.empty or primary.empty:
            return backfill, None
        overlap = sorted(set(backfill["date"]) & set(primary["date"]))
        if not overlap:
            return backfill, None
        bf_close = backfill.set_index("date")["close"]
        pr_close = primary.set_index("date")["close"]
        ratio = float((pr_close.loc[overlap] / bf_close.loc[overlap]).median())
        rebased = backfill.copy()
        for c in ("open", "high", "low", "close"):
            if c in rebased.columns:
                rebased[c] = pd.to_numeric(rebased[c], errors="coerce") * ratio
        return rebased[~rebased["date"].isin(overlap)], ratio

    @staticmethod
    def _repair_pct_change(df: pd.DataFrame) -> pd.DataFrame:
        """Derive pct_change from close wherever a source left it zero/missing.

        Some sources (e.g. AKShare stock_zh_a_daily) do not return 涨跌幅 and
        would otherwise persist 0.0, silently flattening every momentum feature.
        close is the single source of truth; a real flat day recomputes to ~0,
        so overwriting zeros is always safe.
        """
        if df.empty or "close" not in df.columns:
            return df
        df = df.sort_values("date").reset_index(drop=True)
        close = pd.to_numeric(df["close"], errors="coerce")
        if "pct_change" not in df.columns:
            df["pct_change"] = close.pct_change() * 100.0
            return df
        pct = pd.to_numeric(df["pct_change"], errors="coerce")
        bad = (pct.isna() | (pct == 0)) & close.notna()
        if bad.any():
            df.loc[bad, "pct_change"] = close.pct_change().mul(100.0).loc[bad]
        return df

    def _record_failure(self, name: str):
        self._failure_counts[name] = self._failure_counts.get(name, 0) + 1
        if self._failure_counts[name] >= self._failure_threshold:
            self._circuit_open[name] = time.time()

    def _record_success(self, name: str):
        self._failure_counts[name] = 0
        self._circuit_open.pop(name, None)

    def _is_circuit_open(self, name: str) -> bool:
        if name not in self._circuit_open:
            return False
        if time.time() - self._circuit_open[name] >= self._cooldown_sec:
            del self._circuit_open[name]
            self._failure_counts[name] = 0
            return False
        return True
