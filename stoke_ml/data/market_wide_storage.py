"""Storage for market-wide data types (dragon-tiger, margin, northbound).

Partitions: data/a_shares/{data_type}/{year}/{month}/{stock_code}.parquet
"""
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

_LOCK_TIMEOUT = 600.0
_LOCK_STALE = 900.0

# Destructive window replacement must reject if the new output would drop too
# many previously-present dates, or silently remove columns that downstream
# readers depend on.  Allow up to 20% date loss by default.
DEFAULT_DEGRADE_THRESHOLD = 0.2
_SCHEMA_RESERVED = frozenset({"date", "stock_code"})


def _acquire_lock(target: str, timeout: float = _LOCK_TIMEOUT) -> str:
    """Exclusive per-file lock via atomic mkdir. Returns the lock dir path."""
    lock_dir = target + ".lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            os.mkdir(lock_dir)
            return lock_dir
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_dir) > _LOCK_STALE:
                    os.rmdir(lock_dir)  # steal stale lock from a crashed process
                    continue
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError(f"could not acquire lock: {lock_dir}")
            time.sleep(0.05)


def _release_lock(lock_dir: str) -> None:
    try:
        os.rmdir(lock_dir)
    except OSError:
        pass

MARKET_DATA_TYPES = [
    "dragon_tiger", "margin", "northbound",
    "capital_flow", "limit_up_zt", "limit_up_zb", "limit_up_dt", "limit_up_yzt",
    "limit_up_sentiment", "block_trade", "shareholder", "lockup", "lockup_upcoming",
    "dividend", "industry_ranking", "concept_blocks",
    "sina_fund_flow",
    # Processed output variants
    "capital_flow_processed", "block_trade_processed", "shareholder_processed",
    "lockup_processed", "dividend_processed", "industry_ranking_processed",
    "concept_blocks_processed", "board_processed", "valuation",
]


class MarketWideStorage:
    """Save/load market-wide data exploded to per-stock Parquet files."""

    def __init__(self, data_dir: str, data_type: str):
        if data_type not in MARKET_DATA_TYPES:
            raise ValueError(
                f"Unknown market data type: {data_type}. "
                f"Must be one of {MARKET_DATA_TYPES}"
            )
        self._root = data_dir
        self._data_type = data_type

    def _base_dir(self) -> str:
        p = os.path.join(self._root, "a_shares", self._data_type)
        os.makedirs(p, exist_ok=True)
        return p

    def save(
        self,
        df: pd.DataFrame,
        replace_range: bool = False,
        *,
        degrade_threshold: float = DEFAULT_DEGRADE_THRESHOLD,
        provenance: dict | None = None,
        force: bool = False,
        replace_window: tuple | None = None,
    ) -> int:
        """Save per-stock market data to flat files, merging with existing.

        Returns the number of stocks whose replace_range write was rejected by
        the degradation guard.

        Loads existing flat file, concatenates new rows, drops duplicate
        rows (identical across all columns), and writes back atomically.
        Multi-row-per-day events (e.g. block_trade) are preserved.
        Thread-safe: uses temp file + atomic rename per stock.

        ``replace_range=True`` marks this as a *derived-view* write (used by
        preprocessing reprocessing): existing rows whose ``date`` falls inside
        the new rows' [min, max] range are dropped before merging, so a rerun
        yields exactly the current transform output for that range instead of
        accumulating stale rows (e.g. after a logic fix changes values for the
        same date). Rows outside the range are preserved for partial runs.

        ``replace_window`` overrides the [min, max] used to derive the replaced
        range (default: the new rows' own extent).  Pass the intended coverage
        range so a partial output (e.g. only a few dates regenerated because the
        upstream raw load came back short) is measured against the full range
        and rejected instead of slipping through a narrow default window.

        A destructive replace is REJECTED (the old file is left untouched)
        when it would delete more than
        ``degrade_threshold`` of the dates previously present inside the
        window, or when it drops columns the old file carried — either signals
        an upstream failure that produced a partial output.  Pass ``force=True``
        to bypass the guard for an intentional rewrite.  When ``provenance`` is
        given, a sidecar manifest recording the write (run_id / source_snapshot
        / config_hash / git_commit / coverage_report / decision) is written to
        ``{base}/.manifests/{code}.json`` for both accepted and rejected writes.
        """
        if df.empty:
            return 0
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return 0

        base = self._base_dir()
        rejected = 0
        for code, group in df.groupby("stock_code"):
            out_path = os.path.join(base, f"{code}.parquet")
            # Read-modify-write must be exclusive: atomic rename alone only
            # protects readers from torn files, not concurrent writers from
            # overwriting each other's merged rows (parallel year backfills).
            lock_dir = _acquire_lock(out_path)
            try:
                new_rows = group.sort_values("date")
                decision = "accepted"
                report: dict = {}
                if os.path.isfile(out_path):
                    existing = pd.read_parquet(out_path)
                    existing["date"] = pd.to_datetime(existing["date"])
                    # Backward compat: older files may lack stock_code column
                    if "stock_code" not in existing.columns:
                        existing["stock_code"] = code
                    if replace_range:
                        if replace_window is not None:
                            lo = pd.Timestamp(replace_window[0])
                            hi = pd.Timestamp(replace_window[1])
                        else:
                            lo = new_rows["date"].min()
                            hi = new_rows["date"].max()
                        report = self._replace_range_report(existing, new_rows, lo, hi)
                        dropped_cols = report["schema"]["dropped_cols"]
                        if (
                            not force
                            and report["degradation_ratio"] > degrade_threshold
                        ):
                            decision = "rejected"
                            rejected += 1
                            self._log_rejection(code, report, degrade_threshold, reason="date")
                        elif not force and dropped_cols:
                            decision = "rejected"
                            rejected += 1
                            self._log_rejection(
                                code, report, degrade_threshold,
                                reason=f"columns={dropped_cols}",
                            )
                        if decision == "accepted":
                            existing = existing[
                                (existing["date"] < lo) | (existing["date"] > hi)
                            ]
                    new_rows = pd.concat([existing, new_rows], ignore_index=True)
                # Dedup identical rows (not by date only — block_trade has
                # multiple trades per day that must all be preserved).
                new_rows = new_rows.drop_duplicates(keep="last")
                new_rows = new_rows.sort_values("date")
                if decision == "accepted":
                    fd, tmp_path = tempfile.mkstemp(
                        suffix=".parquet", dir=base, prefix=f".tmp_{code}_",
                    )
                    os.close(fd)
                    try:
                        new_rows.to_parquet(tmp_path, index=False, compression='lz4')
                        os.replace(tmp_path, out_path)
                    except Exception:
                        if os.path.isfile(tmp_path):
                            os.unlink(tmp_path)
                        raise
                if provenance is not None:
                    self._write_manifest(code, provenance, report, decision, new_rows)
            finally:
                _release_lock(lock_dir)
        return rejected

    @staticmethod
    def _replace_range_report(
        existing: pd.DataFrame, new_rows: pd.DataFrame, lo, hi,
    ) -> dict:
        """Coverage + schema diff for the [lo, hi] window being replaced."""
        old_in_window = existing[
            (existing["date"] >= lo) & (existing["date"] <= hi)
        ]
        old_dates = set(pd.to_datetime(old_in_window["date"]).dt.date)
        new_in_window = new_rows[
            (new_rows["date"] >= lo) & (new_rows["date"] <= hi)
        ]
        new_dates = set(pd.to_datetime(new_in_window["date"]).dt.date)
        missing = sorted(old_dates - new_dates)
        degradation = (
            len(missing) / len(old_dates) if old_dates else 0.0
        )
        old_cols = set(existing.columns)
        new_cols = set(new_rows.columns)
        dropped = sorted(old_cols - new_cols - _SCHEMA_RESERVED)
        added = sorted(new_cols - old_cols - _SCHEMA_RESERVED)
        return {
            "window_lo": pd.Timestamp(lo).strftime("%Y-%m-%d"),
            "window_hi": pd.Timestamp(hi).strftime("%Y-%m-%d"),
            "old_dates_in_window": len(old_dates),
            "new_dates_in_window": len(new_dates),
            "missing_dates": len(missing),
            "missing_date_list": [d.isoformat() for d in missing[:20]],
            "degradation_ratio": round(degradation, 4),
            "schema": {
                "dropped_cols": dropped,
                "added_cols": added,
                "old_columns": sorted(old_cols),
            },
        }

    def _log_rejection(
        self, code: str, report: dict, threshold: float, *, reason: str,
    ) -> None:
        logger.error(
            "MarketWideStorage[%s] REJECTED replace_range for %s "
            "(degradation %.1f%% > %.0f%%, %d dates lost, %s): file left "
            "untouched",
            self._data_type, code, report["degradation_ratio"] * 100,
            threshold * 100, report["missing_dates"], reason,
        )

    def _write_manifest(
        self, code: str, provenance: dict, report: dict, decision: str,
        rows: pd.DataFrame,
    ) -> None:
        """Record a replace_range write outcome as a sidecar JSON manifest."""
        base = self._base_dir()
        manifest_dir = os.path.join(base, ".manifests")
        os.makedirs(manifest_dir, exist_ok=True)
        payload = {
            "data_type": self._data_type,
            "stock_code": code,
            "run_id": provenance.get("run_id"),
            "source_snapshot": provenance.get("source_snapshot"),
            "config_hash": provenance.get("config_hash"),
            "git_commit": provenance.get("git_commit"),
            "replace_range": True,
            "decision": decision,
            "coverage": report,
            "rows_written": int(len(rows)),
            "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        path = os.path.join(manifest_dir, f"{code}.json")
        tmp = f"{path}.tmp.{os.getpid()}"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp, path)
        except Exception:
            if os.path.isfile(tmp):
                os.unlink(tmp)
            raise

    def load(
        self, stock_code: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Load market data for a single stock in a date range.

        Prefers consolidated flat file; falls back to year/month partitions.
        """
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        base = self._base_dir()

        if not os.path.exists(base):
            return pd.DataFrame()

        # Prefer consolidated flat file: {type}/{code}.parquet
        flat_path = os.path.join(base, f"{stock_code}.parquet")
        if os.path.isfile(flat_path):
            df = pd.read_parquet(flat_path)
            df["date"] = pd.to_datetime(df["date"])
            mask = (df["date"] >= start) & (df["date"] <= end)
            return df[mask].sort_values("date").reset_index(drop=True)

        # Fallback: partitioned {type}/{year}/{month}/{code}.parquet
        frames = []
        for year in range(start.year, end.year + 1):
            year_dir = os.path.join(base, str(year))
            if not os.path.isdir(year_dir):
                continue
            for month in range(1, 13):
                if year == start.year and month < start.month:
                    continue
                if year == end.year and month > end.month:
                    continue
                file_path = os.path.join(
                    year_dir, f"{month:02d}", f"{stock_code}.parquet",
                )
                if not os.path.exists(file_path):
                    continue
                df = pd.read_parquet(file_path)
                df["date"] = pd.to_datetime(df["date"])
                mask = (df["date"] >= start) & (df["date"] <= end)
                frames.append(df[mask])

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)

    def load_date(self, date_str: str) -> pd.DataFrame | None:
        """Load all stocks for a single date from partitioned storage.

        Returns None if the partition directory doesn't exist, empty
        DataFrame if no data matches the date, or the filtered DataFrame.
        """
        dt = pd.Timestamp(date_str)
        base = self._base_dir()
        part_dir = os.path.join(base, str(dt.year), f"{dt.month:02d}")
        if not os.path.isdir(part_dir):
            return None
        frames = []
        for f in os.listdir(part_dir):
            if not f.endswith(".parquet"):
                continue
            try:
                df = pd.read_parquet(os.path.join(part_dir, f))
                mask = pd.to_datetime(df["date"]).dt.date == dt.date()
                matched = df[mask]
                if not matched.empty:
                    frames.append(matched)
            except Exception:
                logger.debug("Failed to read %s/%s for date %s",
                             self._data_type, f, date_str)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
