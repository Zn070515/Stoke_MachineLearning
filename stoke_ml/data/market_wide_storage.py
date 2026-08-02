"""Storage for market-wide data types (dragon-tiger, margin, northbound).

Partitions: data/a_shares/{data_type}/{year}/{month}/{stock_code}.parquet
"""
import logging
import os
import tempfile
import time

import pandas as pd

logger = logging.getLogger(__name__)

_LOCK_TIMEOUT = 600.0
_LOCK_STALE = 900.0


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

    def save(self, df: pd.DataFrame, replace_range: bool = False) -> None:
        """Save per-stock market data to flat files, merging with existing.

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
        """
        if df.empty:
            return
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return

        base = self._base_dir()
        for code, group in df.groupby("stock_code"):
            out_path = os.path.join(base, f"{code}.parquet")
            # Read-modify-write must be exclusive: atomic rename alone only
            # protects readers from torn files, not concurrent writers from
            # overwriting each other's merged rows (parallel year backfills).
            lock_dir = _acquire_lock(out_path)
            try:
                new_rows = group.sort_values("date")
                if os.path.isfile(out_path):
                    existing = pd.read_parquet(out_path)
                    existing["date"] = pd.to_datetime(existing["date"])
                    # Backward compat: older files may lack stock_code column
                    if "stock_code" not in existing.columns:
                        existing["stock_code"] = code
                    if replace_range:
                        lo = new_rows["date"].min()
                        hi = new_rows["date"].max()
                        existing = existing[
                            (existing["date"] < lo) | (existing["date"] > hi)
                        ]
                    new_rows = pd.concat([existing, new_rows], ignore_index=True)
                # Dedup identical rows (not by date only — block_trade has
                # multiple trades per day that must all be preserved).
                new_rows = new_rows.drop_duplicates(keep="last")
                new_rows = new_rows.sort_values("date")
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
            finally:
                _release_lock(lock_dir)

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
