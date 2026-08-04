"""Data storage — daily K-line, single canonical flat layout (review v7 §五).

Canonical store
---------------
One layout only: ``daily/{code}.parquet`` — a complete per-stock file.  The
legacy year/month partitions (``daily/{year}/{month}/{code}.parquet``) were the
old write target, but keeping two layouts invited a split-brain: a downloader
wrote partitions while training discovered flat files, and ``load_daily`` had
to union the two with an mtime heuristic.  Now ``save_daily`` merges into the
flat file and ``load_daily`` reads it directly; stale partition directories on
disk are ignored (never read, never written).  A stock's full history is one
small parquet, so per-stock atomic read-modify-write is cheap.

``save_daily`` is NON-destructive: it reads the existing flat file, merges the
new rows by ``date``, dedups, sorts, then atomically ``os.replace``s a temp
file.  A per-file lock serializes concurrent downloader processes so a merge
cannot lose the other process's rows.
"""
import os
import time

import pandas as pd

_LOCK_TIMEOUT = 30.0  # seconds to wait for a concurrent writer
_LOCK_STALE = 600.0   # a lock older than this is a crashed writer's leftover


def _acquire_lock(lock_path: str) -> None:
    """Cross-process exclusive lock via O_CREAT|O_EXCL lockfile."""
    deadline = time.time() + _LOCK_TIMEOUT
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            return
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > _LOCK_STALE:
                    os.remove(lock_path)
                    continue
            except OSError:
                pass
            if time.time() >= deadline:
                raise TimeoutError(f"could not acquire lock {lock_path}")
            time.sleep(0.05)


def _release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except OSError:
        pass


class DataStorage:
    """Save and load market data as single-layout flat Parquet files."""

    def __init__(self, data_dir: str):
        self._root = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _daily_dir(self, market: str) -> str:
        return os.path.join(self._root, market, "daily")

    def save_daily(self, df: pd.DataFrame, market: str = "a_shares"):
        """Non-destructively merge ``df`` into ``daily/{code}.parquet``.

        Existing rows (by ``date``) are kept on last-write-wins, new rows are
        appended, the file is sorted by date and atomically replaced under a
        per-file lock.  Only the flat canonical layout is touched.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        drop_cols = [c for c in ("year", "month") if c in df.columns]
        base = self._daily_dir(market)
        os.makedirs(base, exist_ok=True)

        for code, group in df.groupby("stock_code"):
            save_df = group.drop(columns=drop_cols)
            out_path = os.path.join(base, f"{code}.parquet")
            lock_path = out_path + ".lock"
            _acquire_lock(lock_path)
            try:
                existing = None
                if os.path.isfile(out_path):
                    existing = pd.read_parquet(out_path)
                    existing["date"] = pd.to_datetime(existing["date"])
                if existing is not None and len(existing):
                    combined = pd.concat(
                        [existing, save_df], ignore_index=True
                    )
                    combined = (
                        combined.drop_duplicates(subset="date", keep="last")
                        .sort_values("date")
                        .reset_index(drop=True)
                    )
                else:
                    combined = save_df
                tmp_path = f"{out_path}.tmp.{os.getpid()}"
                combined.to_parquet(tmp_path, index=False, compression="lz4")
                os.replace(tmp_path, out_path)
            finally:
                _release_lock(lock_path)

    def load_daily(
        self, stock_code: str, start_date: str, end_date: str,
        market: str = "a_shares"
    ) -> pd.DataFrame:
        """Read ``daily/{code}.parquet`` (the single canonical store)."""
        flat_path = os.path.join(self._daily_dir(market), f"{stock_code}.parquet")
        if not os.path.isfile(flat_path):
            return pd.DataFrame()
        result = pd.read_parquet(flat_path)
        result["date"] = pd.to_datetime(result["date"])
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        mask = (result["date"] >= start) & (result["date"] <= end)
        return result[mask].sort_values("date").reset_index(drop=True)

    def list_stocks(self, market: str = "a_shares") -> list[str]:
        """Discover stocks from the flat store (review v7 §五: discovery must use
        the storage API, not raw ``os.listdir`` on a mix of files and dirs)."""
        base = self._daily_dir(market)
        if not os.path.isdir(base):
            return []
        return sorted(
            f[: -len(".parquet")]
            for f in os.listdir(base) if f.endswith(".parquet")
        )
