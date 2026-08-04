"""Data storage — daily K-line partitioned by year/month (v6 §七).

Canonical store
---------------
Incremental writes go to ``daily/{year}/{month}/{code}.parquet``.  ``save_daily``
is NON-destructive: it merges new rows with the existing month partition by
``date``, dedups, sorts, then atomically ``os.replace``s a temp file.  A per-
partition lock serializes concurrent downloader processes so a merge cannot
lose the other process's rows.

The legacy flat file ``daily/{code}.parquet`` remains as a fast full-history
base.  ``load_daily`` uses it directly when it is at least as fresh as every
partition (mtime scan, no parquet reads); if a partition is newer — i.e. an
increment landed after the flat file was last written — it unions the flat
base with the partition rows and lets the partition (the fresher source) win
per-date.  This guarantees a stale flat file can never shadow newer data.
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
    """Save and load market data as partitioned Parquet files."""

    def __init__(self, data_dir: str):
        self._root = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def save_daily(self, df: pd.DataFrame, market: str = "a_shares"):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        for (year, month, code), group in df.groupby(["year", "month", "stock_code"]):
            out_dir = os.path.join(
                self._root, market, "daily", str(year), f"{month:02d}"
            )
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{code}.parquet")
            save_df = group.drop(columns=["year", "month"])

            # Read-merge-write under a partition lock so concurrent downloader
            # processes cannot overwrite each other's increment (v6 §七).
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
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        base = os.path.join(self._root, market, "daily")
        if not os.path.exists(base):
            return pd.DataFrame()

        flat_path = os.path.join(base, f"{stock_code}.parquet")
        flat_exists = os.path.isfile(flat_path)
        flat_mtime = os.path.getmtime(flat_path) if flat_exists else -1.0

        # stat-scan every partition (cheap, no parquet reads) to decide whether
        # a fresh increment landed after the flat file was last written.
        partition_paths = []
        latest_part_mtime = -1.0
        for entry in sorted(os.listdir(base)):
            ydir = os.path.join(base, entry)
            if not os.path.isdir(ydir) or not entry.isdigit():
                continue
            for mdir in os.listdir(ydir):
                p = os.path.join(ydir, mdir, f"{stock_code}.parquet")
                if os.path.isfile(p):
                    partition_paths.append(p)
                    latest_part_mtime = max(latest_part_mtime, os.path.getmtime(p))

        # Fast path: flat is at least as fresh as every partition, so it is the
        # canonical full history — no need to read any partition parquet.
        need_union = (not flat_exists) or latest_part_mtime > flat_mtime

        if not flat_exists and not partition_paths:
            return pd.DataFrame()

        chunks = []
        if flat_exists:
            chunks.append(pd.read_parquet(flat_path))
        if need_union:
            for p in partition_paths:
                chunks.append(pd.read_parquet(p))

        result = pd.concat(chunks, ignore_index=True)
        result["date"] = pd.to_datetime(result["date"])
        # Partitions appended last + keep="last" ⇒ the fresher partition value
        # wins on dates present in both flat and partitions.
        result = (
            result.drop_duplicates(subset="date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        mask = (result["date"] >= start) & (result["date"] <= end)
        return result[mask].sort_values("date").reset_index(drop=True)
