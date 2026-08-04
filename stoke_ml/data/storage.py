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
import datetime as dt
import hashlib
import json
import os
import time

import pandas as pd

_LOCK_TIMEOUT = 30.0  # seconds to wait for a concurrent writer
_LOCK_STALE = 600.0   # a lock older than this is a crashed writer's leftover


def _schema_hash(df: pd.DataFrame) -> str:
    """Stable hash of the column set — the feature-consuming schema contract.

    Column renames/additions/removals change the hash so a stale parquet (built
    with a different feature schema) is caught by `validate_manifest` instead
    of being silently trusted (review v8 §二-1).  Column-only on purpose: the
    dtype can drift across a parquet round-trip without the content being
    wrong, while the provenance (source / adjust / date range / rows) lives in
    the manifest's other fields.
    """
    sig = "|".join(sorted(map(str, df.columns)))
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


def _write_manifest(base: str, code: str, df: pd.DataFrame,
                    source: str, adjust: str) -> None:
    """Atomically write the per-stock contract manifest (review v8 §二-1).

    The manifest pins stock / start / end / rows / source / adjust / schema
    hash / write time so "file exists" never silently implies "data complete".
    """
    manifest = {
        "stock": code,
        "start": (df["date"].min().strftime("%Y-%m-%d") if len(df) else None),
        "end": (df["date"].max().strftime("%Y-%m-%d") if len(df) else None),
        "rows": int(len(df)),
        "source": source,
        "adjust": adjust,
        "schema_hash": _schema_hash(df),
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    path = os.path.join(base, f"{code}.manifest.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


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
        # Provenance stamped by the downloader onto the frame (failover.py sets
        # df.attrs["source"] / ["adjustment_mode"]); falls back to "unknown" so
        # the manifest is always written and always self-describing.
        source = df.attrs.get("source", "unknown")
        adjust = df.attrs.get("adjustment_mode", "unknown")

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
                # Contract manifest written atomically alongside the parquet,
                # still under the lock so readers see a consistent pair.
                _write_manifest(base, code, combined, source, adjust)
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

    def _manifest_path(self, market: str, stock_code: str) -> str:
        return os.path.join(self._daily_dir(market), f"{stock_code}.manifest.json")

    def manifest(self, stock_code: str, market: str = "a_shares") -> dict | None:
        """Read the per-stock contract manifest, or None if it is absent.

        The manifest is the record that "file exists" ≠ "data complete": it
        pins stock / start / end / rows / source / adjust / schema hash / write
        time (review v8 §二-1), so a stale or re-configured parquet is
        detectable instead of silently trusted.
        """
        path = self._manifest_path(market, stock_code)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def validate_manifest(
        self, stock_code: str, market: str = "a_shares"
    ) -> dict:
        """Cross-check the on-disk parquet against its contract manifest.

        Returns a report that is ``ok`` only when the manifest exists AND the
        parquet's actual row count / date range / schema hash match what the
        manifest claims.  A schema change, a partial write or a re-adjustment
        surfaces here instead of silently producing wrong training features.
        """
        flat_path = os.path.join(self._daily_dir(market), f"{stock_code}.parquet")
        if not os.path.isfile(flat_path):
            return {"exists": False, "ok": False, "reason": "parquet missing"}
        manifest = self.manifest(stock_code, market)
        if manifest is None:
            return {"exists": True, "ok": False,
                    "reason": "manifest missing — file exists ≠ data complete"}
        try:
            df = pd.read_parquet(flat_path)
        except Exception as exc:  # pragma: no cover - corruption shape varies
            return {"exists": True, "ok": False, "reason": f"unreadable: {exc}"}
        df["date"] = pd.to_datetime(df["date"])
        actual = {
            "rows": int(len(df)),
            "start": (df["date"].min().strftime("%Y-%m-%d") if len(df) else None),
            "end": (df["date"].max().strftime("%Y-%m-%d") if len(df) else None),
            "schema_hash": _schema_hash(df),
        }
        mismatches = [
            f"{key}: manifest={manifest.get(key)!r} actual={value!r}"
            for key, value in actual.items()
            if manifest.get(key) != value
        ]
        return {
            "exists": True,
            "ok": not mismatches,
            "mismatches": mismatches,
            "manifest": manifest,
            "actual": actual,
        }
