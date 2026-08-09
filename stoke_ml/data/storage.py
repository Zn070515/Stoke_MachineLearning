"""Data storage — daily K-line, single canonical flat layout.

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

Manifest contract
------------------------------------------------------
Each flat parquet carries a sidecar ``daily/{code}.manifest.json`` pinning
stock / start / end / rows / source / adjust / schema hash / provenance /
write time.  The schema hash now covers columns, dtypes, declared units, price
basis, calendar epoch, dataset-convention version AND a content checksum, so a
dtype, unit (元→千元), lot (手→股), adjustment (raw→qfq) or value
drift is caught by ``validate_manifest`` instead of silently trusted.
Provenance comes from ``df.attrs``, which pandas round-trips through the
parquet footer, with stable ``unknown`` defaults for files written before the
attrs feature, so the hash is recomputable from the file on disk.

Formal reads (training / feature build / preprocessing) pass
``require_valid_manifest=True``: a file that exists without a valid
manifest raises instead of being read, so the manifest is a hard constraint,
not a report.  ``load_daily`` keeps a lenient default for exploratory scripts.

The parquet replace and the manifest write are two ``os.replace`` calls, not
one atomic transaction.  A crash in between leaves a stale sidecar
(``torn state`` — new parquet + old manifest or vice versa), but the manifest's
rows/start/end/schema-hash cross-check detects that pair on the next validated
read, and the lock + heartbeat keep a live writer's lock from being stolen.
A generation-directory + current-pointer design that switches data + manifest
as one unit would close the torn-state window entirely (v14 §十三-2); that is a
P2 engineering refactor, deliberately not part of this change.
"""
import datetime as dt
import hashlib
import json
import os
import socket
import threading
import time
import uuid

import pandas as pd

from stoke_ml.data.calendar import VERIFIED_UNTIL, TradingCalendar
from stoke_ml.data.codes import is_a_share_equity_code, normalize_stock_code_series
from stoke_ml.data.contract import RESEARCH_QFQ_DAILY, validate_contract
from stoke_ml.data.date_normalize import as_date_us

_LOCK_TIMEOUT = 30.0  # seconds to wait for a concurrent writer
_LOCK_STALE = 600.0   # a dead lock older than this is a crashed writer's leftover
_LOCK_HEARTBEAT = 5.0  # a live writer refreshes its lock mtime this often

# Canonical K-line value conventions.  A future convention change (元→千元,
# 手→股, raw→qfq) must bump ``_DATA_CONVENTION_VERSION`` so old data is
# flagged by the schema-hash contract instead of silently re-consumed.
_DEFAULT_UNITS = {
    "open": "cny", "high": "cny", "low": "cny", "close": "cny",
    "volume": "shares", "amount": "cny",
}
_DATA_CONVENTION_VERSION = "kline-v1"
_CALENDAR_VERSION = f"a_shares:{VERIFIED_UNTIL['a_shares'].isoformat()}"

# Official trading-day set per (data_dir, lo, hi) span.  A full download
# revisits the same date spans across thousands of save_daily calls, so the
# calendar set is built once per span instead of per write (§八-1).
_TRADING_CACHE: dict[tuple, frozenset] = {}


def _units_tag() -> str:
    return ";".join(f"{k}={v}" for k, v in sorted(_DEFAULT_UNITS.items()))


def _content_checksum(df: pd.DataFrame, columns: list[str]) -> str:
    """Deterministic hash of the actual values, stable across parquet round-trip.

    Numeric/datetime columns hash raw bytes where cheap; object/string columns
    hash the joined UTF-8 text (``.values.tobytes()`` on object dtype would
    hash Python pointers, which differ between processes).
    """
    h = hashlib.sha256()
    for c in columns:
        s = df[c]
        if s.dtype == object:
            h.update(b"\x00".join(str(v).encode("utf-8") for v in s.tolist()))
        elif s.dtype.kind in "biufc":
            h.update(s.to_numpy().tobytes())
        else:
            h.update(str(s.dtype).encode("utf-8") + b"\x00" +
                     b"\x00".join(str(v).encode("utf-8") for v in s.tolist()))
    return h.hexdigest()


def _provenance_from_attrs(df: pd.DataFrame) -> dict:
    """Normalized provenance contract, from ``df.attrs`` with stable defaults.

    Files written before provenance was stamped (legacy parquets) restore no
    attrs, so defaults keep the hash recomputable and the manifest honest
    (``unknown`` source/adjust is the only truthful value for those rows).
    """
    return {
        "source": df.attrs.get("source") or "unknown",
        "adjust": df.attrs.get("adjustment_mode") or "unknown",
        "units": df.attrs.get("units") or _units_tag(),
        "price_basis": df.attrs.get("adjustment_mode") or "unknown",
        "calendar_version": df.attrs.get("calendar_version") or _CALENDAR_VERSION,
        "dataset_version": df.attrs.get("data_convention_version") or _DATA_CONVENTION_VERSION,
    }


def _schema_hash(df: pd.DataFrame) -> str:
    """Stable hash of the data contract.

    Columns + dtypes + declared units + price basis + calendar epoch + dataset
    convention version + content checksum.  A parquet that drifts in any of
    these — renamed/dropped column, dtype change, 元→千元, 手→股, raw→qfq, or
    values edited in place — no longer hashes to the manifest's recorded value
    and ``validate_manifest`` flags it.
    """
    columns = sorted(map(str, df.columns))
    dtypes = [f"{c}:{df[c].dtype}" for c in columns]
    prov = _provenance_from_attrs(df)
    sig = "|".join([
        "cols=" + ",".join(columns),
        "dtypes=" + ",".join(dtypes),
        "units=" + prov["units"],
        "price_basis=" + prov["price_basis"],
        "calendar_version=" + prov["calendar_version"],
        "dataset_version=" + prov["dataset_version"],
        "content=" + _content_checksum(df, columns),
    ])
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:16]


def _build_source_segments(
    old_manifest: dict | None, combined: pd.DataFrame,
    new_dates, source: str, adjust: str,
    batch_segments: list[dict] | None = None,
) -> list[dict]:
    """Per-date source provenance audit.

    A merged file can be early Baostock + late Efinance + some Tushare; the
    flat ``source``/``adjust`` fields only name the latest batch.  This derives
    an attribution for every date — the new batch owns every date it touches
    (last-write-wins, matching the merge) and old segments keep the rest — then
    collapses adjacent dates into disjoint ``{source, adjust, start, end,
    rows}`` segments so the manifest describes which provider fed which part
    of the history.  Legacy manifests degrade to a single un-typed segment,
    which is all that can be known about pre-upgrade rows.

    ``batch_segments`` carries row-level source attribution from the fetch
    layer: a single ``fetch_daily`` batch may itself mix a Baostock
    backfill with the primary source, and without these ranges every new date
    would be attributed to the flat ``source``.  When a new date falls inside a
    batch segment's ``[start, end]``, that segment's ``(source, adjust)`` wins
    over the flat default.
    """
    dates = sorted(set(pd.to_datetime(combined["date"]).dt.normalize()))
    new_set = set(pd.to_datetime(new_dates).dt.normalize())
    if not dates:
        return []
    if old_manifest:
        old_segs = old_manifest.get("source_segments")
        if old_segs:
            base = [dict(s) for s in old_segs]
        else:
            base = [{
                "source": old_manifest.get("source", "unknown"),
                "adjust": old_manifest.get("adjust", "unknown"),
                "start": old_manifest.get("start"),
                "end": old_manifest.get("end"),
                "rows": int(old_manifest.get("rows") or 0),
            }]
    else:
        base = []
    owned: dict[pd.Timestamp, tuple[str, str]] = {}
    for seg in base:
        lo = (pd.Timestamp(seg["start"]).normalize()
              if seg.get("start") else pd.Timestamp.min)
        hi = (pd.Timestamp(seg["end"]).normalize()
              if seg.get("end") else pd.Timestamp.max)
        key = (seg.get("source", "unknown"), seg.get("adjust", "unknown"))
        for d in dates:
            if d in new_set or d in owned:
                continue
            if lo <= d <= hi:
                owned[d] = key
    batch_ranges = []
    for seg in batch_segments or []:
        lo = (pd.Timestamp(seg["start"]).normalize()
              if seg.get("start") else pd.Timestamp.min)
        hi = (pd.Timestamp(seg["end"]).normalize()
              if seg.get("end") else pd.Timestamp.max)
        batch_ranges.append(
            (lo, hi, (seg.get("source", "unknown"), seg.get("adjust", "unknown")))
        )
    for d in dates:
        if d in new_set and d not in owned:
            for lo, hi, key in batch_ranges:
                if lo <= d <= hi:
                    owned[d] = key
                    break
    for d in dates:
        owned.setdefault(d, (source, adjust))
    segments: list[dict] = []
    for d in dates:
        key = owned[d]
        if (segments and segments[-1]["source"] == key[0]
                and segments[-1]["adjust"] == key[1]):
            segments[-1]["end"] = d.date().isoformat()
            segments[-1]["rows"] += 1
        else:
            segments.append({"source": key[0], "adjust": key[1],
                             "start": d.date().isoformat(),
                             "end": d.date().isoformat(), "rows": 1})
    return segments


def _write_manifest(base: str, code: str, df: pd.DataFrame,
                    segments: list[dict], run_id: str) -> None:
    """Atomically write the per-stock contract manifest."""
    prov = _provenance_from_attrs(df)
    manifest = {
        "stock": code,
        "start": (df["date"].min().strftime("%Y-%m-%d") if len(df) else None),
        "end": (df["date"].max().strftime("%Y-%m-%d") if len(df) else None),
        "rows": int(len(df)),
        "source": prov["source"],
        "adjust": prov["adjust"],
        "units": prov["units"],
        "price_basis": prov["price_basis"],
        "calendar_version": prov["calendar_version"],
        "dataset_version": prov["dataset_version"],
        "schema_hash": _schema_hash(df),
        "source_segments": segments,
        "run_id": run_id,
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    path = os.path.join(base, f"{code}.manifest.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


class _LockHandle:
    """Cross-process lock that refreshes its own heartbeat."""

    def __init__(self, path: str):
        self.path = path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def _heartbeat(self) -> None:
        while not self._stop.is_set():
            time.sleep(_LOCK_HEARTBEAT)
            try:
                os.utime(self.path, None)
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours to signal
    except OSError:
        return False


def _lock_is_stale(lock_path: str) -> bool:
    """Whether a lockfile can be reclaimed.

    Never stale on another host; never stale while its writer PID is alive;
    stale only when the PID is dead/unknown AND the lock has not been touched
    for ``_LOCK_STALE`` (a live writer refreshes mtime via heartbeat).  A
    legacy pid-only lock falls back to the time-only heuristic.
    """
    info = None
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        pass
    if not isinstance(info, dict):
        info = None  # legacy pid-only lock → time-only heuristic below
    if info and info.get("hostname") != socket.gethostname():
        return False
    pid = info.get("pid") if info else None
    if pid is not None and _pid_alive(int(pid)):
        return False
    return time.time() - os.path.getmtime(lock_path) > _LOCK_STALE


def _acquire_lock(lock_path: str) -> _LockHandle:
    """Cross-process exclusive lock via O_CREAT|O_EXCL lockfile."""
    deadline = time.time() + _LOCK_TIMEOUT
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, json.dumps({
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "run_id": uuid.uuid4().hex,
                "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }).encode("utf-8"))
            os.close(fd)
            return _LockHandle(lock_path)
        except FileExistsError:
            if _lock_is_stale(lock_path):
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue
            if time.time() >= deadline:
                raise TimeoutError(f"could not acquire lock {lock_path}")
            time.sleep(0.05)


def _release_lock(lock) -> None:
    path = lock.path if isinstance(lock, _LockHandle) else lock
    if isinstance(lock, _LockHandle):
        lock.stop()
    try:
        os.remove(path)
    except OSError:
        pass


class DataStorage:
    """Save and load market data as single-layout flat Parquet files."""

    def __init__(self, data_dir: str):
        self._root = data_dir
        os.makedirs(data_dir, exist_ok=True)

    def _daily_dir(self, market: str) -> str:
        return os.path.join(self._root, market, "daily")

    def save_daily(self, df: pd.DataFrame, market: str = "a_shares",
                   run_id: str | None = None, *, formal: bool = True):
        """Non-destructively merge ``df`` into ``daily/{code}.parquet``.

        Existing rows (by ``date``) are kept on last-write-wins, new rows are
        appended, the file is sorted by date and atomically replaced under a
        per-file lock.  Only the flat canonical layout is touched.

        §八 write gate: before anything lands on disk the frame must survive
        the canonical ``RESEARCH_QFQ_DAILY`` contract — every stock code run
        through the single sanitizer, then schema / PK / OHLC / units / dates
        (official calendar) / provenance / adjustment-mode validated — and the
        price basis checked against the existing manifest.  The parquet is
        written to staging, read back and re-validated before the atomic swap,
        so a frame that violates the contract (or a basis that would splice
        onto a legacy ``unknown`` history) is refused instead of persisted: a
        manifest that matches a corrupt file is still corrupt.

        ``formal=True`` (v13 §五-1/§十) — the default for the canonical write
        path — enforces the strict contract: an ``unknown`` adjustment basis is
        refused, and only A-share common-equity codes (``is_a_share_equity_code``)
        may become canonical daily files.  ``formal=False`` keeps the legacy
        ``unknown`` exemption and is used only by the explicit migration seam
        :meth:`save_daily_repair`.
        """
        if "stock_code" not in df.columns:
            raise ValueError("save_daily: frame has no stock_code column")
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        if df.empty:
            return
        # §八-3: Storage is the last canonical boundary — every code goes
        # through the single sanitizer, so 600001.0 / SH600001 / -1 can never
        # become a mangled file name or a split-key pair of files.
        norm = normalize_stock_code_series(df["stock_code"])
        bad = norm.isna()
        if bad.any():
            offenders = sorted({str(v) for v in df.loc[bad, "stock_code"].tolist()})[:5]
            raise ValueError(
                f"save_daily: {int(bad.sum())} unusable stock_code value(s) "
                f"{offenders} — refusing to write a corrupted canonical key"
            )
        df = df.assign(stock_code=norm)
        # §十 (v13): the canonical daily store only holds A-share common equity.
        # A format-legal but non-equity code (100xxx index, 200xxx/900xxx B-share,
        # 500xxx fund) must not become a canonical daily file.
        non_equity = norm.map(lambda c: not is_a_share_equity_code(c))
        if non_equity.any():
            offenders = sorted({str(c) for c in norm[non_equity].tolist()})[:5]
            raise ValueError(
                f"save_daily: {int(non_equity.sum())} non-A-share-equity "
                f"stock_code value(s) {offenders} — the canonical daily store "
                f"only accepts A-share common stocks"
            )
        drop_cols = [c for c in ("year", "month") if c in df.columns]
        base = self._daily_dir(market)
        os.makedirs(base, exist_ok=True)
        source = df.attrs.get("source", "unknown")
        adjust = df.attrs.get("adjustment_mode", "unknown")
        # Row-level source ranges stamped by the fetch layer (failover.py
        # §十-4) survive here as the batch attribution; read before the
        # groupby slicing so group.attrs cannot drop them.
        batch_segments = df.attrs.get("source_segments")
        run_id = run_id or uuid.uuid4().hex

        for code, group in df.groupby("stock_code"):
            save_df = group.drop(columns=drop_cols)
            out_path = os.path.join(base, f"{code}.parquet")
            lock_path = out_path + ".lock"
            handle = _acquire_lock(lock_path)
            try:
                old_manifest = None
                manifest_path = os.path.join(base, f"{code}.manifest.json")
                if os.path.isfile(manifest_path):
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        old_manifest = json.load(f)
                # §八-2 price-basis gate BEFORE the merge: a concrete basis
                # must never splice onto (a) an explicitly-declared DIFFERENT
                # concrete basis, or (b) a legacy file whose basis is
                # "unknown" — that history cannot be proven consistent with the
                # new rows, so "unknown" is only legal for migration /
                # exploration / audit, never as a seam to a formal QFQ store
                # (§四.1 / P0-2 / v12 §八-2).
                old_adjust = (old_manifest or {}).get("adjust", "unknown")
                has_legacy = old_manifest is not None or os.path.isfile(out_path)
                old_concrete = old_adjust not in ("unknown", "n/a", "")
                new_concrete = adjust not in ("unknown", "n/a", "")
                if has_legacy and not old_concrete and new_concrete:
                    raise ValueError(
                        f"{code}: refusing to append {adjust!r} rows to a "
                        f"legacy file that declares adjust={old_adjust!r} — "
                        f"the existing basis is unknown, so a mixed-basis "
                        f"history would be undetectable; run save_daily_repair() "
                        f"(an explicit migration) instead"
                    )
                if old_concrete and new_concrete and old_adjust != adjust:
                    raise ValueError(
                        f"{code}: refusing to merge price-basis segments "
                        f"(old={old_adjust!r}, new={adjust!r}) — run an "
                        f"explicit re-adjustment migration instead of "
                        f"silently splicing mixed-basis history"
                    )
                existing = None
                if os.path.isfile(out_path):
                    existing = pd.read_parquet(out_path)
                    existing["date"] = pd.to_datetime(existing["date"])
                if existing is not None and len(existing):
                    combined = pd.concat(
                        [existing, save_df], ignore_index=True
                    )
                    combined = combined.drop_duplicates(
                        subset="date", keep="last")
                else:
                    combined = save_df
                combined = combined.sort_values("date").reset_index(drop=True)
                # §十三-3 (v14): a same-date merge overwrite may have CORRECTED
                # a close (t day), which leaves the adjacent row's pct_change
                # still based on the pre-correction price.  Recompute the whole
                # (per-stock) series in the qfq frame — cheap at one stock per
                # file — so every stored return is consistent with the stored
                # closes BEFORE the contract gate and manifest hash are
                # computed.  Row 0 becomes NaN (the honest listing-day value),
                # which the contract permits.
                combined["pct_change"] = (
                    pd.to_numeric(combined["close"], errors="coerce")
                    .pct_change() * 100.0
                )
                # §八-1: the date check must cover the FULL combined span.  A
                # set built only from the new batch's dates would flag every
                # existing row that predates the batch as non_trading_day on
                # the next incremental merge.
                trading_days = self._official_trading_days(combined["date"])
                # Stamp provenance into the frame so both the parquet footer
                # (round-tripped by pandas) and the manifest record it.
                combined.attrs["source"] = source
                combined.attrs["adjustment_mode"] = adjust
                combined.attrs["units"] = _units_tag()
                combined.attrs["calendar_version"] = _CALENDAR_VERSION
                combined.attrs["data_convention_version"] = _DATA_CONVENTION_VERSION
                # §八-1: the FULL contract gate runs before the write — a file
                # whose manifest "matches itself" but whose economic semantics
                # are corrupt must never land in the canonical store.
                violations = validate_contract(
                    combined, RESEARCH_QFQ_DAILY, code=code,
                    trading_days=trading_days, manifest=old_manifest,
                    formal=formal,
                )
                if violations:
                    raise ValueError(
                        f"{code}: refusing to persist a frame violating the "
                        f"{RESEARCH_QFQ_DAILY.dataset_name} contract: "
                        f"{'; '.join(violations[:8])}"
                    )
                tmp_path = f"{out_path}.tmp.{os.getpid()}"
                combined.to_parquet(tmp_path, index=False, compression="lz4")
                # Round-trip read: parquet writes can coerce dtypes / drop
                # attrs, so the exact bytes that will be atomically swapped are
                # re-validated against the contract before the replace.
                back = pd.read_parquet(tmp_path)
                rt_violations = validate_contract(
                    back, RESEARCH_QFQ_DAILY, code=code,
                    trading_days=trading_days, manifest=old_manifest,
                    formal=formal,
                )
                if rt_violations:
                    raise ValueError(
                        f"{code}: parquet round-trip failed the "
                        f"{RESEARCH_QFQ_DAILY.dataset_name} contract: "
                        f"{'; '.join(rt_violations[:8])}"
                    )
                os.replace(tmp_path, out_path)
                segments = _build_source_segments(
                    old_manifest, combined, save_df["date"], source, adjust,
                    batch_segments=batch_segments,
                )
                # Contract manifest written atomically alongside the parquet,
                # still under the lock so readers see a consistent pair.  The
                # schema_hash must describe the ACTUAL bytes on disk — the
                # round-tripped ``back`` frame, whose ``date`` dtype can differ
                # from the in-memory ``combined`` (e.g. datetime64[ns] in memory
                # round-trips to datetime64[ms] on disk).  Hashing ``combined``
                # would record a schema_hash that validate_manifest can never
                # match.
                _write_manifest(base, code, back, segments, run_id)
            finally:
                _release_lock(handle)

    def _official_trading_days(self, dates: pd.Series) -> frozenset:
        """Frozenset of official-calendar trading days covering a batch's span.

        Cached per ``(data_dir, lo, hi)`` because a full download revisits the
        same date spans across thousands of writes.
        """
        dts = pd.to_datetime(dates)
        lo, hi = dts.min().date(), dts.max().date()
        key = (self._root, lo, hi)
        cached = _TRADING_CACHE.get(key)
        if cached is None:
            cal = TradingCalendar("a_shares", calendar_dir=self._root)
            cached = frozenset(cal.get_trading_days(lo, hi))
            _TRADING_CACHE[key] = cached
        return cached

    def save_daily_repair(
        self, df: pd.DataFrame, market: str = "a_shares",
        run_id: str | None = None,
    ):
        """Save a repaired canonical-daily frame, carrying each stock's existing
        manifest ``source``/``adjust`` forward (§八-1).

        In-place maintenance scripts (clip negatives, re-derive ``pct_change``,
        merge a provider column) change values but not the underlying provider,
        so a raw :meth:`save_daily` would degrade the manifest provenance to the
        ``attrs`` default.  This preserves the current attribution for every
        stock in ``df`` before routing through :meth:`save_daily`.

        Repair is the explicit legacy-migration seam, so it routes with
        ``formal=False`` (v13 §五-1): a file whose honest provenance is
        ``unknown`` may still be repaired / re-persisted, while a fresh
        canonical write (the default ``formal=True`` path) refuses it.
        """
        for code, group in df.groupby("stock_code"):
            m = self.manifest(code, market)
            group.attrs["source"] = (m or {}).get("source", "unknown")
            group.attrs["adjustment_mode"] = (m or {}).get("adjust", "unknown")
            self.save_daily(group, market=market, run_id=run_id, formal=False)

    def load_daily(
        self, stock_code: str, start_date: str, end_date: str,
        market: str = "a_shares", require_valid_manifest: bool = False
    ) -> pd.DataFrame:
        """Read ``daily/{code}.parquet`` (the single canonical store).

        ``require_valid_manifest=True`` (formal reads) raises if the
        file exists but its contract manifest is missing, stale or mismatched,
        instead of silently reading a possibly-corrupt parquet.

        Formal reads are also the second enforcement point of the strict
        contract (v13 §五-1/§十): the code must be an A-share common-equity
        code and the frame must survive ``validate_contract(..., formal=True)``
        — an ``unknown`` adjustment basis on disk is refused, not read.
        """
        flat_path = os.path.join(self._daily_dir(market), f"{stock_code}.parquet")
        if not os.path.isfile(flat_path):
            return pd.DataFrame()
        if require_valid_manifest:
            if not is_a_share_equity_code(stock_code):
                raise ValueError(
                    f"refusing to read {stock_code} with require_valid_manifest=True: "
                    f"not an A-share common-equity code"
                )
            report = self.validate_manifest(stock_code, market)
            if not report["ok"]:
                raise ValueError(
                    f"refusing to read {stock_code} with require_valid_manifest=True: "
                    f"{report.get('reason') or report.get('mismatches')}"
                )
        # §v19: on-disk daily parquets mix datetime64[ms] and datetime64[us]
        # date columns; pandas 3.0 raises MergeError on mismatched merge keys.
        # Canonical-us coercion at the read layer (in-memory only, files kept).
        result = as_date_us(pd.read_parquet(flat_path))
        if require_valid_manifest:
            # §十三-1 (v14): pass the ALREADY-VALIDATED manifest and the
            # official trading-day set into the contract, instead of depending
            # on the parquet engine round-tripping ``df.attrs``.  Different
            # parquet engines may persist attrs differently, so provenance is
            # read from the manifest (source/adjust) and the date membership
            # check runs against the official calendar — the same inputs the
            # write path uses (:meth:`save_daily`).
            violations = validate_contract(
                result, RESEARCH_QFQ_DAILY, code=stock_code, formal=True,
                manifest=report["manifest"],
                trading_days=self._official_trading_days(result["date"]),
            )
            if violations:
                raise ValueError(
                    f"refusing to read {stock_code} with require_valid_manifest=True: "
                    f"{'; '.join(violations[:8])}"
                )
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        mask = (result["date"] >= start) & (result["date"] <= end)
        return result[mask].sort_values("date").reset_index(drop=True)

    def list_stocks(self, market: str = "a_shares") -> list[str]:
        """Discover stocks from the flat store (discovery must use the storage
        API, not raw ``os.listdir`` on a mix of files and dirs)."""
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
        pins stock / start / end / rows / source / adjust / units / price basis
        / calendar epoch / dataset version / schema hash / source segments /
        write time, so a stale or re-configured parquet is
        detectable instead of silently trusted.
        """
        path = self._manifest_path(market, stock_code)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def rebuild_manifest(self, stock_code: str, market: str = "a_shares") -> None:
        """Write a contract manifest for an existing parquet (one-time migration).

        Legacy parquets predate the manifest / provenance features, so their
        source is unknowable; the manifest honestly records ``unknown``
        source/adjust with the canonical units/version.  Non-destructive: the
        parquet is only read, never rewritten.
        """
        flat_path = os.path.join(self._daily_dir(market), f"{stock_code}.parquet")
        if not os.path.isfile(flat_path):
            raise FileNotFoundError(flat_path)
        df = pd.read_parquet(flat_path)
        df["date"] = pd.to_datetime(df["date"])
        segments = [{
            "source": "unknown", "adjust": "unknown",
            "start": (df["date"].min().strftime("%Y-%m-%d") if len(df) else None),
            "end": (df["date"].max().strftime("%Y-%m-%d") if len(df) else None),
            "rows": int(len(df)),
        }] if len(df) else []
        _write_manifest(self._daily_dir(market), stock_code, df, segments,
                        run_id=uuid.uuid4().hex)

    def validate_manifest(
        self, stock_code: str, market: str = "a_shares"
    ) -> dict:
        """Cross-check the on-disk parquet against its contract manifest.

        Returns a report that is ``ok`` only when the manifest exists AND the
        parquet's actual row count / date range / schema hash / declared
        provenance match what the manifest claims.  A schema change, a partial
        write, a unit/basis/convention drift or a re-adjustment surfaces here
        instead of silently producing wrong training features.
        """
        flat_path = os.path.join(self._daily_dir(market), f"{stock_code}.parquet")
        if not os.path.isfile(flat_path):
            return {"exists": False, "ok": False, "reason": "parquet missing"}
        manifest = self.manifest(stock_code, market)
        if manifest is None:
            return {"exists": True, "ok": False,
                    "reason": "manifest missing — file exists ≠ data complete"}
        # §二十一 (v14): only genuinely "unreadable file" conditions map to an
        # unreadable report.  ``KeyError``/``OSError``/``ValueError`` cover
        # pyarrow's common corruption surface (ArrowIOError ⊂ OSError,
        # ArrowInvalid ⊂ ValueError); anything else is a programming error and
        # must propagate loudly rather than masquerade as data corruption.
        try:
            df = pd.read_parquet(flat_path)
        except (KeyError, OSError, ValueError) as exc:  # pragma: no cover - corruption shape varies
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
        prov = _provenance_from_attrs(df)
        for key in ("source", "adjust", "units", "price_basis",
                    "calendar_version", "dataset_version"):
            if manifest.get(key) != prov[key]:
                mismatches.append(
                    f"{key}: manifest={manifest.get(key)!r} actual={prov[key]!r}"
                )
        return {
            "exists": True,
            "ok": not mismatches,
            "mismatches": mismatches,
            "manifest": manifest,
            "actual": actual,
        }
