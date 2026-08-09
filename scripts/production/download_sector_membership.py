"""Build genuine-PIT sector membership from CNINFO industry-change history.

The 证监会 (CSRC) industry classification (门类, single-letter A–S) is stable
across the 2001 / 2012 / 中国上市公司协会 renames of the standard, so a stock's
gate letter can be merged across all three CNINFO ``分类标准`` labels (§v19
P0#1).  CNINFO records CHANGE events only: a stock whose CSRC gate never changed
yields ONE event (the most recent standard rename).

Honest-PIT design rule: a stock's gate is asserted ONLY from its FIRST CSRC
event's ``变更日期`` forward — earlier dates stay unclassified (excluded from
``industry_ranking``).  This is genuinely PIT: it never present-backfills today's
classification onto historical rows; it only asserts what CNINFO proves.

Output: ``a_shares/sector_membership.parquet``
    ``[date, stock_code, sector_code, sector_name]`` — per-date long membership
    over each stock's OWN trading days (from ``DataStorage.load_daily``),
    plus a per-year coverage audit in the asset manifest (``coverage_by_year``)
    and a fail-closed run manifest.

Resumability: per-stock parsed intervals are cached under
``a_shares/sector_membership_pit/_stocks/{code}.json``; a re-run skips already-
fetched stocks.  The cache payload is versioned (``cache_version`` + a
``parser_hash`` digest of the parse logic), so a parser or gate-map change is
detected and forces a fail-closed refetch instead of trusting stale intervals
(§v19 §十九).

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_sector_membership.py
"""
import datetime as dt
import functools
import hashlib
import inspect
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.asset_contract import (
    AtomicCommit,
    DataAssetContract,
    contract_for_channel,
    write_asset_manifest,
)
from stoke_ml.data.csrc_gate import (
    CSRC_GATE_CODES,
    CSRC_STANDARD_LABELS,
    csrc_gate_code,
)
from stoke_ml.data.download_manifest import write_run_manifest_or_exit
from stoke_ml.data.storage import DataStorage

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

#: CNINFO columns the CSRC gate parse needs.  A schema drift here is a FAILURE
#: (raise), never a silent empty-gate that would look like a legit no-gate stock.
_CNINFO_REQUIRED_COLS = frozenset({"分类标准", "行业门类", "变更日期"})

#: The CNINFO schema akshare returns after its column rename (no 最新记录标识).
#: A stock with ZERO records gets normalized to an empty frame carrying THESE
#: columns, so it parses + caches as a legitimate no-gate stock (excluded), not
#: a failure that is re-fetched on every run (§v19 P0#1 review #1).
_CNINFO_STANDARD_COLUMNS = [
    "新证券简称", "行业中类", "行业大类", "行业次类", "行业门类",
    "机构名称", "行业编码", "分类标准", "分类标准编码", "证券代码", "变更日期",
]


def _empty_cninfo_frame() -> pd.DataFrame:
    """A zero-row DataFrame carrying the standard CNINFO columns."""
    return pd.DataFrame(columns=_CNINFO_STANDARD_COLUMNS)

_MAX_WORKERS = 8
_FETCH_ATTEMPTS = 3
_FETCH_BACKOFF_BASE = 2.0
_FETCH_INTERVAL = 0.5  # seconds between CNINFO calls (polite rate limit)

SECTOR_MEMBERSHIP_ASSET: DataAssetContract = contract_for_channel(
    "sector_membership",
    data_type="sector_membership",
    partition="single_file",
    extent_column="date",
    effective_date_policy="event_date",
)


def parse_cninfo_events(stock_code: str, events: pd.DataFrame) -> pd.DataFrame:
    """CNINFO change events → per-date long membership ``[date, stock_code,
    sector_code, sector_name]`` (证监会 门类 level, honest-PIT: gate asserted
    only from its first CSRC event's 变更日期 forward).

    Interval boundaries are taken from EVERY CSRC-standard event's 变更日期, so
    a change to an UNRECOGNIZED 门类 name ends the previous interval exactly at
    its 变更日期 − 1 — the previous gate is never asserted PAST the event that
    disproves it (reverse-PIT, the mirror of present-backfill).  The gate-letter
    mapping only decorates each interval; an unrecognized gate yields
    ``sector_code=None`` and that interval is EXCLUDED from the returned rows.
    """
    empty = pd.DataFrame(columns=["date", "stock_code", "sector_code", "sector_name"])
    if events is None or events.empty:
        return empty
    sub = events[events["分类标准"].isin(CSRC_STANDARD_LABELS)].copy()
    if sub.empty:
        return empty
    sub["变更日期"] = pd.to_datetime(sub["变更日期"], errors="coerce")
    sub = sub.dropna(subset=["变更日期"])
    if sub.empty:
        return empty
    # the most recent gate per change date (a rename can emit several rows same-day)
    sub = sub.sort_values("变更日期")
    latest = sub.groupby("变更日期").last().reset_index()
    latest["sector_code"] = latest["行业门类"].map(csrc_gate_code)
    latest["sector_name"] = latest["行业门类"]
    latest = latest.sort_values("变更日期")
    # interval boundaries from EVERY CSRC-standard event (see docstring)
    rows: list[dict] = []
    for i, (_, row) in enumerate(latest.iterrows()):
        start = row["变更日期"]
        end = latest["变更日期"].iloc[i + 1] - pd.Timedelta(days=1) \
            if i + 1 < len(latest) else pd.Timestamp("2099-12-31")
        rows.append({"date": start, "stock_code": stock_code,
                     "sector_code": row["sector_code"],
                     "sector_name": row["sector_name"], "_end": end})
    out = pd.DataFrame(rows)
    # drop intervals whose gate letter is unrecognized (excluded, never asserted)
    return out.dropna(subset=["sector_code"])


def _expand(intervals: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Interval rows → one long row per trading day ``days`` inside the interval."""
    out = []
    for _, iv in intervals.iterrows():
        mask = (days >= iv["date"]) & (days <= iv["_end"])
        for d in days[mask]:
            out.append({"date": d, "stock_code": iv["stock_code"],
                        "sector_code": iv["sector_code"],
                        "sector_name": iv["sector_name"]})
    return pd.DataFrame(out)


def _expand_stock(intervals: pd.DataFrame, storage: DataStorage,
                  code: str) -> pd.DataFrame:
    """Expand one stock's CSRC intervals over its OWN trading days.

    Returns an empty frame for a legit no-gate stock (no intervals to expand —
    the daily store is not touched).  Raises when the canonical daily formal
    read fails (corrupt/missing manifest) or returns no rows — a stock with
    CSRC intervals is expected to have daily history, so an empty or unreadable
    daily is a defect that must land in ``failed``, never a silent skip (§v19
    §十五).
    """
    if intervals is None or intervals.empty:
        return pd.DataFrame(columns=["date", "stock_code", "sector_code",
                                     "sector_name"])
    d = storage.load_daily(
        code, "1970-01-01", "2099-12-31", require_valid_manifest=True)
    if d is None or d.empty:
        raise ValueError(
            f"sector_membership[{code}]: no daily history — expected rows, "
            f"got empty (defect)")
    days = pd.DatetimeIndex(pd.to_datetime(d["date"]).drop_duplicates()
                            .sort_values())
    return _expand(intervals, days)


def _expansion_bucket(intervals: pd.DataFrame | None) -> str:
    """Which bucket the expansion step lands a code in.

    ``"fetch_failed"`` — ``None`` means the CNINFO fetch never succeeded (the
    code is already in ``fetch_failed``), NOT a legit no-gate.  ``"no_gate"`` —
    an empty frame is a legitimate "no CSRC records" result (→ ``complete``).
    ``"expand"`` — non-empty intervals need the canonical daily formal read +
    expansion.  Distinguishing ``None`` from empty is what keeps a failed stock
    out of ``complete`` (§v19 §十五).
    """
    if intervals is None:
        return "fetch_failed"
    if intervals.empty:
        return "no_gate"
    return "expand"


# ── per-stock interval cache (resumable crawl) ──────────────────────────────

#: Cache payload schema version (§v19 §十九).  Bump when the cache format or
#: the parser semantics change; a mismatched version forces a fail-closed
#: refetch instead of trusting stale intervals.
_CACHE_VERSION = "v2"


@functools.lru_cache(maxsize=1)
def _parser_hash() -> str:
    """Deterministic digest of the parse logic (gate map + labels + source).

    The cache must be invalidated whenever the same raw CNINFO events would
    parse into DIFFERENT intervals, so the digest covers both the CSRC gate
    mapping and ``parse_cninfo_events`` itself (§v19 §十九).  Memoized — it is
    read on every cache check in the crawl (hot path), and the inputs are the
    module-level gate map + static function source, so the result is constant
    for the life of the process.
    """
    return hashlib.sha256(
        (repr(sorted(CSRC_GATE_CODES.items())) + "|"
         + repr(sorted(CSRC_STANDARD_LABELS)) + "|"
         + inspect.getsource(parse_cninfo_events)).encode("utf-8")
    ).hexdigest()[:12]


def _write_intervals_cache(path: str, intervals: pd.DataFrame,
                           start_date: str, end_date: str) -> None:
    """Atomically write the per-stock interval cache under the current schema.

    The payload carries the cache/parser identity and the fetch range so a
    stale or differently-shaped cache is detected on load and refetched (§v19
    §十九).
    """
    payload = [{
        "date": pd.Timestamp(iv["date"]).strftime("%Y-%m-%d"),
        "end": pd.Timestamp(iv["_end"]).strftime("%Y-%m-%d"),
        "stock_code": str(iv["stock_code"]),
        "sector_code": str(iv["sector_code"]),
        "sector_name": str(iv["sector_name"]),
    } for _, iv in intervals.iterrows()]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Atomic write (tmp + os.replace): a crash mid-write must never leave a
    # truncated cache file — a present-but-unreadable cache would make the stock
    # permanently failed (§v19 P0#1 review #3).
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({
            "cache_version": _CACHE_VERSION,
            "parser_hash": _parser_hash(),
            "source": "cninfo",
            "start_date": start_date,
            "end_date": end_date,
            "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "intervals": payload,
        }, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _load_intervals_cache(path: str) -> tuple[pd.DataFrame, dict]:
    """Load a cached interval frame plus its meta (top-level JSON minus
    ``intervals``) so the caller can verify cache/parser/range identity."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    meta = {k: v for k, v in raw.items() if k != "intervals"}
    rows = raw.get("intervals", [])
    if not rows:
        return (pd.DataFrame(columns=["date", "stock_code", "sector_code",
                                      "sector_name", "_end"]), meta)
    iv = pd.DataFrame(rows)
    iv["date"] = pd.to_datetime(iv["date"])
    iv["_end"] = pd.to_datetime(iv["end"])
    return (iv.drop(columns=["end"]), meta)


# ── CNINFO fetch (rate-limited, retried) ─────────────────────────────────────

_FETCH_LOCK = threading.Lock()
_LAST_FETCH_AT = 0.0


def _rate_limited_fetch(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """One CNINFO call, gated by a global min-interval so 8 workers stay polite.

    akshare's ``stock_industry_change_cninfo`` builds ``pd.DataFrame([])`` (0×0)
    when a stock has ZERO records and then indexes ``temp_df["变更日期"]`` on a
    column-less frame → deterministic ``KeyError``.  That is a LEGITIMATE "stock
    has no CSRC gate" result, so it is normalized HERE to an empty frame carrying
    the standard CNINFO columns — it parses + caches as empty (excluded), and
    never burns the retry/backoff loop on a non-transient error (§v19 P0#1
    review #1).
    """
    global _LAST_FETCH_AT
    with _FETCH_LOCK:
        wait = _LAST_FETCH_AT + _FETCH_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        _LAST_FETCH_AT = time.time()
    import akshare as ak
    try:
        return ak.stock_industry_change_cninfo(
            symbol=stock_code, start_date=start_date, end_date=end_date)
    except KeyError as exc:
        # §v19 §十八: akshare's known empty-result signature is a 0×0 frame then
        # temp_df["变更日期"] → KeyError('变更日期').  ONLY this exact key means
        # "stock has no CSRC records".  A schema drift that fails a DIFFERENT
        # column index must NOT be swallowed as a legit no-gate stock — re-raise
        # so _ensure_parseable() / the retry loop handle it.
        if not (exc.args and exc.args[0] == "变更日期"):
            raise
        logger.info("sector_membership[%s]: CNINFO returned no records — "
                    "legit-empty (no CSRC gate), not a failure", stock_code)
        return _empty_cninfo_frame()


def _fetch_cninfo_events(stock_code: str, start_date: str, end_date: str,
                         attempts: int = _FETCH_ATTEMPTS) -> pd.DataFrame:
    last_exc: Exception | None = None
    for i in range(attempts):
        try:
            return _rate_limited_fetch(stock_code, start_date, end_date)
        except Exception as exc:
            last_exc = exc
            if i < attempts - 1:
                time.sleep(_FETCH_BACKOFF_BASE * (2 ** i))
    assert last_exc is not None
    raise last_exc


def _ensure_parseable(events: pd.DataFrame) -> None:
    """Fail loudly on a CNINFO schema drift instead of a silent empty gate."""
    if events is None or events.empty:
        return
    missing = _CNINFO_REQUIRED_COLS - set(events.columns)
    if missing:
        raise ValueError(
            f"CNINFO events missing columns {sorted(missing)} — schema drift")


def _fetch_stock(stock_code: str, cache_dir: str, start_date: str,
                 end_date: str) -> pd.DataFrame:
    """Fetch + parse one stock's CSRC intervals, caching them; raises on failure.

    A cache hit (from a prior run) skips the network call entirely — but only
    when the cached payload's version/parser-hash/source/range all match the
    current run.  A mismatched or legacy cache is NOT trusted: it is removed
    and refetched once (fail closed — one-time full crawl is the honest
    behavior) (§v19 §十九).  An unreadable/corrupt cache is treated as a
    mismatch (removed + refetched) so a bad cache file never permanently fails
    a stock.  An empty return is a LEGITIMATE result (the stock has no CSRC
    gate) — it is still cached so a re-run skips it.
    """
    cache_path = os.path.join(cache_dir, f"{stock_code}.json")
    if os.path.isfile(cache_path):
        try:
            intervals, meta = _load_intervals_cache(cache_path)
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            # A corrupt/unreadable cache must NOT permanently fail a stock —
            # treat it as a mismatch, remove it, and refetch once (§v19 §十九).
            logger.warning("sector_membership[%s]: unreadable cache (%s) — "
                           "removing and refetching", stock_code, str(exc)[:120])
            os.remove(cache_path)
        else:
            if (meta.get("cache_version") == _CACHE_VERSION
                    and meta.get("parser_hash") == _parser_hash()
                    and meta.get("source") == "cninfo"
                    and meta.get("start_date") == start_date
                    and meta.get("end_date") == end_date):
                return intervals
            logger.info("sector_membership[%s]: cache mismatch — refetching "
                        "(version=%r, parser_ok=%s, source=%r, range=%s..%s)",
                        stock_code, meta.get("cache_version"),
                        meta.get("parser_hash") == _parser_hash(),
                        meta.get("source"), meta.get("start_date"),
                        meta.get("end_date"))
            os.remove(cache_path)
    events = _fetch_cninfo_events(stock_code, start_date, end_date)
    _ensure_parseable(events)
    intervals = parse_cninfo_events(stock_code, events)
    _write_intervals_cache(cache_path, intervals, start_date, end_date)
    return intervals


# ── per-year coverage audit ──────────────────────────────────────────────────

def _active_stocks_by_year(daily_dir: str, codes: list[str]) -> dict[int, int]:
    """``{calendar year: count of stocks whose daily span covers that year}``.

    A stock is active for every year in ``[start_year, end_year]`` taken from
    its ``daily/{code}.manifest.json`` ``start``/``end`` (ISO).  A missing or
    unreadable manifest is ignored — that stock is active NOWHERE
    (conservative), so the coverage denominator never over-counts (§v19 §十六).
    """
    counts: dict[int, int] = {}
    for code in codes:
        try:
            with open(os.path.join(daily_dir, f"{code}.manifest.json"),
                      "r", encoding="utf-8") as f:
                m = json.load(f)
            start_s = m.get("start")
            end_s = m.get("end")
            if not start_s or not end_s:
                continue
            start = pd.Timestamp(start_s).year
            end = pd.Timestamp(end_s).year
        except (OSError, ValueError, TypeError, AttributeError,
                json.JSONDecodeError):
            continue
        for y in range(start, end + 1):
            counts[y] = counts.get(y, 0) + 1
    return counts


def _coverage_by_year(membership: pd.DataFrame,
                      active_by_year: dict[int, int]) -> dict:
    """``{year: fraction of stocks ACTIVE that year with an asserted gate}``.

    The honest-PIT audit the review demanded: pre-gate years (and no-gate
    stocks) simply have no membership rows, so the per-year fraction against
    the stocks actually trading that year reveals exactly how far back CNINFO
    can assert a classification (§v19 §十六).  A year with no membership rows
    reports ``0.0``; an empty membership reports ``0.0`` for every active year.
    """
    if membership is None or membership.empty:
        return {str(int(y)): 0.0 for y in active_by_year}
    g = membership.copy()
    g["year"] = pd.to_datetime(g["date"]).dt.year
    cov = g.groupby("year")["stock_code"].nunique()
    out = {}
    for y, active in active_by_year.items():
        v = cov.get(y, 0) / max(int(active), 1)
        out[str(int(y))] = float(round(v, 4))
    return out


def main() -> None:
    cfg = load_config()
    data_dir = cfg.project.data_dir
    base = os.path.join(data_dir, "a_shares")
    storage = DataStorage(data_dir)
    codes = storage.list_stocks("a_shares")
    if not codes:
        logger.error("No stocks under %s — run download_data.py first",
                     os.path.join(base, "daily"))
        sys.exit(1)

    cache_dir = os.path.join(base, "sector_membership_pit", "_stocks")
    os.makedirs(cache_dir, exist_ok=True)
    start_date = "19900101"
    end_date = dt.date.today().strftime("%Y%m%d")

    # 1. Fetch + parse CNINFO change events (8-worker pool, resumable cache).
    #    ``complete`` is NOT granted here — it requires the full pipeline below
    #    (§v19 §十五).  Fetch failures collect into ``fetch_failed``.
    logger.info("Fetching CNINFO industry-classification change events for "
                "%d stocks (%s → %s)...", len(codes), start_date, end_date)
    intervals_by_code: dict[str, pd.DataFrame] = {}
    complete: set[str] = set()
    fetch_failed: list[str] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futures = {
            ex.submit(_fetch_stock, code, cache_dir, start_date, end_date): code
            for code in codes
        }
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                intervals_by_code[code] = fut.result()
            except Exception as exc:
                fetch_failed.append(code)
                logger.warning("sector_membership[%s]: FAILED %s",
                               code, str(exc)[:120])
    logger.info("Fetched %d/%d stocks in %.1fs (%d failed)",
                len(intervals_by_code), len(codes), time.time() - t0,
                len(fetch_failed))

    # 2. Expand intervals to per-trading-day rows over each stock's OWN daily
    #    dates.  ``complete`` is granted ONLY here — after fetch + parse + the
    #    canonical daily formal read + expansion (or a legitimate no-gate).
    #    A daily failure (corrupt/missing manifest, empty history) is a FAILURE
    #    that lands in ``failed``, never a silent skip (§v19 §十五).
    logger.info("Expanding intervals to per-day membership over each stock's "
                "own trading days...")
    frames: list[pd.DataFrame] = []
    expansion_failed: list[str] = []
    for code in codes:
        bucket = _expansion_bucket(intervals_by_code.get(code))
        if bucket == "fetch_failed":
            # CNINFO fetch never succeeded — already in fetch_failed; NEVER
            # upgrade a failed stock to complete (§v19 §十五).
            continue
        if bucket == "no_gate":
            # legit no-gate: no CSRC records → no intervals to expand
            complete.add(code)
            continue
        intervals = intervals_by_code[code]
        try:
            exp = _expand_stock(intervals, storage, code)
        except (ValueError, OSError) as exc:
            expansion_failed.append(code)
            logger.warning("sector_membership[%s]: FAILED daily read — this "
                           "stock's expansion FAILED (not skipped): %s",
                           code, str(exc)[:120])
            continue
        if exp.empty:
            continue
        frames.append(exp)
        complete.add(code)
    if not frames:
        logger.error("No sector membership could be built — no stock had CSRC "
                     "gate data (or no daily history). Refusing to write an "
                     "empty sector_membership.parquet.")
        sys.exit(1)
    membership = pd.concat(frames, ignore_index=True)
    membership["date"] = pd.to_datetime(membership["date"])
    membership = membership.sort_values(["date", "stock_code"]).reset_index(drop=True)
    membership = membership[["date", "stock_code", "sector_code", "sector_name"]]

    # 3. Write parquet + asset manifest + per-year coverage audit.  The
    #    coverage denominator is the stocks ACTIVE each year (from each daily
    #    manifest's start/end span), not the whole universe (§v19 §十六).
    out_path = os.path.join(base, "sector_membership.parquet")
    daily_dir = os.path.join(base, "daily")
    active_by_year = _active_stocks_by_year(daily_dir, codes)
    coverage = _coverage_by_year(membership, active_by_year)
    membership.attrs["source"] = (
        "CNINFO industry-classification change history "
        "(AKShare stock_industry_change_cninfo)")
    with AtomicCommit(out_path) as ac:
        membership.to_parquet(ac.tmp_path, index=False, compression="lz4")
    write_asset_manifest(out_path, SECTOR_MEMBERSHIP_ASSET, membership,
                         coverage_by_year=coverage)
    logger.info("Saved %d membership rows (%d stocks) to %s",
                len(membership), membership["stock_code"].nunique(), out_path)

    # 4. Fail-closed run manifest (§五-5 / §v19 §十五): ``complete`` requires the
    #    full pipeline; ``failed`` = fetch failures + expansion failures
    #    (dedup, keep order).  A partial run can never pass complete.
    failed = list(dict.fromkeys(fetch_failed + expansion_failed))
    write_run_manifest_or_exit(
        data_dir, "a_shares/sector_membership",
        requested=codes, failed=failed, complete=complete,
        success_count=len(complete),
    )
    logger.info("Run manifest: %d requested, %d complete, %d failed",
                len(codes), len(complete), len(failed))
    logger.info("Per-year gate coverage (fraction of stocks ACTIVE that year "
                "with an asserted 门类 gate):")
    if coverage:
        for y in sorted(coverage):
            logger.info("  %s: %.1f%%", y, coverage[y] * 100.0)
    else:
        logger.info("  (none)")


if __name__ == "__main__":
    main()
