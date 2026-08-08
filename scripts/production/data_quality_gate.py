"""Data quality gate —常驻数据质量门禁，整合分散的一次性验证。

Each check scans the on-disk data for a known corruption signature and either
passes or reports the offending files. The gate exits non-zero when any enabled
check fails, so it can gate CI / a post-download hook. Run it after any
download or feature rebuild.

The run/report layer (CLI parse, formal-profile override, report assembly,
JSON write) lives in ``data_quality_gate_run.py``; ``main`` is re-exported here
so ``gate_mod.main()`` / the ``__main__`` entry are unchanged (§二十一).  The
check functions + §P1-7 universe reconciliation live in
``data_quality_gate_checks.py`` and are re-exported here; they read THIS
module's mutable state through the ``_g`` module object at call time, so the
test seam (monkeypatching these module globals) keeps working (§T16).

Checks:
  datasets         : required-dataset pre-gate — dir exists, file /
                     stock / row minimums, date-span coverage, freshness.
                     DEFAULT: 0 file or 0 row = FAIL; only --allow-empty lets
                     an empty/missing dataset pass.
  daily_internal   : daily flat pct_change == close.pct_change()*100 (fill-0 pollution)
  aux_pct_aligned  : board/industry processed pct_change == daily pct_change (stale-0)
  aux_close_aligned: processed OHLC == canonical daily close (调整基准漂移)
  feature_pct      : built feature pct_change == daily (feature-layer pollution)
  sparsity         : per-feature effective non-zero coverage (NaN-excluded)
  ohlc_sanity      : dates unique/sorted/no-weekend, code==filename, prices>0,
                     low<=open/close<=high, volume/amount>=0
  contract_schema  : DAILY_EQUITY contract per file — schema, pk, dates on the
                     OFFICIAL trading calendar, units, finite ratios, OHLC,
                     source/adjustment_mode provenance
  manifest         : per-stock manifest must exist and match the parquet
                     (range/completeness/schema-hash/provenance, §四-4)
  universe         : OPT-IN per-requested-stock reconciliation (§P1-7).  Given a
                     requested universe (--requested-universe / --request-manifest
                     / --universe-codes), every requested code is checked for a
                     present parquet + valid manifest + not-degraded coverage,
                     and the gate fails on any missing/degraded stock beyond the
                     tolerated ratios.  A download run manifest source also
                     enriches the report's universe_reconciliation with the
                     run's own completion status (§八-2).  Never runs unless a
                     universe is supplied.

Any read error or missing column FAILS its check — a problem recorded in the
report must also flip the gate.

Sampling is exchange-stratified with a fixed seed so a --quick run is
not biased toward low-code stocks.

Output: reports/data_quality_gate.json (machine-readable) + console summary.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/data_quality_gate.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/data_quality_gate.py --quick --sample 200
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/data_quality_gate.py --check daily_internal,feature_pct
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/data_quality_gate.py --data-dir <train-root> \
      --require daily,features --max-stale-days 10
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/data_quality_gate.py --allow-empty  # dev bootstrap
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/data_quality_gate.py \
      --request-manifest data/a_shares/download_manifest.json   # §P1-7 reconcile the run's requested universe
  PYTHONPATH=. ./.venv/Scripts/python scripts/production/data_quality_gate.py \
      --requested-universe universe.txt --min-universe-rows 500
"""
import datetime as dt
import glob
import hashlib
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stoke_ml.data.asset_contract import manifest_body_digest
from stoke_ml.data.calendar import (
    TradingCalendar,
    calendar_artifact_hash,
    get_research_calendar,
    load_calendar,
    most_recent_completed_trading_day,
)
from stoke_ml.data.channel_sources import CHANNEL_SOURCE, processed_data_type
from stoke_ml.data.contract import get_contract
from stoke_ml.config import get_project_root
from stoke_ml.utils.error_summary import classify_error

logger = logging.getLogger(__name__)

PROJECT = get_project_root()
A_SHARES = PROJECT / "data" / "a_shares"
DAILY_DIR = A_SHARES / "daily"
FEAT_DIR = PROJECT / "data" / "features"

# Contract / manifest / gate division (§四-4): the contract owns row-level
# schema, the per-stock manifest owns range/completeness/source/version, and the
# gate enforces both plus cross-sectional consistency.  A file whose manifest is
# missing or whose start/end/rows/schema-hash no longer match the parquet fails
# here — "file exists" never stands in for "data complete".

# Official trading calendar for A-shares.  The contract's date check is
# mandatory (not opt-in): the gate always passes the official trading-day set
# so official holidays are caught, not just weekends.  Cached per (start, end)
# span because a full run revisits the same ranges across files.
_TRADING_CACHE: dict[tuple, frozenset] = {}

# §九-3: the formal profile must not bless forward-estimate trading days as
# verified exchange fact.  Only enforced under --profile formal (see main()).
ENFORCE_VERIFIED_UNTIL = False

# §九: the frozen calendar is resolved per DATA ROOT (``A_SHARES.parent``), NOT
# as a module-import singleton — a --data-dir redirect must validate the SAME
# calendar artifact the feature pipeline reads from that root.  Cached per root;
# main() clears the cache when --data-dir changes.
_CALENDAR_CACHE: dict[str, TradingCalendar] = {}


def _get_calendar() -> TradingCalendar:
    """The frozen calendar for the CURRENT data root (``A_SHARES.parent``).

    Resolved through ``get_research_calendar(data_dir=...)`` so the artifact at
    ``<root>/exchange_calendar/a_shares.parquet`` is authoritative when present;
    a bootstrap run with no artifact transparently falls back to the code holiday
    set (formal mode refuses that fallback in main()).

    A present-but-corrupt artifact (wrong schema / empty / dup-date / gapped)
    must NOT crash the gate mid-check with a bare traceback — that would abort
    before the report is written.  We fall back to the code-derived calendar so
    every check still completes; ``_calendar_status()`` records the corruption
    (``present=False`` / ``unusable``) and formal mode fails the run regardless.
    """
    root = str(Path(A_SHARES.parent).resolve())
    cal = _CALENDAR_CACHE.get(root)
    if cal is None:
        try:
            cal = get_research_calendar(data_dir=root)
        except Exception as exc:
            logger.warning(
                "calendar artifact unusable at %s (category=%s): %s",
                root, classify_error(exc).value, exc,
            )
            cal = TradingCalendar("a_shares")
        _CALENDAR_CACHE[root] = cal
    return cal


def _now() -> pd.Timestamp:
    """Wall-clock now, normalized to midnight (the freshness reference date).

    A test seam: monkeypatching this simulates running the gate on a specific
    date (e.g. a 春节/国庆 closure day) so holiday-safe freshness is provable
    without waiting for a real holiday.
    """
    return pd.Timestamp.now().normalize()


def _official_trading_days(dates: pd.Series) -> frozenset:
    """Frozenset of official-calendar trading days covering a frame's span."""
    lo, hi = dates.min().date(), dates.max().date()
    key = (lo, hi)
    cached = _TRADING_CACHE.get(key)
    if cached is None:
        cached = frozenset(_get_calendar().get_trading_days(lo, hi))
        _TRADING_CACHE[key] = cached
    return cached

# Processed dirs whose embedded close/OHLC must equal canonical daily.
# Derived from CHANNEL_SOURCE (§T2): the ``*_processed`` dir NAME is the last
# segment of the channel's processed_dir, so it lives in ONE place.
_AUX_CLOSE_CHANNELS = ("block_trade", "board", "dividend", "sector", "lockup",
                       "shareholder")
AUX_CLOSE_DIRS = [
    processed_data_type(CHANNEL_SOURCE[ch]) for ch in _AUX_CLOSE_CHANNELS
]
AUX_PCT_DIRS = [
    processed_data_type(CHANNEL_SOURCE[ch]) for ch in ("board", "sector")
]

# Sparsity canary: features with non-zero ratio below this across the sampled
# panel are reported as event-sparse (they carry signal only for a small subset).
SPARSE_NONZERO_RATIO = 0.005

# Gate's own version, recorded in every report so a consuming run (train_panel's
# required quality-gate check, §六-2) can verify it reviewed THIS gate.
# 3.0 (§二十-8): the daily dataset fingerprint is now a Merkle root over the
# per-stock manifest content hashes — old gate reports' data_manifest_hash are
# NO LONGER comparable, so a consuming run must re-run the gate before it can
# trust the report-to-training binding.
# 2.0 (§七.2/§七.3/§七.4): content-aware dataset fingerprint + explicit run
# scope/sample fields + always-full manifest/contract scan.  Any semantic change
# to what PASS means MUST bump this — a consuming run refuses a gate whose
# semantics it can't name.
QUALITY_GATE_VERSION = "3.0"

# Frozen formal-research profile (§六-4).  The bootstrap defaults are fine for
# dev, but a 5530-stock research run must clear these floors: readable stocks
# >= 98% of the scanned pool, latest date within ~1-2 trading days of target
# end, coverage span >= 5 years, zero unreadable files, zero contract failures.
FORMAL_PROFILE = {
    "min_span_days": 365 * 5,
    "max_stale_days": 4,
    "max_unreadable_ratio": 0.0,
    "stock_ratio": 0.98,
}

# §P1-7: per-requested-stock reconciliation state.  ``_UNIVERSE_REQUEST`` is set
# by main() when a requested universe is supplied (--requested-universe /
# --request-manifest / --universe-codes); ``check_universe`` no-ops when it is
# None, so default runs are byte-for-byte unaffected.  The ``universe`` check is
# deliberately NOT in ``CHECKS`` — it only joins the run when a universe is given.
_UNIVERSE_REQUEST: dict | None = None


@dataclass
class CheckResult:
    name: str
    passed: bool
    summary: str
    files_scanned: int = 0
    rows_scanned: int = 0
    scanned_files: int = 0  # files actually row-read (<= files_scanned under --quick, §九.2)
    unreadable_files: int = 0  # files that existed but failed to parse (§六-3)
    issues: list = field(default_factory=list)  # list of (file, detail)
    details: dict | None = None  # optional structured report (universe §P1-7)


# Union of columns every _load_daily caller needs; cached once per code so the
# aux-alignment checks (which hit the same daily file once per aux file, ~3万×)
# don't re-read it from disk every time — cold-cache runs took ~35 min.
_DAILY_CACHE_COLS = ("date", "open", "high", "low", "close", "pct_change")
_DAILY_CACHE: dict[str, pd.DataFrame] = {}


def _load_daily(code: str, cols: list[str]) -> pd.DataFrame | None:
    """Canonical daily OHLCV frame for ``code`` (cached; returns a safe copy)."""
    cached = _DAILY_CACHE.get(code)
    if cached is None:
        p = DAILY_DIR / f"{code}.parquet"
        if not p.exists():
            return None
        try:
            d = pd.read_parquet(p, columns=list(_DAILY_CACHE_COLS))
        except (OSError, ValueError) as exc:
            # Distinguish a corrupt parquet from a missing file: missing is a
            # normal no-data state (None, callers fail closed), corruption is
            # a data-integrity signal that must be visible, not swallowed.
            logger.warning(
                "daily %s unreadable (category=%s): %s",
                code, classify_error(exc).value, exc,
            )
            return None
        d["date"] = pd.to_datetime(d["date"])
        cached = d.drop_duplicates("date", keep="last").reset_index(drop=True)
        _DAILY_CACHE[code] = cached
    return cached[list(cols)].copy()


# ── required-dataset pre-gate ───────────────────────────────────────────────
# A gate that PASSes on an empty directory is worse than no gate: a wrong
# path, an aborted download, or a config pointing elsewhere all get a green
# check.  Every required dataset must satisfy: dir exists, file count >= min,
# valid stock count >= min, total rows >= min, date-span coverage, freshness.
# Default 0 file / 0 row = FAIL; only --allow-empty permits empty data.
SAMPLE_SEED = 20240804
REQUIRED_DATASETS = ["daily"]  # CLI: --require daily,features,features_panel
MIN_FILES = 1
MIN_STOCKS = 1
MIN_ROWS = 1
MIN_SPAN_DAYS = 180
MAX_STALE_DAYS = 30
# Readable-valid stock fraction floor (0.0 = disabled); --profile formal raises
# it to 0.98 so a dataset with >2% unreadable/empty files cannot pass (§六-4).
FORMAL_STOCK_RATIO = 0.0
# Max tolerated unreadable-file ratio per required dataset; formal profile = 0.0
# so a single corrupt file fails the required-dataset pre-gate (§六-3).
MAX_UNREADABLE_RATIO = 0.05
ALLOW_EMPTY = False


def _dataset_dir(name: str) -> Path:
    """Resolve a required-dataset directory from the current data root.

    §九.1: names resolve against the REAL data root, never a fixed basename
    whitelist.  The three canonical names map to their canonical dirs; any
    other required dataset (e.g. a custom prebuilt like ``features_panel_v2``)
    resolves to ``<data_root>/<name>`` so the gate validates the ACTUAL dir a
    consuming run reads instead of silently scanning DAILY_DIR.
    """
    if name == "daily":
        return DAILY_DIR
    if name == "features":
        return FEAT_DIR
    if name == "features_panel":
        return A_SHARES.parent / "features_panel"
    return A_SHARES.parent / name


def contract_version() -> str:
    """Content hash of the frozen daily contract's schema-defining fields.

    ``DataContract`` has no version field, so a consuming run (train_panel's
    required quality-gate check, §六-2) derives one: any schema / units /
    basis / calendar change flips the digest and the gate-to-training binding
    fails loudly instead of silently evaluating against a different contract.
    """
    c = get_contract("daily_equity")
    payload = json.dumps({
        "required_columns": sorted(c.required_columns),
        "primary_key": list(c.primary_key),
        "units": dict(sorted(c.units.items())),
        "price_basis": c.price_basis,
        "adjustment_mode": c.adjustment_mode,
        "calendar": c.calendar,
        "timezone": c.timezone,
        "required_metadata": sorted(c.required_metadata),
        "price_columns": sorted(c.price_columns),
        "required_finite_ratio": dict(sorted(c.required_finite_ratio.items())),
        "minimum_valid_rows": c.minimum_valid_rows,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def dataset_fingerprint(root: Path, datasets: list[str]) -> str:
    """Deterministic content-aware hash over the required dataset directories.

    §v18-8: for the ``daily`` dataset the hash is a MERKLE ROOT over the
    per-stock manifest CONTENT hashes (``daily/{code}.manifest.json``, with
    the per-write bookkeeping keys — ``written_at`` / ``updated`` / ``run_id``
    — excluded) — the manifest is the data's identity, bound at write time.  A
    formal re-run reads only the small sidecars, never re-scans the full-market
    Daily bytes (tens of GB); a content-identical re-save (a no-op re-download
    that only bumps ``updated`` / ``run_id``) does NOT flip the digest.  A
    parquet whose manifest was rewritten to reflect new content (the canonical
    write path) flips the digest; a manifest-less parquet hashes a distinct
    marker so it never silently equals a manifested one.

    Non-daily datasets (features / features_panel / custom) keep the legacy
    byte-streaming hash — they have no per-stock manifest contract to bind.
    Any dataset a training run consumes that changed after the gate PASS flips
    the digest and train_panel's report-match check fails (§六-2).
    """
    def _resolve(name: str) -> Path:
        # §九.1: mirror _dataset_dir — a custom (non-canonical) dataset resolves
        # under the root by its real name instead of silently re-hashing daily.
        if name == "daily":
            return Path(root) / "a_shares" / "daily"
        if name == "features":
            return Path(root) / "features"
        if name == "features_panel":
            return Path(root) / "features_panel"
        return Path(root) / name

    def _hash_file(fp: Path, h) -> None:
        # LEGACY non-daily branch (features / features_panel / custom): stream
        # the raw bytes in bounded chunks — a content-aware fingerprint cannot
        # trust stat() metadata, which an in-place same-size rewrite preserves
        # verbatim.  Unreadable files hash a marker so a corrupt /
        # permission-blocked file still changes the digest from "absent".
        try:
            with open(fp, "rb") as fh:
                while True:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        break
                    h.update(chunk)
        except OSError:
            h.update(fp.name.encode("utf-8"))
            h.update(b":unreadable;")

    h = hashlib.sha1()
    for ds in sorted(datasets):
        d = _resolve(ds)
        h.update(ds.encode("utf-8"))
        h.update(b"=")
        h.update(str(d.resolve()).encode("utf-8"))
        h.update(b"\x00")
        if not d.is_dir():
            h.update(b"missing")
            h.update(b";")
            continue
        if ds == "daily":
            # §v18-8: Merkle root over per-stock manifest content digests.
            for fp in sorted(d.glob("*.parquet")):
                code = fp.stem
                mp = d / f"{code}.manifest.json"
                digest = (
                    manifest_body_digest(str(mp))
                    if mp.is_file() else "<no-manifest>")
                h.update(fp.name.encode("utf-8"))
                h.update(b":")
                h.update(digest.encode("utf-8"))
                h.update(b";")
            continue
        for fp in sorted(d.glob("*.parquet")):
            h.update(fp.name.encode("utf-8"))
            h.update(b":")
            _hash_file(fp, h)
            h.update(b";")
    return h.hexdigest()[:16]


def _calendar_status() -> dict:
    """Presence + content hash + path of the frozen calendar artifact for the
    CURRENT data root (``A_SHARES.parent``).

    ``present`` is False when the artifact is absent or unusable.  ``hash`` is
    the content hash of the calendar the gate ACTUALLY resolved — the artifact
    when present, else ``None`` (formal mode refuses that code fallback) — so
    the report records what was really validated (§九), not a version string.
    """
    root = A_SHARES.parent
    p = Path(root) / "exchange_calendar" / "a_shares.parquet"
    try:
        present = load_calendar(root, "a_shares") is not None
    except Exception as exc:
        return {"present": False, "hash": None, "path": str(p),
                "reason": f"unusable: {exc}"}
    if not present:
        return {"present": False, "hash": None, "path": str(p),
                "reason": "missing"}
    return {"present": True,
            "hash": calendar_artifact_hash(root, "a_shares"),
            "path": str(p), "reason": ""}


def _exchange_of(code: str) -> str:
    """A-share exchange bucket from the stock-code first digit."""
    first = code[0] if code else ""
    if first in ("0", "3"):
        return "SZ"
    if first == "6":
        return "SH"
    if first in ("4", "8"):
        return "BJ"
    return "other"


def _sample_files(files: list[str], n: int, seed: int = SAMPLE_SEED) -> list[str]:
    """Fixed-seed exchange-stratified sample.

    Plain ``files[:n]`` on the sorted code list is biased toward low-code
    stocks (00xxxx dominate the head).  Stratify by exchange so a --quick run
    covers SH/SZ/BSE proportionally, deterministically per seed.
    """
    if n <= 0 or len(files) <= n:
        return list(files)
    rng = np.random.default_rng(seed)
    buckets: dict[str, list[str]] = {}
    for f in files:
        buckets.setdefault(_exchange_of(Path(f).stem), []).append(f)
    out = []
    for bucket in sorted(buckets):
        items = buckets[bucket]
        want = max(1, round(len(items) / len(files) * n))
        items = list(items)
        rng.shuffle(items)
        out.extend(items[:want])
    return out[:n]


def _scan_dataset(name: str, d: Path, sample: int) -> tuple[list, int, int, int, int]:
    """Return (issues, n_files, n_scanned, n_rows, unreadable).

    ``n_files`` is the true on-disk parquet count; ``n_scanned`` is how many
    of those were actually row-read (a stratified sample when ``sample > 0``,
    --quick).  The gap between the two is the audit scope a consumer reads
    (§九.2).  ``unreadable`` counts files that existed but failed to parse —
    each is reported by name (file, "unreadable") and the dataset FAILS when
    the unreadable share exceeds ``MAX_UNREADABLE_RATIO`` (§六-3).  Under the
    formal profile ``FORMAL_STOCK_RATIO`` additionally demands readable
    stocks >= that fraction of the scanned pool (§六-4).
    """
    issues: list = []
    if not d.exists():
        issues.append((f"{name}", "missing_dir"))
        return issues, 0, 0, 0, 0
    files = sorted(glob.glob(str(d / "*.parquet")))
    if len(files) < MIN_FILES:
        issues.append((f"{name}", f"files={len(files)} < min={MIN_FILES}"))
    scan = _sample_files(files, sample)
    valid_stocks = total_rows = unreadable = 0
    unreadable_stems: list[str] = []
    dates: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for fp in scan:
        try:
            df = pd.read_parquet(fp, columns=["date"])
        except Exception:
            unreadable += 1
            unreadable_stems.append(Path(fp).stem)
            continue
        if "date" not in df:
            continue
        dts = pd.to_datetime(df["date"], errors="coerce").dropna()
        total_rows += len(dts)
        if len(dts):
            valid_stocks += 1
            dates.append((dts.min(), dts.max()))
    # §六-3: unreadable files are always COUNTED (recorded in the report even
    # when tolerated), but only FAIL the dataset when their share exceeds the
    # max ratio.  Per-file entries join the issues list in that case.
    scanned = len(scan)
    if scanned and unreadable / scanned > MAX_UNREADABLE_RATIO:
        issues.append((f"{name}",
                       f"unreadable={unreadable}/{scanned} "
                       f"> max_ratio={MAX_UNREADABLE_RATIO}"))
        issues.extend((stem, "unreadable") for stem in unreadable_stems)
    if FORMAL_STOCK_RATIO and scanned and \
            valid_stocks < max(1, math.ceil(FORMAL_STOCK_RATIO * scanned)):
        issues.append((f"{name}",
                       f"valid_stocks={valid_stocks}/{scanned} < "
                       f"{FORMAL_STOCK_RATIO * 100:.0f}%"))
    if valid_stocks < MIN_STOCKS:
        issues.append((f"{name}", f"valid_stocks={valid_stocks} < min={MIN_STOCKS}"))
    if total_rows < MIN_ROWS:
        issues.append((f"{name}", f"rows={total_rows} < min={MIN_ROWS}"))
    if dates:
        lo = min(a for a, _ in dates)
        hi = max(b for _, b in dates)
        span = (hi - lo).days
        if span < MIN_SPAN_DAYS:
            issues.append((f"{name}", f"span={span}d < min={MIN_SPAN_DAYS}d"))
        # §九: freshness is positional against the frozen calendar, not natural
        # age.  A dataset current through the most recent COMPLETED trading day
        # is fresh even across 春节/国庆 7-8 day closures (a fully-current dataset
        # never trips a natural-day ceiling).  ``behind`` counts trading days in
        # (latest, most_recent_completed]; > MAX_STALE_DAYS is stale.
        cal = _get_calendar()
        most_recent = most_recent_completed_trading_day(cal, _now().date())
        latest = hi.date()
        behind = 0
        if latest < most_recent:
            behind = len(cal.get_trading_days(
                latest + dt.timedelta(days=1), most_recent))
        if behind > MAX_STALE_DAYS:
            issues.append((f"{name}",
                           f"stale={behind} trading-days > max={MAX_STALE_DAYS}"))
        # §九-3: forward-estimate trading days (2027+ A-share closures) are not
        # verified exchange fact — the formal profile refuses data that uses
        # them rather than validating it against guessed holidays.
        if ENFORCE_VERIFIED_UNTIL and latest > cal.verified_until:
            issues.append((f"{name}",
                           f"extends_past_verified_until={latest} > "
                           f"{cal.verified_until}"))
    elif scan:
        issues.append((f"{name}", "empty_rows"))
    return issues, len(files), len(scan), total_rows, unreadable


# ── check functions (§T16: moved to data_quality_gate_checks.py) ────────
# The check functions + §P1-7 universe reconciliation live in
# data_quality_gate_checks.py; they read THIS module's mutable state via the
# _g module object at call time.  Re-exported here so
# gate_mod.<check_name> / `from ... import <check_name>` keep working (§T16).
from scripts.production.data_quality_gate_checks import (  # noqa: E402,F401  re-exported
    _build_universe_request,
    _read_requested_file,
    check_aux_close_aligned,
    check_aux_pct_aligned,
    check_contract_schema,
    check_daily_internal,
    check_datasets,
    check_feature_pct,
    check_manifest,
    check_ohlc_sanity,
    check_sparsity,
    check_universe,
    reconcile_requested_universe,
)

CHECKS = {
    "datasets": check_datasets,
    "daily_internal": check_daily_internal,
    "aux_pct_aligned": check_aux_pct_aligned,
    "aux_close_aligned": check_aux_close_aligned,
    "feature_pct": check_feature_pct,
    "sparsity": check_sparsity,
    "ohlc_sanity": check_ohlc_sanity,
    "contract_schema": check_contract_schema,
    "manifest": check_manifest,
}

# §P1-7: the universe check is deliberately NOT in ``CHECKS`` (a default run
# must be unchanged) but IS runnable — via ``--check universe`` or when a
# requested universe is supplied.  The run loop resolves through this combined
# registry so a KeyError can never hide an opt-in check.
RUN_CHECKS = dict(CHECKS)
RUN_CHECKS["universe"] = check_universe


def _manifest_contract_full_scan(results, total_daily) -> bool:
    """True only when the manifest + contract_schema full-coverage floor is
    really met (v14 §八-1).

    Both full-scan checks must have actually run, both must have passed, and both
    must have scanned every daily parquet.  An unreadable file inside either
    audit already flips that check's ``passed`` to False (§六-3: read errors
    surface as a ``read_err`` issue), so there is no separate ``unreadable_files``
    clause to enforce — the count is only populated by the dataset check, which
    is not part of this pair.  A run scoped to ``--check manifest``
    (contract_schema never joined ``results``) returns False — closing the old
    ``bool(full_audit) and files_scanned == total`` bypass that ignored
    ``passed`` and the absent partner check.
    """
    audit = {r.name: r for r in results if r.name in ("manifest", "contract_schema")}
    if set(audit) != {"manifest", "contract_schema"}:
        return False
    return all(
        r.passed and r.files_scanned == total_daily
        for r in audit.values()
    )


# §二十一: the run/report layer (CLI parse, formal-profile override, global
# mutation, check dispatch, report assembly, JSON write) moved to
# data_quality_gate_run.py; ``main`` is re-exported here so ``gate_mod.main()``,
# the ``__main__`` entry and the build_features.py subprocess invocation are
# unchanged.  The run module reads/mutates the gate's mutable state through the
# module object at call time, so the test seam (monkeypatching gate module
# globals) keeps working.
from scripts.production.data_quality_gate_run import main  # noqa: E402,F401  re-exported (§二十一 run/report layer)


if __name__ == "__main__":
    sys.exit(main())
