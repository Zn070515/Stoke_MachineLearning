"""Data quality gate —常驻数据质量门禁，整合分散的一次性验证。

Each check scans the on-disk data for a known corruption signature and either
passes or reports the offending files. The gate exits non-zero when any enabled
check fails, so it can gate CI / a post-download hook. Run it after any
download or feature rebuild.

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
import argparse
import glob
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.codes import normalize_stock_code
from stoke_ml.data.contract import get_contract, validate_contract
from stoke_ml.data.download_manifest import load_manifest
from stoke_ml.config import get_project_root
from stoke_ml.data.storage import DataStorage
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
_CALENDAR = TradingCalendar("a_shares")
_TRADING_CACHE: dict[tuple, frozenset] = {}

# §九-3: the formal profile must not bless forward-estimate trading days as
# verified exchange fact.  Only enforced under --profile formal (see main()).
VERIFIED_UNTIL = _CALENDAR.verified_until
ENFORCE_VERIFIED_UNTIL = False


def _official_trading_days(dates: pd.Series) -> frozenset:
    """Frozenset of official-calendar trading days covering a frame's span."""
    lo, hi = dates.min().date(), dates.max().date()
    key = (lo, hi)
    cached = _TRADING_CACHE.get(key)
    if cached is None:
        cached = frozenset(_CALENDAR.get_trading_days(lo, hi))
        _TRADING_CACHE[key] = cached
    return cached

# Processed dirs whose embedded close/OHLC must equal canonical daily.
AUX_CLOSE_DIRS = [
    "block_trade_processed",
    "board_processed",
    "dividend_processed",
    "industry_ranking_processed",
    "lockup_processed",
    "shareholder_processed",
]
AUX_PCT_DIRS = ["board_processed", "industry_ranking_processed"]

# Sparsity canary: features with non-zero ratio below this across the sampled
# panel are reported as event-sparse (they carry signal only for a small subset).
SPARSE_NONZERO_RATIO = 0.005

# Gate's own version, recorded in every report so a consuming run (train_panel's
# required quality-gate check, §六-2) can verify it reviewed THIS gate.
# 2.0 (§七.2/§七.3/§七.4): content-aware dataset fingerprint + explicit run
# scope/sample fields + always-full manifest/contract scan.  Any semantic change
# to what PASS means MUST bump this — a consuming run refuses a gate whose
# semantics it can't name.
QUALITY_GATE_VERSION = "2.0"

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
        except Exception as exc:
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

    §七.3: hashes the parquet file BYTES (streamed, never trusting size/mtime),
    so a same-size replacement that preserves mtime_ns still flips the digest —
    name+size+mtime hashing let an in-place overwrite of identical size/mtime
    slip past.  Any dataset a training run consumes that changed after the gate
    PASS — a rebuild, a partial overwrite, an incremental download, a
    re-adjustment — flips the digest and train_panel's report-match check fails
    (§六-2).  ``datasets`` resolve against ``root`` explicitly (NOT the
    module-level dir globals), so a consuming script that imports this function
    hashes the root it actually reads, not the gate's last --data-dir.
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
        # Stream the raw bytes in bounded chunks: a content-aware fingerprint
        # cannot trust stat() metadata, which an in-place same-size rewrite
        # preserves verbatim.  Unreadable files hash a marker so a corrupt /
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
        for fp in sorted(d.glob("*.parquet")):
            h.update(fp.name.encode("utf-8"))
            h.update(b":")
            _hash_file(fp, h)
            h.update(b";")
    return h.hexdigest()[:16]


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
        stale = (pd.Timestamp.now().normalize() - pd.Timestamp(hi)).days
        if stale > MAX_STALE_DAYS:
            issues.append((f"{name}", f"stale={stale}d > max={MAX_STALE_DAYS}d"))
        # §九-3: forward-estimate trading days (2027+ A-share closures) are not
        # verified exchange fact — the formal profile refuses data that uses
        # them rather than validating it against guessed holidays.
        if ENFORCE_VERIFIED_UNTIL and hi.date() > VERIFIED_UNTIL:
            issues.append((f"{name}",
                           f"extends_past_verified_until={hi.date()} > "
                           f"{VERIFIED_UNTIL}"))
    elif scan:
        issues.append((f"{name}", "empty_rows"))
    return issues, len(files), len(scan), total_rows, unreadable


def check_datasets(sample: int) -> CheckResult:
    """Required-dataset pre-gate: empty/missing data must FAIL."""
    res = CheckResult("datasets", True, "")
    if ALLOW_EMPTY:
        res.summary = "skipped (--allow-empty)"
        return res
    n_files = n_scanned = n_rows = 0
    issues: list = []
    root = A_SHARES.parent.resolve()
    canonical = ("daily", "features", "features_panel")
    for name in REQUIRED_DATASETS:
        d = _dataset_dir(name)
        # §九.1: a custom (non-canonical) required dataset must live INSIDE
        # the data root — resolve against the REAL path, not the basename.  A
        # name that escapes the root (e.g. "../features") is refused outright
        # instead of being scanned somewhere the gate never intended.
        # Canonical names map to explicit dirs and skip the guard.
        if name not in canonical and not d.resolve().is_relative_to(root):
            issues.append((name, "outside_data_root"))
            continue
        iss, nf, ns, nr, nu = _scan_dataset(name, d, sample)
        n_files += nf
        n_scanned += ns
        n_rows += nr
        res.unreadable_files += nu
        issues.extend(iss)
    res.files_scanned = n_files
    res.rows_scanned = n_rows
    res.issues = issues
    # §九.2: record the audit denominator — how many files really exist vs how
    # many were row-read under --quick.  A consumer (train_panel) reads
    # scanned_files/total_files to judge whether a sampled gate is enough.
    res.scanned_files = n_scanned
    if issues:
        res.passed = False
    first = issues[0] if issues else ("", "")
    res.summary = (
        f"{'FAIL' if issues else 'OK'} files={n_files} rows={n_rows} "
        f"unreadable={res.unreadable_files} "
        f"datasets={','.join(REQUIRED_DATASETS)}"
        + (f" first={first[0]}:{first[1]}" if issues else "")
    )
    return res


# ── checks ──────────────────────────────────────────────────────────────

def check_daily_internal(sample: int) -> CheckResult:
    """pct_change must equal close.pct_change()*100 (fill-0 pollution)."""
    res = CheckResult("daily_internal", True, "")
    files = _sample_files(sorted(glob.glob(str(DAILY_DIR / "*.parquet"))), sample)
    res.files_scanned = len(files)
    max_diff = 0.0
    poll_total = 0
    for fp in files:
        code = Path(fp).stem
        d = _load_daily(code, ["date", "close", "pct_change"])
        if d is None or "pct_change" not in d or "close" not in d:
            res.issues.append((code, "missing_col/read_err"))
            res.passed = False
            continue
        close = d["close"].astype("float64")
        pc = d["pct_change"].astype("float64")
        recomputed = close.pct_change() * 100.0
        diff = (pc - recomputed).abs().dropna()
        md = float(diff.max()) if len(diff) else 0.0
        max_diff = max(max_diff, md)
        res.rows_scanned += int(len(diff))
        poll = int(((pc == 0.0) & (recomputed.abs() > 1e-4)).sum())
        poll_total += poll
        if md > 0.01 or poll:
            res.passed = False
            res.issues.append((code, f"md={md:.4f} poll={poll}"))
    res.summary = (
        f"max_diff={max_diff:.6f} pollution_rows={poll_total} "
        f"problem_files={len(res.issues)}"
    )
    return res


def check_aux_pct_aligned(sample: int) -> CheckResult:
    """aux pct_change must equal daily pct_change on every overlapping date."""
    res = CheckResult("aux_pct_aligned", True, "")
    files = []
    for d in AUX_PCT_DIRS:
        files += glob.glob(str(A_SHARES / d / "*.parquet"))
    files = _sample_files(sorted(files), sample)
    res.files_scanned = len(files)
    max_diff = 0.0
    for fp in files:
        d = os.path.basename(os.path.dirname(fp))
        code = Path(fp).stem
        daily = _load_daily(code, ["date", "pct_change"])
        if daily is None:
            res.issues.append((f"{d}/{code}", "no_daily"))
            res.passed = False
            continue
        try:
            a = pd.read_parquet(fp, columns=["date", "pct_change"])
        except Exception:
            res.issues.append((f"{d}/{code}", "read_err"))
            res.passed = False
            continue
        if "pct_change" not in a:
            res.issues.append((f"{d}/{code}", "missing_col"))
            res.passed = False
            continue
        a["date"] = pd.to_datetime(a["date"])
        daily["pct_change"] = daily["pct_change"].fillna(0.0)
        aligned = a["date"].map(daily.set_index("date")["pct_change"])
        diff = (a["pct_change"].astype("float64") - aligned).abs().dropna()
        md = float(diff.max()) if len(diff) else 0.0
        max_diff = max(max_diff, md)
        res.rows_scanned += int(len(diff))
        if md > 1e-6:
            res.passed = False
            res.issues.append((f"{d}/{code}", f"md={md:.6f}"))
    res.summary = f"max_diff={max_diff:.8f} problem_files={len(res.issues)}"
    return res


def check_aux_close_aligned(sample: int) -> CheckResult:
    """Processed OHLC must equal canonical daily close (调整基准漂移)."""
    res = CheckResult("aux_close_aligned", True, "")
    files = []
    for d in AUX_CLOSE_DIRS:
        files += glob.glob(str(A_SHARES / d / "*.parquet"))
    files = _sample_files(sorted(files), sample)
    res.files_scanned = len(files)
    max_diff = 0.0
    OHLC = ["open", "high", "low", "close"]
    for fp in files:
        d = os.path.basename(os.path.dirname(fp))
        code = Path(fp).stem
        try:
            a = pd.read_parquet(fp, columns=["date"] + OHLC)
        except Exception:
            # Some event dirs embed only close (no open/high/low); retry that.
            try:
                a = pd.read_parquet(fp, columns=["date", "close"])
            except Exception:
                res.issues.append((f"{d}/{code}", "no_ohlc/read_err"))
                res.passed = False
                continue
        if "close" not in a or "date" not in a:
            res.issues.append((f"{d}/{code}", "missing_col"))
            res.passed = False
            continue
        daily = _load_daily(
            code, ["date"] + [c for c in ["open", "high", "low", "close"] if c in a]
        )
        if daily is None:
            res.issues.append((f"{d}/{code}", "no_daily"))
            res.passed = False
            continue
        a["date"] = pd.to_datetime(a["date"])
        daily_idx = daily.set_index("date")
        for c in ["close"] + [x for x in ("open", "high", "low") if x in a]:
            aligned = a["date"].map(daily_idx[c])
            diff = (a[c].astype("float64") - aligned).abs().dropna()
            if len(diff):
                md = float(diff.max())
                max_diff = max(max_diff, md)
                res.rows_scanned += int(len(diff))
                # Relative tolerance: allow float rounding but catch basis drift.
                rel = (diff / a[c].astype("float64").abs().clip(lower=1.0)).max()
                if rel > 1e-3:
                    res.passed = False
                    res.issues.append((f"{d}/{code}", f"{c} rel={rel:.5f}"))
    res.summary = f"max_abs_diff={max_diff:.6f} problem_files={len(res.issues)}"
    return res


def check_feature_pct(sample: int) -> CheckResult:
    """Feature pct_change must equal daily (feature-layer pollution canary)."""
    res = CheckResult("feature_pct", True, "")
    feats = _sample_files(sorted(glob.glob(str(FEAT_DIR / "*.parquet"))), sample)
    res.files_scanned = len(feats)
    max_diff = 0.0
    CUTOFF = pd.Timestamp("2026-06-18")
    for fp in feats:
        code = Path(fp).stem
        daily = _load_daily(code, ["date", "pct_change"])
        if daily is None:
            res.issues.append((code, "no_daily"))
            res.passed = False
            continue
        try:
            f = pd.read_parquet(fp, columns=["date", "pct_change"])
        except Exception:
            res.issues.append((code, "read_err"))
            res.passed = False
            continue
        if "pct_change" not in f:
            res.issues.append((code, "no_feat_pc"))
            res.passed = False
            continue
        f["date"] = pd.to_datetime(f["date"])
        daily = daily.set_index("date")["pct_change"].fillna(0.0)
        fv = f.set_index("date")["pct_change"].astype("float64")
        m = pd.concat([fv, daily], axis=1, keys=["feat", "daily"]).dropna()
        if m.empty:
            continue
        diff = (m["feat"] - m["daily"]).abs()
        md = float(diff.max())
        max_diff = max(max_diff, md)
        res.rows_scanned += int(len(diff))
        post = m.index >= CUTOFF
        poll = 0
        if post.any():
            poll = int(((m["feat"] == 0) & (m["daily"].abs() > 1e-4))[post].sum())
        if md > 0.5 or poll:
            res.passed = False
            res.issues.append((code, f"md={md:.4f} poll_zero={poll}"))
    res.summary = f"max_diff={max_diff:.6f} problem_files={len(res.issues)}"
    return res


def check_sparsity(sample: int) -> CheckResult:
    """Per-feature non-zero coverage across a sampled panel (event-sparse canary).

    ``(x != 0).mean()`` counts NaN as non-zero (NaN != 0 is True), which
    inflates coverage for missing-heavy features. Report finite-excluded ratios:
      finite_cov        = np.isfinite(x).mean()
      effective_nonzero = (np.isfinite(x) & (x != 0)).mean()
    """
    res = CheckResult("sparsity", True, "")
    feats = _sample_files(sorted(glob.glob(str(FEAT_DIR / "*.parquet"))), sample)
    res.files_scanned = len(feats)
    finite: dict[str, list] = {}
    nz: dict[str, list] = {}
    for fp in feats:
        try:
            df = pd.read_parquet(fp)
        except Exception:
            continue
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                continue
            x = df[c].to_numpy()
            if x.size == 0:
                continue
            finite.setdefault(c, []).append(float(np.isfinite(x).mean()))
            nz.setdefault(c, []).append(float((np.isfinite(x) & (x != 0)).mean()))
    sparse = []
    for c, vals in nz.items():
        eff = float(np.mean(vals))
        if eff < SPARSE_NONZERO_RATIO:
            sparse.append((c, round(eff, 5)))
    sparse.sort(key=lambda t: t[1])
    finite_covs = [float(np.mean(vals)) for vals in finite.values()]
    avg_finite_cov = float(np.mean(finite_covs)) if finite_covs else 0.0
    if len(sparse) > 200:
        res.issues = [(f"{c}", f"nz={nz}") for c, nz in sparse[:200]]
        res.summary = (
            f"event-sparse features={len(sparse)} (showing 200) "
            f"min_nz={sparse[0][1] if sparse else 0} avg_finite_cov={avg_finite_cov:.3f}"
        )
    else:
        res.issues = [(f"{c}", f"nz={nz}") for c, nz in sparse]
        res.summary = (
            f"event-sparse features={len(sparse)} (non_zero_ratio<{SPARSE_NONZERO_RATIO}) "
            f"avg_finite_cov={avg_finite_cov:.3f}"
        )
    # Sparsity is informational — never fails the gate by itself.
    return res


def check_ohlc_sanity(sample: int) -> CheckResult:
    """Raw daily files must be internally consistent.

    Dates unique / sorted / no-weekend (A-shares never trade weekends, even on
    调休 makeup days), stock_code == filename, prices > 0, low <= open/close <=
    high, volume/amount >= 0. Any violation fails the gate.
    """
    res = CheckResult("ohlc_sanity", True, "")
    files = _sample_files(sorted(glob.glob(str(DAILY_DIR / "*.parquet"))), sample)
    res.files_scanned = len(files)
    TOL = 1e-9
    dup_total = 0
    weekend_total = 0
    OHLC = ["open", "high", "low", "close"]
    for fp in files:
        code = Path(fp).stem
        try:
            d = pd.read_parquet(fp)
        except Exception:
            res.issues.append((code, "read_err"))
            res.passed = False
            continue
        if "date" not in d:
            res.issues.append((code, "missing_date"))
            res.passed = False
            continue
        dates = pd.to_datetime(d["date"], errors="coerce")
        if dates.isna().any():
            res.issues.append((code, f"na_dates={int(dates.isna().sum())}"))
            res.passed = False
            continue
        res.rows_scanned += int(len(dates))
        n_dup = int(dates.duplicated().sum())
        dup_total += n_dup
        if n_dup:
            res.passed = False
            res.issues.append((code, f"dup_dates={n_dup}"))
        if not dates.is_monotonic_increasing:
            res.passed = False
            res.issues.append((code, "unsorted_dates"))
        wk = int(dates.dt.dayofweek.isin([5, 6]).sum())
        weekend_total += wk
        if wk:
            res.passed = False
            res.issues.append((code, f"weekend_rows={wk}"))
        if "stock_code" in d:
            sc = d["stock_code"].astype(str).str.strip()
            mism = int((sc != code).sum())
            if mism:
                res.passed = False
                res.issues.append((code, f"stock_code_mismatch={mism}"))
        else:
            res.passed = False
            res.issues.append((code, "missing_stock_code"))
        missing_ohlc = [c for c in OHLC if c not in d]
        if missing_ohlc:
            res.passed = False
            res.issues.append((code, f"missing_ohlc={','.join(missing_ohlc)}"))
            continue
        o = d["open"].astype("float64").to_numpy()
        h = d["high"].astype("float64").to_numpy()
        l = d["low"].astype("float64").to_numpy()
        cl = d["close"].astype("float64").to_numpy()
        for c, v in (("open", o), ("high", h), ("low", l), ("close", cl)):
            n_neg = int((v <= 0).sum())
            if n_neg:
                res.passed = False
                res.issues.append((code, f"{c}<=0 n={n_neg}"))
        # NaN comparisons are False, so NaN cells don't false-trigger here.
        if int((l > h + TOL).sum()):
            res.passed = False
            res.issues.append((code, "low>high"))
        outside = ((cl < l - TOL) | (cl > h + TOL) | (o < l - TOL) | (o > h + TOL)).sum()
        if int(outside):
            res.passed = False
            res.issues.append((code, f"ohlc_outside n={int(outside)}"))
        for c in ("volume", "amount"):
            if c not in d:
                continue
            n_neg = int((pd.to_numeric(d[c], errors="coerce").to_numpy() < 0).sum())
            if n_neg:
                res.passed = False
                res.issues.append((code, f"{c}<0 n={n_neg}"))
    res.summary = (
        f"dup_dates={dup_total} weekend_rows={weekend_total} "
        f"problem_files={len(res.issues)}"
    )
    return res


def check_contract_schema(sample: int) -> CheckResult:
    """Every daily file must satisfy the frozen DAILY_EQUITY contract.

    Schema, primary-key uniqueness, date rules, unit sign constraints AND
    official-trading-calendar membership all come from
    ``stoke_ml.data.contract`` instead of ad-hoc local checks, so the gate and
    the storage/downloader layers share one source of truth.  The trading-day
    set is always supplied (never opt-in), so official holidays are caught, not
    just weekends.

    §七.2: this check is ALWAYS full-scan — the ``sample`` argument is ignored.
    A sampled schema audit could miss a corrupt file that a formal training run
    then consumes; the reviewer's floor for a non-full run is "at least full
    manifest/contract + sampled deep feature audit", and contract conformance is
    the cheapest full-coverage layer worth enforcing unconditionally.
    """
    res = CheckResult("contract_schema", True, "")
    files = sorted(glob.glob(str(DAILY_DIR / "*.parquet")))
    res.files_scanned = len(files)
    contract = get_contract("daily_equity")
    for fp in files:
        code = Path(fp).stem
        try:
            d = pd.read_parquet(fp)
        except Exception:
            res.issues.append((code, "read_err"))
            res.passed = False
            continue
        # Pass the strongly-bound manifest so required_metadata (source /
        # adjustment_mode) is satisfied by the sidecar even for legacy files
        # whose parquet carries no provenance attrs.
        manifest = None
        mfp = Path(fp).with_suffix(".manifest.json")
        if mfp.is_file():
            try:
                manifest = json.loads(mfp.read_text(encoding="utf-8"))
            except Exception as exc:
                # A present-but-unparseable manifest is corruption, not absence.
                # Fall back to no-manifest (validate_contract still runs) but
                # surface the category so it is not silently swallowed.
                logger.warning(
                    "manifest %s unparseable (category=%s): %s",
                    mfp.name, classify_error(exc).value, exc,
                )
                manifest = None
        violations = validate_contract(
            d, contract, code=code, manifest=manifest,
            trading_days=_official_trading_days(d["date"]),
        )
        if violations:
            res.passed = False
            res.issues.append((code, ";".join(violations[:8])))
    res.summary = f"problem_files={len(res.issues)}"
    return res


def check_manifest(sample: int) -> CheckResult:
    """Per-stock manifest must exist and match the parquet it describes.

    The manifest is the range/completeness authority (start/end/rows/source/
    adjust/units/basis/versions + content schema-hash).  A file that exists
    without a valid manifest — or whose manifest no longer matches the parquet
    after an in-place edit, a partial write or a re-adjustment — fails here, so
    the gate independently enforces the §四-4 division rather than trusting
    "file exists".

    §七.2: this check is ALWAYS full-scan — the ``sample`` argument is ignored
    (a manifest gap in a skipped file is a silently unverifiable range claim).
    The sidecar is a few KB per stock, so full coverage costs little.
    """
    res = CheckResult("manifest", True, "")
    files = sorted(glob.glob(str(DAILY_DIR / "*.parquet")))
    res.files_scanned = len(files)
    # Storage root is derived from the daily dir the gate is actually scanning,
    # so a --data-dir redirect validates the same root (not a hardcoded data/).
    storage = DataStorage(str(DAILY_DIR.parents[1]))
    for fp in files:
        code = Path(fp).stem
        report = storage.validate_manifest(code, "a_shares")
        if not report["ok"]:
            res.passed = False
            detail = report.get("reason")
            if not detail:
                detail = ";".join(report.get("mismatches", [])[:8])
            res.issues.append((code, detail))
    res.summary = f"problem_files={len(res.issues)}"
    return res


# ── §P1-7: per-requested-stock reconciliation ─────────────────────────────
# The dataset checks above gate the POOL (files/stocks/rows/spans) but never
# answer "did the run that requested 5530 stocks actually deliver every one of
# them".  This layer reconciles a requested universe code-by-code: parquet
# present, contract manifest valid, data not degraded.  It is a new, additive
# layer — it only runs when a requested universe is passed; without one the
# gate behaves exactly as before.


def _tolerance_count(requested: int, ratio: float) -> int:
    """Max tolerated failing stocks for a ratio (0.0 → strict zero tolerance)."""
    if ratio <= 0.0:
        return 0
    return int(math.ceil(ratio * requested))


def _requested_from_manifest(data: dict) -> dict:
    """Extract codes + requested date range from a download run manifest dict.

    ``requested`` is the full universe the run set out to fetch (§五-4).  The
    requested interval is start_date → requested_end (the bounded request end,
    recorded verbatim), falling back to effective_end then end_date (§七-2).
    """
    requested = data.get("requested")
    if not isinstance(requested, list):
        raise ValueError("download run manifest 'requested' must be a list")
    codes = []
    for tok in requested:
        code = normalize_stock_code(tok)
        if code:
            codes.append(code)
        else:
            logger.warning("requested-universe: dropping unusable requested code %r", tok)
    end = data.get("requested_end") or data.get("effective_end") or data.get("end_date")
    return {
        "codes": codes,
        "requested_start": data.get("start_date"),
        "requested_end": end,
        # §八-2: carry the run-level download status so the gate report's
        # universe_reconciliation exposes the run's OWN completion claim
        # (status / counts / failed / missing / all_complete) next to the
        # on-disk reconciliation.  Only a download run manifest source can
        # enrich the report this way.
        "run_manifest": {
            "status": data.get("status"),
            "requested_count": data.get("requested_count"),
            "complete_count": data.get("complete_count"),
            "failed_count": data.get("failed_count"),
            "missing_count": data.get("missing_count"),
            "all_complete": data.get("all_complete"),
            "effective_end": data.get("effective_end"),
            "latest_available_end": data.get("latest_available_end"),
            "failed": data.get("failed") or [],
            "missing": data.get("missing") or [],
        },
    }


def _parse_requested_json(text: str) -> dict | None:
    """Parse a requested-universe JSON source.

    Recognizes (a) a download run manifest object with a ``requested`` list
    (also carries the requested date range), or (b) a plain JSON list of codes.
    Returns ``None`` when ``text`` is not JSON (caller falls back to line-per-
    code).
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict) and "requested" in data:
        return _requested_from_manifest(data)
    if isinstance(data, list):
        return {
            "codes": [c for c in (normalize_stock_code(t) for t in data) if c],
            "requested_start": None,
            "requested_end": None,
        }
    return None


def _read_requested_file(path: str, require_manifest: bool = False) -> dict:
    """Read a requested-universe source file into codes + optional date range.

    ``require_manifest`` (--request-manifest) reads the file through the
    download-run manifest module's own loader and refuses anything without a
    ``requested`` field.  Otherwise the format is auto-detected: (a) a download
    run manifest JSON, (b) a plain JSON list of codes, or (c) a text/CSV file
    with one code per line (blank lines and ``#`` comments ignored).  Codes
    normalize through ``normalize_stock_code`` (shared with the downloader /
    storage layers).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"requested-universe file not found: {path}")
    if require_manifest:
        data = load_manifest(path)
        if data is None:
            raise ValueError(f"--request-manifest unreadable or empty: {path}")
        if not isinstance(data, dict) or "requested" not in data:
            raise ValueError("--request-manifest file has no 'requested' field")
        info = _requested_from_manifest(data)
    else:
        text = p.read_text(encoding="utf-8")
        info = _parse_requested_json(text)
        if info is None:
            info = {"codes": [], "requested_start": None, "requested_end": None}
            for line in text.splitlines():
                token = line.strip()
                if not token or token.startswith("#"):
                    continue
                code = normalize_stock_code(token)
                if code is None:
                    logger.warning("requested-universe: dropping unusable code token %r", token)
                    continue
                info["codes"].append(code)
    # Dedup (e.g. "000001" + "000001.0" collapse to one code) preserving order.
    seen: set[str] = set()
    info["codes"] = [c for c in info["codes"] if not (c in seen or seen.add(c))]
    return info


def _degradation_reason(
    storage: DataStorage,
    code: str,
    market: str,
    *,
    req_days: frozenset | None,
    min_rows: int,
    min_coverage_ratio: float,
) -> str | None:
    """Why a PRESENT stock is degraded, or ``None`` if it is sound.

    A present parquet is degraded when (1) its contract manifest is missing /
    stale / mismatched — ``validate_manifest`` reuses the §四-4 authority
    instead of re-implementing it; (2) its valid row count is below the
    optional ``min_rows`` floor; or (3) its actual trading-day coverage of the
    requested interval is below the optional ``min_coverage_ratio`` (a stock
    whose history is a thin fragment of the request is not "complete").
    """
    report = storage.validate_manifest(code, market)
    if not report["ok"]:
        detail = report.get("reason")
        if not detail:
            detail = ";".join(report.get("mismatches", [])[:8])
        return f"manifest_invalid: {detail}" if detail else "manifest_invalid"
    actual = report.get("actual") or {}
    rows = actual.get("rows")
    if min_rows and rows is not None and rows < min_rows:
        return f"rows={rows} < min={min_rows}"
    if req_days is not None and min_coverage_ratio > 0.0:
        a_lo = pd.Timestamp(actual["start"]).date() if actual.get("start") else None
        a_hi = pd.Timestamp(actual["end"]).date() if actual.get("end") else None
        if a_lo is not None and a_hi is not None and len(req_days):
            covered = sum(1 for dd in req_days if a_lo <= dd <= a_hi)
            ratio = covered / len(req_days)
            if ratio < min_coverage_ratio:
                return f"coverage_ratio={ratio:.3f} < min={min_coverage_ratio}"
    return None


def reconcile_requested_universe(
    codes: list[str],
    *,
    daily_dir: Path | None = None,
    requested_start: str | None = None,
    requested_end: str | None = None,
    min_rows: int = 0,
    min_coverage_ratio: float = 0.0,
    max_missing_ratio: float = 0.0,
    max_degraded_ratio: float = 0.0,
) -> dict:
    """Reconcile a requested stock universe against on-disk daily files (§P1-7).

    For every requested code, in order: (1) the flat parquet must exist — an
    absent parquet is a MISSING stock; (2) the contract manifest must be valid
    — a present file whose sidecar is missing/stale/mismatched is DEGRADED
    ("file exists" never stands in for "data complete", §四-4); (3) the data
    must not be degraded per the optional ``min_rows`` / ``min_coverage_ratio``
    floors (a request interval must be supplied for the coverage ratio).

    Reuses ``DataStorage.validate_manifest`` (the same authority the
    ``manifest`` check uses) rather than re-reading parquets by hand.

    Returns a structured report: ``requested_count`` / ``present_count`` /
    ``missing_codes`` / ``degraded_codes`` (each with a reason) / ``ok``.
    ``ok`` honours the tolerance ratios ``max_missing_ratio`` /
    ``max_degraded_ratio`` (default 0.0 = any gap fails, matching the gate's
    "a problem recorded must flip the gate" rule).
    """
    d = Path(daily_dir) if daily_dir is not None else DAILY_DIR
    market = d.parent.name
    storage = DataStorage(str(d.parents[1]))
    requested = sorted({c for c in codes if c})
    req_days: frozenset | None = None
    if requested_start and requested_end:
        try:
            req_lo = pd.Timestamp(requested_start).date()
            req_hi = pd.Timestamp(requested_end).date()
            req_days = frozenset(_CALENDAR.get_trading_days(req_lo, req_hi))
        except Exception:
            req_days = None
    missing: list[str] = []
    degraded: list[dict] = []
    present = 0
    for code in requested:
        if not (d / f"{code}.parquet").is_file():
            missing.append(code)
            continue
        present += 1
        reason = _degradation_reason(
            storage, code, market,
            req_days=req_days,
            min_rows=min_rows,
            min_coverage_ratio=min_coverage_ratio,
        )
        if reason:
            degraded.append({"code": code, "reason": reason})
    n = len(requested)
    return {
        "ok": (
            len(missing) <= _tolerance_count(n, max_missing_ratio)
            and len(degraded) <= _tolerance_count(n, max_degraded_ratio)
        ),
        "requested_count": n,
        "present_count": present,
        "missing_count": len(missing),
        "degraded_count": len(degraded),
        "missing_codes": missing,
        "degraded_codes": degraded,
    }


def check_universe(sample: int) -> CheckResult:
    """Per-requested-stock reconciliation against the requested universe (§P1-7).

    ``sample`` is ignored — a requested universe is a complete accounting, never
    a sample.  No-ops (PASS) when no requested universe was supplied.
    """
    res = CheckResult("universe", True, "")
    req = _UNIVERSE_REQUEST
    if req is None:
        res.summary = "skipped (no requested universe)"
        return res
    report = reconcile_requested_universe(
        req["codes"],
        daily_dir=DAILY_DIR,
        requested_start=req.get("requested_start"),
        requested_end=req.get("requested_end"),
        min_rows=req.get("min_rows", 0),
        min_coverage_ratio=req.get("min_coverage_ratio", 0.0),
        max_missing_ratio=req.get("max_missing_ratio", 0.0),
        max_degraded_ratio=req.get("max_degraded_ratio", 0.0),
    )
    # §八-2: a download run manifest source enriches the reconciliation with
    # the run's OWN completion claim (status / counts / failed / missing) —
    # the gate report then records BOTH what the run claimed and what actually
    # landed on disk.  A plain code-list source has no run_manifest and the
    # report shape stays unchanged.
    rm = req.get("run_manifest")
    if rm is not None:
        report["run_manifest"] = rm
    res.details = report
    res.files_scanned = report["requested_count"]
    res.issues = [(c, "missing") for c in report["missing_codes"]]
    res.issues += [(item["code"], item["reason"]) for item in report["degraded_codes"]]
    res.passed = report["ok"]
    res.summary = (
        f"{'FAIL' if not report['ok'] else 'OK'} "
        f"requested={report['requested_count']} present={report['present_count']} "
        f"missing={report['missing_count']} degraded={report['degraded_count']}"
    )
    return res


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


def _build_universe_request(args) -> dict | None:
    """Assemble the §P1-7 universe request from CLI args, or None if none given.

    Sources may combine: ``--requested-universe <file>`` (auto-detected: a
    download run manifest, a JSON code list, or a line-per-code file),
    ``--request-manifest <file>`` (explicit download run manifest), and
    ``--universe-codes a,b,c`` (inline).  Codes are unioned and de-duplicated;
    the requested date range comes from a manifest source (start_date /
    requested_end).  An unreadable / non-manifest --request-manifest raises.
    """
    codes: list[str] = []
    requested_start = requested_end = None
    run_manifest: dict | None = None
    sources: list[str] = []
    if args.requested_universe:
        info = _read_requested_file(args.requested_universe)
        codes += info["codes"]
        requested_start = requested_start or info["requested_start"]
        requested_end = requested_end or info["requested_end"]
        run_manifest = run_manifest or info.get("run_manifest")
        sources.append(args.requested_universe)
    if args.request_manifest:
        info = _read_requested_file(args.request_manifest, require_manifest=True)
        codes += info["codes"]
        requested_start = requested_start or info["requested_start"]
        requested_end = requested_end or info["requested_end"]
        run_manifest = run_manifest or info.get("run_manifest")
        sources.append(args.request_manifest)
    if args.universe_codes:
        codes += [
            c for c in (normalize_stock_code(t) for t in args.universe_codes.split(","))
            if c
        ]
        sources.append("inline")
    if not sources:
        return None
    seen: set[str] = set()
    uniq = [c for c in codes if not (c in seen or seen.add(c))]
    if not uniq:
        raise ValueError(
            "requested universe resolved to no usable codes (empty / all "
            "unusable entries)"
        )
    return {
        "codes": uniq,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "min_rows": args.min_universe_rows,
        "min_coverage_ratio": args.min_universe_coverage,
        "max_missing_ratio": args.max_universe_missing_ratio,
        "max_degraded_ratio": args.max_universe_degraded_ratio,
        "sources": sources,
        "run_manifest": run_manifest,
    }


def main():
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    global MIN_FILES, MIN_STOCKS, MIN_ROWS, MIN_SPAN_DAYS, MAX_STALE_DAYS
    global MAX_UNREADABLE_RATIO, FORMAL_STOCK_RATIO
    global ALLOW_EMPTY, A_SHARES, DAILY_DIR, FEAT_DIR
    global ENFORCE_VERIFIED_UNTIL, _UNIVERSE_REQUEST
    ap = argparse.ArgumentParser(description="Data quality gate")
    ap.add_argument("--check", default=None,
                    help="comma-separated checks (default: all)")
    ap.add_argument("--sample", type=int, default=0,
                    help="cap files per check (0 = all)")
    ap.add_argument("--quick", action="store_true",
                    help="shorthand for --sample 300 (CI / post-build gate)")
    ap.add_argument("--output", default="reports",
                    help="report dir (default reports/)")
    ap.add_argument("--data-dir", default=None,
                    help="data root (default: <repo>/data) — gate the same root "
                         "training reads so gate-PASS and train-read can't diverge")
    ap.add_argument("--require", default="daily",
                    help="comma-separated required datasets: "
                         "daily,features,features_panel (default: daily)")
    ap.add_argument("--allow-empty", action="store_true",
                    help="permit empty/missing required datasets (dev bootstrap)")
    ap.add_argument("--min-files", type=int, default=MIN_FILES,
                    help="minimum parquet files per required dataset")
    ap.add_argument("--min-stocks", type=int, default=MIN_STOCKS,
                    help="minimum readable stocks per required dataset")
    ap.add_argument("--min-rows", type=int, default=MIN_ROWS,
                    help="minimum total rows per required dataset")
    ap.add_argument("--min-span-days", type=int, default=MIN_SPAN_DAYS,
                    help="minimum earliest→latest span per required dataset")
    ap.add_argument("--max-stale-days", type=int, default=MAX_STALE_DAYS,
                    help="max calendar days since latest date before FAIL")
    ap.add_argument("--max-unreadable-ratio", type=float, default=None,
                    help="max unreadable-file share per required dataset "
                         "(default: 0.05; formal profile forces 0.0, §六-3)")
    ap.add_argument("--stock-ratio", type=float, default=None,
                    help="min readable-stock fraction of the scanned pool "
                         "(0.0 = disabled; formal profile forces 0.98, §六-4)")
    ap.add_argument("--profile", type=str, default="bootstrap",
                    choices=["bootstrap", "formal"],
                    help="required-dataset strictness profile (§六-4): "
                         "bootstrap (default, dev) or formal — a 5530-stock "
                         "research run must clear: span >= 5y, stale <= 4d, "
                         "unreadable = 0, readable stocks >= 98%")
    # §P1-7: per-requested-stock reconciliation — OPT-IN; without one of these
    # the gate runs exactly as before (the universe check never joins the run).
    ap.add_argument("--requested-universe", default=None,
                    help="requested-universe file: a download run manifest JSON "
                         "(data/a_shares/download_manifest.json, 'requested' "
                         "field), a JSON code list, or a line-per-code text/CSV")
    ap.add_argument("--request-manifest", default=None,
                    help="explicit download run manifest JSON (must carry "
                         "'requested'); also supplies the requested date range")
    ap.add_argument("--universe-codes", default=None,
                    help="comma-separated inline requested code list")
    ap.add_argument("--min-universe-rows", type=int, default=0,
                    help="§P1-7 degraded floor: a requested stock whose valid "
                         "rows fall below this is DEGRADED (0 = disabled)")
    ap.add_argument("--min-universe-coverage", type=float, default=0.0,
                    help="§P1-7 degraded floor: a requested stock whose "
                         "trading-day coverage of the requested interval is "
                         "below this ratio is DEGRADED (0 = disabled; needs a "
                         "manifest source for the requested interval)")
    ap.add_argument("--max-universe-missing-ratio", type=float, default=0.0,
                    help="max tolerated missing-stock share of the requested "
                         "universe before FAIL (0.0 = any missing fails)")
    ap.add_argument("--max-universe-degraded-ratio", type=float, default=0.0,
                    help="max tolerated degraded-stock share of the requested "
                         "universe before FAIL (0.0 = any degraded fails)")
    args = ap.parse_args()

    # §六-4 frozen formal profile: research-run floors override loose dev
    # defaults so a production build/train can't pass on thin or corrupt data.
    if args.profile == "formal":
        args.min_span_days = FORMAL_PROFILE["min_span_days"]
        args.max_stale_days = FORMAL_PROFILE["max_stale_days"]
        args.max_unreadable_ratio = FORMAL_PROFILE["max_unreadable_ratio"]
        args.stock_ratio = FORMAL_PROFILE["stock_ratio"]
        # §九-3: refuse data that extends past verified_until — forward-estimate
        # holidays are not exchange fact and must not be validated as such.
        ENFORCE_VERIFIED_UNTIL = True

    MIN_FILES = args.min_files
    MIN_STOCKS = args.min_stocks
    MIN_ROWS = args.min_rows
    MIN_SPAN_DAYS = args.min_span_days
    MAX_STALE_DAYS = args.max_stale_days
    if args.max_unreadable_ratio is not None:
        MAX_UNREADABLE_RATIO = args.max_unreadable_ratio
    if args.stock_ratio is not None:
        FORMAL_STOCK_RATIO = args.stock_ratio
    ALLOW_EMPTY = args.allow_empty
    REQUIRED_DATASETS[:] = [x.strip() for x in args.require.split(",") if x.strip()]
    if args.data_dir:
        root = Path(args.data_dir).resolve()
        _DAILY_CACHE.clear()
        A_SHARES = root / "a_shares"
        DAILY_DIR = A_SHARES / "daily"
        FEAT_DIR = root / "features"

    # §P1-7/§八-2: build the optional requested-universe reconciliation.  The
    # check only joins the run when a universe source is supplied (additive);
    # without one, the run is identical to before.  A missing --request-manifest
    # (or one that resolves to no usable codes) FAILS cleanly instead of
    # tracebacking — §八-2 formal mode refuses to silently resolve to whatever
    # happens to be on disk.
    try:
        _UNIVERSE_REQUEST = _build_universe_request(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"requested-universe ERROR: {exc}", file=sys.stderr)
        return 1

    names = (args.check.split(",") if args.check else list(CHECKS))
    if _UNIVERSE_REQUEST is not None and "universe" not in names:
        names.append("universe")
    available = set(CHECKS) | {"universe"}
    unknown = [n for n in names if n not in available]
    if unknown:
        print(f"unknown checks: {unknown}; available: {sorted(available)}")
        return 2
    sample = args.sample or (300 if args.quick else 0)

    results = []
    for name in names:
        t0 = time.time()
        r = RUN_CHECKS[name](sample)
        dt = time.time() - t0
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name:18s} ({dt:.1f}s) {r.summary}")
        for file, detail in r.issues[:15]:
            print(f"         {file}: {detail}")

    passed = all(r.passed for r in results)
    os.makedirs(args.output, exist_ok=True)
    # §七.2: record the run's audit scope so a consumer can tell a full scan
    # from a --quick sample.  manifest/contract_schema are always full-scan
    # (see their docstrings), so formal training can accept a sampled run only
    # when those two really covered every file — the reviewer's "at least full
    # manifest/contract + sampled deep feature audit" floor.  A consumer reads
    # manifest_contract_full_scan to prove that half before trusting a sample.
    total_daily = len(glob.glob(str(DAILY_DIR / "*.parquet")))
    # v14 §八-1: manifest_contract_full_scan is true only when BOTH the manifest
    # and contract_schema checks actually ran, both passed, both covered every
    # daily file and neither reported an unreadable file.  A `--check manifest`
    #-only run leaves contract_schema unproven and must NOT satisfy the floor.
    manifest_contract_full_scan = _manifest_contract_full_scan(results, total_daily)
    # §六-2: a consuming run (train_panel) must be able to verify this report
    # really covers the data it reads — gate version, data root, calendar +
    # contract fingerprints, the required-dataset list and the run-level
    # dataset fingerprint are frozen alongside PASS so a stale/mismatched
    # report is refused instead of silently accepted.
    # §九.1: dataset_paths binds each required dataset to the ABSOLUTE dir the
    # gate validated, so a consumer compares it against the real path it reads
    # (a custom prebuilt basename can no longer pass a wrong-dir gate).
    datasets_check = next((r for r in results if r.name == "datasets"), None)
    total_files = datasets_check.files_scanned if datasets_check else total_daily
    scanned_files = (
        datasets_check.scanned_files if datasets_check else total_daily
    )
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "passed": passed,
        "quality_gate_version": QUALITY_GATE_VERSION,
        "data_root": str(A_SHARES.parent),
        "calendar_version": TradingCalendar.CALENDAR_VERSION,
        "contract_version": contract_version(),
        "required_datasets": list(REQUIRED_DATASETS),
        "dataset_paths": {
            name: str(_dataset_dir(name).resolve())
            for name in REQUIRED_DATASETS
        },
        "profile": args.profile,
        "scope": "full" if sample == 0 else "sample",
        "sample_size": sample,
        "sample_seed": SAMPLE_SEED,
        "scanned_files": scanned_files,   # files actually row-read (§九.2)
        "total_files": total_files,       # true on-disk parquet count (§九.2)
        "manifest_contract_full_scan": manifest_contract_full_scan,
        "data_manifest_hash": dataset_fingerprint(A_SHARES.parent, REQUIRED_DATASETS),
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "summary": r.summary,
                "files_scanned": r.files_scanned,
                "rows_scanned": r.rows_scanned,
                "unreadable_files": r.unreadable_files,
                "issue_count": len(r.issues),
                "sample_issues": [{"file": f, "detail": d} for f, d in r.issues[:50]],
            }
            for r in results
        ],
    }
    # §P1-7: attach the structured universe reconciliation only when it ran —
    # a default run's report keeps its previous shape exactly.
    universe_res = next((r for r in results if r.name == "universe"), None)
    if universe_res is not None and universe_res.details is not None:
        report["universe_reconciliation"] = universe_res.details
    out_path = os.path.join(args.output, "data_quality_gate.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n{'PASS' if passed else 'FAIL'} — wrote {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
