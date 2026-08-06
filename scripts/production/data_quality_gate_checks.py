"""Data quality gate — check functions + §P1-7 universe reconciliation (§T16).

This module holds the gate's CHECK FUNCTIONS (``check_datasets``,
``check_daily_internal``, ...) and the §P1-7 per-requested-stock universe
reconciliation (``reconcile_requested_universe``, ``check_universe``,
``_build_universe_request``, and their internal loaders/helpers).

ALL mutable state lives in ``data_quality_gate.py`` — the constants
(``DAILY_DIR``, ``MIN_FILES``, ``MAX_STALE_DAYS``, ``_UNIVERSE_REQUEST``, ...),
the caches, the helpers (``_load_daily``, ``_scan_dataset``,
``_sample_files``, ``_get_calendar``, ``_official_trading_days``, ...) and
``CheckResult``.  This module never owns state and never binds it at import
time; every reference is resolved through the gate MODULE OBJECT ``_g`` at
CALL TIME, inside the function bodies.  That is exactly what keeps the test
seam working: tests monkeypatch gate-module globals on the gate module object
(``monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR",
...)``) and then call the check functions directly — a call-time ``_g.X`` read
sees the patch, a module-level ``X = _g.SOME_CONST`` (or ``from ... import
DAILY_DIR``) would bind at import time and miss it.

For that reason this module never dereferences ``_g`` at module top level — the
gate module imports this module (after defining all its state) and re-exports
the moved names, so ``gate_mod.<check_name>`` / ``from ... import
<check_name>`` keep resolving on the gate module as before (§T16).

``from __future__ import annotations`` keeps the ``-> _g.CheckResult`` return
annotations from being evaluated at def time — ``_g`` is only bound at the very
bottom of this module (see the NOTE there), so a function signature annotated
with ``_g.X`` must not be resolved during module load (§T16).
"""
from __future__ import annotations

import glob
import json
import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from stoke_ml.data.codes import normalize_stock_code
from stoke_ml.data.contract import get_contract, validate_contract
from stoke_ml.data.download_manifest import load_manifest
from stoke_ml.data.storage import DataStorage
from stoke_ml.utils.error_summary import classify_error

logger = logging.getLogger(__name__)


def check_datasets(sample: int) -> _g.CheckResult:
    """Required-dataset pre-gate: empty/missing data must FAIL."""
    res = _g.CheckResult("datasets", True, "")
    if _g.ALLOW_EMPTY:
        res.summary = "skipped (--allow-empty)"
        return res
    n_files = n_scanned = n_rows = 0
    issues: list = []
    root = _g.A_SHARES.parent.resolve()
    canonical = ("daily", "features", "features_panel")
    for name in _g.REQUIRED_DATASETS:
        d = _g._dataset_dir(name)
        # §九.1: a custom (non-canonical) required dataset must live INSIDE
        # the data root — resolve against the REAL path, not the basename.  A
        # name that escapes the root (e.g. "../features") is refused outright
        # instead of being scanned somewhere the gate never intended.
        # Canonical names map to explicit dirs and skip the guard.
        if name not in canonical and not d.resolve().is_relative_to(root):
            issues.append((name, "outside_data_root"))
            continue
        iss, nf, ns, nr, nu = _g._scan_dataset(name, d, sample)
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
        f"datasets={','.join(_g.REQUIRED_DATASETS)}"
        + (f" first={first[0]}:{first[1]}" if issues else "")
    )
    return res


# ── checks ──────────────────────────────────────────────────────────────

def check_daily_internal(sample: int) -> _g.CheckResult:
    """pct_change must equal close.pct_change()*100 (fill-0 pollution)."""
    res = _g.CheckResult("daily_internal", True, "")
    files = _g._sample_files(sorted(glob.glob(str(_g.DAILY_DIR / "*.parquet"))), sample)
    res.files_scanned = len(files)
    max_diff = 0.0
    poll_total = 0
    for fp in files:
        code = Path(fp).stem
        d = _g._load_daily(code, ["date", "close", "pct_change"])
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


def check_aux_pct_aligned(sample: int) -> _g.CheckResult:
    """aux pct_change must equal daily pct_change on every overlapping date."""
    res = _g.CheckResult("aux_pct_aligned", True, "")
    files = []
    for d in _g.AUX_PCT_DIRS:
        files += glob.glob(str(_g.A_SHARES / d / "*.parquet"))
    files = _g._sample_files(sorted(files), sample)
    res.files_scanned = len(files)
    max_diff = 0.0
    for fp in files:
        d = os.path.basename(os.path.dirname(fp))
        code = Path(fp).stem
        daily = _g._load_daily(code, ["date", "pct_change"])
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


def check_aux_close_aligned(sample: int) -> _g.CheckResult:
    """Processed OHLC must equal canonical daily close (调整基准漂移)."""
    res = _g.CheckResult("aux_close_aligned", True, "")
    files = []
    for d in _g.AUX_CLOSE_DIRS:
        files += glob.glob(str(_g.A_SHARES / d / "*.parquet"))
    files = _g._sample_files(sorted(files), sample)
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
        daily = _g._load_daily(
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


def check_feature_pct(sample: int) -> _g.CheckResult:
    """Feature pct_change must equal daily (feature-layer pollution canary)."""
    res = _g.CheckResult("feature_pct", True, "")
    feats = _g._sample_files(sorted(glob.glob(str(_g.FEAT_DIR / "*.parquet"))), sample)
    res.files_scanned = len(feats)
    max_diff = 0.0
    CUTOFF = pd.Timestamp("2026-06-18")
    for fp in feats:
        code = Path(fp).stem
        daily = _g._load_daily(code, ["date", "pct_change"])
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


def check_sparsity(sample: int) -> _g.CheckResult:
    """Per-feature non-zero coverage across a sampled panel (event-sparse canary).

    ``(x != 0).mean()`` counts NaN as non-zero (NaN != 0 is True), which
    inflates coverage for missing-heavy features. Report finite-excluded ratios:
      finite_cov        = np.isfinite(x).mean()
      effective_nonzero = (np.isfinite(x) & (x != 0)).mean()
    """
    res = _g.CheckResult("sparsity", True, "")
    feats = _g._sample_files(sorted(glob.glob(str(_g.FEAT_DIR / "*.parquet"))), sample)
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
        if eff < _g.SPARSE_NONZERO_RATIO:
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
            f"event-sparse features={len(sparse)} (non_zero_ratio<{_g.SPARSE_NONZERO_RATIO}) "
            f"avg_finite_cov={avg_finite_cov:.3f}"
        )
    # Sparsity is informational — never fails the gate by itself.
    return res


def check_ohlc_sanity(sample: int) -> _g.CheckResult:
    """Raw daily files must be internally consistent.

    Dates unique / sorted / no-weekend (A-shares never trade weekends, even on
    调休 makeup days), stock_code == filename, prices > 0, low <= open/close <=
    high, volume/amount >= 0. Any violation fails the gate.
    """
    res = _g.CheckResult("ohlc_sanity", True, "")
    files = _g._sample_files(sorted(glob.glob(str(_g.DAILY_DIR / "*.parquet"))), sample)
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


def check_contract_schema(sample: int) -> _g.CheckResult:
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
    res = _g.CheckResult("contract_schema", True, "")
    files = sorted(glob.glob(str(_g.DAILY_DIR / "*.parquet")))
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
            trading_days=_g._official_trading_days(d["date"]),
        )
        if violations:
            res.passed = False
            res.issues.append((code, ";".join(violations[:8])))
    res.summary = f"problem_files={len(res.issues)}"
    return res


def check_manifest(sample: int) -> _g.CheckResult:
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
    res = _g.CheckResult("manifest", True, "")
    files = sorted(glob.glob(str(_g.DAILY_DIR / "*.parquet")))
    res.files_scanned = len(files)
    # Storage root is derived from the daily dir the gate is actually scanning,
    # so a --data-dir redirect validates the same root (not a hardcoded data/).
    storage = DataStorage(str(_g.DAILY_DIR.parents[1]))
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
    d = Path(daily_dir) if daily_dir is not None else _g.DAILY_DIR
    market = d.parent.name
    storage = DataStorage(str(d.parents[1]))
    requested = sorted({c for c in codes if c})
    req_days: frozenset | None = None
    if requested_start and requested_end:
        try:
            req_lo = pd.Timestamp(requested_start).date()
            req_hi = pd.Timestamp(requested_end).date()
            req_days = frozenset(_g._get_calendar().get_trading_days(req_lo, req_hi))
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


def check_universe(sample: int) -> _g.CheckResult:
    """Per-requested-stock reconciliation against the requested universe (§P1-7).

    ``sample`` is ignored — a requested universe is a complete accounting, never
    a sample.  No-ops (PASS) when no requested universe was supplied.
    """
    res = _g.CheckResult("universe", True, "")
    req = _g._UNIVERSE_REQUEST
    if req is None:
        res.summary = "skipped (no requested universe)"
        return res
    report = reconcile_requested_universe(
        req["codes"],
        daily_dir=_g.DAILY_DIR,
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


# NOTE: the gate import sits at the BOTTOM of this module (after every check
# function + helper is defined) — the same circular-import-safe pattern
# ``data_quality_gate_run.py`` uses.  The gate module imports this module after
# defining all its state/constants/helpers/CheckResult, so importing the gate
# here can re-enter THIS module while it is only partially initialized.  By
# importing the gate at the very end, every name the gate module needs to
# re-export is already defined, so a re-entrant ``from
# data_quality_gate_checks import (...)`` succeeds (this is what makes
# ``python data_quality_gate.py`` — which runs the gate as ``__main__``, a
# DIFFERENT module object than ``scripts.production.data_quality_gate`` —
# resolve cleanly).  ``_g`` is only dereferenced inside function bodies at CALL
# TIME, so binding the (possibly partially initialized) module object here is
# always safe and the test seam keeps working.
from scripts.production import data_quality_gate as _g  # noqa: E402  circular-import-safe (§T16)
