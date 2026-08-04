"""Data quality gate —常驻数据质量门禁，整合分散的一次性验证。

Each check scans the on-disk data for a known corruption signature and either
passes or reports the offending files. The gate exits non-zero when any enabled
check fails, so it can gate CI / a post-download hook. Run it after any
download or feature rebuild.

Checks:
  daily_internal   : daily flat pct_change == close.pct_change()*100 (fill-0 pollution)
  aux_pct_aligned  : board/industry processed pct_change == daily pct_change (stale-0)
  aux_close_aligned: processed OHLC == canonical daily close (调整基准漂移)
  feature_pct      : built feature pct_change == daily (feature-layer pollution)
  sparsity         : per-feature effective non-zero coverage (NaN-excluded)
  ohlc_sanity      : dates unique/sorted/no-weekend, code==filename, prices>0,
                     low<=open/close<=high, volume/amount>=0

Any read error or missing column FAILS its check — a problem recorded in the
report must also flip the gate (v6 §九).

Output: reports/data_quality_gate.json (machine-readable) + console summary.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/data_quality_gate.py
  PYTHONPATH=. ./.venv/Scripts/python scripts/data_quality_gate.py --quick --sample 200
  PYTHONPATH=. ./.venv/Scripts/python scripts/data_quality_gate.py --check daily_internal,feature_pct
"""
import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stoke_ml.data.contract import get_contract, validate_contract

PROJECT = Path(__file__).resolve().parent.parent
A_SHARES = PROJECT / "data" / "a_shares"
DAILY_DIR = A_SHARES / "daily"
FEAT_DIR = PROJECT / "data" / "features"

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


@dataclass
class CheckResult:
    name: str
    passed: bool
    summary: str
    files_scanned: int = 0
    rows_scanned: int = 0
    issues: list = field(default_factory=list)  # list of (file, detail)


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
        except Exception:
            return None
        d["date"] = pd.to_datetime(d["date"])
        cached = d.drop_duplicates("date", keep="last").reset_index(drop=True)
        _DAILY_CACHE[code] = cached
    return cached[list(cols)].copy()


# ── checks ──────────────────────────────────────────────────────────────

def check_daily_internal(sample: int) -> CheckResult:
    """pct_change must equal close.pct_change()*100 (fill-0 pollution)."""
    res = CheckResult("daily_internal", True, "")
    files = sorted(glob.glob(str(DAILY_DIR / "*.parquet")))
    if sample:
        files = files[:sample]
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
    files.sort()
    if sample:
        files = files[:sample]
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
    files.sort()
    if sample:
        files = files[:sample]
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
    feats = sorted(glob.glob(str(FEAT_DIR / "*.parquet")))
    if sample:
        feats = feats[:sample]
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

    v6 §九: ``(x != 0).mean()`` counts NaN as non-zero (NaN != 0 is True), which
    inflates coverage for missing-heavy features. Report finite-excluded ratios:
      finite_cov        = np.isfinite(x).mean()
      effective_nonzero = (np.isfinite(x) & (x != 0)).mean()
    """
    res = CheckResult("sparsity", True, "")
    feats = sorted(glob.glob(str(FEAT_DIR / "*.parquet")))
    if sample:
        feats = feats[:sample]
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
    """Raw daily files must be internally consistent (v6 §九).

    Dates unique / sorted / no-weekend (A-shares never trade weekends, even on
    调休 makeup days), stock_code == filename, prices > 0, low <= open/close <=
    high, volume/amount >= 0. Any violation fails the gate.
    """
    res = CheckResult("ohlc_sanity", True, "")
    files = sorted(glob.glob(str(DAILY_DIR / "*.parquet")))
    if sample:
        files = files[:sample]
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
    """Every daily file must satisfy the frozen DAILY_EQUITY contract (v6 §十六).

    Schema, primary-key uniqueness, date rules and unit sign constraints all
    come from ``stoke_ml.data.contract`` instead of ad-hoc local checks, so the
    gate and the storage/downloader layers share one source of truth.
    """
    res = CheckResult("contract_schema", True, "")
    files = sorted(glob.glob(str(DAILY_DIR / "*.parquet")))
    if sample:
        files = files[:sample]
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
        violations = validate_contract(d, contract, code=code)
        if violations:
            res.passed = False
            res.issues.append((code, ";".join(violations[:8])))
    res.summary = f"problem_files={len(res.issues)}"
    return res


CHECKS = {
    "daily_internal": check_daily_internal,
    "aux_pct_aligned": check_aux_pct_aligned,
    "aux_close_aligned": check_aux_close_aligned,
    "feature_pct": check_feature_pct,
    "sparsity": check_sparsity,
    "ohlc_sanity": check_ohlc_sanity,
    "contract_schema": check_contract_schema,
}


def main():
    ap = argparse.ArgumentParser(description="Data quality gate")
    ap.add_argument("--check", default=None,
                    help="comma-separated checks (default: all)")
    ap.add_argument("--sample", type=int, default=0,
                    help="cap files per check (0 = all)")
    ap.add_argument("--quick", action="store_true",
                    help="shorthand for --sample 300 (CI / post-build gate)")
    ap.add_argument("--output", default="reports",
                    help="report dir (default reports/)")
    args = ap.parse_args()

    names = (args.check.split(",") if args.check else list(CHECKS))
    unknown = [n for n in names if n not in CHECKS]
    if unknown:
        print(f"unknown checks: {unknown}; available: {list(CHECKS)}")
        return 2
    sample = args.sample or (300 if args.quick else 0)

    results = []
    for name in names:
        t0 = time.time()
        r = CHECKS[name](sample)
        dt = time.time() - t0
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name:18s} ({dt:.1f}s) {r.summary}")
        for file, detail in r.issues[:15]:
            print(f"         {file}: {detail}")

    passed = all(r.passed for r in results)
    os.makedirs(args.output, exist_ok=True)
    report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "passed": passed,
        "checks": [
            {
                "name": r.name,
                "passed": r.passed,
                "summary": r.summary,
                "files_scanned": r.files_scanned,
                "rows_scanned": r.rows_scanned,
                "issue_count": len(r.issues),
                "sample_issues": [{"file": f, "detail": d} for f, d in r.issues[:50]],
            }
            for r in results
        ],
    }
    out_path = os.path.join(args.output, "data_quality_gate.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n{'PASS' if passed else 'FAIL'} — wrote {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
