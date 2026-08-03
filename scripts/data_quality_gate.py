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
  sparsity         : per-feature non-zero coverage — flags event-sparse columns

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


def _load_daily(code: str, cols: list[str]) -> pd.DataFrame | None:
    p = DAILY_DIR / f"{code}.parquet"
    if not p.exists():
        return None
    try:
        d = pd.read_parquet(p, columns=cols)
    except Exception:
        return None
    d["date"] = pd.to_datetime(d["date"])
    return d.drop_duplicates("date", keep="last")


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
            continue
        try:
            a = pd.read_parquet(fp, columns=["date", "pct_change"])
        except Exception:
            res.issues.append((f"{d}/{code}", "read_err"))
            continue
        if "pct_change" not in a:
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
            continue  # file lacks OHLC entirely — not in scope
        if "close" not in a or "date" not in a:
            continue
        daily = _load_daily(code, ["date"] + OHLC)
        if daily is None:
            res.issues.append((f"{d}/{code}", "no_daily"))
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
            continue
        try:
            f = pd.read_parquet(fp, columns=["date", "pct_change"])
        except Exception:
            res.issues.append((code, "read_err"))
            continue
        if "pct_change" not in f:
            res.issues.append((code, "no_feat_pc"))
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
    """Per-feature non-zero coverage across a sampled panel (event-sparse canary)."""
    res = CheckResult("sparsity", True, "")
    feats = sorted(glob.glob(str(FEAT_DIR / "*.parquet")))
    if sample:
        feats = feats[:sample]
    res.files_scanned = len(feats)
    counts: dict[str, list] = {}
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
            counts.setdefault(c, []).append(float((x != 0).mean()))
    sparse = []
    for c, vals in counts.items():
        nz = float(np.mean(vals))
        if nz < SPARSE_NONZERO_RATIO:
            sparse.append((c, round(nz, 5)))
    sparse.sort(key=lambda t: t[1])
    if len(sparse) > 200:
        res.issues = [(f"{c}", f"nz={nz}") for c, nz in sparse[:200]]
        res.summary = f"event-sparse features={len(sparse)} (showing 200) min_nz={sparse[0][1] if sparse else 0}"
    else:
        res.issues = [(f"{c}", f"nz={nz}") for c, nz in sparse]
        res.summary = f"event-sparse features={len(sparse)} (non_zero_ratio<{SPARSE_NONZERO_RATIO})"
    # Sparsity is informational — never fails the gate by itself.
    return res


CHECKS = {
    "daily_internal": check_daily_internal,
    "aux_pct_aligned": check_aux_pct_aligned,
    "aux_close_aligned": check_aux_close_aligned,
    "feature_pct": check_feature_pct,
    "sparsity": check_sparsity,
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
