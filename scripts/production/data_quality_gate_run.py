"""CLI / run / report layer of the data quality gate — extracted from
``scripts.production.data_quality_gate`` (§二十一).  ``main()`` is re-exported
by the gate module, so ``gate_mod.main()``, the ``python data_quality_gate.py``
entry and the build_features.py subprocess invocation are all unchanged.

The gate's check functions + ALL mutable state (``DAILY_DIR``, ``MIN_FILES``,
``MAX_STALE_DAYS``, ``_UNIVERSE_REQUEST``, ...) stay in ``data_quality_gate.py``
because the test seam monkeypatches those module globals on the gate MODULE
OBJECT (``monkeypatch.setattr("scripts.production.data_quality_gate.DAILY_DIR",
...)``) and then calls the check functions / ``gate_mod.main()`` directly.
To keep that seam working, ``main()`` reads and mutates the gate's mutable state
through the gate MODULE OBJECT (``_g.X``) at CALL TIME — never
``from ... import DAILY_DIR`` (which would bind at import time and miss the
patch).
"""
import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.data.vintage_policy import VintagePolicy, vintage_report
# NOTE: the gate import sits BELOW ``def main()`` on purpose — importing the
# gate module first would re-trigger the gate module's bottom ``from
# data_quality_gate_run import main`` re-export while this module is still
# partially initialized (``main`` not yet defined), raising ImportError.  By
# defining ``main`` before importing the gate, BOTH ``import
# data_quality_gate`` (which loads this module as a dependency) and ``import
# data_quality_gate_run`` (which loads the gate as a dependency) resolve the
# circular pair cleanly and ``gate_mod.main`` is always the real function.
# (The import itself lives at the BOTTOM of this module, after ``def main()``.)
def main():
    logging.basicConfig(
        level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s"
    )
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
    ap.add_argument("--min-files", type=int, default=_g.MIN_FILES,
                    help="minimum parquet files per required dataset")
    ap.add_argument("--min-stocks", type=int, default=_g.MIN_STOCKS,
                    help="minimum readable stocks per required dataset")
    ap.add_argument("--min-rows", type=int, default=_g.MIN_ROWS,
                    help="minimum total rows per required dataset")
    ap.add_argument("--min-span-days", type=int, default=_g.MIN_SPAN_DAYS,
                    help="minimum earliest→latest span per required dataset")
    ap.add_argument("--max-stale-days", type=int, default=_g.MAX_STALE_DAYS,
                    help="max TRADING days the dataset may lag the most recent "
                         "completed trading day (per the frozen calendar) "
                         "before FAIL — natural days across 春节/国庆 closures "
                         "do not count")
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
                         "research run must clear: span >= 5y, stale <= 4 "
                         "trading days (behind the most recent completed "
                         "session), unreadable = 0, readable stocks >= 98%%")
    ap.add_argument("--vintage-policy", type=str, default="revision-safe",
                    choices=["revision-safe", "allow-revised", "headline-strict"],
                    help="§T2/§T3: vintage-admission policy recorded in the report "
                         "and enforced in formal mode (default: revision-safe; "
                         "the legacy name \"safe-only\" is the pre-T3 alias).")
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
        args.min_span_days = _g.FORMAL_PROFILE["min_span_days"]
        args.max_stale_days = _g.FORMAL_PROFILE["max_stale_days"]
        args.max_unreadable_ratio = _g.FORMAL_PROFILE["max_unreadable_ratio"]
        args.stock_ratio = _g.FORMAL_PROFILE["stock_ratio"]
        # §九-3: refuse data that extends past verified_until — forward-estimate
        # holidays are not exchange fact and must not be validated as such.
        _g.ENFORCE_VERIFIED_UNTIL = True

    _g.MIN_FILES = args.min_files
    _g.MIN_STOCKS = args.min_stocks
    _g.MIN_ROWS = args.min_rows
    _g.MIN_SPAN_DAYS = args.min_span_days
    _g.MAX_STALE_DAYS = args.max_stale_days
    if args.max_unreadable_ratio is not None:
        _g.MAX_UNREADABLE_RATIO = args.max_unreadable_ratio
    if args.stock_ratio is not None:
        _g.FORMAL_STOCK_RATIO = args.stock_ratio
    _g.ALLOW_EMPTY = args.allow_empty
    _g.REQUIRED_DATASETS[:] = [x.strip() for x in args.require.split(",") if x.strip()]
    if args.data_dir:
        root = Path(args.data_dir).resolve()
        _g._DAILY_CACHE.clear()
        # §九: a --data-dir redirect must re-resolve the calendar + trading-day
        # caches for the NEW root, never reuse the previous root's entries.
        _g._CALENDAR_CACHE.clear()
        _g._TRADING_CACHE.clear()
        _g.A_SHARES = root / "a_shares"
        _g.DAILY_DIR = _g.A_SHARES / "daily"
        _g.FEAT_DIR = root / "features"

    # §P1-7/§八-2: build the optional requested-universe reconciliation.  The
    # check only joins the run when a universe source is supplied (additive);
    # without one, the run is identical to before.  A missing --request-manifest
    # (or one that resolves to no usable codes) FAILS cleanly instead of
    # tracebacking — §八-2 formal mode refuses to silently resolve to whatever
    # happens to be on disk.
    try:
        _g._UNIVERSE_REQUEST = _g._build_universe_request(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"requested-universe ERROR: {exc}", file=sys.stderr)
        return 1

    names = (args.check.split(",") if args.check else list(_g.CHECKS))
    if _g._UNIVERSE_REQUEST is not None and "universe" not in names:
        names.append("universe")
    available = set(_g.CHECKS) | {"universe"}
    unknown = [n for n in names if n not in available]
    if unknown:
        print(f"unknown checks: {unknown}; available: {sorted(available)}")
        return 2
    sample = args.sample or (300 if args.quick else 0)

    results = []
    for name in names:
        t0 = time.time()
        r = _g.RUN_CHECKS[name](sample)
        dt = time.time() - t0
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name:18s} ({dt:.1f}s) {r.summary}")
        for file, detail in r.issues[:15]:
            print(f"         {file}: {detail}")

    passed = all(r.passed for r in results)
    # §九: formal mode refuses to silently fall back to code holiday rules when
    # the frozen calendar artifact is absent — the gate must validate the SAME
    # calendar the feature pipeline reads from this data root.  All checks still
    # run so the report stays informative; the run is failed regardless.
    cal_status = _g._calendar_status()
    if args.profile == "formal" and not cal_status["present"]:
        passed = False
        print(
            f"ERROR: --profile formal requires the frozen calendar artifact at "
            f"{cal_status['path']} ({cal_status['reason']}) — refusing to "
            f"silently fall back to code holiday rules. Run save_calendar() for "
            f"this data root first.",
            file=sys.stderr,
        )
    # §T2/§十五: the run's vintage-admission policy + per-channel allowed flags.
    # Computed ONCE so the report and the formal enforcement share the same
    # view.  Informational for bootstrap; formal mode ENFORCES it below.
    vintage = vintage_report(VintagePolicy(args.vintage_policy))
    # §T2/§T7: formal mode refuses an incomplete or self-contradictory 3-dim
    # vintage declaration.  An undeclared documented channel (denied by
    # default) or a declaration whose source_vintage/transform/pit_alignment is
    # outside the known vocabularies / the reserved "unknown" fallback must
    # surface as a hard FAIL rather than let a consumer guess.  daily_qfq must
    # always be admissible (no model trains without the price channel).
    if args.profile == "formal":
        if vintage["missing_channels"]:
            passed = False
            print(
                f"ERROR: --profile formal requires every documented use_* channel "
                f"to carry a vintage declaration; missing: "
                f"{vintage['missing_channels']} — refusing an incomplete vintage "
                f"declaration.",
                file=sys.stderr,
            )
        elif not vintage["declaration_complete"]:
            # Missing_channels forces declaration_complete=False by construction,
            # so this elif fires ONLY for the declared-but-incomplete case: every
            # documented channel declared, but one carries an out-of-vocabulary
            # or "unknown" dim.
            passed = False
            print(
                "ERROR: --profile formal requires a COMPLETE 3-dim vintage "
                "declaration (every declared channel must declare "
                "source_vintage/transform/pit_alignment within the known "
                "vocabularies, none 'unknown'); refusing an incomplete "
                "declaration.",
                file=sys.stderr,
            )
        if not vintage["daily_qfq_allowed"]:
            passed = False
            print(
                f"ERROR: --profile formal vintage policy {args.vintage_policy} "
                f"denies daily_qfq — a model cannot train without the price "
                f"channel.",
                file=sys.stderr,
            )
    os.makedirs(args.output, exist_ok=True)
    # §七.2: record the run's audit scope so a consumer can tell a full scan
    # from a --quick sample.  manifest/contract_schema are always full-scan
    # (see their docstrings), so formal training can accept a sampled run only
    # when those two really covered every file — the reviewer's "at least full
    # manifest/contract + sampled deep feature audit" floor.  A consumer reads
    # manifest_contract_full_scan to prove that half before trusting a sample.
    total_daily = len(glob.glob(str(_g.DAILY_DIR / "*.parquet")))
    # v14 §八-1: manifest_contract_full_scan is true only when BOTH the manifest
    # and contract_schema checks actually ran, both passed, both covered every
    # daily file and neither reported an unreadable file.  A `--check manifest`
    #-only run leaves contract_schema unproven and must NOT satisfy the floor.
    manifest_contract_full_scan = _g._manifest_contract_full_scan(results, total_daily)
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
        "quality_gate_version": _g.QUALITY_GATE_VERSION,
        "data_root": str(_g.A_SHARES.parent),
        "calendar_version": TradingCalendar.CALENDAR_VERSION,
        # §九: the report binds the ACTUAL frozen artifact of the validated data
        # root (content hash, not a version string) so a consuming run can verify
        # the gate reviewed the same calendar the feature pipeline reads.
        "calendar_artifact_hash": cal_status["hash"],
        "calendar_artifact_path": cal_status["path"],
        "calendar_artifact_present": cal_status["present"],
        "contract_version": _g.contract_version(),
        "required_datasets": list(_g.REQUIRED_DATASETS),
        "dataset_paths": {
            name: str(_g._dataset_dir(name).resolve())
            for name in _g.REQUIRED_DATASETS
        },
        "profile": args.profile,
        "scope": "full" if sample == 0 else "sample",
        "sample_size": sample,
        "sample_seed": _g.SAMPLE_SEED,
        "scanned_files": scanned_files,   # files actually row-read (§九.2)
        "total_files": total_files,       # true on-disk parquet count (§九.2)
        "manifest_contract_full_scan": manifest_contract_full_scan,
        "data_manifest_hash": _g.dataset_fingerprint(_g.A_SHARES.parent, _g.REQUIRED_DATASETS),
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
    # §T2/§十五: the report carries the run's vintage-admission policy and marks
    # every declared channel as allowed/denied under it.  Informational for
    # bootstrap; formal mode ENFORCES the declaration above.
    report["vintage_policy"] = vintage["vintage_policy"]
    report["channel_vintage"] = vintage["channels"]
    out_path = os.path.join(args.output, "data_quality_gate.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"\n{'PASS' if passed else 'FAIL'} — wrote {out_path}")
    return 0 if passed else 1


from scripts.production import data_quality_gate as _g  # noqa: E402 — see NOTE above (circular-import-safe)<｜end▁of▁thinking｜>
