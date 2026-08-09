"""qfq provenance stamp (#85 Phase 2, ADR-046).

Phase 1 normalized the 手/股 volume-unit mix.  Phase 2 declares the price
adjustment basis that the download chain (efinance / akshare / baostock, all
qfq by default) actually produced: every unknown-adjust file whose data
passes a per-file integrity gate is re-declared ``adjust=qfq`` in both the
parquet attrs and the manifest, so formal reads (``require_valid_manifest=
True``) unblock.

Per-file safety gate — a file is stamped ONLY when ALL hold:

  * unit-clean: :func:`migrate_daily_units.classify_file` reports flips==0 AND
    junk==0 (no residual 手 row, no corrupt-year residue).  A file still
    carrying a hidden unit defect must NOT be declared ``units=volume=shares``.
  * qfq signature — per-year median of ``amount/volume/close`` (on the
    unit-clean frame all rows are 股):
      - latest-year median in [0.4, 2.5] (a per-share current price has
        VWAP ~ close; qfq factor is 1 at the present).
      - no TEMPORARY forward spike: an older year's median below the next
        newer year's median / 2 AND that newer year's median more than 1.3x
        the max of the following 3 years.  qfq factors only push old ratios
        UP, so a spike that then collapses is the signature of a volume/amount
        anomaly classify_file's year-median walk (SWITCH=15) missed.  A
        *permanent* forward level shift (older ratios uniformly low, then ~1
        and staying ~1) is the reverse-split (缩股) qfq signature and is NOT
        blocked — it is recorded as a non-blocking ``level_shift`` flag.
  * content invariant: re-serializing the frame must not change its content
    checksum (the stamp is a pure provenance relabel, zero value change).

Files failing the gate keep ``adjust=unknown`` (fail-closed: formal reads
still refuse them) and are listed in the report for human review.

Write is DESTRUCTIVE (rewrites canonical parquet + manifest in place).
``--write`` backs up the in-scope files first.  A full formal re-read is
deferred to the post-write verification step, which runs
``validate_manifest`` on every written file and a formal
``load_daily(require_valid_manifest=True)`` on a sample.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
import uuid

import numpy as np
import pandas as pd

from stoke_ml.data.storage import (DataStorage, _write_manifest,
                                   _content_checksum, _units_tag,
                                   _CALENDAR_VERSION,
                                   _DATA_CONVENTION_VERSION)
from scripts.maintenance.current.migrate_daily_units import classify_file

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DAILY = os.path.join(ROOT, "data", "a_shares", "daily")

LATEST_BAND = (0.4, 2.5)   # latest-year median ratio must sit near 1
MONO_TOL = 2.0             # older med below newer med / 2 => forward increase
SPIKE_DECAY = 1.3          # newer med > following-3yr max * 1.3 => temp spike


def qfq_signature(df: pd.DataFrame) -> dict:
    """Per-year median ``amount/volume/close`` checks.  Returns ``problems``
    (blocking) and ``flags`` (informational)."""
    vol = df["volume"].astype("float64").to_numpy()
    amt = df["amount"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()
    ok = np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close)
    ok &= (vol > 0) & (amt > 0) & (close > 0)
    problems: list[str] = []
    flags: list[str] = []
    if not ok.any():
        problems.append("no_valid_rows")
        return {"problems": problems, "flags": flags}
    ratio = amt[ok] / vol[ok] / close[ok]
    years = pd.to_datetime(df.loc[ok, "date"]).dt.year.to_numpy()
    s = pd.Series(ratio, index=years)
    med = s.groupby(level=0).median()
    cnt = s.groupby(level=0).count()
    use = cnt.to_numpy() >= 5
    ys = med.index[use].to_numpy()
    ms = med.to_numpy()[use]
    if len(ms) == 0:
        latest = float(np.median(ratio))
        if not (LATEST_BAND[0] <= latest <= LATEST_BAND[1]):
            problems.append(f"latest_med={latest:.2f}")
        return {"problems": problems, "flags": flags}
    latest = float(ms[-1])
    if not (LATEST_BAND[0] <= latest <= LATEST_BAND[1]):
        problems.append(f"latest_med={latest:.2f}")
    for i in range(len(ms) - 1):
        if ms[i] < ms[i + 1] / MONO_TOL:  # forward increase (anti-qfq / rev-split)
            hi = float(ms[i + 1])
            fut = ms[i + 2: i + 2 + 3]
            if not len(fut):
                # rise at the very end: the latest-band check covers a large one
                if hi > ms[i] * 3.0:
                    problems.append(
                        f"end_spike:{int(ys[i])}:{ms[i]:.1f}->{int(ys[i+1])}:{hi:.1f}")
                else:
                    flags.append(
                        f"end_rise:{int(ys[i])}:{ms[i]:.2f}->{int(ys[i+1])}:{hi:.2f}")
            elif hi > max(fut) * SPIKE_DECAY:
                problems.append(
                    f"temp_spike:{int(ys[i])}:{ms[i]:.1f}->{int(ys[i+1])}:{hi:.1f}")
            else:
                flags.append(
                    f"level_shift:{int(ys[i])}:{ms[i]:.2f}->{int(ys[i+1])}:{hi:.2f}")
    return {"problems": problems, "flags": flags}


def stamp_one(df: pd.DataFrame, code: str, manifest: dict) -> tuple[str, dict]:
    """Re-declare one file qfq.  Returns (route, detail).  Routes:
    ``clean`` (stamped), ``skip`` (gate failed — reason in detail), or the
    exception message on failure (``failed`` handled by the caller)."""
    res = classify_file(df)
    flips = int(res["flip"].sum())
    junks = int(res["junk"].sum())
    if flips or junks:
        return "skip", {"reason": f"unit_flips={flips}_junk={junks}",
                        "flags": res["flags"]}
    sig = qfq_signature(df)
    if sig["problems"]:
        return "skip", {"reason": ";".join(sig["problems"]),
                        "flags": sig["flags"]}
    return _stamp_write(df, code, manifest, sig["flags"])


def _stamp_write(df: pd.DataFrame, code: str, manifest: dict,
                 flags: list[str]) -> tuple[str, dict]:
    pre = pd.read_parquet(os.path.join(DAILY, f"{code}.parquet"))
    cols = sorted(map(str, pre.columns))
    pre_content = _content_checksum(pre, cols)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df.attrs["source"] = manifest.get("source", "unknown")
    df.attrs["adjustment_mode"] = "qfq"
    df.attrs["units"] = _units_tag()
    df.attrs["calendar_version"] = _CALENDAR_VERSION
    df.attrs["data_convention_version"] = _DATA_CONVENTION_VERSION
    out_path = os.path.join(DAILY, f"{code}.parquet")
    tmp = f"{out_path}.tmp.{os.getpid()}"
    df.to_parquet(tmp, index=False, compression="lz4")
    back = pd.read_parquet(tmp)
    back_cols = sorted(map(str, back.columns))
    back_content = _content_checksum(back, back_cols)
    if back_cols != cols or back_content != pre_content:
        os.remove(tmp)
        return "skip", {"reason": "content_changed",
                        "flags": flags + ["content_checksum_mismatch"]}
    os.replace(tmp, out_path)
    back["date"] = pd.to_datetime(back["date"])
    segs = manifest.get("source_segments")
    if not segs:
        segs = [{
            "source": manifest.get("source", "unknown"), "adjust": "unknown",
            "start": (back["date"].min().strftime("%Y-%m-%d") if len(back) else None),
            "end": (back["date"].max().strftime("%Y-%m-%d") if len(back) else None),
            "rows": int(len(back)),
        }]
    qfq_segs = []
    for s in segs:
        o = dict(s)
        o["adjust"] = "qfq"
        qfq_segs.append(o)
    _write_manifest(DAILY, code, back, qfq_segs, run_id=uuid.uuid4().hex)
    return "clean", {"flags": flags}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="actually stamp the canonical store (backup first). "
                         "Default is a read-only dry-run audit.")
    ap.add_argument("--stocks", default=None,
                    help="comma-separated stock codes to restrict to "
                         "(validation).  Default: whole universe.")
    ap.add_argument("--backup-dir", default=None,
                    help="backup destination for --write (default: "
                         "data/a_shares/daily_stamp_backup_YYYYMMDD_HHMMSS).")
    ap.add_argument("--report", default=None,
                    help="JSON report path (default: reports/stamp_qfq.json).")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(DAILY, "*.parquet")))
    if args.stocks:
        want = {c.strip() for c in args.stocks.split(",") if c.strip()}
        files = [f for f in files
                 if os.path.basename(f)[: -len(".parquet")] in want]
    if not files:
        print("no daily files to process", file=sys.stderr)
        sys.exit(1)

    backup_dir = None
    if args.write:
        backup_dir = (args.backup_dir or os.path.join(
            ROOT, "data", "a_shares",
            f"daily_stamp_backup_{time.strftime('%Y%m%d_%H%M%S')}"))
        os.makedirs(backup_dir, exist_ok=True)
        n_backed = 0
        for f in files:
            base = os.path.basename(f)
            shutil.copy2(f, os.path.join(backup_dir, base))
            mf = f"{f[:-len('.parquet')]}.manifest.json"
            if os.path.isfile(mf):
                shutil.copy2(mf, os.path.join(backup_dir,
                                              os.path.basename(mf)))
            n_backed += 1
        print(f"backup -> {backup_dir} ({n_backed} files)", flush=True)

    print(f"{'WRITE' if args.write else 'DRY-RUN'} {len(files)} files "
          f"({DAILY})", flush=True)
    t0 = time.time()
    n_already = 0
    n_pass = 0
    n_skip = 0
    n_failed = 0
    written: list[str] = []
    skipped: list[dict] = []
    failed: list[dict] = []
    post_verify_fail: list[dict] = []
    storage = DataStorage(os.path.join(ROOT, "data"))

    for path in files:
        code = os.path.basename(path)[: -len(".parquet")]
        mf = os.path.join(DAILY, f"{code}.manifest.json")
        manifest = (json.load(open(mf, encoding="utf-8"))
                    if os.path.isfile(mf) else {})
        if manifest.get("adjust") == "qfq":
            n_already += 1
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        if not args.write:
            res = classify_file(df)
            flips = int(res["flip"].sum())
            junks = int(res["junk"].sum())
            sig = qfq_signature(df)
            route = "pass" if (flips == 0 and junks == 0
                               and not sig["problems"]) else "skip"
            if route == "pass":
                n_pass += 1
            else:
                n_skip += 1
                skipped.append({"code": code, "reason": (
                    f"unit_flips={flips}_junk={junks}"
                    if (flips or junks) else ";".join(sig["problems"]))})
            if len(files) <= 20 or code in {"000001", "000725", "600795",
                                            "600519", "000155", "600381",
                                            "000338", "000520"}:
                print(f"  {code}: {route} flips={flips} junk={junks} "
                      f"flags={res['flags']} sig_problems={sig['problems']}",
                      flush=True)
            continue

        try:
            route, detail = stamp_one(df, code, manifest)
        except Exception as exc:  # noqa: BLE001 - per-file, keep going
            failed.append({"code": code, "error": str(exc)})
            n_failed += 1
            print(f"  {code}: FAILED — {exc}", flush=True)
            continue
        if route == "skip":
            n_skip += 1
            skipped.append({"code": code, "reason": detail["reason"],
                            "flags": detail.get("flags", [])})
            continue
        written.append(code)
        n_pass += 1
        rep = storage.validate_manifest(code)
        if not rep.get("ok"):
            post_verify_fail.append({
                "code": code,
                "reason": rep.get("reason") or rep.get("mismatches"),
            })
        if len(written) % 500 == 0 and written:
            print(f"  stamped {len(written)} files ({time.time()-t0:.0f}s)",
                  flush=True)

    if args.write and written:
        print(f"post-write validate_manifest: {len(written)} written, "
              f"{len(post_verify_fail)} failures", flush=True)
        for f in post_verify_fail[:20]:
            print(f"    {f['code']}: {f['reason']}", flush=True)

    out = {
        "mode": "write" if args.write else "dry-run",
        "backup_dir": backup_dir,
        "n_files": len(files),
        "n_already_qfq": n_already,
        "n_pass": n_pass,
        "n_skip": n_skip,
        "n_failed": n_failed,
        "skipped": skipped,
        "failed": failed,
        "post_write_validate_failures": post_verify_fail,
    }
    report = args.report or os.path.join(ROOT, "reports", "stamp_qfq.json")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nn_files={len(files)} already_qfq={n_already} pass={n_pass} "
          f"skip={n_skip} failed={n_failed}  ({time.time()-t0:.0f}s)",
          flush=True)
    print(f"report: {report}", flush=True)
    if skipped:
        print("skipped (kept unknown — fail-closed):", flush=True)
        for s in skipped[:30]:
            print(f"  {s['code']}: {s['reason']}", flush=True)
    if args.write and n_failed:
        print("WRITE COMPLETED WITH ERRORS — see report 'failed'.", flush=True)


if __name__ == "__main__":
    main()
