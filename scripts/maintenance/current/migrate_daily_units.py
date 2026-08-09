"""手/股 mixed-unit daily-store migration (§ADR-046, #85 Phase 1).

The canonical daily store's ``volume`` column mixes 手 (lots) and 股
(shares) rows because the download failover chain swapped sources mid-
history.  Manifests uniformly declare ``units=volume=shares`` — TRUE only
after every 手 row is normalized ×100.

Classifier (validated in probes against 000001 / 000725 / 600795 / 600519):

  ratio = amount / volume / close   (per valid row, vol/amt/close>0, finite)

  * per-year median ratio (years with >= 5 valid rows), unit from a
    backward walk: r = newer_med / older_med >= SWITCH(15)  -> older is 股
    (GU); r <= 1/SWITCH -> older is 手 (HU).  Base from the LAST year's
    median: < AMBIGUOUS_LO(40) -> GU, > AMBIGUOUS_HI(60) -> HU, else GU
    + ambiguous flag.
  * row-level: within a GU year, ratio/year_med in [STRAY, STRAY_HI] =
    [30, 500] is a stray 手 row (×100); ratio/year_med > 500 is corrupt
    (junk — NOT ×100, flagged for separate review).  Within a HU year,
    ratio/year_med in [1/STRAY, 30] is 手 (×100); < 1/STRAY_HI is junk.

  The 100× separation between the two clusters is exact (both scale with
  the same qfq factor), so the relative bands are robust across the whole
  universe even though an absolute threshold is NOT (600795's 股 rows reach
  ratio 51.9; 手 rows start ~85).

Junk rows (000858 ratio 293k, etc.) are data corruption, not unit errors:
the migration preserves them unchanged and lists them for review.  A file
that still contains junk rows will STILL fail the formal
``amount_volume_unit_mismatch`` band after migration — this is surfaced in
the report, not silently hidden.

Write is DESTRUCTIVE (rewrites canonical parquet + manifest in place).
``--write`` backs up the in-scope files first, routes clean files through
the sanctioned ``DataStorage.save_daily_repair`` seam (preserves
provenance, formal=False), uses a direct write for files carrying junk rows
(save_daily_repair would refuse them) and for files save_daily_repair
refuses on a pre-existing contract violation (e.g. >1% NaN volume/amount),
and afterwards verifies every migrated file's manifest schema_hash against
the bytes on disk.  A
full formal re-read is deferred to Phase 2: until the qfq provenance
migration an ``unknown`` adjustment basis refuses formal reads anyway.
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
                                   _units_tag, _CALENDAR_VERSION,
                                   _DATA_CONVENTION_VERSION)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
DAILY = os.path.join(ROOT, "data", "a_shares", "daily")

# --- classifier constants (validated in diagnostics probes) --------------
SWITCH = 15.0        # year-median jump: newer/older >= SWITCH -> older is 股
STRAY = 30.0         # row vs year-median beyond this many x = the other unit
STRAY_HI = 500.0     # beyond this many x of the median = corrupt, not 手
GU, HU = 1.0, 100.0  # canonical ratio centers (股 ~1, 手 ~100 at qfq~1)
AMBIGUOUS_LO, AMBIGUOUS_HI = 40.0, 60.0   # last-year median ambiguous band
# A year whose MEDIAN ratio is outside [GU_MED_MIN, HU_MED_MAX] is corrupt —
# no unit can explain it.  股 medians sit in [0.3, 84] (600795 reaches 51.9),
# 手 medians in [30, ~15000] (observed cluster max).  The bounds are generous
# so legit qfq extremes (000155-2017 med 53.3, 000830-2016 med 5035) are NOT
# flagged, while true corruption spikes (000408-2020 med 44901, 000858-2015
# med 249992, 000651/000568-2015 med ~200k) are.  A flagged year's rows are
# left untouched (never ×100) and reported for manual review.
GU_MED_MIN, HU_MED_MAX = 0.05, 25000.0


def classify_file(df: pd.DataFrame) -> dict:
    """Per-row unit classification.  Returns flip/junk masks + metadata.

    ``flip`` / ``junk`` are boolean arrays ALIGNED TO THE FULL FRAME (same
    length as ``df``).  ``flip[i]`` == True -> row i is a 手 row and must be
    volume ×100.  ``junk[i]`` == True -> row i is corrupt (ratio implausible
    for BOTH units); the migration leaves it unchanged and reports it.
    """
    vol = df["volume"].astype("float64").to_numpy()
    amt = df["amount"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()
    ok = np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close)
    ok &= (vol > 0) & (amt > 0) & (close > 0)
    n = len(df)
    flip = np.zeros(n, dtype=bool)
    junk = np.zeros(n, dtype=bool)
    res = {"n_valid": int(ok.sum()), "flip": flip, "junk": junk,
           "flags": []}
    if not ok.any():
        res["flags"].append("no_valid_rows")
        return res
    ok_idx = np.where(ok)[0]
    ratio = amt[ok] / vol[ok] / close[ok]
    years = pd.to_datetime(df.loc[ok, "date"]).dt.year.to_numpy()
    s = pd.Series(ratio, index=years)
    med = s.groupby(level=0).median()
    cnt = s.groupby(level=0).count()
    years_all = med.index.to_numpy()
    use = cnt.to_numpy() >= 5
    years_u = years_all[use]
    med_u = med.to_numpy()[use]
    if len(med_u) == 0:
        # no year has >=5 rows: classify the whole file by its overall median
        over = float(np.median(ratio))
        if over < AMBIGUOUS_LO:
            year_unit = {int(y): GU for y in np.unique(years)}
        elif over > AMBIGUOUS_HI:
            year_unit = {int(y): HU for y in np.unique(years)}
        else:
            res["flags"].append(f"overall_median_ambiguous:{over:.2f}")
            return res  # cannot establish the unit — safe default: no ×100
        year_med = {int(y): float(np.median(ratio[years == y]))
                    for y in np.unique(years)}
        return _apply_row_rules(ratio, years, ok_idx, year_med, year_unit,
                                flip, junk, res)

    # backward walk: base from last-year median, then older years
    n_u = len(med_u)
    unit = np.full(n_u, -1.0)
    last_med = med_u[-1]
    if last_med < AMBIGUOUS_LO:
        base = GU
    elif last_med > AMBIGUOUS_HI:
        base = HU
    else:
        base = GU
        res["flags"].append(f"last_year_median_ambiguous:{last_med:.2f}")
    unit[-1] = base
    for i in range(n_u - 2, -1, -1):
        r = med_u[i + 1] / med_u[i] if med_u[i] > 0 else np.inf
        if r >= SWITCH:
            unit[i] = GU
        elif r <= 1.0 / SWITCH:
            unit[i] = HU
        else:
            unit[i] = unit[i + 1]
    year_unit = {int(y): u for y, u in zip(years_u.tolist(), unit.tolist())}
    # corrupt-year detection: a year whose median ratio is outside the
    # [GU_MED_MIN, HU_MED_MAX] plausibility band has no explainable unit — its
    # rows must not be auto-migrated (the year median is unusable as a
    # baseline too).  Flagged years are reported for manual review.
    corrupt_years: set[int] = set()
    for y, m in zip(years_u.tolist(), med_u.tolist()):
        if m < GU_MED_MIN or m > HU_MED_MAX:
            corrupt_years.add(int(y))
            res["flags"].append(f"corrupt_year:{int(y)}:{m:.1f}")
    # sparse years (<5 rows, not in the walk) inherit the nearest classified
    # year's unit so their rows can still be classified
    for y in np.unique(years):
        if int(y) not in year_unit:
            year_unit[int(y)] = _nearest_unit(y, years_u, unit)
    year_med = {int(y): float(m)
                for y, m in zip(years_all.tolist(), med.to_numpy().tolist())}
    return _apply_row_rules(ratio, years, ok_idx, year_med, year_unit,
                            flip, junk, res, corrupt_years)


def _nearest_unit(y: int, years_u: np.ndarray, unit: np.ndarray) -> float:
    d = np.abs(years_u - y)
    return float(unit[int(np.argmin(d))])


def _apply_row_rules(ratio, years, ok_idx, year_med, year_unit, flip, junk,
                     res, corrupt_years=None) -> dict:
    corrupt_years = corrupt_years or set()
    n_flip = 0
    n_junk = 0
    for i in range(len(ratio)):
        full_i = ok_idx[i]
        y = int(years[i])
        if y in corrupt_years:
            junk[full_i] = True
            n_junk += 1
            continue
        ym = year_med.get(y)
        u = year_unit.get(y)
        if ym is None or u is None:
            continue
        r = ratio[i] / ym
        if u == GU:
            if r >= STRAY:
                if r <= STRAY_HI:
                    flip[full_i] = True
                    n_flip += 1
                else:
                    junk[full_i] = True
                    n_junk += 1
            elif r < 1.0 / STRAY_HI:
                junk[full_i] = True
                n_junk += 1
        else:  # HU year — the whole year is 手, except 股 strays / junk
            if r <= STRAY and r >= 1.0 / STRAY:
                flip[full_i] = True
                n_flip += 1
            elif r < 1.0 / STRAY_HI or r > STRAY:
                junk[full_i] = True
                n_junk += 1
    if n_flip:
        res["flags"].append(f"flips:{n_flip}")
    if n_junk:
        res["flags"].append(f"junk:{n_junk}")
    return res


def rows_out_of_band_after(df: pd.DataFrame, flip: np.ndarray) -> int:
    """Rows whose ratio stays outside the formal [0.01, 100] implied-price
    band even AFTER applying the ×100 flip — the amount_volume_unit_mismatch
    residue (junk rows).  Mirrors contract.py's band check."""
    vol = df["volume"].astype("float64").to_numpy().copy()
    amt = df["amount"].astype("float64").to_numpy()
    close = df["close"].astype("float64").to_numpy()
    ok = np.isfinite(vol) & np.isfinite(amt) & np.isfinite(close)
    ok &= (vol > 0) & (amt > 0) & (close > 0)
    if not ok.any():
        return 0
    vol[flip] *= 100.0
    implied = amt[ok] / vol[ok]
    c = close[ok]
    return int(((implied < c / 100.0) | (implied > c * 100.0)).sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="actually rewrite the canonical store (backup first). "
                         "Default is a read-only dry-run audit.")
    ap.add_argument("--stocks", default=None,
                    help="comma-separated stock codes to restrict to "
                         "(validation).  Default: whole universe.")
    ap.add_argument("--backup-dir", default=None,
                    help="backup destination for --write (default: "
                         "data/a_shares/daily_backup_YYYYMMDD_HHMMSS).")
    ap.add_argument("--report", default=None,
                    help="JSON report path (default: "
                         "reports/daily_unit_migration.json).")
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
            f"daily_backup_{time.strftime('%Y%m%d_%H%M%S')}"))
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
    total_rows = 0
    total_flip = 0
    total_junk = 0
    n_with_flip = 0
    n_with_junk = 0
    n_skipped = 0
    per_file: dict[str, dict] = {}
    written: list[str] = []
    failed: list[dict] = []
    still_out_of_band: list[dict] = []

    for path in files:
        code = os.path.basename(path)[: -len(".parquet")]
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        res = classify_file(df)
        flips = int(res["flip"].sum())
        junks = int(res["junk"].sum())
        total_rows += len(df)
        total_flip += flips
        total_junk += junks
        if flips:
            n_with_flip += 1
        if junks:
            n_with_junk += 1
        oob = rows_out_of_band_after(df, res["flip"]) if (flips or junks) else 0
        per_file[code] = {
            "rows": int(len(df)),
            "flips": flips,
            "junk": junks,
            "out_of_band_after": oob,
            "flags": res["flags"],
        }
        if flips == 0:
            n_skipped += 1
        if not args.write:
            if len(files) <= 20 or code in {"000001", "000725", "600795",
                                            "600519", "000858", "301487"}:
                print(f"  {code}: rows={len(df)} flips={flips} junk={junks} "
                      f"oob={oob} flags={res['flags']}", flush=True)
            continue

        # ---- WRITE path --------------------------------------------------
        if flips == 0:
            per_file[code]["skipped"] = "no_flips"
            continue
        try:
            route = _migrate_one(df, code, res["flip"], oob)
            written.append(code)
            per_file[code]["write_route"] = route
            if oob:
                still_out_of_band.append(
                    {"code": code, "out_of_band_after": oob})
        except Exception as exc:  # noqa: BLE001 - surface per-file, keep going
            failed.append({"code": code, "error": str(exc)})
            print(f"  {code}: FAILED — {exc}", flush=True)
        if len(written) % 500 == 0 and len(written):
            print(f"  wrote {len(written)} files ({time.time()-t0:.0f}s)",
                  flush=True)

    manifest_failures: list[dict] = []
    if args.write and written:
        # Post-write verification: the manifest schema_hash must match the bytes
        # actually on disk.  A full formal re-read is deferred to Phase 2 (the
        # unknown adjustment basis would refuse it until the qfq provenance
        # migration), so this is the contract check that is valid right now.
        from stoke_ml.data.storage import DataStorage
        storage = DataStorage(os.path.join(ROOT, "data"))
        for code in written:
            rep = storage.validate_manifest(code)
            if not rep.get("ok"):
                manifest_failures.append({
                    "code": code,
                    "reason": rep.get("reason") or rep.get("mismatches"),
                })
        if manifest_failures:
            print(f"  POST-WRITE MANIFEST FAILURES: "
                  f"{len(manifest_failures)} / {len(written)}", flush=True)
            for f in manifest_failures[:20]:
                print(f"    {f['code']}: {f['reason']}", flush=True)
        else:
            print(f"  post-write manifest check: {len(written)} files OK",
                  flush=True)

    out = {
        "mode": "write" if args.write else "dry-run",
        "backup_dir": backup_dir,
        "n_files": len(files),
        "n_rows": total_rows,
        "n_flip_rows": total_flip,
        "n_junk_rows": total_junk,
        "n_files_with_flips": n_with_flip,
        "n_files_with_junk": n_with_junk,
        "n_files_skipped_no_flip": n_skipped,
        "n_files_written": len(written),
        "n_files_failed": len(failed),
        "manifest_failures": manifest_failures,
        "failed": failed,
        "still_out_of_band_after_write": still_out_of_band,
        "per_file": per_file,
    }
    report = args.report or os.path.join(ROOT, "reports",
                                         "daily_unit_migration.json")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nrows={total_rows} flips={total_flip} junk={total_junk} "
          f"files_with_flips={n_with_flip} files_with_junk={n_with_junk} "
          f"skipped_no_flip={n_skipped} written={len(written)} "
          f"failed={len(failed)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"report: {report}", flush=True)
    if total_junk:
        print("files with junk rows (still fail formal reads until reviewed):",
              flush=True)
        for code, info in per_file.items():
            if info["junk"]:
                print(f"  {code}: junk={info['junk']} "
                      f"oob={info['out_of_band_after']} "
                      f"flags={info['flags']}", flush=True)
    if args.write and failed:
        print("WRITE COMPLETED WITH ERRORS — see report 'failed'.", flush=True)


def _direct_write(df: pd.DataFrame, code: str) -> str:
    """Hash-and-write a migrated frame directly, bypassing save_daily's
    contract gate.  Used for files carrying junk / corrupt rows (the
    amount_volume_unit_mismatch band would refuse them) and as a fallback
    when save_daily_repair refuses a frame on a pre-existing contract
    violation (e.g. >1% NaN volume/amount — the finite-ratio requirement).
    The manifest is rewritten on the SAME (date-sorted) frame so its
    schema_hash matches the written bytes; provenance is carried forward
    from the existing manifest, so an honest unknown basis stays unknown."""
    df = df.sort_values("date").reset_index(drop=True)
    storage = DataStorage(os.path.join(ROOT, "data"))
    manifest = storage.manifest(code) or {}
    df.attrs["source"] = manifest.get("source", "unknown")
    df.attrs["adjustment_mode"] = manifest.get("adjust", "unknown")
    df.attrs["units"] = _units_tag()
    df.attrs["calendar_version"] = _CALENDAR_VERSION
    df.attrs["data_convention_version"] = _DATA_CONVENTION_VERSION
    base_dir = os.path.join(ROOT, "data", "a_shares", "daily")
    out_path = os.path.join(base_dir, f"{code}.parquet")
    tmp = f"{out_path}.tmp.{os.getpid()}"
    df.to_parquet(tmp, index=False, compression="lz4")
    os.replace(tmp, out_path)
    segments = manifest.get("source_segments") or [{
        "source": "unknown", "adjust": "unknown",
        "start": (df["date"].min().strftime("%Y-%m-%d") if len(df) else None),
        "end": (df["date"].max().strftime("%Y-%m-%d") if len(df) else None),
        "rows": int(len(df)),
    }]
    _write_manifest(base_dir, code, df, segments, run_id=uuid.uuid4().hex)
    return "direct"


def _migrate_one(df: pd.DataFrame, code: str, flip: np.ndarray,
                 oob: int) -> str:
    """Normalize one file's 手 rows.  Returns the write route used.

    Clean files (no residual out-of-band rows) route through the sanctioned
    ``DataStorage.save_daily_repair`` seam (formal=False, preserves
    provenance, recomputes schema_hash, heals pct_change).  Files with
    residual out-of-band rows (junk / corrupt years) fall back to a direct
    write because save_daily_repair would refuse them on the
    amount_volume_unit_mismatch band: genuine 手 rows are still ×100 while
    corrupt rows are preserved unchanged and the manifest is rewritten on the
    SAME (date-sorted) frame so its schema_hash matches the written bytes.
    A frame save_daily_repair refuses on a DIFFERENT pre-existing contract
    violation (e.g. >1% NaN volume/amount) falls back to the same direct
    write, so the migration never silently skips a file that has flips.
    """
    base = df["volume"]
    if pd.api.types.is_integer_dtype(base.dtype):
        df["volume"] = base.mask(flip, base.astype("int64") * 100)
    else:
        df["volume"] = base.mask(flip, base.astype("float64") * 100.0)

    if oob:
        return _direct_write(df, code)
    # sanctioned seam: preserves provenance + recomputes schema_hash + heals
    # pct_change, all under formal=False
    storage = DataStorage(os.path.join(ROOT, "data"))
    try:
        storage.save_daily_repair(df.assign(stock_code=code))
        return "clean"
    except ValueError as exc:
        # save_daily_repair refused the frame on a pre-existing contract
        # violation (finite-ratio NaN on volume/amount — not migration-
        # induced).  Fall back to the direct write so the 手 rows are still
        # ×100 and the honest (violating) frame is persisted as-is.
        print(f"    {code}: save_daily_repair refused ({exc}); "
              f"falling back to direct write", flush=True)
        return _direct_write(df, code)


if __name__ == "__main__":
    main()
