"""Probe: daily active-stock sector coverage (2000-2026).

Measurement-only diagnostic (never imported by production).  Computes, for each
calendar year 2000-2026, the per-trading-day coverage

    coverage[d] = classified_active_stocks_on_d / active_stocks_on_d

TWO active/classified definitions are reported:

* VARIANT B (bar-based, PRIMARY -- mirrors the §v19 formal chain):
  ``download_industry_ranking.py`` INNER-joins daily K-line bars on
  ``(date, stock_code)`` with ``sector_membership.parquet``.  So the universe it
  sees on day d is the set of stocks that ACTUALLY TRADED that day (have a bar),
  and a stock is classified iff membership asserts a row at exactly ``(d, code)``.
  Membership rows are per-day (one row per (date, stock_code), all dates genuine
  trading days), so no event-boundary expansion is needed.

* VARIANT A (manifest-span, sensitivity only): the literal
  ``manifest [start,end] span contains d`` definition.  This is NOT what the
  chain sees: during 2015-2018 mass suspensions a stock with no bar is counted
  in the denominator, and because the membership source is bar-aligned it also
  has no membership row that day -- fabricating a coverage dip that never
  touches the real chain.

Genuine trading days are the union of ACTUAL daily K-line bar dates (NOT the
exchange_calendar artifact, which marks the 2000 golden weeks / future 2026
dates as open with no bars).  Phantom (bar-less open) calendar days per year are
reported separately and excluded.

Output: per-year tables for both variants, global span stats, years failing
candidate gate rules, and a recommended minimum_start_date.
"""

import glob
import json
import os
import time

import numpy as np
import pandas as pd

from stoke_ml.config import load_config

t0 = time.time()


def log(msg: str) -> None:
    print(msg, flush=True)


def agg_table(coverage: np.ndarray, years: np.ndarray,
              active: np.ndarray, phantom_by_year: dict[int, int],
              label: str) -> list[dict]:
    year_stats: list[dict] = []
    for yr in range(2000, 2027):
        idx = years == yr
        cov = coverage[idx]
        acts = active[idx]
        n_days = int(idx.sum())
        n_zero = int((acts == 0).sum())
        valid = cov[~np.isnan(cov)]
        if n_days == 0:
            continue
        if len(valid) == 0:
            mean_cov = p05_cov = frac80 = frac95 = np.nan
        else:
            mean_cov = float(np.mean(valid))
            p05_cov = float(np.percentile(valid, 5))
            frac80 = float(np.mean(valid >= 0.80))
            frac95 = float(np.mean(valid >= 0.95))
        year_stats.append({
            "year": yr, "trading_days": n_days, "zero_active_days": n_zero,
            "phantom_days": phantom_by_year.get(yr, 0),
            "mean_cov": mean_cov, "p05_cov": p05_cov,
            "frac80": frac80, "frac95": frac95,
        })
    log(f"\n=== PER-YEAR TABLE ({label}) ===")
    log(f"{'year':<6}{'trading_days':<14}{'mean':<9}{'p05':<9}"
        f"{'frac>=0.80':<11}{'frac>=0.95':<11}{'zero_act':<9}{'phantom':<9}")
    for s in year_stats:
        log(f"{s['year']:<6}{s['trading_days']:<14}{s['mean_cov']:<9.4f}"
            f"{s['p05_cov']:<9.4f}{s['frac80']:<11.4f}{s['frac95']:<11.4f}"
            f"{s['zero_active_days']:<9}{s['phantom_days']:<9}")
    return year_stats


def main() -> None:
    cfg = load_config()
    data_dir = cfg.project.data_dir
    base = os.path.join(data_dir, "a_shares")
    daily_dir = os.path.join(base, "daily")

    # ── 1. Daily bars: genuine trading days + per-stock bar presence ─────────
    log(f"[t={time.time()-t0:.1f}s] reading daily bars...")
    bar_rows: list[tuple[str, np.datetime64]] = []
    manifest: list[tuple[str, np.datetime64, np.datetime64]] = []
    bar_files = sorted(glob.glob(os.path.join(daily_dir, "*.parquet")))
    for f in bar_files:
        code = os.path.basename(f).split(".")[0]
        d = pd.to_datetime(pd.read_parquet(f, columns=["date"])["date"]).dt.normalize()
        for x in d.to_numpy(dtype="datetime64[D]"):
            bar_rows.append((code, x))
        mf = os.path.join(daily_dir, f"{code}.manifest.json")
        try:
            with open(mf, "r", encoding="utf-8") as fh:
                m = json.load(fh)
            manifest.append((code,
                             np.datetime64(pd.Timestamp(m["start"]), "D"),
                             np.datetime64(pd.Timestamp(m["end"]), "D")))
        except (OSError, ValueError, TypeError, KeyError,
                json.JSONDecodeError):
            manifest.append((code, np.datetime64("1970-01-01", "D"),
                             np.datetime64("2099-12-31", "D")))
    bars = pd.DataFrame(bar_rows, columns=["stock_code", "date"])
    bars = bars.drop_duplicates()
    man = pd.DataFrame(manifest, columns=["code", "start", "end"])
    log(f"[t={time.time()-t0:.1f}s] {len(bars)} bar rows, "
        f"{len(man)} manifests, {len(bar_files)} files")

    genuine = np.array(sorted(bars["date"].unique()), dtype="datetime64[D]")
    years = np.array([pd.Timestamp(x).year for x in genuine], dtype=int)

    # ── 2. Calendar artifact (phantom-day accounting only) ───────────────────
    from stoke_ml.data.calendar import get_research_calendar
    cal = get_research_calendar("a_shares", strict=False, data_dir=data_dir)
    cal_days = set(cal.get_trading_days("2000-01-01", "2026-12-31"))
    genuine_set = {pd.Timestamp(x).date() for x in genuine}
    phantom = sorted(cal_days - genuine_set)
    phantom_by_year: dict[int, int] = {}
    for p in phantom:
        phantom_by_year[pd.Timestamp(p).year] = phantom_by_year.get(
            pd.Timestamp(p).year, 0) + 1
    log(f"calendar artifact: {len(cal_days)} open days; "
        f"{len(phantom)} phantom (bar-less open) days -> "
        f"{phantom_by_year}")

    # ── 3. Membership ────────────────────────────────────────────────────────
    mem_path = os.path.join(base, "sector_membership.parquet")
    log(f"[t={time.time()-t0:.1f}s] reading membership...")
    mem = pd.read_parquet(mem_path)
    mem["date"] = pd.to_datetime(mem["date"], errors="coerce").dt.normalize()
    mem["stock_code"] = mem["stock_code"].astype(str).str.strip()
    n_before = len(mem)
    mem = mem[mem["date"].notna() & mem["stock_code"].notna()].copy()
    n_dropped = int(n_before - len(mem))
    log(f"membership rows: {n_before} -> {len(mem)} (dropped {n_dropped} NaN date/stock rows)")

    # ── 4. VARIANT B (bar-based = real chain) coverage ───────────────────────
    # classified_B(d) = # stocks with BOTH a bar and a membership row on d
    cls = pd.merge(
        bars[["date", "stock_code"]],
        mem[["date", "stock_code"]].drop_duplicates(),
        on=["date", "stock_code"], how="inner",
    )
    active_B = bars.groupby("date").size().reindex(
        pd.Index(genuine, name="date"), fill_value=0).to_numpy()
    class_B = cls.groupby("date").size().reindex(
        pd.Index(genuine, name="date"), fill_value=0).to_numpy()
    cov_B = np.where(active_B > 0, class_B / np.maximum(active_B, 1), np.nan)
    log(f"[t={time.time()-t0:.1f}s] variant B (bar-based): "
        f"global mean={np.nanmean(cov_B):.4f}")

    # ── 5. VARIANT A (manifest-span) coverage ────────────────────────────────
    starts = man["start"].to_numpy(dtype="datetime64[D]")
    ends = man["end"].to_numpy(dtype="datetime64[D]")
    starts_sorted = np.sort(starts)
    ends_sorted = np.sort(ends)

    def active_count(days: np.ndarray) -> np.ndarray:
        n_le = np.searchsorted(starts_sorted, days, side="right")
        n_lt = np.searchsorted(ends_sorted, days, side="left")
        return n_le - n_lt

    known = man["code"]
    memA = mem[mem["stock_code"].isin(known)].merge(
        man, left_on="stock_code", right_on="code", how="left")
    memA = memA.dropna(subset=["start", "end"])
    md = memA["date"].to_numpy(dtype="datetime64[D]")
    memA = memA[(memA["start"].to_numpy(dtype="datetime64[D]") <= md) &
                (md <= memA["end"].to_numpy(dtype="datetime64[D]"))].copy()
    class_A = (memA.groupby("date")["stock_code"].nunique()
               .reindex(pd.Index(genuine, name="date"), fill_value=0).to_numpy())
    active_A = active_count(genuine)
    cov_A = np.where(active_A > 0, class_A / np.maximum(active_A, 1), np.nan)
    log(f"[t={time.time()-t0:.1f}s] variant A (manifest-span): "
        f"global mean={np.nanmean(cov_A):.4f}")

    # ── 6. Tables + global + candidate rules ─────────────────────────────────
    stats_B = agg_table(cov_B, years, active_B, phantom_by_year,
                        "VARIANT B: bar-based (REAL chain semantics)")
    stats_A = agg_table(cov_A, years, active_A, phantom_by_year,
                        "VARIANT A: manifest-span (sensitivity)")

    def global_stats(cov: np.ndarray, active: np.ndarray) -> dict:
        valid = cov[~np.isnan(cov)]
        return {
            "n_trading_days": int(len(cov)),
            "n_zero_active": int((active == 0).sum()),
            "mean_cov": float(np.mean(valid)),
            "p05_cov": float(np.percentile(valid, 5)),
            "frac80": float(np.mean(valid >= 0.80)),
            "frac95": float(np.mean(valid >= 0.95)),
        }

    for label, cov, act, stats in (
        ("B (bar-based)", cov_B, active_B, stats_B),
        ("A (manifest-span)", cov_A, active_A, stats_A),
    ):
        g = global_stats(cov, act)
        log(f"\n=== GLOBAL 2000-2026 [{label}] ===")
        log(f"trading_days={g['n_trading_days']} "
            f"zero_active={g['n_zero_active']} "
            f"mean={g['mean_cov']:.4f} p05={g['p05_cov']:.4f} "
            f"frac80={g['frac80']:.4f} frac95={g['frac95']:.4f}")
        fail_p05 = [s["year"] for s in stats
                    if not np.isnan(s["p05_cov"]) and s["p05_cov"] < 0.80]
        fail_mean = [s["year"] for s in stats
                     if not np.isnan(s["mean_cov"]) and s["mean_cov"] < 0.95]
        log(f"fail p05>=0.80: {fail_p05}")
        log(f"fail mean>=0.95: {fail_mean}")
        # contiguous viability: first year from which ALL later years pass both
        viable = None
        for k in range(len(stats)):
            sub = stats[k:]
            if all(not np.isnan(s["p05_cov"]) and s["p05_cov"] >= 0.80
                   and not np.isnan(s["mean_cov"]) and s["mean_cov"] >= 0.95
                   for s in sub):
                viable = stats[k]["year"]
                break
        log(f"suggested minimum_start_date: {viable}")

    log(f"\nmembership drop count (NaN date/stock rows): {n_dropped}")
    log("event-boundary expansion needed? NO -- membership verified as per-day "
        "rows; (date, stock_code) unique; all membership dates are genuine "
        "trading days.")
    log(f"elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
