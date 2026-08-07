"""Download historical index-constituent membership from Baostock.

Queries constituent sets on a monthly grid for the 3 major indices Baostock
covers (000300 / 000905 / 000016) and reconstructs per-stock membership
intervals. Baostock has NO historical data for 000852 (CSI1000) or 000688
(科创50) — documented coverage gaps.

Baostock row semantics: a query at date D returns every constituent as of D,
each row carrying `updateDate`. In baostock 0.9.20 `updateDate` is Baostock's
monthly data-refresh date (the last trading day of the prior month), re-stamped
on EVERY query for every stock — NOT an index adjustment date. A stock present
in consecutive monthly snapshots is therefore a continuous member; an absence
from one or more snapshots marks a removal.

Reconstruction: for each (index, stock), a membership spell is a run of
consecutive snapshot dates in which the stock appears. in_date = the earliest
refresh date (updateDate) seen in the run (a proxy for when membership became
effective); out_date = the FIRST ABSENT snapshot — the first grid month that no
longer reported the stock (last present + 1 month) — NaT if the stock was still
a member at the final grid query (open-ended). Every consumer
(stoke_ml.data.universe.load_index_membership, the panel normalizer) reads the
half-open interval in_date <= d < out_date, so out_date must be the first month
the stock was GONE, never the last month it was confirmed. Spans are resolved
to the monthly grid — a stock joining/leaving between snapshots is attributed
to the boundary of its presence run.

Outputs:
  data/a_shares/index_constituents_hist/snapshots/{index}/{YYYY-MM-DD}.parquet
      raw query result per index-date (audit trail)
  data/a_shares/index_constituents_hist/membership.parquet
      long-form intervals: stock_code, index_code, in_date, out_date
"""
import argparse
import os
import sys
import time

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.download_manifest import write_run_manifest

INDICES = {
    "000300": ("query_hs300_stocks", "沪深300"),
    "000905": ("query_zz500_stocks", "中证500"),
    "000016": ("query_sz50_stocks", "上证50"),
}
GRID_START = "2015-01-01"
GRID_END = "2026-12-31"
RETRIES = 3


def query_flat(bs, fn_name, date, retries=RETRIES):
    """One constituent query -> DataFrame, with retry."""
    fn = getattr(bs, fn_name)
    last_err = None
    for attempt in range(retries):
        try:
            rs = fn(date=date)
            if rs.error_code != "0":
                raise RuntimeError(f"{fn_name}@{date}: {rs.error_code} {rs.error_msg}")
            cols = rs.fields  # baostock 0.9.20 stores field names here (no get_fields())
            df = rs.get_data()  # full DataFrame; pagination handled internally
            if df.empty:
                df = pd.DataFrame(columns=cols)
            df["query_date"] = pd.Timestamp(date)
            return df
        except Exception as e:
            last_err = e
            print(f"  retry {fn_name}@{date} ({attempt+1}/{retries}): {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{fn_name}@{date}: failed after {retries} attempts: {last_err}")


def rebuild_membership(snap_dir):
    """Concatenate cached snapshots -> membership interval DataFrame.

    Baostock 0.9.20 re-stamps updateDate (its monthly refresh date) on every
    query for every stock, so spells are keyed on CONTIGUOUS PRESENCE across
    snapshot dates rather than on updateDate: a stock present in consecutive
    monthly snapshots is one continuous membership spell. in_date = the run's
    earliest refresh date (updateDate); out_date = the FIRST ABSENT snapshot
    month (last present + 1 month), NaT if the run reaches the final grid query.
    Every consumer reads the half-open interval in_date <= d < out_date, so
    out_date must be the first grid month the stock was gone, never the last
    month it was confirmed.
    """
    frames = []
    for idx in INDICES:
        sd = os.path.join(snap_dir, idx)
        if not os.path.isdir(sd):
            continue
        for f in sorted(os.listdir(sd)):
            if not f.endswith(".parquet"):
                continue
            df = pd.read_parquet(os.path.join(sd, f))
            if df.empty:
                continue
            df["index_code"] = idx
            frames.append(df)
    if not frames:
        raise SystemExit("no snapshots collected — run without --resume first")
    allq = pd.concat(frames, ignore_index=True)
    allq["stock_code"] = allq["code"].astype(str).str.rsplit(".", n=1).str[-1]
    allq["updateDate"] = pd.to_datetime(allq["updateDate"], errors="coerce").dt.normalize()
    allq["query_date"] = pd.to_datetime(allq["query_date"]).dt.normalize()
    allq = allq.dropna(subset=["stock_code", "updateDate"])

    last_grid = allq["query_date"].max()
    rows = []
    for (idx, code), g in allq.groupby(["index_code", "stock_code"]):
        g = g.sort_values("query_date")
        # A new spell starts when > 1 calendar month elapses between two
        # consecutive snapshots (the stock was absent -> a removal + re-add).
        month_ids = g["query_date"].dt.to_period("M").astype("int64")
        new_run = month_ids.diff().gt(1)
        new_run.iloc[0] = True
        for _rid, rg in g.groupby(new_run.cumsum()):
            in_date = rg["updateDate"].min()
            # out_date is the FIRST ABSENT snapshot, not the last present one.
            # Consumers read the half-open interval in_date <= d < out_date, so
            # writing the last-present month here would wrongly drop the final
            # confirmed member month.  Spells are contiguous (the split above on
            # month_ids.diff().gt(1)), so the month right after the last present
            # snapshot is exactly the first one the stock was absent from.
            last_present = rg["query_date"].max()
            out_date = last_present + pd.DateOffset(months=1)
            # Query dates are month starts: out_date == last_grid means the
            # final grid month WAS queried and the stock absent there (contiguous
            # spell → the month after last_present is empty) → closed, not
            # open-ended.  STRICT > also preserves open-endedness when the final
            # grid month was never successfully queried (last_grid is the max
            # query_date among SUCCESSFUL snapshots, i.e. earlier than out_date).
            still_active = out_date > last_grid
            rows.append({"stock_code": code, "index_code": idx,
                         "in_date": in_date,
                         "out_date": pd.NaT if still_active else out_date})
    return pd.DataFrame(rows, columns=["stock_code", "index_code", "in_date", "out_date"])


def _finalize(data_dir, start_date, end_date, requested, failed,
              complete, skipped_existing, snap_dir, base) -> int:
    """Write the run manifest, then rebuild + save membership.

    Returns a non-zero exit code when the run is not fully successful.  The
    manifest is ALWAYS written (a partial run is recorded honestly, §五-5); a
    manifest-write failure RAISES — a run that cannot record its own coverage
    must fail loudly, never be swallowed (§七).
    """
    write_run_manifest(
        data_dir, "a_shares/index_constituents_hist",
        start_date=start_date, end_date=end_date,
        requested=requested, failed=failed, complete=complete,
        success_count=len(complete),
        skipped_existing_count=skipped_existing,
    )  # no try/except: a manifest-write failure must propagate → non-zero exit
    if failed:
        # A partial run can never pass for complete (download_manifest.py's own
        # principle): the manifest records the partial status AND the process
        # exits non-zero so QA sees the gap.
        return 1
    mem = rebuild_membership(snap_dir)
    mem.to_parquet(os.path.join(base, "membership.parquet"), index=False,
                   compression="lz4")
    print(f"membership.parquet: {len(mem)} spells, {mem['stock_code'].nunique()} stocks")
    print(mem.groupby("index_code")["stock_code"].nunique().to_string())
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=GRID_START)
    ap.add_argument("--end", default=GRID_END)
    ap.add_argument("--resume", action="store_true",
                    help="skip (index, date) snapshots already cached")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = cfg.project.data_dir
    base = os.path.join(data_dir, "a_shares", "index_constituents_hist")
    snap_dir = os.path.join(base, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    for idx in INDICES:
        os.makedirs(os.path.join(snap_dir, idx), exist_ok=True)

    try:
        import baostock as bs
    except ImportError:
        raise SystemExit("baostock not installed: pip install baostock")

    lg = bs.login()
    if lg.error_code != "0":
        raise SystemExit(f"bs.login failed: {lg.error_code} {lg.error_msg}")
    # Best-effort: avoid indefinite hang on a dead connection (retry loop recovers).
    try:
        _sock = bs.__session.sock
        if _sock is not None:
            _sock.settimeout(90)
    except Exception:
        pass  # internal attribute may differ across baostock versions — non-fatal
    # baostock 0.9.20 has no module-level __session; the live socket lives at
    # baostock.common.context.default_socket. Without a timeout a dead connection
    # hangs send_msg forever; the retry loop only helps if the recv raises.
    try:
        import baostock.common.context as _bs_ctx
        _sock = getattr(_bs_ctx, "default_socket", None)
        if _sock is not None:
            _sock.settimeout(90)
    except Exception:
        pass  # non-fatal; --resume reruns cover any hang

    grid = pd.date_range(args.start, args.end, freq="MS")
    requested = [f"{idx}/{d.strftime('%Y-%m-%d')}"
                 for d in grid for idx in INDICES]
    done_snapshots: set[str] = set()
    failed_snapshots: list[str] = []
    skipped_existing = 0

    try:
        for i, d in enumerate(grid):
            day = d.strftime("%Y-%m-%d")
            for idx, (fn, _name) in INDICES.items():
                out = os.path.join(snap_dir, idx, f"{day}.parquet")
                snap_id = f"{idx}/{day}"
                if args.resume and os.path.exists(out):
                    done_snapshots.add(snap_id)
                    skipped_existing += 1
                    continue
                try:
                    df = query_flat(bs, fn, day)
                except Exception as e:
                    failed_snapshots.append(snap_id)
                    print(f"  {snap_id}: query failed: {str(e)[:100]}")
                    continue
                df["index_code"] = idx
                # Atomic write: a killed process must not leave a truncated
                # snapshot that --resume would skip as "done" and lose forever.
                tmp = out + ".tmp"
                df.to_parquet(tmp, index=False, compression="lz4")
                os.replace(tmp, out)
                done_snapshots.add(snap_id)
            if (i + 1) % 12 == 0:
                print(f"  {i+1}/{len(grid)} grid months done")
    finally:
        bs.logout()

    # Unified run manifest (§五-5): a partial run can never pass for complete.
    # _finalize writes the manifest, then — only on a clean full run — rebuilds
    # + saves membership.  It returns a non-zero exit code when the run is not
    # fully successful and never swallows a manifest-write failure.
    rc = _finalize(
        data_dir, args.start, args.end, requested, failed_snapshots,
        done_snapshots, skipped_existing, snap_dir, base,
    )
    if rc != 0:
        sys.exit(rc)


if __name__ == "__main__":
    main()
