"""One-shot data inventory: scan all parquet directories and print a summary table."""
import os
import sys
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "a_shares")

# Define all expected data layers with directory paths
LAYERS = [
    ("K-line (daily)", os.path.join(DATA_DIR, "daily")),
    ("K-line (minute)", os.path.join(DATA_DIR, "minute")),
    ("News (raw)", os.path.join(DATA_DIR, "news_raw")),
    ("News (silver)", os.path.join(DATA_DIR, "news_silver")),
    ("News (Gold/sentiment)", os.path.join(DATA_DIR, "news_sentiment")),
    ("Guba (raw)", os.path.join(DATA_DIR, "guba_raw")),
    ("Guba (silver)", os.path.join(DATA_DIR, "guba_silver")),
    ("Guba (Gold/sentiment)", os.path.join(DATA_DIR, "guba_sentiment")),
    ("Comment (features)", os.path.join(DATA_DIR, "comment")),
    ("Announcement (raw)", os.path.join(DATA_DIR, "announcement_raw")),
    ("Announcement (Gold)", os.path.join(DATA_DIR, "announcement_sentiment")),
    ("Margin", os.path.join(DATA_DIR, "margin")),
    ("Northbound", os.path.join(DATA_DIR, "northbound")),
    ("Dragon & Tiger", os.path.join(DATA_DIR, "dragon_tiger")),
    ("Fundamental", os.path.join(DATA_DIR, "fundamental")),
    ("ETF flow", os.path.join(DATA_DIR, "etf_flow")),
    ("Universe (IPO/ST/index)", os.path.join(DATA_DIR, "universe")),
    ("Analyst", os.path.join(DATA_DIR, "analyst")),
    ("Pledge", os.path.join(DATA_DIR, "pledge")),
]


def find_parquet_files(root_dir):
    """Recursively find all .parquet files under root_dir. Returns list of paths."""
    if not os.path.exists(root_dir):
        return []
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith(".parquet"):
                paths.append(os.path.join(dirpath, fn))
    return paths


def sample_parquet(path, max_rows=5):
    """Read a parquet file safely, returning (row_count, columns, sample_df)."""
    try:
        df = pd.read_parquet(path)
        return len(df), list(df.columns), df.head(1)
    except Exception:
        return 0, [], pd.DataFrame()


def main():
    print("=" * 100)
    print(f"{'Data Layer':<28s} {'Files':>8s} {'Total Rows':>14s} {'Row Scale':>10s} {'Date Start':>12s} {'Date End':>12s}")
    print("=" * 100)

    totals = {"files": 0, "rows": 0}

    for layer_name, layer_dir in LAYERS:
        files = find_parquet_files(layer_dir)
        n_files = len(files)

        if n_files == 0:
            print(f"{layer_name:<28s} {'—':>8s} {'—':>14s} {'—':>10s} {'—':>12s} {'—':>12s}")
            continue

        # Sample first 3 files for column names and row count estimation
        sample_paths = files[:3]
        total_rows = 0
        all_cols = set()
        date_start = "N/A"
        date_end = "N/A"

        for i, p in enumerate(sample_paths):
            n, cols, sample = sample_parquet(p)
            total_rows += n
            all_cols.update(cols)
            if i == 0 and not sample.empty:
                # Try to find date columns
                for dc in ["date", "trade_date", "aligned_date", "timestamp", "day"]:
                    if dc in sample.columns:
                        try:
                            vals = pd.to_datetime(sample[dc].dropna())
                            if not vals.empty:
                                date_start = vals.min().strftime("%Y-%m-%d")
                                date_end = "..."  # need full scan for end date
                        except Exception:
                            pass

        # Estimate total rows: average of samples * n_files
        avg_rows = total_rows / len(sample_paths) if sample_paths else 0
        est_total = int(avg_rows * n_files)

        # Try to get actual total by summing all files (only for small directories)
        if n_files <= 50:
            actual_total = 0
            for p in files:
                n, _, _ = sample_parquet(p)
                actual_total += n
            est_total = actual_total

        # Get date range from first and last file (by doing a quick scan)
        if n_files > 0 and n_files <= 100:
            first_n, _, first_df = sample_parquet(files[0])
            last_n, _, last_df = sample_parquet(files[-1])
            for dc in ["date", "trade_date", "aligned_date", "timestamp", "day"]:
                if not first_df.empty and dc in first_df.columns:
                    try:
                        s = pd.to_datetime(first_df[dc].dropna())
                        if not s.empty:
                            date_start = s.min().strftime("%Y-%m-%d")
                    except Exception:
                        pass
                if not last_df.empty and dc in last_df.columns:
                    try:
                        e = pd.to_datetime(last_df[dc].dropna())
                        if not e.empty:
                            date_end = e.max().strftime("%Y-%m-%d")
                    except Exception:
                        pass

        # Human-readable row scale
        if est_total >= 1_000_000:
            scale = f"{est_total/1_000_000:.1f}M"
        elif est_total >= 1_000:
            scale = f"{est_total/1_000:.1f}K"
        else:
            scale = str(est_total)

        print(f"{layer_name:<28s} {n_files:>8d} {est_total:>14,d} {scale:>10s} {date_start:>12s} {date_end:>12s}")

        totals["files"] += n_files
        totals["rows"] += est_total

    print("=" * 100)
    print(f"{'TOTAL':<28s} {totals['files']:>8d} {totals['rows']:>14,d}")
    print()

    # Column-level detail for key layers
    print("=" * 100)
    print("COLUMN DETAILS (sampled from first file in each layer)")
    print("=" * 100)
    for layer_name, layer_dir in LAYERS:
        files = find_parquet_files(layer_dir)
        if not files:
            continue
        _, cols, _ = sample_parquet(files[0])
        col_str = ", ".join(cols[:30])
        if len(cols) > 30:
            col_str += f" ... (+{len(cols)-30})"
        print(f"\n[{layer_name}] {len(cols)} columns:")
        print(f"  {col_str}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
