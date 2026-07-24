"""BERTopic text quality chain benchmark.

Loads Silver data for N stocks, runs the full text chain including topic
modeling, reports topic coverage, feature count, and runtime.
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.features.news_nlp import NewsSentimentAnalyzer
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _collect_silver(data_dir, source, stocks, date_start, date_end):
    """Load and combine silver data from multiple stocks."""
    calendar = TradingCalendar("a_shares")
    frames = []
    for code in stocks:
        try:
            if source == "news":
                storage = NewsStorage(data_dir, calendar)
                df = storage.load_silver_news(code)
            else:
                storage = GubaStorage(data_dir, calendar)
                df = storage.load_silver(code)
            if df is not None and not df.empty:
                if "aligned_date" in df.columns:
                    df["aligned_date"] = pd.to_datetime(df["aligned_date"])
                    df = df[(df["aligned_date"] >= date_start) & (df["aligned_date"] <= date_end)]
                frames.append(df.assign(stock_code=code))
        except Exception as e:
            logger.warning("Failed to load %s for %s: %s", source, code, e)

    if not frames:
        return pd.DataFrame(), []

    combined = pd.concat(frames, ignore_index=True)
    loaded_codes = sorted(combined["stock_code"].unique())
    logger.info("Loaded %d posts from %d stocks", len(combined), len(loaded_codes))
    return combined, loaded_codes


def main():
    parser = argparse.ArgumentParser(description="BERTopic text chain benchmark")
    parser.add_argument("--source", choices=["news", "guba"], default="news")
    parser.add_argument("--stocks", type=int, default=10)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    t0 = time.time()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    date_start = cfg.markets.a_shares.start_date
    date_end = datetime.now().strftime("%Y-%m-%d")

    # ---- Stock selection -------------------------------------------------
    if args.source == "news":
        silver_dir = os.path.join(data_dir, "a_shares", "news_silver")
    else:
        silver_dir = os.path.join(data_dir, "a_shares", "guba_silver")

    if not os.path.isdir(silver_dir):
        logger.error("Silver directory not found: %s", silver_dir)
        sys.exit(1)

    all_codes = sorted([
        f.replace(".parquet", "")
        for f in os.listdir(silver_dir)
        if f.endswith(".parquet")
    ])
    if len(all_codes) == 0:
        logger.error("No silver data found in %s", silver_dir)
        sys.exit(1)

    if args.stocks < len(all_codes):
        step = max(len(all_codes) // args.stocks, 1)
        codes = [all_codes[i * step] for i in range(args.stocks)]
    else:
        codes = all_codes

    logger.info("Source: %s, %d/%d stocks", args.source, len(codes), len(all_codes))

    # ---- Collect all posts -----------------------------------------------
    t_load = time.time()
    all_silver, loaded_codes = _collect_silver(
        data_dir, args.source, codes, date_start, date_end
    )
    if all_silver.empty:
        logger.error("No silver data loaded")
        sys.exit(1)
    logger.info("Data loading: %.1fs", time.time() - t_load)

    # ---- Build preprocessing pipeline ------------------------------------
    pp_cfg = cfg.get("preprocessing", {})
    pp = PreprocessingPipeline.from_config(pp_cfg)

    # Ensure topic modeler is enabled for benchmark
    tm = pp.topic_modeler
    if tm is None:
        logger.warning(
            "Topic modeler not configured (preprocessing.text.topic_model.enabled=false). "
            "Only text_pre + aggregation will run."
        )
    else:
        logger.info("Topic modeler: min_topic_size=%d, embedding=%s",
                     tm.min_topic_size, tm.embedding_model)

    analyzer = NewsSentimentAnalyzer(force_lexicon=True)

    # ---- Fit topic modeler on all posts ----------------------------------
    if tm is not None and tm._enabled:
        t_fit = time.time()
        logger.info("Fitting topic modeler on %d posts...", len(all_silver))
        # Run quality filter first on combined data
        clean = pp.run("text_pre", all_silver)
        tm.fit(clean, source=args.source)
        fit_time = time.time() - t_fit
        logger.info("Topic modeler fit: %.1fs", fit_time)
        if tm._model is not None:
            try:
                n_topics = len(tm._model.get_topic_info())
                logger.info("Topics discovered: %d", n_topics)
            except Exception:
                n_topics = "unknown"
        else:
            n_topics = "disabled (fit failed)"
    else:
        fit_time = 0
        n_topics = "disabled"

    # ---- Per-stock: run chain + aggregate --------------------------------
    t_proc = time.time()
    results = []

    for code in loaded_codes:
        try:
            if args.source == "news":
                storage = NewsStorage(data_dir)
                gold = storage.silver_to_gold(
                    code, analyzer=analyzer, preprocessing_pipeline=pp
                )
            else:
                storage = GubaStorage(data_dir)
                gold = storage.silver_to_gold(
                    code, analyzer=analyzer, preprocessing_pipeline=pp
                )

            if gold.empty:
                continue

            topic_cols = [c for c in gold.columns if c.startswith("topic_")]
            results.append({
                "stock": code,
                "n_days": len(gold),
                "n_topic_features": len(topic_cols),
                "topic_entropy_mean": (
                    gold["topic_entropy"].mean()
                    if "topic_entropy" in gold.columns else 0.0
                ),
                "topic_entropy_std": (
                    gold["topic_entropy"].std()
                    if "topic_entropy" in gold.columns else 0.0
                ),
                "n_feature_cols": len(gold.columns) - 2,  # excl date, stock_code
            })

        except Exception as e:
            logger.warning("  %s: FAILED — %s", code, e)

    proc_time = time.time() - t_proc
    total_time = time.time() - t0

    # ---- Report ----------------------------------------------------------
    if not results:
        logger.error("No per-stock results")
        sys.exit(1)

    results_df = pd.DataFrame(results)
    print()
    print("=" * 64)
    print(f"BERTopic TEXT CHAIN BENCHMARK — {args.source.upper()}")
    print("=" * 64)
    print(f"  Stocks processed:      {len(results_df)}")
    print(f"  Topics discovered:     {n_topics}")
    print(f"  Total posts:           {len(all_silver)}")
    print(f"  Fit time:              {fit_time:.1f}s")
    print(f"  Per-stock proc time:   {proc_time:.1f}s")
    print(f"  Total time:            {total_time:.1f}s")
    print()
    print("Per-stock feature summary:")
    print(results_df.describe().round(3).to_string())
    print()
    print("Sample rows:")
    print(results_df.head(10).to_string())

    if "topic_entropy_mean" in results_df.columns:
        valid_entropy = results_df["topic_entropy_mean"].replace(0.0, np.nan).dropna()
        if len(valid_entropy) > 0:
            print(f"\n  Topic entropy (non-zero stocks): "
                  f"mean={valid_entropy.mean():.3f}, "
                  f"std={valid_entropy.std():.3f}")
            print(f"  (Higher entropy = more diverse discussion topics)")

    # Save
    output_dir = args.output or cfg.project.model_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"topic_benchmark_{args.source}.csv")
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
