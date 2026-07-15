"""End-to-end validation: compare new preprocessing pipeline vs legacy gold.

Regenerates gold data for N stocks, shows feature differences, and runs a
quick XGBoost comparison to measure MCC impact of the new text chain.

Usage:
    PYTHONPATH=. ./.venv/Scripts/python scripts/compare_pipelines.py --stocks 3
    PYTHONPATH=. ./.venv/Scripts/python scripts/compare_pipelines.py --stocks 3 --train
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
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.features.news_nlp import NewsSentimentAnalyzer
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import compute_classification_metrics
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

QUICK_EVAL = [
    "000001", "600519", "000725", "600276", "000651",
    "601318", "600900", "002415", "000858", "600036",
    "002594", "601088", "300750", "688981", "002493",
]


def main():
    parser = argparse.ArgumentParser(description="Compare preprocessing pipelines")
    parser.add_argument("--stocks", type=int, default=3)
    parser.add_argument("--source", default="news", choices=["news", "guba"])
    parser.add_argument("--train", action="store_true",
                        help="Run XGBoost training comparison (slower)")
    args = parser.parse_args()

    cfg = load_config()
    data_dir = cfg.project.data_dir
    calendar = TradingCalendar("a_shares")
    analyzer = NewsSentimentAnalyzer()

    pp_cfg = cfg.get("preprocessing", {})
    pp = PreprocessingPipeline.from_config(pp_cfg)

    if args.source == "news":
        storage = NewsStorage(data_dir, calendar)
    else:
        storage = GubaStorage(data_dir, calendar)

    codes = QUICK_EVAL[:args.stocks]

    # Fit topic modeler on combined silver data
    t0 = time.time()
    tm = pp.topic_modeler
    topic_info = None
    if tm is not None and tm._enabled:
        all_silver = []
        for code in codes:
            df = (storage.load_silver_news(code) if args.source == "news"
                  else storage.load_silver(code))
            if not df.empty:
                all_silver.append(df)
        if all_silver:
            combined = pd.concat(all_silver, ignore_index=True)
            tm.fit(combined, source=args.source)
            if tm._model is not None:
                topic_info = tm._model.get_topic_info()
                logger.info("Topic modeler: %d topics from %d posts (%.1fs)",
                             len(topic_info), len(combined), time.time() - t0)

    # Per-stock comparison
    rows = []
    for code in codes:
        logger.info("--- %s ---", code)
        try:
            row = _compare_stock(storage, analyzer, pp, code, args)
            rows.append(row)
        except Exception as e:
            logger.error("%s: %s", code, e, exc_info=True)

    if not rows:
        logger.error("No stocks completed")
        sys.exit(1)

    summary = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("PIPELINE COMPARISON — FEATURE DIFF")
    print("=" * 70)
    feat_cols = ["stock_code", "n_days", "n_new_cols", "n_legacy_cols", "n_extra_cols"]
    if "mcc_new" in summary.columns:
        feat_cols += ["mcc_new", "mcc_legacy", "mcc_delta"]
    print(summary[feat_cols].to_string(index=False))

    if "n_extra_cols" in summary.columns:
        print(f"\nNew pipeline adds {summary['n_extra_cols'].mean():.0f} extra "
              f"columns on average vs legacy")

    if "mcc_delta" in summary.columns:
        mean_delta = summary["mcc_delta"].mean()
        print(f"Mean MCC delta (new - legacy): {mean_delta:+.4f}")

    return summary


def _compare_stock(storage, analyzer, pp, code, args):
    """Generate gold with both pipelines, compare features and optionally train."""
    date_start = "2015-01-01"
    date_end = datetime.now().strftime("%Y-%m-%d")

    # New pipeline path
    gold_new = storage.silver_to_gold(code, analyzer, preprocessing_pipeline=pp)

    # Legacy path
    gold_legacy = storage.silver_to_gold(code, analyzer, preprocessing_pipeline=None)

    if gold_new.empty or gold_legacy.empty:
        return {"stock_code": code, "n_days": 0, "error": "empty gold"}

    new_cols = set(gold_new.columns) - {"date", "stock_code"}
    legacy_cols = set(gold_legacy.columns) - {"date", "stock_code"}
    extra_cols = sorted(new_cols - legacy_cols)
    topic_cols = [c for c in extra_cols if c.startswith("topic_")]

    row = {
        "stock_code": code,
        "n_days": len(gold_new),
        "n_new_cols": len(new_cols),
        "n_legacy_cols": len(legacy_cols),
        "n_extra_cols": len(extra_cols),
        "n_topic_cols": len(topic_cols),
        "extra_cols": ",".join(extra_cols[:10]) if extra_cols else "",
    }

    if extra_cols:
        logger.info("  Extra columns: %s", extra_cols[:8])

    if args.train:
        train_result = _train_compare(code, gold_new, gold_legacy, args.source)
        row.update(train_result)

    return row


def _train_compare(code, gold_new, gold_legacy, source):
    """Quick XGBoost train on both gold variants, return MCC delta."""
    cfg = load_config()
    storage = DataStorage(cfg.project.data_dir)
    date_end = datetime.now().strftime("%Y-%m-%d")

    ohlcv = storage.load_daily(code, start_date="2015-01-01", end_date=date_end)
    if ohlcv.empty:
        return {}

    kwargs = {"sentiment_df" if source == "news" else "guba_df": None}
    aux_key = "sentiment_df" if source == "news" else "guba_df"

    def _train_one(gold_df):
        kwargs = {aux_key: gold_df}
        fp = FeaturePipeline(
            seq_len=60, flat_mode=True,
            use_sentiment=(source == "news"),
            use_guba=(source == "guba"),
            use_comment=False, use_margin=False, use_northbound=False,
            use_dragon_tiger=False, use_announcements=False,
            use_fundamental=False, use_etf_flow=False, use_xueqiu=False,
        )
        X, y, _ = fp.build_features(ohlcv, **kwargs)
        if len(X) == 0:
            return np.nan

        splitter = WalkForwardSplitter(train_years=2, val_months=3)
        n = len(X)
        pseudo = pd.date_range("2000-01-01", periods=n, freq="B")
        folds = list(splitter.split(pseudo))
        if not folds:
            return np.nan

        train_idx, val_idx = folds[0]
        if train_idx[-1] >= n or val_idx[-1] >= n:
            return np.nan

        model = XGBoostBaseline(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        metrics = compute_classification_metrics(y[val_idx], preds)
        return metrics.get("mcc", np.nan)

    mcc_new = _train_one(gold_new)
    mcc_legacy = _train_one(gold_legacy)

    return {
        "mcc_new": round(mcc_new, 4) if not np.isnan(mcc_new) else np.nan,
        "mcc_legacy": round(mcc_legacy, 4) if not np.isnan(mcc_legacy) else np.nan,
        "mcc_delta": round(mcc_new - mcc_legacy, 4)
        if not np.isnan(mcc_new) and not np.isnan(mcc_legacy) else np.nan,
    }


if __name__ == "__main__":
    main()
