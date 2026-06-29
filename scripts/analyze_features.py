"""Feature dimension ablation study with bootstrap confidence intervals.

Trains XGBoost with different feature subsets across 35+ stocks and uses
paired bootstrap to compute 95% confidence intervals for MCC deltas,
quantifying how much each data dimension contributes to prediction accuracy.
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.announcement_storage import AnnouncementStorage
from stoke_ml.data.comment_storage import CommentStorage
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import compute_classification_metrics
from stoke_ml.models.baseline import XGBoostBaseline, LGBMBaseline
from stoke_ml.features.selection import FeatureSelector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

N_ESTIMATORS = 50
MAX_DEPTH = 4
LEARNING_RATE = 0.1
FEATURE_FRACTION = 0.7  # LightGBM EFB regularization
MAX_WINDOWS = 2
N_BOOTSTRAP = 1000
SEED = 42

rng = np.random.default_rng(SEED)


def load_stock_data(code, ds, ns, as_, gs, cs, ms, nbs, start, end):
    result = {"kl": ds.load_daily(code, start, end)}
    sentiment = ns.load_daily_sentiment(code, start, end)
    if not sentiment.empty:
        result["sentiment"] = sentiment
    announcement = as_.load_daily_sentiment(code, start, end)
    if not announcement.empty:
        result["announcement"] = announcement
    guba = gs.load_daily_sentiment(code, start, end)
    if not guba.empty:
        result["guba"] = guba
    comment = cs.build_features(code, start, end)
    if not comment.empty:
        result["comment"] = comment
    margin = ms.load(code, start, end)
    if not margin.empty:
        result["margin"] = margin
    northbound = nbs.load(code, start, end)
    if not northbound.empty:
        result["northbound"] = northbound
    return result


def _make_model(model_type):
    if model_type == "lgbm":
        return LGBMBaseline(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE, feature_fraction=FEATURE_FRACTION,
        )
    return XGBoostBaseline(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
    )


def train_and_eval(pipe, data, splitter, model_type="xgb", selector=None):
    X, y, _ = pipe.build_features(
        data["kl"],
        sentiment_df=data.get("sentiment"),
        announcement_df=data.get("announcement"),
        guba_df=data.get("guba"),
        comment_df=data.get("comment"),
        margin_df=data.get("margin"),
        northbound_df=data.get("northbound"),
    )
    if len(X) < 50:
        return None
    n_samples, seq_len, n_feats = X.shape
    X_flat = X.reshape(n_samples, seq_len * n_feats)
    metrics_list = []
    for i, (train_idx, val_idx) in enumerate(splitter.split(X_flat)):
        if i >= MAX_WINDOWS:
            break
        X_train, X_val = X_flat[train_idx], X_flat[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue
        # Apply feature selection on training data only
        if selector is not None:
            try:
                X_train = selector.fit_transform(X_train, y_train)
                X_val = selector.transform(X_val)
            except Exception:
                pass  # fall through with full features on selection failure
        model = _make_model(model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        metrics_list.append(compute_classification_metrics(y_val, y_pred))
    if not metrics_list:
        return None
    return {
        "mcc": np.mean([m["mcc"] for m in metrics_list]),
        "accuracy": np.mean([m["accuracy"] for m in metrics_list]),
        "f1": np.mean([m["f1"] for m in metrics_list]),
        "n_windows": len(metrics_list),
    }


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap 95% CI for the mean of *values*."""
    values = np.array(values)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    means = np.array(means)
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return np.mean(values), lo, hi


def bootstrap_delta_ci(baseline_vals, config_vals, n_bootstrap=N_BOOTSTRAP, alpha=0.05):
    """Paired bootstrap CI for mean(config - baseline)."""
    baseline_vals = np.array(baseline_vals)
    config_vals = np.array(config_vals)
    assert len(baseline_vals) == len(config_vals)
    deltas = []
    n = len(baseline_vals)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        delta = np.mean(config_vals[idx]) - np.mean(baseline_vals[idx])
        deltas.append(delta)
    deltas = np.array(deltas)
    lo = np.percentile(deltas, 100 * alpha / 2)
    hi = np.percentile(deltas, 100 * (1 - alpha / 2))
    return np.mean(config_vals - baseline_vals), lo, hi


def main():
    parser = argparse.ArgumentParser(description="Feature dimension ablation study")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=str, default=None)
    parser.add_argument("--start", type=str, default="2024-01-01")
    parser.add_argument("--end", type=str, default="2026-06-27")
    parser.add_argument("--n-stocks", type=int, default=35)
    parser.add_argument("--model", type=str, default="xgb",
                        choices=["xgb", "lgbm", "all"],
                        help="Model type: xgb (XGBoost), lgbm (LightGBM+EFB), all (both)")
    parser.add_argument("--feature-select", action="store_true",
                        help="Apply MI(200) + SFS(50) feature selection per window")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    ds = DataStorage(data_dir)
    ns = NewsStorage(data_dir)
    as_ = AnnouncementStorage(data_dir)
    gs = GubaStorage(data_dir)
    cs = CommentStorage(data_dir)
    ms = MarketWideStorage(data_dir, "margin")
    nbs = MarketWideStorage(data_dir, "northbound")

    # Pick stocks
    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        base = os.path.join(data_dir, "a_shares", "daily")
        all_codes = set()
        for root, _dirs, files in os.walk(base):
            for f in files:
                if f.endswith(".parquet"):
                    all_codes.add(f.replace(".parquet", ""))
        per_prefix = max(1, args.n_stocks // 7)
        codes = []
        for prefix in ["000", "002", "300", "600", "601", "603", "688"]:
            prefix_codes = sorted([c for c in all_codes if c.startswith(prefix)])[:per_prefix]
            codes.extend(prefix_codes)

    # Load data
    logger.info("Loading data for %d stocks...", len(codes))
    all_data = {}
    for code in codes:
        data = load_stock_data(code, ds, ns, as_, gs, cs, ms, nbs, args.start, args.end)
        if data.get("kl") is not None and len(data["kl"]) >= 300:
            all_data[code] = data
    logger.info("Loaded %d stocks with sufficient data", len(all_data))

    splitter = WalkForwardSplitter(train_years=1, val_months=3, step_months=3)

    # --- Define configurations ---
    configurations = [
        ("technical", dict(use_sentiment=False, use_announcements=False,
                           use_guba=False, use_comment=False,
                           use_margin=False, use_northbound=False)),
        ("+ sentiment", dict(use_sentiment=True, use_announcements=False,
                             use_guba=False, use_comment=False,
                             use_margin=False, use_northbound=False)),
        ("+ guba", dict(use_sentiment=False, use_announcements=False,
                        use_guba=True, use_comment=False,
                        use_margin=False, use_northbound=False)),
        ("+ comment", dict(use_sentiment=False, use_announcements=False,
                           use_guba=False, use_comment=True,
                           use_margin=False, use_northbound=False)),
        ("+ margin", dict(use_sentiment=False, use_announcements=False,
                          use_guba=False, use_comment=False,
                          use_margin=True, use_northbound=False)),
        ("+ northbound", dict(use_sentiment=False, use_announcements=False,
                              use_guba=False, use_comment=False,
                              use_margin=False, use_northbound=True)),
        ("+ margin+nb", dict(use_sentiment=False, use_announcements=False,
                             use_guba=False, use_comment=False,
                             use_margin=True, use_northbound=True)),
        ("ALL", dict(use_sentiment=True, use_announcements=True,
                     use_guba=True, use_comment=True,
                     use_margin=True, use_northbound=True)),
    ]

    # --- Determine model types to evaluate ---
    model_types = ["xgb", "lgbm"] if args.model == "all" else [args.model]
    codes_list = list(all_data.keys())

    selector = FeatureSelector(mi_k=200, sfs_k=50, model_type="lgbm") if args.feature_select else None

    for mtype in model_types:
        logger.info("=" * 60)
        logger.info("MODEL: %s (feature_select=%s)", mtype.upper(), args.feature_select)
        all_results = {}  # config_name -> list of per-stock MCCs

        for label, kwargs in configurations:
            logger.info("CONFIG: %s", label)
            pipe = FeaturePipeline(
                seq_len=cfg.features.seq_len,
                use_technical=True, use_scoring=True, use_temporal=True,
                **kwargs,
            )
            per_stock_mccs = []
            for i, code in enumerate(codes_list):
                t0 = time.time()
                metrics = train_and_eval(pipe, all_data[code], splitter, model_type=mtype, selector=selector)
                dt = time.time() - t0
                mcc = metrics["mcc"] if metrics else None
                if mcc is not None:
                    per_stock_mccs.append(mcc)
                    logger.info("  [%d/%d] %s: MCC=%.4f (%.1fs)",
                                i + 1, len(codes_list), code, mcc, dt)
                else:
                    logger.info("  [%d/%d] %s: no result (%.1fs)",
                                i + 1, len(codes_list), code, dt)
            all_results[label] = per_stock_mccs
            mean, lo, hi = bootstrap_ci(per_stock_mccs)
            logger.info("  => MCC=%.4f [%.4f, %.4f] (n=%d stocks)",
                        mean, lo, hi, len(per_stock_mccs))

        # --- Baseline for deltas ---
        baseline_mccs = all_results.get("technical", [])
        if not baseline_mccs:
            logger.error("No baseline results for %s — skipping", mtype)
            continue

        # --- Summary with bootstrap CIs ---
        print("\n" + "=" * 80)
        print(f"{mtype.upper()} ABLATION SUMMARY — 95% BOOTSTRAP CI")
        print(f"({len(codes_list)} stocks, {MAX_WINDOWS} windows, {N_BOOTSTRAP} bootstrap samples)")
        print("=" * 80)
        print(f"{'Configuration':<20} {'MCC':>8} {'95% CI':>22} {'Δ MCC':>8} {'Δ 95% CI':>22}")
        print("-" * 86)

        base_mean, base_lo, base_hi = bootstrap_ci(baseline_mccs)
        config_order = ["technical", "+ sentiment", "+ guba", "+ comment",
                        "+ margin", "+ northbound", "+ margin+nb", "ALL"]
        for label in config_order:
            mccs = all_results.get(label, [])
            if not mccs:
                continue
            mean, lo, hi = bootstrap_ci(mccs)
            if label == "technical":
                delta_str = "—"
                delta_ci_str = "—"
            else:
                delta_mean, delta_lo, delta_hi = bootstrap_delta_ci(baseline_mccs, mccs)
                delta_str = f"{delta_mean:+.4f}"
                delta_ci_str = f"[{delta_lo:+.4f}, {delta_hi:+.4f}]"
            print(f"{label:<20} {mean:8.4f} [{lo:.4f}, {hi:.4f}]   {delta_str:>8} {delta_ci_str:>22}")

        # --- Significance summary ---
        print("\n--- Statistical Significance ---")
        for label in ["+ sentiment", "+ guba", "+ comment",
                       "+ margin", "+ northbound", "+ margin+nb", "ALL"]:
            mccs = all_results.get(label, [])
            if not mccs:
                continue
            delta_mean, delta_lo, delta_hi = bootstrap_delta_ci(baseline_mccs, mccs)
            sig = "SIGNIFICANT" if delta_lo > 0 else ("NEGATIVE" if delta_hi < 0 else "not significant")
            print(f"  {label:<20}: Δ={delta_mean:+.4f} 95% CI [{delta_lo:+.4f}, {delta_hi:+.4f}] — {sig}")

        # --- Per-stock best config ---
        print("\n--- Best Config Per Stock ---")
        config_names = list(all_results.keys())
        best_counts = {k: 0 for k in config_names}
        for i, code in enumerate(codes_list):
            stock_results = {}
            for name, mccs in all_results.items():
                if i < len(mccs):
                    stock_results[name] = mccs[i]
            if stock_results:
                best = max(stock_results, key=stock_results.get)
                best_counts[best] += 1
        for name in config_names:
            print(f"  {name}: {best_counts[name]} stocks")

        # --- Per-stock deltas for guba ---
        print("\n--- Per-Stock Guba Delta vs Baseline ---")
        guba_mccs = all_results.get("+ guba", [])
        for i, code in enumerate(codes_list):
            bl = baseline_mccs[i] if i < len(baseline_mccs) else None
            gb = guba_mccs[i] if i < len(guba_mccs) else None
            if bl is not None and gb is not None:
                delta = gb - bl
                bar = "+" if delta > 0 else "-"
                print(f"  {code}: {bl:.4f} → {gb:.4f} (Δ{delta:+.4f}) {bar * max(1, int(abs(delta)*50))}")


if __name__ == "__main__":
    main()
