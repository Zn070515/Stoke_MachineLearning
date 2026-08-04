"""Panel baselines benchmark (review v8 四-2).

The VSN+xLSTM panel model is complex; without simple reference points there is
no way to tell whether the complexity earns its keep.  This script trains
Linear Ridge / LightGBM / MLP / naive-momentum on the SAME prebuilt panel
features and the SAME non-overlapping walk-forward folds as train_panel.py, then
scores each with the SAME ``evaluate_portfolio`` sleeve-account evaluator.

- Fold boundaries are imported from train_panel.py (same module), so the two
  runs are aligned by construction, not by copy.
- Feature contract is point-in-time: a sample entering at day ``e`` uses the
  feature cross-section at ``e-1`` (the decision column the xLSTM also sees).
- The naive "past return mean" baseline is the floor the review names: if the
  complex model cannot beat it, the complexity adds no value.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/train_baselines_panel.py \
      --prebuilt data/features_panel --stocks 200 --max-folds 2 \
      --models ridge,lgbm,mlp,momentum --outdir reports/experiments/baselines
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Import fold helpers from train_panel.py so boundaries stay identical.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stoke_ml.config import load_config
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.models.baseline.panel_baselines import (
    FittedScoreAdapter,
    PrecomputedScoreAdapter,
    ScaledPredictor,
    build_flat_samples,
    build_momentum_grid,
)
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.evaluate import EVALUATOR_VERSION, evaluate_portfolio
from train_panel import (  # noqa: E402  (same-module fold logic)
    _cross_sectional_normalize,
    _discover_stocks,
    _experiment_version,
    _fmt_date,
    _fold_eligible_stocks,
    _mask_stocks,
    _prebuilt_channel_coverage,
    _resolve_universe,
    _slice_panel,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class _LGBMWrapper:
    """LightGBM regression booster with a sklearn-like fit/predict surface."""

    _PARAMS = {
        "objective": "regression",
        "metric": "l2",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.3,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "min_data_in_leaf": 100,
        "verbosity": -1,
        "n_jobs": -1,
    }

    def __init__(self, seed: int):
        self._params = dict(self._PARAMS)
        self._params["seed"] = seed
        self._bst = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import lightgbm as lgb

        self._bst = lgb.train(
            self._params, lgb.Dataset(X, label=y), num_boost_round=150
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._bst is None:
            raise RuntimeError("not fitted")
        it = self._bst.best_iteration or 150
        return self._bst.predict(X, num_iteration=it)


def make_model(name: str, seed: int):
    """Construct the sklearn-style regressor for a baseline name."""
    if name == "ridge":
        return Ridge(alpha=10.0, solver="lsqr")
    if name == "lgbm":
        return _LGBMWrapper(seed)
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-3,
            batch_size=256,
            max_iter=25,
            random_state=seed,
            early_stopping=False,
        )
    raise ValueError(f"unknown baseline model: {name}")


def main():
    parser = argparse.ArgumentParser(description="Panel baselines benchmark (v8 四-2)")
    parser.add_argument("--prebuilt", type=str, default="data/features_panel",
                        help="Panel-mode prebuilt features dir (same as train_panel.py)")
    parser.add_argument("--stocks", type=int, default=200,
                        help="Universe size / cap (default: 200)")
    parser.add_argument("--universe", type=str, default="random",
                        choices=["first", "random", "stratified", "all",
                                 "csi300", "csi500", "csi800"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=str, default="2000-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--max-folds", type=int, default=2,
                        help="Limit number of walk-forward folds (default: 2)")
    parser.add_argument("--lockbox-months", type=int, default=12)
    parser.add_argument("--models", type=str, default="ridge,lgbm,mlp,momentum",
                        help="Comma-separated baselines to run")
    parser.add_argument("--max-train-rows", type=int, default=100000,
                        help="Cap on baseline training rows per fold (benchmark "
                             "memory guard; biases toward earlier train columns only)")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Override seq_len (default: 60)")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.end is None:
        args.end = datetime.now().strftime("%Y-%m-%d")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    cfg = load_config()
    data_dir = cfg.project.data_dir

    all_stocks = _discover_stocks(data_dir, None)
    stock_list, universe_desc = _resolve_universe(
        all_stocks, args.universe, args.stocks, args.seed, data_dir,
    )
    if not stock_list:
        logger.error("No stocks found")
        sys.exit(1)
    universe_resolved = list(stock_list)
    universe_used = list(stock_list)

    logger.info("Universe: %s", universe_desc)
    logger.info("Loading K-line data for %d stocks from %s to %s...",
                len(stock_list), args.start, args.end)

    from stoke_ml.data.storage import DataStorage
    ds = DataStorage(data_dir)
    frames = []
    for code in stock_list:
        df = ds.load_daily(code, args.start, args.end)
        if df is not None and not df.empty:
            df["stock_code"] = code
            frames.append(df)
    if not frames:
        logger.error("No data loaded for any stock")
        sys.exit(1)

    panel = pd.concat(frames, ignore_index=True)
    panel_stocks = sorted(panel["stock_code"].unique())

    seq_len = args.seq_len or 60
    fp = FeaturePipeline(
        seq_len=seq_len,
        use_sentiment=True, use_announcements=True,
        use_guba=True, use_comment=True, use_margin=True,
        use_northbound=True, use_dragon_tiger=True,
        use_fundamental=True, use_etf_flow=True,
        use_capital_flow=True, use_block_trade=True,
        use_shareholder=True, use_lockup=True, use_dividend=True,
        use_valuation=True,
        use_board=False, use_sector=False, use_concept=False,
    )
    panel_data = fp.build_panel_features(
        panel, aux_data=None, horizon=args.horizon, prebuilt_dir=args.prebuilt,
    )
    channel_manifest = _prebuilt_channel_coverage(panel_data)

    global_dates = panel_data.get("global_dates")
    n_stocks = panel_data["static_features"].shape[0]
    n_timesteps = panel_data["past_known"].shape[1]
    static_dim = panel_data["static_features"].shape[2]
    pk_dim = panel_data["past_known"].shape[2]
    po_dim = panel_data["past_observed"].shape[2]
    logger.info("Panel data: %d stocks × %d timesteps  dims S=%d PK=%d PO=%d "
                "horizon=%d", n_stocks, n_timesteps, static_dim, pk_dim, po_dim,
                args.horizon)

    config = PanelConfig(
        seq_len=seq_len,
        static_dim=static_dim,
        past_known_dim=pk_dim,
        past_observed_dim=po_dim,
        horizon=args.horizon,
        seed=args.seed,
    )

    version_info = _experiment_version(
        data_dir, universe_used, args.prebuilt,
        static_dim, pk_dim, po_dim, config, args.start, args.end, args.seed,
    )
    logger.info("Version freeze: commit=%s data=%s feat=%s uni=%s cal=%s",
                version_info["git_commit"][:10],
                version_info["data_manifest_hash"],
                version_info["feature_schema_hash"],
                version_info["universe_hash"],
                version_info["calendar_version"])

    val_len = 126
    step = val_len
    purge = config.seq_len
    lockbox_len = int(args.lockbox_months * 21)
    lockbox_start = max(0, n_timesteps - lockbox_len)
    if lockbox_start <= 0:
        logger.error("Lockbox (%d steps) leaves no trainable panel (n=%d)",
                     lockbox_len, n_timesteps)
        sys.exit(1)

    outdir = args.outdir or os.path.join(
        "reports", "experiments", "baselines_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(outdir, exist_ok=True)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    summary_rows: list[dict] = []
    fold_records: list[dict] = []

    last_val_start = n_timesteps - val_len - lockbox_len
    val_start = last_val_start
    fold = 0
    while val_start >= 0:
        if args.max_folds and fold >= args.max_folds:
            break
        train_end = val_start - purge
        if train_end < config.seq_len + 1:
            break
        fold += 1
        val_end = min(val_start + val_len, n_timesteps)
        val_context_start = val_start - config.seq_len

        fold_eligible = _fold_eligible_stocks(panel_data, train_end)
        fold_stocks = [panel_stocks[i] for i in np.where(fold_eligible)[0]]
        if len(fold_stocks) < 20:
            logger.warning("Fold %d: only %d stocks eligible PIT (need >= 20) — "
                           "skipping fold", fold, len(fold_stocks))
            val_start -= step
            continue

        inner_train_data = _mask_stocks(
            _slice_panel(panel_data, slice(0, train_end), price_pad=config.horizon),
            fold_eligible,
        )
        outer_test_data = _mask_stocks(
            _slice_panel(panel_data, slice(val_context_start, val_end),
                         price_pad=config.horizon),
            fold_eligible,
        )

        # Same target normalization as train_panel.py: cross-sectional z-score
        # per date (clean open-to-open returns), then clip to [-5, 5].
        inner_train_data["y_return"] = _cross_sectional_normalize(
            inner_train_data["y_return"], inner_train_data["return_target_mask"])
        outer_test_data["y_return"] = _cross_sectional_normalize(
            outer_test_data["y_return"], outer_test_data["return_target_mask"])
        for dd in (inner_train_data, outer_test_data):
            np.clip(dd["y_return"], -5.0, 5.0, out=dd["y_return"])

        n_windows = outer_test_data["past_known"].shape[1] - config.seq_len
        logger.info("Fold %d/%d: inner_train [0:%d] outer_test [%d:%d] "
                    "(%d eligible stocks, %d entry days)",
                    fold, args.max_folds or "∞", train_end,
                    val_context_start, val_end, len(fold_stocks), n_windows)

        for model_name in models:
            t0 = time.time()
            if model_name == "momentum":
                grid = build_momentum_grid(
                    outer_test_data, config.seq_len, config.seq_len + n_windows,
                    config.seq_len)
                adapter = PrecomputedScoreAdapter(grid)
            else:
                # PIT label gate: y_return[:, e] is realized at open[e+horizon],
                # so a training sample is usable only when its label is fully
                # known by the fold cutoff train_end (e + horizon <= train_end-1,
                # i.e. entry_end = train_end - horizon).  train_panel.py's inner
                # train ends at inner_train_end = train_end - inner_val_len -
                # seq_len, which guarantees the same property — without this
                # gate the baseline would train on labels unfolding into the
                # purge gap right before the test window.
                Xtr, ytr = build_flat_samples(
                    inner_train_data, config.seq_len, train_end - config.horizon,
                    max_rows=args.max_train_rows)
                if len(ytr) < 100:
                    logger.warning("  fold %d %s: only %d training rows — skipping",
                                   fold, model_name, len(ytr))
                    continue
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr)
                model = make_model(model_name, args.seed)
                model.fit(Xtr_s, ytr)
                adapter = FittedScoreAdapter(ScaledPredictor(model, scaler))
            adapter.reset()
            outer_m = evaluate_portfolio(
                adapter, outer_test_data, config, device,
                horizon=config.horizon,
                top_fraction=0.1,
                raw_returns=outer_test_data["realized_return"],
                require_price_path=True,
                n_boot=500,
            )
            elapsed = time.time() - t0
            rec = {
                "model": model_name,
                "fold": fold,
                "train_rows": len(ytr) if model_name != "momentum" else 0,
                "ls_sharpe": outer_m.get("ls_sharpe", 0.0),
                "ls_gross_sharpe": outer_m.get("ls_gross_sharpe", 0.0),
                "ic_mean": outer_m.get("ic_mean", 0.0),
                "ic_ir": outer_m.get("ic_ir", 0.0),
                "long_sharpe": outer_m.get("long_sharpe", 0.0),
                "q5mq1_ret": outer_m.get("q5mq1_ret", 0.0),
                "eligible_ew_sharpe": outer_m.get("eligible_ew_sharpe", 0.0),
                "universe_ew_sharpe": outer_m.get("universe_ew_sharpe", 0.0),
                "entry_start": _fmt_date(global_dates, val_start),
                "entry_end": _fmt_date(global_dates, val_start + val_len - 1),
                "seconds": round(elapsed, 1),
            }
            fold_records.append(rec)
            logger.info(
                "  %-8s fold %d | LS_Sharpe=%.2f IC=%.4f Long=%.2f Q5-Q1=%.1fbp "
                "ElgEW=%.2f UniEW=%.2f (%.1fs)",
                model_name, fold, rec["ls_sharpe"], rec["ic_mean"],
                rec["long_sharpe"], rec["q5mq1_ret"] * 10000,
                rec["eligible_ew_sharpe"], rec["universe_ew_sharpe"], elapsed,
            )
        val_start -= step

    if not fold_records:
        logger.error("No baseline fold completed")
        sys.exit(1)

    rows = pd.DataFrame(fold_records)
    rows.to_csv(os.path.join(outdir, "baseline_table.csv"), index=False)

    summary = {
        "version": version_info,
        "universe": universe_desc,
        "models": models,
        "n_folds": int(rows["fold"].nunique()),
        "eval_note": (
            "Same prebuilt features + same non-overlapping folds + same "
            "evaluate_portfolio sleeve account as train_panel.py; baselines "
            "train on the full [0, train_end) span (no inner_val carve-out)."
        ),
        "models_result": {},
    }
    for model_name in models:
        sub = rows[rows["model"] == model_name]
        if sub.empty:
            continue
        def _agg(col):
            vals = sub[col].dropna().astype(float)
            return {
                "mean": float(vals.mean()) if len(vals) else None,
                "std": float(vals.std()) if len(vals) > 1 else None,
                "n": int(len(vals)),
            }
        summary["models_result"][model_name] = {
            "ls_sharpe": _agg("ls_sharpe"),
            "ic_mean": _agg("ic_mean"),
            "long_sharpe": _agg("long_sharpe"),
            "q5mq1_ret": _agg("q5mq1_ret"),
            "eligible_ew_sharpe": _agg("eligible_ew_sharpe"),
            "universe_ew_sharpe": _agg("universe_ew_sharpe"),
        }

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(outdir, "universe_resolved.txt"), "w", encoding="utf-8") as f:
        f.write(f"# {universe_desc}\n# n={len(universe_resolved)}\n")
        f.write("\n".join(universe_resolved))
        f.write("\n")

    logger.info("=== Baseline Summary (%d folds, disjoint OOS windows) ===",
                summary["n_folds"])
    header = f"{'model':<10} {'LS_sharpe':>10} {'IC':>8} {'long_sharpe':>12} {'q5mq1(bp)':>10} {'elgEW':>8} {'uniEW':>8}"
    logger.info(header)
    for model_name in models:
        r = summary["models_result"].get(model_name)
        if not r:
            continue
        logger.info("%-10s %10.2f %8.4f %12.2f %10.1f %8.2f %8.2f",
                    model_name,
                    r["ls_sharpe"]["mean"] or 0.0,
                    r["ic_mean"]["mean"] or 0.0,
                    r["long_sharpe"]["mean"] or 0.0,
                    (r["q5mq1_ret"]["mean"] or 0.0) * 10000,
                    r["eligible_ew_sharpe"]["mean"] or 0.0,
                    r["universe_ew_sharpe"]["mean"] or 0.0)
    logger.info("Artifacts saved to %s", outdir)


if __name__ == "__main__":
    main()
