"""Label type benchmark — absolute direction vs sector-relative outperformance.

Compares 2 label types:
  1. abs  — price[t+1] > price[t]  (current default)
  2. rel  — stock_return > sector_median_return

Uses return_dates from FeaturePipeline to align sector-relative labels exactly
with feature samples (avoiding the off-by-NaN-drop mismatch).
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.stock_sector_mapper import StockSectorMapper
from stoke_ml.data.xueqiu_storage import XueqiuStorage
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import compute_classification_metrics
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _build_sector_return_map(codes, date_start, date_end, data_dir):
    """Build a per-date sector median return map from panel data.

    Returns dict: {(date, sector): median_return}
    """
    storage = DataStorage(data_dir)
    mapper = StockSectorMapper()

    close_frames = []
    sectors = {}
    for code in codes:
        df = storage.load_daily(code, date_start, date_end)
        if df.empty or len(df) < 120:
            continue
        try:
            sec = mapper.get_sector(code) or "未知"
        except Exception:
            sec = "未知"
        sectors[code] = sec
        close_frames.append(df[["date", "close"]].assign(stock_code=code))

    if not close_frames:
        return {}, sectors

    panel = pd.concat(close_frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["sector"] = panel["stock_code"].map(sectors)
    panel = panel.sort_values(["date", "stock_code"])

    # Daily return per stock
    panel["ret"] = panel.groupby("stock_code")["close"].pct_change()
    panel = panel.dropna(subset=["ret"])

    # Sector median return per date (key: (date_obj, sector_str))
    panel["date_obj"] = panel["date"].dt.date
    sector_med = panel.groupby(["date_obj", "sector"])["ret"].median()

    logger.info("Sector map sample keys: %s",
                list(sector_med.index[:5]))
    return sector_med.to_dict(), sectors


def _compute_rel_labels(y_abs, dates, sector_map, sector_of_code):
    """Replace absolute labels with sector-relative labels.

    y_abs: original binary labels from pipeline
    dates: prediction dates from pipeline (same length as y_abs)
    sector_map: {(date, sector): median_return}
    sector_of_code: sector name for this stock
    """
    y_rel = y_abs.copy()
    for i, d in enumerate(dates):
        key = (pd.Timestamp(d).date(), sector_of_code)
        # If sector median return > 0, stock needs to beat it
        if key in sector_map:
            med_ret = sector_map[key]
            # y_rel = 1 if the stock outperformed sector median
            # y_abs[i] = 1 means stock went up (close[t+1] > close[t])
            # We need actual return to compare, but all we have is up/down
            # Instead: y_rel = 1 if (stock_up AND sector_down) OR (stock_up > sector_avg_up)
            # Simplification: treat sector median > 0 as "sector up", then
            # y_rel = 1 if stock_up > sector_up
            # Since y_abs is binary up/down, approximate:
            #   if sector went up and stock went up → y_rel = 1 (outperform)
            #   if sector went down and stock went up → y_rel = 1 (outperform)
            #   if sector went up and stock went down → y_rel = 0
            #   if sector went down and stock went down → y_rel = 0
            # This is just: y_rel = y_abs (same!) when sector-neutral
            # Better: use continuous labels from close data
            pass
    return y_rel


def main():
    parser = argparse.ArgumentParser(description="Label type benchmark")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--stocks", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir
    storage = DataStorage(data_dir)
    date_start = cfg.markets.a_shares.start_date
    date_end = datetime.now().strftime("%Y-%m-%d")

    # ---- Stock selection ----
    all_codes = sorted([
        f.replace(".parquet", "")
        for f in os.listdir(os.path.join(data_dir, "a_shares", "daily"))
        if f.endswith(".parquet")
    ])
    if args.stocks and args.stocks < len(all_codes):
        step = max(len(all_codes) // args.stocks, 1)
        codes = [all_codes[i * step] for i in range(args.stocks)]
    else:
        codes = all_codes

    # ---- Build sector return map ----
    logger.info("Building sector return map for %d stocks...", len(codes))
    sector_ret_map, stock_sectors = _build_sector_return_map(
        codes, date_start, date_end, data_dir
    )
    logger.info("Sector map: %d entries, %d stocks with sectors",
                len(sector_ret_map), len(stock_sectors))

    # ---- Pipeline ----
    pipeline = FeaturePipeline(
        seq_len=cfg.features.get("flat_seq_len", cfg.features.seq_len),
        horizon=cfg.features.target_horizon,
        flat_mode=True,
        use_technical=cfg.features.technical_indicators,
        use_scoring=cfg.features.rule_based_scoring,
        use_temporal=cfg.features.temporal_features,
        use_sentiment=True, use_announcements=False,
        use_guba=True, use_comment=False, use_xueqiu=True,
        use_margin=False, use_northbound=False,
        use_dragon_tiger=False, use_fundamental=False,
        use_etf_flow=False, use_interaction=False,
    )

    model_params = dict(cfg.model.params)
    model_params["n_estimators"] = 50
    model_params["max_depth"] = 4
    splitter = WalkForwardSplitter(
        train_years=cfg.training.validation.train_years,
        val_months=cfg.training.validation.val_months,
        step_months=6,
    )

    all_results = []
    stock_count = 0

    for code in codes:
        if code not in stock_sectors:
            continue

        df = storage.load_daily(code, date_start, date_end)
        if df.empty or len(df) < 200:
            continue

        # Load aux data
        aux = {}
        for key, loader_fn in [
            ("sentiment", lambda c=code: _safe_load(
                NewsStorage(data_dir).load_daily_sentiment, c, date_start, date_end)),
            ("xueqiu", lambda c=code: _safe_load(
                XueqiuStorage(data_dir).load_daily_sentiment, c, date_start, date_end)),
            ("guba", lambda c=code: _safe_load(
                GubaStorage(data_dir).load_daily_sentiment, c, date_start, date_end)),
        ]:
            try:
                d = loader_fn()
                aux[key] = d if (isinstance(d, pd.DataFrame) and not d.empty) else None
            except Exception:
                aux[key] = None

        # Build features with dates
        try:
            X, y_abs, _, dates = pipeline.build_features(
                df, sentiment_df=aux.get("sentiment"),
                guba_df=aux.get("guba"), xueqiu_df=aux.get("xueqiu"),
                return_dates=True,
            )
        except Exception as e:
            logger.warning("  %s features: FAILED — %s", code, e)
            continue

        if len(X) == 0 or len(dates) == 0:
            continue

        # Ensure alignment of dates and labels
        min_len = min(len(y_abs), len(dates))
        y_abs = y_abs[:min_len]
        dates = dates[:min_len]

        # Build sector-relative labels using close data and dates
        sector = stock_sectors[code]
        y_rel = np.zeros(min_len, dtype=int)
        horizon = cfg.features.target_horizon

        # Get close prices aligned with dates
        close_series = df.set_index("date")["close"]
        date_idx = pd.DatetimeIndex(dates)

        for i, d in enumerate(date_idx):
            # d is the prediction date. We need close[d+horizon] vs close[d]
            try:
                if d not in close_series.index:
                    y_rel[i] = y_abs[i]
                    continue
                close_t = close_series.loc[d]
                # Find close at d+horizon trading days later
                future_closes = close_series.loc[d:]
                if len(future_closes) > horizon:
                    close_t_plus = future_closes.iloc[horizon]
                    stock_ret = (close_t_plus - close_t) / close_t

                    # Sector median return for this date
                    key = (d.date(), sector)
                    if key in sector_ret_map:
                        med_ret = sector_ret_map[key]
                        y_rel[i] = 1 if stock_ret > med_ret else 0
                    else:
                        y_rel[i] = y_abs[i]  # fallback to absolute
                else:
                    y_rel[i] = y_abs[i]
            except Exception:
                y_rel[i] = y_abs[i]

        # Balance check
        rel_pos = y_rel.mean()
        abs_pos = y_abs.mean()
        if 0.0 < rel_pos < 1.0:
            n_samples = len(y_abs)
            pseudo_dates = pd.date_range("2000-01-01", periods=n_samples, freq="B")
            folds = list(splitter.split(pseudo_dates))[:5]

            for label_name, Y in [("abs", y_abs), ("rel", y_rel)]:
                for fold_idx, (train_idx, val_idx) in enumerate(folds):
                    if train_idx[-1] >= n_samples or val_idx[-1] >= n_samples:
                        break

                    Y_train = Y[train_idx]
                    Y_val = Y[val_idx]

                    if len(np.unique(Y_train)) < 2 or len(np.unique(Y_val)) < 2:
                        continue

                    try:
                        model = XGBoostBaseline(**model_params)
                        model.fit(X[train_idx], Y_train)
                        preds = model.predict(X[val_idx])
                        metrics = compute_classification_metrics(Y_val, preds)
                    except Exception as e:
                        logger.warning("  %s %s fold %d: FAILED — %s",
                                       code, label_name, fold_idx, e)
                        continue

                    all_results.append({
                        "stock": code,
                        "label": label_name,
                        "fold": fold_idx,
                        **metrics,
                    })

        stock_count += 1
        if stock_count % 10 == 0:
            logger.info("  [%d/%d] %s done (abs_pos=%.2f, rel_pos=%.2f)",
                        stock_count, len(codes), code, abs_pos, rel_pos)
        else:
            logger.debug("  [%d] %s: abs_pos=%.2f rel_pos=%.2f",
                         stock_count, code, abs_pos, rel_pos)

    # ==================================================================
    if not all_results:
        logger.error("No results — check data and label alignment")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    logger.info("\n%s", "=" * 64)
    logger.info("LABEL TYPE BENCHMARK (%d stocks)", stock_count)
    logger.info("%s", "=" * 64)

    summary = results_df.groupby("label").agg(
        mcc_mean=("mcc", "mean"),
        mcc_std=("mcc", "std"),
        acc_mean=("accuracy", "mean"),
        n_folds=("mcc", "count"),
    ).round(4)
    logger.info("\n%s", summary.to_string())

    if "abs" in summary.index and "rel" in summary.index:
        delta = summary.loc["rel", "mcc_mean"] - summary.loc["abs", "mcc_mean"]
        logger.info("\nDelta (rel - abs): %+.4f", delta)
        best = "rel" if delta > 0 else "abs"
        logger.info("Best: %s", best)

    output_dir = args.output or cfg.project.model_dir
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "label_benchmark.csv")
    results_df.to_csv(out_path, index=False)
    logger.info("\nSaved to %s", out_path)


def _safe_load(loader_fn, code, start, end):
    try:
        return loader_fn(code, start, end)
    except Exception:
        return pd.DataFrame()


if __name__ == "__main__":
    main()
