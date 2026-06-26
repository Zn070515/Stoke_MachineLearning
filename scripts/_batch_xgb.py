"""XGBoost baseline batch evaluation — same 15 stocks as LSTM for comparison."""
import sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.data.market_wide_storage import MarketWideStorage
from stoke_ml.data.fundamental_storage import FundamentalStorage
from stoke_ml.data.etf_storage import ETFStorage
from stoke_ml.data.stock_sector_mapper import StockSectorMapper
from stoke_ml.features.pipeline import FeaturePipeline
from stoke_ml.evaluation.splitter import WalkForwardSplitter
from stoke_ml.evaluation.metrics import mcc_score
from stoke_ml.models.baseline.xgboost_model import XGBoostBaseline

STOCKS = [
    '000001', '600519', '000725', '600276', '000651', '601318',
    '600900', '002415', '000858', '600036', '002594', '601088',
    '300750', '688981', '002493',
]

cfg = load_config()
storage = DataStorage(cfg.project.data_dir)
news_storage = NewsStorage(cfg.project.data_dir)
margin_storage = MarketWideStorage(cfg.project.data_dir, 'margin')
nb_storage = MarketWideStorage(cfg.project.data_dir, 'northbound')
dt_storage = MarketWideStorage(cfg.project.data_dir, 'dragon_tiger')
fund_storage = FundamentalStorage(cfg.project.data_dir)
etf_storage = ETFStorage(cfg.project.data_dir)
sector_mapper = StockSectorMapper()
pipeline = FeaturePipeline(seq_len=5, horizon=1, flat_mode=True)
splitter = WalkForwardSplitter(train_years=2, val_months=3)
date_end = datetime.now().strftime('%Y-%m-%d')

results = []
for code in STOCKS:
    print(f'[{code}] Loading...', end=' ', flush=True)
    df = storage.load_daily(code, '2015-01-01', date_end)
    if df.empty:
        print('SKIP')
        continue
    sentiment_df = news_storage.load_daily_sentiment(code, '2015-01-01', date_end)
    margin_df = margin_storage.load(code, '2015-01-01', date_end)
    nb_df = nb_storage.load(code, '2015-01-01', date_end)
    dt_df = dt_storage.load(code, '2015-01-01', date_end)
    fundamental_df = fund_storage.forward_fill_to_daily(code, '2015-01-01', date_end)
    sector = sector_mapper.get_sector(code)
    etf_df = etf_storage.load_sector_flow(sector, '2015-01-01', date_end) if sector else pd.DataFrame()
    X, y, aligned_close = pipeline.build_features(
        df,
        sentiment_df=sentiment_df if not sentiment_df.empty else None,
        margin_df=margin_df if not margin_df.empty else None,
        northbound_df=nb_df if not nb_df.empty else None,
        dragon_tiger_df=dt_df if not dt_df.empty else None,
        fundamental_df=fundamental_df if not fundamental_df.empty else None,
        etf_flow_df=etf_df if not etf_df.empty else None,
    )
    n_features = X.shape[1]
    n_samples = len(X)
    print(f'X={X.shape} f={n_features}', flush=True)

    pseudo_dates = pd.date_range('2000-01-01', periods=n_samples, freq='B')
    folds_list = list(splitter.split(pseudo_dates))
    stock_mccs = []

    for train_idx, val_idx in folds_list:
        if train_idx[-1] >= n_samples or val_idx[-1] >= n_samples:
            break
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        model = XGBoostBaseline()
        model.fit(X_train, y_train)
        val_preds = model.predict(X_val)
        stock_mccs.append(mcc_score(y_val, val_preds))

    mean_mcc = np.mean(stock_mccs)
    std_mcc = np.std(stock_mccs)
    n_folds = len(stock_mccs)
    print(f'  => MCC={mean_mcc:+.4f} +/- {std_mcc:.4f} ({n_folds} folds) [{sector or "?"}]', flush=True)
    results.append({
        'code': code, 'sector': sector or '?', 'n_features': n_features,
        'mcc_mean': mean_mcc, 'mcc_std': std_mcc, 'n_folds': n_folds,
    })

print()
print('=' * 72)
print(f'{"Stock":<10} {"Sector":<8} {"Features":>8} {"MCC":>8} {"±Std":>8} {"Folds":>6}')
print('-' * 72)
for r in sorted(results, key=lambda x: x['mcc_mean'], reverse=True):
    print(f'{r["code"]:<10} {r["sector"]:<8} {r["n_features"]:>8} '
          f'{r["mcc_mean"]:>+8.4f} {r["mcc_std"]:>8.4f} {r["n_folds"]:>6}')
if results:
    avg = np.mean([r['mcc_mean'] for r in results])
    print('-' * 72)
    print(f'{"AVERAGE":<10} {"":<8} {"":>8} {avg:>+8.4f}')
print('Done')
