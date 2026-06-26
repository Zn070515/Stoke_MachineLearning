"""Batch LSTM evaluation across multiple stocks — one script, clean output."""
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
from stoke_ml.models.dl.dataset import StockDataset
from stoke_ml.models.dl.lightning_module import StockLightningModule
from stoke_ml.models.dl.lstm_model import LSTMModel
import torch, pytorch_lightning as pl
from torch.utils.data import DataLoader
import logging
logging.getLogger('pytorch_lightning').setLevel(logging.ERROR)
logging.getLogger('lightning').setLevel(logging.ERROR)
logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
os.environ['LIGHTNING_SUPPRESS_LOGS'] = '1'

STOCKS = [
    '000001',  # 银行
    '600519',  # 白酒
    '000725',  # 科技
    '600276',  # 医药
    '000651',  # 家电
    '601318',  # 保险
    '600900',  # 电力
    '002415',  # 科技(海康)
    '000858',  # 白酒(五粮液)
    '600036',  # 银行(招行)
    '002594',  # 汽车(比亚迪)
    '601088',  # 煤炭(神华)
    '300750',  # 电池(宁德时代)
    '688981',  # 半导体(中芯国际)
    '002493',  # 化工(荣盛石化)
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
pipeline = FeaturePipeline(seq_len=60, horizon=1, flat_mode=False)
splitter = WalkForwardSplitter(train_years=2, val_months=3)
date_end = datetime.now().strftime('%Y-%m-%d')

results = []
for code in STOCKS:
    print(f'[{code}] Loading...', end=' ', flush=True)
    df = storage.load_daily(code, '2015-01-01', date_end)
    if df.empty:
        print('SKIP (no data)')
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
    n_features = X.shape[2]
    n_samples = len(X)
    news_days = sentiment_df['has_news'].sum() if not sentiment_df.empty else 0
    print(f'X={X.shape} f={n_features} news={news_days}', flush=True)

    pseudo_dates = pd.date_range('2000-01-01', periods=n_samples, freq='B')
    folds_list = list(splitter.split(pseudo_dates))
    stock_mccs = []

    for train_idx, val_idx in folds_list:
        if train_idx[-1] >= n_samples or val_idx[-1] >= n_samples:
            break
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        class_weight = [1.0, n_neg / n_pos] if n_pos > 0 and n_neg > 0 else None

        model = LSTMModel(input_dim=n_features, hidden_dim=128, num_layers=2, dropout=0.3)
        lit = StockLightningModule(model=model, learning_rate=0.001,
                                   class_weight=class_weight, use_scheduler=False)
        ckpt = pl.callbacks.ModelCheckpoint(
            dirpath='/tmp/batch_ckpt', filename='fold', monitor='val_mcc',
            mode='max', save_top_k=1)
        es = pl.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')
        train_ds = StockDataset(X_train, y_train)
        val_ds = StockDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
        trainer = pl.Trainer(
            max_epochs=20, accelerator='auto', devices=1,
            callbacks=[ckpt, es],
            enable_progress_bar=False, enable_model_summary=False,
        )
        trainer.fit(lit, train_loader, val_loader)
        best_path = ckpt.best_model_path
        if best_path:
            best = StockLightningModule.load_from_checkpoint(best_path, weights_only=False)
        else:
            best = lit
        best.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        best.to(device)
        all_preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                logits = best(xb.to(best.device))
                all_preds.append(torch.argmax(logits, -1).cpu().numpy())
        val_preds = np.concatenate(all_preds)
        stock_mccs.append(mcc_score(y_val, val_preds))

    mean_mcc = np.mean(stock_mccs)
    std_mcc = np.std(stock_mccs)
    n_folds = len(stock_mccs)
    sector_name = sector or 'unknown'
    print(f'  => MCC={mean_mcc:+.4f} +/- {std_mcc:.4f} ({n_folds} folds) [{sector_name}]', flush=True)
    results.append({
        'code': code, 'sector': sector_name, 'n_features': n_features,
        'n_samples': n_samples, 'news_days': news_days,
        'mcc_mean': mean_mcc, 'mcc_std': std_mcc, 'n_folds': n_folds,
    })

print()
print('=' * 72)
print(f'{"Stock":<10} {"Sector":<8} {"Features":>8} {"News":>6} {"MCC":>8} {"±Std":>8} {"Folds":>6}')
print('-' * 72)
for r in sorted(results, key=lambda x: x['mcc_mean'], reverse=True):
    print(f'{r["code"]:<10} {r["sector"]:<8} {r["n_features"]:>8} {r["news_days"]:>6} '
          f'{r["mcc_mean"]:>+8.4f} {r["mcc_std"]:>8.4f} {r["n_folds"]:>6}')

if results:
    avg = np.mean([r['mcc_mean'] for r in results])
    print('-' * 72)
    print(f'{"AVERAGE":<10} {"":<8} {"":>8} {"":>6} {avg:>+8.4f}')
print('Done')
