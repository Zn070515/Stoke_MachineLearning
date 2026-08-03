"""Trace pct_change through load_daily -> _engineer_features."""
import os

from stoke_ml.config import load_config
from stoke_ml.data.storage import DataStorage
from stoke_ml.features.pipeline import FeaturePipeline

cfg = load_config()
storage = DataStorage(cfg.project.data_dir)
print("data_dir:", cfg.project.data_dir)

df = storage.load_daily("000001", cfg.markets.a_shares.start_date, "2026-07-31")
print("load_daily rows:", len(df), "cols:", len(df.columns))
m = df["date"] >= "2026-06-18"
if "pct_change" in df.columns:
    print("load_daily pct_change zero in 06-18+ window:",
          int((df.loc[m, "pct_change"] == 0).sum()), "/", int(m.sum()))
    print(df.loc[m, ["date", "pct_change"]].tail(6).to_string())
else:
    print("load_daily has NO pct_change column")

pipe = FeaturePipeline(
    seq_len=cfg.features.seq_len, horizon=cfg.features.target_horizon,
    flat_mode=False,
    use_technical=True, use_scoring=True, use_temporal=True,
    use_sentiment=True, use_guba=True, use_comment=True,
    use_limit_up=False, use_pledge=True, use_market_env=True,
    use_market_env_refine=True, use_index_membership=True,
)
feats = pipe._engineer_features(df)
print("\nfeats rows:", len(feats), "cols:", len(feats.columns))
print("pct_change in feats:", "pct_change" in feats.columns)
if "pct_change" in feats.columns:
    m2 = feats["date"] >= "2026-06-18"
    print("feats pct_change zero in 06-18+ window:",
          int((feats.loc[m2, "pct_change"] == 0).sum()), "/", int(m2.sum()))
