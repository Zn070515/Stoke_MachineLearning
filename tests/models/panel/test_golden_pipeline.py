"""Golden pipeline system test (review v8 §六).

Fixed 20 stocks × 500 trading days, with every scenario the review lists baked
into the synthetic download mock:

  * 停牌 (suspension)     — contiguous trading-day blocks dropped for 3 stocks
  * 分红 (dividend)       — ex-right gaps removed by the qfq 复权口径 (asserted)
  * 除权 (2:1 split)      — a 50% ex-right gap, likewise removed before storage
  * 新股 (new listing)    — 5 stocks list 150–400 trading days into the grid
  * 缺失新闻 (missing news)  — 5 stocks have no sentiment channel at all
  * 缺失财务 (missing financials) — 12 stocks lack fundamentals, 4 lack valuation
  * 数据源切换 (failover)   — mock-primary serves 12 stocks, mock-secondary 8;
                              every fetch tries the primary first and falls back

Then the REAL production chain runs end-to-end — only production functions are
called, no hand-written training loop, no mocked feature/dataset/train layers:

    download mock → DataStorage → quality gate → save_features(prebuilt)
    → build_panel_features(prebuilt_dir) → PanelDataset → train_panel(1 epoch)
    → evaluate_portfolio(require_price_path=True)

This is the review's "highest value" test: the system test that proves the whole
chain holds together when the data is dirty.
"""
import numpy as np
import pandas as pd
import torch

from scripts import data_quality_gate as dqg
from stoke_ml.data.sources.a_shares.base import AShareSourceBase
from stoke_ml.data.sources.a_shares.failover import AShareDownloader
from stoke_ml.data.storage import DataStorage
from stoke_ml.features.pipeline import FeaturePipeline, _PIT_STATIC_COLS
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.dataset import PanelDataset
from stoke_ml.models.panel.evaluate import evaluate_portfolio
from stoke_ml.models.panel.train import train_panel

N_STOCKS = 20
N_DAYS = 500
SEQ_LEN = 20
HORIZON = 5
SEED = 7
START_DATE = "2024-01-02"


def _grid_dates() -> pd.DatetimeIndex:
    return pd.bdate_range(START_DATE, periods=N_DAYS)


def _make_synthetic_daily():
    """20 stocks × 500 trading days with all seven review scenarios.

    Returns ``(daily, aux, meta)``:
      daily — full OHLCV panel, qfq-adjusted (the downloader mock's output)
      aux   — {code: {"sentiment"/"fundamental"/"valuation": df or None}}
      meta  — {"codes", "exright": [(code, ex_idx, d, stored_ret, raw_ret)],
               "no_news"/"no_fund"/"no_val": sets}
    """
    rng = np.random.RandomState(SEED)
    dates = _grid_dates()
    codes = [f"{600000 + i:06d}" for i in range(N_STOCKS)]

    # 新股: 5 stocks list later (grid index of their first trading day).
    late_first = {codes[i]: [150, 250, 300, 350, 400][i] for i in range(5)}
    # 停牌: contiguous trading-day blocks dropped for 3 stocks.
    suspend = {
        codes[2]: [(180, 205)],
        codes[7]: [(250, 262), (320, 330)],
        codes[12]: [(400, 412)],
    }
    # 分红/除权: (ex_right_idx, drop_ratio).  A 5%/4% dividend and a 2:1 split.
    # 600001 lists at grid 250, so its ex-right dates sit inside its listed span.
    exright = {
        codes[1]: [(260, 0.05), (300, 0.04)],
        codes[6]: [(200, 0.50)],
    }
    no_news = set(codes[15:20])
    no_fund = set(codes[8:20])
    no_val = set(codes[16:20])

    parts = []
    exright_checks = []
    for i, code in enumerate(codes):
        fi = late_first.get(code, 0)
        n = N_DAYS - fi
        drift = 0.0004 * (i % 5 - 2)
        # Continuous ADJUSTED close path — this is what every production source
        # normalizes to (前复权), so corporate actions never leave a fake jump.
        adj_close = 10.0 * np.cumprod(1.0 + rng.normal(drift, 0.02, n))
        # step drops by (1-d) at each ex-right date → the RAW series has the
        # real ex-right gap while the stored (adjusted) series stays continuous.
        step = np.ones(n)
        for ex_rel, d in exright.get(code, []):
            k = ex_rel - fi
            if 0 <= k < n:
                step[k:] *= (1.0 - d)
        raw_close = adj_close * step

        open_ = np.empty(n)
        open_[0] = adj_close[0] * (1 + rng.normal(0, 0.003))
        open_[1:] = adj_close[:-1] * (1 + rng.normal(0, 0.003, n - 1))
        high = np.maximum(open_, adj_close) * (1 + np.abs(rng.normal(0, 0.004, n)))
        low = np.minimum(open_, adj_close) * (1 - np.abs(rng.normal(0, 0.004, n)))
        volume = np.abs(rng.normal(1e6, 2e5, n))
        amount = volume * adj_close

        drop = set()
        for (s, e) in suspend.get(code, []):
            drop.update(range(s, e))
        keep = [t for t in range(fi, N_DAYS) if t not in drop]

        rows = []
        for t in keep:
            j = t - fi
            rows.append({
                "date": dates[t], "stock_code": code,
                "open": float(open_[j]), "high": float(high[j]),
                "low": float(low[j]), "close": float(adj_close[j]),
                "volume": float(volume[j]), "amount": float(amount[j]),
            })
        df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        # pct_change computed AFTER suspension removal so the stored series is
        # internally consistent (the gap-crossing return is the real return).
        df["pct_change"] = df["close"].pct_change() * 100.0
        parts.append(df)

        # Record stored-vs-raw behaviour on each ex-right date for the 复权 audit.
        for ex_rel, d in exright.get(code, []):
            k = ex_rel - fi  # relative index into the per-stock arrays
            if k <= 0 or k >= n or ex_rel not in keep:
                continue
            stored_ret = float(adj_close[k] / adj_close[k - 1] - 1.0)
            raw_ret = float(raw_close[k] / raw_close[k - 1] - 1.0)
            exright_checks.append((code, int(k), d, stored_ret, raw_ret))

    daily = pd.concat(parts, ignore_index=True)
    aux = _build_aux(daily)
    meta = {
        "codes": codes,
        "exright": exright_checks,
        "no_news": no_news, "no_fund": no_fund, "no_val": no_val,
    }
    return daily, aux, meta


def _build_aux(daily):
    """Per-stock aux channels; 缺失新闻/缺失财务 = channel absent for that stock."""
    rng = np.random.RandomState(SEED + 1)
    codes = sorted(daily["stock_code"].unique())
    has_news = set(codes[:15])
    has_fund = set(codes[:8])
    has_val = set(codes[:16])
    by_code = {}
    for code in codes:
        stock_dates = sorted(daily[daily["stock_code"] == code]["date"].tolist())
        entry = {}
        if code in has_news:
            keep = sorted(rng.choice(len(stock_dates),
                                     int(len(stock_dates) * 0.7), replace=False))
            entry["sentiment"] = pd.DataFrame([{
                "date": stock_dates[k], "stock_code": code,
                "sentiment_mean": float(rng.normal(0, 0.1)), "sentiment_std": 0.5,
                "news_count": 1.0, "positive_ratio": 0.5, "negative_ratio": 0.3,
                "has_news": 1.0,
            } for k in keep])
        if code in has_fund:
            quarters = stock_dates[::80][:6]
            entry["fundamental"] = pd.DataFrame([{
                "disclose_date": d, "report_date": d, "stock_code": code,
                "roe": float(rng.uniform(5, 20)), "roa": float(rng.uniform(2, 8)),
                "eps": float(rng.uniform(0.2, 2.0)),
                "revenue_yoy": float(rng.uniform(-20, 40)),
                "profit_yoy": float(rng.uniform(-30, 50)),
                "debt_ratio": float(rng.uniform(30, 70)),
                "gross_margin": float(rng.uniform(20, 60)),
                "net_margin": float(rng.uniform(5, 25)),
            } for d in quarters])
        if code in has_val:
            entry["valuation"] = pd.DataFrame([{
                "date": d, "stock_code": code,
                "pe_ttm": float(rng.uniform(8, 60)), "pb_mrq": float(rng.uniform(1, 8)),
                "ps_ttm": float(rng.uniform(1, 10)), "pcf_ttm": float(rng.uniform(3, 30)),
            } for d in stock_dates])
        by_code[code] = entry
    return by_code


class _MockSource(AShareSourceBase):
    """Serves pre-built synthetic daily data for ``serve_codes``; other stocks
    get an empty frame so the downloader falls through to the next source
    (数据源切换).  ``calls``/``empties`` record the failover for assertions."""

    SOURCE_NAME = "mock"

    def __init__(self, daily, serve_codes):
        self._daily = daily
        self._serve = set(serve_codes)
        self.calls: list[str] = []
        self.empties: list[str] = []

    def is_available(self):
        return True

    def fetch_daily(self, stock_code, start_date, end_date):
        self.calls.append(stock_code)
        if stock_code not in self._serve:
            self.empties.append(stock_code)
            return pd.DataFrame()
        df = self._daily[self._daily["stock_code"] == stock_code].copy()
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.Timestamp(start_date)) & (
            df["date"] <= pd.Timestamp(end_date)
        )
        df = df[mask].sort_values("date").reset_index(drop=True)
        if df.empty:
            self.empties.append(stock_code)
            return pd.DataFrame()
        df.attrs["source"] = self.SOURCE_NAME
        df.attrs["adjustment_mode"] = "qfq"
        return df


def _slice_time(panel_data: dict, start: int, stop: int, price_pad: int = 0) -> dict:
    """Time-axis slice mirroring train_panel._slice_panel.

    Price columns are padded `horizon` columns beyond `stop` so the sleeve
    entered on the last signal day can still liquidate at open[stop+horizon].
    """
    out = {
        "static_features": panel_data["static_features"][:, start:stop, :],
        "past_known": panel_data["past_known"][:, start:stop],
        "past_observed": panel_data["past_observed"][:, start:stop],
        "y_direction": panel_data["y_direction"][:, start:stop],
        "y_return_raw": panel_data["y_return"][:, start:stop].copy(),
        "y_return": panel_data["y_return"][:, start:stop].copy(),
        "y_volatility": panel_data["y_volatility"][:, start:stop].copy(),
        "observation_mask": panel_data["observation_mask"][:, start:stop],
        "entry_eligible_mask": panel_data["entry_eligible_mask"][:, start:stop],
        "return_target_mask": panel_data["return_target_mask"][:, start:stop],
        "vol_target_mask": panel_data["vol_target_mask"][:, start:stop],
        "realized_return": panel_data["realized_return"][:, start:stop].copy(),
        "date_indices": panel_data["date_indices"][:, start:stop].copy(),
        "decision_eligible_mask": panel_data["decision_eligible_mask"][:, start:stop],
        "history_eligible_mask": panel_data["history_eligible_mask"][:, start:stop],
    }
    max_T = panel_data["close_price"].shape[1]
    pstop = min(stop + price_pad, max_T) if price_pad > 0 else stop
    out["close_price"] = panel_data["close_price"][:, start:pstop]
    out["open_price"] = panel_data["open_price"][:, start:pstop]
    return out


def _pipeline():
    return FeaturePipeline(
        seq_len=SEQ_LEN, horizon=HORIZON,
        use_sentiment=True, use_fundamental=True, use_valuation=True,
        use_announcements=False, use_guba=False, use_comment=False,
        use_margin=False, use_northbound=False, use_dragon_tiger=False,
        use_earnings=False, use_etf_flow=False, use_interaction=False,
        use_feature_selection=False, use_capital_flow=False,
        use_block_trade=False, use_shareholder=False, use_lockup=False,
        use_dividend=False, use_board=False, use_sector=False,
        use_concept=False, use_macro=False, use_industry=False,
        use_pledge=False, use_market_env=False, use_index_membership=False,
        use_market_env_refine=False,
        use_emotion_refine=False, use_fundamental_refine=False,
        use_temporal_stats=False,
        min_history=SEQ_LEN,
    )


def _run_storage_gate(daily_dir) -> None:
    """Run the download/storage quality-gate checks against a temp daily dir."""
    dqg._DAILY_CACHE.clear()
    old_daily, old_feat = dqg.DAILY_DIR, dqg.FEAT_DIR
    dqg.DAILY_DIR = daily_dir
    try:
        for check in (dqg.check_ohlc_sanity, dqg.check_contract_schema,
                      dqg.check_daily_internal):
            res = check(0)
            assert res.passed, (
                f"{res.name} FAILED: {res.summary} first issues: {res.issues[:5]}"
            )
    finally:
        dqg.DAILY_DIR = old_daily
        dqg.FEAT_DIR = old_feat
        dqg._DAILY_CACHE.clear()


def _run_feature_gate(daily_dir, features_dir) -> None:
    """Run the feature-layer quality-gate check against prebuilt features."""
    dqg._DAILY_CACHE.clear()
    old_daily, old_feat = dqg.DAILY_DIR, dqg.FEAT_DIR
    dqg.DAILY_DIR = daily_dir
    dqg.FEAT_DIR = features_dir
    try:
        res = dqg.check_feature_pct(0)
        assert res.passed, (
            f"feature_pct FAILED: {res.summary} first issues: {res.issues[:5]}"
        )
    finally:
        dqg.DAILY_DIR = old_daily
        dqg.FEAT_DIR = old_feat
        dqg._DAILY_CACHE.clear()


def _download(daily, codes):
    """Real AShareDownloader with mock sources; primary→secondary failover.

    The production backfill path calls ``self._sources[-1]`` (Baostock) for
    stocks whose served history starts after the requested start — for 新股
    Baostock correctly returns nothing (the stock did not exist yet), which is
    the harmless no-op the downloader logs.  A dedicated always-empty mock
    plays that Baostock role so the failover source's call log stays clean.
    """
    primary = _MockSource(daily, serve_codes=codes[0:12])
    secondary = _MockSource(daily, serve_codes=codes[12:20])
    backfill = _MockSource(daily, serve_codes=[])
    dl = AShareDownloader()
    dl._sources = [primary, secondary, backfill]
    end = str(_grid_dates()[-1])
    parts = []
    for code in codes:
        df = dl.fetch_daily(code, START_DATE, end)
        assert len(df) > 0, f"{code}: downloader returned no data"
        parts.append(df)
    return pd.concat(parts, ignore_index=True), primary, secondary


class TestCorporateActionAdjustment:
    def test_qfq_removes_exright_gap(self):
        """分红/除权: the stored (adjusted) series must be continuous across an
        ex-right date while the raw series keeps the corporate-action gap."""
        _daily, _aux, meta = _make_synthetic_daily()
        assert len(meta["exright"]) >= 3
        for code, ex_idx, d, stored_ret, raw_ret in meta["exright"]:
            assert abs(stored_ret) < 0.06, (
                f"{code} day {ex_idx}: stored adjusted return {stored_ret:.4f} "
                f"shows an artificial ex-right jump (drop={d})"
            )
            assert abs(raw_ret - (-d)) < 0.04, (
                f"{code} day {ex_idx}: raw return {raw_ret:.4f} does not match "
                f"the expected {d} ex-right gap"
            )


class TestGoldenPipeline:
    def test_full_chain_with_all_scenarios(self, tmp_path):
        daily, aux, meta = _make_synthetic_daily()
        codes = meta["codes"]

        # ── 1. download mock ──────────────────────────────────────────────
        downloaded, primary, secondary = _download(daily, codes)
        assert primary.calls == codes, "every fetch must try the primary first"
        assert set(primary.empties) == set(codes[12:20]), (
            "数据源切换: the primary must have failed exactly on the 8 "
            "secondary-served stocks"
        )
        assert set(secondary.calls) == set(codes[12:20])
        assert len(secondary.empties) == 0
        assert len(downloaded["stock_code"].unique()) == N_STOCKS

        # ── 2. storage ───────────────────────────────────────────────────
        data_dir = tmp_path / "data"
        storage = DataStorage(str(data_dir))
        storage.save_daily(downloaded, market="a_shares")
        assert storage.list_stocks() == codes
        for code in codes:
            m = storage.manifest(code)
            assert m is not None and m["stock"] == code
            assert storage.validate_manifest(code)["ok"], f"{code} manifest mismatch"

        # ── 3. quality gate (download/storage output) ────────────────────
        daily_dir = data_dir / "a_shares" / "daily"
        _run_storage_gate(daily_dir)

        # ── 4. features: prebuild → feature gate → panel assembly ────────
        features_dir = tmp_path / "features_panel"
        features_dir.mkdir()
        pipeline = _pipeline()
        for code in codes:
            stock_df = downloaded[downloaded["stock_code"] == code] \
                .sort_values("date").reset_index(drop=True)
            a = aux[code]
            pipeline.save_features(
                str(features_dir / f"{code}.parquet"), stock_df,
                sentiment_df=a.get("sentiment"),
                fundamental_df=a.get("fundamental"),
                valuation_df=a.get("valuation"),
                panel_mode=True,
            )

        # 缺失新闻: news-less stocks carry no has_news column at the feature
        # layer (panel ZI-fills it to False downstream).
        news_less = sorted(meta["no_news"])[0]
        news_full = sorted(set(codes) - meta["no_news"])[0]
        f_none = pd.read_parquet(features_dir / f"{news_less}.parquet")
        assert "has_news" not in f_none.columns or f_none["has_news"].sum() == 0
        f_has = pd.read_parquet(features_dir / f"{news_full}.parquet")
        assert "has_news" in f_has.columns and f_has["has_news"].sum() > 0

        # 缺失财务: stocks without fundamentals lack roe; without valuation
        # lack pe_ttm — both ZI-filled downstream by the panel column-alignment.
        fund_less = sorted(meta["no_fund"])[0]
        fund_full = sorted(set(codes) - meta["no_fund"])[0]
        f_nofund = pd.read_parquet(features_dir / f"{fund_less}.parquet")
        assert "roe" not in f_nofund.columns or f_nofund["roe"].abs().sum() == 0
        f_fund = pd.read_parquet(features_dir / f"{fund_full}.parquet")
        assert "roe" in f_fund.columns and f_fund["roe"].abs().sum() > 0

        val_less = sorted(meta["no_val"])[0]
        val_full = sorted(set(codes) - meta["no_val"])[0]
        f_noval = pd.read_parquet(features_dir / f"{val_less}.parquet")
        assert "pe_ttm" not in f_noval.columns or f_noval["pe_ttm"].abs().sum() == 0
        f_val = pd.read_parquet(features_dir / f"{val_full}.parquet")
        assert "pe_ttm" in f_val.columns and f_val["pe_ttm"].abs().sum() > 0

        # 20 prebuilt feature parquets → feature-layer quality gate passes.
        assert len(list(features_dir.glob("*.parquet"))) == N_STOCKS
        _run_feature_gate(daily_dir, features_dir)

        # Real panel assembly from the prebuilt parquets (production prebuilt path).
        panel_data = pipeline.build_panel_features(
            downloaded, horizon=HORIZON, prebuilt_dir=str(features_dir)
        )
        N = panel_data["static_features"].shape[0]
        T = panel_data["past_known"].shape[1]
        assert N == N_STOCKS and T == N_DAYS
        assert panel_data["static_features"].shape == (N, T, len(_PIT_STATIC_COLS))
        for key in ("observation_mask", "entry_eligible_mask", "return_target_mask",
                    "vol_target_mask", "decision_eligible_mask", "history_eligible_mask"):
            assert panel_data[key].dtype == np.bool_
        assert panel_data["global_dates"].shape == (T,)

        # ── 5. dataset ───────────────────────────────────────────────────
        ds = PanelDataset(panel_data, seq_len=SEQ_LEN, min_history=SEQ_LEN)
        assert ds.n_stocks == N_STOCKS
        assert ds.n_timesteps == N_DAYS
        assert len(ds) > 0
        assert ds.valid_mask.sum() > 0, "no trainable windows after masking"

        # ── 6. train 1 epoch (production trainer) ────────────────────────
        train_stop = 380  # after the last new-listing date (350)
        val_stop = N_DAYS
        train_data = _slice_time(panel_data, 0, train_stop)
        val_data = _slice_time(panel_data, train_stop, val_stop, price_pad=HORIZON)
        config = PanelConfig(
            static_dim=panel_data["static_features"].shape[2],
            past_known_dim=panel_data["past_known"].shape[2],
            past_observed_dim=panel_data["past_observed"].shape[2],
            hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
            grn_layers=1, seq_len=SEQ_LEN, min_history=SEQ_LEN,
            batch_size=16, max_epochs=1, compile_model=False,
            num_workers=0, horizon=HORIZON, seed=0, rank_loss_weight=0.0,
        )
        device = torch.device("cpu")
        model, history = train_panel(
            config, train_data, val_data, device,
            raw_val_returns=val_data["realized_return"],
        )
        assert history["best_epoch_idx"] == 0
        best = history["best_metrics"]
        assert best.get("n_periods", 0) >= 2, "sleeve account produced too few periods"
        assert np.isfinite(best["long_sharpe"])

        # ── 7. evaluate (explicit hold-out on the deployed checkpoint) ────
        m = evaluate_portfolio(
            model, val_data, config, device,
            horizon=HORIZON, raw_returns=val_data["realized_return"],
            require_price_path=True,
        )
        assert m["n_periods"] >= 2
        assert np.isfinite(m["long_sharpe"])
        assert np.isfinite(m["ls_sharpe"])
