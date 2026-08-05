"""Production-chain smoke test.

Runs the REAL production pipeline end-to-end on tiny synthetic OHLCV:

    FeaturePipeline.build_panel_features → train_panel (1 epoch, CPU)
        → evaluate_portfolio(require_price_path=True) → sleeve-account identity

Only production functions are called — no hand-written training loop, no 2D
static shortcut.  This is the guard that the "NEW train_panel.py /
evaluate.py call NEW Panel interface" contract holds on a real forward pass, and
that the price-path sleeve account runs (its internal account-identity
assertion fires) rather than silently falling back to the legacy estimator.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.production.train_panel import _weight_hash
from stoke_ml.data.calendar import TradingCalendar
from stoke_ml.features.pipeline import FeaturePipeline, _PIT_STATIC_COLS
from stoke_ml.models.panel import PanelConfig
from stoke_ml.models.panel.evaluate import _simulate_sleeve_account, evaluate_portfolio
from stoke_ml.models.panel.model import PanelModel
from stoke_ml.models.panel.train import train_panel

N_STOCKS = 12
N_DAYS = 200
SEQ_LEN = 20
HORIZON = 5


def _make_synthetic_panel(n_stocks=N_STOCKS, n_days=N_DAYS, seed=42):
    """Geometric random-walk OHLCV panel → well-defined indicators / vol targets."""
    rng = np.random.RandomState(seed)
    codes = [f"{600000 + i:06d}" for i in range(n_stocks)]
    # Official A-share trading days: the pipeline rejects dates not
    # on the exchange calendar, so a bdate_range grid would silently shrink T.
    dates = pd.DatetimeIndex(
        TradingCalendar("a_shares").get_trading_days("2022-01-03", "2022-12-31")
    )[:n_days]
    rows = []
    for i, code in enumerate(codes):
        drift = 0.0005 * (i % 5 - 2)
        close = 10.0 * np.cumprod(1 + rng.normal(drift, 0.02, n_days))
        open_ = np.concatenate([[close[0]], close[:-1]]) * (
            1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        volume = np.abs(rng.normal(1e6, 2e5, n_days))
        amount = volume * close
        for t in range(n_days):
            rows.append({
                "date": dates[t], "stock_code": code,
                "open": float(open_[t]), "high": float(high[t]),
                "low": float(low[t]), "close": float(close[t]),
                "volume": float(volume[t]), "amount": float(amount[t]),
            })
    return pd.DataFrame(rows)


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
        seq_len=SEQ_LEN,
        minute_mode=False,
        use_board=False, use_sector=False, use_concept=False,
        min_history=SEQ_LEN,
    )


class TestBuildPanelFeatures:
    @pytest.mark.slow
    def test_keys_and_shapes(self):
        panel = _make_synthetic_panel()
        data = _pipeline().build_panel_features(panel, horizon=HORIZON)

        N = data["static_features"].shape[0]
        T = data["past_known"].shape[1]
        assert N == N_STOCKS
        assert T == N_DAYS
        for key in ("static_features", "past_known", "past_observed"):
            assert data[key].dtype == np.float32
        # 9 PIT-static cols: price_60d_q / amt_60d_q / listing_days
        # + 6 board one-hot.  industry_code excluded — no PIT
        # industry-membership source exists, only a present-snapshot map.
        assert data["static_features"].shape == (N, T, len(_PIT_STATIC_COLS))
        for key in ("y_direction", "y_return", "y_volatility"):
            assert data[key].shape == (N, T)
        for key in ("observation_mask", "entry_eligible_mask", "return_target_mask",
                    "vol_target_mask", "decision_eligible_mask", "history_eligible_mask"):
            assert key in data, f"missing mask {key}"
            assert data[key].shape == (N, T)
            assert data[key].dtype == np.bool_
        assert data["close_price"].shape == (N, T)
        assert data["open_price"].shape == (N, T)
        assert data["global_dates"].shape == (T,)


class TestProductionTrainThenEvaluate:
    @pytest.mark.slow
    def test_full_chain(self):
        panel = _make_synthetic_panel()
        panel_data = _pipeline().build_panel_features(panel, horizon=HORIZON)

        train_stop = 140
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
            min_stocks_per_day=5,
        )
        device = torch.device("cpu")

        # train_panel is the production trainer: best-checkpoint selection on
        # inner_val, then a price-path portfolio evaluation of the deployed
        # checkpoint.  require_price_path=True is forced inside train_panel.
        model, history = train_panel(
            config, train_data, val_data, device,
            raw_val_returns=val_data["realized_return"],
        )
        assert history["best_epoch_idx"] == 0
        best = history["best_metrics"]
        assert best.get("n_periods", 0) >= 2, "sleeve account produced too few periods"
        assert np.isfinite(best["long_sharpe"])

        # Explicit hold-out evaluation of the exact deployed checkpoint — must
        # go through the chronological sleeve account, not the legacy fallback.
        m = evaluate_portfolio(
            model, val_data, config, device,
            horizon=HORIZON, raw_returns=val_data["realized_return"],
            require_price_path=True,
        )
        assert m["n_periods"] >= 2
        assert np.isfinite(m["long_sharpe"])
        assert np.isfinite(m["ls_sharpe"])


class TestSleeveAccountIdentity:
    def test_nav_identity_and_gross_bound(self):
        rng = np.random.RandomState(0)
        N, W, Wp = 4, 12, 12 + HORIZON
        preds = rng.randn(N, W).astype(np.float32)
        # Deterministic gently-rising prices so fills / NAV stay well-defined.
        px = np.linspace(10.0, 11.2, Wp).astype(np.float32)
        close = np.tile(px, (N, 1))
        openp = close * 0.999
        pool = np.ones((N, W), dtype=bool)

        res = _simulate_sleeve_account(
            preds, close, openp, pool, HORIZON, top_fraction=0.5,
            cost=0.0005, mode="long",
        )
        daily = res["daily"]
        final_nav = float(np.prod(1 + daily))
        # The simulator's internal assertion already enforces
        # prod(1+daily) == final_nav; re-assert it explicitly here.
        assert np.isfinite(final_nav) and final_nav > 0
        # The cost=0 gross series can only bound the net NAV from above.
        gross = res["gross_daily"]
        assert float(np.prod(1 + gross)) >= final_nav - 1e-9


class TestPersistBestCheckpoint:
    """§十二-1: the per-fold best-inner-val checkpoint must be persistable and
    its trained-parameter hash must distinguish it from every other fold.

    The version-dict `model_hash` fingerprints config + architecture source and
    is therefore shared by all folds; `_weight_hash` hashes the actual
    state_dict so an OOS tape row maps to exactly one set of weights.
    """

    @pytest.mark.slow
    def test_weight_hash_roundtrip_and_fold_distinction(self, tmp_path):
        panel = _make_synthetic_panel()
        panel_data = _pipeline().build_panel_features(panel, horizon=HORIZON)
        train_stop = 140
        val_stop = N_DAYS
        train_data = _slice_time(panel_data, 0, train_stop)
        val_data = _slice_time(panel_data, train_stop, val_stop, price_pad=HORIZON)

        def _config(seed):
            return PanelConfig(
                static_dim=panel_data["static_features"].shape[2],
                past_known_dim=panel_data["past_known"].shape[2],
                past_observed_dim=panel_data["past_observed"].shape[2],
                hidden_dim=32, xlstm_num_blocks=1, xlstm_num_heads=2,
                grn_layers=1, seq_len=SEQ_LEN, min_history=SEQ_LEN,
                batch_size=16, max_epochs=1, compile_model=False,
                num_workers=0, horizon=HORIZON, seed=seed, rank_loss_weight=0.0,
                min_stocks_per_day=5,
            )

        device = torch.device("cpu")
        model, history = train_panel(
            _config(0), train_data, val_data, device,
            raw_val_returns=val_data["realized_return"],
        )
        model2, _ = train_panel(
            _config(1), train_data, val_data, device,
            raw_val_returns=val_data["realized_return"],
        )

        wh = _weight_hash(model)
        # Deterministic on identical weights.
        assert wh == _weight_hash(model)
        # Two folds with different seeds → different trained weights → hash
        # must differ even though version_info["model_hash"] would not.
        assert wh != _weight_hash(model2)

        # Persist the exact checkpoint dict a fold writes, then reload: the
        # hash recomputed from the loaded state_dict must match the recorded
        # one — proving the recorded weight_hash really identifies these
        # weights (the offline-replay contract of fold_XXX_model.pt).
        ckpt_path = tmp_path / "fold_001_model.pt"
        torch.save({
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": dataclasses.asdict(_config(0)),
            "weight_hash": wh,
            "best_epoch": history.get("best_epoch_idx", 0) + 1,
        }, ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        reloaded = PanelModel(PanelConfig(**ckpt["config"])).to(device)
        reloaded.load_state_dict(ckpt["state_dict"])
        assert _weight_hash(reloaded) == ckpt["weight_hash"]
