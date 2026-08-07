"""§T2 golden test: the LIVE and PREBUILT feature paths must produce
byte-identical model inputs.

The v17 audit found the SAME channel's LIVE and PREBUILT feature paths
hard-coded in five consumer modules (live ``a_shares/capital_flow/`` vs
prebuilt ``a_shares/capital_flow_processed/``).  Rewiring them to
``CHANNEL_SOURCE`` must NOT change what the feature pipeline produces.  This
golden test feeds the SAME synthetic raw data down BOTH paths:

  LIVE    : ``load_aux_data(...)`` → ``build_panel_features(..., aux_data=...)``
            — per-stock aux dfs read from the LIVE dir
            (``a_shares/capital_flow/``) and merged live in
            ``_engineer_features``;
  PREBUILT: ``build_features.build_one(...)`` →
            ``build_panel_features(..., prebuilt_dir=...)`` — per-stock aux dfs
            read from the PROCESSED dir (``a_shares/capital_flow_processed/``),
            pre-engineered and saved to parquet.

and asserts the ``past_known`` / ``past_observed`` / ``static_features`` grids
are element-wise identical (tight tolerance).

Scope decision (documented): **capital_flow** — the MarketWideStorage channel
with a DISTINCT ``*_processed`` prebuilt variant (the exact live-vs-prebuilt
split the audit flagged) — plus **block_trade**, a sparse event channel with
the same split.  Every other channel carries NO synthetic data, so it is
ZI-zero / ``has_*=False`` on BOTH paths — identical.  If a REAL raw-vs-processed
preprocessing difference were exposed by this test, it would be recorded and
reconciled here, not papered over.
"""
import importlib.util
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stoke_ml.data.calendar import TradingCalendar, save_calendar
from stoke_ml.data.channel_sources import CHANNEL_SOURCE
from stoke_ml.data.storage import DataStorage
from stoke_ml.features.pipeline import FeaturePipeline

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "production"

SEQ_LEN = 20
HORIZON = 1
START = "2023-01-03"
END = "2023-06-30"
CODES = ["000001", "600519"]
TOL = {"rtol": 1e-9, "atol": 1e-9, "equal_nan": True}


def _load_script(name: str):
    path = _SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def build_features_mod():
    return _load_script("build_features")


@pytest.fixture(scope="module")
def train_panel_panel_mod():
    return _load_script("train_panel_panel")


class _ConfigStub(dict):
    """A config-like object serving BOTH access styles the pipeline uses:

    * ``load_config().project.data_dir`` — aux_aligner's _merge_macro /
      _merge_market_env / _merge_industry (attribute access);
    * ``load_config().get("universe", {})`` — EligibilityBuilder (dict access).
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)


def _config_stub(data_dir: str) -> _ConfigStub:
    return _ConfigStub({
        "project": SimpleNamespace(data_dir=data_dir),
        "universe": {
            "long_suspension_days": 60,
            "suspension_lookback": 60,
            "min_amount_60d": 5_000_000,
        },
    })


def _pipeline():
    # MUST construct the EXACT pipeline build_one builds from args: only these
    # switches are passed there; every other use_* keeps its FeaturePipeline
    # default (True).  The SAME instance is used for the live AND prebuilt
    # build_panel_features calls so the two paths configure identically.
    return FeaturePipeline(
        seq_len=SEQ_LEN, horizon=HORIZON, flat_mode=False, min_history=SEQ_LEN,
        use_technical=True, use_scoring=True, use_temporal=True,
        use_sentiment=False, use_guba=False, use_comment=False,
        use_limit_up=False,
        use_pledge=False, use_market_env=False, use_market_env_refine=False,
        use_index_membership=False,
    )


def _trading_days(data_dir: str):
    # get_trading_days yields datetime64[s]; normalize to ns (the pandas default
    # the download layer produces) so the save_daily parquet round-trip keeps
    # the dtype and the contract manifest's schema_hash still matches on read.
    return pd.DatetimeIndex(
        TradingCalendar("a_shares", calendar_dir=data_dir)
        .get_trading_days(START, END)
    ).astype("datetime64[ns]")


def _daily_frame(code: str, tdays: pd.DatetimeIndex, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = len(tdays)
    close = 10.0 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    open_ = np.concatenate([[close[0]], close[:-1]]) * (
        1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = np.abs(rng.normal(1e6, 2e5, n))
    return pd.DataFrame({
        "date": tdays, "stock_code": code,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "amount": volume * close,
    })


def _aux_frame(code: str, channel: str, tdays: pd.DatetimeIndex,
               seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    n = len(tdays)
    if channel == "capital_flow":
        return pd.DataFrame({
            "date": tdays, "stock_code": code,
            "main_net_inflow": rng.randn(n) * 1e7,
            "retail_net_inflow": rng.randn(n) * 1e7,
            "has_capital_flow": True,
        })
    # block_trade — names avoid the K-line volume/amount collision so the
    # _merge_daily_aux merge keeps them (not the drop-collision path).
    return pd.DataFrame({
        "date": tdays, "stock_code": code,
        "block_amount": rng.randn(n) * 1e8,
        "block_price": 10.0 + rng.randn(n),
        "has_block_trade": True,
    })


def _write_aux_identical(data_dir: str, code: str, channel: str,
                         tdays: pd.DatetimeIndex, seed: int) -> None:
    """Write the SAME bytes to the LIVE dir AND the *_processed dir.

    §T2: the live path (load_aux_data → MarketWideStorage.load) reads the
    LIVE dir; the prebuilt path (build_one → _load_stock_parquet) reads the
    PROCESSED dir.  Identical bytes ⇒ identical merge inputs.
    """
    spec = CHANNEL_SOURCE[channel]
    frame = _aux_frame(code, channel, tdays, seed)
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False, compression="lz4")
    payload = buf.getvalue()
    for rel in (spec.live_dir, spec.processed_dir):
        path = os.path.join(data_dir, rel, f"{code}.parquet")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(payload)


def _build_one_args(build_features_mod, data_dir, pdir, code):
    return {
        "code": code,
        "seq_len": SEQ_LEN, "horizon": HORIZON,
        "use_technical": True, "use_scoring": True, "use_temporal": True,
        "use_sentiment": False, "use_guba": False, "use_comment": False,
        "use_limit_up": False,
        "use_pledge": False, "use_market_env": False,
        "use_market_env_refine": False, "use_index_membership": False,
        "storage": DataStorage(data_dir),
        "news_storage": None, "margin_storage": None, "nb_storage": None,
        "dt_storage": None, "fund_storage": None, "earnings_storage": None,
        "etf_storage": None, "sector_mapper": None,
        "guba_storage": None, "comment_storage": None, "ann_storage": None,
        "data_dir": data_dir,
        "output_dir": str(pdir),
        "start": START, "end": END,
        "force": True, "panel_mode": True,
        # force=True skips the manifest-matches cache probe, so the config hash
        # is only recorded into the sidecar (not compared) — a fixed value keeps
        # the test free of the real config.yaml dependency.
        "_git_commit": "t2-golden",
        "_config_hash": "t2-golden-config-hash",
    }


def test_live_and_prebuilt_paths_identical(tmp_path, monkeypatch,
                                           build_features_mod,
                                           train_panel_panel_mod):
    data_dir = str(tmp_path / "data")
    # Neutralize config-root reads (macro/market_env/industry) AND the
    # EligibilityBuilder universe params — both go through load_config.
    import stoke_ml.config as config_mod
    monkeypatch.setattr(config_mod, "load_config",
                        lambda: _config_stub(data_dir))

    # ── Hermetic data root: calendar artifact + canonical daily + aux ──
    save_calendar(data_dir, "a_shares")
    tdays = _trading_days(data_dir)
    assert len(tdays) > 2 * SEQ_LEN  # enough history for seq windows

    storage = DataStorage(data_dir)
    daily_frames = []
    for i, code in enumerate(CODES):
        df = _daily_frame(code, tdays, seed=100 + i)
        df.attrs["source"] = "efinance"
        df.attrs["adjustment_mode"] = "qfq"
        storage.save_daily(df)
        daily_frames.append(
            storage.load_daily(code, START, END, require_valid_manifest=True))
        _write_aux_identical(data_dir, code, "capital_flow", tdays, seed=200 + i)
        _write_aux_identical(data_dir, code, "block_trade", tdays, seed=300 + i)
    panel = pd.concat(daily_frames, ignore_index=True)

    # ── LIVE path: load_aux_data (live dirs) → build_panel_features ──
    aux_data, _manifest = train_panel_panel_mod.load_aux_data(
        CODES, data_dir, START, END, required_channels={"capital_flow",
                                                        "block_trade"})
    for code in CODES:
        assert "capital_flow" in aux_data[code]
        assert "block_trade" in aux_data[code]
    pipeline = _pipeline()
    live = pipeline.build_panel_features(
        panel, horizon=HORIZON, aux_data=aux_data, data_dir=data_dir)

    # ── PREBUILT path: build_one (processed dirs) → build_panel_features ──
    pdir = tmp_path / "prebuilt"
    os.makedirs(pdir, exist_ok=True)
    for code in CODES:
        code, status, _cat = build_features_mod.build_one(
            _build_one_args(build_features_mod, data_dir, pdir, code))
        assert status == "built"
    prebuilt = pipeline.build_panel_features(
        panel, horizon=HORIZON, prebuilt_dir=str(pdir), data_dir=data_dir)

    # ── Same shape, same values ──
    assert live["past_known"].shape == prebuilt["past_known"].shape
    assert live["past_observed"].shape == prebuilt["past_observed"].shape
    assert live["static_features"].shape == prebuilt["static_features"].shape
    np.testing.assert_allclose(
        live["past_known"], prebuilt["past_known"], **TOL)
    np.testing.assert_allclose(
        live["past_observed"], prebuilt["past_observed"], **TOL)
    np.testing.assert_allclose(
        live["static_features"], prebuilt["static_features"], **TOL)
    # Targets come from the shared panel → trivially identical, but assert it
    # so a future target change cannot silently diverge.
    np.testing.assert_allclose(live["y_direction"], prebuilt["y_direction"])
    np.testing.assert_allclose(live["y_return"], prebuilt["y_return"])
