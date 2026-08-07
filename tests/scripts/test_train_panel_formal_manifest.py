"""Unit tests for T4: formal load_aux_data() enforcing required-channel Asset
Manifests (§五 + §十九-9).

Formal mode (``formal=True``) must FAIL HARD (SystemExit) when a required
channel's asset manifest is missing / content-mismatched / extent-mismatched,
and must FAIL loudly for required channels with NO manifest support (the cninfo
announcement-sentiment path).  Explore mode (``formal=False``) keeps the legacy
warn-and-proceed: the same broken manifests load leniently and the run proceeds.

None of these read real market data — synthetic per-stock parquets + manifests
on ``tmp_path``, no network.
"""
import importlib.util
import json
import os

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "production", "train_panel.py")


@pytest.fixture(scope="module")
def tp():
    spec = importlib.util.spec_from_file_location("train_panel_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── synthetic guba fixture ───────────────────────────────────────────────

def _guba_df(code="000001"):
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "stock_code": [code, code],
        "guba_sentiment_mean": [0.1, -0.2],
        "guba_sentiment_std": [0.3, 0.1],
        "guba_post_count": [2, 1],
        "guba_positive_ratio": [0.6, 0.4],
        "guba_negative_ratio": [0.2, 0.5],
        "has_guba_post": [True, True],
    })


def _write_guba(tmp_path, code="000001"):
    """Write a guba sentiment parquet + a VALID sidecar asset manifest."""
    from stoke_ml.data.guba_storage import GubaStorage
    data_dir = str(tmp_path / "data")
    GubaStorage(data_dir).save_daily_sentiment(_guba_df(code))
    return data_dir


def _guba_flat(data_dir, code="000001"):
    return os.path.join(data_dir, "a_shares", "guba_sentiment",
                        f"{code}.parquet")


def _guba_manifest(data_dir, code="000001"):
    return _guba_flat(data_dir, code) + ".manifest.json"


def _tamper_content(data_dir, code="000001"):
    """Rewrite the parquet body so the manifest's schema_hash no longer matches."""
    flat = _guba_flat(data_dir, code)
    df = pd.read_parquet(flat)
    df.loc[0, "guba_sentiment_mean"] = 9.9
    df.to_parquet(flat, index=False, compression="lz4")


def _tamper_extent(data_dir, code="000001"):
    """Edit the manifest's start/end so it no longer matches the file's extent."""
    path = _guba_manifest(data_dir, code)
    with open(path, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["start"] = "1999-01-01"
    m["end"] = "2099-12-31"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f)


# ── formal=True: three tamper states FAIL HARD ───────────────────────────

def test_formal_valid_manifest_passes(tp, tmp_path):
    """Positive control: a required channel with a valid manifest passes the
    formal gate and loads normally."""
    data_dir = _write_guba(tmp_path)
    result, manifest = tp.load_aux_data(
        ["000001"], data_dir, "2024-01-01", "2024-12-31",
        required_channels={"guba"}, formal=True)
    assert manifest["guba"]["status"] == "OK"
    assert not result["000001"]["guba"].empty


def test_formal_missing_manifest_fails(tp, tmp_path):
    """Manifest MISSING (parquet exists, sidecar deleted) → hard fail."""
    data_dir = _write_guba(tmp_path)
    os.remove(_guba_manifest(data_dir))
    with pytest.raises(SystemExit) as ei:
        tp.load_aux_data(
            ["000001"], data_dir, "2024-01-01", "2024-12-31",
            required_channels={"guba"}, formal=True)
    msg = str(ei.value)
    assert "formal mode" in msg
    assert "manifest missing" in msg
    assert "guba" in msg


def test_formal_tampered_content_fails(tp, tmp_path):
    """Manifest content MISMATCH (parquet body edited, schema_hash drifted)
    → hard fail."""
    data_dir = _write_guba(tmp_path)
    _tamper_content(data_dir)
    with pytest.raises(SystemExit) as ei:
        tp.load_aux_data(
            ["000001"], data_dir, "2024-01-01", "2024-12-31",
            required_channels={"guba"}, formal=True)
    msg = str(ei.value)
    assert "formal mode" in msg
    assert "schema_hash" in msg


def test_formal_tampered_extent_fails(tp, tmp_path):
    """Manifest coverage MISMATCH (recorded start/end no longer match the
    file's extent) → hard fail."""
    data_dir = _write_guba(tmp_path)
    _tamper_extent(data_dir)
    with pytest.raises(SystemExit) as ei:
        tp.load_aux_data(
            ["000001"], data_dir, "2024-01-01", "2024-12-31",
            required_channels={"guba"}, formal=True)
    msg = str(ei.value)
    assert "formal mode" in msg
    assert "start:" in msg


def test_formal_cninfo_announcement_fails(tp, tmp_path):
    """A required channel with NO manifest support (cninfo announcement-
    sentiment path) fails loudly with guidance under formal."""
    data_dir = str(tmp_path / "data")
    cninfo_dir = os.path.join(
        data_dir, "a_shares", "cninfo_announcements", "sentiment")
    os.makedirs(cninfo_dir, exist_ok=True)
    pd.DataFrame({"date": pd.to_datetime(["2024-01-02"]),
                  "v": [1.0]}).to_parquet(
        os.path.join(cninfo_dir, "000001.parquet"))
    with pytest.raises(SystemExit) as ei:
        tp.load_aux_data(
            ["000001"], data_dir, "2024-01-01", "2024-12-31",
            required_channels={"announcement"}, formal=True)
    msg = str(ei.value)
    assert "no asset-manifest support" in msg
    assert "announcement" in msg
    assert "--prebuilt" in msg


def test_formal_unadopted_market_wide_channel_fails(tp, tmp_path):
    """shareholder is loaded by load_aux_data but has NO MarketWideStorage
    asset contract — under formal a required shareholder channel fails loudly."""
    data_dir = str(tmp_path / "data")
    with pytest.raises(SystemExit) as ei:
        tp.load_aux_data(
            ["000001"], data_dir, "2024-01-01", "2024-12-31",
            required_channels={"shareholder"}, formal=True)
    msg = str(ei.value)
    assert "no asset-manifest support" in msg or "no asset-manifest contract" in msg


def test_formal_multi_channel_aggregates_all_failures(tp, tmp_path):
    """TWO required channels failing at once → BOTH channel names appear in the
    single SystemExit message, locking the aggregate diagnostics join format."""
    data_dir = _write_guba(tmp_path)
    _tamper_content(data_dir)  # guba: content (schema_hash) mismatch
    # sentiment: parquet present but manifest MISSING → formal read raises
    sentiment_dir = os.path.join(data_dir, "a_shares", "sentiment")
    os.makedirs(sentiment_dir, exist_ok=True)
    pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "stock_code": ["000001", "000001"],
        "sentiment_mean": [0.1, -0.2],
        "sentiment_std": [0.3, 0.1],
        "news_count": [2, 1],
        "positive_ratio": [0.6, 0.4],
        "negative_ratio": [0.2, 0.5],
        "has_news": [True, True],
    }).to_parquet(os.path.join(sentiment_dir, "000001.parquet"),
                  index=False, compression="lz4")
    with pytest.raises(SystemExit) as ei:
        tp.load_aux_data(
            ["000001"], data_dir, "2024-01-01", "2024-12-31",
            required_channels={"guba", "sentiment"}, formal=True)
    msg = str(ei.value)
    assert "guba" in msg
    assert "sentiment" in msg
    assert "schema_hash" in msg
    assert "manifest missing" in msg


# ── formal=False (explore): the same states WARN and PROCEED ─────────────

def test_explore_missing_manifest_proceeds(tp, tmp_path, caplog):
    data_dir = _write_guba(tmp_path)
    os.remove(_guba_manifest(data_dir))
    with caplog.at_level("DEBUG"):
        result, manifest = tp.load_aux_data(
            ["000001"], data_dir, "2024-01-01", "2024-12-31",
            required_channels={"guba"}, formal=False)
    assert manifest["guba"]["status"] == "OK"
    assert not result["000001"]["guba"].empty


def test_explore_tampered_content_proceeds(tp, tmp_path, caplog):
    data_dir = _write_guba(tmp_path)
    _tamper_content(data_dir)
    with caplog.at_level("WARNING"):
        result, manifest = tp.load_aux_data(
            ["000001"], data_dir, "2024-01-01", "2024-12-31",
            required_channels={"guba"}, formal=False)
    assert manifest["guba"]["status"] == "OK"
    assert not result["000001"]["guba"].empty
    assert any("manifest mismatch" in m for m in caplog.messages)


def test_explore_tampered_extent_proceeds(tp, tmp_path, caplog):
    data_dir = _write_guba(tmp_path)
    _tamper_extent(data_dir)
    with caplog.at_level("WARNING"):
        result, manifest = tp.load_aux_data(
            ["000001"], data_dir, "2024-01-01", "2024-12-31",
            required_channels={"guba"}, formal=False)
    assert manifest["guba"]["status"] == "OK"
    assert not result["000001"]["guba"].empty


# ── §T4: _resolve_panel live branch threads formal=_formal_mode(args) ────

def _fake_storage():
    class _FakeStorage:
        def __init__(self, data_dir):
            self.data_dir = data_dir

        def load_daily(self, code, start, end, require_valid_manifest=True):
            return pd.DataFrame({
                "date": pd.to_datetime(["2022-01-04"]),
                "open": [1.0], "high": [1.1], "low": [0.9],
                "close": [1.0], "volume": [100], "amount": [100.0],
            })
    return _FakeStorage


def _capture_pipeline():
    calls = []

    class _FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def build_panel_features(self, panel, **kw):
            calls.append({"panel": panel, **kw})
            return {"__dummy": True}
    return _FakePipeline, calls


def _panel_args(vintage_policy, **overrides):
    base = {
        "vintage_policy": vintage_policy,
        "minute": False,
        "horizon": 1,
        "start": "2020-01-01",
        "end": "2024-12-31",
        "universe": "random",
        "prebuilt": None,
        "seq_len": None,
        "allow_high_risk_universe": False,
        "allow_fundamental_ablation": False,
        "no_formal": False,
        "no_require_quality_gate": False,
        "feature_profile": None,
        "require_aux_channels": "",
        "no_aux": False,
    }
    base.update(overrides)
    import types
    return types.SimpleNamespace(**base)


def _resolve_panel_with_spy(tp, monkeypatch, **args_overrides):
    """Run _resolve_panel live branch; return the formal kwarg load_aux_data got."""
    import stoke_ml.data.storage as storage_mod
    fake_pipe, _calls = _capture_pipeline()
    monkeypatch.setattr(
        "scripts.production.train_panel_panel.FeaturePipeline", fake_pipe)
    monkeypatch.setattr(storage_mod, "DataStorage", _fake_storage())
    captured = {}

    def _spy_aux(stock_list, data_dir, start_date, end_date,
                 required_channels=None, formal=False):
        captured["formal"] = formal
        captured["required"] = set(required_channels or ())
        return {}, {}

    monkeypatch.setattr(
        "scripts.production.train_panel_panel.load_aux_data", _spy_aux)
    args = _panel_args("revision-safe", **args_overrides)
    args.panel_store = None
    args.require_feature_manifest = False
    tp._resolve_panel(
        args, ["000001"], 60, "data", {"sentiment"}, _store_load=False)
    return captured


def test_resolve_panel_live_branch_forwards_formal_true(tp, monkeypatch):
    """A formal run (--no-formal absent) threads formal=True into load_aux_data."""
    captured = _resolve_panel_with_spy(tp, monkeypatch)
    assert captured["formal"] is True
    assert captured["required"] == {"sentiment"}


def test_resolve_panel_live_branch_forwards_formal_false(tp, monkeypatch):
    """An exploratory run (--no-formal) threads formal=False into load_aux_data."""
    captured = _resolve_panel_with_spy(tp, monkeypatch, no_formal=True)
    assert captured["formal"] is False


def test_load_aux_data_defaults_formal_false(tp, tmp_path):
    """Backward compat: no formal kwarg → explore (warn-proceed), exactly the
    legacy behavior."""
    data_dir = _write_guba(tmp_path)
    os.remove(_guba_manifest(data_dir))
    result, manifest = tp.load_aux_data(
        ["000001"], data_dir, "2024-01-01", "2024-12-31",
        required_channels={"guba"})
    assert manifest["guba"]["status"] == "OK"
    assert not result["000001"]["guba"].empty
