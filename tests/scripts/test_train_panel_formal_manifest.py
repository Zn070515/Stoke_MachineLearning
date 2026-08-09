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


def test_formal_manifest_gate_covers_consumed_fundamental_ablation(
        tp, tmp_path):
    """§v18-2: a consumed-but-unrequired channel (fundamental via ablation) is
    manifest-gated, not silently skipped — corrupting its manifest aborts.

    ``fundamental`` is denied under revision-safe unless the ablation flag
    forces it ON, so under a real run it enters the CONSUMED set only via the
    explicit ablation.  With it in the consumed set, a present-but-manifest-less
    fundamental parquet must FAIL the formal gate (SystemExit) — the same way a
    required channel's broken manifest does.
    """
    data_dir = str(tmp_path / "data")
    fundamental_dir = os.path.join(data_dir, "a_shares", "fundamentals")
    os.makedirs(fundamental_dir, exist_ok=True)
    # A present fundamental parquet with NO sidecar asset manifest — the
    # FormalStorage formal read refuses it (require_valid_manifest=True).
    pd.DataFrame({
        "report_date": pd.to_datetime(["2024-03-31"]),
        "stock_code": ["000001"],
        "pe": [10.0],
    }).to_parquet(os.path.join(fundamental_dir, "000001.parquet"),
                  index=False, compression="lz4")
    # Route through load_aux_data: formal + consumed_channels (the full channel
    # set the run opens) drives the gate, with fundamental consumed-but-unrequired.
    with pytest.raises(SystemExit) as ei:
        tp.load_aux_data(
            ["000001"], data_dir, "2020-01-01", "2024-12-31",
            required_channels=set(),
            consumed_channels={"sentiment", "fundamental"}, formal=True)
    msg = str(ei.value)
    assert "formal mode" in msg
    assert "fundamental" in msg
    assert "manifest missing" in msg


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
                 required_channels=None, consumed_channels=None, formal=False):
        captured["formal"] = formal
        captured["required"] = set(required_channels or ())
        captured["consumed"] = set(consumed_channels or ())
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
    # §v18-2: the live branch ALSO threads the full CONSUMED set — the gate
    # covers every channel the pipeline opens, with required ⊆ consumed.
    assert "sentiment" in captured["consumed"]
    assert captured["required"] <= captured["consumed"]


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


# ── §v19 P0#2: market_env derived-asset lineage freshness gate ─────────────

def test_formal_gate_aborts_stale_market_env_lineage(tmp_path, monkeypatch):
    """§v19 P0#2: a market_env whose manifest lineage no longer matches the
    CURRENT upstreams must abort a formal run with a STALE diagnosis."""
    from scripts.production.train_panel_panel import _enforce_formal_manifests
    from stoke_ml.data.asset_contract import write_asset_manifest
    from stoke_ml.data.broadcast_assets import MARKET_ENV_ASSET

    me_dir = tmp_path / "a_shares" / "market_breadth"
    me_dir.mkdir(parents=True)
    out = me_dir / "market_env_daily.parquet"
    df = pd.DataFrame({
        "high_low_ratio": [0.5], "market_adv_ratio": [0.6],
        "market_turnover_z": [1.0],
        "mkt_cap_total_z": [0.0], "avg_account_cap_z": [0.0],
        "investor_new_num": [1.0], "investor_new_z": [0.0],
    }, index=pd.to_datetime(["2024-01-02"]))
    df.index.name = "date"
    df.to_parquet(str(out))
    write_asset_manifest(
        str(out), MARKET_ENV_ASSET, df, parts={"price": {}, "account": {}},
        upstream_roots={"daily": "AAA"}, transform_code_hash="ccc",
        transform_config_hash="ddd")
    # force stale: the real compute_lineage returns DIFFERENT upstreams
    monkeypatch.setattr(
        "scripts.production.build_market_env.compute_lineage",
        lambda data_dir, parts: {
            "upstream_roots": {"daily": "ZZZ"},
            "transform_code_hash": "ccc", "transform_config_hash": "ddd"})

    with pytest.raises(SystemExit) as ei:
        _enforce_formal_manifests(["000001"], str(tmp_path), "2024-01-01",
                                  "2024-01-31", {"market_env"})
    assert "DERIVED-ASSET STALE" in str(ei.value)


def test_formal_gate_passes_fresh_market_env_lineage(tmp_path, monkeypatch):
    """§v19 P0#2 positive control: a market_env whose lineage MATCHES the
    CURRENT upstreams / transform code / config passes the gate (no SystemExit)
    — the freshness check must not false-positive abort a valid run."""
    from scripts.production.train_panel_panel import _enforce_formal_manifests
    from stoke_ml.data.asset_contract import write_asset_manifest
    from stoke_ml.data.broadcast_assets import MARKET_ENV_ASSET

    me_dir = tmp_path / "a_shares" / "market_breadth"
    me_dir.mkdir(parents=True)
    out = me_dir / "market_env_daily.parquet"
    df = pd.DataFrame({
        "high_low_ratio": [0.5], "market_adv_ratio": [0.6],
        "market_turnover_z": [1.0],
        "mkt_cap_total_z": [0.0], "avg_account_cap_z": [0.0],
        "investor_new_num": [1.0], "investor_new_z": [0.0],
    }, index=pd.to_datetime(["2024-01-02"]))
    df.index.name = "date"
    df.to_parquet(str(out))
    write_asset_manifest(
        str(out), MARKET_ENV_ASSET, df, parts={"price": {}, "account": {}},
        upstream_roots={"daily": "AAA"}, transform_code_hash="ccc",
        transform_config_hash="ddd")
    # fresh: compute_lineage recomputes the SAME lineage the manifest records
    monkeypatch.setattr(
        "scripts.production.build_market_env.compute_lineage",
        lambda data_dir, parts: {
            "upstream_roots": {"daily": "AAA"},
            "transform_code_hash": "ccc", "transform_config_hash": "ddd"})

    assert _enforce_formal_manifests(
        ["000001"], str(tmp_path), "2024-01-01", "2024-01-31",
        {"market_env"}) is None


# ── §v19 P0.3/§十七: industry_ranking lineage + sector active-stock coverage ──
# The market_env chain is Canonical Daily + CNINFO Sector Membership → Industry
# Ranking → Market Env.  These checks run only when a formal run CONSUMES
# market_env AND the underlying chain files exist — a non-market_env run or an
# absent industry_ranking/sector_membership file never triggers them.

def _write_fresh_market_env(tmp_path, monkeypatch):
    """Write a market_env_daily.parquet whose derived lineage matches the
    CURRENT upstreams (fresh) so the market_env chain gate passes; the
    downstream industry_ranking / sector-coverage checks run under it."""
    from stoke_ml.data.asset_contract import write_asset_manifest
    from stoke_ml.data.broadcast_assets import MARKET_ENV_ASSET

    me_dir = tmp_path / "a_shares" / "market_breadth"
    me_dir.mkdir(parents=True)
    out = me_dir / "market_env_daily.parquet"
    df = pd.DataFrame({
        "high_low_ratio": [0.5], "market_adv_ratio": [0.6],
        "market_turnover_z": [1.0],
        "mkt_cap_total_z": [0.0], "avg_account_cap_z": [0.0],
        "investor_new_num": [1.0], "investor_new_z": [0.0],
    }, index=pd.to_datetime(["2024-01-02"]))
    df.index.name = "date"
    df.to_parquet(str(out))
    write_asset_manifest(
        str(out), MARKET_ENV_ASSET, df, parts={"price": {}, "account": {}},
        upstream_roots={"daily": "AAA"}, transform_code_hash="ccc",
        transform_config_hash="ddd")
    # fresh: compute_lineage recomputes the SAME lineage the manifest records
    monkeypatch.setattr(
        "scripts.production.build_market_env.compute_lineage",
        lambda data_dir, parts: {
            "upstream_roots": {"daily": "AAA"},
            "transform_code_hash": "ccc", "transform_config_hash": "ddd"})
    return str(tmp_path)


def _write_industry_ranking(data_dir, upstream_roots):
    """Write industry_ranking.parquet + a valid INDUSTRY_RANKING_ASSET manifest
    carrying the given recorded upstream_roots lineage."""
    from scripts.production.download_industry_ranking import INDUSTRY_RANKING_ASSET
    from stoke_ml.data.asset_contract import write_asset_manifest

    ir_path = os.path.join(data_dir, "a_shares", "industry_ranking.parquet")
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "sector_code": ["SEC0001"], "sector_name": ["制造业"],
        "change_pct": [0.5], "ret_std": [0.1], "n_stocks": [3],
        "up_count": [2], "down_count": [1], "rank": [1],
        "leader": ["000001"], "leader_change": [0.9],
    })
    df.to_parquet(ir_path, index=False, compression="lz4")
    write_asset_manifest(
        ir_path, INDUSTRY_RANKING_ASSET, df,
        upstream_roots=upstream_roots, transform_code_hash="ccc",
        transform_config_hash="ddd",
        membership_source="pit", pit_alignment="verified")
    return ir_path


def _write_sector_membership(data_dir, coverage_by_year):
    """Write sector_membership.parquet + a valid SECTOR_MEMBERSHIP_ASSET manifest
    carrying the given per-year coverage audit."""
    from scripts.production.download_sector_membership import SECTOR_MEMBERSHIP_ASSET
    from stoke_ml.data.asset_contract import write_asset_manifest

    sm_path = os.path.join(data_dir, "a_shares", "sector_membership.parquet")
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02"]),
        "stock_code": ["000001"], "sector_code": ["C"], "sector_name": ["制造业"],
    })
    df.to_parquet(sm_path, index=False, compression="lz4")
    write_asset_manifest(sm_path, SECTOR_MEMBERSHIP_ASSET, df,
                         coverage_by_year=coverage_by_year)
    return sm_path


def test_formal_gate_aborts_stale_industry_ranking_lineage(tmp_path, monkeypatch):
    """§v19 P0.3: a market_env run whose industry_ranking lineage is stale — its
    sector_membership upstream changed WITHOUT an industry_ranking rebuild (the
    §十四 "Tuesday bug") — must abort with an INDUSTRY-RANKING STALE diagnosis."""
    from scripts.production.train_panel_panel import _enforce_formal_manifests

    data_dir = _write_fresh_market_env(tmp_path, monkeypatch)
    _write_industry_ranking(
        data_dir, {"daily": "AAA", "sector_membership": "BBB"})
    # force stale: the real lineage recomputes a DIFFERENT sector_membership root
    monkeypatch.setattr(
        "scripts.production.download_industry_ranking.compute_lineage",
        lambda data_dir, prov: {
            "upstream_roots": {"daily": "AAA", "sector_membership": "ZZZ"},
            "transform_code_hash": "ccc", "transform_config_hash": "ddd"})

    with pytest.raises(SystemExit) as ei:
        _enforce_formal_manifests(["000001"], data_dir, "2024-01-01",
                                  "2024-01-31", {"market_env"})
    msg = str(ei.value)
    assert "INDUSTRY-RANKING STALE" in msg
    assert "sector_membership" in msg
    assert "download_industry_ranking.py" in msg


def test_formal_gate_aborts_low_sector_coverage(tmp_path, monkeypatch):
    """§v19 §十七: a market_env run over a year whose sector chain active-stock
    coverage is below the 0.80 floor must abort with the coverage diagnosis."""
    from scripts.production.train_panel_panel import _enforce_formal_manifests

    data_dir = _write_fresh_market_env(tmp_path, monkeypatch)
    _write_sector_membership(data_dir, {"2024": 0.5})

    with pytest.raises(SystemExit) as ei:
        _enforce_formal_manifests(["000001"], data_dir, "2024-01-01",
                                  "2024-12-31", {"market_env"})
    msg = str(ei.value)
    assert "sector active-stock coverage" in msg
    assert "2024=0.50" in msg
    assert "< 0.8" in msg


def test_formal_gate_passes_adequate_sector_coverage(tmp_path, monkeypatch):
    """§v19 §十七 positive control: per-year coverage at/above the 0.80 floor
    passes the gate (no SystemExit) — the coverage check must not false-positive
    abort a valid run."""
    from scripts.production.train_panel_panel import _enforce_formal_manifests

    data_dir = _write_fresh_market_env(tmp_path, monkeypatch)
    _write_sector_membership(data_dir, {"2024": 0.9})

    assert _enforce_formal_manifests(
        ["000001"], data_dir, "2024-01-01", "2024-12-31",
        {"market_env"}) is None
