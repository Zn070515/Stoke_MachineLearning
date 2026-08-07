"""Unit tests for train_panel.py universe selection and honest eval reporting.

These cover the Track B credibility fixes:
- `_best_eval_metrics` reports the eval nearest the deployed best-val-RankIC
  checkpoint instead of the post-hoc max.
- `_resolve_universe` implements seeded/stratified/index universes that remove
  the alphabetical `sorted(...)[:N]` sampling bias.
- `_save_artifacts` persists the resolved/used stock lists + summary.

None of these read the 109GB feature store.
"""

import dataclasses
import importlib.util
import json
import os
import types

import numpy as np
import pandas as pd
import pytest

import stoke_ml.data.channel_vintage as cv
from stoke_ml.data.calendar import calendar_artifact_hash

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "scripts", "production", "train_panel.py")


@pytest.fixture(scope="module")
def tp():
    spec = importlib.util.spec_from_file_location("train_panel_mod", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── _best_eval_metrics ────────────────────────────────────────────────

def _history(metrics, best_epoch_idx, eval_epochs=None):
    h = {
        "val_metrics": metrics,
        "best_epoch_idx": best_epoch_idx,
    }
    if eval_epochs is not None:
        h["val_eval_epochs"] = eval_epochs
    return h


def test_best_eval_picks_epoch_nearest_best_not_max(tp):
    h = _history(
        [
            {"ls_sharpe": 1.0, "ic_mean": 0.02},
            {"ls_sharpe": 2.5, "ic_mean": 0.05},  # post-hoc max — must NOT win
            {"ls_sharpe": 0.5, "ic_mean": 0.01},
        ],
        best_epoch_idx=10,  # deployed checkpoint at 1-based epoch 11
        eval_epochs=[5, 10, 15],
    )
    m, epoch = tp._best_eval_metrics(h)
    assert epoch == 10
    assert m["ls_sharpe"] == 2.5


def test_best_eval_before_first_eval_falls_back_to_epoch5(tp):
    h = _history([{"ls_sharpe": 1.0}], best_epoch_idx=0, eval_epochs=[5])
    m, epoch = tp._best_eval_metrics(h)
    assert epoch == 5
    assert m["ls_sharpe"] == 1.0


def test_best_eval_legacy_history_uses_epoch5_grid(tp):
    # Old histories lack val_eval_epochs — assume evals at epochs 5,10,15...
    h = _history(
        [{"ls_sharpe": 0.3}, {"ls_sharpe": 0.9}],
        best_epoch_idx=8,  # 1-based 9 → nearest grid point is 10
    )
    m, epoch = tp._best_eval_metrics(h)
    assert epoch == 10
    assert m["ls_sharpe"] == 0.9


def test_best_eval_empty_metrics(tp):
    m, epoch = tp._best_eval_metrics({"val_metrics": []})
    assert m == {}
    assert epoch == 0


# ── _resolve_universe ─────────────────────────────────────────────────

ALL = [
    "000001", "000002", "000004", "000006", "000008", "000010",
    "300001", "300002", "300005",
    "600000", "600004", "600006", "600010",
    "430001", "830799",
]


def test_first_is_sorted_prefix(tp):
    r, desc = tp._resolve_universe(ALL, "first", 5, 42, "data")
    assert r == sorted(ALL)[:5]


def test_random_is_seeded_and_not_code_prefix(tp):
    r1, _ = tp._resolve_universe(ALL, "random", 5, 42, "data")
    r2, _ = tp._resolve_universe(ALL, "random", 5, 42, "data")
    assert r1 == r2  # reproducible
    assert sorted(r1) != sorted(ALL)[:5]  # no code-order bias
    assert len(r1) == 5


def test_random_different_seed_different_sample(tp):
    r1, _ = tp._resolve_universe(ALL, "random", 5, 42, "data")
    r2, _ = tp._resolve_universe(ALL, "random", 5, 42 + 1, "data")
    assert r1 != r2


def test_stratified_covers_all_exchanges(tp):
    r, desc = tp._resolve_universe(ALL, "stratified", 9, 7, "data")
    assert len(r) == 9
    prefixes = {c[0] for c in r}
    assert prefixes & {"6"}      # SH
    assert prefixes & {"0", "3"}  # SZ
    assert prefixes & {"4", "8"}  # BJ


def test_all_ignores_limit(tp):
    r, desc = tp._resolve_universe(ALL, "all", 5, 42, "data")
    assert len(r) == len(ALL)


def test_csi_intersects_available_stocks(tp):
    members_parquet = os.path.join(ROOT, "data", "a_shares",
                                   "index_constituents_hist", "membership.parquet")
    if not os.path.isfile(members_parquet):
        pytest.skip("index membership parquet not on disk")
    # Only 3 stocks "available" — csi union must shrink to those with data.
    subset = ["000001", "600519", "300750"]
    r, desc = tp._resolve_universe(subset, "csi300", None, 42, ROOT + "/data")
    assert set(r) <= set(subset)
    assert r == sorted(subset)  # all three are CSI300 ever-members


def test_csi_drop_refuses_when_gate_enforced(tp, monkeypatch):
    """§八-2: when universe reconciliation is ENFORCED (formal=True, i.e.
    _gate_enforced(args)), csi* members missing daily K-line must refuse with
    the missing list — the requested index is never silently shrunk to whatever
    happens to be on disk."""
    monkeypatch.setattr(
        "scripts.production.train_panel_universe._load_index_universe",
        lambda data_dir, idx_codes: ["000001", "600519", "300750", "600999"])
    with pytest.raises(SystemExit) as ei:
        tp._resolve_universe(ALL, "csi300", None, 42, "data", formal=True)
    msg = str(ei.value)
    assert "600999" in msg
    assert "no daily K-line" in msg


def test_csi_drop_warns_and_records_when_not_enforced(tp, monkeypatch, caplog):
    """§八-2: when reconciliation is NOT enforced (formal=False, e.g.
    --no-require-quality-gate), csi* members missing daily K-line degrade to a
    prominent warning + a recorded drop in the description — the run proceeds
    but the artifact still exposes the gap."""
    import logging
    monkeypatch.setattr(
        "scripts.production.train_panel_universe._load_index_universe",
        lambda data_dir, idx_codes: ["000001", "600519", "300750", "600999"])
    with caplog.at_level(logging.WARNING, logger="train_panel_mod"):
        r, desc = tp._resolve_universe(
            ALL, "csi300", None, 42, "data", formal=False)
    assert "600999" not in r
    assert "dropped 3" in desc   # ALL only contains 000001 of the 4 mocked members
    assert any("600999" in m for m in caplog.messages)
    assert any("no daily K-line" in m for m in caplog.messages)


# ── _save_artifacts ───────────────────────────────────────────────────

def test_save_artifacts_writes_all_files(tp, tmp_path):
    class Args:
        universe = "random"
        seed = 42
        stocks = 3

    out = str(tmp_path)
    tp._save_artifacts(
        out, Args(), ["000001", "600519"], ["000001"],
        "random 2 (seed=42)", {"n_folds": 1},
    )
    assert os.path.isfile(os.path.join(out, "args.json"))
    assert os.path.isfile(os.path.join(out, "universe_resolved.txt"))
    assert os.path.isfile(os.path.join(out, "universe_used.txt"))
    assert os.path.isfile(os.path.join(out, "summary.json"))

    resolved = open(os.path.join(out, "universe_resolved.txt"), encoding="utf-8").read()
    assert "000001" in resolved and "600519" in resolved
    used = open(os.path.join(out, "universe_used.txt"), encoding="utf-8").read()
    assert "600519" not in used.split()
    summary = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8"))
    assert summary["n_folds"] == 1


# ── _check_verified_until_scope (§九-3) ──────────────────────────────

def test_verified_until_refuses_panel_past_2026(tp, tmp_path):
    """A formal run whose panel axis reaches 2027 (forward-estimate closures)
    must be refused — the strict calendar raises past verified_until."""
    gd = np.array(["2024-01-02", "2027-01-04"], dtype="datetime64[D]")
    msg = tp._check_verified_until_scope(
        gd, enforce=True, data_dir=str(tmp_path))
    assert msg is not None
    assert "formal run refused" in msg


def test_verified_until_accepts_within_verified_window(tp, tmp_path):
    gd = np.array(["2024-01-02", "2024-12-31"], dtype="datetime64[D]")
    assert tp._check_verified_until_scope(
        gd, enforce=True, data_dir=str(tmp_path)) is None


def test_verified_until_opt_out_not_enforced(tp, tmp_path):
    """--no-require-quality-gate (enforce=False) lets an exploratory run use a
    panel that reaches past verified_until."""
    gd = np.array(["2024-01-02", "2027-01-04"], dtype="datetime64[D]")
    assert tp._check_verified_until_scope(
        gd, enforce=False, data_dir=str(tmp_path)) is None


def test_verified_until_empty_panel_is_noop(tp, tmp_path):
    assert tp._check_verified_until_scope(
        None, enforce=True, data_dir=str(tmp_path)) is None
    empty = np.array([], dtype="datetime64[D]")
    assert tp._check_verified_until_scope(
        empty, enforce=True, data_dir=str(tmp_path)) is None


def test_verified_until_scope_forwards_data_dir(tp, tmp_path, monkeypatch):
    """§九: _check_verified_until_scope must thread the data_dir it actually
    reads into get_research_calendar — the strict calendar follows the frozen
    exchange_calendar artifact at that root, never the process config default."""
    captured = {}

    class _Cal:
        def get_trading_days(self, lo, hi):
            return []

    def fake_get_research_calendar(*args, **kwargs):
        captured.update(kwargs)
        return _Cal()

    monkeypatch.setattr(
        "scripts.production.train_panel_gates.get_research_calendar",
        fake_get_research_calendar)
    gd = np.array(["2024-01-02", "2024-12-31"], dtype="datetime64[D]")
    assert tp._check_verified_until_scope(
        gd, enforce=True, data_dir=str(tmp_path)) is None
    assert captured.get("data_dir") == str(tmp_path)
    assert captured.get("strict") is True


# ── _calendar_freeze (§九-4) ──────────────────────────────────────────

def test_calendar_freeze_records_checksum_verified_source(tp, tmp_path):
    """§九-4: the version freeze saves calendar content hash + verified_until +
    source, and it is invariant to the artifact's generated_at stamp (two
    saves at different times hash the same)."""
    from stoke_ml.data.calendar import save_calendar
    save_calendar(str(tmp_path), "a_shares")
    a = tp._calendar_freeze(str(tmp_path))
    b = tp._calendar_freeze(str(tmp_path))
    assert a == b
    assert set(a) == {"calendar_artifact_hash", "verified_until", "calendar_source"}
    assert len(a["calendar_artifact_hash"]) == 16
    assert a["verified_until"] == "2026-12-31"
    assert a["calendar_source"]


def test_calendar_freeze_falls_back_to_code_formula(tp, tmp_path):
    """No artifact on disk → the freeze uses the code-derived frame (what the
    panel pipeline actually consumes) and still records all three fields."""
    a = tp._calendar_freeze(str(tmp_path))
    assert set(a) == {"calendar_artifact_hash", "verified_until", "calendar_source"}
    assert a["verified_until"] == "2026-12-31"
    assert len(a["calendar_artifact_hash"]) == 16


def test_calendar_freeze_artifact_and_formula_hash_agree(tp, tmp_path):
    """The artifact is generated from the same holiday set, and the hash strips
    generated_at — so a disk artifact and the code formula must hash equal."""
    from stoke_ml.data.calendar import save_calendar
    formula = tp._calendar_freeze(str(tmp_path))
    save_calendar(str(tmp_path), "a_shares")
    artifact = tp._calendar_freeze(str(tmp_path))
    assert formula == artifact


def test_calendar_freeze_sensitive_to_holiday_change(tp, tmp_path, monkeypatch):
    """A single holiday flip must flip the content hash — the whole point of
    hashing calendar CONTENT rather than a manual version string."""
    base = tp._calendar_freeze(str(tmp_path))

    def fake_build(market="a_shares"):
        from stoke_ml.data.calendar import build_calendar_frame
        df = build_calendar_frame(market)
        df.loc[df.index[0], "is_open"] = not bool(df.iloc[0]["is_open"])
        return df

    # _calendar_freeze was split into train_panel_registry (§二十一); it
    # resolves build_calendar_frame from THAT module's namespace, so the
    # holiday-flip monkeypatch must land there, not on the train_panel shell.
    import scripts.production.train_panel_registry as registry
    monkeypatch.setattr(registry, "build_calendar_frame", fake_build)
    changed = tp._calendar_freeze(str(tmp_path))
    assert changed["calendar_artifact_hash"] != base["calendar_artifact_hash"]
    assert changed["verified_until"] == base["verified_until"]
    assert changed["calendar_source"] == base["calendar_source"]


def test_experiment_version_embeds_calendar_freeze(tp, tmp_path):
    """The version dict that lands in version.json/summary.json carries the
    §九-4 calendar content fields alongside the manual version string."""
    from stoke_ml.models.panel import PanelConfig
    ver = tp._experiment_version(
        str(tmp_path), [], None, 0, 0, 0, PanelConfig(),
        "2020-01-01", "2024-12-31", 42,
    )
    assert ver["calendar_version"] == tp.TradingCalendar.CALENDAR_VERSION
    assert len(ver["calendar_artifact_hash"]) == 16
    assert ver["verified_until"] == "2026-12-31"
    assert ver["calendar_source"]


# ── _require_universe_artifacts (§P0-7) ────────────────────────────────

def _write_universe_artifacts(root, *, membership=False, delisted=False, ipo=False):
    base = root / "a_shares"
    if membership:
        d = base / "index_constituents_hist"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"stock_code": ["600000"], "index_code": ["000300"],
                      "in_date": [pd.Timestamp("2015-01-31")],
                      "out_date": [pd.NaT]}).to_parquet(d / "membership.parquet")
    if delisted:
        d = base / "universe"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"公司代码": ["000002"],
                      "暂停上市日期": [pd.Timestamp("2015-06-30")]}).to_parquet(
            d / "delisted.parquet")
    if ipo:
        d = base / "universe"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"stock_code": ["000001"],
                      "list_date": [pd.Timestamp("2000-01-01")]}).to_parquet(
            d / "ipo.parquet")


def test_formal_missing_membership_blocks_csi_universe(tp, tmp_path):
    """§P0-7: csi300 with no membership.parquet must REFUSE to start in formal
    mode — the per-day membership gate would silently no-op into the historical
    union, measuring the wrong task."""
    _write_universe_artifacts(tmp_path, delisted=True)
    with pytest.raises(SystemExit):
        tp._require_universe_artifacts(str(tmp_path), "csi300", formal=True)


def test_formal_missing_delist_records_blocks_every_universe(tp, tmp_path):
    """§P0-7: delisting records feed the force-sell exit policy, so a formal
    run with no delisted.parquet must fail even for the default random
    universe."""
    _write_universe_artifacts(tmp_path, ipo=True)
    with pytest.raises(SystemExit):
        tp._require_universe_artifacts(str(tmp_path), "random", formal=True)


def test_formal_missing_ipo_blocks_all_universe(tp, tmp_path):
    """§P0-7: the `all` universe merges delisted stocks via IPO records; without
    them the survivorship-free merge is fake, so formal mode must fail."""
    _write_universe_artifacts(tmp_path, delisted=True)
    with pytest.raises(SystemExit):
        tp._require_universe_artifacts(str(tmp_path), "all", formal=True)


def test_exploratory_missing_artifact_degrades_without_exit(tp, tmp_path):
    """§P0-7: exploratory (--no-formal) may proceed with a degraded gate — the
    run is explicitly marked, not silently treated as complete."""
    _write_universe_artifacts(tmp_path, delisted=True)
    tp._require_universe_artifacts(str(tmp_path), "csi300", formal=False)


def test_formal_all_artifacts_present_passes(tp, tmp_path):
    _write_universe_artifacts(tmp_path, membership=True, delisted=True, ipo=True)
    tp._require_universe_artifacts(str(tmp_path), "csi300", formal=True)
    tp._require_universe_artifacts(str(tmp_path), "all", formal=True)


def test_empty_membership_counts_as_missing(tp, tmp_path):
    """§P0-7: an EMPTY membership.parquet is as silent a no-op as a missing one
    — formal mode must treat it as missing too."""
    d = tmp_path / "a_shares" / "index_constituents_hist"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=["stock_code", "index_code", "in_date", "out_date"]).to_parquet(
        d / "membership.parquet")
    _write_universe_artifacts(tmp_path, delisted=True)
    with pytest.raises(SystemExit):
        tp._require_universe_artifacts(str(tmp_path), "csi500", formal=True)


# ── §八.3 gate labeling + strict-index-training ────────────────────────

def test_gate_inner_train_membership_ands_into_entry_mask(tp):
    """§八.3 strict mode: per-day membership ANDs into the inner-train
    entry_eligible_mask; both the stock's own eligibility and membership must
    be True for the sample to stay trainable (only True&True survives).  mem is
    the FULL panel grid (rows index into it); rows/cols pull the fold's
    submatrix."""
    inner = {"entry_eligible_mask": np.ones((3, 5), dtype=bool)}
    mem = np.zeros((12, 5), dtype=bool)
    mem[7] = [True, True, False, False, True]    # off-member mid-window
    mem[9] = [False, True, True, True, True]     # not a member day 0
    mem[11] = [True, True, True, True, True]     # always a member
    tp._gate_inner_train_membership(
        inner, mem, rows=np.array([7, 9, 11]), cols=np.arange(5))
    # Entry mask is AND of all-True with the selected membership rows.
    assert inner["entry_eligible_mask"].tolist() == [
        [True, True, False, False, True],
        [False, True, True, True, True],
        [True, True, True, True, True],
    ]


def test_gate_inner_train_membership_never_restores_disqualified(tp):
    """Membership is ANDed, never ORed: a stock already disqualified by the
    pipeline (entry_eligible_mask False) stays False even if it IS a member."""
    inner = {"entry_eligible_mask": np.array([
        [True, True, True],
        [False, False, False],   # stock 1 already excluded everywhere
    ])}
    mem = np.ones((2, 3), dtype=bool)
    tp._gate_inner_train_membership(
        inner, mem, rows=np.array([0, 1]), cols=np.arange(3))
    assert inner["entry_eligible_mask"][1].tolist() == [False, False, False]
    assert inner["entry_eligible_mask"][0].tolist() == [True, True, True]


def test_gate_inner_train_membership_row_col_subselection(tp):
    """rows/cols must pick the right submatrix from the full grid — a fold that
    keeps only some stocks (rows) and some inner-train columns (cols) gets
    exactly the corresponding membership cells, aligned to the fold's own
    row/column order."""
    mem = np.full((4, 6), False, dtype=bool)
    mem[1, 2] = True   # (grid row 1, grid col 2) is the only member cell
    inner = {"entry_eligible_mask": np.ones((2, 3), dtype=bool)}
    tp._gate_inner_train_membership(
        inner, mem, rows=np.array([0, 1]), cols=np.array([2, 3, 4]))
    # inner row 0 ← grid row 0 (no members), inner row 1 ← grid row 1:
    # grid col 2 → inner col 0 is True, rest False.
    assert inner["entry_eligible_mask"].tolist() == [
        [False, False, False],
        [True, False, False],
    ]


def test_gate_descriptions_default_vs_strict(tp):
    """§八.3 summary labeling: default trains on the broad union (ungated)
    while evaluation gates 未退市 (+ membership when consumed); strict mode
    gates the train loss by per-day membership."""
    # csi universe consumes membership; strict off → train union.
    assert tp._gate_descriptions(True, False) == (
        "not_delisted + per-day-membership", "union (ungated)")
    # csi universe consumes membership; strict on → train membership-gated.
    assert tp._gate_descriptions(True, True) == (
        "not_delisted + per-day-membership", "per-day-membership")
    # non-membership universe (all/stratified) → eval is not_delisted only;
    # strict has no membership to gate, so train stays union.
    assert tp._gate_descriptions(False, False) == ("not_delisted", "union (ungated)")
    assert tp._gate_descriptions(False, True) == ("not_delisted", "union (ungated)")


# ── _experiment_signature (§十二.5 / §P1-8) ───────────────────────────

def test_signature_changes_with_seed(tp):
    """§P1-8: two runs differing ONLY in the random seed are DIFFERENT
    trials — the review's canonical keep-best-of-N-seeds case.  The old
    signature (no seed) conflated them into one trial, undercounting DSR N."""
    from stoke_ml.models.panel import PanelConfig

    base = {
        "data_manifest_hash": "d1", "feature_schema_hash": "f1",
        "universe_hash": "u1", "model_hash": "m1",
        "evaluator_version": "ev1", "calendar_version": "cv1",
        "calendar_artifact_hash": "ch1",
    }
    cfg = PanelConfig(seq_len=60, static_dim=5, past_known_dim=10,
                      past_observed_dim=20, horizon=1, seed=42)
    s42 = tp._experiment_signature(base, cfg)
    assert s42 != tp._experiment_signature(base, dataclasses.replace(cfg, seed=7))
    # Same research choices → same signature (a re-run, not a new trial).
    assert tp._experiment_signature(base, cfg) == s42


def test_signature_changes_with_research_choices(tp):
    """§P1-8: horizon / seq_len / txn_cost / calendar / evaluator all enter
    the signature — changing any research choice is a NEW trial."""
    from stoke_ml.models.panel import PanelConfig

    base = {
        "data_manifest_hash": "d1", "feature_schema_hash": "f1",
        "universe_hash": "u1", "model_hash": "m1",
        "evaluator_version": "ev1", "calendar_version": "cv1",
        "calendar_artifact_hash": "ch1",
    }
    cfg = PanelConfig(seq_len=60, static_dim=5, past_known_dim=10,
                      past_observed_dim=20, horizon=1, seed=42)
    base_sig = tp._experiment_signature(base, cfg)
    assert (tp._experiment_signature(base, dataclasses.replace(cfg, horizon=5))
            != base_sig)
    assert (tp._experiment_signature(base, dataclasses.replace(cfg, seq_len=30))
            != base_sig)
    assert (tp._experiment_signature(base, dataclasses.replace(cfg, txn_cost=0.001))
            != base_sig)
    assert (tp._experiment_signature({**base, "evaluator_version": "ev2"}, cfg)
            != base_sig)
    assert (tp._experiment_signature({**base, "calendar_artifact_hash": "ch2"}, cfg)
            != base_sig)


def test_signature_binds_vintage_policy(tp):
    """§T19: two runs differing ONLY in the vintage-admission policy train on
    materially different channels — safe-only denies latest_revised_aligned
    (fundamental/macro/earnings/valuation/pledge/shareholder/index_membership/
    market_env_refine) that allow-revised admits — so they MUST be distinct
    trials, never conflated into one experiment."""
    from stoke_ml.models.panel import PanelConfig

    base = {
        "data_manifest_hash": "d1", "feature_schema_hash": "f1",
        "universe_hash": "u1", "model_hash": "m1",
        "evaluator_version": "ev1", "calendar_version": "cv1",
        "calendar_artifact_hash": "ch1",
    }
    cfg = PanelConfig(seq_len=60, static_dim=5, past_known_dim=10,
                      past_observed_dim=20, horizon=1, seed=42)
    s_safe = tp._experiment_signature(base, cfg, vintage_policy="safe-only")
    assert (tp._experiment_signature(base, cfg, vintage_policy="allow-revised")
            != s_safe)
    # Same policy + same everything else → same signature (a re-run).
    assert tp._experiment_signature(base, cfg, vintage_policy="safe-only") == s_safe


def test_signature_binds_feature_profile(tp):
    """§T19: a different frozen feature profile is a different feature recipe
    (different required_channels + per-channel coverage minimums) — two runs
    differing ONLY in the profile must be distinct trials."""
    from stoke_ml.models.panel import PanelConfig

    base = {
        "data_manifest_hash": "d1", "feature_schema_hash": "f1",
        "universe_hash": "u1", "model_hash": "m1",
        "evaluator_version": "ev1", "calendar_version": "cv1",
        "calendar_artifact_hash": "ch1",
    }
    cfg = PanelConfig(seq_len=60, static_dim=5, past_known_dim=10,
                      past_observed_dim=20, horizon=1, seed=42)
    s_none = tp._experiment_signature(base, cfg, feature_profile="none")
    assert (tp._experiment_signature(base, cfg, feature_profile="headline_v1")
            != s_none)
    # Same profile + same everything else → same signature (a re-run).
    assert tp._experiment_signature(base, cfg, feature_profile="none") == s_none


def test_signature_vintage_profile_default_to_none(tp):
    """§T19: callers that omit the new levers (None defaults — the baseline
    script) must hash exactly as if the literal 'none' were passed, so their
    signatures stay stable across this upgrade."""
    from stoke_ml.models.panel import PanelConfig

    base = {
        "data_manifest_hash": "d1", "feature_schema_hash": "f1",
        "universe_hash": "u1", "model_hash": "m1",
        "evaluator_version": "ev1", "calendar_version": "cv1",
        "calendar_artifact_hash": "ch1",
    }
    cfg = PanelConfig(seq_len=60, static_dim=5, past_known_dim=10,
                      past_observed_dim=20, horizon=1, seed=42)
    assert (tp._experiment_signature(base, cfg)
            == tp._experiment_signature(
                base, cfg, vintage_policy="none", feature_profile="none"))


# ── _require_quality_gate (§九.1 custom-prebuilt binding / §八-2 universe) ──

_VALID_RECON = {
    "ok": True, "requested_count": 2, "present_count": 2,
    "missing_count": 0, "degraded_count": 0,
    "missing_codes": [], "degraded_codes": [],
}


def _gate_report(tp, data_dir, required, dataset_paths=None, passed=True,
                 scope="full", manifest_contract_full_scan=True,
                 fingerprint=None, universe_reconciliation=_VALID_RECON,
                 checks=None):
    """Build a report that would PASS every §六-2 check except the ones the
    caller intentionally varies, so a test isolates the §九.1/§八-2 behavior.

    Defaults to a full-scope PASS: a checks array with ``manifest`` and
    ``contract_schema`` both passed (the §八-2 full-scope floor), and a valid
    ``universe_reconciliation`` — §八-2 requires reconciliation for EVERY
    enforced run, so the happy path carries one."""
    d = os.path.realpath(str(data_dir))
    req = sorted(required)
    if checks is None:
        checks = [
            {"name": "manifest", "passed": True},
            {"name": "contract_schema", "passed": True},
        ]
    return {
        "passed": passed,
        "quality_gate_version": tp.QUALITY_GATE_VERSION,
        "data_root": d,
        "calendar_version": tp.TradingCalendar.CALENDAR_VERSION,
        # §八: the report must BIND the calendar artifact's content hash so the
        # gate can vouch for the calendar training reads, not just its version
        # string.  In a tmp data dir with no artifact both the fixture value and
        # the live value are the deterministic code-derived hash.
        "calendar_artifact_hash": calendar_artifact_hash(d, "a_shares"),
        "contract_version": tp.contract_version(),
        "required_datasets": req,
        "dataset_paths": dataset_paths or {},
        "scope": scope,
        "manifest_contract_full_scan": manifest_contract_full_scan,
        "data_manifest_hash": fingerprint or tp.dataset_fingerprint(d, req),
        "checks": checks,
        "universe_reconciliation": universe_reconciliation,
    }


def test_require_quality_gate_custom_prebuilt_consumed(tp, tmp_path):
    """§九.1: a custom prebuilt basename (not features/features_panel) is added
    to ``consumed`` and must be covered by the gate's required datasets — the
    old fixed-basename whitelist silently skipped it."""
    data_dir = tmp_path / "data"
    prebuilt = data_dir / "features_panel_v2"
    report_path = tmp_path / "gate.json"
    report = _gate_report(tp, data_dir, ["daily", "features_panel_v2"],
                          dataset_paths={
                              "daily": str((data_dir / "a_shares" / "daily").resolve()),
                              "features_panel_v2": str(prebuilt.resolve()),
                          })
    report_path.write_text(json.dumps(report), encoding="utf-8")
    out = tp._require_quality_gate(str(data_dir), str(prebuilt), str(report_path))
    assert out["required_datasets"] == ["daily", "features_panel_v2"]


def test_require_quality_gate_refuses_missing_custom_dataset(tp, tmp_path):
    """§九.1: a custom prebuilt that the gate did NOT validate is refused —
    consumed must be a subset of required_datasets."""
    data_dir = tmp_path / "data"
    prebuilt = data_dir / "features_panel_v2"
    report_path = tmp_path / "gate.json"
    report = _gate_report(tp, data_dir, ["daily"])
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit):
        tp._require_quality_gate(str(data_dir), str(prebuilt), str(report_path))


def test_require_quality_gate_refuses_path_mismatch(tp, tmp_path):
    """§九.1: if the gate validated the prebuilt at a DIFFERENT absolute dir
    than training reads, the run is refused instead of trusted."""
    data_dir = tmp_path / "data"
    prebuilt = data_dir / "features_panel_v2"
    report_path = tmp_path / "gate.json"
    report = _gate_report(tp, data_dir, ["daily", "features_panel_v2"],
                          dataset_paths={
                              "daily": str((data_dir / "a_shares" / "daily").resolve()),
                              "features_panel_v2": str(
                                  (data_dir / "elsewhere").resolve()),
                          })
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit):
        tp._require_quality_gate(str(data_dir), str(prebuilt), str(report_path))


# ── §八: the gate binds the calendar artifact's CONTENT hash, not just the
# version string.  A content edit that keeps CALENDAR_VERSION, or a report that
# never bound a hash, must be REFUSED. ────────────────────────────────

def test_require_quality_gate_refuses_calendar_content_change(tp, tmp_path):
    """§八: a gate PASS report whose calendar artifact CONTENT changed since the
    gate (a holiday row flipped, CALENDAR_VERSION untouched) must be refused —
    the version check alone cannot catch a same-version content edit."""
    from stoke_ml.data.calendar import load_calendar, save_calendar
    data_dir = tmp_path / "data"
    prebuilt = data_dir / "features_panel"
    report_path = tmp_path / "gate.json"
    # Report bound the calendar as it stood when the gate PASSed — at this point
    # no artifact exists, so the fixture records the code-derived hash.
    report = _gate_report(tp, data_dir, ["daily", "features_panel"],
                          dataset_paths={
                              "daily": str((data_dir / "a_shares" / "daily").resolve()),
                              "features_panel": str(prebuilt.resolve()),
                          })
    report_path.write_text(json.dumps(report), encoding="utf-8")
    # Edit the artifact CONTENT with the SAME CALENDAR_VERSION: flip one real
    # trading day to closed.  load_calendar still accepts the frame (no date
    # gaps), so only the content hash changes.
    save_calendar(str(data_dir), "a_shares")
    frame = load_calendar(str(data_dir), "a_shares")
    assert str(frame["version"].iloc[0]) == tp.TradingCalendar.CALENDAR_VERSION
    frame.loc[frame["date"] == pd.Timestamp("2010-02-24"), "is_open"] = False
    frame.to_parquet(str(data_dir / "exchange_calendar" / "a_shares.parquet"))
    with pytest.raises(SystemExit) as ei:
        tp._require_quality_gate(str(data_dir), str(prebuilt), str(report_path))
    assert "calendar artifact changed since the gate PASS" in str(ei.value)


def test_require_quality_gate_missing_calendar_hash_refuses(tp, tmp_path):
    """§八: a report WITHOUT a bound calendar_artifact_hash (an old gate report,
    or a gate that ran with NO calendar artifact present) must be REFUSED — the
    gate cannot vouch for calendar content it never bound.  A None/missing hash
    on one side is a refusal, never a skip-the-comparison escape."""
    data_dir = tmp_path / "data"
    prebuilt = data_dir / "features_panel"
    report_path = tmp_path / "gate.json"
    report = _gate_report(tp, data_dir, ["daily", "features_panel"],
                          dataset_paths={
                              "daily": str((data_dir / "a_shares" / "daily").resolve()),
                              "features_panel": str(prebuilt.resolve()),
                          })
    del report["calendar_artifact_hash"]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        tp._require_quality_gate(str(data_dir), str(prebuilt), str(report_path))
    assert "calendar artifact changed since the gate PASS" in str(ei.value)


# ── §八-2: requested-universe reconciliation is REQUIRED ──────────────

def _recon_with_missing(*missing, requested=3, present=1, degraded=()):
    return {
        "ok": not missing and not degraded,
        "requested_count": requested, "present_count": present,
        "missing_count": len(missing), "degraded_count": len(degraded),
        "missing_codes": list(missing), "degraded_codes": list(degraded),
    }


def _universe_report(tp, tmp_path, *, recon, checks):
    """A full-scope report whose ONLY problem is the universe reconciliation."""
    data_dir = tmp_path / "data"
    prebuilt = data_dir / "features_panel"
    report_path = tmp_path / "gate.json"
    report = _gate_report(
        tp, data_dir, ["daily", "features_panel"],
        universe_reconciliation=recon,
        passed=all(c["passed"] for c in checks),
        checks=checks,
        dataset_paths={
            "daily": str((data_dir / "a_shares" / "daily").resolve()),
            "features_panel": str(prebuilt.resolve()),
        },
    )
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return data_dir, prebuilt, report_path


def test_require_quality_gate_refuses_report_without_reconciliation(tp, tmp_path):
    """§八-2: EVERY enforced run (not just csi*/requested) requires the gate
    report to carry a universe_reconciliation — a report that never accounted
    for the requested universe cannot vouch for training on it."""
    data_dir = tmp_path / "data"
    prebuilt = data_dir / "features_panel"
    report_path = tmp_path / "gate.json"
    report = _gate_report(tp, data_dir, ["daily", "features_panel"],
                          universe_reconciliation=None,
                          dataset_paths={
                              "daily": str((data_dir / "a_shares" / "daily").resolve()),
                              "features_panel": str(prebuilt.resolve()),
                          })
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        tp._require_quality_gate(str(data_dir), str(prebuilt), str(report_path))
    assert "universe reconciliation missing" in str(ei.value)


def test_require_quality_gate_refuses_missing_stocks_lists_codes(tp, tmp_path):
    """§八-2: missing requested stocks are refused, and the missing codes are
    listed explicitly (never silently dropped)."""
    recon = _recon_with_missing("300750", "600519")
    checks = [
        {"name": "manifest", "passed": True},
        {"name": "contract_schema", "passed": True},
        {"name": "universe", "passed": False},
    ]
    data_dir, prebuilt, report_path = _universe_report(tp, tmp_path, recon=recon, checks=checks)
    with pytest.raises(SystemExit) as ei:
        tp._require_quality_gate(str(data_dir), str(prebuilt), str(report_path))
    msg = str(ei.value)
    assert "300750" in msg and "600519" in msg
    assert "missing stocks" in msg


def test_allow_missing_universe_escape_proceeds_and_surfaces_missing(tp, tmp_path):
    """§八-2: --allow-missing-universe is the explicit escape — a report that
    FAILED purely on the universe check (missing stocks, nothing degraded) is
    accepted, and the missing list is surfaced for the caller to record."""
    recon = _recon_with_missing("300750", "600519")
    checks = [
        {"name": "manifest", "passed": True},
        {"name": "contract_schema", "passed": True},
        {"name": "universe", "passed": False},
    ]
    data_dir, prebuilt, report_path = _universe_report(tp, tmp_path, recon=recon, checks=checks)
    out = tp._require_quality_gate(
        str(data_dir), str(prebuilt), str(report_path), allow_missing=True)
    assert out["passed"] is False            # the report stays honest
    assert out["universe_reconciliation"]["missing_codes"] == ["300750", "600519"]


def test_allow_missing_does_not_escape_degraded_stocks(tp, tmp_path):
    """§八-2: the escape is scoped to MISSING stocks — a report that also flags
    present-but-degraded stocks is still refused (degraded is never escapable)."""
    recon = _recon_with_missing("600519", degraded=[{"code": "000002", "reason": "manifest_invalid"}])
    checks = [
        {"name": "manifest", "passed": True},
        {"name": "contract_schema", "passed": True},
        {"name": "universe", "passed": False},
    ]
    data_dir, prebuilt, report_path = _universe_report(tp, tmp_path, recon=recon, checks=checks)
    with pytest.raises(SystemExit) as ei:
        tp._require_quality_gate(
            str(data_dir), str(prebuilt), str(report_path), allow_missing=True)
    assert "degraded stocks" in str(ei.value)


# ── _exchange_group (§六 single market authority + BJ fallback) ───────

def test_exchange_group_known_equity_prefixes(tp):
    assert tp._exchange_group("600519") == "SH"
    assert tp._exchange_group("000001") == "SZ"
    assert tp._exchange_group("300750") == "SZ"
    assert tp._exchange_group("830799") == "BJ"


def test_exchange_group_unknown_prefix_falls_back_to_bj(tp):
    """Anything market_of_code does not recognize (legacy 老三板 4/8 codes AND
    non-equity/garbage prefixes) must bucket as BJ — never silently SH."""
    assert tp._exchange_group("400001") == "BJ"   # legacy 老三板
    assert tp._exchange_group("820001") == "BJ"
    assert tp._exchange_group("123456") == "BJ"   # not a known equity prefix


# ── _require_single_use_lockbox (§二十 default-off) ───────────────────

def test_lockbox_default_off_never_touches_marker(tp, tmp_path):
    """§二十: lockbox_months=0 (the new DEFAULT) must never write the marker
    and must never refuse — even a formal run with a prior marker present."""
    marker = str(tmp_path / "lockbox_used.json")
    # Prior use exists; a default (0-month) run must ignore it and not raise.
    tp._mark_lockbox_used(marker, {"universe": "prior"})
    tp._require_single_use_lockbox(
        0, formal=True, marker_path=marker,
        info={"universe": "default", "lockbox_months": 0})
    # The prior marker is untouched (still the prior run's record).
    with open(marker, encoding="utf-8") as fh:
        assert json.load(fh)["universe"] == "prior"


def test_lockbox_default_off_no_marker_written_when_absent(tp, tmp_path):
    marker = str(tmp_path / "lockbox_used.json")
    tp._require_single_use_lockbox(
        0, formal=True, marker_path=marker,
        info={"universe": "default", "lockbox_months": 0})
    assert not os.path.isfile(marker)


def test_lockbox_exploratory_run_never_touches_marker(tp, tmp_path):
    """§二十: non-formal (--no-require-quality-gate / --no-formal) runs never
    touch the marker even when a lockbox is requested."""
    marker = str(tmp_path / "lockbox_used.json")
    tp._require_single_use_lockbox(
        12, formal=False, marker_path=marker,
        info={"universe": "explore", "lockbox_months": 12})
    assert not os.path.isfile(marker)


def test_lockbox_open_refuses_second_formal_use(tp, tmp_path):
    """§二十: the lockbox is single-use — a formal run that opens it with a
    prior marker present is refused (this is the behavior the default-off
    change keeps opt-in)."""
    marker = str(tmp_path / "lockbox_used.json")
    tp._mark_lockbox_used(marker, {"universe": "final", "opened_at": "2026-08-05"})
    with pytest.raises(SystemExit) as ei:
        tp._require_single_use_lockbox(
            12, formal=True, marker_path=marker,
            info={"universe": "sneak", "lockbox_months": 12})
    assert "单次开启" in str(ei.value) or "single" in str(ei.value).lower()


# ── §七-P0 universe memory guard ─────────────────────────────────────

def test_require_all_universe_prebuilt_refuses_without_prebuilt(tp):
    """--universe all without --prebuilt is refused outright — the full market
    cannot be feature-engineered in RAM (§七-P0)."""
    with pytest.raises(SystemExit):
        tp._require_all_universe_prebuilt("all", None)


def test_require_all_universe_prebuilt_allows_prebuilt(tp):
    """--universe all WITH --prebuilt proceeds; a non-all universe needs no
    prebuilt at all."""
    tp._require_all_universe_prebuilt("all", "data/features_panel")
    tp._require_all_universe_prebuilt("random", None)


def test_panel_memory_gb_formula(tp):
    """§七-P0 estimate = n_stocks × n_timesteps × n_features × 4B ÷ 1024³.

    (The full-market 5530×5000×8000 float32 panel is ~824 GB — well past the
    96 GB ceiling; the check pins the function to the documented formula.)"""
    expected = 5530 * 5000 * 8000 * 4 / (1024 ** 3)
    assert tp._panel_memory_gb(5530, 5000, 8000) == pytest.approx(expected, abs=0.5)


def test_enforce_all_universe_refuses_by_default(tp):
    """--universe all above the 48 GB refuse line is refused by default, with
    an estimate, the host-memory caveat, and the escape hatch named."""
    with pytest.raises(SystemExit) as ei:
        tp._enforce_universe_memory("all", 5530, 5000, 8000)
    msg = str(ei.value)
    assert "universe=all" in msg
    assert "GB" in msg
    assert "--allow-high-risk-universe" in msg


def test_enforce_all_universe_small_panel_ok(tp):
    """A small --universe all cap (est ≪ 48 GB) is 'ok' and never raises."""
    est, action = tp._enforce_universe_memory("all", 50, 1000, 100)
    assert action == "ok"
    assert est == pytest.approx(50 * 1000 * 100 * 4 / (1024 ** 3), abs=0.5)


def test_enforce_csi800_above_hard_ceiling_refuses(tp):
    """csi800 above the 96 GB hard ceiling is refused (est ≈ 111.8 GB)."""
    with pytest.raises(SystemExit) as ei:
        tp._enforce_universe_memory("csi800", 3000, 5000, 2000)
    assert "universe=csi800" in str(ei.value)
    assert "--allow-high-risk-universe" in str(ei.value)


def test_enforce_csi800_warn_band_warns(tp):
    """csi800 in the warn band (48 < est < 96 GB) warns and does NOT raise."""
    est, action = tp._enforce_universe_memory("csi800", 1500, 5000, 2000)
    assert action == "warn"
    assert 48.0 < est < 96.0


def test_enforce_override_downgrades_refuse_to_warning(tp, caplog):
    """--allow-high-risk-universe downgrades the refusal to a prominent
    WARNING; the UN-overridden verdict is still reported as 'refuse'."""
    import logging
    with caplog.at_level(logging.WARNING, logger="train_panel_mod"):
        est, action = tp._enforce_universe_memory(
            "all", 5530, 5000, 8000, allow_override=True)
    assert action == "refuse"   # the un-overridden verdict stays "refuse"
    assert any("§七-P0 risk" in m for m in caplog.messages)
    assert any("--allow-high-risk-universe" in m for m in caplog.messages)


def test_enforce_available_gb_precheck_refuses(tp):
    """When host available memory is known and est > available, the panel
    cannot fit THIS host — refused even though est is below the 48 GB line."""
    with pytest.raises(SystemExit) as ei:
        tp._enforce_universe_memory("all", 300, 2000, 1000, available_gb=2.0)
    msg = str(ei.value)
    assert "universe=all" in msg
    assert "available" in msg


def test_enforce_available_gb_precheck_skips_other_universes(tp):
    """§七-P0: the available-memory precheck applies ONLY to all/csi800 — a
    transiently low host `available` snapshot must not refuse a documented
    default run on a smaller universe."""
    est, action = tp._enforce_universe_memory(
        "random", 500, 5000, 8000, available_gb=1.0)
    assert action == "ok"      # est ~74.5 GB > 1.0 GB, but random is not prechecked
    assert est > 1.0


# ── §七-P0 pre-build (T5) memory estimate / early guard ────────────────

def _prebuilt_schema_df(extra_cols):
    """A tiny multi-row DataFrame whose SCHEMA (only) the estimate reads."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "stock_code": ["000001", "000001"],
        "close": [10.0, 10.5],
        "ma_5": [9.9, 10.2],
        **extra_cols,
    })


def test_estimate_panel_memory_live_build_none(tp):
    """Live build (--prebuilt unset) → no pre-build estimate → the early guard
    is skipped; the post-build actual-dims backstop covers live builds."""
    args = _panel_args("allow-revised", prebuilt=None)
    assert tp._estimate_panel_memory(args, ["000001"], "irrelevant") is None


def test_estimate_panel_memory_missing_prebuilt_none(tp, tmp_path):
    """A prebuilt dir with no *.parquet → None (never crash the estimate path)."""
    args = _panel_args("allow-revised", prebuilt=str(tmp_path))
    assert tp._estimate_panel_memory(args, ["000001"], str(tmp_path)) is None


def test_estimate_panel_memory_reads_schema(tp, tmp_path):
    """The estimate reads the FIRST prebuilt parquet's SCHEMA (not its data) and
    drops exactly the columns the build drops: *_lag{N}, topic_* when use_topic
    is off, FUNDAMENTAL_COLS when use_fundamental is off, plus the date/
    stock_code identifiers."""
    df = _prebuilt_schema_df({
        "ma_5_lag1": [9.8, 9.9],      # *_lag{N} — always dropped
        "topic_entropy": [0.1, 0.2],  # topic_* — dropped (use_topic off)
        "roe": [0.1, 0.1],            # FUNDAMENTAL_COLS — dropped (safe-only)
    })
    df.to_parquet(str(tmp_path / "000001.parquet"))
    stock_list = ["000001", "000002", "000003"]
    args = _panel_args(
        "safe-only", prebuilt=str(tmp_path), seq_len=60,
        start="2024-01-01", end="2024-12-31", universe="all",
    )
    n, t, d = tp._estimate_panel_memory(args, stock_list, str(tmp_path))
    assert n == len(stock_list)
    assert t >= 1                       # a full year has positive trading days
    assert d == 2                       # survivors: close, ma_5


def test_estimate_panel_memory_fundamental_ablation_keeps_roe(tp, tmp_path):
    """--allow-fundamental-ablation forces use_fundamental=True → the roe column
    survives the estimate (the prebuilt parquet's fundamental cols are NOT
    scrubbed on an ablation run)."""
    df = _prebuilt_schema_df({"roe": [0.1, 0.1]})
    df.to_parquet(str(tmp_path / "000001.parquet"))
    stock_list = ["000001"]
    args = _panel_args(
        "safe-only", allow_fundamental_ablation=True,
        prebuilt=str(tmp_path), seq_len=60,
        start="2024-01-01", end="2024-12-31", universe="all",
    )
    n, t, d = tp._estimate_panel_memory(args, stock_list, str(tmp_path))
    assert d == 3                       # close, ma_5, roe


def test_early_enforce_universe_memory_refuses_oversized(tp, tmp_path):
    """End-to-end: a huge resolved universe + a many-column prebuilt schema →
    the estimate exceeds the 48 GB line and the early guard REFUSES (SystemExit
    naming --allow-high-risk-universe) BEFORE any build happens."""
    df = pd.DataFrame({f"f{i}": [0.0, 0.0] for i in range(10200)})
    df["date"] = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df["stock_code"] = ["000001", "000001"]
    df.to_parquet(str(tmp_path / "000001.parquet"))
    stock_list = [f"{i:06d}" for i in range(5530)]
    args = _panel_args(
        "allow-revised", universe="all", prebuilt=str(tmp_path), seq_len=60,
        start="2024-01-01", end="2024-12-31",
    )
    with pytest.raises(SystemExit) as ei:
        tp._early_panel_memory_guard(
            args, stock_list, str(tmp_path), store_load=False)
    msg = str(ei.value)
    assert "universe=all" in msg
    assert "--allow-high-risk-universe" in msg


def test_early_enforce_universe_memory_override_allows_oversized(tp, tmp_path, caplog):
    """--allow-high-risk-universe downgrades the early refusal to a warning; the
    UN-overridden verdict is still 'refuse'."""
    import logging
    df = pd.DataFrame({f"f{i}": [0.0, 0.0] for i in range(10200)})
    df["date"] = pd.to_datetime(["2024-01-02", "2024-01-03"])
    df["stock_code"] = ["000001", "000001"]
    df.to_parquet(str(tmp_path / "000001.parquet"))
    stock_list = [f"{i:06d}" for i in range(5530)]
    args = _panel_args(
        "allow-revised", universe="all", allow_high_risk_universe=True,
        prebuilt=str(tmp_path), seq_len=60,
        start="2024-01-01", end="2024-12-31",
    )
    with caplog.at_level(logging.WARNING, logger="train_panel_mod"):
        est, action = tp._early_panel_memory_guard(
            args, stock_list, str(tmp_path), store_load=False)
    assert action == "refuse"
    assert est > 48.0
    assert any("--allow-high-risk-universe" in m for m in caplog.messages)


def test_early_guard_ok_small_panel(tp, tmp_path):
    """A small universe + small prebuilt schema → the early guard does NOT raise
    and reports the UN-overridden verdict 'ok' (a legitimate run is not blocked)."""
    df = _prebuilt_schema_df({"topic_entropy": [0.1, 0.2], "roe": [0.1, 0.1]})
    df.to_parquet(str(tmp_path / "000001.parquet"))
    stock_list = ["000001", "000002", "000003"]
    args = _panel_args(
        "safe-only", universe="random", prebuilt=str(tmp_path), seq_len=60,
        start="2024-01-01", end="2024-12-31",
    )
    est, action = tp._early_panel_memory_guard(
        args, stock_list, str(tmp_path), store_load=False)
    assert action == "ok"
    assert est > 0.0


def test_early_guard_skipped_on_store_load(tp, tmp_path):
    """A store-load re-run (no build, lazy mmap) must NEVER be refused by the
    pre-build estimate — the store's surviving subset may be far smaller than
    the requested universe; the post-build actual-dims check covers it."""
    args = _panel_args("allow-revised", universe="all", prebuilt=None)
    assert tp._early_panel_memory_guard(
        args, ["000001"], str(tmp_path), store_load=True) is None


# ── §T2: vintage-policy-driven feature switches ────────────────────────

def _panel_args(vintage_policy, **overrides):
    """A minimal train_panel args namespace for the switch/fingerprint tests."""
    base = {
        "vintage_policy": vintage_policy,
        "minute": False,
        "horizon": 1,
        "start": "2020-01-01",
        "end": "2024-12-31",
        "universe": "random",
        # §七-P0 pre-build memory guard fields (T5).
        "prebuilt": None,
        "seq_len": None,
        "allow_high_risk_universe": False,
        "allow_fundamental_ablation": False,
        # §十四 (T7) feature-profile gate fields.
        "no_formal": False,
        "no_require_quality_gate": False,
        "feature_profile": None,
        "require_aux_channels": "",
    }
    base.update(overrides)
    return types.SimpleNamespace(**base)


def test_allow_revised_reproduces_todays_switch_set(tp):
    """allow-revised must reproduce the pre-T2 effective switch set EXACTLY:
    every base-True dim on (incl. the revised-aligned ones), board/sector/
    concept off, seq_len + minute_mode passthrough."""
    kw = tp._panel_pipeline_kwargs(_panel_args("allow-revised"), seq_len=60)
    assert kw["use_sentiment"] is True
    assert kw["use_announcements"] is True
    assert kw["use_fundamental"] is True
    assert kw["use_earnings"] is True
    assert kw["use_macro"] is True
    assert kw["use_market_env_refine"] is True
    assert kw["use_valuation"] is True
    assert kw["use_board"] is False
    assert kw["use_sector"] is False
    assert kw["use_concept"] is False
    assert kw["seq_len"] == 60
    assert kw["minute_mode"] is False


def test_safe_only_denies_revised_aligned_dims(tp):
    """safe-only additionally turns OFF the base-True, latest_revised_aligned
    dims (fundamental/macro/earnings/valuation/index_membership/
    market_env_refine/pledge/shareholder) while keeping derived_versioned
    (market_env/industry) and raw_vintage_safe (sentiment) ON."""
    kw = tp._panel_pipeline_kwargs(_panel_args("safe-only"), seq_len=60)
    for dim in ("fundamental", "macro", "earnings", "valuation",
                "index_membership", "market_env_refine", "pledge", "shareholder"):
        assert kw[f"use_{dim}"] is False, dim
    assert kw["use_sentiment"] is True
    assert kw["use_market_env"] is True
    assert kw["use_industry"] is True
    for dim in ("board", "sector", "concept", "limit_up", "topic"):
        assert kw[f"use_{dim}"] is False, dim


def test_base_dim_preference_matches_documented_dims(tp):
    """Drift canary: _BASE_DIM_PREFERENCE is exactly the documented 27 use_*
    dimension list — a new documented dimension must be added to both."""
    assert set(tp._BASE_DIM_PREFERENCE) == set(cv.DOCUMENTED_USE_DIMS)


def test_announcement_switch_key_special_case(tp):
    """The announcement channel maps to the constructor kwarg use_announcements
    (with an "s"); the bare use_announcement key must never be emitted (it is
    not a FeaturePipeline constructor parameter)."""
    kw = tp._panel_pipeline_kwargs(_panel_args("allow-revised"), seq_len=60)
    assert "use_announcements" in kw
    assert "use_announcement" not in kw


def test_panel_store_meta_fingerprints_vintage_policy(tp):
    """§T2: the vintage policy enters the panel-store meta fingerprint via
    feature_switches, so a policy change auto-invalidates a stale store."""
    allow = tp._panel_store_meta(_panel_args("allow-revised"), seq_len=60, stock_list=[f"{i:06d}" for i in range(100)])
    safe = tp._panel_store_meta(_panel_args("safe-only"), seq_len=60, stock_list=[f"{i:06d}" for i in range(100)])
    assert allow["feature_switches"] != safe["feature_switches"]
    assert allow["feature_switches"]["use_fundamental"] is True
    assert safe["feature_switches"]["use_fundamental"] is False
    assert safe["feature_switches"]["use_sentiment"] is True


def test_allow_fundamental_ablation_reincludes_fundamental_under_safe_only(tp):
    """T3 decision #1: --allow-fundamental-ablation is the ONLY way a
    fundamental channel enters a safe-only run.  It forces use_fundamental=True
    while the OTHER 7 policy-denied channels stay False — the flag is
    fundamental-only, never a blanket allow-revised."""
    kw = tp._panel_pipeline_kwargs(
        _panel_args("safe-only", allow_fundamental_ablation=True), seq_len=60)
    assert kw["use_fundamental"] is True
    for dim in ("macro", "earnings", "valuation",
                "index_membership", "market_env_refine", "pledge", "shareholder"):
        assert kw[f"use_{dim}"] is False, dim


def test_allow_fundamental_ablation_changes_store_fingerprint(tp):
    """T3: the ablation flag must change the panel-store meta fingerprint — an
    ablation store must never be reused by a non-ablation run (nor vice-versa),
    exactly as a policy change does."""
    base = tp._panel_store_meta(_panel_args("safe-only"), seq_len=60, stock_list=[f"{i:06d}" for i in range(100)])
    ablated = tp._panel_store_meta(
        _panel_args("safe-only", allow_fundamental_ablation=True),
        seq_len=60, stock_list=[f"{i:06d}" for i in range(100)])
    assert base["feature_switches"] != ablated["feature_switches"]
    assert base["feature_switches"]["use_fundamental"] is False
    assert ablated["feature_switches"]["use_fundamental"] is True


def test_allow_fundamental_ablation_missing_flag_defaults_off(tp):
    """T3: an args stub WITHOUT the new attr (a caller that predates the flag)
    must not crash — the defensive read defaults the flag to off."""
    kw = tp._panel_pipeline_kwargs(_panel_args("safe-only"), seq_len=60)
    assert kw["use_fundamental"] is False


# ── §T6 decision 2: strict-CSI daily-member normalization + strict training ──

def test_is_csi_universe(tp):
    """csi300/csi500/csi800 are the strict-CSI universes; everything else
    (including the historical-member-universe modes) is not."""
    for u in ("csi300", "csi500", "csi800"):
        assert tp._is_csi_universe(u) is True, u
    for u in ("random", "all", "first", "stratified"):
        assert tp._is_csi_universe(u) is False, u


def test_strict_index_training_effective(tp):
    """§T6 decision 2: the default (None) decides from the universe — ON for
    the strict-CSI universes, OFF otherwise; an explicit flag always wins."""
    assert tp._strict_index_training_effective(None, "csi300") is True
    assert tp._strict_index_training_effective(None, "csi800") is True
    assert tp._strict_index_training_effective(None, "random") is False
    assert tp._strict_index_training_effective(None, "all") is False
    assert tp._strict_index_training_effective(True, "random") is True
    assert tp._strict_index_training_effective(False, "csi300") is False


def test_panel_store_meta_csi_marks_daily_membership_norm(tp):
    """§T6: a CSI universe bakes daily-member cross-section normalization into
    the panel arrays, so the store fingerprint records the pseudo-switch — a
    stale store built for the all-stock z-norm must refuse to mix.  Non-CSI
    universes never carry the key."""
    csi = tp._panel_store_meta(
        _panel_args("safe-only", universe="csi300"),
        seq_len=60, stock_list=[f"{i:06d}" for i in range(100)])
    assert csi["feature_switches"].get("daily_membership_norm") is True
    # The pseudo-switch is ADDED to the real switch set, never replacing it.
    assert csi["feature_switches"]["seq_len"] == 60
    assert csi["feature_switches"]["use_sentiment"] is True
    non_csi = tp._panel_store_meta(
        _panel_args("safe-only", universe="random"),
        seq_len=60, stock_list=[f"{i:06d}" for i in range(100)])
    assert "daily_membership_norm" not in non_csi["feature_switches"]


def _fake_storage():
    """DataStorage replacement whose load_daily yields one tiny row per stock —
    enough for _resolve_panel's K-line concat without reading real disk data."""
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
    """FeaturePipeline replacement that records build_panel_features kwargs."""
    calls = []
    class _FakePipeline:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def build_panel_features(self, panel, **kw):
            calls.append({"panel": panel, **kw})
            return {"__dummy": True}
    return _FakePipeline, calls


def test_resolve_panel_passes_membership_for_csi(tp, monkeypatch, caplog):
    """§T6 decision 2 wiring: for a CSI universe, _resolve_panel loads the
    index-membership intervals and hands them to build_panel_features as
    daily_membership (so the cross-section z-norm is member-limited)."""
    import stoke_ml.data.storage as storage_mod
    fake_pipe, calls = _capture_pipeline()
    monkeypatch.setattr("scripts.production.train_panel_panel.FeaturePipeline",
                        fake_pipe)
    monkeypatch.setattr(storage_mod, "DataStorage", _fake_storage())
    mem = pd.DataFrame({
        "stock_code": ["600000"],
        "index_code": ["000300"],
        "in_date": pd.to_datetime(["2022-01-01"]),
        "out_date": pd.to_datetime([pd.NaT]),
    })
    monkeypatch.setattr(
        "scripts.production.train_panel_panel.load_index_membership",
        lambda data_dir, indices: mem)
    args = _panel_args("safe-only", universe="csi300", no_aux=True)
    args.panel_store = None
    args.require_feature_manifest = False
    panel_data, channel_manifest = tp._resolve_panel(
        args, ["600000"], 60, "data", {"sentiment"}, _store_load=False)
    assert panel_data == {"__dummy": True}
    assert len(calls) == 1
    assert calls[0]["daily_membership"] is mem


def test_resolve_panel_empty_membership_warns_and_degrades(tp, monkeypatch, caplog):
    """§T6: an EMPTY/missing membership for a CSI universe must not crash — it
    warns and degrades to daily_membership=None (the all-stock z-norm)."""
    import stoke_ml.data.storage as storage_mod
    fake_pipe, calls = _capture_pipeline()
    monkeypatch.setattr("scripts.production.train_panel_panel.FeaturePipeline",
                        fake_pipe)
    monkeypatch.setattr(storage_mod, "DataStorage", _fake_storage())
    empty = pd.DataFrame(columns=["stock_code", "index_code", "in_date", "out_date"])
    monkeypatch.setattr(
        "scripts.production.train_panel_panel.load_index_membership",
        lambda data_dir, indices: empty)
    args = _panel_args("safe-only", universe="csi800", no_aux=True)
    args.panel_store = None
    args.require_feature_manifest = False
    with caplog.at_level("WARNING"):
        tp._resolve_panel(
            args, ["600000"], 60, "data", {"sentiment"}, _store_load=False)
    assert "empty/missing" in caplog.text
    assert calls[0]["daily_membership"] is None


# ── §十四 (T7): feature-profile required channels + min-coverage gate ────

def _manifest_entry(loaded, coverage):
    return {"requested": True, "required": True, "loaded_stocks": loaded,
            "coverage": coverage, "errors": 0, "status": "OK"}


def test_resolve_required_set_profile_adds_channels(tp):
    """A formal, gate-enforced run with a named profile unions the profile's
    required_channels into the explicit --require-aux-channels set and carries
    the profile's minimum-coverage thresholds."""
    from stoke_ml.config.feature_profile import FEATURE_PROFILES
    args = _panel_args("safe-only", feature_profile="headline_v1",
                       require_aux_channels="sentiment,extra_ch")
    required_set, min_cov, name = tp._resolve_required_set(args)
    assert name == "headline_v1"
    assert required_set == (
        set(FEATURE_PROFILES["headline_v1"].required_channels) | {"extra_ch"})
    assert min_cov == FEATURE_PROFILES["headline_v1"].minimum_coverage


def test_resolve_required_set_cli_default_activates_headline_v1(tp):
    """The CLI default --feature-profile headline_v1 (a formal, gate-enforced
    run) activates the profile — no explicit flag needed."""
    args = _panel_args("safe-only", feature_profile="headline_v1")
    required_set, min_cov, name = tp._resolve_required_set(args)
    assert name == "headline_v1"
    assert min_cov == {"sentiment": 0.90, "guba": 0.90, "comment": 0.90,
                       "announcement": 0.70, "margin": 0.95, "northbound": 0.90,
                       "capital_flow": 0.90, "etf_flow": 0.80, "block_trade": 0.30,
                       "lockup": 0.30, "dividend": 0.30, "industry": 0.95,
                       "market_env": 0.95}
    assert "margin" in required_set


def test_resolve_required_set_none_disables_profile(tp):
    args = _panel_args("safe-only", feature_profile="none",
                       require_aux_channels="guba")
    required_set, min_cov, name = tp._resolve_required_set(args)
    assert name == "none"
    assert required_set == {"guba"}
    assert min_cov == {}


def test_resolve_required_set_no_formal_skips_profile(tp):
    """--no-formal (exploratory) never activates the profile — the set is just
    the explicit channels, even when a profile is named."""
    args = _panel_args("safe-only", feature_profile="headline_v1",
                       no_formal=True, require_aux_channels="guba")
    required_set, min_cov, name = tp._resolve_required_set(args)
    assert name == "none"
    assert required_set == {"guba"}
    assert min_cov == {}


def test_resolve_required_set_no_gate_skips_profile(tp):
    """--no-require-quality-gate (dev smoke) also never activates the profile."""
    args = _panel_args("safe-only", feature_profile="headline_v1",
                       no_require_quality_gate=True, require_aux_channels="guba")
    required_set, min_cov, name = tp._resolve_required_set(args)
    assert name == "none"
    assert required_set == {"guba"}
    assert min_cov == {}


def test_resolve_required_set_unknown_profile_aborts(tp):
    """A TYPO'd profile name on an active gate must abort loudly, not silently
    skip the coverage gate."""
    args = _panel_args("safe-only", feature_profile="bogus")
    with pytest.raises(SystemExit) as ei:
        tp._resolve_required_set(args)
    assert "unknown feature profile" in str(ei.value)


def test_enforce_channel_coverage_aborts_below_minimum(tp, caplog):
    """A probeable required channel below its profile minimum aborts the run."""
    import logging
    manifest = {"sentiment": _manifest_entry(100, 0.50)}
    with caplog.at_level(logging.ERROR, logger="train_panel_mod"):
        with pytest.raises(SystemExit) as ei:
            tp._enforce_channel_coverage({"sentiment"}, manifest,
                                         {"sentiment": 0.90})
    assert ei.value.code == 1
    assert any("coverage 0.5000 < minimum 0.9000" in m for m in caplog.messages)


def test_enforce_channel_coverage_meets_minimum_passes(tp):
    manifest = {"sentiment": _manifest_entry(100, 0.95)}
    # Must NOT raise.
    tp._enforce_channel_coverage({"sentiment"}, manifest, {"sentiment": 0.90})


def test_enforce_channel_coverage_absent_channel_warns_not_abort(tp, caplog):
    """A required/min-cov channel with NO manifest entry (prebuilt panel without
    a has_* flag, e.g. margin/northbound/capital_flow) warns — coverage cannot
    be verified — but NEVER aborts.  Without this scope the default
    headline_v1 profile would refuse every prebuilt run."""
    import logging
    with caplog.at_level(logging.WARNING, logger="train_panel_mod"):
        tp._enforce_channel_coverage({"margin"}, {}, {"margin": 0.95})
    assert any("margin" in m for m in caplog.messages)
    assert any("no coverage probe" in m for m in caplog.messages)


def test_enforce_channel_coverage_zero_coverage_aborts(tp, caplog):
    """A required channel that IS probed but has ZERO coverage aborts instead of
    silently training on air."""
    import logging
    manifest = {"sentiment": _manifest_entry(0, 0.0)}
    with caplog.at_level(logging.ERROR, logger="train_panel_mod"):
        with pytest.raises(SystemExit) as ei:
            tp._enforce_channel_coverage({"sentiment"}, manifest)
    assert ei.value.code == 1
    assert any("ZERO coverage" in m for m in caplog.messages)


def test_enforce_channel_coverage_non_numeric_coverage_skipped(tp):
    """coverage=None (an entry that is not coverage-probeable) must NOT be
    compared as a numeric 0 — it is skipped, never a min-coverage abort."""
    manifest = {"sentiment": {"loaded_stocks": 100, "coverage": None}}
    tp._enforce_channel_coverage({"sentiment"}, manifest, {"sentiment": 0.90})


def test_enforce_channel_coverage_unrequested_channel_ignored(tp):
    """A channel in the manifest but NOT in required_set is not gated at all."""
    manifest = {"sentiment": _manifest_entry(0, 0.0)}
    tp._enforce_channel_coverage({"guba"}, manifest, {})  # must not abort
