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
        tp, "_load_index_universe",
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
        tp, "_load_index_universe",
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

def test_verified_until_refuses_panel_past_2026(tp):
    """A formal run whose panel axis reaches 2027 (forward-estimate closures)
    must be refused — the strict calendar raises past verified_until."""
    gd = np.array(["2024-01-02", "2027-01-04"], dtype="datetime64[D]")
    msg = tp._check_verified_until_scope(gd, enforce=True)
    assert msg is not None
    assert "formal run refused" in msg


def test_verified_until_accepts_within_verified_window(tp):
    gd = np.array(["2024-01-02", "2024-12-31"], dtype="datetime64[D]")
    assert tp._check_verified_until_scope(gd, enforce=True) is None


def test_verified_until_opt_out_not_enforced(tp):
    """--no-require-quality-gate (enforce=False) lets an exploratory run use a
    panel that reaches past verified_until."""
    gd = np.array(["2024-01-02", "2027-01-04"], dtype="datetime64[D]")
    assert tp._check_verified_until_scope(gd, enforce=False) is None


def test_verified_until_empty_panel_is_noop(tp):
    assert tp._check_verified_until_scope(None, enforce=True) is None
    empty = np.array([], dtype="datetime64[D]")
    assert tp._check_verified_until_scope(empty, enforce=True) is None


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
    allow = tp._panel_store_meta(_panel_args("allow-revised"), seq_len=60, n_stocks=100)
    safe = tp._panel_store_meta(_panel_args("safe-only"), seq_len=60, n_stocks=100)
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
    base = tp._panel_store_meta(_panel_args("safe-only"), seq_len=60, n_stocks=100)
    ablated = tp._panel_store_meta(
        _panel_args("safe-only", allow_fundamental_ablation=True),
        seq_len=60, n_stocks=100)
    assert base["feature_switches"] != ablated["feature_switches"]
    assert base["feature_switches"]["use_fundamental"] is False
    assert ablated["feature_switches"]["use_fundamental"] is True


def test_allow_fundamental_ablation_missing_flag_defaults_off(tp):
    """T3: an args stub WITHOUT the new attr (a caller that predates the flag)
    must not crash — the defensive read defaults the flag to off."""
    kw = tp._panel_pipeline_kwargs(_panel_args("safe-only"), seq_len=60)
    assert kw["use_fundamental"] is False
