"""Unit tests for train_panel.py universe selection and honest eval reporting.

These cover the Track B credibility fixes:
- `_best_eval_metrics` reports the eval nearest the deployed best-val-RankIC
  checkpoint instead of the post-hoc max.
- `_resolve_universe` implements seeded/stratified/index universes that remove
  the alphabetical `sorted(...)[:N]` sampling bias.
- `_save_artifacts` persists the resolved/used stock lists + summary.

None of these read the 109GB feature store.
"""

import importlib.util
import json
import os

import numpy as np
import pytest

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

    monkeypatch.setattr(tp, "build_calendar_frame", fake_build)
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
