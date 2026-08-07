"""§十二 wiring of scripts/production/preprocess_new_data.py.

Formal mode defaults to STRICT — error-level quality problems block and nothing
is persisted.  `--allow-degraded` opts out explicitly (degraded output is
written), `--strict` still wins when both are given, and the `--no-formal` dev
path keeps its legacy degrade-silently semantics.
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stoke_ml.preprocessing.base import PreprocessingChain
from stoke_ml.preprocessing.monitor.quality import QualityMonitor
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "production" / "preprocess_new_data.py"
_spec = importlib.util.spec_from_file_location("preprocess_new_data_mod", _SCRIPT)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)

_RAW = pd.DataFrame({"x": [1.0] + [np.inf] * 9})  # trips the quality monitor


class _FakeStorageFactory:
    """Replaces MarketWideStorage; records every write to the last (dest)."""

    instances = []

    def __init__(self, data_dir, data_type):
        self.data_dir = data_dir
        self.data_type = data_type
        self.saved = []
        _FakeStorageFactory.instances.append(self)

    def load(self, code, start, end):
        return _RAW.copy()

    def save(self, df, replace_range=False, *, degrade_threshold=0.2,
             provenance=None, replace_window=None):
        self.saved.append(df.copy())
        return 0


class _FakeDS:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_daily(self, code, start, end, require_valid_manifest=True):
        return pd.DataFrame({"close": [10.0, 11.0, 12.0]})


class _FakeCalendar:
    def get_trading_days(self, start, end):
        return [pd.Timestamp("2024-01-02").date()]


def _quality_pipeline():
    pp = PreprocessingPipeline()
    pp.register_chain("event_block_trade", PreprocessingChain([], name="event_block_trade"))
    pp._quality_monitor = QualityMonitor(missing_error_threshold=0.5)
    return pp


def _args(**overrides):
    base = dict(
        no_formal=False, allow_degraded=False, strict=False,
        save_to=None, degrade_threshold=0.2,
        start="2024-01-01", end="2024-12-31",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(autouse=True)
def _patch_deps(monkeypatch):
    monkeypatch.setattr(mod, "MarketWideStorage", _FakeStorageFactory)
    monkeypatch.setattr("stoke_ml.data.storage.DataStorage", _FakeDS)
    monkeypatch.setattr("stoke_ml.data.calendar.get_research_calendar",
                        lambda strict=True, data_dir=None: _FakeCalendar())
    _FakeStorageFactory.instances = []
    yield


def _effective_strict(args):
    return args.strict or (not args.no_formal and not args.allow_degraded)


def _run_process(args):
    mod._process_standard(
        "block_trade", "block_trade", _quality_pipeline(), "event_block_trade",
        ["000001"], "/tmp/data", args, {"run_id": "r1"},
        strict=_effective_strict(args),
    )
    return _FakeStorageFactory.instances[-1].saved


class TestCLIParser:
    def test_allow_degraded_flag_registered(self):
        parser = mod.build_parser()
        assert parser.parse_args(["--type", "flow"]).allow_degraded is False
        assert parser.parse_args(["--type", "flow", "--allow-degraded"]).allow_degraded is True

    def test_strict_and_no_formal_flags_preserved(self):
        parser = mod.build_parser()
        assert parser.parse_args(["--type", "flow"]).strict is False
        assert parser.parse_args(["--type", "flow", "--strict"]).strict is True
        assert parser.parse_args(["--type", "flow", "--no-formal"]).no_formal is True


class TestFormalStrictWiring:
    def test_formal_default_blocks_quality_failure(self):
        """formal (no --allow-degraded): quality failure → nothing persisted."""
        assert _run_process(_args()) == []

    def test_formal_allow_degraded_writes_output(self):
        """formal + --allow-degraded: degraded output IS persisted."""
        saved = _run_process(_args(allow_degraded=True))
        assert len(saved) == 1
        assert len(saved[0]) == len(_RAW)

    def test_explicit_strict_wins_over_allow_degraded(self):
        """--strict + --allow-degraded: strict still blocks."""
        assert _run_process(_args(strict=True, allow_degraded=True)) == []

    def test_no_formal_preserves_legacy_degrade(self):
        """--no-formal (dev path): quality failures degrade silently."""
        saved = _run_process(_args(no_formal=True))
        assert len(saved) == 1


class TestBuildProvenanceNarrow:
    """§T18: the git rev-parse provenance catch narrows to
    (OSError, subprocess.CalledProcessError)."""

    def test_git_failure_falls_back_to_unknown(self, monkeypatch):
        def _fail(*a, **k):
            raise mod.subprocess.CalledProcessError(
                128, ["git", "rev-parse", "HEAD"]
            )

        monkeypatch.setattr(mod.subprocess, "run", _fail)
        prov = mod._build_provenance({"a": 1})
        assert prov["git_commit"] == "unknown"
        assert prov["config_hash"] != "unknown"

    def test_git_non_oserror_calledprocesserror_propagates(self, monkeypatch):
        def _raise(*a, **k):
            raise RuntimeError("unexpected")

        monkeypatch.setattr(mod.subprocess, "run", _raise)
        with pytest.raises(RuntimeError, match="unexpected"):
            mod._build_provenance({"a": 1})


class TestErrorSummaryAggregation:
    """§T18: the per-stock preprocess loops keep their broad catch but count
    every failure through an ErrorSummary (reported via log_summary)."""

    def test_process_standard_records_failure(self, monkeypatch, caplog):
        def _raise(*a, **k):
            raise RuntimeError("load boom")

        monkeypatch.setattr(_FakeStorageFactory, "load", _raise)
        with caplog.at_level("WARNING", logger="preprocess_new_data_mod"):
            _run_process(_args(allow_degraded=True))
        assert any("Error summary" in r.getMessage() for r in caplog.records)
        assert any("preprocess:block_trade" in r.getMessage() for r in caplog.records)

    def test_process_board_records_failure(self, tmp_path, monkeypatch, caplog):
        def _raise(*a, **k):
            raise RuntimeError("board boom")

        monkeypatch.setattr(_FakeDS, "load_daily", _raise)
        args = _args()
        with caplog.at_level("WARNING", logger="preprocess_new_data_mod"):
            mod._process_board(
                _quality_pipeline(), "board", ["000001"], str(tmp_path), args,
                {"run_id": "r1"}, strict=_effective_strict(args),
            )
        assert any("Error summary" in r.getMessage() for r in caplog.records)
        assert any("preprocess:board" in r.getMessage() for r in caplog.records)

    def test_process_sector_records_failure(self, tmp_path, monkeypatch, caplog):
        a_shares = tmp_path / "a_shares"
        a_shares.mkdir()
        # A minimal industry_ranking: sector_code present so the broadcaster's
        # drop_duplicates subset is satisfiable, no change_pct so it early-returns
        # without heavy cross-sectional math + a snapshot cache.
        pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "sector_code": ["01"],
        }).to_parquet(a_shares / "industry_ranking.parquet", index=False)
        pd.DataFrame({"stock_code": ["000001"], "sector": ["银行"]}).to_csv(
            a_shares / "stock_sector_cache.csv", index=False)

        def _raise(*a, **k):
            raise RuntimeError("sector boom")

        monkeypatch.setattr(_FakeDS, "load_daily", _raise)
        pp = PreprocessingPipeline()
        pp.register_chain("sector", PreprocessingChain([], name="sector"))
        pp._quality_monitor = QualityMonitor(missing_error_threshold=0.5)
        args = _args(sector_snapshot_asof=None)
        with caplog.at_level("WARNING", logger="preprocess_new_data_mod"):
            mod._process_sector(
                pp, "sector", ["000001"], str(tmp_path), args,
                {"run_id": "r1"}, strict=_effective_strict(args),
            )
        assert any("Error summary" in r.getMessage() for r in caplog.records)
        assert any("preprocess:sector" in r.getMessage() for r in caplog.records)
