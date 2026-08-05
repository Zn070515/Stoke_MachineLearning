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
                        lambda strict=True: _FakeCalendar())
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
