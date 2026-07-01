"""Integration tests for PreprocessingPipeline — end-to-end text+numeric chains."""
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import pytest
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.preprocessing.base import PreprocessingChain
from stoke_ml.preprocessing.text.bipolar import BipolarClassifier
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter
from stoke_ml.preprocessing.text.aggregation import DailyAggregator
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector
from stoke_ml.preprocessing.numeric.missing import MissingImputer


class TestPreprocessingPipeline:
    def test_registers_chains(self):
        pp = PreprocessingPipeline()
        pp.register_chain("text", PreprocessingChain(
            [BipolarClassifier(), TimeDecayWeighter()], name="text"
        ))
        assert "text" in pp.list_chains()

    def test_runs_text_chain(self):
        pp = PreprocessingPipeline()
        pp.register_chain("text", PreprocessingChain(
            [BipolarClassifier(), DailyAggregator()], name="text"
        ))
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "sentiment_title": [0.5, -0.3, 0.8],
        })
        result = pp.run("text", df.copy())
        assert "bipolar_sent" in result.columns
        assert "agreement" in result.columns

    def test_runs_numeric_chain(self):
        pp = PreprocessingPipeline()
        pp.register_chain("numeric", PreprocessingChain(
            [OutlierDetector(threshold=5.0), MissingImputer()], name="numeric"
        ))
        df = pd.DataFrame({
            "close": [10.0, 10.5, np.nan, 10.2, 10.8],
            "volume": [1e6, 1.1e6, 1e6, 1e6, 1e6],
        })
        result = pp.run("numeric", df.copy())
        # Missing value in close should be filled
        assert not result["close"].isna().all()

    def test_run_nonexistent_chain_raises(self):
        pp = PreprocessingPipeline()
        with pytest.raises(KeyError):
            pp.run("nonexistent", pd.DataFrame())

    def test_full_text_pipeline(self):
        pp = PreprocessingPipeline()
        chain = PreprocessingChain([
            BipolarClassifier(),
            TimeDecayWeighter(halflife_days=7),
            DailyAggregator(windows=(3, 5)),
        ], name="text_full")
        pp.register_chain("text", chain)

        n_posts = 100
        rng = np.random.RandomState(42)
        dates = pd.date_range("2024-01-01", "2024-06-30", freq="B")
        df = pd.DataFrame({
            "aligned_date": rng.choice(dates, n_posts),
            "sentiment_title": rng.uniform(-1, 1, n_posts).astype(np.float32),
        })

        result = pp.run("text", df)
        # Should have daily aggregation with rolling windows
        assert "bipolar_sent" in result.columns
        assert "bipolar_sent_3d_mean" in result.columns
        assert "attention" in result.columns

    def test_run_with_stock_code_context(self):
        pp = PreprocessingPipeline()
        pp.register_chain("text", PreprocessingChain(
            [BipolarClassifier(), DailyAggregator()], name="text"
        ))
        df = pd.DataFrame({
            "aligned_date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "sentiment_title": [0.5, 0.8],
        })
        result = pp.run("text", df.copy(), stock_code="000001")
        assert "bipolar_sent" in result.columns

    def test_from_config_builds_pipeline(self):
        """Verify from_config builds pipeline with text+numeric chains."""
        config = {
            "text": {
                "bipolar": {"threshold_positive": 0.25, "threshold_negative": -0.25},
                "time_decay": {"halflife_days": 14},
                "aggregation": {"windows": [5, 10]},
            },
            "numeric": {
                "outlier": {"threshold": 4.0},
                "missing": {"short_gap_max": 3},
                "cross_section": {"enabled": True},
                "scaling": {"window_days": 120},
                "higher_order": {"enabled": True},
            },
        }
        pp = PreprocessingPipeline.from_config(config)
        assert "text" in pp.list_chains()
        assert "numeric" in pp.list_chains()

    def test_list_chains_empty(self):
        pp = PreprocessingPipeline()
        assert pp.list_chains() == []

    def test_multiple_chains(self):
        pp = PreprocessingPipeline()
        pp.register_chain("xueqiu", PreprocessingChain(
            [BipolarClassifier()], name="xueqiu"
        ))
        pp.register_chain("guba", PreprocessingChain(
            [BipolarClassifier()], name="guba"
        ))
        assert set(pp.list_chains()) == {"guba", "xueqiu"}
