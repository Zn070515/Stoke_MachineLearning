"""End-to-end smoke test: run preprocessing on synthetic realistic data."""
import pytest
import pandas as pd
import numpy as np
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.preprocessing.base import PreprocessingChain
from stoke_ml.preprocessing.text.bipolar import BipolarClassifier
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter
from stoke_ml.preprocessing.text.aggregation import DailyAggregator
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector
from stoke_ml.preprocessing.numeric.missing import MissingImputer
from stoke_ml.preprocessing.numeric.scaling import RobustScaler
from stoke_ml.preprocessing.monitor.quality import QualityMonitor


class TestFullPipelineSmoke:
    """Run the full text + numeric chains on synthetic realistic data."""

    def test_text_chain_realistic(self):
        """Simulate Guba silver data: 500 posts over 2 years."""
        rng = np.random.RandomState(42)
        n = 500
        dates = pd.date_range("2024-01-01", "2026-06-30", freq="D")
        df = pd.DataFrame({
            "aligned_date": rng.choice(dates, n),
            "sentiment_title": rng.uniform(-1, 1, n).astype(np.float32),
            "sentiment_body": rng.uniform(-1, 1, n).astype(np.float32),
        })
        df = df.sort_values("aligned_date").reset_index(drop=True)

        pp = PreprocessingPipeline()
        chain = PreprocessingChain([
            BipolarClassifier(sentiment_cols=["sentiment_title", "sentiment_body"]),
            TimeDecayWeighter(halflife_days=7),
            DailyAggregator(windows=(3, 5, 10)),
        ], name="text_full")
        pp.register_chain("guba", chain)

        result = pp.run("guba", df)

        # Assert daily output
        assert "bipolar_sent" in result.columns
        assert "agreement" in result.columns
        assert "attention" in result.columns
        assert "body_sent_mean" in result.columns

        # Bipolar is in [-1, 1]
        assert result["bipolar_sent"].between(-1, 1).all()
        # Agreement is in [0, 1]
        assert result["agreement"].between(0, 1).all()
        # Attention is non-negative
        assert (result["attention"] >= 0).all()

        # Rolling window derivatives exist
        assert "bipolar_sent_5d_mean" in result.columns

        # At least some days have data
        assert len(result) > 0

    def test_numeric_chain_realistic(self):
        """Simulate daily K-line data: 500 trading days."""
        rng = np.random.RandomState(42)
        n = 500
        close = 100 + rng.randn(n).cumsum() * 0.5
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": close * (1 + rng.uniform(-0.01, 0.01, n)),
            "high": close * (1 + rng.uniform(0.0, 0.02, n)),
            "low": close * (1 - rng.uniform(0.0, 0.02, n)),
            "close": close,
            "volume": np.abs(rng.randn(n) * 1e6) + 1e5,
        })
        # Insert some gaps
        df.loc[50:52, "close"] = np.nan
        df.loc[200:210, "close"] = np.nan
        df.loc[300, "volume"] = 1e10  # outlier

        pp = PreprocessingPipeline()
        chain = PreprocessingChain([
            OutlierDetector(threshold=5.0),
            MissingImputer(short_gap_max=2, medium_gap_max=10),
            RobustScaler(window_days=60, min_periods=20),
        ], name="numeric_mini")
        pp.register_chain("daily", chain)

        result = pp.run("daily", df)

        # Short gaps should be filled
        assert not np.isnan(result["close"].iloc[51])

        # Outlier should be clipped
        assert result["volume"].max() < 1e10

        # Long gaps (>10 days) may still have NaN or be filled
        assert "close" in result.columns
        assert len(result) == n

    def test_quality_monitor_integration(self):
        """QualityMonitor reports on output data."""
        rng = np.random.RandomState(42)
        df_good = pd.DataFrame({
            "bipolar_sent": rng.uniform(-1, 1, 200),
            "agreement": rng.uniform(0, 1, 200),
            "attention": np.log(1 + rng.poisson(5, 200)),
        })

        qm = QualityMonitor(
            missing_warn_threshold=0.2,
            missing_error_threshold=0.5,
        )
        qm.fit_transform(df_good)
        assert not qm.has_errors

        # Inject bad data
        df_bad = pd.DataFrame({
            "bipolar_sent": [np.inf, -np.inf, 0.5],
            "agreement": [0.5, 0.5, 0.5],
        })
        qm2 = QualityMonitor()
        qm2.fit_transform(df_bad)
        assert qm2.has_errors

    def test_from_config_smoke(self):
        """Verify PreprocessingPipeline.from_config builds runnable chains."""
        config = {
            "text": {
                "bipolar": {"threshold_positive": 0.2, "threshold_negative": -0.2},
                "time_decay": {"halflife_days": 7},
                "aggregation": {"windows": [3, 5]},
            },
            "numeric": {
                "outlier": {"threshold": 5.0},
                "missing": {"short_gap_max": 2, "medium_gap_max": 10},
                "cross_section": {"enabled": False},
                "scaling": {"window_days": 120, "winsorize_sigma": 3.0},
                "higher_order": {"enabled": False},
            },
        }
        pp = PreprocessingPipeline.from_config(config)

        # Text chain smoke
        rng = np.random.RandomState(42)
        df_text = pd.DataFrame({
            "aligned_date": pd.to_datetime(rng.choice(
                pd.date_range("2024-01-01", "2024-06-30", freq="D"), 200
            )),
            "sentiment_title": rng.uniform(-1, 1, 200).astype(np.float32),
        })
        result_text = pp.run("text", df_text)
        assert "bipolar_sent" in result_text.columns
        assert len(result_text) > 0

        # Numeric chain smoke
        n = 300
        close = 100 + rng.randn(n).cumsum() * 0.5
        df_num = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "open": close * (1 + rng.uniform(-0.01, 0.01, n)),
            "high": close * (1 + rng.uniform(0.0, 0.02, n)),
            "low": close * (1 - rng.uniform(0.0, 0.02, n)),
            "close": close,
            "volume": np.abs(rng.randn(n) * 1e6) + 1e5,
        })
        result_num = pp.run("numeric", df_num)
        assert "close" in result_num.columns
        assert len(result_num) == n
