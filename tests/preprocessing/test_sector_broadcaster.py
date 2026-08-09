"""§十-4: SectorBroadcaster must not present-backfill the current sector snapshot
onto history.

A static ``sector_map`` (built from the current-snapshot
``stock_sector_cache.csv``) is valid only from ``sector_map_valid_from`` on;
older rows get NaN sector_code so their broadcast sector features read as
unknown (zeroed).  A caller that already resolved a per-row ``sector_code``
(genuine PIT membership) is respected, not overwritten by the static map.
"""
import numpy as np
import pandas as pd

from stoke_ml.preprocessing.cross_sectional.sector import SectorBroadcaster


def _sector_features(n=6):
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "sector_code": ["SEC0000"] * n,
        "change_pct": [0.01] * n,
        "rank": [1] * n,
        "leader": ["000001"] * n,
        "momentum_5d": [0.05] * n,
        "sector_rrg_y": [0.5] * n,
        "sector_rrg_x": [0.2] * n,
        "sector_relative_strength": [0.1] * n,
        "sector_breadth_z": [1.0] * n,
        "sector_alpha": [0.3] * n,
    })


def _stock_df(code="000001", n=6):
    # Starts on the same day as _sector_features so every row has coverage.
    dates = pd.date_range("2024-01-02", periods=n, freq="B")
    return pd.DataFrame({
        "date": dates,
        "stock_code": code,
        "volume": [1e6] * n,
    })


class TestSectorBroadcasterPIT:
    def test_valid_from_bounds_static_map(self):
        df = _stock_df()
        feat = _sector_features()
        out = SectorBroadcaster().transform(
            df.copy(),
            sector_map={"000001": "SEC0000"},
            sector_features=feat,
            sector_map_valid_from="2024-01-03",
        )
        pre = out["date"] < pd.Timestamp("2024-01-03")
        post = ~pre
        # Pre-asof rows: unknown sector → broadcast features zeroed/NaN.
        assert out.loc[pre, "sector_code"].isna().all()
        assert (out.loc[pre, "sector_relative_strength"] == 0).all()
        assert (out.loc[pre, "sector_breadth_z"] == 0).all()
        assert out.loc[pre, "momentum_5d"].isna().all()
        # Post-asof rows: mapped and broadcast normally.
        assert (out.loc[post, "sector_code"] == "SEC0000").all()
        assert (out.loc[post, "sector_relative_strength"] == 0.1).all()
        assert (out.loc[post, "momentum_5d"] == 0.05).all()

    def test_valid_from_beyond_window_zeros_everything(self):
        """A snapshot fetched after the window carries no valid history at all."""
        df = _stock_df()
        feat = _sector_features()
        out = SectorBroadcaster().transform(
            df.copy(),
            sector_map={"000001": "SEC0000"},
            sector_features=feat,
            sector_map_valid_from="2025-01-01",
        )
        assert out["sector_code"].isna().all()
        assert (out["sector_relative_strength"] == 0).all()

    def test_pre_resolved_sector_code_respected(self):
        """Per-row PIT membership wins over the static map (no re-stamping)."""
        df = _stock_df()
        df["sector_code"] = "SEC0001"
        feat = _sector_features()  # only SEC0000 → SEC0001 rows won't match
        out = SectorBroadcaster().transform(
            df.copy(),
            sector_map={"000001": "SEC0000"},
            sector_features=feat,
        )
        assert (out["sector_code"] == "SEC0001").all()
        # Unmatched sector → relative-strength broadcast is zeroed, not backfilled.
        assert (out["sector_relative_strength"] == 0).all()

    def test_no_valid_from_maps_all_rows(self):
        """Generic broadcaster default: without a boundary the static map is
        applied everywhere (the script supplies the boundary, §十-4)."""
        df = _stock_df()
        feat = _sector_features()
        out = SectorBroadcaster().transform(
            df.copy(),
            sector_map={"000001": "SEC0000"},
            sector_features=feat,
        )
        assert (out["sector_code"] == "SEC0000").all()
        assert (out["momentum_5d"] == 0.05).all()


def test_transform_handles_all_nan_sector_code():
    """A stock with no valid sector assignment (all-NaN float64 sector_code
    key) must not crash merging against a str sector panel — sector features
    are simply absent (unknown sector)."""
    sb = SectorBroadcaster()
    industry_ranking = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "sector_code": ["A", "A", "B"],
        "sector_name": ["s1", "s1", "s2"],
        "change_pct": [0.5, -0.2, 0.3],
        "up_count": [10, 8, 5],
        "down_count": [2, 4, 6],
        "rank": [1, 1, 2],
        "leader": ["000001", "000001", "000002"],
    })
    sector_features = sb.build_sector_features(industry_ranking)

    base = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "stock_code": ["920305", "920305", "920305"],
        "close": [10.0, 10.1, 10.2],
    })
    base["sector_code"] = float("nan")  # float64 all-NaN, as .where() produces

    out = sb.transform(base, sector_features=sector_features)
    assert len(out) == 3  # unchanged, no crash
    # no sector_* broadcast columns (unknown sector); the sector_code key
    # column itself is not a broadcast column and survives.
    assert not any(
        c.startswith("sector_") and c != "sector_code" for c in out.columns
    )
