"""L3 Deepen: market-environment factors from raw macro + breadth columns.

MarketEnvRefiner compresses the ~28 raw macro columns (already merged as PO by
_merge_macro) into six regime factors, then assembles a composite regime score.
All outputs use the ``menv_`` prefix so TemporalTransformer treats them as
past-observed. Graceful when any input column is absent (sparse configs).
"""
import numpy as np
import pandas as pd

# factor name -> (source col, z-window); window=None means use the raw value.
_FACTOR_Z = {
    "menv_shibor_1m_z": ("shibor_1M", 60),
    "menv_fx_usd_cny_z": ("fx_usd_cny", 60),
    "menv_cpi_z": ("cpi_yoy", 60),
}
_FACTOR_RAW = {
    "menv_bond_10y2y_spread": "bond_cn_10y2y_spread",
    "menv_us_cn_10y_spread": ("bond_us_10y", "bond_cn_10y"),
    "menv_m1_m2_spread": ("m1_yoy", "m2_yoy"),
}
_COMPOSITE = [
    "menv_shibor_1m_z", "menv_us_cn_10y_spread", "menv_fx_usd_cny_z",
    "menv_m1_m2_spread", "menv_cpi_z", "menv_bond_10y2y_spread",
]


def _rolling_z(s: pd.Series, win: int) -> pd.Series:
    m = s.rolling(win, min_periods=20).mean()
    sd = s.rolling(win, min_periods=20).std()
    return ((s - m) / sd.replace(0, np.nan)).fillna(0.0).astype(np.float32)


class MarketEnvRefiner:
    """Compress raw macro cols into menv_* market-environment factors."""

    def refine(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "date" not in out.columns:
            return out
        for col, (src, win) in _FACTOR_Z.items():
            if src in out.columns and col not in out.columns:
                out[col] = _rolling_z(out[src], win)
        for col, src in _FACTOR_RAW.items():
            if col in out.columns:
                continue
            if isinstance(src, tuple):
                a, b = src
                if a in out.columns and b in out.columns:
                    out[col] = (out[a] - out[b]).astype(np.float32)
            elif src in out.columns:
                out[col] = out[src].astype(np.float32)
        present = [c for c in _COMPOSITE if c in out.columns]
        if present:
            out["menv_regime_z"] = out[present].mean(axis=1).fillna(0.0).astype(np.float32)
        return out
