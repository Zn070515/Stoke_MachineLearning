"""FlowDecomposer: decompose raw capital flow into multi-layer factor suite.

6-layer decomposition (spec §3.1):
  L1 — size-tier ratios
  L2 — OFI intensity (rolling z-score)
  L3 — persistence (consecutive inflow days)
  L4 — divergence (price-flow confirmation failure)
  L5 — residualization (strip return contamination)
  L6 — size-tier spread (large minus small)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep

logger = logging.getLogger(__name__)

FLOW_COLS = ["main_net", "super_net", "large_net", "mid_net", "small_net"]


def _zscore_last(s: np.ndarray) -> float:
    """Z-score of the last element within the rolling window."""
    std = np.std(s, ddof=0)
    if std < 1e-12:
        return 0.0
    return float((s[-1] - np.mean(s)) / std)


class FlowDecomposer(PreprocessingStep):
    """Decompose capital flow into ratios, intensity, persistence, divergence.

    Parameters:
        persistence_windows: rolling windows for consecutive inflow count.
        intensity_window: rolling window for z-score computation.
        divergence_window: window for price-flow divergence check.
        flow_halflife: decay half-life (days) for flow momentum EMA.
        extreme_threshold: |z| > threshold flags extreme flow day.
        residualize: run cross-sectional regression to strip return contamination.
    """

    def __init__(
        self,
        persistence_windows: tuple[int, ...] = (5, 10, 20),
        intensity_window: int = 20,
        divergence_window: int = 5,
        flow_halflife: int = 7,
        extreme_threshold: float = 1.8,
        residualize: bool = True,
    ):
        self.persistence_windows = persistence_windows
        self.intensity_window = intensity_window
        self.divergence_window = divergence_window
        self.flow_halflife = flow_halflife
        self.extreme_threshold = extreme_threshold
        self.residualize = residualize

    def transform(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        flow_cols_present = [c for c in FLOW_COLS if c in df.columns]
        if not flow_cols_present:
            return df

        df = df.sort_values(["stock_code", "date"]) if "stock_code" in df.columns and "date" in df.columns else df

        self._compute_ratios(df, flow_cols_present)
        self._compute_intensity(df, flow_cols_present)
        self._compute_persistence(df, flow_cols_present)
        self._compute_divergence(df, flow_cols_present)
        if self.residualize and "close" in df.columns:
            self._compute_residual(df)
        if "large_ratio" in df.columns and "small_ratio" in df.columns:
            df["large_minus_small"] = (df["large_ratio"] - df["small_ratio"]).astype(
                np.float32
            )

        return df

    # ── L1: size-tier ratios ──────────────────────────────────────────

    def _compute_ratios(self, df, flow_cols):
        present = set(flow_cols)
        total = pd.Series(0.0, index=df.index)
        for c in ["super_net", "large_net", "mid_net", "small_net"]:
            if c in present:
                total += df[c].abs()
        eps = 1e-8

        tier_map = {
            "super_net": "super_ratio",
            "large_net": "large_ratio",
            "mid_net": "mid_ratio",
            "small_net": "small_ratio",
        }
        for src, dst in tier_map.items():
            if src in present:
                df[dst] = (df[src] / (total + eps)).astype(np.float32)

        if "main_net" in present:
            df["main_ratio"] = (df["main_net"] / (total + eps)).astype(np.float32)

    # ── L2: OFI intensity ─────────────────────────────────────────────

    def _compute_intensity(self, df, flow_cols):
        if "main_net" not in flow_cols:
            return
        has_stock = "stock_code" in df.columns
        w = self.intensity_window
        if has_stock:
            grp = df.groupby("stock_code")["main_net"]
            roll_mean = (
                grp.rolling(w, min_periods=5).mean().reset_index(level=0, drop=True)
            )
            roll_std = (
                grp.rolling(w, min_periods=5).std(ddof=0).reset_index(level=0, drop=True)
            )
        else:
            roll_mean = df["main_net"].rolling(w, min_periods=5).mean()
            roll_std = df["main_net"].rolling(w, min_periods=5).std(ddof=0)

        df["flow_z"] = (
            ((df["main_net"] - roll_mean) / (roll_std + 1e-8))
            .clip(-5, 5)
            .astype(np.float32)
        )
        df["flow_intensity"] = df["flow_z"].abs().astype(np.float32)
        df["is_extreme_flow"] = df["flow_intensity"] > self.extreme_threshold

        alpha = np.exp(-np.log(2) / max(self.flow_halflife, 1))
        if has_stock:
            df["flow_momentum"] = (
                df.groupby("stock_code")["flow_z"]
                .transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
                .astype(np.float32)
            )
        else:
            df["flow_momentum"] = (
                df["flow_z"].ewm(alpha=alpha, adjust=False).mean().astype(np.float32)
            )

    # ── L3: persistence ───────────────────────────────────────────────

    def _compute_persistence(self, df, flow_cols):
        if "main_net" not in flow_cols or "stock_code" not in df.columns:
            return
        df["_pos"] = (df["main_net"] > 0).astype(np.int8)
        for w in self.persistence_windows:
            col = f"consecutive_inflow_{w}d"
            df[col] = (
                df.groupby("stock_code")["_pos"]
                .rolling(w, min_periods=1)
                .sum()
                .reset_index(level=0, drop=True)
                .astype(np.int16)
            )
        df.drop(columns=["_pos"], inplace=True)

    # ── L4: divergence ────────────────────────────────────────────────

    def _compute_divergence(self, df, flow_cols):
        if "close" not in df.columns or "main_net" not in flow_cols:
            return
        w = max(self.divergence_window, 1)
        has_stock = "stock_code" in df.columns

        if has_stock:
            price_chg = df.groupby("stock_code")["close"].pct_change()
            price_z = (
                price_chg.groupby(df["stock_code"])
                .rolling(w, min_periods=3)
                .apply(_zscore_last, raw=True)
                .reset_index(level=0, drop=True)
            )
            flow_cumsum = (
                df.groupby("stock_code")["main_net"]
                .rolling(w, min_periods=3)
                .sum()
                .reset_index(level=0, drop=True)
            )
            flow_z = (
                flow_cumsum.groupby(df["stock_code"])
                .rolling(w, min_periods=3)
                .apply(_zscore_last, raw=True)
                .reset_index(level=0, drop=True)
            )
        else:
            price_chg = df["close"].pct_change()
            price_z = price_chg.rolling(w, min_periods=3).apply(_zscore_last, raw=True)
            flow_cumsum = df["main_net"].rolling(w, min_periods=3).sum()
            flow_z = flow_cumsum.rolling(w, min_periods=3).apply(_zscore_last, raw=True)

        df["flow_price_divergence"] = (
            (np.sign(price_z.fillna(0)) != np.sign(flow_z.fillna(0))).astype(np.int8)
        )

    # ── L5: residualization ───────────────────────────────────────────

    def _compute_residual(self, df):
        """Strip return contamination from flow signal.

        Regresses flow_z on contemporaneous return and keeps the residual:
            flow_z = α + β·ret + ε  →  ε = flow_z - (α + β·ret)
        ε is the purified alpha — flow signal orthogonal to price movement.
        """
        if "flow_z" not in df.columns:
            return
        ret = (
            df.groupby("stock_code")["close"].pct_change()
            if "stock_code" in df.columns
            else df["close"].pct_change()
        )
        mask = ret.notna() & df["flow_z"].notna()
        if mask.sum() < 10:
            df["flow_alpha_residual"] = df["flow_z"].fillna(0).astype(np.float32)
            return
        from numpy.polynomial import polynomial as P

        x = ret.loc[mask].values       # regressor: return
        y = df.loc[mask, "flow_z"].values  # regressand: flow_z
        c = P.polyfit(x, y, 1)         # flow_z = α + β·ret
        fitted = c[0] + c[1] * ret.fillna(0)
        df["flow_alpha_residual"] = (
            df["flow_z"].fillna(0) - fitted
        ).astype(np.float32)


