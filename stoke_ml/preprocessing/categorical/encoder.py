"""ConceptBlockEncoder: multi-label concept membership → multi-hot + derived.

6-layer encoding (spec §3.5):
  L1 — concept vocabulary construction (top-N by frequency)
  L2 — multi-hot encoding (cb_0..cb_{N-1})
  L3 — derived per-stock features (board_count, momentum, has_hot_board)
  L4 — concept heat score (国信 3D: volume + momentum + news)
  L5 — concept momentum (multi-window: 3/6/12 months)
  L6 — board co-occurrence (overlap score, concept leader)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingStep

logger = logging.getLogger(__name__)


class ConceptBlockEncoder(PreprocessingStep):
    """Encode multi-label concept board membership as features.

    Parameters:
        top_n: number of most frequent concepts to encode.
        min_stocks_per_board: filter out micro-boards.
        momentum_months: windows (months) for concept momentum.
    """

    def __init__(
        self,
        top_n: int = 100,
        min_stocks_per_board: int = 5,
        momentum_months: tuple[int, ...] = (3, 6, 12),
    ):
        if top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {top_n}")
        if min_stocks_per_board < 1:
            raise ValueError(f"min_stocks_per_board must be >= 1, got {min_stocks_per_board}")
        for m in momentum_months:
            if m <= 0:
                raise ValueError(f"momentum_months values must be > 0, got {m}")
        self.top_n = top_n
        self.min_stocks_per_board = min_stocks_per_board
        self.momentum_months = momentum_months
        self._vocabulary: Optional[list[str]] = None  # fitted vocabulary

    def fit(self, df: pd.DataFrame, **kwargs) -> ConceptBlockEncoder:
        """Build concept vocabulary from all observed board names.

        This should be called on the full cross-stock dataset before transform.
        """
        if df.empty or "board_name" not in df.columns:
            self._vocabulary = []
            return self

        # Count board frequency, filter small boards
        board_counts = df.groupby("board_name").size()
        board_counts = board_counts[board_counts >= self.min_stocks_per_board]
        # Select top N
        self._vocabulary = (
            board_counts.nlargest(self.top_n)
            .index.tolist()
        )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        vocab = self._vocabulary
        if not vocab:
            logger.warning(
                "ConceptBlockEncoder not fitted — inferring vocabulary from "
                "current batch. This may produce different columns between "
                "train/test. Call fit() on the full dataset first."
            )
            if "board_name" in df.columns:
                counts = df.groupby("board_name").size()
                vocab = counts.nlargest(self.top_n).index.tolist()

        if not vocab:
            return df

        # L2: multi-hot encoding
        self._add_multihot(df, vocab)

        # L3: derived features
        self._add_derived(df)

        # L4: concept heat (simplified — full version needs volume/news data)
        self._add_concept_heat(df)

        # L5: concept momentum
        self._add_concept_momentum(df)

        # L6: board co-occurrence
        self._add_cooccurrence(df)

        return df

    # ── L2: multi-hot ──────────────────────────────────────────────────

    def _add_multihot(self, df, vocab):
        """Create multi-hot columns cb_0..cb_{N-1} via vectorized get_dummies.

        Each cb_j maps to vocab[j] regardless of which boards are present
        in the current batch — column semantics are stable across stocks.
        """
        if "board_name" not in df.columns or "stock_code" not in df.columns:
            return
        dummies = pd.get_dummies(df["board_name"], dtype=np.int8)
        for j, board in enumerate(vocab):
            col = f"cb_{j}"
            if board in dummies.columns:
                df[col] = dummies[board]
            else:
                df[col] = np.int8(0)

    # ── L3: derived ────────────────────────────────────────────────────

    def _add_derived(self, df):
        """Per-stock derived features from multi-hot columns."""
        if "board_change_pct" in df.columns:
            df["_board_pct"] = pd.to_numeric(df["board_change_pct"], errors="coerce")

        # Board count: number of distinct boards per stock-date (aggregate from long format)
        if "date" in df.columns and "stock_code" in df.columns:
            board_counts = df.groupby(["date", "stock_code"]).size().reset_index(name="_n_boards")
            board_counts = board_counts.set_index(["date", "stock_code"])
            df["board_count"] = (
                df.set_index(["date", "stock_code"])
                .index.map(board_counts["_n_boards"].to_dict())
            )
            df["board_count"] = df["board_count"].fillna(0).astype(np.int16)
            df.reset_index(drop=True, inplace=True)

        if "board_change_pct" in df.columns:
            # Mean/max momentum per stock across its boards (use cleaned column)
            df["board_momentum_mean"] = (
                df.groupby(["date", "stock_code"])["_board_pct"]
                .transform("mean")
                .astype(np.float32)
            )
            df["board_momentum_max"] = (
                df.groupby(["date", "stock_code"])["_board_pct"]
                .transform("max")
                .astype(np.float32)
            )

        if "_board_pct" in df.columns:
            daily_top10 = (
                df.groupby("date")["_board_pct"]
                .transform(lambda s: s.rank(pct=True) > 0.9)
            )
            df["has_hot_board"] = daily_top10.astype(np.int8)
            df.drop(columns=["_board_pct"], inplace=True)

    # ── L4: concept heat ───────────────────────────────────────────────

    def _add_concept_heat(self, df):
        """Compute concept heat score (国信 3D framework, simplified).

        Uses board_change_pct as a proxy for momentum_score.
        Full 3D would need volume_ratio and news_count per concept.
        """
        if "board_change_pct" not in df.columns:
            return
        df["_pct_clean"] = pd.to_numeric(df["board_change_pct"], errors="coerce")
        df["avg_concept_heat"] = (
            df.groupby(["date", "stock_code"])["_pct_clean"]
            .transform(lambda s: s.rank(pct=True).mean())
            .astype(np.float32)
        )
        df.drop(columns=["_pct_clean"], inplace=True)

    # ── L5: concept momentum ───────────────────────────────────────────

    def _add_concept_momentum(self, df):
        """Compute concept momentum over multiple windows (months→trading days)."""
        if "board_change_pct" not in df.columns:
            return
        df["_pct_clean"] = pd.to_numeric(df["board_change_pct"], errors="coerce")
        # Sort by date within each board before rolling
        if "date" in df.columns:
            df = df.sort_values(["board_name", "date"])
        days_per_month = 21
        for m in self.momentum_months:
            w = m * days_per_month
            col = f"concept_momentum_{m}m"
            df[col] = (
                df.groupby("board_name")["_pct_clean"]
                .transform(
                    lambda s: s.rolling(w, min_periods=max(5, w // 4)).sum()
                )
                .astype(np.float32)
            )
        df.drop(columns=["_pct_clean"], inplace=True)

    # ── L6: co-occurrence ──────────────────────────────────────────────

    def _add_cooccurrence(self, df):
        """Compute board overlap and leader flags."""
        if "lead_stock" in df.columns and "stock_code" in df.columns:
            df["is_concept_leader"] = (
                df["lead_stock"].astype(str) == df["stock_code"].astype(str)
            ).astype(np.int8)

        # Board overlap: per stock, average Jaccard with other stocks in same boards
        # Simplified: board_count as a proxy (more boards = higher neighborhood density)
        if "board_count" in df.columns:
            # Normalize to [0, 1]
            max_bc = df["board_count"].max()
            if max_bc > 0:
                df["board_overlap_score"] = (
                    df["board_count"] / max_bc
                ).astype(np.float32)
            else:
                df["board_overlap_score"] = 0.0
