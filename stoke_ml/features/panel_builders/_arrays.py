"""Panel array allocation + finalization (T8 memmap seam).

``PanelArrays`` encapsulates all dense panel arrays — allocation, per-stock
write (scatter through public attributes), sanitization, and final dict
assembly.  T8 swaps the ``np.zeros`` backing for chunked memmap without
touching any builder loop.

When ``sink_dir`` is set, the three big feature grids (static/pk/po) are
allocated as ``np.lib.format.open_memmap`` files directly on disk, so the
complete ``(N, T, D)`` grids never reside in RAM.  The 2-D target/mask arrays
are small (N x T) and stay dense.
"""
import os
import numpy as np


class PanelArrays:
    """Container for all dense panel arrays (T8 memmap seam).

    Created early (once ``N_stocks`` and ``max_T`` are known); builders write
    into the public ndarray attributes directly during their per-stock loops.
    ``alloc_features()`` is called later (after column discovery gives the
    dimensionalities); ``sanitize()`` + ``assemble()`` run last.

    Parameters
    ----------
    N_stocks : int
    max_T : int
    sink_dir : str or None
        When set, the three big (N, T, D) grids are memmap-backed to disk
        files under this directory.  Small 2-D arrays stay dense in RAM.
        None (default) preserves the original all-dense behaviour.
    """

    def __init__(self, N_stocks: int, max_T: int, sink_dir: str | None = None):
        self.N = N_stocks
        self.T = max_T
        self._sink_dir = sink_dir
        self._sink_paths: dict[str, str] = {}
        self._alloc_target_arrays()

    # -- allocation -------------------------------------------------------

    def _alloc_target_arrays(self):
        N, T = self.N, self.T
        # Targets — always dense (small N x T).
        self.y_dir = np.full((N, T), -100, dtype=np.int64)
        self.y_ret = np.zeros((N, T), dtype=np.float32)
        self.y_vol = np.zeros((N, T), dtype=np.float32)
        self.forward_vol_nobs = np.zeros((N, T), dtype=np.int32)
        # Fill-probability accumulators (per-date, not per-stock)
        self.entry_counts = np.zeros(T, dtype=np.int64)
        self.filled_counts = np.zeros(T, dtype=np.int64)
        # Observation / entry / target masks
        self.obs = np.zeros((N, T), dtype=bool)
        self.entry = np.zeros((N, T), dtype=bool)
        self.ret_tgt = np.zeros((N, T), dtype=bool)
        self.vol_tgt = np.zeros((N, T), dtype=bool)
        self.realized = np.zeros((N, T), dtype=np.float32)
        # Raw price paths (NaN outside a stock's trading days)
        self.close_price = np.full((N, T), np.nan, dtype=np.float32)
        self.open_price = np.full((N, T), np.nan, dtype=np.float32)
        # PIT-static raw inputs (pre-normalization)
        self.amt60_raw = np.zeros((N, T), dtype=np.float32)
        self.first_col = np.full(N, -1, dtype=np.int32)
        self.has_amount = np.ones(N, dtype=bool)
        # Per-stock position map + per-stock row count.  stock_pos is written
        # by TargetBuilder.compute() (which pre-sizes it) — leave it unset here
        # so a read before the target loop fails loudly rather than silently
        # indexing an empty list.
        self.stock_pos: list | None = None
        self.stock_T = np.zeros(N, dtype=np.int32)

    def alloc_features(self, static_dim: int, pk_dim: int, po_dim: int):
        """Allocate the three feature grids (called after column discovery).

        When ``sink_dir`` is set, the grids are backed by disk files via
        ``np.lib.format.open_memmap`` (writes proper .npy headers so
        ``np.load(mmap_mode='r')`` can re-open them lazily).  The 2-D
        target/mask arrays stay dense regardless.
        """
        N, T = self.N, self.T
        if self._sink_dir is not None:
            os.makedirs(self._sink_dir, exist_ok=True)
            self.static = np.lib.format.open_memmap(
                os.path.join(self._sink_dir, "static_features.npy"),
                mode="w+", dtype=np.float32, shape=(N, T, static_dim),
            )
            self.pk = np.lib.format.open_memmap(
                os.path.join(self._sink_dir, "past_known.npy"),
                mode="w+", dtype=np.float32, shape=(N, T, pk_dim),
            )
            self.po = np.lib.format.open_memmap(
                os.path.join(self._sink_dir, "past_observed.npy"),
                mode="w+", dtype=np.float32, shape=(N, T, po_dim),
            )
            self._sink_paths = {
                "static": os.path.join(self._sink_dir, "static_features.npy"),
                "pk": os.path.join(self._sink_dir, "past_known.npy"),
                "po": os.path.join(self._sink_dir, "past_observed.npy"),
            }
        else:
            self.static = np.zeros((N, T, static_dim), dtype=np.float32)
            self.pk = np.zeros((N, T, pk_dim), dtype=np.float32)
            self.po = np.zeros((N, T, po_dim), dtype=np.float32)

    def flush_sink(self):
        """Flush pending data to the sink files and close underlying mappings.

        Called AFTER the build is finished and ``assemble()`` has captured
        the memmap references into the returned dict.  On Windows the caller
        must close the sink before ``save_panel_memmap`` can write to the same
        directory (open memmaps keep their backing files locked).
        """
        if not self._sink_paths:
            return
        for attr in ("po", "pk", "static"):
            arr = getattr(self, attr, None)
            if arr is not None and isinstance(arr, np.memmap):
                arr.flush()
                if hasattr(arr, "_mmap") and arr._mmap is not None:
                    arr._mmap.close()

    # -- sanitization -----------------------------------------------------

    def sanitize(self):
        """Replace NaN/Inf with zeros and clip extreme values on all feature
        and target arrays (post-normalization)."""
        for attr in ("pk", "po", "static"):
            arr = getattr(self, attr, None)
            if arr is not None:
                np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
                np.clip(arr, -10.0, 10.0, out=arr)
        for attr in ("y_ret", "y_vol"):
            arr = getattr(self, attr)
            np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    # -- final assembly ---------------------------------------------------

    def assemble(
        self,
        global_dates: np.ndarray,
        decision_arr: np.ndarray,
        history_arr: np.ndarray,
        universe_eligible_arr: np.ndarray,
        fill_prob_arr: np.ndarray,
        pk_cols: list,
        po_cols: list,
        valid_codes: list,
    ) -> dict:
        """Build the return dict expected by callers (train_panel, tests)."""
        T = self.T
        N = self.N
        date_idx_arr = np.tile(np.arange(T, dtype=np.int32), (N, 1))
        return {
            "static_features": self.static,
            "past_known": self.pk,
            "past_observed": self.po,
            "y_direction": self.y_dir,
            "y_return": self.y_ret,
            "y_volatility": self.y_vol,
            "date_indices": date_idx_arr,
            "global_dates": global_dates,
            "observation_mask": self.obs,
            "entry_eligible_mask": self.entry,
            "return_target_mask": self.ret_tgt,
            "vol_target_mask": self.vol_tgt,
            "forward_vol_nobs": self.forward_vol_nobs,
            "realized_return": self.realized,
            "fill_prob": fill_prob_arr,
            "decision_eligible_mask": decision_arr,
            "history_eligible_mask": history_arr,
            "universe_eligible_mask": universe_eligible_arr,
            "close_price": self.close_price,
            "open_price": self.open_price,
            "past_known_cols": list(pk_cols),
            "past_observed_cols": list(po_cols),
            "stock_codes": list(valid_codes),
        }
