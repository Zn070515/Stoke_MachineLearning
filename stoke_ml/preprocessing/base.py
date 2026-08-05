"""Abstract base class and chain container for preprocessing steps.

Every step is scikit-learn compatible: fit() learns parameters from
training data, transform() applies them.  A PreprocessingChain composes
multiple steps into a single fit/transform pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect

# Every step declares which preprocessing regime it belongs to (§十-1):
#   stateless_pit   — deterministic per-date transform, no learned state.  Safe
#                     to run over full history offline; fit() only records range.
#   fold_train_only — learns parameters from training rows (vocabulary, scaler
#                     baseline, drift baseline).  MUST be re-fit per fold on
#                     train-only data; a formal full-history pass REFUSES these.
#   global_frozen   — fit once on a pinned reference (e.g. a corpus cutoff) and
#                     frozen; safe offline ONLY when the artifact is pinned and
#                     not re-fit on the full window.
FIT_SCOPES = ("stateless_pit", "fold_train_only", "global_frozen")


class PreprocessingStep(ABC):
    """One preprocessing operation with fit/transform/fit_transform.

    Every step records ``fit_start`` / ``fit_end`` — the date range of the
    data it was last fit on (None until first fit) — so a reviewer can audit
    that no step was fit over full history instead of per-fold.
    PIT-safe steps (per-date / rolling normalizers) leave fit() stateless and
    compute statistics point-in-time inside transform(); the recorded range
    documents the caller's fit discipline.
    """

    # One of FIT_SCOPES; see the module docstring above (§十-1).  A subclass
    # must set this when it learns parameters from the data it is fit on.
    fit_scope = "stateless_pit"

    # PIT fit-range provenance, set by _record_fit_range() during fit().
    fit_start = None
    fit_end = None

    def fit(self, df, **kwargs):
        """Learn parameters from *df*. Default is no-op, return self."""
        self._record_fit_range(df)
        return self

    @abstractmethod
    def transform(self, df, **kwargs):
        """Apply the learned transformation to *df*."""
        ...

    def fit_transform(self, df, **kwargs):
        """Fit then transform in one call."""
        self.fit(df, **kwargs)
        return self.transform(df, **kwargs)

    def _record_fit_range(self, df):
        """Store min/max date of *df* as fit_start/fit_end for auditing."""
        lo, hi = _frame_date_range(df)
        if lo is not None:
            self.fit_start = lo
            self.fit_end = hi

    def __repr__(self) -> str:
        init_params = _init_param_repr(self)
        return f"{type(self).__name__}({init_params})"


class PreprocessingChain(PreprocessingStep):
    """Ordered sequence of PreprocessingSteps.

    Each step's transform output becomes the next step's input.
    fit() calls fit() on every step in order using the same df.
    transform() pipes df through each step.
    fit_transform() fits on the *first* step's input, then transforms
    through all steps.
    """

    def __init__(self, steps=None, name="chain"):
        self.steps = list(steps or [])
        self.name = name
        self.fit_start = None
        self.fit_end = None

    def fit(self, df, **kwargs):
        current = df.copy()
        self._record_fit_range(current)
        for step in self.steps:
            step.fit(current, **kwargs)
            current = step.transform(current, **kwargs)
        return self

    def transform(self, df, **kwargs):
        current = df.copy()
        for step in self.steps:
            current = step.transform(current, **kwargs)
        return current

    def fit_transform(self, df, **kwargs):
        """Fit then transform in a single pass — each step runs once.

        Mirrors fit() so the chain records its own fit_start/fit_end — the
        formal PreprocessingPipeline.run() path uses fit_transform(), and
        without this the chain-level provenance would stay None while every
        step records a range.
        """
        current = df.copy()
        self._record_fit_range(current)
        for step in self.steps:
            step.fit(current, **kwargs)
            current = step.transform(current, **kwargs)
        return current

    def add(self, step: PreprocessingStep) -> PreprocessingChain:
        if step.fit_scope not in FIT_SCOPES:
            raise ValueError(
                f"step {type(step).__name__} has unknown fit_scope "
                f"{step.fit_scope!r} (expected one of {FIT_SCOPES})"
            )
        self.steps.append(step)
        return self

    def fold_train_only_steps(self) -> list[PreprocessingStep]:
        """Steps that must be re-fit per fold on train-only rows (§十-1)."""
        return [s for s in self.steps if s.fit_scope == "fold_train_only"]

    def offline_pit_chain(self) -> "PreprocessingChain":
        """Sub-chain safe to run in full-history offline preprocessing.

        Drops every ``fold_train_only`` step — those must be fit per fold on
        train-only data and are refused by the formal offline path (§十-1).
        """
        sub = PreprocessingChain(name=f"{self.name}_offline_pit")
        sub.steps = [s for s in self.steps if s.fit_scope != "fold_train_only"]
        return sub

    def fold_fitted_chain(self) -> "PreprocessingChain":
        """Sub-chain of steps that MUST be fit per fold on train-only rows.

        The complement of :meth:`offline_pit_chain` — the steps a fold-aware
        trainer fits on its training slice before transforming the fold (§十-1).
        """
        sub = PreprocessingChain(name=f"{self.name}_fold_fitted")
        sub.steps = [s for s in self.steps if s.fit_scope == "fold_train_only"]
        return sub

    def to_config(self) -> dict:
        recorded = []
        for s in self.steps:
            params = {
                k: v for k, v in s.__dict__.items()
                if not k.endswith("_") and not callable(v)
                and not k.startswith("_")
            }
            recorded.append({
                "type": type(s).__name__,
                "fit_scope": s.fit_scope,
                "params": params,
            })
        return {"name": self.name, "steps": recorded}

    def __repr__(self) -> str:
        step_names = " → ".join(type(s).__name__ for s in self.steps)
        return f"PreprocessingChain('{self.name}': {step_names or 'empty'})"


def _frame_date_range(df):
    """Return (min_date, max_date) of *df* via its 'date' column or DatetimeIndex.

    Returns (None, None) when no date axis is available, so fit provenance is
    simply not recorded for row-id-indexed frames.
    """
    import pandas as pd

    if df is None:
        return None, None
    if "date" in getattr(df, "columns", ()):
        dates = df["date"]
    elif isinstance(getattr(df, "index", None), pd.DatetimeIndex):
        dates = df.index
    else:
        return None, None
    try:
        return dates.min(), dates.max()
    except (TypeError, ValueError):
        return None, None


def _init_param_repr(obj) -> str:
    """Reconstruct how __init__ was called from stored attributes."""
    try:
        sig = inspect.signature(type(obj).__init__)
        params = []
        for name, param in sig.parameters.items():
            if name in ("self", "args", "kwargs"):
                continue
            if hasattr(obj, name):
                val = getattr(obj, name)
                params.append(f"{name}={val!r}")
        return ", ".join(params)
    except (ValueError, TypeError):
        return "..."
