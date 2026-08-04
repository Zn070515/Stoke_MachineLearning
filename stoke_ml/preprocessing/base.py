"""Abstract base class and chain container for preprocessing steps.

Every step is scikit-learn compatible: fit() learns parameters from
training data, transform() applies them.  A PreprocessingChain composes
multiple steps into a single fit/transform pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect


class PreprocessingStep(ABC):
    """One preprocessing operation with fit/transform/fit_transform.

    Every step records ``fit_start`` / ``fit_end`` — the date range of the
    data it was last fit on (None until first fit) — so a reviewer can audit
    that no step was fit over full history instead of per-fold (v8 §三-1).
    PIT-safe steps (per-date / rolling normalizers) leave fit() stateless and
    compute statistics point-in-time inside transform(); the recorded range
    documents the caller's fit discipline.
    """

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
        """Fit then transform in a single pass — each step runs once."""
        current = df.copy()
        for step in self.steps:
            step.fit(current, **kwargs)
            current = step.transform(current, **kwargs)
        return current

    def add(self, step: PreprocessingStep) -> PreprocessingChain:
        self.steps.append(step)
        return self

    def to_config(self) -> dict:
        recorded = []
        for s in self.steps:
            params = {
                k: v for k, v in s.__dict__.items()
                if not k.endswith("_") and not callable(v)
                and not k.startswith("_")
            }
            recorded.append({"type": type(s).__name__, "params": params})
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
