"""Abstract base class and chain container for preprocessing steps.

Every step is scikit-learn compatible: fit() learns parameters from
training data, transform() applies them.  A PreprocessingChain composes
multiple steps into a single fit/transform pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect


class PreprocessingStep(ABC):
    """One preprocessing operation with fit/transform/fit_transform."""

    def fit(self, df, **kwargs):
        """Learn parameters from *df*. Default is no-op, return self."""
        return self

    @abstractmethod
    def transform(self, df, **kwargs):
        """Apply the learned transformation to *df*."""
        ...

    def fit_transform(self, df, **kwargs):
        """Fit then transform in one call."""
        self.fit(df, **kwargs)
        return self.transform(df, **kwargs)

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

    def fit(self, df, **kwargs):
        current = df.copy()
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
