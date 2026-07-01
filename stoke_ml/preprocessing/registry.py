"""Feature registry for governance: definitions, lineage, baseline stats.

Each feature's full life-cycle is recorded — raw columns → transforms →
final column name — plus distribution snapshots for drift detection.
Tags enable one-command ablation group selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import logging
from pathlib import Path
from collections.abc import Iterator

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeatureDefinition:
    """One feature column with full lineage and metadata."""

    name: str
    category: str
    display_name: str = ""
    source: str = "unknown"
    dtype: str = "float32"
    value_range: tuple | None = None

    parents: list[str] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    step_version: str = "0.1.0"

    baseline_stats: dict = field(default_factory=dict)
    calibration_date: str = ""

    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.name

    def to_dict(self) -> dict:
        d = asdict(self)
        d["value_range"] = list(self.value_range) if self.value_range else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeatureDefinition:
        vr = d.get("value_range")
        if vr and isinstance(vr, list):
            d = {**d, "value_range": tuple(vr)}
        return cls(**{k: v for k, v in d.items()
                      if k in cls.__dataclass_fields__})


class FeatureRegistry:
    """Collect, query, and persist feature definitions."""

    def __init__(self, features: list[FeatureDefinition] | None = None):
        self._features: dict[str, FeatureDefinition] = {}
        for fd in (features or []):
            self.register(fd)

    # -- mutation -------------------------------------------------------

    def register(self, feature: FeatureDefinition) -> None:
        self._features[feature.name] = feature

    # -- query ----------------------------------------------------------

    def get(self, name: str) -> FeatureDefinition | None:
        return self._features.get(name)

    def get_by_group(self, tag: str) -> list[str]:
        """Return feature names that carry *tag*."""
        return sorted(
            name for name, fd in self._features.items()
            if tag in fd.tags
        )

    def get_by_source(self, source: str) -> list[str]:
        return sorted(
            name for name, fd in self._features.items()
            if fd.source == source
        )

    def list_names(self) -> list[str]:
        return sorted(self._features.keys())

    def validate_matrix(self, df) -> tuple[list[str], list[str]]:
        """Return (missing_cols, extra_cols) relative to registry."""
        registered = set(self._features.keys())
        present = set(df.columns)
        missing = sorted(registered - present)
        extra = sorted(present - registered)
        return missing, extra

    def export_lineage(self, fmt: str = "json") -> str:
        if fmt == "json":
            return json.dumps(
                {name: fd.to_dict() for name, fd in self._features.items()},
                ensure_ascii=False, indent=2,
            )
        raise ValueError(f"Unknown format: {fmt}")

    def check_drift(
        self,
        new_stats: dict[str, dict],
        p_threshold: float = 0.01,
    ) -> list[dict]:
        """Two-sample KS approximation: compare new_stats to baseline_stats.

        *new_stats* is {feature_name: {mean, std}}.
        Returns list of {feature, p_value, new_mean, baseline_mean}
        for features with p < p_threshold.
        """
        alerts = []
        try:
            from scipy import stats
        except ImportError:
            return alerts

        for name, baseline in self._features.items():
            if not baseline.baseline_stats or name not in new_stats:
                continue
            bm = baseline.baseline_stats
            nm = new_stats[name]
            try:
                ks_stat, p_val = stats.ks_2samp(
                    np.random.normal(bm["mean"], bm.get("std", 1.0), 100),
                    np.random.normal(nm["mean"], nm.get("std", 1.0), 100),
                )
                if p_val < p_threshold:
                    alerts.append({
                        "feature": name,
                        "p_value": float(p_val),
                        "new_mean": nm["mean"],
                        "baseline_mean": bm["mean"],
                    })
            except Exception:
                pass
        return alerts

    # -- persistence ----------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [fd.to_dict() for fd in self._features.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> FeatureRegistry:
        path = Path(path)
        if not path.exists():
            logger.warning("Registry file not found: %s", path)
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        features = [FeatureDefinition.from_dict(d) for d in data]
        return cls(features)

    def __len__(self) -> int:
        return len(self._features)

    def __iter__(self) -> Iterator[FeatureDefinition]:
        return iter(self._features.values())
