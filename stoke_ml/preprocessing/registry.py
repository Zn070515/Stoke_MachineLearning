"""Feature registry for governance: definitions, lineage, baseline stats.

Each feature's full life-cycle is recorded — raw columns → transforms →
final column name — plus distribution snapshots for drift detection.
Tags enable one-command ablation group selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import logging
import os
import tempfile
from pathlib import Path
from collections.abc import Iterator

import numpy as np

logger = logging.getLogger(__name__)


class _RegistryEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars to Python native types."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _to_json_safe(obj):
    """Recursively convert numpy types in nested structures to Python natives."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


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
        d["value_range"] = list(self.value_range) if self.value_range is not None else None
        d = _to_json_safe(d)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> FeatureDefinition:
        vr = d.get("value_range")
        if isinstance(vr, list):
            d = {**d, "value_range": tuple(vr) if vr else None}
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        unknown = set(d.keys()) - set(cls.__dataclass_fields__)
        if unknown:
            logger.debug("FeatureDefinition.from_dict: ignoring unknown keys %s", unknown)
        return cls(**known)


class FeatureRegistry:
    """Collect, query, and persist feature definitions."""

    def __init__(self, features: list[FeatureDefinition] | None = None):
        self._features: dict[str, FeatureDefinition] = {}
        for fd in (features or []):
            self.register(fd)

    # -- mutation -------------------------------------------------------

    def register(self, feature: FeatureDefinition) -> None:
        if feature.name in self._features:
            existing = self._features[feature.name]
            if existing.to_dict() != feature.to_dict():
                logger.warning(
                    "FeatureRegistry: overwriting '%s' with different definition",
                    feature.name,
                )
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
                ensure_ascii=False, indent=2, cls=_RegistryEncoder,
            )
        raise ValueError(f"Unknown format: {fmt}")

    def check_drift(
        self,
        new_stats: dict[str, dict],
        sigma_threshold: float = 3.0,
    ) -> list[dict]:
        """Deterministic drift detection: compare new_stats to baseline_stats.

        Uses sigma-based threshold on mean shift: |new_mean - baseline_mean|
        divided by baseline_std. A feature triggers an alert when the shift
        exceeds *sigma_threshold* standard deviations.

        *new_stats* is {feature_name: {mean, std}}.
        Returns list of {feature, sigma_shift, new_mean, baseline_mean}.
        """
        alerts = []
        for name, baseline in self._features.items():
            if not baseline.baseline_stats or name not in new_stats:
                continue
            bm = baseline.baseline_stats
            nm = new_stats[name]
            try:
                b_mean = float(bm.get("mean", 0.0))
                b_std = float(bm.get("std", 1.0))
                n_mean = float(nm.get("mean", 0.0))
                if not np.isfinite(b_mean) or not np.isfinite(n_mean):
                    continue
                if b_std <= 0:
                    continue
                sigma_shift = abs(n_mean - b_mean) / b_std
                if sigma_shift > sigma_threshold:
                    alerts.append({
                        "feature": name,
                        "sigma_shift": round(sigma_shift, 3),
                        "new_mean": round(n_mean, 4),
                        "baseline_mean": round(b_mean, 4),
                    })
            except (TypeError, ValueError) as e:
                logger.debug(
                    "check_drift: skipping '%s' — bad stats: %s", name, e,
                )
        return alerts

    # -- persistence ----------------------------------------------------

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [_to_json_safe(fd.to_dict()) for fd in self._features.values()]
        payload = json.dumps(data, ensure_ascii=False, indent=2)

        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".json", prefix=".registry_", dir=str(path.parent)
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, str(path))
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

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
