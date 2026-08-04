"""PreprocessingPipeline: orchestration engine for preprocessing chains.

Chains are registered per source (e.g. 'news', 'guba') and can be run
independently.  The pipeline is configuration-driven and compatible with
both the existing FeaturePipeline and a future backtesting system.
"""

from __future__ import annotations

import logging

import pandas as pd

from stoke_ml.preprocessing.base import PreprocessingChain

logger = logging.getLogger(__name__)


class PreprocessingQualityError(Exception):
    """Raised in strict mode when a chain's output fails error-level quality checks.

    Carries the full quality report (``.report``) and the staged output
    DataFrame (``.df``) so callers can persist the partial result rather than
    losing it (v6 §十: 禁止 commit，保留 staging 输出).
    """

    def __init__(self, chain_name: str, report: list[dict], df: pd.DataFrame):
        self.chain_name = chain_name
        self.report = report
        self.df = df
        n_errors = sum(1 for r in report if r["level"] == "ERROR")
        messages = " | ".join(
            r.get("message", str(r)) for r in report if r["level"] == "ERROR"
        )
        super().__init__(
            f"QualityMonitor[{chain_name}] {n_errors} error-level issue(s): {messages}"
        )


class PreprocessingPipeline:
    """Register and run named preprocessing chains.

    Usage:
        pp = PreprocessingPipeline()
        pp.register_chain("news", text_chain)
        clean = pp.run("news", raw_articles, stock_code="000001")
    """

    def __init__(self):
        self._chains: dict[str, PreprocessingChain] = {}

    def register_chain(self, name: str, chain: PreprocessingChain) -> None:
        self._chains[name] = chain

    def run(self, chain_name: str, df: pd.DataFrame, *, strict: bool = False,
            **kwargs) -> pd.DataFrame:
        """Run a named chain on *df*, returning transformed DataFrame.

        With ``strict=True``, error-level quality problems raise
        :class:`PreprocessingQualityError` instead of degrading silently —
        the caller must decide whether to persist the staged output.
        """
        chain = self._chains.get(chain_name)
        if chain is None:
            raise KeyError(
                f"Chain '{chain_name}' not found. "
                f"Available: {self.list_chains()}"
            )
        out = chain.fit_transform(df, **kwargs)
        qm = getattr(self, "_quality_monitor", None)
        if qm is not None:
            qm.transform(out)  # pure check, does not modify data
            report = qm.report
            errors = [r for r in report if r["level"] == "ERROR"]
            warns = [r for r in report if r["level"] == "WARN"]
            logger.info(
                "QualityMonitor[%s]: %d errors, %d warnings",
                chain_name, len(errors), len(warns),
            )
            if errors:
                logger.warning(
                    "QualityMonitor[%s] errors: %s", chain_name, errors[:3]
                )
            if strict and errors:
                raise PreprocessingQualityError(chain_name, report, out)
        return out

    def get_chain(self, name: str):
        """Return the named PreprocessingChain, or None if not registered."""
        return self._chains.get(name)

    def list_chains(self) -> list[str]:
        return sorted(self._chains.keys())

    @property
    def topic_modeler(self):
        """The TopicModeler instance, if configured. May be None."""
        return getattr(self, "_topic_modeler", None)

    @property
    def quality_monitor(self):
        """The QualityMonitor instance, if configured. May be None."""
        return getattr(self, "_quality_monitor", None)

    @property
    def drift_monitor(self):
        """The DriftMonitor instance, if configured. May be None."""
        return getattr(self, "_drift_monitor", None)

    @property
    def registry(self):
        """The FeatureRegistry instance, if configured. May be None."""
        return getattr(self, "_registry", None)

    @classmethod
    def from_config(cls, config: dict) -> PreprocessingPipeline:
        """Build pipeline from configuration dict.

        *config* is the 'preprocessing' section from config.yaml.
        Accepts plain dict or OmegaConf DictConfig.
        """
        from stoke_ml.preprocessing.config import build_pipeline_from_config

        if config is not None and not isinstance(config, dict):
            try:
                from omegaconf import OmegaConf
                config = OmegaConf.to_container(config, resolve=True)
            except Exception:
                config = {}
        return build_pipeline_from_config(config)
