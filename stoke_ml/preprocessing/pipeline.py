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
    losing it (禁止 commit，保留 staging 输出).
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


class PreprocessingScopeError(Exception):
    """Raised by the formal full-history path when a chain holds fold_train_only steps.

    A ``fold_train_only`` step must be fit per fold on train-only rows; running
    it over full history bakes validation/future information into its learned
    parameters (e.g. a concept vocabulary or a drift baseline).  The formal
    offline path refuses such chains rather than silently fitting on everything
    (§十-1).  Carry the split chains so the caller can re-route the step to the
    per-fold path instead of just losing the run.
    """

    def __init__(
        self,
        chain_name: str,
        step_types: list[str],
        offline_chain=None,
        fold_chain=None,
    ):
        self.chain_name = chain_name
        self.step_types = list(step_types)
        self.offline_chain = offline_chain
        self.fold_chain = fold_chain
        super().__init__(
            f"PreprocessingScope[{chain_name}] fold_train_only step(s) must not "
            f"run in full-history offline preprocessing: {self.step_types}"
        )


class DriftMonitorError(Exception):
    """Raised by ``PreprocessingPipeline.drift_check(..., on_drift="raise")``
    when the eval window drifted beyond the fitted baseline (§十-3).

    Carries the full drift report (``.report``) for the caller to persist.
    """

    def __init__(self, report: list[dict]):
        self.report = report
        features = [r.get("feature") for r in report]
        super().__init__(
            f"DriftMonitor: {len(features)} drifted/missing feature(s): {features}"
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
            formal: bool = False, **kwargs) -> pd.DataFrame:
        """Run a named chain on *df*, returning transformed DataFrame.

        With ``strict=True``, error-level quality problems raise
        :class:`PreprocessingQualityError` instead of degrading silently —
        the caller must decide whether to persist the staged output.

        With ``formal=True`` the caller declares this is full-history offline
        preprocessing; any ``fold_train_only`` step in the chain raises
        :class:`PreprocessingScopeError` (§十-1).  Such steps must be fit per
        fold on train-only rows via :meth:`PreprocessingChain.fold_fitted_chain`,
        not over the whole window.
        """
        chain = self._chains.get(chain_name)
        if chain is None:
            raise KeyError(
                f"Chain '{chain_name}' not found. "
                f"Available: {self.list_chains()}"
            )
        if formal:
            forbidden = chain.fold_train_only_steps()
            if forbidden:
                raise PreprocessingScopeError(
                    chain_name,
                    [type(s).__name__ for s in forbidden],
                    offline_chain=chain.offline_pit_chain(),
                    fold_chain=chain.fold_fitted_chain(),
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

    def offline_pit_chains(self) -> dict[str, "PreprocessingChain"]:
        """Map chain name → offline-safe sub-chain (fold_train_only steps dropped).

        A full-history offline preprocess pass runs exactly these sub-chains —
        the ``fold_train_only`` steps are routed to the per-fold path instead
        (§十-1).
        """
        return {name: c.offline_pit_chain() for name, c in self._chains.items()}

    def fold_fitted_chains(self) -> dict[str, "PreprocessingChain"]:
        """Map chain name → per-fold sub-chain (fold_train_only steps only).

        These are the steps a fold-aware trainer fits on its training slice
        before transforming the fold (§十-1).
        """
        return {name: c.fold_fitted_chain() for name, c in self._chains.items()}

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

    def drift_fit(self, df: pd.DataFrame) -> None:
        """Fit the DriftMonitor baseline on a TRAIN slice (§十-3).

        The baseline must be fit on training-fold rows only — fitting on the
        full window would benchmark each window against itself.  Raises when
        no DriftMonitor is configured (preprocessing.monitor.enabled=false).
        """
        dm = self.drift_monitor
        if dm is None:
            raise RuntimeError(
                "No DriftMonitor configured "
                "(preprocessing.monitor.enabled=false)."
            )
        dm.fit(df)
        logger.info(
            "DriftMonitor baseline fit on %d rows, %d features",
            len(df), len(dm.baseline_),
        )

    def drift_check(self, df: pd.DataFrame, *, on_drift: str = "warn") -> list[dict]:
        """Compare an EVAL slice against the fitted baseline (§十-3).

        Returns the drift report (one entry per drifted / missing feature).
        ``on_drift="raise"`` raises :class:`DriftMonitorError` when any feature
        drifted; ``"warn"`` (default) logs and returns the report.  Raises
        RuntimeError when no baseline was fitted — call ``drift_fit`` on the
        train fold first, so monitoring is never silently skipped.
        """
        if on_drift not in ("warn", "raise"):
            raise ValueError(
                f"on_drift must be 'warn' or 'raise', got {on_drift!r}"
            )
        dm = self.drift_monitor
        if dm is None:
            raise RuntimeError(
                "No DriftMonitor configured "
                "(preprocessing.monitor.enabled=false)."
            )
        if not dm.baseline_:
            raise RuntimeError(
                "DriftMonitor has no baseline — call pp.drift_fit(train_df) on "
                "the training fold before checking eval data (§十-3)."
            )
        dm.transform(df)
        report = dm.drift_report
        if report:
            logger.warning(
                "DriftMonitor: %d drifted/missing feature(s): %s",
                len(report), report[:5],
            )
        else:
            logger.info("DriftMonitor: no drifted features")
        if on_drift == "raise" and report:
            raise DriftMonitorError(report)
        return report

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

        # A None config is a legitimate "no preprocessing section" → empty
        # pipeline.  A malformed non-dict config is a real error and must
        # BLOCK, not silently degrade to an empty pipeline.
        if config is None:
            config = {}
        elif not isinstance(config, dict):
            try:
                from omegaconf import OmegaConf
                config = OmegaConf.to_container(config, resolve=True)
            except Exception as exc:
                raise ValueError(
                    f"preprocessing config could not be parsed: {exc}"
                ) from exc
        return build_pipeline_from_config(config)
