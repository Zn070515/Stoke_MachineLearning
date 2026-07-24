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

    def run(self, chain_name: str, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run a named chain on *df*, returning transformed DataFrame."""
        chain = self._chains.get(chain_name)
        if chain is None:
            raise KeyError(
                f"Chain '{chain_name}' not found. "
                f"Available: {self.list_chains()}"
            )
        return chain.fit_transform(df, **kwargs)

    def get_chain(self, name: str):
        """Return the named PreprocessingChain, or None if not registered."""
        return self._chains.get(name)

    def list_chains(self) -> list[str]:
        return sorted(self._chains.keys())

    @property
    def topic_modeler(self):
        """The TopicModeler instance, if configured. May be None."""
        return getattr(self, "_topic_modeler", None)

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
