"""PreprocessingPipeline: orchestration engine for preprocessing chains.

Chains are registered per source (e.g. 'xueqiu', 'news') and can be run
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
        pp.register_chain("xueqiu", text_chain)
        clean = pp.run("xueqiu", raw_posts, stock_code="000001")
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
                f"Available: {list(self._chains.keys())}"
            )
        return chain.fit_transform(df, **kwargs)

    def list_chains(self) -> list[str]:
        return sorted(self._chains.keys())

    @classmethod
    def from_config(cls, config: dict) -> PreprocessingPipeline:
        """Build pipeline from configuration dict.

        *config* is the 'preprocessing' section from config.yaml.
        """
        from stoke_ml.preprocessing.config import build_pipeline_from_config
        return build_pipeline_from_config(config)
