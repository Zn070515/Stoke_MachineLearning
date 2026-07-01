"""Config-driven pipeline construction.

Reads the 'preprocessing' section of config.yaml and assembles
PreprocessingChains for each text/numeric source.
"""

from __future__ import annotations

import logging

from stoke_ml.preprocessing.base import PreprocessingChain
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.preprocessing.text.bipolar import BipolarClassifier
from stoke_ml.preprocessing.text.decay import TimeDecayWeighter
from stoke_ml.preprocessing.text.quality import QualityFilter
from stoke_ml.preprocessing.text.topics import TopicModeler
from stoke_ml.preprocessing.text.aggregation import DailyAggregator
from stoke_ml.preprocessing.numeric.outlier import OutlierDetector
from stoke_ml.preprocessing.numeric.missing import MissingImputer
from stoke_ml.preprocessing.numeric.scaling import RobustScaler
from stoke_ml.preprocessing.numeric.cross_section import CrossSectionNormalizer
from stoke_ml.preprocessing.numeric.higher_order import HigherOrderDeriver

logger = logging.getLogger(__name__)

_STEP_REGISTRY = {
    "BipolarClassifier": BipolarClassifier,
    "TimeDecayWeighter": TimeDecayWeighter,
    "DailyAggregator": DailyAggregator,
    "QualityFilter": QualityFilter,
    "TopicModeler": TopicModeler,
    "OutlierDetector": OutlierDetector,
    "MissingImputer": MissingImputer,
    "RobustScaler": RobustScaler,
    "CrossSectionNormalizer": CrossSectionNormalizer,
    "HigherOrderDeriver": HigherOrderDeriver,
}


def build_pipeline_from_config(cfg: dict) -> PreprocessingPipeline:
    """Assemble PreprocessingPipeline from config dict.

    Accepts plain dict or OmegaConf DictConfig.
    """
    if cfg is not None and not isinstance(cfg, dict):
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.to_container(cfg, resolve=True)
        except Exception:
            cfg = {}

    pp = PreprocessingPipeline()
    pp_cfg = cfg if isinstance(cfg, dict) else {}
    if not pp_cfg:
        return pp

    # Text chain
    text_cfg = pp_cfg.get("text", {})
    bipolar = BipolarClassifier(
        pos_threshold=text_cfg.get("bipolar", {}).get("threshold_positive", 0.2),
        neg_threshold=text_cfg.get("bipolar", {}).get("threshold_negative", -0.2),
    )
    decay_cfg = text_cfg.get("time_decay", {})
    decay = TimeDecayWeighter(
        halflife_days=decay_cfg.get("halflife_days", 7),
    )
    agg_cfg = text_cfg.get("aggregation", {})
    aggregator = DailyAggregator(
        windows=tuple(agg_cfg.get("windows", [3, 5, 10, 20])),
    )

    # text_pre: QualityFilter → BipolarClassifier → TimeDecayWeighter (per-post)
    text_pre = PreprocessingChain(name="text_pre")
    qf_cfg = text_cfg.get("quality_filter", {})
    text_pre.add(QualityFilter(
        min_text_length=qf_cfg.get("min_text_length", 5),
        max_duplicate_similarity=qf_cfg.get("max_duplicate_similarity", 0.9),
        remove_html=qf_cfg.get("remove_html", True),
    ))
    text_pre.add(bipolar)
    text_pre.add(decay)
    pp.register_chain("text_pre", text_pre)

    # text_aggregate: DailyAggregator only
    text_agg = PreprocessingChain(name="text_aggregate")
    text_agg.add(aggregator)
    pp.register_chain("text_aggregate", text_agg)

    # text: full chain for standalone use
    text_full = PreprocessingChain(name="text")
    for step in text_pre.steps:
        text_full.add(step)
    text_full.add(aggregator)
    pp.register_chain("text", text_full)

    # Topic modeler (cross-stock fit, per-stock transform — not in chain)
    topic_cfg = text_cfg.get("topic_model", {})
    if topic_cfg.get("enabled", False):
        pp._topic_modeler = TopicModeler(
            enabled=True,
            n_topics=topic_cfg.get("n_topics", "auto"),
            min_topic_size=topic_cfg.get("min_topic_size", 50),
            model_cache_dir=topic_cfg.get("model_cache_dir", "models/bertopic"),
            embedding_model=topic_cfg.get("embedding_model", "finbert"),
        )

    # Numeric cleaning chain (FeaturePipeline-safe: no scaling, no higher-order)
    num_cfg = pp_cfg.get("numeric", {})
    clean_chain = PreprocessingChain(name="numeric_clean")
    oc = num_cfg.get("outlier", {})
    clean_chain.add(OutlierDetector(
        threshold=oc.get("threshold", 5.0),
        clip=oc.get("clip", True),
    ))
    mc = num_cfg.get("missing", {})
    clean_chain.add(MissingImputer(
        short_gap_max=mc.get("short_gap_max", 2),
        medium_gap_max=mc.get("medium_gap_max", 10),
    ))
    cs = num_cfg.get("cross_section", {})
    clean_chain.add(CrossSectionNormalizer(
        enabled=cs.get("enabled", True),
        stages=cs.get("stages", ["sector", "size", "adaptive"]),
    ))
    pp.register_chain("numeric", clean_chain)

    # Numeric full chain (standalone: includes scaling and higher-order features)
    full_chain = PreprocessingChain(name="numeric_full")
    for step in clean_chain.steps:
        full_chain.add(step)
    sc = num_cfg.get("scaling", {})
    full_chain.add(RobustScaler(
        window_days=sc.get("window_days", 252),
        winsorize_sigma=sc.get("winsorize_sigma", 3.0),
        min_periods=min(sc.get("min_periods", 63), sc.get("window_days", 252)),
    ))
    ho = num_cfg.get("higher_order", {})
    if ho.get("enabled", True):
        full_chain.add(HigherOrderDeriver(enabled=True))
    pp.register_chain("numeric_full", full_chain)

    return pp
