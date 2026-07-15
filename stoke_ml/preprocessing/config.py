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
from stoke_ml.preprocessing.daily_continuous.flow import FlowDecomposer
from stoke_ml.preprocessing.event_sparse.aggregator import EventToDaily
from stoke_ml.preprocessing.cross_sectional.board import BoardBroadcaster
from stoke_ml.preprocessing.cross_sectional.sector import SectorBroadcaster
from stoke_ml.preprocessing.categorical.encoder import ConceptBlockEncoder

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
    "FlowDecomposer": FlowDecomposer,
    "EventToDaily": EventToDaily,
    "BoardBroadcaster": BoardBroadcaster,
    "SectorBroadcaster": SectorBroadcaster,
    "ConceptBlockEncoder": ConceptBlockEncoder,
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

    # text: full chain for standalone use (independent instances, not shared)
    text_full = PreprocessingChain(name="text")
    text_full.add(QualityFilter(
        min_text_length=qf_cfg.get("min_text_length", 5),
        max_duplicate_similarity=qf_cfg.get("max_duplicate_similarity", 0.9),
        remove_html=qf_cfg.get("remove_html", True),
    ))
    text_full.add(BipolarClassifier(
        pos_threshold=text_cfg.get("bipolar", {}).get("threshold_positive", 0.2),
        neg_threshold=text_cfg.get("bipolar", {}).get("threshold_negative", -0.2),
    ))
    text_full.add(TimeDecayWeighter(
        halflife_days=decay_cfg.get("halflife_days", 7),
    ))
    text_full.add(DailyAggregator(
        windows=tuple(agg_cfg.get("windows", [3, 5, 10, 20])),
    ))
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

    # ── Shape A: daily_continuous chain "flow" ──
    flow_cfg = pp_cfg.get("flow", {})
    if flow_cfg.get("enabled", True):
        flow_chain = PreprocessingChain(name="flow")
        flow_chain.add(FlowDecomposer(
            persistence_windows=tuple(flow_cfg.get("persistence_windows", [5, 10, 20])),
            intensity_window=flow_cfg.get("intensity_window", 20),
            residualize=flow_cfg.get("residualize", True),
        ))
        flow_chain.add(OutlierDetector(
            threshold=flow_cfg.get("outlier_threshold", oc.get("threshold", 5.0)),
            clip=oc.get("clip", True),
        ))
        flow_chain.add(MissingImputer(
            short_gap_max=flow_cfg.get("short_gap_max", mc.get("short_gap_max", 2)),
            medium_gap_max=flow_cfg.get("medium_gap_max", mc.get("medium_gap_max", 10)),
        ))
        # NOTE: CrossSectionNormalizer is NOT included here — it requires
        # cross-stock peers per date and is applied at the panel level in
        # FeaturePipeline.build_features_from_panel() instead.
        pp.register_chain("flow", flow_chain)

    # ── Shape B: event_sparse chains (one per event type) ──
    event_cfg = pp_cfg.get("event", {})
    for etype in ["block_trade", "shareholder", "lockup", "dividend"]:
        ecfg = event_cfg.get(etype, {})
        echain = PreprocessingChain(name=f"event_{etype}")
        echain.add(EventToDaily(
            event_type=etype,
            decay_halflife_days=ecfg.get("decay_halflife_days", 90),
            forward_fill_max=ecfg.get("forward_fill_max", 5),
        ))
        echain.add(MissingImputer(
            short_gap_max=ecfg.get("short_gap_max", 5),
            medium_gap_max=ecfg.get("medium_gap_max", 60),
        ))
        pp.register_chain(f"event_{etype}", echain)

    # ── Shape C: cross_sectional chains ──
    cs_cfg = pp_cfg.get("cross_sectional", {})
    if cs_cfg.get("board", {}).get("enabled", True):
        board_chain = PreprocessingChain(name="board")
        board_chain.add(BoardBroadcaster(
            consecutive_lookback=cs_cfg.get("board", {}).get("consecutive_lookback", 20),
        ))
        pp.register_chain("board", board_chain)

    if cs_cfg.get("sector", {}).get("enabled", True):
        sector_chain = PreprocessingChain(name="sector")
        sector_chain.add(SectorBroadcaster(
            momentum_windows=tuple(
                cs_cfg.get("sector", {}).get("momentum_windows", [5, 20, 60, 252])
            ),
            breadth_normalize_window=cs_cfg.get("sector", {}).get(
                "breadth_normalize_window", 252
            ),
        ))
        pp.register_chain("sector", sector_chain)

    # ── Shape D: categorical chain ──
    concept_cfg = pp_cfg.get("concept", {})
    if concept_cfg.get("enabled", True):
        concept_chain = PreprocessingChain(name="concept")
        concept_chain.add(ConceptBlockEncoder(
            top_n=concept_cfg.get("top_n", 100),
            min_stocks_per_board=concept_cfg.get("min_stocks_per_board", 5),
        ))
        pp.register_chain("concept", concept_chain)

    return pp
