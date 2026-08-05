"""Fit + persist the TopicModeler on a pinned production corpus.

Production step (§十-2): builds the topic model on silver text up to a pinned
TRAIN_END cutoff, saves the BERTopic model + a corpus content hash + the
cutoff to the model cache dir, and writes a run manifest.  The news/guba
storage paths then call ``TopicModeler.transform(source=...)`` which
auto-loads this pinned artifact via ``preprocessing.text.topic_model.corpus_cutoff``.

Usage:
    PYTHONPATH=. ./.venv/Scripts/python scripts/production/fit_topic_model.py \
        --source news --cutoff 2025-12-31 --stocks 200

Exits non-zero when the fit fails (no posts, deps missing, model disabled), so
a training pipeline treats a missing topic model as an error rather than
silently producing data without topic features.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from stoke_ml.config import load_config
from stoke_ml.data.calendar import get_research_calendar
from stoke_ml.data.guba_storage import GubaStorage
from stoke_ml.data.news_storage import NewsStorage
from stoke_ml.preprocessing.pipeline import PreprocessingPipeline
from stoke_ml.preprocessing.text.topics import dependency_versions

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def _discover_stocks(data_dir: str, source: str) -> list[str]:
    """List stocks that have silver text under the source's silver dir."""
    sub = "news_silver" if source == "news" else "guba_silver"
    silver_dir = os.path.join(data_dir, "a_shares", sub)
    if not os.path.isdir(silver_dir):
        logger.error("Silver directory not found: %s", silver_dir)
        return []
    return sorted(
        f.replace(".parquet", "")
        for f in os.listdir(silver_dir)
        if f.endswith(".parquet")
    )


def _collect_silver(data_dir: str, source: str, codes: list[str], cutoff: str):
    """Load + combine silver text up to *cutoff* (PIT) for the given stocks."""
    calendar = get_research_calendar(strict=True)
    frames = []
    for code in codes:
        try:
            if source == "news":
                storage = NewsStorage(data_dir, calendar)
                df = storage.load_silver_news(code)
            else:
                storage = GubaStorage(data_dir, calendar)
                df = storage.load_silver(code)
            if df is not None and not df.empty:
                if "aligned_date" in df.columns:
                    df["aligned_date"] = pd.to_datetime(df["aligned_date"])
                    if cutoff is not None:
                        df = df[df["aligned_date"] <= pd.Timestamp(cutoff)]
                frames.append(df.assign(stock_code=code))
        except Exception as exc:
            logger.warning("Failed to load %s for %s: %s", source, code, exc)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _corpus_file_info(data_dir: str, source: str, codes: list[str]) -> tuple:
    """SHA-256 per contributing silver parquet + a combined corpus-file hash.

    Returns ``(entries, combined_hash)`` where each entry is
    ``{"file": abs_path, "sha256": hex}``.  Missing/unreadable files are
    recorded as ``"unreadable"`` — they contributed no rows, so they are
    excluded from the combined digest (which feeds the manifest's
    ``corpus_file_hash``, §十三).
    """
    sub = "news_silver" if source == "news" else "guba_silver"
    entries = []
    for code in codes:
        path = os.path.join(data_dir, "a_shares", sub, f"{code}.parquet")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "rb") as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            digest = "unreadable"
        entries.append({"file": path, "sha256": digest})
    combined = hashlib.sha256()
    for e in entries:
        combined.update(e["file"].encode("utf-8"))
        combined.update(b"\n")
        combined.update(e["sha256"].encode("utf-8"))
        combined.update(b"\n")
    return entries, combined.hexdigest()


def _sha256_file(path: str) -> str:
    """Full SHA-256 of a file, or ``"unknown"`` when missing/unreadable."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "unknown"


def _config_hash(cfg) -> str:
    """SHA-1 of the resolved preprocessing config used for the fit (§十三)."""
    try:
        from omegaconf import OmegaConf
        return hashlib.sha1(
            OmegaConf.to_yaml(cfg.preprocessing).encode("utf-8")
        ).hexdigest()[:16]
    except Exception:
        return "unknown"


def _calendar_hash(data_dir: str) -> str:
    """SHA-1 of the research calendar identity used to map PIT dates (§十三).

    Uses the same factory the fit's silver load goes through
    (``get_research_calendar``) and hashes market + calendar version +
    verified-until, so a calendar change is visible in the fit provenance.
    """
    try:
        cal = get_research_calendar(
            market="a_shares", strict=False, data_dir=data_dir,
        )
        ident = f"{cal.market}|{cal.CALENDAR_VERSION}|{cal.verified_until}"
        return hashlib.sha1(ident.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Fit + persist the TopicModeler (§十-2)"
    )
    parser.add_argument("--source", choices=["news", "guba"], default="news")
    parser.add_argument(
        "--cutoff", type=str, required=True,
        help="Pinned training cutoff (YYYY-MM-DD, TRAIN_END).  Only posts with "
             "aligned_date <= cutoff enter the topic representation space.",
    )
    parser.add_argument(
        "--stocks", type=int, default=None,
        help="Limit to N stocks (default: all stocks with silver data).",
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Ignore a valid cached model (same cutoff + corpus) and retrain.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the UMAP step (deterministic reproducibility, "
             "recorded in the manifest §十三).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = cfg.project.data_dir

    all_codes = _discover_stocks(data_dir, args.source)
    if not all_codes:
        logger.error("No %s silver data found under %s", args.source, data_dir)
        sys.exit(1)
    if args.stocks is not None and args.stocks <= 0:
        parser.error("--stocks must be a positive integer (or omitted for all)")
    codes = all_codes
    if args.stocks is not None and args.stocks < len(all_codes):
        step = max(len(all_codes) // args.stocks, 1)
        codes = [all_codes[i * step] for i in range(args.stocks)]
    logger.info(
        "Source: %s, cutoff: %s, %d/%d stocks",
        args.source, args.cutoff, len(codes), len(all_codes),
    )

    all_silver = _collect_silver(data_dir, args.source, codes, args.cutoff)
    if all_silver.empty:
        logger.error("No %s posts up to %s", args.source, args.cutoff)
        sys.exit(1)
    logger.info(
        "Loaded %d posts from %d stocks",
        len(all_silver), all_silver["stock_code"].nunique(),
    )

    pp_cfg = cfg.get("preprocessing", {})
    pp = PreprocessingPipeline.from_config(pp_cfg)
    tm = pp.topic_modeler
    if tm is None:
        logger.error("preprocessing.text.topic_model.enabled is false in config")
        sys.exit(1)

    clean = pp.run("text_pre", all_silver)
    tm.fit(
        clean,
        source=args.source,
        corpus_cutoff=args.cutoff,
        force_retrain=args.force_retrain,
        seed=args.seed,
        stock_codes=codes,
    )

    if tm._model is None:
        logger.error(
            "TopicModeler fit FAILED — no model produced.  Check that "
            "BERTopic/UMAP/HDBSCAN are installed and the corpus meets "
            "min_topic_size."
        )
        sys.exit(1)

    try:
        n_topics = len(tm._model.get_topic_info())
    except Exception:
        n_topics = "unknown"
    logger.info(
        "Topic model ready: %s topics, corpus_hash=%s, cutoff=%s",
        n_topics, tm.corpus_hash_, args.cutoff,
    )
    logger.info(
        "Production transform() auto-loads this artifact only after you set "
        "preprocessing.text.topic_model.corpus_cutoff=%s in config.yaml "
        "(§十-2).",
        args.cutoff,
    )

    # §十三 provenance: every reproducibility-relevant fact about this fit is
    # pinned in the manifest.  All keys are ADDITIVE — old manifests remain
    # valid, and readers that only know the original fields keep working.
    corpus_key = tm._corpus_key(args.cutoff)
    cache_path = os.path.join(
        tm.model_cache_dir, f"bertopic_{args.source}_cutoff_{corpus_key}.pkl"
    )
    corpus_files, corpus_file_hash = _corpus_file_info(
        data_dir, args.source, codes,
    )
    versions = dependency_versions()
    if args.stocks is not None and args.stocks < len(all_codes):
        sampling_method = "deterministic_equidistant_sorted"
    else:
        sampling_method = "all_stocks"
    embedding_model_used = tm._read_meta(
        args.source, corpus_key
    ).get("embedding_model", tm.embedding_model)

    manifest_dir = os.path.join(cfg.project.model_dir, "topic")
    os.makedirs(manifest_dir, exist_ok=True)
    manifest_path = os.path.join(manifest_dir, f"topic_fit_{args.source}.json")
    manifest = {
        "source": args.source,
        "corpus_cutoff": args.cutoff,
        "corpus_hash": tm.corpus_hash_,
        "n_topics": n_topics,
        "n_posts": len(all_silver),
        "fit_date": datetime.now().isoformat(),
        # §十三 additions
        "model_pickle_sha256": _sha256_file(cache_path),
        "corpus_files": corpus_files,
        "corpus_file_hash": corpus_file_hash,
        "embedding_model": tm.embedding_model,
        "embedding_model_used": embedding_model_used,
        "min_topic_size": tm.min_topic_size,
        "seed": args.seed,
        "bertopic_version": versions["bertopic"],
        "sentence_transformers_version": versions["sentence-transformers"],
        "python_version": versions["python"],
        "sampling_method": sampling_method,
        "stock_codes": list(codes),
        "config_hash": _config_hash(cfg),
        "calendar_hash": _calendar_hash(data_dir),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    logger.info("Run manifest written to %s", manifest_path)


if __name__ == "__main__":
    main()
