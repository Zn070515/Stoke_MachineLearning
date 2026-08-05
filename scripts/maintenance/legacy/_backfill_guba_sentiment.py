# ARCHIVED (maintenance/legacy): historical/one-off — NOT part of the canonical pipeline. See scripts/README.md.
"""GPU-accelerated Guba sentiment backfill — RTX 4090, CPU-friendly.

Optimization stack:
  L1: FP16 model → Tensor Cores (2× over FP32)
  L2: torch.inference_mode() → zero autograd overhead
  L3: Streaming tokenization → tokenize one batch at a time, CPU stays cool
  L4: Length-sorted batching → minimal padding waste (30-50% less compute)
  L5: Multi-stock grouping → amortize Python overhead
  L6: GPU-resident inference → no back-and-forth during forward passes

Key design: tokenizer runs per-batch (2k texts), not on the entire group.
This keeps CPU usage moderate and steady — 8 download shards still breathe.

Usage:
  PYTHONPATH=. ./.venv/Scripts/python scripts/maintenance/legacy/_backfill_guba_sentiment.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent


# ── Fast inference engine ─────────────────────────────────────────────

class FinBERTFast:
    """Direct FinBERT inference — FP16 model, streaming tokenization."""

    def __init__(self, analyzer):
        import torch
        pipe = analyzer._pipe
        self._tokenizer = pipe.tokenizer
        self._model = pipe.model.half().eval()

        id2label = self._model.config.id2label
        self._pos_idx = next(i for i, lbl in id2label.items() if lbl == "Positive")
        self._neg_idx = next(i for i, lbl in id2label.items() if lbl == "Negative")

        logger.info("FinBERTFast: FP16 on CUDA, pos=%d neg=%d",
                     self._pos_idx, self._neg_idx)

    def predict_streaming(
        self, texts: List[str], batch_size: int = 2048, max_len: int = 200,
    ) -> np.ndarray:
        """Length-sorted → stream-tokenize → GPU predict.

        Tokenizes ONE batch at a time (not all texts), keeping CPU
        usage low and steady.  Sorting by length first minimizes
        padding within each batch.
        """
        import torch

        n = len(texts)
        # Sort by char length → uniform batches → minimal padding
        lengths = np.array([len(t) if isinstance(t, str) else 0 for t in texts])
        order = np.argsort(lengths)

        sorted_scores = np.empty(n, dtype=np.float32)

        with torch.inference_mode():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                positions = order[start:end]
                batch = [texts[p][:300] if isinstance(texts[p], str) and texts[p] else ""
                         for p in positions]

                # Tokenize only this batch (CPU, low memory)
                tok = self._tokenizer(
                    batch, return_tensors="pt", padding=True,
                    truncation=True, max_length=max_len,
                )
                ids = tok["input_ids"].to("cuda")
                mask = tok["attention_mask"].to("cuda")

                # GPU forward (FP16, Tensor Cores)
                out = self._model(input_ids=ids, attention_mask=mask)
                probs = torch.softmax(out.logits.float(), dim=-1)
                batch_scores = (probs[:, self._pos_idx] - probs[:, self._neg_idx])
                sorted_scores[start:end] = batch_scores.cpu().numpy()

        # Unsort → original order
        scores = np.empty(n, dtype=np.float32)
        scores[order] = sorted_scores
        return scores


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GPU Guba sentiment backfill")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="GPU inference batch size (default: 2048)")
    parser.add_argument("--group-size", type=int, default=20,
                        help="Stocks per group (default: 20, higher = more GPU amortization)")
    parser.add_argument("--stocks", type=str, default=None,
                        help="Comma-separated stock codes (default: all with Bronze)")
    parser.add_argument("--force", action="store_true",
                        help="Recompute all stocks even if Gold is current")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT))
    from stoke_ml.config import load_config
    from stoke_ml.data.calendar import TradingCalendar
    from stoke_ml.data.guba_storage import GubaStorage
    from stoke_ml.features.news_nlp import NewsSentimentAnalyzer

    cfg = load_config()
    data_dir = cfg.project.data_dir
    raw_dir = Path(data_dir) / "a_shares" / "guba_raw"
    gold_dir = Path(data_dir) / "a_shares" / "guba_sentiment"

    # ── Load FinBERT ──
    logger.info("Loading FinBERT...")
    t0 = time.time()
    analyzer = NewsSentimentAnalyzer(force_lexicon=False)
    analyzer._ensure_loaded()
    if analyzer._device != "cuda":
        logger.error("FinBERT on CPU — CUDA not available.")
        sys.exit(1)
    logger.info("FinBERT on CUDA in %.1fs", time.time() - t0)

    engine = FinBERTFast(analyzer)
    del analyzer

    # Warm-up CUDA context
    import torch
    logger.info("GPU warm-up...")
    engine.predict_streaming(["测试"] * 64, batch_size=64)
    logger.info("Ready.")

    # ── Storage ──
    calendar = TradingCalendar("a_shares")
    guba_storage = GubaStorage(data_dir, calendar)

    # ── Discover stocks ──
    if args.stocks:
        codes = [c.strip() for c in args.stocks.split(",")]
    else:
        codes = sorted([
            f.stem for f in raw_dir.glob("*.parquet")
            if not f.name.startswith(".tmp_")
        ])

    if not codes:
        logger.error("No Bronze files found in %s", raw_dir)
        sys.exit(1)

    # ── Filter ──
    todo = []
    skipped = 0
    for code in codes:
        bronze_path = raw_dir / f"{code}.parquet"
        gold_path = gold_dir / f"{code}.parquet"
        if args.force:
            todo.append(code)
        elif not gold_path.exists():
            todo.append(code)
        elif bronze_path.stat().st_mtime > gold_path.stat().st_mtime + 5:
            todo.append(code)
        else:
            skipped += 1

    logger.info("Stocks: %d total, %d need sentiment, %d current (skipped)",
                len(codes), len(todo), skipped)

    if not todo:
        logger.info("Nothing to do.")
        return 0

    # ── Grouped processing ──
    success, fail = 0, 0
    total_posts = 0
    t_start = time.time()

    for g_start in range(0, len(todo), args.group_size):
        group_codes = todo[g_start:g_start + args.group_size]
        g_t0 = time.time()

        # ── Load Silver for group ──
        silvers: Dict[str, Tuple[pd.DataFrame, int, int]] = {}
        all_titles: List[str] = []

        for code in group_codes:
            gold_path_check = gold_dir / f"{code}.parquet"
            bronze_mtime = (raw_dir / f"{code}.parquet").stat().st_mtime
            if not args.force and gold_path_check.exists():
                if gold_path_check.stat().st_mtime >= bronze_mtime - 5:
                    continue

            try:
                silver = guba_storage.load_silver(code)
            except Exception:
                continue
            if silver.empty:
                continue

            silver = silver.drop(
                columns=["sentiment_title", "sentiment_body"], errors="ignore"
            )

            start = len(all_titles)
            all_titles.extend(silver["title"].tolist())
            end = len(all_titles)
            silvers[code] = (silver, start, end)

        if not all_titles:
            continue

        # ── GPU inference: streaming, one batch at a time ──
        all_scores = engine.predict_streaming(all_titles, args.batch_size)

        # ── Scatter → aggregate → save ──
        for code, (silver, start, end) in silvers.items():
            try:
                silver = silver.copy()
                silver["sentiment_title"] = all_scores[start:end].astype(np.float32)
                gold = guba_storage._silver_to_gold_legacy(silver, code)
                if gold.empty:
                    continue
                guba_storage.save_daily_sentiment(gold)
            except Exception as e:
                logger.error("%s: save failed: %s", code, e)
                fail += 1
                continue

            post_count = int(gold["guba_post_count"].sum())
            total_posts += post_count
            success += 1

        g_elapsed = time.time() - g_t0
        total_elapsed = time.time() - t_start
        logger.info(
            "[%d-%d/%d] %d stocks %d titles in %.1fs (%.1f stk/min) | total: %d ok %.1f stk/min",
            g_start + 1, min(g_start + args.group_size, len(todo)), len(todo),
            len(silvers), len(all_titles), g_elapsed,
            len(silvers) / g_elapsed * 60 if g_elapsed > 0 else 0,
            success, success / total_elapsed * 60 if total_elapsed > 0 else 0,
        )

    elapsed = time.time() - t_start
    logger.info(
        "Done: %d ok, %d fail, %d skip — %.0f posts in %.1f min (%.1f stk/min)",
        success, fail, skipped, total_posts, elapsed / 60,
        success / elapsed * 60 if elapsed > 0 else 0,
    )

    try:
        if torch.cuda.is_available():
            logger.info("GPU memory: %.2f GB alloc / %.2f GB reserved",
                        torch.cuda.memory_allocated() / 1024**3,
                        torch.cuda.memory_reserved() / 1024**3)
    except Exception:
        pass

    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
