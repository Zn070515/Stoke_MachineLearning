"""Feature-cache sidecar manifests.

Each prebuilt feature parquet carries a JSON manifest recording the code
version, config signature, feature schema, and per-channel source file
fingerprints.  Cache hits compare these hashes — not just file size — so a
config / code / data change invalidates a stale feature instead of silently
reusing it during training.
"""
import functools
import hashlib
import json
import logging
import os
import subprocess
from datetime import datetime, timezone

from stoke_ml.utils.error_summary import classify_error

logger = logging.getLogger(__name__)

# Per-stock channel directories under data/a_shares/{subdir}/{code}.parquet.
SOURCE_SUBDIRS = {
    "daily": ("daily",),
    "sentiment": ("sentiment",),
    "guba_sentiment": ("guba_sentiment",),
    "comment_sentiment": ("comment_sentiment",),
    "announcements_sentiment": ("announcements", "sentiment"),
    "fundamentals": ("fundamentals",),
    "margin": ("margin",),
    "northbound": ("northbound",),
    "dragon_tiger": ("dragon_tiger",),
    "valuation": ("valuation",),
    "capital_flow": ("capital_flow_processed",),
    "board": ("board_processed",),
    "sector": ("industry_ranking_processed",),
    "block_trade": ("block_trade_processed",),
    "dividend": ("dividend_processed",),
    "lockup": ("lockup_processed",),
    "shareholder": ("shareholder_processed",),
    "concept": ("concept_blocks_processed",),
    "pledge": ("pledge_processed",),
    "index_membership": ("index_membership_processed",),
}

# Market-wide snapshot files shared by every stock.
SNAPSHOT_FILES = {
    "earnings_forecasts": ("earnings", "forecasts.parquet"),
    "earnings_express": ("earnings", "express.parquet"),
}

# Market-wide shared inputs (relative to data_dir) that CHANGE FEATURE VALUES
# but are not per-stock files (§十-1).  Macro rates, market-environment breadth,
# industry cross-sectional returns, the stock→sector mapper (which sector ETF
# flow a stock reads), the trading calendar (date alignment) and the whole
# sector-ETF-flow directory must each invalidate a prebuilt feature when they
# change.  Without them a stale macro/calendar/mapper would be silently reused.
# Panel-composition artifacts (universe/ipo.parquet, universe/delisted.parquet,
# index_constituents_hist/membership.parquet) are deliberately NOT here: they
# change which stocks/dates enter the training panel, not per-stock feature
# values, so they are covered by the fold-level universe/membership hash
# (train_panel FoldResearchContext), not the per-stock feature manifest.
SHARED_FILES = {
    # P1-10: the market-env lineage path must match where the channel is actually
    # READ/WRITTEN — aux_aligner._merge_market_env reads
    # a_shares/market_breadth/market_env_daily.parquet (written by
    # _preprocess_market_env.py).  A path here that misses `market_breadth` would
    # fingerprint None forever, so a market-env change never invalidated a stale
    # feature cache.
    "market_env": ("a_shares", "market_breadth", "market_env_daily.parquet"),
    "industry": ("a_shares", "industry", "industry_returns.parquet"),
    "sector_mapper": ("a_shares", "stock_sector_cache.csv"),
    "calendar": ("exchange_calendar", "a_shares.parquet"),
}

# Shared DIRECTORY inputs — hashed as a whole (any file change invalidates).
# macro lives under a generation root (a_shares/macro/macro_daily_gen/) since
# §十三-2 — the legacy flat file froze, so fingerprinting the directory (not the
# flat file) is what keeps a macro update from silently reusing a stale cache.
SHARED_DIRS = {
    "etf_flow": ("a_shares", "etf_flow"),
    "macro": ("a_shares", "macro", "macro_daily_gen"),
}

_SHARED_NAMES = frozenset((*SHARED_FILES, *SHARED_DIRS))

# Build-time config keys whose values change the feature output (flat form).
# The production path hashes a full config snapshot instead (config_snapshot),
# so this whitelist only backs the legacy flat-dict form (unit tests / callers
# that carry just the pipeline flags).
_CONFIG_KEYS = [
    "seq_len", "flat_seq_len", "horizon",
    "use_technical", "use_scoring", "use_temporal",
    "use_sentiment", "use_announcements", "use_guba", "use_comment",
    "use_margin", "use_northbound", "use_dragon_tiger",
    "use_fundamental", "use_earnings", "use_valuation", "use_etf_flow",
    "use_capital_flow", "use_block_trade", "use_shareholder", "use_lockup",
    "use_dividend", "use_board", "use_sector", "use_concept",
    "use_macro", "use_industry", "use_limit_up", "use_pledge",
    "use_market_env", "use_market_env_refine", "use_index_membership",
    "use_interaction", "use_feature_selection", "feature_selection_k",
    "use_new_preprocessing", "use_emotion_refine", "use_fundamental_refine",
    "use_temporal_stats", "drop_dead_features", "min_history",
    "panel_mode", "start", "end",
    # §七: topic_* features are OFF by default (headline representation
    # leakage from the global_frozen topic model); flipping use_topic changes
    # the feature COLUMN set, so the flat-dict config hash must track it too
    # (production config_hash already covers topic enablement via the
    # preprocessing section; schema_hash catches the column-set change).
    "use_topic",
]

# config.yaml sections that change prebuilt feature VALUES.  Hashed verbatim
# by config_hash() so a same-commit config.yaml edit (technical windows,
# missing-value handling, thresholds, cross-sectional normalization params,
# source effective-date / persistence strategy, universe gates) invalidates
# the feature cache — the git-commit field alone cannot see it (§十一-3).
_CONFIG_SECTIONS = ["features", "preprocessing", "universe", "fundamental"]

# Package subdirectories whose source CODE determines feature VALUES.  When git
# is unavailable git_commit degrades to 'unknown' and code drift would go
# undetected (§十-2) — feature_code_tree_hash() hashes these trees as a
# fallback.  Model-layer code is deliberately excluded: it consumes features but
# does not change them, so hashing it would invalidate the cache on every model
# edit.
_FEATURE_CODE_DIRS = ("features", "preprocessing")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _stable_dumps(obj) -> str:
    """Deterministic JSON for hashing: sorted keys, compact separators."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
    )


def _to_plain(value):
    """Recursively normalize an OmegaConf node / container to plain Python."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        # OmegaConf is optional here: when it is not importable, the pure-Python
        # recursion below still normalizes dict/list/tuple, and any leftover
        # scalar is stringified by _stable_dumps' default=str — a correct,
        # deterministic fallback.  A genuine conversion failure is NOT swallowed;
        # it propagates so the hashing layer cannot silently degrade (§二十一).
        pass
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def config_snapshot(cfg) -> dict:
    """Feature-affecting config sections, normalized to plain containers.

    Covers technical-indicator switches, missing-value handling, thresholds,
    cross-sectional normalization params and the source effective-date /
    persistence strategy (§十一-3), plus the universe gates that
    build_panel_features reads at runtime.  ``start``/``end`` and
    ``panel_mode`` are deliberately excluded: they are resolved differently
    by the build and training processes, so they are captured by the
    manifest's source-file range and the feature schema hash instead.
    """
    snap = {}
    for name in _CONFIG_SECTIONS:
        raw = cfg.get(name) if isinstance(cfg, dict) else getattr(cfg, name, None)
        snap[name] = _to_plain(raw) if raw is not None else {}
    return snap


def current_config_hash() -> str | None:
    """config_hash of the active config.yaml's feature-affecting sections.

    Returns None (callers then skip the comparison) if config cannot load.
    Only the config LOAD degrades to None — a missing / malformed config is an
    environment condition.  Any programming error inside ``config_snapshot`` /
    ``config_hash`` propagates instead of being swallowed (§二十一).
    """
    try:
        from stoke_ml.config import load_config
        cfg = load_config()
    except Exception:
        return None
    return config_hash(config_snapshot(cfg))


def git_head() -> str:
    """Current git HEAD, or 'unknown' when not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError) as exc:
        # Not a git repo (git rev-parse fails) or git unavailable (OSError):
        # the documented non-repo degradation.  Any other exception propagates.
        logger.warning("git rev-parse failed (category=%s)", classify_error(exc).value)
        return "unknown"


@functools.lru_cache(maxsize=1)
def feature_code_tree_hash() -> str:
    """Content hash of the feature/preprocessing source tree; 'unknown' if absent.

    When git is unavailable (ZIP distribution, non-git checkout) git_commit
    degrades to 'unknown' and two code versions become indistinguishable — a
    stale feature cache would then survive a code change (§十-2).  Hashing the
    feature code tree's bytes restores that distinction: any edit to a feature /
    preprocessing module changes the hash and invalidates the cache.  Memoized
    because build pipelines call it once per stock and the tree cannot change
    mid-process.
    """
    pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    entries: dict[str, str] = {}
    for name in _FEATURE_CODE_DIRS:
        base = os.path.join(pkg_root, name)
        if not os.path.isdir(base):
            continue
        for root, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(root, fn)
                digest = _content_hash(path)
                if digest is None:
                    continue
                entries[os.path.relpath(path, pkg_root)] = digest
    if not entries:
        return "unknown"
    joined = "\n".join(f"{p}:{h}" for p, h in sorted(entries.items()))
    return _sha1(joined)


def _upstream_manifest_hash(path: str) -> str | None:
    """Content checksum recorded in an upstream storage sidecar manifest.

    The daily K-line storage already writes ``daily/{code}.manifest.json`` whose
    ``schema_hash`` is a value-level content checksum (storage.py ``_schema_hash``)
    — re-reading that sidecar is cheaper and more robust than re-hashing the
    parquet bytes.  Any channel that adds a content-checksummed sidecar at
    ``{stem}.manifest.json`` is picked up the same way.
    """
    sidecar = os.path.splitext(path)[0] + ".manifest.json"
    if not os.path.isfile(sidecar):
        return None
    try:
        with open(sidecar, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as exc:
        # The sidecar EXISTS but cannot be read / parsed — deliberately distinct
        # from "no sidecar" above.  Not fatal: the caller falls back to hashing
        # the parquet bytes (still a correct fingerprint), but a corrupt storage
        # sidecar is a real anomaly, so log it instead of swallowing it.
        logger.warning(
            "upstream manifest %s unreadable (category=%s), falling back to "
            "byte hash", sidecar, classify_error(exc).value,
        )
        return None
    sh = m.get("schema_hash")
    return sh if isinstance(sh, str) and sh else None


def _content_hash(path: str) -> str | None:
    """Streaming SHA-256 of a file's bytes; None if unreadable."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _dir_content_hash(path: str) -> str | None:
    """Content digest of every file under ``path`` (relative names + bytes).

    Used for shared DIRECTORY inputs (e.g. sector ETF flows) where a change to
    ANY file must invalidate the feature cache.  Returns None only when nothing
    readable is under the directory.
    """
    entries: dict[str, str] = {}
    for root, _dirs, files in os.walk(path):
        for fn in files:
            full = os.path.join(root, fn)
            digest = _content_hash(full)
            if digest is None:
                continue
            entries[os.path.relpath(full, path)] = digest
    if not entries:
        return None
    joined = "\n".join(f"{p}:{h}" for p, h in sorted(entries.items()))
    return _sha1(joined)


def file_fingerprint(path: str) -> str | None:
    """Content-based fingerprint of a source file; None if the file is absent.

    Prefers the upstream storage sidecar manifest's content checksum (daily
    K-line), falling back to streaming the file bytes through SHA-256.  Either
    way the fingerprint changes when the *content* changes — a same-size
    replacement, a preserved-mtime copy, or a filesystem sync cannot slip
    through the way the old size+mtime proxy could (§十一-2).
    """
    if not os.path.isfile(path):
        return None
    upstream = _upstream_manifest_hash(path)
    if upstream is not None:
        return _sha1(f"upstream:{upstream}")
    return _content_hash(path)


@functools.lru_cache(maxsize=128)
def _shared_fingerprint(path: str) -> str | None:
    """Fingerprint of a market-wide shared input, memoized by path.

    Shared inputs are byte-identical for every stock, so re-hashing one per
    stock would cost O(#stocks × file_size).  Memoizing by path hashes each
    shared input exactly once per process.  The tradeoff (accepted, as for
    ``feature_code_tree_hash``) is that a mid-process data edit is not re-read
    — shared data does not change during a build run.
    """
    if os.path.isdir(path):
        return _dir_content_hash(path)
    return file_fingerprint(path)


def _input_fingerprint(name: str, path: str) -> str | None:
    """Fingerprint for one lineage entry: shared market-wide inputs memoized."""
    if name in _SHARED_NAMES:
        return _shared_fingerprint(path)
    return file_fingerprint(path)


def schema_hash(path: str) -> str:
    """Hash of a parquet's column set (names + types). Reads only the header.

    The ``except`` is a narrow whitelist (``KeyError`` / ``OSError`` /
    ``ValueError``) so an unreadable path degrades to ``"unknown"`` without
    crashing manifest writes, while ANY other exception — a corrupt schema or a
    pyarrow-level bug — propagates instead of being swallowed (§二十一 D2).  The
    old ``except Exception`` masked genuine schema-read bugs behind ``"unknown"``
    and let a stale cache survive.
    """
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        cols = [(f.name, str(f.type)) for f in schema]
        return _sha1(json.dumps(cols, sort_keys=True))
    except (KeyError, OSError, ValueError) as exc:
        logger.warning(
            "schema_hash failed for %s (category=%s)", path, classify_error(exc).value,
        )
        return "unknown"


def config_hash(config: dict) -> str:
    """Signature of the config a feature's values depend on.

    A full config snapshot (see ``config_snapshot``) is hashed verbatim — the
    production path — so every feature-affecting config.yaml value is covered.
    A flat legacy dict (unit tests / callers that only carry pipeline flags)
    is filtered through ``_CONFIG_KEYS`` instead.
    """
    if isinstance(config.get("features"), dict) and isinstance(
        config.get("preprocessing"), dict
    ):
        return _sha1(_stable_dumps(config))
    sig = json.dumps({k: config.get(k) for k in _CONFIG_KEYS}, sort_keys=True)
    return _sha1(sig)


def shared_inputs_hash(data_dir: str) -> str:
    """Aggregate fingerprint of every market-wide shared input.

    §十二.3/§P1-1: the per-channel ``source_files`` entries already fingerprint
    each shared input individually; this aggregate is a lineage-DEFINITION
    guard.  If the SHARED_FILES/SHARED_DIRS schema later grows (a new shared
    input that changes feature values), the aggregate differs from older
    manifests and the cache rebuilds with complete lineage instead of silently
    trusting a manifest written before that input existed.  Feature-selection
    is deliberately not a shared input here: it is derived deterministically
    from the source data + config, so the config hash covers it.
    """
    entries: dict[str, str | None] = {}
    for name in sorted((*SHARED_FILES, *SHARED_DIRS)):
        rel = SHARED_FILES.get(name) or SHARED_DIRS.get(name)
        entries[name] = _shared_fingerprint(os.path.join(data_dir, *rel))
    return _sha1(_stable_dumps(entries))


def source_paths(data_dir: str, code: str) -> dict[str, str]:
    """Canonical per-channel source file path for a stock.

    Includes per-stock channel files, market-wide snapshot files and the
    shared market-wide inputs (§十-1) — the latter are identical for every
    code but must still be fingerprinted so a macro / market-env / industry /
    sector-mapper / calendar / ETF-flow change invalidates a stale feature.
    """
    a_shares = os.path.join(data_dir, "a_shares")
    out = {}
    for name, sub in SOURCE_SUBDIRS.items():
        out[name] = os.path.join(a_shares, *sub, f"{code}.parquet")
    for name, rel in SNAPSHOT_FILES.items():
        out[name] = os.path.join(a_shares, *rel)
    for name, rel in SHARED_FILES.items():
        out[name] = os.path.join(data_dir, *rel)
    for name, rel in SHARED_DIRS.items():
        out[name] = os.path.join(data_dir, *rel)
    return out


def channels_and_source_files(
    data_dir: str, code: str, start: str, end: str,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Per-channel status + per-channel fingerprint for a fresh manifest."""
    channels, source_files = {}, {}
    for name, path in source_paths(data_dir, code).items():
        fp = _input_fingerprint(name, path)
        source_files[name] = {"hash": fp}
        channels[name] = "complete" if fp is not None else "missing_optional"
    source_files["daily"]["range"] = [start, end]
    channels["daily"] = "complete"  # a stock only builds when daily loaded
    return channels, source_files


def write_manifest(payload: dict, manifest_path: str) -> None:
    """Atomically write a manifest (tmp + os.replace).

    Any write failure propagates after the temp file is cleaned up — the
    ``finally`` guarantees cleanup while preserving the original exception,
    without a catch-and-reraise wrapper (§二十一).
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp = f"{manifest_path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, manifest_path)
    finally:
        if os.path.isfile(tmp):
            os.unlink(tmp)


def make_manifest(
    code: str, config: dict, output_path: str, data_dir: str,
    commit: str, cfg_hash: str,
) -> dict:
    """Assemble the sidecar manifest for a freshly built feature file."""
    channels, source_files = channels_and_source_files(
        data_dir, code, config.get("start"), config.get("end")
    )
    return {
        "stock_code": code,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "feature_code_tree_hash": feature_code_tree_hash(),
        "config_hash": cfg_hash,
        "feature_schema_hash": schema_hash(output_path),
        "horizon": config.get("horizon"),
        "seq_len": config.get("seq_len"),
        "panel_mode": config.get("panel_mode"),
        "source_files": source_files,
        "channels": channels,
        # §十二.3/§P1-1: aggregate over all market-wide shared inputs — a
        # lineage-definition guard (see shared_inputs_hash).
        "shared_inputs_hash": shared_inputs_hash(data_dir),
    }


def manifest_matches(
    manifest_path: str, code: str, config: dict, output_path: str,
    data_dir: str, commit: str, cfg_hash: str,
) -> bool:
    """Cache hit iff every recorded hash still matches the current inputs.

    Thin bool wrapper over :func:`manifest_matches_detailed` for callers that
    only need a yes/no (e.g. ``build_features.py`` cache-hit probe); the
    detailed variant additionally reports WHICH lineage entry went stale.
    """
    return manifest_matches_detailed(
        manifest_path, code, config, output_path, data_dir, commit, cfg_hash,
    )[0]


def manifest_matches_detailed(
    manifest_path: str, code: str, config: dict, output_path: str,
    data_dir: str, commit: str, cfg_hash: str,
) -> tuple[bool, list[str]]:
    """Cache-hit check returning ``(matches, failure_reasons)``.

    Identical semantics to :func:`manifest_matches` but does NOT short-circuit:
    every recorded check is evaluated so the caller can report the structured
    reason each stale feature failed.  ``matches`` is True iff ``reasons`` is
    empty.  Reason vocabulary (stable identifiers for triage / dashboards):

      * ``code_changed``            — git commit, feature code-tree hash, or
                                      stock_code disagree (the code-tree hash is
                                      ALWAYS compared, git or not, §十二.2).
      * ``config_changed``          — recorded config_hash / horizon / seq_len
                                      disagree with the current config.
      * ``schema_changed``          — the feature parquet's column set drifted
                                      from what the manifest recorded.
      * ``source_changed``          — a per-stock source channel (daily,
                                      sentiment, margin, ...) fingerprint changed.
      * ``shared_input_changed``    — a market-wide shared input (macro,
                                      industry, sector_mapper, etf_flow dir, or
                                      the shared-inputs aggregate) changed, or a
                                      manifest predates shared-input lineage.
      * ``calendar_changed``        — the exchange-calendar artifact changed.
      * ``range_changed``           — the recorded daily source date-window no
                                      longer matches the requested start/end.
      * ``manifest_missing`` / ``manifest_unreadable`` — no usable sidecar.

    ``cfg_hash`` may be None (config could not load); the config_hash comparison
    is then skipped, mirroring the pre-existing warn path.
    """
    reasons: list[str] = []
    if not os.path.isfile(manifest_path):
        return False, ["manifest_missing"]
    try:
        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError) as exc:
        # An unreadable / malformed manifest is treated as stale (rebuild),
        # never trusted — the safe direction for a disposable cache artifact.
        logger.warning(
            "manifest %s unreadable (category=%s), treating as stale",
            manifest_path, classify_error(exc).value,
        )
        return False, ["manifest_unreadable"]

    if m.get("stock_code") != code:
        reasons.append("code_changed")
    if cfg_hash is not None and m.get("config_hash") != cfg_hash:
        reasons.append("config_changed")
    if m.get("git_commit") != commit:
        reasons.append("code_changed")
    if m.get("feature_schema_hash") != schema_hash(output_path):
        reasons.append("schema_changed")
    if m.get("horizon") != config.get("horizon"):
        reasons.append("config_changed")
    if m.get("seq_len") != config.get("seq_len"):
        reasons.append("config_changed")
    # §十-1: a manifest written before shared inputs were fingerprinted cannot
    # vouch for them — rebuild to record complete lineage.  Without this, a
    # macro / calendar / ETF-flow change would silently keep old features valid.
    missing_shared = _SHARED_NAMES - set((m.get("source_files") or {}).keys())
    if missing_shared:
        reasons.append("shared_input_changed")
    # §十二.2: the feature code-tree hash is ALWAYS compared, git or not.  In a
    # repo, git_commit can match while uncommitted source edits change the
    # feature code — trusting the commit alone would let a stale cache survive;
    # outside a repo both sides record git_commit=unknown and only this hash
    # distinguishes versions.  Both git_commit AND the code-tree hash must match
    # for reuse; a manifest that cannot vouch for its code provenance (missing /
    # None field) is treated as stale rather than trusted.
    if m.get("feature_code_tree_hash") != feature_code_tree_hash():
        reasons.append("code_changed")
    # §十二.3/§P1-1: the shared-inputs aggregate must match too.  A manifest
    # written before a shared input existed (or before this field was recorded)
    # cannot vouch for it — rebuild to record complete lineage.
    if m.get("shared_inputs_hash") != shared_inputs_hash(data_dir):
        reasons.append("shared_input_changed")
    # start/end are NOT part of config_hash (build and training resolve them
    # differently) — they are recorded as the daily source range and compared
    # here so a date-window change still invalidates the cache (§十一-3).
    rng = (m.get("source_files") or {}).get("daily", {}).get("range")
    if rng != [config.get("start"), config.get("end")]:
        reasons.append("range_changed")
    for name, rec in (m.get("source_files") or {}).items():
        path = source_paths(data_dir, code).get(name)
        if path is None:
            continue
        if rec.get("hash") != _input_fingerprint(name, path):
            if name == "calendar":
                reasons.append("calendar_changed")
            elif name in _SHARED_NAMES:
                reasons.append("shared_input_changed")
            else:
                reasons.append("source_changed")
    return not reasons, reasons
