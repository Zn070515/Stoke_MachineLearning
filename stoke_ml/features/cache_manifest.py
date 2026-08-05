"""Feature-cache sidecar manifests.

Each prebuilt feature parquet carries a JSON manifest recording the code
version, config signature, feature schema, and per-channel source file
fingerprints.  Cache hits compare these hashes — not just file size — so a
config / code / data change invalidates a stale feature instead of silently
reusing it during training.
"""
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
]

# config.yaml sections that change prebuilt feature VALUES.  Hashed verbatim
# by config_hash() so a same-commit config.yaml edit (technical windows,
# missing-value handling, thresholds, cross-sectional normalization params,
# source effective-date / persistence strategy, universe gates) invalidates
# the feature cache — the git-commit field alone cannot see it (§十一-3).
_CONFIG_SECTIONS = ["features", "preprocessing", "universe", "fundamental"]


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
    except Exception:
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
    """
    try:
        from stoke_ml.config import load_config
        return config_hash(config_snapshot(load_config()))
    except Exception:
        return None


def git_head() -> str:
    """Current git HEAD, or 'unknown' when not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception as exc:
        logger.warning("git rev-parse failed (category=%s)", classify_error(exc).value)
        return "unknown"


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
    except Exception:
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


def schema_hash(path: str) -> str:
    """Hash of a parquet's column set (names + types). Reads only the header."""
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        cols = [(f.name, str(f.type)) for f in schema]
        return _sha1(json.dumps(cols, sort_keys=True))
    except Exception as exc:
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


def source_paths(data_dir: str, code: str) -> dict[str, str]:
    """Canonical per-channel source file path for a stock."""
    a_shares = os.path.join(data_dir, "a_shares")
    out = {}
    for name, sub in SOURCE_SUBDIRS.items():
        out[name] = os.path.join(a_shares, *sub, f"{code}.parquet")
    for name, rel in SNAPSHOT_FILES.items():
        out[name] = os.path.join(a_shares, *rel)
    return out


def channels_and_source_files(
    data_dir: str, code: str, start: str, end: str,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Per-channel status + per-channel fingerprint for a fresh manifest."""
    channels, source_files = {}, {}
    for name, path in source_paths(data_dir, code).items():
        fp = file_fingerprint(path)
        source_files[name] = {"hash": fp}
        channels[name] = "complete" if fp is not None else "missing_optional"
    source_files["daily"]["range"] = [start, end]
    channels["daily"] = "complete"  # a stock only builds when daily loaded
    return channels, source_files


def write_manifest(payload: dict, manifest_path: str) -> None:
    """Atomically write a manifest (tmp + os.replace)."""
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    tmp = f"{manifest_path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, manifest_path)
    except Exception:
        if os.path.isfile(tmp):
            os.unlink(tmp)
        raise


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
        "config_hash": cfg_hash,
        "feature_schema_hash": schema_hash(output_path),
        "horizon": config.get("horizon"),
        "seq_len": config.get("seq_len"),
        "panel_mode": config.get("panel_mode"),
        "source_files": source_files,
        "channels": channels,
    }


def manifest_matches(
    manifest_path: str, code: str, config: dict, output_path: str,
    data_dir: str, commit: str, cfg_hash: str,
) -> bool:
    """Cache hit iff every recorded hash still matches the current inputs."""
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)
    except Exception as exc:
        logger.warning(
            "manifest %s unreadable (category=%s), treating as stale",
            manifest_path, classify_error(exc).value,
        )
        return False
    if not all([
        m.get("stock_code") == code,
        m.get("config_hash") == cfg_hash,
        m.get("git_commit") == commit,
        m.get("feature_schema_hash") == schema_hash(output_path),
        m.get("horizon") == config.get("horizon"),
        m.get("seq_len") == config.get("seq_len"),
    ]):
        return False
    # start/end are NOT part of config_hash (build and training resolve them
    # differently) — they are recorded as the daily source range and compared
    # here so a date-window change still invalidates the cache (§十一-3).
    rng = (m.get("source_files") or {}).get("daily", {}).get("range")
    if rng != [config.get("start"), config.get("end")]:
        return False
    for name, rec in (m.get("source_files") or {}).items():
        path = source_paths(data_dir, code).get(name)
        if path is None:
            continue
        if rec.get("hash") != file_fingerprint(path):
            return False
    return True
