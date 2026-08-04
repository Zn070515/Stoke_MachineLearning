"""Feature-cache sidecar manifests (v6 §十二).

Each prebuilt feature parquet carries a JSON manifest recording the code
version, config signature, feature schema, and per-channel source file
fingerprints.  Cache hits compare these hashes — not just file size — so a
config / code / data change invalidates a stale feature instead of silently
reusing it during training.
"""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

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

# Build-time config keys whose values change the feature output.
_CONFIG_KEYS = [
    "seq_len", "horizon",
    "use_technical", "use_scoring", "use_temporal",
    "use_sentiment", "use_guba", "use_comment", "use_limit_up",
    "use_pledge", "use_market_env", "use_market_env_refine",
    "use_index_membership", "panel_mode", "start", "end",
]


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def git_head() -> str:
    """Current git HEAD, or 'unknown' when not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def file_fingerprint(path: str) -> str | None:
    """Cheap content fingerprint (size + mtime_ns); None if the file is absent.

    A make-style proxy for content hashing: re-downloading or appending to a
    source parquet changes its size/mtime, so any stale feature built from it
    is invalidated at the next build.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return _sha1(f"{st.st_size}:{st.st_mtime_ns}")


def schema_hash(path: str) -> str:
    """Hash of a parquet's column set (names + types). Reads only the header."""
    try:
        import pyarrow.parquet as pq
        schema = pq.read_schema(path)
        cols = [(f.name, str(f.type)) for f in schema]
        return _sha1(json.dumps(cols, sort_keys=True))
    except Exception:
        return "unknown"


def config_hash(config: dict) -> str:
    """Signature of the build-time config a feature's values depend on."""
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
    except Exception:
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
    for name, rec in (m.get("source_files") or {}).items():
        path = source_paths(data_dir, code).get(name)
        if path is None:
            continue
        if rec.get("hash") != file_fingerprint(path):
            return False
    return True
