"""Memmap persistence for the panel feature grids (§十六).

``build_panel_features`` returns a dense (N, T, D) feature grid that, for a
full A-share universe, is the dominant resident-memory cost of a training run
(past_known / past_observed alone are hundreds of GB at 5530 × 6500 × ~300 ×
4B).  The plan (§十六) calls this the "一次性建立整个稠密 (N,T,D) 内存数组"
problem and asks for a lazy storage layer so a run never holds the whole dense
grid in RAM at once.

This module implements that layer with **numpy memmap (.npy)**: save the panel
once with :func:`save_panel_memmap`, then re-load it with
:func:`load_panel_memmap`, which returns ``np.memmap`` objects backed by the
files.  Memmap slices are lazy — a window read in ``PanelDataset.__getitem__``
(or a fold slice in ``_slice_panel``) page-faults only the rows/columns it
actually touches, so a large universe's epoch startup never needs to
materialize the whole array.

No new third-party dependency: ``np.save``/``np.load(mmap_mode='r')`` are stock
numpy.  (Zarr was considered and rejected — it would add a pinned dependency
to ``uv.lock`` for no benefit over memmap here.)

Windows file-lock note: an open memmap keeps its backing file locked on
Windows — attempts to ``os.remove``/``os.replace`` (re-saving over the same
path, or a ``TemporaryDirectory`` cleanup while the loaded dict is still
referenced) raise ``PermissionError``.  Drop the loaded dict and let the arrays
be garbage-collected before removing / re-creating a store.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Canonical array keys of the build_panel_features output dict.  These are
# what a complete store must contain; any other ndarray key present in the
# source dict is persisted too (so the store round-trips faithfully) but a
# store missing any of these is refused by load_panel_memmap.
_PANEL_ARRAY_KEYS: tuple[str, ...] = (
    "static_features", "past_known", "past_observed",
    "y_direction", "y_return", "y_volatility",
    "date_indices", "global_dates",
    "observation_mask", "entry_eligible_mask", "return_target_mask",
    "vol_target_mask", "decision_eligible_mask", "history_eligible_mask",
    "universe_eligible_mask", "forward_vol_nobs",
    "realized_return", "close_price", "open_price",
)

# Non-array metadata persisted as JSON lists (row/column identity).  train_panel
# needs stock_codes for row identity; the *_cols lists feed per-fold dead-column
# drop and the has_* channel-coverage probe.
_PANEL_JSON_KEYS: tuple[str, ...] = (
    "stock_codes", "past_known_cols", "past_observed_cols",
)

# Marker written ONLY after every array has been durably replaced, so an
# interrupted save never looks complete to a later run.
_COMPLETE_MARKER = "complete.json"

# Build-time fingerprint written BEFORE the completeness marker.  A store that
# has complete.json but no meta.json is a legacy/partial store that cannot
# vouch for its config, so an expected_meta load refuses it.
_META_FILE = "meta.json"

# Fields a load MUST match to trust the stored panel's targets.  Silently
# training on stale targets — a --horizon 1 store reused with --horizon 5, or
# a different universe/feature switch set — is the failure this guard exists
# to block.  ``config_hash`` is only compared when both sides have it
# (None = config could not load, mirroring cache_manifest).  Fields NOT here
# (git_commit) drift loudly-but-proceeds.
_CRITICAL_META_KEYS: tuple[str, ...] = (
    "horizon", "seq_len", "start", "end", "universe", "n_stocks",
    "feature_switches", "config_hash",
)


def _atomic_npy(out: Path, name: str, arr: np.ndarray) -> None:
    """Write ``arr`` to ``{name}.npy`` atomically (temp file + os.replace)."""
    tmp = out / f"{name}.npy.tmp"
    # np.save appends '.npy' to a filename not ending in '.npy', so write via an
    # open handle to keep the temp suffix unambiguous.
    with open(tmp, "wb") as fh:
        np.save(fh, arr, allow_pickle=False)
    os.replace(tmp, out / f"{name}.npy")


def _atomic_json(out: Path, name: str, value) -> None:
    """Write ``value`` to ``{name}.json`` atomically (temp file + os.replace)."""
    tmp = out / f"{name}.json.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False)
    os.replace(tmp, out / f"{name}.json")


def save_panel_memmap(
    panel_data: dict, out_dir: str | Path, meta: dict | None = None,
) -> list[str]:
    """Persist every array of a build_panel_features output dict to disk.

    Each array is written to ``{out_dir}/{name}.npy``; ``stock_codes`` and the
    ``*_cols`` column lists go to ``{name}.json``.  ``has_*``-style aux keys
    (not part of the panel contract) are skipped; any other non-None value
    that is neither an array nor a known JSON metadata key is dropped with a
    warning so a missing required key surfaces immediately rather than as a
    confusing "incomplete store" on load.  Every write is atomic (temp file +
    ``os.replace``); ``meta.json`` (when ``meta`` is given — the build-time
    config fingerprint, see :func:`load_panel_memmap`) is written BEFORE a
    ``complete.json`` marker, which is written only after all files are in
    place so a partially-written store is never mistaken for complete (see
    :func:`panel_store_complete`).

    Returns the sorted list of written file names (``<name>.npy`` /
    ``<name>.json`` / ``meta.json``).
    """
    out = Path(out_dir)
    if out.exists() and not out.is_dir():
        raise ValueError(
            f"panel store path {out} exists but is not a directory — a panel "
            "store is a directory of .npy/.json files.  Point at a new/empty "
            "directory or remove the conflicting file."
        )
    out.mkdir(parents=True, exist_ok=True)
    # Remove the completeness marker up front: from now until the final
    # write the dir is explicitly in-progress.
    marker = out / _COMPLETE_MARKER
    if marker.exists():
        marker.unlink()
    written: list[str] = []
    for name, value in panel_data.items():
        if value is None:
            continue
        if isinstance(value, np.ndarray):
            if name.startswith("has_"):
                continue
            _atomic_npy(out, name, np.asarray(value))
            written.append(f"{name}.npy")
        elif name in _PANEL_JSON_KEYS:
            _atomic_json(out, name, value)
            written.append(f"{name}.json")
        else:
            logger.warning(
                "save_panel_memmap: dropping key %r (type %s) — not an array "
                "or a known JSON metadata key, so it is not part of the panel "
                "store contract", name, type(value).__name__,
            )
    # meta.json first (so a store is never complete without it), then the
    # completeness marker — both via the same atomic temp+os.replace path as
    # every other file, so a crash can't leave a truncated file.
    if meta is not None:
        _atomic_json(out, "meta", meta)
        written.append(_META_FILE)
    _atomic_json(out, "complete", {"complete": True})
    return sorted(written)


def _load_meta(out: Path) -> dict | None:
    """meta.json contents of a store, or None when the store has no meta file."""
    p = out / _META_FILE
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _validate_meta(out: Path, expected_meta: dict) -> None:
    """Refuse a store whose meta fingerprint disagrees with the current build.

    Research-critical fields (``_CRITICAL_META_KEYS``) must match or the load
    is refused — the stored panel's targets / column set / calendar would be
    stale.  Non-critical fields (git_commit) drift with a loud warning and a
    proceed, mirroring cache_manifest's stale-manifest logic.  ``config_hash``
    is compared only when BOTH sides have it (None means config could not
    load).  A store with no meta.json at all cannot vouch for its config and
    is refused rather than trusted.
    """
    recorded = _load_meta(out)
    if recorded is None:
        raise RuntimeError(
            f"panel store at {out} has no {_META_FILE} — cannot verify the "
            "stored panel matches this run's config (horizon/seq_len/start/end/"
            "universe/feature switches).  Rebuild the store for the current "
            "flags (--panel-store DIR) or point at a fresh directory."
        )
    mismatches: list[str] = []
    for key in _CRITICAL_META_KEYS:
        rec = recorded.get(key)
        exp = expected_meta.get(key)
        if key == "config_hash" and (exp is None or rec is None):
            continue  # config could not load on one side — nothing to compare
        if rec != exp:
            mismatches.append(f"{key}: stored={rec!r} requested={exp!r}")
    if mismatches:
        raise RuntimeError(
            f"panel store at {out} does not match this run's config — refusing "
            "to load stale targets: " + "; ".join(mismatches) +
            ".  Rebuild the store for this configuration "
            "(--panel-store DIR with the current flags)."
        )
    exp_git = expected_meta.get("git_commit")
    rec_git = recorded.get("git_commit")
    if exp_git and rec_git and exp_git != rec_git:
        logger.warning(
            "panel store %s was built at git commit %s (current %s) — "
            "proceeding since model-layer code does not change feature values; "
            "rebuild if you changed feature/preprocessing code", out, rec_git, exp_git,
        )


def load_panel_memmap(
    out_dir: str | Path, expected_meta: dict | None = None,
) -> dict:
    """Load a store written by :func:`save_panel_memmap` back into a dict
    matching the build_panel_features contract.

    Numeric arrays are loaded read-only via ``np.load(mmap_mode='r')`` — the
    returned ``np.memmap`` objects page-fault only the rows/columns actually
    read, so a full universe never materializes in RAM.  The JSON-list keys
    (``stock_codes``, ``past_known_cols``, ``past_observed_cols``) come back
    as plain Python lists, and ``global_dates`` as a datetime64 memmap — all
    directly consumable by ``PanelDataset`` / ``_slice_panel`` / train_panel.

    When ``expected_meta`` is given (the current run's build fingerprint), the
    store's ``meta.json`` is validated against it first — a mismatch on any
    research-critical field (horizon, seq_len, start/end, universe, feature
    switches, config_hash) raises RuntimeError naming the mismatch, so a stale
    store can never silently feed wrong targets.  git_commit drift only warns.

    Raises FileNotFoundError naming every required file that is missing.
    """
    out = Path(out_dir)
    missing = [f"{name}.npy" for name in _PANEL_ARRAY_KEYS
               if not (out / f"{name}.npy").is_file()]
    missing += [f"{name}.json" for name in _PANEL_JSON_KEYS
                if not (out / f"{name}.json").is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete panel memmap store at {out} — missing "
            f"{len(missing)} required file(s): {', '.join(missing)}"
        )
    data: dict = {}
    for path in sorted(out.glob("*.npy")):
        if path.name.endswith(".tmp"):
            continue
        data[path.name[:-4]] = np.load(path, mmap_mode="r")
    for path in sorted(out.glob("*.json")):
        if path.name in (_COMPLETE_MARKER, _META_FILE):
            continue
        with open(path, encoding="utf-8") as fh:
            data[path.name[:-5]] = json.load(fh)
    if expected_meta is not None:
        _validate_meta(out, expected_meta)
    return data


def panel_store_complete(out_dir: str | Path) -> bool:
    """True when ``out_dir`` holds a complete, fully-written panel store."""
    out = Path(out_dir)
    if not (out / _COMPLETE_MARKER).is_file():
        return False
    return all((out / f"{name}.npy").is_file() for name in _PANEL_ARRAY_KEYS) and all(
        (out / f"{name}.json").is_file() for name in _PANEL_JSON_KEYS
    )
