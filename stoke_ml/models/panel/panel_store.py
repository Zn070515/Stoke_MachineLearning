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

import hashlib
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

# Warn-and-proceed keys (T4 §八): bind the store to the EXTERNAL data
# artifacts it was built from — data manifest, calendar, universe
# status/delist, index membership, prebuilt feature manifest.  A mismatch
# warns loudly but proceeds (each is re-derivable by rebuilding the store), so
# stale is suspicious, not fatal.  Compared recorded-meta-vs-expected ONLY
# when BOTH sides carry a value (None on either side = skip), mirroring
# config_hash.
_WARN_META_KEYS: tuple[str, ...] = (
    "data_manifest_hash", "calendar_hash", "universe_status_hash",
    "membership_hash", "prebuilt_feature_manifest_hash",
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


def _hash_sha1(text: str) -> str:
    """16-hex-char content digest, matching the codebase's hash convention."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _stock_order_hash(panel_data: dict) -> str | None:
    """Canonical hash of the panel's STOCK ROW ORDER, self-derived.

    ``build_panel_features`` writes stock_codes = valid_codes (stocks that
    survived cleaning), which may be a SUBSET of the requested universe.  So the
    meaningful binding is the store's OWN stock_codes order: tamper
    stock_codes.json and this recompute diverges from the recorded value,
    refusing a store whose row identity no longer matches its arrays (a
    misaligned stock_codes silently trains the WRONG stocks via the dataset's
    positional randperm).  None when the panel lacks stock_codes (callers
    defensively skip the binding).
    """
    codes = panel_data.get("stock_codes")
    if codes is None:
        return None
    return _hash_sha1("\n".join(str(c) for c in codes))


def _array_dtype(value) -> str:
    """dtype of an array-like, reading it DIRECTLY off ndarray objects.

    A loaded store's feature arrays are ``np.memmap`` (an ndarray subclass), and
    ``.dtype`` is a header property — no file bytes are touched, so reading it
    off the object keeps the lazy-memmap path clean.  ``np.asarray(value).dtype``
    would give the same result but routes through ``asarray``'s
    copy/view-conversion semantics for no benefit here; only genuinely
    non-array inputs (lists/tuples from tests) need the ``np.asarray`` fallback,
    which materializes a tiny array.
    """
    if isinstance(value, np.ndarray):
        return str(value.dtype)
    return str(np.asarray(value).dtype)


def _feature_schema_hash(panel_data: dict) -> str | None:
    """Canonical hash of the panel's FEATURE SCHEMA (cols + array dtypes).

    Covers the ordered ``past_known_cols`` / ``past_observed_cols`` lists plus
    the dtypes of the three feature arrays, so a tampered col list (a renamed /
    removed column) or a dtype change on disk is refused at load instead of
    silently feeding a mismatched schema.  None when any required key is missing
    (callers defensively skip the binding).
    """
    cols_ok = (
        panel_data.get("past_known_cols") is not None
        and panel_data.get("past_observed_cols") is not None
    )
    arrays_ok = all(
        panel_data.get(k) is not None
        for k in ("past_known", "past_observed", "static_features")
    )
    if not cols_ok or not arrays_ok:
        return None
    payload = {
        "past_known_cols": list(panel_data["past_known_cols"]),
        "past_observed_cols": list(panel_data["past_observed_cols"]),
        "past_known_dtype": _array_dtype(panel_data["past_known"]),
        "past_observed_dtype": _array_dtype(panel_data["past_observed"]),
        "static_features_dtype": _array_dtype(panel_data["static_features"]),
    }
    return _hash_sha1(json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


# §八 (T4): SINGLE SOURCE OF TRUTH for the self-consistency bindings.  Hard-fail
# keys validated by SELF-CONSISTENCY (not by the expected-vs-recorded loop
# above): ``save_panel_memmap`` recomputes them from the ACTUAL ``panel_data``
# and ``load_panel_memmap`` re-derives them from the store's own arrays/lists,
# so a tampered stock_codes.json / past_known_cols.json / past_observed_cols.json
# / feature dtype is refused instead of silently training the WRONG stocks:
# ``PanelDataset.__getitem__``'s ``max_stocks_per_date`` randperm samples rows by
# position (dataset.py), so a misaligned stock_codes would train the wrong codes
# with no error.  The current run's expected_meta CANNOT carry these — a
# store-backed re-run never rebuilds to discover the schema, and
# ``build_panel_features`` writes stock_codes = valid_codes (stocks that SURVIVED
# cleaning), a SUBSET of the requested universe — so they live outside the
# expected-vs-recorded loop.  A recorded hash the store can no longer recompute
# (source keys missing) is also refused — the record claims a binding the store
# cannot vouch for.  Every key here must have a recompute that NEVER raises and
# returns None (skip) when the panel lacks the source keys — the save path
# iterates the same table, so an unimplemented/raising recompute would fail the
# save, not silently record a bogus binding.
_SELF_CONSISTENCY_RECOMPUTE: dict[str, callable] = {
    "feature_schema_hash": _feature_schema_hash,
    "stock_order_hash": _stock_order_hash,
}

# Derived from the recompute table so the set of RECORDED keys can never drift
# from the set of VALIDATED keys: adding a binding means implementing its
# recompute here, and every recompute key is then automatically both recorded at
# save and validated at load (adding to one table only would otherwise either
# record-never-validate or KeyError at runtime).
_SELF_CONSISTENCY_META_KEYS: tuple[str, ...] = tuple(_SELF_CONSISTENCY_RECOMPUTE)


def _merge_self_fingerprints(meta: dict, panel_data: dict) -> None:
    """Overwrite ``meta``'s self-fingerprints with values recomputed from the ACTUAL
    ``panel_data`` (authoritative), dropping a key the panel cannot recompute.

    The caller's meta wins for every OTHER key; these two can only be verified
    at load by recomputing from the store's own arrays/lists, so the recorded
    value must derive from the panel actually written — never from the caller's
    request side (which may describe a universe that got trimmed by cleaning).
    A key the panel lacks is dropped rather than inherited, so a stale caller
    value can never masquerade as a binding the store cannot vouch for.
    """
    for key, recompute in _SELF_CONSISTENCY_RECOMPUTE.items():
        value = recompute(panel_data)
        if value is None:
            meta.pop(key, None)
        else:
            meta[key] = value


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
    # every other file, so a crash can't leave a truncated file.  Before
    # writing, the caller-supplied meta is merged with the panel's OWN
    # self-consistency fingerprints (feature schema + stock order), recomputed
    # from the ACTUAL panel_data being written — those two can only be
    # verified at load by recomputing from the store's own arrays/lists, so
    # they must derive from the panel actually written, never from the
    # caller's request side (which may describe a universe trimmed by
    # cleaning).
    if meta is not None:
        meta = dict(meta)
        _merge_self_fingerprints(meta, panel_data)
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


def _validate_meta(
    out: Path, expected_meta: dict, recorded: dict | None = None,
) -> None:
    """Refuse a store whose meta fingerprint disagrees with the current build.

    Research-critical fields (``_CRITICAL_META_KEYS``) must match or the load
    is refused — the stored panel's targets / column set / calendar would be
    stale.  ``_WARN_META_KEYS`` (external-artifact hashes: data manifest,
    calendar, universe status/delist, membership, prebuilt feature manifest)
    warn-and-proceed on mismatch, compared only when BOTH sides carry a value
    (each is re-derivable by rebuilding the store, so stale is suspicious, not
    fatal).  Non-critical fields (git_commit) drift with a loud warning and a
    proceed, mirroring cache_manifest's stale-manifest logic.  ``config_hash``
    is compared only when BOTH sides have it (None means config could not
    load).  A store with no meta.json at all cannot vouch for its config and
    is refused rather than trusted.

    ``recorded`` is the already-loaded meta.json dict when the caller has it
    (load_panel_memmap reads the file once and threads it through to both
    validators); when None it is read here.
    """
    if recorded is None:
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
    for key in _WARN_META_KEYS:
        rec = recorded.get(key)
        exp = expected_meta.get(key)
        if rec is None or exp is None:
            continue  # either side lacks the binding — nothing to compare
        if rec != exp:
            logger.warning(
                "panel store %s: %s differs (stored=%r requested=%r) — "
                "proceeding since it is re-derivable by rebuilding the store; "
                "a persistent mismatch is suspicious and worth investigating",
                out, key, rec, exp,
            )
    exp_git = expected_meta.get("git_commit")
    rec_git = recorded.get("git_commit")
    if exp_git and rec_git and exp_git != rec_git:
        logger.warning(
            "panel store %s was built at git commit %s (current %s) — "
            "proceeding since model-layer code does not change feature values; "
            "rebuild if you changed feature/preprocessing code", out, rec_git, exp_git,
        )


def _validate_self_consistency(
    data: dict, out: Path, recorded: dict | None = None,
) -> None:
    """Recompute the store's self-consistency fingerprints from its OWN arrays
    /lists and refuse a store that can no longer recompute a recorded binding.

    ``stock_order_hash`` / ``feature_schema_hash`` are recorded at save time
    from the ACTUAL ``panel_data`` (see :func:`_merge_self_fingerprints`).
    They are NOT compared against the current run's expected_meta — a
    store-backed re-run never rebuilds to discover the schema, and
    ``build_panel_features`` writes stock_codes = valid_codes, a SUBSET of the
    requested universe — so the only meaningful check is recorded-vs-recompute
    from the loaded store's own arrays/lists.  A tampered stock_codes.json /
    past_known_cols.json / past_observed_cols.json / feature dtype is refused
    instead of silently training the WRONG stocks:
    ``PanelDataset.__getitem__``'s ``max_stocks_per_date`` randperm samples
    rows by position (dataset.py), so a misaligned stock_codes would train
    wrong codes with no error.  A recorded hash the store can no longer
    recompute (source keys missing) is also refused — the record claims a
    binding the store cannot vouch for.

    Runs whenever the store has a meta.json (the saved fingerprints live
    there), regardless of whether ``expected_meta`` was given for the
    expected-vs-recorded loop.  ``recorded`` is the already-loaded meta.json
    dict when the caller has it (load_panel_memmap reads the file once and
    threads it through); when None it is read here.
    """
    if recorded is None:
        recorded = _load_meta(out)
    if recorded is None:
        return
    violated: list[str] = []
    for key in _SELF_CONSISTENCY_META_KEYS:
        rec = recorded.get(key)
        if rec is None:
            continue  # legacy store saved before the binding existed
        recompute = _SELF_CONSISTENCY_RECOMPUTE[key]
        recomputed = recompute(data)
        if recomputed is None or recomputed != rec:
            violated.append(
                f"{key}: recorded={rec!r} recomputed={recomputed!r}"
            )
    if violated:
        raise RuntimeError(
            f"panel store at {out} fails self-consistency check — the store's "
            "own metadata no longer matches its arrays/lists: "
            + "; ".join(violated) + ".  Rebuild the store for this "
            "configuration (--panel-store DIR with the current flags)."
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
    store can never silently feed wrong targets.  git_commit drift only warns,
    and ``_WARN_META_KEYS`` (external-artifact hashes) drift with a loud
    warning and a proceed (each is re-derivable by rebuilding the store).
    Independently of ``expected_meta``, the store's self-consistency
    fingerprints (feature schema + stock order, see
    :func:`_validate_self_consistency`) are recomputed from the store's own
    arrays/lists whenever a meta.json exists — a tampered identity file is
    refused rather than silently training the wrong stocks.

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
    # Read meta.json ONCE and thread the same dict through both validators
    # (the expected-vs-recorded guard and the self-consistency guard), so a
    # load never re-reads the file a second time.
    recorded = _load_meta(out)
    if expected_meta is not None:
        _validate_meta(out, expected_meta, recorded=recorded)
    _validate_self_consistency(data, out, recorded=recorded)
    return data


def panel_store_complete(out_dir: str | Path) -> bool:
    """True when ``out_dir`` holds a complete, fully-written panel store."""
    out = Path(out_dir)
    if not (out / _COMPLETE_MARKER).is_file():
        return False
    return all((out / f"{name}.npy").is_file() for name in _PANEL_ARRAY_KEYS) and all(
        (out / f"{name}.json").is_file() for name in _PANEL_JSON_KEYS
    )
