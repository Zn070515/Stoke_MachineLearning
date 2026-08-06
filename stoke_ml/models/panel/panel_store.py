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
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

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


def save_panel_memmap(panel_data: dict, out_dir: str | Path) -> list[str]:
    """Persist every array of a build_panel_features output dict to disk.

    Each array is written to ``{out_dir}/{name}.npy``; ``stock_codes`` and the
    ``*_cols`` column lists go to ``{name}.json``.  ``has_*``-style aux keys
    (not part of the panel contract) are skipped, as are any other non-array
    values.  Every write is atomic (temp file + ``os.replace``); a
    ``complete.json`` marker is written only after all files are in place so a
    partially-written store is never mistaken for complete (see
    :func:`panel_store_complete`).

    Returns the sorted list of written file names (``<name>.npy`` /
    ``<name>.json``).
    """
    out = Path(out_dir)
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
    marker.write_text(json.dumps({"complete": True}), encoding="utf-8")
    return sorted(written)


def load_panel_memmap(out_dir: str | Path) -> dict:
    """Load a store written by :func:`save_panel_memmap` back into a dict
    matching the build_panel_features contract.

    Numeric arrays are loaded read-only via ``np.load(mmap_mode='r')`` — the
    returned ``np.memmap`` objects page-fault only the rows/columns actually
    read, so a full universe never materializes in RAM.  The JSON-list keys
    (``stock_codes``, ``past_known_cols``, ``past_observed_cols``) come back
    as plain Python lists, and ``global_dates`` as a datetime64 memmap — all
    directly consumable by ``PanelDataset`` / ``_slice_panel`` / train_panel.

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
        if path.name == _COMPLETE_MARKER:
            continue
        with open(path, encoding="utf-8") as fh:
            data[path.name[:-5]] = json.load(fh)
    return data


def panel_store_complete(out_dir: str | Path) -> bool:
    """True when ``out_dir`` holds a complete, fully-written panel store."""
    out = Path(out_dir)
    if not (out / _COMPLETE_MARKER).is_file():
        return False
    return all((out / f"{name}.npy").is_file() for name in _PANEL_ARRAY_KEYS) and all(
        (out / f"{name}.json").is_file() for name in _PANEL_JSON_KEYS
    )
