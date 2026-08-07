"""Source-tree + canonical-JSON hashing for the panel store's §七 provenance.

The panel store's ``panel_input_hash`` (and its ``feature_code_tree_hash``
component) must answer "was this store built from the code that is on disk
RIGHT NOW" — the §七 fake-provenance guard.  Two properties matter:

* Content, not git: uncommitted edits must be visible.  A store bound to a
  ``git_commit`` alone would silently pass the staleness guard after a local
  edit that changed the feature code, so this module hashes the ON-DISK bytes
  of every ``*.py`` file under ``stoke_ml/`` + ``scripts/production/`` (the
  feature pipeline and the production entry points that consume it) —
  independent of any VCS state.
* Determinism, not mtime: the same tree always yields the same hash, so a
  rebuild is byte-identical.  Each file is hashed by streaming SHA-256 over its
  bytes, then the SORTED ``relpath → digest`` mapping is hashed via canonical
  JSON (``sort_keys`` + compact separators — the repo convention).

``feature_code_tree_hash`` deliberately does NOT use
``functools.lru_cache``: the whole point is to observe an edit on the next
call, and a long-lived process (or a test that edits a file then re-hashes)
must see the change immediately.
"""
from __future__ import annotations

import hashlib
import json
import os

#: The two source subtrees the panel-input provenance covers: the feature/data
#: library plus the production scripts that drive it.  Relative to the project
#: root (the repo top level).  Everything else (tests, docs, plans) is excluded
#: by design — a test change is not a feature-code change.
_SOURCE_DIRS: tuple[str, ...] = ("stoke_ml", "scripts/production")


def canonical_json(obj) -> str:
    """Canonical JSON string for ``obj`` — sorted keys, compact separators.

    The repo's canonical serialization (matches ``cache_manifest._stable_dumps``
    minus the ``default=str`` fallback, which this module never needs — the
    provenance payload is plain JSON scalars / lists / dicts).  Sorting the keys
    makes two dicts with the same content serialize identically regardless of
    insertion order, which is what makes the aggregate hash deterministic.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def hash_json(obj) -> str:
    """SHA-256 hex digest of the canonical JSON serialization of ``obj``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def _file_digest(path: str) -> str | None:
    """Streaming SHA-256 of a file's bytes; None on any read error.

    Streaming (64 KB chunks) so a large module never loads fully into memory.
    Returns None — not raises — on a read failure: the tree hash treats an
    unreadable file as "no digest for this relpath" and, since a digest-missing
    relpath is still recorded under its sorted key, the tree hash still changes
    when a previously-readable file becomes unreadable.  The hash is over the
    RAW BYTES (not text), so no encoding/canonicalization ambiguity.
    """
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _project_root() -> str:
    """The repo top level, derived from this module's location.

    ``stoke_ml/models/panel/code_tree_hash.py`` → four levels up → the project
    root.  A supplied ``root`` overrides this in ``feature_code_tree_hash``
    (tests point at a ``tmp_path`` fixture).
    """
    return os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))


def feature_code_tree_hash(root: str | None = None) -> str:
    """Content hash of every ``*.py`` under ``root/{stoke_ml,scripts/production}``.

    Walks the two source subtrees (``_SOURCE_DIRS``) of ``root`` (default: the
    project root derived from this module), hashing each ``*.py`` file's raw
    bytes and the SORTED ``relpath → digest`` mapping via ``hash_json``.  The
    mapping key is the path RELATIVE to the repo root (``stoke_ml/...``,
    ``scripts/production/...``), so renaming or moving a file changes the hash
    even when its bytes are unchanged.  Non-``.py`` files are ignored (a data
    artifact under the tree is not feature code).  An empty/absent tree returns
    ``"unknown"`` — a store built with no source to hash cannot claim a binding.

    NOT cached (deliberately): an edit must be observed on the next call.  Cost
    is ~O(tree size), typically well under 100 ms for this repo — one hash per
    store build is negligible.
    """
    base = os.path.abspath(root) if root is not None else _project_root()
    entries: dict[str, str] = {}
    for sub in _SOURCE_DIRS:
        top = os.path.join(base, sub)
        if not os.path.isdir(top):
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            # Deterministic traversal: sort so the walk order never leaks into
            # the hash (though the mapping hash is order-independent anyway).
            dirnames.sort()
            for name in sorted(filenames):
                if not name.endswith(".py"):
                    continue
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, base).replace("\\", "/")
                digest = _file_digest(path)
                if digest is None:
                    # Record an explicit null so an unreadable file still
                    # differentiates this tree from one where it is absent.
                    entries[rel] = None
                else:
                    entries[rel] = digest
    if not entries:
        return "unknown"
    return hash_json(entries)
