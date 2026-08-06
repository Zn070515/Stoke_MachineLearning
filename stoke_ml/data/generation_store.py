"""Generation-directory + CURRENT-pointer atomic aux-data writes (§十三-2).

数据文件与 Manifest 不是单一原子对象：单独替换 parquet 再单独替换 manifest，
两次 ``os.replace`` 之间崩溃会留下 torn state（新数据 + 旧 Manifest，反之亦然）。
generation 目录 + CURRENT 指针把数据与 Manifest 作为一代整体切换——读者只跟随
CURRENT，而 CURRENT 在 data.parquet 与 manifest.json 都落盘之后才最后翻转，
因此任何时刻都读不到「半成品」一代。

Data + manifest are not a single atomic object: replacing the parquet and then
the manifest as two separate ``os.replace`` calls leaves a torn window (new
data + old manifest, or vice versa).  A generation directory plus a CURRENT
pointer switches the pair as one unit — readers only follow CURRENT, and
CURRENT is flipped LAST, only after both ``data.parquet`` and ``manifest.json``
are in place, so a partially-written generation is never observable.

Layout::

    <data_dir>/<rel>_gen/
        gen_00000001/data.parquet + manifest.json
        gen_00000002/data.parquet + manifest.json
        CURRENT            # file whose content is the active generation name

Writes are serialized by a single-writer lock on the generation root
(``file_lock``, §十三): generation numbering, parquet, manifest and the CURRENT
flip all run under it, so a concurrent writer is REFUSED with
:class:`GenerationStoreError` rather than interleaved.  Readers need no lock —
CURRENT is flipped atomically and old generations are never deleted.  Old
generations are retained (no pruning) — retention is a separate concern.

Each write stamps a ``schema_hash`` over the FULL content (including the index,
via ``reset_index()``) into the manifest; read_generation recomputes it and
refuses a tampered parquet unconditionally.  A legacy generation written before
T10 has no ``schema_hash`` — it cannot be verified and is read with a warning.

Refusal is unconditional — there is no formal/``require_valid`` mode.  A torn
generation must never be silently read in ANY mode; raising
:class:`GenerationStoreError` is the only safe answer, and the feature build
fails loudly rather than merging a partial dataset (§十三-2).
"""
import json
import logging
import os

import pandas as pd

from stoke_ml.data.asset_contract import AtomicCommit, file_lock, schema_hash

logger = logging.getLogger(__name__)

GEN_SUFFIX = "_gen"
CURRENT_NAME = "CURRENT"


class GenerationStoreError(RuntimeError):
    """A generation layout is present but torn / incomplete."""


def _generation_schema_hash(df: pd.DataFrame) -> str:
    """Hash the FULL content INCLUDING the index — the macro frame's
    DatetimeIndex named "date" is the primary-key content, and
    ``asset_contract.schema_hash`` only covers ``df.columns``.

    ``reset_index()`` turns the index into a column so the hash covers its
    values; the parquet round-trip preserves the index, so read_generation
    applies the same canonicalization and recomputes an identical hash.
    """
    try:
        canonical = df.reset_index()
    except ValueError:
        # Index name collides with an existing column (degenerate layout) —
        # fall back to hashing the frame as-is; read_generation hits the same
        # path, so the hash is still recomputable.
        canonical = df
    return schema_hash(canonical)


def _stamp_manifest(manifest: dict, df: pd.DataFrame, gen_name: str) -> dict:
    """Stamp generation + content-fingerprint fields into the manifest.

    ``schema_hash`` is always (re)computed over the full content including the
    index so read_generation can detect a tampered parquet.  Caller-supplied
    ``rows`` / ``columns`` are preserved when present (backward compat);
    defaults derived from ``df`` are filled in otherwise.
    """
    stamped = dict(manifest)
    stamped["generation"] = gen_name
    stamped["schema_hash"] = _generation_schema_hash(df)
    if "rows" not in stamped:
        stamped["rows"] = int(len(df))
    if "columns" not in stamped:
        col_names = [str(c) for c in df.columns]
        if df.index.name is not None:
            col_names.append(str(df.index.name))
        stamped["columns"] = sorted(col_names)
    return stamped


def write_generation(
    data_dir: str, rel: str, df: pd.DataFrame, manifest: dict,
    *, lock_timeout: float = 600.0,
) -> str:
    """Write ``df`` + ``manifest`` as the next generation and flip CURRENT.

    Returns the generation name (e.g. ``"gen_00000002"``).  Writes each file via
    temp-file + ``os.replace``; CURRENT is flipped LAST so a crash mid-write
    leaves CURRENT pointing at the previous complete generation (§十三-2).

    The ENTIRE read-modify-write (generation numbering + parquet + manifest +
    CURRENT flip) runs under a single-writer lock keyed on the generation root,
    so two concurrent writers cannot compute the same next generation number or
    interleave CURRENT flips.  On contention past ``lock_timeout`` seconds the
    write is REFUSED with :class:`GenerationStoreError` — no partial write.
    """
    gen_root = os.path.join(data_dir, rel + GEN_SUFFIX)
    # Create the generation root (and its parents) before taking the lock — the
    # lock dir lives NEXT to the root, so its parent must exist for os.mkdir.
    os.makedirs(gen_root, exist_ok=True)
    try:
        with file_lock(gen_root, timeout=lock_timeout):
            existing = [
                int(name[len("gen_"):])
                for name in os.listdir(gen_root)
                if name.startswith("gen_") and name[len("gen_"):].isdigit()
            ]
            next_n = max(existing, default=0) + 1
            gen_name = f"gen_{next_n:08d}"
            gen_dir = os.path.join(gen_root, gen_name)
            os.makedirs(gen_dir, exist_ok=True)

            stamped = _stamp_manifest(manifest, df, gen_name)
            data_path = os.path.join(gen_dir, "data.parquet")
            with AtomicCommit(data_path) as ac:
                df.to_parquet(ac.tmp_path)
            manifest_path = os.path.join(gen_dir, "manifest.json")
            with AtomicCommit(manifest_path) as ac:
                with open(ac.tmp_path, "w", encoding="utf-8") as f:
                    json.dump(stamped, f, indent=2, ensure_ascii=False)

            current_path = os.path.join(gen_root, CURRENT_NAME)
            with AtomicCommit(current_path) as ac:
                with open(ac.tmp_path, "w", encoding="utf-8") as f:
                    f.write(gen_name)
            return gen_name
    except TimeoutError as exc:
        raise GenerationStoreError(
            f"could not acquire single-writer lock on {gen_root} within "
            f"{lock_timeout}s — another writer in progress"
        ) from exc


def read_generation(data_dir: str, rel: str) -> pd.DataFrame | None:
    """Read the active generation's ``data.parquet``, or ``None`` if no
    generation layout exists (caller falls back to a legacy layout).

    Raises :class:`GenerationStoreError` whenever a generation layout is present
    but torn: CURRENT missing, CURRENT not a ``gen_\\d{8}`` name, CURRENT
    pointing at a missing generation dir, or the active generation missing
    either ``data.parquet`` or ``manifest.json``.  This refusal applies in ALL
    modes — a torn generation must never be silently read.

    Beyond structure, the parquet is validated against the manifest's
    ``schema_hash`` (recomputed over the full content including the index).  A
    tampered or drifted file no longer matches and is refused unconditionally
    with :class:`GenerationStoreError`.  A legacy generation written before T10
    has no ``schema_hash`` — it cannot be verified and is read with a warning.
    """
    gen_root = os.path.join(data_dir, rel + GEN_SUFFIX)
    if not os.path.isdir(gen_root):
        return None

    current_path = os.path.join(gen_root, CURRENT_NAME)
    if not os.path.isfile(current_path):
        raise GenerationStoreError(
            f"generation layout present but CURRENT pointer missing: {gen_root}"
        )
    with open(current_path, encoding="utf-8") as f:
        gen_name = f.read().strip()
    if not (
        gen_name.startswith("gen_")
        and len(gen_name) == 12
        and gen_name[4:].isdigit()
    ):
        raise GenerationStoreError(
            f"CURRENT does not name a valid generation (expected gen_XXXXXXXX): "
            f"{gen_name!r} at {current_path}"
        )

    gen_dir = os.path.join(gen_root, gen_name)
    if not os.path.isdir(gen_dir):
        raise GenerationStoreError(
            f"CURRENT points to a missing generation dir: {gen_dir}"
        )
    for fname in ("data.parquet", "manifest.json"):
        if not os.path.isfile(os.path.join(gen_dir, fname)):
            raise GenerationStoreError(
                f"active generation incomplete (missing {fname}): {gen_dir}"
            )

    manifest_path = os.path.join(gen_dir, "manifest.json")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        raise GenerationStoreError(
            f"active generation manifest unreadable: {manifest_path} ({exc})"
        ) from exc

    data_path = os.path.join(gen_dir, "data.parquet")
    try:
        df = pd.read_parquet(data_path)
    except Exception as exc:
        # The parquet is user-controlled on-disk data; corruption can surface
        # as any of several pyarrow/OS errors, all of which mean "cannot read
        # this generation" and must refuse uniformly.
        raise GenerationStoreError(
            f"active generation data unreadable: {data_path} ({exc})"
        ) from exc

    stored_hash = manifest.get("schema_hash")
    if stored_hash is None:
        logger.warning(
            "generation %s manifest has no schema_hash (legacy, pre-T10); "
            "content cannot be verified — reading anyway",
            gen_name,
        )
    else:
        actual = _generation_schema_hash(df)
        if actual != stored_hash:
            raise GenerationStoreError(
                f"active generation {gen_name} failed content validation: "
                f"data.parquet no longer matches manifest schema_hash "
                f"(manifest={stored_hash!r}, actual={actual!r}) — {gen_dir}"
            )
    return df
