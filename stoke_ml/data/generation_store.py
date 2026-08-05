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

Single-writer assumption: macro downloads run one at a time, so no file locks
(consistent with the previous flat-write script).  Old generations are retained
(no pruning) — retention is a separate concern.

Refusal is unconditional — there is no formal/``require_valid`` mode.  A torn
generation must never be silently read in ANY mode; raising
:class:`GenerationStoreError` is the only safe answer, and the feature build
fails loudly rather than merging a partial dataset (§十三-2).
"""
import json
import os

import pandas as pd

GEN_SUFFIX = "_gen"
CURRENT_NAME = "CURRENT"


class GenerationStoreError(RuntimeError):
    """A generation layout is present but torn / incomplete."""


def write_generation(data_dir: str, rel: str, df: pd.DataFrame, manifest: dict) -> str:
    """Write ``df`` + ``manifest`` as the next generation and flip CURRENT.

    Returns the generation name (e.g. ``"gen_00000002"``).  Writes each file via
    temp-file + ``os.replace``; CURRENT is flipped LAST so a crash mid-write
    leaves CURRENT pointing at the previous complete generation (§十三-2).
    """
    gen_root = os.path.join(data_dir, rel + GEN_SUFFIX)
    os.makedirs(gen_root, exist_ok=True)

    existing = [
        int(name[len("gen_"):])
        for name in os.listdir(gen_root)
        if name.startswith("gen_") and name[len("gen_"):].isdigit()
    ]
    next_n = max(existing, default=0) + 1
    gen_name = f"gen_{next_n:08d}"
    gen_dir = os.path.join(gen_root, gen_name)
    os.makedirs(gen_dir, exist_ok=True)

    data_path = os.path.join(gen_dir, "data.parquet")
    tmp = os.path.join(gen_dir, "data.parquet.tmp")
    df.to_parquet(tmp)
    os.replace(tmp, data_path)

    stamped = dict(manifest)
    stamped["generation"] = gen_name
    manifest_path = os.path.join(gen_dir, "manifest.json")
    tmp = os.path.join(gen_dir, "manifest.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stamped, f, indent=2, ensure_ascii=False)
    os.replace(tmp, manifest_path)

    current_path = os.path.join(gen_root, CURRENT_NAME)
    tmp = os.path.join(gen_root, CURRENT_NAME + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(gen_name)
    os.replace(tmp, current_path)
    return gen_name


def read_generation(data_dir: str, rel: str) -> pd.DataFrame | None:
    """Read the active generation's ``data.parquet``, or ``None`` if no
    generation layout exists (caller falls back to a legacy layout).

    Raises :class:`GenerationStoreError` whenever a generation layout is present
    but torn: CURRENT missing, CURRENT not a ``gen_\\d{8}`` name, CURRENT
    pointing at a missing generation dir, or the active generation missing
    either ``data.parquet`` or ``manifest.json``.  This refusal applies in ALL
    modes — a torn generation must never be silently read.
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
        and gen_name[5:].isdigit()
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
    return pd.read_parquet(os.path.join(gen_dir, "data.parquet"))
