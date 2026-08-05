#!/usr/bin/env python
"""Docs-vs-code consistency guard.

Derives ground-truth values from the codebase (AST-parsed dataclass defaults,
FeaturePipeline constructor signature, on-disk file counts) and verifies that
CLAUDE.md / CONTEXT.md / README.md still state those values. Exits 1 on any
mismatch so it can gate CI.

Data-dependent checks (stock count) are skipped when the 109GB data dir is
absent — everything else runs anywhere.

Run:
    PYTHONPATH=. ./.venv/Scripts/python scripts/production/check_docs_consistency.py
"""

import ast
import glob
import os
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

def _find_root(start=None):
    """Walk up from the script until the dir holding config.yaml (repo root).

    Layout-agnostic so the guard keeps working regardless of how deeply
    scripts/ is nested (it moved scripts/ → scripts/production/ in §十七-2).
    """
    d = os.path.dirname(os.path.abspath(start or __file__))
    while True:
        if os.path.isfile(os.path.join(d, "config.yaml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("repo root (config.yaml) not found above script")
        d = parent


ROOT = _find_root()
DOCS = {
    "CLAUDE.md": os.path.join(ROOT, "CLAUDE.md"),
    "CONTEXT.md": os.path.join(ROOT, "CONTEXT.md"),
    "README.md": os.path.join(ROOT, "README.md"),
}

# FeaturePipeline constructor params that toggle processing rather than
# carrying auxiliary data columns — excluded from the data-dimension count.
PIPELINE_SWITCH_FLAGS = frozenset({
    "use_technical", "use_scoring", "use_temporal", "use_interaction",
    "use_feature_selection", "use_new_preprocessing", "use_emotion_refine",
    "use_fundamental_refine", "use_temporal_stats",
})


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_src(rel_path):
    return read(os.path.join(ROOT, rel_path))


def ast_panel_config():
    tree = ast.parse(read_src(os.path.join("stoke_ml", "models", "panel", "config.py")))
    defaults = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PanelConfig":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    defaults[item.target.id] = ast.literal_eval(item.value)
    return defaults


def ast_aux_dim_count():
    tree = ast.parse(read_src(os.path.join("stoke_ml", "features", "pipeline.py")))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FeaturePipeline":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    use_params = [a.arg for a in item.args.args if a.arg.startswith("use_")]
                    return len([p for p in use_params if p not in PIPELINE_SWITCH_FLAGS])
    raise SystemExit("FeaturePipeline.__init__ not found")


def stock_count():
    daily = os.path.join(ROOT, "data", "a_shares", "daily")
    if not os.path.isdir(daily):
        return None
    return len([p for p in os.listdir(daily) if p.endswith(".parquet")])


def test_count():
    return len(glob.glob(os.path.join(ROOT, "tests", "**", "test_*.py"), recursive=True))


def doc_test_number(text):
    m = re.search(r"~?\s*(\d+)\s*(?:test files|个测试文件)", text)
    return int(m.group(1)) if m else None


def main():
    cfg = ast_panel_config()
    aux_dims = ast_aux_dim_count()
    n_tests = test_count()
    n_stocks = stock_count()

    docs = {name: read(path) for name, path in DOCS.items()}
    results = []

    def check(name, ok, detail):
        results.append((name, ok, detail))

    # ── PanelConfig constants → CONTEXT.md ──
    for const, doc_key, needle in [
        ("xlstm_num_blocks", "CONTEXT.md", r"xlstm_num_blocks \| %d" % cfg["xlstm_num_blocks"]),
        ("batch_size", "CONTEXT.md", r"batch_size \| %d" % cfg["batch_size"]),
        ("hidden_dim", "CONTEXT.md", r"hidden_dim \| %d" % cfg["hidden_dim"]),
        ("past_known_dim", "CONTEXT.md", r"\(%d维\)" % cfg["past_known_dim"]),
        ("past_observed_dim", "CONTEXT.md", r"\(%d维\)" % cfg["past_observed_dim"]),
    ]:
        check(f"CONTEXT {const}", re.search(needle, docs["CONTEXT.md"]) is not None,
              f"expect {const}={cfg[const]} in CONTEXT.md")

    # ── PanelConfig format line → README.md ──
    fmt = r"%d PastKnown \+ %d PastObserved \+ %d Static" % (
        cfg["past_known_dim"], cfg["past_observed_dim"], cfg["static_dim"])
    check("README panel dims", re.search(fmt, docs["README.md"]) is not None,
          f"expect '{fmt}' in README.md")

    # ── Aux dimension count → CLAUDE.md / README.md ──
    check("CLAUDE aux dims",
          re.search(r"%d `use_\*` data dimensions" % aux_dims, docs["CLAUDE.md"]) is not None,
          f"expect {aux_dims} `use_*` data dimensions in CLAUDE.md")
    check("README aux dims",
          re.search(r"%d个 `use_\*`" % aux_dims, docs["README.md"]) is not None,
          f"expect {aux_dims}个 `use_*` 数据维度 in README.md")

    # ── Test count → CLAUDE.md / README.md (tolerance ±5) ──
    for doc_key in ("CLAUDE.md", "README.md"):
        doc_n = doc_test_number(docs[doc_key])
        ok = doc_n is not None and abs(doc_n - n_tests) <= 5
        check(f"{doc_key} test count", ok,
              f"expect ~{n_tests} test files in {doc_key} (got {doc_n})")

    # ── Stock count → all three docs (skip if data absent) ──
    if n_stocks is None:
        results.append(("stock count", True, "data/a_shares/daily absent — SKIPPED"))
    else:
        for doc_key in DOCS:
            check(f"{doc_key} stock count",
                  str(n_stocks) in docs[doc_key],
                  f"expect {n_stocks} in {doc_key}")

    # ── Fold schedule → train_panel.py + docs ──
    # Non-overlapping OOS folds (step = val_len), inner_val
    # carve selects best epoch, outer_test evaluated once, lockbox reserved.
    tp = read_src(os.path.join("scripts", "production", "train_panel.py"))
    val_ok = "inner_val" in tp and "step = val_len" in tp
    check("train_panel fold", val_ok,
          "expect inner_val carve + step=val_len (non-overlapping folds) in train_panel.py")
    for doc_key in ("CONTEXT.md", "README.md"):
        ok = (re.search(r"inner[ _]val", docs[doc_key], re.IGNORECASE) is not None
              and re.search(r"63\s*天", docs[doc_key]) is None)
        check(f"{doc_key} fold schedule", ok,
              "expect inner_val/outer_test described, no stale 63天 step in " + doc_key)

    # ── Architectural facts ─────────────────────────────────────────────
    # Each check derives ground truth from code; the docs must state the same
    # fact so an agent editing either side can't silently drift.

    def docs_all(pattern, name):
        ok = True
        for doc_key in DOCS:
            hit = re.search(pattern, docs[doc_key], re.IGNORECASE) is not None
            check(f"{doc_key} {name}", hit, f"expect '{pattern}' in {doc_key}")
            ok = ok and hit
        return ok

    storage_src = read_src(os.path.join("stoke_ml", "data", "storage.py"))
    contract_src = read_src(os.path.join("stoke_ml", "data", "contract.py"))
    calendar_src = read_src(os.path.join("stoke_ml", "data", "calendar.py"))
    evaluate_src = read_src(os.path.join("stoke_ml", "models", "panel", "evaluate.py"))
    baselines_src = read_src(os.path.join("stoke_ml", "models", "baseline", "panel_baselines.py"))

    # storage layout: single canonical flat daily/{code}.parquet (+ sidecar manifest)
    if "daily/{code}.parquet" in storage_src:
        docs_all(r"daily/\{code\}\.parquet", "flat storage layout")
    else:
        check("storage flat layout", False, "code lost daily/{code}.parquet canonical layout")

    # price basis: forward-adjusted qfq research series (unified qfq basis)
    if 'price_basis="qfq"' in contract_src:
        docs_all(r"qfq|前复权", "qfq price basis")
    else:
        check("price basis qfq", False, "code lost price_basis=qfq contract")

    # calendar: externalized exchange_calendar artifact with verified_until
    if "verified_until" in calendar_src and "bse_notices" in calendar_src:
        docs_all(r"verified_until", "verified_until calendar")
    else:
        check("calendar verified_until", False, "code lost externalized verified_until calendar")

    # headline evaluator version (EVALUATOR_VERSION in evaluate.py)
    m_ev = re.search(r'EVALUATOR_VERSION\s*=\s*"([^"]+)"', evaluate_src)
    if m_ev:
        ev = m_ev.group(1)
        pattern = r"evaluator(?:[_ ]version)?\s*[=:：]?\s*" + re.escape(ev)
        ok = any(re.search(pattern, docs[k], re.IGNORECASE) for k in DOCS)
        check("evaluator version", ok, f"expect evaluator version {ev} in a doc")
    else:
        check("evaluator version", False, "EVALUATOR_VERSION missing in evaluate.py")

    # required manifest: formal reads enforce manifest verification
    if "require_valid_manifest" in storage_src:
        docs_all(r"manifest", "manifest-required reads")
    else:
        check("required manifest", False, "code lost require_valid_manifest")

    # baseline list: Ridge / LightGBM / MLP panel baselines
    if all(w in baselines_src for w in ("Ridge", "LightGBM", "MLP")):
        for doc_key in DOCS:
            ok = all(re.search(w, docs[doc_key], re.IGNORECASE) for w in ("Ridge", "LightGBM", "MLP"))
            check(f"{doc_key} baseline list", ok, f"expect Ridge/LightGBM/MLP in {doc_key}")
    else:
        check("baseline list", False, "code lost Ridge/LightGBM/MLP baselines")

    # ── Report ──
    width = max(len(n) for n, _, _ in results)
    failed = 0
    for name, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"[{mark}] {name:<{width}}  {detail}")
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    if failed:
        print("Docs drift detected — update the docs above to match the code, then re-run.")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
