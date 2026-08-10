# V19 Review Fixes (7-Item Infrastructure Closeout) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all 7 infrastructure audit items from `ChatGPT_v19.md` §二十一 — genuine-PIT sector membership, a real derived-asset lineage gate, formal-prebuilt-only training, fail-closed canonical reads, an enforced Market Env column contract, a frozen fundamental-ablation profile, and fail-closed run manifests everywhere.

**Architecture:** Four P0 items harden the *provenance chain* of derived assets (market_env / industry_ranking): (P0#1) rebuild sector membership as genuinely PIT from CNINFO's per-stock industry-classification change history (证监会 CSRC standard, 门类 level) and derive `industry_ranking`/`market_adv_ratio` from it; (P0#2) add a generic `validate_derived_asset` gate that recomputes `upstream_roots` + `transform_code_hash` + `transform_config_hash` and fails a formal run on a STALE derived asset; (P0#4) make the two builders read daily via the canonical validated store (`require_valid_manifest=True`, fail-closed, no silent skip); (P0#3) require the prebuilt feature mainline for ALL formal named-profile runs, demoting live feature engineering to debug/smoke/exploratory. Three P1 items tidy the edges: (P1#5) split Market Env DataContract into required-PRICE / optional-ACCOUNT and actually enforce `column_contract` in `validate_asset_manifest`; (P1#6) freeze a `fundamental_ablation_v1` FeatureProfile with coverage contracts; (P1#7) migrate every remaining production downloader's run-manifest write to fail-closed `write_run_manifest_or_exit`.

**Tech Stack:** pandas/pyarrow, AKShare (CNINFO industry-change API), torch (only in training context), the existing data-governance stack (`asset_contract.py`, `contract.py`, `feature_profile.py`, `train_panel_*` gates).

---

## Scope note (P1#7 vs §十五)

`ChatGPT_v19.md` §十五 names ETF Flow / Macro / Fundamental / CNINFO / Analyst / Pledge as offenders and item #7 says "剩余…全部统一为 fail-closed", §十五 last line says "直接做统一 wrapper". This plan therefore migrates **all 12** remaining `write_run_manifest(...)` call sites under `scripts/production/` (a superset of the 6 named in earlier session notes): `download_etf_flow`, `download_macro`, `download_valuation`, `download_analyst`, `download_index_hist`, `download_datacenter`, `download_cninfo_announcements`, `download_minute`, `download_ipo_st`, `download_pledge`, `download_fundamentals`, `download_shareholder`. Each is a mechanical swap to `write_run_manifest_or_exit(...)` (same arguments).

## Pre-flight facts (verified in recon)

- `sector_membership.parquet` does NOT exist; `stock_sector_cache.csv` (5105 rows, 22 curated sectors) is the current-snapshot backfill the review flags. `preprocess_new_data.py:518-538` already prefers `sector_membership.parquet` with schema `[date, stock_code, sector_code]` and falls back to the snapshot with `sector_map_valid_from`.
- CNINFO `ak.stock_industry_change_cninfo(symbol, start_date, end_date)` WORKS from this machine (~0.5–1.8 s/call; columns `新证券简称/行业中类/行业大类/行业次类/行业门类/机构名称/行业编码/分类标准/分类标准编码/证券代码/变更日期`). It returns per-stock classification-change events across many standards; the 证监会 family = `证监会行业分类标准（2001）`, `证监会行业分类标准（2012）`, `中国上市公司协会上市公司行业分类标准` (the 2012 standard under its post-2022 name). The CSRC 门类 letters (A–S) are stable across the renames, so merging the three labels at 门类 level is legitimate.
- `market_env_daily.parquet` exists (0.1MB) but is a LEGACY build with NO `.manifest.json` — P0#2/P1#5 formal gates need it rebuilt by `build_market_env.py` first (P0#1 rebuild covers this).
- All 5530 daily files carry `{code}.manifest.json` sidecars, so `DataStorage.load_daily(..., require_valid_manifest=True)` (storage.py:581) is usable fail-closed for P0#4.
- Only `MARKET_ENV_ASSET` (broadcast_assets.py:34-41) declares `column_contract="market_env_daily"`; `INDUSTRY_ASSET` has none → P1#5 enforcement only newly gates market_env.
- Fundamental data: all 5530 files exist, sampled 400/400 non-empty, stock_coverage 1.0 and date_coverage 1.0 over 2024–25 → a 0.90/0.90 contract is meaningful AND passable.
- `stoke_ml/models/panel/__init__.py` imports torch-dependent modules → importing ANY `stoke_ml.models.panel.*` pulls torch. Keep torch-free code (data layer) out of that import path.
- `_require_prebuilt_mainline` (train_panel_universe.py:184) is called from `train_panel.py:228`; the formal branch is `formal_research and n_resolved > threshold and not prebuilt and not store_complete`. Tests asserting the small-formal-live-allowed behavior are in `tests/scripts/test_train_panel_universe.py:1057,1062`.

---

## Task 1: P1#5 — Split Market Env DataContract + enforce `column_contract`

**Files:**
- Modify: `stoke_ml/data/contract.py` (DataContract + MARKET_ENV)
- Modify: `stoke_ml/data/asset_contract.py` (`validate_asset_manifest` column_contract enforcement)
- Test: `tests/data/test_contract.py`, `tests/data/test_asset_contract.py`

- [ ] **Step 1: Write the failing contract test**

Append to `tests/data/test_contract.py`:

```python
def test_market_env_contract_price_required_account_optional():
    from stoke_ml.data.contract import get_contract
    c = get_contract("market_env_daily")
    price = {"high_low_ratio", "market_adv_ratio", "market_turnover_z"}
    account = {"mkt_cap_total_z", "avg_account_cap_z", "investor_new_num", "investor_new_z"}
    assert set(c.required_columns) == price
    assert set(c.optional_columns) == account
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_contract.py::test_market_env_contract_price_required_account_optional -q`
Expected: FAIL — `DataContract` has no attribute `optional_columns`.

- [ ] **Step 3: Add `optional_columns` to DataContract and split MARKET_ENV**

In `stoke_ml/data/contract.py`, add the field after `required_columns` (line 60):

```python
    required_columns: tuple[str, ...]
    #: Columns that MAY be present but are not required (e.g. the market_env
    #: ACCOUNT part is proxy-PIT and ablation-only — a file missing them is
    #: still schema-valid).  Enforcement: required columns must all be present;
    #: optional columns are never demanded (§v19-11).
    optional_columns: tuple[str, ...] = ()
```

Replace the `MARKET_ENV` block (contract.py:639-657) with:

```python
MARKET_ENV = DataContract(
    dataset_name="market_env_daily",
    primary_key=("date",),  # DatetimeIndex-backed broadcast file
    required_columns=(
        "high_low_ratio", "market_adv_ratio", "market_turnover_z",
    ),
    optional_columns=(
        "mkt_cap_total_z", "avg_account_cap_z",
        "investor_new_num", "investor_new_z",
    ),
    units={},
    price_basis="n/a",
    timezone="Asia/Shanghai",
    calendar="SSE_SZSE",
    allowed_missingness={
        "mkt_cap_total_z": "account part absent before account_stats coverage",
        "avg_account_cap_z": "account part absent before account_stats coverage",
        "investor_new_num": "account part absent before account_stats coverage",
        "investor_new_z": "account part absent before account_stats coverage",
    },
)
```

- [ ] **Step 4: Run the contract test to verify it passes**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_contract.py -q`
Expected: PASS.

- [ ] **Step 5: Write the failing column_contract enforcement test**

Append to `tests/data/test_asset_contract.py`:

```python
def test_validate_asset_manifest_enforces_column_contract(tmp_path):
    import pandas as pd
    from stoke_ml.data.asset_contract import (
        DataAssetContract, validate_asset_manifest, write_asset_manifest)
    asset = DataAssetContract(
        data_type="market_env_daily", partition="single_file",
        extent_column="date", column_contract="market_env_daily")
    p = tmp_path / "me.parquet"
    # only the 3 PRICE columns -> valid under the split contract
    price_only = pd.DataFrame({
        "high_low_ratio": [0.5], "market_adv_ratio": [0.6],
        "market_turnover_z": [1.0]}, index=pd.to_datetime(["2024-01-02"]))
    price_only.index.name = "date"
    write_asset_manifest(str(p), asset, price_only)
    assert validate_asset_manifest(str(p), asset)["ok"]

    # missing a REQUIRED price column -> must fail
    p2 = tmp_path / "bad.parquet"
    bad = pd.DataFrame({
        "high_low_ratio": [0.5], "market_turnover_z": [1.0],
        "mkt_cap_total_z": [0.0]}, index=pd.to_datetime(["2024-01-02"]))
    bad.index.name = "date"
    write_asset_manifest(str(p2), asset, bad)
    report = validate_asset_manifest(str(p2), asset)
    assert not report["ok"]
    assert any("missing_required_column:market_adv_ratio" in m for m in report["mismatches"])
```

- [ ] **Step 6: Run it to verify it fails**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_asset_contract.py::test_validate_asset_manifest_enforces_column_contract -q`
Expected: FAIL — the price-only file is flagged because `validate_asset_manifest` does not yet check columns at all (the mismatches list is empty for the present-only-file case → the second assertion fires, or the first does).

- [ ] **Step 7: Enforce column_contract in `validate_asset_manifest`**

In `stoke_ml/data/asset_contract.py`, add a module-level import at the top (contract.py imports only stdlib/numpy/pandas — no cycle, no torch):

```python
from stoke_ml.data.contract import get_contract as _get_contract
```

Inside `validate_asset_manifest` (asset_contract.py), in the `mismatches` construction block, after the `actual.update(_extent(...))` line, add:

```python
    if asset.column_contract:
        try:
            col_contract = _get_contract(asset.column_contract)
        except KeyError:
            mismatches.append(
                f"column_contract: manifest={asset.column_contract!r} has no "
                f"registered DataContract")
        else:
            missing_required = [
                c for c in col_contract.required_columns
                if c not in actual_df.columns]
            if missing_required:
                mismatches.append(
                    "missing_required_column:"
                    + ",".join(missing_required))
```

Note: this must run against `actual_df` (the already-loaded frame), so place it AFTER `actual_df` is defined and use `actual_df.columns`, not the `actual` dict (which only carries rows/schema_hash/extent).

- [ ] **Step 8: Run the enforcement test to verify it passes**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_asset_contract.py::test_validate_asset_manifest_enforces_column_contract -q`
Expected: PASS.

- [ ] **Step 9: Run the two affected suites**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_asset_contract.py tests/data/test_contract.py -q`
Expected: PASS (no existing test asserts the old 7-required contract).

- [ ] **Step 10: Commit**

```bash
git add stoke_ml/data/contract.py stoke_ml/data/asset_contract.py tests/data/test_contract.py tests/data/test_asset_contract.py
git commit -m "feat(v19-P1#5): split Market Env DataContract (PRICE required / ACCOUNT optional) and enforce column_contract in validate_asset_manifest"
```

---

## Task 2: P0#2 — `validate_derived_asset` lineage gate

**Files:**
- Modify: `stoke_ml/data/asset_contract.py` (pure helper — no torch import)
- Modify: `scripts/production/build_market_env.py` (extract `compute_lineage`)
- Modify: `scripts/production/train_panel_panel.py` (`_enforce_formal_manifests` market_env branch)
- Test: `tests/data/test_asset_contract.py`, `tests/scripts/test_train_panel_formal_manifest.py`, `tests/scripts/test_build_market_env.py`

- [ ] **Step 1: Write the failing `validate_derived_asset` test**

Append to `tests/data/test_asset_contract.py`:

```python
def test_validate_derived_asset_ok_and_stale():
    from stoke_ml.data.asset_contract import validate_derived_asset
    manifest = {
        "upstream_roots": {"daily": "AAA", "industry_ranking": "BBB"},
        "transform_code_hash": "ccc",
        "transform_config_hash": "ddd",
    }
    ok = validate_derived_asset(
        manifest,
        current_upstream_roots={"daily": "AAA", "industry_ranking": "BBB"},
        current_transform_code_hash="ccc",
        current_transform_config_hash="ddd")
    assert ok["ok"] and not ok["stale"]

    stale = validate_derived_asset(
        manifest,
        current_upstream_roots={"daily": "ZZZ", "industry_ranking": "BBB"},
        current_transform_code_hash="ccc",
        current_transform_config_hash="ddd")
    assert not stale["ok"] and stale["stale"]
    assert any("upstream_roots.daily" in m for m in stale["mismatches"])

    missing = validate_derived_asset(
        {"rows": 5}, current_upstream_roots={}, current_transform_code_hash="x",
        current_transform_config_hash="y")
    assert not missing["ok"] and missing["stale"]
    assert any("no recorded lineage" in m for m in missing["mismatches"])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_asset_contract.py::test_validate_derived_asset_ok_and_stale -q`
Expected: FAIL — `ImportError: cannot import name 'validate_derived_asset'`.

- [ ] **Step 3: Implement `validate_derived_asset` in asset_contract.py**

Append to `stoke_ml/data/asset_contract.py`:

```python
#: The three derivation-lineage keys a derived asset manifest records (§v18-7).
_DERIVED_LINEAGE_KEYS = (
    "upstream_roots", "transform_code_hash", "transform_config_hash",
)


def validate_derived_asset(
    manifest: dict,
    *,
    current_upstream_roots: dict,
    current_transform_code_hash: str,
    current_transform_config_hash: str,
) -> dict:
    """Freshness/lineage gate for a DERIVED asset manifest (§v19 P0#2).

    ``manifest`` is the on-disk ``{parquet}.manifest.json`` (a write-time
    snapshot of the derivation).  The ``current_*`` arguments are recomputed
    from the assets/code/config on disk RIGHT NOW.  Returns::

        {
          "ok": bool,
          "stale": bool,           # False when lineage is recorded and current
          "mismatches": [str, ...],
        }

    Integrity (the file matches its manifest) is a SEPARATE check — this is
    freshness: "was this derived asset built from the CURRENT upstreams /
    transform code / transform config".  Any recorded key that no longer
    matches its recomputed value means STALE → the asset must be rebuilt.
    A manifest with NO recorded lineage is stale-by-default (fail-closed):
    a pre-lineage derived asset cannot prove freshness.
    """
    recorded = {
        k: manifest.get(k)
        for k in _DERIVED_LINEAGE_KEYS
        if k in manifest
    }
    if len(recorded) != len(_DERIVED_LINEAGE_KEYS):
        return {
            "ok": False,
            "stale": True,
            "mismatches": [
                "no recorded lineage (missing "
                f"{sorted(set(_DERIVED_LINEAGE_KEYS) - set(recorded))}) — "
                "derived asset predates the lineage extension; rebuild"],
        }
    current = {
        "upstream_roots": current_upstream_roots,
        "transform_code_hash": current_transform_code_hash,
        "transform_config_hash": current_transform_config_hash,
    }
    mismatches: list[str] = []
    if recorded["upstream_roots"] != current["upstream_roots"]:
        diffs = []
        keys = sorted(set(recorded["upstream_roots"]) | set(current["upstream_roots"]))
        for k in keys:
            r = recorded["upstream_roots"].get(k)
            c = current["upstream_roots"].get(k)
            if r != c:
                diffs.append(f"{k}: recorded={r!r} current={c!r}")
        mismatches.append("upstream_roots changed: " + "; ".join(diffs))
    for key in ("transform_code_hash", "transform_config_hash"):
        if recorded[key] != current[key]:
            mismatches.append(
                f"{key}: recorded={recorded[key]!r} current={current[key]!r}")
    return {"ok": not mismatches, "stale": bool(mismatches), "mismatches": mismatches}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/data/test_asset_contract.py::test_validate_derived_asset_ok_and_stale -q`
Expected: PASS.

- [ ] **Step 5: Refactor build_market_env.py to expose `compute_lineage`**

In `scripts/production/build_market_env.py`, extract the config-hash computation and expose a single recompute entry point. Replace the inline `transform_config_hash=hash_json({...})` in `write_market_env` (lines 360-363) with a call to a new module-level function, and add:

```python
def _transform_config_hash(parts: dict) -> str:
    """Config identity of a market_env build (§v18-7): output columns + parts."""
    return hash_json({
        "output_columns": sorted(OUTPUT_COLUMNS),
        "parts": parts,
    })


def compute_lineage(data_dir: str, parts: dict) -> dict:
    """The §v19 P0#2 derivation lineage, recomputable at read time.

    ``write_market_env`` records this at write time; the formal gate
    (``train_panel_panel._enforce_formal_manifests``) recomputes it from the
    CURRENT on-disk upstreams / this builder's current source / the recorded
    ``parts`` and compares via :func:`asset_contract.validate_derived_asset`.
    """
    return {
        "upstream_roots": _upstream_roots(data_dir),
        "transform_code_hash": _transform_code_hash(),
        "transform_config_hash": _transform_config_hash(parts),
    }
```

Then in `write_market_env`, change:

```python
    write_asset_manifest(
        out_path, MARKET_ENV_ASSET, df, parts=parts,
        upstream_roots=_upstream_roots(data_dir),
        transform_code_hash=_transform_code_hash(),
        transform_config_hash=hash_json({
            "output_columns": sorted(OUTPUT_COLUMNS),
            "parts": parts,
        }),
    )
```

to:

```python
    write_asset_manifest(
        out_path, MARKET_ENV_ASSET, df, parts=parts,
        **compute_lineage(data_dir, parts),
    )
```

- [ ] **Step 6: Run build_market_env tests to verify the refactor is behavior-preserving**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_build_market_env.py -q`
Expected: PASS (lineage hashes byte-identical to before — `_transform_config_hash` produces exactly the same JSON as the inline `hash_json`).

- [ ] **Step 7: Wire the gate into `_enforce_formal_manifests`**

In `scripts/production/train_panel_panel.py`, the market_env branch (lines 1599-1608). After the existing `validate_asset_manifest(path, asset)` check passes for `ch == "market_env"`, add the lineage check:

```python
        elif ch in ("industry", "market_env"):
            fname = ("industry_returns.parquet" if ch == "industry"
                     else "market_env_daily.parquet")
            asset = INDUSTRY_ASSET if ch == "industry" else MARKET_ENV_ASSET
            rel = os.path.join(*source_dir(CHANNEL_SOURCE[ch]).split("/"), fname)
            path = os.path.join(data_dir, rel)
            if os.path.isfile(path):
                report = validate_asset_manifest(path, asset)
                if not report["ok"]:
                    problems.append(_fmt_manifest_problem(path, report))
                elif ch == "market_env":
                    from scripts.production.build_market_env import compute_lineage
                    from stoke_ml.data.asset_contract import validate_derived_asset
                    parts = (report["manifest"] or {}).get("parts", {})
                    lineage = validate_derived_asset(
                        report["manifest"] or {},
                        current_upstream_roots=compute_lineage(data_dir, parts)["upstream_roots"],
                        current_transform_code_hash=compute_lineage(data_dir, parts)["transform_code_hash"],
                        current_transform_config_hash=compute_lineage(data_dir, parts)["transform_config_hash"],
                    )
                    if lineage["stale"]:
                        problems.append(
                            f"{path}: DERIVED-ASSET STALE — "
                            + "; ".join(lineage["mismatches"])
                            + "; rebuild with build_market_env.py")
```

Add `compute_lineage` to the module-level import from `scripts.production.build_market_env` if the file already imports it; otherwise the function-local import above is fine (train_panel_panel is a torch context, so importing build_market_env — which imports `code_tree_hash.hash_json` → torch — is already safe here).

- [ ] **Step 8: Write the failing formal-gate stale-lineage test**

Append to `tests/scripts/test_train_panel_formal_manifest.py` (reuse its existing fixture/helper style — look at how the file builds `args`, `_panel_args`, and asserts SystemExit for manifest problems):

```python
def test_formal_gate_aborts_stale_market_env_lineage(tmp_path, monkeypatch):
    """§v19 P0#2: a market_env whose manifest lineage no longer matches the
    CURRENT upstreams must abort a formal run with a STALE diagnosis."""
    import json
    import pandas as pd
    from scripts.production.train_panel_panel import _enforce_formal_manifests
    from stoke_ml.data.asset_contract import write_asset_manifest
    from stoke_ml.data.broadcast_assets import MARKET_ENV_ASSET

    me_dir = tmp_path / "a_shares" / "market_breadth"
    me_dir.mkdir(parents=True)
    out = me_dir / "market_env_daily.parquet"
    df = pd.DataFrame({
        "high_low_ratio": [0.5], "market_adv_ratio": [0.6],
        "market_turnover_z": [1.0],
        "mkt_cap_total_z": [0.0], "avg_account_cap_z": [0.0],
        "investor_new_num": [1.0], "investor_new_z": [0.0],
    }, index=pd.to_datetime(["2024-01-02"]))
    df.index.name = "date"
    write_asset_manifest(
        str(out), MARKET_ENV_ASSET, df, parts={"price": {}, "account": {}},
        upstream_roots={"daily": "AAA"}, transform_code_hash="ccc",
        transform_config_hash="ddd")
    # force stale: the real compute_lineage returns DIFFERENT upstreams
    monkeypatch.setattr(
        "scripts.production.build_market_env.compute_lineage",
        lambda data_dir, parts: {
            "upstream_roots": {"daily": "ZZZ"},
            "transform_code_hash": "ccc", "transform_config_hash": "ddd"})

    with pytest.raises(SystemExit) as ei:
        _enforce_formal_manifests(["000001"], str(tmp_path), "2024-01-01",
                                  "2024-01-31", {"market_env"})
    assert "DERIVED-ASSET STALE" in str(ei.value)
```

Note: this test must NOT import torch (tests/scripts is the ml slice, torch is fine there) and must write a parquet + manifest into `tmp_path/a_shares/market_breadth/`. Adjust to match the module's actual `_panel_args`/fixture conventions when the implementer opens the file.

- [ ] **Step 9: Run the formal-gate tests to verify the wiring**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_train_panel_formal_manifest.py -q`
Expected: PASS (new test passes; existing formal tests unaffected — the lineage check only fires when the market_env manifest records lineage, and existing test fixtures either lack a market_env manifest or record matching lineage).

- [ ] **Step 10: Commit**

```bash
git add stoke_ml/data/asset_contract.py scripts/production/build_market_env.py scripts/production/train_panel_panel.py tests/data/test_asset_contract.py tests/scripts/test_train_panel_formal_manifest.py tests/scripts/test_build_market_env.py
git commit -m "feat(v19-P0#2): add validate_derived_asset lineage gate and wire market_env freshness into the formal manifest gate"
```

---

## Task 3: P0#4 — fail-closed canonical daily reads in `build_market_env.build_turnover_daily`

**Files:**
- Modify: `scripts/production/build_market_env.py` (`build_turnover_daily`)
- Test: `tests/scripts/test_build_market_env.py`

- [ ] **Step 1: Write the failing fail-closed test**

Append to `tests/scripts/test_build_market_env.py`:

```python
def test_build_turnover_daily_fails_closed_on_bad_daily(tmp_path):
    import os
    import pandas as pd
    from scripts.production.build_market_env import build_turnover_daily
    base = tmp_path / "a_shares"
    daily = base / "daily"
    daily.mkdir(parents=True)
    # one healthy stock
    ok = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "amount": [100.0, 200.0],
    })
    ok.to_parquet(daily / "000001.parquet", index=False)
    from stoke_ml.data.storage import _write_manifest
    _write_manifest(str(daily), "000001", ok, [{
        "source": "test", "adjust": "qfq",
        "start": "2024-01-02", "end": "2024-01-03", "rows": 2}],
        run_id="x")
    # a stock whose parquet exists but manifest is missing
    bad = pd.DataFrame({"date": ["2024-01-02"], "amount": [50.0]})
    bad.to_parquet(daily / "000002.parquet", index=False)
    with pytest.raises(SystemExit) as ei:
        build_turnover_daily(str(base))
    assert "manifest missing" in str(ei.value) or "require_valid_manifest" in str(ei.value)
```

(Adjust the manifest-write call to the real `DataStorage.save_daily`/`_write_manifest` signature the implementer finds in `storage.py`; the essence is: a present parquet without a valid manifest must abort, not silently skip.)

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_build_market_env.py::test_build_turnover_daily_fails_closed_on_bad_daily -q`
Expected: FAIL — current code warns-and-continues, returns a series, no SystemExit.

- [ ] **Step 3: Rewrite `build_turnover_daily` to fail-closed canonical reads**

Replace the body of `build_turnover_daily` (build_market_env.py:90-112) with a version that reads every daily file through the canonical validated store and aborts on any invalid file:

```python
def build_turnover_daily(base: str) -> pd.Series:
    """Sum 'amount' across all daily flat files per date -> z-scored turnover.

    ``base`` is the ``a_shares`` dir (``data_dir/a_shares``).  Same-day trade
    data, so this is the VERIFIED price part.  §v19 P0#4: a formal builder must
    NEVER silently skip an unreadable / un-manifested daily file — every daily
    parquet is read through ``DataStorage``'s canonical validated path
    (``require_valid_manifest=True``), and ANY file that fails validation aborts
    the whole build (a missing stock would silently change market turnover).
    """
    from stoke_ml.data.storage import DataStorage
    storage = DataStorage(os.path.dirname(base))
    codes = storage.list_stocks("a_shares")
    problems: list[str] = []
    amounts: list[pd.Series] = []
    for code in codes:
        try:
            d = storage.load_daily(code, "1970-01-01", "2099-12-31",
                                   require_valid_manifest=True)
        except (ValueError, OSError) as exc:
            problems.append(f"{code}: {exc}")
            continue
        if d is None or d.empty:
            continue
        amounts.append(d.groupby(d["date"])["amount"].sum())
    if problems:
        raise SystemExit(
            "build_market_env: %d/%d daily files FAILED canonical validation — "
            "refusing to build a turnover series over incomplete inputs "
            "(§v19 P0#4):\n  " + "\n  ".join(problems[:20]))
    if not amounts:
        return pd.Series(dtype="float64")
    tot = pd.concat(amounts).groupby(level=0).sum()
    return _z(tot)
```

- [ ] **Step 4: Run the fail-closed test to verify it passes**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_build_market_env.py -q`
Expected: PASS (new test aborts; existing turnover tests that use fully-manifested temp daily dirs still pass — if an existing test writes a daily dir without manifests, that test's fixture must be updated to write manifests, since the fail-closed behavior now requires them).

- [ ] **Step 5: Commit**

```bash
git add scripts/production/build_market_env.py tests/scripts/test_build_market_env.py
git commit -m "feat(v19-P0#4): build_market_env.build_turnover_daily reads daily via require_valid_manifest=True and fails closed on any invalid file"
```

---

## Task 4: P0#1 — genuine-PIT sector membership + PIT `industry_ranking` (+ fail-closed reads in `download_industry_ranking`)

**Files:**
- Create: `scripts/production/download_sector_membership.py`
- Create: `stoke_ml/data/csrc_gate.py` (pure CSRC 门类 → code map; data-layer, no torch)
- Modify: `scripts/production/download_industry_ranking.py` (PIT membership + fail-closed daily reads)
- Modify: `scripts/production/build_market_env.py` — NO change needed (already reads `industry_ranking.parquet`)
- Test: `tests/scripts/test_download_sector_membership.py` (new), `tests/scripts/test_industry_ranking_pit.py` (new)

### Data-source facts for the implementer

- AKShare: `ak.stock_industry_change_cninfo(symbol="002594", start_date="19900101", end_date="20260809")` returns columns `新证券简称 / 行业中类 / 行业大类 / 行业次类 / 行业门类 / 机构名称 / 行业编码 / 分类标准 / 分类标准编码 / 证券代码 / 变更日期`. Works from this machine (~0.5–1.8 s/call).
- The 证监会 (CSRC) classification family = 分类标准 ∈ {`证监会行业分类标准（2001）`, `证监会行业分类标准（2012）`, `中国上市公司协会上市公司行业分类标准`}. The CSRC 门类 letters (A–S) are identical across the renames → merging the three labels at 门类 level is legitimate.
- `行业门类` holds the top-level gate name (e.g. 金融业, 制造业). The gate letter is the FIRST character of `行业编码` for CSRC standards (e.g. `J66` → `J`), but the reliable universal source is the explicit gate-name→letter map below.
- CNINFO records CHANGE events only. A stock whose CSRC gate never changed yields ONE event (the most recent standard rename). Design rule (honest PIT): a stock's gate is asserted only from its FIRST CSRC event's `变更日期` forward; earlier dates stay unclassified (excluded from industry_ranking) and the per-year coverage is recorded in the manifest. This is genuinely PIT — the review's complaint was present-backfill, and this only asserts what CNINFO proves.

- [ ] **Step 1: Create `stoke_ml/data/csrc_gate.py`**

```python
"""CSRC 证监会 industry-gate (门类) mapping for the PIT sector membership.

The 证监会 industry classification (门类, single-letter A–S) is stable across
the 2001 / 2012 / 中国上市公司协会 renames of the standard, so a stock's gate
letter can be merged across all three CNINFO ``分类标准`` labels
(§v19 P0#1).  This map is pure (stdlib only) so the data layer and the
downloader share it without pulling torch.
"""

#: CSRC 门类 name (as CNINFO's ``行业门类`` reports it) → gate letter.
CSRC_GATE_CODES: dict[str, str] = {
    "农、林、牧、渔业": "A",
    "采矿业": "B",
    "制造业": "C",
    "电力、热力、燃气及水生产和供应业": "D",
    "建筑业": "E",
    "批发和零售业": "F",
    "交通运输、仓储和邮政业": "G",
    "住宿和餐饮业": "H",
    "信息传输、软件和信息技术服务业": "I",
    "金融业": "J",
    "房地产业": "K",
    "租赁和商务服务业": "L",
    "科学研究和技术服务业": "M",
    "水利、环境和公共设施管理业": "N",
    "居民服务、修理和其他服务业": "O",
    "教育": "P",
    "卫生和社会工作": "Q",
    "文化、体育和娱乐业": "R",
    "综合": "S",
    "制造业门类": "C",   # 2001-standard spelling variant
    "金融业门类": "J",
}

#: CNINFO ``分类标准`` labels that all denote the CSRC standard family.
CSRC_STANDARD_LABELS: frozenset[str] = frozenset({
    "证监会行业分类标准（2001）",
    "证监会行业分类标准（2012）",
    "中国上市公司协会上市公司行业分类标准",
})


def csrc_gate_code(gate_name: str) -> str | None:
    """Gate letter for a ``行业门类`` name, or None when unrecognized."""
    return CSRC_GATE_CODES.get(gate_name)
```

- [ ] **Step 2: Write the failing downloader-parse test**

Create `tests/scripts/test_download_sector_membership.py`:

```python
import pandas as pd

from scripts.production.download_sector_membership import parse_cninfo_events


def test_parse_cninfo_events_merges_csrc_labels_and_expands():
    # 002594-style events: one 2012-standard change + one 中国上市公司协会 rename
    events = pd.DataFrame({
        "证券代码": ["002594", "002594"],
        "行业门类": ["制造业", "制造业"],
        "分类标准": ["证监会行业分类标准（2012）",
                      "中国上市公司协会上市公司行业分类标准"],
        "行业编码": ["C36", "C36"],
        "变更日期": ["2011-06-30", "2024-02-08"],
    })
    df = parse_cninfo_events("002594", events)
    # only the CSRC gate letter C survives; intervals: [2011-06-30, 2024-02-07]
    # then [2024-02-08, end-of-time], both with sector_code C
    assert {"date", "stock_code", "sector_code", "sector_name"} <= set(df.columns)
    assert set(df["sector_code"]) == {"C"}
    assert (df["date"] >= "2011-06-30").all()
```

- [ ] **Step 3: Run it to verify it fails**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_download_sector_membership.py -q`
Expected: FAIL — `ModuleNotFoundError: scripts.production.download_sector_membership`.

- [ ] **Step 4: Implement `scripts/production/download_sector_membership.py`**

The downloader must: (1) enumerate the universe from `DataStorage.list_stocks()`; (2) fetch CNINFO change events per stock (8-worker pool, per-stock retry ×3 + backoff, resumable per-stock interval cache under `a_shares/sector_membership_pit/_stocks/{code}.json`); (3) parse to intervals via `parse_cninfo_events`; (4) expand intervals to per-date long `[date, stock_code, sector_code, sector_name]` over the stock's OWN daily dates (from `DataStorage.load_daily`); (5) write `a_shares/sector_membership.parquet` + asset manifest (a new `DataAssetContract` with `data_type="sector_membership"`, `column_contract=None`, `effective_date_policy="event_date"`, `contract_for_channel("sector_membership", ...)`), plus a per-year coverage audit; (6) `write_run_manifest_or_exit`.

Key parsing function (pure, unit-tested):

```python
def parse_cninfo_events(stock_code: str, events: pd.DataFrame) -> pd.DataFrame:
    """CNINFO change events → per-date long membership ``[date, stock_code,
    sector_code, sector_name]`` (证监会 门类 level, honest-PIT: gate asserted
    only from its first CSRC event's 变更日期 forward).
    """
    from stoke_ml.data.csrc_gate import CSRC_STANDARD_LABELS, csrc_gate_code
    if events is None or events.empty:
        return pd.DataFrame(columns=["date", "stock_code", "sector_code", "sector_name"])
    sub = events[events["分类标准"].isin(CSRC_STANDARD_LABELS)].copy()
    sub["变更日期"] = pd.to_datetime(sub["变更日期"], errors="coerce")
    sub = sub.dropna(subset=["变更日期"])
    if sub.empty:
        return pd.DataFrame(columns=["date", "stock_code", "sector_code", "sector_name"])
    # the most recent gate per change date (a rename can emit several rows same-day)
    sub = sub.sort_values("变更日期")
    latest = sub.groupby("变更日期").last().reset_index()
    latest["sector_code"] = latest["行业门类"].map(csrc_gate_code)
    latest["sector_name"] = latest["行业门类"]
    latest = latest.dropna(subset=["sector_code"]).sort_values("变更日期")
    if latest.empty:
        return pd.DataFrame(columns=["date", "stock_code", "sector_code", "sector_name"])
    # expand each interval to every day in the stock's daily span
    rows: list[dict] = []
    for i, (_, row) in enumerate(latest.iterrows()):
        start = row["变更日期"]
        end = latest["变更日期"].iloc[i + 1] - pd.Timedelta(days=1) \
            if i + 1 < len(latest) else pd.Timestamp("2099-12-31")
        rows.append({"date": start, "stock_code": stock_code,
                     "sector_code": row["sector_code"],
                     "sector_name": row["sector_name"], "_end": end})
    return pd.DataFrame(rows)
```

The interval→per-trading-day expansion (in `main()`, after loading each stock's daily dates `days`):

```python
    def _expand(intervals: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
        out = []
        for _, iv in intervals.iterrows():
            mask = (days >= iv["date"]) & (days <= iv["_end"])
            for d in days[mask]:
                out.append({"date": d, "stock_code": iv["stock_code"],
                            "sector_code": iv["sector_code"],
                            "sector_name": iv["sector_name"]})
        return pd.DataFrame(out)
```

The `main()` flow (resumable crawl + audit + write) follows the codebase's established downloader pattern (see `download_industry.py` for the run-manifest shape and `download_data.py` for the session-pool/rate-limiter shape). The per-year audit records `{year: fraction of stocks with a gate}` into the manifest's `coverage_by_year` field and logs it.

- [ ] **Step 5: Run the downloader-parse test to verify it passes**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_download_sector_membership.py -q`
Expected: PASS.

- [ ] **Step 6: Write the failing PIT industry_ranking test**

Create `tests/scripts/test_industry_ranking_pit.py`:

```python
import os
import pandas as pd

from scripts.production import download_industry_ranking as dir_mod


def test_build_industry_ranking_uses_pit_membership(tmp_path):
    """industry_ranking must derive sectors from sector_membership.parquet
    (per-date) rather than the current-snapshot cache."""
    base = tmp_path / "a_shares"
    (base / "daily").mkdir(parents=True)
    for code, sector in [("000001", "J"), ("600519", "C")]:
        df = pd.DataFrame({"date": ["2024-01-02", "2024-01-03"],
                           "pct_change": [1.0, -0.5]})
        df.to_parquet(base / "daily" / f"{code}.parquet", index=False)
    mem = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03", "2024-01-02", "2024-01-03"],
        "stock_code": ["000001", "000001", "600519", "600519"],
        "sector_code": ["J", "J", "C", "C"],
        "sector_name": ["金融业", "金融业", "制造业", "制造业"],
    })
    mem.to_parquet(base / "sector_membership.parquet", index=False)
    df = dir_mod.build_industry_ranking(str(base))
    assert set(df["sector_code"]) == {"J", "C"}
    assert set(df["sector_name"]) == {"金融业", "制造业"}
```

- [ ] **Step 7: Run it to verify it fails**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_industry_ranking_pit.py -q`
Expected: FAIL — `build_industry_ranking` does not exist / reads the snapshot cache.

- [ ] **Step 8: Refactor `download_industry_ranking.py`**

Extract the ranking computation into a module-level `build_industry_ranking(base: str) -> pd.DataFrame` that (a) loads `sector_membership.parquet` when present (join daily rows on `(date, stock_code)` → `sector_code`/`sector_name`), else falls back to the snapshot cache; (b) reads each daily file through `DataStorage.load_daily(code, "1970-01-01", "2099-12-31", require_valid_manifest=True)` and FAILS on any invalid file (§v19 P0#4 — same pattern as Task 3); (c) computes the exact same sector aggregates as today (sector equal-weighted return / std / n_stocks, up/down counts, leader, rank). `main()` then calls `build_industry_ranking` and writes the parquet + run manifest. The `sector_code_map` SEC#### assignment is replaced by the membership's `sector_code` directly (the market_env `market_adv_ratio` only groups by sector_code, so the semantic change is absorbed by the downstream z-score).

The P0#4 fail-closed daily read (replacing lines 61-72):

```python
    from stoke_ml.data.storage import DataStorage
    storage = DataStorage(os.path.dirname(base))
    problems: list[str] = []
    frames: list[pd.DataFrame] = []
    for code in codes:
        try:
            d = storage.load_daily(code, "1970-01-01", "2099-12-31",
                                   require_valid_manifest=True)
        except (ValueError, OSError) as exc:
            problems.append(f"{code}: {exc}")
            continue
        if d is None or d.empty:
            continue
        frames.append(d)
    if problems:
        raise SystemExit(
            "download_industry_ranking: %d daily files FAILED canonical "
            "validation — refusing to build industry ranking over incomplete "
            "inputs (§v19 P0#4):\n  " + "\n  ".join(problems[:20]))
```

- [ ] **Step 9: Run the PIT industry_ranking test to verify it passes**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_industry_ranking_pit.py -q`
Expected: PASS.

- [ ] **Step 10: Rebuild the derived assets (data operation — run once)**

Run the new downloader (8-worker CNINFO crawl, resumable):

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_sector_membership.py > logs/v19_sector_membership.log 2>&1
```

Then rebuild industry ranking (PIT) and market_env (which reads industry_ranking):

```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_industry_ranking.py > logs/v19_industry_ranking.log 2>&1
PYTHONPATH=. ./.venv/Scripts/python scripts/production/build_market_env.py > logs/v19_market_env.log 2>&1
```

Inspect the audit: `Read logs/v19_sector_membership.log` — confirm per-year coverage and that `sector_membership.parquet` exists with `[date, stock_code, sector_code, sector_name]`. Confirm `industry_ranking.parquet` + `market_env_daily.parquet` + their `.manifest.json` sidecars now exist (market_env now carries lineage fields).

Expected: three log files end with success lines; `a_shares/sector_membership.parquet`, `a_shares/industry_ranking.parquet`, `a_shares/market_breadth/market_env_daily.parquet` + `.manifest.json` all exist.

- [ ] **Step 11: Commit**

```bash
git add scripts/production/download_sector_membership.py stoke_ml/data/csrc_gate.py scripts/production/download_industry_ranking.py tests/scripts/test_download_sector_membership.py tests/scripts/test_industry_ranking_pit.py
git commit -m "feat(v19-P0#1): build genuine-PIT sector membership from CNINFO CSRC gate changes; derive industry_ranking per-date; fail-closed canonical daily reads"
```

---

## Task 5: P0#3 — formal runs always require the prebuilt mainline

**Files:**
- Modify: `scripts/production/train_panel_universe.py` (`_require_prebuilt_mainline`)
- Modify: `scripts/production/train_panel.py` (pass-through — check the call site)
- Test: `tests/scripts/test_train_panel_universe.py`

- [ ] **Step 1: Write the failing 收口 test**

In `tests/scripts/test_train_panel_universe.py`, replace the body of `test_require_prebuilt_mainline_small_formal_allowed` (line 1057) so the small formal run is now REFUSED:

```python
def test_require_prebuilt_mainline_small_formal_refused(tp):
    # §v19 P0#3 收口: a small formal run may NO LONGER use live feature
    # engineering — formal named-profile research is prebuilt-only.
    with pytest.raises(SystemExit):
        tp._require_prebuilt_mainline("random", None, n_resolved=500,
                                      formal_research=True)
```

and add (next to `test_require_prebuilt_mainline_large_formal_allows_prebuilt_and_store`):

```python
def test_require_prebuilt_mainline_formal_allows_prebuilt_any_size(tp):
    tp._require_prebuilt_mainline("random", "data/features_panel",
                                  n_resolved=500, formal_research=True)
    tp._require_prebuilt_mainline("random", None, n_resolved=500,
                                  formal_research=True, store_complete=True)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_train_panel_universe.py -k require_prebuilt_mainline -q`
Expected: FAIL — small-formal-live still allowed.

- [ ] **Step 3: 收口 the gate**

In `scripts/production/train_panel_universe.py`, change the formal branch (lines 216-227):

```python
    if formal_research and not prebuilt and not store_complete:
        raise SystemExit(
            f"formal research requires the prebuilt feature mainline: live "
            f"feature engineering is reserved for debug / smoke / exploratory "
            f"runs (§v19 P0#3).  Run "
            f"scripts/production/build_features.py --panel-mode to build "
            f"data/features_panel once, then re-run with --prebuilt "
            f"data/features_panel (or point --panel-store at a previously-built "
            f"complete store).  To run live anyway, pass --no-formal "
            f"(exploratory) or --no-require-quality-gate (dev smoke) explicitly."
        )
```

Keep the `universe == "all"` branch unchanged. The `threshold`/`n_resolved` parameters remain in the signature (the call site in `train_panel.py:228` passes them) but are no longer consulted by the formal branch; update the function docstring to say the threshold is retained only for the universe-all branch and signature compatibility.

- [ ] **Step 4: Run the prebuilt-mainline tests to verify the 收口**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_train_panel_universe.py -k require_prebuilt_mainline -q`
Expected: PASS. Then run the whole file to catch any other test that relied on small-formal-live:

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_train_panel_universe.py -q`
Expected: PASS (any test asserting a live formal run that now aborts must be updated to pass `prebuilt`/`store_complete` or `formal_research=False`).

- [ ] **Step 5: Commit**

```bash
git add scripts/production/train_panel_universe.py scripts/production/train_panel.py tests/scripts/test_train_panel_universe.py
git commit -m "feat(v19-P0#3): formal research runs always require the prebuilt feature mainline; live FE demoted to debug/smoke/exploratory"
```

---

## Task 6: P1#6 — frozen `fundamental_ablation_v1` FeatureProfile

**Files:**
- Modify: `stoke_ml/config/feature_profile.py`
- Test: `tests/config/test_feature_profile.py`, `tests/scripts/test_train_panel_gates.py` (if present)

- [ ] **Step 1: Write the failing profile test**

Append to `tests/config/test_feature_profile.py`:

```python
def test_fundamental_ablation_v1_profile():
    from stoke_ml.config.feature_profile import FEATURE_PROFILES, profile_for
    prof = profile_for("fundamental_ablation_v1")
    assert prof is not None
    assert "fundamental" in prof.required_channels
    # superset of headline_v1
    base = profile_for("headline_v1")
    assert set(base.required_channels) <= set(prof.required_channels)
    assert prof.vintage_policy == "allow-revised"
    fc = prof.coverage_contracts["fundamental"]
    assert fc.metric == "stock_coverage"
    assert fc.threshold == 0.90
    assert fc.requires == ("date_coverage", 0.90)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/config/test_feature_profile.py::test_fundamental_ablation_v1_profile -q`
Expected: FAIL — `profile_for("fundamental_ablation_v1")` is None.

- [ ] **Step 3: Add the frozen profile**

In `stoke_ml/config/feature_profile.py`, add `fundamental_ablation_v1` to `FEATURE_PROFILES` (after `headline_v1`, line 361). Build it from the headline base to guarantee the superset:

```python
    "fundamental_ablation_v1": FeatureProfile(
        name="fundamental_ablation_v1",
        required_channels=FEATURE_PROFILES["headline_v1"].required_channels
            + ("fundamental",),
        coverage_contracts={
            **FEATURE_PROFILES["headline_v1"].coverage_contracts,
            # §v19 P1#6: the fundamental ablation is only meaningful when the
            # fundamental channel is actually covered — stock coverage >= 0.90
            # AND date/report coverage >= 0.90 (the composite form, mirroring
            # the era_coverage contract).  fundamental is latest_revised-sourced,
            # so the profile is validated under the allow-revised policy.
            "fundamental": CoverageContract(
                "stock_coverage", 0.90,
                requires=("date_coverage", 0.90)),
        },
        vintage_policy="allow-revised",
    ),
```

Also add a short module comment above the new entry explaining that `fundamental` is a `latest_revised` channel (denied under `revision-safe`) and this profile is the explicit, frozen opt-in — superseding the `--allow-fundamental-ablation` CLI override as the canonical ablation route.

- [ ] **Step 4: Run the profile test to verify it passes**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/config/test_feature_profile.py -q`
Expected: PASS.

- [ ] **Step 5: Verify the vintage-policy match enforcement works for the new profile**

Add a gate-level assertion (in `tests/scripts/test_train_panel_gates.py` if it exists, else `tests/config/test_feature_profile.py`): a run with `--feature-profile fundamental_ablation_v1` and `--vintage-policy revision-safe` must be refused by `train_panel_gates` (which enforces `args_vintage == profile.vintage_policy`). Assert the enforced policy equals `allow-revised`.

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_train_panel_gates.py -q` (or the file where you placed the assertion)
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add stoke_ml/config/feature_profile.py tests/config/test_feature_profile.py
git commit -m "feat(v19-P1#6): freeze fundamental_ablation_v1 FeatureProfile with stock/date coverage contracts under allow-revised"
```

---

## Task 7: P1#7 — fail-closed run manifests for all remaining production downloaders

**Files (all under `scripts/production/`):**
- Modify: `download_analyst.py`, `download_cninfo_announcements.py`, `download_datacenter.py`, `download_etf_flow.py`, `download_fundamentals.py`, `download_index_hist.py`, `download_macro.py`, `download_minute.py`, `download_ipo_st.py`, `download_pledge.py`, `download_shareholder.py`, `download_valuation.py`
- Test: `tests/scripts/test_download_run_manifest.py` (+ the smoke suite)

- [ ] **Step 1: Write a small failing guard test for one migrated script**

In `tests/scripts/test_download_run_manifest.py`, add a helper-import assertion that each migrated downloader imports `write_run_manifest_or_exit` (the fail-closed symbol) and NOT the legacy bare `write_run_manifest` call in its `main`:

```python
import importlib

MIGRATED = [
    "download_analyst", "download_cninfo_announcements", "download_datacenter",
    "download_etf_flow", "download_fundamentals", "download_index_hist",
    "download_macro", "download_minute", "download_ipo_st", "download_pledge",
    "download_shareholder", "download_valuation",
]

def test_all_production_downloaders_use_fail_closed_run_manifest():
    import inspect
    for mod in MIGRATED:
        m = importlib.import_module(f"scripts.production.{mod}")
        src = inspect.getsource(m)
        assert "write_run_manifest_or_exit" in src, f"{mod} not migrated"
```

(If any of these modules import torch at top level and break under tests/data, the importlib check lives in tests/scripts — the ml slice — where torch is installed. Confirm the modules import cleanly there.)

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_download_run_manifest.py -q`
Expected: FAIL — several modules still call the legacy symbol.

- [ ] **Step 3: Migrate each downloader**

For each of the 12 files, find the `try: write_run_manifest(...) except Exception: logger.warning(...)` block in `main()` and replace the call with `write_run_manifest_or_exit(...)` using the SAME keyword arguments, dropping the now-unneeded try/except (the wrapper logs and `sys.exit(1)` on failure). Also update the import line:

```python
from stoke_ml.data.download_manifest import write_run_manifest_or_exit
```

Concretely — `download_etf_flow.py:85` currently:

```python
    try:
        write_run_manifest(
            data_dir, "a_shares/etf_flow",
            requested=..., failed=..., complete=...,
        )
    except Exception as e:
        logger.warning("run manifest write failed: %s", e)
```

becomes:

```python
    write_run_manifest_or_exit(
        data_dir, "a_shares/etf_flow",
        requested=..., failed=..., complete=...,
    )
```

Repeat for `download_macro.py:62`, `download_valuation.py:158`, `download_analyst.py:74`, `download_index_hist.py:150`, `download_datacenter.py:261`, `download_cninfo_announcements.py:272`, `download_minute.py:240`, `download_ipo_st.py:57`, `download_pledge.py:156`, `download_fundamentals.py:110`, `download_shareholder.py:126` (line numbers are the call sites found in recon — re-locate by grepping `write_run_manifest\(`).

- [ ] **Step 4: Run the guard test to verify all are migrated**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_download_run_manifest.py -q`
Expected: PASS.

- [ ] **Step 5: Run the smoke suite**

Run: `PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/ -q`
Expected: PASS (any existing test that asserted the warn-on-failure behavior for a migrated downloader must be updated to expect `SystemExit`; the guard test above is the new contract).

- [ ] **Step 6: Commit**

```bash
git add scripts/production/download_analyst.py scripts/production/download_cninfo_announcements.py scripts/production/download_datacenter.py scripts/production/download_etf_flow.py scripts/production/download_fundamentals.py scripts/production/download_index_hist.py scripts/production/download_macro.py scripts/production/download_minute.py scripts/production/download_ipo_st.py scripts/production/download_pledge.py scripts/production/download_shareholder.py scripts/production/download_valuation.py tests/scripts/test_download_run_manifest.py
git commit -m "feat(v19-P1#7): fail-closed run manifests (write_run_manifest_or_exit) for all remaining production downloaders"
```

---

## Self-Review (run after writing — fix inline)

**Spec coverage (ChatGPT_v19.md §二十一):**
- P0#1 方案 A (真 PIT sector membership + rebuild) → Task 4 (CNINFO downloader + PIT industry_ranking + rebuild). Note: `market_adv_ratio` becomes genuinely PIT after the rebuild, satisfying the review's "做不到就暂时从 strict headline 移除" (we DID it, so no removal needed).
- P0#2 (`upstream_roots + transform_code_hash + transform_config_hash` verified) → Task 2 (`validate_derived_asset` + formal-gate wiring).
- P0#3 (formal named-profile 一律 prebuilt; live demoted) → Task 5.
- P0#4 (no silent skip; canonical validated input) → Task 3 (build_market_env) + Task 4 Step 8 (download_industry_ranking).
- P1#5 (Price/Account contract split + real column_contract execution) → Task 1.
- P1#6 (fundamental_ablation_v1 frozen profile + coverage contract) → Task 6.
- P1#7 (剩余 run manifest 全部 fail-closed) → Task 7 (all 12).

**Placeholder scan:** thresholds are concrete (0.90/0.90); the CNINFO parse/expand code and the fail-closed read code are given verbatim; the 12-file migration is enumerated by name + line. The one deliberate abstraction is "match the existing `_panel_args`/fixture conventions in test_train_panel_formal_manifest.py" — the implementer opens that file and reuses its pattern (acceptable: the exact fixture name varies and is mechanical).

**Type consistency:** `sector_membership.parquet` columns `[date, stock_code, sector_code, sector_name]` are consistent across Task 4 Step 2/4/6/8; `validate_derived_asset` signature matches Task 2 Step 3 and Step 7; `MARKET_ENV.required_columns` (3 price) / `optional_columns` (4 account) match Task 1 Step 1/3/5.

**Sequencing:** Task 1 (P1#5 contract) is first because Task 2's formal gate and Task 3's fail-closed reads depend on a coherent contract layer. Task 4 (P0#1 rebuild) must run its Step 10 data-rebuild BEFORE any formal training, because Tasks 2's gate requires a lineage-bearing market_env manifest that only the rebuild produces. Tasks 5/6/7 are independent of the data rebuild.

**Execution handoff:** after all code tasks pass review, run the full verification (smoke + `check_docs_consistency.py`), update CONTEXT.md docs for the formal-prebuilt-only rule and the PIT sector membership, then push + watch CI.
