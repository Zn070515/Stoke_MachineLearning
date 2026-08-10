# Sector/Derived Lineage Closeout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining `ChatGPT_v19.md` Sector/Derived lineage items — P0.2 (no silent snapshot fallback), P0.3 (full DataAsset/lineage for `sector_membership` + `industry_ranking`), P0.4 (completion/coverage semantics), §十八 (narrow CNINFO KeyError), §十九 (cache versioning), §二十 (feature metadata), and P0.5 (quality-gate rerun) — so the derived chain `Canonical Daily + CNINFO Sector Membership → Industry Ranking → Market Env` is fully provenance-closed.

**Architecture:** Every derived asset records a write-time lineage snapshot (`upstream_roots` + `transform_code_hash` + `transform_config_hash`, reusing `asset_contract.validate_derived_asset`); the formal gate recomputes and fails on staleness. The sector chain is closed with *explicit two-level validation* (§十四): `industry_ranking` records its upstream (`daily` manifest-root + `sector_membership` asset root), and `market_env` records `industry_ranking` as an upstream — so a `sector_membership` change without an `industry_ranking` rebuild is caught when the formal gate validates `industry_ranking`'s own freshness. Snapshot fallback becomes a debug-only opt-in flag that declares `pit_alignment="proxy"`.

**Tech Stack:** pandas/pyarrow, the existing data-governance stack (`asset_contract.py`, `broadcast_assets.py`, `download_manifest.py`, `data_quality_gate.dataset_fingerprint`, `code_tree_hash.hash_json`), AKShare CNINFO.

---

## Pre-flight facts (verified in recon)

- `sector_membership.parquet` + `.manifest.json` EXIST (16M rows, `[date, stock_code, sector_code, sector_name]`, `coverage_by_year` recorded). Current coverage is universe-denominated (deflates early years).
- `industry_ranking.parquet` exists but has **NO** `.manifest.json`. `download_industry_ranking.py` reads `sector_membership.parquet` via plain `pd.read_parquet` (no `validate_asset_manifest`) and still defaults to the snapshot-cache fallback (lines 115-140).
- `download_sector_membership.py`: `complete.add(code)` fires on fetch+parse success (line 298) BEFORE the daily expansion (lines 310-331); daily read at line 315 lacks `require_valid_manifest=True` and warns-continues on failure; `_rate_limited_fetch` catches ALL `KeyError` (line 200); the per-stock intervals cache carries only `{"intervals": [...]}` (no version).
- `build_market_env.build_industry_advance` (lines 123-145) reads `industry_ranking.parquet` via plain `pd.read_parquet` with NO manifest validation.
- `train_panel_panel._enforce_formal_manifests` (line 1472) already validates `market_env` + its derived lineage (lines 1612-1634) but does NOT validate `industry_ranking`'s own lineage nor sector coverage.
- `test_industry_ranking_pit.py:100` (`test_build_industry_ranking_falls_back_to_snapshot_cache`) asserts the DEFAULT snapshot fallback — P0.2 makes the default FAIL, so this test must switch to `allow_snapshot_fallback=True`.
- `test_build_market_env.py` `_write_industry` (lines 71-83) writes a bare `industry_ranking.parquet` WITHOUT a manifest — Task 3's `check_asset_read(require_valid_manifest=True)` requires the fixture to write a manifest.
- `code_tree_hash.hash_json` is torch-free (imports hashlib/json/os only). `dataset_fingerprint(root, ["daily"])` gives the daily manifest-root Merkle digest.
- `SECTOR_MEMBERSHIP_ASSET` is defined in `download_sector_membership.py`; the codebase pattern is "asset contract next to its writer", so `INDUSTRY_RANKING_ASSET` goes in `download_industry_ranking.py`.

---

## Task 1: Harden `download_sector_membership.py` (§十八 + §十九 + §十五 + §十六)

**Files:**
- Modify: `scripts/production/download_sector_membership.py`
- Modify: `tests/scripts/test_download_sector_membership.py`

### Step 1 (§十八): narrow the CNINFO KeyError catch

In `_rate_limited_fetch` (line 197-203), change the `except KeyError:` handler so ONLY the known empty-result signature is normalized, everything else re-raises:

```python
    try:
        return ak.stock_industry_change_cninfo(
            symbol=stock_code, start_date=start_date, end_date=end_date)
    except KeyError as exc:
        # §v19 §十八: akshare's known empty-result signature is a 0×0 frame then
        # temp_df["变更日期"] → KeyError('变更日期').  ONLY this exact key means
        # "stock has no CSRC records".  A schema drift that fails a DIFFERENT
        # column index must NOT be swallowed as a legit no-gate stock — re-raise
        # so _ensure_parseable() / the retry loop handle it.
        if not (exc.args and exc.args[0] == "变更日期"):
            raise
        logger.info("sector_membership[%s]: CNINFO returned no records — "
                    "legit-empty (no CSRC gate), not a failure", stock_code)
        return _empty_cninfo_frame()
```

### Step 2 (§十九): versioned intervals cache

Replace `_write_intervals_cache` / `_load_intervals_cache` / `_fetch_stock` so the cache payload carries the parser/config identity and a mismatched cache is refetched:

- New module constant: `_CACHE_VERSION = "v2"`.
- `_parser_hash()` — a deterministic digest of the parse logic: `hashlib.sha256((repr(sorted(CSRC_GATE_CODES.items())) + "|" + repr(sorted(CSRC_STANDARD_LABELS)) + "|" + inspect.getsource(parse_cninfo_events)).encode("utf-8")).hexdigest()[:12]` (import `inspect`, `hashlib`; import the two csrc_gate symbols at module level — they are pure, torch-free).
- `_write_intervals_cache(path, intervals, start_date, end_date)` writes:
  ```python
  {"cache_version": _CACHE_VERSION, "parser_hash": _parser_hash(),
   "source": "cninfo", "start_date": start_date, "end_date": end_date,
   "fetched_at": <iso utc now>, "intervals": [...]}
  ```
- `_load_intervals_cache(path)` returns `(intervals_df, meta_dict)` where `meta_dict` is the top-level JSON minus `intervals`.
- `_fetch_stock`: on a cache-file present, load `(intervals, meta)` and verify `meta.get("cache_version") == _CACHE_VERSION`, `meta.get("parser_hash") == _parser_hash()`, `meta.get("source") == "cninfo"`, `meta.get("start_date") == start_date`, `meta.get("end_date") == end_date`. ANY mismatch → treat as no cache: `os.remove(path)` (if present) and refetch. A legacy cache (no `cache_version`) is a mismatch → refetch (one-time full crawl is the honest fail-closed behavior).

### Step 3 (§十五): completion requires the full pipeline

Restructure `main()` so `complete` is only granted after fetch + parse + canonical daily formal read + expansion (or a legitimate no-gate), and a daily failure lands in `failed` (not a silent skip):

- Fetch step (lines 289-304) stays — it fills `intervals_by_code` and a `fetch_failed` list.
- Replace the expansion loop (lines 309-331) with one that:
  - `intervals.empty` (legit no-gate) → `complete.add(code)`; skip daily read.
  - else `storage.load_daily(code, "1970-01-01", "2099-12-31", require_valid_manifest=True)`; on `(ValueError, OSError)` → `failed.append(code)` + log a FAILED warning (not "skipping") and continue.
  - daily `None`/empty → `failed.append(code)` (a stock with daily history is expected; empty is a defect).
  - `_expand(intervals, days)`; append to `frames` when non-empty; `complete.add(code)`.
- The final `failed` = `fetch_failed + expansion_failed` (dedup, keep order). `write_run_manifest_or_exit(..., requested=codes, failed=failed, complete=complete, ...)` unchanged.

### Step 4 (§十六): active-denominator coverage

- Add `_active_stocks_by_year(daily_dir: str, codes: list[str]) -> dict[int, int]` that reads each `daily/{code}.manifest.json` `start`/`end` (ISO strings) and counts, per calendar year, how many stocks have `start_year <= year <= end_year`. A missing/unreadable manifest for a code is ignored (that stock is not active — conservative).
- Change `_coverage_by_year(membership, active_by_year)` signature: for every year in `active_by_year`, `coverage = (distinct stocks with membership rows that year) / active_by_year[year]`; years with no membership rows report `0.0`; `membership` empty → return `{str(y): 0.0 for y in active_by_year}`. Return `{str(int(y)): float(round(v,4))}`.
- In `main()`: `daily_dir = os.path.join(base, "daily")`; `active_by_year = _active_stocks_by_year(daily_dir, codes)`; call `_coverage_by_year(membership, active_by_year)`.

### Step 5: tests

- §十八: new test — a `KeyError("something_else")` (a fake schema drift) from the mocked akshare must NOT be normalized: `_fetch_stock` raises (the retry loop then marks failed). Existing `test_fetch_stock_normalizes_true_empty_to_cached_empty` must still pass (it raises `KeyError('变更日期')`).
- §十九: new test — a cache written with `cache_version`/`parser_hash`/range matching is reused (network not hit); a cache with `cache_version="v1"` (or missing version) triggers a refetch (network hit). Update any existing cache-format assertions.
- §十五: new test — `main()`-level behavior is hard to unit-test (network + 5530 stocks); instead unit-test the decision: a fake `intervals_by_code` where one code has intervals and its daily read raises → that code is in `failed`, not `complete`. Refactor the expansion into a helper `_expand_stock(intervals, storage, code) -> pd.DataFrame` (raises on daily failure, returns empty frame for no-gate) and test that helper directly with a monkeypatched storage whose `load_daily(..., require_valid_manifest=True)` raises.
- §十六: new test — `_coverage_by_year(membership, {2010: 10, 2011: 10})` with membership covering 5 stocks in 2010 and 0 in 2011 → `{"2010": 0.5, "2011": 0.0}`.

### Step 6: run suites + commit

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_download_sector_membership.py -q
```
Expected: PASS. Then `git commit -m "fix(v19): sector_membership — narrow CNINFO KeyError (§十八), versioned intervals cache (§十九), completion-after-expansion (§十五), active-stock coverage denominator (§十六)"`.

---

## Task 2: Fail-closed `download_industry_ranking.py` + `INDUSTRY_RANKING_ASSET` lineage (P0.2 + §十三 + §十四)

**Files:**
- Modify: `scripts/production/download_industry_ranking.py`
- Modify: `tests/scripts/test_industry_ranking_pit.py`

### Step 1 (P0.2): default-FAIL membership, opt-in snapshot fallback

- Change `build_industry_ranking(base)` → `build_industry_ranking(base, *, allow_snapshot_fallback: bool = False)` and return `(df, provenance)` where `provenance = {"membership_source": "pit" | "snapshot_fallback", "pit_alignment": "verified" | "proxy"}`.
- Membership-present branch: provenance `pit_alignment="verified"`, `membership_source="pit"`.
- Membership-absent branch: if `not allow_snapshot_fallback` → `raise SystemExit("download_industry_ranking: sector_membership.parquet missing — snapshot fallback is disabled by default (§v19 P0.2); pass --allow-snapshot-sector-fallback to force the legacy current-snapshot cache (produces pit_alignment='proxy', never strict headline)")`. If allowed → use the snapshot cache as today, provenance `pit_alignment="proxy"`, `membership_source="snapshot_fallback"`.
- `main()`: add `ap.add_argument("--allow-snapshot-sector-fallback", action="store_true")`; call `build_industry_ranking(base, allow_snapshot_fallback=args.allow_snapshot_sector_fallback)`.

### Step 2 (§十三): validate `sector_membership` before use

In the membership-present branch, before `pd.read_parquet(membership_path)`:
```python
from stoke_ml.data.asset_contract import validate_asset_manifest
from scripts.production.download_sector_membership import SECTOR_MEMBERSHIP_ASSET
report = validate_asset_manifest(membership_path, SECTOR_MEMBERSHIP_ASSET)
if not report["ok"]:
    raise SystemExit(
        "download_industry_ranking: sector_membership.parquet FAILED its asset "
        "manifest check (§v19 P0.3): " + "; ".join(report.get("mismatches") or [])
        + "; re-run download_sector_membership.py")
```

### Step 3 (§十四): `INDUSTRY_RANKING_ASSET` + lineage manifest

- Add `INDUSTRY_RANKING_ASSET` to `download_industry_ranking.py` (writer-owned contract, mirroring `SECTOR_MEMBERSHIP_ASSET`):
  ```python
  from stoke_ml.data.asset_contract import DataAssetContract, contract_for_channel
  INDUSTRY_RANKING_ASSET: DataAssetContract = contract_for_channel(
      "industry_ranking",
      data_type="industry_ranking",
      partition="single_file",
      extent_column="date",
      effective_date_policy="record_date",
  )
  ```
- Add `_file_sha256(path)` and `_transform_code_hash()` mirroring `build_market_env.py` (lines 294-345), and a `compute_lineage(data_dir, provenance)` public helper:
  ```python
  def compute_lineage(data_dir, provenance):
      from stoke_ml.models.panel.code_tree_hash import hash_json
      from scripts.production.data_quality_gate import dataset_fingerprint
      base = os.path.join(data_dir, "a_shares")
      mem_path = os.path.join(base, "sector_membership.parquet")
      return {
          "upstream_roots": {
              "daily": dataset_fingerprint(data_dir, ["daily"]),
              "sector_membership": _file_sha256(mem_path),
          },
          "transform_code_hash": _transform_code_hash(),
          "transform_config_hash": hash_json({
              "output_columns": sorted({c for c in result_columns}),
              "membership_source": provenance["membership_source"],
              "allow_snapshot_fallback": provenance["membership_source"] == "snapshot_fallback",
          }),
      }
  ```
  (`result_columns` = the actual output columns of the built ranking.)
- In `main()` after building `result`, write the manifest:
  ```python
  from stoke_ml.data.asset_contract import AtomicCommit, write_asset_manifest
  provenance = ...  # from build_industry_ranking
  with AtomicCommit(output_path) as ac:
      result.to_parquet(ac.tmp_path, index=False, compression="lz4")
  write_asset_manifest(
      output_path, INDUSTRY_RANKING_ASSET, result,
      upstream_roots=..., transform_code_hash=..., transform_config_hash=...,
      membership_source=provenance["membership_source"],
      pit_alignment=provenance["pit_alignment"],
  )
  ```
  (Reuse the `compute_lineage` helper so write-time and gate-time recomputation are the same function.)

### Step 4: tests

- Update `test_build_industry_ranking_falls_back_to_snapshot_cache` → call `build_industry_ranking(str(base), allow_snapshot_fallback=True)` and assert provenance `pit_alignment == "proxy"`.
- New: default (no flag) with membership missing → `SystemExit` (or `ValueError`) mentioning "sector_membership.parquet missing".
- New: membership present → provenance `pit_alignment == "verified"`, `membership_source == "pit"`.
- New: membership present but its parquet is a bare file with NO manifest (or a tampered one) → `SystemExit` (manifest check fails). Use the `SECTOR_MEMBERSHIP_ASSET` write path (`write_asset_manifest`) for the healthy case in existing tests, since the builder now validates.
- New: `compute_lineage` returns the three keys; changing `sector_membership.parquet` bytes flips `upstream_roots["sector_membership"]`.
- Existing `test_build_industry_ranking_uses_pit_membership` / `test_build_industry_ranking_excludes_unclassified_stocks` must be updated to write a `sector_membership.parquet` WITH a manifest (the builder now validates). Give the fixture's membership frame `attrs["source"]` and call `write_asset_manifest(mem_path, SECTOR_MEMBERSHIP_ASSET, mem)`.

### Step 5: run suites + commit

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_industry_ranking_pit.py -q
```
Expected: PASS. Commit: `feat(v19-P0.2/P0.3): industry_ranking fails closed without sector_membership; validate membership; write INDUSTRY_RANKING_ASSET lineage manifest`.

---

## Task 3: `build_market_env.py` consumption validation + §二十 metadata

**Files:**
- Modify: `scripts/production/build_market_env.py`
- Modify: `tests/scripts/test_build_market_env.py`
- Modify: `stoke_ml/config/feature_profile.py` (comment only)

### Step 1: validate `industry_ranking` on consumption + thread pit

Change `build_industry_advance(base)` → `build_industry_advance(base) -> tuple[pd.Series, str]` (series, pit_alignment):
- When `industry_ranking.parquet` is absent → `(empty_series, "verified")` (missing is not a manifest problem; the required-price-col assertion downstream handles absence).
- When present: `raw = pd.read_parquet(path)`; `check_asset_read(path, INDUSTRY_RANKING_ASSET, raw, require_valid_manifest=True)` (import `INDUSTRY_RANKING_ASSET` from `scripts.production.download_industry_ranking`); then compute the advance series as today; read the manifest's `pit_alignment` (via `validate_asset_manifest` or the written manifest) and return it (default `"verified"` if the manifest carries none — but the manifest always carries it after Task 2).
- Update the caller in `build_market_env` (line 219):
  ```python
  adv, adv_pit = build_industry_advance(base)
  if not adv.empty:
      series["market_adv_ratio"] = adv
  ```
- Record the pit in `parts` so the honest declaration is manifest-visible: add to `parts["price"]` a key `industry_advance_pit: adv_pit` and extend the note to mention "CSRC broad-sector advance ratio" (§二十).

### Step 2: §二十 feature metadata

In `feature_profile.py` near `MARKET_ENV_PRICE_COLS` (line 101), add a short comment: `market_adv_ratio` is the **CSRC broad-sector advance ratio** — the fraction of 证监会 门类 sectors (A–S) with positive equal-weighted return, derived from the PIT `industry_ranking` (§二十). The parts note in `build_market_env` is updated in Step 1.

### Step 3: tests

- Update `_write_industry` (test_build_market_env.py lines 71-83) to write the manifest too:
  ```python
  rows.to_parquet(daily / "industry_ranking.parquet", index=False)
  from stoke_ml.data.asset_contract import write_asset_manifest
  from scripts.production.download_industry_ranking import INDUSTRY_RANKING_ASSET
  write_asset_manifest(str(daily / "industry_ranking.parquet"), INDUSTRY_RANKING_ASSET,
                       rows, pit_alignment="verified", membership_source="pit")
  ```
- New: a bare `industry_ranking.parquet` WITHOUT a manifest → `build_industry_advance` raises (fail-closed, `require_valid_manifest=True`).
- New: `parts["price"]["industry_advance_pit"] == "verified"` in `test_price_part_declared_verified`.
- Existing manifest/lineage tests (`test_manifest_written_and_formal_read_passes`, `test_write_market_env_manifest_carries_lineage`, etc.) must still pass — verify the new `parts["price"]["industry_advance_pit"]` key does not break any exact-dict assertion.

### Step 4: run suites + commit

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_build_market_env.py -q
```
Expected: PASS. Commit: `feat(v19-P0.3/§二十): build_market_env validates industry_ranking on consumption; declare CSRC broad-sector advance-ratio metadata`.

---

## Task 4: Formal gate — `industry_ranking` chain freshness + sector coverage threshold (§十四 wiring + §十七)

**Files:**
- Modify: `scripts/production/train_panel_panel.py`
- Modify: `tests/scripts/test_train_panel_formal_manifest.py` (or a new `tests/scripts/test_train_panel_sector_coverage.py`)

### Step 1: `industry_ranking` lineage check in the market_env branch

In `_enforce_formal_manifests`, inside the `elif ch in ("industry", "market_env")` branch, after the market_env derived-lineage check (lines 1612-1634), add an `industry_ranking` chain check that only fires when `ch == "market_env"`:

```python
                elif ch == "market_env":
                    ir_path = os.path.join(data_dir, "a_shares", "industry_ranking.parquet")
                    if os.path.isfile(ir_path):
                        from scripts.production.download_industry_ranking import (
                            INDUSTRY_RANKING_ASSET, compute_lineage as ir_lineage)
                        from stoke_ml.data.broadcast_assets import MARKET_ENV_ASSET
                        ir_report = validate_asset_manifest(ir_path, INDUSTRY_RANKING_ASSET)
                        if not ir_report["ok"]:
                            problems.append(_fmt_manifest_problem(ir_path, ir_report))
                        else:
                            prov = {
                                "membership_source": (ir_report["manifest"] or {}).get("membership_source", "pit"),
                                "pit_alignment": (ir_report["manifest"] or {}).get("pit_alignment", "verified"),
                            }
                            ir_now = ir_lineage(data_dir, prov)
                            ir_line = validate_derived_asset(
                                ir_report["manifest"] or {},
                                current_upstream_roots=ir_now["upstream_roots"],
                                current_transform_code_hash=ir_now["transform_code_hash"],
                                current_transform_config_hash=ir_now["transform_config_hash"],
                            )
                            if ir_line["stale"]:
                                problems.append(
                                    f"{ir_path}: INDUSTRY-RANKING STALE — "
                                    + "; ".join(ir_line["mismatches"])
                                    + "; rebuild with download_industry_ranking.py")
```
This catches the §十四 Tuesday bug: a `sector_membership` change without an `industry_ranking` rebuild flips `upstream_roots.sector_membership` → STALE → the market_env chain fails.

### Step 2 (§十七): sector active-stock coverage threshold

Add a module constant near the gate:
```python
#: §v19 §十七: a formal market_env run requires, per calendar year in its date
#: range, that the sector chain's active-stock coverage meet this floor.
SECTOR_COVERAGE_THRESHOLD = 0.80
```
In the same `ch == "market_env"` block, after the lineage checks, read the `sector_membership` manifest and enforce per-year coverage over the run's date range:
```python
                    sm_path = os.path.join(data_dir, "a_shares", "sector_membership.parquet")
                    if os.path.isfile(sm_path):
                        from scripts.production.download_sector_membership import SECTOR_MEMBERSHIP_ASSET
                        sm_report = validate_asset_manifest(sm_path, SECTOR_MEMBERSHIP_ASSET)
                        if not sm_report["ok"]:
                            problems.append(_fmt_manifest_problem(sm_path, sm_report))
                        else:
                            cov = (sm_report["manifest"] or {}).get("coverage_by_year", {})
                            start_y = int(pd.to_datetime(start_date).year)
                            end_y = int(pd.to_datetime(end_date).year)
                            for y in range(start_y, end_y + 1):
                                frac = cov.get(str(y), 0.0)
                                if frac < SECTOR_COVERAGE_THRESHOLD:
                                    problems.append(
                                        f"{sm_path}: sector active-stock coverage {y}={frac:.2f} "
                                        f"< {SECTOR_COVERAGE_THRESHOLD} (§v19 §十七) — re-run "
                                        "download_sector_membership.py / expand CNINFO gate history")
```
(A missing year in `coverage_by_year` is read as `0.0` — fail-closed.)

### Step 3: tests

- New: a market_env formal gate test (mirror the existing `test_formal_gate_aborts_stale_market_env_lineage` pattern) where `industry_ranking.parquet` has a manifest whose `upstream_roots["sector_membership"]` is stale → SystemExit with "INDUSTRY-RANKING STALE".
- New: a sector-coverage gate test — write a `sector_membership.parquet` + manifest with `coverage_by_year` below threshold for a year in the run range → SystemExit with "sector active-stock coverage"; and a passing case at/above threshold.
- Existing formal-manifest tests must still pass (the new checks only fire when `market_env` is consumed AND the files exist).

### Step 4: run suites + commit

Run:
```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/scripts/test_train_panel_formal_manifest.py -q
```
Expected: PASS. Commit: `feat(v19-P0.3/§十七): formal gate validates industry_ranking lineage and sector active-stock coverage for market_env runs`.

---

## Task 5: Rebuild the derived chain + verify + P0.5 quality gate (data operation)

Dependency: Tasks 1-4 must be merged first (the rebuild exercises the new manifests/coverage/cache semantics).

- [ ] Rebuild sector membership (the §十九 cache-version change forces a one-time full CNINFO refetch; 8 workers, resumable):
```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_sector_membership.py > logs/v19_sector_membership_closeout.log 2>&1
```
Verify: log ends with per-year coverage (active-stock denominator — early years higher than the old 0.15-0.33); `Read` the log and confirm `coverage_by_year` recent years ≥ 0.88.
- [ ] Rebuild industry ranking (now with INDUSTRY_RANKING_ASSET manifest + lineage):
```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/production/download_industry_ranking.py > logs/v19_industry_ranking_closeout.log 2>&1
```
Verify: `data/a_shares/industry_ranking.parquet.manifest.json` exists with `upstream_roots.daily` + `upstream_roots.sector_membership` + transform hashes + `pit_alignment="verified"`.
- [ ] Rebuild market_env (validates industry_ranking on consumption):
```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/production/build_market_env.py > logs/v19_market_env_closeout.log 2>&1
```
Verify: `market_env_daily.parquet.manifest.json` still valid; `parts.price.industry_advance_pit == "verified"`.
- [ ] Full smoke suite (the code paths the gates exercise must not regress):
```bash
PYTHONPATH=. ./.venv/Scripts/python -m pytest tests/ -q > logs/v19_closeout_smoke.log 2>&1
```
Expected: PASS (no regressions; any test asserting the old snapshot-fallback default or bare industry_ranking read must have been updated in Tasks 2/3).
- [ ] P0.5 — rerun the current-generation full quality gate:
```bash
PYTHONPATH=. ./.venv/Scripts/python scripts/production/data_quality_gate.py > logs/v19_data_quality_gate.log 2>&1
```
Verify: `reports/data_quality_gate.json` reflects the CURRENT data generation (not the 2026-08-03 one) and the gate PASSes.
- [ ] Update CONTEXT.md for the sector chain: PIT `sector_membership` + fail-closed `industry_ranking` (no default snapshot fallback; `--allow-snapshot-sector-fallback` produces `pit_alignment=proxy`), the `INDUSTRY_RANKING_ASSET` lineage, the §十七 coverage threshold, and `market_adv_ratio` as the CSRC broad-sector advance ratio. Run `scripts/production/check_docs_consistency.py`; exit 0.
- [ ] Commit: `chore(v19): rebuild sector-derived chain with provenance + rerun quality gate (P0.5)`.

---

## Self-Review

**Spec coverage (ChatGPT_v19.md §十二-§二十 + conclusion P0.2-P0.5):**
- P0.2 (§十二): default FAIL on missing sector_membership + opt-in flag + `pit_alignment="proxy"` → Task 2 Step 1.
- P0.3 (§十三): validate sector_membership before use → Task 2 Step 2; `INDUSTRY_RANKING_ASSET` + lineage → Task 2 Step 3; consumption-side validation → Task 3 Step 1; formal-gate chain check → Task 4 Step 1.
- P0.4 (§十五/§十六/§十七): completion-after-expansion + formal daily read → Task 1 Step 3; active-denominator coverage → Task 1 Step 4; formal per-year threshold → Task 4 Step 2.
- §十八: narrow KeyError → Task 1 Step 1.
- §十九: versioned cache → Task 1 Step 2.
- §二十: CSRC broad-sector advance-ratio metadata → Task 3 Step 2.
- P0.5: quality-gate rerun → Task 5.

**Placeholder scan:** thresholds concrete (0.80); the KeyError narrowing, cache payload, expansion-completion restructure, lineage helper, and gate blocks are given verbatim. One deliberate abstraction: "read the manifest's pit_alignment via validate_asset_manifest" — the implementer opens the exact file.

**Type consistency:** `compute_lineage(data_dir, provenance)` signature matches Task 2 Step 3 and Task 4 Step 1; `build_industry_ranking(base, *, allow_snapshot_fallback)` returns `(df, provenance)` consistently; `build_industry_advance(base) -> (series, pit)` matches Task 3 Step 1 and its caller; `_coverage_by_year(membership, active_by_year)` matches Task 1 Step 4 and Task 5's expected manifest.

**Sequencing:** Tasks 1-2 are independent (different files). Task 3 needs Task 2's `INDUSTRY_RANKING_ASSET`. Task 4 needs Tasks 1-3. Task 5 (rebuild) runs after all code tasks pass review, then P0.5.
