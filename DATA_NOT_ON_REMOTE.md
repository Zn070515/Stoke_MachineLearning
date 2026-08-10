# Data Is Not on the Remote — Remote-Only Review Blind Spots

**Status:** informational
**Last updated:** 2026-08-10 (v19 milestone)

## Why this document exists

This repository is a single-context research repo whose **actual data lives only on
the development machine** (`data/` is git-ignored). The remote `origin/master`
contains code, contracts, config, plans, and audit reports — but none of the
parquet that the code reads and the gate validates. Anyone reviewing the remote
(an external reviewer, ChatGPT, CI on a fresh clone) must know exactly which
claims are **code-verifiable** and which are **data-dependent assertions that
cannot be re-run from the remote**.

## What is excluded from the remote

`data/` is excluded wholesale (`.gitignore`: `data/a_shares/`, `*.parquet`, `data/`).
It holds:

| Dataset | Contents |
|---|---|
| `data/a_shares/daily/` | 5530 daily K-line parquet, 2000-01-04 → 2026-08-07, canonical qfq source of truth |
| `data/a_shares/*_processed/` | 6 aux channels: block_trade, board, dividend, industry_ranking, lockup, shareholder |
| `data/features/` | FLAT prebuilt features, 5530 files, ALL-mode (~24,300 cols, ~109 GB) |
| `data/features_panel/` | PANEL prebuilt features (cross-sectional z-score) |
| raw channels | news/guba/comment/fundamentals/margin/northbound/etc., exchange_calendar, sector membership |

Model checkpoints (`models/`) are likewise local-only. `config.yaml` carries no
secrets (Tushare token is read from the `TUSHARE_TOKEN` env var), so the remote is
safe to share — it is simply **data-free**.

## Consequence: repo claims are assertions, not locally re-runnable facts

- **Gate results cannot be reproduced.** `scripts/production/data_quality_gate*.py`
  read `data/`. A reviewer can inspect the check logic but cannot re-run it, so a
  `PASS`/`FAIL` in `reports/data_quality_gate.json` is a claim about data the
  reviewer cannot see.
- **`--quick` sampling cannot be re-audited.** The post-build gate samples 300 of
  5530 files (exchange-stratified, fixed seed). A remote reviewer cannot extend the
  sample to all 5530 to check whether it hid anything.
- **Contract / schema / manifest conformance is unverified against reality.**
  `contract_schema`, `manifest`, `aux_close_aligned` compare on-disk parquet to
  `DataContract`s and manifests; remote-only, the contract code is reviewed but its
  conformance is asserted, not demonstrated.

## Concrete cases found locally that a remote-only review could not see

1. **Aux close seam (2026-08-10).** After the daily tail refresh to 08-07, all six
   aux channels carried forward-filled `close` frozen at the pre-refresh daily end —
   a near-global misalignment. The gate's `aux_close_aligned` check is SAMPLED, so
   its report under-counted the true set. Only a local full scan and full reprocess
   found and fixed it. From code alone the seam is invisible.
2. **Stale prebuilt-feature divergence (2026-08-10).** `check_feature_pct` compares
   the FLAT `data/features/{code}.parquet` `pct_change` against daily. After the qfq
   migration + tail refresh, the old prebuilt features carried stale values — e.g.
   `000009` on 2026-07-20 showed feature `pct_change=-2.0934` vs daily `+0.4959`
   (max_diff 2.589266). The check reads the FLAT dir (`FEAT_DIR`) independently of
   `--require`, so a `--require daily,features_panel` run still validates flat
   features. A remote reviewer can read the check but cannot see the wrong row.
3. **qfq provenance / mixed-unit migration.** The 手/股 mixed-unit ×100
   normalization and unknown→qfq provenance migration rewrote every daily row. The
   migration code and the audit evidence (`reports/external_sample_verify.json`,
   `reports/qfq_price_verify.json`) are on the remote, but the actual pre/post
   values and the external cross-checks cannot be re-run.
4. **Sparsity / coverage numbers.** `check_sparsity` reports per-feature non-zero
   coverage (e.g. `comment_* nz=0.0`) computed from real feature matrices.
   Remote-only these are unverifiable assertions.

## Failure modes for a fresh environment trusting the remote alone

- `git clone` + `pytest` → data-dependent tests fail (no data).
- `build_features.py` / `train_panel.py` on a clone → fail closed
  (`require_valid_manifest`, formal-prebuilt-only, `verified_until`) because data is
  absent — by design, not by accident.
- A reviewer approving a migration from code alone can miss value-level corruption —
  exactly the class of bug found locally in cases 1–2 above.
- Data freshness is a local property: `data/` can lag or lead the committed code,
  and `MAX_STALE_DAYS` is only checkable where the data exists.

## What mitigations exist / are proposed

In place:
- **Gate-report provenance.** `reports/data_quality_gate.json` is committed after a
  formal run with profile + required datasets, so a remote reader knows what was
  checked and when — but cannot re-run it.
- **Contract versioning.** `contract_version()` (in `data_quality_gate.py`) digests
  the daily schema; any schema/units/basis change flips the binding and a consuming
  run fails loudly — a code-side guard that works remote-only.
- **Audit evidence on remote.** `reports/external_sample_verify.json` (independent
  source cross-check) and `reports/qfq_price_verify.json` (price-basis audit) are
  committed.

Proposed:
- Commit a small sampled dataset (a few stocks × key channels) or a data-bootstrap
  script so a fresh reviewer/environment can reproduce at least a subset of the
  gate.
- Commit data checksums/manifests so the remote can detect drift between a clone's
  provisioned data and the local canonical set.

## How a remote reviewer can still add value

- Static review of gate logic, preprocessing PIT rules, qfq provenance code,
  `DataContract`s, calendar logic, crawler failover — all in-repo.
- Verify that the CODE enforces fail-closed semantics (require-valid-manifest,
  formal-prebuilt-only, verified-until) so an absent/misprovisioned dataset cannot
  silently pass.
- Cross-file consistency: does the committed `reports/data_quality_gate.json` match
  the claims in plans/docs? Does the review record match the report? This is fully
  checkable from the remote.
