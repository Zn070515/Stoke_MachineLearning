# scripts/ 布局（runbook）

四目录布局（§十七-2）：

| 目录 | 内容 | 是否 canonical |
|---|---|---|
| `scripts/production/` | 规范入口：数据下载、特征工程、训练、评测、门禁、基准 | ✅ 是 |
| `scripts/maintenance/current/` | 仍有效的维护脚本（经 DataStorage 路由，可重放） | ✅ 是 |
| `scripts/maintenance/legacy/` | 归档的一次性 repair/backfill/merge/fix 脚本 | ❌ 否 |
| `scripts/diagnostics/` | 只读 probe/verify/scan 脚本 | ❌ 否 |

## 规范入口（canonical，`scripts/production/`）

- 数据下载：`download_data.py`、`download_news.py`、`download_guba.py`、
  `download_comment.py`、`download_market_data.py`、`download_fundamentals.py`，
  以及 `download_*.py` 各单一数据源入口
- 特征工程：`build_features.py`、`preprocess_new_data.py`、`build_calendar.py`
- 训练：`train_panel.py`、`train_baselines_panel.py`、`train_baseline.py`
- 门禁/评测：`data_quality_gate.py`、`check_docs_consistency.py`、
  `benchmark_*.py`、`feature_*_report.py`、`analyze_features.py`

文档（CLAUDE.md / README.md / CONTEXT.md）里出现的命令路径统一指向这里。
`check_docs_consistency.py` 校验文档与代码一致性时也以此为唯一事实源。

## 维护（`scripts/maintenance/`）

- `current/`：仍可用于当前 canonical 数据的维护脚本。必须经 DataStorage 等
  存储层读写，不绕过数据契约。额外：
  - `gen_requirements.py`：从 `pyproject.toml`（唯一依赖源）生成 `requirements.txt`
    （§十八-3）
  - `ci.py`：本地最小 CI —— compileall → docs consistency → fast pytest →
    production smoke，与 `.github/workflows/ci.yml` 一致（§十八-4）
- `legacy/`：历史一次性脚本（backfill / repair / merge / consolidate /
  redownload / fix），仅对当时的脏数据/旧布局有效。文件头已加
  `# ARCHIVED (maintenance/legacy)` 标记，**不可直接用于当前 canonical 数据**
  ——当前数据统一走 DataStorage 的 manifest 契约（§八）。

## 诊断（`scripts/diagnostics/`）

只读探测/校验/扫描脚本（`_probe_*` / `_verify_*` / `_inventory_data.py` /
`_audit_data.ps1`）。文件头已加 `# DIAGNOSTIC (diagnostics)` 标记，不修改任何
数据，仅供人工排查问题。

## 约定

- 新脚本放对应目录；新增 canonical 入口 → `production/`，并同步更新文档。
- 脚本名带 `_` 前缀 = 非 canonical（维护/诊断），不进文档命令清单。
- 历史文档（`docs/superpowers/**`、`docs/research-findings/**`）是时间点快照，
  其中的旧 `scripts/` 路径**不改写**。
