# 数据预处理全面重构 — 设计文档

> **Status:** Draft → Review
> **日期:** 2026-07-01
> **目标:** 将原始数据处理从"简单FinBERT打分+日均值+ZI填零"升级为完整的模块化预处理系统

---

## 1. 动机

### 1.1 当前问题

| 问题 | 严重程度 | 具体表现 |
|------|----------|----------|
| 无归一化 | **致命** | 原始值直接入模，close=100 vs volume=1e8，LSTM/Transformer 严重受梯度尺度影响 |
| ZI 填零粗暴 | **高** | 所有缺失用 0 填充，无分布考量，虚构信号污染训练 |
| 文本仅均值/std | **高** | 一维情感分数丢失方向性、共识度、主题信息、时间衰减 |
| 无截面标准化 | **高** | 所有特征 per-stock 时间序列，全市场涨 5% vs 单只涨 5% 无区分 |
| 情感 body 废弃 | **中** | `sentiment_body` 算出来但从未在聚合层使用 |
| dropna() 丢整段 | **中** | `_create_sequences` 中任一 NaN 丢弃整条序列 |
| 无数据质量监控 | **中** | 异常值、缺失率、分布偏移无感知 |
| 无特征治理 | **低** | 消融实验需手写列名，特征血缘不可追溯 |

### 1.2 目标

1. 建立 **可插拔的模块化预处理系统**，每步独立可测试
2. 引入 **特征注册与版本管理**，特征血缘可追溯
3. 内置 **数据质量监控**，关键异常自动告警
4. 文本处理从"一维情感"升级为**多维文本特征**（方向性+共识度+主题+时间衰减）
5. 数值处理加入**截面标准化 + 高级插值 + 异常检测**

---

## 2. 架构概览

```
                        ┌─────────────────────────────┐
                        │      FeatureRegistry        │
                        │  (特征目录+血缘+分布快照)      │
                        └─────────────────────────────┘
                                    ▲
                                    │ register
                                    │
 ┌──────────┐    ┌──────────────────┴───────────────────┐    ┌──────────┐
 │  Bronze  │───▶│         PreprocessingPipeline         │───▶│  Model   │
 │  (raw)   │    │                                       │    │  Ready   │
 └──────────┘    │  ┌─────────────────────────────────┐  │    └──────────┘
                 │  │ Per-source Chains:              │  │
                 │  │                                 │  │
                 │  │  Text  → quality → bipolar      │  │
                 │  │        → decay → topics → agg   │  │
                 │  │                                 │  │
                 │  │  Numeric → outlier → missing    │  │
                 │  │         → cross_section → scale │  │
                 │  │         → higher_order          │  │
                 │  │                                 │  │
                 │  │  Monitor (drift/quality report) │  │
                 │  └─────────────────────────────────┘  │
                 └───────────────────────────────────────┘
```

**目录结构：**

```
stoke_ml/preprocessing/
├── __init__.py
├── base.py              # PreprocessingStep 抽象基类 + PreprocessingChain
├── pipeline.py          # PreprocessingPipeline 编排引擎
├── registry.py          # FeatureRegistry: 特征定义+版本+血缘
├── config.py            # YAML → chain 配置解析
├── text/
│   ├── __init__.py
│   ├── quality.py       # 文本质量过滤（去重/去噪/短文本）
│   ├── bipolar.py       # 牛熊分类 + 净情感 + 共识度
│   ├── decay.py         # 时间衰减加权
│   ├── topics.py        # BERTopic 主题建模 + 主题特征
│   └── aggregation.py   # 多维日聚合（替代简单mean/std）
├── numeric/
│   ├── __init__.py
│   ├── outlier.py       # MAD/IQR 异常检测与修正
│   ├── missing.py       # 缺口分类插值（线性/Kalman/标记）
│   ├── scaling.py       # 滚动 RobustScaler
│   ├── cross_section.py # 行业中性化 + 市场相对化
│   └── higher_order.py  # 高阶矩 + 波动率曲面 + 量价衍生
├── monitor/
│   ├── __init__.py
│   ├── quality.py       # 缺失率/重复率/零值率检测
│   └── drift.py         # KS-test 分布偏移检测
└── chains/
    ├── text_default.yaml
    ├── numeric_default.yaml
    └── full_pipeline.yaml
```

**核心原则：**
1. **可插拔** — 每个 `PreprocessingStep` 有统一接口 `fit() / transform() / fit_transform()`
2. **可追溯** — 每个特征注册到 Registry，记录 lineage（来源→变换→版本）
3. **PIT 安全** — 所有时序变换严格向后看（rolling/expanding），lag 逻辑内置
4. **纯文件** — 输入 Parquet，输出 Parquet，不引入外部服务依赖
5. **不耦合训练代码** — 预处理模块独立于 `train_baseline.py` / `train_lstm.py`，通过配置驱动接入
6. **兼容回测系统** — 预留 `start_date`/`end_date` 参数化接口，回测框架可按时间窗口调用

---

## 3. 文本预处理链

### 3.1 流程

```
原始帖子
 │
 ├── QualityFilter ──► 去HTML标签, 去重(title+body相似度>0.9),
 │                     去短文本(<5字), 去纯表情/符号帖
 │
 ├── BipolarClassifier ──► FinBERT 三分类:
 │     bull: sentiment > +0.2
 │     bear: sentiment < -0.2
 │     neutral: -0.2 ≤ sentiment ≤ +0.2
 │
 ├── TimeDecayWeighter ──► w_i = exp(-λ × days_since_post)
 │     λ = ln(2) / halflife_days (默认7天半衰期)
 │
 ├── TopicModeler ──► BERTopic(embedding=FinBERT, 
 │     cluster=HDBSCAN, dim_reduce=UMAP)
 │     每帖 → topic_id, topic_probability
 │     自适应主题数 20-50
 │
 └── DailyAggregator ──► 每日每股票聚合
```

### 3.2 聚合产出特征

**核心情感特征（6列→扩展为~20列）：**

| 特征 | 公式 | 范围 | 描述 |
|------|------|------|------|
| `bipolar_sent` | (N_bull - N_bear) / (N_bull + N_bear + 1) | [-1, 1] | 牛熊净情感 |
| `agreement` | 1 - sqrt(1 - bipolar_sent²) | [0, 1] | 共识度（0=分歧, 1=一致） |
| `attention` | ln(1 + N_total) | [0, ∞) | 关注热度 |
| `weighted_sent` | Σ(s_i × w_i) / Σ(w_i) | [-1, 1] | 时间衰减加权情感 |
| `sent_divergence` | σ(s) / (|μ(s)| + 0.01) | [0, ∞) | 情感分歧度（变异系数） |
| `sent_skew` | skew(s) | [-2, 2] | 情感偏度 |
| `bull_ratio` | N_bull / (N_bull + N_bear + 1) | [0, 1] | 看多占比 |
| `bear_ratio` | N_bear / (N_bull + N_bear + 1) | [0, 1] | 看空占比 |
| `body_sent_mean` | mean(sentiment_body) | [-1, 1] | 正文情感（当前废弃） |
| `body_sent_weighted` | weighted_mean(sentiment_body) | [-1, 1] | 正文时间衰减情感 |

**主题特征（per topic, 20-50 topics）：**

| 特征组 | 数量 | 描述 |
|--------|------|------|
| `topic_{k}_sent` | K列 | 每个主题的当日平均情感 |
| `topic_{k}_ratio` | K列 | 当日该主题帖子占比 |
| `topic_entropy` | 1列 | 主题分布的香农熵（主题集中度） |
| `topic_dominant` | 1列 | 当日主导主题ID |
| `topic_sent_dispersion` | 1列 | 各主题间情感的标准差 |

**多时间窗口衍生（对所有日频指标做 rolling mean/std）：**

```
窗口: [3, 5, 10, 20] 日
每个基础指标 × 4窗口 × 2统计量(mean/std) = 8列
```

### 3.3 设计决策

- BERTopic 模型 **per source** 首次全量训练，后续周度用 `.merge_models()` 合并（先训新数据独立模型→merge到主模型，保留 UMAP+HDBSCAN 质量）
- Body 情感与标题情感**分别聚合**，产出两套独立指标
- 主题数由 HDBSCAN 自动确定（`min_cluster_size=50` 控制粒度）
- 所有文本特征的 PIT 安全由 lag(1) 保证（在 merge 时 shift，同现有逻辑）

---

## 4. 数值预处理链

### 4.1 流程

```
数值序列 (K线/基本面/市场数据等)
 │
 ├── OutlierDetector ──► MAD法: |x - median| > threshold×MAD → clip到边界
 │     threshold默认 5.0
 │     涨跌停(±9.5%)不clip（真实信号, 标记 is_limit_up/down）
 │     Winsorize 截尾用 σ 倍数而非分位数: ±3σ for normal, ±4σ for heavy-tail
 │
 ├── MissingImputer ──► 按缺口长度分类处理:
 │     1-2日: 线性插值
 │     3-10日: Kalman 平滑器 (statsmodels.tsa.statespace)
 │     >10日: 保留 NaN + 生成 has_gap_{col} 标记
 │     不再使用简单 ZI 填零
 │
 ├── CrossSectionNormalizer ──► 截面标准化 (三阶段):
 │     Stage 1 — 行业中性化:
 │       回归: X ~ Σ(industry_dummies) → 取残差
 │       或: X - median(同行业) / MAD(同行业)
 │     Stage 2 — 市值中性化:
 │       对Stage1残差再回归: residual ~ log(market_cap) + log²(market_cap) → 取残差
 │     Stage 3 — 自适应强度 (高波动期加强中性化):
 │       α = α₀ × (1 + β × (σ_short - σ_long) / σ_long)
 │     使用已有 StockSectorMapper + 现有市值数据
 │
 ├── RobustScaler ──► 滚动窗口标准化:
 │     252日 rolling window (backward only)
 │     winsorize(±3σ) → median → MAD scale
 │     训练集fit参数，验证集复用
 │
 └── HigherOrderDeriver ──► 高阶衍生特征:
       skew_20d, kurt_20d (收益率分布形态)
       realized_vol_{5,10,20,60}d (已实现波动率曲面)
       amihud_illiq (Amihud非流动性指标)
       vwap_deviation (VWAP偏离)
       max_drawdown_{20,60}d
       up_days_ratio_{20}d
```

### 4.2 设计决策

- 所有 rolling 统计用 `min_periods` 保证质量（窗口内样本不够→NaN，不做假值）
- 截面标准化复用已有 `StockSectorMapper`，不新增行业数据依赖
- 归一化参数在 walk-forward 的 train 窗口 fit，validation 窗口 transform
- 高阶衍生特征仅对 OHLCV 做，不对已标准化的截面数据再做
- **截面标准化采用 Hybrid 方案**（2024 业界共识）：特征在行业内做 z-score 标准化，但模型在全截面训练。这比纯 Generalist（忽略行业）和纯 Specialist（分行业训练，数据不够）都更好，Sharpe 更高、回撤更低
- **目标变量考虑改用截面相对收益**：不预测绝对涨跌（close[t+1] > close[t]），而预测行业相对收益（stock_return - sector_median_return），消除行业 beta 干扰

---

## 5. 特征注册与治理（FeatureRegistry）

```python
@dataclass
class FeatureDefinition:
    name: str                    # "bipolar_sent_ema7d"
    display_name: str            # "牛熊净情感(EMA 7日衰减)"
    category: str                # "text_sentiment / numeric_scaled / text_topic"
    source: str                  # "xueqiu / news / guba / comment / announcement"
    dtype: str                   # "float32"
    value_range: tuple           # (-1.0, 1.0)
    
    # 血缘
    parents: list[str]           # 上游原始列
    transformations: list[str]   # 变换步骤序列
    step_version: str            # 语义版本
    
    # 分布快照（漂移检测基准）
    baseline_stats: dict         # {mean, std, p01, p50, p99, missing_rate}
    calibration_date: str
    
    # 标签系统
    tags: list[str]              # ["ablation=xueqiu", "lag=1", "needs_scaling", "window=daily"]
```

**Registry 核心方法：**

| 方法 | 功能 |
|------|------|
| `register(feature)` | 注册新特征定义 |
| `get_by_group(tag)` | 按标签取特征，一键消融 |
| `get_by_source(source)` | 按数据源取所有衍生特征 |
| `export_lineage(format)` | 导出完整血缘图（JSON/Mermaid） |
| `validate_matrix(df)` | 验证特征矩阵列的完整性和类型 |
| `check_drift(new_stats)` | vs baseline KS-test，超阈值告警 |
| `save() / load()` | 持久化到 JSON，纳入 git version control |

**消融使用示例：**
```python
registry = FeatureRegistry.load("models/features/feature_registry.json")
xq_cols = registry.get_by_group("ablation=xueqiu")
# → ["bipolar_sent_xq", "agreement_xq", "attention_xq", ...]
# 直接传给 FeaturePipeline(use_xueqiu=False, exclude_cols=xq_cols)
```

---

## 6. 数据质量监控

```
PreprocessingMonitor
├── 输入层 (每次数据更新后)
│   ├── missing_rate: 任何列 > 20% → WARN
│   ├── duplicate_rate: > 5% → WARN
│   ├── zero_rate: ZI占比 > 50% → WARN  
│   └── freshness: max(date) 距今 > 3天 → WARN
│
├── 变换层 (每个 preprocessing step 后)
│   ├── outlier_rate: MAD异常占比 > 10% → WARN
│   ├── infinity_check: 任何inf → ERROR
│   ├── constant_check: variance=0 的列 → WARN
│   └── shape_check: 行/列数不符预期 → ERROR
│
├── 输出层 (模型消费前)
│   ├── distribution_drift: KS-test p < 0.01 → WARN
│   ├── correlation_shift: 关键特征对 |Δcorr| > 0.3 → WARN
│   ├── target_balance: |P(y=1) - 0.5| > 0.2 → INFO
│   └── feature_cardinality: 分类特征新类别 → WARN
│
└── 报告输出
    └── logs/quality/{YYYY-MM-DD}/{source}_{step}.json
        含: 所有告警, 关键指标值, 与上次的 diff
```

**日志级别：**
- **ERROR**: 阻塞训练（inf值、列缺失、shape不匹配）
- **WARN**: 可训练但需关注（高缺失率、漂移、新类别）
- **INFO**: 记录供回溯（分布统计更新、例行报告）

---

## 7. 配置设计

```yaml
# config.yaml 新增段
preprocessing:
  enabled: true
  output_dir: "data/preprocessed"
  registry_path: "models/features/feature_registry.json"
  
  text:
    quality_filter:
      min_text_length: 5
      max_duplicate_similarity: 0.9
      remove_html: true
    bipolar:
      threshold_positive: 0.2
      threshold_negative: -0.2
    time_decay:
      method: "ema"
      halflife_days: 7
    topic_model:
      enabled: true
      n_topics: "auto"
      min_topic_size: 50
      model_cache_dir: "models/bertopic"
      embedding_model: "finbert"      # 复用现有 FinBERT 做嵌入
    aggregation:
      windows: [1, 3, 5, 10, 20]
      use_body_sentiment: true
      
  numeric:
    outlier:
      method: "mad"
      threshold: 5.0
      clip: true
    missing:
      short_gap_method: "linear"
      short_gap_max: 2
      medium_gap_method: "kalman"
      medium_gap_max: 10
    cross_section:
      enabled: true
      stages: ["sector", "size", "adaptive"]   # 行业中性化 → 市值中性化 → 自适应强度
    scaling:
      method: "robust"
      window_days: 252
      winsorize_sigma: 3.0          # ±3σ 截尾 (替代分位数, 更保守)
    higher_order:
      enabled: true
      
  monitor:
    enabled: true
    log_dir: "logs/quality"
    drift_p_threshold: 0.01
    missing_warn_threshold: 0.2
    zero_rate_warn_threshold: 0.5
    
  registry:
    enabled: true
    baseline_update_freq: "monthly"
```

---

## 8. 与现有系统的接口

### 8.1 接入 FeaturePipeline

```python
# 方式1: 渐进式 — 通过配置开关
pipeline = FeaturePipeline(
    preprocessing_config="config.yaml",  # 新增
    use_legacy_preprocessing=False,      # False=新链
    # ... 其余参数不变
)

# 方式2: 独立使用 — 预处理和特征工程分离
from stoke_ml.preprocessing import PreprocessingPipeline
pp = PreprocessingPipeline.from_config("config.yaml")
clean_data = pp.run(source="xueqiu", stock_code="000001", 
                     start_date="2024-01-01", end_date="2026-06-30")
# clean_data 直接可以喂给模型或继续 FeaturePipeline
```

### 8.2 兼容性

- 现有 `FeaturePipeline.build_features()` 的 `aux_df` 参数**保持不变**
- 预处理产出仍然是 `pd.DataFrame` with `date` column，与现有存储格式兼容
- `use_legacy_preprocessing=True` 时，行为完全等同于当前版本
- 回测系统通过 `start_date`/`end_date` 调用预处理链，按时间窗口产出数据切片

---

## 9. 迁移路线

| Phase | 内容 | 时间估计 | 产出 |
|-------|------|----------|------|
| **P1: 基础设施** | base.py + registry.py + monitor.py + bipolar.py + decay.py + aggregation.py | 2-3天 | 文本链 MVP，与现有 pipeline 并行运行对比 |
| **P2: 数值链** | outlier.py + missing.py + scaling.py + cross_section.py + higher_order.py | 2-3天 | 全数值预处理可用，10只股票 MCC 验证 |
| **P3: 高级文本** | topics.py + quality.py + BERTopic 集成 | 3-4天 | 主题特征上线，245只全量文本重处理 |
| **P4: 全量验证** | 245只 walk-forward + 新旧 MCC 对比 + 性能优化 | 2-3天 | 完整对比报告，确定 MCC 提升幅度 |

**总估计: 9-13 天**

---

## 10. 成功标准

| 指标 | 当前基准 | 目标 |
|------|----------|------|
| XGBoost MCC (全特征) | ~0.026 | ≥ 0.035 (+35%) |
| 特征缺失填充质量 | ZI=0 | Kalman > 线性插值 > ZI (按 RMSE vs 真实值) |
| 文本特征贡献 (SHAP) | 情感列 ≈ top-20% | 主题+情感列进入 top-10% |
| 数据质量告警响应时间 | 无 | 异常发生后 1 次预处理内检测 |
| 消融实验效率 | 手写列名 | registry.get_by_group() 一键 |
| 回测兼容 | 无接口 | start/end date 切片可用 |

---

## 11. 待定与风险

| 项目 | 风险 | 缓解 |
|------|------|------|
| BERTopic 推理速度 | 每帖需 FinBERT 嵌入 → 全量慢 | 预计算嵌入缓存，增量新帖用 merge_models 合并 |
| Kalman 平滑器依赖 | pykalman 年久失修 (未更新) | **改用 statsmodels.tsa.statespace.UnobservedComponents** (活跃维护, 原生支持缺失值) |
| 截面标准化需实时行业分类 | StockSectorMapper 可能缺覆盖率 | 三阶段 fallback: 行业→市值→自适应，每步失败跳过 |
| 特征列爆炸（主题50×窗口4×统计2=400列） | 维度灾难 | FeatureSelection (已有 MI filter) 截断 |
| 回测系统未建成 | 接口设计可能不匹配 | 用最简单的 date range 参数化(PIT-safe)，最小约定 |
| 多阶段中性化可能过度 | 去掉太多信号 | 每阶段可选开关，通过消融实验验证各阶段贡献 |
