# Stoke_MachineLearning `(7)` 全面立体审核

## 一、审核基准

本轮只以最新上传的：

```text
Stoke_MachineLearning-master(7).zip
SHA-256:
4b12cc923458236e41c745cd6b5fbe018ee119e0f4f2a4d2c7a6cc527ae19fcb
```

为准。压缩包包含 377 个文件。

审核覆盖：

```text
数据源与市场代码
→ 下载与断点续传
→ Storage / Contract / Manifest
→ 交易日历与 Universe
→ 文本、基本面和其他辅助数据
→ Feature Pipeline / Cache / Lineage
→ Panel / Label / Mask
→ 模型、训练与 Baseline
→ Walk-forward / Lockbox / Continuous OOS
→ 账户会计与统计推断
→ 性能规模、测试、CI 和维护性
```

本轮只进行轻量动态审核，没有运行真实全量下载、109GB 特征重建或正式长训练。

---

# 二、动态审核结果

## 已通过

| 项目                              |             结果 |
| ------------------------------- | -------------: |
| 全仓 Python 编译                    |             通过 |
| 文档一致性守卫                         |   **53/53 通过** |
| 代码、Contract、下载、预处理、Topic、路径快速测试 | **248 passed** |
| Feature 行身份机制                   |         已进入主路径 |
| 下载状态与 Run Manifest              |         基础行为正确 |
| Continuous OOS 策略一致性            |          已明显加强 |
| TopicModel 正式拟合脚本               |            已加入 |
| 模型 checkpoint 与权重哈希             |            已加入 |

测试出现 3 个 Pandas `GroupBy.apply` 弃用警告，位于：

```text
stoke_ml/preprocessing/text/aggregation.py
```

当前审计环境没有 `pyarrow/fastparquet`，因此涉及真实 Parquet 读写的测试无法全部执行。这是本次动态审核的明确边界。

---

## 动态反向测试发现

### 1. 北交所代码路由发生内部矛盾

对 `920001`：

```text
codes.py       → BJ
Tushare        → 920001.SZ
Baostock       → sh.920001
AKShare/Sina   → sz920001
分钟源         → sh920001
资金流备用源   → sh920001
```

对 `430047`：

```text
codes.py       → BJ
Tushare        → 430047.BJ
AKShare/Sina   → sz430047
分钟源         → sz430047
```

也就是说，Universe 和 Storage 会接受这些北交所股票，但不同 Provider 可能请求完全不同的市场。

---

### 2. 下载覆盖 override 生成互相矛盾的 Manifest

人工构造：

```text
请求范围：2000-01-01 至 2024-01-03
实际数据：仅 2024-01-02 至 2024-01-03
provider_exhausted=True
covers_request=True
```

写出的 Manifest 同时包含：

```json
{
  "status": "COMPLETE",
  "covers_request": true,
  "request_covered": false
}
```

Resume 匹配器随后又根据 `request_covered=false` 拒绝跳过。

因此它目前不会制造错误跳过，但“上市日期已知时显式覆盖请求”的设计实际上无法工作，会导致反复下载和状态语义矛盾。

---

### 3. 成交量/成交额单位检查仍会漏掉典型错误

当前弱检查采用：

```text
amount / volume
```

与 `close` 比较，并容忍 100 倍区间。

动态构造：

| 输入错误                                 | Contract 结果 |
| ------------------------------------ | ----------- |
| Tushare 原始 `vol=手`、`amount=千元` 同时未转换 | **通过**      |
| Volume 正好多错 100 倍                    | **通过**      |
| `volume=0` 但 `amount>0`              | **通过**      |
| Amount 多错 1000 倍                     | 拒绝          |

所以这项检查目前只能抓到部分极端错误，不能证明单位已经正确。

---

# 三、总体判断

这一版继续修复了大量早期基础问题：

* 股票数组行身份已经闭环；
* Canonical 股票代码更严格；
* QFQ/RAW 明确混写已被拒绝；
* 下载失败能记录 Run Manifest；
* Feature Manifest 采用完整 lineage 校验；
* Continuous OOS 开始检查模型、策略、Universe 和日历身份；
* Baseline 与深度模型的候选池和账户口径更接近；
* TopicModel 有了正式拟合入口；
* Evaluator 的会计实现已经相当成熟。

所以，当前最主要的风险不再是“模型或账户代码明显写错”，而是上游研究语义：

> **一部分输入在历史时点实际上并不可知，或者其历史值会因未来事件而重写。**

本轮发现两个比普通工程 Bug 更重要的问题：

1. 基本面数据直接把报告期末当成披露日；
2. 全系统以最新前复权价格作为 canonical，并使用绝对价格水平做特征。

这两项会影响整个 OOS 研究结论，应排在下一次正式训练之前修复。

---

# 四、P0：基本面数据存在直接前视泄漏

涉及：

```text
stoke_ml/data/sources/a_shares/fundamental_source.py
stoke_ml/data/fundamental_storage.py
stoke_ml/features/aux_aligner.py
```

当前数据源明确写着：

```python
# No disclose_date from this API — use report_date as proxy
result["disclose_date"] = result["report_date"]
```

而 `FundamentalStorage` 又根据 `disclose_date` 将数据向后填充。

## 为什么这是直接泄漏

例如 2025 年一季度财务数据：

```text
报告期末：2025-03-31
实际披露：可能在 2025-04 月下旬
```

当前系统会从：

```text
2025-03-31
```

开始让模型看到这份财报，相当于提前数周甚至一个月知道：

* ROE；
* 营收增长；
* 净利润；
* 毛利率；
* 负债率；
* EPS。

这不是“保守近似”，而是明确的未来信息。

更重要的是，正式训练入口默认：

```python
use_fundamental=True
```

所以它不是未启用的备用模块，而会进入主实验。

## 处理建议

正式 headline 实验在修复前应直接关闭基本面 Channel。

长期正确方案是获取真实：

```text
disclose_date
announcement_time
revision_date
```

并以实际首次公开时间映射到下一可交易时点。

如果短期内拿不到真实披露时间，宁可使用非常保守的法定披露截止日，也不能使用报告期末。法定截止日仍不够精确，但至少不会提前公开。

修复后必须重建：

```text
fundamentals
→ daily alignment
→ features
→ panel
→ 全部实验
```

---

# 五、P0：最新 QFQ 价格并不完全 Point-in-Time

当前 canonical 日线强制：

```text
前复权 qfq
```

Feature Pipeline 又将：

```text
open
high
low
close
price_60d_q
```

作为特征，并进行历史横截面归一化。

## 问题本质

前复权历史价格通常会随未来除权、拆股、分红等公司行为重新调整。

假设某股票 2025 年价格是 100 元，2026 年发生 2:1 拆股。今天重新下载 QFQ 历史后，2025 年价格可能被重写成 50 元。

于是模型在模拟 2025 年决策时看到的：

```text
绝对股价
价格横截面排名
价格相对其他股票的 z-score
```

包含了 2026 年公司行为对历史价格尺度的影响。

## 哪些特征受影响

相对安全的主要是尺度不变的量：

* 单段收益率；
* 百分比变化；
* `close / MA - 1`；
* 同一调整尺度下的动量；
* 归一化 K 线形状。

存在泄漏风险的是尺度敏感量：

* 原始 QFQ `open/high/low/close` 横截面 z-score；
* `price_60d_q`；
* 绝对价格阈值；
* 依赖绝对价格尺度的交互特征。

目前 `_discover_pk_columns()` 明确将 OHLC 纳入模型输入，所以风险不只存在于一个 static 特征。

## 增量更新的另一个问题

QFQ 历史会因新公司行为重锚。即使旧文件和新批次都声明：

```text
adjustment_mode=qfq
```

它们也可能属于不同的 QFQ vintage。

如果只追加新尾部而不刷新完整历史，可能产生人工 seam，而 Storage 的 raw/qfq gate 无法发现，因为两边都叫 qfq。

## 推荐架构

Canonical 原始层应保存：

```text
未复权 OHLCV
公司行为/复权因子
数据版本
```

研究层按用途生成：

```text
total-return / adjusted returns
scale-invariant technical features
PIT raw nominal price
```

对于当前版本，最低限度应：

1. 从模型输入中移除绝对 QFQ OHLC 和 `price_60d_q`；
2. 只保留比例、收益率、滚动相对位置等尺度不变量；
3. 使用未复权价格检查成交额/成交量经济一致性；
4. 每次公司行为后完整刷新受影响历史，而不是只追加尾部。

---

# 六、P0：北交所 Provider 路由不统一

内部规范化器认为：

```text
4xxxxx / 8xxxxx / 920xxx → BJ
```

但 Provider 各自重新用前缀猜市场，并得出不同结论。

这会造成：

* 请求到另一交易所的同代码证券；
* Provider 返回空数据后被误计为失败；
* Circuit breaker 被无效请求触发；
* Failover 将错误市场的数据混入；
* 北交所股票在不同 Channel 中指向不同证券。

## 正确修法

将市场判断集中到唯一函数：

```python
market_of_code(code) -> SH | SZ | BJ
```

所有 Provider 只使用该函数。

然后每个 Provider 明确声明能力：

```text
supports(SH)
supports(SZ)
supports(BJ)
```

不支持北交所的 Provider 应返回：

```text
UNSUPPORTED_MARKET
```

而不是把 BJ 代码伪装成 SZ 或 SH 请求。

必须增加矩阵测试：

| 代码     | 期望 |
| ------ | -- |
| 600519 | SH |
| 000001 | SZ |
| 300750 | SZ |
| 430047 | BJ |
| 830799 | BJ |
| 920001 | BJ |

并对日线、分钟、资金流、新闻、行情备用源逐一测试。

---

# 七、P0：`--all` 模式在你的 96GB 机器上不可行

README 描述 Panel 维度大致为：

```text
255 PastKnown
1418 PastObserved
9 Static
合计 1682 个 float32 特征
```

2000—2026 的统一日期轴约 6500 个交易列。

仅三块稠密特征数组估算：

|  股票数 |         仅特征数组 |
| ---: | ------------: |
|  300 |    约 12.2 GiB |
|  500 |    约 20.4 GiB |
|  800 |    约 32.6 GiB |
| 1500 |    约 61.1 GiB |
| 5530 | 约 **225 GiB** |

这还不包括：

* 原始和预构建 Pandas DataFrame；
* `all_feat` 横截面拼接；
* 标签、价格和多个 Mask；
* Fold 切片；
* PyTorch Tensor；
* DataLoader；
* 数据增强副本；
* Python 对象开销。

因此：

> **RTX 4090 的 24GB 显存不是主要瓶颈，96GB 主内存才是。**

默认 500 股票大概率可运行，但峰值内存可能达到数十 GB；`csi800` 历史成员并集已经接近风险区；`--universe all` 明确不可运行。

## 训练时间也很重

`PanelDataset` 是股票—日期窗口级样本。500 股票 × 数千日期，单 Fold 很容易达到数十万到数百万有效窗口。

`DateGroupedSampler` 每个 epoch 还会：

```python
indices = []
```

构造所有有效样本的 Python 整数列表，而不是流式 `yield`。

这会增加：

* Python 内存；
* 采样时间；
* 大 Universe 的 epoch 启动延迟。

## 推荐重构

将 Dataset 改成以“日期”为一级样本：

```text
一个 batch = 一个日期 × 一组股票
```

优点：

* Rank loss 获得完整或受控采样的横截面；
* 每个 epoch 主要遍历日期，而不是股票×日期；
* 不再构造数百万 Python index；
* 可以按日期 lazy 读取 Memmap/Zarr；
* 直接解决当前 Rank loss 被 batch_size 切碎的问题。

特征网格应改为：

* Memmap；
* Zarr；
* 分股票或分日期 chunk；
* Lazy window gathering；

而不是一次性建立整个稠密 `(N,T,D)` 内存数组。

---

# 八、Quality Gate 仍有两个可绕过点

## 1. `manifest_contract_full_scan` 不要求两个检查都存在

当前：

```python
full_audit = [
    r for r in results
    if r.name in ("manifest", "contract_schema")
]

manifest_contract_full_scan = (
    bool(full_audit)
    and all(r.files_scanned == total_daily for r in full_audit)
)
```

如果用户只执行：

```text
--check manifest
```

并且 Manifest 扫描了全部文件，就会得到：

```text
manifest_contract_full_scan = true
```

即使 `contract_schema` 根本没有运行。

训练端只看这个布尔值，因此 Sample-scope Formal Gate 可能在没有全量 Contract 检查时通过。

## 修法

必须要求：

```python
names == {"manifest", "contract_schema"}
```

且二者都：

* Passed；
* Files scanned 等于完整 Daily 文件数；
* Unreadable 为 0。

训练端也应直接检查报告中的 `checks`，不要只信聚合布尔值。

---

## 2. Requested Universe 对账仍不是 Formal 训练的必需条件

Quality Gate 已支持：

```text
--requested-universe
--request-manifest
```

但 `build_features.py --quality-gate` 默认没有传入下载 Run Manifest。

训练入口也没有要求报告必须包含 Universe reconciliation。

因此 CSI Universe 的解析逻辑仍是：

```python
历史指数成员
∩
当前磁盘上实际存在的股票
```

缺失数据的成员会被静默删除。

这会产生：

* 数据可用性偏差；
* 退市股票缺失；
* 早期历史成员缺失；
* 下载失败股票从研究 Universe 中消失；
* Gate 仍然显示“现有文件的 98% 有效”。

正式训练应要求：

```text
下载 Run Manifest
→ Requested universe 对账
→ 每只股票文件/Manifest/范围状态
→ Quality Gate
→ 训练
```

不能让“磁盘上有什么就研究什么”成为正式 Universe 定义。

---

# 九、交易日历仍不是全仓唯一对象

核心 Formal 路径已经采用：

```python
get_research_calendar(...)
```

这是进步。

但仓库中仍有多个模块直接：

```python
TradingCalendar("a_shares")
```

例如：

* `data_quality_gate.py`；
* 新闻和股吧下载；
* DataCenter 下载；
* Fundamentals 下载；
* 新闻 Gold 构建；
* 部分 Margin、龙虎榜、涨跌停和 Sector 数据源。

## 直接影响

Quality Gate 的 `_CALENDAR` 在模块导入时创建，没有根据：

```text
--data-dir
```

加载对应 Calendar artifact。

所以可能出现：

```text
Feature Pipeline 使用 data_dir A 的冻结 Calendar
Quality Gate 使用代码内置 Calendar
报告只记录 CALENDAR_VERSION
```

而不是实际 Calendar artifact 的内容哈希。

`get_research_calendar(strict=True)` 在 artifact 缺失时也会退回代码规则，并不会因“Formal”自动失败。

## 还存在 Freshness 假阳性

Formal Gate 使用：

```python
today - latest_date > 4 calendar days
```

判断过期。

春节、国庆长假期间，一份完全最新的数据也可能超过 4 个自然日，从而错误失败。

应改成：

```text
latest_date
==
最近一个理论上已发布完成的正式交易日
```

而不是自然日差值。

---

# 十、TopicModel：默认关闭是正确的，启用时仍不够 PIT

当前 Feature Pipeline 默认关闭 Topic 特征，这是非常重要的安全选择。

但启用后仍有以下风险。

## 1. 一个全局 Topic 模型不能覆盖所有 Walk-forward Fold

如果 TopicModel 用截至 2025 年的语料拟合，然后为 2018 年新闻分配 Topic，那么 2018 年的表示空间受到：

* 2019—2025 词汇；
* 后续主题结构；
* 后续聚类结果；

影响。

原始新闻时间没有错，但表示模型看过未来。

正式方案只能是：

* 每个 Fold 按 `train_end` 单独拟合；
* 或在第一个 OOS 开始前冻结一个永不更新的模型；
* 或 headline 实验持续关闭 Topic。

## 2. Cutoff 列缺失时不会失败

`_collect_silver()` 只有在存在：

```text
aligned_date
```

时才执行 cutoff 过滤。

如果 Silver Schema 漂移或列缺失，整个历史语料会直接进入 TopicModel，而不是 Formal fail。

应要求 `aligned_date` 为必需列；没有就立即终止。

## 3. 单股票加载失败被 warning 后跳过

Topic 拟合可能在只加载部分股票的情况下成功完成。

Manifest 记录了文件信息，但没有严格的：

```text
requested_stocks
loaded_stocks
failed_stocks
coverage threshold
status=DEGRADED
```

Formal Topic fit 应有最低覆盖率，任何读取失败都应进入最终状态。

## 4. Transform 失败仍会静默变成 -1

`TopicModeler.transform()` 捕获异常后：

```python
topic_id = -1
topic_probability = 0
```

这意味着一个正式 Topic-enabled 实验可能：

* 模型文件成功加载；
* Transform 运行失败；
* 整个 Topic Channel 退化成常数；
* 训练仍继续。

需要：

```python
transform(..., formal=True)
```

Formal 模式下异常直接终止。

---

# 十一、Download Resume 的覆盖定义需要统一

当前同时保存：

```text
covers_request
request_covered
```

并允许调用方 override，但匹配器又重新根据实际日期验证。

设计目标互相冲突。

更简单可靠的做法是：

在下载前先构建每只股票的：

```text
effective_requested_start = max(global_start, listing_date)
effective_requested_end   = min(global_end, delist_date, latest_available)
```

之后 Manifest 只记录一套范围：

```text
requested_start/end        用户原始请求
effective_start/end        股票实际要求
actual_start/end           获得的数据
effective_range_covered    唯一完成结论
```

不要允许一个布尔 override 绕过日期事实。

---

# 十二、Contract 的量价单位检查需要重新设计

除了动态测试发现的漏检，当前检查还有一个概念问题：

```text
amount / volume
```

是未复权实际成交均价，而 `close` 是 QFQ 价格。

二者并不保证处于同一个历史价格尺度。公司行为后，QFQ 历史值可能大幅缩放，而实际成交额/成交量对应的是当时原始名义价格。

因此即使数据单位都正确，也可能因为复权尺度不同被误判；反过来，100 倍宽容带又会放过典型单位错误。

## 更可靠的检查位置

Provider Adapter 必须用真实响应 fixture 检查：

* Tushare `vol` 手→股；
* Tushare `amount` 千元→元；
* Baostock 股/元；
* AKShare/Efinance 的实际单位；
* 空值与零成交语义。

Canonical 层最好使用：

```text
raw VWAP = amount / volume
```

与未复权 `open/high/low/close` 比较。

还应明确拒绝：

```text
volume == 0 且 amount > 0
volume > 0 且 amount == 0
```

---

# 十三、Storage 的几个残余一致性问题

## 1. Formal load 未将已验证 Manifest 传给 Contract

`load_daily(require_valid_manifest=True)` 先验证 Manifest，之后调用：

```python
validate_contract(result, ..., formal=True)
```

但没有传：

* Manifest；
* Official trading days。

于是 Provenance 依赖 Parquet 是否保留 `DataFrame.attrs`。

项目允许多个 Parquet Engine，而不同 Engine 的 attrs 行为可能不一致。更可靠的方式是将刚验证过的 Manifest 直接传给 Contract。

## 2. Parquet 和 Manifest 不是单一原子对象

当前在同一个锁内执行：

```text
replace parquet
→ replace manifest
```

若进程在二者之间崩溃：

```text
新 Parquet + 旧 Manifest
```

Formal read 会拒绝，安全性尚可；但 non-formal read 仍可读取 torn state。

更严格的设计可采用：

```text
generation directory
+ current pointer
```

让数据文件和 Manifest 作为一代整体切换。

## 3. 历史修正后没有重新计算相邻 `pct_change`

Storage 合并采用：

```text
同日期新行覆盖旧行
```

如果修正了 t 日 close，t+1 日旧 `pct_change` 仍可能基于修正前 close。

Contract 只检查 `pct_change` 是否有限，不检查其是否等于：

```text
close[t] / close[t-1] - 1
```

Quality Gate 可能随后发现，但 canonical 写入本身已经接受了不一致文件。

建议每次合并后统一重算，或至少对受影响边界重算并在写入前验证。

---

# 十四、Feature Lineage 已经很强，仍缺 Topic 和部分辅助资产

当前 lineage 已包括：

* 日线及主要 per-stock Channel；
* Macro；
* Market environment；
* Industry；
* Sector mapper；
* Calendar；
* ETF Flow；
* Feature code tree；
* Config；
* Schema；
* Date range。

这是优秀设计。

剩余问题：

1. TopicModel 的模型文件和 fit manifest 没有作为 Feature shared input；
2. 一部分生产辅助数据仍直接 `to_parquet()`，没有统一 Manifest；
3. 行业/宏观等数据可能包含修订值，没有 vintage 标识；
4. Universe 虽在 Fold artifact 中哈希，但 Quality Gate 未强制 Requested Universe 完整性。

如果 Topic 保持关闭，第 1 项暂时不影响 headline。

---

# 十五、宏观和财务数据仍缺少“数据版本时间”

即使修复真实 `disclose_date`，还要区分：

```text
首次公布值
后来修订值
```

当前没有看到完整的：

```text
vintage_time
revision_time
retrieved_at
version_id
```

体系。

例如：

* 财报更正；
* 宏观统计修订；
* 历史行业分类重构；
* 供应商对历史数据回补。

如果今天下载到的是修订后的历史值，再把它映射到最初披露日，仍会形成 revision leakage。

高标准 PIT 系统需要：

```text
event_date
first_publication_time
revision_publication_time
retrieval_snapshot
```

至少正式报告中要标明哪些 Channel 是真正 vintage-safe，哪些只是“按发布日期对齐的最新修订历史”。

---

# 十六、模型与训练审核

## 成熟部分

当前已经具备：

* 任务级 Mask；
* Inactive task 安全处理；
* Raw RankIC 选 checkpoint；
* 最低每日股票数；
* 日期级 Ranking；
* Pair coverage；
* 有效样本加权 Validation；
* Gradient diagnostics；
* Determinism 声明；
* Fold checkpoint；
* 真实 Weight hash；
* Inference 模块；
* LSTM/VSN/xLSTM/多任务/Ranking/Static 消融；
* Entry selection bias 报告。

模型训练代码本身已经不是当前最大风险。

## 剩余问题

### Entry-fill selection bias

训练样本要求未来入场 Open 可用，因此模型没有学习：

```text
决策时可选
但次日突然停牌或无开盘
```

的样本。

评价时则是先排名、后检查成交。

更严格的设计应：

* 信号学习基于 decision eligibility；
* Return loss 再使用收益目标 Mask；
* 可增加 entry-fill probability head。

### Ranking 仍是子横截面

同一天股票超过 batch size 后会被分成多个 batch，不同 batch 之间没有 pair。

将 Dataset 改成日期级 batch，可以同时解决规模和完整横截面 Ranking 问题。

### 数据增强

默认关闭是正确的。启用后仍主要是每 Fold 一次性生成固定损坏副本，不是真正在线增强。

---

# 十七、Baseline 还有一个身份哈希缺口

Baseline tape 的 `model_config_hash` 主要来自：

```text
模型名 + PanelConfig
```

但没有明确绑定：

* `--with-seq-features`；
* Baseline 输入构造版本；
* `max_train_rows`；
* 特定 Baseline 超参数；
* Scaler/Input recipe。

因此同一模型名下，带 Sequence summary 和不带 Sequence summary 的两次运行，可能拥有相同 `model_config_hash`，并被 Continuous replay 当成同一策略架构。

应单独保存：

```text
baseline_input_recipe_hash
baseline_hyperparameter_hash
scaler_hash
training_sample_policy_hash
```

模型文件 SHA-256 只能证明每 Fold 的拟合结果不同，不能证明它们属于同一个策略定义。

---

# 十八、Continuous OOS 的剩余问题

当前已正确检查：

* Data version；
* Universe；
* Membership；
* Calendar；
* Horizon；
* Cost；
* Top fraction；
* Evaluator；
* Price convention；
* Exit policy；
* Strategy mode；
* Model source/config/schema。

这是很好的闭环。

仍建议增加：

## 1. Formal tape 必须含 `weight_hash`

Weight hash 不需要跨 Fold 相等，但每一折都必须存在，并最好与对应 checkpoint 文件重新计算后的 SHA-256 一致。

否则 Continuous OOS 可以从预测重放账户，却无法证明预测来自保留下来的具体模型权重。

## 2. 预测日期重复应失败

当前不同 Fold 的价格重叠有一致性断言，但若两个 tape 在同一股票同一信号日均有预测，后写可能覆盖前写。

Fold 理论上不重叠，但仍应将其做成强不变量。

## 3. 报告措辞仍有误

代码 Summary 仍称：

```text
disjoint OOS return windows
```

实际上严格不重叠的是：

```text
signal / entry windows
```

最后一批持仓的退出和 Price padding 可能进入下一 Fold 的日期范围。Continuous account 已正确处理，但文字应改为“disjoint signal windows”。

---

# 十九、账户与执行模型

Evaluator 是当前最成熟的模块之一，核心会计测试已经很强。

但它仍是研究级执行近似，不是 A 股完整成交仿真：

* 只要 Open 有效就假设能买入；
* 未检查一字涨停无法买入；
* 未检查一字跌停无法卖出；
* 未区分 ST、主板、创业板、科创板、北交所涨跌幅；
* 没有 100 股手数；
* 没有成交量占比与市场冲击；
* 成本是对称单一费率；
* 没有卖出印花税非对称；
* 没有最低佣金；
* Short 没有融券可得性、借券费和保证金。

所以正式 headline 应继续使用：

```text
Continuous long-only net OOS
```

Long-short 应明确标为：

```text
theoretical long-short factor diagnostic
```

而不是可执行 A 股多空策略。

---

# 二十、统计推断

目前已有：

* Newey–West；
* Moving-block bootstrap；
* PSR；
* 有效样本量；
* 历史实验 Registry；
* DSR；
* Block-bootstrap max-mean reality check；
* Experiment signature 和锁。

这套体系已经高于大多数个人项目。

剩余边界：

1. Registry 引入前的大量历史试验未被统计，DSR 的真实 trial 数仍可能偏低；
2. 随机种子、输入 recipe 和全部人工研究选择必须进入 experiment signature；
3. Outer folds 在开发期间被反复查看后，会逐渐承担验证集角色；
4. Lockbox 只有在设计冻结后单次开启，才能保持真正的检验意义。

---

# 二十一、工程与维护性

## 优点

* 没有发现硬编码的 API Key、Token 或 Password；
* 依赖已进入 `pyproject.toml` 和 `uv.lock`；
* CI、Python 版本、脚本分层、文档守卫较完善；
* Production、Maintenance、Legacy、Diagnostics 分层方向正确。

## 主要问题

核心文件仍然过大：

| 文件                     |       规模 |
| ---------------------- | -------: |
| `train_panel.py`       | 约 2900 行 |
| `features/pipeline.py` | 约 2000 行 |
| `evaluate.py`          | 约 1700 行 |
| `data_quality_gate.py` | 约 1400 行 |
| `aux_aligner.py`       |  约 900 行 |

生产和核心代码中仍有大量：

```python
except Exception
```

数据源降级可以宽容，但 Storage、Manifest、Feature build、Formal preprocessing 和 Experiment registry 不应广泛吞掉编程错误。

---

# 二十二、当前评分

| 维度                     |         评分 |
| ---------------------- | ---------: |
| 总体架构                   | **9.5/10** |
| 研究设计意识                 | **9.5/10** |
| 时间轴与 Mask              | **9.2/10** |
| 日线下载和 Resume           | **8.2/10** |
| Provider 市场路由          | **6.0/10** |
| Canonical Storage      | **8.8/10** |
| Data Contract          | **8.0/10** |
| Quality Gate           | **8.1/10** |
| 交易日历闭环                 | **7.5/10** |
| Universe / 退市 / 指数成员   | **8.5/10** |
| 基本面 PIT                | **2.5/10** |
| QFQ PIT 语义             | **5.0/10** |
| 预处理因果性                 | **8.4/10** |
| Topic PIT              | **6.5/10** |
| Feature Pipeline       | **9.0/10** |
| Feature lineage        | **8.4/10** |
| Dataset / Label / Mask | **9.2/10** |
| 模型与训练                  | **9.0/10** |
| Baseline 公平性           | **8.5/10** |
| Long-only Evaluator    | **9.2/10** |
| Continuous OOS         | **8.8/10** |
| 统计推断                   | **8.3/10** |
| 测试体系                   | **9.2/10** |
| 96GB 机器的全市场可运行性        | **4.5/10** |
| 工程维护性                  | **7.3/10** |
| 当前研究结果可信度              | **6.5/10** |
| 已证明稳定 Alpha            |  **仍不能评分** |

---

# 二十三、修复优先级

## 第一批：再次正式训练前必须完成

1. **停用或修复 `disclose_date = report_date` 的基本面数据；**
2. 移除所有尺度敏感的 QFQ 绝对价格特征，规划 Raw + Adjustment Factor 架构；
3. 统一北交所市场路由，并为不支持 BJ 的 Provider 显式降级；
4. 修正 Quality Gate 的“Manifest+Contract 必须同时存在”判断；
5. Formal Gate 强制 Requested Universe 对账；
6. 让 Quality Gate 使用实际 `data_dir` 的冻结 Calendar artifact；
7. Formal Topic 启用时要求 Fold-safe cutoff、完整日期列和 strict transform；
8. 明确限制 `--all`，避免 96GB 主机直接 OOM。

## 第二批：Lockbox 前完成

1. 重构成交量/成交额单位校验；
2. 统一下载 Effective requested range；
3. Storage Formal read 传入 Manifest 和官方 Calendar；
4. 合并历史修正后重新计算 `pct_change`；
5. Baseline input recipe 纳入模型身份；
6. Continuous tape 要求 Weight hash 和 checkpoint 对账；
7. Macro/Fundamental 增加 revision/vintage 语义；
8. 辅助数据逐步统一 Contract、Manifest 和原子写入；
9. 将 Dataset 重构为 Date-centric / lazy storage；
10. 修正文档中的 “disjoint return windows” 表述。

## 第三批：工程优化

* 拆分 FeaturePipeline、TrainPanel、Evaluate 和 Gate；
* 将 `DateGroupedSampler` 改为流式生成；
* 清理关键路径 broad exception；
* 增加真实 Provider 响应 fixture；
* 统一所有市场映射、股票代码和日历入口；
* 处理 Pandas 3.0 `GroupBy.apply` 行为变更。

---

# 最终评价

这版的模型、账户、OOS、测试与工程骨架已经相当强。前几轮发现的大量基础错误确实已经修复，不再是简单“边改边冒新洞”的状态。

但本轮从整个研究链路重新审视后，最关键的结论是：

> **现在阻碍项目可信度的主要问题，已经从代码实现错误转移到了数据在历史时点是否真实可知。**

当前最严重的两个问题：

```text
报告期末被当成财报披露日
最新 QFQ 绝对价格被当成历史 PIT 价格
```

它们都可能让模型在没有明显报错、没有异常收益尖峰、所有测试全绿的情况下获得未来信息。

因此下一步不应继续加特征或调 xLSTM。应先冻结一套真正的：

```text
Raw price + Corporate actions
真实披露时间
数据 vintage
统一 Calendar
完整 Requested Universe
Provider 市场能力
```

修复这些之后，这个项目才真正进入：

> **结果可能没有 Alpha，但不会因为时间语义和数据版本问题制造假 Alpha。**
