# 股票预测 + 新闻情绪仓库学习分析报告

> 2026-06-20 | 调研 30+ GitHub 仓库，聚焦新闻-价格绑定、反爬技术、特征工程三个维度

---

## 一、新闻与价格绑定的 SOTA 方法对比

### 1.1 当前项目做法

```
新闻爬取 → 日期对齐(无时间戳=同日) → 日聚合(6特征) → LEFT JOIN on date → ZI填零 → 时序滞后(1/2/3/5/10/20)
预测目标: close[t+1] > close[t]
```

**关键缺陷**：
- **同日泄露风险**：当天 9:30 发出的新闻，被用来预测当天收盘价（未区分盘中/盘后）
- **无 PIT 时间戳**：新浪新闻页面只有日期，无法判断 15:00 前后
- **ZI 方法过于粗糙**：填充零假设"没新闻=中性"，但"没新闻"本身可能是重要信号

### 1.2 STONK (2025, sarthak-12/thesis-dsaa) ⭐ 最严谨

```
FinBERT 嵌入新闻 → 交叉模态注意力层 + 数值特征(LSTM编码) → 涨跌分类
```

**防泄漏机制**（本可项目直接借鉴）：
1. 除开盘价外，所有数值特征**滞后 1 天**
2. 5 折**时间序列交叉验证**（非随机划分），明确指出随机划分"会泄露未来信息"
3. 对比了拼接融合(concat) vs 交叉注意力(cross-attention) — 后者 F1=0.73

**启示**：我们应改为 `sentiment[t-1] → predict price[t]`，而非同天绑定

### 1.3 GHOST (2025, WHUT-zwj/GHOST) ⭐ 架构最新颖

```
股票级Token化 → Mamba状态空间模型 → 分层门控融合情绪 → 跨股票注意力
```

核心创新：
- **股票维度 Token 化**：将时间维度转为股票维度，复杂度 O(T²)→O(N²)
- **Mamba 替代 Transformer**：线性时间复杂度，适用于长序列
- **分层门控**：根据市场波动率动态调节情绪影响权重

### 1.4 TFT 情绪预测 (awu8732/sentiment)

```
多源新闻 → FinBERT + VADER 评分 → Temporal Fusion Transformer → 多步预测
```

关键机制：
- **因果掩码**：注意力层强制只看过去，不看未来
- **Walk-forward 验证**：按月滚动窗口，训练集逐步扩展
- **变量选择网络(VSN)**：自动降权无效特征（如某天无新闻）

### 1.5 各方法对比

| 方法 | 情绪模型 | 预测模型 | 绑定策略 | 防泄漏 | F1/准确率 |
|---|---|---|---|---|---|
| **当前项目** | SnowNLP | XGBoost/LSTM | 同日 LEFT JOIN + ZI | 弱 | 未系统评估 |
| **STONK** | FinBERT/DeBERTa | 交叉注意力 | 滞后1天+时间序列CV | 强 | 0.73 / 0.68 |
| **GHOST** | GDELT情绪 | Mamba + 分层门控 | 按日门控融合 | 中等 | SOTA on CSI300 |
| **TFT情绪** | FinBERT+VADER | TFT | 因果掩码+Walk-forward | 强 | - |
| **Adv-ALSTM** | - | LSTM+对抗训练 | 时序注意力 | 中等 | 0.713 / 0.56 |

---

## 二、反爬绕过技术深度分析

### 2.1 中国金融网站反爬格局

```
新浪财经: 基础反爬，curl_cffi 可绕过
东方财富: 中等级别，session复用即可
同花顺(10jqka): 阿里云WAF + JS挑战
雪球(xueqiu):  阿里云WAF + acw_sc__v2 cookie算法 + xq_a_token登录态
```

### 2.2 雪球 `acw_sc__v2` 算法破解 (最关键发现)

雪球每次冷访问返回一段混淆JS，在浏览器VM中执行后计算出 `acw_sc__v2` cookie。无需该 cookie 则所有请求返回 403。

**最高效方案（已公开）**：纯 Python 重现算法
```
JS混淆代码 → 逆向工程 → Python重写(SHA256+密钥派生+时间戳编码)
→ 直接生成正确cookie字符串 → 免浏览器，速度快100倍+
```

已知工具包包含 `xueqiu.js`（带注释的逆向代码）和 Python 封装器。这比 Playwright/curl_cffi 更可靠，因为：
- 不需要完整浏览器环境（省内存）
- 不需要 TLS 指纹伪装（cookie 正确即放行）
- 速度接近原生 requests

### 2.3 工具选型对比

| 工具 | 速度 | Cloudflare成功率 | 内存 | 适用场景 |
|---|---|---|---|---|
| `curl_cffi` (chrome120) | 快(125ms) | 78-82% | 55MB | API抓取 |
| Playwright + stealth | 慢(500ms+) | ~70% 无隐身 | 250MB/实例 | 登录流程 |
| Scrapling | 中等 | 85%+ | 模块化 | 综合方案 |
| 纯算法cookie重现 | 最快(50ms) | 99%+ (雪球) | 10MB | 绕过阿里云WAF |

### 2.4 pysnowball 分析

- `uname-yang/pysnowball` (1800⭐, 月下载22K)
- 本质是雪球 API 的薄封装，需要用户**手动从浏览器提取 `xq_a_token` cookie**
- 无 token 刷新、无重试、无 403 处理
- 12 个未处理 issue，近6个月无提交
- **仅适合原型验证，不适合生产环境**

### 2.5 生产级反爬最佳实践

基于 AKShare 35K⭐ 的大规模实践：

```
1. 每协程 session 隔离 (contextvars) → QPS 从 3.49 提升至 32.4
2. 双模式 IP 轮换: 定时轮换 + 触发式轮换(403/429立即切)
3. 指数退避: 1s → 2s → 4s → 8s
4. Header 熵管理: 随机化 Accept/Accept-Language/Sec-Fetch 头顺序
5. 混合方案: Playwright 登录拿 cookie → curl_cffi 批量API抓取
```

---

## 三、特征工程对标分析

### 3.1 Microsoft Qlib (35K+⭐) — 工业级标杆

Qlib 的 Alpha158/Alpha360 因子集是业界最成熟的开源实现：

- **Alpha158**: 158个表达式因子 (ROC/MA/STD/BETA/RSV)，可配置回溯窗口
- **Alpha360**: 60天平减价格/成交量历史，对当前 close 做归一化
- **PIT 修订链**：每条基本面记录跟踪 `(date, period, value, _next)`，`P()` 操作符自动取"该观测日期最新已发布值"，彻底消除 look-ahead bias
- **20+模型**已集成，统一 `fit/predict` 接口

**差距**：我们当前仅有 ~20 个技术因子 + 6 个情绪因子，缺乏 Qlib 级别的系统化因子库

### 3.2 多模态融合前沿

| 模型 | 模态 | 融合方式 | 增益 vs 单模态 |
|---|---|---|---|
| MSGCA | 指标+文本+关系图 | 门控交叉注意力 | 8-31% |
| Dual-Path Transformer | 文本+价格 | 交叉模态注意力对齐 | - |
| Uni-FinLLM | 文本+时序+基本面+宏观 | 共享Transformer+模块化任务头 | 6pt准确率 |

### 3.3 缺失值处理最佳方案

| 方法 | 仓库 | 原理 |
|---|---|---|
| ZI (当前) | 本项目 | 填零+has_news标记 |
| VSN | TFT/pytorch-forecasting | 自动学习降权噪声特征 |
| Binary Mask | TimeAutoDiff | 标记张量 M[t,f]∈{0,1} 区分真值/填充值 |
| Patch Masking | ChannelTokenFormer | 训练时随机遮蔽patch，推理时缺整片则排除注意力 |

**ZI 方法在国际基准中属于最基础级别**。TFT 的 VSN 和 Binary Mask 是值得升级的方向。

---

## 四、对当前项目的改进建议

### 高优先级（立即改善模型质量）

1. **修复同日信息泄露**：情感特征改为 `sentiment[t-1] → predict direction[t]`，而非 `sentiment[t] → direction[t+1]`。STONK 已验证此改动显著提升 OOS 性能。

2. **升级情感模型**：SnowNLP → 中文 FinBERT (如 `yhangzzz/RoBERTa-wwm-ext-finetuned-finance` 或 `MengLee/finbert-chinese`)。SnowNLP 基于 2013 年语料训练，对金融领域术语覆盖率差。

3. **引入时间序列交叉验证**：当前 split 方式可能泄露未来信息。改用 Walk-forward 按年滚动窗口。

### 中优先级（扩展数据覆盖）

4. **破解雪球 acw_sc__v2**：找到公开的 JS 逆向代码，用 Python 重现算法，绕过阿里云 WAF 而非对抗。搜索 GitHub 关键词 `acw_sc__v2 python` + `xueqiu cookie algorithm`。

5. **东方财富新闻 API**：通过 AKShare `stock_news_em()` 获取5年历史新闻（东方财富有完整的新闻归档），每只股票可能数百篇文章，是目前最可行的扩展历史新闻覆盖的方法。

6. **引入 Qlib Alpha158 因子**：可直接调用 Qlib Python 包，生成158个表达式因子与我们的技术指标互补。

### 低优先级（探索性）

7. **Mamba 模型**：GHOST 已证明 Mamba 在股票预测上优于 Transformer（线性复杂度 + 长序列友好），可作为 LSTM 的替代方案。

8. **GDELT 新闻数据**：GDELT 提供免费全球新闻情绪数据（含中国），可作为第三方情绪验证源。

---

## 五、关键参考仓库索引

| 仓库 | Stars | 核心价值 |
|---|---|---|
| `microsoft/qlib` | 35K+ | 工业级因子引擎+PIT数据处理 |
| `akfamily/akshare` | 35K+ | A股全品类数据一站式接口 |
| `virattt/ai-hedge-fund` | 45K+ | 多智能体架构参考 |
| `fulifeng/Adv-ALSTM` | 1K+ | StockNet SOTA 时序注意力 |
| `WHUT-zwj/GHOST` | 新 | Mamba+股票Token化 |
| `sarthak-12/thesis-dsaa` | 新 | 最严谨防泄漏设计 |
| `yumoxu/stocknet-dataset` | 500+ | 标准评测基准 |
| `uname-yang/pysnowball` | 1.8K | 雪球API参考(不推荐直接使用) |
| `awu8732/sentiment` | 新 | TFT+Walk-forward验证 |
| `ClementPerroud/Adv-ALSTM` | 100+ | Adv-ALSTM复现 |
| `doncollins1985/FinBERT_LSTM` | 100+ | 简单两阶段流水线参考 |
