# 磁盘资源扫描 + 针对性WebSearch 研究结果

> 2026-06-29 | 回答：消除瓶颈与缺陷的可利用资源与方案

---

## 一、磁盘可用资源

### 1.1 可直接调用的库

| 资源 | 位置 | 状态 |
|------|------|------|
| **LightGBM 4.6.0** | `.venv/Lib/site-packages/lightgbm/` | 已安装但**项目中未使用** |
| PyTorch 2.7.1+cu128 | `.venv` | 正常 |
| Transformers 5.9.0 | `.venv` | 正常 |
| XGBoost | `.venv` | 正在使用 |
| SnowNLP | `.venv` | 备用 |
| AKShare / Tushare / Baostock / Efinance | `.venv` | 正常 |
| curl-cffi | `.venv` | 正常 |
| Optuna 4.9.0 | Anaconda 全局 | 未在 venv 安装但可访问 |
| Scikit-learn 1.7.2 | Anaconda 全局 | 正常 |
| Statsmodels 0.14.5 | Anaconda 全局 | 正常 |
| TensorFlow 2.21.0 | Anaconda 全局 | 可用 |
| Dask 2025.11.0 | Anaconda 全局 | 可用（分布式计算）|
| NLTK 3.9.2 | Anaconda 全局 | 可用 |

### 1.2 本地相关项目

| 项目 | 路径 | 可利用内容 |
|------|------|-----------|
| **Stock-Prediction-Models** | `GitHub_Project/Stock-Prediction-Models-master/` | 30个DL模型(LSTM/GRU/Attention/Seq2Seq/VAE)、23个RL Agent、stacking集成(AE+XGBoost+ARIMA)、sentiment-consensus情绪融合notebook |
| **daily_stock_analysis** | `GitHub_Project/daily_stock_analysis-main/` | 完整A股分析系统，同源数据(akshare/efinance/tushare/baostock)，丰富的技术指标(MACD/RSI/趋势/量能/支撑压力/乖离率综合评分)，趋势分析器(700行成熟代码) |
| **MediaCrawler** | `GitHub_Project/MediaCrawler-main/` | 多平台爬虫框架，反爬虫技术集(指纹浏览器/代理池/验证码识别)，可复用的反爬模式 |
| **WonderTrader** | `GitHub_Project/wondertrader-master/` | C++交易框架，策略/组合示例 |

### 1.3 模型与数据缓存

| 缓存 | 路径 | 内容 |
|------|------|------|
| HF Hub | `~/.cache/huggingface/hub/` | GPT-2 (小)、WikiText — **无FinBERT权重** |
| ModelScope | `~/.cache/modelscope/hub/` | `yiyanghkust/finbert-tone-chinese` 仅 `.mdl` 元数据，**权重未下载**；ModelScope上该模型不存在 |
| Torch Hub | `~/.cache/torch/hub/checkpoints/` | 空 |
| KaggleHub | `~/.cache/kagglehub/datasets/` | 有缓存目录 |
| Chroma | `~/.cache/chroma/` | 向量数据库可用 |
| ONNX | `~/.cache/onnx_models/` | 可用 |

---

## 二、WebSearch 结果

### 2.1 Guba Body 替代获取方案

**方案A: Selenium + stealth.min.js**
- 来源: `zcyeee/EastMoney_Crawler` (GitHub)
- 方法: Selenium模拟浏览器 + `stealth.min.js` 去除自动化特征
- 直接解析正文/评论页面HTML
- 内置反反爬：自动重启WebDriver(~660页触发)、断点续爬
- 评估: 可尝试，用Playwright + stealth JS逐个访问，单进程slow模式可能不被封

**方案B: 分布式请求**
- 来源: 阿里云/腾讯云开发者实践
- 方法: 中间件任务去重+分发，每IP每次少量请求
- 评估: 需要多IP资源，成本较高

### 2.2 Qlib Alpha因子体系

**Alpha158 (158因子，适合树模型):**

| 类别 | 因子数 | 示例公式 |
|------|--------|---------|
| K线基础 | 9 | KMID=($close-$open)/$open, KLEN=($high-$low)/$open, KUP(上影线), KLOW(下影线), KSFT(收盘位置) |
| 价格标准化 | 4 | OPEN0=$open/$close, HIGH0=$high/$close, LOW0=$low/$close, VWAP0=$vwap/$close |
| 滚动窗口(5/10/20/30/60) | 29×5=145 | MA, STD, MAX, MIN, QTL(分位数), RANK, RSV(KDJ基础), CORR(价量相关系数), ROC(变化率), BETA(趋势斜率), RSQR(R²趋势线性度), RESI(残差), SUMP/SUMN/SUMD(类RSI), VMA(量均线), WVMA(量加权波动率), IMAX/IMIN(Aroon高低点位置), CNTP/CNTN/CNTD(涨跌天数统计) |

**Alpha360 (360因子，适合深度学习):** 60天 × 6字段(OPEN/HIGH/LOW/CLOSE/VWAP/VOLUME) = 360个，纯原始序列无预计算

**标签定义:** `Ref($close, -2) / Ref($close, -1) - 1` (T+1买入T+2卖出，考虑A股T+1制度)

**Qlib CSI300 Benchmark (2024):**

| 模型 | 数据集 | IC | Rank IC | 年化收益 | 最大回撤 |
|------|--------|-----|---------|---------|---------|
| DoubleEnsemble | Alpha158 | 0.0521 | 0.0502 | 11.58% | -9.20% |
| LightGBM | Alpha158 | 0.0448 | 0.0469 | 9.01% | -10.38% |
| XGBoost | Alpha158 | 0.0498 | 0.0505 | 7.80% | -11.68% |
| HIST | Alpha360 | 0.0522 | 0.0667 | 9.87% | -6.81% |
| IGMTF | Alpha360 | 0.0480 | 0.0606 | 9.46% | -7.16% |
| GRU | Alpha360 | 0.0493 | 0.0584 | 7.20% | -8.21% |

### 2.3 高维特征降维最佳实践

多个2024年Nature/IEEE/ACM研究结论:

| 方法 | 效果 | 适合场景 |
|------|------|---------|
| **LightGBM EFB** (Exclusive Feature Bundling) | 自动捆绑互斥稀疏特征，信息无损 | 首选，24,300→更少特征自动 |
| LightGBM `feature_fraction=0.6-0.8` | 每棵树随机采样部分特征 | 正则化防过拟合 |
| PCA + LightGBM | Accuracy 74%→80% (+6.3%, 2024 Nature) | 线性降维首选 |
| Autoencoder + XGBoost | 已有项目验证 | 非线性降维 |
| Sequential Forward Selection (SFS) | 精确度最高但训练成本最高 | 最终精筛 |
| Mutual Information Filter | AUROC最优 (0.774, 2024 Nature) | 快速粗筛 |

**核心发现:** LightGBM EFB + `feature_fraction` 在2024研究中全面优于XGBoost处理高维金融数据。我们有LightGBM 4.6.0可直接使用。

### 2.4 融资融券+北向资金整合模式

| 特征 | 构建方式 | 预测价值 |
|------|---------|---------|
| 融资余额变化率 | `(balance_t - balance_{t-1}) / balance_{t-1}` | 杠杆资金情绪 |
| 两融余额占流通市值比 | `margin_balance / market_cap` | 杠杆参与度(~2.5%) |
| 北向净流入(N日累计) | `sum(net_flow_{t-N+1:t})` | 外资趋势 |
| 北向持股变化 | `hold_pct_t - hold_pct_{t-1}` | 外资增减仓信号 |

浦银国际2024报告使用14维度情绪指数，包含上述全部指标，验证了其预测能力。

### 2.5 中文金融NLP替代模型

| 模型 | 准确率 | F1 | 离线 | 说明 |
|------|--------|-----|------|------|
| `yiyanghkust/finbert-tone-chinese` ★ | 0.88 | 0.87 | ✅ | 分析师报告训练，已在用 |
| **`bardsai/finance-sentiment-zh-base`** | **0.973** | **0.966** | ✅ | 金融短语库训练，准确率显著更高 |
| Fine-tune RoBERTa-wwm-ext | 可定制 | - | ✅ | 需自己标注数据 |

---

## 三、优化路线图（按投入产出比排序）

| 优先级 | 行动 | 预期收益 | 难度 | 耗时 |
|--------|------|---------|------|------|
| **P0** | LightGBM替换XGBoost + 启用EFB | 自动降维，训练速度3-5x，MCC+0.005~0.01 | 低 | 1h |
| **P0** | 扩展技术指标到Alpha158风格(~150因子) | 信息量提升~4x，无需seq_len即可训树模型 | 中 | 4h |
| **P1** | Playwright+stealth.js慢速补采Guba正文 | body覆盖0%→10-30% | 中 | 2h实现+后台跑 |
| **P1** | 融资融券+北向资金维度ablation | 验证253MB+36MB数据的预测价值 | 低 | 1h |
| **P2** | 尝试bardsai/finance-sentiment-zh-base | 情绪质量潜在提升 | 低 | 30min |
| **P2** | Mutual Information/SFS特征选择 | 24,300→~50关键特征 | 中 | 3h |

---

## 四、现有瓶颈对应解决方案矩阵

| 瓶颈 | 严重性 | 解决方案 | 来源 |
|------|--------|---------|------|
| Guba body WAF封禁(0%覆盖) | 高 | Playwright+stealth.js慢速爬取 | zcyeee/EastMoney_Crawler |
| 维度爆炸(24,300维ALL配置) | 高 | LightGBM EFB + feature_fraction | 2024 Nature/IEEE ×3 |
| MCC偏低(<0.03) | 高 | Alpha158因子扩展 + 融资融券/北向资金 | Qlib benchmark + 浦银国际2024 |
| 融资融券253MB未测试 | 中 | 构建4个核心特征后ablation | 浦银国际14维情绪指数模型 |
| 纯技术面仅40个指标 | 中 | 扩展到Alpha158(158因子，含价量相关/分位数/趋势线性度) | Microsoft Qlib |
| FinBERT首次加载需网络 | 低 | 已修复(hf-mirror→cache→lexicon 3级回退) | 已提交 |
| 情感词库命中率仅13% | 中 | FinBERT已修复(100%命中)；可试bardsai模型 | 已提交 / WebSearch |
| Δ CI跨0(统计不显著) | 中 | 增加样本量(100→200+stock) + 信号增强 | 浦银国际方法 |
