# López de Prado 回测过拟合检测框架

> 来源: Bailey & López de Prado (2012-2015), pypbo (GitHub), TradingStrategy.ai

## 核心组件

### 1. Probabilistic Sharpe Ratio (PSR)
计算策略真实 Sharpe 超过给定基准阈值的**概率**，纳入收益分布的偏度和峰度（全部四阶矩）:

```
PSR(SR*) = Φ((SR̂ − SR*) / σ̂(SR̂))
```

- **MinTRL (Minimum Track Record Length)**: 使测量 Sharpe 在给定置信水平下统计显著所需的最小观测数

### 2. Deflated Sharpe Ratio (DSR)
扩展 PSR，修正**多重测试/选择偏差**。当研究者尝试数百组参数组合只报告最优时，DSR 调整阈值以反映真实的独立试验次数。

### 3. Probability of Backtest Overfitting (PBO)
正式定义: **IS 最优的模型配置在 OOS 中表现低于所有 N 个配置中位数的概率**。
- PBO → 1: 严重过拟合
- PBO → 0: 回测可信

### 4. Combinatorially Symmetric Cross-Validation (CSCV)
PBO 的实现框架:
1. 形成性能矩阵 M (T × N)
2. 分区为 S 个等大不相交子矩阵
3. 形成所有 C(S, S/2) 组合
4. 每组合: IS 选最优 → 记录 OOS 排名
5. 计算 logit λ = ln(ω/(1−ω))
6. PBO = logit 分布在零以下的积分

CSCV 特性: **无模型、非参数、对称** — 每个子样本在不同组合中既是训练也是测试。

## 额外输出
- **Performance Degradation**: OOS vs IS 回归 (典型负 β)
- **Probability of Loss**: IS 最优策略在 OOS 负收益的组合比例
- **Stochastic Dominance**: 策略选择是否优于随机选择
- **MinBTL (Minimum Backtest Length)**: 回测需要多长才能可信

## 关键阈值
- t-statistic 门槛应接近 **3.0**（非传统 1.96）
- 约 5 年日线数据 + >45 次策略变体测试 → 几乎必然找到虚假优胜者
