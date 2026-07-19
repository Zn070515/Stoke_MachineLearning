# QuantBench 全流水线评估基准

> 来源: IEEE FITEE 2025, github.com/SaizhuoWang/quantbench, HKUST + IDEA Research

## 定位
工业级全流水线 AI 量化投资评估平台

## 流水线覆盖
1. **数据管理** (Data Curation)
2. **特征工程** (Feature Engineering)
3. **预测建模** (Predictive Modeling)
4. **投资组合管理** (Portfolio Management)
5. **算法交易** (Algorithmic Trading)

## 支持的 AI 方法
- 传统 ML
- 深度学习
- 强化学习
- GenAI/LLM

## 指标分类

### 任务特定指标 (Task-Specific)
- 预测任务: IC, RankIC, RMSE, MAE, 方向准确率
- 组合任务: Sharpe, MaxDD, Calmar, 换手率

### 任务无关指标 (Task-Agnostic)
- 计算效率
- 可复现性
- 鲁棒性

## 关键设计原则
1. 全流水线标准化
2. 行业对齐
3. 市场模拟框架
4. 开源可复现
