# TFT 替代架构选型分析

> **日期**: 2026-07-17
> **背景**: TFT 在金融数据上存在梯度坍塌的结构性缺陷（FinFusion 450+ 实验确认，无完整解决方案），决定更换架构。
> **约束**: RTX 4090 24GB, 488 只 A 股, 日频预测, 3 任务输出（方向+收益率+波动率）, walk-forward 验证。

---

## 三个候选方向

| # | 方向 | 一句话 | 风险等级 |
|---|------|--------|---------|
| A | **VSN + xLSTM** | Oxford 2025 金融基准测试 Sharpe 第一 | 🟢 低 |
| B | **Mamba 双尺度** | 选择性状态空间，线性复杂度，梯度稳定 | 🟡 中 |
| C | **Multi-Task Panel Transformer (类 Stockformer)** | 面板原生架构，小波分解+图嵌入+多任务 | 🔴 高 |

---

## 淘汰的标准（为什么只留这三个）

从调研中淘汰的方向和原因：

| 方向 | 淘汰原因 |
|------|---------|
| PatchTST | Channel-independent 设计，不利用跨股票信息 |
| iTransformer | 变量维度 attention，在低 SNR 金融数据上未见优势 |
| TimesNet | 强周期假设，股市日频不满足 |
| Crossformer | Exchange 基准最差（MSE 0.94 vs iTransformer 0.36） |
| N-BEATS/N-HiTS | 单变量设计，无法处理 panel 结构 |
| Informer/Autoformer | 长序列优化（>96步），我们的 60 步不需要 |
| LTR-Net | 与方向 C 重叠（都是 Transformer 变体），被 Stockformer 覆盖 |
| RiskAwareTNet | BiLSTM+Transformer，本质上还有 attention，梯度问题可能复现 |

---

## 评估维度

每个方向从以下 5 个维度打分（1-5）：

1. **金融实证** — 是否有金融数据的 benchmark 结果
2. **梯度稳定性** — 是否有已知的梯度坍塌风险
3. **Panel 适配** — 是否原生支持跨股票学习
4. **实现复杂度** — 从我们现有代码出发的改造量
5. **代码可参考性** — 是否有高质量开源实现

---

## 各方向详细分析见子文档

- [方向 A: VSN + xLSTM](./architecture-a-vsn-xlstm.md)
- [方向 B: Mamba 双尺度](./architecture-b-mamba.md)
- [方向 C: Multi-Task Panel Transformer](./architecture-c-panel-transformer.md)

---

## 初步建议

**推荐 A（VSN + xLSTM）为首选**，理由：
- Oxford 基准实证最强（Sharpe 第一 + 最大 breakeven 交易成本缓冲）
- xLSTM 无 attention → 无梯度坍塌风险
- 我们的 VSN 代码（标量路径）可直接复用
- 实现复杂度最低，最快出结果验证方向

**B（Mamba）为备选**，理由：
- 如果 xLSTM 效果不够，Mamba 的选择性状态空间是更现代的方案
- 双尺度设计（CMDMamba）天然适配金融数据的微观/宏观分离
- 但开源实现成熟度不如 xLSTM

**C 为远期储备**，理由：
- 面板原生设计理论上最适合我们的数据形态
- 但包含 Wavelet + Graph + Attention，实现复杂度和风险都高
- 适合 A 或 B 跑通后作为升级方向
