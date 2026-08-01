# xLSTM NaN 频发根因分析与修复方案

> 2026-07-19 | 三次训练实证 + NX-AI 官方实现 vs 本仓库逐行对比 + HuggingFace Transformers 交叉验证

---

## 现象

| 配置 | Stocks | lr | 结果 |
|------|--------|-----|------|
| hidden=256, blocks=3 | 188 | 1e-3 | 3 折全稳定，均值 Sharpe=2.69 |
| hidden=320, blocks=3 | 388 | 5e-4 | Fold 1 OK (Sharpe 4.14)，Fold 2 epoch 10 NaN |
| hidden=384, blocks=4 | 388 | 1e-3 | 早期 NaN |

规律：hidden_dim ≥ 320 突破稳定边界；NaN 只在 epoch 5-10 出现（warmup 峰值 lr + 模型学到有效信号）。

---

## 根因：稳定器 (m) 更新顺序错误

### 三源交叉验证

对 NX-AI 官方实现、HuggingFace Transformers (v5.5.3)、terrylica/cc-skills 金融时序实现进行逐行对比后，**三者完全一致**，而本仓库在**关键一行**上不同。

#### 官方正确实现 (NX-AI/xlstm, `vanilla/slstm.py`)

```python
# Step 1: 计算 log-space 遗忘门 + 之前稳定器
logfplusm = m + logsigmoid(fraw)

# Step 2: 先用当前步的 iraw 更新稳定器
mnew = torch.max(iraw, logfplusm)     # ← iraw 参与 m 的计算

# Step 3: 再用新的 m 来稳定指数门控
igate = torch.minimum(torch.exp(iraw - mnew), torch.ones_like(iraw))   # ∈ [0, 1]
fgate = torch.minimum(torch.exp(logfplusm - mnew), torch.ones_like(iraw))  # ∈ [0, 1]
```

**核心不变式：`mnew = max(iraw, ...)` → `iraw - mnew ≤ 0` → `exp(iraw - mnew) ≤ 1` 永远成立。**

#### HuggingFace Transformers 的实现 (v5.5.3, 2025-07)

```python
# 同一模式，加上了 gate_soft_cap=15.0 的 tanh 软截断
scaM_state_new = torch.max(scaF_log + scaM_old, igate)   # 先更新 m
scaF_act = torch.exp(scaF_log + scaM_old - scaM_state_new)  # 再用新 m 算门
```

额外保护层：`gate_soft_cap(15.0) = 15.0 * tanh(x / 15.0)`，防止 raw logit 超过 ±15。

#### 本仓库的实现 (错误)

```python
# stoke_ml/models/panel/xlstm.py:98-112

i = torch.exp(self._stabilize(i_raw, m))   # ← m 是上一步的旧值！
f = torch.sigmoid(f_raw)
z = torch.tanh(z)
o = torch.sigmoid(o)

c = f * c + i * z
n = f * n + i
m = torch.max(m, i_raw).detach()           # ← m 在 i 之后才更新！
h = o * (c / n.clamp(min=1e-8))
```

**Bug:** 计算 `i = exp(i_raw - m_prev)` 时，`m_prev` 是旧的稳定器值。如果当前步的 `i_raw` 超过之前所有步的最大值，则 `i_raw - m_prev > 0`，导致 `i = exp(正数) >> 1`。

### 数值模拟

假设序列前两步，`i_raw` 分别为 5 和 3：

| 步骤 | 官方 (正确) | 本仓库 (错误) |
|------|------------|--------------|
| t=1: i_raw=5 | `mnew=max(5,0)=5`, `i=min(exp(0),1)=1` | `i=exp(5-0)=148` |
| t=2: i_raw=3 | `mnew=max(3,5)=5`, `i=min(exp(-2),1)=0.135` | `i=exp(3-5)=0.135` |

第一步官方 `i=1 ← 正确`，而我们 `i=148`，导致 `n` 从 0 跳到 148。在 60 步序列中，如果出现 5-10 次新纪录，`n` 可以膨胀到数百甚至数千。此时 `c/n` 的梯度 `-c/n²` 被稀释到极其微小，模型实质上在"盲训"。

当 hidden_dim 从 256 增大到 320：
- W_in 权重矩阵参数量 +56%
- `i_raw` 的值域更宽，更容易产生新纪录
- epoch 5-10 warmup lr 达到峰值时，权重更新幅度最大，i_raw 新纪录出现频率最高
- 最终，某次 `exp(i_raw - m_prev)` 超出 float32 范围 (`exp(88.7) ≈ 3.4e38`) → NaN

### 其他次要差异

| 方面 | 官方做法 | 本仓库做法 | 影响程度 |
|------|---------|-----------|---------|
| **m 更新顺序** | m 在 i 之前更新 | m 在 i 之后更新 | 🔴 核心 bug |
| **f 门控形式** | `logsigmoid(fraw)` + log-space | `sigmoid(f_raw)` 普通 | 🟡 中等 |
| **门控截断** | `torch.minimum(exp(...), 1.0)` | 无 | 🟡 中等 |
| **soft cap** | `15.0 * tanh(x/15.0)` (HF 另加) | 无 | 🟢 防御性 |

`logsigmoid` vs `sigmoid` 的差异：官方在稳定器追踪中考虑了遗忘门的 log-space 衰减 `m + logsigmoid(fraw)`，确保旧的稳定器值通过遗忘门正确衰减。我们用 `max(m_prev, i_raw)` 会导致 m 被历史值"卡住"——即使遗忘门已经清空了所有旧信息，m 仍可能保持一个旧的高值。

---

## 修复方案

### 方案 A：对齐官方实现 (~15 行改动，推荐)

```python
# 替换 xlstm.py sLSTMBlock.forward() 中的循环体 (行 93-113)

for t in range(T):
    x_t = x[:, t, :]
    h_flat = h.reshape(B, D)
    W_in = self.W(torch.cat([x_t, h_flat], dim=-1))
    i_raw, f_raw, z_raw, o_raw = W_in.chunk(4, dim=-1)
    i_raw = i_raw.reshape(B, self.num_heads, self.head_dim)
    f_raw = f_raw.reshape(B, self.num_heads, self.head_dim)
    z_raw = z_raw.reshape(B, self.num_heads, self.head_dim)
    o_raw = o_raw.reshape(B, self.num_heads, self.head_dim)

    # ── 对齐 NX-AI 官方实现 ──
    LOG_GATE_MIN, LOG_GATE_MAX = -10.0, 10.0
    log_i = torch.clamp(i_raw, LOG_GATE_MIN, LOG_GATE_MAX)
    log_f_plus_m = m + torch.clamp(F.logsigmoid(f_raw), LOG_GATE_MIN, 0.0)

    m_new = torch.maximum(log_i, log_f_plus_m)               # (1) 先更新m
    i_gate = torch.exp(log_i - m_new).clamp(max=1.0)         # (2) 用新m算门
    f_gate = torch.exp(log_f_plus_m - m_new).clamp(max=1.0)

    c = f_gate * c + i_gate * torch.tanh(z_raw)              # (3) 无除法的cell更新
    n = f_gate * n + i_gate
    o = torch.sigmoid(o_raw)
    h = o * (c / n.clamp(min=1e-8))                           # (4) 官方仍用normalizer归一化
    m = m_new.detach()

    outputs.append(h.reshape(B, 1, D))
```

关键改动：
1. **m 更新移到 exp 之前** — 消弭 i > 1 的可能性
2. **`logsigmoid` 替代 `sigmoid`** — log-space 遗忘门计算
3. **`log_i` clamp `[-10, 10]`** — 防御层
4. **`.clamp(max=1.0)`** — 显式门控上限

### 方案 B：快速止血（1 行，最小改动）

```python
# xlstm.py:100，在 exp 之前加 clamp
i_raw = i_raw.reshape(B, self.num_heads, self.head_dim)
i_raw = torch.clamp(i_raw, -10.0, 10.0)  # ← 新增这一行
i = torch.exp(self._stabilize(i_raw, m))
```

这个只能防止 `exp(88)` 溢出的直接爆炸，但不能解决门控 >1 和 prediction collapse 的问题。建议作为**临时止血**，然后合并方案 A。

### 方案 C：加 HF 风格 soft_cap（可选防御层）

```python
def _soft_cap(values: torch.Tensor, cap: float = 15.0) -> torch.Tensor:
    return cap * torch.tanh(values / cap)

# 在 chunk 之后对所有 gate raw logits 应用
i_raw = self._soft_cap(i_raw, 15.0)
f_raw = self._soft_cap(f_raw, 15.0)
```

这是 HuggingFace 的额外防御层，和我们现有的 grad_clip 不冲突。可在方案 A 基础上叠加。

---

## 建议执行顺序

1. **立即合入方案 A**（对齐官方实现）→ 消除根因
2. 重跑 hidden=320, lr=5e-4 验证不再 NaN
3. 如果仍不稳定，叠加方案 C（soft_cap）

---

## 受影响的文件

| 文件 | 行号 | 问题 |
|------|------|------|
| `stoke_ml/models/panel/xlstm.py` | 98-99 | `m` 在 `i=exp(...)` 之后更新，导致 `i` 可能 > 1 |
| `stoke_ml/models/panel/xlstm.py` | 100 | `f` 用普通 sigmoid 而非 logsigmoid |
| `stoke_ml/models/panel/xlstm.py` | 121-122 | `_stabilize` 减的是旧 m，不保护反向传播 |

## 参考文献

- Beck et al. "xLSTM: Extended Long Short-Term Memory" (NeurIPS 2024). arXiv:2405.04517
- **NX-AI/xlstm** official: `xlstm/blocks/slstm/src/vanilla/slstm.py` — `slstm_forward_pointwise()` 的正确更新顺序
- **HuggingFace Transformers** v5.5.3: `modeling_xlstm.py` — `soft_cap()` + `gate_soft_cap=15.0` 稳定化模式
- **terrylica/cc-skills** `xlstm-implementation.md` — 金融时序上的 normalizer collapse → max-stabilizer 修复实证
- **TNodeCode/pytorch-sequence-models PR #10** — sLSTM 状态维度不匹配等 6 个补充 bug
