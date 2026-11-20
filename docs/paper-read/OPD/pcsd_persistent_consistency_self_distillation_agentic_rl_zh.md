# PCSD：Persistent Consistency for Self-Distillation in Agentic RL 论文解读

论文：

```text
PCSD: Persistent Consistency for Self-Distillation in Agentic Reinforcement Learning
```

链接：

```text
https://arxiv.org/abs/2608.01837
```

提交时间：

```text
2026-08-03
```

## 一句话总结

PCSD 解决的是 Agentic RL 里 OPSD 的一个细节但关键的问题：

```text
privileged teacher 平均更强，
但它在每个 token 位置都可靠吗？
```

这篇的核心不是换 teacher，也不是换 GRPO，而是：

```text
给每个 student-sampled token 计算一个 teacher 信号可靠性权重，
并且这个权重不只看单点 gap，
而是看 teacher-favoring signal 是否在局部窗口内持续存在。
```

所以 PCSD 可以理解成：

```text
GRPO + token-level OPSD + persistent-consistency weighting
```

## 术语表

| 英文术语 | 建议中文 | 在本文中的具体含义 |
|---|---|---|
| discrepancy | 差异 / 分歧度 | teacher 和 student 对同一个 token 的支持程度差异 |
| teacher-student discrepancy | 教师-学生分歧度 | 通常对应 `log pi_T - log pi_S`，不是“答案错误” |
| gap | 差距 / 间隔 | 本文基本指 sampled log-probability gap |
| sampled log-probability gap | 采样 token 对数概率差 | 对 student 已采样 token，比较 teacher 和 student 给它的 logprob |
| persistent consistency | 持续一致性 | teacher support 在局部窗口内持续存在，而不是单点 spike |
| teacher support | 教师支持度 | teacher 对当前 token 给出的概率 / logprob 支持程度 |
| teacher-favoring signal | 偏向教师的信号 | teacher 相比 student 更支持该 token 的信号 |
| credibility-aware distillation | 可信度感知蒸馏 | 蒸馏前估计 teacher 信号是否可靠，再决定 token 权重 |
| adaptive aggregation | 自适应聚合 | 根据局部方差在短窗口和长窗口之间插值 |
| trend modulation | 趋势调制 | 如果后续局部 gap 下降，就降低当前 token 的蒸馏权重 |
| sigmoid gating | sigmoid 门控 | 用 sigmoid 把连续 gap 映射成 `[0,1]` 权重 |
| privileged context | 特权上下文 | teacher 可额外看到的 retrieved skills / task hints |

## 背景

长程 agent 任务中，环境 reward 通常很稀疏：

```text
几十轮交互、几百个 token，
最后只有一个成功 / 失败 reward。
```

GRPO 这类方法能用 group-relative reward 做 trajectory-level 更新，但 credit assignment 很粗：

```text
同一条 trajectory 里的所有 token 基本共享同一个 trajectory advantage。
```

OPSD 用 privileged teacher 给 student trajectory 上的 token 提供 dense supervision：

```text
student: 只看环境可见信息
teacher: 看同样轨迹 prefix + privileged skills
```

但 privileged teacher 不是每一步都可靠：

```text
1. skill retrieval 可能不相关；
2. teacher context 可能有噪声；
3. 某些 token 位置本身存在任务歧义；
4. 单个 token 的 teacher-student gap 可能只是采样波动。
```

因此问题变成：

```text
哪些 token 位置值得蒸馏？
```

## 论文认为已有方法的问题

已有 credibility-aware distillation 大致有两类。

这里的 `discrepancy` 建议翻译成“差异 / 分歧度”。在 PCSD 里它不是“错误”的意思，而是 teacher 和 student 对同一个 student sampled token 的支持程度差异。

更具体地说，通常就是 sampled log-probability gap：

```text
delta = log pi_T(y | h_T) - log pi_S(y | h_S)
```

如果 `delta > 0`，说明 teacher 比 student 更支持这个 token；如果 `delta < 0`，说明 teacher 并不比 student 更认可这个 token。

相关术语建议：

```text
teacher-student discrepancy: 教师-学生分歧度 / 差异
sampled log-probability gap: 采样 token 对数概率差
```

### 1. Token-level 方法

它们直接看单个位置的 teacher-student discrepancy，例如：

```text
teacher 比 student 更支持当前 sampled token 吗？
```

优点：

```text
细粒度，能区分不同 token 位置。
```

问题：

```text
单点信号噪声大；
一个 spike 可能是 teacher 真有优势，也可能只是瞬时波动。
```

### 2. Step-level 方法

它们把一个 reasoning/action step 内的信号聚合成一个权重。

优点：

```text
更稳，抗局部噪声。
```

问题：

```text
太粗，同一个 step 内不同 token 的 teacher reliability 可能差很多。
```

PCSD 的目标是折中：

```text
保留 token-level 分辨率，
但用局部持续性降低单点噪声。
```

## Teacher / Student 是什么

student policy：

```text
π_θ
```

只接收标准环境 prompt：

```text
task instruction
observable environment feedback
causal interaction history
admissible actions
```

teacher policy：

```text
π_T
```

是从同一个 base checkpoint 初始化、冻结的 privileged-context teacher。它接收：

```text
student 可见的同样环境 prefix
+ retrieved task-relevant skills
```

关键点：

```text
teacher 不采样新动作；
teacher 只给 student 已经生成的 token 打 logprob。
```

所以 PCSD 的 teacher-student 对比发生在同一条 student on-policy trajectory 上。

## 基础信号：teacher-student sampled log-probability gap

对第 `k` 个 turn 的第 `i` 个 token：

```text
δ_{k,i}
= log π_T(y_{k,i} | h^T_{k,i})
  - log π_θ(y_{k,i} | h^S_{k,i})
```

中文可以叫：

```text
教师-学生采样 token 对数概率差
```

含义：

```text
δ 越大，说明 teacher 相对 student 更支持这个 student sampled token；
δ 越小，说明 teacher 并不比 student 更认可这个 token。
```

PCSD 不直接用 `δ_{k,i}` 做权重，而是看：

```text
teacher support 是否在邻近 token 上持续存在。
```

## 1. Persistent Consistency Assessment：持续一致性评估

官方术语：

```text
Persistent Consistency
```

较正式中文：

```text
持续一致性
```

它的意思是：

```text
一个可靠 teacher 信号不应该只在单个 token 突然冒出来，
而应该在局部邻域内持续支持 student sampled tokens。
```

对窗口大小 `N`，PCSD 用 forward window 做指数加权聚合：

```text
δbar^{(N)}_{k,i}
= sum_{j=0}^{N-1} α^j m_{k,i+j} δ_{k,i+j}
  / sum_{j=0}^{N-1} α^j m_{k,i+j}
```

其中：

```text
α: 衰减系数
m: response mask
```

这是 forward-looking window：

```text
当前 token i 会看 i, i+1, ..., i+N-1 的 gap。
```

但注意：

```text
这是 response 完成后用于构造训练 objective；
不是 inference-time 未来泄漏。
```

作用：

```text
1. 保留 token-level 位置；
2. 抑制孤立 spike；
3. 更重视离当前 token 近的位置。
```

## 2. Adaptive Aggregation：自适应聚合

PCSD 不用固定窗口，而是在短窗口和长窗口之间软插值。

先用最大窗口 `N_max` 计算局部均值和方差：

```text
μ^{(N_max)}_{k,i}
= sum_j m_{k,i+j} δ_{k,i+j} / sum_j m_{k,i+j}
```

```text
σ^2_{k,i}
= sum_j m_{k,i+j}(δ_{k,i+j}-μ^{(N_max)}_{k,i})^2
  / sum_j m_{k,i+j}
```

再把方差映射成插值系数：

```text
r_{k,i}
= clip(
    (σ^2_{k,i} - τ_low) / (τ_high - τ_low),
    0, 1
  )
```

最后：

```text
δbar^{adaptive}_{k,i}
= (1-r_{k,i}) δbar^{(N_min)}_{k,i}
  + r_{k,i} δbar^{(N_max)}_{k,i}
```

直觉：

```text
低方差区域：
  gap 比较稳定，用短窗口，保留 token-level 精度。

高方差区域：
  gap 波动更大，用长窗口，增强平滑和抗噪。
```

论文强调：

```text
高方差不是直接等价于 teacher 不可靠；
它只是说明单点估计更不稳，需要更强局部聚合。
```

## 3. Trend Modulation：趋势调制

持续支持不仅要“平均高”，还要看趋势。

如果当前 gap 很高，但后面快速下降：

```text
这可能是一个正在消失的 teacher advantage，
不应该给太大蒸馏权重。
```

PCSD 用最大窗口 `N_max` 拟合一个 OLS slope：

```text
slope_{k,i}
= sum_j m_{k,i+j}(j-jbar_{k,i})δ_{k,i+j}
  / (sum_j m_{k,i+j}(j-jbar_{k,i})^2 + ε_slope)
```

然后定义 one-sided trend factor：

```text
η_{k,i}
= clip(
    1 - γ · ReLU(-slope_{k,i}/δ_scale,k),
    0, 1
  )
```

其中：

```text
δ_scale,k = mean_i |δ_{k,i}| + ε_scale
```

含义：

```text
slope >= 0:
  teacher support 没有下降，η=1，不额外放大。

slope < 0:
  teacher support 正在下降，η<1，只做衰减。
```

这个设计是 one-sided 的：

```text
趋势项只能减权，不能加权。
```

目的：

```text
防止趋势估计自己制造额外 credit。
```

## 4. Sigmoid Gating：连续门控

最终 token weight 是：

```text
w_{k,i}
= σ(β_gate · δbar^{adaptive}_{k,i}) · η_{k,i}
```

含义：

```text
δbar adaptive 越大，基础蒸馏权重越高；
如果局部 teacher support 下降，η 会把权重压低。
```

为什么用 sigmoid：

```text
1. 权重连续，不是 hard selection；
2. 映射有界，更稳定；
3. β_gate 控制门控锐度。
```

论文附录还指出，sigmoid 的导数有界：

```text
|σ'(x)| <= 1/4
```

所以更稳定的 gap 估计会传导为更稳定的 token weight。

## PCSD loss

PCSD 的 distillation objective 是：

```text
L_PCSD
= 1/M · sum_{k,i} m_{k,i} w_{k,i} δ_{k,i}
```

其中：

```text
M = sum_{k,i} m_{k,i}
```

但优化时：

```text
teacher logprob 和 w_{k,i} 都 detach。
```

因此实际梯度等价于：

```text
对 student sampled token 做 weighted NLL，
权重是 PCSD 计算出来的 teacher reliability。
```

附录伪代码写得更明确：

```text
L_PCSD
= 1/M sum m w [ sg(log π_T) - log π_S ]
```

也就是说，真正更新 student 的项是：

```text
- w_{k,i} log π_θ(y_{k,i}|h^S_{k,i})
```

注意它不是 full-vocab KL，而是：

```text
sampled-token distillation / weighted behavior cloning on student tokens
```

## 总 loss

PCSD 和 GRPO 联合优化：

```text
L_total = L_GRPO + λ_PCSD L_PCSD
```

其中 GRPO 使用：

```text
1. group-relative trajectory reward
2. clipped policy-ratio objective
3. reference-policy KL regularization
```

所以：

```text
GRPO 提供 sparse environmental reward 的方向；
PCSD 提供 token-level dense teacher supervision。
```

## 一个具体计算例子

假设某个 response 的连续 5 个 token 上，teacher-student gap 是：

```text
δ = [0.8, 0.7, 0.6, -0.2, 0.9]
```

如果只看第 1 个 token 的点值：

```text
δ_1 = 0.8
```

会觉得 teacher 很支持。

如果用窗口 `N=3`、`α=0.8`：

```text
δbar_1
= (0.8 + 0.8*0.7 + 0.8^2*0.6) / (1 + 0.8 + 0.8^2)
= (0.8 + 0.56 + 0.384) / 2.44
= 0.715
```

这说明第 1 个 token 附近确实有持续 teacher support。

再看第 3 个 token：

```text
窗口: [0.6, -0.2, 0.9]
δbar_3 = (0.6 + 0.8*(-0.2) + 0.64*0.9) / 2.44
       = 1.016 / 2.44
       = 0.416
```

虽然 `δ_3=0.6` 为正，但局部持续性弱很多，权重会低一些。

如果这个窗口的 slope 是负的，比如：

```text
slope / δ_scale = -0.5
γ = 0.3
η = 1 - 0.3 * 0.5 = 0.85
```

最终：

```text
w = sigmoid(β_gate * δbar_adaptive) * 0.85
```

趋势项只会压低权重，不会因为正趋势额外放大。

## 端到端计算案例：从 logprob 到 PCSD loss

假设 student 在某个环境状态下生成 5 个 token：

```text
y = [go, to, fridge, open, it]
```

teacher 不重新 rollout，只在这条 student trajectory 上给同一批 token 打 logprob。

### 1. teacher / student logprob 与 discrepancy

| i | token | teacher logprob | student logprob | delta_i |
|---:|---|---:|---:|---:|
| 1 | go | -0.2 | -1.0 | 0.8 |
| 2 | to | -0.3 | -1.0 | 0.7 |
| 3 | fridge | -0.4 | -1.0 | 0.6 |
| 4 | open | -1.4 | -1.2 | -0.2 |
| 5 | it | -0.5 | -1.4 | 0.9 |

```text
delta_i = log pi_T(y_i | h_i^T) - log pi_S(y_i | h_i^S)
delta = [0.8, 0.7, 0.6, -0.2, 0.9]
```

`delta_i > 0` 表示 teacher 比 student 更支持这个 student token；`delta_i < 0` 表示 teacher 并不比 student 更认可它。

### 2. 对第 3 个 token 计算持续一致性

第 3 个 token 的单点 gap 是 `delta_3 = 0.6`。如果只看单点，会认为 teacher 明显支持它。但 PCSD 还会看局部窗口。

用简化窗口 `N=3`、`alpha=0.8`，第 3 个 token 的 forward window 是：

```text
[delta_3, delta_4, delta_5] = [0.6, -0.2, 0.9]

delta_bar_3^{N=3}
= (0.6 + 0.8*(-0.2) + 0.8^2*0.9) / (1 + 0.8 + 0.8^2)
= (0.6 - 0.16 + 0.576) / 2.44
= 0.416
```

这说明第 3 个 token 虽然单点为正，但附近 support 不算稳定。

### 3. 自适应窗口插值

论文主实验是 `N_min=1, N_max=8`。这里为了手算，简化为 `N_min=1, N_max=3, tau_low=0.05, tau_high=0.5`。

```text
delta_bar_3^{N_min=1} = delta_3 = 0.6
delta_bar_3^{N_max=3} = 0.416
```

窗口 `[0.6, -0.2, 0.9]` 的均值和方差近似为：

```text
mu = (0.6 - 0.2 + 0.9) / 3 = 0.433

sigma^2
= [(0.6-0.433)^2 + (-0.2-0.433)^2 + (0.9-0.433)^2] / 3
= 0.216
```

插值系数：

```text
r = clip((sigma^2 - tau_low) / (tau_high - tau_low), 0, 1)
  = clip((0.216 - 0.05) / (0.5 - 0.05), 0, 1)
  = 0.369
```

自适应 gap：

```text
delta_bar_3^{adaptive}
= (1-r) * delta_bar_3^{N_min} + r * delta_bar_3^{N_max}
= 0.631 * 0.6 + 0.369 * 0.416
= 0.532
```

含义：局部波动不小，所以 PCSD 不完全相信单点 `0.6`，而是向更平滑的长窗口估计 `0.416` 靠近。

### 4. 趋势调制和最终权重

假设第 3 个 token 的窗口 slope 估计为负：

```text
slope / delta_scale = -0.5
gamma = 0.3
```

趋势因子：

```text
eta_3 = clip(1 - gamma * ReLU(-slope / delta_scale), 0, 1)
      = 1 - 0.3 * 0.5
      = 0.85
```

再用 `beta_gate=5.0` 做 sigmoid gating：

```text
w_3 = sigmoid(beta_gate * delta_bar_3^{adaptive}) * eta_3
    = sigmoid(5.0 * 0.532) * 0.85
    = sigmoid(2.66) * 0.85
    = 0.935 * 0.85
    = 0.795
```

### 5. 进入 PCSD loss

PCSD loss 是：

```text
L_PCSD = 1/M * sum_i m_i w_i [sg(log pi_T(y_i|h_i^T)) - log pi_S(y_i|h_i^S)]
```

`sg(log pi_T)` 和 `w_i` 都 detach，所以对 student 真正有梯度的是：

```text
- w_i * log pi_S(y_i | h_i^S)
```

第 3 个 token 上：

```text
log pi_S(fridge | h_3^S) = -1.0
w_3 = 0.795

loss contribution = -0.795 * (-1.0) = 0.795
```

这个例子说明：PCSD 不是看到 `delta_i > 0` 就直接强学，而是先检查 teacher support 是否在局部持续、是否波动太大、后续趋势是否下降，再把得到的可靠性权重乘到 sampled-token distillation loss 上。

## 超参数

论文主实验中 PCSD 的关键超参：

| Parameter | Value | 含义 |
|---|---:|---|
| `N_min` | 1 | 短窗口 |
| `N_max` | 8 | 长窗口 / 方差窗口 / 趋势窗口 |
| `alpha` | 0.8 | 指数衰减 |
| `tau_low` | 0.05 | 方差下阈值 |
| `tau_high` | 0.5 | 方差上阈值 |
| `gamma` | 0.3 | 负趋势调制强度 |
| `beta_gate` | 5.0 | sigmoid 门控锐度 |
| `lambda_PCSD` | 0.01 | 蒸馏 loss 系数 |

共同训练设置：

```text
optimizer: AdamW
learning rate: 1e-6
KL penalty coefficient: 0.01
gradient clipping: 1.0
training steps: 150
group size G: 8
GRPO clip epsilon: 0.2
```

ALFWorld：

```text
batch = 16 tasks × 8 rollouts = 128 trajectories
max prompt length = 2048
max response length = 512
episode action limit = 50
reward: success 10, otherwise 0
```

WebShop：

```text
1000 training tasks
128 fixed validation tasks
batch = 16 tasks × 8 rollouts
max prompt length = 4096
max response length = 512
episode step limit = 15
reward: success 10, otherwise 0
```

## 实验设置

Benchmarks：

```text
ALFWorld
WebShop
```

Backbones：

```text
Qwen2.5-3B-Instruct
Qwen3-1.7B-Instruct
```

模型命名说明：论文表格里写的是 Qwen3-1.7B-Instruct，但 Qwen 官方 Hugging Face 仓库通常对应的是 Qwen/Qwen3-1.7B，而不是 Qwen/Qwen3-1.7B-Instruct。

所以这里的 Qwen3-1.7B-Instruct 更像作者对 post-trained / chat-ready Qwen3-1.7B 的描述性写法，不一定是官方仓库名。复现实验时应优先检查作者代码配置；如果按官方 Qwen HF 命名，通常应理解为 Qwen/Qwen3-1.7B。

Baselines：

```text
Vanilla
Skill-Prompt*
OPSD
GRPO
Skill-GRPO
Skill-GRPO*
GRPO+OPSD
RLSD
SDAR
PCSD
```

`*` 表示 inference-time 也使用 retrieved skills。

PCSD 训练时 teacher 可使用 retrieved skills，但：

```text
student 训练和评测都不直接看到 privileged skills；
PCSD 评测时也不需要 inference-time skill retrieval。
```

## 完整主结果

### Qwen2.5-3B-Instruct

| Method | WebShop Score | WebShop Acc | Pick | Look | Clean | Heat | Cool | Pick2 | ALF Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 6.7 | 0.8 | 44.4 | 11.1 | 6.2 | 15.4 | 28.6 | 12.5 | 21.9 |
| Skill-Prompt* | 0.2 | 0.8 | 51.7 | 66.7 | 48.4 | 0.0 | 4.3 | 10.0 | 28.9 |
| OPSD | 11.3 | 3.1 | 48.8 | 41.7 | 16.7 | 0.0 | 15.8 | 16.7 | 28.1 |
| GRPO | 79.8 | 63.3 | 91.2 | 62.5 | 96.2 | 61.9 | 65.0 | 47.4 | 75.0 |
| Skill-GRPO | 77.3 | 60.9 | 88.9 | 71.4 | 58.8 | 70.6 | 40.7 | 29.2 | 60.2 |
| Skill-GRPO* | 76.3 | 66.4 | 94.3 | 57.1 | 100.0 | 66.7 | 73.1 | 57.1 | 80.5 |
| GRPO+OPSD | 77.8 | 66.4 | 100.0 | 82.4 | 85.7 | 75.0 | 70.0 | 60.0 | 81.2 |
| RLSD | 84.4 | 66.4 | 87.9 | 75.0 | 90.9 | 75.0 | 73.1 | 68.4 | 79.7 |
| SDAR | 83.4 | 67.2 | 97.1 | 62.5 | 100.0 | 61.9 | 75.0 | 84.2 | 84.4 |
| PCSD | 85.0 | 67.2 | 100.0 | 63.6 | 82.1 | 83.3 | 94.4 | 100.0 | 90.6 |

### Qwen3-1.7B-Instruct

| Method | WebShop Score | WebShop Acc | Pick | Look | Clean | Heat | Cool | Pick2 | ALF Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 46.5 | 4.7 | 25.0 | 22.2 | 3.1 | 0.0 | 21.4 | 4.2 | 12.5 |
| Skill-Prompt* | 23.0 | 2.3 | 10.3 | 50.0 | 16.1 | 0.0 | 0.0 | 5.0 | 9.4 |
| OPSD | 47.4 | 9.3 | 26.3 | 33.3 | 9.1 | 0.0 | 4.5 | 5.3 | 14.1 |
| GRPO | 67.3 | 38.3 | 71.1 | 41.7 | 36.4 | 40.0 | 31.8 | 31.6 | 46.1 |
| Skill-GRPO | 73.4 | 46.1 | 27.6 | 54.5 | 22.7 | 27.3 | 0.0 | 19.2 | 21.1 |
| Skill-GRPO* | 80.4 | 50.0 | 31.4 | 42.9 | 51.9 | 8.3 | 11.5 | 7.1 | 28.1 |
| GRPO+OPSD | 70.7 | 38.3 | 38.2 | 50.0 | 30.8 | 28.6 | 30.0 | 21.1 | 32.0 |
| RLSD | 74.0 | 50.8 | 50.0 | 37.5 | 61.5 | 19.0 | 50.0 | 21.1 | 42.2 |
| SDAR | 76.8 | 58.6 | 73.5 | 25.0 | 76.9 | 33.3 | 40.0 | 36.8 | 53.9 |
| PCSD | 78.9 | 58.6 | 63.3 | 68.8 | 50.0 | 69.2 | 48.1 | 63.6 | 59.4 |

## 主结果结论

论文的主要结论：

```text
1. PCSD 在两个 backbone 上都取得最高 ALFWorld Overall。
2. Qwen2.5-3B 上，PCSD 比 GRPO 高 15.6 点，比 SDAR 高 6.2 点。
3. Qwen3-1.7B 上，PCSD 比 GRPO 高 13.3 点，比 SDAR 高 5.5 点。
4. PCSD 不需要 inference-time skills，却超过 Skill-GRPO* 的 ALFWorld Overall。
5. WebShop 上 PCSD 保持竞争力：3B 上 Score 第一，Acc 与 SDAR 并列第一；1.7B 上 Acc 与 SDAR 并列第一，Score 第二。
```

## Training dynamics

论文跟踪了：

```text
1. teacher-student average gap
2. gate activation ratio，也就是 w_{k,i} > 0.5 的 token 比例
```

观察：

```text
mean gap 仍为负，但逐渐上升；
20%-28% 的 valid tokens 有 w > 0.5。
```

解释：

```text
PCSD 不是 hard-select 大量 token，
而是把蒸馏集中到一部分更有持续 teacher support 的 token 上。
```

## Unseen split 泛化

ALFWorld unseen split：

```text
PCSD Overall = 86.7
GRPO Overall = 70.9
SDAR Overall = 72.7
```

结论：

```text
PCSD 的持续一致性权重不只是 seen split 上有效，
在 unseen environment layout 上也比 GRPO / SDAR 更稳。
```

## 消融实验

### Component ablation

Qwen2.5-3B-Instruct on ALFWorld：

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | Overall |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCSD | 100.0 | 63.6 | 82.1 | 83.3 | 94.4 | 100.0 | 90.6 |
| Fixed N=1 | 90.5 | 82.4 | 100.0 | 69.2 | 84.0 | 64.0 | 82.8 |
| Fixed N=4 | 100.0 | 63.6 | 100.0 | 83.3 | 94.4 | 65.1 | 88.3 |
| w/o trend | 96.4 | 85.7 | 90.9 | 91.7 | 72.7 | 64.0 | 83.6 |
| w/o decay | 86.1 | 63.6 | 96.4 | 83.3 | 100.0 | 69.6 | 85.1 |

结论：

```text
1. 固定 N=1 退化为点式 token weighting，Overall 掉到 82.8。
2. 固定 N=4 比 N=1 好，但仍低于 adaptive PCSD。
3. 去掉 trend modulation，Overall 掉到 83.6，说明下降趋势衰减很重要。
4. 去掉 exponential decay，Overall 掉到 85.1，说明近邻加权比 uniform window 更稳。
5. 各组件单独看有些 subtask 会更高，但完整 PCSD Overall 最好。
```

### Distillation coefficient sensitivity

| lambda_PCSD | Pick | Look | Clean | Heat | Cool | Pick2 | Overall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 100.0 | 63.6 | 82.1 | 83.3 | 94.4 | 100.0 | 90.6 |
| 0.005 | 96.0 | 90.0 | 86.4 | 79.1 | 71.4 | 96.0 | 87.5 |
| 0.05 | 100.0 | 72.7 | 83.9 | 100.0 | 72.0 | 71.4 | 83.6 |
| 0.0 | 91.2 | 62.5 | 96.2 | 61.9 | 65.0 | 47.4 | 75.0 |

结论：

```text
1. lambda=0 等价于去掉 PCSD distillation，Overall 回到 GRPO 的 75.0。
2. lambda=0.005 已经显著提升到 87.5。
3. lambda=0.01 最好，Overall 90.6。
4. lambda=0.05 反而下降到 83.6，说明蒸馏过强会破坏 reward optimization 和 teacher guidance 的平衡。
```

## 和 RLSD / SDAR 的关系

RLSD：

```text
用 privileged teacher signal 调节 PG 的 token-level advantage，
方向仍由 verifier / environment reward 决定。
```

SDAR：

```text
更关注 agent self-distillation，但权重信号相对更局部。
```

PCSD 的区别：

```text
不只看单点 teacher-student gap，
而是看 teacher support 是否在局部窗口内持续存在；
同时用 adaptive window、exponential decay、trend attenuation、sigmoid gating 生成连续 token weights。
```

所以它更像：

```text
从“teacher 当前是否支持这个 token”
升级到“teacher 是否持续、稳定地支持这个局部片段”。
```

## 局限

论文自己也指出：

```text
1. aggregation / gating 超参是固定的；
2. teacher 是 frozen 的；
3. 权重机制没有随着训练自适应学习；
4. abrupt change 可能被局部平滑部分抹掉；
5. 未来可学习 context-dependent window / gate，或做 self-evolving distillation。
```

## 一句话结论

PCSD 的贡献可以概括为：

```text
在 Agentic OPSD 中，
不要盲目相信 teacher 的每个 token-level gap；
要看 teacher-favoring signal 是否在局部邻域中持续存在，
再把这种持续性变成连续 token-level distillation weight。
```

它的实验结果说明：

```text
这种 token-level persistent consistency weighting
比单点 weighting、固定窗口 weighting、朴素 GRPO+OPSD、RLSD、SDAR
在 ALFWorld 长程交互任务上更稳。
```

## BibTeX

```bibtex
@misc{lv2026pcsdpersistentconsistencyselfdistillation,
  title={PCSD: Persistent Consistency for Self-Distillation in Agentic Reinforcement Learning},
  author={Lv, Chunji and Wei, Yangguang and Liu, Junlin and Gao, Yang and Liu, Ming and Wang, Xinming and Wu, Jinyang and Wang, Guoren and Li, Changsheng},
  year={2026},
  eprint={2608.01837},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2608.01837}
}
```

