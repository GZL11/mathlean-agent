# SMOPD：Specialize-and-Merge Online Policy Distillation 论文解读

论文：

```text
SMOPD: Multi-Reward Reinforcement Learning via
Specialize-and-Merge Online Policy Distillation
```

链接：

```text
https://arxiv.org/abs/2608.03092
https://arxiv.org/pdf/2608.03092
```

作者单位：

```text
Qwen Large Model Application Team, Alibaba
University of Chinese Academy of Sciences
```

方法名：

```text
SMOPD = Specialize-and-Merge Online Policy Distillation
中文：专门化-合并式在线策略蒸馏
```

## 一句话总结

这篇论文处理的不是普通单 reward OPD，而是多 reward RL 的 reward balancing 问题。

核心问题是：

```text
多个 reward 同时训练一个 policy 时，即使每个 reward 都做 group-wise normalization，
稀疏 reward 仍然可能因为组内没有变化而几乎不给梯度，
最后被更细粒度、更稠密的 reward 掩盖。
```

SMOPD 的解法是两阶段：

```text
Stage 1: Specialize
  用不同 reward priority profile 训练多个 reward-specialized teachers。

Stage 2: Merge
  student 自己 rollout；
  多个 frozen teachers 在 student prefix 上给 top-k next-token distribution；
  把多个 teacher distribution 混成 teacher mixture；
  用 forward KL 做 token-level OPD；
  同时加一个 balanced GDPO anchor 维持 sequence-level 任务目标。
```

一句话公式：

```text
L_total = L_GDPO_anchor + lambda * L_OPD
```

所以它不是“只用 OPD loss”，也不是“多个 reward 直接加权求和”的方法，而是：

```text
先让每个 reward 在自己最容易学的 regime 中训练出 teacher，
再用 on-policy distillation 把这些 teacher 的能力合并回一个 student。
```

## 背景：为什么 GDPO 还不够

多 reward RL 里，每条 rollout 有多个 reward：

```text
r_i = (r_i^(1), ..., r_i^(K))
```

GRPO 通常先把 reward 求和，再在 group 内归一化：

```text
A_i^GRPO
= (sum_k r_i^(k) - mu_S) / (sigma_S + epsilon)
```

问题是不同 reward 的量纲、范围、密度不同，直接求和会互相遮蔽。

GDPO 的改进是先对每个 reward 维度单独 group-wise normalization，再加权求和：

```text
A_hat_i^(k)
= (r_i^(k) - mu^(k)) / sigma^(k)

A_i^sum
= sum_k w_k * A_hat_i^(k)

A_hat_i^GDPO
= (A_i^sum - mu_batch(A^sum)) / (sigma_batch(A^sum) + epsilon)
```

这能解决 reward scale 不同的问题，但不能解决 reward density 不同的问题。

论文重点指出：

```text
如果某个 reward 是 sparse / binary reward，
它在很多 prompt group 内可能所有 rollout 都是同一个值。
此时这个 reward 的组内方差为 0 或接近 0，
即使做 per-reward normalization，也没有有效排序信号。
```

比如 tool calling 任务里：

```text
accuracy reward: r_acc in [-3, 3]
  细粒度，能区分函数名、参数、参数值等。

format reward: r_fmt in {0, 1}
  二值，只看是否满足 XML 格式。
```

如果一个 group 里 8 条 rollout 的 format reward 全是 0：

```text
[0,0,0,0,0,0,0,0]
```

那么 format 维度在这个 group 内没有任何“哪条更好”的信息。

这就是论文说的：

```text
reward-density imbalance
中文：奖励密度不均衡
```

## 术语表

| 英文术语 | 建议中文 | 含义 |
|---|---|---|
| multi-reward RL | 多奖励强化学习 | 同一个 response 同时由多个 reward 维度打分 |
| reward scalarization | 奖励标量化 | 把多个 reward 加权合成一个标量目标 |
| reward masking | 奖励遮蔽 | 一个 reward 的训练信号被另一个 reward 掩盖 |
| reward-density imbalance | 奖励密度不均衡 | 不同 reward 给出有效组内差异的频率不同 |
| dense reward | 稠密奖励 | 经常能给 rollout 排序的细粒度 reward |
| sparse reward | 稀疏奖励 | 常常只有 0/1 或大多数样本同值的 reward |
| reward-priority profile | 奖励优先级配置 | GDPO 中不同 reward 的权重向量 `w` |
| reward-specialized teacher | 奖励专门化教师 | 用偏向某个 reward 的 profile 训练出的 teacher |
| teacher mixture | 教师混合分布 | 多个 teacher next-token distribution 的加权混合 |
| GDPO anchor | GDPO 锚点损失 | 用均衡 reward 权重继续做 sequence-level RL |
| imitation ceiling | 模仿上限 | 纯 OPD 最多匹配 teacher mixture，无法继续超过它 |
| component-wise top-k | 按 teacher 分量取 top-k | 每个 teacher 先取自己的 top-k token，再合并 |
| confidence-failure gate | 置信-失败门控 | 根据 teacher confidence、student failure、teacher disagreement 自适应混合 teacher |

## 方法：两阶段 SMOPD

### Stage 1：Reward-Specialized Teacher Training

假设有 `K` 个 reward 维度。对目标 reward `k`，训练一个偏向它的 teacher：

```text
A_teacher,k
= w_k^high * A_hat^(k)
  + sum_{j != k} w_j^low * A_hat^(j)
```

例如 accuracy + format 两个 reward：

```text
accuracy teacher:
  w = [0.9, 0.1]

format teacher:
  w = [0.1, 0.9]
```

因为 GDPO 已经把每个 reward 维度归一化到近似同尺度，所以 `0.9:0.1` 这个权重比会直接改变梯度贡献比例。论文明确说，在 `K=2` 时，这相当于把目标 reward 的信号相对另一个 reward 放大约 `9x`。

这样做的目的不是最终部署多个 teacher，而是先获得互补能力：

```text
accuracy teacher 学会函数/参数正确性；
format teacher 学会 XML wrapper / 输出格式；
后面再把两者合并成一个 student。
```

### Stage 2：Multi-Teacher Online Policy Distillation

给定 `M` 个 teacher：

```text
pi_1, ..., pi_M
```

student `pi_theta` 自己生成 response，也就是 on-policy rollout。对 student 生成序列的每个 prefix：

```text
s_t = x + y_<t
```

每个 teacher 在同一个 prefix 上给 next-token distribution：

```text
p_{pi_m}(. | s_t)
```

然后混成 teacher mixture：

```text
p_T^mix(v | s_t) propto sum_m alpha_m * p_{pi_m}(v | s_t)
```

默认：

```text
alpha_m = 1/M
```

即两个 teacher 时：

```text
alpha = [0.5, 0.5]
```

### 为什么只取 top-k

完整 vocab 上对多个 teacher 计算分布太贵，所以论文用 component-wise top-k：

```text
kappa = 16 by default
```

做法：

```text
1. 每个 teacher m 返回自己的 top-kappa token set T_m^kappa；
2. 每个 token 的 logprob 加上 log alpha_m；
3. 把所有 M * kappa 个候选 token pool 到一起；
4. 从 pool 里再取最大的 kappa 个作为 mixture target。
```

公式：

```text
p_tilde_T^mix(. | s_t)
= top-kappa_{m=1..M, v in T_m^kappa}
  [log alpha_m + log p_{pi_m}(v | s_t)]
```

注意：这是 exact mixture 的近似，不是完整 vocab 的精确 `logsumexp`。论文的解释是 teacher 的 top-k mass 很高，实测 top-16 已保留约 `0.994` 的 teacher probability mass，所以 `kappa=16` 基本够用。

### OPD loss

SMOPD 使用 forward KL：

```text
L_OPD
= sum_t KL(
     p_tilde_T^mix(. | s_t)
     ||
     p_theta(. | s_t)
   )
```

其中：

```text
teacher distribution = 多个 reward-specialized teachers 的 mixture
student distribution = 当前可训练 policy
prefix = student 自己 rollout 出来的 on-policy prefix
```

这里的 OPD 是 full-distribution / top-k distribution distillation，不是 sampled-token reward shaping。

### GDPO anchor

只做 OPD 会有一个问题：student 最多学到 teacher mixture，本身没有新的 reward optimization 方向。

所以论文加 balanced GDPO anchor：

```text
L_anchor = L_GDPO(pi_theta; w_equal)
```

其中：

```text
w_equal = (1/K, ..., 1/K)
```

两个 reward 时就是：

```text
w_equal = [0.5, 0.5]
```

总损失：

```text
L_total = L_anchor + lambda * L_OPD
```

默认：

```text
lambda = 1.0
```

两项作用不同：

```text
OPD:
  token-level dense supervision
  告诉 student 每个位置该像哪个 teacher mixture

GDPO anchor:
  sequence-level reward optimization
  告诉 student 整条 response 在均衡多 reward 下该升权还是降权
```

## 具体计算案例

假设一个 tool calling prompt，需要输出：

```xml
<think>...</think>
<tool_call>{"name": "get_weather", "arguments": {"city": "Beijing"}}</tool_call>
<response>...</response>
```

有两个 reward：

```text
r_acc: 函数名/参数是否正确，范围 [-3, 3]
r_fmt: XML 格式是否正确，范围 {0, 1}
```

### Stage 1：训练两个 teacher

accuracy teacher 用：

```text
w = [0.9, 0.1]
```

它更关注：

```text
get_weather
city
Beijing
```

format teacher 用：

```text
w = [0.1, 0.9]
```

它更关注：

```text
<think>
</think>
<tool_call>
</tool_call>
<response>
</response>
```

### Stage 2：student rollout 到某个 prefix

student 当前生成到：

```text
s_t = x + "<think>I need weather.</think>\n"
```

此时下一个 token 应该大概率是：

```text
<tool_call>
```

两个 teacher 在该 prefix 上给分布：

```text
format teacher:
  "<tool_call>" -> 0.70
  "<response>"  -> 0.10
  "{"           -> 0.05
  other         -> 0.15

accuracy teacher:
  "{"           -> 0.35
  "<tool_call>" -> 0.25
  "get_weather" -> 0.15
  other         -> 0.25
```

均匀混合：

```text
p_mix("<tool_call>")
= 0.5 * 0.70 + 0.5 * 0.25
= 0.475

p_mix("{")
= 0.5 * 0.05 + 0.5 * 0.35
= 0.200

p_mix("<response>")
= 0.5 * 0.10 + 0.5 * 0
= 0.050

p_mix("get_weather")
= 0.5 * 0 + 0.5 * 0.15
= 0.075
```

假设 student 当前分布是：

```text
student:
  "{"           -> 0.45
  "<tool_call>" -> 0.20
  "<response>"  -> 0.10
  "get_weather" -> 0.05
```

那么 OPD 用：

```text
KL(p_mix || p_student)
= sum_v p_mix(v) * log(p_mix(v) / p_student(v))
```

这会把 student 往 teacher mixture 拉：

```text
提高 "<tool_call>" 概率；
保留 accuracy teacher 对 "{" / "get_weather" 的偏好；
不让 student 只学格式或只学参数。
```

同时，这条完整 response 还会被 GDPO anchor 根据：

```text
r_acc + r_fmt 的均衡优势
```

决定整条 response 的升权/降权。

## 为什么需要 anchor：imitation ceiling

论文附录给了一个很重要的解释。

把 frozen teacher mixture 写成：

```text
q(. | s)
```

纯 OPD 目标是：

```text
L_OPD(theta)
= E_{s ~ d_theta} [
     KL(q(. | s) || p_theta(. | s))
   ]
```

KL 非负，且当：

```text
p_theta(. | s) = q(. | s)
```

时达到最小值 0。此时梯度：

```text
grad_theta KL(q || p_theta)
= - sum_v q(v | s) grad_theta log p_theta(v | s)
```

当 `p_theta = q`：

```text
- sum_v q(v | s) grad log p_theta(v | s)
= - sum_v grad p_theta(v | s)
= - grad 1
= 0
```

所以纯 OPD 学到 teacher mixture 后就没有继续优化信号了。这就是：

```text
imitation ceiling
中文：模仿上限
```

GDPO anchor 的梯度是 reward-driven policy gradient：

```text
-grad L_anchor
= E_{o ~ pi_theta} [
    A_hat^GDPO(o) grad log pi_theta(o)
  ]
```

它的固定点是 reward 的局部最优，而不是 teacher mixture。因此在 `p_theta = q` 时：

```text
grad L_total
= grad L_anchor + lambda * 0
```

一般仍然非零，可以继续把 student 推过 teacher mixture。

这也是为什么 SMOPD 不是只用 OPD loss。

## 为什么 OPD 能帮 sparse reward

对二值 reward，假设当前 policy 的成功率是：

```text
p_succ
```

一个 prompt group 里采样 `G` 条 rollout。如果所有 rollout 的 binary reward 都相同，则这个 reward 维度没有组内差异。

全 1 或全 0 的概率：

```text
(p_succ)^G + (1 - p_succ)^G
```

有混合信号的概率：

```text
1 - (p_succ)^G - (1 - p_succ)^G
```

当 `p_succ -> 0` 时近似：

```text
G * p_succ
```

所以如果模型几乎不会输出正确格式，format reward 的 group 内差异概率非常低，GDPO anchor 在 format 维度几乎没有有效信号。

OPD 不依赖这个组内 reward 方差。format teacher 在 `<tool_call>`、`</tool_call>` 等 token 上给 dense distribution guidance，可以先把 `p_succ` 拉起来。等成功率不再接近 0，group 内开始出现 0/1 混合，GDPO anchor 才能继续发挥作用。

## 为什么用 Forward KL

SMOPD 默认：

```text
Forward KL = KL(teacher mixture || student)
```

论文理由是：多 teacher mixture 里不同 teacher 可能负责不同 mode。

例如同一个 prefix：

```text
format teacher 高概率给 XML wrapper token；
accuracy teacher 高概率给 function / argument token。
```

Forward KL 更 mode-covering，会鼓励 student 覆盖 teacher mixture 的高概率区域：

```text
不要只学其中一个 teacher 的 mode。
```

Reverse KL 更 mode-seeking，可能只追某个 dominant mode，丢掉另一个 teacher 的专门能力。

消融也支持这个选择：

```text
Forward KL kappa=16: Overall 5.646
Reverse KL sampled PG: Overall 5.563
```

## 训练设置

### Complementary rewards：RLLA tool calling

| 项目 | 设置 |
|---|---|
| Base models | Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct |
| Training data | RLLA-4K train split from ToolRL |
| Reward dimensions | `r_acc in [-3,3]`, `r_fmt in {0,1}` |
| Teacher weights | acc `[0.9,0.1]`, fmt `[0.1,0.9]` |
| Anchor weights | `[0.5,0.5]` |
| Teacher mixing | uniform, `alpha=0.5` |
| Training steps | 150 for 1.5B, 100 for 3B |
| Rollouts per prompt | `G=8` |
| Learning rate | `1e-6` |
| OPD top-kappa | 16 |
| OPD coefficient lambda | 1.0 |
| Distillation loss | forward KL over top-kappa support |
| Eval | greedy, temp 0.0, top-p 1.0 |
| Max new tokens | 1024 |
| Hardware | 8 x NVIDIA H100 |

### Conflicting rewards：helpful + harmless

| 项目 | 设置 |
|---|---|
| Base models | Qwen2.5-3B/7B-Instruct, Llama-3.2-3B-Instruct |
| Training prompts | Alpaca prompt-only train split |
| Useful reward | Qwen2.5-7B-SafeRLHF-RM |
| Harmless reward | Qwen2.5-7B-SafeRLHF-CM |
| Teacher weights | useful `[0.7,0.3]`, harmless `[0.3,0.7]` |
| Anchor weights | `[0.5,0.5]` |
| Rollouts per prompt | `G=4` |
| Training steps | 100 for 3B, 66 for 7B |
| Rollout temp / top-p | 0.7 / 1.0 |
| Learning rate | `1e-6` |
| OPD top-kappa / lambda | 16 / 1.0 |
| Eval benchmarks | HH-RLHF, PKU-SafeRLHF, Alpaca |
| Hardware | 8 x NVIDIA H100 |

## 主结果：Complementary rewards / RLLA

指标：

```text
Acc Reward: accuracy reward
Format Pass: XML format pass rate
RLLA Mean: r_acc + r_fmt
BFCL AST: BFCL-v4 function-calling AST accuracy
API-Bank L1/L2/L3/Avg: LLM-judge functional accuracy
```

| Backbone | Method | Acc Reward | Format Pass | RLLA Mean | BFCL AST | API L1 | API L2 | API L3 | API Avg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B | GRPO Baseline | 1.737 | 5.0% | 1.787 | 68.6% | 79.2 | 71.6 | 84.0 | 79.4 |
| Qwen2.5-1.5B | GDPO Baseline | 1.761 | 8.8% | 1.849 | 72.6% | 84.7 | 74.6 | 84.7 | 83.6 |
| Qwen2.5-1.5B | GD^2PO Baseline | 1.776 | 8.8% | 1.864 | 73.3% | 82.2 | 71.6 | 82.4 | 81.1 |
| Qwen2.5-1.5B | Accuracy Teacher [0.9,0.1] | 1.779 | 1.2% | 1.791 | 70.5% | 81.2 | 73.1 | 84.0 | 80.9 |
| Qwen2.5-1.5B | Format Teacher [0.1,0.9] | 1.771 | 97.5% | 2.746 | 70.5% | 85.5 | 73.1 | 67.9 | 80.2 |
| Qwen2.5-1.5B | SMOPD | 1.765 | 97.5% | 2.740 | 70.7% | 90.0 | 82.1 | 80.9 | 87.1 |
| Qwen2.5-3B | GRPO Baseline | 1.783 | 97.5% | 2.758 | 75.8% | 81.5 | 65.7 | 47.3 | 72.2 |
| Qwen2.5-3B | GDPO Baseline | 1.753 | 97.5% | 2.728 | 75.3% | 80.7 | 74.6 | 64.1 | 76.4 |
| Qwen2.5-3B | GD^2PO Baseline | 1.732 | 97.5% | 2.707 | 76.5% | 82.0 | 67.2 | 61.1 | 75.7 |
| Qwen2.5-3B | Accuracy Teacher [0.9,0.1] | 1.795 | 95.0% | 2.745 | 77.2% | 83.2 | 68.7 | 61.8 | 76.9 |
| Qwen2.5-3B | Format Teacher [0.1,0.9] | 1.740 | 97.2% | 2.712 | 76.1% | 84.2 | 70.2 | 62.6 | 77.9 |
| Qwen2.5-3B | SMOPD | 1.763 | 97.5% | 2.738 | 77.2% | 83.5 | 64.2 | 65.7 | 77.4 |

结论：

```text
1. 1.5B 上 GDPO / GD^2PO 的 format pass 只有 8.8%，说明二值 format reward 被 accuracy reward 掩盖。
2. Format Teacher 把 format pass 拉到 97.5%，证明该能力可学，只是在 balanced scalarized RL 下信号太弱。
3. SMOPD 也达到 97.5% format pass，同时 API-Bank Avg 达到 87.1，是 1.5B 最好。
4. 3B 上 base 已经能学会 format，所以 SMOPD 的 in-domain 增益变小，但 BFCL AST / API Avg 仍有竞争力。
```

## 主结果：Conflicting rewards / Safe alignment

指标：

```text
U. = Useful score
H. = Harmless score
Avg = (U + H) / 2
Overall = HH-RLHF / PKU-SafeRLHF / Alpaca 三个 Avg 的均值
```

| Backbone | Method | HH U | HH H | HH Avg | PKU U | PKU H | PKU Avg | Alpaca U | Alpaca H | Alpaca Avg | Overall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B | GRPO | 4.400 | 5.690 | 5.045 | 5.372 | 6.878 | 6.125 | 5.281 | 6.020 | 5.650 | 5.607 |
| Qwen2.5-3B | GDPO | 4.382 | 5.704 | 5.043 | 5.373 | 6.883 | 6.128 | 5.247 | 6.001 | 5.624 | 5.598 |
| Qwen2.5-3B | GD^2PO | 4.485 | 5.730 | 5.107 | 5.406 | 6.879 | 6.143 | 5.379 | 6.023 | 5.701 | 5.650 |
| Qwen2.5-3B | Useful Teacher | 4.448 | 5.472 | 4.960 | 5.303 | 6.700 | 6.001 | 5.407 | 5.736 | 5.572 | 5.511 |
| Qwen2.5-3B | Harmless Teacher | 4.320 | 5.784 | 5.052 | 5.375 | 6.919 | 6.147 | 5.149 | 6.176 | 5.662 | 5.620 |
| Qwen2.5-3B | SMOPD | 4.511 | 5.743 | 5.127 | 5.426 | 6.869 | 6.147 | 5.392 | 6.071 | 5.732 | 5.669 |
| Qwen2.5-7B | GRPO | 4.237 | 5.469 | 4.853 | 5.386 | 6.834 | 6.110 | 4.976 | 5.647 | 5.312 | 5.425 |
| Qwen2.5-7B | GDPO | 4.318 | 5.439 | 4.878 | 5.329 | 6.766 | 6.048 | 5.323 | 5.813 | 5.568 | 5.498 |
| Qwen2.5-7B | GD^2PO | 4.379 | 5.511 | 4.945 | 5.390 | 6.801 | 6.096 | 5.434 | 5.909 | 5.672 | 5.571 |
| Qwen2.5-7B | Useful Teacher | 4.515 | 5.313 | 4.914 | 5.430 | 6.658 | 6.044 | 5.513 | 5.572 | 5.543 | 5.500 |
| Qwen2.5-7B | Harmless Teacher | 4.220 | 5.607 | 4.913 | 5.354 | 6.889 | 6.122 | 5.021 | 6.041 | 5.531 | 5.522 |
| Qwen2.5-7B | SMOPD | 4.475 | 5.656 | 5.065 | 5.448 | 6.837 | 6.143 | 5.421 | 6.036 | 5.729 | 5.646 |
| Llama-3.2-3B | GRPO | 4.264 | 5.510 | 4.887 | 5.117 | 6.645 | 5.881 | 5.340 | 5.891 | 5.615 | 5.461 |
| Llama-3.2-3B | GDPO | 4.414 | 5.642 | 5.028 | 5.315 | 6.761 | 6.038 | 5.366 | 5.998 | 5.682 | 5.583 |
| Llama-3.2-3B | GD^2PO | 4.438 | 5.657 | 5.047 | 5.319 | 6.776 | 6.047 | 5.429 | 6.012 | 5.721 | 5.605 |
| Llama-3.2-3B | Useful Teacher | 4.174 | 5.177 | 4.676 | 5.198 | 6.512 | 5.855 | 5.296 | 5.489 | 5.393 | 5.308 |
| Llama-3.2-3B | Harmless Teacher | 3.917 | 5.271 | 4.594 | 5.046 | 6.626 | 5.836 | 4.869 | 5.724 | 5.296 | 5.242 |
| Llama-3.2-3B | SMOPD | 4.437 | 5.633 | 5.035 | 5.349 | 6.756 | 6.052 | 5.406 | 5.962 | 5.684 | 5.590 |

结论：

```text
1. Qwen2.5-3B: SMOPD Overall 5.669，高于 GD^2PO 5.650。
2. Qwen2.5-7B: SMOPD Overall 5.646，高于 GD^2PO 5.571，提升更明显。
3. Llama-3.2-3B: SMOPD 5.590，高于 GDPO 5.583，但低于 GD^2PO 5.605。
4. 论文解释 Llama 上的差距主要来自 anchor：SMOPD 用的是 GDPO anchor，而 GD^2PO 改进了 advantage estimator。换更强 anchor 可能是正交改进方向。
```

## 消融实验 1：Anchor 是否必要

Qwen2.5-7B safe alignment，`kappa=32`：

| Method | Useful | Harmless | Overall |
|---|---:|---:|---:|
| GDPO baseline / anchor only | 4.990 | 6.006 | 5.498 |
| Useful teacher | 5.153 | 5.848 | 5.500 |
| Harmless teacher | 4.865 | 6.179 | 5.522 |
| OPD only / no anchor | 5.011 | 6.077 | 5.544 |
| SMOPD / OPD + anchor | 5.099 | 6.178 | 5.639 |

结论：

```text
OPD only 可以超过单个 teacher，但接近 teacher mixture 的模仿上限；
加 anchor 后 Overall 从 5.544 到 5.639，说明 balanced reward PG 继续提供优化方向。
```

## 消融实验 2：top-k support 和 KL 方向

Qwen2.5-7B safe alignment：

| Objective | Support | Useful | Harmless | Overall |
|---|---|---:|---:|---:|
| GDPO baseline | -- | 4.990 | 6.006 | 5.498 |
| Forward KL | kappa=16 | 5.115 | 6.176 | 5.646 |
| Forward KL | kappa=32 | 5.099 | 6.178 | 5.639 |
| Forward KL | kappa=64 | 5.102 | 6.181 | 5.642 |
| Reverse KL / sampled PG | sampled | 5.034 | 6.093 | 5.563 |

结论：

```text
1. kappa=16/32/64 差异很小，Overall 只在 5.639-5.646 间波动。
2. top-16 已保留约 0.994 的 teacher probability mass，所以更大 kappa 帮助有限。
3. Forward KL 明显优于 sampled-token Reverse KL。
4. 原因是多 teacher mixture 需要 mode-covering，RKL 的 mode-seeking 更容易丢掉某个 teacher 的专门能力。
```

## 消融实验 3：reward priority profile 强度

Qwen2.5-3B RLLA，使用更温和的 teacher 权重：

```text
aggressive: [0.9,0.1] / [0.1,0.9]
moderate:   [0.7,0.3] / [0.3,0.7]
```

| Backbone | Method | Acc Reward | Format Pass | RLLA Mean | BFCL AST |
|---|---|---:|---:|---:|---:|
| Qwen2.5-3B | GDPO Baseline | 1.753 | 97.5% | 2.728 | 75.3% |
| Qwen2.5-3B | Accuracy Teacher [0.7,0.3] | 1.763 | 94.4% | 2.707 | 76.4% |
| Qwen2.5-3B | Format Teacher [0.3,0.7] | 1.734 | 97.5% | 2.709 | 73.4% |
| Qwen2.5-3B | SMOPD | 1.782 | 97.5% | 2.757 | 76.9% |

结论：

```text
SMOPD 不一定需要特别激进的 teacher specialization。
只要 teachers 仍然互补，moderate profile 也能合并出更好的 student。
这里 moderate SMOPD 的 RLLA 2.757 甚至高于 main aggressive 3B 的 2.738。
```

## 消融实验 4：teacher mixing gate

论文比较：

```text
uniform mixing: alpha_m = 1/M
sequence-level confidence-failure gate
token-level confidence-failure gate
```

gate 用三个信号：

```text
teacher confidence:
  top-k distribution 是否尖锐，由负熵 + top2 margin 计算。

student reward failure:
  student 在该 teacher 对应 reward 维度上失败程度。

teacher disagreement:
  teacher top-1 token 是否不同。
```

teacher confidence：

```text
c_{m,t}
= 1/2 * (1 - H[p_m(.|s_t)] / log kappa)
  + 1/2 * (p_m^(1)(s_t) - p_m^(2)(s_t))
```

reward failure：

```text
f_m
= 1 - clamp((r_m - r_min) / (r_max - r_min), 0, 1)
```

### RLLA mixing results

| Backbone | Mixing | Acc Reward | Format Pass | RLLA Mean | BFCL AST |
|---|---|---:|---:|---:|---:|
| 1.5B | SMOPD uniform | 1.765 | 97.5% | 2.740 | 70.7% |
| 1.5B | SMOPD seq gate | 1.741 | 94.3% | 2.684 | 70.9% |
| 1.5B | SMOPD token gate | 1.787 | 90.3% | 2.690 | 70.9% |
| 3B | SMOPD uniform | 1.782 | 97.5% | 2.757 | 76.9% |
| 3B | SMOPD seq gate | 1.765 | 96.9% | 2.733 | 76.6% |
| 3B | SMOPD token gate | 1.755 | 97.5% | 2.730 | 77.1% |

### Safe alignment mixing results

| Backbone | Mixing | HH Avg | PKU Avg | Alpaca Avg | Overall |
|---|---|---:|---:|---:|---:|
| Qwen2.5-3B | uniform | 5.127 | 6.147 | 5.732 | 5.669 |
| Qwen2.5-3B | seq gate | 5.134 | 6.152 | 5.720 | 5.669 |
| Qwen2.5-3B | token gate | 5.110 | 6.142 | 5.717 | 5.656 |
| Qwen2.5-7B | uniform | 5.065 | 6.143 | 5.729 | 5.646 |
| Qwen2.5-7B | seq gate | 5.080 | 6.136 | 5.687 | 5.635 |
| Qwen2.5-7B | token gate | 5.082 | 6.149 | 5.723 | 5.651 |
| Llama-3.2-3B | uniform | 5.035 | 6.052 | 5.684 | 5.590 |
| Llama-3.2-3B | seq gate | 5.001 | 6.034 | 5.701 | 5.578 |
| Llama-3.2-3B | token gate | 5.016 | 6.034 | 5.667 | 5.572 |

结论：

```text
1. 三种 mixing 方法差距很小。
2. RLLA 上 uniform 最强。
3. safe alignment 上 token gate 只在 Qwen2.5-7B 略高：5.651 vs 5.646。
4. 复杂 gate 没有稳定收益，所以默认 uniform mixing。
```

## 和 MOPD / 普通 OPD 的区别

MOPD 是多 teacher OPD，用来合并不同 domain / capability teachers，例如 math、code、instruction following。

SMOPD 的问题不同：

```text
不是多个 domain；
而是同一个 prompt / 同一个 domain 内有多个 reward 同时作用。
```

因此不能靠 routing 把样本分给不同 teacher，因为同一条 response 同时要满足：

```text
accuracy + format
helpful + harmless
```

SMOPD 的 teacher 是按 reward 维度专门化出来的：

```text
teacher_m = reward-specialized policy
```

OPD 的作用也不是单纯压缩 teacher，而是：

```text
把不同 reward 专门化 teacher 的 token-level 偏好合并到一个 student，
同时用 balanced GDPO anchor 保持原始多 reward 目标。
```

## 局限和适用条件

适用：

```text
1. 多 reward 之间密度不同，比如 dense accuracy + sparse format。
2. 某些 reward 需要专门提高权重才能学到。
3. 能负担训练多个 reward-specialized teachers。
4. 希望最终只部署一个 unified policy。
```

局限：

```text
1. 成本更高：Stage 1 要训练多个 teacher，Stage 2 还要多 teacher scoring。
2. teacher profile 需要设计：虽然 moderate/aggressive 都有效，但仍需要确定每个 reward 的偏置权重。
3. 如果 teacher 本身不互补，或某个 reward teacher 学不到目标能力，合并收益会受限。
4. 当前 anchor 是 GDPO，论文自己也指出换更强的 anchor 可能继续提升。
```

## 我对这篇论文的理解

SMOPD 的关键贡献不是发明了新的 KL，也不是发明了复杂 teacher gate，而是把 multi-reward RL 拆成两个更容易处理的问题：

```text
问题 1：每个 reward 如何先学好？
答案：给它高 priority，训练 reward-specialized teacher。

问题 2：多个 reward-specialized policies 如何合并？
答案：在 student on-policy prefix 上，用 multi-teacher FKL distillation 合并 token distribution；
     再用 balanced GDPO anchor 保持最终任务目标。
```

它背后的判断是：

```text
多 reward 难的不是 reward 归一化公式本身，
而是不同 reward 的有效梯度出现频率不同。
```

因此与其在一个 policy 里同时平衡所有 reward，不如：

```text
先允许每个 reward 在偏置环境中充分表达；
再用 OPD 把这些表达合并成一个 policy。
```

