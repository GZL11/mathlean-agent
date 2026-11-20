# ShortOPD：用 Short-to-Long OPD 恢复结构化剪枝后的生成能力

本文档解读论文：

```text
ShortOPD: Recovering Pruned LLMs with Short-to-Long On-Policy Distillation
https://arxiv.org/abs/2607.13124
```

## 一句话总结

ShortOPD 是一篇 **结构化剪枝后的生成能力恢复** 论文。

它的核心判断是：

```text
剪枝后的 LLM 在多选题上可能还行，但自由生成会崩；
能力没有完全消失，而是被降级到了采样分布里；
恢复时应该在剪枝模型自己的 on-policy states 上，用剪枝前模型做 token-level distillation。
```

普通 OPD 已经满足：

```text
on-policy states + dense token-level teacher distribution
```

但剪枝学生早期长 rollout 很容易掉进重复尾巴。ShortOPD 解决的是：

```text
不要早期就浪费大量 token 和 teacher compute 在低信息量重复后缀上；
先短后长，等模型恢复后再放开 rollout horizon。
```

## Teacher / Student 设置

实验主对象：

```text
teacher:
  未剪枝的 Qwen3-4B-Instruct-2507

student:
  结构化剪枝后的 Qwen3-4B-Instruct-2507
```

剪枝方式：

```text
Block-Influence depth pruning
```

具体配置：

```text
原始层数: 36 transformer blocks
删除层: 25–33，按 zero-based block id
剪后层数: 27
参数减少: 约 25%
```

teacher 始终冻结：

```text
π_T = pre-compression self
```

student 从剪枝模型初始化：

```text
π_θ = compressed student
```

## 论文发现的问题

### 1. 生成能力崩得比识别任务更严重

很多结构化剪枝论文主要看：

```text
MMLU
HellaSwag
ARC
```

这种 recognition / multiple-choice 任务。  
但部署时更多是自由生成：

```text
数学推理
代码生成
开放问答
摘要
多轮对话
```

论文发现，剪掉 25% 参数后：

```text
greedy pass@1 几乎崩掉
```

比如 Qwen3-4B-Instruct 剪掉 4 层时，GSM8K greedy 只有 49.0，但 best-of-64 能到 91.2，甚至超过 teacher greedy 的 88.1。

这说明：

```text
正确轨迹还在 student 的采样分布里，
只是 greedy 解码排不到前面。
```

作者称之为：

```text
useful generations are demoted, not erased
```

### 2. 主要失败模式是 suffix repetition

在 25% 剪枝点，模型还不是完全乱码，而是常常：

```text
前面有一点可用内容，
后面进入周期性重复 suffix。
```

论文在 192-prompt probe 上统计：

```text
25% 参数剪枝后 suffix repetition 率很高
```

更关键的是，重复后缀上的 teacher-student 差异很低：

```text
重复 suffix tokens 平均 JSD: 0.0014
普通 tokens 平均 JSD: 0.051
```

teacher NLL 也极低：

```text
重复 suffix teacher NLL: 3e-5
普通 tokens teacher NLL: 0.68
```

含义是：

```text
进入重复 loop 后，teacher 也很容易沿着这个 prefix 继续预测重复 token；
因此这些尾部 token 的 OPD 梯度信息很少。
```

所以早期如果固定长 horizon，例如 2048 / 8192：

```text
大量 rollout 和 teacher scoring 预算被浪费在低信息重复尾巴上。
```

## OPD 基线怎么做

对每个 prompt `x`：

```text
1. student 在 x 上 on-policy rollout:
   y ~ π_θ(· | x)

2. teacher 在同一条 student trajectory 上重新打分:
   π_T(· | x, y_<t)

3. student 匹配 teacher 的 next-token distribution
```

损失：

```text
L_OPD(θ)
= E_x E_{y ~ π_θ(·|x)}
  [ 1/|y| * sum_t D_α(π_T(·|x,y_<t), π_θ(·|x,y_<t)) ]
```

论文用的是 generalized JSD：

```text
α = 0.5
```

实现上：

```text
top-100 logits + aggregated tail mass
importance weights clipped at 2.0
```

注意：

```text
没有 policy-gradient loss；
没有 RL reward；
没有 verifier。
```

这就是 label-free 的 on-policy self-distillation。

## ShortOPD 的核心：动态 rollout horizon

ShortOPD 在普通 OPD 外面加了一个 budget controller。

它控制每步的 response rollout budget：

```text
H_s <= H_max
```

不是一直固定跑满 `H_max`。

它每步观察三个统计量：

```text
r_s: suffix-repetitive rollouts 的比例
q_s: clean truncation rate
L_s: 平均 effective length
```

### r_s：重复率

对每条 rollout `y_i`，检测最后一个窗口是否出现严重周期性重复：

```text
a_i = TerminalLoop(y_i)
r_s = mean_i a_i
```

`a_i = 1` 表示这条输出尾部有严重重复 loop。

### q_s：干净截断率

只统计：

```text
没有重复
但因为达到当前 H_s 而被截断
```

形式是：

```text
b_i = 1[ |y_i| = H_s and a_i = 0 ]
q_s = mean_i b_i
```

含义：

```text
如果很多非重复输出都打满了当前长度，
说明当前 horizon 可能太短，需要增长。
```

### L_s：effective length

如果检测到重复 suffix，就把重复开始前的 prefix 长度作为有效长度：

```text
ℓ_i = loop onset position
```

如果没有重复：

```text
ℓ_i = |y_i|
```

平均：

```text
L_s = mean_i ℓ_i
```

然后对 `r_s, q_s, L_s` 都做 EMA 平滑：

```text
(r_bar_s, q_bar_s, L_bar_s) = EMA(r_s, q_s, L_s)
```

## horizon 控制规则

ShortOPD 先生成一个目标 horizon：

```text
H*_s =
  min(H_s, λ L_bar_s),              if r_bar_s > ρ_high
  γ_up H_s,                         if r_bar_s < ρ_low and q_bar_s > τ
  H_s,                              otherwise
```

含义：

```text
高重复:
  shrink，缩到 effective length 附近，但保留一点余量

低重复 + 高 clean truncation:
  grow，说明模型已经能干净地用满当前 horizon

否则:
  hold
```

最后再做 budget EMA：

```text
H_{s+1} = clip[H_min, H_max](
  round_16(β_H H_s + (1 - β_H) H*_s)
)
```

论文默认超参：

```text
ρ_low = 0.20
ρ_high = 0.45
τ = 0.10
λ = 1.15
γ_up = 1.25
EMA decay = 0.7
H_min / H_max = 1024 / 2048
budget rounding = multiple of 16
```

## TerminalLoop 是怎么检测重复的

检测器只看 response 最后：

```text
W = 512 tokens
```

检查周期：

```text
p = 1..10
```

对每个周期 p，比较：

```text
z_{p+k} == z_k
```

如果最后 `A=32` 个比较位置的 agreement 高于：

```text
η = 0.90
```

说明周期性重复一直延伸到结尾。

再要求：

```text
至少 C = 3 个周期
尾部长度至少 L_min = 64
并且满足严重条件:
  tail >= 128 tokens
  或 tail >= 30% suffix window
```

检测器返回：

```text
是否 severe
重复 tail 的开始位置
period p
```

注意：

```text
低 loss / 低 teacher NLL 只用于局部 refine loop boundary；
是否 shrink 只由 repetition rate gate 决定。
```

这避免 controller 被低 loss 本身误导。

## 数据构造

恢复语料只有 prompt，不依赖 label / verifier。

总规模：

```text
45,447 prompts
```

三类来源：

```text
Math:
  GSM8K train: 7,473 prompts
  MATH train: 7,500 prompts
  MATH train 与 MATH-500 eval 不重叠

Code:
  NVIDIA OpenCodeInstruct: 15,000 filtered prompts
  MBPP non-test: 474 unique tasks
  过滤掉 unit-test signal 非正的样本
  去重
  删除 function name 与 HumanEval / MBPP test 重叠的样本

Open-ended:
  15,000 instruction prompts
  来自 ShareGPT / Vicuna-style conversations / UltraChat-200K
```

所有 recovery 方法用同一个 prompt distribution，但消费方式不同：

```text
ShortOPD / OPD:
  只用 prompt 字段；
  response 由 student 在线采样；
  teacher 只负责对 student trajectory 打分。

SFT w/o KD:
  用原始 assistant responses 做 hard-label SFT。

SeqKD:
  用 pre-compression teacher 生成固定 responses；
  再对这些固定 responses 做 SFT。

KD:
  用固定 sequences；
  但监督从 hard labels 换成 teacher next-token distributions。
```

所以 ShortOPD 的关键是：

```text
训练状态来自 student 自己生成的 trajectory，
不是 gold answer，也不是 teacher pre-generated answer。
```

## 主实验结果

25% 剪枝后的 Qwen3-4B-Instruct：

| Model / Method | GSM8K | MATH | HumanEval | MBPP | Alpaca | QA | Sum | MT-B | Avg | T% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Dense teacher | 88.10 | 89.40 | 75.00 | 80.54 | 6.64 | 4.38 | 8.86 | 6.95 | 75.17 | 100.0 |
| Pruned untrained | 1.14 | 1.60 | 0.00 | 0.00 | 1.14 | 1.00 | 1.06 | 1.09 | 5.71 | 7.6 |
| SFT w/o KD | 25.93 | 9.40 | 26.83 | 29.18 | 1.91 | 1.19 | 3.12 | 1.59 | 21.19 | 28.2 |
| SeqKD | 32.83 | 12.20 | 29.88 | 41.25 | 2.52 | 1.38 | 5.26 | 2.10 | 28.60 | 38.0 |
| KD | 29.34 | 9.60 | 29.27 | 39.30 | 2.94 | 1.48 | 6.89 | 2.36 | 30.52 | 40.6 |
| ShortOPD | 62.70 | 42.00 | 43.90 | 49.81 | 4.68 | 2.04 | 7.94 | 4.28 | 48.46 | 64.5 |

结论：

```text
剪枝后 generation Avg 从 75.17 掉到 5.71；
ShortOPD 恢复到 48.46；
约等于 untrained pruned 的 9 倍；
比最强 off-policy baseline KD 高约 18 分。
```

## 效率结果

与固定 horizon 对比：

```text
Vanilla OPD fixed H=2048:
  337M rollout tokens
  11.1h
  Avg 49.6

ShortOPD init H=2048:
  250M rollout tokens
  8.5h
  Avg 48.5

Vanilla OPD fixed H=8192:
  869M rollout tokens
  35.9h
  Avg 50.2
```

所以：

```text
ShortOPD 与 fixed 8192 只差 1.7 Avg，
但 rollout tokens 少 71%，
wall-clock 从 35.9h 降到 8.5h。
```

## epoch scaling

ShortOPD 多训几轮仍然涨：

| Checkpoint | GSM8K | MATH | HumanEval | MBPP | Alpaca | QA | Sum | MT-B | Avg | T% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 epoch / step 710 | 62.70 | 42.00 | 43.90 | 49.81 | 4.68 | 2.04 | 7.94 | 4.28 | 48.46 | 64.5 |
| 2 epochs / step 1420 | 70.66 | 50.60 | 51.83 | 56.42 | 4.98 | 2.20 | 8.16 | 4.47 | 53.45 | 71.1 |
| 3 epochs / step 2127 | 72.71 | 55.20 | 54.27 | 57.20 | 5.14 | 2.21 | 8.15 | 4.89 | 55.41 | 73.7 |

说明：

```text
on-policy recovery 可以继续随 exposure 增长。
```

## sparse RLVR 对比

GSM8K-only 对比：

```text
Pruned:
  1.1

RLVR PPO:
  0.23

RLVR GRPO:
  1.6

ShortOPD GSM8K-only:
  37.8

ShortOPD multi-domain:
  62.7
```

原因：

```text
剪枝模型很少采到正确答案；
稀疏 reward 几乎没有正信号；
dense teacher distribution 每个 token 都能给修复方向。
```

## recovery corpus 消融

| Recovery corpus | GSM8K | MATH | HumanEval | MBPP | Alpaca | QA | Sum | MT-B | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full mixture | 62.70 | 42.00 | 43.90 | 49.81 | 4.68 | 2.04 | 7.94 | 4.28 | 48.46 |
| w/o code | 58.45 | 33.40 | 0.61 | 2.33 | 4.47 | 2.15 | 7.88 | 3.99 | 34.96 |
| w/o math | 8.04 | 6.00 | 28.66 | 43.19 | 3.96 | 1.73 | 7.19 | 2.94 | 30.51 |
| w/o general | 54.21 | 28.80 | 48.17 | 50.19 | 3.01 | 1.51 | 5.41 | 3.15 | 39.02 |

结论：

```text
ShortOPD 是 reward-free，但不是 data-free；
它只能修复 student on-policy distribution 访问到的状态。
所以 recovery prompt corpus 的能力覆盖很重要。
```

## 多选题 sanity check

多选题上反而 SFT / KD 更强：

| Method | ARC-C | HellaSwag | MMLU | WinoG. | Avg |
|---|---:|---:|---:|---:|---:|
| Dense teacher | 89.93 | 80.60 | 70.93 | 65.19 | 76.66 |
| Pruned untrained | 49.49 | 41.67 | 41.65 | 49.80 | 45.65 |
| SFT w/o KD | 89.33 | 76.89 | 69.09 | 64.09 | 74.85 |
| SeqKD | 83.36 | 72.65 | 64.19 | 63.06 | 70.82 |
| KD | 87.71 | 77.13 | 67.59 | 65.04 | 74.37 |
| ShortOPD 1 epoch | 79.52 | 68.89 | 60.70 | 61.33 | 67.61 |
| ShortOPD 3 epochs | 84.47 | 75.59 | 64.59 | 61.72 | 71.59 |
| SFT-init ShortOPD | 89.93 | 79.67 | 69.99 | 67.64 | 76.81 |

这说明：

```text
recognition recovery 和 generation recovery 不是一回事。
```

SFT / KD 更像恢复 fixed-sequence likelihood；
ShortOPD 更像恢复自由生成分布。

组合起来最好：

```text
SFT-init ShortOPD
```

## 和普通 OPD 的区别

普通 OPD：

```text
固定 rollout horizon
student rollout
teacher token-level distillation
```

ShortOPD：

```text
同样 student rollout + teacher token-level distillation，
但额外动态控制 rollout horizon：
早期重复多 -> 缩短
后期干净截断多 -> 增长
```

所以 ShortOPD 不是换 loss，而是：

```text
给 OPD 加了 short-to-long budget controller。
```

## 一句话结论

ShortOPD 的核心贡献是：

```text
把 on-policy self-distillation 用作结构化剪枝后的生成恢复信号，
并用 repetition-gated short-to-long horizon control，
把早期训练预算从低信息重复尾巴转移到更有用的前缀恢复上。
```

