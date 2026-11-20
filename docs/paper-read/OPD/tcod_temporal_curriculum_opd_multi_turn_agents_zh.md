# TCOD：Temporal Curriculum On-Policy Distillation 技术解读

本文档解读论文：

```text
TCOD: Exploring Temporal Curriculum in On-Policy Distillation for Multi-turn Autonomous Agents
https://arxiv.org/abs/2604.24005
```

## 一句话总结

TCOD 解决的是多轮 agent 场景下 vanilla OPD 的训练不稳定：

```text
单轮任务里，OPD 在 student rollout 上蒸馏 teacher token 分布通常可行。

多轮 agent 里，student 前面几轮一旦出错，
后续 observation / history 都会被带偏，
teacher 在这些偏离状态上给出的监督会越来越不可靠，
导致 trajectory-level KL 不断升高，success rate 甚至崩到 0。
```

论文把这个问题叫：

```text
Trajectory-Level KL Instability
```

TCOD 的核心做法很简单：

```text
不要一上来就让 student 学完整长轨迹。
先学短轨迹 / 靠近终点的轨迹，
再逐步扩展到完整 trajectory。
```

## Student / Teacher 模型速览

论文用了两类 teacher-student 设置。

### Qwen3 系列

```text
Teacher:
  Qwen3-30B-A3B-Instruct

Student:
  Qwen3-0.6B
  Qwen3-1.7B
  Qwen3-4B
```

用于 WebShop / ALFWorld / ScienceWorld 跨 benchmark 实验。

### Qwen2.5 系列

```text
Teacher:
  Qwen2.5-7B-RL
  也就是在 ALFWorld domain 上用 GRPO 训练过的 Qwen2.5-7B。

Student:
  Qwen2.5-0.5B
  Qwen2.5-1.5B
  Qwen2.5-3B
  Qwen2.5-7B
```

这里有一个重要结论：

```text
更大的 teacher 不一定更好；
target domain 上表现强、且和 student capacity 匹配的 teacher 更重要。
```

## 多轮 OPD 的基本形式

多轮 agent 每一轮：

```text
看到 observation o_t
根据 history h_t 生成 response/action a_t
环境返回下一轮 observation o_{t+1}
```

history 定义为：

```text
h_t = (o_0, a_0, o_1, a_1, ..., o_{t-1}, a_{t-1}, o_t)
```

vanilla OPD 目标：

```text
L_OPD(theta)
  = E_{tau ~ pi_theta} [
      sum_{t=0}^{T-1}
        KL( pi_phi(a_t | h_t) || pi_theta(a_t | h_t) )
    ]
```

其中：

```text
pi_phi:
  teacher policy。

pi_theta:
  student policy。

tau:
  student 自己 rollout 出来的完整多轮轨迹。
```

## 发现的问题：Trajectory-Level KL Instability

论文在 ALFWorld 上观察到两个现象。

### 现象 1：KL 上升和成功率崩溃同时出现

对小 student，例如：

```text
Qwen3-0.6B / 1.7B
Qwen2.5-0.5B / 1.5B
```

用 vanilla OPD 蒸馏强 teacher 时：

```text
trajectory-level KL 快速升高；
success rate 接近 0。
```

这和单轮 math / QA OPD 很不一样。单轮任务里 KL 通常会下降并收敛。

### 现象 2：即使能收敛，初始 KL 也极高

有些更大的 student 最终 KL 会下降，但初始 KL 非常高：

```text
initial KL ~ 1000
converged KL ~ 60
```

这说明训练早期非常不稳定。

### 根因：跨 turn 错误累积

论文的解释是：

```text
student 在前几轮生成的 action 会改变后续 environment state；
错误 action 会进入 history；
后续 teacher 看到的是 student 带偏后的状态；
teacher 对这些状态下的 token 概率越来越低；
KL 随 turn index 上升。
```

也就是说，多轮 agent 的问题不只是 response 长，而是：

```text
每一轮 action 都会改变后续状态分布。
```

这和 Long-CoT 不同：

```text
Long-CoT:
  只是同一个环境状态下 response 更长。

Multi-turn agent:
  每一轮都会改变环境 observation 和 history。
```

## TCOD 的核心思路

TCOD 把多轮 OPD 改成一个时间课程学习问题：

```text
先让 student 只接触较短、较容易的轨迹片段；
再逐步增加 student 需要负责的 interaction depth。
```

论文提出两个变体：

```text
TCOD-F2B:
  Forward-to-Backward，从前往后扩展。

TCOD-B2F:
  Backward-to-Forward，从后往前扩展。
```

课程步长由：

```text
k = k_start + floor(n / eta)
```

控制。

其中：

```text
n:
  当前 training step。

k_start:
  初始 student 负责的 turn 数，实验里通常设为 1。

eta:
  curriculum growth rate。
  eta 越大，扩展越慢。
```

## TCOD-F2B：从前往后学

F2B 是最直接的版本：

```text
训练早期:
  student 只 rollout 前 k 轮。

随着训练推进:
  k 逐渐增大，直到完整 horizon。
```

目标：

```text
L_TCOD_F2B(theta)
  = E_{tau ~ pi_theta} [
      sum_{t=0}^{k-1}
        KL( pi_phi(a_t | h_t) || pi_theta(a_t | h_t) )
    ]
```

直觉：

```text
先学前几轮相对干净的状态；
不要早期就把后面已经被错误污染的 turns 也纳入 loss。
```

优点：

```text
1. 实现简单。
2. 不需要预先收集 teacher successful trajectories。
3. 比 B2F 更省训练时间。
```

## TCOD-B2F：从后往前学

B2F 更像 reverse curriculum。

流程：

```text
1. 先用 teacher 采集成功轨迹 tau*。
2. 训练早期，teacher 执行前 L-k 步，把环境导航到接近成功的位置。
3. student 从这个中间状态接管，完成后 k 步。
4. 随着训练推进，k 增大，teacher prefix 逐渐变短。
5. 最后 k = L，teacher prefix 为 0，student 从初始状态完整执行。
```

目标：

```text
L_TCOD_B2F(theta)
  = E_{tau ~ (pi_phi, pi_theta)} [
      sum_{t=L-k}^{L}
        KL( pi_phi(a_t | h_t) || pi_theta(a_t | h_t) )
    ]
```

注意：

```text
teacher prefix 不反传梯度；
teacher 只是把 student 放到“快要成功”的状态附近。
```

论文的说法是：

```text
teacher serves only to place student on the doorstep of success。
```

## B2F 的 train-test mismatch 如何解决

B2F 训练早期 student 是从 teacher 导航后的中间状态开始，但测试时必须从头开始。

TCOD 的解决方式是：

```text
逐渐减少 teacher prefix。
```

也就是：

```text
早期:
  teacher 做 L-1 步，student 做最后 1 步。

中期:
  teacher 做 L-k 步，student 做后 k 步。

后期:
  teacher 做 0 步，student 从头做到尾。
```

所以到训练末期：

```text
训练分布和测试分布重新对齐。
```

## 异步训练细节

论文还实现了异步 rollout / training：

```text
actors:
  持续采样 trajectories。

learner:
  从 shared buffer 取样训练。
```

硬件分配：

```text
8 x NVIDIA H20

4 x H20:
  actors

2 x H20:
  learners

2 x H20:
  teachers
```

为降低 off-policy 风险，论文使用 staleness filter：

```text
n_current - n_old <= Delta_max
```

经验上：

```text
Delta_max = 2
```

## 实验设置

benchmark：

```text
ALFWorld:
  embodied navigation / object manipulation
  max turns = 30

WebShop:
  e-commerce navigation
  max turns = 15

ScienceWorld:
  scientific reasoning in text environment
  max turns = 30
```

ALFWorld 额外有：

```text
seen split
unseen split
hard split
```

hard split 是：

```text
teacher 在 train split 上 pass@10 都失败的 121 个任务。
```

这个 split 用来验证 student 是否能超过 teacher 的能力边界。

## 主要结果：ALFWorld

Teacher：

```text
Qwen2.5-7B-RL
```

### Qwen2.5-3B Student

```text
Valid Seen success:
  Vanilla OPD: 65.72
  TCOD-B2F:    77.86
  TCOD-F2B:    81.43

Valid Unseen success:
  Vanilla OPD: 60.45
  TCOD-B2F:    70.90
  TCOD-F2B:    79.19

Hard success:
  Vanilla OPD: 10.74
  TCOD-B2F:    13.22
  TCOD-F2B:     9.92
```

### Qwen2.5-7B Student

```text
Valid Seen success:
  Vanilla OPD: 75.37
  TCOD-B2F:    86.43
  TCOD-F2B:    82.14

Valid Unseen success:
  Vanilla OPD: 72.14
  TCOD-B2F:    77.61
  TCOD-F2B:    76.12

Hard success:
  Teacher:      6.61
  Vanilla OPD: 13.22
  TCOD-B2F:    20.66
  TCOD-F2B:    18.18
```

这里很有意思：

```text
在 hard split 上，teacher pass@10 都失败；
但 student 经过 TCOD 后超过 teacher。
```

论文认为这说明 TCOD 不是单纯复制 teacher，而是帮助 student 学到更 robust 的策略。

## 跨 benchmark 结果

Teacher：

```text
Qwen3-30B-A3B-Instruct
```

### Qwen3-1.7B Student

vanilla OPD 几乎崩掉：

```text
WebShop:      0.14
ALFWorld:     0.32
ScienceWorld: 0.05
Avg:          0.17
```

TCOD 则显著恢复：

```text
best Avg around 18.84
```

### Qwen3-4B Student

vanilla OPD 本身能跑：

```text
Avg: 27.64
```

TCOD 有小幅提升：

```text
best Avg around 29.54
```

结论：

```text
TCOD 对小模型救崩效果最明显；
对较大模型更多是稳定训练和小幅提效。
```

## curriculum growth rate 影响

论文测试：

```text
eta in {2, 4, 6}
```

结论：

```text
成功率变化小于 2%，不太敏感。
```

但：

```text
eta 越大，课程扩展越慢；
student 在当前 horizon 学得更久；
KL 更稳定。
```

实践建议：

```text
先用较小 eta 快速推进；
如果观察到 KL instability，再增大 eta。
```

## 训练效率

TCOD 比 vanilla OPD 更省时间：

```text
最多减少约 32% total training time。
```

原因：

```text
训练早期 student 只跑短 horizon；
trajectory 更短；
rollout collection 更快。
```

F2B 通常比 B2F 更快：

```text
F2B:
  直接限制最大 interaction steps = k。

B2F:
  虽然从中间状态开始，但 student 仍可能探索额外动作。
```

## 和 SDAR / SIRI / DOPD 的区别

```text
SDAR:
  用 skill-conditioned self-teacher 做 gated token distillation。

SIRI:
  自己挖 skill、验证 skill、内化 skill-guided action。

DOPD:
  解决 privileged information 引入的 privilege illusion，
  动态选择 teacher / student 监督源。

TCOD:
  不引入 skill 或 privileged input；
  只控制多轮 OPD 中 student 暴露的 trajectory depth。
```

所以 TCOD 的创新点更像：

```text
时间维度上的 curriculum OPD。
```

## 局限

论文提到几个限制：

```text
1. TCOD-B2F 需要预先收集 teacher successful trajectories。
2. 现在使用固定 schedule，未来可以根据 KL EMA 做 adaptive horizon。
3. 只验证了 text-based multi-turn benchmarks。
4. 还没扩展到 multimodal / physical embodied environments。
```

## 我的理解

TCOD 的核心非常朴素，但抓住了多轮 OPD 的关键：

```text
多轮 agent 的难点不是 token 多，
而是每个错误 action 都会改变后续状态分布。
```

vanilla OPD 把完整轨迹全拿来蒸馏，相当于：

```text
把后面已经被 student 自己带偏的状态也交给 teacher 打分。
```

这些监督信号很可能已经不在 teacher 的有效支持范围内，所以 KL 越训越大。

TCOD 的解决方案就是：

```text
先别学那么远。
先学短 horizon 或接近成功的后半段，
等稳定后再扩展到完整任务。
```

从工程上看，F2B 是更容易落地的版本；B2F 更像利用 teacher successful trajectory 做 reverse curriculum，效果可能更强，但需要额外收集 teacher 成功轨迹。

