# ReflectRL / GLOW 论文解读

论文正式标题：

```text
Learning from Mistakes: Negative Reasoning Samples Enhance Out-of-Domain Generalization
```

方法名：

```text
GLOW
```

你提到的 `ReflectRL` 大概率是指这篇围绕“reflective-to-direct reasoning”做 on-policy self-distillation 的工作。这里按论文原文来解读。

链接：

```text
https://aclanthology.org/2026.acl-long.1370/
```

## 一句话总结

这篇论文想做的是：

```text
不用 ground-truth solution，也不用外部 teacher，
只用模型自己的多次 rollout，
把“多数票正确的轨迹”当作 teacher context，
把“多数票不一致的轨迹”当作 student 纠错对象，
从而提升数学推理的 OOD 泛化。
```

它和传统 RLVR / OPSD 的区别是：

```text
不靠 gold answer 监督；
不靠 scalar reward；
而是把 self-consistency vote 变成 dense token-level distillation。
```

## 核心问题

论文观察到：

```text
RLVR / GRPO 这类方法依赖 gold answer 来给 rollout 打 0/1 奖励；
OPSD 依赖 gold solution 来构造 privileged teacher context。
```

但在很多场景里：

```text
标签贵、标签少、或者根本没有。
```

所以作者问的是：

```text
能不能只用模型自己的多次采样结果，构造一个“自举老师”？
```

答案是可以，做法就是：

```text
先 majority vote 找 pseudo-answer，
再把 agreeing rollout 当 teacher reference，
把 disagreeing rollouts 当 student distillation targets。
```

## 术语表

| 英文术语 | 建议中文 | 含义 |
|---|---|---|
| golden negative trajectories | 黄金负轨迹 / 高价值负轨迹 | 多数票不一致、但仍携带可学习信息的错误推理轨迹 |
| reflective-to-direct reasoning | 反思式到直接式推理 | teacher 看到更丰富的反思上下文，student 只看问题和自身前缀 |
| pseudo-answer | 伪答案 | 多数票得到的 final answer |
| agreeing rollout | 一致轨迹 | final answer 等于 pseudo-answer 的轨迹 |
| disagreeing rollout | 不一致轨迹 | final answer 与 pseudo-answer 不同的轨迹 |
| self-consistency score | 自一致分数 | 多数票占全部采样的比例 |
| competence frontier | 能力前沿 | 模型能 vote 出较可靠答案，但仍会产生冲突轨迹的区域 |
| forward KL | 前向 KL | `KL(teacher || student)` |
| generalized JSD | 广义 Jensen-Shannon 散度 | 论文里 `D_beta` 的通用形式，`beta -> 0` 时变成 forward KL |

## 方法框架

### 1. Sample

对一个无标签题目 `x`，采样 `G` 条 rollout：

```text
y^(1), ..., y^(G) ~ pi_bar(. | x)
```

其中：

```text
pi_bar = stop-gradient copy of pi_theta
```

最终答案从 rollout 里解析：

```text
a^(g) = Ans(y^(g))
```

如果没有可解析答案，就是 `invalid`。

### 2. Vote

在有效答案里做 plurality vote：

```text
a_tilde(x) = argmax_a sum_g 1[a^(g)=a]
```

然后分组：

```text
Y^+ = { y^(g) : a^(g) = a_tilde(x) }
Y^- = { y^(g) : a^(g) != a_tilde(x) }
```

自一致分数：

```text
c(x) = (1/G) * sum_g 1[a^(g)=a_tilde(x)]
```

默认阈值：

```text
tau = 0.5
```

即多数票必须过半，这样这个 prompt 才参与训练。

### 3. Distill

从 `Y^+` 里选一条 agreeing rollout 作为 teacher reference：

```text
y^+ in Y^+
```

默认是 **shortest agreeing rollout**，但消融里发现 **longest agreeing rollout** 更好。

从 `Y^-` 里选一条或几条 disagreeing rollout 作为 student target：

```text
y^- in Y^-
```

然后对每个 student prefix 计算 teacher / student 的 next-token divergence：

```text
L_GLOW(theta)
= sum_{i in B^-} sum_t
  D_beta(
    pi_bar(. | x, y^+, y^-_<t)
    ||
    pi_theta(. | x, y^-_<t)
  )
```

这里的意思是：

```text
student 只看题目 x 和错误轨迹前缀 y^-_<t
teacher 额外看到 pseudo-solution y^+
```

所以 teacher 是“反思式”的，student 是“直接式”的。

## 具体例子

题目：

```text
Solve 3x + 2 = 11.
```

采样 8 条 rollout，答案如下：

```text
3, 3, 3, 3, 3, 4, 13/3, invalid
```

多数票是 `3`，于是：

```text
Y^+ = 前 5 条
Y^- = 第 6、7 条
```

选一条 `Y^+` 作为 teacher reference：

```text
y^+ = "3x+2=11, subtract 2 gives 3x=9, divide by 3, x=3"
```

选一条 `Y^-` 作为 student target：

```text
y^- = "3x+2=11, 3x=13, x=13/3"
```

当 student 已走到前缀：

```text
x + "3x+2=11, 3x="
```

teacher 看到的是：

```text
x + y^+ + "3x+2=11, 3x="
```

此时 teacher 会更倾向输出：

```text
9
```

student 可能还在纠结：

```text
13
```

最小化 `KL(teacher || student)` 就是在这个错误前缀上纠正 student。

## 为什么叫 golden negative trajectories

因为这里的 negative 不是“完全没用的坏样本”，而是：

```text
final answer 错了，
但轨迹前缀里仍包含大量可学习的中间 reasoning。
```

论文把这种“多数票不一致的轨迹”当作最值得纠正的对象。

它不是普通 self-training 的“只模仿好样本”，而是：

```text
在模型自己最容易走歪的前缀上，
用老师分布把错误轨迹拉回多数票解法。
```

## 为什么是 token-level distillation，而不是 scalar reward

如果把 majority vote 只变成一个 0/1 reward：

```text
答案对 -> 1
答案错 -> 0
```

那信息太稀疏了。

GLOW 的做法是：

```text
把“多数票正确”变成 teacher context，
把“多数票不一致”变成 token-level supervision 的载体。
```

所以它比 self-consistency RL 更密。

## 训练设置

| 项目 | 设置 |
|---|---|
| Models | Qwen3-4B, Qwen3-8B, Qwen3-30B-A3B-Instruct-2507, Qwen3-4B-Instruct-2507 |
| Data | OpenThoughts 30k subset |
| Train objective | forward KL / generalized JSD with `beta=0` |
| Teacher update | fixed initial policy（默认） |
| LoRA | rank 64, alpha 128 |
| LR | `5e-6` |
| Grad clip | 0.1 |
| Sampling temp | 1.1 |
| top-p / top-k | 0.95 / 20 |
| Steps | 150 |
| Checkpoint interval | 25 |
| `G` | 8 |
| `tau` | 0.5 |
| Max completion length | 4096 |

## 评测 benchmark

论文只测数学推理：

```text
AIME24
AIME25
HMMT25
MATH500
AMC23
```

其中：

```text
AIME24 / AIME25 / HMMT25: avg@12
MATH500 / AMC23: avg@4
```

没有泛化到通用对话或 agent benchmark，范围比较窄，但很干净。

## 主结果

### Non-thinking mode

| Model | Base | Method | Avg |
|---|---:|---:|---:|
| Qwen3-4B | 40.96 | 49.49 | +8.53 |
| Qwen3-8B | 43.57 | 54.31 | +10.74 |

并且它超过了同设定下的：

```text
SFT
GRPO
OPSD
TTRL
RENT
Intuitor
```

### Thinking mode

| Model | Base | Method | Avg |
|---|---:|---:|---:|
| Qwen3-4B | 74.85 | 77.05 | +2.20 |
| Qwen3-8B | 76.09 | 77.99 | +1.90 |

思考模式下提升变小，但仍然稳定。

### Instruct models

| Model | Base | Method | Avg |
|---|---:|---:|---:|
| Qwen3-30B-A3B-Instruct-2507 | 75.77 | 77.46 |
| Qwen3-4B-Instruct-2507 | 67.00 | 68.78 |

## 消融实验

### 1. Self-consistency threshold `tau`

结论很直接：

```text
tau 越高，效果越差。
```

在 Qwen3-8B non-thinking、longest-1、G=8 的设置里：

| tau | Avg(best) |
|---|---:|
| 0.0 | 61.83 |
| 0.3 | 58.59 |
| 0.5 | 57.10 |
| 0.7 | 48.18 |
| 0.9 | 44.40 |

作者的解释是：

```text
过滤掉 low-consensus prompt 的收益，不如它损失的训练信号大。
```

### 2. Rollouts per prompt `G`

| G | Avg(best) |
|---|---:|
| 4 | 56.99 |
| 8 | 57.10 |
| 12 | 61.79 |
| 16 | 59.37 |

结论：

```text
G=12 最好，但 G=8 是为了和 OPSD 的预算对齐。
```

### 3. Teacher update

固定 teacher 是默认设置：

```text
fixed initial policy
```

EMA 也测了，最好的是 `EMA 0.995`，比 fixed 更强。

这说明：

```text
teacher 跟着 student 轻微移动，有时能提升自举效果。
```

### 4. Teacher reference / distillation target

论文比较了：

```text
teacher reference: shortest / random / longest
distillation target: longest / random / shortest
```

结论：

```text
最长的 agreeing rollout 当 teacher reference 最好；
最长的 disagreeing rollout 当 student target 也最好。
```

label-only 版本最差，说明：

```text
只用最终答案不够，必须保留完整 reasoning trace。
```

### 5. Disagreeing rollout cap

`disagree-1` 最稳，`disagree-all` 最差。

说明：

```text
不是把所有负轨迹都喂进去就更好；
要控制噪声和计算量。
```

### 6. Divergence family

默认是：

```text
Forward KL = KL(teacher || student)
```

消融结果：

| Objective | Avg(best) |
|---|---:|
| Forward KL | 57.10 |
| JSD | 43.34 |
| Reverse KL | training diverges |

所以这篇论文很明确：

```text
用 FKL，不用 RKL。
```

原因也是典型的 mode-covering vs mode-seeking：

```text
多个正确 reasoning mode 需要一起保留；
RKL 太容易塌到某个 mode，甚至发散。
```

### 7. Learning rate

论文还做了 LR ablation，结论是匹配的 150-step linear decay 更稳。

## 这篇论文到底在讲什么

如果压成一句工程视角的话：

```text
它把 self-consistency 从“测试时投票”改造成“训练时的 dense teacher context”。
```

更具体一点：

```text
1. 模型多次 rollout，自己投票。
2. 投票赢的轨迹变成 teacher 参考。
3. 投票输的轨迹变成 student 的纠错路径。
4. teacher 在这些错误前缀上提供 next-token 分布。
5. 用 FKL 逐 token 蒸馏。
```

这比只把票数当 reward 更密，也比只模仿 gold solution 更无监督。

## 局限

论文自己也说了几个限制：

```text
1. 需要可解析 final answer，open-ended generation 不适合直接用。
2. 主要验证在数学推理，域外泛化没测。
3. 效果依赖 base model 自己已经有一定 competence。
4. 多采样会带来明显 generation cost。
```

作者还指出：

```text
non-thinking 模式收益最大；
thinking 模式因为 base 已经很强，headroom 变小，所以增益缩小。
```

## 我的理解

这篇工作的核心不是“负样本”本身，而是：

```text
把“负样本里仍然有价值的部分”显式拿来做 teacher conditioning。
```

它和普通 self-training 的差异在于：

```text
不是只学多数票答案，
而是在错误轨迹上逐 token 纠偏。
```

所以它更像一种：

```text
反思式 teacher + 直接式 student 的 on-policy distillation。
```

