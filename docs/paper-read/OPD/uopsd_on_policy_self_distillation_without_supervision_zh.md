# u-OPSD：On-Policy Self-Distillation without Any Supervision 论文解读

论文：

```text
On-Policy Self-Distillation without Any Supervision
```

链接：

```text
https://arxiv.org/abs/2608.06296
https://arxiv.org/pdf/2608.06296
```

作者单位：

```text
UC San Diego
Georgia Institute of Technology
University of Maryland, College Park
ByteDance
```

方法名：

```text
u-OPSD = unsupervised on-policy self-distillation
中文：无监督在线策略自蒸馏
```

## 一句话总结

这篇论文想回答一个问题：

```text
OPSD 的 teacher 一定要看 ground-truth solution 吗？
```

它的回答是：不一定。

u-OPSD 用模型自己的多次 on-policy rollouts 做 majority vote，得到一个 pseudo-answer；如果多数票足够一致，就选一条 agreeing rollout 作为 pseudo-solution，让 teacher branch 看到这个 pseudo-solution，再去监督那些和多数票不一致的 disagreeing rollouts。

所以它的核心是：

```text
用模型内部一致性构造 privileged context，
用模型内部不一致性定位需要纠正的 on-policy trajectory。
```

一句话公式：

```text
OPSD:    teacher sees x + GT solution
u-OPSD:  teacher sees x + self-consensus pseudo-solution
```

关键点：

```text
1. 不需要 ground truth solution；
2. 不需要 verifier reward；
3. 不需要外部 teacher model；
4. 仍然是 full-distribution token-level OPSD；
5. 训练只发生在模型“多数票有信心但仍会生成反对答案”的 frontier 上。
```

## 术语表

| 英文术语 | 建议中文 | 含义 |
|---|---|---|
| u-OPSD | 无监督在线策略自蒸馏 | 不用 GT / reward / 外部 teacher 的 OPSD |
| internal consistency | 内部一致性 | 模型多次 rollout 的答案是否一致 |
| majority vote | 多数投票 | 从多个 rollout 的 final answer 中选票数最多的答案 |
| pseudo-answer | 伪答案 | majority vote 得到的答案 `a_tilde` |
| pseudo-solution | 伪解答 | 与 pseudo-answer 一致的一条完整 rollout，作为 teacher privileged context |
| agreeing rollout | 一致轨迹 | final answer 等于 pseudo-answer 的 rollout |
| disagreeing rollout | 不一致轨迹 | final answer 不等于 pseudo-answer 的 rollout |
| self-consistency threshold `tau` | 自一致阈值 | 多数票比例达到该阈值才接受 pseudo-label |
| full-vocabulary distillation | 全词表蒸馏 | 对整个 vocab 的 teacher/student 分布算 divergence |
| sampled-token distillation | 采样 token 蒸馏 | 只在 student 实际采样 token 上算 logprob gap |
| teacher reference | 教师参考解 | teacher branch 条件中的 pseudo-solution |
| distillation target | 蒸馏目标轨迹 | 被 teacher 监督的 disagreeing rollout |
| non-thinking mode | 非思考模式 | Qwen3 不显式长 CoT 的模式 |
| thinking mode | 思考模式 | Qwen3 显式长 reasoning 的模式 |

## 背景：OPSD 的“self”还不彻底

普通 OPD：

```text
student 在自己的 on-policy rollouts 上学习；
teacher 是外部更强模型。
```

OPSD 进一步去掉外部 teacher：

```text
teacher 和 student 是同一个模型；
但 teacher branch 额外看到 ground-truth solution y*。
```

OPSD 目标：

```text
L_OPSD(theta)
= E_{(x,y*)} E_{y ~ pi_bar(.|x)} sum_t
  D_beta(
    pi_bar(. | x, y*, y_<t)
    ||
    pi_theta(. | x, y_<t)
  )
```

问题是：

```text
teacher 参数是 self 的，
但让 teacher 变强的信息 y* 仍来自外部监督。
```

u-OPSD 的动机就是去掉这个 `y*`：

```text
能不能由模型自己的多次生成构造 teacher privileged context？
```

## 方法流程

对一个 unlabeled problem `x`：

### 1. Sample：采样多个 rollouts

```text
y^(1), ..., y^(G) ~ pi_bar(. | x)
```

其中：

```text
pi_bar = stop-gradient copy of pi_theta
```

默认：

```text
G = 8
```

对每条 rollout 抽取最终答案：

```text
a^(g) = Ans(y^(g))
```

`Ans` 是答案抽取和规范化函数，例如从最后一个 `\boxed{...}` 里抽答案。如果没有可解析答案，则记为：

```text
empty / invalid
```

### 2. Vote：多数投票构造 pseudo-answer

在 valid answers 里做 plurality vote：

```text
a_tilde(x)
= argmax_a sum_g 1[a^(g) = a]
```

然后分组：

```text
Y_x^+ = { y^(g): a^(g) = a_tilde(x) }
Y_x^- = { y^(g): a^(g) != a_tilde(x), a^(g) != empty }
```

自一致分数：

```text
c(x) = (1/G) * sum_g 1[a^(g) = a_tilde(x)]
```

注意分母是所有 sampled rollouts `G`，不是 valid rollout 数。因此 truncation / invalid 会降低 confidence。

如果：

```text
c(x) < tau
```

则这个 prompt 不参与训练。

默认：

```text
tau = 0.5
```

即多数票必须过半。

### 3. Select：选 teacher reference 和 distillation targets

如果多数票可信，选择一个 agreeing rollout 作为 pseudo-solution：

```text
y^+ in Y_x^+
```

默认正文里写的是：

```text
shortest agreeing rollout
```

但消融表发现最佳组合其实是：

```text
longest agreeing rollout as teacher reference
longest disagreeing rollout as target
```

默认 distillation target 是 disagreeing rollouts 的一个子集：

```text
B_x^- subset Y_x^-
```

论文主要设置：

```text
k = 1
```

即每个 prompt 最多蒸馏 1 条 disagreeing rollout，默认可随机选，也有 longest-1 变体。

如果：

```text
Y_x^- = empty
```

也不训练，因为模型所有 valid rollouts 都同意，没有需要纠正的不一致轨迹。

## Teacher / Student 是什么

student branch：

```text
pi_theta(. | x, y^-_<t)
```

也就是只看 problem `x` 和 disagreeing rollout 的 prefix。

teacher branch：

```text
pi_bar(. | x, y^+, y^-_<t)
```

也就是同一个模型的 stop-gradient / teacher copy，但额外看到 pseudo-solution `y^+`。

关键区别：

```text
student sees: x + disagreeing prefix
teacher sees: x + pseudo-solution + disagreeing prefix
```

所以 teacher 不是更大模型，也不是 verifier，而是：

```text
同一个模型 + 自己多数票选出的解答轨迹作为 privileged context。
```

## Loss 公式

论文的 u-OPSD 目标：

```text
L_u-OPSD(theta)
= sum_{i in W_k} sum_{t=1}^{|y^(i)|}
  D_beta(
    pi_bar(. | x, y^+, y^(i)_<t)
    ||
    pi_theta(. | x, y^(i)_<t)
  )
```

更完整的期望形式：

```text
L_u-OPSD(theta)
= E_{x ~ U} E_{rollouts ~ pi_bar}
  1[Y_x^- != empty] * 1/|Y_x^-|
  sum_{y^- in Y_x^-} 1/|y^-| sum_n
  D_beta(
    pi_bar(. | x, y^+, y^-_<n)
    ||
    pi_theta(. | x, y^-_<n)
  )
```

默认 divergence：

```text
Forward KL: KL(pi_T || pi_S)
beta = 0
```

即：

```text
D_beta(P || Q) = KL(P || Q)
```

其中：

```text
P = teacher distribution = pi_bar(. | x, y^+, prefix)
Q = student distribution = pi_theta(. | x, prefix)
```

注意：

```text
这是 full-vocabulary token-level distillation，
不是 sampled-token OPD reward，
也不是 GRPO / policy gradient loss。
```

## 具体计算案例

假设一个数学题 `x`，模型采样 `G=8` 条 rollouts，抽出的 final answers 是：

```text
1: 42
2: 42
3: 42
4: 42
5: 42
6: 41
7: 39
8: invalid / truncated
```

多数票：

```text
a_tilde = 42
```

自一致分数：

```text
c(x) = 5 / 8 = 0.625
```

默认 `tau=0.5`，所以这个 prompt 被接受。

分组：

```text
Y^+ = 5 条答案为 42 的 rollouts
Y^- = 2 条答案为 41 / 39 的 rollouts
invalid 不参与 Y^+ 或 Y^-
```

选择 teacher reference：

```text
y^+ = Y^+ 中选一条 rollout
```

例如正文默认选 shortest agreeing rollout。

选择 distillation target：

```text
y^- = Y^- 中选一条 disagreeing rollout
```

对这条 disagreeing rollout 的每个 prefix 做蒸馏。

假设某个 prefix 下：

```text
x + y^-_<t = 已经走到错误推理分支前缀
```

student 预测下一个 token 分布：

```text
pi_S(. | x, y^-_<t)
```

teacher 多看 pseudo-solution：

```text
pi_T(. | x, y^+, y^-_<t)
```

如果 teacher 认为应该转向正确解法，它的 next-token distribution 会不同。训练就是最小化：

```text
KL(pi_T || pi_S)
```

从而让 student 在自己曾经走错的 prefix 上，学习“知道 42 这条 pseudo-solution 后的模型会怎么继续”。

## 为什么只蒸馏 disagreeing rollouts

因为 u-OPSD 的目标不是普通 self-training：

```text
不是让模型直接模仿多数票那条完整解答；
而是在模型自己会犯错的 on-policy prefixes 上提供 dense correction。
```

如果所有 valid rollouts 都同意：

```text
Y^- = empty
```

说明这个问题上模型已经自洽，没有明显的自我矛盾可纠正。

如果多数票不够强：

```text
c(x) < tau
```

说明模型自己也不确定，pseudo-solution 不可靠。

所以训练集中在：

```text
模型多数时候知道一个答案，
但仍有一部分概率会生成冲突答案。
```

论文称这是模型的 competence frontier。

## 和 TTRL / self-consistency RL 的区别

TTRL / self-consistency RL 通常把多数票转成 scalar reward：

```text
rollout answer == majority answer -> reward 1
else -> reward 0
```

u-OPSD 不这么做，而是：

```text
majority answer / agreeing rollout -> teacher context
teacher distribution -> dense token-level supervision
```

论文结果也说明：

```text
consensus 作为 teacher conditioning context，
比 consensus 作为 scalar reward 更有效。
```

## 实验设置

Models：

```text
Qwen3-4B
Qwen3-8B
Qwen3-4B-Instruct-2507
Qwen3-30B-A3B-Instruct-2507
```

模式：

```text
non-thinking mode
thinking mode
```

训练数据：

```text
OpenThoughts 30k subset
```

u-OPSD 只使用 problem statement，不使用 solution field。

评测 benchmark：

```text
AIME24, AIME25, HMMT25: avg@12
MATH500, AMC23: avg@4
```

训练超参：

| Hyperparameter | Value |
|---|---:|
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LR | 5e-6 |
| grad norm clipping | 0.1 |
| sampling temperature | 1.1 |
| top-p | 0.95 |
| top-k | 20 |
| train steps | 150 |
| checkpoint interval | 25 |
| rollouts per prompt `G` | 8 |
| self-consistency threshold `tau` | 0.5 |
| completion budget | 4096 |
| student/teacher scoring context max | 20000 tokens |
| objective | full-vocab forward KL with per-token pointwise clipping |
| teacher update default | fixed initial policy |

硬件和时间：

```text
8 GPUs, 80GB class, colocated vLLM generation
u-OPSD: 150 steps, G=8, 4096 budget, about 9 hours
supervised OPSD: about 3.5 hours
```

所以 u-OPSD 的代价更高：

```text
它用采样计算换掉人工标签 / verifier / external teacher。
```

## 主结果：non-thinking mode

| Model | Method | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | Base | 25.83 | 17.78 | 10.83 | 84.10 | 66.25 | 40.96 |
| Qwen3-4B | + SFT | 26.67 | 19.72 | 13.06 | 84.85 | 69.38 | 42.73 |
| Qwen3-4B | + GRPO | 25.00 | 22.50 | 15.00 | 86.20 | 80.62 | 45.86 |
| Qwen3-4B | + OPSD | 32.22 | 20.83 | 16.39 | 85.75 | 76.25 | 46.29 |
| Qwen3-4B | + TTRL | 25.00 | 20.83 | 11.94 | 83.60 | 67.50 | 41.77 |
| Qwen3-4B | + RENT | 22.22 | 20.28 | 11.39 | 84.00 | 74.38 | 42.45 |
| Qwen3-4B | + Intuitor | 23.89 | 20.00 | 11.67 | 83.70 | 70.62 | 41.98 |
| Qwen3-4B | + u-OPSD | 37.50 | 27.78 | 14.44 | 86.50 | 81.25 | 49.49 |
| Qwen3-8B | Base | 27.50 | 23.33 | 13.61 | 84.05 | 69.38 | 43.57 |
| Qwen3-8B | + SFT | 26.94 | 21.67 | 11.94 | 84.10 | 72.50 | 43.43 |
| Qwen3-8B | + GRPO | 30.56 | 21.94 | 13.06 | 87.85 | 73.75 | 45.43 |
| Qwen3-8B | + OPSD | 41.67 | 28.06 | 18.33 | 87.15 | 85.00 | 52.04 |
| Qwen3-8B | + TTRL | 27.22 | 21.11 | 13.06 | 84.45 | 71.88 | 43.54 |
| Qwen3-8B | + RENT | 28.33 | 21.67 | 10.83 | 84.00 | 70.00 | 42.97 |
| Qwen3-8B | + Intuitor | 26.11 | 22.50 | 11.67 | 84.20 | 73.12 | 43.52 |
| Qwen3-8B | + u-OPSD | 45.56 | 34.72 | 18.61 | 89.55 | 83.12 | 54.31 |

结论：

```text
Qwen3-4B: u-OPSD 比 Base +8.53， 比 supervised OPSD +3.20。
Qwen3-8B: u-OPSD 比 Base +10.74，比 supervised OPSD +2.27。
```

non-thinking mode 是这篇论文最强的结果。

## 主结果：thinking mode

| Model | Method | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | Base | 74.17 | 64.72 | 45.56 | 94.80 | 95.00 | 74.85 |
| Qwen3-4B | + SFT | 73.06 | 69.17 | 41.67 | 95.40 | 94.38 | 74.73 |
| Qwen3-4B | + GRPO | 73.89 | 69.44 | 43.61 | 95.45 | 99.38 | 76.35 |
| Qwen3-4B | + OPSD | 75.28 | 68.06 | 43.06 | 95.20 | 99.38 | 76.20 |
| Qwen3-4B | + TTRL | 72.78 | 68.33 | 45.56 | 95.90 | 96.25 | 75.76 |
| Qwen3-4B | + RENT | 74.72 | 65.83 | 43.06 | 95.25 | 99.38 | 75.65 |
| Qwen3-4B | + Intuitor | 76.39 | 68.33 | 42.78 | 95.35 | 97.50 | 76.07 |
| Qwen3-4B | + u-OPSD | 76.39 | 68.06 | 46.94 | 95.75 | 98.12 | 77.05 |
| Qwen3-8B | Base | 75.56 | 66.67 | 45.00 | 96.35 | 96.88 | 76.09 |
| Qwen3-8B | + SFT | 76.39 | 69.72 | 43.89 | 95.80 | 95.00 | 76.16 |
| Qwen3-8B | + GRPO | 76.94 | 69.17 | 47.78 | 95.70 | 95.00 | 76.92 |
| Qwen3-8B | + OPSD | 80.83 | 69.72 | 46.67 | 95.75 | 96.88 | 77.97 |
| Qwen3-8B | + TTRL | 77.22 | 68.61 | 46.94 | 95.75 | 96.25 | 76.95 |
| Qwen3-8B | + RENT | 77.50 | 70.28 | 45.83 | 95.95 | 96.25 | 77.16 |
| Qwen3-8B | + Intuitor | 76.94 | 70.28 | 44.17 | 96.20 | 95.62 | 76.64 |
| Qwen3-8B | + u-OPSD | 76.94 | 71.39 | 47.50 | 96.00 | 98.12 | 77.99 |

结论：

```text
Qwen3-4B thinking: u-OPSD Avg 77.05，高于 OPSD 76.20 / GRPO 76.35。
Qwen3-8B thinking: u-OPSD Avg 77.99，基本持平 OPSD 77.97，高于 GRPO 76.92。
```

thinking mode 增益明显小很多，因为 base 已经很强，而且 rollout 更长，同样 150 step budget 下有效训练样本更少。

## Instruction-tuned models：pass@1

| Model | Method | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B-Instruct-2507 | Base | 80.00±0.00 | 63.33±2.72 | 43.33±2.72 | 96.33±0.25 | 95.83±1.18 | 75.77±0.85 |
| Qwen3-30B-A3B-Instruct-2507 | OPSD | 78.89±3.14 | 61.11±3.14 | 47.78±1.57 | 97.20±0.43 | 96.67±2.36 | 76.33±1.58 |
| Qwen3-30B-A3B-Instruct-2507 | u-OPSD | 75.56±5.67 | 65.56±4.16 | 50.00±0.00 | 97.00±0.43 | 99.17±1.18 | 77.46±2.09 |
| Qwen3-4B-Instruct-2507 | Base | 66.67±7.20 | 53.33±5.44 | 27.78±4.16 | 93.87±0.34 | 93.33±1.18 | 67.00±2.25 |
| Qwen3-4B-Instruct-2507 | OPSD | 62.22±5.09 | 52.22±5.09 | 31.11±5.09 | 94.93±0.64 | 95.00±0.00 | 67.10±1.13 |
| Qwen3-4B-Instruct-2507 | u-OPSD | 68.89±5.67 | 57.78±1.57 | 28.89±3.14 | 94.20±0.86 | 94.17±1.18 | 68.78±1.70 |

## 消融实验

### 1. pseudo-label quality

在 64 个 training prompts 上：

```text
96.3% rollouts 可以解析出 boxed answer；
94.0% prompts 通过 tau=0.5 自一致阈值；
86.7% pseudo-label 与数据源 gold answer 一致；
少于 10% valid rollouts 与多数票不一致。
```

这说明 majority vote 并不完美，但质量已经足够高。

### 2. self-consistency threshold `tau`

Qwen3-8B non-thinking, longest-1, G=8：

| tau | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Avg |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 53.33 / 57.50 | 41.39 / 46.67 | 22.50 / 27.22 | 90.70 / 92.15 | 88.12 / 89.38 | 59.21 / 61.83 |
| 0.3 | 50.83 / 53.33 | 34.44 / 41.67 | 20.28 / 23.06 | 89.85 / 90.65 | 85.00 / 88.12 | 56.08 / 58.59 |
| 0.5 default | 43.06 / 53.89 | 35.83 / 37.50 | 20.28 / 20.28 | 88.00 / 89.90 | 82.50 / 85.62 | 53.93 / 57.10 |
| 0.7 | 32.78 / 37.50 | 23.61 / 26.67 | 13.06 / 14.44 | 85.95 / 87.70 | 74.38 / 76.25 | 45.96 / 48.18 |
| 0.9 | 32.22 / 32.22 | 20.83 / 22.50 | 13.61 / 13.61 | 84.70 / 84.85 | 70.62 / 72.50 | 44.40 / 44.40 |

格式：

```text
step150 / best checkpoint
```

结论：

```text
tau 越高，效果越差；
严格过滤 pseudo-label 去掉了太多有用训练信号。
```

虽然默认用 `tau=0.5` 做公平主实验，但消融显示 `tau=0` 的 best checkpoint 最高。

### 3. rollouts per prompt `G`

| G | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Avg |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 47.22 / 50.28 | 35.83 / 37.78 | 18.06 / 21.11 | 88.15 / 90.20 | 84.38 / 87.50 | 54.73 / 56.99 |
| 8 default | 43.06 / 53.89 | 35.83 / 37.50 | 20.28 / 20.28 | 88.00 / 89.90 | 82.50 / 85.62 | 53.93 / 57.10 |
| 12 | 53.61 / 57.22 | 40.83 / 43.33 | 20.56 / 26.39 | 91.00 / 92.00 | 86.25 / 90.00 | 58.45 / 61.79 |
| 16 | 51.67 / 55.28 | 41.67 / 44.17 | 22.22 / 22.22 | 89.90 / 89.90 | 87.50 / 87.50 | 58.59 / 59.37 |

结论：

```text
G 是主要成本项，sampling cost 线性随 G 增加；
G=12 的 best checkpoint 最好；
G=16 没继续提升；
主实验用 G=8 是为了和 OPSD generation budget 匹配。
```

### 4. teacher update

| Teacher | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| fixed default | 43.06 / 53.89 | 35.83 / 37.50 | 20.28 / 20.28 | 88.00 / 89.90 | 82.50 / 85.62 | 53.93 / 57.10 |
| EMA 0.999 | 43.61 / 52.78 | 36.67 / 41.39 | 18.33 / 23.33 | 88.40 / 90.50 | 87.50 / 87.50 | 54.90 / 58.82 |
| EMA 0.99 | 52.78 / 54.17 | 39.72 / 43.89 | 23.33 / 23.33 | 89.85 / 91.00 | 83.75 / 86.88 | 57.89 / 58.88 |
| EMA 0.995 | 54.17 / 55.00 | 40.28 / 43.33 | 21.94 / 22.50 | 90.10 / 90.75 | 83.75 / 89.38 | 58.05 / 59.47 |

结论：

```text
主实验 default 固定 teacher 是为了和 OPSD 设置一致；
EMA teacher 其实更好，EMA 0.995 best checkpoint 比 fixed 高 2.37 points。
```

### 5. teacher reference / distillation target 选择

| Teacher ref \ distill target | longest | random | shortest |
|---|---:|---:|---:|
| label-only | 43.40 (125) | 43.00 (100) | 41.55 (25) |
| shortest default | 57.10 (125) | 54.37 (75) | 55.23 (150) |
| random | 57.96 (75) | 56.93 (75) | 55.77 (75) |
| longest | 59.00 (75) | 57.90 (100) | 55.01 (125) |

结论：

```text
只给 boxed pseudo-label 明显不够；
teacher 需要看到完整 reasoning trace。
```

最佳组合是：

```text
teacher reference = longest agreeing rollout
distill target = longest disagreeing rollout
Avg = 59.00
```

这和正文一开始“shortest pseudo-solution”直觉不同。实际消融说明更长的完整推理轨迹给 teacher 更多条件信息。

### 6. disagreeing-rollout selection

| Variant | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 |
|---|---:|---:|---:|---:|---:|
| OPSD supervised | 27.50 / 27.50 | 20.00 / 23.61 | 12.50 / 13.33 | 83.80 / 84.65 | 71.88 / 72.50 |
| disagree-1 default | 33.89 / 35.28 | 27.78 / 27.78 | 14.72 / 16.11 | 87.60 / 87.60 | 79.38 / 79.38 |
| disagree-2 | 29.72 / 31.11 | 23.33 / 27.50 | 15.28 / 16.67 | 86.20 / 86.30 | 75.62 / 80.00 |
| disagree-3 | 27.78 / 34.72 | 25.83 / 31.11 | 13.33 / 18.33 | 85.45 / 87.10 | 72.50 / 76.25 |
| disagree-all | 29.44 / 30.83 | 22.50 / 25.83 | 10.00 / 13.33 | 85.25 / 86.00 | 73.75 / 74.38 |
| longest-1 | 33.06 / 34.72 | 23.33 / 26.67 | 14.72 / 18.33 | 85.50 / 86.80 | 73.75 / 78.12 |

结论：

```text
capped variants 都强于 supervised OPSD；
disagree-1 最稳定；
disagree-all 最差，说明把所有不一致轨迹都蒸馏会引入太多噪声。
```

### 7. full vocab vs sampled token vs top-k

| Variant | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| Student token | 29.44 | 19.72 | 12.78 | 84.70 | 70.62 | 43.45 |
| top-20 | 50.00 | 37.78 | 19.44 | 90.20 | 88.12 | 56.94 |
| top-50 | 48.06 | 37.50 | 24.17 | 89.45 | 85.62 | 55.85 |
| top-100 | 53.89 | 43.33 | 20.56 | 91.65 | 85.62 | 59.01 |
| top-200 | 51.67 | 40.00 | 22.50 | 90.15 | 88.75 | 58.11 |
| Full-vocabulary | 53.89 | 37.50 | 20.28 | 89.90 | 85.62 | 57.10 |

结论：

```text
只在 student sampled token 上做 distillation 很弱；
full-distribution / top-k distribution 明显更强；
top-100 甚至略高于 full-vocab，是一个计算优化方向。
```

### 8. divergence family

| Objective | AIME24 | AIME25 | HMMT25 | MATH500 | AMC23 | Avg |
|---|---:|---:|---:|---:|---:|---:|
| Base | 27.50 | 23.33 | 13.61 | 84.05 | 69.38 | 43.57 |
| Forward KL KL(pi_T || pi_S), beta=0 default | 53.89 | 37.50 | 20.00 | 89.75 | 84.38 | 57.10 |
| Reverse KL KL(pi_S || pi_T), beta=1 | training diverges | training diverges | training diverges | training diverges | training diverges | training diverges |
| JSD beta=0.5 | 29.44 | 21.94 | 12.22 | 83.70 | 69.38 | 43.34 |

结论：

```text
u-OPSD 强烈偏好 FKL；
JSD 基本回到 base；
RKL 训练发散。
```

论文对 RKL 发散的观察：

```text
step 25: 平均 generation 2.7k chars
step 50: 12k chars
step 75: 76k chars
step 125+: 达到 99k token ceiling
parsable boxed answer 比例从 99% 掉到 33%
```

也就是说 RKL 主要导致 loss of termination / 长度失控，而不是生成更连贯的 reasoning。

## 为什么 FKL 更适合这篇

默认是：

```text
KL(pi_T || pi_S)
```

这让 student 覆盖 teacher distribution 支持的区域。对于 u-OPSD，teacher 看到的是 pseudo-solution，可能仍有噪声，但它提供了“如何从错误 prefix 转回多数票解法”的完整 next-token 分布。

如果用 RKL：

```text
KL(pi_S || pi_T)
```

student 更倾向 mode-seeking，且当 teacher 分布在某些 prefix 下很尖锐时，student 容易通过重复某些高概率片段来降低局部 loss，导致 termination 崩溃。

这和很多 OPD 里 RKL 可用的情况不同，因为这里 teacher context 是模型自己生成的 pseudo-solution，不是外部稳定强 teacher 或可靠 GT conditioning。

## 和 supervised OPSD 的区别

| 维度 | OPSD | u-OPSD |
|---|---|---|
| 数据 | `(x, y*)` labeled solution | unlabeled `x` |
| teacher context | ground-truth solution `y*` | agreeing rollout `y+` |
| student trajectory | model rollout `y` | disagreeing rollout `y-` |
| 监督来源 | 外部 GT | 内部 majority vote |
| 是否需要 verifier | 不需要，但需要 GT solution | 不需要 |
| 是否需要外部 teacher | 不需要 | 不需要 |
| loss | full-vocab FKL | full-vocab FKL |
| 核心风险 | GT leakage / privileged context bias | pseudo-label 错误、采样成本高 |

## 和 GRPO / TTRL 的区别

| 维度 | GRPO | TTRL / self-consistency RL | u-OPSD |
|---|---|---|---|
| 信号 | GT verifier reward | majority vote scalar reward | majority vote conditioned teacher distribution |
| 粒度 | sequence-level | sequence-level | token-level full distribution |
| 是否无监督 | 否 | 是 | 是 |
| 是否 on-policy | 是 | 是 | 是 |
| 主要优势 | reward 方向可靠 | 不需要 GT | 不需要 GT 且有 dense supervision |

## 局限性

论文明确指出：

```text
1. 只覆盖 Qwen3 family；
2. 只覆盖 competition math；
3. 需要答案可抽取、可 canonicalize；
4. open-ended generation 需要 soft consensus 替代 exact-match vote；
5. 增益取决于 base 模型能力：模型必须已经能多数时候投出好答案；
6. 训练成本高于 supervised OPSD。
```

我补充一个关键风险：

```text
majority vote 可能稳定地投错。
```

论文测到 pseudo-label 与 gold 一致率为 86.7%，说明约 13.3% pseudo-label 是错的。u-OPSD 的成功依赖一个经验事实：

```text
多数投票的噪声 < 它提供的内部 correction 信号收益。
```

## 一句话结论

u-OPSD 的核心贡献是把 OPSD 里的 external privileged context 换成 internal self-consensus：

```text
用 agreeing rollout 作为 teacher reference，
用 disagreeing rollout 作为 student correction target，
在模型自己的错误 prefixes 上做 full-distribution FKL self-distillation。
```

它证明了一个重要点：

```text
在数学 reasoning 的 non-thinking 模式里，
OPSD 的瓶颈未必是有没有 gold solution，
而是能不能把模型自己的不一致性挖出来并转成 dense token supervision。
```
