# DASD: Direction-Adaptive Self-Distillation 论文解读

论文：

```text
Tailoring Teaching to Aptitude: Direction-Adaptive Self-Distillation for LLM Reasoning
```

来源：

```text
arXiv:2605.22263
```

## 一句话总结

DASD 认为 OPSD 的失败不是“teacher 太强”本身，而是：

```text
所有 token 被施加了同一个方向的 teacher pressure。
```

这会带来两个冲突：

```text
高 entropy token:
  需要探索和分叉，
  但 uniform teacher pressure 会把探索压死。

低 entropy token:
  需要稳定执行，
  但如果故意远离 teacher，又会伤 step accuracy。
```

所以 DASD 把 self-distillation 改写成：

```text
高 entropy tokens -> pushed away from teacher
低 entropy tokens -> pulled toward teacher
```

也就是方向自适应，而不是单向 imitation。

## 背景：为什么 uniform OPSD 会坏

论文的核心诊断是：

```text
OPSD 压缩了预测不确定性。
```

而复杂 reasoning 里，不确定性并不总是坏事，它支持：

```text
exploration
hypothesis revision
functional self-correction
```

如果 teacher pressure 对所有 token 一视同仁，那么：

```text
1. 高熵分叉会被压掉；
2. 低熵执行位如果不被拉稳，又会丢步骤准确率。
```

## DASD 怎么做

DASD 用 `token-level entropy` 作为 routing signal。

论文的方向图是：

```text
低 entropy -> toward teacher
高 entropy -> away from teacher
```

它还加了两个关键件：

```text
1. gap-reliability gate
2. smooth signed routing
```

直观上：

```text
不是简单按 entropy 硬切，
而是让方向变化更平滑，并且只在 teacher-student gap 可信时才放大。
```

### teacher-student gap 是怎么计算的

这里的 gap 不是泛泛的“差距”，而是 **student 和 teacher 对同一个 token 的分布差异**。

论文里可写成：

```text
gap_t = KL(p_t || q_t)
```

其中：

```text
p_t = π_θ(. | x, y_<t)      # student distribution
q_t = sg[π_T(. | x, y_<t)]  # teacher distribution
```

如果把它展开到 student 采样 token `y_t` 上，也可以理解成：

```text
student 选中的 token 在 teacher 和 student 下的支持程度差了多少。
```

更直白一点，teacher-student gap 反映的是：

```text
teacher 认为当前 token 分布和 student 现在的分布差多远。
```

这和 entropy 不一样：

```text
entropy 看的是 student 自己有多不确定；
gap 看的是 student 和 teacher 在这个位置分歧有多大。
```

## 什么叫 gap 可信

DASD 里的 gap-reliability gate 不是说“gap 绝对正确”，而是说：

```text
这个位置上的 teacher-student 分歧，值不值得拿来决定方向。
```

它大致包含两层意思：

### 1. 分歧要和方向信号一致

如果 student entropy 已经说明这是一个高熵分叉点，而 teacher-student gap 也确实在这个位置很大，那么这个 gap 更可信，说明 teacher 在这里确实提供了有意义的偏好差异。

如果 entropy 很低，但 gap 却异常大，可能只是 teacher 和 student 在一个已经收敛的位置上出现了偶然尖峰，这时不应该过度放大。

### 2. 分歧最好来自局部稳定区，而不是噪声 spike

论文强调的是 smooth signed routing，而不是单点硬切。也就是说，gate 更偏好：

```text
在局部窗口里持续存在的 gap
```

而不是：

```text
只在一个 token 上突然冒出来的尖峰 gap
```

所以“gap 可信”更准确的中文可以叫：

```text
教师-学生分歧可靠
```

意思是这个分歧足够稳定、足够和当前 entropy 状态一致，可以用来决定“靠近 teacher”还是“远离 teacher”的方向。

## 数学上 DASD 怎么表达这个方向

### 理论依据是什么

先说结论：**这不是一个严格定理，而是“诊断实验 + 符号化设计 + 消融验证”的经验性依据**。

你的疑问是关键：

```text
高熵 token 正是 student 不确定的位置，
为什么不是更靠近 teacher，而是远离 teacher？
```

DASD 的回答不是“高熵 token 不重要”，而是：

```text
高熵 token 往往是 reasoning 的分叉点；
在这些位置强行 imitation teacher，会把 student 的多个候选 continuation 压成 teacher 的单一路径，
造成 premature collapse，也就是过早塌缩。
```

这里的 teacher 通常不是“绝对真理”，而是一个 solution-conditioned / self-teacher target。它能提供稳定方向，但也会带来 conformity pressure，也就是“向 teacher 分布靠拢的压力”。对所有 token 都加同向 teacher pressure，会把不同功能的 token 混在一起处理。

## 论文的三条诊断依据

### 1. Uniform OPSD 会压缩不确定性

OPSD 类方法把所有 token 都往 teacher 拉：

```text
all tokens -> toward teacher
```

论文观察到这会降低 token entropy、response length 和 exploration density。问题在于，复杂 reasoning 里有些高熵 token 是有功能的：

```text
1. 提出不同假设；
2. 尝试不同解法；
3. 为后续 self-correction 留空间；
4. 在不确定步骤上保留多个 plausible continuation。
```

如果这些位置也被强行拉向 teacher，模型会更快变得自信，但不一定更会推理。

### 2. Uniform repulsion 也不行

如果所有 token 都远离 teacher：

```text
all tokens -> away from teacher
```

探索会增加，但 step accuracy 会受损。原因是很多低熵 token 其实是已经学会的执行位，比如：

```text
代入公式、展开等式、写出中间结论、执行已确定操作。
```

这些位置不需要再被推开，反而应该靠 teacher 稳住。

### 3. Student entropy 是最好的 routing signal

论文比较了：

```text
position proxy
token frequency
gradient norm
attention entropy
student entropy H_t
```

结果 student entropy 最好。它的含义是：当前 student 自己在这个位置有多不确定，比“位置靠前/靠后”“token 频率高/低”更直接。

## 数学上 DASD 怎么表达这个方向

对 student 当前 token distribution：

```text
p_t = pi_theta(. | x, y_<t)
```

token entropy 是：

```text
H_t = - sum_v p_t(v) log p_t(v)
```

teacher distribution 记为：

```text
q_t = sg[pi_T(. | x, y_<t)]
```

普通 OPSD 的方向可以理解成：

```text
min KL(p_t || q_t)
```

也就是所有 token 都被拉向 teacher。

DASD 改成 signed objective：

```text
L_DASD,t = omega_t * KL(p_t || q_t)
```

其中 `omega_t` 是可正可负的方向权重：

```text
omega_t > 0: minimize KL -> 靠近 teacher
omega_t < 0: minimize omega_t * KL -> 等价于增大 KL -> 远离 teacher
```

方向由 entropy 决定，论文使用 smooth signed routing。更贴近原文的写法是：

```text
direction_t = tanh((tau_rho - H_t) / (sigma_H + epsilon))
```

其中：

```text
tau_rho = Quantile_rho({H_t}_t)
sigma_H = MeanAbsDev({H_t}_t)
```

论文默认 `rho = 0.20`，也就是用每条 trajectory 自己的 entropy 分位数当边界，而不是全局固定阈值。

直觉：

```text
H_t < tau_rho  -> direction_t > 0 -> 低熵，靠近 teacher
H_t > tau_rho  -> direction_t < 0 -> 高熵，远离 teacher
```

同时再乘一个 gap-reliability gate：

```text
g_t = sigma(|bar_delta_t| - 1)
omega_t = direction_t * g_t
```

其中：

```text
bar_delta_t = delta_t / (median_t |delta_t| + epsilon)
delta_t = log q_t(o_t) - log p_t(o_t)
```

所以 gap 不是简单的“原始 KL 大”，而是 **归一化后的 teacher-student logprob gap** 相对该轨迹自己的典型尺度是否足够大。

直觉上：

```text
|bar_delta_t| > 1  -> gate 接近 1，说明这个 gap 已经大到足够可信
|bar_delta_t| < 1  -> gate 较小，说明更像弱分歧或噪声，压低更新
```

所以 DASD 不是简单地说“高熵就反着学”，而是：

```text
高熵 + teacher gap 可信 -> 适度远离 teacher；
高熵 + gap 不可信 -> gate 压低，少动；
低熵 + teacher gap 可信 -> 靠近 teacher；
低熵 + gap 不可信 -> gate 压低，少动。
```

## 为什么高熵不是更应该靠近 teacher

关键区别在于：

```text
“student 不确定”不等价于“应该立刻被 teacher 决定”。
```

在 reasoning 任务里，高熵位置经常是：

```text
1. 方法选择点：用代数、枚举、几何还是构造；
2. 假设分叉点：先尝试 A 还是 B；
3. 修正触发点：发现前一步可能错了，要不要回退；
4. 语义转折点：therefore / however / alternatively 这类方向改变。
```

这些 token 的价值在于保留搜索空间。OPSD 如果在这里把模型强行拉向 teacher，会让模型更像 teacher 的单一路径，但也更容易失去：

```text
多样性
自我修正
pass@k 能力
长程探索
```

因此 DASD 的逻辑是：

```text
低熵位置：模型已经知道怎么走，teacher 用来校准和稳定；
高熵位置：模型还在探索，teacher 不应过早收缩它的分布。
```

这就是“高熵远离 teacher”的理论/经验依据。

## 和其他方法的差别

| 方法 | 对 high-entropy token | 对 low-entropy token | 问题 |
|---|---|---|---|
| OPSD | 靠近 teacher | 靠近 teacher | 高熵探索被压缩 |
| Novelty / uniform repulsion | 远离 teacher | 远离 teacher | 低熵执行位被破坏 |
| EGRSD 类 entropy reweight | 少学或多学 teacher | 仍然同向靠近 teacher | 只调强度，不改方向 |
| DASD | 远离 teacher | 靠近 teacher | 同时保探索和执行稳定 |

所以 DASD 的贡献不是简单 reweight，而是：

```text
把 distillation 从“该不该学 teacher / 学多少”
改成“哪些 token 该靠近 teacher，哪些 token 该远离 teacher”。
```

## 诊断和消融

论文把 routing signal 选了很多替代方案来比：

```text
Position proxy
Token frequency
Gradient norm
Attention entropy
Student entropy H_t
```

结果是：

```text
Student entropy H_t 最好。
```

方向图也做了对照：

```text
const +1   -> OPSD 式 uniform attraction
const -1   -> uniform repulsion
hard threshold
linear ramp
tanh direction
```

结果显示：

```text
tanh direction 最稳。
```

gate 也做了对照：

```text
no gate
fixed threshold
magnitude-only
gap-reliability gate
```

结果是：

```text
gap-reliability gate 最好。
```

这说明 DASD 不是简单 reweight，而是有完整的 routing 设计。

## 实验结果

### 主表：六个数学 reasoning benchmark

论文的主 benchmark 是：

```text
MATH500
Minerva
HMMT25
OlympiadBench
AIME24
AIME25
```

主结果里，DASD 在三种模型规模上都取得了最佳 macro Avg@16：

| Model | Base | GRPO | OPSD | DASD |
|---|---:|---:|---:|---:|
| Qwen3-1.7B | 23.5 | 33.8 | 23.6 | 37.7 |
| Qwen3-4B | 31.9 | 45.8 | 36.0 | 49.5 |
| Qwen3-8B | 33.0 | 46.5 | 36.3 | 52.5 |

### Pass@16

论文还报告：

```text
Qwen3-1.7B: 57.9
Qwen3-4B:   68.0
Qwen3-8B:   69.0
```

这说明 DASD 不只是把单次准确率抬高，还保住了多样性。

### Cross-domain / cross-family

Qwen3 交叉领域：

| Method | Code | Scientific | Agentic/tool | Avg |
|---|---:|---:|---:|---:|
| Qwen3-8B | 72.1 | 62.5 | 69.6 | 68.1 |
| +GRPO | 72.3 | 63.0 | 70.1 | 68.5 |
| +OPSD | 69.9 | 64.3 | 70.3 | 68.1 |
| +DASD | 73.7 | 65.2 | 71.1 | 70.0 |

跨 family 的 math average：

| Family | Base | GRPO | OPSD | DASD |
|---|---:|---:|---:|---:|
| Olmo | 33.8 | 49.4 | 34.5 | 54.2 |
| Llama | 31.1 | 44.7 | 31.4 | 49.1 |

### 消融实验整体在验证什么

DASD 的设计有三个核心问题：

```text
1. 用什么信号判断 token 是该探索还是该收敛？
2. 判断出来以后，方向函数应该怎么从“靠近 teacher”过渡到“远离 teacher”？
3. teacher-student gap 什么时候值得相信，什么时候只是噪声？
```

所以论文的消融分三组：

```text
1. Routing signal ablation: 换掉 entropy，看谁最适合做路由信号。
2. Direction map ablation: 固定路由信号，比较不同方向函数。
3. Gate ablation: 固定路由和方向，比较 gap 可靠性门控。
```

评价指标里：

```text
AIME24 / AIME25: reasoning accuracy
Exploration density: 探索密度，越高说明 response 中保留了更多探索/分叉行为
```

### 设计空间消融

Qwen3-1.7B 的消融很关键：

这一组只回答一个问题：**用什么 token-level signal 来决定 toward / away teacher 的方向？**

| Variant | 含义 | 预期问题 |
|---|---|---|
| Position proxy | 用 token 位置作为不确定性代理，比如前后位置不同权重 | 位置不等价于不确定性，早晚 token 都可能是关键分叉 |
| Token frequency | 用 token 出现频率作为代理 | 低频 token 不一定是探索，高频 token 也可能是推理关键位 |
| Gradient norm | 用梯度范数判断 token 是否重要 | 梯度大表示更新强，不一定表示应该靠近还是远离 teacher |
| Attention entropy | 用 attention 分布熵作为不确定性 | attention 不确定性和 next-token 决策不确定性不完全一致 |
| Student entropy H_t | 直接用 student next-token distribution 的熵 | 最贴近“当前 policy 在这个位置是否犹豫” |

结果是 student entropy 最好，说明 DASD 需要的是 **当前 policy 的局部决策不确定性**，不是外部 proxy。

| Variant | AIME24 | AIME25 | Exploration density |
|---|---:|---:|---:|
| Position proxy | 17.8 | 15.9 | 1.6 |
| Token frequency | 20.4 | 17.7 | 2.1 |
| Gradient norm | 25.1 | 22.8 | 2.8 |
| Attention entropy | 30.9 | 27.9 | 3.9 |
| Student entropy `H_t` | 35.0 | 32.1 | 4.3 |

方向图消融：

这一组固定 routing signal 为 student entropy，比较“方向函数”怎么设计。

| Variant | 含义 | 问题 / 作用 |
|---|---|---|
| OPSD: const +1 | 所有 token 都 toward teacher | 等价于普通 OPSD，最容易压缩探索 |
| Novelty: const -1 | 所有 token 都 away from teacher | 探索多，但会破坏低熵执行位 |
| Hard threshold | 熵低于阈值 toward，高于阈值 away，硬切换 | 阈值附近不连续，训练容易抖 |
| Linear ramp | 方向随熵线性变化 | 比 hard threshold 平滑，但边界仍不够柔和 |
| tanh direction | 用 tanh 做平滑有界方向 | 平滑、可正可负、有界，主方法采用 |

结果说明：不是“远离 teacher”本身带来收益，也不是“靠近 teacher”本身带来收益，而是 **按 entropy 平滑切换方向** 才有效。

| Variant | AIME24 | AIME25 | Exploration density |
|---|---:|---:|---:|
| OPSD: const +1 | 15.0 | 9.2 | 0.7 |
| Novelty: const -1 | 12.0 | 10.6 | 5.6 |
| Hard threshold | 31.5 | 28.4 | 4.0 |
| Linear ramp | 33.6 | 30.6 | 4.1 |
| tanh direction | 35.0 | 32.1 | 4.3 |

gate 消融：

这一组固定 entropy routing 和 tanh direction，比较 gap gate 怎么设计。

| Variant | 含义 | 问题 / 作用 |
|---|---|---|
| No gate | 不看 teacher-student gap，direction 直接生效 | 噪声 token 也会被推近/推远 |
| Fixed threshold | gap 超过固定阈值才开 gate | 不同样本/trajectory 的 gap 尺度不同，固定阈值不稳 |
| Magnitude-only | 只看 gap 大小，不做相对尺度归一化 | 大 gap 不一定可信，可能只是该轨迹整体 gap 都大 |
| Gap-reliability gate | 用 trajectory 内 median 归一化后的 gap 控制 gate | 自适应每条 trajectory 的 gap 尺度，过滤弱分歧 |

主方法的 gate 是：

```text
g_t = sigma(|bar_delta_t| - 1)
bar_delta_t = delta_t / (median_t |delta_t| + epsilon)
```

所以它不是问“gap 是否绝对大”，而是问：

```text
这个 token 的 gap 是否比这条 trajectory 的典型 gap 更显著。
```

结果说明：DASD 的方向切换需要可靠性过滤；否则 high-entropy token 远离 teacher、low-entropy token 靠近 teacher 都可能被单点噪声误导。

| Variant | AIME24 | AIME25 | Exploration density |
|---|---:|---:|---:|
| No gate | 29.4 | 26.3 | 3.8 |
| Fixed threshold | 28.6 | 25.6 | 3.7 |
| Magnitude-only | 30.8 | 27.7 | 4.0 |
| Gap-reliability gate | 35.0 | 32.1 | 4.3 |

消融总解释：

```text
1. Routing signal 表证明：student entropy 是最好的 token-level 不确定性代理。
2. Direction map 表证明：只靠近 teacher 或只远离 teacher 都不行，必须方向自适应。
3. Gate 表证明：方向自适应还不够，必须过滤不可靠的 teacher-student gap。
```

因此 DASD 的收益来自三件事叠加：

```text
student entropy routing
+ smooth signed direction
+ gap-reliability gate
```

少任何一个，都会掉分或损失探索密度。

### 训练动态

论文还分析了训练曲线：

```text
1. DASD 能同时保持 reward 增长和 response length；
2. token entropy 不会像 OPSD 那样快速塌缩；
3. high-entropy forks 会保留下来，后续确实能触发修正。
```

## 和 OPSD 的关系

可以把 DASD 看成：

```text
OPSD 的方向修正版
```

而不是单纯的强弱缩放版。

对比起来：

```text
OPSD:
  所有 token 向 teacher 靠拢。

PW-OPSD:
  所有 token 仍然向 teacher 靠拢，
  但按位置调权重。

EGRSD:
  所有 token 仍然向 teacher 靠拢，
  但按 teacher entropy 调权重。

DASD:
  有些 token 向 teacher 靠拢，
  有些 token 要远离 teacher。
```

这也是它最本质的地方：

```text
不是“更弱的 OPSD”，而是“方向分裂的 OPSD”。
```

## 我怎么理解它

DASD 的价值在于它把 reasoning 的两个目标拆开了：

```text
探索和执行不是同一件事。
```

所以 token-level supervision 也不能只有一种方向。

这比单纯的 token reweight 更进一步，因为它承认：

```text
teacher 有时该被信，
有时该被逆着走。
```

