# VAPO: Vision-Anchored Policy Optimization 论文解读

论文：

```text
More Thought, Less Accuracy? On the Dual Nature of Reasoning in Vision-Language Models
```

来源：

```text
arXiv:2509.25848
ICLR 2026
```

这里的 VAPO 指：

```text
Vision-Anchored Policy Optimization
```

可以翻成：

```text
视觉锚定策略优化
```

注意不要和另一个纯文本长 CoT RL 方向的 VAPO 混淆。后者指 `Value-model-based Augmented Proximal Policy Optimization`，单独见：

```text
docs/paper-read/vapo_value_model_augmented_ppo_zh.md
```

## 一句话总结

这篇论文的核心不是“让 VLM 想得更长”，而是反过来说：

```text
多模态推理越长，模型可能越不看图。
```

作者把这个现象称为：

```text
visual forgetting
```

可以翻成：

```text
视觉遗忘
```

VAPO 的做法是在 GRPO 的 reward 里加入一个新的 `perception reward`：

```text
推理过程中随机插入若干 visual anchors，
让模型判断一些关于图像的真假 claim，
如果模型在后半段 reasoning 仍能答对这些视觉 claim，
说明它没有把图像忘掉。
```

所以 VAPO 本质上是：

```text
GRPO + 视觉锚点检测 + 感知奖励
```

## 背景：为什么需要 VAPO

Vision-R1、R1-OneVision、VLAA-Thinker 这类模型已经证明：

```text
RL 可以让 VLM 产生更强的 reasoning 能力。
```

但本文指出，多模态场景和纯文本场景不一样。纯文本长 CoT 通常有利于复杂数学和代码推理；但 VLM 的长 CoT 可能带来副作用：

```text
模型越写越像在语言空间里自洽推理，
但越来越少回看图像证据。
```

作者用 `early decision mode` 做分析：

```text
让模型在 reasoning 的不同中间位置停下来直接给答案。
```

结果发现：

```text
1. 早期 reasoning 通常有帮助；
2. reasoning 继续变长后，收益会饱和；
3. 在 MMStar、HallusionBench 这类视觉密集任务上，后半段 reasoning 甚至会让准确率下降；
4. 很多 full reasoning 的错误，其实在早一点停下时能答对。
```

这就是标题里的：

```text
More Thought, Less Accuracy
```

可以翻成：

```text
思考更多，准确率反而更低。
```

## 视觉遗忘是什么

`visual forgetting` 指：

```text
随着文本 reasoning 变长，模型对图像 token 的注意力逐渐下降，
最终推理更依赖语言先验和前文文本，而不是输入图像。
```

论文做了两个 test-time remedy 来验证这个解释。

### Visual Replay

```text
视觉重放
```

做法是在推理中间重新插入图像，让模型再次看到视觉输入。

### Focus Prompt

```text
聚焦提示
```

做法是在推理中间插入类似“我需要重新看图”的提示，让模型主动回看图像。

这两个方法都会让模型的 visual attention ratio 突然上升，并缓解后半段 reasoning 的掉分。作者据此认为：问题不是 reasoning 本身没用，而是 reasoning 过程中视觉 grounding 丢了。

## VAPO 怎么做

VAPO 仍然沿用 GRPO 框架。对每个样本：

```text
输入: question q, image I, ground-truth answer y
policy: pi_theta
rollout: o_i
```

GRPO 先采样一组 responses：

```text
o_1, o_2, ..., o_G
```

每条 response 得到一个 sequence-level reward `R_i`，再做组内归一化 advantage：

```text
A_i = (R_i - mean(R)) / std(R)
```

VAPO 的新增部分是 reward。

## Visual Claim

`visual claim` 可以翻成：

```text
视觉断言
```

它是关于图像的一个简短真假命题，例如：

```text
The gray bus is facing toward the left.        [CORRECT]
There are three different motorcycles.        [WRONG]
The only small car is yellow.                 [CORRECT]
```

论文用 GPT-5 为每张图生成一组 visual claims。要求有两个：

```text
1. balanced：
   正确 claim 和错误 claim 数量均衡，避免模型总猜 yes / no。

2. independent：
   claim 只依赖图像，不依赖原问题和历史 reasoning。
```

`balanced` 可以翻成：

```text
正负样本均衡
```

`independent` 可以翻成：

```text
问题无关 / 独立视觉断言
```

这里的关键是：claim 不应该是“根据问题才能判断”的东西，而应该是“只看图就能判断”的东西。这样它才能检测模型在 reasoning 中是否还保留视觉感知能力。

## Visual Anchor

`visual anchor` 可以翻成：

```text
视觉锚点
```

给定一条 rollout：

```text
o_i = (o_i,1, o_i,2, ..., o_i,T)
```

VAPO 在这条 reasoning trajectory 里选 `K` 个位置：

```text
A_i = {a_1, a_2, ..., a_K},  a_k in [1, T]
```

这些位置就是 anchor。到达某个 anchor 时，拿一个 visual claim `c_k` 去问模型：

```text
<anchor> claim c_k
Is this claim correct? Answer (yes/no):
```

模型只需要输出 yes/no。然后和 claim 的标签 `l_k` 比较。

单个 anchor 的得分是：

```text
s_k = 1[
  argmax_{j in {yes,no}} pi_theta(j | q, I, o_i,<a_k, c_k) = l_k
]
```

也就是：

```text
claim 判断对了，s_k = 1；
claim 判断错了，s_k = 0。
```

这个 `s_k` 不是最终任务 reward，而是推理中间的视觉感知检测分。

## Perception Reward

`perception reward` 可以翻成：

```text
感知奖励
```

它把所有 anchor 的 `s_k` 聚合起来：

```text
R_perc = sum_k w_k s_k / sum_k w_k
```

其中权重是：

```text
w_k = exp(beta * a_k / T)
```

这里 `a_k / T` 表示 anchor 在整条 reasoning 里的相对位置。

`beta` 是 late-emphasis weight，可以翻成：

```text
后段强调权重
```

因为 `w_k` 随着 `a_k/T` 增大而增大，所以越靠后的 anchor 权重越大。

为什么要强调后段？

```text
因为 visual forgetting 主要发生在 reasoning 后半段。
```

所以 VAPO 不是平均检测整条 reasoning，而是更重视后半段还会不会看图。

## 最终 Reward

VAPO 的最终 reward 是：

```text
R_i = R_acc + R_fmt + gamma * 1[R_acc = 1] * R_perc
```

其中：

```text
R_acc:
  最终答案正确性奖励。

R_fmt:
  格式奖励。

R_perc:
  感知奖励。

gamma:
  感知奖励权重。
```

最重要的是这个门控：

```text
1[R_acc = 1]
```

它可以翻成：

```text
准确率门控 / 正确答案门控
```

意思是：

```text
只有最终答案正确时，perception reward 才会生效。
```

这样做是为了防止 reward hacking。

如果没有这个门控，模型可能学会：

```text
少做任务推理，
专门回答 anchor claim，
靠感知奖励拿分。
```

有了这个门控以后，模型必须先把原任务答对，才有资格拿视觉锚点奖励。

## 一个具体计算例子

假设某个 rollout 最终：

```text
R_acc = 1
R_fmt = 1
gamma = 0.1
K = 3
T = 100
beta = 1.5
```

三个 anchor 位置：

```text
a_1 = 20
a_2 = 60
a_3 = 90
```

模型对三个 visual claims 的判断：

```text
s_1 = 1
s_2 = 0
s_3 = 1
```

对应权重：

```text
w_1 = exp(1.5 * 20/100) = exp(0.3) 约 1.35
w_2 = exp(1.5 * 60/100) = exp(0.9) 约 2.46
w_3 = exp(1.5 * 90/100) = exp(1.35) 约 3.86
```

感知奖励：

```text
R_perc
= (1.35 * 1 + 2.46 * 0 + 3.86 * 1) / (1.35 + 2.46 + 3.86)
= 5.21 / 7.67
约 0.68
```

最终 reward：

```text
R_i = 1 + 1 + 0.1 * 1 * 0.68 = 2.068
```

如果最终答案错了：

```text
R_acc = 0
R_fmt = 1
R_perc = 0.68
```

则：

```text
R_i = 0 + 1 + 0.1 * 0 * 0.68 = 1
```

也就是说，视觉 claim 答得再好，只要最终答案错，感知奖励不会进入最终 reward。

## 训练设置

论文默认设置：

```text
base model:
  Qwen2.5-VL-3B / Qwen2.5-VL-7B

training data:
  ViRL39K

training epochs:
  2

learning rate:
  5e-6

responses per example:
  G = 5

sampling temperature:
  1.0

anchor number:
  K = 20

late-emphasis weight:
  beta = 1.5

perception reward weight:
  gamma = 0.1
```

## 实验结果

论文评测了两组 benchmark。

### 数学 / 推理类 benchmark

```text
MathVerse
MathVista
MathVision
LogicVista
WeMath
Geometry3k
```

主要结果：

| Model | MathVerse | MathVista | MathVision | LogicVista | WeMath | Geo3k | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-7B | 40.7 | 62.3 | 23.2 | 42.6 | 33.1 | 38.5 | 40.1 |
| InternVL2.5-8B | 34.5 | 68.2 | 25.6 | 38.3 | 38.6 | 44.8 | 41.7 |
| R1-OneVision-7B | 46.4 | 64.1 | 29.9 | 45.6 | 44.6 | 46.1 | 46.1 |
| VLAA-Thinker-7B | 48.2 | 68.0 | 26.4 | 48.5 | 41.5 | 50.6 | 47.2 |
| Vision-R1-7B | 52.4 | 73.5 | 28.2 | 49.7 | 41.6 | 49.0 | 49.1 |
| VAPO-Thinker-3B | 35.8 | 67.1 | 23.9 | 39.7 | 35.4 | 44.2 | 41.0 |
| VAPO-Thinker-7B | 53.3 | 75.6 | 31.9 | 50.9 | 43.6 | 51.3 | 51.1 |

相对最强 7B baseline `Vision-R1-7B`：

```text
49.1 -> 51.1
提升 2.0 points
```

### 通用多模态 benchmark

```text
MMMU
MMStar
HallusionBench
MMVet
```

主要结果：

| Model | MMMU | MMStar | HallBench | MMVet | Avg |
|---|---:|---:|---:|---:|---:|
| Qwen-VL | 52.7 | 54.9 | 50.0 | 64.8 | 55.6 |
| R1-OneVision | 54.3 | 54.1 | 52.5 | 65.2 | 56.5 |
| VLAA | 59.1 | 49.7 | 54.7 | 70.0 | 58.4 |
| Vision-R1 | 57.6 | 61.4 | 49.5 | 71.1 | 59.9 |
| VAPO | 60.2 | 63.0 | 57.4 | 71.9 | 63.1 |

这里提升更明显：

```text
59.9 -> 63.1
提升 3.2 points
```

这符合论文主张：VAPO 最主要补的是视觉 grounding，所以在视觉密集任务上更明显。

### 全部十个 benchmark

完整表里，VAPO-Thinker-7B 的平均分是：

```text
55.9
```

对比：

```text
Qwen2.5-VL-7B:   46.3
InternVL2.5-8B:  48.1
R1-OneVision-7B: 50.3
VLAA-Thinker-7B: 51.7
Vision-R1-7B:    53.4
VAPO-Thinker-7B: 55.9
```

## 和 test-time remedy 的对比

论文还比较了两个测试时补救方法：

```text
Focus Prompt
Visual Replay
```

结果：

| Model | MathVerse | MathVista | MathVision | LogicVista | WeMath | Geo3k | MMMU | MMStar | HallBench | MMVet | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| VLAA-Thinker-7B | 48.2 | 68.0 | 26.4 | 48.5 | 41.5 | 50.6 | 59.1 | 49.7 | 54.7 | 70.0 | 51.7 |
| + Focus Prompt | 48.8 | 68.4 | 27.3 | 49.2 | 41.8 | 50.7 | 59.7 | 51.1 | 55.2 | 70.7 | 52.3 |
| + Visual Replay | 49.8 | 69.8 | 27.9 | 49.9 | 42.3 | 51.1 | 60.5 | 52.9 | 56.2 | 71.7 | 53.2 |
| Vision-R1-7B | 52.4 | 73.5 | 28.2 | 49.7 | 41.6 | 49.0 | 57.6 | 61.4 | 49.5 | 71.1 | 53.4 |
| + Focus Prompt | 52.8 | 73.8 | 29.3 | 50.2 | 42.1 | 49.7 | 58.7 | 61.8 | 50.5 | 71.4 | 54.0 |
| + Visual Replay | 53.4 | 75.2 | 29.9 | 50.7 | 42.5 | 50.5 | 59.4 | 62.1 | 51.8 | 71.9 | 54.7 |
| VAPO-Thinker-7B | 53.3 | 75.6 | 31.9 | 50.9 | 43.6 | 51.3 | 60.2 | 63.0 | 57.4 | 71.9 | 55.9 |

结论：

```text
test-time 插提示 / 重放图像确实能缓解视觉遗忘，
但 VAPO 作为训练期方法效果更强，而且不需要推理时反复插图。
```

## 消融实验

### Anchor 数量 K

| K | MathVerse | MathVista | MathVision | LogicVista | WeMath | Geo3k | MMMU | MMStar | HallBench | MMVet | Avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 48.2 | 70.6 | 26.1 | 45.5 | 39.1 | 47.3 | 56.6 | 58.9 | 53.2 | 69.9 | 51.5 |
| 5 | 50.9 | 73.1 | 28.4 | 47.8 | 41.2 | 50.5 | 58.4 | 61.3 | 55.6 | 70.4 | 53.8 |
| 10 | 52.8 | 74.8 | 30.7 | 49.5 | 42.5 | 51.2 | 59.9 | 62.2 | 56.7 | 71.2 | 55.1 |
| 15 | 53.2 | 75.8 | 31.5 | 50.4 | 43.9 | 51.0 | 60.4 | 62.7 | 57.0 | 71.6 | 55.8 |
| 20 | 53.3 | 75.6 | 31.9 | 50.9 | 43.6 | 51.3 | 60.2 | 63.0 | 57.4 | 71.9 | 55.9 |

结论：

```text
K 越大通常越好，但 15 到 20 已经接近饱和。
```

### Late-emphasis 权重 beta

| beta | MathVerse | MathVista | MathVision | LogicVista | WeMath | Geo3k | MMMU | MMStar | HallBench | MMVet | Avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 51.7 | 73.8 | 28.4 | 48.6 | 42.0 | 49.9 | 58.9 | 60.2 | 53.8 | 70.3 | 53.8 |
| 0.5 | 52.7 | 74.8 | 30.2 | 49.6 | 42.8 | 50.7 | 59.5 | 62.0 | 55.4 | 71.3 | 54.9 |
| 1.0 | 53.1 | 75.7 | 31.5 | 51.3 | 43.4 | 51.4 | 60.0 | 62.5 | 56.8 | 72.0 | 55.8 |
| 1.5 | 53.3 | 75.6 | 31.9 | 50.9 | 43.6 | 51.3 | 60.2 | 63.0 | 57.4 | 71.9 | 55.9 |
| 2.0 | 53.0 | 75.5 | 31.1 | 50.5 | 43.1 | 51.5 | 60.0 | 63.3 | 56.6 | 71.6 | 55.6 |
| 2.5 | 51.8 | 74.8 | 30.4 | 50.0 | 42.6 | 50.5 | 59.8 | 62.2 | 55.3 | 70.2 | 54.7 |

结论：

```text
beta = 0 等于不强调后段，效果明显较弱。
beta 太大也会掉分。
默认 beta = 1.5 最好。
```

这说明“后半段更需要视觉锚定”这个假设是有效的，但不能过度。

### Anchor 插入位置

| Position | MathVerse | LogicVista | Geometry3k | MMMU | MMStar | HallBench | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 44.2 | 44.4 | 44.5 | 53.6 | 55.2 | 51.2 | 49.0 |
| Uniform | 48.4 | 47.1 | 47.5 | 56.1 | 58.4 | 55.0 | 52.1 |
| Front-loaded | 48.1 | 46.7 | 47.0 | 55.9 | 58.5 | 54.4 | 51.8 |
| Back-loaded | 48.7 | 46.9 | 47.7 | 56.2 | 59.0 | 55.8 | 52.4 |
| Random | 48.9 | 47.3 | 47.7 | 56.7 | 59.1 | 55.5 | 52.5 |

结论：

```text
随机插入最好；
后段插入接近最好；
前段插入较弱。
```

这再次支持 visual forgetting 主要发生在后半段。

### 是否只是数据增强

作者专门做了一个对照：

```text
把 visual claims 直接变成额外训练样本，然后用普通 GRPO。
```

结果：

```text
GRPO + visual-claim data augmentation:
  49.0 -> 49.3

VAPO with same claims as anchors:
  49.0 -> 51.0
```

结论：

```text
VAPO 的收益不是因为多了几条合成样本，
而是因为这些 claim 被用作 reasoning 中间的视觉锚点 reward。
```

### Self-supervised anchors

作者还试了不用 GPT-5，而是让模型自己生成 visual claims：

| Method | MathVerse | LogicVista | Geometry3k | MMMU | MMStar | HallBench | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base Model | 40.7 | 42.6 | 38.5 | 52.7 | 54.9 | 50.0 | 46.6 |
| GRPO | 44.2 | 44.4 | 44.5 | 53.6 | 55.2 | 51.2 | 49.0 |
| VAPO self-supervised | 48.1 | 46.5 | 48.3 | 55.9 | 57.9 | 55.1 | 51.9 |
| VAPO GPT | 48.9 | 47.3 | 47.7 | 56.7 | 59.1 | 55.5 | 52.5 |

结论：

```text
外部 GPT claim 更好，但 self-supervised anchor 也保留了大部分效果。
```

这点很重要，因为它说明 VAPO 不一定必须依赖强闭源模型。

## 和 Vision-R1 / PTST 的关系

Vision-R1 解决的是：

```text
MLLM 缺少高质量 multimodal reasoning cold start；
直接 RL 不稳定；
长 CoT 容易无效变长。
```

它的 PTST 做法是：

```text
先限制 reasoning 长度，
再逐步放宽长度预算。
```

VAPO 解决的是另一个问题：

```text
不是 reasoning 太长本身错，
而是 reasoning 变长后模型忘了看图。
```

所以 VAPO 不通过长度惩罚或长度截断来解决问题。论文还明确说，它没有显式长度 regularization，VAPO-Thinker 的平均 reasoning 长度和 R1-OneVision / Vision-R1 接近。

两者可以这样对比：

| 方法 | 主要控制对象 | 解决的问题 | 核心机制 |
|---|---|---|---|
| Vision-R1 PTST | 输出长度预算 | cold-start 后无效长思考 | staged max generation length |
| VAPO | reasoning 中的视觉依赖 | 长 reasoning 导致视觉遗忘 | visual anchors + perception reward |

直观上：

```text
PTST 是“别一开始想太长”；
VAPO 是“想的时候别忘了看图”。
```

## 和 OPD 的关系

VAPO 不是标准 OPD，因为它没有做：

```text
student rollout 上匹配 teacher token distribution。
```

它更接近：

```text
RLVR + process-level perceptual reward
```

也就是：

```text
最终答案 reward 负责 correctness；
visual anchor reward 负责 grounding；
GRPO 负责 policy update。
```

但它和多模态 OPD 的关注点高度相似：

```text
不要把所有 token / trajectory 都平均看待；
要找出哪些位置真的需要视觉监督。
```

如果把它接到 OPD 线上，可以想象两种扩展：

```text
1. 在 anchor 附近加 teacher-student KL，而不是只给 yes/no reward。
2. 用 visual-claim score 给 OPD loss 做 token/turn weighting。
```

## 论文的核心贡献

我认为主要有三点。

### 1. 提出 multimodal reasoning 的双重性

论文不是简单说 reasoning 有用，而是指出：

```text
reasoning 同时增强 logic inference，
也可能削弱 perception grounding。
```

这个观点很重要，因为它解释了为什么 VLM RL 不能直接照搬文本 R1 经验。

### 2. 把 visual forgetting 做成可训练信号

visual forgetting 本身是分析现象。VAPO 的贡献是把它变成了 reward：

```text
推理中间还能不能判断图像 claim？
```

这比只看 final answer 更细，也比单纯看 attention 更可优化。

### 3. 证明训练期 anchor 优于推理期补丁

Focus Prompt / Visual Replay 能救一部分，但会增加推理成本，也不改变模型本身。

VAPO 让模型在训练阶段学会：

```text
推理过程中持续保持视觉 grounding。
```

## 局限

### 1. 依赖 visual claim 质量

默认 claim 来自 GPT-5。claim 如果不够视觉相关、标签错、或者太简单，perception reward 就会失真。

论文虽然做了 self-supervised anchors，但 GPT 版本仍然更强。

### 2. Reward 仍然是 proxy

会判断 visual claim，不等价于真的在主任务 reasoning 中正确使用视觉证据。

例如模型可能对 claim 判断很好，但在原问题里仍然把视觉关系用错。

### 3. 主要验证单图任务

论文有 multi-image pilot，但主设置仍是 single-image。视频、多图长程 reasoning、agentic visual environment 还需要进一步验证。

### 4. Anchor 插入改变了训练分布

训练时模型会在 reasoning 中遇到 `<anchor>` 判断任务；推理时通常没有这些 anchor。这是一个 train-test mismatch。

论文结果说明这种 mismatch 没有压垮性能，但它仍然是方法设计上的一个需要注意的点。

## 对后续 MLLM + RL / OPD 的启发

VAPO 给出的启发非常直接：

```text
多模态 RL 不能只奖励最终答案。
```

更合理的 post-training 可能需要同时优化：

```text
1. answer correctness；
2. reasoning format；
3. visual grounding；
4. evidence usage；
5. token-level / step-level credit assignment。
```

如果和 OPD 合起来看，VAPO 说明多模态 OPD 的 teacher signal 最好不要平均加在整段输出上，而应该更关注：

```text
1. 后半段 reasoning；
2. 视觉依赖强的 token；
3. claim / evidence / object relation 附近的 token；
4. 容易发生 hallucination drift 的位置。
```

一句话：

```text
VAPO 把“看图”从输入条件，变成了训练过程中持续被检查和奖励的对象。
```
