# Vision-R1：Incentivizing Reasoning Capability in Multimodal Large Language Models 技术解读

本文档解读论文：

```text
Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models
https://arxiv.org/abs/2503.06749
Official repo: https://github.com/Osilly/Vision-R1
```

## 一句话总结

Vision-R1 研究的是：

```text
DeepSeek-R1-Zero 式 RL 能否直接迁移到多模态大模型 MLLM，
让模型在图文数学题上自然涌现复杂推理能力？
```

论文的结论是：

```text
直接 RL 不够。

MLLM 需要先用高质量 multimodal CoT 做 cold-start，
再用 GRPO + PTST 做渐进式 RL，
才能稳定学出较长、正确、带反思的视觉推理过程。
```

这篇论文的核心不是 OPD，也不是 teacher-student distillation，而是：

```text
multimodal reasoning cold start + RLVR-style training
```

可以把它看成 R1-like RL 在多模态推理上的早期系统化探索。

## 论文背景

DeepSeek-R1-Zero 证明了一个很强的现象：

```text
纯文本 LLM 可以通过 RL，在没有 SFT CoT 的情况下涌现复杂 reasoning。
```

但 MLLM 直接照搬这套流程会遇到两个额外困难：

```text
1. 多模态 reasoning 数据稀缺；
2. 图像信息需要先被正确理解，再进入语言推理链。
```

如果只给图像和问题，然后靠 sparse reward 做 RL，模型很难自己学出：

```text
questioning
reflection
self-correction
long-form CoT
```

这些词可以翻成：

```text
questioning:
  自我提问 / 质疑。

reflection:
  反思。

self-correction:
  自我纠错。

long-form CoT:
  长链式推理。
```

论文把这件事和 DeepSeek-R1 里的 `Aha moment` 联系起来：

```text
Aha moment:
  “啊哈时刻”，指模型在推理中主动发现前面可能有错，
  然后修正思路或重新检查条件。
```

## 直接 RL 为什么不够

论文先做了一个 Vision-R1-Zero 尝试：

```text
base MLLM + 10K multimodal math problems + GRPO
```

奖励包括：

```text
format reward:
  输出必须符合 <think>...</think><answer>...</answer> 格式。

result reward:
  最终答案必须和 ground truth 一致。
```

这种设定模仿 R1-Zero：

```text
只靠 RL 奖励，希望模型自己学出复杂推理。
```

但实验观察是：

```text
1. 直接 RL 很难激活 MLLM 的复杂推理模式；
2. 模型不容易生成高质量、长链条、带反思的 CoT；
3. 随训练推进，模型可能写得更长，但准确率不一定提高。
```

这说明多模态场景里有一个额外瓶颈：

```text
模型不只是不会推理，
还可能没有稳定地把图像信息转成可推理的文本状态。
```

因此 Vision-R1 没有走纯 RL 路线，而是采用：

```text
cold-start initialization + RL training
```

## 整体训练管线

Vision-R1 分两大阶段：

```text
Stage 1:
  构造 Vision-R1-cold 数据集，
  用 SFT 做 cold-start initialization，
  得到 Vision-R1-CI。

Stage 2:
  在 10K multimodal math 数据上，
  用 GRPO + PTST 做 RL training，
  得到 Vision-R1。
```

可以写成：

```text
Base MLLM
  -> SFT on Vision-R1-cold
  -> Vision-R1-CI
  -> GRPO + PTST
  -> Vision-R1
```

论文和官方 repo 的主线模型基于 Qwen2.5-VL 系列，并报告了 7B、32B、72B 规模结果。

## Vision-R1-cold 数据集怎么构造

Vision-R1-cold 是一个约 200K 样本的 multimodal CoT 数据集。关键点是：

```text
不依赖人工 CoT 标注。
```

它靠两个模型配合：

```text
existing MLLM:
  负责把图像信息转成文字描述和初步推理。

DeepSeek-R1:
  负责基于文本化后的视觉信息生成高质量 reasoning trace。
```

论文把这套方法叫：

```text
Modality Bridging
```

可以翻成：

```text
模态桥接。
```

它解决的问题是：

```text
DeepSeek-R1 是文本推理强模型，但不能直接看图；
MLLM 能看图，但复杂推理能力弱。
```

所以用 MLLM 先把图像内容桥接到文本空间，再让 DeepSeek-R1 在文本空间生成高质量 CoT。

## Modality Bridging 具体流程

对一个图像数学题：

```text
(image, question, answer)
```

Vision-R1-cold 的构造大致是：

```text
1. 把 image + question 输入现有 MLLM；
2. 让 MLLM 生成 Pseudo-CoT；
3. 把 image + question + Pseudo-CoT 再输入 MLLM；
4. 让 MLLM 生成更详细的视觉描述；
5. 把详细文本描述 + question 输入 DeepSeek-R1；
6. DeepSeek-R1 生成高质量 CoT 和 final answer；
7. 过滤 final answer 不匹配 ground truth 的样本；
8. 做规则过滤和后处理，保留更一致的样本。
```

这里 `Pseudo-CoT` 可以翻成：

```text
伪 CoT / 初步链式推理。
```

它不是最终训练目标，而是一个视觉信息抽取工具。它的作用是让 MLLM 在生成详细描述时暴露更多和推理相关的视觉细节。

一个简化例子：

```text
题目：
  图中三角形 ABC，已知两条边和一个角，求面积。

普通 caption:
  图中有一个三角形。

Pseudo-CoT 辅助后的 detailed description:
  图中三角形 ABC，AB 标注为 5，AC 标注为 6，
  夹角 A 标注为 30 度，需要使用 1/2 ab sin C。

DeepSeek-R1 看到后：
  可以在纯文本空间完成公式选择、代入、计算和反思。
```

这就是模态桥接的核心：

```text
把“看图问题”变成“信息充分的文本推理问题”。
```

## Cold Start 的作用

用 Vision-R1-cold 做 SFT 后，得到：

```text
Vision-R1-CI
```

其中 `CI` 可以理解为：

```text
Cold-Initialized model
```

中文可以写成：

```text
冷启动初始化模型。
```

它的作用不是直接追求最终分数，而是给后续 RL 一个更好的初始策略：

```text
1. 学会 <think>/<answer> 输出结构；
2. 学会把视觉信息组织进 reasoning；
3. 学会较长 CoT 的基本语言模式；
4. 初步具备 questioning / reflection / self-correction 行为。
```

论文的判断是：

```text
没有这个 cold start，RL 很难从 sparse reward 里自己摸到复杂推理模式。
```

## 过度思考优化问题

Cold start 之后直接做 RL，又会出现另一个问题：

```text
Overthinking Optimization Problem
```

可以翻成：

```text
过度思考优化问题。
```

现象是：

```text
模型生成更长的 CoT，
但并没有带来更高准确率，
甚至长 reasoning 更容易错误。
```

这和纯文本 R1 场景的直觉有差异。对于 MLLM，长推理不一定更好，因为：

```text
1. 视觉信息本身可能已经被误读；
2. 推理越长，错误视觉假设被放大的机会越多；
3. sparse reward 只看最终答案，不能稳定惩罚中间错误；
4. 模型可能学到“写得更长像推理”，而不是“推得更对”。
```

因此 Vision-R1 提出 PTST。

## PTST 是什么

PTST 全称：

```text
Progressive Thinking Suppression Training
```

可以翻成：

```text
渐进式思考抑制训练。
```

这个名字容易误解。它不是要永久压短 reasoning，而是：

```text
训练早期先限制 CoT 长度，
迫使模型学会短而正确的推理；
随着训练推进，再逐步放宽长度限制，
让模型学习更复杂的 reasoning。
```

论文方法概述里给出的阶段性长度/采样安排是：

```text
Stage 1: L_s = 4K, G_s = 16
Stage 2: L_s = 8K, G_s = 8
Stage 3: L_s = 16K, G_s = 4
```

但实验设置里又写成了：

```text
Stage 1: 100 steps, max generation length = 8K, 16 samples/input
Stage 2: 100 steps, max generation length = 8K, 8 samples/input
```

所以更稳妥的读法是：

```text
1. PTST 的核心思想是“先压短，再放宽”。
2. 具体数值在方法概述和实验设置里有轻微不一致。
3. 最终 Vision-R1 使用的是两阶段版本，Stage 3 没有作为最终 checkpoint。
```

PTST 的直觉是：

```text
先学会正确思考，再允许复杂思考。
```

## PTST 有没有长度惩罚

严格说，论文里没有看到一个单独的显式长度惩罚项，例如：

```text
- lambda * |y|
```

它主要是通过**长度约束**而不是**长度惩罚**来控制输出：

```text
O^(s) = { o | |o| <= L_s }
```

也就是第 `s` 个阶段只在满足最大生成长度约束的输出空间里采样和优化。再配合 GRPO 的组内相对更新，模型会被推向“在更短预算内拿到正确答案”的策略。

换句话说：

```text
PTST 控制的是可生成长度上限和训练采样分布，
不是在 reward 里直接扣长度分。
```

## GRPO 训练目标

Vision-R1 使用 GRPO：

```text
Group Relative Policy Optimization
```

可以翻成：

```text
组相对策略优化。
```

对同一个问题 `x`，当前 policy 采样一组 responses：

```text
y_1, y_2, ..., y_G
```

每条 response 得到 reward：

```text
r_i
```

组内归一化 advantage：

```text
A_i = (r_i - mean(r_1, ..., r_G)) / std(r_1, ..., r_G)
```

然后用 PPO-style ratio 更新当前 policy。简化写法：

```text
L_GRPO(theta)
= - E [
    min(
      ratio_i,t * A_i,
      clip(ratio_i,t, 1 - eps, 1 + eps) * A_i
    )
  ]
```

其中：

```text
ratio_i,t =
pi_theta(y_i,t | x, y_i,<t)
/
pi_old(y_i,t | x, y_i,<t)
```

GRPO 的特点是：

```text
不训练 value model；
用同组样本的相对 reward 做 baseline。
```

## HFRRF 是什么

Vision-R1 的 RL reward 使用：

```text
Hard Formatting Result Reward Function
```

可以翻成：

```text
硬格式结果奖励函数。
```

它比普通的：

```text
format reward + result reward
```

更严格。

可以简化写成：

```text
r_i =
  1, 如果格式正确且答案正确；
  0, 否则。
```

其中格式正确通常指：

```text
<think> reasoning process </think>
<answer> final answer </answer>
```

这种 hard reward 的作用是：

```text
不要让模型只靠格式拿分；
也不要让答案对但输出结构混乱的样本被过度奖励。
```

它把 reward 锚定到：

```text
正确答案 + 合法 reasoning format
```

## 一个具体训练例子

假设同一道图形题，模型采样 4 条 responses：

```text
y1: 格式正确，答案正确      r1 = 1
y2: 格式正确，答案错误      r2 = 0
y3: 格式错误，答案正确      r3 = 0
y4: 格式错误，答案错误      r4 = 0
```

组内均值和标准差简化为：

```text
mean = 0.25
std 约等于 0.433
```

则 advantage 大致是：

```text
A1 = (1 - 0.25) / 0.433  =  1.73
A2 = (0 - 0.25) / 0.433  = -0.58
A3 = (0 - 0.25) / 0.433  = -0.58
A4 = (0 - 0.25) / 0.433  = -0.58
```

GRPO 更新会鼓励 `y1` 的 token，压低 `y2/y3/y4` 的 token。

注意：

```text
如果 y3 最终答案正确但格式错，它仍然拿 0。
```

这就是 HFRRF 的 hard 之处。

## PTST 如何作用到这个例子

如果处在 Stage 1：

```text
max reasoning length = 4K / 8K
group size = 16
```

模型不能靠无限拉长 CoT 去“碰运气”。它必须在 4K 内找到正确路线。

如果进入 Stage 2：

```text
max reasoning length = 8K
group size = 8
```

模型已经有一定正确推理模式后，再允许更长链条，处理更复杂问题。

这不是简单的 curriculum learning，而是针对 MLLM 的一个特殊控制：

```text
先压制无效长思考，
再释放有效长思考。
```

更具体地说，它做的是：

```text
在每个阶段固定一个长度预算 L_s，
在该预算下采样 G_s 条候选轨迹，
用 HFRRF 给每条轨迹打 0/1 奖励，
再用 GRPO 更新 policy。
```

因此，PTST 改变的是：

```text
1. 当前阶段允许模型输出多长；
2. 当前阶段每个问题采多少条候选。
```

而不是：

```text
给长输出额外罚分。
```

## 实验结果怎么看

### 评测任务

论文主实验把 benchmark 分成两组。

```text
Math reasoning:
  MM-Math
  MathVista
  MathVerse

General multimodal:
  MMStar
  ChartQA
  MME_sum
  HallBench
```

其中 MathVista 还报告了若干子任务：

```text
GEO:
  geometry reasoning

ARI:
  algebraic reasoning

GPS:
  geometry problem solving

MWP:
  math word problems
```

论文主表以 `MathVista / MathVerse / MM-Math / Avg.` 为主，官方 repo 额外汇总了 `MathVerse mini Vision_Only / DynaMath`。

### 主要 baseline

主表 baseline 大致分四类：

```text
闭源通用 MLLM:
  OpenAI O1
  GPT-4o
  GPT-4V
  Claude-3.5 Sonnet

开源通用 MLLM:
  Qwen2.5-VL-7B
  Qwen2.5-VL-72B
  InternVL2.5-78B

开源数学 MLLM:
  Math-LLaVA-13B
  Math-PUMA-Qwen-2-7B
  Multimath-7B
  URSA-8B

开源 reasoning MLLM:
  LLaVA-CoT-11B
  Mulberry-7B
```

### Table 1 主表关键数值

`-` 表示论文未报告该项。

| Model | MathVista ALL | MathVerse | MM-Math | Avg. |
|---|---:|---:|---:|---:|
| OpenAI O1 | 73.9 | - | - | - |
| GPT-4o | 63.8 | 37.6 | 31.8 | 44.4 |
| GPT-4V | 49.9 | 39.4 | 23.1 | 37.5 |
| Claude-3.5 Sonnet | 67.7 | 26.5 | - | - |
| Qwen2.5-VL-7B | 68.1 | 46.7 | 34.1 | 49.6 |
| Qwen2.5-VL-72B | 73.5 | 51.3 | 45.6 | 56.8 |
| InternVL2.5-78B | 71.9 | 25.4 | 17.8 | 38.4 |
| LLaVA-CoT-11B | - | 54.8 | 20.3 | 30.5 |
| Mulberry-7B | - | 63.1 | 23.7 | - |
| Vision-R1-7B | 73.5 | 52.4 | 40.2 | 55.4 |

Vision-R1-7B 还在 MathVista 子任务上给出：

```text
GEO = 80.3
ARI = 79.0
GPS = 83.2
MWP = 80.6
```

这也是论文最强调的数学推理能力提升点。

### Table 3 通用多模态 benchmark

| Model | MMStar | ChartQA | MME_sum | HallBench | MathVista | MathVerse | MM-Math |
|---|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.2-11B-V | 49.8 | 83.4 | 1787 | 40.3 | 48.6 | 8.4 | 4.1 |
| Mulberry-Llama-11B | 51.3 | 83.5 | 2035 | 48.9 | 61.1 | 20.3 | 16.5 |
| Vision-R1-LlamaV-CI-11B | 61.4 | 83.9 | 2190 | 49.5 | 62.7 | 27.1 | 26.1 |

这张表说明 cold-start 数据不仅对 math 有用，对通用多模态能力也有正向增益。

### 主要结果

论文和官方 repo 报告的核心结论是：

```text
Vision-R1 在多个 multimodal math reasoning benchmarks 上平均提升约 6%。
```

官方 repo 给出的主表包括：

```text
MathVista
MathVerse
MathVerse mini Vision_Only
MM-Math
DynaMath
```

7B 结果：

```text
Qwen2.5-VL-7B:
  AVG = 47.9

Vision-R1-7B:
  AVG = 53.8
  improvement = +5.9
```

其中 MathVista：

```text
Vision-R1-7B = 73.5
```

更细一点看，Vision-R1-7B 在 MathVista 的四个数学相关子任务上表现为：

```text
GEO = 80.3
ARI = 79.0
GPS = 83.2
MWP = 80.6
```

在通用多模态 benchmark 上，它也保持了较强泛化：

```text
MMStar = 52.4
ChartQA = 40.2
MME_sum = 55.4
HallBench = 33.6
```

32B / 72B 结果也提升明显：

```text
Vision-R1-32B:
  AVG = 63.7
  improvement = +11.1

Vision-R1-72B:
  AVG = 65.0
  improvement = +9.2
```

需要注意：

```text
官方 repo 标注 32B 和 72B 使用了额外 RL 数据。
```

所以 7B 结果更适合作为方法干净性判断，32B/72B 更像 scale-up 结果。

### Cold-start 数据质量

Table 2 不看 accuracy，而是统计三种数据里的“自反思标记”出现频次：

```text
Word             LLaVA-CoT(100K)   Mulberry(260K)   Vision-R1-cold(200K)
Wait             2,300             1,122            585,719
Hmm              1                 0                75,853
Mistake          183               8,784            26,697
Alternatively    251               68               188,187
Check            8,332             26,421           100,148
```

论文的解释是：

```text
Vision-R1-cold 里的 questioning / reflection / inspection
明显更密集，因此更适合做 cold-start CoT。
```

更重要的是，作者还用 Llama-3.2-11B-V 在这些 cold-start 数据上做了 SFT：

```text
Vision-R1-LlamaV-CI-11B
  > LLaVA-CoT-11B
  > Mulberry-Llama-11B
```

并且在 MM-Math 上比 Mulberry-Llama-11B 高出 7.4 个点，说明 cold-start 数据本身质量确实更高。

## 消融结论

论文的 cold-start 消融非常关键。简化看：

```text
Vision-R1-Zero:
  直接 RL，效果有限。

Zero + PTST:
  只加 PTST，有提升但不够。

Zero + SFT + PTST:
  如果 SFT 数据没有高质量 CoT annotation，效果可能更差。

Vision-R1:
  高质量 multimodal CoT cold start + PTST + GRPO，效果最好。
```

更具体地看 Table 4：

```text
Vision-R1-Zero:
  Avg. Len = 1285
  Avg. Acc = 50.7

Vision-R1-CI:
  Avg. Len = 3566
  Avg. Acc = 44.5

Vision-R1-Long:
  Avg. Len = 3107
  Avg. Acc = 47.7

Vision-R1:
  Avg. Len = 2057
  Avg. Acc = 55.4
```

这里能读出三层结论：

```text
1. 只做 RL 的 Vision-R1-Zero 不够，推理不稳定。
2. 只做 cold start 的 Vision-R1-CI 会变得很长，但错误链条也更多。
3. 直接把 CI 拉到长长度（Vision-R1-Long）会让优化更难，反而不如 PTST。
4. 两阶段 PTST 的 Vision-R1 最平衡，准确率最高，输出长度也更合理。
```

这说明真正关键的不是“有没有 SFT”，而是：

```text
SFT 的内容是否提供了可学习的 multimodal reasoning pattern。
```

也不是“有没有 RL”，而是：

```text
RL 是否从一个足够好的 reasoning policy 开始优化。
```

## 和 OPD / RLSD 的关系

Vision-R1 不是 OPD，因为它没有做：

```text
student rollout 上的 teacher-student distribution matching。
```

它的 RL 阶段主要是：

```text
student rollout + verifier / rule reward + GRPO update。
```

但它和 OPD/RLSD 线有一个共同主题：

```text
dense reasoning supervision 和 sparse outcome reward 如何配合。
```

Vision-R1 的答案是：

```text
先用 high-quality CoT SFT 提供 dense reasoning prior；
再用 GRPO 的 hard result reward 做 outcome-level refinement。
```

RLSD 的答案是：

```text
RLVR reward 决定方向；
self-distillation teacher signal 决定 token-level 幅度。
```

所以两者可以这样对比：

| 方法 | Dense signal 来自哪里 | Sparse reward 怎么用 | 核心风险 |
|---|---|---|---|
| Vision-R1 | cold-start CoT SFT | GRPO + HFRRF | CoT 数据质量和视觉描述质量决定上限 |
| RLSD | privileged self-teacher logprob gap | RLVR 决定 sign | privileged signal 仍需 clipping 和校准 |
| 普通 OPD | external teacher logits | 可选，不一定需要 | teacher token preference 未必等于最终正确性 |

## 论文真正解决了什么

这篇论文最重要的发现不是：

```text
GRPO 可以训练 MLLM。
```

更准确地说是：

```text
MLLM 的 R1-like RL 需要一个 multimodal reasoning cold start。
```

它解决的是三个具体问题：

```text
1. 数据问题：
   没有足够高质量的 multimodal CoT，直接 RL 难以激活复杂 reasoning。

2. 模态问题：
   文本 R1 很强，但不能看图；
   MLLM 能看图，但 reasoning 弱；
   Modality Bridging 把两者接起来。

3. 优化问题：
   cold start 后模型可能过度长思考；
   PTST 先压短再放长，降低 long wrong CoT 的优化难度。
```

## 局限

我认为主要局限有五点。

### 1. 依赖强文本 reasoning teacher

Vision-R1-cold 的质量依赖 DeepSeek-R1。也就是说：

```text
多模态推理能力的一部分来自文本 R1 的迁移。
```

这不是坏事，但它说明方法并不是纯 RL 涌现。

### 2. 模态桥接可能丢视觉信息

如果 MLLM 在 detailed description 阶段漏掉图像细节，DeepSeek-R1 后续再强也只能在错误文本上推理。

典型风险：

```text
图形题中的长度、角度、方向、遮挡、坐标关系被描述错。
```

### 3. HFRRF 仍然是 sparse reward

HFRRF 很干净，但仍只看：

```text
格式是否正确；
最终答案是否正确。
```

它不能直接指出中间哪一步错了。

### 4. PTST 的阶段设置偏经验

`8K -> 8K -> 16K` 和 `16 -> 8 -> 4` 的 group size 设计有效，但更像经验工程选择。不同模型、数据、任务可能需要重新调。

### 5. 长 CoT 不等于真实视觉推理

模型出现 questioning / reflection 并不必然证明它真的理解图像。它可能学到一种 R1-style reasoning 语言风格。

因此评估时不能只看 CoT 是否像人，而要看：

```text
1. final answer accuracy；
2. visual grounding 是否正确；
3. 对 adversarial visual details 是否稳健；
4. shorter direct-answer setting 下能力是否仍保留。
```

## 对我们阅读 OPD/RL 论文的启发

Vision-R1 给后续多模态 OPD/RL 研究一个很重要的启发：

```text
不要指望 sparse RL reward 自动补齐所有 reasoning structure。
```

对于 MLLM，比较稳的路线可能是：

```text
1. 用 teacher / MLLM / text-R1 构造高质量 multimodal reasoning prior；
2. 用 RLVR 校准 final-answer correctness；
3. 用 OPD / RLSD / DOPD 类方法提供 token-level 或 step-level credit assignment；
4. 对视觉 grounding 单独引入 reward 或 verifier。
```

也就是说，Vision-R1 是一条：

```text
CoT cold start -> GRPO refinement
```

路线，而不是：

```text
OPD teacher logits -> student distribution matching
```

但它和 OPD 的未来结合空间很明显：

```text
Vision-R1-cold 提供 reasoning prior；
RLSD/OPD 提供 token-level credit assignment；
visual verifier 提供 grounding reward。
```

## 最终判断

Vision-R1 的核心贡献可以压成三句话：

```text
1. 直接 R1-Zero 式 RL 不足以稳定激活 MLLM 的复杂推理。

2. 通过 Modality Bridging，可以利用现有 MLLM + DeepSeek-R1
   构造大规模 multimodal CoT cold-start 数据。

3. Cold start 后不能直接鼓励越想越长，
   需要 PTST 先抑制无效长思考，再逐步释放复杂推理长度。
```

如果放到多模态 reasoning post-training 脉络里，它的定位是：

```text
第一批把 R1-like RL 系统迁移到 MLLM 的代表工作之一。
```

它的技术重心不在 teacher model 构造，也不在 OPD loss，而在：

```text
如何为 MLLM 准备足够好的 reasoning initialization，
以及如何避免 RL 把模型推向冗长但错误的视觉 CoT。
```

## BibTeX

```bibtex
@misc{huang2025visionr1,
  title={Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models},
  author={Wenxuan Huang and Bohan Jia and Zijie Zhai and Shaosheng Cao and Zheyu Ye and Fei Zhao and Zhe Xu and Yao Hu and Shaohui Lin},
  year={2025},
  eprint={2503.06749},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2503.06749}
}
```
