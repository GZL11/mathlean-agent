# DOPD：Dual On-policy Distillation 技术解读

本文档解读论文：

```text
DOPD: Dual On-policy Distillation
https://arxiv.org/abs/2606.30626
```

## 一句话总结

DOPD 解决的是一个很具体的 OPD 问题：

```text
当 teacher 或 student 被喂入 privileged information 时，
teacher 比 student 强这件事，可能并不全是“可迁移能力差距”，
也可能只是“信息更占优”。
```

论文把这个问题叫做 **privilege illusion**。

所以 DOPD 不是简单地“更强 teacher 蒸馏更弱 student”，而是：

```text
对每个 token 先看：
1. teacher 和 student 在 privileged context 下的优势差多大；
2. teacher 和 student 对这个 token 的相对概率谁更高。

再决定：
用 teacher 蒸馏、用 student 自蒸馏，
还是只做轻量约束。
```

## Student / Teacher 模型速览

论文主实验使用的是非思维版本的 Qwen3 / Qwen3-VL：

```text
LLM 主实验:
  Teacher:
    Qwen3-8B

  Student:
    Qwen3-1.7B

VLM 主实验:
  Teacher:
    Qwen3-VL-8B

  Student:
    Qwen3-VL-2B
```

此外还做了多组尺度泛化实验：

```text
Qwen3-8B -> Qwen3-0.6B
Qwen3-4B -> Qwen3-0.6B
Qwen3-4B -> Qwen3-1.7B
Qwen3-1.7B -> Qwen3-0.6B
```

这里的 privileged input 不是模型自己随便想出来的，而是训练时额外生成的提示信息：

```text
LLM:
  step-wise decomposition hints / verified rationales

VLM:
  structured visual annotations，
  包括 query-related bounding boxes、object labels、quadruple coordinates
```

这些 privileged input 由 GPT-5.4 生成，并经过质量复核后保留高质量样本。

## 背景：普通 OPD 为什么不够

普通 OPD 是：

```text
student 先 on-policy rollout；
teacher 在 student rollout 上给 token-level 蒸馏信号。
```

这类方法默认一个前提：

```text
teacher 比 student 强，因此 teacher 的 token 偏好应该更值得学。
```

但如果 teacher 输入了 privileged information，这个前提会变得不纯。

举例：

```text
teacher 看到了额外的提示，
student 没看到。

teacher 在某个 token 上概率更高，
到底是因为它真的更懂，
还是只是因为它多看了提示？
```

如果不区分，student 很容易学到：

```text
privileged shortcut
```

也就是论文说的 privilege illusion。

## Privilege Advantage Gap

DOPD 的关键量是 **privilege advantage gap**：

```text
A = | log P_T(y_n | x, p, y_<n) - log P_S(y_n | x, p, y_<n) |
```

这里：

```text
x:
  原始输入。

p:
  privileged information。

y_n:
  当前 token。

P_T:
  privileged teacher policy。

P_S:
  privileged student policy。
```

注意这一步很重要：

```text
teacher 和 student 都先看 privileged input，
再比较它们对同一个 token 的差异。
```

所以这个 gap 不是“老师和学生裸分差”，而是：

```text
在同样 privileged 条件下，
teacher 相对 student 到底多了多少真正的能力差。
```

作者认为：

```text
gap 大:
  更像可迁移的能力差距。

gap 小:
  更像 privileged 信息本身造成的表面优势。
```

## 四类 token 路由

DOPD 把 token 分成四类。

### 1. Low A + High qS & qT

```text
I_LH = (A_n < A_bar) AND (qS + qT >= qS_bar + qT_bar)
```

这类 token 里，teacher 和 student 都比较自信，但两者优势差不大。

论文认为这更像：

```text
大家都已经会了，
更多是 privileged 信息在帮忙，
没必要强行 full teacher imitation。
```

所以用 **轻量 teacher 蒸馏**：

```text
L_LH = beta_l * KL_reverse(P_S(.|x,y_<n) || P_T(.|x,p,y_<n))
```

### 2. Low A + Low qS & qT

```text
I_LL = (A_n < A_bar) AND (qS + qT < qS_bar + qT_bar)
```

这类 token 两边都不自信，说明都可能在 competence 边缘。

如果这时强行 teacher forcing，容易把噪声学进去。

所以论文改用 **弱 student 自约束**：

```text
L_LL = beta_w * KL_reverse(P_S(.|x,y_<n) || sg[P_S(.|x,p,y_<n)] )
```

这里 privileged student 只当稳定锚点，不当知识源。

### 3. High A + High qT

```text
I_HT = (A_n >= A_bar) AND (qT >= qS)
```

这类 token 的特点是：

```text
teacher 在 privileged 条件下明显比 student 更强。
```

因此这是最值得迁移的能力信号。

论文在这里用 **full-vocabulary teacher distillation**，并且用 JS divergence：

```text
L_HT = JS(P_S(.|x,y_<n) || P_T(.|x,p,y_<n))
```

这是最强的监督。

### 4. High A + High qS

```text
I_HS = (A_n >= A_bar) AND (qT < qS)
```

这里 student 自己更自信，teacher 反而不占优。

这时如果硬拉向 teacher，可能抑制 student 的有效探索。

所以还是用 **轻量 privileged-student 蒸馏**：

```text
L_HS = beta_l * KL_reverse(P_S(.|x,y_<n) || sg[P_S(.|x,p,y_<n)] )
```

## 总损失

四类 token 的损失用 mask 组合：

```text
L_DOPD = I_LH * L_LH + I_LL * L_LL + I_HT * L_HT + I_HS * L_HS
```

整体目标是：

```text
E_x [ E_{y ~ P_S} [ 1/|y| * sum_n L_DOPD_n ] ]
```

直觉上就是：

```text
谁真的体现了可迁移能力，就加大蒸馏；
谁更像 privileged shortcut，就轻蒸馏或弱约束；
谁 student 自己更确定，就别强行压回 teacher。
```

## 为什么这和普通 OPD 不一样

普通 OPD 的问题是：

```text
所有 token 默认同一种监督源、同一种强度、同一种目标。
```

DOPD 的问题意识更细：

```text
token 之间的信息价值不一样；
privileged teacher 的优势也不一定都能迁移。
```

所以 DOPD 做的是：

```text
动态路由监督源、目标和强度。
```

这也是它名字里的 **Dual**。

## tensor 视角

可以粗略看成：

```text
N:
  batch 中 token 数

V:
  vocab size

qS, qT:
  [N]

A:
  [N]

mask:
  [N]
```

训练时会多出两次 forward：

```text
privileged teacher forward
privileged student forward
```

然后根据 token regime 选择不同 loss。

## 实验设置

### LLM

训练数据：

```text
RaR-Science-20K
DAPO-Math-17K
Skywork-OR1-Coding-14K
```

评测：

```text
C-Eval
LiveBench
MATH500
AIME25
ZebraLogic
AutoLogi
BFCLv3
LCBv5
```

### VLM

训练数据：

```text
ViRL39K
```

评测：

```text
RealWorldQA
MMStar
MathVision
DynaMath
LogicVista
MMMU
MMMU-Pro
VSI-Bench
```

## 主要结果

### LLM

平均分：

```text
Teacher Policy 52.8
Student Policy 39.1
Vanilla OPD    43.9
Uni-OPD        46.6
EOPD           46.1
TIP            44.8
DOPD           51.4
```

这说明 DOPD 已经很接近 teacher，甚至在某些挑战任务上超过 teacher。

### VLM

平均分：

```text
Teacher Policy 62.9
Student Policy 48.3
Vanilla OPD    52.4
Uni-OPD        54.2
Vision-OPD     55.6
VA-OPD         56.3
DOPD           58.4
```

同样是 DOPD 最好。

## 消融结论

### 去掉 privileged input

性能明显下降。

说明 privileged input 不是装饰，而是 privilege advantage gap 的基础。

### 去掉 teacher distillation

性能掉得很明显，说明 teacher 仍然是主要知识来源。

### 去掉 student distillation

也会掉，说明 student 自约束不是多余的。

### 去掉 advantage-aware routing

性能和稳定性都变差。

说明不是“多蒸馏就好”，而是“该蒸馏谁、蒸馏多强”很重要。

### 去掉 adaptive divergence / strategy

也会退化。

说明 sampled token / top-k / full vocabulary 不是随便选，得按 token 类型来。

## 论文的核心判断

这篇论文的核心判断可以压成一句话：

```text
privileged teacher 的高分，不一定都该学；
只有真正体现能力差的 token，才值得强迁移。
```

所以 DOPD 的意义不是“再做一个 teacher-student distillation”，而是：

```text
把 privileged info 带来的表面优势和真实能力差拆开，
再按 token 级别分别处理。
```

## 和前面几篇的关系

它和 Uni-OPD、SDAR、SIRI 这些方法都沾 on-policy distillation，但重点不同：

```text
Uni-OPD:
  用 outcome reward 校准 teacher token 监督。

SDAR:
  用 gated self-distillation 把 skill-guided 行为内化。

SIRI:
  先挖 skill，再验证 skill，再内化 skill。

DOPD:
  解决 privileged information 带来的 privilege illusion，
  用双路蒸馏把“可迁移能力”和“信息幻觉”拆开。
```

## 我的理解

这篇论文最值得记住的不是公式，而是判断：

```text
teacher 更强，不代表所有强都是可迁移的；
privileged 信息更丰富，也不代表蒸馏收益都是真的能力提升。
```

所以 DOPD 做了一个很实用的事：

```text
把 token 按“能力差”和“自信程度”分层，
然后决定是强蒸馏 teacher，
还是轻蒸馏，
还是让 student 自己稳住。
```

