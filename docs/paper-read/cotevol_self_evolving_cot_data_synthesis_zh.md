# CoTEvol：Self-Evolving Chain-of-Thoughts 技术解读

本文档解读论文：

```text
CoTEvol: Self-Evolving Chain-of-Thoughts for Data Synthesis in Mathematical Reasoning
https://arxiv.org/abs/2604.14768
ACL Findings 2026
```

这篇论文研究的是 **数学推理 CoT 训练数据合成**，不是 RL policy training。它把 CoT 生成看成一个 population-based search 问题，用类似遗传算法的方式，在同一道题的多条 reasoning trajectory 上做选择、交叉和变异，最后选出高质量 CoT 用于 SFT。

## 一句话总结

CoTEvol 的核心是：

```text
不要只独立采样很多条 CoT，也不要只对单条 CoT 反复 self-refine。

把多条 CoT 当成一个种群：
  用 fitness function 选出较好的父代；
  用 reflective global crossover 融合不同 CoT 的优点；
  用 uncertainty-guided local mutation 修复高不确定步骤；
  迭代若干轮后选最高 fitness 的 CoT 作为训练数据。
```

它解决的问题是：

```text
高质量人工 CoT 贵；
大模型蒸馏 CoT 成本高且有 teacher bias；
Best-of-N / Self-Refine 这类 test-time search 计算利用率低。
```

## 方法定位

CoTEvol 属于数据合成方法：

```text
input:
  math problem q

output:
  evolved high-quality CoT solution x

usage:
  用 evolved CoT 做 SFT，训练数学推理模型。
```

它不是：

```text
1. OPD / distillation loss。
2. GRPO / PPO 训练策略。
3. reward model 训练。
```

它更像：

```text
用 LLM 自己做 CoT search，
把找到的好轨迹沉淀成 SFT 数据。
```

## 整体流程

对每道数学题：

```text
1. Population Initialization
   采样 N_pop 条初始 CoT。

2. Fitness Evaluation
   用 rule-based verifiers 给每条 CoT 打分。

3. Elite Selection
   根据 fitness 选父代。

4. Reflective Global Crossover
   两条父代 CoT 通过反思反馈做全局融合。

5. Uncertainty-Guided Local Mutation
   找出单条 CoT 中最不稳定的 reasoning step，局部重写。

6. Population Update
   把 offspring 加入种群，按 fitness 保持固定种群规模。

7. Iteration
   重复若干轮。

8. Final Selection
   选择最高 fitness CoT 作为训练样本。
```

论文默认设置：

```text
base generator:
  Qwen2.5-7B-Instruct

population size:
  N_pop = 4

evolution iterations:
  t = 3

selected parents:
  k = 2

generation temperature:
  0.6

max output length:
  2048

mutation base temperature:
  tau_0 = 0.6

mutation entropy coefficient:
  lambda = 5

inference engine:
  vLLM

SFT framework:
  OpenRLHF
```

如果 evolution 后仍然没有正确解：

```text
用 D-CoT 补齐。
```

论文说这个 fallback 平均只占：

```text
6.8%
```

## Fitness Function

fitness function 是进化搜索的评价器，由三个 rule-based verifier 组成：

```text
R = R_ac + R_fmt + R_len
```

### Answer Correctness Verifier

判断最终答案是否正确：

```text
R_ac = 1    if predicted answer correct
R_ac = 0.5  if answer is numerically interpretable but incorrect
R_ac = 0    otherwise
```

这里需要 ground-truth answer。

### Format Matching Verifier

检查最终答案格式是否满足要求：

```text
final answer should be enclosed in \boxed{}
```

如果格式正确：

```text
R_fmt = 0.5
```

否则：

```text
R_fmt = 0
```

### Length-based Reward

长度 reward 用 cosine scaling，鼓励 reasoning 足够完整但不要过度冗长。

设：

```text
L:
  当前 CoT 长度

L_max:
  当前 population 中最大 CoT 长度
```

如果答案正确：

```text
R_len
  = C_min + 0.5 * (C_max - C_min)
    * (1 + cos(pi * L / L_max))
```

如果答案错误：

```text
R_len
  = W_min + 0.5 * (W_max - W_min)
    * (1 + cos(pi * L / L_max))
```

直觉：

```text
正确答案:
  偏好简洁、完整的 CoT。

错误答案:
  也用长度项做轻微调节，但不会盖过正确性。
```

## Population Initialization

对同一道题独立采样 `N_pop` 条候选 CoT：

```text
temperature = 0.6
```

为了避免种群里都是近似重复解，论文做去重：

```text
如果两条 CoT 的 ROUGE-L > 0.7，
随机保留一条。
```

同时过滤：

```text
1. incomplete response
2. malformed response
3. model failure / crash-like output
```

直到得到 `N_pop` 条有效且有一定差异的 CoT。

## Elite Selection

论文用 Boltzmann Tournament Selection：

```text
1. 对每条 CoT 的 fitness 做 softmax。
2. 按概率采样 parent candidates。
3. 在每轮选择 top-k candidates 作为父代。
```

作用：

```text
高 fitness CoT 更可能被选中，
但低 fitness CoT 也保留非零概率，
避免过早收敛和多样性损失。
```

## Reflective Global Crossover

crossover 是全局结构层面的重组。

给定两条 parent CoT，根据 answer correctness 分三种情况生成 reflection feedback：

### 1. Elite Merging

两个 parent 都正确：

```text
提炼二者共同优点和独特技巧，
总结 best practices，
指导 offspring 生成更简洁有效的 CoT。
```

### 2. Success-Error Fusion

一个正确，一个错误：

```text
从正确 CoT 中提取成功策略；
从错误 CoT 中分析错误模式；
指导 offspring 继承有效步骤并规避已知错误。
```

### 3. Failure Pattern Summary

两个 parent 都错误：

```text
总结共同错误和 reasoning pitfalls，
形成 negative knowledge list，
指导模型避开这些路径、探索替代解法。
```

这不是传统 GA 里的硬切片拼接，而是：

```text
LLM-mediated crossover
```

也就是让 LLM 根据父代和反思反馈生成新的 offspring CoT。

## Uncertainty-Guided Local Mutation

mutation 是局部细粒度修复。

核心思想：

```text
高不确定步骤更可能是 reasoning 分叉点或错误点，
应该优先局部重写。
```

### Token Entropy

对 token 位置 `j`：

```text
H_j = - sum_{v in V} p(v | x_<j) log p(v | x_<j)
```

这表示模型对下一个 token 的不确定性。

### Step Entropy

把完整 CoT 切成 reasoning steps：

```text
x = [x_1, ..., x_s]
```

每个 step 覆盖一段 token 索引 `T_s`。

step entropy：

```text
H_step_s
  = (1 / |T_s|) * sum_{j in T_s} H_j
```

选择最不稳定 step：

```text
s* = argmax_s H_step_s
```

### Mutation Temperature

从 `s*` 开始重写，保留之前的 prefix。

mutation temperature：

```text
tau_mut
  = tau_0 * (1 + lambda * H_step_s*)
```

其中：

```text
tau_0 = 0.6
lambda = 5
```

直觉：

```text
不确定性越高，
mutation temperature 越高，
局部探索越强。
```

这样可以：

```text
保留已稳定的前缀推理，
只在最不确定步骤附近探索替代路径。
```

## 无 Ground Truth 版本

论文还定义了不使用 GT answer 的 self-supervised variant。

变化：

```text
1. 移除 answer correctness verifier。
2. 用 LLM self-evaluation 给候选 CoT 打分。
3. crossover 改成基于 parent trajectories 的 self-reflection。
4. mutation 保持不变，但 prompt 中移除 answer-related content。
```

无 GT 版本仍能提升，但效果弱于有 GT。

成功率：

```text
S1K:
  Original SR = 0.359
  Evol. SR w/o GT = 0.491
  Evol. SR w/ GT  = 0.825

LIMO:
  Original SR = 0.420
  Evol. SR w/o GT = 0.537
  Evol. SR w/ GT  = 0.881
```

含义：

```text
即使没有标准答案，CoTEvol 仍可通过 self-evaluation + entropy mutation 找到更好的 CoT；
但 ground-truth correctness verifier 对搜索质量帮助很大。
```

## 训练数据和评测

训练数据：

```text
S1K
LIMO
MATH
DeepMath103k
```

主表重点报告：

```text
S1K
LIMO
```

评测 benchmarks：

```text
GSM8K
MATH500
Minerva Math
GaokaoEn23
OlympiadBench
CollegeMath
AIME24
AMC23
```

评测设置：

```text
zero-shot prompt
greedy decoding
Pass@1
```

## Baselines

### H-CoT

原数据集中的人工或原始 CoT：

```text
Human-annotated CoT
```

### D-CoT

从更强模型蒸馏 CoT：

```text
Qwen2.5-Math-72B-Instruct
Qwen-Distil-32B
Gemini-Flash-2.5
DeepSeek-R1
```

### S-CoT

使用模型自身能力合成 CoT：

```text
Best-of-N
Self-Refine
```

## 主结果

Base model：

```text
Qwen2.5-7B-Instruct
AVG = 46.4
```

### S1K

```text
H-CoT:
  AVG = 47.8

Best D-CoT, DeepSeek-R1:
  AVG = 50.6

Best S-CoT, Self-Refine:
  AVG = 50.9

CoTEvol w/o GT:
  AVG = 51.3

CoTEvol w/ GT:
  AVG = 53.2
```

论文总结：

```text
CoTEvol 比 strongest D-CoT 高 +2.6，
比 strongest S-CoT 高 +2.3。
```

### LIMO

```text
H-CoT:
  AVG = 45.8

Best D-CoT, Qwen-Distil-32B:
  AVG = 49.3

Best S-CoT, Self-Refine:
  AVG = 50.6

CoTEvol w/o GT:
  AVG = 51.3

CoTEvol w/ GT:
  AVG = 52.7
```

论文总结：

```text
CoTEvol 比 strongest D-CoT 高 +3.5，
比 strongest S-CoT 高 +2.1。
```

## 消融实验

消融在 S1K 上做，平均结果：

```text
w/o Crossover:
  AVG = 51.2

w/o Mutation:
  AVG = 52.0

w/o Fitness:
  AVG = 51.3

Full CoTEvol:
  AVG = 53.2
```

结论：

```text
1. 三个组件都有效。
2. Crossover 和 Fitness 对整体效果最关键。
3. Mutation 的下降较小，但稳定贡献局部探索能力。
```

更细分看：

```text
w/o Crossover:
  少了全局重组，难以融合不同 CoT 的有效策略。

w/o Fitness:
  少了选择压力，搜索方向变弱。

w/o Mutation:
  少了局部修复，高不确定步骤无法被定向探索。
```

## 难度分析

论文按 LIMO 设置把问题分成：

```text
Simple-500
Complex-500
Advanced-500
```

成功率：

```text
Simple-500:
  Original 0.899
  w/o GT 0.910
  w/ GT 0.910

Complex-500:
  Original 0.755
  w/o GT 0.820
  w/ GT 0.852

Advanced-500:
  Original 0.405
  w/o GT 0.450
  w/ GT 0.622
```

结论：

```text
简单题已经接近饱和，提升空间小。
中等难度题最适合 evolution，因为初始轨迹有信息但不完美。
高难题受模型能力限制，初始轨迹太差会限制 evolution。
```

## 迭代轮数和 lambda

迭代轮数：

```text
iter 越多效果越好，
但第一轮提升最大，
后续收益递减。
```

论文报告：

```text
iter = 3 时：
  S1K gain = 6.8
  LIMO gain = 6.3
```

mutation 的 `lambda`：

```text
lambda = 5 最好。
```

解释：

```text
lambda 太小:
  mutation 温度不够，探索不足。

lambda 太大:
  mutation 太激进，搜索空间过大，破坏已有推理结构。
```

## 失败样本处理

如果 evolution 失败，论文比较三种处理：

```text
1. discard failed samples
2. replace with H-CoT
3. replace with D-CoT
```

结论：

```text
D-CoT patch 最好。
```

原因：

```text
少量复杂问题超出当前模型自演化能力，
用强模型蒸馏 CoT 补齐可以兼顾质量和成本。
```

这构成一个实际可用方案：

```text
大部分数据用 CoTEvol 自生成；
少量失败样本用强模型 D-CoT patch。
```

## 推理时 search 效果

论文也把 CoTEvol 当 test-time search 方法比较 Pass@10：

```text
Qwen2.5-7B-Instruct:
  MATH500 69.4
  AMC23   47.4

+ Best-of-N:
  MATH500 85.4
  AMC23   82.5

+ Self-Refine:
  MATH500 85.4
  AMC23   85.0

+ CoTEvol:
  MATH500 86.2
  AMC23   87.5
```

说明：

```text
CoTEvol 不只适合生成训练数据，
也可以作为 inference-time search。
```

不过论文主线仍是：

```text
用 CoTEvol 合成 CoT，再做 SFT。
```

## 计算效率

FLOPs 对比，单位为 `10^12`：

```text
Best-of-N:
  1689.53

Self-Refine:
  733.95

CoTEvol:
  453.83
```

原因：

```text
Best-of-N:
  多次采样 + 大 reward model 评估，成本最高。

Self-Refine:
  对所有 solution 做 critique/refine，成本较高。

CoTEvol:
  初始只有 4 条 solution；
  每轮只对 top-2 做 crossover/mutation；
  fitness 是轻量 rule-based verifier。
```

## 和 Best-of-N / Self-Refine 的区别

### Best-of-N

```text
独立采样 N 条。
用 reward model / verifier 选最好。
```

问题：

```text
每条路径互不共享中间信息；
如果采样落在低质量区域，计算利用率低。
```

### Self-Refine

```text
对单条轨迹反复 critique/refine。
```

问题：

```text
局限在单一路径附近，
全局探索能力弱。
```

### CoTEvol

```text
population-level search。
crossover 做全局重组；
mutation 做局部高不确定步骤修复。
```

优势：

```text
能复用不同 CoT 中的有效 reasoning 片段，
也能定向探索不稳定步骤。
```

## 和 RLVR / MaxRL / OPD 的关系

CoTEvol 不是 RL 训练算法，但 fitness function 类似 verifiable reward：

```text
answer correctness
format
length
```

可以把它理解为：

```text
用 rule-based reward 指导 test-time evolutionary search，
再把搜索结果转成 SFT 数据。
```

和 RLVR 的区别：

```text
RLVR:
  reward 用于更新 policy 参数。

CoTEvol:
  reward/fitness 用于搜索和选择 CoT，
  最后用 SFT 更新模型。
```

和 MaxRL 的区别：

```text
MaxRL:
  改 binary correctness RL 的 advantage / objective。

CoTEvol:
  改训练数据生成方式。
```

和 OPD 系列区别：

```text
OPD:
  teacher/student token-level supervision。

CoTEvol:
  不依赖 teacher token logprob，
  主要利用模型自身生成和 rule-based fitness。
```

## 局限

论文指出当前训练时只选每题最高 fitness 的一条 CoT：

```text
best-only selection
```

这保证了监督质量，但没有充分利用 evolution 过程中发现的多条不同有效轨迹。

论文也尝试了 multi-trajectory training，但发现：

```text
naively using multiple solutions does not outperform best-only.
```

因此，如何有效聚合和利用多样化 evolved trajectories 是后续方向。

另外，从方法本身看还有几个实际限制：

```text
1. 有 GT answer 时效果明显更强。
2. 对极难题依赖模型初始能力，初始种群太差时 evolution 受限。
3. 失败样本仍需少量 D-CoT patch。
4. 需要多次推理生成，虽然比 Best-of-N 省，但仍是离线数据合成成本。
```

## 总结

CoTEvol 的本质是：

```text
用遗传算法思想组织 LLM 的 CoT self-synthesis。
```

它的关键设计：

```text
1. CoT population，而不是单条 trajectory。
2. rule-based fitness，提供搜索方向。
3. reflective crossover，融合不同 CoT 的全局策略。
4. entropy mutation，局部修复高不确定步骤。
5. failure patch，用 D-CoT 补齐少量失败样本。
```

最终目标不是在推理时一直搜索，而是：

```text
离线生成高质量 CoT 训练集，
再通过 SFT 提升模型数学推理能力。
```
