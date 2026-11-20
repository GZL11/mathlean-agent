# Post-Training Shifts Confidence 论文解读

链接：<https://arxiv.org/abs/2607.13753v2>

标题：Post-Training Shifts Confidence: A Three-Stage Analysis of How SFT, RL, and OPD Shape CoT Calibration

## 一句话结论

这篇论文研究的不是“哪个后训练方法最终准确率最高”，而是：`SFT / RL / OPD` 会怎样改变模型在 CoT 推理过程中的置信度校准。

它的核心结论是：

- `OPD` 最擅长在推理开始前判断题目难不难。
- `SFT` 最适合在推理过程中做 early stopping。
- `RL` 最适合在多条完整 CoT 生成后，用置信度挑可靠答案。
- 置信度不是全局可靠的，而是和 CoT 中的相对位置强相关。

基于这个发现，作者提出 `PosConf`：只在模型置信度可靠的位置区间使用置信度。

## 研究问题

推理模型通常只看最终答案准确率，例如一题答对还是答错。但这不能回答下面几个问题：

- 还没开始推理时，模型能不能判断这题自己会不会做？
- 推理到一半时，模型能不能发现当前 CoT 轨迹质量差，提前停掉？
- 生成多条 CoT 后，模型自己的 token 概率能不能帮助选出更可靠的答案？

这篇论文把这些问题拆成三个阶段：

```text
Pre-CoT   : 推理前，预测题目难度
Intra-CoT : 推理中，判断当前轨迹是否该提前终止
Post-CoT  : 推理后，对多条候选答案做聚合或筛选
```

## 对比模型

所有模型都基于同一个 backbone：

```text
Qwen2.5-7B-Instruct
```

论文比较四个模型：

- `Qwen-Instruct`：原始 instruct 模型
- `Qwen-SFT`：在正确 teacher CoT 上做监督微调
- `Qwen-RL`：用可验证答案奖励做 GRPO
- `Qwen-OPD`：学生自己 rollout，teacher 在学生轨迹上提供 token-level dense KL 蒸馏

teacher 是：

```text
DeepSeek-R1-Distill-Qwen-32B
```

训练数据来自：

```text
DeepScaler + SimpleRL
```

这样设计是为了控制变量：模型家族和数据源固定，只比较后训练目标对置信度的影响。

## 置信度怎么计算

给定前缀：

```text
s_l = (P_x, t_1, ..., t_{l-1})
```

模型给出下一个 token 的概率分布：

```text
p(v | s_l)
```

取 top-k token 集合：

```text
V_k(s_l)
```

token-level confidence 定义为：

```text
c_l = - 1/k * sum_{v in V_k(s_l)} log p(v | s_l)
```

论文里用：

```text
k = 20
```

直观理解：

- 如果 top-k token 的概率分布更集中，模型越“确定”。
- 如果分布更平，模型越“不确定”。

注意：这个公式本质上接近 top-k 平均负对数概率。论文把它作为 confidence 使用，数值解释依赖其实现和排序方向。实验中更重要的是它能否把正确轨迹排到错误轨迹前面。

## Pre-CoT：推理前难度估计

Pre-CoT 用 prompt 后第一个 token 的置信度：

```text
C_pre(x) = c_1(P_x)
```

它想衡量：

```text
还没开始写 CoT 时，模型能不能知道这题自己会不会做？
```

难度标签不是人工标的，而是模型特定的：

- 对同一道题采样多条推理轨迹
- 用 majority voting 得到答案
- 如果 majority answer 正确，则这题对该模型是 solved
- 否则是 unsolved

评价指标：

- `AUROC`
- `PRR`

主结果：

| Model | Avg. AUROC | Avg. PRR |
|---|---:|---:|
| Qwen-Instruct | 0.594 | 0.146 |
| Qwen-SFT | 0.526 | 0.095 |
| Qwen-RL | 0.320 | -0.478 |
| Qwen-OPD | 0.644 | 0.345 |

结论：`OPD` 的推理前置信度最好，`RL` 最差，甚至出现明显反校准。

## Intra-CoT：推理中 early stopping

推理过程中，论文用滑动窗口平滑 token confidence：

```text
G_{l,w}(τ) = 1/w * sum_{r=l-w+1}^{l} c_r
```

如果窗口置信度低于阈值：

```text
G_{l,w}(τ) < θ
```

就提前终止这条轨迹。

这个阶段关心的是：

```text
能不能少生成 token，同时保留或提升最终准确率？
```

关键结论：

- `Qwen-SFT` 的推理中置信度最稳定，最适合 early stopping。
- `Qwen-RL` 早期信号弱，但中后期逐渐可靠。
- `Qwen-OPD` 低预算下有收益，但后期会出现反校准风险。

例子：

- Qwen-SFT 在 AIME 2024 上从 full budget 的 `18.84` 提到 30% token budget 的 `26.90`。
- Qwen-SFT 在 AIME 2025 上从 `15.14` 提到 `23.68`。
- Qwen-OPD 在 30% budget 下 AIME 2024 从 `20.78` 提到 `26.49`。

这里不是 early stopping 本身提高了单条轨迹质量，而是它过滤掉了低置信度、不可靠的轨迹，让最后聚合时剩下的轨迹更好。

## Post-CoT：推理后答案聚合

Post-CoT 阶段已经生成完整 CoT。此时有多条 candidate trace：

```text
τ_1, τ_2, ..., τ_n
```

标准方法是 majority voting。

论文测试了几种置信度聚合：

- `Most Confident`：只取最高置信度轨迹的答案
- `Top-5% Conf.`：保留置信度最高的 5% 轨迹，再 majority voting
- `Top-10% Conf.`：保留置信度最高的 10% 轨迹，再 majority voting
- `PosConf-*`：只使用可靠位置区间的 confidence

主结果：

| Model | Majority Vote Avg. | Best Confidence Avg. |
|---|---:|---:|
| Qwen-Instruct | 46.2 | 46.4 |
| Qwen-SFT | 59.8 | 54.5 |
| Qwen-RL | 57.4 | 63.5 |
| Qwen-OPD | 58.5 | 55.0 |

结论：

- `RL` 的完整轨迹置信度最有用。
- `SFT` 和 `OPD` 直接做置信度筛选反而会伤害 majority voting。

这说明最终准确率高，不代表 trace-level confidence 可靠。

## 为什么不同后训练方法表现不同

### SFT

SFT 直接在正确 teacher CoT 上做 token-level cross entropy。

所以它的局部 token 概率比较稳定，正确轨迹和错误轨迹在推理过程中有较稳定 gap。

因此：

- Intra-CoT 好
- Post-CoT 不一定好

原因是 SFT 会让生成轨迹更像 teacher，轨迹之间置信度分布重叠更大，用 trace-level confidence 很难挑出真正正确的轨迹。

### RL

RL 用 outcome reward，只看最终答案是否正确。

所以推理前的置信度不一定知道题目难度。

但一旦模型进入某条推理路径，正确轨迹和错误轨迹的 confidence 开始分离。

因此：

- Pre-CoT 差
- Post-CoT 好

这也是为什么 RL 的 confidence filtering 在多样本聚合时收益最大。

### OPD

OPD 是学生自己 rollout，然后 teacher 在学生轨迹上给 token-level dense KL 指导。

它早期置信度很好，说明 teacher 的监督帮助学生保留了 query-level uncertainty。

但后期会出现一个反直觉现象：

```text
错误轨迹的置信度可能高于正确轨迹
```

这叫 inverse calibration。

因此：

- Pre-CoT 好
- 后期 confidence 可能误导
- naive Post-CoT confidence filtering 会伤害性能

## PosConf 是什么

论文发现置信度可靠性和 CoT 的相对位置有关。

定义相对位置：

```text
ρ_t = t / m
```

其中：

- `t` 是当前 token 位置
- `m` 是整条 trace 长度
- `ρ_t = 0` 接近开头
- `ρ_t = 1` 接近最终答案

PosConf 的核心思想是：

```text
不要整条 CoT 平均 confidence，只取可靠位置区间。
```

对于完整轨迹：

```text
C_w(τ) = sum_{t=1}^{m} w(t/m)c_t / sum_{t=1}^{m} w(t/m)
```

其中 `w(t/m)` 控制哪些位置参与计算。

论文给出的可靠区间：

| Model | Intra-CoT interval | Post-CoT interval |
|---|---|---|
| Qwen-SFT | [0, 1] | [0.6, 1] |
| Qwen-RL | [0.2, 1] | [0.6, 1] |
| Qwen-OPD | [0, 0.6] | [0, 0.6] |

解释：

- SFT：整段都还可以，后段更适合聚合。
- RL：前 20% 不可靠，要等路径开始成型。
- OPD：前 60% 可靠，后面可能反校准。

## PosConf 的效果

对 RL：

- 避免太早使用置信度。
- 在 Post-CoT 聚合上，`PosConf-Top-5%` 平均达到 `63.5`，高于 majority voting 的 `57.4`。

对 OPD：

- 避免后期 inverse calibration。
- 在低 token budget early stopping 上稳定提升。
- AIME 2025 上最多增加 `4.32` 点。

## 这篇论文和 OPD 的关系

这篇论文不是提出新的 OPD 训练算法，而是分析 OPD 训练出来的模型有什么置信度行为。

它对 OPD 的结论很有意思：

- OPD 的前期置信度很有价值。
- OPD 的后期置信度可能反向。
- 所以 OPD 模型不能简单用整条轨迹平均 confidence 做筛选。

这对我们之前读 OPD 系列论文很有启发：OPD 可能优化了 teacher 的 token-level supervision，但这种监督未必让所有位置上的 confidence 都可靠。

## 评测 benchmark

论文使用四个数学推理 benchmark：

- AIME 2024
- AIME 2025
- AMC 2023
- MATH500

生成设置：

- temperature `0.6`
- top-p `0.95`
- maximum generation length `32768`
- 每题离线生成 `320` 条 trace
- 聚合实验固定用 `256` 条 trace
- confidence 计算使用 top-k，`k=20`
- 每个实验重复 5 个随机种子

## 论文的价值

我觉得这篇最重要的价值不是 PosConf 这个技巧本身，而是它提醒我们：

```text
置信度校准不是一个模型级别的单一属性。
```

更准确地说：

```text
同一个模型的 confidence，可能在推理前可靠，在推理后不可靠；
也可能推理前没用，但完整轨迹上很好用。
```

所以在使用 token logprob、entropy、confidence 做 test-time compute 控制时，不能默认整条 CoT 都等价。

## 局限

- 只研究数学推理任务。
- 只使用 Qwen2.5-7B-Instruct 一个 backbone。
- confidence 只来自 token probability，没有系统比较 entropy、margin、hidden state 等信号。
- PosConf 区间来自 AIME 2024 轨迹分析，跨任务泛化还需要更大规模验证。

## 速记

- 论文主题：SFT/RL/OPD 如何改变 CoT 置信度校准
- 核心框架：Pre-CoT / Intra-CoT / Post-CoT
- 核心发现：置信度可靠性是阶段相关、位置相关的
- OPD：早期 confidence 好，后期可能 inverse calibration
- SFT：推理中 confidence 稳，适合 early stopping
- RL：完整轨迹 confidence 好，适合 confidence-based aggregation
- 方法补充：PosConf 只取可靠相对位置区间的 confidence

