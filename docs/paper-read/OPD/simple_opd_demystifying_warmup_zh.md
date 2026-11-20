# Simple-OPD：解读 On-policy Distillation 的 Warm-up

> 论文：**Simple-OPD: Demystifying Warm-up for On-policy Distillation**  
> arXiv：<https://arxiv.org/abs/2608.06802>  
> 提交时间：2026-08-07

## 1. 核心结论

这篇论文不提出新的 OPD divergence（散度）或新的 teacher-student loss，而是研究一个很实际的问题：**为什么 OPD 前的 warm-up（预热）有时有效，有时会损害 OOD 泛化？**

论文给出的答案是：

1. warm-up 数据必须包含 CoT（Chain-of-Thought，思维链），因为只学习最终答案不足以初始化后续 OPD 所需的推理行为；
2. CoT 最好由后续 OPD 使用的同一个 teacher（教师模型）生成。关键不是外部模型“更强”，而是 CoT 与 downstream teacher（后续蒸馏教师）兼容；
3. teacher rollout 的最终答案是否正确影响相对次要。只要错误 rollout 仍然保留与正确 rollout 类似的推理、分解和自检结构，也能产生有效 warm-up；
4. full SFT（全参数监督微调）容易损害 OOD 泛化，LoRA 这种低秩、受约束的更新能保留更多原模型能力；
5. Simple-OPD 就是：**用 OPD teacher 生成的 CoT 做 LoRA warm-up，然后继续运行原本的 OPD**。它不修改 OPD objective（目标函数）。

一句话：Simple-OPD 主要迁移的是 **teacher-compatible thinking pattern（与教师兼容的思维模式）**，而不是简单复制正确答案。

## 2. 它和普通 OPD 的关系

Simple-OPD 仍然是 OPD，只是在 OPD 之前增加一个初始化阶段：

```text
student base model
        |
        | teacher rollout with CoT
        v
LoRA SFT warm-up on teacher CoT
        |
        v
merge/use warm-up checkpoint
        |
        v
standard OPD on student on-policy rollouts
```

因此：

- warm-up 阶段是 SFT loss；
- 正式 OPD 阶段仍使用 student 自己 rollout 的 response；
- teacher 在正式 OPD 阶段提供 token-level dense supervision（token 级稠密监督）；
- 没有把 teacher trajectory 当成 OPD 的 student on-policy trajectory；
- 没有引入额外的 PG loss、GRPO 或 reward model；
- 没有改变 OPD 的 RKL objective。

## 3. OPD 预备知识

令 student policy 为 $\pi_\theta$，固定 teacher policy 为 $\pi_T$。给定 prompt $x$，student 自己采样：

$$
\hat y=(\hat y_1,\ldots,\hat y_L)\sim\pi_\theta(\cdot|x).
$$

第 `t` 个 token 的上下文是：

$$
c_t=(x,\hat y_{<t}).
$$

论文采用 token-level reverse KL（token 级反向 KL）：

$$
\ell_t^{\mathrm{RKL}}(\theta)
=D_{\mathrm{KL}}\left(
\pi_\theta(\cdot|c_t)\Vert\pi_T(\cdot|c_t)
\right).
$$

整条 response 的 OPD loss 为：

$$
\mathcal L_{\mathrm{OPD}}(\theta)
=\mathbb E_{x\sim\mathcal D_x,\ \hat y\sim\pi_\theta}
\left[\frac1L\sum_{t=1}^{L}
\ell_t^{\mathrm{RKL}}(\theta)\right].
$$

这里的 on-policy 体现在上下文 `c_t` 是由当前 student 自己采样的前缀构成，而不是 teacher 轨迹的前缀。

### 3.1 一个 token 级 OPD 计算例子

假设 student 在某个 student-generated context 下的下一 token 分布为：

| token | student $p$ | teacher $q$ |
|---|---:|---:|
| $\texttt{verify}$ | 0.60 | 0.80 |
| $\texttt{answer}$ | 0.30 | 0.15 |
| $\texttt{stop}$ | 0.10 | 0.05 |

该位置的 RKL 是：

$$
D_{\mathrm{KL}}(p\Vert q)
=0.6\log\frac{0.6}{0.8}
+0.3\log\frac{0.3}{0.15}
+0.1\log\frac{0.1}{0.05}.
$$

数值约为：

$$
-0.1726+0.2079+0.0693=0.1046.
$$

OPD 会在 student 自己 rollout 的每个 token context 上计算这样的分布差异并平均。Simple-OPD 只改变 OPD 开始前的 student 参数初始化，不改变这个计算。

## 4. Warm-up 数据研究了什么

论文把 warm-up 数据的质量拆成三个问题：

### 4.1 是否有 CoT

比较 teacher response 中包含思维过程的 SFT 与只保留最终答案的 SFT。

观察结果：

- CoT warm-up 在训练过程中持续优于 no-CoT warm-up；
- 同一个训练 step 下，含 CoT checkpoint 的表现更高，且差距通常随 warm-up 继续扩大；
- 后续 OPD 可以改善 no-CoT checkpoint，但不能完全弥补较弱初始化；
- CoT checkpoint 在不同训练时刻的结果更稳定，no-CoT checkpoint 波动更大。

解释：OPD 的后续 token-level teacher supervision 需要 student 已经具备能够继续展开推理的行为基础。只学习最终 answer token，不能充分提供这种过程初始化。

### 4.2 CoT 是否必须来自 OPD teacher

论文比较：

- 后续 OPD 使用的 teacher 生成的 CoT；
- GPT-5.5 生成的 CoT。

结果：即使 GPT-5.5 是更强的外部模型，GPT-5.5 CoT warm-up 仍然接近 student 初始性能，而 OPD teacher CoT 会随着 warm-up 稳定提升 student。

后续 OPD 后，这个差异仍然保留：OPD 只能部分修复 GPT-5.5 warm-up 造成的弱初始化，且 warm-up 越长，结果越接近 direct OPD baseline，甚至变差。

因此作者的结论不是“越强的 teacher 越适合 warm-up”，而是：**warm-up 的 CoT 来源要与正式 OPD 的 teacher 对齐**。

### 4.3 rollout 最终答对是否重要

论文分别用 teacher 的 correct rollout（正确 rollout）和 wrong rollout（错误 rollout）构造 warm-up 数据。

结果：在 SFT 和后续 OPD 后，错误 rollout 的效果与正确 rollout 接近。这里的“错误也可以”有明确前提：错误 rollout 仍可能包含正确的问题分解、中间计算、自检和验证结构；错误通常只是局部计算或最后答案出错。

所以不能把论文结论误读为“错误数据和正确数据永远等价”。更准确的说法是：当 teacher-aligned CoT 已经存在时，**最终 answer correctness（答案正确性）不是 warm-up 质量的唯一主导因素**。

## 5. 为什么 full SFT 会伤害 OOD

论文观察到：

- direct OPD：ID（in-domain，域内）性能逐渐提升，但 OOD 性能较早达到峰值，继续训练后下降；
- full SFT warm-up：初始 ID 性能高、收敛更快，但 warm-up 本身会显著降低 OOD，后续 OPD 不能恢复丢失的泛化能力；
- LoRA warm-up：初期 ID 可能略低于 full SFT，但能较好保留 OOD；之后随着 OPD 继续训练，ID 最终达到相近水平。

直观理解：full SFT 可以自由改变整个模型的参数，使模型快速贴合 teacher CoT 数据分布，但也更容易覆盖原模型中与训练任务无关的通用能力。LoRA 只允许一个低秩方向的参数增量，适应能力受限，因此获得推理模式的同时对原能力干扰更小。

## 6. Simple-OPD 的具体做法

### 6.1 数据构造

1. 从 OPD training set（OPD 训练集）随机采样 prompt；
2. 用正式 OPD teacher 生成 response；
3. 保留 teacher response 中的 CoT 和最终答案；
4. 用这些 $(\text{prompt},\text{teacher CoT response})$ 对 student 做 LoRA SFT；
5. LoRA warm-up 训练到合适的 checkpoint 后，继续执行标准 OPD。

工具返回、环境动作等不涉及本文；实验是数学推理模型的 response distillation。

### 6.2 LoRA 参数化

对于原始权重矩阵 $W_0$，LoRA 不直接更新 $W_0$，而是：

$$
W=W_0+\frac{\alpha}{r}BA,
$$

其中：

- $A\in\mathbb{R}^{r\times k}$、$B\in\mathbb{R}^{d\times r}$ 是可训练矩阵；
- $r$ 是 adaptation rank（适配秩），远小于原矩阵维度；
- $\alpha$ 是 LoRA scaling coefficient（缩放系数）；
- $W_0$ 保持冻结。

### 6.3 不是长度截断式 warm-up

Simple-OPD 的核心不是把 response 截断到固定长度，也不是只挑正确答案。它的设计变量是：

- 是否保留 teacher CoT；
- CoT 是否来自正式 OPD teacher；
- rollout 是否正确；
- full SFT 还是 LoRA；
- LoRA rank 和 warm-up steps。

## 7. 实验设置

### 7.1 评测集

| 类型 | Benchmark | 中文解释 |
|---|---|---|
| ID | AMC23 | 美国高中数学竞赛 2023 |
| ID | MATH-500 | 数学推理测试集 |
| ID | AIME24 / AIME25 | 2024/2025 AIME 数学竞赛题 |
| OOD | IFEval | 指令遵循评测 |
| OOD | GPQA-Diamond | 高难度科学问答 |
| OOD | HumanEval | 代码生成 |
| OOD | MMLU-Pro Physics/Chemistry/History | 物理、化学、历史子集 |

ID Avg. 是 AMC23、MATH-500、AIME24、AIME25 的平均；OOD Avg. 是 IFEval、GPQA、Physics、Chemistry、History 的平均。论文主表将 `HumanEval` 没有列入该汇总表，表中 OOD 五列为 IFEval、GPQA、Phys.、Chem.、Hist.。

### 7.2 三组模型设置

- 非思考模式主设置：Qwen3-1.7B student，GRPO-trained Qwen3-4B teacher；
- 思考模式设置：Qwen3-0.6B-Thinking student，Qwen3-4B-Thinking-2507 teacher；
- OPD variant 设置：Qwen3-1.7B non-thinking student，GRPO-trained Qwen3-4B non-thinking teacher。

### 7.3 超参数

论文附录报告：

| 阶段 | 配置 |
|---|---|
| SFT warm-up | batch size $16$ |
| Full SFT | learning rate $5\times10^{-6}$ |
| LoRA SFT | learning rate $5\times10^{-5}$ |
| OPD | micro batch size $128$，mini batch size $128$，rollout $n=1$ |
| OPD | maximum prompt length $2048$，maximum response length $8192$ |
| OPD | temperature $1.0$，top-$p$ $1.0$，learning rate $1\times10^{-6}$，KL coefficient $0.0$ |

教师基础/指令/思考模式使用不同的最大 response length；附录给出的推理配置分别为 16,384、24,576 和 32,768。论文重点是方法比较，未把所有 warm-up checkpoint 的曲线数据逐点列成表。

## 8. 完整主结果：非思考模式

student 是 Qwen3-1.7B，teacher 是 GRPO-trained Qwen3-4B。数值为百分比。

| 方法 | AIME24 | AIME25 | ID Avg | IFEval | GPQA | Physics | Chemistry | History | OOD Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher | 63.75 | 56.25 | 60.00 | 80.96 | 43.43 | 78.21 | 77.47 | 48.29 | 65.67 |
| Student | 13.75 | 11.87 | 12.81 | 65.80 | 28.79 | 42.03 | 40.46 | 25.46 | 40.51 |
| OPD | 40.21 | 36.46 | 38.34 | **64.88** | 20.71 | 57.89 | **60.60** | **33.07** | 47.43 |
| **Simple-OPD** | **42.50** | **36.88** | **39.69** | 63.96 | **26.77** | **60.51** | 60.51 | **33.07** | **48.96** |
| G-OPD | 44.79 | 38.33 | 41.56 | **64.51** | **13.64** | 54.04 | 55.65 | **35.70** | **44.71** |
| **G-OPD + Simple-OPD** | **46.88** | **39.37** | **43.13** | 63.96 | 13.13 | **55.74** | **57.69** | 32.55 | 44.61 |
| PowerOPD | **43.54** | 34.58 | 39.06 | 62.66 | **21.72** | **59.74** | **59.98** | 29.92 | 46.80 |
| **PowerOPD + Simple-OPD** | 43.13 | **36.88** | **40.01** | **65.80** | 20.20 | 57.97 | 59.01 | **35.17** | **47.63** |

论文在每个 OPD variant 内单独比较 Simple-OPD 与对应 baseline；不能把不同 variant 的绝对值直接当作同一个方法的公平排序。

## 9. 思考模式完整结果

student 是 Qwen3-0.6B thinking，teacher 是 Qwen3-4B-Thinking-2507。数值为百分比。

| 方法 | AIME24 | AIME25 | AMC23 | MATH500 | ID Avg | IFEval | GPQA | Physics | Chemistry | History | OOD Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher | 76.25 | 75.00 | 99.69 | 97.70 | 87.16 | 30.50 | 66.67 | 83.91 | 84.72 | 58.79 | 64.92 |
| Student | 9.17 | 18.75 | 49.69 | 74.15 | 37.94 | 33.09 | 22.22 | 43.42 | 45.49 | 17.85 | 32.41 |
| OPD | **14.17** | 22.08 | 54.37 | 78.80 | 42.36 | 29.76 | 24.24 | **50.73** | **52.21** | 20.21 | 35.43 |
| **Simple-OPD** | 12.92 | **23.33** | **57.19** | **81.80** | **43.81** | **30.87** | **26.77** | 50.65 | 50.80 | **22.31** | **36.28** |

Simple-OPD 的 ID Avg 从 42.36 提升到 43.81，OOD Avg 从 35.43 提升到 36.28。它不保证每个单项都提升，例如 AIME24、Physics、Chemistry 在该设置下略低。

## 10. 消融与结论

### 10.1 CoT 与 no-CoT

结论：CoT warm-up 在同一训练 step 下稳定优于只保留答案的 warm-up；后续 OPD 不能完全修复 no-CoT 的弱初始化。作者将此解释为 CoT 提供了持续的学习信号和可迁移的推理过程。

### 10.2 OPD teacher CoT 与 GPT-5.5 CoT

结论：OPD teacher 的 CoT 在 SFT 和后续 OPD 后都优于 GPT-5.5 CoT。GPT-5.5 的“更强”并不能抵消 teacher mismatch（教师不匹配）。warm-up 数据需要与后续 OPD teacher 的行为分布兼容。

### 10.3 correct rollout 与 wrong rollout

结论：正确和错误 rollout 在 SFT、后续 OPD 后表现相近。作者进一步展示了一个配对例子：两条回答有相同的容斥计数、问题分解和自检结构，但错误回答在局部交集求和处出错并把错误传播到最终答案。由此推断，warm-up 主要迁移 verification pattern（验证模式）和 reasoning skeleton（推理骨架），而不是只迁移最后答案。

### 10.4 Full SFT 与 LoRA warm-up

这是最重要的训练方式消融：

| 方式 | ID 收敛 | OOD 泛化 | 论文解释 |
|---|---|---|---|
| Direct OPD | 较慢；继续训练能提升 | 早期达到峰值，之后下降 | 没有预热，student 起点弱 |
| Full SFT warm-up | 初始 ID 高、收敛快 | 明显下降，后续难恢复 | 全参数更新干扰预训练能力 |
| LoRA warm-up | 初始 ID 可能略低，后续追平 | 明显强于 full SFT | 低秩约束减少参数干扰 |

### 10.5 LoRA rank 与 warm-up steps

附录比较了不同 LoRA rank 和 warm-up steps：

- rank 16 取得最强 OOD trajectory（泛化轨迹）；
- rank 32、64 的 OOD 泛化略弱；
- 增大 rank 对 ID 的额外收益很小，不是 rank 越大越好；
- 增加 warm-up steps 带来更清晰、更一致的提升，尤其是 OOD；
- 作者因此认为 warm-up duration（预热时长）比 LoRA rank 更重要。

### 10.6 与不同 OPD objective 的兼容性

主表显示 Simple-OPD 可以叠加 standard OPD、G-OPD 和 PowerOPD，不需要改这些 OPD objective。非思考模式下的 ID Avg 变化：

| OPD objective | baseline ID Avg | +Simple-OPD ID Avg | 变化 |
|---|---:|---:|---:|
| OPD | 38.34 | 39.69 | +1.35 |
| G-OPD | 41.56 | 43.13 | +1.57 |
| PowerOPD | 39.06 | 40.01 | +0.95 |

对应 OOD Avg 变化分别为 `+1.53`、`-0.10`、`+0.83`。这说明 Simple-OPD 的主要稳定收益在 ID；OOD 不是每个 OPD variant 都提升，但总体没有出现 full SFT 那种系统性退化。

## 11. 论文中“warm-up 更快”的准确含义

这里的“更快”主要指：在相同 OPD update step 下达到更高性能，或达到某个竞争性 ID 性能所需的 OPD 更新更少。它不是说每个 token 的 forward 计算更快，也不是说 teacher rollout 成本降低。

原因是 LoRA warm-up 先把 student 放到一个更接近后续 teacher reasoning behavior 的初始化区域：

$$
\theta_{\mathrm{warm}}
\approx \theta_{\mathrm{base}}+\Delta\theta_{\mathrm{teacher\text{-}aligned}}.
$$

后续 OPD 不必从完全不会展开该思维模式的 student 起步，因此同样数量的 OPD 更新可以更快获得有效的 student contexts 和更有用的 teacher token supervision。

## 12. 局限与我的判断

- 论文实验规模主要是 Qwen3 0.6B/1.7B student 与 4B teacher，不能直接外推到任意规模；
- teacher-aligned CoT 需要额外 teacher rollout 和 warm-up 训练成本；
- wrong rollout 有效的前提是仍保留合理推理结构，随机乱码或严重错误轨迹不应直接等同；
- LoRA rank、warm-up steps 的详细曲线主要以图呈现，论文没有把所有 checkpoint 的数值表完整列出；
- 论文的主贡献是实验诊断和简单配方，不是新的 OPD 理论目标。

我认为最值得记住的结论是：**OPD warm-up 的关键不是把答案提前教给 student，而是让 student 先学会以正式 teacher 能继续监督的方式思考；LoRA 则控制这种初始化对原有能力的破坏。**

## 13. 参考信息

- 论文标题：Simple-OPD: Demystifying Warm-up for On-policy Distillation
- arXiv：2608.06802
- OPD 公式、实验表和消融数据均根据论文 arXiv v1 整理；图中只有曲线、没有明确离散数值的部分不做估读。
