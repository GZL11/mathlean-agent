# TROLL：用 Token-level Trust Region 替代 PPO Clip 的 LLM RL 训练

> 论文：**TROLL: Trust Regions improve Reinforcement Learning for Large Language Models**  
> arXiv：<https://arxiv.org/abs/2510.03817>  
> 版本信息：arXiv v3, 2026-02-23；ICLR 2026 conference paper  
> 项目页：<https://niklasfreymuth.github.io/troll/>

## 1. 背景

LLM 后训练里的 PPO / GRPO / Dr.GRPO / GSPO / REINFORCE++ 通常都依赖 PPO-style clipping。最近很多工作主要改的是 advantage estimation、reward normalization 或 length normalization，但很少直接改 policy update 里的 clipping 机制。

TROLL 的核心观点是：

> PPO clipping 只是 trust region 的启发式近似。对 LLM 来说，更直接的做法是对每个生成 token 的完整输出分布施加 KL trust region，把新策略投影回旧策略附近。

它不是新的 reward、不是新的 advantage estimator，也不是 OPD / DPO。更准确地说，TROLL 是一个 **optimizer / policy objective 层面的替换件**：把 PPO-like clip objective 换成 token-level KL trust-region projection，可以接在 GRPO、PPO、Dr.GRPO、GSPO、REINFORCE++，甚至 GPG 这类不使用 clipping 的 policy gradient 方法上。

## 2. 问题定义

标准 ratio objective 可以写成：

$$
J_{\text{ratio}}(\theta)
=
\mathbb{E}_{o \sim \pi_{\text{old}}(o|q), D(q)}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
\frac{\tilde{\pi}_\theta(o_t|q,o_{<t})}
{\pi_{\text{old}}(o_t|q,o_{<t})}
A(o_t,q,o_{<t})
\right].
$$

PPO clipping 使用：

$$
J_{\text{ppo}}(\theta)
=
\mathbb{E}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
\min
\left(
r_t A_t,\,
\operatorname{clip}(r_t, 1-\epsilon_{\text{ppo}}, 1+\epsilon_{\text{ppo}}) A_t
\right)
\right],
$$

其中：

$$
r_t =
\frac{\tilde{\pi}_\theta(o_t|q,o_{<t})}
{\pi_{\text{old}}(o_t|q,o_{<t})}.
$$

论文认为 clipping 有两个问题：

1. 它约束的是 sampled token 的 ratio，不是真正约束完整 token distribution 的 KL。
2. ratio 越界后会切断梯度，导致学习效率和稳定性下降。

## 3. 核心方法

### 3.1 Token-level KL Trust Region Projection

对每个输出 token 位置，TROLL 解一个投影问题：

$$
\pi_\theta(o_t|q,o_{<t})
=
\arg\min_{\hat{\pi}_\theta(o_t|q,o_{<t})}
\operatorname{KL}
\left(
\hat{\pi}_\theta(o_t|q,o_{<t})
\middle\|
\tilde{\pi}_\theta(o_t|q,o_{<t})
\right)
$$

约束为：

$$
\operatorname{KL}
\left(
\hat{\pi}_\theta(o_t|q,o_{<t})
\middle\|
\pi_{\text{old}}(o_t|q,o_{<t})
\right)
\le \epsilon.
$$

直观解释：

- $\tilde{\pi}_\theta$ 是当前模型更新后给出的新 token distribution；
- $\pi_{\text{old}}$ 是采样 rollout 时的旧策略分布；
- $\pi_\theta$ 是投影后的策略分布；
- 如果新策略已经在 trust region 内，就不动；
- 如果新策略离旧策略太远，就投影回 KL ball 内。

这个投影有闭式方向：

$$
\pi_\theta(o_t|q,o_{<t})
\propto
\exp
\left(
\frac{
\eta^* \log \pi_{\text{old}}(o_t|q,o_{<t})
+ \log \tilde{\pi}_\theta(o_t|q,o_{<t})
}{
\eta^* + 1
}
\right).
$$

也就是在 logit / log probability 空间里做旧策略和新策略的几何插值。$\eta^*$ 控制投影回旧策略的强度，通过一维 convex dual optimization 求解。

### 3.2 TROLL Objective

投影后，用投影策略 $\pi_\theta$ 做 ratio objective，并额外加一个 regression 项，让原始 LLM 输出 $\tilde{\pi}_\theta$ 靠近投影后的 $\pi_\theta$：

$$
J_{\text{TROLL}}(\theta)
=
\mathbb{E}
\left[
\frac{1}{|o|}
\sum_{t=1}^{|o|}
\left(
\frac{\pi_\theta(o_t|q,o_{<t})}
{\pi_{\text{old}}(o_t|q,o_{<t})}
A_t
\right)
-
\alpha
\operatorname{KL}
\left(
\tilde{\pi}_\theta(o_t|q,o_{<t})
\middle\|
\left\lfloor
\pi_\theta(o_t|q,o_{<t})
\right\rfloor
\right)
\right].
$$

其中 $\lfloor \cdot \rfloor$ 表示 stop-gradient。论文所有实验里 $\alpha=1$。

这个 objective 不假设 $A_t$ 的来源，所以 advantage 可以来自 PPO value model、GRPO group reward、Dr.GRPO、GSPO 或 REINFORCE++。

### 3.3 Sparse Projection

直接对完整 vocab 做投影成本太高。以 Qwen3 tokenizer 为例，vocab size 是 `151,936`，每个 response token 都保存完整 float32 分布会非常贵。

TROLL 的做法是稀疏化：

1. 对每个 token distribution 选概率质量最大的 top-$K$ token；
2. 只保留覆盖累计概率质量 $1-\delta$ 的 token；
3. 始终保留模型真实采样出的 token，保证 sampled token 上仍有梯度；
4. 对丢弃 token 使用很小的 default probability $p_d > 0$；
5. 默认设置是 $K=64$，$\delta=10^{-5}$，也就是保留约 `99.999%` 概率质量。

论文观察到，在多数模型和任务上，平均 `5-10` 个 token 就足以覆盖 `99.999%` 的分布质量。

## 4. 实验设置

### 4.1 任务和数据

| 数据集 / benchmark | 用途 | 规模 / 组成 | 奖励 |
|---|---|---|---|
| DAPO-Math | 主数学 RLVR 训练 | `16,893` train；从原训练集划出 `1,024` 作为 DAPO-Eval | 正则解析 final answer，与 ground truth 匹配，binary reward |
| Math-Eval | 数学 OOD 评测 | MATH500、AMC、AIME2024、AIME2025、OMNI-MATH、OlympiadBench、Minerva | AIME2024 / AIME2025 / AMC 用 `32` responses 取均值，其余 single response |
| GSM8K | 额外数学任务 | 约 `8.5k` train，`1.3k` test | final integer answer 匹配 |
| Eurus-Math | 额外数学 RLVR | `455,261` train，`1,024` eval | 数学答案解析 |
| Eurus-Code | 代码生成 RLVR | `25,276` train，`1,024` eval | 执行测试用例，reward 是通过测试比例；评测 pass@1 |

Eurus-Code 的验证集由 APPS、CodeContests、TACO、Codeforces 组成，数量分别是 `142`、`377`、`382`、`123`。评测时分别算 pass@1 后取平均。

### 4.2 Baseline

论文主要比较两类 update：

- **Clip**：各方法原始 PPO-like clipping 版本；
- **TROLL**：把 clipping 替换为 TROLL trust-region projection。

涉及的训练算法包括：

| 方法 | 侧重点 |
|---|---|
| GRPO | group-relative advantage，无 value model |
| Dr.GRPO | 修正 GRPO 中偏向长回答的优化 bias |
| PPO | value model + GAE |
| GSPO | sequence-level importance ratio |
| REINFORCE++ / RF++ | global advantage normalization |
| BAPO | adaptive clipping baseline |
| GPG | clipping-free policy gradient baseline |

## 5. 完整实验表格

### 5.1 Table 1：DAPO 主实验，Qwen3-8B 和 Qwen2.5-7B-Instruct

| Model | Metric | Update | GRPO | Dr.GRPO | PPO | GSPO | RF++ |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-8B | DAPO Train | Clip | 0.667 | 0.678 | 0.640 | 0.000 | 0.648 |
| Qwen3-8B | DAPO Train | TROLL | 0.721 | 0.704 | 0.744 | 0.736 | 0.742 |
| Qwen3-8B | DAPO Eval | Clip | 0.640 | 0.653 | 0.602 | 0.000 | 0.626 |
| Qwen3-8B | DAPO Eval | TROLL | 0.691 | 0.674 | 0.715 | 0.706 | 0.728 |
| Qwen3-8B | MATH Eval | Clip | 0.541 | 0.549 | 0.508 | 0.000 | 0.520 |
| Qwen3-8B | MATH Eval | TROLL | 0.551 | 0.546 | 0.591 | 0.580 | 0.578 |
| Qwen2.5-7B-Instruct | DAPO Train | Clip | 0.443 | 0.467 | 0.444 | 0.159 | 0.429 |
| Qwen2.5-7B-Instruct | DAPO Train | TROLL | 0.495 | 0.513 | 0.431 | 0.481 | 0.486 |
| Qwen2.5-7B-Instruct | DAPO Eval | Clip | 0.323 | 0.331 | 0.324 | 0.093 | 0.323 |
| Qwen2.5-7B-Instruct | DAPO Eval | TROLL | 0.389 | 0.389 | 0.353 | 0.390 | 0.380 |
| Qwen2.5-7B-Instruct | MATH Eval | Clip | 0.313 | 0.317 | 0.319 | 0.127 | 0.311 |
| Qwen2.5-7B-Instruct | MATH Eval | TROLL | 0.350 | 0.359 | 0.349 | 0.333 | 0.344 |

主结论：

1. TROLL 在多数方法和指标上优于 Clip。
2. GSPO + Clip 在两个模型上都出现明显 divergence，Qwen3-8B 最终为 `0.000`；GSPO + TROLL 稳定训练。
3. 论文总结数学任务上通常有 `3-10` percentage points 提升。

### 5.2 Table 2：实验使用的模型 checkpoint

| Model | Checkpoint |
|---|---|
| Qwen3-0.6B | <https://huggingface.co/Qwen/Qwen3-0.6B> |
| Qwen3-1.7B | <https://huggingface.co/Qwen/Qwen3-1.7B> |
| Qwen3-4B | <https://huggingface.co/Qwen/Qwen3-4B> |
| Qwen3-8B | <https://huggingface.co/Qwen/Qwen3-8B> |
| Qwen3-14B | <https://huggingface.co/Qwen/Qwen3-14B> |
| Qwen2.5-0.5B-Instruct | <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct> |
| Qwen2.5-1.5B-Instruct | <https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct> |
| Qwen2.5-3B-Instruct | <https://huggingface.co/Qwen/Qwen2.5-3B-Instruct> |
| Qwen2.5-7B-Instruct | <https://huggingface.co/Qwen/Qwen2.5-7B-Instruct> |
| Llama-3.1-8B | <https://huggingface.co/meta-llama/Llama-3.1-8B> |
| Llama-3.1-8B-Instruct | <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> |
| Llama-3.2-3B | <https://huggingface.co/meta-llama/Llama-3.2-3B> |
| Llama-3.2-3B-Instruct | <https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct> |
| FineMath-Llama-3B | <https://huggingface.co/HuggingFaceTB/FineMath-Llama-3B> |
| Apertus-8B | <https://huggingface.co/swiss-ai/Apertus-8B-2509> |
| Apertus-8B-Instruct | <https://huggingface.co/swiss-ai/Apertus-8B-Instruct-2509> |
| SmolLM3-3B | <https://huggingface.co/HuggingFaceTB/SmolLM3-3B> |

### 5.3 Table 3：默认超参数

| Hyperparameter | Symbol / Variable | Value |
|---|---|---:|
| Trust Region Size | $\epsilon$ | 0.05 |
| KL Regression Factor | $\alpha$ | 1.0 |
| Sparsity Remaining Mass | $1-\delta$ | 0.99999 |
| Max Sparse Tokens | $K$ | 64 |
| Chunk Size |  | 1024 |
| Clip Value | $\epsilon_{\text{ppo}}$ | 0.2 |
| Learning Rate |  | $10^{-6}$ |
| Gradient Max Norm |  | 1.0 |
| Weight Decay |  | 0.0 |
| Learning Rate Schedule |  | constant |
| Learning Rate Critic (PPO only) |  | $10^{-5}$ |
| Weight Decay Critic (PPO only) |  | 0.01 |
| Sampler Per Query |  | 8 |
| Batch Size |  | 32 |
| Batches Per Step |  | 8 |

### 5.4 Table 4：Math-Eval 各子集完整结果

| Model | Method | Update | AIME24 | AIME25 | AMC | MATH | Omni-Math | Olympiad | Minerva |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | GRPO | Clip | 0.066 | 0.075 | 0.535 | 0.683 | 0.239 | 0.286 | 0.304 |
| Qwen2.5-7B-Instruct | GRPO | TROLL | 0.168 | 0.129 | 0.587 | 0.712 | 0.254 | 0.317 | 0.284 |
| Qwen2.5-7B-Instruct | Dr.GRPO | Clip | 0.103 | 0.067 | 0.560 | 0.662 | 0.242 | 0.288 | 0.295 |
| Qwen2.5-7B-Instruct | Dr.GRPO | TROLL | 0.168 | 0.135 | 0.605 | 0.706 | 0.259 | 0.320 | 0.317 |
| Qwen2.5-7B-Instruct | PPO | Clip | 0.092 | 0.064 | 0.503 | 0.706 | 0.251 | 0.316 | 0.299 |
| Qwen2.5-7B-Instruct | PPO | TROLL | 0.162 | 0.093 | 0.547 | 0.734 | 0.258 | 0.320 | 0.332 |
| Qwen2.5-7B-Instruct | GSPO | Clip | 0.026 | 0.002 | 0.188 | 0.344 | 0.106 | 0.102 | 0.120 |
| Qwen2.5-7B-Instruct | GSPO | TROLL | 0.159 | 0.076 | 0.531 | 0.699 | 0.257 | 0.297 | 0.310 |
| Qwen2.5-7B-Instruct | RF++ | Clip | 0.096 | 0.033 | 0.556 | 0.666 | 0.239 | 0.291 | 0.299 |
| Qwen2.5-7B-Instruct | RF++ | TROLL | 0.166 | 0.102 | 0.552 | 0.687 | 0.263 | 0.320 | 0.319 |
| Qwen3-8B | GRPO | Clip | 0.439 | 0.293 | 0.720 | 0.889 | 0.465 | 0.547 | 0.431 |
| Qwen3-8B | GRPO | TROLL | 0.547 | 0.353 | 0.790 | 0.812 | 0.465 | 0.497 | 0.391 |
| Qwen3-8B | Dr.GRPO | Clip | 0.458 | 0.305 | 0.743 | 0.891 | 0.477 | 0.547 | 0.425 |
| Qwen3-8B | Dr.GRPO | TROLL | 0.447 | 0.337 | 0.769 | 0.880 | 0.466 | 0.522 | 0.403 |
| Qwen3-8B | PPO | Clip | 0.380 | 0.234 | 0.694 | 0.874 | 0.439 | 0.531 | 0.405 |
| Qwen3-8B | PPO | TROLL | 0.524 | 0.408 | 0.780 | 0.910 | 0.521 | 0.567 | 0.425 |
| Qwen3-8B | GSPO | Clip | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Qwen3-8B | GSPO | TROLL | 0.474 | 0.407 | 0.813 | 0.897 | 0.514 | 0.547 | 0.408 |
| Qwen3-8B | RF++ | Clip | 0.356 | 0.251 | 0.728 | 0.897 | 0.459 | 0.538 | 0.415 |
| Qwen3-8B | RF++ | TROLL | 0.490 | 0.355 | 0.803 | 0.891 | 0.528 | 0.563 | 0.415 |

这张表更细地说明：TROLL 不是只在平均分上提升，而是在 AIME、AMC、MATH、Omni-Math、OlympiadBench 等多个子集上普遍更稳。例外也存在，比如 Qwen3-8B 的 GRPO / Dr.GRPO 在 MATH、Olympiad、Minerva 上 Clip 个别更高，论文解释为 Qwen3 GRPO / Dr.GRPO 在 OOD Math-Eval 上后期有轻微 overfit。

### 5.5 Table 5：TROLL 的显存和运行时开销

设置：Qwen3 小模型在 GSM8K 上用 GRPO，4x Nvidia A100-40GB，控制 response length 为约 `256` token。

| Metric / Method | Qwen3-0.6B | Qwen3-1.7B | Qwen3-4B |
|---|---:|---:|---:|
| VRAM (Clip) | 25.415 GiB | 28.418 GiB | 34.574 GiB |
| VRAM (TROLL) | 27.868 GiB | 30.663 GiB | 36.837 GiB |
| VRAM (TROLL Chunked) | 27.227 GiB | 29.994 GiB | 36.157 GiB |
| VRAM Delta (Chunked) | +1.812 GiB (+7.1%) | +1.576 GiB (+5.5%) | +1.583 GiB (+4.6%) |
| Runtime (Clip) | 30.874 s | 43.372 s | 85.133 s |
| Runtime (TROLL) | 46.715 s | 49.053 s | 90.570 s |
| Runtime (TROLL Chunked) | 47.600 s | 50.629 s | 92.906 s |
| Runtime Delta (Chunked) | +16.726 s (+54.2%) | +7.257 s (+16.7%) | +7.773 s (+9.1%) |

论文强调 projection overhead 主要和 vocab size 相关，随模型参数量增加相对占比会下降。0.6B 因为没有充分打满 GPU，runtime delta 看起来偏大；到 4B 时 runtime overhead 已低于 `10%`。

## 6. 其它主实验结论

### 6.1 Qwen3 多尺寸模型

Figure 3 比较了 Qwen3 `0.6B / 1.7B / 4B / 8B / 14B` 在 DAPO 和 Eurus-Code 上的 GRPO 训练。

结论：

1. DAPO 上，TROLL 在训练集、DAPO-Eval、Math-Eval 都提升 sample efficiency 和最终成功率。
2. Qwen3-4B + TROLL 的表现接近 Qwen3-14B + Clip。
3. Eurus-Code 上，TROLL 对所有 Qwen3 尺寸都有明显提升，论文报告提升约 `7-18` percentage points，relative gain 为 `18-30%`。

### 6.2 其它模型族

Figure 4、Figure 10、Figure 11 比较了 Llama、SmolLM3、FineMath-Llama、Apertus 等模型在 GSM8K 上的训练。

结论：

1. 对 Llama 系列，Clip 往往需要更多 steps 才出现正向训练信号；TROLL 学得更快。
2. 对 Apertus，TROLL 更稳定，Clip 在部分设置中失败。
3. 对本来已经接近饱和的模型，TROLL 仍通常有小幅提升。

### 6.3 其它训练算法

Figure 14 比较 GRPO(TROLL)、GRPO(Clip)、GRPO(BAPO)、GPG(TROLL)、GPG。

结论：

1. BAPO 的 adaptive clipping 在 evaluation 上略好于普通 Clip，但仍明显低于 TROLL。
2. vanilla GPG 初期能学，但后期会 divergence。
3. GPG + TROLL 后训练稳定性恢复，表现接近 GRPO(TROLL)。

## 7. 消融实验

论文的消融主要以曲线图给出，没有为每条曲线提供逐点数值表。因此下面区分两类信息：原文表格数值只来自 Table 5；其它为原文 Figure 5 / 15 / 16 / 17 / 18 的曲线结论。

| 消融项 | 设置 | 模型 / 数据 | 结果与结论 |
|---|---|---|---|
| KL bound | 默认 $\epsilon=0.05$；对比 $\epsilon=0.01$ 和 $\epsilon=0.25$ | Qwen3-1.7B + GRPO + DAPO | 更小 KL bound 学得更慢，但最终可接近默认；更大 KL bound 表现更差，说明 update 太激进会伤害训练 |
| Sparse top-K | 默认 $K=64$；对比 $K=16$ 和 $K=256$ | Qwen3-1.7B + GRPO + DAPO | $K=16$ 明显变差，可能因为 KL 估计不准；$K=256$ 成本更高但不优于 $K=64$ |
| Sparsity mass | 默认 $\delta=10^{-5}$；随 $K$ 调整到 $10^{-4}$ / $10^{-6}$ | Qwen3-1.7B + GRPO + DAPO | 默认设置已足够，平均 `5-10` 个 token 通常能覆盖 `99.999%` 概率质量 |
| Batch size / stale data | batch size 从 `256` 增到 `2048` | Qwen3-1.7B + GRPO + DAPO | TROLL 对更大 batch size 仍稳定；Clip 随 batch size 增大逐步退化 |
| Entropy | 比较训练中 token entropy | Qwen3-14B + GRPO；Eurus-Code 额外分析 | Clip 往往通过降低 entropy 提升 reward；TROLL 更能保留 entropy，同时提升成功率 |
| Projection fraction | Clip 的 clipped token fraction vs TROLL 的 projected token fraction | Qwen3 / Qwen2.5 多尺寸 | 两者影响 token 比例都大约略高于 `0.1%`，所以稳定性提升不是因为 TROLL 简单地限制了更多 token |
| Response length | 训练中 response length 变化 | Qwen3 / Qwen2.5 多尺寸 | TROLL 更快把回答长度调整到适合任务的范围；Qwen3 倾向缩短，Qwen2.5-Instruct 倾向变长 |
| Runtime / VRAM | Table 5 | Qwen3-0.6B / 1.7B / 4B + GSM8K | 4B 上 chunked TROLL 只增加 `+1.583 GiB` 显存和 `+9.1%` runtime |

消融结论：

1. TROLL 需要足够准确的 KL projection；过小 $K$ 会影响 KL 估计，性能下降。
2. 合理范围内超参数不敏感，默认 $\epsilon=0.05$、$K=64$、$\delta=10^{-5}$ 是较稳的选择。
3. TROLL 的稳定性不仅来自“更保守”，因为 projected token fraction 和 Clip 的 clipped token fraction 数量级接近。
4. TROLL 对较大 batch 和较 stale 的 rollout 更稳，这对大规模 RL 训练很关键。
5. 稀疏化是可用性的关键，没有 sparse projection，完整 vocab distribution 的内存不可接受。

## 8. 这篇论文核心想说什么

一句话：

> LLM RL 里很多不稳定性不一定来自 reward 或 advantage，而可能来自 PPO clipping 这个粗糙的 update 约束；把它换成真正的 token-level KL trust-region projection，可以在多个模型、任务和 RL 算法上更稳、更快、更高分。

它的贡献不是训练数据，也不是新 benchmark，而是 **训练算法 / policy objective**。

对 agentic RL / tool-use RL 的启发是：

1. 如果 GRPO / PPO 在长链路任务上不稳定，不一定只调 reward，也可以检查 update constraint 本身。
2. TROLL 可以和过程奖励、outcome reward、RLAIF、tool-use verifier reward 并用，因为它不关心 advantage 怎么来。
3. 对长程 agent 来说，大 batch、stale rollout、multi-step minibatch 更新很常见；TROLL 对 batch size/staleness 更稳这一点有实际价值。
4. 它不会改变 inference 行为，只改变训练时 objective，因此工程落地成本主要在 rollout buffer 需要保存 sparse old logits，以及实现 projection layer。

## 9. 局限

1. 主实验集中在 RLVR 数学和代码任务，还没有直接验证多轮工具调用、网页 agent、创作 agent。
2. 需要保存旧策略的 sparse logits，比普通 GRPO rollout buffer 更复杂。
3. Projection 和 sparse distribution 实现有工程成本，尤其需要处理大 vocab、chunking、默认概率和数值稳定。
4. 消融曲线没有提供完整数值表，很多分析是趋势性结论。
5. 它解决的是 policy update 稳定性，不解决 reward hacking、credit assignment、环境设计或数据覆盖问题。

## 10. 总结

TROLL 可以看作 PPO clipping 的原则化替代：

- PPO Clip：限制 sampled token 的 ratio，越界后切梯度；
- TROLL：对每个 token 的完整输出分布做 KL trust-region projection，并让原始 LLM 输出回归到投影分布。

从实验看，它在数学 RLVR、代码 RLVR、不同模型族和不同 advantage estimator 上都比 Clip 更稳。对当前做 agentic RL 的价值在于：它不是替代 GRPO / PPO / process reward，而是可以作为这些方法底层的 policy update 约束，用来减少长链路训练中的 collapse、divergence 和 sample inefficiency。

## 11. 附：TROLL 投影的数学过程

对每个 token 位置，TROLL 先解一个带 KL 约束的投影问题：

$$
\pi_\theta(o_t|q,o_{<t})
=
\arg\min_{\hat{\pi}_\theta(o_t|q,o_{<t})}
\operatorname{KL}\left(
\hat{\pi}_\theta(o_t|q,o_{<t})
\middle\|
\tilde{\pi}_\theta(o_t|q,o_{<t})
\right)
$$

约束是：

$$
\operatorname{KL}\left(
\hat{\pi}_\theta(o_t|q,o_{<t})
\middle\|
\pi_{\text{old}}(o_t|q,o_{<t})
\right)\le\epsilon,
\qquad
\sum_{o_t}\pi_\theta(o_t|q,o_{<t})=1.
$$

引入拉格朗日乘子 $\eta>0$ 和 $\lambda$ 后：

$$
\mathcal{L}(\pi_\theta,\eta,\lambda)
=
\operatorname{KL}(\pi_\theta\|\tilde{\pi}_\theta)
-\eta\left(\epsilon-\operatorname{KL}(\pi_\theta\|\pi_{\text{old}})\right)
-\lambda\left(1-\sum_{o_t}\pi_\theta(o_t|q,o_{<t})\right).
$$

对 $\pi_\theta$ 求一阶条件，可以得到闭式形式：

$$
\log \pi_\theta(o_t|q,o_{<t})
=
\frac{
\log \tilde{\pi}_\theta(o_t|q,o_{<t})
+\eta\log\pi_{\text{old}}(o_t|q,o_{<t})
}{\eta+1}
-\text{const},
$$

因此

$$
\pi_\theta(o_t|q,o_{<t})
\propto
\exp\left(
\frac{
\log \tilde{\pi}_\theta(o_t|q,o_{<t})
+\eta\log\pi_{\text{old}}(o_t|q,o_{<t})
}{\eta+1}
\right).
$$

这里的 $\eta^*$ 由一维 dual optimization 决定。把上式代回，可把 dual 写成：

$$
D(\eta)
=
-\eta\epsilon
-(\eta+1)\log
\sum_{o_t}
\exp\left(
\frac{
\log \tilde{\pi}_\theta(o_t|q,o_{<t})
+\eta\log\pi_{\text{old}}(o_t|q,o_{<t})
}{\eta+1}
\right),
$$

然后求：

$$
\eta^*=\arg\max_{\eta\ge 0}D(\eta).
$$

这个标量优化可以用几轮 bracketing / ternary search 很快求到。论文随后用 implicit differentiation 把 $\eta^*$ 对 $\tilde{\pi}_\theta$ 的梯度传回去，使整个投影可微。

### 稀疏化版本

由于完整 vocab 太大，TROLL 先稀疏化再投影：

- 取 top-$K$ token；
- 保留累计概率质量达到 $1-\delta$ 的 token；
- 强制保留 sampled token；
- 对丢弃 token 用小的 default mass $p_d$ 近似。

论文给出的 sparse KL 误差界可写成：

$$
\operatorname{KL}(p\|q)
\le
\gamma^{-1}\operatorname{KL}(p'\|q')
+\delta\log\frac{\delta}{q_{\min}}.
$$

默认 $K=64$、$\delta=10^{-5}$ 时，通常已经能覆盖约 `99.999%` 的概率质量。

## 12. 附：`adv_compute` 代码示意

TROLL 本身不改变 advantage 的定义。它只替换 policy update 里的 clipping / trust-region 部分。因此 `adv_compute` 可以继续沿用 GRPO、Dr.GRPO、PPO、GSPO 或 REINFORCE++ 的实现。

下面是一个 GRPO / RLVR 场景里常见的 outcome-only advantage 计算写法。它假设每个 prompt 采样 `group_size` 条 response，用 verifier / reward function 得到每条 response 的最终奖励，然后在同一个 prompt 内做 group normalization。

```python
import torch


def adv_compute(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    计算 GRPO 风格的 token-level advantage。

    Args:
        rewards:
            shape = [batch_size]。
            每条 response 的 outcome reward，例如数学题答对为 1，答错为 0。
            注意这里是一条 response 一个标量 reward，不是每个 token 一个 reward。

        response_mask:
            shape = [batch_size, response_len]。
            response token 的有效位置 mask。
            有效 token 为 1，padding token 为 0。

        group_size:
            每个 prompt 采样的 response 数量。
            GRPO 会在同一个 prompt 的 group 内比较好坏。

        eps:
            防止标准差为 0 的数值稳定项。

    Returns:
        advantages:
            shape = [batch_size, response_len]。
            每个有效 response token 共享同一条 response 的 normalized advantage。
    """
    batch_size = rewards.shape[0]
    assert batch_size % group_size == 0

    # 1. 把 batch 还原成 [num_prompts, group_size]。
    #    每一行对应同一个 prompt 采样出的多条 response。
    group_rewards = rewards.view(-1, group_size)

    # 2. 在同一个 prompt 的 group 内计算均值。
    #    这相当于给每个 prompt 构造一个局部 baseline。
    group_mean = group_rewards.mean(dim=1, keepdim=True)

    # 3. 在同一个 prompt 的 group 内计算标准差。
    #    标准差越大，说明这个 prompt 下不同 response 的区分度越强。
    group_std = group_rewards.std(dim=1, keepdim=True)

    # 4. 做 group normalization。
    #    reward 高于同组平均值，则 advantage 为正；
    #    reward 低于同组平均值，则 advantage 为负。
    normalized_adv = (group_rewards - group_mean) / (group_std + eps)

    # 5. 展平回 [batch_size]，每条 response 一个 advantage 标量。
    response_adv = normalized_adv.view(batch_size)

    # 6. 扩展到 token 维度。
    #    outcome-only RLVR 没有过程奖励，所以同一条 response 的所有有效 token
    #    使用同一个 advantage。
    advantages = response_adv[:, None] * response_mask

    return advantages
```

在 TROLL 里，这个 `advantages` 会进入类似下面的 policy objective：

```python
ratio = torch.exp(projected_logprob - old_logprob)
policy_loss = -(ratio * advantages * response_mask).sum() / response_mask.sum()
```

这里的关键区别是：

- PPO / GRPO + Clip 用的是 clipped ratio；
- GRPO + TROLL 用的是投影后的 `projected_logprob` 来算 ratio；
- advantage 本身不因为 TROLL 改变。

如果换成 PPO，`adv_compute` 通常不是上面的 group normalization，而是用 value model 和 GAE：

```python
delta_t = reward_t + gamma * value_{t+1} - value_t
adv_t = delta_t + gamma * lambda_ * adv_{t+1}
```

但 TROLL 仍然只作用在 policy update 上，不负责 value / advantage 的估计。

## 13. 附：TROLL 的 token-level KL 投影代码示意

下面这段才是 TROLL 的核心，不是 `adv_compute`。
它描述的是：在每个 token 位置上，如何把新策略分布 `new_logits` 投影回旧策略 `old_logits` 周围的 KL trust region。

```python
import torch


def sparse_support_from_probs(
    probs: torch.Tensor,
    top_k: int = 64,
    mass_threshold: float = 0.99999,
    sampled_token_id: int | None = None,
):
    """
    从完整 vocab 分布中选出稀疏支持集。

    Args:
        probs:
            shape = [vocab_size]，已经 softmax 后的概率。
        top_k:
            最多保留的 token 数。
        mass_threshold:
            累计概率质量阈值，例如 0.99999。
        sampled_token_id:
            本轮真正采样到的 token。必须保留，否则 sampled token 没有梯度。

    Returns:
        support_ids:
            保留下来的 token id。
    """
    # 1) 先按概率排序，取 top_k 候选。
    topk_probs, topk_ids = torch.topk(probs, k=min(top_k, probs.numel()))

    # 2) 取累计质量达到阈值的位置。
    cumsum = torch.cumsum(topk_probs, dim=0)
    keep_n = int((cumsum < mass_threshold).sum().item()) + 1
    keep_n = min(keep_n, topk_ids.numel())

    support_ids = topk_ids[:keep_n]

    # 3) 强制保留 sampled token。
    if sampled_token_id is not None and (support_ids == sampled_token_id).sum() == 0:
        support_ids = torch.cat(
            [support_ids, torch.tensor([sampled_token_id], device=probs.device)]
        )

    return torch.unique(support_ids)


def normalize_on_support(logits: torch.Tensor, support_ids: torch.Tensor):
    """
    只在 support 上做 softmax，得到稀疏化后的分布。
    """
    support_logits = logits[support_ids]
    support_log_probs = torch.log_softmax(support_logits, dim=0)
    support_probs = support_log_probs.exp()
    return support_log_probs, support_probs


def sparse_kl(
    p_log_probs: torch.Tensor,
    q_log_probs: torch.Tensor,
    p_probs: torch.Tensor,
) -> torch.Tensor:
    """
    在 support 上计算 KL(p || q) = sum p * (log p - log q)
    """
    return torch.sum(p_probs * (p_log_probs - q_log_probs))


def project_token_distribution(
    new_logits: torch.Tensor,
    old_logits: torch.Tensor,
    sampled_token_id: int,
    epsilon: float = 0.05,
    top_k: int = 64,
    mass_threshold: float = 0.99999,
    max_iter: int = 40,
):
    """
    对单个 token 位置做 TROLL projection。

    这个函数对应论文里的：
        min KL(pi || new_pi)
        s.t. KL(pi || old_pi) <= epsilon

    直观上，它做的是：
        1. 先看当前新分布和旧分布离得有多远；
        2. 如果还在 trust region 内，就直接用新分布；
        3. 如果超了，就把新分布往旧分布方向拉回去；
        4. 这个“拉回去”的结果不是硬截断，而是一个可微的投影分布。

    返回：
        projected_log_probs:
            投影后的 token 分布 log prob，只在 support 上有值。
        support_ids:
            稀疏支持集。
        kl_value:
            投影前或投影后的 KL 值，取决于 caller 怎么用。
    """
    # 1) 把 logits 变成概率。
    #    先做完整 softmax，是为了后面挑 support 时知道每个 token 的质量。
    new_probs = torch.softmax(new_logits, dim=-1)
    old_probs = torch.softmax(old_logits, dim=-1)

    # 2) 构造稀疏支持集。
    #    不直接在整个 vocab 上算 KL，因为 vocab 太大；
    #    先保留高概率 token、累计质量达到阈值的 token，
    #    再强制把 sampled token 也放进去，保证这个实际采样到的 token 仍然有梯度。
    support_ids = sparse_support_from_probs(
        new_probs,
        top_k=top_k,
        mass_threshold=mass_threshold,
        sampled_token_id=sampled_token_id,
    )

    # 3) 只在 support 上重新归一化。
    #    这样 support 上就形成了一个合法分布，后面可以直接算 KL。
    new_lp, new_p = normalize_on_support(new_logits, support_ids)
    old_lp, old_p = normalize_on_support(old_logits, support_ids)

    # 4) 先算投影前的 KL。
    #    如果新分布本来就没越界，就没有必要做 projection，直接返回即可。
    kl_before = sparse_kl(new_lp, old_lp, new_p)
    if kl_before <= epsilon:
        return new_lp, support_ids, kl_before

    # 5) 如果越界，就解一个标量 eta。
    #    eta 越大，投影结果越偏向 old policy；
    #    eta 越小，投影结果越接近 new policy。
    #    这就是论文里说的几何插值形式。
    def build_projected_log_probs(eta: torch.Tensor):
        # 投影后的 log-prob：
        #   log pi_proj(v) ∝ (log pi_new(v) + eta * log pi_old(v)) / (eta + 1)
        # 再减掉 logsumexp 做归一化，确保总概率和为 1。
        mixed = (new_lp + eta * old_lp) / (eta + 1.0)
        return mixed - torch.logsumexp(mixed, dim=0)

    def kl_after_eta(eta_value: float) -> torch.Tensor:
        # 给定一个 eta，先构造投影分布，再算它相对旧分布的 KL。
        eta = torch.tensor(eta_value, device=new_logits.device, dtype=new_logits.dtype)
        proj_lp = build_projected_log_probs(eta)
        proj_p = proj_lp.exp()
        return sparse_kl(proj_lp, old_lp, proj_p)

    # 6) 用一维搜索找 eta。
    #    目标是让投影后的 KL 尽量贴近 epsilon，同时保持 KL <= epsilon。
    left, right = 0.0, 1.0
    while kl_after_eta(right).item() > epsilon and right < 1e6:
        right *= 2.0

    for _ in range(max_iter):
        mid = (left + right) / 2.0
        kl_mid = kl_after_eta(mid).item()
        if kl_mid > epsilon:
            # KL 还太大，说明投影还不够保守，需要更靠近 old policy。
            left = mid
        else:
            # KL 已经压下来了，可以尝试减小 eta，让结果别离 new policy 太远。
            right = mid

    # 7) 用找到的 eta_star 构造最终投影分布。
    #    这个 projected_log_probs 就是后续拿去做 policy ratio 的分布。
    eta_star = torch.tensor(right, device=new_logits.device, dtype=new_logits.dtype)
    projected_log_probs = build_projected_log_probs(eta_star)
    projected_kl = sparse_kl(projected_log_probs, old_lp, projected_log_probs.exp())
    return projected_log_probs, support_ids, projected_kl
```

### 这段代码在做什么

1. `new_logits` 和 `old_logits` 分别对应更新后的新策略和采样时的旧策略。
2. 先做 sparse support，避免对整个 vocab 直接算 KL。
3. 在 support 上计算：

$$
\mathrm{KL}(p\|q)=\sum_{v\in S}p(v)\log\frac{p(v)}{q(v)}.
$$

4. 如果 `KL <= epsilon`，就直接用新分布。
5. 如果超了，就通过搜索 `eta` 找到投影后的分布。
6. 投影后的分布本质上是：

$$
\pi_{\text{proj}}
\propto
\exp\left(\frac{\log \pi_{\text{new}}+\eta^*\log\pi_{\text{old}}}{\eta^*+1}\right).
$$

### 一句实话

这段代码是按论文数学过程写的可读版，不是作者仓库的逐行原始实现；但它表达的就是 TROLL 的核心计算逻辑。
