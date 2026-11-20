# Learning Dynamics of LLM Finetuning 论文解读

论文：

```text
Learning Dynamics of LLM Finetuning
```

链接：

```text
https://arxiv.org/abs/2407.10490
```

会议：

```text
ICLR 2025
```

## 一句话总结

这篇论文不是提出一个新的 SFT / DPO 损失，而是：

```text
用 learning dynamics 的视角解释 LLM 微调时，
一次梯度更新到底如何改变模型对其他 response 的概率。
```

它的核心发现是：

```text
SFT 不只会拉高被监督 response，也会间接拉高相似 response；
DPO 的负梯度会产生 squeezing effect，把概率质量挤到模型原本最自信的区域。
```

这给很多现象提供了解释：

```text
1. SFT 后某些 hallucination 会被放大；
2. off-policy DPO 中 chosen / rejected 乃至很多其他 response 的置信度都会下降；
3. on-policy DPO 往往比 off-policy DPO 更稳；
4. DPO 可能加剧重复、模板化或 prior bias。
```

## 背景：为什么要看 learning dynamics

很多 LLM 微调分析是从最终目标看问题：

```text
SFT 最大化标注答案似然；
DPO 最大化 chosen 相对 rejected 的偏好 margin；
RLHF / PPO 最大化 reward，同时受 KL 约束。
```

但这篇论文问的是另一个问题：

```text
当模型在一个样本上做一次梯度更新后，
它对另一个 prompt-response 的 log-likelihood 会怎么变？
```

也就是从：

```text
loss objective
```

转到：

```text
prediction evolution / learning dynamics
```

这个视角更适合解释训练中的反直觉曲线，例如：

```text
DPO 明明鼓励 chosen、惩罚 rejected，
为什么很多 response 的 likelihood 都一起下降？
```

## 核心分解：三项决定一次更新的影响

论文先从 softmax 分类问题出发，给出 one-step learning dynamics：

```text
Δ log π^t(y | x_o)
= - η · A^t(x_o) · K^t(x_o, x_u) · G^t(x_u, y_u)
  + O(η^2)
```

三项分别是：

```text
A^t: 当前输出分布带来的 softmax adaptation 项
K^t: eNTK，相当于 x_o 和 x_u 在模型梯度空间里的相似度
G^t: 由训练 loss 决定的 residual / energy / direction 项
```

可以直观翻译成：

```text
一次训练样本如何影响另一个样本，
取决于：
1. 另一个样本当前预测分布长什么样；
2. 两个输入/上下文在模型内部有多相似；
3. 当前 loss 给出的更新方向和强度。
```

其中最关键的是 `G^t`：

```text
不同 finetuning 方法的本质差异，
主要体现在 G^t 给模型什么方向、多少能量。
```

## 扩展到 LLM

LLM 的输出是序列，不是单个 label。论文把 prompt 和 response 拼成：

```text
χ = [x; y]
```

第 `l` 个 token 的预测为：

```text
π(y_l | x, y_<l)
```

通过 causal mask，可以把每个 token 位置看作一个 softmax 分类问题。于是 SFT 的 token 级分解为：

```text
[Δ log π^t(y | χ_o)]_m
= - Σ_l η · [A^t(χ_o)]_m
      · [K^t(χ_o, χ_u)]_{m,l}
      · [G^t(χ_u)]_l
  + O(η^2)
```

其中 SFT 的 residual 是：

```text
G^t_SFT(χ_u) = π_{θ^t}(y | χ_u) - y_u
```

这就是标准交叉熵的 softmax 梯度。

## SFT 学到了什么

SFT loss 是：

```text
L_SFT(x_u, y_u^+)
= - Σ_l log π(y_l^+ | x_u, y^+_<l)
```

直接效果很简单：

```text
把 y_u^+ 的概率拉高。
```

但论文强调一个更重要的间接效果：

```text
如果某个 response y' 和 y_u^+ 在模型梯度空间里相似，
即 K^t(χ_o, χ_u) 较大，
那么 SFT 也会间接拉高 y'。
```

所以 SFT 的影响不是只落在标注答案上，而是会扩散到：

```text
1. chosen response 的改写；
2. 语义不同但格式/表达风格相似的 response；
3. 训练集中其他问题的 preferred response；
4. 某些模型熟悉的模板化表达。
```

论文用 probing dataset 观察到：

```text
1. y_u^+ 的 likelihood 持续上升；
2. y_u^+ 的 rephrase 起初也会上升，之后随训练推进开始下降；
3. 完全随机词序 / 非自然句子的 likelihood 持续下降；
4. 其他问题的 preferred response 可能持续上升。
```

这解释了一类 SFT hallucination：

```text
模型可能把别的问题的高质量答案片段，
错误迁移到当前问题上。
```

## DPO 的分解

off-policy DPO loss 是：

```text
L_DPO(θ)
= - E log σ(
    β log [π_θ(y_u^+|χ_u^+) / π_ref(y_u^+|χ_u^+)]
    - β log [π_θ(y_u^-|χ_u^-) / π_ref(y_u^-|χ_u^-)]
  )
```

论文把 DPO 的 one-step dynamics 分成 chosen 和 rejected 两个方向：

```text
Δ log π^t(y | χ_o)
= - Σ_l η A^t(χ_o)
  (
    K^t(χ_o, χ_u^+) G^t_DPO+
    - K^t(χ_o, χ_u^-) G^t_DPO-
  )
  + O(η^2)
```

其中：

```text
G^t_DPO+ = β(1-a)(π_{θ^t}(y | χ_u^+) - y_u^+)
G^t_DPO- = β(1-a)(π_{θ^t}(y | χ_u^-) - y_u^-)
```

`a` 是 DPO margin 经过 sigmoid 后的值，表示当前 policy 相对 reference 已经多好地区分了 chosen / rejected。

直觉是：

```text
chosen 方向：像 SFT 一样拉高 y_u^+；
rejected 方向：施加一个大的负向压力，压低 y_u^-。
```

## Squeezing Effect：挤压效应

这是论文最重要的概念。

官方英文：

```text
squeezing effect
```

正式中文可以译为：

```text
挤压效应
```

含义是：

```text
当 softmax 输出上对一个 label / token 施加强负梯度时，
被压低的概率质量不会均匀分给所有其他 label，
而是主要被挤到更新前模型最自信的 label 上。
```

论文总结了几个性质：

```text
1. 被负梯度作用的 y^- 概率会下降。
2. 下降的概率质量会大量流向原本最高置信的 y*。
3. 高概率维度更容易继续升高，低概率维度更容易继续降低。
4. 当前分布越尖锐，挤压效应越强。
5. 如果 y^- 本来就是低概率区域，挤压效应更严重。
```

这能解释 off-policy DPO 的一个反直觉现象：

```text
chosen / rejected / rephrase / 其他观察 response 的 likelihood 都在下降，
但 greedy teacher-forcing response 的 likelihood 快速上升。
```

论文中的一个具体观察是：

```text
DPO 8 个 epoch 内，
teacher-forcing greedy response 的 log-likelihood 从 -113 上升到 -63；
而 SFT 中 y_u^+ 的 log-likelihood 从 -130 上升到 -90。
```

也就是说，DPO 中很多概率质量被挤到了模型原本最自信的路径，而这个路径不一定是 preferred response。

## 为什么 off-policy DPO 更容易出问题

off-policy DPO 的 chosen / rejected 通常来自：

```text
人类标注
外部模型
固定离线数据集
```

因此对当前 policy 来说，`y_u^+` 和 `y_u^-` 都可能位于低概率区域。尤其是 `y_u^-`：

```text
如果 rejected response 本来就是 current policy 很不可能生成的东西，
DPO 还要对它施加强负梯度，
就等于在 valley region 上强行往下压。
```

这会导致：

```text
概率质量被挤到模型自己原本最相信的输出；
模型 prior bias 被强化；
生成可能更重复、更模板化。
```

这也给 on-policy sampling 的好处提供了一个解释：

```text
on-policy 样本来自当前 policy，
通常不会处在特别深的低概率 valley；
因此负梯度造成的 squeezing effect 更弱。
```

## 论文提出的简单改进

作者提出一个非常反直觉但简单的方法：

```text
在 SFT 阶段，不只训练 (x, y_u^+)，
也把 (x, y_u^-) 加进 SFT 数据。
```

也就是先做 dataset extension：

```text
baseline:
  SFT on (x, y^+)
  then DPO on (x, y^+, y^-)

extend:
  SFT on both (x, y^+) and (x, y^-)
  then DPO on (x, y^+, y^-)
```

为什么这样有用？

```text
先把 y^- 从 valley region 拉高一些；
DPO 阶段再压 y^- 时，负梯度不再作用于极低概率区域；
因此 squeezing effect 被削弱。
```

这不是说 `y^-` 是好答案，而是：

```text
先避免在一个极低概率点施加强负梯度，
再让 DPO 去区分 y^+ 和 y^-。
```

## 实验设置

论文主要做的是诊断实验，而不是标准 leaderboard。

训练数据：

```text
Anthropic-HH
UltraFeedback
```

训练集采样：

```text
从 training split 随机选 5000 条
```

probing dataset：

```text
从训练集中随机选 500 条
```

模型：

```text
Pythia-410M / 1B / 1.4B / 2.8B
Qwen1.5-0.5B / 1.8B
```

观察对象包括：

```text
y_u^+              : preferred response
y_u^-              : rejected response
y_gpts^+ / y_gpts^-: 保持语义、改变格式的 ChatGPT rephrase
y_gptf^+ / y_gptf^-: 改变语义、保持格式的 ChatGPT rephrase
y_test^+           : 测试集中其他问题的 preferred response
y_hum              : ChatGPT 生成的随机英文句子
y_rnd              : 随机词序
y_urnd^+           : y_u^+ 词语打乱后的句子
```

指标主要是：

```text
不同 response 类型的平均 log-likelihood 曲线
```

## 改进方法的结果

作者用 Qwen1.5-1.8B 在 Anthropic-HH 上验证 dataset extension 方法。

比较对象：

```text
B: baseline，标准 SFT -> DPO
E: extend，扩展 SFT 数据 -> DPO
```

评价方式：

```text
从 Anthropic-HH test split 随机选 1000 个问题；
生成 response；
用 GPT3.5-Turbo 和 Claude3-Haiku 做 pairwise judge；
报告 extend 相对 baseline 的 win rate。
```

完整 win-rate 表：

| DPO epoch | ChatGPT judge | Claude judge |
|---:|---:|---:|
| 0 | 0.4729 | 0.4679 |
| 2 | 0.6518 | 0.5151 |
| 4 | 0.6928 | 0.6045 |
| 6 | 0.6667 | 0.5432 |

结论：

```text
1. DPO 前，extend 反而略弱，因为它在 SFT 阶段也学了 rejected response。
2. DPO 后，extend 明显强于 baseline。
3. 第 4 个 DPO epoch 最好，ChatGPT / Claude judge 都给出最高 win rate。
4. 这支持作者的解释：缓解 squeezing effect 后，DPO alignment 更有效。
```

附录还报告：

```text
B_{2-4} vs B_1: GPT 0.6727, Claude 0.6411
E_{2-4} vs E_1: GPT 0.6898, Claude 0.7321
E_{2-4} vs E_{2-2}: GPT 0.6853, Claude 0.5517
E_{2-4} vs E_{2-6}: GPT 0.6324, Claude 0.5316
```

这说明：

```text
DPO 本身有效；
extend + DPO 也有效；
但 DPO epoch 不是越多越好，4 epoch 在该设置下更优。
```

## 这篇和 OPD / on-policy 的关系

这篇不是 OPD 论文，但对 OPD / on-policy preference tuning 很有启发。

它提供了一个解释：

```text
off-policy 的问题不只是 distribution shift，
还包括负梯度打在 current policy 的低概率 valley 上，
从而触发 squeezing effect。
```

因此 on-policy 的价值可以理解为：

```text
让训练样本位于 current policy 更自然的支持集上，
避免对极低概率输出施加强负梯度。
```

这和 OPD 里反复强调的：

```text
student prefix / student distribution 下做 teacher signal
```

是同一类思想：

```text
训练信号必须落在当前模型真的会访问到的区域，
否则梯度方向和概率质量迁移都可能变得很奇怪。
```

## 局限

1. 理论分解依赖一阶近似和 eNTK 相对稳定假设。
2. squeezing effect 的严格证明主要在 logistic regression / softmax head 场景，LLM 中是通过类比和实验证据支撑。
3. 论文的改进方法很简单，但不是完整的最优 DPO 算法。
4. 主要指标是 log-likelihood 动态和 LLM judge win-rate，不是大规模能力 benchmark。

## 一句话结论

这篇论文的价值在于：

```text
它把 SFT / DPO 的效果从“目标函数想优化什么”，
转化为“梯度一步步把概率质量搬到哪里”。
```

最值得记住的是：

```text
DPO 的负梯度如果打在低概率 rejected response 上，
会把概率质量挤到模型原本最自信的输出，
这就是 squeezing effect。
```

## BibTeX

```bibtex
@misc{ren2024learningdynamicsllmfinetuning,
  title={Learning Dynamics of LLM Finetuning},
  author={Ren, Jie and Zhao, Qipeng and Li, Han and Liu, Dongrui and Liu, Ting and Zhou, Sihang and Zhang, Junxian and Qiu, Xipeng},
  year={2024},
  eprint={2407.10490},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2407.10490}
}
```

