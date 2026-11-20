# Vision-OPD：用局部裁剪视角把细粒度视觉能力蒸馏到全图推理

本文档解读论文：

```text
Vision-OPD: Learning to See Fine Details for Multimodal LLMs via On-Policy Self-Distillation
https://arxiv.org/abs/2605.18740
```

## 一句话总结

这篇论文是一个 **多模态版 OPD**。

它的核心思想不是“再找一个更强外部 teacher”，而是：

```text
同一个多模态模型在 crop 视角下往往比在 full-image 视角下更会看细节；
那就把 crop 视角下的行为当成 privileged teacher，
再用 on-policy distillation 把这种能力迁移到 full-image policy 上。
```

## 论文要解决什么

细粒度视觉问答里，很多错误不是因为模型完全看不懂局部证据，而是因为：

```text
局部证据被全图上下文淹没了。
```

作者把这个现象叫做：

```text
regional-to-global perception gap
```

也就是：

```text
同一个模型在 crop 上回答正确，
但在 full image 上答错。
```

论文认为，这说明瓶颈不是纯粹的识别能力，而是：

```text
如何在全图里把相关证据“盯住”。
```

## 系统设置

论文把同一个 MLLM 拆成两个条件策略：

```text
teacher policy  = p_T(· | x_crop, q)
student policy  = p_S(· | x_global, q)
```

其中：

```text
x_crop   : 包含目标证据的裁剪图
x_global : 原始全图
q        : 视觉问题
```

这两个 policy 共享同一个底座模型参数，只是输入视角不同。

所以这里的 teacher 不是外部大模型，而是：

```text
同一个模型在“更有利的局部视角”下形成的 privileged behavior。
```

## 训练数据怎么来

作者不是人工标一大堆真实答案，而是用自动合成管线构造训练样本：

```text
(x, x', q)
```

其中：

```text
x  : 全图
x' : 从图中自动提取出来的 evidence-centered crop
q  : 问题
```

训练数据总量是：

```text
6.2K
```

论文还用一个较大的模型生成答案，并只保留达到高一致性的样本，用来给对比基线提供 ground-truth label。

## 方法主线

### 1. student 先在全图上 on-policy rollout

给定样本 `(x, x', q)`，student 先基于全图生成自己的回答轨迹：

```text
y ~ p_S(· | x, q)
```

注意这里是 on-policy：

```text
后续的监督都发生在 student 自己生成的轨迹上。
```

### 2. teacher 和 student 在同一条轨迹上重新算 next-token 分布

对 student 生成的每个前缀 `y_<n`，计算：

```text
p_S(· | x, q, y_<n)
p_T(· | x', q, y_<n)
```

然后最小化两者的逐 token divergence。

所以它不是让 teacher 重新生成整段答案，而是：

```text
在 student 自己走出来的轨迹上，比较两种视觉条件下的 token 分布。
```

## 损失函数

论文的目标是：

```text
L = E_{(x,x',q)} E_{y ~ p_S} [ D(p_T || p_S)(y | x, x', q) ]
```

直白一点就是：

```text
student 自己先走一步，
然后对每一步 token，逼近 teacher 在 crop 视角下的分布。
```

论文默认用 JSD：

```text
JSD_beta(p_T || p_S) = beta * KL(p_T || m) + (1 - beta) * KL(p_S || m)
m = beta * p_T + (1 - beta) * p_S
```

展开到词表上的计算是：

```text
KL(p_T || m)
= sum_v p_T(v | h_t^T) * log [ p_T(v | h_t^T) / m(v) ]

KL(p_S || m)
= sum_v p_S(v | h_t^S) * log [ p_S(v | h_t^S) / m(v) ]

m(v)
= beta * p_T(v | h_t^T) + (1 - beta) * p_S(v | h_t^S)
```

所以单个 prefix / token position 的 JSD 是：

```text
JSD_beta,t
= beta * sum_v p_T(v | h_t^T)
    * log [ p_T(v | h_t^T) / m(v) ]
  + (1 - beta) * sum_v p_S(v | h_t^S)
    * log [ p_S(v | h_t^S) / m(v) ]
```

整条 student response 上再做 token 平均：

```text
L_Vision-OPD
= E_{(x,x',q)} E_{y ~ p_S(.|x,q)}
  [ 1/T * sum_{t=1}^T JSD_beta,t ]
```

其中：

```text
h_t^S = (x, q, y_<t)   # full-image student prefix
h_t^T = (x', q, y_<t)  # crop-conditioned teacher prefix
```

实现上论文使用 top-K logits distillation 来近似全词表分布，而不是每一步都对完整词表求和。

实验里使用 `beta = 0.5`。

### 只用了 OPD loss 吗

是的，Vision-OPD 主方法只用 on-policy distribution matching / OPD-style distillation loss，不是 `RL loss + OPD loss`。

训练时 student 在 full image 上采样 response；teacher 在 crop image 条件下，对同一条 student response 的每个 prefix 给 next-token distribution；student 去匹配 teacher 的分布。

因此 GRPO、DAPO、RLVR 在这篇里是 baseline，不是 Vision-OPD loss 的组成项。Vision-OPD 没有额外的 environment reward、verifier reward、group advantage 或 policy-gradient loss。

更准确地说：

```text
Vision-OPD loss = on-policy token-level divergence loss
```

### 为什么用 JSD

论文比较了 Forward KL、Reverse KL 和 JSD，最终主实验用 `beta=0.5` 的 JSD。

这里的直觉是：Vision-OPD 不是典型“大 teacher 压小 student”的蒸馏，而是同一个模型在两种视觉条件下的 self-distillation。如果只用 FKL `KL(p_T || p_S)`，会更偏向覆盖 teacher 分布；如果只用 RKL `KL(p_S || p_T)`，会更偏向 mode-seeking。

JSD 把 teacher 和 student 都约束到混合分布 `m`，更对称、更有界，适合这种同模型、不同输入视角之间的信息迁移。论文消融里 JSD 的平均分也最高。

## 为什么一定要 on-policy

因为如果直接拿 teacher 轨迹喂给 student，会回到普通 SFT 的问题：

```text
训练分布和推理分布不一致
```

Vision-OPD 让 student 用自己的 rollout 当 prefix，目的就是：

```text
让训练时看到的状态分布更接近推理时的状态分布。
```

## 为什么一定要 dense token-level supervision

作者强调，细粒度视觉问题不能只给 sequence-level reward。

因为这样会出现一个问题：

```text
如果整条回答最终对了/错了，
中间每个 token 的信用分配都太粗。
```

Vision-OPD 直接用每个 token 的分布差异做学习信号，因此每个 token 都有梯度。

## teacher 为什么需要正则化

这篇论文里的 teacher 还是“同一个模型”，所以 teacher 和 student 容易一起漂。

如果 teacher 不约束，会出现：

```text
teacher 和 student 共同退化，训练崩掉。
```

作者对比了几种 teacher regularization：

```text
1. 不加正则
2. 冻结初始 teacher
3. trust-region regularization
4. EMA regularization
```

实验里 EMA 最稳，所以最终采用：

```text
EMA teacher
```

## 为什么用 top-K logits distillation

直接全词表蒸馏太贵。

所以作者只保留 top-K logits，再加上 tail probability term 来近似完整分布。

实验里用：

```text
K = 100
```

作者还比较了另一种简化版本：

```text
只在 student 采样到的 token 上做 per-token shaping
```

结果是 top-K logits distillation 更好，说明：

```text
更密的分布级监督，比只看采样 token 的标量 shaping 更有效。
```

## 和普通 OPD 的关系

普通 OPD 的典型形式是：

```text
teacher 和 student 是两个策略，
student 在自己的 rollout 上，
向 teacher 的 token 分布靠拢。
```

Vision-OPD 只是把这个框架搬到视觉场景里，并把 teacher 改成：

```text
同一模型的 crop-conditioned privileged policy
```

所以它本质上是：

```text
用“看得更清楚的局部视角”教“看全图的视角”。
```

## 实验结果

论文在两个层面上报告结果：

```text
1. 细粒度视觉理解 benchmark
2. 保留能力的 holdout benchmark
```

### Benchmark 是哪些

细粒度视觉理解任务：

| Benchmark | 关注点 |
|---|---|
| V* Bench | tiny visual targets 的定位和识别 |
| ZoomBench | 需要不同 zoom level 细节的问题 |
| HR-Bench 4K / 8K | 高分辨率视觉感知 |
| MME-RealWorld EN / CN | 真实世界高清照片理解，中英文两版 |

Holdout tasks：

| Benchmark | 用途 |
|---|---|
| MMVP | 通用多模态视觉偏差 / 对抗样本能力 |
| CV-Bench | 通用视觉理解 |
| MMStar | 通用多模态能力 |
| POPE | object hallucination / 泛化保持 |

### Baselines 是哪些

第一组是和 SOTA MLLM 对比：

```text
Thinking-with-Images agentic models:
DeepEyes, Thyme, DeepEyesV2, SenseNova-MARS

Closed-source single-forward models:
GPT-5.2, GPT-5.4, Gemini-3.1-Pro, Gemini-3.5-Flash

Open-source single-forward models:
Qwen3.5-4B/9B/397B, MiMo-VL-RL, Qwen3-VL-Instruct,
ZwZ, MiniCPM-V-4.5, GLM-4.6V, Kimi-K2.6
```

第二组是在同样数据和 backbone 下比较训练策略：

```text
Vanilla
SFT on Self-Teacher
GRPO
DAPO
OPSD
Vision-OPD
```

其中 SFT / RLVR / OPSD 的 ground-truth label 来自 Qwen3.5-397B 生成并经过多数一致性筛选；Vision-OPD 本身不需要 ground-truth label、reward verifier 或外部 teacher。

主要结论是：

```text
Vision-OPD 在细粒度任务上明显优于原始基线，
同时没有把原有通用视觉能力明显训坏。
```

表 2 显示，它在 Qwen3.5-4B 和 9B 上都优于 SFT、GRPO、DAPO、OPSD 等基线。

## 完整主结果：和 SOTA MLLMs 对比

单位是 accuracy (%)。

| Model | Param | V* | Zoom | HR-4K | HR-8K | MME-RW EN | MME-RW CN | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepEyes | 7B | 85.86 | 46.51 | 75.13 | 72.63 | 64.10 | 64.09 | 68.05 |
| Thyme | 7B | 82.20 | 45.09 | 77.00 | 72.00 | 64.80 | 64.59 | 67.61 |
| DeepEyesV2 | 7B | 81.68 | 44.97 | 77.88 | 73.75 | 64.90 | 65.07 | 68.04 |
| SenseNova-MARS | 8B | 92.15 | 47.81 | 83.13 | 78.38 | 67.90 | 68.90 | 73.05 |
| GPT-5.2 | - | 79.06 | 50.89 | 81.12 | 78.38 | 72.60 | 68.80 | 71.81 |
| GPT-5.4 | - | 76.96 | 52.66 | 84.00 | 77.88 | 74.20 | 70.93 | 72.77 |
| Gemini-3.1-Pro | - | 87.96 | 61.18 | 89.63 | 86.88 | 76.53 | 73.31 | 79.25 |
| Gemini-3.5-Flash | - | 89.01 | 61.42 | 89.12 | 86.62 | 75.31 | 73.97 | 79.24 |
| Qwen3.5 | 4B | 84.29 | 47.69 | 84.38 | 80.13 | 63.86 | 63.70 | 70.68 |
| Qwen3.5 | 9B | 82.72 | 52.07 | 85.75 | 80.63 | 71.40 | 67.67 | 73.37 |
| MiMo-VL-RL | 7B | 83.25 | 45.68 | 73.50 | 69.38 | 62.73 | 55.89 | 65.07 |
| Qwen3-VL-Instruct | 8B | 84.82 | 42.96 | 79.63 | 75.25 | 63.19 | 64.61 | 68.41 |
| ZwZ | 8B | 87.96 | 56.69 | 83.63 | 81.75 | 66.57 | 68.09 | 74.12 |
| MiniCPM-V-4.5 | 9B | 70.68 | 42.60 | 69.63 | 61.50 | 62.65 | 61.64 | 61.45 |
| GLM-4.6V | 106B | 86.91 | 50.06 | 82.13 | 78.88 | 65.57 | 65.62 | 71.53 |
| Qwen3-VL-Instruct | 235B | 91.10 | 56.09 | 86.13 | 80.38 | 71.74 | 69.04 | 75.75 |
| Qwen3.5 | 397B | 87.96 | 57.16 | 89.38 | 85.50 | 74.82 | 69.82 | 77.44 |
| Kimi-K2.6 | 1T | 88.48 | 53.14 | 81.88 | 78.00 | 69.22 | 66.13 | 72.81 |
| Vision-OPD | 4B | 92.15 | 59.76 | 84.50 | 80.38 | 74.88 | 70.76 | 77.07 |
| Vision-OPD | 9B | 94.76 | 65.80 | 88.13 | 85.50 | 73.40 | 70.46 | 79.68 |

主结论：

```text
1. Vision-OPD-9B 平均 79.68，是表中最高。
2. Vision-OPD-9B 超过 Qwen3.5-397B 的 77.44，也超过 GPT-5.4 的 72.77。
3. Vision-OPD-4B 平均 77.07，已经接近 / 超过很多大得多的开源模型。
4. Vision-OPD 不需要 inference-time crop / zoom tool，只需要 single forward pass。
```

## 完整主结果：训练策略对比

### Qwen3.5-4B

| Method | V* | Zoom | HR-4K | HR-8K | MMVP | CV-Bench | MMStar | POPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 84.29 | 47.69 | 84.38 | 80.13 | 76.67 | 87.13 | 78.53 | 88.28 |
| SFT on Self-Teacher | 78.01 | 54.67 | 79.75 | 76.38 | 78.33 | 85.70 | 68.40 | 87.48 |
| GRPO | 83.77 | 55.38 | 82.63 | 78.25 | 79.33 | 87.24 | 70.60 | 86.37 |
| DAPO | 84.82 | 55.74 | 84.00 | 78.00 | 79.33 | 86.95 | 72.27 | 86.62 |
| OPSD | 85.34 | 53.85 | 82.25 | 79.38 | 78.68 | 87.27 | 75.07 | 88.83 |
| Vision-OPD | 92.15 | 59.76 | 84.50 | 80.38 | 79.67 | 87.27 | 79.60 | 89.14 |

### Qwen3.5-9B

| Method | V* | Zoom | HR-4K | HR-8K | MMVP | CV-Bench | MMStar | POPE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 82.72 | 52.07 | 85.75 | 80.63 | 83.33 | 88.29 | 83.07 | 88.88 |
| SFT on Self-Teacher | 82.20 | 58.46 | 83.50 | 80.25 | 80.67 | 87.97 | 73.33 | 87.47 |
| GRPO | 85.34 | 57.51 | 86.88 | 83.25 | 81.67 | 87.78 | 73.40 | 87.78 |
| DAPO | 88.48 | 55.62 | 86.25 | 84.13 | 79.33 | 87.30 | 75.67 | 87.54 |
| OPSD | 89.53 | 57.51 | 84.00 | 81.25 | 80.67 | 87.45 | 79.13 | 87.48 |
| Vision-OPD | 94.76 | 65.80 | 88.13 | 85.50 | 83.67 | 88.40 | 83.20 | 89.13 |

结论：

```text
1. Vision-OPD 在细粒度任务上全面优于 SFT、GRPO、DAPO、OPSD。
2. SFT on Self-Teacher 虽然能提高 ZoomBench，但明显伤害 MMStar 等 holdout 能力。
3. GRPO / DAPO 使用 binary outcome reward，token-level credit assignment 弱。
4. OPSD 需要 ground-truth label 提供 reward signal；Vision-OPD 不需要 label / verifier。
5. Vision-OPD 在 holdout 上基本保持或略升，说明不是简单过拟合到细粒度训练集。
```

## 消融实验

### 1. Teacher regularization

基于 Qwen3.5-9B。Trust-region 和 EMA 的 update coefficient 都是 `alpha = 0.05`。

| Teacher | V* | Zoom | HR-4K | HR-8K | MME-RW EN | MME-RW CN | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current Policy | 0.00 | 0.00 | 0.00 | 0.00 | 3.51 | 0.05 | 0.59 |
| Initial Policy | 93.72 | 63.91 | 88.00 | 86.75 | 73.35 | 70.68 | 79.40 |
| Trust-Region Regularization | 93.19 | 63.79 | 88.25 | 86.25 | 72.96 | 70.86 | 79.22 |
| EMA Regularization | 94.76 | 65.80 | 88.13 | 85.50 | 73.40 | 70.46 | 79.68 |

结论：

```text
不加 regularization、直接用 current policy 当 teacher 会 collapse；
冻结 initial teacher 已经很强；
EMA 最好，所以主实验采用 EMA teacher。
```

这里的 collapse 很关键：Vision-OPD 虽然叫 self-distillation，但 teacher 不能和 student 完全同步，否则 teacher target 会跟着 student 漂，训练会失去稳定锚点。

### 2. Divergence objective

基于 Qwen3.5-9B。

| Objective | V* | Zoom | HR-4K | HR-8K | MME-RW EN | MME-RW CN | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Forward KL `KL(p_T || p_S)` | 93.19 | 64.02 | 87.88 | 84.13 | 72.83 | 70.17 | 78.70 |
| Reverse KL `KL(p_S || p_T)` | 90.05 | 62.72 | 87.88 | 86.00 | 74.60 | 69.92 | 78.53 |
| JSD `beta = 0.5` | 94.76 | 65.80 | 88.13 | 85.50 | 73.40 | 70.46 | 79.68 |

结论：

```text
JSD 平均分最高；FKL 略低，RKL 更低。
```

解释上，JSD 更适合这里的同模型、不同视觉条件 self-distillation；它不像 FKL 那样单向覆盖 teacher，也不像 RKL 那样强 mode-seeking。

### 3. On-policy generation length

基于 Qwen3.5-9B。

| Generation Length | V* | Zoom | HR-4K | HR-8K | MME-RW EN | MME-RW CN | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| 512 tokens | 91.62 | 64.38 | 88.00 | 86.13 | 72.15 | 69.44 | 78.62 |
| 1024 tokens | 94.76 | 65.80 | 88.13 | 85.50 | 73.40 | 70.46 | 79.68 |

结论：

```text
1024-token rollout 更好。
```

原因是 Vision-OPD 的监督发生在 token level：rollout 越长，可用于 teacher-student distribution matching 的 prefix / token 状态越多，dense supervision 更充分。

### 4. Top-K logits distillation vs sampled-token distillation

基于 Qwen3.5-9B。

| Method | V* | Zoom | HR-4K | HR-8K | MME-RW EN | MME-RW CN | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sampled-token distillation | 93.72 | 61.54 | 86.38 | 85.38 | 76.02 | 68.65 | 78.62 |
| Top-K logits distillation | 94.76 | 65.80 | 88.13 | 85.50 | 73.40 | 70.46 | 79.68 |

结论：

```text
Top-K logits distillation 平均更好。
```

sampled-token 版本只看 student 当前采样 token 的 teacher / student logprob ratio，本质更像 per-token scalar shaping；Top-K logits 版本比较的是局部分布，能提供更密的分布级 credit assignment。

## Regional-to-Global Gap 是否真的缩小

论文在训练中反复测试同一个问题：

```text
full image 输入时的准确率 vs crop 输入时的准确率。
```

二者差值就是 regional-to-global gap，较正式中文可以叫：

```text
区域到全图感知差距
```

训练前，不同模型在 ZoomBench 上都存在大约 `18-22` 个点的 gap；训练后，Vision-OPD 的 gap 持续下降。论文据此认为：

```text
模型不是学会了新的局部识别能力，
而是把原本需要 crop 才能触发的细粒度证据聚焦能力内化到了 full-image policy 中。
```

## 推理效率

Vision-OPD 训练时使用 crop-conditioned teacher，但推理时不使用 crop tool。

```text
inference = full image + question -> single forward pass
```

因此它和 DeepEyes / Thyme / SenseNova-MARS 这类 inference-time zoom / crop / search 的 agentic 方法不同：后者需要多步视觉工具调用，Vision-OPD 把这种 zoom benefit 蒸馏进模型参数里。

## 这篇论文真正强调的点

它的贡献不只是“又一个蒸馏方法”，而是这组判断：

```text
1. 模型往往已经会看局部证据
2. 失败主要发生在全图上下文干扰下
3. 所以可以把局部视角当 privileged supervision
4. 再用 on-policy distillation 迁移回全图推理
```

## 一句话结论

Vision-OPD 本质上是：

```text
把同一个多模态模型的“局部清晰视角”蒸馏到“全图推理视角”里，
用 on-policy + token-level distillation 解决细粒度视觉理解中的 regional-to-global gap。
```

