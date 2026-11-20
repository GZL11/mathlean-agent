# MSRL：Multi-Stage RL for Generative Multimodal Reward Modeling 技术解读

本文档解读论文：

```text
MSRL: Scaling Generative Multimodal Reward Modeling via Multi-Stage Reinforcement Learning
https://arxiv.org/abs/2603.25108
CVPR 2026
```

论文研究的是 **multimodal reward model**，不是直接训练主策略模型。目标是训练一个可以判断多模态输出好坏的 generative reward model，用于视觉理解、图像生成、视频理解、视频生成等任务的 preference prediction / reranking。

## 一句话总结

MSRL 的核心是：

```text
多模态偏好数据很贵，不能靠大规模 multimodal RLVR 硬堆。

先用大量文本偏好数据训练 reward reasoning 能力，
再通过 caption-based RL 和 cross-modal KD 过渡到多模态，
最后只用少量 multimodal preference data 做适配。
```

三阶段：

```text
Stage 1:
  Textual RL
  在大规模文本偏好数据上学通用 reward reasoning。

Stage 2:
  Caption-Based RL + Experience Replay
  把图像/视频替换成 caption，用纯文本形式桥接 multimodal task gap。

Stage 2.5:
  Cross-Modal Knowledge Distillation, CMKD
  用 caption-trained MRM 生成 rationale，蒸馏到真实多模态输入下。

Stage 3:
  Multimodal RL
  用有限 multimodal preference data 做最终适配。
```

论文最核心的观点是：

```text
reward reasoning ability 可以先从 text preference 中学到，
再迁移到 multimodal preference modeling。
```

## 背景：MRM 是什么

MRM 即 multimodal reward model。给定输入 `x` 和候选输出 `y`，模型输出一个 reward / preference。

论文考虑两类 multimodal preference：

```text
Visual Understanding:
  输入 = 图像/视频 + query
  候选 = 两个文本回答 A/B
  目标 = 判断哪个回答更好

Visual Generation:
  输入 = 文本 caption / prompt
  候选 = 两张图像或两个视频 A/B
  目标 = 判断哪个生成结果更符合 prompt
```

四个具体任务：

```text
Image Understanding:
  (<Query>, <Image>, <Res A>, <Res B>)

Image Generation:
  (<Caption>, <Image A>, <Image B>)

Video Understanding:
  (<Query>, <Video>, <Res A>, <Res B>)

Video Generation:
  (<Caption>, <Video A>, <Video B>)
```

## Discriminative vs Generative MRM

### Discriminative MRM

传统 reward model 直接输出 scalar：

```text
r_phi(x, y)
```

对 preference pair `(y_a, y_b)`，用 Bradley-Terry loss：

```text
L_d
  = - E[ log sigmoid(r_phi(x, y_a) - r_phi(x, y_b)) ]
```

其中：

```text
y_a ≻ y_b
```

表示人类偏好 A。

### Generative MRM

Generative MRM 不直接输出 scalar，而是让 MLLM 生成 reasoning 和 preference token：

```text
<think>
  分析 A/B 哪个更好
</think>
<answer>
  A 或 B
</answer>
```

训练目标：

```text
L_g
  = - E[ log pi_phi(w = l | s) ]
```

其中：

```text
s = [prompt, x, y_a, y_b]
l = A 或 B
w = 模型生成的 preference token
```

MSRL 基于 generative MRM。

## RLVR for MRM

论文使用 RLVR 来提升 MRM 的 reward reasoning。由于 preference label 已知，可以构造 verifiable reward：

```text
r_format(s, o):
  输出格式正确则 1，否则 0。

r_accuracy(s, o):
  <answer> 中的 A/B 和 ground-truth preference 一致则 1，否则 0。
```

总 reward：

```text
r_v(s, o)
  = r_format(s, o) + r_accuracy(s, o)
```

RLVR 目标：

```text
L_RLVR
  = - E_{o ~ pi_theta} [ r_v(s, o) ]
    - beta * D_KL(pi_theta || pi_theta_old)
```

实际优化算法：

```text
GRPO
```

这里的 RL 不是训练 policy 去回答用户，而是训练 reward model 更准确地产生 preference reasoning 和 A/B 判断。

## 为什么需要 MSRL

问题：

```text
高质量 multimodal preference data 很贵。
```

直接做大规模 multimodal RLVR 受限于标注量。

MSRL 的假设：

```text
偏好判断中的“reward reasoning”有很大一部分是跨模态可迁移的。

例如：
  事实一致性、幻觉识别、回答完整性、语义对齐、风格匹配。
```

这些能力可以先从大量 text preference 学到，再迁移到 image/video preference。

## Stage 1：Textual RL

Stage 1 先训练纯文本 reward reasoning。

流程：

```text
1. SFT cold-start：
   用带 rationale 的文本偏好数据教模型输出 <think>/<answer> 格式。

2. Textual RLVR：
   在大规模文本偏好数据上用 GRPO 强化 preference prediction。
```

数据：

```text
SFT:
  40k rationale-based preference examples from HelpSteer3

RLVR:
  400k preference examples sampled from GRAM-R^2
```

注意：

```text
GRAM-R^2 虽然有 synthetic rationales，
但论文 Stage 1 RLVR 不使用这些 rationale，
只使用 preference label token 构造 reward。
```

Stage 1 时：

```text
freeze vision encoder
freeze projector
```

原因是：

```text
全部监督来自 text-only data，不需要更新视觉模块。
```

Stage 1 的作用：

```text
获得通用 reward reasoning capability。
```

消融显示 Stage 1 是最关键的部分。

## Stage 2：Caption-Based RL

Stage 2 解决 task gap。

做法：

```text
把原始 multimodal preference data 中的 image/video 替换成 caption。
```

于是训练样本变成纯文本形式，但保留 multimodal 任务语义：

```text
原始：
  image + query + response A/B

caption-based：
  image caption + query + response A/B
```

或：

```text
原始：
  text prompt + image A/B

caption-based：
  text prompt + caption of image A/B
```

论文使用 GPT-5 生成图像/视频 caption。

Stage 2 数据规模：

```text
caption-based preference examples:
  19,442
```

数据来源覆盖：

```text
IU:
  Image Understanding

IG:
  Image Generation

VU:
  Video Understanding

VG:
  Video Generation
```

Stage 2 继续使用 RLVR / GRPO，但额外加入 task recognition reward：

```text
r_task = 0.2 if predicted task tag is correct else 0
```

模型需要先输出任务类型标签，例如：

```text
<type>Image Understanding</type>
```

这个 reward 的目的：

```text
让 reward model 区分当前是在评估图像理解、图像生成、视频理解还是视频生成。
```

### Experience Replay

Stage 2 还加入 text preference replay，避免忘掉 Stage 1 学到的文本 reward reasoning。

主文写的是：

```text
new : previous data = 5 : 1
```

附录额外分析 caption:text mixing ratio，发现：

```text
caption:text = 4:1
  accuracy = 75.5，最好

caption:text = 5:1
  accuracy = 75.2
```

这里主文和附录表述略有不一致。可以理解为主实验使用了约 5:1 的混合设置，但附录 ratio sweep 显示 4:1 在 caption-based test set 上最佳。

## Stage 2.5：Cross-Modal Knowledge Distillation, CMKD

Stage 2 仍然是 caption/text 输入，还没有真正用视觉输入训练 reward reasoning。

CMKD 的目标：

```text
把 caption-conditioned reward reasoning 蒸馏到 multimodal input 条件下。
```

给定 multimodal preference sample `s` 和 caption `c`：

```text
s:
  原始图像/视频 preference sample

c:
  对图像/视频的 caption
```

先用 Stage 2 得到的 caption-trained MRM：

```text
pi_theta_text
```

采样 `n` 条 rationale：

```text
{o_1, ..., o_n} ~ pi_theta_text(. | s, c)
```

然后用三步筛选得到高质量 distilled rationale `o*`：

```text
1. 多数投票：
   pseudo label = mode({w_j})
   丢弃 predicted label != pseudo label 的 rationale。

2. 格式过滤：
   丢弃不满足 <think>/<answer> 格式的 rationale。

3. 置信度选择：
   在剩余 rationale 中选择 confidence 最高的一条。
```

最后用 `o*` 做 SFT：

```text
- log pi_theta([c, o*] | s)
```

论文还说，在之后的 multimodal RL stage 中，模型会先生成视觉 caption：

```text
<caption> ... </caption>
```

再做 reward reasoning：

```text
<think> ... </think>
<answer> A/B </answer>
```

这使模型在真实视觉输入下也能继承 caption-based reasoning 行为。

## Stage 3：Multimodal RL

Stage 3 使用真实 multimodal input 做最终适配。

数据规模：

```text
multimodal examples:
  20,038
```

输入不再替换为 caption，而是保留原始：

```text
image / video
```

仍然使用：

```text
RLVR + GRPO
```

并保留 task recognition reward。

论文强调：

```text
由于 Stage 1/2 已经建立 reward reasoning 和跨模态桥接，
Stage 3 只需要少量 multimodal data 就能有效适配。
```

## 训练设置

backbone：

```text
InternVL3.5
  1B, 4B, 8B, 14B
```

主实验重点报告 4B 和 8B。

MSRL GRPO 超参：

```text
framework:
  ms-swift

sampling size:
  8

learning rate:
  1e-6

batch size:
  128
```

附录补充：

```text
Stage 1 SFT:
  lr = 1e-5
  batch size = 128
  epochs = 3

Stage 1 RL:
  sampling = 8
  lr = 1e-6

Stage 2 Caption-Based RL:
  lr = 1e-6
  sampling = 8
  task recognition reward
  experience replay

Stage 3 / CMKD:
  appendix 写 lr = 2e-6, sampling = 8
```

注意：附录把 “Stage 3” 描述为 CMKD，但主文把 CMKD 放在 preference generalization 后、再接 multimodal RL。这里可以理解为实验实现中 CMKD 和最终多模态适配阶段紧密相连。

## Benchmark 和 Baseline

### Evaluation Benchmarks

Image Understanding：

```text
VL-RewardBench
Multimodal RewardBench
```

Image Generation：

```text
GenAI-Bench
```

Video Understanding：

```text
ShareGPTVideo
```

Video Generation：

```text
GenAI-Bench
VideoGen-RewardBench
```

### Baselines

MLLM-as-a-Judge：

```text
GPT-4o
GPT-5-mini
Gemini-1.5-Pro
Gemini-3-Pro
Claude-3.7-Sonnet
Qwen3-VL-8B-Instruct
InternVL3.5-8B / 14B
```

Open-source MRMs：

```text
MM-RLHF-Reward
R1-Reward
UnifiedReward
UnifiedReward-think
LLaVA-Critic
VisionReward
```

Same-data baselines：

```text
Discriminative MRM
Generative MRM
```

这些 same-data baseline 使用同样 backbone 和同样 multimodal preference data，用来公平比较 MSRL 的 staged textual RL 是否带来额外收益。

## 主要结果

### Image Understanding

InternVL3.5-8B，同样 multimodal preference data：

```text
VL-RewardBench Avg.

Discriminative MRM:
  64.3

Generative MRM:
  66.6

MSRL:
  75.9

MSRL + voting@16:
  77.5
```

Multimodal RewardBench Avg.：

```text
Discriminative MRM:
  74.9

Generative MRM:
  76.2

MSRL:
  80.5

MSRL + voting@16:
  81.3
```

论文摘要中的代表性提升：

```text
VL-RewardBench:
  66.6 -> 75.9

GenAI-Bench:
  70.2 -> 75.7
```

### Image / Video Generation and Video Understanding

InternVL3.5-8B：

```text
Generative MRM average:
  74.3

MSRL average:
  80.7

MSRL + voting@16 average:
  81.9
```

具体：

```text
Image Gen, GenAIBench:
  Generative MRM 70.2
  MSRL 75.7

Video Under, ShareGPT:
  Generative MRM 80.6
  MSRL 85.5

Video Gen, GenAIBench:
  Generative MRM 68.3
  MSRL 81.4

Video Gen, VideoGen:
  Generative MRM 78.2
  MSRL 80.1
```

## 消融实验

消融在：

```text
InternVL3.5-8B
VL-RewardBench
```

结果：

```text
Generative Baseline:
  Avg. 66.6

MSRL:
  Avg. 75.9

w/o Stage 1:
  Avg. 68.8

w/o Stage 2 Caption Data:
  Avg. 74.3

w/o Stage 2 CMKD:
  Avg. 73.4

w/o Stage 3:
  Avg. 72.6
```

结论：

```text
Stage 1 textual RL 最关键：
  移除后从 75.9 掉到 68.8。

Stage 2 caption data 有用：
  移除后掉到 74.3。

CMKD 有用：
  移除后掉到 73.4。

Stage 3 multimodal RL 有用：
  移除后掉到 72.6。
```

有意思的是：

```text
w/o Stage 3 仍然有 72.6，
明显高于 Generative Baseline 66.6。
```

这说明：

```text
大部分 reward reasoning 能力确实来自 Stage 1/2，
最终 multimodal RL 更多是适配和 refinement。
```

## Scaling 结论

论文展示了不同模型规模：

```text
1B, 4B, 8B, 14B
```

和不同 textual preference data 规模下的表现。

核心结论：

```text
1. 文本 RL 数据越多，MRM 性能越好。
2. 更大模型从 textual RL 中获益更明显。
3. 一旦 Stage 1/2 学到 reward reasoning，只需要少量 multimodal data 就能适配。
```

这就是论文标题里的：

```text
Scaling Generative Multimodal Reward Modeling
```

不是通过无限增加多模态偏好标注来 scale，而是通过：

```text
scale textual RL
```

来间接 scale MRM。

## 和前面几篇论文的关系

### 和 MaxRL

MaxRL 讨论的是：

```text
binary correctness RL 的 objective 应该怎么设计。
```

MSRL 使用的是：

```text
RLVR / GRPO
```

它关注的是训练数据和迁移路径：

```text
text preference -> caption preference -> multimodal preference
```

### 和 X-OPD

X-OPD 是：

```text
用 text teacher 对 speech student 的 on-policy trajectory 做 token-level supervision。
```

MSRL 是：

```text
用 textual preference RL 学 reward reasoning，
再迁移到 multimodal reward modeling。
```

二者都依赖跨模态迁移，但对象不同：

```text
X-OPD:
  policy model / speech LLM

MSRL:
  reward model / multimodal judge
```

### 和 TrOPD / Uni-OPD

TrOPD 和 Uni-OPD 都在处理 OPD 信号可靠性。

MSRL 不是 OPD 方法，它没有 teacher-student token-level OPD 信号。它是：

```text
multi-stage RLVR training recipe for generative MRMs。
```

## 局限

可以看到几个限制：

```text
1. 仍然需要少量 multimodal preference data 做 Stage 2/3。
2. caption 质量依赖 GPT-5，如果 caption 错，会影响迁移。
3. RLVR reward 仍然主要来自 A/B preference label，无法保证 rationale 本身完全正确。
4. 主要验证在 reward model accuracy 上，没有完整展示用该 MRM 反过来训练 policy 的收益。
```

## 总结

MSRL 的本质是一个 staged curriculum：

```text
Text preference RL:
  学 reward reasoning。

Caption-based RL:
  用 caption 桥接 multimodal task structure。

CMKD:
  把 caption reasoning 蒸馏到真实视觉输入。

Multimodal RL:
  用少量 multimodal preference data 做最终适配。
```

它的关键贡献不是新的 GRPO 公式，而是证明：

```text
多模态 reward modeling 的瓶颈不一定要靠更多 multimodal annotation 解决；
大规模 text preference RL 可以成为可扩展的训练轴。
```
