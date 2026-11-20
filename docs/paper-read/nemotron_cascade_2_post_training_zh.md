# Nemotron-Cascade 2 后训练全流程技术解读

本文档解读论文 **Nemotron-Cascade 2: Post-Training LLMs with Cascade RL and Multi-Domain On-Policy Distillation**。

论文地址：

```text
https://arxiv.org/abs/2603.19220
```

模型和数据：

```text
Model:
  https://huggingface.co/nvidia/Nemotron-Cascade-2-30B-A3B

SFT Data:
  https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-SFT-Data

RL Data:
  https://huggingface.co/datasets/nvidia/Nemotron-Cascade-2-RL-Data
```

## 一句话总结

Nemotron-Cascade 2 是一个 **30B 总参数、3B 激活参数的 MoE 模型**，核心不是提出一个单点 RL 算法，而是给出一套完整的后训练工程配方：

```text
高质量 SFT
  -> Cascade RL 按能力域串行强化
  -> Multi-domain On-Policy Distillation 做能力再平衡
  -> RLHF / Long-context RL / Code RL / SWE RL 继续专项增强
```

它的目标是解决一个大模型后训练中的常见矛盾：

```text
单个能力域 RL 能把该域做强，
但多轮跨域 RL 容易带来能力漂移、遗忘和 benchmark regression。
```

论文的回答是：

```text
1. 不把所有 RL 任务粗暴混在一起。
2. 按能力域进行 Cascade RL。
3. 对冲突小、验证成本接近的任务做 multi-domain RL。
4. 在中间插入 MOPD，从每个能力域最强 checkpoint 蒸馏回统一 student。
```

## 模型定位

论文模型：

```text
Nemotron-Cascade-2-30B-A3B
```

含义：

```text
30B:
  总参数量。

A3B:
  每 token 激活约 3B 参数。

MoE:
  mixture-of-experts 架构，靠稀疏激活提高单位计算量下的能力密度。
```

基础模型：

```text
Nemotron-3-Nano-30B-A3B-Base
```

论文主张该模型在很小激活参数下达到强 reasoning 和 agentic 能力，并在 IMO 2025、IOI 2025、ICPC World Finals 2025 上达到 gold-medal level。

## 总体训练流程

论文的后训练 pipeline 可以拆成：

```text
0. Base model
   Nemotron-3-Nano-30B-A3B-Base

1. SFT
   大规模多域监督微调，建立 reasoning、chat、instruction following、tool use、agentic、SWE 基础能力。

2. IF-RL
   instruction-following RL，优先把严格指令遵循能力拉起来。

3. Multi-domain RL
   把若干冲突小、验证成本接近的任务合并 RL。

4. MOPD
   从 Cascade RL 过程中不同能力域的最强 checkpoint 选 teacher，
   做 multi-domain on-policy distillation，修复 regression，统一能力。

5. RLHF
   用 generative reward model 做人类偏好对齐。

6. Long-context RL
   专门强化长上下文理解和长上下文 reasoning。

7. Code RL
   竞争编程专项 RL。

8. SWE RL
   软件工程任务 RL，包括 agentless repair 和 execution-based agentic SWE。
```

用图式表达：

```text
Base
  -> SFT
  -> IF-RL
  -> Multi-domain RL
  -> MOPD
  -> RLHF
  -> Long-context RL
  -> Code RL
  -> SWE RL
  -> Final model
```

## 为什么叫 Cascade RL

Cascade RL 的核心是：

```text
把不同能力域的 RL 按顺序串起来训练，而不是一次性混合所有任务。
```

论文给出的理由主要有三个：

```text
1. 降低工程复杂度
   每个阶段只处理相对同质的任务和 reward。

2. 减少跨域干扰
   不同任务的输出长度、验证成本、reward 类型差异很大，混训容易低效或互相伤害。

3. 便于针对每个能力域设置合适超参
   例如 response length、rollout size、overlong filtering、reward function、验证器都可以单独配置。
```

论文也强调，Cascade 的顺序不是通用常数，而是依赖模型行为和学习轨迹。它们选择当前顺序，是为了先建立 foundational priors，再进行专项能力 refinement。

## 严格 On-Policy GRPO

论文在 Cascade RL 主体中采用 GRPO，并强调严格 on-policy：

```text
每个 iteration:
  1. 用当前 policy 生成一组 rollouts
  2. 只做一次 gradient update
```

这样：

```text
采样 policy == 被更新 policy
importance sampling ratio = 1
```

这和很多 PPO 式多 epoch 复用 rollout batch 的做法不同。它牺牲了一些样本复用效率，但换来更稳定的 RL，尤其避免长链 reasoning RL 中 policy lag、entropy collapse 和 off-policy bias。

论文还去掉了 KL divergence 项：

```text
KL coefficient = 0
entropy coefficient = 0
```

GRPO objective 退化成：

```text
group-normalized reward + token-level policy gradient
```

这点和 Uni-OPD / Miles 代码里可以配置 PPO ratio、TIS、KL loss 的工程框架不同。Nemotron-Cascade 2 的论文主线更偏严格 on-policy。

## SFT 阶段

SFT 目标是建立基础能力，覆盖：

```text
math
code reasoning
science
long context
general chat
instruction following
safety
conversational agent
software engineering agent
terminal agent
```

训练框架：

```text
packed sequence length:
  up to 256K tokens

训练方式:
  单阶段混合 SFT

经验观察:
  约 1.5 epochs 达到较优效果
```

### Math SFT

主要包括：

```text
1.8M tool-calling math samples
2.6M non-tool competition math samples
816K mathematical proof / proof verification samples
```

teacher / 数据生成来源包括：

```text
DeepSeek-V3.2
DeepSeek-V3.2-Speciale
GPT-OSS-120B
```

### Code SFT

代码推理数据来自：

```text
OpenCode-Stage2
OpenCodeReasoning
HardTests
Codeforces / AtCoder / AIZU / CodeChef 等平台
```

处理方式：

```text
1. 严格去重
2. 对有 test cases 的 prompt 进行 correctness filtering
3. 对无 test cases 的 prompt 倾向选择更长、更完整 reasoning trace
```

规模：

```text
1.9M Python reasoning traces
1.0M C++14 reasoning traces
1.3M Python tool-calling reasoning traces
1.1M scientific coding samples
```

### General Chat / IF / Agentic SFT

general chat 同时包含 reasoning-on 和 reasoning-off 样本。

SWE SFT 同时使用：

```text
agentic trajectories
agentless repair / localization / test generation
```

论文中特别指出，将 agentless SWE 数据和 agentic 数据合并，比只训 agentic 数据能提升 SWE-bench Verified 表现。

Terminal agent SFT 使用 Terminal-Task-Gen，包含静态数据改造成终端任务，以及 seed-based / skill-based synthetic tasks。

## Cascade RL 阶段 1：IF-RL

目标：

```text
提升 strict instruction following。
```

为什么放第一阶段：

```text
1. IF-RL 会损害一些 human alignment benchmark。
2. 后续 RLHF 可以恢复 human preference alignment。
3. 早期 IF-RL 得到的 checkpoint 可以作为后续 MOPD 的 instruction-following teacher。
```

关键技巧：

```text
dynamic filtering:
  过滤掉同一 prompt 下 rollout 全对或全错的样本。

overlong penalty:
  对超过最大长度未完成的样本给 0 reward。

thinking mode only:
  只在 thinking mode 下训练，不额外引入 reward model。
```

超参：

```text
batch size:
  128 prompts

rollouts per prompt:
  16

temperature:
  1.0

top-p:
  1.0

learning rate:
  3e-6

KL / entropy:
  0

steps:
  约 180

max response length:
  49K

overlong filtering:
  False
```

## Cascade RL 阶段 2：Multi-domain RL

目标：

```text
联合强化一组互相冲突较小、验证成本相近的任务。
```

任务组成：

```text
55% MCQA / STEM
30% agentic tool calling
15% structured output
```

为什么可以混训：

```text
1. 论文观察到这些任务不会造成明显 benchmark degradation。
2. response length 和 verification time 接近，训练吞吐更稳定。
3. 能同时提升 MMLU-Pro、agent benchmark、IF-Bench 等指标。
```

超参：

```text
batch size:
  128

rollouts per prompt:
  16

temperature / top-p:
  1.0 / 1.0

steps:
  约 70

max response length:
  49K

overlong filtering:
  True
```

## Cascade RL 阶段 3：MOPD

MOPD 是本文最值得和 Uni-OPD 对照的部分。

全称：

```text
Multi-domain On-Policy Distillation
```

它解决的问题：

```text
Cascade RL 虽然比随机顺序的 sequential RL 稳定，
但训练环境越来越多之后，仍然会出现 capability drift 和 benchmark fluctuation。
```

典型冲突：

```text
RLVR:
  可能降低 entropy、缩短 reasoning trace，影响 math reasoning。

RLHF:
  可能和 instruction following 形成 trade-off。
```

MOPD 的做法：

```text
1. 从 Cascade RL 过程中挑出每个能力域的最强 checkpoint。
2. 把这些 checkpoint 组成 teacher pool。
3. 对每个训练样本选择对应 domain teacher。
4. 用 teacher 对 student 自己采样的 response token 做 token-level distillation advantage。
5. 在多个 domain 上把专长蒸馏回一个统一 student policy。
```

### Teacher 选择

论文使用三个 domain teacher：

```text
Math teacher:
  initial SFT checkpoint
  因为 SFT 已经有很强 math reasoning。

RLHF teacher:
  从 initial SFT 经过 RLHF 优化得到的 checkpoint。

Multi-domain teacher:
  从 IF-RL + Multi-domain RL 之后的 checkpoints 中选择。
```

这些 teacher 都来自同一个 SFT 初始化，所以：

```text
tokenizer 相同
vocabulary 相同
模型族相同
distribution shift 较小
```

### MOPD Objective

论文定义 token-level distillation advantage，直觉是：

```text
如果 domain teacher 比当前训练 policy 更支持 student 采样出来的 token，
这个 token 的 advantage 为正，训练鼓励 student 提高该 token 概率。
```

形式上就是类似 OPD 的：

```text
A_t^MOPD ≈ log pi_teacher(a_t | s_t) - log pi_train(a_t | s_t)
```

注意：

```text
只计算 student sampled token 的 logprob 差，
不需要对 full vocabulary 做完整 KL。
```

这和 Uni-OPD 的 token-level OPD 信号很接近：

```text
Uni-OPD / OPD:
  teacher_logprob - student_logprob

Nemotron MOPD:
  domain_teacher_logprob - train_policy_logprob
```

区别是：

```text
Nemotron-Cascade 2 的 MOPD 是多 domain teacher pool + Cascade RL 中间能力再平衡；
Uni-OPD 重点是 outcome reward 对 teacher token supervision 做 margin calibration。
```

### Train-Inference Mismatch

论文明确写到：response 由 inference engine 中的 student policy 采样，但优化在 training engine 中完成，因此需要处理 train-inference mismatch。

它使用 truncated importance weighting：

```text
w_t = clip(
  pi_train(a_t | s_t) / pi_infer(a_t | s_t),
  lower,
  upper
)
```

这和我们前面整理的 TIS / ICEPOP 语义一致：

```text
rollout / inference engine:
  生成 token 时的 policy

train engine:
  训练时重新 forward 的 policy

TIS:
  用 train / rollout 的 logprob ratio 修正 mismatch
```

论文中的主线不是 PPO old-policy IS，而是 train-inference mismatch correction。

### MOPD 超参

论文给出的设置：

```text
默认 rollout size:
  4

prompts per update:
  128

effective batch:
  512 responses

后续实验中:
  512 prompts * rollout size 1 更稳定，最终效果相近

warmup:
  前 30 optimization steps linear warm-up

convergence:
  通常 40-50 optimization steps

max response length:
  98K

steps:
  52

overlong filtering:
  False
```

论文强调 warm-up 很重要，因为训练初期 gradient norm 很大，warm-up 后迅速下降。

### MOPD 为什么高效

GRPO / RLVR 的 reward 通常是 sequence-level sparse reward：

```text
整条 response 一个 reward，
然后 broadcast 到 token-level loss。
```

MOPD 给的是 dense token-level advantage：

```text
每个 token 都有 teacher-student logprob difference。
```

因此 sample efficiency 和 step efficiency 更高。

论文例子：

```text
AIME25 math-only:
  GRPO 25 steps: 89.9 -> 91.0
  MOPD 30 steps: 89.9 -> 92.0

ArenaHard v2:
  MOPD 52 steps:
    Hard Prompt 71.5 -> 85.5
    Creative Writing 40.6 -> 71.0
```

## Cascade RL 阶段 4：RLHF

目标：

```text
恢复和提升 human preference alignment。
```

训练特点：

```text
thinking mode only
使用 generative reward model
```

论文说，相比同时训练 thinking / non-thinking，只训 thinking mode 虽然可能收敛慢一点或少一些局部收益，但能避免 instruction-following 明显退化。

超参：

```text
batch size:
  128

rollouts per prompt:
  16

temperature / top-p:
  1.0 / 1.0

max response length:
  16K

learning rate:
  3e-6

entropy coefficient:
  0

KL coefficient:
  0.03

steps:
  约 25-30

overlong filtering:
  True
```

注意：这是少数保留 KL 的阶段，用来保持其他 domain 能力。

## Cascade RL 阶段 5：Long-context RL

目标：

```text
增强长上下文理解和长上下文 reasoning。
```

训练数据：

```text
NVIDIA Nano-v3 RL data blend 中的 long-context 子集
```

reward / judge：

```text
Nemo-Gym RL environment
Qwen3-235B-A22B-Instruct-2507 作为 LLM judge
```

设置：

```text
input sequence limit:
  32K tokens

max sequence length:
  49K tokens

batch size:
  128

rollouts per prompt:
  16

temperature / top-p:
  1.0 / 1.0

steps:
  约 30

KL / entropy:
  0

overlong filtering:
  True
```

论文观察到，如果在这个阶段混入其他 domain，会伤害无关 benchmark，因此 long-context 单独成阶段。

## Cascade RL 阶段 6：Code RL

目标：

```text
提升竞争编程和长链代码 reasoning。
```

数据来源：

```text
Nemotron-Cascade coding corpus
AtCoder / Codeforces / AIZU 等平台
```

数据筛选：

```text
用 GPT-OSS-120B 过滤过于简单的问题：
  如果 8/8 rollouts 全部解对，则认为太简单，剔除。

最终保留:
  约 3.5K 高难样本
```

训练特点：

```text
strict binary reward
fully on-policy
```

为什么严格 binary reward：

```text
避免 reward hacking，
让代码训练稳定对齐真实 test case correctness。
```

超参：

```text
batch size:
  128

rollouts per prompt:
  16

max response length:
  118K

temperature:
  1.0

top-p:
  0.95

steps:
  22

overlong filtering:
  True
```

工程上，论文部署异步 reward verification server，用 384 CPU cores 完成大批量代码执行验证。

## Cascade RL 阶段 7：SWE RL

SWE RL 分两类：

```text
1. Agentless RL
2. Execution-based RL for agentic SWE scaffold
```

### Agentless RL

目标：

```text
提升 code repair 能力。
```

数据和 reward：

```text
数据来源:
  Nemotron-Cascade 1 的 agentless code repair 数据

reward model:
  GPT-OSS-120B 评估 patch 质量
```

训练细节：

```text
batch:
  128 prompts * 16 rollouts

max sequence length:
  98,304

temperature / top-p:
  1.0 / 1.0

难样本 mask:
  如果某 prompt 下没有 rollout reward > 0.5，则 mask loss
```

原因：

```text
全错的极难 prompt 会降低 agentless SWE RL 的稳定性和有效性。
```

论文观察到，agentless RL 不只提升 agentless repair，也能迁移到 OpenHands 这类 agentic scaffold。

### Execution-Based Agentic SWE RL

目标：

```text
让模型在真实软件工程 agent scaffold 中端到端优化完整 problem-solving trajectory。
```

训练环境：

```text
OpenHands framework
executable software environments
```

episode 内容：

```text
1. 读仓库
2. 搜索代码
3. 定位 bug
4. 修改文件
5. 运行测试
6. 根据测试反馈迭代
```

reward：

```text
编译结果
unit test outcome
patch correctness
```

超参：

```text
batch size:
  1024 responses
  = 16 prompts * 64 rollouts

max context length:
  256K tokens

max turns:
  200

training data:
  SWE-Gym
  R2E-Subset
```

这个阶段最像真实 agent RL：reward 来自环境执行，而不是静态答案匹配。

## 训练超参总览

论文附录给出的主要超参：

```text
SFT:
  packed sequence length: 256K
  warmup steps: 200
  scheduler: cosine
  max steps: 40,000
  actual training steps: 33,000
  optimizer: AdamW

IF-RL:
  max response length: 49K
  batch size: 128
  rollout size: 16
  steps: 180
  temp/top-p: 1.0/1.0
  overlong filtering: False

Multi-domain RL:
  max response length: 49K
  batch size: 128
  rollout size: 16
  steps: 70
  temp/top-p: 1.0/1.0
  overlong filtering: True

MOPD:
  max response length: 98K
  batch size: 128
  rollout size: 4
  steps: 52
  temp/top-p: 1.0/1.0
  overlong filtering: False

RLHF:
  max response length: 16K
  batch size: 128
  rollout size: 16
  steps: 25
  temp/top-p: 1.0/1.0
  overlong filtering: True

Long-context RL:
  max response length: 49K
  batch size: 128
  rollout size: 16
  steps: 30
  temp/top-p: 1.0/1.0
  overlong filtering: True

Code RL:
  max response length: 118K
  batch size: 128
  rollout size: 16
  steps: 22
  temp/top-p: 1.0/0.95
  overlong filtering: True
```

注意：arXiv HTML 对部分 learning rate 数值渲染不完整，正文中明确给出 IF-RL 和 RLHF 为 `3e-6`，其他阶段部分 LR 需要以 PDF 表格或 release 配置为准。

## 和 Uni-OPD 的关系

两者都属于 on-policy distillation，但关注点不同。

### 共同点

```text
1. 都使用 student 自己 rollout 的 trajectory。
2. 都用 teacher 对 student sampled token 做 logprob 级别监督。
3. 都不需要 full vocabulary KL，只需要 sampled token logprob。
4. 都把 teacher-student logprob difference 作为 dense token-level advantage。
5. 都关心 train-inference mismatch，并使用 importance weighting / clipping 类手段。
```

### 差异

```text
Uni-OPD:
  重点是 outcome-guided calibration。
  当 teacher token-level preference 和 outcome correctness 冲突时，
  用 margin shift / margin mask 修正 OPD advantage。

Nemotron-Cascade 2 MOPD:
  重点是 Cascade RL 中的 multi-domain capability re-balancing。
  从不同阶段选出 domain-best teacher，
  把 math / RLHF / multi-domain 等专长蒸馏回统一 student。
```

简化理解：

```text
Uni-OPD 解决:
  teacher token supervision 是否和 outcome reward 一致。

Nemotron MOPD 解决:
  多阶段 Cascade RL 后不同能力域如何统一且不互相遗忘。
```

## 和 Miles / 本仓库代码的关系

本仓库中 Qwen3-30B-A3B 的 GRPO 示例和论文有相似工程结构，但不是同一套训练配方。

相关脚本：

```text
miles/scripts/run_qwen3_30b_a3b.py
miles/scripts/run-qwen3-30B-A3B.sh
miles/scripts/models/qwen3-30B-A3B.sh
```

相似点：

```text
1. 30B-A3B MoE 训练。
2. GRPO / RLVR 风格训练。
3. rollout size > 1。
4. 使用 SGLang / rollout engine 生成。
5. 有 train-inference mismatch 相关机制可选。
```

区别：

```text
Nemotron-Cascade 2:
  是完整后训练 pipeline：
  SFT -> IF-RL -> Multi-domain RL -> MOPD -> RLHF -> Long-context RL -> Code RL -> SWE RL。

本仓库 Qwen3-30B-A3B 示例:
  主要是单个 GRPO 训练任务示例，
  例如 dapo-math-17k + deepscaler reward + AIME eval。
```

本仓库脚本中的关键设置：

```text
prompt data:
  dapo-math-17k

reward:
  deepscaler

rollout:
  n_samples_per_prompt = 8
  rollout_max_response_len = 8192
  rollout_temperature = 1

global batch:
  256

advantage estimator:
  grpo

KL:
  use-kl-loss, but kl-loss-coef = 0.00

optimizer:
  Adam
  lr = 1e-6
  weight_decay = 0.1
```

也就是说，Miles 脚本可以看作实现某个 math RL 阶段的工程骨架，而不是完整复现 Nemotron-Cascade 2 的 Cascade post-training。

## 关键方法理解

### 1. 为什么先 IF-RL

严格指令遵循是后续 agent、tool use、structured output 的基础能力。先把它练好，后续 RLHF 再恢复偏好对齐，比先做 RLHF 再训 IF 更符合论文观察。

### 2. 为什么 MOPD 放在中间

因为前两个 RL 阶段会把模型推向某些能力域，导致别的 benchmark 波动。MOPD 用多个 domain teacher 做 dense token-level 蒸馏，把已经学到的专长重新融合。

### 3. 为什么 Code RL 放得比较后

竞争编程需要长 reasoning、大 token budget 和昂贵验证。它是专项能力，不适合作为早期 foundational stage；放后面能在已有基础能力上做深度强化。

### 4. 为什么 SWE RL 最后

SWE RL 是最高复杂度的 agentic RL，包含多轮工具调用、仓库状态、测试反馈和长上下文。它的环境成本最高、轨迹最长、变量最多，因此适合作为最终专项 refinement。

### 5. 为什么强调严格 on-policy

长链 reasoning RL 中，policy drift 会放大训练不稳定。论文选择每组 rollout 只做一次更新，让 importance ratio 恒为 1，减少 PPO-style off-policy correction 的复杂度。

## 可复现性注意点

这篇论文比很多 post-training paper 更开放，因为 release 了：

```text
model checkpoint
SFT data
RL data
method details
```

但完整复现仍然非常重：

```text
1. SFT 数据规模巨大，packed sequence 长度到 256K。
2. RL 阶段多，顺序和 checkpoint selection 很关键。
3. Code RL 需要大规模 test execution。
4. SWE RL 需要真实 executable environment 和 OpenHands scaffold。
5. MOPD 需要维护多个 domain-best teacher checkpoint。
6. 长上下文和 agentic RL 对显存、调度、环境吞吐要求很高。
```

如果工程上借鉴，建议先复现最小闭环：

```text
1. 单域 GRPO / RLVR
2. dynamic filtering
3. train-inference mismatch logging / TIS
4. 单 teacher OPD / MOPD
5. 多 teacher domain routing
6. 再扩展到 Cascade RL
```

## 最终总结

Nemotron-Cascade 2 的核心贡献不是某一个新 loss，而是一套后训练系统设计：

```text
Cascade RL:
  用领域顺序降低多域 RL 干扰。

Multi-domain RL:
  对冲突小、验证成本接近的任务合并训练，提高效率。

MOPD:
  从中间最强 domain teacher 做 dense token-level on-policy distillation，
  修复 regression，统一多域能力。

Strict on-policy GRPO:
  每轮 rollout 只做一次 update，让 IS ratio = 1，换稳定性。

Agentic RL:
  把后训练扩展到真实工具调用、终端、软件工程和 executable environment。
```

和 Uni-OPD 对照看：

```text
Uni-OPD 更像“如何让 OPD 信号更可靠”；
Nemotron-Cascade 2 更像“如何把 OPD/MOPD 插入完整多域后训练流程里”。
```

