# X-OPD：Cross-Modal On-Policy Distillation 技术解读

本文档解读论文：

```text
X-OPD: Cross-Modal On-Policy Distillation for Capability Alignment in Speech LLMs
https://arxiv.org/abs/2603.24596
```

这篇论文把 OPD 从纯文本模型扩展到 Speech LLM 的跨模态能力对齐。核心问题是：端到端语音 LLM 虽然交互延迟更低，也能建模语调、情绪等声学信息，但在复杂指令、推理和知识类任务上，通常明显弱于对应的文本 LLM。

## Student / Teacher 模型速览

论文主实验：

```text
Student:
  Qwen3-Omni-A3B-Instruct

Teacher:
  Qwen3-A3B-Instruct
```

这里的 teacher 是文本 LLM，student 是语音/多模态 Omni 模型。消融中还测试了：

```text
Teacher:
  Qwen3-A22B-Instruct
```

一个关键结论是：更大的文本 teacher 不一定更适合。Qwen3-A22B-Instruct 文本能力更强，但在 X-OPD 中不如 Qwen3-A3B-Instruct，因为 A3B teacher 与 Qwen3-Omni-A3B-Instruct student 的语言侧规模和分布更匹配。

## 一句话总结

X-OPD 的核心是：

```text
让 Speech LLM student 自己生成 on-policy response，
再让更强或同源的 text LLM teacher 在这些 student trajectory 上给 token-level logprob 信号，
用 teacher/student logprob gap 构造 advantage，
通过 RL-style policy gradient 把文本模型的推理能力对齐到语音输入场景。
```

它不是普通 SFT，也不是离线 KD：

```text
SFT / Offline KD:
  训练目标来自固定答案或固定 teacher response。
  student 学的是离线轨迹，容易有 exposure bias。

X-OPD:
  response 来自 student 当前策略自己的 rollout。
  teacher 只评价 student 已经生成的 token。
  训练信号作用在 student 自己会访问到的分布上。
```

## 要解决的问题

论文认为 Speech LLM 的能力退化主要来自两个因素：

```text
1. 高质量 speech-reasoning 配对数据稀缺。
2. 连续声学表示和文本 LLM 的离散逻辑空间天然不完全对齐。
```

因此，文本 LLM 中常用的大量 SFT/RL 数据不能直接无损迁移到 Speech LLM。即使模型背后继承了强文本 LLM，经过语音模态接入和多模态训练后，也可能出现：

```text
speech input performance drop:
  语音输入下能力下降。

text input performance drop:
  即使输入文本，Speech LLM 版本也弱于原始文本模型。

catastrophic forgetting:
  对原有音频、音乐、环境声等能力产生遗忘。
```

X-OPD 的目标就是降低这种 alignment tax。

## 模型和数据设定

论文使用并行数据：

```text
D = { (S_i, T_i) }
```

其中：

```text
S_i:
  speech prompt，语音输入。

T_i:
  text prompt，与 S_i 语义严格一致的文本输入。
```

这种配对数据可以通过两种方式获得：

```text
1. 从文本 instruction 合成语音。
2. 对已有音频 prompt 做 ASR 转写。
```

论文实验中，训练数据来自：

```text
Tulu 3:
  10,934 pairs，136.7 hours。

NaturalReasoning:
  16,913 pairs，95.5 hours。
```

数据处理流程：

```text
1. 从 Tulu 3 / NaturalReasoning 采样文本 prompts。
2. 用 Gemini-3-Flash-Preview 改写成更口语化的 instruction。
3. 用 CosyVoice3 合成 24kHz 语音。
4. 用 SenseVoice 做 ASR back-translation。
5. 过滤 WER > 5% 的样本。
```

## Student / Teacher 设置

论文主实验：

```text
Student:
  Qwen3-Omni-A3B-Instruct

Teacher:
  Qwen3-A3B-Instruct
```

消融中还测试了：

```text
Teacher:
  Qwen3-A22B-Instruct
```

一个重要结论是：更大的 teacher 不一定更好。Qwen3-A22B-Instruct 虽然文本能力更强，但在 X-OPD 中不如 Qwen3-A3B-Instruct teacher。这里不是说 A22B 和 A3B 同规模，而是说 Qwen3-A3B-Instruct 作为 text teacher，与 Qwen3-Omni-A3B-Instruct student 的语言侧激活规模更匹配。论文解释为 capacity gap / knowledge gap 太大时，teacher 的偏好可能不够适合 student 对齐。

训练设置：

```text
framework:
  verl

optimization:
  RL-style policy optimization，pure on-policy sampling

trainable parameters:
  full-parameter training on language model backbone

frozen modules:
  audio tower
  modal adapter

optimizer:
  Adam

learning rate:
  2e-6

batch size:
  256

rollouts per prompt:
  n = 4
```

## X-OPD 训练流程

对每个配对样本 `(S, T)`，student 会在两个条件下生成或评估 response：

```text
speech-conditioned path:
  student 输入 speech prompt S。

text-conditioned path:
  student 输入 text prompt T。
```

teacher 始终是 text LLM，只接收文本 prompt `T`。

完整流程可以写成：

```text
1. 给定配对 prompt (S, T)。
2. student 当前策略 pi_theta 做 on-policy rollout，采样 m 条 response y_j。
3. 对每条 response 的每个 token y_{j,t}：
   - teacher 计算 log pi_phi(y_{j,t} | T, y_{j,<t})
   - student 计算 log pi_theta(y_{j,t} | T, y_{j,<t})
   - student 计算 log pi_theta(y_{j,t} | S, y_{j,<t})
4. 用 teacher/student logprob 差构造 advantage。
5. 用 policy ratio r(theta) * advantage 做 policy gradient 更新。
```

注意：teacher 不需要生成 response。teacher 是在 student 已经生成的 token 上打分。

## Tensor Shape 视角

假设：

```text
B:
  prompt batch size

m:
  每个 prompt 的 rollout 数量，论文设置 n=4

L:
  response token length，实际训练通常 padding 到 max_response_len

V:
  vocab size
```

rollout token：

```text
response_ids:
  shape = [B, m, L]
```

如果合并 batch 和 rollout 维度，常见实现会变成：

```text
response_ids:
  shape = [B * m, L]
```

teacher sampled-token logprob：

```text
teacher_log_probs:
  log pi_phi(y_t | T, y_<t)
  shape = [B * m, L]
```

student text-conditioned sampled-token logprob：

```text
student_text_log_probs:
  log pi_theta(y_t | T, y_<t)
  shape = [B * m, L]
```

student speech-conditioned sampled-token logprob：

```text
student_speech_log_probs:
  log pi_theta(y_t | S, y_<t)
  shape = [B * m, L]
```

old policy logprob：

```text
old_log_probs:
  log pi_old(y_t | condition, y_<t)
  shape = [B * m, L]
```

attention / response mask：

```text
response_mask:
  1 表示有效 response token
  0 表示 padding token
  shape = [B * m, L]
```

优势函数：

```text
A_im:
  shape = [B * m, L]

A_cm:
  shape = [B * m, L]
```

## In-Modal Advantage

in-modal advantage 用于稳定 student 在文本输入条件下的基础能力。

论文公式：

```text
A_im(y_t)
  = log pi_phi(y_t | T, y_<t)
    - log pi_theta(y_t | T, y_<t)
```

含义：

```text
teacher 在文本 prompt T 下认为 token y_t 更合理，
但 student 在同样文本 prompt T 下给的概率偏低，
则 A_im 为正，训练会提高 student 对该 token 的概率。
```

如果：

```text
teacher_logprob > student_text_logprob
```

则：

```text
A_im > 0
```

这个 token 对 student 来说是“teacher 更认可但 student 还没学好”的 token。

如果：

```text
teacher_logprob < student_text_logprob
```

则：

```text
A_im < 0
```

这个 token 是 student 比 teacher 更偏好的 token，policy gradient 会压低它。

## Cross-Modal Advantage

cross-modal advantage 用于把 teacher 的文本逻辑迁移到 speech-conditioned student 上。

论文公式：

```text
A_cm(y_t)
  = log pi_phi(y_t | T, y_<t)
    - log pi_theta(y_t | S, y_<t)
```

含义：

```text
teacher 在文本 prompt T 下评价 student rollout token；
student 在语音 prompt S 下生成或评估同一个 token；
二者 logprob 差表示“语音条件下 student 离文本 teacher 还有多远”。
```

这就是 X-OPD 的跨模态桥：

```text
T:
  teacher 可理解的文本语义输入。

S:
  student 实际需要处理的语音输入。

y:
  student 自己生成的 response trajectory。
```

因为 `(S, T)` 语义一致，所以 teacher 在 `T` 下对 `y_t` 的评价，可以作为 student 在 `S` 下的 token-level 训练信号。

## Policy Gradient Objective

论文把 loss 写成两个部分：

```text
L_im(theta)
  = E [
      1/m * sum_{j=1}^m
      1/|y_j| * sum_{t=1}^{|y_j|}
      r_{j,t}(theta) * A_im(y_{j,t})
    ]

L_cm(theta)
  = E [
      1/m * sum_{j=1}^m
      1/|y_j| * sum_{t=1}^{|y_j|}
      r_{j,t}(theta) * A_cm(y_{j,t})
    ]
```

其中：

```text
r_{j,t}(theta)
  = pi_theta(y_{j,t} | condition, y_{j,<t})
    / pi_old(y_{j,t} | condition, y_{j,<t})
```

更常见的 logprob 写法：

```text
r_{j,t}(theta)
  = exp(new_log_probs_{j,t} - old_log_probs_{j,t})
```

shape：

```text
new_log_probs:
  [B * m, L]

old_log_probs:
  [B * m, L]

ratio:
  [B * m, L]

advantage:
  [B * m, L]

token_loss:
  ratio * advantage
  [B * m, L]
```

最终目标：

```text
L(theta)
  = lambda * L_im(theta)
    + (1 - lambda) * L_cm(theta)
```

其中：

```text
lambda = 1:
  只做 text-conditioned in-modal OPD。

lambda = 0:
  只做 speech-conditioned cross-modal OPD。

lambda = 0.5:
  论文默认设置，同时对齐文本和语音能力。
```

实现时需要注意符号约定。论文公式写的是 policy gradient objective，但代码中的 loss 通常要最小化，所以经常会写成：

```text
pg_loss = - ratio * advantage
```

即：

```text
advantage > 0:
  增大当前 token 概率。

advantage < 0:
  减小当前 token 概率。
```

## 为什么需要 Multi-Sampling Rollout

论文认为单条 rollout 方差太高，因此每个 prompt 采样多条 response：

```text
m = 4
```

作用：

```text
1. 覆盖 student 当前 policy 下更多可能路径。
2. 缓解单个采样结果偶然性导致的梯度噪声。
3. 让 teacher 在更多 student 自己会生成的 token 上提供反馈。
```

这和 GRPO 多 sample 的直觉相近，但 X-OPD 的 advantage 不是 outcome reward group normalization，而是 teacher/student token logprob gap。

## 和普通 KL Distillation 的区别

普通 forward KL distillation 常见形式：

```text
D_KL(pi_teacher(. | x, y_<t) || pi_student(. | x, y_<t))
  = sum_v pi_teacher(v) * log(pi_teacher(v) / pi_student(v))
```

这需要 teacher/student 在每个 position 的 full vocab distribution。

X-OPD 论文公式更接近 sampled-token OPD：

```text
只在 student rollout 已经采样出的 token y_t 上计算：

log pi_teacher(y_t | T, y_<t)
- log pi_student(y_t | condition, y_<t)
```

因此它的监督信号不是 full distribution KL，而是对 sampled action 的 token-level advantage。

区别可以概括为：

```text
Full-logit KD / KL:
  比较整个 vocab 分布。
  dense supervision，但显存和计算更重。

X-OPD:
  只评价 student 实际采样 token。
  更 on-policy，更贴近 RL policy gradient。
```

## 和 Uni-OPD 的关系

Uni-OPD 的核心是：

```text
teacher 的 token-level OPD 信号可能和 outcome reward 不一致，
所以要用 outcome correctness 对 teacher supervision 做校准。
```

例如：

```text
correct trajectory 的 teacher score 反而低于 incorrect trajectory，
说明 teacher token-level preference 和最终正确性存在冲突。
```

Uni-OPD 用 margin shift、correct/incorrect balance 等机制修正这种冲突。

X-OPD 的核心是：

```text
把 OPD 用到 speech/text 跨模态对齐。
```

它关注的是：

```text
text teacher 如何给 speech student 提供 token-level 训练信号。
```

这篇论文没有引入 Uni-OPD 那种 outcome reward 校准，也没有讨论 correct / incorrect trajectory balance。它默认 teacher 在文本输入下的 token-level logprob 可以作为可靠的对齐信号。

## 和 OVD 的关系

OVD 使用的是 verbal score：

```text
teacher supervision = trajectory-level / step-level verbal score
```

例如让 teacher 输出 0-9 分，再用 rejection sampling / GRPO-style RL 训练。

X-OPD 使用的是 token-level logprob：

```text
teacher supervision = log pi_teacher(y_t | T, y_<t)
```

因此：

```text
OVD:
  更适合黑盒 teacher 或无法拿 logits/logprobs 的场景。
  监督更稀疏，但工程成本低。

X-OPD:
  需要 teacher 能计算 student token 的 logprob。
  监督更细粒度，能给每个 token credit assignment。
```

## 和 GOLD / GKD 的关系

GKD / GOLD 主要讨论纯文本 on-policy distillation：

```text
student 生成 response，
teacher 对 response 做 distribution-level 蒸馏。
```

GOLD 进一步解决跨 tokenizer 问题：

```text
teacher 和 student tokenizer 不同时，
通过 sequence alignment + vocab alignment / ULD 等方式蒸馏。
```

X-OPD 关注点不同：

```text
不是跨 tokenizer，
而是跨 input modality。
```

它假设 speech prompt `S` 和 text prompt `T` 语义一致，然后用文本 teacher 在 `T` 上评价 student 的 response token，从而给 speech-conditioned student 提供 advantage。

## 实验结果

论文主要比较了不同 Speech LLM 相对文本模型的能力掉点。

在 Qwen3-Omni-A3B-Instruct 上：

```text
Original Qwen3-Omni-A3B-Instruct:
  speech Avg. Drop = 11.29%
  text Avg. Drop   = 5.51%

X-OPD:
  speech Avg. Drop = 3.43%
  text Avg. Drop   = 0.97%
```

这说明 X-OPD 不只是提升 speech input，也同时修复了 Speech LLM 在 text input 下相对原始 text LLM 的能力退化。

论文还发现，使用同一批数据训练的 baseline 反而更差：

```text
SFT:
  speech Avg. Drop = 22.76%
  text Avg. Drop   = 17.49%

Offline KD:
  speech Avg. Drop = 19.62%
  text Avg. Drop   = 15.02%

GKD (Forward KL):
  speech Avg. Drop = 20.23%
  text Avg. Drop   = 15.81%

X-OPD:
  speech Avg. Drop = 3.43%
  text Avg. Drop   = 0.97%
```

论文的解释是：

```text
naive alignment 可能和 base model 内部 prior 冲突，
导致 alignment tax 变大。

X-OPD 因为在 student 当前策略分布上训练，
并且 token-level advantage 来自 text teacher，
所以更容易稳定地把文本逻辑能力迁移到语音条件下。
```

## 消融结果

### Teacher 大小

```text
Qwen3-A3B-Instruct teacher:
  speech Avg. Drop = 3.43%
  text Avg. Drop   = 0.97%

Qwen3-A22B-Instruct teacher:
  speech Avg. Drop = 5.55%
  text Avg. Drop   = 2.23%
```

结论：

```text
teacher 不是越大越好。
teacher/student 能力过远时，student 未必能有效吸收 teacher 的偏好。
```

### lambda

论文默认：

```text
lambda = 0.5
```

消融：

```text
lambda = 1.0:
  Text-only OPD，只使用 L_im。

lambda = 0.0:
  Speech-only OPD，只使用 L_cm。

lambda = 0.5:
  同时使用 L_im 和 L_cm。
```

结果显示：

```text
lambda = 0.5 最好。
```

论文认为 text objective 和 speech objective 有互补性：

```text
textual distillation 可以帮助 speech performance；
speech-focused distillation 也会改善 text-only scores。
```

## Catastrophic Forgetting

论文用 MMAR benchmark 评估灾难性遗忘。MMAR 包含 sound、music、speech reasoning，用于检查模型是否保留原有音频能力。

结果：

```text
Original Model:
  71.3

SFT:
  60.3，drop 11.0

Offline KD:
  59.9，drop 11.4

GKD:
  60.0，drop 11.3

X-OPD lambda=0:
  69.8，drop 1.5

X-OPD lambda=0.5:
  69.3，drop 2.0

X-OPD lambda=1:
  70.7，drop 0.6
```

这说明 X-OPD 在提升 speech/text 对齐的同时，对原有音频能力破坏更小。

一个合理解释是：

```text
SFT / Offline KD 强制模型拟合固定答案分布，
容易覆盖掉原有多模态能力。

X-OPD 在 student 自己生成的 trajectory 上做相对温和的 policy update，
且 audio tower / modal adapter 冻结，
因此对原始声学能力干扰更小。
```

## 关键局限

### 没有 outcome reward 校准

X-OPD 直接使用 teacher/student logprob gap 作为 advantage：

```text
A = teacher_logprob - student_logprob
```

但它没有检查：

```text
这个 trajectory 最终答案是否正确。
```

因此，如果 teacher 的 token-level preference 和最终正确性不一致，X-OPD 本身没有 Uni-OPD 那样的 margin shift / outcome calibration 机制。

### 依赖高质量 speech-text semantic invariance

X-OPD 要求：

```text
S_i 和 T_i 语义严格一致。
```

如果 TTS、ASR 或 prompt rewrite 引入语义偏差，那么 teacher 在 `T` 上给出的 token-level 信号就不一定适合 student 在 `S` 上学习。

### 需要 teacher logprob

X-OPD 需要：

```text
log pi_phi(y_t | T, y_<t)
```

也就是 teacher 至少要能对给定 token sequence 做 forward，并返回 sampled-token logprob。黑盒只返回文本答案的 teacher 不适合直接做 X-OPD。

### 论文没有给出完整工程实现

论文说明使用 verl 实现 RL-style policy optimization，但没有公开完整训练代码或给出每个张量在工程中的变量名。根据公式，可以对应到常见 RLHF/OPD 实现中的：

```text
new_log_probs:
  当前 student policy 对 rollout token 的 logprob。

old_log_probs:
  采样时 student policy 的 logprob。

teacher_log_probs:
  text teacher 对 student rollout token 的 logprob。

advantages:
  teacher_log_probs - student_log_probs。

ratio:
  exp(new_log_probs - old_log_probs)。

pg_loss:
  - ratio * advantages。
```

## 和当前 Uni-OPD 代码库的对应关系

当前代码库主要实现的是 Uni-OPD / OPD 相关训练技巧，不是 X-OPD 原论文代码。可以类比的模块包括：

```text
miles/miles/backends/training_utils/loss.py:
  policy loss、PPO/GRPO ratio、TIS/ICEPOP 等训练侧 loss 逻辑。

miles/Uni_OPD_utils/margin_calibration/margin_shift.py:
  Uni-OPD 的 margin shift，用 outcome correctness 修正 OPD advantage。

miles/Uni_OPD_utils/online_balance/rollout_sample_balance.py:
  在线 correct / incorrect trajectory balance。
```

如果要在当前代码库中实现 X-OPD，需要额外补充：

```text
1. speech/text paired prompt dataloader。
2. student speech-conditioned rollout。
3. student text-conditioned forward logprob。
4. text teacher forward logprob。
5. A_im / A_cm 两套 advantage 计算。
6. lambda 加权后的 policy loss。
```

## 最小伪代码

```python
# prompt batch:
# speech_inputs: [B, ...]
# text_inputs:   [B, T_prompt]

# 1. student on-policy rollout
# response_ids: [B * m, L]
# old_log_probs: [B * m, L]
response_ids, old_log_probs = student.rollout(
    speech_inputs,
    num_rollouts=m,
)

# 2. teacher evaluates student response under text prompt
# teacher_log_probs: [B * m, L]
teacher_log_probs = teacher.forward_log_probs(
    prompts=text_inputs.repeat_interleave(m, dim=0),
    responses=response_ids,
)

# 3. student text-conditioned logprob
# student_text_log_probs: [B * m, L]
student_text_log_probs = student.forward_log_probs(
    prompts=text_inputs.repeat_interleave(m, dim=0),
    responses=response_ids,
)

# 4. student speech-conditioned logprob
# student_speech_log_probs: [B * m, L]
student_speech_log_probs = student.forward_log_probs(
    prompts=speech_inputs.repeat_interleave(m, dim=0),
    responses=response_ids,
)

# 5. advantages
# A_im / A_cm: [B * m, L]
A_im = teacher_log_probs - student_text_log_probs.detach()
A_cm = teacher_log_probs - student_speech_log_probs.detach()
A = lambda_ * A_im + (1.0 - lambda_) * A_cm

# 6. current policy logprob for policy gradient
# new_log_probs: [B * m, L]
new_log_probs = student.forward_log_probs(
    prompts=speech_inputs.repeat_interleave(m, dim=0),
    responses=response_ids,
)

ratio = torch.exp(new_log_probs - old_log_probs)

# response_mask: [B * m, L]
pg_loss = -ratio * A
loss = masked_mean(pg_loss, response_mask)
```

上面伪代码只表达核心思想。真实实现里还需要处理：

```text
1. speech/text 两条路径是否分别计算 ratio。
2. PPO clip / KL penalty / entropy bonus 等稳定项。
3. padding mask、EOS mask、loss aggregation。
4. teacher/student tokenizer 是否一致。
5. rollout engine 和 training engine 的 logprob 一致性。
```

## 总结

X-OPD 的本质是：

```text
用 text teacher 的 token-level logprob，
给 speech student 的 on-policy trajectory 做 advantage shaping，
从而把文本模型的推理和指令能力迁移到语音输入条件下。
```

它的创新点不在于新的 RL 优化器，而在于把 OPD 的“student 自采样 + teacher 打 token-level 分”放到了跨模态语音对齐场景，并设计了：

```text
1. speech/text semantic-invariant paired prompts。
2. in-modal advantage A_im。
3. cross-modal advantage A_cm。
4. lambda 加权的双目标训练。
5. multi-sampling rollout 降低方差。
```

相比 SFT / Offline KD / GKD，它在论文实验中显著降低了 speech/text 双模态的能力掉点，并更好地保留原始音频能力。
