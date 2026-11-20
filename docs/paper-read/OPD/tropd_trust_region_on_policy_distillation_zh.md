# TrOPD：Trust Region On-Policy Distillation 技术解读

本文档解读论文：

```text
Trust Region On-Policy Distillation
https://arxiv.org/abs/2606.01249
version: v3, 2026-06-17
```

## Student / Teacher 模型速览

TrOPD 的 teacher 是比 student 更强或经过 RL 的 reasoning teacher。论文主要设置：

```text
数学单域:
  Student:
    DeepSeek-Distilled-Qwen-1.5B

  Teacher:
    Skywork-OR1-Math-7B

多域 reasoning:
  Student:
    DeepSeek-Distilled-Qwen-1.5B

  Teacher:
    Skywork-OR1-7B

Qwen3 多任务:
  Student:
    Qwen3-SFT-1.7B

  Teacher:
    Qwen3-Nemotron-4B
```

所以 TrOPD 讨论的是 **teacher/student 分布差距较大时，teacher 的 sampled-token 监督是否可信**，不是 outcome reward 校准，也不是跨模态对齐。

论文目标是解决 reasoning-oriented OPD 中的训练不稳定问题：当 teacher 和 student 分布差距较大时，student 自己生成的 token 可能落在 teacher 低置信度区域，直接用 token-level reverse-KL reward 会产生极端负梯度，导致训练不稳甚至崩掉。

## 一句话总结

TrOPD 的核心是：

```text
不要对所有 student-generated token 都无脑做 OPD。

先判断这个 token 是否落在 teacher 可验证的 trust region：
  - trust region 内：继续用 on-policy RKL / K1 estimator。
  - trust region 外：不要强行用 sampled-token RKL，改用 mask / clip / top-k FKL 处理 outlier。
  - 另外加入 teacher prefix off-policy guidance，让 student 从 teacher 支持的轨迹附近开始探索。
```

它关注的是：

```text
OPD token-level credit assignment 的可靠性。
```

而不是像 Uni-OPD 那样用 outcome correctness 校准 teacher signal，也不是像 X-OPD 那样处理 speech/text 跨模态对齐。

## 背景：为什么普通 OPD 会不稳定

普通 on-policy distillation 让 student 自己生成 response：

```text
x ~ pi_S
```

然后 teacher 在 student token 上打分，常见 reverse KL objective：

```text
D_KL(pi_S || pi_T)
  = E_{x ~ pi_S} [ log pi_S(x) / pi_T(x) ]
```

最大化负 RKL：

```text
J_KD
  = - D_KL(pi_S || pi_T)
  = - E_{x ~ pi_S} [ log pi_S(x) / pi_T(x) ]
```

对 sampled token `x`，token-level reward / advantage 可写成：

```text
log pi_T(x) / pi_S(x)
  = log pi_T(x) - log pi_S(x)
```

如果 teacher 对 student 采样 token 极不认可：

```text
pi_T(x) << pi_S(x)
```

则：

```text
log pi_T(x) / pi_S(x) << 0
```

这个 token 会带来很强的负向 policy-gradient 信号。论文认为，当 teacher/student 分布差距很大时，这种信号不一定可靠，因为 token 已经落在 teacher 几乎不覆盖的区域，teacher 的 sampled-token 监督可能变成 outlier。

## K1 Estimator 是什么

full-vocabulary RKL 需要在每个位置计算整个词表：

```text
sum_{v in V} pi_S(v) log pi_S(v) / pi_T(v)
```

这对长 reasoning response 很贵：

```text
memory roughly O(n * |V|)
```

其中：

```text
n:
  sequence length

|V|:
  vocab size
```

reasoning 模型常有几千到几万 token 的 response，因此 full-vocab KD 成本很高。OPD 论文和这篇 TrOPD 都使用 sampled-token 的 K1 estimator：

```text
x ~ pi_S

D_KL(pi_S || pi_T)
  = E_{x ~ pi_S} [ log pi_S(x) / pi_T(x) ]
```

实际只需要 student 采出来的 token logprob：

```text
student_logprob:
  log pi_S(x)

teacher_logprob:
  log pi_T(x)

rkl_reward:
  log pi_T(x) - log pi_S(x)
```

memory 变成：

```text
O(n)
```

问题是：K1 在分布 mismatch 很大时方差和极端梯度会变严重。

## RKL / FKL 的“视角”到底是什么意思

这里的“学生视角”和“教师视角”不是指整条 trajectory 一定来自谁，而是指 KL 里面的期望/加权分布来自谁。

Reverse KL：

```text
D_KL(pi_S || pi_T)
  = E_{x ~ pi_S} [
      log pi_S(x) / pi_T(x)
    ]
```

它的 token sample 来自 student：

```text
x ~ pi_S
```

所以它问的是：

```text
student 会生成哪些 token？
这些 student token 在 teacher 看来概率高不高？
```

因此 RKL 是 student perspective。它更像：

```text
学生先写一个 token，老师判断这个 token 是否接受。
```

Forward KL：

```text
D_KL(pi_T || pi_S)
  = E_{x ~ pi_T} [
      log pi_T(x) / pi_S(x)
    ]
```

如果写成 full distribution：

```text
D_KL(pi_T || pi_S)
  = sum_v pi_T(v) log pi_T(v) / pi_S(v)
```

它由 teacher 分布加权，所以它问的是：

```text
teacher 认为哪些 token 应该有高概率？
student 有没有覆盖这些 teacher-supported token？
```

因此 FKL 是 teacher perspective。它更像：

```text
老师给出候选方向，学生去覆盖这些方向。
```

需要特别注意：

```text
在 TrOPD 的 on-policy outlier FKL 中，
trajectory/context 仍然来自 student rollout，
只是 token-level supervision 换成 teacher top-k distribution。
```

也就是说，在第 `t` 个位置：

```text
c_t = (prompt, y_1, ..., y_{t-1})
```

这个 context `c_t` 是 student rollout prefix。但 RKL 和 FKL 使用的 token 对象不同：

```text
RKL / K1:
  context = student rollout prefix c_t
  supervised token = student sampled token y_t
  question = teacher 是否接受 y_t

outlier FKL:
  context = student rollout prefix c_t
  supervised tokens = teacher top-k tokens under c_t
  question = student 是否覆盖 teacher 在 c_t 下支持的 token
```

因此，不能把“FKL 是教师视角”简单理解成“FKL 一定使用 teacher rollout trajectory”。在 TrOPD 中，只有 teacher prefix guidance 那部分才是真正的 teacher-generated trajectory。

## TrOPD 的三个组成部分

TrOPD 包含三块：

```text
1. Trust-Region On-Policy Learning
   对 teacher 认可的 student token 使用 RKL / K1 estimator。

2. Outlier Estimation
   对 teacher 不认可的 outlier token，不直接用 RKL；
   可以 mask、clip，最好是用 teacher top-k FKL。

3. Off-Policy Guidance
   用 teacher-generated prefix 引导 student 从更高质量轨迹附近继续 on-policy rollout。
```

论文统一 objective：

```text
J_TrOPD(x)
  =
  - I[x ~ pi_S] * M_bar_x
    * sum_{v in V_T^k} pi_T(v) log pi_T(v) / pi_S(v)

  - I[x ~ pi_S] * M_x
    * log pi_S(x) / pi_T(x)

  - beta * I[x ~ pi_T]
    * log pi_T(x) / pi_S(x)
```

其中：

```text
M_x:
  trust region mask。

M_bar_x:
  outlier mask，等于 1 - M_x。

V_T^k:
  teacher top-k vocabulary。
```

因为论文写的是要最大化的 objective，工程实现里如果用最小化 loss，通常要加负号。

## Trust Region 如何判定

对 student 采样出的 token：

```text
x ~ pi_S
```

论文定义 token 落入 trust region 的概率：

```text
P_trust(x)
  = min( pi_T(x) / pi_S(x), 1 )
```

然后：

```text
M_x ~ Bernoulli(P_trust(x))
```

也就是说：

```text
如果 teacher 对这个 token 的概率 >= student：
  pi_T(x) >= pi_S(x)
  P_trust = 1
  这个 token 必定进入 trust region。

如果 teacher 对这个 token 的概率 < student：
  pi_T(x) < pi_S(x)
  P_trust = pi_T(x) / pi_S(x)
  按比例随机接受。
```

用 logprob 写：

```text
ratio_TS = exp(teacher_logprob - student_logprob)
P_trust = min(ratio_TS, 1)
```

tensor shape：

```text
teacher_log_probs:
  [B * G, L]

student_log_probs:
  [B * G, L]

ratio_TS:
  [B * G, L]

trust_prob:
  [B * G, L]

trust_mask:
  [B * G, L]
```

其中：

```text
B:
  prompt batch size

G:
  每个 prompt 的 rollout 数量，论文设置 4

L:
  response token length
```

这个设计来自 speculative decoding 的接受概率直觉：

```text
accept probability ~= min(pi_teacher(x) / pi_student(x), 1)
```

如果 teacher 会接受 student decode 的 token，则认为它处于 teacher 可验证区域，RKL/K1 信号更可信。

## Trust Region 内：继续用 RKL / K1

trust region 内使用：

```text
- KL(pi_S || pi_T)
```

sampled-token estimator：

```text
J_trust(x)
  = - log pi_S(x) / pi_T(x)
  = log pi_T(x) - log pi_S(x)
```

乘上 mask：

```text
J_trust(x)
  = M_x * (log pi_T(x) - log pi_S(x))
```

直觉：

```text
teacher 比 student 更认可的 token：
  log pi_T - log pi_S > 0
  增强 token 概率。

teacher 比 student 更不认可的 token：
  log pi_T - log pi_S < 0
  降低 token 概率。
```

但这个 RKL 信号只在 trust region 使用。

## Outlier 区域：为什么不用 RKL

outlier token 满足：

```text
M_x = 0
```

这通常意味着：

```text
pi_T(x) / pi_S(x) 较小，
teacher 对这个 student token 低置信。
```

如果继续使用：

```text
log pi_T(x) - log pi_S(x)
```

可能出现极端负 reward，使 policy gradient 出现 outlier。

论文比较了三类 outlier 处理：

```text
1. Mask Outlier
   outlier 区域直接不给梯度。

2. Clip Outlier
   把极端 reward clip 到阈值 tau。

3. FKL Outlier
   对 outlier 区域不用 sampled token RKL，
   改用 teacher top-k forward KL。
```

实验显示，单纯 mask/clip 都能比 vanilla OPD 稳定，但 top-k FKL outlier 效果最好。

## Outlier 区域：Top-k FKL

outlier 区域仍然处在 student rollout trajectory 上。设第 `t` 个位置的上下文为：

```text
c_t = (prompt, y_1, ..., y_{t-1})
```

这里的 `y_<t` 来自 student 之前已经生成的 token。因此 outlier FKL 不是 teacher rollout，而是：

```text
student trajectory context
+ teacher top-k token distribution
```

forward KL：

```text
D_KL(pi_T || pi_S)
  = sum_{v in V} pi_T(v) log pi_T(v) / pi_S(v)
```

TrOPD 在 outlier 区域只用 teacher top-k vocab 近似：

```text
V_T^k = TopK(pi_T)

J_outlier_FKL(x)
  =
  - sum_{v in V_T^k}
      pi_T(v) log pi_T(v) / pi_S(v)
```

等价写成优化 student 的交叉熵形式：

```text
sum_{v in V_T^k} pi_T(v) log pi_S(v)
```

忽略 teacher entropy 常数后，就是让 student 在 teacher top-k token 上提高概率。

为什么 outlier 区域要从 RKL 换成 FKL？

```text
RKL on outlier:
  盯着 student 已经采出来、teacher 很不认可的 token x。
  信号主要是“强烈压低这个 token”。
  如果 teacher/student mismatch 很大，这个负梯度可能是 outlier。

FKL on outlier:
  不再盯着 student 采错的 token x。
  改为看 teacher 在同一 context 下认为哪些 top-k token 合理。
  信号变成“把 student 概率质量拉向 teacher-supported tokens”。
```

所以它的作用不是简单惩罚错误 token，而是给 student 一个更明确的修正方向：

```text
不要只告诉 student “这个 token 不行”，
而是告诉 student “teacher 在这里更希望这些 token”。
```

memory：

```text
O(n * k)
```

论文统一设置：

```text
k = 64
```

为什么不在所有 token 上都用 top-k FKL？

论文实验发现：

```text
Full FKL / standalone top-k FKL 表现很差，
在 math-domain 中甚至 AIME 接近 0。
```

原因是 top-k FKL 是 full-vocab FKL 的有偏近似。如果全程只用小 top-k support，梯度会越来越偏。TrOPD 只在 outlier 区域使用它，目的是从 teacher 视角补回一些有用监督，而不是把它作为主 objective。

## Off-Policy Guidance

纯 OPD 的另一个问题是：

```text
student 太弱时，自己生成的 SoG 质量低；
teacher 虽然能评价，但 student 很少探索到高质量轨迹区域。
```

TrOPD 加入 teacher prefix：

```text
x[:l] 由 teacher 生成
x[l:] 由 student 继续生成
```

也就是：

```text
teacher 先写一段高质量 prefix，
student 从这个 prefix 后面继续 on-policy rollout。
```

off-policy prefix 使用 forward KL imitation：

```text
J_off(x)
  = - beta * I[x ~ pi_T] * log pi_T(x) / pi_S(x)
```

因为 prefix token 来自 teacher：

```text
x ~ pi_T
```

所以可以用 K1 estimator 估计 forward KL：

```text
KL(pi_T || pi_S)
  = E_{x ~ pi_T} [ log pi_T(x) / pi_S(x) ]
```

最小化 loss 时：

```text
loss_off
  = beta * log pi_T(x) / pi_S(x)
  = beta * (log pi_T(x) - log pi_S(x))
```

其中：

```text
log pi_T(x)
```

对 student 不反传，是常数。因此对 student 的梯度等价于：

```text
loss_off gradient
  = - beta * log pi_S(x)
```

也就是 teacher-prefix token 上的 CE / SFT loss：

```text
提高 student 对 teacher prefix token 的概率。
```

所以 teacher prefix guidance 可以同时理解为：

```text
形式上：
  sampled-token FKL / K1 estimator of KL(pi_T || pi_S)

梯度上：
  beta 加权的 teacher-forcing CE loss

作用上：
  off-policy distillation / imitation learning
```

它和 outlier top-k FKL 的区别：

```text
Outlier top-k FKL:
  context = student rollout prefix
  supervised tokens = teacher top-k token distribution
  memory = O(n * k)

Teacher prefix guidance:
  context/token prefix = teacher-generated prefix
  supervised token = teacher sampled token
  memory = O(n)
  gradient equivalent to CE on teacher prefix tokens
```

论文设置：

```text
beta = 0.001
```

此外，teacher prefix 的最大长度不是固定不变的：

```text
初始：
  max off-policy trajectory length = max training sequence length

训练中：
  cosine schedule 逐渐 anneal 到 0

结束：
  完全 on-policy
```

直觉：

```text
早期 student 弱：
  多用 teacher prefix，把 rollout 引导到高质量区域。

后期 student 变强：
  减少 teacher prefix，恢复纯 on-policy 训练。
```

## 最小伪代码

```python
# shapes:
# response_ids:       [B * G, L]
# student_log_probs:  [B * G, L]  log pi_S(x)
# teacher_log_probs:  [B * G, L]  log pi_T(x)
# response_mask:      [B * G, L]

# 1. trust probability
ratio_ts = torch.exp(teacher_log_probs - student_log_probs)
trust_prob = torch.clamp(ratio_ts, max=1.0)

# 2. stochastic trust mask
trust_mask = torch.bernoulli(trust_prob).bool()
outlier_mask = ~trust_mask

# 3. trust-region RKL / K1 objective
# objective maximize: log pi_T - log pi_S
# loss minimize: -(log pi_T - log pi_S)
loss_trust = -(teacher_log_probs - student_log_probs)
loss_trust = loss_trust * trust_mask * response_mask

# 4. outlier top-k FKL
# teacher_topk_probs:       [B * G, L, k]
# student_topk_log_probs:   [B * G, L, k]
# teacher_topk_log_probs:   [B * G, L, k]
loss_outlier = (
    teacher_topk_probs
    * (teacher_topk_log_probs - student_topk_log_probs)
).sum(dim=-1)
loss_outlier = loss_outlier * outlier_mask * response_mask

# 5. off-policy guidance on teacher-prefix tokens
# prefix_teacher_log_probs: [B, Lp]
# prefix_student_log_probs: [B, Lp]
# prefix_mask:              [B, Lp]
loss_off = beta * (
    prefix_teacher_log_probs - prefix_student_log_probs
) * prefix_mask

loss = (
    loss_trust.sum()
    + loss_outlier.sum()
    + loss_off.sum()
) / (
    response_mask.sum() + prefix_mask.sum()
)
```

真实实现还要处理：

```text
1. teacher prefix 和 student continuation 的拼接。
2. prefix length 的 cosine annealing。
3. top-k token id 对齐和 gather student logprob。
4. rollout old policy / current policy 的策略更新细节。
5. PPO/GRPO 框架中是否还额外使用 policy ratio。
```

论文的公式主要描述 distillation objective，不是完整 PPO-style clipped surrogate。

## 和 PPO Trust Region 的区别

TrOPD 名字里有 Trust Region，但它不是 PPO/TRPO 里那种：

```text
限制 current policy 和 old policy 的 KL / ratio。
```

PPO 常见 ratio 是：

```text
ratio_policy
  = pi_current(x) / pi_old(x)
```

控制的是：

```text
一次 policy update 不要离采样策略太远。
```

TrOPD 的 trust ratio 是：

```text
ratio_teacher_student
  = pi_T(x) / pi_S(x)
```

控制的是：

```text
teacher 对 student sampled token 的监督是否可靠。
```

所以二者不是同一个 ratio：

```text
PPO ratio:
  current policy vs old rollout policy。

TrOPD trust ratio:
  teacher policy vs student policy。
```

## 和 SAO / DIS 的区别

SAO 的 DIS 处理的是异步 rollout 中的 train/infer mismatch：

```text
ratio = train_policy_prob / rollout_policy_prob
```

超出 trust region 的 token 直接不反传，避免 stale rollout 带来的错误梯度。

TrOPD 处理的是 teacher/student supervision mismatch：

```text
ratio = teacher_prob / student_prob
```

如果 teacher 很不认可 student token，就认为 sampled-token RKL 监督不可靠。

二者相似点：

```text
都用 token-level mask 抑制不可靠梯度。
```

关键区别：

```text
SAO / DIS:
  关心异步训练中 rollout policy 和 train policy 是否一致。

TrOPD:
  关心 teacher 是否能可靠监督 student 自己采出的 token。
```

## 和 Uni-OPD 的区别

Uni-OPD 的核心问题：

```text
teacher token-level OPD signal 可能和 outcome reward 冲突。
```

例如：

```text
correct trajectory 的 teacher score 低于 incorrect trajectory。
```

Uni-OPD 用：

```text
1. margin shift
2. correct / incorrect trajectory balance
3. outcome-aware advantage calibration
```

来修正 teacher signal。

TrOPD 的核心问题：

```text
teacher/student 分布差距大时，
sampled-token RKL 的梯度可能成为 outlier。
```

它不看最终答案正确与否，而是看：

```text
pi_T(x) / pi_S(x)
```

即 teacher 是否“接受”这个 student token。

所以：

```text
Uni-OPD:
  outcome-aware。

TrOPD:
  distribution-reliability-aware。
```

二者可以互补：

```text
先用 TrOPD 判断 token-level supervision 是否可靠；
再用 Uni-OPD 的 outcome reward 修正 trajectory-level preference。
```

## 和 REOPOLD / EOPD / AOPD 的关系

论文比较了多个 OPD baseline：

```text
OPD:
  vanilla token-level RKL / K1 estimator。

EOPD / Entropy OPD:
  根据 token entropy 选择训练 token，例如只训练 top 20% high-entropy tokens。

REOPOLD:
  使用 reward clipping 降低 outlier reward 的影响。

AOPD:
  concurrent work，和 TrOPD 思路正交。
```

TrOPD 认为：

```text
entropy 不一定是 OPD token 可靠性的好标准；
普通低熵 token 也可能有 teacher 提供的有效监督。
```

因此它不用 entropy 判断，而用：

```text
min(pi_T(x) / pi_S(x), 1)
```

作为 teacher accept / trust region criterion。

和 REOPOLD 相比：

```text
REOPOLD:
  对 outlier reward 做 clip。
  极端梯度被压住，但有用监督也可能被截断。

TrOPD:
  trust region 内保留 RKL。
  outlier 区域改用 teacher top-k FKL 回收监督。
```

## 实验设置

论文评估两类设置：

```text
single-domain distillation:
  math domain

multi-domain distillation:
  math + code + science / STEM + instruction following
```

模型设置：

```text
single-task:
  teacher = Skywork-OR1-Math-7B
  student = DeepSeek-Distilled-Qwen-1.5B

multi-task:
  teacher = Skywork-OR1-7B
  student = DeepSeek-Distilled-Qwen-1.5B

multi-task Qwen3:
  teacher = Qwen3-Nemotron-4B
  student = Qwen3-SFT-1.7B
```

训练数据：

```text
single-domain:
  OpenThoughts3 math prompts

multi-domain:
  OpenThoughts3 prompts from math, code, science
```

训练超参：

```text
training steps:
  200

learning rate:
  5e-6

top-k for FKL:
  k = 64

off-policy guidance beta:
  beta = 0.001

prompt batch size:
  128

rollouts per prompt:
  4

max generation length:
  8096 tokens
```

评测：

```text
math:
  AIME 2024, AIME 2025, AMC 2023
  each result averaged over 32 evaluations

STEM:
  GPQA Diamond, MMLU-Redux v2

instruction following:
  IFBench

code:
  LiveCodeBench v6
```

## 实验结论

### DeepSeek-Qwen2.5-1.5B single-domain

平均分：

```text
Base:
  34.69

OPD:
  37.11

REOPOLD:
  38.79

TrOPD:
  40.63
```

TrOPD 相比 OPD：

```text
+3.52 average
```

### DeepSeek-Qwen2.5-1.5B multi-domain

平均分：

```text
OPD:
  32.99

REOPOLD:
  35.58

TrOPD:
  37.61
```

TrOPD 相比 OPD：

```text
+4.62 average
```

### Qwen3-SFT-1.7B multi-domain

平均分：

```text
Qwen3-SFT-1.7B:
  39.87

OPD:
  48.29

REOPOLD:
  48.56

TrOPD:
  51.73
```

TrOPD 相比 OPD：

```text
+3.44 average
```

## Ablation：哪个部分有用

math-domain ablation：

```text
OPD:
  46.79

Mask Outlier:
  47.72

Clip Outlier:
  47.86

FKL Outlier:
  49.00

TrOPD Mask:
  48.79

TrOPD Clip:
  48.73

TrOPD FKL:
  49.85
```

结论：

```text
1. 只要抑制 outlier，mask/clip 都比 vanilla OPD 好。
2. outlier 上用 top-k FKL 比单纯 mask/clip 更好。
3. 加 off-policy guidance 后，三种 outlier 方案都进一步提升。
4. TrOPD FKL 最强。
```

## Appendix 信息

论文 appendix 只有 Qwen3-Nemotron-4B teacher 的训练细节，没有额外算法证明。

Qwen3-Nemotron-4B：

```text
base:
  Qwen3-4B-Base

training:
  SFT + RLVR
```

SFT：

```text
data:
  Nemotron 3 Nano public datasets

size:
  about 14M samples

optimizer:
  Adam

learning rate:
  5e-5

weight decay:
  0.1

warmup:
  10%

batch size:
  512

average response length:
  about 7K tokens
```

RLVR：

```text
data:
  Nemotron-3-Nano-RL-Training-Blend

domains:
  Math, Coding, Science, Instruction Following

algorithm:
  GRPO

group size:
  16

batch size:
  128

update frequency:
  update every 2048 rollouts

max generation length:
  32K tokens

temperature:
  1.0

extra:
  masked importance sampling
```

## 对当前代码库的启发

当前 Uni-OPD 代码库已有的相关概念：

```text
miles/miles/backends/training_utils/loss.py:
  PPO/GRPO policy loss、TIS/ICEPOP、logprob ratio 等。

miles/Uni_OPD_utils/margin_calibration/margin_shift.py:
  outcome-aware margin shift。

miles/Uni_OPD_utils/online_balance/rollout_sample_balance.py:
  correct / incorrect rollout balance。
```

如果要实现 TrOPD，需要新增或改造：

```text
1. teacher_log_probs 与 student_log_probs 的 token-level ratio：
   exp(teacher_log_probs - student_log_probs)

2. trust_mask：
   Bernoulli(min(ratio, 1))

3. trust region RKL loss：
   -(teacher_log_probs - student_log_probs)

4. outlier top-k FKL：
   teacher top-k token ids/probs
   gather student logprob on teacher top-k ids

5. teacher prefix generation：
   teacher 先生成 prefix，student 从 prefix 继续 rollout

6. prefix length cosine annealing：
   从全 teacher prefix 逐步降到 0
```

需要特别注意，TrOPD 的 ratio 不是：

```text
current policy / old policy
```

也不是：

```text
train policy / rollout policy
```

而是：

```text
teacher / student
```

这决定了它解决的是 teacher supervision reliability，而不是 PPO 更新步长或训推不一致。

## 总结

TrOPD 的本质是：

```text
在 OPD 中，不是所有 student-generated token 都值得信任地接受 teacher sampled-token RKL 监督。
```

它用：

```text
P_trust(x) = min(pi_T(x) / pi_S(x), 1)
```

把 token 分成：

```text
trust region:
  teacher 可验证，使用 RKL / K1 estimator。

outlier:
  teacher 不太认可，避免 RKL 极端负梯度；
  改用 mask / clip / top-k FKL。
```

再通过：

```text
teacher prefix off-policy guidance
```

让 student 早期更容易探索到 teacher 支持的高质量 reasoning 轨迹。

这篇论文和 Uni-OPD 的关注点互补：

```text
TrOPD:
  解决 teacher/student distribution mismatch 下的 token-level supervision reliability。

Uni-OPD:
  解决 teacher token-level preference 和 outcome correctness 不一致的问题。
```

如果把二者结合，一个自然方向是：

```text
1. 用 TrOPD 过滤/修正不可靠 token-level OPD 信号。
2. 用 Uni-OPD 的 outcome reward 再校准 trajectory-level 正负样本 margin。
```
