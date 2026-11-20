# UP：Unbounded Positive Asymmetric Optimization 技术解读

本文档解读论文：

```text
UP: Unbounded Positive Asymmetric Optimization for Breaking the Exploration-Stability Dilemma
https://arxiv.org/abs/2607.06987
```

## 一句话总结

这篇论文解决的是 GRPO / DAPO / GSPO 这类基于 importance sampling ratio 的 RL 算法里的一个矛盾：

```text
不 clip:
  ratio = pi_theta / pi_old 可能爆炸，训练不稳定。

clip:
  正优势样本一旦 ratio 超过上界，梯度直接变 0，
  罕见但正确的长尾推理路径无法继续被强化，探索能力受限。
```

UP 的做法是：

```text
正优势轨迹:
  去掉 pi_old anchor 和上界 clip，
  改成稳定的 REINFORCE-like log-prob loss。

负优势轨迹:
  保留原来的 clipped objective，
  避免对错误轨迹做过激惩罚导致训练崩。
```

所以它的本质不是一个新的 reward，也不是蒸馏方法，而是一个可以插到 GxPO policy loss 里的 asymmetric objective。

## 背景：为什么 ratio 和 clipping 会形成两难

GRPO / DAPO 的 token-level importance sampling ratio 是：

```text
r_{i,t}(theta)
  = pi_theta(o_{i,t} | q, o_{i,<t})
    / pi_old(o_{i,t} | q, o_{i,<t})
```

这里：

```text
q:
  prompt / query。

o_i:
  第 i 条 rollout response。

o_{i,t}:
  第 i 条 response 的第 t 个 token。

pi_old:
  生成这批 rollout 时的行为策略。

pi_theta:
  当前正在训练更新的 policy。
```

纯 IS 目标的梯度大致是：

```text
grad J_IS
  = A * r(theta) * grad log pi_theta(token)
```

问题在于 `r(theta)` 直接乘在梯度上。如果某个 token 在 `pi_old` 下概率很小，例如：

```text
pi_old(token) = 0.0001
pi_theta(token) = 0.01
r = 100
```

那么这个 token 的梯度会被放大 100 倍。长推理里这种低概率 token 很常见，所以不加约束的 IS 容易带来梯度爆炸。

为了解决这个问题，PPO / GRPO / DAPO 会 clip ratio：

```text
positive advantage:
  min(r * A, clip(r, 1 - eps_low, 1 + eps_high) * A)
```

当 `A > 0` 时，如果：

```text
r > 1 + eps_high
```

目标进入常数区，梯度变成 0。也就是说，正确轨迹上某个 token 只要被提升到超过 trust region 上界，就不能继续被强化。

## Probability Capacity

论文提出 Probability Capacity，简称 Cap，用来衡量：

```text
在梯度被 clip 掉之前，
某个 token 的 pi_theta 还能增加或减少多少。
```

对正优势 token，DAPO 的上界是：

```text
pi_theta <= (1 + eps_high) * pi_old
```

同时概率不能超过 1，所以正向更新容量是：

```text
Cap(pi_theta, pi_old)
  = min(1, (1 + eps_high) * pi_old) - pi_theta
    if pi_theta < (1 + eps_high) * pi_old

  = 0
    otherwise
```

一个具体例子：

```text
pi_old = 0.01
eps_high = 0.28

max allowed pi_theta
  = 1.28 * 0.01
  = 0.0128

capacity
  = 0.0128 - pi_theta
```

如果当前 `pi_theta = 0.0128`，即使这条推理路径是正确的，且 advantage 很大，梯度也已经被截断了。

这就是论文说的 exploration-stability dilemma：

```text
clip 太紧:
  稳定，但探索被压死。

不 clip:
  能探索，但 IS ratio 会爆。
```

## UP 的核心方法

UP 的关键不是简单把正优势上界设成无穷大。论文做了两个动作：

```text
1. 正优势分支去掉 pi_old 分母，改成 self-anchored stop-gradient ratio。
2. 只对正优势这么做；负优势仍保留 clipping。
```

### Self-Anchored Ratio

普通 IS ratio 是：

```text
r(theta) = pi_theta / pi_old
```

UP 对正优势样本定义：

```text
tilde_r(theta)
  = pi_theta(token) / sg(pi_theta(token))
```

其中 `sg(.)` 是 stop-gradient：

```text
forward 数值:
  sg(pi_theta) = pi_theta

backward:
  sg(pi_theta) 被当成常数，不反传梯度。
```

所以 forward 上：

```text
tilde_r(theta) = 1
```

但 backward 上：

```text
grad tilde_r(theta)
  = grad pi_theta / sg(pi_theta)
  = grad pi_theta / pi_theta
  = grad log pi_theta
```

因此：

```text
grad J_UP^+
  = A * grad log pi_theta(token)
```

这和 REINFORCE 的梯度形式一致，但样本仍然来自 `pi_old` 生成的 rollout batch。它的直觉是：

```text
保留“用当前 batch 多步训练”的工程方式，
但正优势分支不再让 pi_old 出现在梯度分母里。
```

这样就同时避免了两个问题：

```text
没有 pi_old 分母:
  不会因为 pi_old 很小导致 ratio 梯度爆炸。

没有 upper clip:
  正确长尾 token 可以持续被增强。
```

在 UP 正优势分支里，论文认为 Cap 不再被历史策略限制，最大可提升空间变成：

```text
Cap = 1 - pi_theta
```

也就是只受概率本身上界约束。

## UP-DAPO 公式

UP-DAPO 把 token 按 advantage 正负分流：

```text
J_UP-DAPO(theta)
  = E [
      1 / sum_i |o_i|
      * sum_i sum_t objective_{i,t}
    ]
```

其中：

```text
if A_{i,t} > 0:
  objective_{i,t}
    = A_{i,t} * log pi_theta(o_{i,t} | q, o_{i,<t})

if A_{i,t} <= 0:
  objective_{i,t}
    = min(
        r_{i,t}(theta) * A_{i,t},
        clip(r_{i,t}(theta), 1 - eps_low, 1 + eps_high) * A_{i,t}
      )
```

注意：论文主公式里正分支直接写成 `A * log pi_theta`，这是因为 self-anchored ratio 的梯度已经等价于 REINFORCE-like log-prob gradient。

训练代码里如果按最小化 loss 写，通常会变成：

```python
# log_probs: [N, L]
# old_log_probs: [N, L]
# advantages: [N] 或 [N, L]
# response_mask: [N, L]

ratio = torch.exp(log_probs - old_log_probs)       # [N, L]
ratio_clip = torch.clamp(
    ratio,
    1.0 - eps_low,
    1.0 + eps_high,
)                                                   # [N, L]

adv = advantages
if adv.dim() == 1:
    adv = adv[:, None]                              # [N, 1] -> broadcast 到 [N, L]

pos_mask = (adv > 0).float() * response_mask        # [N, L]
neg_mask = (adv <= 0).float() * response_mask       # [N, L]

# 正优势：不使用 old_log_probs，不使用 ratio，不使用 upper clip
obj_pos = adv * log_probs                           # [N, L]

# 负优势：保留原 GRPO/DAPO clipped objective
obj_neg_unclipped = ratio * adv                     # [N, L]
obj_neg_clipped = ratio_clip * adv                  # [N, L]
obj_neg = torch.minimum(obj_neg_unclipped, obj_neg_clipped)

objective = pos_mask * obj_pos + neg_mask * obj_neg # [N, L]
loss = -masked_mean(objective, response_mask)
```

这里 `N = B * G`：

```text
B:
  prompt 数量。

G:
  每个 prompt 的 rollout 数。

L:
  response token 最大长度。
```

如果 advantage 是 group-level 的，例如 GRPO 常见的 `[B, G]` 或 flatten 后 `[B * G]`，会 broadcast 到每个 response token。DAPO 也可以使用 token-level advantage `[B * G, L]`。

## UP-GRPO 公式

UP-GRPO 和 UP-DAPO 的区别主要是：

```text
GRPO 使用 sequence/group advantage A_i，
同一条 response 内所有 token 共享一个 advantage。

GRPO 还可能包含 reference KL penalty。
```

论文写成：

```text
if A_i > 0:
  A_i * log pi_theta(o_{i,t} | q, o_{i,<t})
  - beta * KL(pi_theta || pi_ref)

if A_i <= 0:
  min(
    r_{i,t}(theta) * A_i,
    clip(r_{i,t}(theta), 1 - eps, 1 + eps) * A_i
  )
  - beta * KL(pi_theta || pi_ref)
```

也就是说：

```text
UP 改的是 policy-gradient surrogate；
KL penalty 是否存在取决于底层算法配置。
```

## UP-GSPO 公式

GSPO 是 sequence-level ratio，不是 token-level ratio。它定义：

```text
s_i(theta)
  = ( pi_theta(o_i | q) / pi_old(o_i | q) ) ^ (1 / |o_i|)

  = exp(
      1 / |o_i|
      * sum_t [
          log pi_theta(o_{i,t} | q, o_{i,<t})
          - log pi_old(o_{i,t} | q, o_{i,<t})
        ]
    )
```

UP-GSPO 同样按 advantage 正负分流：

```text
if A_i > 0:
  A_i * 1 / |o_i| * sum_t log pi_theta(o_{i,t} | q, o_{i,<t})

if A_i <= 0:
  min(
    s_i(theta) * A_i,
    clip(s_i(theta), 1 - eps, 1 + eps) * A_i
  )
```

附录证明了正优势分支的 self-anchored sequence ratio：

```text
( pi_theta(o_i | q) / sg(pi_theta(o_i | q)) ) ^ (1 / |o_i|)
```

求导后正好变成：

```text
A_i * 1 / |o_i| * sum_t grad log pi_theta(o_{i,t} | ...)
```

所以 UP-GSPO 的正分支是 length-normalized REINFORCE gradient。

## 为什么不能直接设置 eps_high = infinity

这是论文消融里最关键的一点。

如果只把 DAPO 的 `eps_high` 设成无穷大，那么正优势分支变成：

```text
objective = r(theta) * A
```

但 ratio 仍然是：

```text
r(theta) = pi_theta / pi_old
```

梯度仍然是：

```text
grad = A * r(theta) * grad log pi_theta
```

也就是说，`pi_old` 很小导致的梯度爆炸问题还在。论文实验显示：

```text
DAPO with eps_high = infinity:
  训练早期看起来可行，
  约 80 steps 后 gradient norm 爆到 1e13，
  KL divergence 也快速上升。
```

所以 UP 的关键不是“放开上界”本身，而是：

```text
放开正优势上界之前，
先把梯度里的 pi_old 分母拿掉。
```

## 为什么负优势不能也 unbounded

如果对负优势也使用 unbounded REINFORCE-like 更新，错误 token 的概率会被持续强力压低。由于负样本很多、reward/verifier 也可能有噪声，这种更新会非常激进。

论文的对照实验是 symmetric unbounded off-policy REINFORCE：

```text
positive advantage:
  unbounded

negative advantage:
  also unbounded
```

结果：

```text
约 25 steps 内训练崩溃，
gradient norm 和 KL divergence 都急剧上升。
```

所以 UP 的 asymmetric 很重要：

```text
正优势:
  需要探索，允许持续增强。

负优势:
  需要稳定，保留 trust-region clipping。
```

附录对负优势 Cap 的描述是：

```text
有效更新窗口:
  1 - eps_low <= r(theta) <= c

lower boundary:
  pi_lower = (1 - eps_low) * pi_old

dual clip upper boundary:
  pi_upper = c * pi_old
```

负优势的目标是降低 token 概率，所以可降低容量是：

```text
Cap(pi_theta, pi_old)
  = pi_theta - (1 - eps_low) * pi_old
    if (1 - eps_low) * pi_old <= pi_theta <= c * pi_old

  = 0
    otherwise
```

## 实验设置

论文验证了三个层面：

```text
1. token-level:
   UP-DAPO vs DAPO
   model = Qwen3-14B-Base
   train = DAPO-17K-MATH
   eval = AIME24 Avg@32 / Maj@32 / Best@32

2. broad baseline:
   UP-GRPO vs 11 个 RL baselines
   model = Qwen3-8B
   train = MATH Levels 3-5
   eval = AIME24, AMC23, MATH500, Minerva, OlympiadBench Pass@1

3. universality:
   UP-GSPO on Qwen3-30B-A3B-Base MoE
   UP-GRPO on Qwen3-VL-8B-Instruct VLM
```

主要 baselines：

```text
DAPO, GRPO, GMPO, ASPO, CISPO, Dr.GRPO,
W-REINFORCE, REINFORCE++, RLOO, DPPO, GSPO, SAPO
```

## 主要结果

### UP-DAPO vs DAPO

在 Qwen3-14B-Base + DAPO-17K-MATH 上：

```text
AIME24 Avg@32 peak:
  DAPO:    47.71
  UP-DAPO: 51.15

AIME24 Maj@32 peak:
  DAPO:    58.36
  UP-DAPO: 60.88

AIME24 Best@32 peak:
  DAPO:    80.49
  UP-DAPO: 81.79
```

训练动态上：

```text
UP-DAPO entropy 更高:
  说明探索能力更强。

UP-DAPO gradient norm / KL 与 DAPO 接近甚至略低:
  说明放开正优势上界没有带来额外不稳定。
```

### UP-GRPO vs 其他 RL baseline

在 Qwen3-8B + MATH Levels 3-5 上，Pass@1 结果：

```text
Method        AIME24  AMC23  MATH500  Minerva  OlympiadBench  Average
GRPO          35.73   75.00  86.00    30.88    51.34          55.79
Dr.GRPO       33.33   85.00  85.80    30.15    51.19          57.09
CISPO         38.02   87.50  86.60    29.04    55.65          59.36
DPPO          40.10   87.50  86.20    30.51    53.27          59.52
GMPO          37.50   87.50  87.00    31.25    55.06          59.66
GSPO          40.52   85.00  88.20    31.25    55.80          60.15
SAPO          39.90   82.50  87.20    30.88    55.65          59.23
REINFORCE++   20.52   62.50  78.80    29.78    42.26          46.77
RLOO          31.67   80.00  85.20    28.68    50.60          55.23
W-REINFORCE   35.52   80.00  85.80    30.15    53.27          56.95
ASPO          37.50   85.00  87.60    29.78    58.48          59.67
UP-GRPO       41.04   87.50  88.40    31.25    58.33          61.31
```

UP-GRPO 平均分最高，比 GSPO 高 1.16 个百分点。

论文还观察到：

```text
多数 baseline 的 entropy 会随训练下降，说明策略探索逐渐收缩。
UP-GRPO 的 entropy 持续上升，同时 KL 保持较低。
```

### UP-GSPO 和多模态结果

在 Qwen3-30B-A3B-Base MoE 上：

```text
AIME24 Avg@32 peak:
  GSPO:    52.71
  UP-GSPO: 55.73
```

KL divergence 与 GSPO 几乎一致。

在 Qwen3-VL-8B-Instruct + Geometry3K 上：

```text
Geometry3K test peak accuracy:
  GRPO:    59.30
  UP-GRPO: 62.60
```

KL divergence 与 GRPO 几乎一致。

## 训练超参摘录

论文强调 baseline 与 UP 变体使用相同训练设置，唯一改变是 policy objective。

Qwen3-14B-Base 的 DAPO / UP-DAPO：

```text
prompt / response max:
  2048 / 20480

global batch size:
  512 prompts

rollout n:
  16

PPO mini-batch size:
  32

learning rate:
  1e-6

DAPO:
  eps_low = 0.2
  eps_high = 0.28

UP-DAPO:
  eps_low = 0.2
  eps_high = not used for positive branch
```

Qwen3-8B 的 GRPO / UP-GRPO：

```text
prompt / response max:
  1024 / 3072

global batch size:
  128 prompts

rollout n:
  8

PPO mini-batch size:
  32

learning rate:
  1e-6

GRPO:
  eps_low = 0.2
  eps_high = 0.2

UP-GRPO:
  eps_low = 0.2
  eps_high = not used for positive branch
```

rollout decoding：

```text
training temperature:
  1.0

training top-p:
  1.0

validation top-p:
  0.7

top-k:
  -1
```

## 和前面几篇论文的关系

### 和 MaxRL 的关系

MaxRL 改的是 advantage / objective 的统计解释：

```text
RL 是 ML 的一阶近似；
MaxRL 用 pass@k / higher-order terms 改 advantage 权重。
```

UP 改的是 policy loss 的 ratio / clipping 结构：

```text
正优势不用 pi_theta / pi_old，
改成 A * log pi_theta。
```

二者理论上可以组合：

```text
MaxRL 给 advantage；
UP 决定这个 advantage 如何进入 policy loss。
```

### 和 SAO / async 训推不一致工作的关系

SAO / DIS 类方法关注：

```text
rollout policy 和 training policy 不一致时，
token-level ratio 过大或过小会导致训练不稳。
```

它们通常通过 strict mask / clipping 来砍掉不可信 token。

UP 关注的是另一层问题：

```text
即使是标准 old_policy rollout，
正优势 token 的 upper clip 也会限制探索；
但直接去 clip 又会被 pi_old denominator 放大梯度。
```

所以 UP 不是异步系统优化，而是 GxPO loss 设计。

### 和 TrOPD / OPD 的关系

TrOPD / OPD 关注 teacher-student distillation：

```text
哪些 teacher token 可信？
teacher 和 student 分布差异太大怎么办？
```

UP 没有 teacher：

```text
它只看 RL advantage 的正负，
并调整 policy-gradient surrogate。
```

如果未来把 teacher trajectory 也纳入 RL batch，UP 可以作为底层 policy loss，但它本身不解决 teacher 是否可靠的问题。

## 局限和使用前提

UP 的前提是：

```text
正优势确实代表值得增强的轨迹。
```

如果 reward model / rule verifier 有大量 false positive，UP 会更积极地放大这些错误正样本。因此它更适合：

```text
数学、代码、几何等 verifier 较可靠的任务；
或者 outcome reward 可信度较高的训练阶段。
```

此外，UP 主要解决的是正优势探索受限：

```text
它不会自动解决 reward hacking、数据分布偏移、prompt 难度不均衡等问题。
```

## 我的理解

这篇论文的核心贡献可以压缩成一句话：

```text
对正确轨迹，不再用 pi_theta / pi_old 来决定梯度大小，
而是直接用 A * grad log pi_theta 继续强化；
对错误轨迹，继续用 clipping 保守惩罚。
```

它看起来简单，但关键点是分清了两个问题：

```text
ratio 爆炸来自 pi_old denominator；
探索受限来自 positive upper clip。
```

如果只去掉 clip，不去掉 `pi_old`，会崩；如果正负都去掉 clip，也会崩。UP 的价值就在于同时满足：

```text
正样本:
  unbounded exploration。

负样本:
  bounded stabilization。
```

