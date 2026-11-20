# Direct-OPD：Weak-to-Strong Generalization via Direct On-Policy Distillation 技术解读

本文档解读论文：

```text
Weak-to-Strong Generalization via Direct On-Policy Distillation
https://arxiv.org/abs/2607.05394
```

这篇论文讨论的是：如何把小模型 RL 学到的改进方向迁移到更强的大模型上，而不是在每个大模型上重新跑昂贵的 RLVR。

## Student / Teacher 模型速览

Direct-OPD 的 teacher 不是单个最终模型，而是一对 weak teacher checkpoints：

```text
Teacher pair 1:
  pre-RL teacher reference:
    R1-Distill-1.5B

  post-RL teacher:
    JustRL-1.5B

Teacher pair 2:
  pre-RL teacher reference:
    Nemotron-1.5B

  post-RL teacher:
    QuestA-Nemotron-1.5B
```

Student 是更强模型：

```text
Qwen3-1.7B
Qwen3-4B
R1-Distill-7B
```

关键点：Direct-OPD 不是模仿 post-RL weak teacher 的最终分布，而是蒸馏：

```text
Delta_T = log pi_T_postRL - log pi_T_preRL
```

也就是 weak teacher 经过 RL 学到的 policy shift。

## 一句话总结

Direct-OPD 的核心是：

```text
不要蒸馏 weak post-RL teacher 的最终 policy。

而是蒸馏 weak model 从 pre-RL 到 post-RL 的 policy shift：

  Delta_T = log pi_T - log pi_T_ref

这个 log-ratio 表示 RL 让 weak teacher 更倾向或更不倾向哪些 token。
把这个 policy shift 当成 dense implicit reward，
作用在 strong student 自己 on-policy 访问到的 states 上。
```

一句话说：

```text
迁移的是“弱模型经过 RL 后学到的方向”，
不是“弱模型最后长什么样”。
```

## 为什么普通 OPD 不够

普通 OPD 是：

```text
student 自己 rollout；
teacher 在 student visited states 上提供 token distribution；
student 模仿 teacher 的最终 policy。
```

如果 teacher 比 student 强，这通常合理。

但 weak-to-strong 场景不同：

```text
post-RL weak teacher 可能仍然弱于 strong student。
```

此时直接模仿 weak teacher 的最终分布会有问题：

```text
teacher final policy = RL 带来的有用改进 + weak model 自身能力上限
```

如果 student 已经比 teacher 强，模仿 teacher 可能把 student 往弱模型能力边界拉回去。

论文的例子：

```text
R1-Distill-7B 初始 AIME24 = 56.7
JustRL-1.5B post-RL teacher AIME24 = 51.3

vanilla OPD toward JustRL-1.5B 会把 7B student 拉低到约 50。
```

所以问题不是 weak teacher 没有用，而是：

```text
不应该复制 weak teacher 的最终 policy。
```

## Direct-OPD 转移什么

Direct-OPD 使用一对 teacher checkpoint：

```text
pi_T_ref:
  weak model 的 pre-RL reference checkpoint

pi_T:
  同一个 weak model 的 post-RL checkpoint
```

定义 policy shift：

```text
Delta_T(y | x)
  = log pi_T(y | x) - log pi_T_ref(y | x)
```

token-level 分解：

```text
Delta_T(y | x)
  = sum_t r_t(y_t | s_t)

r_t(v)
  = log pi_T(v | s_t) - log pi_T_ref(v | s_t)
```

其中：

```text
s_t = (x, y_<t)
```

含义：

```text
r_t(v) > 0:
  RL 后 teacher 更喜欢 token v。

r_t(v) < 0:
  RL 后 teacher 压低了 token v。
```

这个 `r_t(v)` 就是 Direct-OPD 的 dense per-token implicit reward。

## Delta_T 和 Token Shift 计算例子

`Delta_T(y|x)` 表示：

```text
teacher 经过 RL 前后，
对整条 response y 的 log-prob 变化量。
```

如果：

```text
Delta_T(y|x) > 0
```

说明 RL 之后，weak teacher 更倾向生成这条 response。

如果：

```text
Delta_T(y|x) < 0
```

说明 RL 之后，weak teacher 压低了这条 response。

由于自回归分解：

```text
log pi(y|x)
  = sum_t log pi(y_t | x, y_<t)
```

所以整条 response 的 shift 等于每个 token shift 的和：

```text
Delta_T(y|x)
  = sum_t [
      log pi_T(y_t | s_t)
      - log pi_T_ref(y_t | s_t)
    ]
```

其中：

```text
s_t = (x, y_<t)
```

下面给一个具体 token-level 例子。

假设 prompt 是：

```text
x = "What is 2 + 3?"
```

student rollout 到某个 prefix：

```text
s_t = "What is 2 + 3? Answer:"
```

student 当前 top-k candidate tokens 是：

```text
S_t = ["5", "6", "4"]
```

pre-RL teacher `pi_T_ref` 对这些 token 的概率：

```text
pi_T_ref("5" | s_t) = 0.40
pi_T_ref("6" | s_t) = 0.25
pi_T_ref("4" | s_t) = 0.20
```

post-RL teacher `pi_T` 对这些 token 的概率：

```text
pi_T("5" | s_t) = 0.70
pi_T("6" | s_t) = 0.10
pi_T("4" | s_t) = 0.05
```

逐 token 计算 policy shift：

```text
r_t("5")
  = log 0.70 - log 0.40
  = log(0.70 / 0.40)
  = log 1.75
  ≈ 0.5606
```

```text
r_t("6")
  = log 0.10 - log 0.25
  = log(0.10 / 0.25)
  = log 0.4
  ≈ -0.9163
```

```text
r_t("4")
  = log 0.05 - log 0.20
  = log(0.05 / 0.20)
  = log 0.25
  ≈ -1.3863
```

含义：

```text
"5":
  RL 后 teacher 更喜欢，positive shift。
  Direct-OPD 会鼓励 student 增大 "5" 的概率。

"6":
  RL 后 teacher 不喜欢，negative shift。
  Direct-OPD 会压低 student 对 "6" 的概率。

"4":
  RL 后 teacher 更明显压低，negative shift 更大。
```

如果 student 在这个 top-k support 上的归一化概率是：

```text
bar p_t("5") = 0.50
bar p_t("6") = 0.30
bar p_t("4") = 0.20
```

则 weighted coefficient：

```text
A_w(v) = stop_gradient(bar p_t(v) * r_t(v))
```

得到：

```text
A_w("5")
  = 0.50 * 0.5606
  = 0.2803
```

```text
A_w("6")
  = 0.30 * (-0.9163)
  = -0.2749
```

```text
A_w("4")
  = 0.20 * (-1.3863)
  = -0.2773
```

local loss 可理解为：

```text
loss_t
  = - sum_{v in S_t} A_w(v) * log pi_student(v | s_t)
```

代入：

```text
loss_t
  = -0.2803 * log pi_student("5" | s_t)
    +0.2749 * log pi_student("6" | s_t)
    +0.2773 * log pi_student("4" | s_t)
```

优化这个 loss 会：

```text
提高 "5" 的概率；
降低 "6" 和 "4" 的概率。
```

所以 Direct-OPD 的 token-level 逻辑是：

```text
看 weak teacher 经过 RL 后，
对 student 当前候选 token 的概率是升了还是降了；
升了就鼓励 student，
降了就抑制 student。
```

## 为什么 log-ratio 可以看成 implicit reward

KL-regularized RL 目标：

```text
max_pi E_{y ~ pi} [
  r(x, y) - beta * log(pi(y|x) / pi_ref(y|x))
]
```

它的最优策略满足：

```text
pi*(y|x) ∝ pi_ref(y|x) * exp(r(x,y) / beta)
```

取 log-ratio：

```text
log pi*(y|x) / pi_ref(y|x)
  = r(x,y) / beta - log Z(x)
```

其中 `log Z(x)` 对同一个 prompt 是常数。

所以：

```text
policy / reference 的 log-ratio
≈ reward 的缩放版本 + prompt-level 常数
```

Direct-OPD 反过来使用这个事实：

```text
已经有 post-RL teacher pi_T 和 pre-RL reference pi_T_ref，
就可以从它们的 log-ratio 里读出 teacher RL 学到的隐式 reward。
```

即：

```text
Delta_T(y|x)
  = log pi_T(y|x) - log pi_T_ref(y|x)
  ≈ r_T(x,y) / beta - log Z_T(x)
```

## Direct-OPD 目标函数

student 初始 checkpoint：

```text
pi_S
```

训练中 policy：

```text
pi_theta
```

理想 sequence-level objective：

```text
J_DirectOPD(theta)
  = E_x [
      E_{y ~ pi_theta(.|x)} [ Delta_T(y|x) ]
      - alpha * KL(pi_theta(.|x) || pi_S(.|x))
    ]
```

含义：

```text
1. student 自己 rollout。
2. 在 student rollout 上读取 weak teacher 的 RL-induced policy shift。
3. 提高那些 RL 后 weak teacher 更偏好的 token/trajectory。
4. 用 KL 把 student anchor 到自己的初始模型 pi_S，避免过度漂移。
```

这个目标的解析最优形式：

```text
pi*(y|x)
  ∝ pi_S(y|x) * exp(Delta_T(y|x) / alpha)

  = pi_S(y|x) *
    (pi_T(y|x) / pi_T_ref(y|x))^(1/alpha)
```

所以 Direct-OPD 不是让 student 变成 teacher，而是：

```text
在 student 自己的初始分布 pi_S 上，
乘上 weak teacher 的 RL 改进方向。
```

## Top-k Token Surrogate

论文沿用 OPD 的 on-policy top-k interface。

在每个 student visited prefix：

```text
s_t = (x, y_<t)
```

取 student 当前 top-k token：

```text
S_t = TopK_v pi_theta(v | s_t)
```

默认：

```text
k = 16
```

在 `S_t` 上归一化 student 概率：

```text
bar p_t(v)
  = pi_theta(v|s_t) / sum_{u in S_t} pi_theta(u|s_t)
```

对每个 candidate token 计算 teacher policy shift：

```text
r_t(v)
  = log pi_T(v|s_t) - log pi_T_ref(v|s_t)
```

然后用 top-k 上的解析期望代替单个 sampled token 的高方差估计：

```text
sum_{v in S_t}
  bar p_t(v) * r_t(v) * grad log pi_theta(v|s_t)
```

这相当于 Rao-Blackwellization：

```text
trajectory 仍然由 student rollout 采样；
但每个 visited prefix 的 action gradient 不只用 sampled token，
而是在 student top-k support 上求期望。
```

## Stop-gradient 为什么需要

论文定义：

```text
A_t^w(v)
  = stop_gradient( bar p_t(v) * r_t(v) )
```

最终局部 surrogate：

```text
grad J
≈ E [
    sum_t sum_{v in S_t}
      A_t^w(v) * grad log pi_theta(v|s_t)
  ]
  - alpha * grad KL(pi_theta || pi_S)
```

为什么要 stop-gradient？

```text
bar p_t(v) 本身依赖 pi_theta。
如果不 detach，会引入 top-k softmax 权重的额外 Jacobian 项，
不再是标准 policy-gradient 形式。
```

所以实现上：

```text
bar p_t(v) * r_t(v)
```

只作为静态 advantage coefficient，不对它反传。

## Adaptive KL

Direct-OPD 的 dense reward scale 来自 teacher pair：

```text
Delta_T = log pi_T - log pi_T_ref
```

但这个 scale 取决于 weak teacher 当初 RL 的 reward scale 和 KL budget，student 并不知道。

因此固定 KL 系数 `alpha` 很难跨 teacher-student pair 通用。

论文采用一个简单 adaptive KL controller：

```text
alpha_{m+1}
  = clip(
      alpha_m * (1 + epsilon * sign(bar r_m)),
      alpha_min,
      alpha_max
    )
```

默认：

```text
epsilon = 0.01
alpha_min = 0.5
alpha_max = 2.5
```

其中：

```text
bar r_m:
  当前 batch 中 student-weighted shift 的均值

bar r_m = mean over visited prefixes/top-k [
  bar p_t(v) * r_t(v)
]
```

直觉：

```text
如果 bar r_m > 0:
  当前 visited tokens 平均受到 teacher shift 鼓励；
  增大 alpha，加强 KL anchor，避免过度放大局部信号。

如果 bar r_m < 0:
  当前 visited tokens 平均被 teacher shift 压低；
  减小 alpha，允许 student 离开这些不被鼓励的区域。
```

它和普通 PPO adaptive KL 不同：

```text
普通 adaptive KL:
  根据实际 KL 是否超过 target 来调节。

Direct-OPD adaptive KL:
  根据 dense teacher-shift reward 的 batch mean sign 来调节。
```

## 和普通 OPD 的区别

普通 OPD：

```text
teacher signal = log pi_T(v|s_t) - log pi_student(v|s_t)
目标是模仿 post-RL teacher final distribution。
```

Direct-OPD：

```text
teacher signal = log pi_T(v|s_t) - log pi_T_ref(v|s_t)
目标是转移 teacher RL-induced policy shift。
```

核心区别：

```text
OPD copies teacher final policy.
Direct-OPD transfers teacher improvement direction.
```

所以 Direct-OPD 可以在：

```text
student already stronger than teacher
```

时仍然有效。

## 和 DPO 的关系

DPO 使用 policy/reference log-ratio 表示隐式 reward：

```text
log pi(y|x) - log pi_ref(y|x)
```

Direct-OPD 也使用同一个 identity，但方向相反：

```text
DPO:
  从 preference data 拟合一个 policy。

Direct-OPD:
  已经有 post-RL policy 和 reference，
  反推出这次 RL 学到的 implicit reward。
```

所以 Direct-OPD 可以理解成：

```text
把 weak teacher 的 RL checkpoint pair 当作一个 reward model。
```

但这个 reward 是 token-level dense log-ratio，不需要单独训练 reward model。

## Tensor Shape 视角

假设：

```text
B:
  prompt batch size

G:
  rollout number per prompt

L:
  response length

k:
  student top-k size，默认 16
```

student rollout：

```text
response_ids:
  [B * G, L]
```

current student top-k：

```text
topk_ids:
  [B * G, L, k]

student_topk_probs:
  [B * G, L, k]

student_topk_logprobs:
  [B * G, L, k]
```

teacher pair 在同一 prefix、同一 top-k ids 上 forward/gather：

```text
teacher_topk_logprobs:
  log pi_T(v|s_t)
  [B * G, L, k]

teacher_ref_topk_logprobs:
  log pi_T_ref(v|s_t)
  [B * G, L, k]
```

dense shift：

```text
r_t(v):
  teacher_topk_logprobs - teacher_ref_topk_logprobs
  [B * G, L, k]
```

weighted advantage coefficient：

```text
A_w:
  stop_gradient(student_topk_probs_renorm * r_t)
  [B * G, L, k]
```

local loss 可以写成：

```text
loss_direct
  = - sum_{v in topk} A_w(v) * log pi_theta(v|s_t)
```

再加：

```text
alpha * KL(pi_theta || pi_S)
```

## 训练设置

Direct-OPD 数据：

```text
math subset of Skywork-OR1-RL-Data
```

也观察到替换为：

```text
DAPO-Math-17K
```

有类似趋势。

训练 prompt：

```text
Solve the following math problem step by step.
The last line of your response should be of the form
Answer: $Answer ...
```

Direct-OPD 默认超参：

```text
framework:
  verl

global batch size:
  64

mini batch size:
  64

rollout n:
  4

max prompt length:
  1024

max response length:
  2048

sampling temperature:
  1.0

top-p:
  1.0

learning rate:
  1e-6

training steps:
  300

student top-k support:
  16

KL coefficient:
  [0.8, 2] when fixed;
  adaptive KL otherwise
```

评测：

```text
benchmarks:
  AIME 2024, AIME 2025

samples per problem:
  32

temperature:
  0.7

top-p:
  0.95

max generation length:
  31744
```

## 实验 1：小 RL teacher 改进更强 student

Teacher pair 1：

```text
R1-Distill-1.5B -> JustRL-1.5B
```

Student：

```text
Qwen3-1.7B
Qwen3-4B
R1-Distill-7B
```

结果：

```text
Qwen3-1.7B:
  AIME24 48.3 -> 58.3 (+10.0)
  AIME25 36.8 -> 43.2 (+6.4)

Qwen3-4B:
  AIME24 72.5 -> 77.6 (+5.1)
  AIME25 65.6 -> 68.8 (+3.2)

R1-Distill-7B:
  AIME24 56.7 -> 63.1 (+6.4)
  AIME25 40.5 -> 48.8 (+8.3)
```

注意：

```text
Qwen3-4B 和 R1-Distill-7B 初始已经超过 post-RL JustRL-1.5B teacher。
```

这说明 Direct-OPD 不是在复制弱 teacher。

Teacher pair 2：

```text
Nemotron-1.5B -> QuestA-Nemotron-1.5B
```

结果：

```text
Qwen3-1.7B:
  AIME24 48.3 -> 59.0 (+10.7)
  AIME25 36.8 -> 43.1 (+6.3)

R1-Distill-7B:
  AIME24 56.3 -> 61.2 (+4.9)
  AIME25 39.5 -> 44.0 (+4.5)
```

说明：

```text
Direct-OPD 不只适用于一个 teacher pair 或一种训练 pipeline。
```

## 实验 2：弱到强路径比直接大模型 RL 更省

比较两条路径：

```text
Direct RL:
  直接在 R1-Distill-7B 上跑 RL。

Weak-to-Strong:
  先在 R1-Distill-1.5B 上跑 RL，
  选 300/600/900/1200/1500 step checkpoints，
  再用 Direct-OPD 转移到 R1-Distill-7B。
```

结论：

```text
同样 RL step budget 下，
小模型 RL + Direct-OPD 转移
优于直接在 7B 上 RL。
```

计算成本：

```text
R1-Distill-1.5B 1500-step RL:
  about 160 hours on 32 A100

R1-Distill-7B RL:
  about 320 hours on 32 A100

Direct-OPD transfer:
  about 4 hours on 8 A100
```

含义：

```text
小模型是更便宜的 RL 探索器；
Direct-OPD 把它发现的 policy-improvement direction 转移给大模型。
```

## 实验 3：Sequential Composition

Direct-OPD 可以串联多个独立 policy shift。

对 Qwen3-1.7B：

```text
Initial:
  AIME24 48.3
  AIME25 36.8

After JustRL shift:
  AIME24 58.3 (+10.0)
  AIME25 43.2 (+6.4)

After QuestA shift:
  AIME24 63.8 (+15.5 total)
  AIME25 46.8 (+10.0 total)
```

这说明：

```text
不同小模型 RL run 学到的 policy shift 可以组合，
像叠加多个 improvement direction。
```

## 关键分析 1：不依赖 top-k overlap imitation

普通 OPD 成功通常要求 teacher/student 在 student visited states 上有较高 top-k overlap。

Direct-OPD 不一定需要。

论文计算：

```text
Overlap_k(S, C; s_t)
  = |TopK_student(s_t) ∩ TopK_comparison(s_t)| / k
```

比较：

```text
student vs post-RL teacher
student vs teacher reference
```

发现：

```text
pattern-aligned case:
  overlap 上升，类似普通 OPD。

cross-pattern transfer:
  overlap 仍较低，
  但 Direct-OPD 仍有效。
```

解释：

```text
Direct-OPD 不要求 student 模仿 teacher 高概率 token；
它只需要在 student 自己 top-k support 内，
用 teacher/reference log-ratio 给这些 token 排序。
```

## 关键分析 2：短 response horizon 可以泛化到长推理

Direct-OPD 训练使用：

```text
max response length = 2048
```

但评测可以生成很长：

```text
max generation length = 31744
```

论文分析发现：

```text
2k 短 horizon 训练产生的行为变化，
会影响更长 rollout 的 later reasoning pattern。
```

同时：

```text
6k horizon 虽然让 actor 更强地移动到 teacher-shift 方向，
但验证效果更差。
```

解释：

```text
长 prefix 后 teacher pair 可能越来越 off-distribution，
teacher/reference log-ratio 变大但不可靠。
2k horizon 避免吸收这些 late-prefix noisy signal。
```

## 关键分析 3：KL 控制信号可靠性

Direct-OPD 的 dense reward 不应该被当成越大越好。

原因：

```text
如果 student 漂移太远，
它会访问 teacher 和 teacher_ref 都不熟悉的 states，
此时 log-ratio 可能数值很大，但不代表真正有用。
```

KL 的作用不仅是 regularization：

```text
它控制 student 访问哪些 states，
也就控制 teacher-shift reward 是否可靠。
```

论文发现：

```text
最佳 fixed KL coefficient 因 teacher-student pair 而异。
adaptive KL 往往把 dense token reward 的 batch mean 拉向接近 0 的平衡区。
```

## 和 TrOPD / Uni-OPD / X-OPD 的关系

### 和 TrOPD

TrOPD：

```text
判断 teacher 对 student sampled token 是否可靠；
outlier 区域改用 top-k FKL。
```

Direct-OPD：

```text
不看 teacher final policy 是否接受 student token；
看 post-RL teacher 相对 pre-RL teacher 的 policy shift。
```

二者都关注 teacher signal reliability，但对象不同：

```text
TrOPD:
  teacher vs student distribution mismatch

Direct-OPD:
  teacher RL-induced direction vs teacher capacity ceiling
```

### 和 Uni-OPD

Uni-OPD：

```text
用 outcome correctness 校准 teacher token-level OPD signal。
```

Direct-OPD：

```text
用 teacher checkpoint pair 提取 implicit reward。
```

Direct-OPD 不直接使用 outcome reward 校准 student trajectory。

### 和 X-OPD

X-OPD：

```text
text teacher -> speech student
跨模态对齐。
```

Direct-OPD：

```text
weak post-RL teacher pair -> strong student
弱到强迁移。
```

二者都不是简单模仿 teacher final policy，但迁移对象不同。

## 局限

论文明确指出：

```text
1. Direct-OPD 依赖 teacher/reference policy shift 在 student visited states 上有意义。
2. response length 和 KL strength 对 teacher-student pair 敏感。
3. 如果 teacher RL 学到的 shift 本身不可靠，迁移也会失败。
```

另一个实际限制：

```text
需要同时加载/查询 post-RL teacher 和 pre-RL teacher reference，
并在 student top-k tokens 上计算 logprob。
```

不过 teacher 是小模型，论文中 scoring 不是主要瓶颈。

## 总结

Direct-OPD 的本质是：

```text
把小模型 RL 的结果从“最终策略”变成“隐式奖励”来复用。
```

它转移的不是：

```text
pi_T
```

而是：

```text
log pi_T - log pi_T_ref
```

这个 policy shift 表示：

```text
RL 让小模型更喜欢什么、更不喜欢什么。
```

然后把这个信号用于 strong student 自己的 on-policy states：

```text
student 自己走到哪里，
就在那里读 teacher pair 的 log-ratio。
```

因此它可以让：

```text
weak teacher < strong student
```

的情况下仍然产生正迁移。
