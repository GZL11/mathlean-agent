# SDAR：Self-Distilled Agentic Reinforcement Learning 技术解读

本文档解读论文：

```text
Self-Distilled Agentic Reinforcement Learning
https://arxiv.org/abs/2605.15155
```

说明：SDAR 和 AgentOPSD 都属于带 privileged skill context 的 agentic self-distillation / OPSD 方向，但二者是不同论文。SDAR 的核心方法是把 OPSD 作为 gated auxiliary objective 接到 agentic GRPO 上；AgentOPSD 2608.05987 则把 self-teacher gap 转成 turn-level credit 来重塑 GRPO advantage，见 [AgentOPSD 解读](OPD/agentopsd_recursive_self_distillation_agentic_rl_zh.md)。

## Student / Teacher 模型设置

这篇论文里的 teacher 不是另一个更大的独立模型，而是同一个策略模型的 teacher branch：

```text
Student:
  当前训练的 policy，不带额外 skill context。

Self-Teacher:
  同一个 policy / 同源参数分支，
  但 forward 时额外看到训练期检索出的 skill context c^+。
```

论文实验使用的 student backbone：

```text
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen3-1.7B-Instruct
```

因此 SDAR 不是“大 teacher 教小 student”的蒸馏，而是：

```text
同一个模型在普通上下文下作为 student；
同一个模型在额外 skill 上下文下作为 self-teacher。
```

## 一句话总结

SDAR 解决的是多轮智能体 RL 里的一个问题：

```text
GRPO / RL:
  有环境结果奖励，方向可靠，但监督很稀疏，只到轨迹级。

OPSD / self-distillation:
  有 token 级稠密监督，但多轮智能体场景里 teacher 信号很容易不稳定。
```

论文的做法是：

```text
RL 仍然是主干，不改 GRPO 的优势值和策略损失；
OPSD 只作为辅助损失；
每个 token 用 teacher-student 差值经过 sigmoid 得到门控权重；
重点蒸馏 teacher 明确认可的 token；
对 teacher 的负向否定做软衰减。
```

核心公式是：

```text
L(theta)
  = L_GRPO(theta) + lambda_SDAR * L_SDAR(theta)
```

其中 `L_SDAR` 是带门控的 token 级自蒸馏。

## 背景：为什么智能体场景不能直接套 OPSD

论文讨论的是多轮智能体：

```text
给定任务 x。
第 k 轮智能体看到 observation o_k。
智能体生成 response/action a_k。
环境返回下一轮 observation o_{k+1}。
```

一条轨迹里有多轮 response，论文为了表示简单，把所有有效 response token 展平成：

```text
y = (y_1, ..., y_T) ~ pi_theta(. | x)
```

其中：

```text
T:
  一整条多轮轨迹里的有效 response token 数。

s_t = (x, y_<t):
  student 在第 t 个 token 看到的上下文。

s_t^+ = (x, c^+, y_<t):
  self-teacher 看到的上下文，多了训练期特权上下文 c^+。
```

这里的 teacher 不是另一个更强模型，而是：

```text
同一个 policy / 同源模型，
但 teacher 分支在训练时额外看到特权上下文。
```

论文里的特权上下文主要是检索到的 skills：

```text
c^+ = 训练时检索出来的任务相关 skill。
```

这些 skills 来自外部 skill library，论文记为：

```text
E = {e_1, ..., e_M}
```

原文把每个 skill 描述为“紧凑、结构化的示范”。它们编码领域知识，例如：

```text
子目标分解
动作模板
任务领域知识
结构化示范
```

论文没有说这些 skill 是训练过程中由模型在线生成的；算法输入里直接要求给定 skill library `E`。训练时从 `E` 中检索一个 skill 放到 self-teacher 的上下文里，推理时 `{skill context}` 为空。目标是把 skill 中有用的知识内化到模型参数里，而不是让模型依赖推理期 skill prompt。

## 论文指出的两个核心问题

### 1. 多轮 OPSD 不稳定

单轮任务里，OPSD 可以让 teacher 在 student 自己采样的 token 上给稠密监督。

但多轮智能体不一样：

```text
第 1 轮的 token 影响第 2 轮 observation。
第 2 轮 response 又影响后续环境状态。
```

一旦 student 轨迹偏离 teacher 支持的轨迹，后面的 token 级监督可能迅速失真。论文称为多轮误差累积：

```text
早期不匹配会在多轮交互中持续放大；
每轮 KL 上升；
任务表现可能崩。
```

### 2. 特权上下文指导是不对称可信的

teacher 分支的优势来自检索到的 skills，但这些 skills 不是绝对可靠：

```text
Skill 质量:
  检索到的 skill 可能不相关、不完整、冗余。

Skill 利用:
  teacher 可能没有正确用好 skill。

多轮漂移:
  轨迹越往后，teacher-student 不匹配越容易变大。
```

因此 teacher 对 student token 的判断要分情况看。

如果 student 采样了 token `y_t`，teacher 给它更高概率：

```text
log pi_T(y_t | s_t^+) > log pi_theta(y_t | s_t)
```

这说明：

```text
student 已经能生成这个 token；
带 skill 的 teacher 更认可这个 token；
这可以视为正向认可。
```

如果 teacher 给它更低概率：

```text
log pi_T(y_t | s_t^+) < log pi_theta(y_t | s_t)
```

这不一定表示 token 错了，也可能是：

```text
skill 检索错了；
teacher 没用好 skill；
多轮上下文已经漂移。
```

所以 SDAR 的核心判断是：

```text
teacher 的正向认可更可信；
teacher 的负向否定要保守处理。
```

## 方法总览

SDAR 不是把蒸馏信号塞进优势值，也不是替换 GRPO。它保留：

```text
L_GRPO:
  verifier / 环境奖励驱动的策略损失。
```

额外加一个带门控的 OPSD 损失：

```text
L(theta)
  = L_GRPO(theta) + lambda_SDAR * L_SDAR(theta)
```

论文明确强调：

```text
RL remains the primary optimization backbone。
OPSD 只是被严格控制的辅助目标。
```

这个设计和 RLSD 不同：

```text
RLSD:
  用 teacher-student 差值重新加权 RL 优势值。
  风险是训练早期不匹配大时，直接放大 RL 更新。

SDAR:
  不改 RL 优势值。
  蒸馏单独作为辅助损失。
```

## GRPO 主损失

对每个输入 `x`，GRPO 采样一组 responses：

```text
{ y^(i) }_{i=1}^G ~ pi_theta(. | x)
```

环境给每条轨迹一个奖励：

```text
R(x, y^(i))
```

组内标准化 advantage：

```text
A^(i)
  = (R(x, y^(i)) - mu_G) / sigma_G
```

token 级 ratio：

```text
r_t^(i)
  = pi_theta(y_t^(i) | s_t^(i))
    / pi_old(y_t^(i) | s_t^(i))
```

GRPO loss 是常规 clipped surrogate：

```text
L_GRPO
  = - 1/G * sum_i Agg(
      min(
        r_t^(i) * A^(i),
        clip(r_t^(i), 1 - eps, 1 + eps) * A^(i)
      )
    )
    + beta_KL * KL(pi_theta || pi_ref)
```

其中 masked token average：

```text
Agg(z_1:T)
  = sum_t m_t * z_t / sum_t m_t
```

`m_t` 是 response mask，表示 token 是否有效。

## OPSD 信号：Teacher-Student Gap

teacher 和 student 在同一个 student-sampled token `y_t` 上计算概率：

```text
student:
  pi_theta(y_t | s_t)

self-teacher:
  pi_T(y_t | s_t^+)
```

论文使用 reverse-KL aligned 的 sampled-token gap：

```text
Delta_t
  = log pi_T(y_t | s_t^+) - log pi_theta(y_t | s_t)
```

这个式子来自 RKL 的单样本估计。

完整 RKL 是：

```text
D_RKL^(t)
  = D_KL(pi_theta(. | s_t) || pi_T(. | s_t^+))
  = sum_{v in V} pi_theta(v | s_t)
      * log [ pi_theta(v | s_t) / pi_T(v | s_t^+) ]
```

如果只在 student 当前采样 token `y_t ~ pi_theta(. | s_t)` 上估计：

```text
D_hat_RKL^(t)
  = log pi_theta(y_t | s_t) - log pi_T(y_t | s_t^+)
```

取负号就是：

```text
Delta_t
  = - D_hat_RKL^(t)
  = log pi_T(y_t | s_t^+) - log pi_theta(y_t | s_t)
```

含义：

```text
Delta_t > 0:
  teacher 比 student 更支持 student 采样的这个 token。

Delta_t < 0:
  teacher 比 student 更不支持这个 token。
```

## Token-Level Gating

SDAR 把 gap 变成 gate：

```text
g_t = sigmoid(beta * Delta_t)
```

并且 gate 是 detached：

```text
Delta_t = sg(log pi_T(y_t | s_t^+) - log pi_theta(y_t | s_t))
g_t = sg(sigmoid(beta * Delta_t))
```

`beta` 是 sigmoid sharpness：

```text
beta 越大:
  gate 越接近 hard 0/1。

beta 越小:
  gate 越平滑，接近平均蒸馏。
```

论文默认：

```text
beta = 5.0
lambda_SDAR = 0.01
```

为什么 sigmoid 有用：

```text
1. g_t 一定在 (0, 1)，避免 gap 直接作为权重时梯度无界。
2. Delta_t 越大，g_t 越大，保留 teacher 的正向认可。
3. Delta_t 为负时，g_t 变小，但不是 hard mask，属于软衰减。
```

一个具体例子：

```text
beta = 5

token A:
  log pi_T = -1.2
  log pi_S = -2.0
  Delta = 0.8
  g = sigmoid(4.0) = 0.982
  几乎完整蒸馏。

token B:
  log pi_T = -3.0
  log pi_S = -2.0
  Delta = -1.0
  g = sigmoid(-5.0) = 0.0067
  teacher 的负向否定基本被压掉。

token C:
  log pi_T = -2.1
  log pi_S = -2.0
  Delta = -0.1
  g = sigmoid(-0.5) = 0.377
  保留一部分弱监督，不做硬裁剪。
```

## SDAR Loss

每个 token 的 SDAR loss：

```text
ell_t^SDAR
  = g_t * (log pi_T(y_t | s_t^+) - log pi_theta(y_t | s_t))
```

整体：

```text
L_SDAR
  = Agg(ell_t^SDAR)
```

由于 `log pi_T` 和 `g_t` 都 detach，优化 `L_SDAR` 等价于：

```text
L_SDAR
  = C - Agg(g_t * log pi_theta(y_t | s_t))
```

其中：

```text
C = Agg(g_t * log pi_T(y_t | s_t^+))
```

对 `theta` 是常数。

所以梯度是：

```text
grad L_SDAR
  = - Agg(g_t * grad log pi_theta(y_t | s_t))
```

这说明 SDAR 的辅助项本质是：

```text
用 gate 加权的 on-policy behavior cloning / weighted likelihood。
```

它不是全词表 KL，也不需要 teacher 全词表分布。

## Tensor Shape 视角

实现上可以理解成：

```text
B:
  prompt / 任务数量。

G:
  每个任务的 rollout 数。

N = B * G:
  展平后的轨迹数。

L:
  response token 最大长度。
```

主要张量：

```text
input_ids / responses:
  [N, L]

response_mask:
  [N, L]

rewards:
  [N]

advantages:
  [N]

student_log_probs:
  [N, L]
  当前 policy 在 student 上下文 s_t 上对采样 token y_t 的 log prob。

old_log_probs:
  [N, L]
  生成 rollout 的 old policy 对采样 token y_t 的 log prob。

teacher_log_probs:
  [N, L]
  self-teacher 在特权上下文 s_t^+ 上对同一个 y_t 的 log prob。

delta:
  [N, L]

gate:
  [N, L]
```

伪代码：

```python
# GRPO branch
ratio = torch.exp(student_log_probs - old_log_probs)       # [N, L]
adv = advantages[:, None]                                  # [N, 1]

obj_unclipped = ratio * adv                                # [N, L]
obj_clipped = torch.clamp(ratio, 1-eps, 1+eps) * adv        # [N, L]
pg_obj = torch.minimum(obj_unclipped, obj_clipped)          # [N, L]

loss_grpo = -masked_mean(pg_obj, response_mask)
loss_grpo = loss_grpo + beta_kl * ref_kl

# SDAR branch
delta = (teacher_log_probs - student_log_probs).detach()    # [N, L]
gate = torch.sigmoid(beta * delta).detach()                 # [N, L]

loss_sdar_token = gate * (teacher_log_probs.detach() - student_log_probs)
loss_sdar = masked_mean(loss_sdar_token, response_mask)

loss = loss_grpo + lambda_sdar * loss_sdar
```

注意：

```text
teacher_log_probs 用特权上下文计算。
student_log_probs 用普通 context 计算。
两者评估的是同一个 student rollout token。
```

## Skill Retrieval

论文把技能库记为：

```text
E = {e_1, ..., e_M}
```

训练时为每个 task 检索一个 skill `c^+` 给 self-teacher。论文测试了四种检索策略：

```text
UCB Retrieval:
  把 skill 选择看成 multi-armed bandit。

Keyword Matching:
  根据 task description 的关键词匹配类别，取对应 skill。

Full Retrieval:
  提供更完整的 skill context。

Random Retrieval:
  随机给 skill，用来测试鲁棒性。
```

UCB score：

```text
score(e)
  = mean_reward(e) + c * sqrt(ln N_ucb / n(e))
```

其中：

```text
mean_reward(e):
  skill e 历史上被使用时获得的平均 reward。

N_ucb:
  同一 task type 的总检索次数。

n(e):
  skill e 被选择的次数。

c:
  exploration-exploitation 系数。
```

一个重要结果是：

```text
即使 Random Retrieval 也比纯 GRPO 好。
```

这说明收益不完全来自检索质量，而来自门控对噪声 skill 信号的过滤。

## 算法流程

论文 Appendix B 的 SDAR 流程可以概括为：

```text
for each training iteration:
  1. sample batch of tasks x
  2. for each x, retrieve skill c^+
  3. 用当前 policy 做 on-policy rollout，采样 G 条轨迹
  4. 环境执行，得到轨迹级奖励
  5. 组内标准化 reward，得到 A^(i)
  6. 计算 GRPO clipped policy loss
  7. 对同一批 student rollout:
       用特权上下文 (x, c^+, y_<t) 做 teacher forward
       用普通 context (x, y_<t) 做 student forward
       计算 Delta_t 和 gate g_t
       计算 token 级 SDAR loss
  8. 最小化 L_GRPO + lambda * L_SDAR
```

关键点：

```text
rollout 由 student / 当前 policy 产生；
teacher 不单独 rollout；
teacher 只在 student 轨迹上做强制前向计算 / logprob 评估。
```

所以这是 on-policy 自蒸馏，而不是模仿 teacher 轨迹。

## 和几个 baseline 的区别

### GRPO

```text
GRPO:
  只有轨迹级奖励。

SDAR:
  GRPO + token 级带门控的 OPSD 辅助损失。
```

### OPSD

```text
OPSD:
  主要依赖 token 级自蒸馏。
  多轮智能体下容易因为轨迹漂移崩。

SDAR:
  RL 是主损失；
  OPSD 被 gate 控制后只做辅助。
```

### GRPO+OPSD

```text
GRPO+OPSD:
  直接把 OPSD loss 加到 GRPO 上。

SDAR:
  加的是 gated OPSD；
  positive gap 强，negative gap 弱。
```

论文结果显示 naive `GRPO+OPSD` 在 Qwen3-1.7B 上明显退化：

```text
ALFWorld:
  GRPO:      46.1
  GRPO+OPSD: 32.0
  SDAR:      53.9
```

### Skill-GRPO

```text
Skill-GRPO:
  训练 prompt 中直接注入 skill。
  测试时如果不注入 skill，可能掉很多。

SDAR:
  skill 只给 teacher 分支作为训练时特权上下文。
  测试时 student 不需要 skill。
```

这体现了论文强调的 skills internalization。

### RLSD

```text
RLSD:
  用 self-teacher gap 重新加权 RL advantage。

SDAR:
  不改 RL advantage；
  差值只控制辅助蒸馏的门控权重。
```

论文认为 RLSD 在训练早期 teacher-student 不匹配大时容易过度放大更新。

## 实验设置

模型：

```text
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
Qwen3-1.7B-Instruct
```

任务：

```text
ALFWorld:
  text-based household task environment。

Search-QA:
  NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle。

WebShop:
  web shopping interactive environment，使用 128 个 fixed validation tasks。
```

训练：

```text
150 steps
8 H800 GPUs

ALFWorld:
  batch = 16 tasks
  rollouts per prompt = 8
  max prompt length = 2048

Search-QA:
  batch = 128 tasks
  max prompt length = 4096
  retriever = E5

WebShop:
  train tasks = 1000
  batch = 16 tasks
  rollouts per prompt = 8
  max prompt length = 4096
```

主要超参：

```text
learning rate:
  1e-6

group size G:
  8

PPO / GRPO clip ratio:
  0.2

lambda_SDAR:
  0.01

beta:
  5.0

reference KL coefficient:
  0.01

skill retrieval:
  KM by default
```

## 主要结果

论文表 1 很大，这里摘核心平均指标。

### Qwen2.5-3B-Instruct

```text
Method        ALFWorld Avg   Search-QA Avg   WebShop Score   WebShop Acc
GRPO          75.0          36.4            79.8            63.3
GRPO+OPSD     81.2          44.6            77.8            66.4
Skill-SD      73.4          44.1            75.9            64.0
RLSD          79.7          43.8            84.4            66.4
SDAR          84.4          43.4            85.0            68.0
```

相对 GRPO：

```text
ALFWorld:
  +9.4

Search-QA:
  +7.0 左右，论文按多个 QA 子集平均报告。

WebShop Acc:
  +4.7
```

### Qwen2.5-7B-Instruct

```text
Method        ALFWorld Avg   Search-QA Avg   WebShop Score   WebShop Acc
GRPO          81.2          42.0            80.9            72.6
Skill-GRPO*   88.3          47.5            87.0            81.2
GRPO+OPSD     80.4          47.0            86.8            76.5
Skill-SD      85.1          47.8            86.1            76.5
RLSD          82.0          49.0            87.4            77.3
SDAR          85.9          49.0            89.4            82.8
```

SDAR 在 WebShop Acc 上相对 GRPO 提升：

```text
82.8 - 72.6 = +10.2
```

### Qwen3-1.7B-Instruct

```text
Method        ALFWorld Avg   Search-QA Avg   WebShop Score   WebShop Acc
GRPO          46.1          40.8            67.3            38.3
Skill-GRPO    21.1          40.4            73.4            46.1
Skill-GRPO*   28.1          40.7            80.4            50.0
GRPO+OPSD     32.0          42.2            70.7            38.3
Skill-SD      52.3          40.8            81.8            53.9
RLSD          42.2          40.6            74.0            50.8
SDAR          53.9          41.9            76.8            58.6
```

小模型上更能看出 gated distillation 的价值：

```text
Skill-GRPO 直接注入 skill 反而可能伤害 ALFWorld；
GRPO+OPSD 也显著低于 GRPO；
SDAR 最稳。
```

## 鲁棒性：不同 skill retrieval 质量

以 Qwen2.5-7B 为例，论文比较：

```text
Method   ALFWorld     WebShop-Score   WebShop-Acc
UCB      86.8 (+5.6)  87.5 (+6.6)     81.2 (+8.6)
KM       85.9 (+4.7)  89.4 (+8.5)     82.8 (+10.2)
Full     83.2 (+2.0)  87.2 (+6.3)     78.1 (+5.5)
Random   83.1 (+1.9)  82.5 (+1.6)     73.6 (+1.0)
w/o OPSD 81.2         80.9            72.6
```

结论：

```text
检索越好，效果越强；
但即使随机检索 skill，也没有明显崩，仍超过纯 GRPO。
```

这支持论文主张：

```text
SDAR 的门控可以过滤低质量 skill 带来的负向 teacher 信号。
```

## 消融实验

### Gating Strategy

论文比较了：

```text
Entropy gating:
  g_t = sigmoid(beta * h_t)

Gap gating:
  g_t = sigmoid(beta * Delta_t)

Soft-OR gating:
  g_t = sigmoid(beta * [1 - (1 - h_t)(1 - Delta_t)])
```

其中：

```text
h_t = - sum_v pi_theta(v | s_t) log pi_theta(v | s_t)
```

结果：

```text
Teacher-Student Gap gating 最好。
```

原因：

```text
差值直接衡量 teacher 对 student 当前 token 的认可程度；
entropy 只是 student uncertainty 的间接代理；
soft-OR 会降低选择性。
```

### beta

论文 sweep：

```text
beta in {0, 1, 5, 10}
```

结论：

```text
beta = 5 最好。

beta = 0:
  等价于 no gate / uniform distillation，继承 naive OPSD 的不稳定。

beta 太大:
  gate 过于二值化，丢掉 borderline token 的 partial credit。
```

### lambda

论文 sweep：

```text
lambda_SDAR in {0.001, 0.01, 0.1}
```

结论：

```text
lambda = 0.01 最好。

lambda = 0.001:
  辅助信号太弱。

lambda = 0.1:
  distillation gradient 压过 RL signal。
```

这也符合论文的核心设定：

```text
RL 必须是主干；
distillation 只能是受控辅助项。
```

### Distillation Objective

论文比较：

```text
Reverse KL
Forward KL
JSD
```

结果：

```text
Reverse KL 最好。
```

原因：

```text
Reverse KL 是 mode-seeking；
它更倾向于让 student 集中到 teacher 支持的模式上。

Forward KL 是 mode-covering；
它会鼓励 student 覆盖 teacher 分布中的更多模式，
在 teacher 信号弱或噪声较大时更容易引入不可靠指导。

JSD 是折中，但仍带有 mode-covering 的副作用。
```

更实际的一点是：

```text
SDAR 的 sampled-token gap 天然对应 RKL 的单样本估计，
不需要全词表 teacher 分布。
```

## 和前面几篇论文的关系

### 和 Uni-OPD / OPD

Uni-OPD / OPD 的核心是：

```text
student trajectory 上使用 teacher token-level signal，
再通过 outcome reward 修正 teacher supervision。
```

SDAR 更像 OPSD：

```text
teacher 不是独立强模型；
teacher 是同一个模型加特权上下文。
```

SDAR 不用 outcome reward 去判断 teacher token 是否正确，而是：

```text
RL outcome reward 走 GRPO 主干；
teacher-student 差值只决定辅助蒸馏强度。
```

### 和 TrOPD / trust-region OPD

TrOPD 关注 teacher-student distribution mismatch，通常会对不可信 token 做 mask 或 trust-region 处理。

SDAR 的 gate 也有 trust 的味道，但更软：

```text
不是 hard mask；
是 sigmoid gate。
```

并且 SDAR 的不对称性来自特权上下文：

```text
positive gap:
  认为 teacher 的正向认可更可信。

negative gap:
  认为 teacher 的负向否定可能来自 skill 噪声或多轮漂移，因此衰减。
```

### 和 SAO / 异步训推不一致

SAO 关注 rollout policy 和 training policy 之间的 lag / mismatch。

SDAR 关注：

```text
student 普通上下文和 self-teacher 特权上下文之间的 token 偏好差值。
```

它不是异步训练方法，也不是解决 rollout engine / train engine 不一致的技术。

### 和 UP

UP 改的是 GxPO 的 policy-gradient surrogate：

```text
正优势去掉 pi_old ratio 和 upper clip；
负优势保留 clipping。
```

SDAR 不改 GRPO policy loss：

```text
它在 GRPO 外面加一个 gated self-distillation loss。
```

二者作用层不同，理论上可以组合：

```text
UP-GRPO 作为 RL 主干；
SDAR 作为特权上下文辅助蒸馏。
```

### 和 MaxRL

MaxRL 改的是 binary reward 下 advantage / objective 的统计目标，让 RL 更接近 maximum likelihood。

SDAR 不改 reward objective：

```text
它补充 token 级稠密监督。
```

二者也可以组合：

```text
MaxRL-style advantage + SDAR auxiliary distillation。
```

## 我的理解

SDAR 的本质是：

```text
在智能体 RL 里，不直接相信 teacher，也不放弃 teacher。
```

它把 teacher with skills 看成一个弱而有用的辅助信号：

```text
当 teacher 比 student 更支持 student 自己采样的 token，
这通常是值得吸收的知识；

当 teacher 不支持 student token，
这可能是 token 错了，也可能是 skill / teacher / 多轮状态错了，
所以不能强行惩罚。
```

因此它不是“teacher 教 student 做完整轨迹”，而是：

```text
student 仍由 RL 和环境 reward 驱动；
teacher 只在 token-level 给局部、加权、可衰减的辅助监督。
```

从工程上看，这篇论文最重要的设计是三点：

```text
1. teacher 不 rollout，只在 student rollout token 上做强制前向计算。
2. RL advantage 不被 distillation gap 改写。
3. 差值经过 detach 后的 sigmoid 门控，避免稠密监督压过 RL 主信号。
```

这也是为什么它比朴素 `GRPO+OPSD` 稳定：它承认带 skill 上下文的 teacher 是不完美的，只强吸收正向认可，谨慎对待负向否定。
