# AgentOPSD：Recursive Self-Distillation for Agentic Reinforcement Learning 论文解读

论文：

```text
AgentOPSD: Recursive Self-Distillation for Agentic Reinforcement Learning
```

链接：

```text
https://arxiv.org/abs/2608.05987
https://arxiv.org/pdf/2608.05987
```

作者单位：

```text
Tsinghua University
Zhejiang University
Meituan Longcat
```

代码：

```text
https://github.com/ZethWang/AgentOPSD
```

## 一句话总结

AgentOPSD 解决的是长程 agent RL 里的 turn-level credit assignment 问题：

```text
GRPO 只有 trajectory-level advantage，
OPSD 有 token-level teacher-student gap，
但 agent 环境真正发生状态转移的是 turn/action 边界。
```

它的核心不是再加一个普通 OPD loss，而是：

```text
把 token-level teacher-student gap 聚合成 turn-level evidence，
再用递归 Bayesian belief update(贝叶斯后验更新) 计算每个 turn 对最终成功概率信念的边际修正，
最后用这个 turn credit 重塑 GRPO advantage。
```

所以 AgentOPSD 可以理解成：

```text
GRPO + recursive turn-level advantage reshaping from OPSD evidence
```

注意一个关键点：

```text
AgentOPSD 不引入单独的 distillation loss。
teacher-student gap 是 detach 的，只通过 reshaped advantage 影响 GRPO。
```

## 术语表

| 英文术语 | 建议中文 | 含义 |
|---|---|---|
| AgentOPSD | 面向 Agent 的递归 OPSD | 把 OPSD gap 转成 turn-level credit 的方法 |
| turn-level credit assignment | 轮级信用分配 | 判断每一轮 action 对最终成功有多大贡献 |
| recursive belief update | 递归信念更新 | 用每轮 evidence 更新“最终会成功”的信念状态 |
| belief state `B_k` | 信念状态 | 到第 `k` 轮为止，trajectory 支持成功的相对概率 |
| log-odds space | 对数胜算空间 | `logit(B)=log(B/(1-B))`，便于累加 evidence |
| turn-level evidence `e_k` | 轮级证据 | 一轮 action 内 token-level gap 的总和 |
| marginal revision `Delta B_k` | 边际信念修正 | 第 `k` 轮让成功信念变化了多少 |
| bounded advantage reshaping | 有界优势重塑 | 用 turn credit 乘到 GRPO advantage，但限制 multiplier 范围 |
| privileged skill `c+` | 特权技能上下文 | 训练期给 teacher branch 的 retrieved skill，推理不用 |

## 背景：为什么 GRPO / OPSD 都不够

Agentic RL 是多轮交互：

```text
turn k:
  state/history s_k
  agent action a_k = tokens y_{k,1:L_k}
  environment returns observation o_k
```

一条 episode 最后只有 binary reward：

```text
R(tau) in {0, 1}
```

GRPO 对同一个 task 采样 `G` 条 trajectories，然后算 sequence-level advantage：

```text
A_seq^(i) = (R^(i) - R_bar) / (sigma_R + epsilon_0)
```

问题是：

```text
同一个 trajectory 里的所有 turn/token 都继承同一个 A_seq。
```

这会把关键 action、冗余 action、甚至误导 action 混在一起。

普通 OPSD / SDAR 则提供 token-level teacher-student gap，但 agent 的环境反馈发生在 turn/action 边界：

```text
多个 token 共同组成一个 action；
环境只在 action 后转移。
```

所以 AgentOPSD 的问题定义是：

```text
如何把 token-level OPSD gap 变成 history-dependent turn-level credit？
```

## Teacher / Student 是什么

student branch：

```text
h_{k,t} = (s_k, y_{k,<t})
```

teacher branch：

```text
h^+_{k,t} = (s_k, c^+, y_{k,<t})
```

其中：

```text
c^+ = training-only retrieved skill
```

论文说明这些 skill 来自 SkillRL 的 SkillBank，通过 keyword matching 检索；训练时给 teacher branch，推理时不使用 external skills。

关键点：

```text
student 和 teacher share parameters theta；
teacher 不是单独 rollout；
teacher 只在 student-generated action/token 上打 logprob；
teacher 比 student 多看到 c+。
```

所以 AgentOPSD 的 teacher 是：

```text
same policy + privileged skill context
```

## 1. Token-level gap

对第 `k` 轮第 `t` 个 student-generated token：

```text
delta_{k,t}
= log pi_theta(y_{k,t} | h^+_{k,t})
  - log pi_theta(y_{k,t} | h_{k,t})
```

中文：

```text
教师-学生 token 对数概率差
```

含义：

```text
delta_{k,t} > 0:
  加入 skill context 后，模型更支持 student 已生成的 token。

delta_{k,t} < 0:
  加入 skill context 后，模型不支持这个 token。
```

注意：

```text
delta detach，只作为 credit evidence，不直接回传 teacher 梯度。
```

## 2. Turn-level evidence

把一轮 action 里的 token gaps 求和：

```text
e_k = sum_{t=1}^{L_k} delta_{k,t}
```

也可写成 action-level likelihood ratio：

```text
e_k
= log pi_theta(a_k | s_k, c^+) / pi_theta(a_k | s_k)
```

解释：

```text
e_k > 0:
  skill-conditioned branch 更认为这个 turn/action 像成功行为。

e_k < 0:
  skill-conditioned branch 不支持这个 turn/action。
```

这一步是 AgentOPSD 和 token-level OPSD 的第一个关键区别：

```text
AgentOPSD 不直接拿 token gap 做 loss；
而是先按环境 turn/action 边界聚合。
```

## 2.1 为什么这个 gap 可以当作 Bayesian evidence

论文先定义事件：

```text
C = trajectory eventually succeeds (轨迹最终获得成功)
```

理想情况下，第 `k` 轮 action 对成功信念的更新是一个 Bayes factor：

```text
logit p(C | s_k, a_k) - logit p(C | s_k)
= log [ p(a_k | s_k, C) / p(a_k | s_k, not C) ]
```

中文：

```text
Bayes factor = 贝叶斯因子 / 贝叶斯证据比
```

它问的是：

```text
这个 action 更像成功轨迹里的行为，还是更像失败轨迹里的行为？
```

但 `p(a_k | s_k, C)` 和 `p(a_k | s_k, not C)` 没法直接拿到，所以论文用 self-teacher contrast (对比) 近似：

```text
e_k = log [ pi_theta(a_k | s_k, c+) / pi_theta(a_k | s_k) ]
```

这里有两个近似假设：

```text
A1. skill-conditioned branch 近似 success-conditional branch：
    pi_theta(a_k | s_k, c+) ≈ p(a_k | s_k, C)

A2. 当当前成功概率 rho_k = p(C | s_k) 较小时，
    普通 policy 的边际分布更接近 failure-conditioned branch：
    pi_theta(a_k | s_k) ≈ p(a_k | s_k, not C)
```

更一般地，在只用 A1 的情况下：

```text
e_k ≈ log [ p(a_k | s_k, C) / p(a_k | s_k) ]
    = log [ p(C | s_k, a_k) / p(C | s_k) ]
```

这不是完整 Bayes factor，而是 pointwise mutual information：

```text
pointwise mutual information, PMI = 点互信息
```

含义：

```text
e_k > 0:
  观察到这个 action 后，成功概率上升。

e_k < 0:
  观察到这个 action 后，成功概率下降。
```

论文强调：`e_k` 不一定是校准后的真实成功概率增量，但它保留了证据方向和相对强弱。因此后续只把它当作 belief update 的 evidence proxy，而不是直接当 reward 或 distillation target。

## 3. Bayesian belief update

AgentOPSD 把 `e_k` 看成一轮新 evidence，用来更新“trajectory 最终成功”的 belief。

先用 group success rate 初始化 prior（先验）：

```text
B_0 = clip(R_bar, epsilon_0, 1 - epsilon_0)
```

其中：

```text
R_bar = S / G
```

也就是一个 GRPO group 里成功 trajectories 的比例。

再维护 evidence accumulator：

```text
c_0 = 0
c_k = gamma * c_{k-1} + e_k
```

`gamma` 是 evidence decay：

```text
gamma = 0.95
```

然后在 log-odds space 里更新：

```text
ell_k = logit(B_0) + c_k
B_k = sigmoid(ell_k)
```

其中：

```text
logit(u) = log(u / (1-u))
```

直觉：

```text
B_k 不是校准后的真实成功概率；
它是“到第 k 轮为止，这条 trajectory 对成功的相对支持程度”。
```

## 4. Marginal revision：每轮到底贡献多少

第 `k` 轮 credit 不是直接用 `e_k`，而是用 belief revision：

```text
Delta B_k = B_k - B_{k-1}
```

论文给了近似：

```text
Delta B_k
≈ B_{k-1}(1 - B_{k-1}) * [ e_k - (1-gamma)c_{k-1} ]
```

这个式子很关键：

```text
1. e_k 是当前 turn 的新 evidence；
2. (1-gamma)c_{k-1} 是历史 evidence 衰减后释放出来的 carry-over；
3. B_{k-1}(1-B_{k-1}) 是 state sensitivity。
```

所以同样大小的 `e_k`，在不同历史状态下作用不同：

```text
B 接近 0.5:
  outcome 仍不确定，这轮 evidence 影响最大。

B 接近 0 或 1:
  belief 已饱和，这轮 evidence 影响变小。
```

这就是论文说的：

```text
local gap is not sequential credit
```

局部 gap 只有放到历史 belief 里看，才知道是不是 pivotal turn。

## 5. Outcome-aligned signed credit

用 terminal outcome 的 GRPO advantage 决定整体方向：

```text
q_k = sign(A_seq) * Delta B_k
```

含义：

```text
成功 trajectory:
  upward belief revision 是 outcome-consistent，应增加 credit。

失败 trajectory:
  upward belief revision 反而和最终失败矛盾，应减少 credit。
```

这一步很重要：

```text
AgentOPSD 不让 self-teacher 反转 GRPO 的最终方向；
它只根据 belief revision 重新分配同一条 trajectory 内不同 turns 的更新强度。
```

## 6. Bounded advantage reshaping

对同一条 trajectory 内的 `q_k` 做标准化：

```text
mu_q = mean_k q_k
sigma_q = std_k q_k
z_k = (q_k - mu_q) / (sigma_q + epsilon_0)
```

构造有界 multiplier：

```text
w_k = clip(1 + b z_k, 1-b, 1+b)
```

论文设置：

```text
b = 0.2
```

所以：

```text
w_k in [0.8, 1.2]
```
w_k 可以理解成从 OPD / self-teacher evidence 得到的、归一化并 clip 后的 turn-level OPD weight

最后重塑 sequence advantage：

```text
A_tilde_k = A_seq * [ (1-lambda) + lambda w_k ]
```

论文默认：

```text
lambda = 0.5
```

所以 multiplier 范围实际是：

```text
(1-lambda) + lambda*w_k
in [0.9, 1.1]
```

这保证：

```text
AgentOPSD 只调整每个 turn 的更新强度；
不会反转 GRPO 的正负方向。
```

## 6.1 这个 reshaping 有哪些理论性质

论文在 appendix 里给了几个性质。

### 1. Boundedness：有界性

因为：

```text
w_k in [1-b, 1+b]
A_tilde_k = A_seq * [(1-lambda) + lambda*w_k]
```

所以：

```text
|A_tilde_k - A_seq| <= lambda*b*|A_seq|
```

也就是：

```text
(1-lambda*b)|A_seq| <= |A_tilde_k| <= (1+lambda*b)|A_seq|
```

默认 `lambda=0.5, b=0.2` 时：

```text
0.9|A_seq| <= |A_tilde_k| <= 1.1|A_seq|
```

所以它是一个很保守的 turn-level reweighting，不会把 GRPO 更新幅度无限放大。

### 2. Sign preservation：符号保持

因为：

```text
(1-lambda) + lambda*w_k >= 1 - lambda*b > 0
```

所以：

```text
sign(A_tilde_k) = sign(A_seq)
```

这就是为什么论文说 verifier / terminal reward 仍然决定全局优化方向：

```text
成功轨迹仍然整体增强；
失败轨迹仍然整体压低；
AgentOPSD 只决定哪一轮更该多更新或少更新。
```

### 3. Recovery of GRPO：退化回 GRPO

当：

```text
lambda = 0
```

则：

```text
A_tilde_k = A_seq
```

所以 AgentOPSD 完全退化为 GRPO。

### 4. Exact budget：信念修正可望远镜求和

因为：

```text
Delta B_k = B_k - B_{k-1}
```

所以：

```text
sum_k Delta B_k = B_K - B_0
```

这说明每个 turn 的 revision 是把整条 trajectory 的 belief change 拆成逐轮贡献。

## 7. 最终训练 loss

每个 token 继承自己所在 turn 的 `A_tilde_k`：

```text
token t in turn k -> A_tilde_k
```

然后进入 GRPO clipped objective：

```text
L_AgentOPSD(theta)
= - 1/G sum_i 1/sum_t M_{i,t}
    sum_t M_{i,t}
      min(
        r_{i,t} A_tilde_{kappa_i(t)}^{(i)},
        clip(r_{i,t}, 1-epsilon, 1+epsilon)
          A_tilde_{kappa_i(t)}^{(i)}
      )
  + beta_KL L_KL
```

其中：

```text
r_{i,t}
= pi_theta(y_{i,t} | h_{i,t})
  / pi_old(y_{i,t} | h_{i,t})
```

论文明确写道：

```text
No separate distillation loss is introduced.
```

也就是说：

```text
AgentOPSD 不是 L_GRPO + lambda * L_OPSD；
而是用 OPSD evidence 计算 turn-level reshaped advantage，
再只优化一个 reshaped GRPO objective。
```

## 一个具体计算例子

假设一个 task 的 group size：

```text
G = 8
成功数 S = 4
R_bar = 4/8 = 0.5
B_0 = 0.5
logit(B_0) = 0
```

某条成功 trajectory 有 3 个 turns，GRPO sequence advantage：

```text
A_seq = +1.0
```

每轮 token gap 聚合后得到：

```text
e_1 = 0.6
e_2 = 0.1
e_3 = 0.8
gamma = 0.95
```

递归 evidence：

```text
c_1 = 0.95*0 + 0.6 = 0.6
ell_1 = 0 + 0.6 = 0.6
B_1 = sigmoid(0.6) = 0.646

c_2 = 0.95*0.6 + 0.1 = 0.67
ell_2 = 0.67
B_2 = sigmoid(0.67) = 0.662

c_3 = 0.95*0.67 + 0.8 = 1.4365
ell_3 = 1.4365
B_3 = sigmoid(1.4365) = 0.808
```

边际 belief revision：

```text
Delta B_1 = 0.646 - 0.500 = 0.146
Delta B_2 = 0.662 - 0.646 = 0.016
Delta B_3 = 0.808 - 0.662 = 0.146
```

因为这是成功 trajectory：

```text
q_k = sign(A_seq) * Delta B_k = Delta B_k
```

所以：

```text
q = [0.146, 0.016, 0.146]
```

这说明：

```text
turn 1 和 turn 3 都显著提升了成功 belief；
turn 2 虽然 e_2 为正，但在历史 belief 下几乎没有带来新增修正，可能是 routine/redundant turn。
```

对 `q` 做 trajectory 内标准化，假设得到：

```text
z ≈ [0.71, -1.41, 0.71]
```

用 `b=0.2`：

```text
w = clip(1 + 0.2*z, 0.8, 1.2)
  ≈ [1.142, 0.800, 1.142]
```

用 `lambda=0.5`：

```text
A_tilde_k = A_seq * [(1-0.5) + 0.5*w_k]
```

得到：

```text
A_tilde ≈ [1.071, 0.900, 1.071]
```

结论：

```text
turn 1 / turn 3 的所有 tokens 被增强更新；
turn 2 的所有 tokens 被削弱更新；
整体正方向不变，仍然是成功 trajectory 的正优势更新。
```

如果这条 trajectory 是失败的，`A_seq < 0`，则：

```text
q_k = sign(A_seq) * Delta B_k
```

会把“与最终失败相矛盾的 upward belief revision”降权，从而避免把失败轨迹里的误导 turn 强化。

## 实验设置

Benchmarks：

```text
ALFWorld
Search-QA
WebShop
```

Search-QA 包括：

```text
single-hop: NQ, TriviaQA, PopQA
multi-hop: HotpotQA, 2Wiki, MuSiQue, Bamboogle
```

其中：

```text
NQ 和 HotpotQA 是 in-domain；
其他是 held-out。
```

Models：

```text
Qwen2.5-3B-Instruct
Qwen2.5-7B-Instruct
```

Privileged skills：

```text
SkillBank from SkillRL
keyword matching retrieval
training only
no skills at inference
```

## Baselines

论文对比三类 baseline：

| 类型 | 方法 | 含义 |
|---|---|---|
| Training-free | Vanilla | base model |
| Training-free | Skill-Prompt* | 推理时直接 prepend retrieved skills |
| Group-relative RL | GRPO | 标准 trajectory-level advantage |
| Group-relative RL | Skill-GRPO | 训练 prompt 里直接注入 skills |
| Group-relative RL | Skill-GRPO* | 训练和验证都使用 skills |
| Self-distillation RL | OPSD | 纯 on-policy self-distillation |
| Self-distillation RL | GRPO+OPSD | GRPO 加朴素 OPSD |
| Self-distillation RL | Skill-SD | skill-conditioned teacher distillation |
| Self-distillation RL | RLSD | 用 teacher-student gap 缩放 token GRPO update |
| Self-distillation RL | SDAR | GRPO 外加 gated auxiliary distillation loss |
| Self-distillation RL | StepOPSD | turn/step-level local gap，但不递归更新 belief |

## 完整主结果

论文主表报告：

```text
ALFWorld: 六类任务成功率 + Avg
Search-QA: 七个 QA 子集准确率 + Avg
WebShop: Score / Acc
```

`*` 表示 validation / testing 时也给 retrieved skills；没有 `*` 的方法推理时不使用 skills。

### Qwen2.5-3B-Instruct

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | ALF Avg | NQ | Triv | Pop | Hotp | 2Wk | MuS | Bam | Search Avg | WebShop Score | WebShop Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 44.4 | 11.1 | 6.2 | 15.4 | 28.6 | 12.5 | 21.9 | 24.6 | 48.1 | 31.0 | 26.3 | 25.3 | 7.2 | 59.7 | 31.7 | 6.7 | 0.8 |
| Skill-Prompt* | 51.7 | 66.7 | 48.4 | 0.0 | 4.3 | 10.0 | 28.9 | 23.7 | 46.2 | 30.6 | 24.4 | 22.1 | 7.5 | 12.5 | 23.9 | 0.2 | 0.8 |
| OPSD | 48.8 | 41.7 | 16.7 | 0.0 | 15.8 | 16.7 | 28.1 | 0.1 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 11.3 | 3.1 |
| GRPO | 91.2 | 62.5 | 96.2 | 61.9 | 65.0 | 47.4 | 75.0 | 39.3 | 60.6 | 41.1 | 37.4 | 34.6 | 15.4 | 26.4 | 36.4 | 79.8 | 63.3 |
| Skill-GRPO | 88.9 | 71.4 | 58.8 | 70.6 | 40.7 | 29.2 | 60.2 | 43.5 | 58.8 | 43.0 | 36.8 | 32.2 | 11.7 | 12.5 | 34.1 | 77.3 | 60.9 |
| Skill-GRPO* | 94.3 | 57.1 | 100.0 | 66.7 | 73.1 | 57.1 | 80.5 | 44.3 | 59.6 | 44.3 | 39.0 | 36.1 | 14.5 | 14.9 | 36.1 | 76.3 | 66.4 |
| GRPO+OPSD | 100.0 | 82.4 | 85.7 | 75.0 | 70.0 | 60.0 | 81.2 | 44.9 | 61.2 | 45.2 | 40.4 | 38.5 | 16.0 | 66.1 | 44.6 | 77.8 | 66.4 |
| Skill-SD | 88.2 | 50.0 | 96.2 | 52.4 | 65.0 | 57.9 | 73.4 | 44.4 | 60.4 | 44.0 | 39.5 | 40.4 | 15.4 | 64.9 | 44.1 | 75.9 | 64.0 |
| RLSD | 87.9 | 75.0 | 90.9 | 75.0 | 73.1 | 68.4 | 79.7 | 41.5 | 58.6 | 42.3 | 40.4 | 40.2 | 16.8 | 66.9 | 43.8 | 84.4 | 66.4 |
| SDAR | 97.1 | 62.5 | 100.0 | 61.9 | 75.0 | 84.2 | 84.4 | 44.8 | 58.1 | 44.3 | 38.6 | 36.2 | 15.7 | 66.1 | 43.4 | 85.0 | 68.0 |
| StepOPSD | 82.4 | 66.7 | 82.6 | 52.2 | 73.7 | 75.0 | 73.4 | 43.6 | 61.2 | 43.8 | 39.2 | 38.1 | 15.8 | 64.5 | 43.7 | 82.4 | 66.4 |
| AgentOPSD | 100.0 | 68.8 | 86.4 | 66.7 | 84.6 | 73.7 | 84.4 | 44.9 | 61.9 | 48.0 | 40.9 | 41.6 | 14.4 | 68.1 | 46.7 | 90.4 | 69.5 |

### Qwen2.5-7B-Instruct

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | ALF Avg | NQ | Triv | Pop | Hotp | 2Wk | MuS | Bam | Search Avg | WebShop Score | WebShop Acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 36.1 | 22.2 | 3.1 | 0.0 | 0.0 | 0.0 | 12.5 | 25.2 | 50.8 | 29.5 | 29.0 | 29.0 | 10.4 | 63.7 | 33.9 | 5.9 | 1.6 |
| Skill-Prompt* | 51.7 | 50.0 | 32.3 | 5.3 | 4.3 | 0.0 | 23.4 | 30.9 | 52.1 | 32.7 | 32.7 | 27.9 | 12.7 | 66.1 | 36.4 | 1.7 | 0.8 |
| OPSD | 50.0 | 60.0 | 22.7 | 21.4 | 17.6 | 9.5 | 32.8 | 8.8 | 8.6 | 17.5 | 2.5 | 4.2 | 0.5 | 1.2 | 6.2 | 4.5 | 2.3 |
| GRPO | 91.2 | 87.5 | 96.2 | 81.0 | 65.0 | 57.9 | 81.2 | 45.1 | 63.7 | 44.0 | 43.6 | 43.2 | 16.8 | 37.6 | 42.0 | 80.9 | 72.6 |
| Skill-GRPO | 88.5 | 66.7 | 65.2 | 61.1 | 57.7 | 73.1 | 69.5 | 45.2 | 63.7 | 45.7 | 43.1 | 43.3 | 19.6 | 21.4 | 40.3 | 80.4 | 71.9 |
| Skill-GRPO* | 100.0 | 83.3 | 96.4 | 83.3 | 75.0 | 78.9 | 88.3 | 44.8 | 63.0 | 45.1 | 43.7 | 43.7 | 20.5 | 71.4 | 47.5 | 87.0 | 81.2 |
| GRPO+OPSD | 91.4 | 61.5 | 100.0 | 87.5 | 76.5 | 52.2 | 80.4 | 47.3 | 64.5 | 46.9 | 43.8 | 39.3 | 18.0 | 69.4 | 47.0 | 86.8 | 76.5 |
| Skill-SD | 93.9 | 93.8 | 90.9 | 100.0 | 69.2 | 68.4 | 85.1 | 47.1 | 64.5 | 47.8 | 44.2 | 42.1 | 20.2 | 69.0 | 47.8 | 86.1 | 76.5 |
| RLSD | 100.0 | 87.5 | 92.3 | 58.8 | 80.0 | 65.2 | 82.0 | 46.8 | 63.0 | 44.4 | 45.5 | 48.9 | 21.5 | 73.0 | 49.0 | 87.4 | 77.3 |
| SDAR | 94.7 | 75.0 | 100.0 | 86.7 | 68.2 | 78.9 | 85.9 | 46.3 | 63.5 | 48.2 | 43.8 | 48.4 | 19.6 | 73.0 | 49.0 | 89.4 | 82.8 |
| StepOPSD | 98.1 | 75.0 | 100.0 | 90.5 | 80.0 | 63.2 | 88.4 | 45.3 | 64.6 | 45.1 | 44.5 | 44.4 | 19.3 | 69.8 | 48.2 | 87.2 | 78.1 |
| AgentOPSD | 91.2 | 87.5 | 100.0 | 90.5 | 75.0 | 84.2 | 89.1 | 47.5 | 64.9 | 46.8 | 45.8 | 45.3 | 20.9 | 70.2 | 49.2 | 90.2 | 79.7 |

主结果结论：

```text
1. AgentOPSD 在 8 个 aggregate comparisons 中全部超过 GRPO+OPSD、Skill-SD、RLSD。
2. 相比 SDAR，AgentOPSD 在 8 个 aggregate comparisons 中赢 6 个。
3. Qwen2.5-7B 上 ALFWorld 达到 89.1，是主表最高。
4. Qwen2.5-3B 上 WebShop Score/Acc 达到 90.4 / 69.5，是主表最高。
5. Search-QA 上提升较小，因为 Search-QA 最大 interaction turns 只有 4，长程 credit assignment 的作用空间更小。
```

## Horizon robustness（长程任务稳定性）

衡量：任务交互轮数变长时，方法性能下降得有多快。
论文在 ALFWorld / Qwen2.5-7B 上做了 horizon sensitivity：

这里的 horizon 就是 agent 完成任务需要的平均 turn/action 数。长程任务里，GRPO 这种 trajectory-level advantage 会越来越难做 credit assignment，因为一条轨迹里 step 越多，同一个成败 reward 要广播给越多 action，误分配越严重。
```text
success points lost per extra turn
每增加一轮交互损失的成功率百分点
```

越接近 0，说明越不怕任务变长。

关键数值：

| Method | success pts per extra turn |
|---|---:|
| AgentOPSD | -0.54 |
| GRPO | -2.91 |
| RLSD | -3.59 |

结论：

```text
AgentOPSD 是最平的，说明长程任务里 turn-level recursive credit 比 uniform trajectory advantage 更稳。
```

## 消融实验

ALFWorld / Qwen2.5-7B：

| Variant | 改动 | Success |
|---|---|---:|
| AgentOPSD full | turn-level, bounded, lambda=0.5 | 89.1 |
| Turn-level granularity | 换成 per-token accumulation | 85.9 |
| Recursive state revision（递归状态修订） | 用 raw local gap `e_k` 替代 `Delta B_k` | 82.8 |
| Signed direction | 只用 `|Delta B_k|`，去掉 outcome sign | 80.5 |
| State prior `B_0` anchor | 去掉 empirical-rate initialization（经验速率初始化） | 78.9 |

逐项解释：

### 1. Turn-level granularity

```text
turn-level -> per-token accumulation
89.1 -> 85.9
```

说明：

```text
agent 环境反馈和状态转移发生在完整 action/turn 后；
token-level accumulation 会把一个决策拆碎，和环境边界错位。
```

### 2. Recursive state revision

```text
Delta B_k -> raw local gap e_k
89.1 -> 82.8
```

说明：

```text
raw e_k 只看当前 turn；
Delta B_k 看当前 turn 如何修正已有历史 belief。
```

这验证了论文核心：

```text
local gap is not sequential credit.
```

### 3. Signed direction

```text
q_k = sign(A_seq) Delta B_k
改成只用 |Delta B_k|
89.1 -> 80.5
```

说明：

```text
belief 变化大小只能说明“这里重要”；
但不知道它和最终成功/失败是否一致。
```

例如：

```text
失败 trajectory 中某一轮让 success belief 上升，
这其实和最终 outcome 矛盾，不能被当成好 turn 强化。
```

### 4. State prior B0 anchor

```text
去掉 B0 = group success rate
89.1 -> 78.9
```

说明：

```text
group success rate 提供了任务难度 / 初始成功概率的 verifier-grounded anchor。
```

没有 `B0`，所有 trajectory 从任意 uncertainty level 开始，早期 turn 的 belief revision 会被错误缩放。

## 超参数

主实验统一设置：

| Hyperparameter | Value | 含义 |
|---|---:|---|
| learning rate `eta` | `1e-6` | 学习率 |
| group size `G` | 8 | 每个 task 的 rollout 数 |
| PPO clip | `0.2 / 0.24` | `epsilon_low / epsilon_high` |
| reference KL `alpha_KL` | 0.01 | 对 reference policy 的 KL penalty |
| reshaping weight `lambda` | 0.5 | turn reshaping 强度 |
| multiplier band `b` | 0.2 | `w_k in [0.8, 1.2]` |
| evidence decay `gamma` | 0.95 | 历史 evidence 几何衰减 |
| skill retrieval | KM | keyword matching |

环境设置：

| Env | Steps | Batch | G | Max prompt | Max response | Max turns | GPUs |
|---|---:|---:|---:|---:|---:|---:|---:|
| ALFWorld | 150 | 16 | 8 | 2048 | 512 | 50 | 8 |
| WebShop | 150 | 16 | 8 | 4096 | 512 | 15 | 2 |
| Search-QA | 150 | 128 | 8 | 4096 | 512 | 4 | 4 |

超参敏感性：

```text
lambda in {0.5, 0.25, 0.1, 0.01}
gamma in {1.0, 0.95, 0.9, 0.8}
epsilon_high in {0.2, 0.24, 0.28}
```

结论：

```text
lambda 最敏感，0.5 最好；
gamma 不太敏感，0.95 作为 mild decay；
epsilon_high 影响较小，说明继承了 GRPO trust-region robustness。
```

## 和 SDAR / StepOPSD 的区别

### 和 SDAR

SDAR：

```text
L = L_GRPO + lambda * L_SDAR
```

也就是：

```text
GRPO 主损失 + gated auxiliary distillation loss
```

AgentOPSD：

```text
没有单独 distillation loss；
OPSD gap 只用于重塑 GRPO advantage。
```

所以：

```text
SDAR 把 gap 当 token-level auxiliary supervision；
AgentOPSD 把 gap 当 turn-level credit evidence。
```

### 和 StepOPSD

StepOPSD：

```text
把 teacher-student signal 聚合到 step/turn，
但每个 turn 仍然按 local gap 独立打分。
```

AgentOPSD：

```text
先聚合成 e_k，
再递归更新 B_k，
用 Delta B_k 衡量这个 turn 在历史上下文中是否 pivotal。
```

所以 AgentOPSD 比 StepOPSD 多了：

```text
history-dependent recursive belief revision
```

## 一句话结论

AgentOPSD 的核心贡献是：

```text
不要把 OPSD gap 直接当 token/turn credit；
要先按 agent turn 聚合，再看它如何递归修正“最终会成功”的 belief。
```

它的实验说明：

```text
长程 agent 任务中，真正有效的不是更多 teacher signal，
而是把 teacher signal 变成和环境 turn 边界、历史 evidence、最终 outcome 一致的 credit。
```

