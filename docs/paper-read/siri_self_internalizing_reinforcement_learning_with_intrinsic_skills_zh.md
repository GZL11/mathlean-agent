# SIRI：Self-Internalizing Reinforcement Learning with Intrinsic Skills 技术解读

本文档解读论文：

```text
SIRI: Self-Internalizing Reinforcement Learning with Intrinsic Skills for LLM Agent Training
https://arxiv.org/abs/2606.02355
```

## 一句话总结

SIRI 解决的是长程智能体训练里的一个现实问题：

```text
技能很有用，但现有技能方法通常依赖外部技能生成器，或者推理时还得挂一个 skill bank。
```

作者想要的是：

```text
训练时可以用 skill，
但最终推理时只用原始 prompt，
不需要额外检索服务，也不需要 runtime memory。
```

所以 SIRI 的核心不是“再造一个技能库”，而是：

```text
先用 RL 热身，
再从自己成功的无技能轨迹里挖技能，
再验证这些技能有没有真实收益，
最后只把有用的 skill-guided action 内化到参数里。
```

## Student / Teacher 模型速览

这篇论文的“teacher”不是一个独立大模型，而是同一个策略在不同上下文下的两种角色：

```text
Student:
  原始无技能上下文下的 policy。

Teacher:
  同一个 policy 在训练期加上 skill context 后的 skill-augmented branch。
```

实验主干模型是：

```text
Qwen2.5-7B-Instruct
Qwen2.5-1.5B-Instruct
```

因此这里更像：

```text
同一个 agent 的 skill-free 版本
和 skill-augmented 版本之间做内部蒸馏。
```

## skills 是哪里的

这是这篇论文最关键的问题之一。

SIRI 的 skills 不是人工手写规则，也不是外部固定老师模型直接吐出来的，而是：

```text
从 agent 自己成功的无技能轨迹里挖出来的。
```

论文把 skill 库记为：

```text
B = {m1, m2, ...}
```

每个 skill 是一个“条件 - 策略”对：

```text
mj = (cj, sj, uj, nj, statej)
```

含义是：

```text
cj:
  这个 skill 适用的条件。

sj:
  可复用的自然语言策略。

uj:
  在线效用估计。

nj:
  被检索次数。

statej:
  candidate / active / retired。
```

论文明确说，这些 skill 是从 **successful skill-free trajectories** 里总结出来的，也就是：

```text
先让 agent 不带 skill 跑出一批成功轨迹，
再把这些轨迹总结成紧凑的自然语言技能。
```

## 核心直觉

SIRI 认为长程智能体里，技能有两个阶段：

```text
1. 训练期：
   skill 可以帮助探索、提高成功率。

2. 推理期：
   skill 不应该成为永久外部依赖。
```

所以它把 skill 从“长期记忆”改成“临时训练信号”。

这带来两个目标：

```text
1. 自己学会挖技能，不依赖外部 skill generator。
2. 把有用 skill-guided behavior 吸收到参数里，推理时不再检索。
```

## 三阶段流程

### Phase 0: Policy Warmup

先用 GiGPO 热身 policy，目的是：

```text
1. 让 agent 具备基本交互能力。
2. 收集足够多的成功无技能轨迹。
```

为什么要这样做：

```text
如果一开始 policy 太弱，
挖出来的 skill 很可能是噪声或偶然行为，
后面整个 skill bank 都会脏掉。
```

### Phase 1: Self-Skill Mining and Validation

这一步分两件事：

```text
1. 从成功的无技能轨迹里挖候选 skill。
2. 用带 skill / 不带 skill 的成对 rollout 验证这个 skill 有没有真实收益。
```

论文定义一个任务组 `g` 的效用差：

```text
Delta_g = R_skill_g - R_base_g
```

其中：

```text
R_skill_g:
  带 skill 的平均回报。

R_base_g:
  不带 skill 的平均回报。
```

然后对每个 skill 做 EMA 更新：

```text
u_j <- alpha * u_j + (1 - alpha) * Delta_g
```

如果某个 skill 的效用持续为正，就晋升为 active；持续为负，就 retire。

这一步的目的很直接：

```text
不是所有自己挖出来的 skill 都值得信。
必须先看它在真实 rollout 里有没有提升整体任务回报。
```

### Phase 2: Advantage-Weighted Skill Internalization

这一步才是真正的“内化”。

流程是：

```text
teacher trajectory:
  在 skill-augmented context 下生成。

student trajectory:
  在 skill-free context 下学习。
```

但不是把整条 teacher 轨迹无脑蒸馏给 student，而是只挑：

```text
1. 带 skill 的轨迹本身是正收益的；
2. 轨迹里真正属于 action span 的 token；
3. 这些 token 还要按 GiGPO 的 advantage 加权。
```

它的 token 权重是：

```text
kappa_{i,t,l}
  = gate * max(A(a_t), 0) / (A_plus + eps)
```

其中 gate 会同时要求：

```text
1. 这条轨迹确实用了 skill；
2. 这个 skill 的整体效用是正的；
3. 这个 token 处于可执行 action span。
```

所以 SIRI 不是“学 teacher 的所有字面输出”，而是：

```text
只吸收那些对任务成功有帮助、且能被 RL 证实有价值的 skill-guided action。
```

## 为什么这样做

作者想解决两个耦合问题：

```text
skill discovery:
  该从哪些轨迹里总结 skill？

skill internalization:
  这些 skill 相关 token 里，哪些真的该进入参数？
```

如果不做验证，skill bank 很容易污染。
如果不做选择性内化，student 可能学到一堆不必要的语言形式，而不是有效动作。

所以 SIRI 的设计逻辑是：

```text
RL credit 不只用来优化动作，
还用来判断 skill 是否有用，以及哪些 skill-guided 行为值得吸收。
```

## 训练目标

总损失：

```text
L_total(theta)
  = L_GiGPO(theta) + beta * L_KL(theta) + lambda(k) * L_int(theta)
```

其中：

```text
L_GiGPO:
  主 RL 损失。

L_KL:
  对 reference policy 的约束。

L_int:
  skill 内化损失。
```

其中 `L_int` 只对被门控选中的 action token 计算交叉熵式蒸馏。

## 训练流程

可以把 Algorithm 1 概括成：

```text
1. 先 warmup，收集成功无技能轨迹。
2. 定期从这些轨迹里挖候选 skill。
3. 用带 skill / 不带 skill 的并行 rollout 算 skill 效用。
4. 晋升有效 skill，淘汰无效 skill。
5. 当 skill 库成熟后，开始只对有收益的 skill-guided action 做内化。
6. 训练结束后丢弃 skill bank。
```

## tensor 视角

可以粗略理解为：

```text
N:
  轨迹数

T:
  一条轨迹的 token 长度

L:
  action token span 长度
```

主要张量：

```text
skill_free_context:
  [N, T]

skill_aug_context:
  [N, T]

trajectory_reward:
  [N]

episode_advantage:
  [N]

step_advantage:
  [N, T]

action_token_mask:
  [N, T]
```

## 实验设置

任务：

```text
ALFWorld
WebShop
```

模型：

```text
Qwen2.5-7B-Instruct
Qwen2.5-1.5B-Instruct
```

基线：

```text
Origin, ReAct, Reflexion, PPO, RLOO, GRPO, GiGPO, MemRL, Evolver, Mem0+GRPO, SimpleMem+GRPO, SkillRL
```

## 主要结果

在 Qwen2.5-7B-Instruct 上：

```text
ALFWorld success:
  GiGPO  0.908
  SIRI   0.930

WebShop success:
  GiGPO  0.728
  SIRI   0.813
```

在 ALFWorld 和 WebShop 上，SIRI 还超过了 SkillRL、MemRL 这类带长期记忆或 skill bank 的方法。

## 消融结论

### 去掉 Phase 0

效果明显下降。

原因：

```text
太早挖 skill，噪声很大。
```

### 只保留 skill，不做内化

比纯 GiGPO 好，但不如完整 SIRI。

说明：

```text
skill 确实有用，
但还得把有用部分吸收到参数里。
```

### 不用 skill

性能会掉回接近 GiGPO。

说明：

```text
skill bank 和内化机制都不是装饰项。
```

### 用外部大模型挖 skill

前期更强，但后期自挖 skill 会逐渐追上。

这说明：

```text
SIRI 的自挖技能不是摆设，
而是会随着训练变好。
```

## 这篇论文的核心创新

我会把它压成三句话：

```text
1. 技能不是外部记忆，而是训练期信号。
2. 技能不是先验真理，而是要用 rollout 效用验证。
3. 最终要蒸馏的是有效 action，而不是整条带 skill 的轨迹。
```

## 和前面几篇的关系

它和 Uni-OPD / TrOPD / SDAR / UP 这些论文不同：

```text
Uni-OPD:
  用 outcome reward 校准 teacher token supervision。

TrOPD:
  用 trust region 过滤 teacher/student 分布差太远的 token。

SDAR:
  用 gated self-distillation 把 skill-guided token 内化。

UP:
  改的是 GxPO policy loss 的 positive branch。

SIRI:
  解决的是“技能从哪来、怎么验证、怎么内化、怎么去掉推理期依赖”。
```

## 我的理解

这篇论文真正有价值的地方不是“又一个 skill bank”，而是把 skill 重新定义成：

```text
训练期可验证、可淘汰、可内化的临时中间产物。
```

它的逻辑链条很完整：

```text
先让 agent 自己变强；
再从成功轨迹里提炼 skill；
再看这些 skill 是否真提升回报；
最后只把真正有帮助的 action 行为吸收到无 skill policy 里。
```

这样最终部署时就不会带着一堆检索组件跑推理。

