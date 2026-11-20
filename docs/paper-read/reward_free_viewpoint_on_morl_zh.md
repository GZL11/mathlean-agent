# A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning 技术解读

本文档解读论文：

```text
A Reward-Free Viewpoint on Multi-Objective Reinforcement Learning
https://arxiv.org/abs/2604.24532
```

## 一句话总结

这篇论文不是 LLM 论文，而是 **多目标强化学习（MORL）** 论文。  
它的核心想法是：

```text
把 reward-free RL（RFRL）的 Forward-Backward 框架迁移到 MORL，
再用偏好引导探索和辅助 Q loss，提升样本效率和偏好泛化。
```

换句话说，它不是单纯做多目标奖励加权，也不是单纯做 Pareto front 搜索，而是：

```text
用“能适配任意奖励函数”的 RFRL 视角，来增强多目标策略学习。
```

## 论文想解决什么

MORL 的难点很直接：

```text
训练时能看到多维奖励，
测试时用户偏好 λ 不知道，
所以要学出一组能适配不同偏好的策略。
```

传统做法一般是：

```text
1. 为每个偏好单独训一套策略
2. 或者训一个 preference-conditioned policy
```

问题是：

```text
1. 数据效率不高
2. 对训练时没见过的偏好泛化一般
3. 在更高维目标下容易退化
```

作者的判断是：

```text
RFRL 其实天然适合 MORL，
因为它本来就是“学任意 reward function 的最优策略”。
```

## 核心设定

MORL 形式化成多目标 MDP：

```text
(S, A, P, R, γ, μ)
```

其中 `R(s,a)` 是 `d` 维奖励向量。  
测试时给定偏好向量 `λ`，目标变成线性标量化：

```text
f_λ(r) = λ^T r
```

所以最终要学的是：

```text
π(s, λ) -> action distribution
```

也就是一个 **偏好条件策略**。

## 方法主线

### 1. 用 Forward-Backward 作为 RFRL 主体

作者选用的是 state-of-the-art 的 FB-RL 作为底座。

FB 把最优 Q 函数分解成两个网络：

```text
Q(s, a, z_R) = F_θ(s, a, z_R)^T z_R
```

其中：

```text
F_θ : forward representation
B_ω : backward representation
z_R : reward-conditioned latent vector
```

直观上：

```text
z_R 表示“当前 reward function 对应的策略方向”
```

给定一个奖励函数 `R`，FB 用回放缓冲区中的样本来构造：

```text
z_R = E_{(s,a)~D}[ B_ω(s,a) R(s,a) ]
```

然后 greedy policy 取：

```text
π(s, z_R) = argmax_a F_θ(s,a,z_R)^T z_R
```

### 2. MORL 里不要只用纯 reward-free 的 z

纯 RFRL 的问题是：

```text
探索范围太散，
不一定覆盖 MORL 真正关心的 Pareto 区域。
```

所以作者提出 **Preference-Guided Exploration（PG-Explore）**。  
它不是随便采一个 latent `z`，而是把偏好 `λ` 和 replay buffer 里的奖励样本结合起来构造：

```text
z_hat_λ = (1 / n_s) * sum_{(s,a,r,s') in D} B_ω(s,a) r^T λ
```

这里 `D` 是从 replay buffer 里采样的 mini-batch。

直观理解是：

```text
不是让 latent 去学“所有可能 reward”，
而是优先学和当前偏好 λ 相关、对 MORL 真有用的 reward 方向。
```

### 3. 把 RFRL 的训练目标当成辅助任务

作者保留了 FB 的 measure loss：

```text
L_M(F_θ, B_ω; z_λ)
```

它负责学 successor measure 的 Bellman consistency。

同时，作者再加一个 **辅助 Q loss**：

```text
L_Q(F_θ; z_λ) = E[(F_θ(s,a,z_λ)^T z_λ - (λ^T r + γ F_θ¯(s', π(s',z_λ), z_λ)^T z_λ))^2]
```

这个 Q loss 用的是环境里真实观测到的多目标奖励向量，而不是原始 FB 里的 pseudo reward。

所以训练目标可以理解成：

```text
measure loss 负责学 reward-free 表示
Q loss 负责把表示拉回到 MORL 的真实奖励结构
```

## 训练流程

算法 1 的流程很简单：

```text
1. 随机采样一个 preference λ
2. 用 PG-Explore 生成 z
3. 用这个 z 跑 rollout
4. 从 replay buffer 采样 transition
5. 更新 FB 网络和 policy
```

所以它不是先把所有偏好都穷举出来再训，而是：

```text
边采样偏好，边构造 z，边做训练。
```

## 为什么这个方法有意义

作者的核心逻辑是：

```text
RFRL 学的是“更广泛的 reward family”
MORL 只需要其中一部分
```

因此 RFRL 提供了天然的辅助任务来源：

```text
1. 让表示更通用
2. 让策略共享更多环境知识
3. 对没见过的偏好更稳
```

但作者也指出，**直接把 RFRL 原样搬来不够**，因为它会在 MORL 里探索得太散。  
所以必须加：

```text
1. preference-guided exploration
2. auxiliary Q loss
```

## 实验设置

论文在 MO-Gymnasium 上评测，覆盖：

```text
离散控制 + 连续控制
```

常见任务包括：

```text
HalfCheetah2d
Walker2d
Hopper3d
Ant3d
Humanoid2d
Humanoid5d
```

评价指标主要是：

```text
UT: utility
HV: hypervolume
ED: episode distance / error-related metric
```

## 对比基线

作者和一批 MORL 方法比，包括：

```text
PD-MORL
Q-Pensieve
CAPQL
PGMORL
MORL/D
PCN
SFOLS
GPI-LS
GPI-PD
FB
```

## 主要结果

论文报告的核心结论是：

```text
1. MORL-FB 在大多数任务上优于现有 SOTA MORL 方法
2. 在偏好样本较少时，优势更明显
3. 对高维目标任务的泛化更稳
4. 样本效率显著更好
```

作者还特别强调：

```text
即使训练步数更少，MORL-FB 也能达到更高的 HV / UT。
```

## 消融结论

消融显示两件事都很关键：

```text
1. 去掉 preference-guided exploration，性能掉很多
2. 去掉 auxiliary Q loss，性能也掉很多
```

这说明：

```text
单纯 reward-free 的 FB 不够，
必须把 MORL 目标显式注入进去。
```

## 这篇论文的本质

可以把它理解成：

```text
RFRL 提供“能适配任意 reward 的表示学习框架”
MORL 提供“偏好条件的多目标优化目标”
MORL-FB 负责把两者接起来
```

它的创新不是提出一个全新的 RL 范式，而是：

```text
第一次系统地把 RFRL 改造成一个可用于 MORL 的实用算法。
```

## 一句话结论

这篇论文的核心是：

```text
把 reward-free 的 Forward-Backward RL 迁移到多目标强化学习里，
再用偏好引导探索 + 辅助 Q loss，
让一个偏好条件策略在多目标任务上学得更快、泛化更好。
```

