# TTT-Discover：在测试时继续训练模型来做科学与工程发现

本文档解读论文：

```text
Learning to Discover at Test Time
https://arxiv.org/abs/2601.16175
```

## 一句话总结

这篇论文提出 **TTT-Discover**：

```text
针对一个具体测试问题，不只是让冻结 LLM 多采样 / 多搜索，
而是在测试时继续对这个问题做 RL，让模型权重也随尝试经验更新。
```

它的目标不是训练一个泛化到很多任务的 policy，而是：

```text
在一个给定科学 / 工程问题上找到一个最强 candidate solution。
```

所以它优化的是：

```text
发现一个 SOTA 解
```

而不是：

```text
让模型平均表现更好
```

## 系统设置

模型：

```text
主实验：OpenAI gpt-oss-120b
部分数学对比：Qwen3-8B
```

训练方式：

```text
test-time RL
LoRA rank = 32
每个问题单独训练
50 training steps
每步 512 rollouts
总采样预算 50 * 512 = 25,600
```

训练平台：

```text
Tinker API
```

论文报告每个问题的测试时训练成本大约是：

```text
几百美元，典型约 $500 / problem
```

## 这篇和普通 test-time scaling 的区别

普通 test-time scaling / evolutionary search 通常是：

```text
冻结 LLM
维护一个历史尝试 buffer
把过去好的解写进 prompt
让 LLM 继续生成新解
```

比如 AlphaEvolve / OpenEvolve 这类方法主要是：

```text
prompt-level search
```

TTT-Discover 则是：

```text
prompt / state 继续复用
同时模型参数也继续更新
```

也就是：

```text
search + learning
```

## Discovery problem 怎么形式化

论文把每个科学 / 工程问题都变成一个环境：

```text
problem description d
state s      : 一个 candidate solution
action a     : LLM 生成的 thinking tokens + code
transition T : 解析 / 执行代码，得到新 candidate solution s'
reward R(s') : 这个 candidate 的连续分数
```

例如：

```text
GPU kernel:
  state  = kernel code
  action = 新的 Triton kernel code
  reward = 1 / runtime

数学构造:
  state  = step function / sequence / circle packing
  action = Python code 修改构造
  reward = 证明出的 bound 或 objective value

AtCoder:
  state  = C++ algorithm
  action = 新 C++ code
  reward = local test score

single-cell denoising:
  state  = analysis algorithm code
  action = 改进算法的 code
  reward = 1 / MSE 或 benchmark score
```

如果代码解析失败、验证失败或超时：

```text
reward = 0
```

## 为什么普通 RL 不适合 discovery

普通 RL 目标是：

```text
学一个平均 reward 高、能反复部署的 policy
```

但 discovery 目标是：

```text
只要找到一个最好的 solution 就行
```

这两个目标不一样。

举个 kernel 例子：

```text
已有 SOTA runtime = 2000 us
某个候选 runtime = 1900 us
```

这对 discovery 很重要，因为它超过了 SOTA。  
但在普通 expected reward 里，它可能只是一个小幅均值提升。

所以 TTT-Discover 的目标函数要更偏向：

```text
高 reward tail / best solution
```

而不是：

```text
平均 reward
```

## 算法整体流程

算法 1 可以理解成：

```text
输入:
  problem description d
  初始 policy π_θ0

初始化:
  archive H0 = {empty solution}

循环:
  1. 从 archive 里选一个 start state s_i 和上下文 c_i
  2. 当前 policy π_θi 基于 (d, s_i, c_i) 生成 action a_i
  3. 环境解析 / 执行 action，得到新 state s'_i
  4. 计算 reward r_i = R(s'_i)
  5. 把这次尝试加入 archive
  6. 用这次 batch 的尝试更新 policy 参数 θ_i

输出:
  archive 中 reward 最高的 solution
```

核心有两个子模块：

```text
train:
  如何更新 policy

reuse:
  下一次从 archive 里选哪个已有 solution 继续改
```

TTT-Discover 对这两个子模块分别做了设计：

```text
train = adaptive entropic objective
reuse = PUCT
```

## 关键设计一：Entropic objective

普通 RL 优化：

```text
E[R]
```

TTT-Discover 优化：

```text
J_β(θ; s) = log E_{τ ~ π_θ(·|s)} [ exp(β r(τ; s)) ]
```

这个目标的直觉是：

```text
β 越大，目标越接近 max reward；
β 越小，目标越接近 expected reward。
```

当 `β -> ∞` 时：

```text
log E[exp(β r)] / β ≈ max r
```

所以它更符合 discovery：

```text
我要的不是平均好，
我要的是找到一个特别好的解。
```

对应的 policy gradient 权重是：

```text
w_β(τ | s) = exp(β r(τ; s)) / E_{π_θ}[exp(β r(τ; s))]
```

优势写成：

```text
A_β(τ | s) = w_β(τ | s) - 1
```

论文还加 KL penalty：

```text
A(τ; s) = w_β(τ | s) - 1 - λ log(π_θ(τ|s) / π_θ0(τ|s))
```

其中 `π_θ0` 是初始模型，用来约束测试时训练不要漂太远。

## 为什么 β 要自适应

固定 `β` 不好调：

```text
训练早期 β 太大:
  少数高分 outlier 会主导更新，容易不稳

训练后期 β 太小:
  改进越来越小，advantage 变弱，学不动
```

所以作者按每个 start state `s` 自适应选择 `β(s)`。

做法是构造一个 tilted distribution：

```text
q_β(τ | s) ∝ π_θ(τ | s) exp(β r(τ; s))
```

然后限制：

```text
KL(q_β(·|s) || π_θ(·|s)) = γ
```

论文固定：

```text
γ = ln 2
```

直觉：

```text
让高分样本更重要，
但不允许权重集中得太离谱。
```

batch 内实际实现是：

```text
1. 同一个 start state 采 N 个 rollouts
2. 根据 reward r_n 做 softmax-like reweighting
3. 用二分搜索找到满足 KL budget 的 β
4. 用 leave-one-out 估计 advantage，减少偏置
```

## 关键设计二：PUCT state reuse

TTT-Discover 维护一个 archive：

```text
H_t = {过去发现过的 candidate states}
```

每次不是都从空解开始，而是从 archive 里选一个已有解继续改。

选择规则来自 PUCT：

```text
score(s) = Q(s) + c * scale * P(s) * sqrt(1 + T) / (1 + n(s))
```

其中：

```text
Q(s):
  从 s 出发曾经得到过的最高 child reward

P(s):
  按 archive reward 排名得到的 prior
  reward 越高，prior 越大

n(s):
  s 或其后代被扩展过多少次

T:
  总扩展次数

scale:
  archive 当前 reward range
```

注意它用的是：

```text
max child reward
```

不是平均 child reward。

原因还是 discovery 目标：

```text
只关心从这个状态出发有没有可能找到特别好的后代。
```

archive 更新时：

```text
1. 每个 expanded parent 只保留 top-2 children
2. 全局 archive 最多保留 top-1000 states
3. 初始 seed states 永远保留
```

## 和 Best-of-N 的区别

Best-of-N：

```text
固定模型
独立采样 25,600 次
取最高 reward
```

TTT-Discover：

```text
每步采样 512 次
用这批经验更新模型
再从 archive 里挑 promising state 继续扩展
总采样预算也是 25,600
```

所以它不是靠更多采样赢，而是：

```text
在相同采样预算下，把尝试经验内化进模型参数。
```

## 主实验结果

### 数学任务

| Method | Model | Erdos min overlap ↓ | AC1 ↓ | AC2 ↑ |
|---|---|---:|---:|---:|
| best human | - | 0.380927 | 1.50973 | 0.9015 |
| AlphaEvolve | Gemini-2.0 Pro + Flash | 0.380924 | 1.50530 | 0.8962 |
| AlphaEvolve V2 | Gemini-2.0 Pro + Flash | 0.380924 | 1.50317 | 0.9610 |
| ThetaEvolve | R1-Qwen3-8B | n/a | 1.50681 | 0.9468 |
| ThetaEvolve w/ SOTA reuse | R1-Qwen3-8B | n/a | 1.50314 | n/a |
| OpenEvolve | gpt-oss-120b | 0.380965 | 1.50719 | 0.9449 |
| Best-of-25600 | gpt-oss-120b | 0.380906 | 1.51004 | 0.9344 |
| TTT-Discover | Qwen3-8B | 0.380932 | 1.50525 | 0.9472 |
| TTT-Discover | gpt-oss-120b | 0.380876 | 1.50287 | 0.9591 |

结论：

```text
Erdos minimum overlap:
  TTT-Discover 达到 0.380876，新 SOTA

AC1:
  TTT-Discover 达到 1.50287，新 SOTA

AC2:
  没有超过 AlphaEvolve V2 的 0.9610
```

### GPU Kernel：TriMul

| Method | Model | A100 ↓ | H100 ↓ | B200 ↓ | AMD MI300X ↓ |
|---|---|---:|---:|---:|---:|
| 1st human | - | 4531.5 | 1371.1 | 1038.9 | 2515.8 |
| 2nd human | - | 4918.5 | 2368.0 | 2362.4 | 5165.0 |
| 3rd human | - | 5182.2 | 2545.7 | 1931.0 | 5359.3 |
| 4th human | - | 6097.8 | 3654.8 | 2248.9 | 5981.4 |
| 5th human | - | 8345.0 | 4233.1 | 6503.8 | 8365.1 |
| Best-of-25600 | gpt-oss-120b | 9219.7 | 5390.3 | 3254.9 | 4902.0 |
| TTT-Discover | gpt-oss-120b | 2198.2 | 1161.2 | 914.2 | 1555.7 |

结论：

```text
TriMul 上 TTT-Discover 在所有 GPU 类型上都超过 human leaderboard 第一名。
训练 reward 只用 H100 runtime，但发现的 kernel 还能泛化到 A100 / B200 / MI300X。
```

### AtCoder Heuristic Contest

| Method | Model | Geometry ahc039 ↑ | Scheduling ahc058 ↑ |
|---|---|---:|---:|
| 1st human | - | 566,997 | 847,674,723 |
| 2nd human | - | 557,212 | 846,938,871 |
| 3rd human | - | 554,334 | 846,350,877 |
| 4th human | - | 552,933 | 845,489,747 |
| 5th human | - | 549,746 | 845,324,831 |
| ALE-Agent | Ensemble | 550,647 | 848,373,282 |
| ShinkaEvolve | Ensemble | 558,026 | n/a |
| Best-of-25600 | gpt-oss-120b | 554,171 | 772,429,752 |
| TTT-Discover | gpt-oss-120b | 567,062 | 848,414,228 |

结论：

```text
如果按当时比赛提交，TTT-Discover 在两个 AHC 任务上都能拿第 1。
```

### Single-cell denoising

| Method | Model | PBMC Score ↑ | PBMC MSE ↓ | PBMC Poisson ↓ | Tabula Score ↑ | Tabula MSE ↓ | Tabula Poisson ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| MAGIC (A, R) | - | 0.64 | 0.19 | 0.05 | 0.64 | 0.18 | 0.03 |
| MAGIC (R) | - | 0.64 | 0.19 | 0.05 | 0.64 | 0.18 | 0.03 |
| ALRA (S, RN) | - | 0.50 | 0.26 | 0.05 | 0.47 | 0.27 | 0.03 |
| MAGIC (A) | - | 0.42 | 0.19 | 0.16 | 0.40 | 0.18 | 0.12 |
| MAGIC | - | 0.42 | 0.19 | 0.16 | 0.40 | 0.18 | 0.12 |
| OpenEvolve | gpt-oss-120b | 0.70 | 0.16 | 0.05 | 0.71 | 0.15 | 0.03 |
| Best-of-25600 | gpt-oss-120b | 0.62 | 0.20 | 0.05 | 0.65 | 0.18 | 0.03 |
| TTT-Discover | gpt-oss-120b | 0.71 | 0.15 | 0.05 | 0.73 | 0.14 | 0.03 |

结论：

```text
TTT-Discover 在 OpenProblems denoising benchmark 上超过 MAGIC / ALRA / OpenEvolve。
```

## 消融实验

消融在 GPUMode TriMul H100 上做：

| Setting | train | reuse | Best runtime ↓ |
|---|---|---|---:|
| Best Human Kernel | - | - | 1371.1 |
| TTT-Discover | TTT with adaptive entropic | PUCT | 1203.10 |
| Constant beta | TTT with constant beta entropic | PUCT | 1483.83 |
| No entropic | TTT with expected reward | PUCT | 1985.67 |
| No TTT | no TTT | PUCT | 2060.70 |
| Epsilon-greedy reuse | TTT with adaptive entropic | epsilon-greedy | 1328.89 |
| No reuse | TTT with adaptive entropic | no reuse | 5274.03 |
| Naive test-time RL | TTT with expected reward | no reuse | 5328.73 |
| Best-of-N | no TTT | no reuse | 5352.36 |

消融说明：

```text
adaptive entropic objective:
  明显优于 expected reward 和 fixed beta

PUCT reuse:
  明显优于 no reuse

TTT 本身:
  明显优于只做 search / Best-of-N
```

## 这篇论文的本质

TTT-Discover 可以理解成：

```text
一个面向单个 test problem 的 online RL discovery loop。
```

它和普通后训练最大的区别是：

```text
普通后训练:
  训练一个以后泛化用的模型

TTT-Discover:
  为当前这个问题临时训练模型，只要最终 archive 里出现一个最优解即可
```

所以它的 policy 只是工具，最终产物是：

```text
best discovered solution
```

## 局限性

这篇方法有几个前提：

```text
1. 必须有自动 reward / verifier
2. reward 最好是连续的，能区分小改进
3. 每次 candidate evaluation 成本不能太离谱
4. 测试时允许修改模型权重
5. 发现的解仍需要专家审核，尤其是科学 / 生物场景
```

论文也明确提醒 single-cell denoising：

```text
benchmark 分数提升不等于一定带来新的生物学洞察。
```

## 一句话结论

TTT-Discover 的核心贡献是：

```text
把 test-time compute 从“冻结模型搜索更多答案”
推进到“针对单个测试问题继续 RL 训练模型”，
并用 entropic objective + PUCT reuse 把优化目标对准 discovery 里的 best-solution search。
```

