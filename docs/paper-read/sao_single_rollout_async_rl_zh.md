# SAO：Single-Rollout Asynchronous Optimization 技术解读

本文档解读论文：

```text
Single-Rollout Asynchronous Optimization for Agentic Reinforcement Learning
Zhenyu Hou, Yujiang Li, Jie Tang, Yuxiao Dong
arXiv:2607.07508
https://arxiv.org/abs/2607.07508
```

asynchronous RL 是 “updating the model as rollouts arrive”，也就是 rollout 到达就训练，而不是等完整同步 batch。

## 一句话总结

SAO 解决的是：

```text
长轨迹 agentic RL 中，同步 GRPO / PPO 等 batch-interleaved pipeline 会被慢 rollout 拖住；
异步 RL 虽然吞吐更高，但 policy lag / off-policy 更严重，GRPO 的 group-wise sampling 也不适配。
```

它的核心方案：

```text
1. 用 asynchronous actor-learner pipeline。
2. 每个 prompt 只采 1 条 rollout，生成完立刻进训练。
3. 不再用 GRPO 的 group reward baseline。
4. 用 value model / critic 估计 advantage。
5. 用 rollout engine 记录的 token logprob 直接做 importance ratio。
6. 对 ratio 做严格双边 token-level mask，超出 trust region 的 token 直接不反传。
```

最终目标：

```text
既保留异步训练的吞吐优势，
又让长链 reasoning / tool-use / coding agent RL 稳定训练上千步。
```

## 背景：为什么同步 GRPO 不适合 agentic RL

常规同步 RL 流程：

```text
1. 当前 policy 生成一个完整 rollout batch。
2. 等 batch 中所有 trajectory 都结束。
3. 再进入训练阶段。
```

对普通短回答任务，这样还可以。但 agentic / coding / tool-use 任务有两个特点：

```text
1. rollout length 方差很大。
   有些 trajectory 很快结束，有些可能跑很多工具调用。

2. trajectory 内部有环境反馈。
   例如代码执行结果、工具 observation、OpenHands 环境状态。
```

同步 batch 的问题：

```text
短 trajectory 已经完成，但必须等最长 trajectory。
GPU / rollout workers 出现 idle。
```

GRPO 还额外要求：

```text
同一个 prompt 生成一组 responses，
用组内 reward mean / std 做 advantage normalization。
```

这会引入 group barrier：

```text
同一 prompt 下 8 条 response 必须全部结束，
才能计算 group-level advantage 并送入训练。
```

在异步 agentic RL 里，这个 barrier 很糟：

```text
1. 慢样本拖住快样本。
2. 快样本等待期间 policy 已经更新，off-policy 更严重。
3. 在线环境里经常每个 prompt 只能拿到单条 feedback，天然没有 group。
```

## 异步到底指什么

SAO 里的异步主要指：

```text
rollout / inference 和 training / learner 异步。
```

也就是：

```text
rollout workers:
  持续用某个 policy 版本生成 trajectory。

learner / trainer:
  rollout 完成后进入队列；
  learner 从队列取样本，凑够 training batch 后再更新 policy。
```

因此，一个典型异步流程是：

```text
1. rollout engine 正在生成 trajectory batch k+1。
2. learner 同时正在用已经完成的 trajectory batch k 做 forward/backward/update。
3. batch k+1 中较短的 trajectory 先完成后，可以更早进入训练队列。
4. learner 不需要等同一个 group 或整个同步 batch 全部结束。
```

注意：

```text
生成完即训练
  不是指 1 条 trajectory 完成后立刻 optimizer.step。

更准确是:
  1 条 trajectory 完成后立刻可入队；
  learner 凑够 batch size 后才做一次 policy update。
```

论文实验中：

```text
batch size = 128
group size = 1
```

含义是：

```text
每个 prompt 只采 1 条 trajectory；
但一次 policy update 仍然使用 128 条完成的 trajectories / samples。
```

### 生成与训练流程图

同步 GRPO 的流程：

```text
Policy theta_k
    |
    v
Rollout group for prompt_1: y_1 ... y_N
Rollout group for prompt_2: y_1 ... y_N
...
    |
    |  等同 prompt 的 N 条 response 全部完成
    v
Compute group reward baseline
    |
    v
Build train batch
    |
    v
Forward / backward / optimizer.step
    |
    v
Policy theta_{k+1}
```

SAO 异步 single-rollout 流程：

```text
                 +------------------------------+
                 | rollout workers / engines    |
                 | use policy snapshots         |
                 +------------------------------+
                    |       |        |
                    v       v        v
                 traj_a   traj_b   traj_c   ...
                    |       |        |
                    |  完成即入队，不等 sibling rollouts
                    v       v        v
             +-----------------------------------+
             | completed trajectory queue        |
             +-----------------------------------+
                         |
                         | learner 取样
                         | 凑够 batch size，例如 128 条
                         v
             +-----------------------------------+
             | training batch                    |
             | value / advantage / DIS mask      |
             +-----------------------------------+
                         |
                         v
             forward / backward / optimizer.step
                         |
                         v
                  policy theta_{k+1}

与此同时:
  rollout workers 已经在继续生成后续 trajectories。
```

两者的核心区别：

```text
GRPO:
  barrier 在 group level。
  同一个 prompt 的 N 条 response 必须全部结束。

SAO:
  barrier 只剩 learner batch level。
  任意完成的 batch size 条 trajectory 都可以组成训练 batch。
```

这和同步 pipeline 的区别：

```text
同步:
  rollout 阶段和 train 阶段基本串行。
  必须等当前 batch 全部 rollout 完成。

异步:
  rollout 和 train overlap。
  trajectory 完成后以 streaming / queue 形式进入 learner。
```

### 异步程度

可以把异步程度分成三档：

```text
轻度异步:
  rollout batch k 生成完；
  learner 训练 batch k；
  rollout engine 同时开始 batch k+1。

中度异步:
  多个 rollout workers 持续生成；
  learner 从队列里不断取已完成 trajectory；
  policy update 和 rollout generation 持续 overlap。

重度异步:
  rollout workers 可能使用明显落后的 policy；
  learner 已经更新很多步；
  队列里仍然有旧 policy 生成的 trajectory。
```

SAO 要解决的是中到重度异步下的问题：

```text
trajectory 是 rollout policy 生成的；
训练时 current policy 已经变了；
所以样本对 current policy 来说是 off-policy。
```

### 这样异步会不会太重

会带来明显 off-policy 风险。SAO 的核心不是否认这个风险，而是：

```text
只训练那些 current policy 和 rollout policy 还没偏太远的 token。
```

具体做法：

```text
ratio
  = pi_current(a_t | s_t) / pi_rollout(a_t | s_t)

  = exp(current_logprob - rollout_logprob)
```

然后：

```text
ratio 在 trust region 内:
  token 保留，用于 policy update。

ratio 太大或太小:
  token 直接 mask，不反传。
```

所以异步可以较重，但训练信号会被 DIS gate：

```text
off-policy 程度可接受的 token:
  用。

off-policy 太严重的 token:
  丢。
```

训练稳定还依赖其他配套设计：

```text
1. single rollout
   减少 GRPO group barrier 和等待造成的 policy lag。

2. DIS hard mask
   off-policy 太严重的 token 不训练。

3. value model
   没有 GRPO group baseline，需要 critic 给 advantage。

4. faster value update
   critic 更新更频繁，跟上 policy 分布变化。

5. skip-observation GAE
   agent trajectory 中跳过 environment observation token，
   避免 credit assignment 被观察 token 噪声污染。
```

一句话：

```text
SAO 确实允许 rollout 和 train 并行 overlap；
这会带来更强 off-policy；
SAO 用 rollout_logprob、双边 token mask 和 value model 把这种异步风险压住。
```

## SAO 的核心设计

SAO = Single-rollout Asynchronous Optimization。

可以拆成四个关键组件：

```text
1. Direct Double-Sided Importance Sampling, DIS
2. Single-rollout sampling
3. 强化 value model 训练
4. Skip-Observation token-level GAE
```

## 组件 1：Direct Double-Sided Importance Sampling

### 异步 RL 的 policy lag

异步训练中：

```text
rollout engine 正在生成 trajectory；
training engine 同时不断更新 policy。
```

所以一条长 trajectory 可能跨越多个 policy 版本：

```text
trajectory 前半段由较旧 policy 生成；
后半段可能由更新后的 rollout engine 生成；
训练时 current policy 又是另一个版本。
```

理想 PPO 里通常有：

```text
current policy:
  pi_theta

old policy:
  pi_old

rollout policy:
  pi_rollout
```

传统分解：

```text
pi_theta / pi_old:
  PPO old-policy correction

pi_old / pi_rollout:
  train-rollout mismatch correction
```

但异步长轨迹里精确维护所有 old policy 很难：

```text
需要保存大量历史 checkpoint；
还要知道每个 token 由哪个 policy version 生成。
```

### SAO 的简化

SAO 直接放弃显式 `pi_old`：

```text
r_t(theta)
  = pi_theta(a_t | s_t) / pi_rollout(a_t | s_t)

  = exp(log pi_theta(a_t | s_t)
        - log pi_rollout(a_t | s_t))
```

其中：

```text
log pi_rollout:
  rollout engine 生成 token 时记录的 logprob。

log pi_theta:
  当前训练 policy 对同一 token 重新 forward 得到的 logprob。
```

这和我们之前看的 TIS / ICEPOP 很接近：

```text
都是 train/current policy 与 rollout engine behavior policy 的 ratio。
```

但 SAO 更激进：

```text
不再维护 old policy；
直接把 rollout policy 当作 behavior policy proxy。
```

### 双边 token-level mask

SAO 的校准函数：

```text
f(r_t; eps_low, eps_high)
  = r_t, if 1 - eps_low < r_t < 1 + eps_high
  = 0,   otherwise
```

policy objective 写成：

```text
L(theta)
  = E_t [
      f(r_t, eps_low, eps_high)
      * A_t
      * log pi_theta(a_t | s_t)
    ]
```

也就是说：

```text
ratio 在 trust region 内:
  token 保留，并乘以 ratio。

ratio 太小或太大:
  token 直接 mask，不参与梯度。
```

这和 PPO clip 的差别：

```text
PPO clip:
  根据 advantage 符号，只裁剪会提升 objective 的方向。

SAO DIS:
  不管 advantage 正负，只要 ratio 超出 [1-eps_low, 1+eps_high]，
  token 直接置 0。
```

### PPO/GRPO clip 为什么不是简单的域外 mask

PPO / GRPO 常用的 clipped objective 不是先把所有 ratio clamp 后再算 loss。

它不是：

```text
ratio = clamp(ratio, 1 - eps, 1 + eps)
loss = -ratio * A
```

如果这样做，那么 clip 区间外确实都会变成常数边界，没有梯度。

实际 PPO 是同时计算两项：

```text
unclipped = ratio * A
clipped   = clamp(ratio, 1 - eps, 1 + eps) * A

objective = min(unclipped, clipped)
loss      = -objective
```

因此，ratio 即使在区间外，最后选中的也不一定是 clipped 项。

假设：

```text
clip range = [0.8, 1.2]
```

#### Case 1：A > 0 且 ratio > 1.2

正 advantage 表示：

```text
这个 token 好，应该提高概率。
```

如果：

```text
ratio = 1.5
A > 0
```

则：

```text
unclipped = 1.5 * A
clipped   = 1.2 * A
objective = min(1.5A, 1.2A) = 1.2A
```

选中 clipped 项，`1.2` 是常数，因此这部分没有继续推高概率的梯度。

含义：

```text
这个好 token 的概率已经提高太多了，不要继续推高。
```

#### Case 2：A > 0 且 ratio < 0.8

如果：

```text
ratio = 0.6
A > 0
```

则：

```text
unclipped = 0.6 * A
clipped   = 0.8 * A
objective = min(0.6A, 0.8A) = 0.6A
```

选中的是 unclipped 项，仍然有梯度。

含义：

```text
这个好 token 的概率反而比 old policy 低了，
应该允许梯度把它拉回来。
```

#### Case 3：A < 0 且 ratio < 0.8

负 advantage 表示：

```text
这个 token 不好，应该降低概率。
```

如果：

```text
ratio = 0.6
A = -1
```

则：

```text
unclipped = 0.6 * (-1) = -0.6
clipped   = 0.8 * (-1) = -0.8
objective = min(-0.6, -0.8) = -0.8
```

选中 clipped 项，没有继续压低概率的梯度。

含义：

```text
这个坏 token 的概率已经降低太多了，不要继续压低。
```

#### Case 4：A < 0 且 ratio > 1.2

如果：

```text
ratio = 1.5
A = -1
```

则：

```text
unclipped = 1.5 * (-1) = -1.5
clipped   = 1.2 * (-1) = -1.2
objective = min(-1.5, -1.2) = -1.5
```

选中的是 unclipped 项，仍然有梯度。

含义：

```text
这个坏 token 的概率反而升高了，
应该允许梯度把它压回去。
```

所以 PPO / GRPO clip 的真实含义是：

```text
如果更新方向会让 policy 更偏离 old policy，就截断梯度；
如果更新方向是在把 policy 拉回合理范围，就保留梯度。
```

而 SAO 的 DIS mask 是：

```text
if ratio < 1 - eps_low or ratio > 1 + eps_high:
  weight = 0
else:
  weight = ratio
```

它不看 advantage 的正负：

```text
ratio 太大:
  直接不反传。

ratio 太小:
  也直接不反传。
```

因此：

```text
PPO / GRPO clip:
  区间外不一定没梯度，取决于 advantage 符号和 min/max 选中了哪一项。

SAO DIS mask:
  区间外一定没梯度。
```

因此它比普通 PPO clip 更像：

```text
严格双边 token-level rejection / IcePop-style mask。
```

论文也说它与 IcePop 类似，但进一步去掉了 `pi_old`，只使用 rollout logprob。

## 组件 2：Single-Rollout Sampling

GRPO：

```text
每个 prompt 采 N 条 response。
用组内 reward baseline 算 advantage。
```

SAO：

```text
每个 prompt 只采 1 条 response。
生成完马上进入训练队列。
```

好处：

```text
1. 没有 group barrier。
2. 快 trajectory 不用等慢 trajectory。
3. 更适合在线学习，每个 prompt 只有一次环境反馈也能训练。
4. 降低 group-wise waiting 引起的 policy lag。
```

代价：

```text
不能再用 GRPO 的组内相对 reward 做 baseline。
```

所以 SAO 必须引入 critic / value model：

```text
A_t = return_t - V(s_t)
```

或者用 GAE：

```text
A_t = sum_l (gamma * lambda)^l * delta_{t+l}
```

这就是它和 critic-free GRPO 的主要差别：

```text
SAO 是 value-based / actor-critic 风格；
GRPO 是 group-relative critic-free 风格。
```

## 组件 3：Value Model 训练策略

Single-rollout 的核心风险是：

```text
单条 trajectory 的 reward variance 很高。
如果 value model 不准，advantage 会很噪，policy update 会被破坏。
```

论文用了三个 value model 设计。

### Faster Value Update

策略：

```text
每做 1 次 policy update，
value model 做 K 次 update。
```

实验中：

```text
K = 2
```

原因：

```text
异步 RL 中 policy 分布变化快；
critic 如果跟不上，会给出过时 baseline。
```

更频繁更新 critic 后：

```text
critic 更快贴近当前 policy 分布；
advantage variance 更低；
policy update 更稳定。
```

### Frozen-Attention Value Training

论文发现：

```text
value model 的梯度 norm 明显大于 policy model；
不稳定主要来自 full attention layers；
MoE layers 相对稳定。
```

因此训练 critic 时：

```text
冻结 attention modules；
主要优化 MoE projections。
```

直觉：

```text
pretrained attention 已经具备语义路由能力；
value head / MoE 适配 reward 更重要；
冻结 attention 可以正则化 critic，降低梯度震荡。
```

### Scaling Value Pretraining

论文强调：

```text
value model cold start 是 single-rollout RL 的主要瓶颈。
```

所以需要扩大 value pretraining 数据：

```text
让 critic 在 RL 开始前已有较好的 return / value 初始化。
```

否则 single-rollout 下没有 group baseline，critic 又不准，训练很容易崩。

## 组件 4：Skip-Observation Token-Level GAE

Agentic trajectory 不是纯模型生成文本，而是：

```text
T = [a_0, o_0, a_1, o_1, ...]
```

其中：

```text
a_i:
  model action，例如回复、工具调用、代码 patch。

o_i:
  environment observation，例如工具返回、执行结果、OpenHands 环境状态。
```

标准 token-level GAE 会在相邻 token 上传播 advantage：

```text
token_t -> token_{t+1}
```

但从 `a_i` 的末尾跨到 `o_i` 的开头是不合理的：

```text
o_i 不是模型生成的；
它是环境反馈；
value model 对 observation token 的 value 可能引入噪声。
```

SAO 做 Skip-Observation GAE：

```text
从当前 action 的最后 token，
直接跳到下一个 action 的第一个 token。
```

论文公式：

```text
A(a_{i,N})
  = delta + gamma * lambda * A(a_{i+1,0})

delta
  = r_t + gamma * V(a_{i+1,0}) - V(a_{i,N})
```

含义：

```text
advantage 只沿模型生成的 action 边界传播；
跳过 environment observation token；
避免把外部环境噪声传播进模型 action credit assignment。
```

论文附录还比较了 step-level GAE：

```text
step average value
last-token value
token-level value
```

结果：

```text
token-level 最好。
```

说明复杂 reasoning / agentic trajectory 仍然需要细粒度 token-level supervision。

## SAO 和 GRPO 的核心区别

| 维度 | GRPO | SAO |
|---|---|---|
| rollout | 每 prompt 多条 | 每 prompt 单条 |
| baseline | group mean / std | value model |
| 是否需要 critic | 不需要 | 需要 |
| 异步适配 | 差，有 group barrier | 好，生成完即训练 |
| off-policy 修正 | PPO/old-policy ratio 或 variants | 直接 current/rollout ratio |
| clipping | PPO-style selected clip | 双边 token-level mask |
| agent observation | 通常不是核心设计 | Skip-Observation GAE |

## 和 TIS / ICEPOP 的关系

我们之前在 Uni-OPD / Miles 代码中区分过：

```text
PPO ratio:
  current_policy / old_policy

TIS / ICEPOP:
  train_policy / rollout_policy
```

SAO 的 DIS 更接近后者：

```text
r_t = current_policy / rollout_policy
```

但 SAO 把它直接用于 policy objective：

```text
f(r_t) * A_t * log pi_theta
```

并且：

```text
ratio 超出双边阈值时直接 mask。
```

可以理解为：

```text
SAO 把异步 RL 中难以维护的 old_policy 直接删掉，
用 rollout logprob 作为 behavior probability，
再用强双边 mask 控制 off-policy token。
```

代价：

```text
会引入 controlled off-policy bias。
```

论文认为这个 bias 是可接受的，因为换来了：

```text
1. 不用保存历史 policy checkpoint。
2. 不用 old-policy inference。
3. 不用追踪每个 token 来自哪个 policy version。
4. 异步长轨迹训练更稳定。
```

## 实验设置

### Agentic Reasoning

模型：

```text
Qwen3-30B-A3B-Thinking-2507
```

初始化：

```text
先用 GPT-OSS-120B 生成的 Tool-Integrated Reasoning 数据做 3 epochs SFT；
再用该模型初始化 policy 和 value model。
```

任务：

```text
TIR:
  natural-language math reasoning + Python tool calls。
```

RL 设置：

```text
batch size:
  128

group size:
  1

max length:
  128k tokens

policy lr:
  1e-6

eps_low:
  0.3

eps_high:
  5.0

value lr:
  5e-6

critic warmup:
  10 steps

critic updates per policy update:
  K = 2
```

GRPO 对比：

```text
16 prompts * 8 rollouts = batch size 128
```

### Coding Agent

模型：

```text
Qwen3-30B-A3B-Thinking-2507
```

SWE-Bench Verified：

```text
OpenHands scaffold
max interaction turns: 300
context budget: 128k tokens
```

和 reasoning 设置基本一致，但：

```text
eps_low = 0.8
eps_high = 3.0
```

### Evaluation

数学 reasoning：

```text
AIME2025
BeyondAIME
HMMT Nov 2025
IMOAnswerBench
```

coding：

```text
SWE-Bench Verified
```

评估设置：

```text
top-p = 1.0
temperature = 1.0
max generation length = 128k
math 最大 50 turns
SWE 最大 300 OpenHands turns
```

## 主要结果

### Math reasoning

论文表 1：

```text
Qwen3-30B-A3B SFT (w/ python):
  AIME2025:        80.4
  BeyondAIME:      53.3
  HMMT Nov 2025:   75.2
  IMOAnswerBench:  53.3

GRPO (w/ python):
  AIME2025:        84.2
  BeyondAIME:      54.8
  HMMT Nov 2025:   76.0
  IMOAnswerBench:  55.8

SAO:
  AIME2025:        97.3
  BeyondAIME:      74.8
  HMMT Nov 2025:   88.3
  IMOAnswerBench:  74.0
```

也就是说，SAO 在四个 benchmark 上都显著优于 SFT 和 GRPO。

论文还报告：

```text
SAO w/ DIS only:
  AIME2025:        94.2
  BeyondAIME:      71.5
  HMMT Nov 2025:   86.7
  IMOAnswerBench:  71.3

GRPO + DIS:
  AIME2025:        93.5
  BeyondAIME:      70.8
  HMMT Nov 2025:   84.0
  IMOAnswerBench:  70.0
```

这说明：

```text
DIS 本身很重要；
但 single-rollout + value model 的 SAO 整体仍然更强。
```

### SWE-Bench Verified

论文表 2：

```text
Qwen3-30B-A3B:
  23.0

GRPO w/ DIS:
  27.0

SAO:
  29.8
```

### 训练稳定性

论文说：

```text
vanilla GRPO 大约 160 steps collapse；
vanilla VAPO 大约 90 steps collapse；
SAO 可以稳定训练约 1000 steps。
```

GRPO + DIS 能稳定很多，但在 400 steps 后和 SAO 拉开差距。

## Ablation

### Faster Value Update

表 3 / 表 4：

```text
SAO:
  AIME2025:   97.3
  BeyondAIME: 74.8

SAO w/o Faster value:
  AIME2025:   95.0
  BeyondAIME: 69.8
```

解释：

```text
单次 critic update 跟不上 policy 分布变化；
critic baseline 不准，advantage 噪声更大。
```

### Frozen Attention

结果：

```text
SAO w/o Frozen attention:
  AIME2025:   90.6
  BeyondAIME: 74.5
```

解释：

```text
full-parameter critic training 梯度 norm 更大；
冻结 attention 能稳定 value model 优化。
```

### Running Mean Baseline

结果：

```text
Running mean baseline:
  AIME2025:   79.8
  BeyondAIME: 55.3
```

说明：

```text
单 rollout 下简单历史 reward mean 可以工作，
但远不如 state-dependent value model。
```

### Step-Level vs Token-Level

附录表 5，400 steps：

```text
Step-level Average:
  AIME2025:   85.8
  BeyondAIME: 60.5

Step-level Last-Token:
  AIME2025:   87.3
  BeyondAIME: 62.8

Token-level:
  AIME2025:   89.8
  BeyondAIME: 66.8
```

结论：

```text
token-level value / advantage 更适合复杂推理轨迹。
```

## Online Learning Simulation

论文设计了一个动态写作风格任务：

```text
模型需要从多个风格中选择并生成回答。
reward preference 会随训练阶段变化：
  cute
  chuunibyou
  classical
```

reward：

```text
r = r_quality * r_style

r_quality ∈ {0, 1}
r_style   ∈ {0, 1}
```

judge：

```text
GLM-4.7
```

对比：

```text
SAO vs Running Mean baseline
```

结果：

```text
当 reward preference 改变时，
SAO 能快速压低旧风格并对齐新风格；
Running Mean 因为历史窗口惯性，适应明显滞后。
```

这个实验强调：

```text
single-rollout + value model 特别适合非平稳在线环境。
```

## 和本仓库 / Miles 的关系

本仓库里我们前面讨论过：

```text
rollout_log_probs:
  rollout engine 生成 token 时记录。

batch["log_probs"]:
  train engine 训练前重新 forward。

current log_probs:
  policy loss 里当前模型 forward。
```

SAO 的 DIS 需要的关键数据就是：

```text
rollout_log_probs
current_log_probs
```

计算：

```text
ratio = exp(current_log_probs - rollout_log_probs)
```

然后：

```text
if ratio in [1 - eps_low, 1 + eps_high]:
  pg_loss *= ratio
else:
  pg_loss = 0
```

这和当前 Miles 里的 `icepop_function` 非常像：

```text
ice_ratio = exp(train_log_probs - rollout_log_probs)

if ice_ratio in range:
  pg_loss *= ice_ratio
else:
  pg_loss *= 0
```

但差异是：

```text
Miles ICEPOP:
  主要校正 train snapshot / rollout engine 的 mismatch。

SAO DIS:
  直接用 current policy / rollout policy ratio 做 policy objective 的 off-policy gate。
  它不维护 old_policy。
```

如果把 SAO 思路接入 Miles，要关注：

```text
1. rollout engine 必须可靠保存每个 token 的 rollout_logprob。
2. policy loss 中 old_log_probs 可能应直接选 rollout_log_probs。
3. clipping 不再是 PPO selected clip，而是双边 token mask。
4. single-rollout 需要 critic/value model，不能继续完全依赖 GRPO group advantage。
5. agentic trajectory 需要区分 action token 和 observation token。
```

## 方法局限

论文自己也指出：

```text
1. 实验集中在 Qwen3-30B-A3B backbone。
2. 结论不一定直接迁移到小模型、短任务、dense reward RLHF。
3. SAO 依赖可靠 value model。
4. SAO 依赖 rollout engine 保存 token-level behavior probabilities。
5. 在线学习实验是模拟 preference shift，不是真实用户在线系统。
```

工程上还要注意：

```text
1. critic 训练成本和显存成本比 GRPO 更高。
2. value pretraining 数据质量很关键。
3. 异步系统中 token-level logprob 的版本一致性必须严格管理。
4. 双边 mask 阈值过严会丢失太多梯度；过松又会引入 off-policy instability。
```

## 最终理解

SAO 的核心贡献不是“又一个 PPO clip 变体”，而是一套面向 agentic 异步 RL 的完整优化组合：

```text
single rollout:
  消除 GRPO group barrier，降低等待和 policy lag。

DIS:
  直接用 rollout logprob 做 behavior probability，
  双边 token-level mask 控制 off-policy token。

strong critic:
  用 value model 替代 group baseline，
  并通过 faster update / frozen attention / value pretraining 稳住 critic。

skip-observation GAE:
  让 advantage 只在模型 action token 之间传播，
  避免 observation token 噪声。
```

一句话：

```text
GRPO 适合同步、多样本、可组内比较的 RLVR；
SAO 更适合异步、长轨迹、单反馈、agentic/online 的 RL。
```
