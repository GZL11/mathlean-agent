# VAPO: Efficient and Reliable RL for Advanced Reasoning Tasks 论文解读

论文：

```text
VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks
```

链接：

```text
https://arxiv.org/abs/2504.05118
```

这里的 VAPO 指：

```text
Value-model-based Augmented Proximal Policy Optimization
```

可以翻成：

```text
基于价值模型增强的近端策略优化
```

注意：这和另一篇多模态论文里的 `Vision-Anchored Policy Optimization` 同名，但不是同一篇。

## 一句话总结

这篇论文的核心观点是：

```text
GRPO / DAPO 这类 value-model-free 方法很稳，
但如果能把 value model 训练好，
value-model-based PPO 在 long-CoT reasoning 上有更高上限。
```

VAPO 做的事情不是发明一个全新的 RL objective，而是把 PPO 在 long-CoT 上容易崩的几个点逐个补上：

```text
1. value model 初始化和长序列 bootstrapping 会有 bias；
2. response 长度差异很大时，固定 GAE lambda 不合适；
3. verifier reward 太稀疏，正确样本太珍贵，不能只靠普通 PPO 慢慢试错。
```

最后它在 `Qwen2.5-32B` base model 上做 RL，AIME 2024 `avg@32` 达到约 `60.4`，超过同设置下的 `DeepSeek-R1-Zero-Qwen-32B` 和 `DAPO`。

## 背景：为什么还要回到 value model

最近 LLM reasoning RL 里，GRPO / DAPO 很受欢迎，因为它们不训练 value model：

```text
同一个 prompt 采多条 responses，
用组内 reward 均值/方差做 baseline，
把 response-level advantage 分给整条轨迹。
```

优点很明显：

```text
1. 不需要额外 value model；
2. 不会被 value model 训练不稳定拖垮；
3. 工程上更简单。
```

但代价也明显：

```text
一条 response 只得到一个 scalar advantage，
每个 token 基本共享同一个方向。
```

对于 long-CoT reasoning，这个 credit assignment 太粗：

```text
某一步推理错了，最终答案错；
但普通 value-free advantage 很难指出是哪一步造成的。
```

VAPO 想重新证明：

```text
value model 如果训练得足够稳，
可以给 token-level 更细的 advantage，
从而比 value-free RL 有更高优化上限。
```

## VAPO 认为 PPO 在 long-CoT 上有哪三个问题

### 1. Value Model Bias over Long Sequences

可以翻成：

```text
长序列上的价值模型偏差
```

典型 PPO 会训练一个 value model：

```text
V(s_t) ≈ 从当前 token 状态 s_t 开始，未来能拿到的期望 return
```

但在 LLM RL 里，value model 常从 reward model 初始化。问题是两者目标不一样：

```text
reward model:
  通常只在 EOS / 最终答案附近打分。

value model:
  需要对每个中间 token 状态估计未来回报。
```

这个 objective mismatch 会导致 value 初始化偏差。长 CoT 里这种偏差会在 GAE 的反向传播里累积，最后使 value model 不再是可靠 baseline。

### 2. Heterogeneous Sequence Lengths

可以翻成：

```text
异质序列长度 / 长短混合响应
```

reasoning RL 里，同一批 response 可能有的几十 token，有的几千 token。固定 `lambda = 0.95` 的 GAE 对长短序列并不公平。

举个论文里的核心现象：

```text
lambda = 0.95
0.95^100 ≈ 0.006
```

也就是说，对于很长的序列，最终 reward 反传到前面 token 时几乎衰减没了。前面 token 的 advantage 更依赖 value bootstrapping，而 bootstrapping 又可能有 bias。

所以固定 lambda 会造成：

```text
短 response:
  方差更明显。

长 response:
  bias 更明显，最终 reward 很难传回前面。
```

### 3. Sparse Verifier Reward

可以翻成：

```text
稀疏验证器奖励
```

数学 reasoning 通常用 verifier / rule reward：

```text
答案对: 1
答案错: 0
```

长 CoT 会让这个问题更严重：

```text
生成更长，采样更贵；
正确样本更少；
一旦采到正确样本，就非常值得利用。
```

所以 VAPO 的策略是：

```text
保留足够探索，
但正确样本一旦出现，要用更强的方式学进去。
```

## VAPO 的七个组件

VAPO 不是单一技巧，而是七个工程/算法组件的组合。

### 1. Value-Pretraining

可以翻成：

```text
价值模型预训练
```

目的：

```text
缓解 reward model 初始化 value model 带来的偏差。
```

做法：

```text
1. 固定一个 policy，例如 pi_sft。
2. 用这个固定 policy 连续采样 responses。
3. 用 Monte-Carlo return 训练 value model。
4. 直到 value loss / explained variance 等指标足够稳定。
5. 保存这个 value checkpoint，再进入 PPO 训练。
```

直觉是：

```text
先让 value model 学会“每个中间状态未来大概能拿多少分”，
不要一上来就让带偏差的 value model 参与 policy 更新。
```

### 2. Decoupled-GAE

可以翻成：

```text
解耦式广义优势估计
```

普通 PPO 里，critic update 和 actor update 往往共用一套 GAE 参数。但 VAPO 把它们拆开：

```text
value / critic:
  lambda_critic = 1.0

policy / actor:
  lambda_policy 单独设置
```

为什么 value 用 `lambda_critic = 1.0`？

```text
它更接近 Monte-Carlo return，
可以减少 bootstrapping bias，
让 critic 学到更少偏的 target。
```

为什么 policy 不一定也用 `1.0`？

```text
policy 更新要考虑收敛速度和方差。
lambda 太大，advantage 方差可能变高；
lambda 太小，长序列 reward 又传不回来。
```

所以 VAPO 用 decoupled-GAE 把两个目标拆开。

### 3. Length-Adaptive GAE

可以翻成：

```text
长度自适应 GAE
```

VAPO 不再固定 `lambda_policy = 0.95`，而是让它随 response 长度 `l` 变化：

```text
lambda_policy = 1 - 1 / (alpha * l)
```

论文默认：

```text
alpha = 0.05
```

具体算一下：

```text
l = 100:
lambda_policy = 1 - 1 / (0.05 * 100)
              = 1 - 1 / 5
              = 0.8

l = 1000:
lambda_policy = 1 - 1 / (0.05 * 1000)
              = 1 - 1 / 50
              = 0.98

l = 2000:
lambda_policy = 1 - 1 / 100
              = 0.99
```

直觉：

```text
短序列不需要特别大的 lambda；
长序列需要更大的 lambda，避免最终 reward 传不到前面。
```

这比固定 `0.95` 更适合 long-CoT，因为它承认：

```text
不同长度 response 的 bias-variance trade-off 不一样。
```

### 4. Token-Level Policy Gradient Loss

可以翻成：

```text
token 级策略梯度损失
```

普通 sample-level loss 通常是：

```text
先对每条 response 内部 token 求平均，
再对 batch 内 response 求平均。
```

这会导致：

```text
长 response 的每个 token 权重更小。
```

例如：

```text
response A 长度 100
response B 长度 1000
```

如果先在每条 response 内平均，A 和 B 作为“样本”的权重相同，但 B 里的每个 token 只有 A 的 1/10 权重。

VAPO 改成 token-level loss：

```text
整个 batch 所有 token 统一平均。
```

简化写法：

```text
L_PPO
= - 1 / sum_i |o_i|
   * sum_i sum_t min(
       r_i,t * A_i,t,
       clip(r_i,t, 1 - eps, 1 + eps) * A_i,t
     )
```

这会让长 response 的 token 不再被系统性低估。

### 5. Clip-Higher

可以翻成：

```text
上界放宽裁剪
```

普通 PPO clip 用同一个范围：

```text
clip(r, 1 - eps, 1 + eps)
```

VAPO 采用非对称 clip：

```text
clip(r, 1 - eps_low, 1 + eps_high)
```

默认：

```text
eps_low = 0.2
eps_high = 0.28
```

直觉：

```text
对正 advantage token，
允许概率上升得更多；
对负方向仍保持较强约束，
避免把低概率探索空间过快压没。
```

它主要是为了缓解 entropy collapse，同时让正确探索结果能更快被利用。

### 6. Positive Example LM Loss

可以翻成：

```text
正样本语言建模损失
```

如果某条 sampled response 答对了，VAPO 额外对它加一个 NLL loss：

```text
L_NLL
= - 1 / sum_{o_i in T} |o_i|
   * sum_{o_i in T} sum_t log pi_theta(a_t | s_t)
```

其中：

```text
T = 当前 rollout batch 中 answer 正确的 responses 集合。
```

最终 policy loss：

```text
L = L_PPO + mu * L_NLL
```

论文默认：

```text
mu = 0.1
```

它的意义是：

```text
正确样本很稀缺，一旦采到，就不只通过 PPO ratio 慢慢推，
而是额外做一次 imitation learning，把正确轨迹更直接地记住。
```

这很接近 self-imitation learning 的思想。

### 7. Group-Sampling

可以翻成：

```text
组采样 / 同题多样本采样
```

固定总采样预算时，有两种分配：

```text
方案 A:
  多采 prompt，每个 prompt 只采 1 条。

方案 B:
  少采 prompt，每个 prompt 采多条。
```

VAPO 选择方案 B：

```text
512 prompts per sampling
每个 prompt 采 16 条
mini-batch size = 512
```

它的好处是：

```text
同一道题里更容易同时出现正确/错误 responses，
形成更强的组内对比信号。
```

这点和 GRPO 的 group-based 思路一致，但 VAPO 仍然是 value-model-based PPO。

## 训练设置

论文实验设置：

```text
base model:
  Qwen2.5-32B pretrained model

是否使用 SFT 数据:
  不使用 SFT data，用于和 DAPO / DeepSeek-R1-Zero-Qwen-32B 保持可比

optimizer:
  AdamW

actor learning rate:
  1e-6

critic learning rate:
  2e-6

scheduler:
  warmup-constant

gamma:
  1.0

base PPO lambda:
  0.95

base PPO clip epsilon:
  0.2
```

VAPO 相对 vanilla PPO 的关键参数：

```text
value warmup:
  50 steps

lambda_critic:
  1.0

lambda_policy:
  1 - 1 / (alpha * l), alpha = 0.05

eps_low:
  0.2

eps_high:
  0.28

positive LM loss weight:
  0.1

sampling:
  512 prompts, 16 samples per prompt
```

## 实验结果

主指标：

```text
AIME 2024 avg@32
top_p = 0.7
temperature = 1.0
```

主要结果：

| Model | AIME24 avg@32 |
|---|---:|
| Vanilla PPO | 5 |
| DeepSeek-R1-Zero-Qwen-32B | 47 |
| DAPO | 50 |
| VAPO w/o Value-Pretraining | 11 |
| VAPO w/o Decoupled-GAE | 33 |
| VAPO w/o Length-adaptive GAE | 45 |
| VAPO w/o Clip-Higher | 46 |
| VAPO w/o Token-level Loss | 53 |
| VAPO w/o Positive Example LM Loss | 54 |
| VAPO w/o Group-Sampling | 55 |
| VAPO | 60 |

论文摘要和正文还报告：

```text
VAPO 最终达到 60.4；
约 5000 steps 内达到 SOTA；
多次独立运行没有训练 crash；
峰值稳定在 60-61。
```

## 消融怎么理解

按 drop 大小看：

```text
VAPO full: 60
```

去掉不同组件：

```text
w/o Value-Pretraining:
  60 -> 11
  几乎崩掉，说明 value 初始化偏差是 PPO long-CoT 的最大风险。

w/o Decoupled-GAE:
  60 -> 33
  说明 value 和 policy 共用 GAE 参数会让长序列 reward 反传失败。

w/o Length-adaptive GAE:
  60 -> 45
  说明固定 lambda 不能同时照顾短 response 和长 response。

w/o Clip-Higher:
  60 -> 46
  说明探索空间和 entropy 管理很关键。

w/o Token-level Loss:
  60 -> 53
  说明长 response token 被低估会伤性能。

w/o Positive Example LM Loss:
  60 -> 54
  说明正确样本太稀缺，采到了就要更直接地学习。

w/o Group-Sampling:
  60 -> 55
  说明同题多样本对比确实比单题单样本更有效。
```

这张表其实说明：

```text
VAPO 的核心不是某一个 magic trick，
而是一套让 value-based PPO 不在 long-CoT 上崩掉的组合拳。
```

## 和 GRPO / DAPO 的区别

### GRPO / DAPO

```text
value-model-free
```

优势：

```text
1. 不训练 value model；
2. 训练更简单；
3. 用 group reward 做 baseline，稳定性强。
```

限制：

```text
1. advantage 更粗；
2. token-level credit assignment 弱；
3. 对 long-CoT 中间步骤的错误定位不够细。
```

### VAPO

```text
value-model-based
```

优势：

```text
1. 可以得到 token-level 更细 advantage；
2. 理论上方差更低；
3. 正确训练时有更高上限。
```

风险：

```text
1. value model bias 会直接带崩 policy；
2. GAE 参数对长短响应很敏感；
3. 需要额外 critic 训练和更多工程细节。
```

VAPO 的定位可以概括为：

```text
不是否定 GRPO / DAPO，
而是证明 value model 不是不能用，
只是必须专门为 long-CoT 修 PPO。
```

## 一个具体例子：为什么 Length-Adaptive GAE 有用

假设最终 reward 只在 EOS 给出：

```text
answer correct => R = 1
answer wrong   => R = 0
```

现在有一条 1000 token 的正确 response。

如果固定：

```text
lambda = 0.95
```

那么从最后 reward 传到前 100 个 token 的系数大约是：

```text
0.95^100 ≈ 0.006
```

更前面的 token 基本拿不到最终 reward 的直接信号，只能依赖：

```text
V(s_{t+1}) - V(s_t)
```

如果 value model 本身还有 bias，就会变成：

```text
前面 token 的 advantage 几乎由有偏 value 决定。
```

Length-Adaptive GAE 对长 response 会给更大的 `lambda_policy`：

```text
l = 1000:
lambda_policy = 0.98
```

则前 100 token 的衰减：

```text
0.98^100 ≈ 0.13
```

虽然仍然衰减，但比 `0.006` 大很多，最终 reward 对前面推理步骤的影响保留更多。

这就是它想解决的问题：

```text
长 reasoning 的前半段不能完全被 bootstrapped value 支配。
```

## 和 OPD / RLSD 的关系

VAPO 不是 OPD，因为它没有：

```text
teacher-student token distribution matching。
```

它是典型的：

```text
RLVR + value-model-based PPO
```

但它和 OPD / RLSD 有共同关心的问题：

```text
稀疏 final reward 如何变成 token-level credit assignment？
```

三者区别：

| 方法 | token-level 信号来自哪里 | final reward 的角色 | 主要风险 |
|---|---|---|---|
| VAPO | value model / GAE | verifier reward 生成 return 和 advantage | value bias 会带崩 |
| RLSD | privileged self-teacher logprob gap | 决定更新方向，teacher 只调幅度 | privileged signal 校准不当 |
| OPD | teacher logits / distributions | 可选，不一定需要 | teacher 偏好未必等于最终正确性 |

所以 VAPO 对 OPD 线的启发是：

```text
OPD 之外，value model 仍然是一种可能的 token-level credit assignment 方案；
但必须处理长序列 bias、长度异质性和 sparse reward。
```

## 局限

### 1. 方法是系统组合，单点创新不集中

VAPO 更像一套 value-based RL 工程系统，集成了：

```text
VC-PPO 的 value pretraining / decoupled GAE；
DAPO 的 Clip-Higher / token-level loss；
SIL 的 positive-example LM loss；
GRPO 的 group-sampling。
```

论文的贡献在于把这些组件系统组合并验证，但它不是一个单公式式的新算法。

### 2. 结果主要集中在 AIME24

主表是 AIME24 `avg@32`。论文提到这些技术也适用于 code reasoning，但主结果展示较集中。

### 3. 需要额外 value model 工程成本

相比 GRPO / DAPO，VAPO 需要：

```text
1. 训练 critic；
2. value pretraining / warmup；
3. explained variance 等指标监控；
4. 长短序列下 GAE 参数管理。
```

这会增加实现和调参复杂度。

### 4. Positive LM Loss 可能带来 imitation bias

对正确样本加 NLL 是高效的，但如果正确答案轨迹里存在偶然、冗余或不可泛化的 reasoning pattern，也可能被直接记住。

## 我对这篇论文的判断

VAPO 的价值在于，它把一个之前很多人回避的问题重新摆上桌面：

```text
value model 不是天然不适合 long-CoT RL，
而是 naive PPO 不适合 long-CoT RL。
```

GRPO / DAPO 的成功说明：

```text
不训练 value model 是一条稳妥路线。
```

VAPO 则说明：

```text
如果愿意付出 critic 训练和工程复杂度，
value-based PPO 仍然可能有更高上限。
```

一句话：

```text
VAPO 是一篇“把 PPO 从 long-CoT 崩溃边缘拉回来”的系统论文。
```

