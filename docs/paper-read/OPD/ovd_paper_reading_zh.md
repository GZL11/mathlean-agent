# OVD 论文技术解读

本文档解读论文 **OVD: On-policy Verbal Distillation**（arXiv:2601.21968），重点说明它和 token-level OPD / Uni-OPD 的区别、teacher-student 设置、核心算法、off-policy 风险和适用场景。

论文地址：

```text
https://arxiv.org/abs/2601.21968
```

## Student / Teacher 模型速览

OVD 的 teacher 不提供 token logits，而是给轨迹或步骤打 verbal score。主要设置：

```text
数学推理:
  Student:
    Qwen2.5-Math-1.5B

  Teacher:
    QwQ-32B

Web Q&A:
  Student:
    Qwen2.5-3B-Base
    LLaMA-3.2-3B-Base

  Teacher / Environment Agent:
    Qwen2.5-7B-Base SFT-based environment agent
    Qwen2.5-14B-Base SFT-based environment agent
```

所以 OVD 的 teacher 更像黑盒评分器 / 环境代理，而不是传统 OPD 里的 token-level logprob teacher。

## 一句话总结

OVD 的核心是：不用 teacher token-level logits/logprobs，而是让 teacher 或 environment agent 给学生生成的 trajectory 打离散 verbal score，然后通过 rejection sampling 和 GRPO-style RL 训练学生。

可以概括为：

```text
传统 OPD:
  teacher supervision = token-level logprob / logits

OVD:
  teacher supervision = verbal score over trajectory / step
```

它要解决的是 token-level distillation 在长轨迹 RL 中的三个问题：

```text
1. 显存开销大
2. 需要访问 teacher token distribution，不适合黑盒 teacher
3. 逐 token 对齐会限制 student 的探索空间
```

## Student / Teacher 模型设置

OVD 按任务类型使用不同的 student 和 teacher。

### 数学推理

```text
Student:
  Qwen2.5-Math-1.5B

Teacher:
  QwQ-32B
```

论文附录写明：math tasks 使用 QwQ-32B 作为 teacher model。teacher 根据 prompt template 对 reasoning trajectories 生成 0-9 verbal score。student 初始化自 pretrained base model，不使用 instruction tuning，数学任务采用 Qwen2.5-Math-1.5B。

训练设置：

```text
base policy:
  Qwen2.5-Math-1.5B

rollout samples per prompt:
  8

rollout temperature:
  0.6

teacher score temperature:
  0.7

max prompt length:
  1024

max response length:
  3072

learning rate:
  1e-6

KL coefficient:
  0.001
```

### Web Q&A

```text
Student:
  Qwen2.5-3B-Base
  LLaMA-3.2-3B-Base

Teacher / Environment Agent:
  Qwen2.5-7B-Base SFT-based environment agent
  Qwen2.5-14B-Base SFT-based environment agent（14B 实验）
```

论文附录还给出模拟搜索环境模型：

```text
sunhaonlp/Simulation_LLM_google_7B_V2
sunhaonlp/Simulation_LLM_google_14B_V2
```

这些 environment agent 参考 ZeroSearch 训练，用来模拟 search engine outputs，并在训练中提供 step-level verbal scores，评价 query quality 和 reasoning coherence。

训练设置：

```text
base policy:
  meta-llama/Llama-3.2-3B
  Qwen/Qwen-2.5-3B

training steps:
  203

training batch size:
  64

validation batch size:
  512

max prompt length:
  4096

max response length:
  500

teacher trajectory generation rejection threshold:
  7
```

## 方法动机

传统 token-level KD / OPD 需要 teacher 在每个 token 上输出 full-vocabulary logits 或 logprobs。

显存复杂度大致是：

```text
O(B * N * L * V)
```

其中：

```text
B:
  batch size

N:
  samples per problem

L:
  trajectory length

V:
  vocabulary size
```

长推理轨迹下，`L` 很大；大模型词表下，`V` 也很大。论文以 Qwen2.5-7B、sequence length 8192 为例讨论 memory bottleneck。

OVD 改为只让 teacher 输出 verbal score：

```text
score ∈ {0, 1, ..., 9}
```

于是监督从 full-vocab token distribution 变成小词表上的离散评价，显存压力大幅下降，也允许 teacher 是黑盒模型或环境 agent。

## 轨迹表示

不同任务的 trajectory 不一样。

数学推理：

```text
y = (s_1, s_2, ..., s_K)
```

其中 `s_k` 是推理步骤，例如一句自然语言推导或一次计算。

Web Q&A：

```text
y = (q_1, d_1, s_1, q_2, d_2, s_2, ..., q_K, d_K, s_K)
```

其中：

```text
q_k:
  search query

d_k:
  retrieved documents / simulated search results

s_k:
  基于检索结果的推理步骤
```

OVD 训练学生同时学习：

```text
reasoning behavior
tool/search interaction behavior
```

## Verbal Rejection Sampling

训练时学生先 on-policy 采样：

```text
tau_i^(j) ~ pi_S(. | x_i)
```

teacher 对该 trajectory 给 verbal score：

```text
S(tau_i^(j)) ∈ {0, ..., 9}
```

同时计算 outcome reward：

```text
R(tau_i^(j))
```

论文算法中的 reject 条件是：

```text
if S(tau_i^(j)) < theta or incorrect:
  replace tau_i^(j) with teacher trajectory ~ pi_T(. | x_i)
  recompute reward
```

所以不是只看 teacher score，也看最终 correctness / outcome reward。

最终进入训练的是一个混合分布：

```text
accepted student trajectories
+
teacher replacement trajectories
```

论文理论部分称其为：

```text
mixture distribution
teacher-induced off-policy objective
```

这点很重要：OVD 虽然从 student on-policy rollout 开始，但经过 rejection / replacement 后，训练 batch 不是纯 on-policy。

## Reward 设计

论文中的 outcome reward：

```text
Web Q&A:
  R(y) = F1(y, y*)

Math:
  R(y) = delta(y, y*)
```

其中：

```text
F1:
  预测答案和参考答案之间的词重叠 soft score

delta:
  数学等价判断，通常为 0/1
```

verbal score 主要用于：

```text
rejection / selection
```

outcome reward 主要用于：

```text
RL reward / advantage
incorrect 判断
```

## Policy Optimization

OVD 使用 GRPO-style group-relative advantages。

对同一个 problem `x_i` 的多条 trajectory，计算：

```text
A(tau_i^(j)) = (R(tau_i^(j)) - mu_i) / (sigma_i + epsilon)
```

其中：

```text
mu_i:
  同一个 problem 下多条 trajectory 的 reward 均值

sigma_i:
  同一个 problem 下 reward 标准差
```

然后使用 PPO-style clipped policy gradient：

```text
L_RL = - E[min(rho * A, clip(rho, 1-eps, 1+eps) * A)]
```

其中：

```text
rho = pi_S(y | x) / pi_S_old(y | x)
```

这和 GRPO / PPO 类方法一致，区别在于训练样本经过 verbal rejection sampling 构造成了 teacher-student mixture。

## Off-Policy 问题

OVD 早期 student 很弱，大量 student trajectory 会被 rejected，因此 teacher trajectory 占比很高。

这意味着：

```text
tau_T ~ pi_T
```

但被用来更新：

```text
pi_S
```

从严格 RL 角度看，这部分是 off-policy。论文也承认它是 teacher-induced off-policy objective。

OVD 的稳定性来自 bias-variance tradeoff：

```text
保留低质量 student trajectory:
  更 on-policy
  但 reward 噪声高、错误多、训练早期不稳定

替换为 teacher trajectory:
  更 off-policy
  但质量高、方差低、训练早期更稳定
```

随着学生变强，acceptance rate 上升，混合分布会逐渐从 teacher-heavy 变成 student-heavy，更接近 on-policy。

论文理论部分给出的直觉是：

```text
初期:
  alpha_t 小，更多依赖 teacher demonstration

后期:
  alpha_t 增大，更多 accepted student trajectories
```

其中 `alpha_t` 是 student trajectory 通过 verbal threshold 的概率。

## 和 DPO 的关系

OVD 和 DPO 有相似点，但不是 DPO。

相似点：

```text
都不需要 teacher token-level logits
都可以使用偏好/质量信号指导 student
都可能使用非当前 student policy 产生的数据
```

关键区别：

```text
DPO:
  使用 fixed preference pairs
  优化 pairwise preference objective
  通常不需要在线 rollout
  不需要环境交互

OVD:
  每轮从当前 student rollout 开始
  teacher score 决定 accept/reject
  低分/错误 trajectory 替换成 teacher trajectory
  用 GRPO/PPO-style reward objective 更新
```

更准确地说，OVD 是：

```text
online rejection-sampling distillation + GRPO
```

或者：

```text
student on-policy exploration + teacher replacement 的 hybrid RL
```

## 和 Uni-OPD 的区别

两篇论文都属于 on-policy distillation 思路，但监督粒度不同。

```text
Uni-OPD:
  使用 teacher token logprob
  r_t = log pi_T - log pi_student
  问题是 token-level teacher signal 可能和 outcome reward 不一致
  解决方法是用 outcome reward 做 margin calibration

OVD:
  不使用 teacher token logprob
  使用 teacher verbal score
  问题是 token-level logits 太贵/不可得/限制探索
  解决方法是 verbal rejection sampling + GRPO
```

一句话对比：

```text
Uni-OPD:
  teacher token signal 很细，但可能方向错，所以用 outcome 校准

OVD:
  teacher token signal 太贵或拿不到，所以改用 verbal score
```

## 适用场景

OVD 更适合：

```text
1. teacher 是黑盒，拿不到 logits/logprobs
2. trajectory 很长，token-level distillation 显存不可接受
3. 任务需要环境交互，例如搜索、工具调用、agent 轨迹
4. 希望 student 保持更自由的探索，不被 teacher token distribution 逐 token 绑定
```

但它也有明显风险：

```text
1. verbal score 质量高度依赖 teacher / environment agent
2. rejection threshold 是关键超参
3. early training teacher trajectory 占比高，off-policy bias 明显
4. 如果 threshold 太高，可能退化成 imitation-heavy
5. 如果 threshold 太低，低质量 student trajectory 会大量进入训练
```

论文 ablation 也指出，完全替换 policy trajectory 为 teacher trajectory（例如 rejection threshold 过高）不一定更好，因为会损害 policy model 的探索多样性。

## 算法流程

简化版：

```text
Input:
  teacher pi_T
  student pi_S
  dataset D
  samples per problem N
  score threshold theta
  PPO clipping threshold epsilon_c

For each training iteration:
  1. sample problem batch {x_i}
  2. for each problem x_i:
       for j in 1..N:
         student 生成 trajectory tau_i^j
         teacher 给 verbal score S(tau_i^j)
         计算 outcome reward R(tau_i^j)

         if S(tau_i^j) < theta or incorrect:
           用 teacher trajectory 替换 tau_i^j
           重新计算 reward

       计算 group-relative advantage:
         A = (R - mean(R_group)) / (std(R_group) + eps)

  3. 用 PPO/GRPO clipped policy gradient 更新 student
```

## 总结

OVD 的核心不是更精细地拟合 teacher，而是反过来：把 teacher supervision 从 token-level probability matching 改成 trajectory/step-level verbal scoring。

它的核心 tradeoff 是：

```text
放弃 token-level 细粒度概率监督
换来更低显存、更强黑盒兼容性、更自然的 agent/environment feedback
```

训练早期它确实更像 teacher-guided off-policy RL / imitation-heavy RL；后期随着 student trajectory 通过率提高，才逐渐接近 on-policy learning。
