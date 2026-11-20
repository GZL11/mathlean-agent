# MaxRL：Maximum Likelihood Reinforcement Learning 技术解读

本文档解读论文：

```text
Maximum Likelihood Reinforcement Learning
https://arxiv.org/abs/2602.02710
```

论文的核心主张很直接：

```text
在 binary correctness / verifier-based 任务里，
标准 RL 优化的是 expected reward / pass@1，
但真正更“像监督学习”的目标其实是 maximum likelihood。
```

作者提出 MaxRL，把 RL 重新解释成一种对 maximum likelihood 的有限采样近似，并证明：

```text
1. RL 只对应 ML 的一阶近似。
2. MaxRL 的 truncated objective 随 rollout compute 增加会逐步逼近 ML。
3. 这种逼近不是只降低方差，而是 objective 本身更接近 ML。
```

## 一句话总结

MaxRL 的本质是：

```text
把“一个输入上模型成功的概率”看成隐式 likelihood，
然后用 on-policy sampling + 成功样本重加权，
构造一个随着 rollout 数增加而逐步逼近 maximum likelihood 的 RL 目标。
```

和普通 RL 的区别是：

```text
普通 RL:
  优化 expected reward / pass@1。

MaxRL:
  优化 log(success probability) 的截断级数近似，
  并把 higher-order pass@k 信息纳入训练。
```

## 问题设定

论文考虑的是 binary correctness settings，例如：

```text
数学题、代码生成、导航、maze、分类、verifier-based reasoning
```

对每个输入 `x`，模型会采样一个 latent generation `z`，然后通过确定性解码函数 `f(z)` 得到最终输出 `y`。成功条件是：

```text
f(z) = y*(x)
```

于是可以定义成功概率：

```text
p_theta(y*(x) | x)
  = E_{z ~ m_theta(.|x)} [ I{f(z) = y*(x)} ]
```

这里：

```text
m_theta(.|x):
  模型在输入 x 下对 latent trajectory 的分布。

f(z):
  从 latent generation 到最终输出的确定性映射。
```

这个成功概率就是论文里的隐式 likelihood。

## ML 和 RL 分别在优化什么

### Maximum Likelihood

ML 目标是最大化：

```text
J_ML(theta)
  = E_{x ~ rho} [ log p_theta(y*(x) | x) ]
```

也就是：

```text
最大化模型在正确答案上的对数成功概率。
```

这和分类里的 cross-entropy 非常像：

```text
CE(x, y*)
  = -log pi_theta(y*|x)
```

### Reinforcement Learning

binary reward 定义为：

```text
r(x, z) = I{f(z) = y*(x)}
```

则 RL 目标是：

```text
J_RL(theta)
  = E_{x ~ rho} E_{z ~ m_theta(.|x)} [ r(x, z) ]
  = E_{x ~ rho} [ passrate(x) ]
```

也就是标准 expected reward / pass@1：

```text
只关心“这次采样成不成功”。
```

## 核心洞察：RL 只是 ML 的一阶近似

论文最重要的推导是把：

```text
log p
```

写成 failure events 的 Maclaurin 展开。令：

```text
p = passrate(x)
```

则：

```text
log p
  = - sum_{k=1}^∞ (1 - p)^k / k
```

而：

```text
fail@k(x) = (1 - p)^k
pass@k(x) = 1 - fail@k(x)
```

所以：

```text
∇ J_ML(x)
  = sum_{k=1}^∞ (1/k) ∇ pass@k(x)
```

这个式子很关键，含义是：

```text
maximum likelihood 不只是优化 pass@1，
而是一个 infinite harmonic mixture：
pass@1, pass@2, pass@3, ...
```

而标准 RL 只保留第一项：

```text
∇ J_RL(x) = ∇ pass@1(x)
```

所以论文的判断是：

```text
RL 优化的是 ML 的一阶近似，
而不是 ML 本身。
```

## MaxRL 的 truncated objective

由于完整的无限级数很难直接估计，MaxRL 定义截断目标：

```text
J_MaxRL^(T)(x)
  = - sum_{k=1}^T (1 - p)^k / k
```

对应梯度：

```text
∇ J_MaxRL^(T)(x)
  = sum_{k=1}^T (1/k) ∇ pass@k(x)
```

它有三个关键性质：

```text
T = 1:
  退化为 RL。

T -> infinity:
  收敛到 ML。

中间的 T:
  在 RL 和 ML 之间插值。
```

所以 MaxRL 的定义方式不是“改 reward shaping”，而是：

```text
直接定义了一个随着 compute 增加而逼近 ML 的目标族。
```

## 关键梯度定理

论文证明 maximum likelihood 的梯度可以写成 success-conditioned expectation：

```text
∇ J_ML(x)
  = E[ ∇ log m_theta(z|x) | f(z)=y*(x) ]
```

含义是：

```text
ML 的梯度 = 成功轨迹上的平均 score function。
```

这使得一个很自然的 estimator 出现：

```text
sample N 条 rollouts
只对成功样本求平均
```

记：

```text
r_i = I{f(z_i)=y*(x)}
S_i = ∇ log m_theta(z_i|x)
K = sum_i r_i
```

则 estimator 是：

```text
g_hat_N(x)
  = (1/K) sum_i r_i S_i,   if K >= 1
  = 0,                     if K = 0
```

论文证明：

```text
E[g_hat_N(x)] = ∇ J_MaxRL^(N)(x)
```

也就是说：

```text
rollout 数 N 越大，
这个 estimator 对 ML 梯度的近似阶数越高。
```

## 为什么它和 REINFORCE / GRPO 不一样

### REINFORCE

REINFORCE 的梯度本质上是在优化：

```text
pass@1
```

它相当于：

```text
对所有采样平均，
只关心有没有成功。
```

### GRPO

GRPO 用 reward 的 group mean/std 做归一化：

```text
A_i = (r_i - mean(r)) / std(r)
```

论文把它视为一个不同的 population-level objective。它不是 ML 的渐近近似，也不会随着更多 rollouts 自动变成 ML。

### MaxRL

MaxRL 的关键差异是：

```text
用成功样本数 K 做归一化，
而不是总样本数 N 或 reward std。
```

论文给出的最终 on-policy 实现是：

```text
g_hat(x)
  = (1 / (N * r_hat(x))) sum_j (r_j - r_hat(x)) S_j
```

其中：

```text
r_hat(x) = (1/N) sum_j r_j
```

如果没有任何成功样本：

```text
g_hat(x) = 0
```

在代码层面，这几乎就是把 GRPO 的 advantage normalization 改成了：

```text
mean reward
```

而不是：

```text
std reward
```

## Tensor / 代码视角

假设：

```text
B:
  batch size

G:
  每个 prompt 的 rollout 数量

L:
  response length
```

对每个 prompt `x`：

```text
rollouts:
  y^(1), ..., y^(G)

rewards:
  r_1, ..., r_G

score functions:
  S_1, ..., S_G
```

在 LLM 训练里，`S_j` 就是：

```text
∇ log pi_theta(y^(j) | x)
```

而 MaxRL 的 token-level loss 仍然是标准 policy gradient 形式，只是 advantage 变成：

```text
A_i = (r_i - mean(r)) / mean(r)
```

所以对单个 token 的 loss 仍可写成：

```text
loss = - logprob * advantage
```

区别在于这个 advantage 的 population-level 含义更接近 ML。

## 为什么说它“更像交叉熵”

在可微分类任务里，ML 就是 cross-entropy。论文用 ImageNet 做了一个可微对照实验：

```text
REINFORCE:
  优化 pass@1，学得很慢。

MaxRL:
  rollouts 越多，越接近 cross-entropy 训练。
```

这说明 MaxRL 的目的不是替代监督学习，而是：

```text
在不可微的 correctness-based 任务里，
把 RL 的训练动态拉回到“像最大似然训练”的方向。
```

## 实验结论

论文做了四类实验。

### 1. ImageNet

这里是一个“可直接算 ML”的控制实验。结果显示：

```text
MaxRL 随 rollout 增加，会逐渐逼近 exact maximum likelihood。
```

而 REINFORCE 很难爬升。

### 2. Maze

在大规模随机迷宫任务里：

```text
MaxRL 在 pass@1、pass@128、pass@256 都优于 GRPO 和 RLOO。
```

说明它在数据多、compute 多时更能利用额外采样。

### 3. GSM8K

在固定数据、多 epoch 训练场景中：

```text
MaxRL 的 pass@1 上升更慢，
但 pass@k 衰减更小，
最终平均性能更高。
```

这说明它比 GRPO 更不容易把模型训得过于尖锐。

### 4. Qwen3 大模型训练

在 1.7B / 4B 规模数学训练中，MaxRL：

```text
1. 在多个 benchmark 上不弱于 GRPO。
2. pass@k 更健康。
3. 测试时可获得很高的 sampling efficiency gain。
```

论文提到最高约：

```text
20x test-time scaling efficiency gain
```

## 论文的真正贡献

这篇论文不是提出一个新 reward model，也不是一个新 verifier，而是提出了一个新的视角：

```text
binary correctness RL 的目标不是 expected reward，
而是 maximum likelihood over success probability。
```

然后它进一步给出：

```text
1. ML 的 pass@k 展开。
2. 一个有限 rollout 下的截断目标族。
3. 一个和这个目标族严格对应的 unbiased estimator。
4. 一个简单的 on-policy 实现。
```

所以 MaxRL 的价值在于：

```text
把“RL 训练越久越尖锐、pass@k 越塌”这个经验现象，
重新解释为 objective choice 问题，而不是单纯优化技巧问题。
```

## 和其他论文的关系

### 和 GRPO 的关系

GRPO 仍然是 expected reward family，核心在 group normalization。

MaxRL 则认为：

```text
如果任务的本质是 binary correctness，
那优化 pass@1 本身就不够，
更合理的是逼近 log likelihood。
```

### 和 pass@k 直接优化的关系

MaxRL 不是直接把目标写成 pass@k，而是通过：

```text
log p = infinite sum of fail@k
```

间接得到一个对所有 pass@k 的 harmonic mixture。

### 和 Uni-OPD / TrOPD / X-OPD 的关系

这些工作关注的是：

```text
teacher-student supervision reliability
```

而 MaxRL 关注的是：

```text
binary correctness 任务的 RL objective 本身应该是什么。
```

它们处在不同层面，但都在试图修正“标准 RL 不一定是最合适的目标”这个问题。

## 局限

论文也明确说了：

```text
1. 目前主要适用于 binary reward / correctness settings。
2. 不直接处理 continuous rewards。
3. 不直接处理 multi-turn 或 PPO-style off-policy 情况。
```

也就是说，它最适合：

```text
verifier-based reasoning / code / maze / classification
```

而不是一般连续奖励控制任务。

## 总结

MaxRL 的一句话总结是：

```text
在 binary correctness 任务中，
RL 不是为了替代 maximum likelihood，
而是一个有限采样下逼近 maximum likelihood 的工具。
```

它最重要的 insight 是：

```text
RL 只是一阶近似；
MaxRL 通过 success-conditioned averaging 和 truncated pass@k expansion，
把更多 higher-order correctness 信息纳入训练。
```

因此它比标准 RL 更像一个“可计算的 ML 近似器”，而不是单纯的 reward maximizer。
