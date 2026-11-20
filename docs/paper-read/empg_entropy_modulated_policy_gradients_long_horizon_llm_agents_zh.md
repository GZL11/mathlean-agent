# Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents 论文解读

论文：

```text
Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents
```

链接：

```text
https://arxiv.org/abs/2509.09265
```

项目页：

```text
https://empgseed-seed.github.io/
```

## 一句话总结

EMPG 解决的不是“再加一个 dense reward”，而是：

```text
标准 policy gradient 的更新幅度天然和 entropy 绑在一起，
会让 confident correct action 更新太小、uncertain action 更新太大。
```

所以它做的是：

```text
用 step-level entropy 去重标定每一步的 advantage，
再加一个 future clarity bonus，鼓励 agent 走更可预测的路径。
```

## 背景

长程 LLM agent 的典型困难是：

```text
reward 很稀疏，credit assignment 很难；
长 horizon 下，前面很多步骤到底有没有贡献，不容易判断。
```

常见做法是：

```text
1. 造 dense reward
2. 用 PRM / verifier / imitation 去补中间步骤
3. 直接做 policy optimization
```

EMPG 的切入点不同：

```text
它认为问题不只在 reward 稀疏，
还在于 policy gradient 本身的学习动力学有偏。
```

作者提出一个核心观察：

```text
对于 softmax policy，
policy gradient 的 norm 和 entropy 是单调耦合的。
```

## 理论动机

项目页给出的命题是：

```text
E_{a ~ π}[ ||∇_z log π(a|s)||^2 ] = 1 - exp(-H_2(π))
```

其中 `H_2(π)` 是 Rényi-2 entropy。

这个式子可以从 softmax 的 log-prob 梯度直接推出来。

设一个状态 `s` 下有 `K` 个动作，logits 是：

```text
z = [z_1, z_2, ..., z_K]
```

softmax policy 是：

```text
π(a|s) = p_a = exp(z_a) / sum_j exp(z_j)
```

所以：

```text
log π(a|s)
= log p_a
= z_a - log sum_j exp(z_j)
```

对任意 logit `z_k` 求导：

```text
∂ log π(a|s) / ∂z_k
= 1{k=a} - p_k
```

拼成向量就是：

```text
∇_z log π(a|s) = one_hot(a) - p
```

因此平方范数是：

```text
||∇_z log π(a|s)||^2
= ||one_hot(a) - p||^2
= 1 - 2p_a + sum_j p_j^2
```

对 `a ~ π` 取期望：

```text
E_{a~π}[||∇_z log π(a|s)||^2]
= sum_a p_a (1 - 2p_a + sum_j p_j^2)
```

逐项展开：

```text
sum_a p_a * 1 = 1
sum_a p_a * 2p_a = 2 sum_a p_a^2
sum_a p_a * sum_j p_j^2
= (sum_j p_j^2) * (sum_a p_a)
= sum_j p_j^2
```

所以：

```text
E
= 1 - 2 sum_a p_a^2 + sum_j p_j^2
= 1 - sum_j p_j^2
```

Rényi-2 entropy 定义为：

```text
H_2(π) = - log sum_j p_j^2
```

因此：

```text
sum_j p_j^2 = exp(-H_2(π))
```

代回去得到：

```text
E_{a ~ π}[ ||∇_z log π(a|s)||^2 ]
= 1 - exp(-H_2(π))
```

一个具体例子：

```text
p = [0.7, 0.2, 0.1]
sum_j p_j^2 = 0.49 + 0.04 + 0.01 = 0.54
E[||grad||^2] = 1 - 0.54 = 0.46
```

逐动作验证：

```text
a=1: one_hot-p = [0.3, -0.2, -0.1], ||grad||^2 = 0.14
a=2: one_hot-p = [-0.7, 0.8, -0.1], ||grad||^2 = 1.14
a=3: one_hot-p = [-0.7, -0.2, 0.9], ||grad||^2 = 1.34

0.7 * 0.14 + 0.2 * 1.14 + 0.1 * 1.34 = 0.46
```

这说明：

```text
entropy 越高，期望梯度范数越大；
entropy 越低，期望梯度范数越小。
```

因此标准 PG 会带来两个问题：

```text
1. confident correct action 的梯度太小，学得慢；
2. uncertain exploratory action 的梯度太大，训练不稳。
```

EMPG 想做的是把这个天然耦合“重新校准”。

## 方法总览

EMPG 不是改 reward，而是改 advantage 的形状。

它由两部分组成：

```text
1. Self-Calibrating Gradient Scaling，自校准梯度缩放
2. Future Clarity Bonus，未来清晰度奖励
```

可以把它理解成：

```text
当前步：按不确定性调节更新力度
未来步：奖励那些会导向更清晰状态的动作
```

## 1. Self-Calibrating Gradient Scaling：自校准梯度缩放

论文里的 `H_step` 不是“跨 step 的均值”，而是：

```text
单个语义 step 内 token-level entropy 的平均值
```

也就是代码里这一句：

```text
step_entropy = batch.batch['old_entropy'][i][start:end].mean().item()
```

然后把一个 batch 里所有 step 的 `H_step` 收集起来，再做 batch 内归一化：

```text
H_norm = (H - min(H)) / (max(H) - min(H) + 1e-8)
```

然后定义一个缩放系数：

```text
g(H) = exp(-k * H_norm) / mean(exp(-k * H_norm))
```

直觉是：

```text
低 entropy step -> g(H) 更大 -> 更新被放大
高 entropy step -> g(H) 更小 -> 更新被压小
```

中文里可以把这个模块理解成：

```text
根据模型当前这一步有多确定，自动校准 policy gradient 的强度。
```

这就把 learning signal 从“谁更随机谁更新更猛”改成了：

```text
谁更可靠，谁更新更猛。
```

## 2. Future Clarity Bonus：未来清晰度奖励

第二项是未来清晰度奖励：

```text
f(H) = exp(-k_f * H_norm)
```

它鼓励 agent 选择那些会把未来状态推向更低 entropy 的路径。

这里的 `H_norm` 仍然是**当前这个 step 的归一化 entropy**。
所谓“future”不是指预测一个显式的未来熵函数，而是指：

```text
拿后一个 step 的 entropy 作为当前 step 的局部未来清晰度代理
```

注意：

```text
H_norm 是对“每个 step 的 entropy”做 batch 内归一化，
不是把 batch 里所有 step 压成同一个数。
```

因此：

```text
f_H[i]   = 第 i 个 step 的清晰度分数
f_H[i+1] = 第 i+1 个 step 的清晰度分数
```

它们通常不同，因为对应的是两个不同 step 的 entropy；只是二者都用同一个 batch 的 min/max 做了归一化。

在伪代码里对应的是：

```text
batch.batch['advantages'][idx][start:end] += zeta * f_H[i+1]
```

也就是说，它使用的是**下一 step 已经观测到的 entropy**，不是对未来 token 的显式建模。

直观上就是：

```text
不要只追求这一步看起来对，
还要追求后面更容易判断、更稳定的轨迹。
```

中文里可以把它理解成：

```text
奖励那些能让后续决策更清楚的动作。
```

这让 EMPG 不只是“重标定梯度大小”，还会“引导方向”。

## 最终的 advantage 调制

项目页的伪代码里，核心过程是：

1. 先收集每个 step 的 entropy。
2. 做 batch-level normalization。
3. 计算 `g(H)` 和 `f(H)`。
4. 用它们去修改 step-level advantage。
5. 再做一次整体 advantage normalization。

也就是说，EMPG 不是直接改 policy loss 的结构，而是：

```text
把原本 uniform 的 token/step advantage 变成 entropy-aware 的 advantage。
```

## 最终的 PG loss

EMPG 最终还是回到标准 policy gradient 目标，只不过把 advantage 换成经过 entropy 调制后的版本。

如果记某条 trajectory 的原始 step-level advantage 为 `A(i,t)`，则调制后的中间量可以写成：

```text
A_mod(i,t) = A(i,t) * g(H_t^{(i)}) + ζ * f(H_{t+1}^{(i)})
```

其中 `A_mod` 是先做 entropy 调制、再做 batch 内 zero-mean normalization 之前的量。  
最终用于更新的优势写成：

```text
A_final(i,t) = A_mod(i,t) - mean_batch(A_mod)
```

于是对应的 policy loss 可以写成：

```text
L_PG
= - E[ sum_t A_final(i,t) * log π_θ(a_t | s_t) ]
```

其中：

```text
A_final(i,t)
不是原始 advantage，
而是结合了 self-calibrating scaling 和 future clarity bonus 的调制版本。
```

所以 EMPG 的训练目标本质上是：

```text
标准 PG + entropy-aware advantage shaping
```

## 训练直觉

EMPG 在正样本和负样本上的行为不同：

```text
成功轨迹：
  confident steps 被强力强化，
  uncertain steps 被相对弱化。

失败轨迹：
  confident errors 被强烈惩罚，
  uncertain exploration 不会被过度打压。
```

这点很关键，因为它解决的是：

```text
hallucinated confidence
```

也就是模型对错误动作很自信、但 reward 又在告诉它错了。

## 训练伪代码的含义

项目页里还有一个实现层面的细节：

```text
step entropy 是按“语义步骤”算的，
不是按单个 token 算的。
```

这对 agent 场景很重要，因为一个 action segment 往往是：

```text
reasoning + act
```

而不是单个 token 的局部变化。

所以 EMPG 的 credit assignment 是：

```text
step-level, entropy-modulated, outcome-aware
```

## 实验设置

论文在三个长程 agent benchmark 上做了评测：

```text
WebShop
ALFWorld
Deep Search
```

对比的都是强 policy gradient baselines。

关键超参是：

```text
k = 1.0
k_f = 1.0
zeta = 0.05
```

step entropy 的定义是：

```text
单个语义 step 内 token-level entropy 的平均值
```

然后对 batch 内所有 step 的 entropy 做 min-max 归一化，再计算 `g(H)` 和 `f(H)`。

## 完整结果

### 1. ALFWorld / WebShop 主结果

| Method | ALFWorld Pick | Look | Clean | Heat | Cool | Pick2 | All | WebShop Score | Succ. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-4o | 75.3 | 60.8 | 31.2 | 56.7 | 21.6 | 49.8 | 48.0 | 31.8 | 23.7 |
| Gemini-2.5-Pro | 92.8 | 63.3 | 62.1 | 69.0 | 26.6 | 58.7 | 60.3 | 42.5 | 35.9 |
| Qwen2.5-1.5B Prompting | 5.9 | 5.5 | 3.3 | 9.7 | 4.2 | 0.0 | 4.1 | 23.1 | 5.2 |
| ReAct | 17.4 | 20.5 | 15.7 | 6.2 | 7.7 | 2.0 | 12.8 | 40.1 | 11.3 |
| Reflexion | 35.3 | 22.2 | 21.7 | 13.6 | 19.4 | 3.7 | 21.8 | 55.8 | 21.9 |
| PPO (with critic) | 64.8 | 40.5 | 57.1 | 60.6 | 46.4 | 47.4 | 54.4 | 73.8 | 51.5 |
| RLOO | 88.3 | 52.8 | 71.0 | 62.8 | 66.4 | 56.9 | 69.7 | 73.9 | 52.1 |
| GRPO* | 87.9 | 40.0 | 78.1 | 35.7 | 65.2 | 44.4 | 65.6 | 78.0 | 58.2 |
| GRPO + EMPG* | 85.5 | 33.5 | 78.9 | 76.2 | 74.7 | 69.1 | 73.7 | 80.4 | 60.8 |
| DAPO* | 88.1 | 61.4 | 82.5 | 90.1 | 83.9 | 69.5 | 80.8 | 85.9 | 73.2 |
| DAPO + EMPG* | 97.7 | 80.7 | 87.5 | 87.0 | 88.3 | 80.0 | 88.1 | 86.8 | 73.8 |
| Qwen2.5-7B Prompting | 33.4 | 21.6 | 19.3 | 6.9 | 2.8 | 3.2 | 14.8 | 26.4 | 7.8 |
| ReAct | 48.5 | 35.4 | 34.3 | 13.2 | 18.2 | 17.6 | 31.2 | 46.2 | 19.5 |
| Reflexion | 62.0 | 41.6 | 44.9 | 30.9 | 36.3 | 23.8 | 42.7 | 58.1 | 28.8 |
| PPO (with critic) | 92.3 | 64.0 | 92.5 | 89.5 | 80.3 | 68.8 | 80.4 | 81.4 | 68.7 |
| RLOO | 87.6 | 78.2 | 87.3 | 81.3 | 71.9 | 48.9 | 75.5 | 80.3 | 65.7 |
| GRPO* | 88.8 | 43.7 | 88.1 | 70.3 | 77.7 | 56.8 | 74.8 | 77.8 | 65.6 |
| GRPO + EMPG* | 92.9 | 75.2 | 74.8 | 86.3 | 73.7 | 65.3 | 78.5 | 81.0 | 69.3 |
| DAPO* | 98.9 | 86.1 | 94.9 | 83.2 | 81.4 | 90.1 | 90.0 | 90.6 | 79.6 |
| DAPO + EMPG* | 99.0 | 86.8 | 97.3 | 94.9 | 75.8 | 90.3 | 91.6 | 92.0 | 82.7 |

表中的 `*` 表示论文作者复现的 baseline / 方法结果。

### 2. Deep Search 主结果与消融

| Method | WebWalker | HotpotQA | 2wiki | Avg. ID | Musique | Bamboogle | Avg. OOD | Avg. Overall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DAPO (Baseline) | 55.1 | 66.4 | 68.9 | 63.5 | 38.8 | 80.8 | 59.8 | 62.0 |
| + Gradient Scaling | 54.9 | 68.8 | 67.4 | 63.7 | 41.0 | 86.4 | 63.7 | 63.7 |
| + Future Bonus | 60.6 | 69.7 | 67.9 | 66.1 | 40.4 | 82.4 | 61.4 | 64.2 |
| + EMPG (Ours) | 57.5 | 71.2 | 71.0 | 66.6 | 41.8 | 84.8 | 63.7 | 65.3 |
| Gain vs. Baseline | +2.4 | +4.8 | +2.1 | +3.1 | +3.0 | +4.0 | +3.9 | +3.3 |

### 3. 消融结论

从表里可以直接读出：

```text
1. Gradient Scaling 单独就能提升基线，说明 entropy-aware reweighting 本身有效。
2. Future Bonus 单独也有效，尤其对 ID / OOD 的综合分有帮助。
3. 两者结合最好，说明“重标定当前步 + 奖励更清晰的未来路径”是互补的。
```

项目页结论是：

```text
EMPG 在三个任务上都显著优于强 PG baseline，
而且训练更稳定。
```

## 论文观察到的现象

项目页给了两个分析图：

```text
1. KL loss dynamics：baseline 后期更不稳定，EMPG 更稳
2. step-level entropy dynamics：不是只有高熵步变化大，低熵步也会被显著重塑
```

这说明 EMPG 不是只在极端不确定的地方动手，
而是在整体上重新分配学习信号。

## 它解决了什么

EMPG 解决的核心问题可以概括成三句：

```text
1. 不要让高熵 step 自动拿到更大的梯度。
2. 不要让 confident correct action 更新太慢。
3. 让未来路径更可解释、更可预测。
```

## 和常见 agent RL 的区别

它和 GRPO / DAPO / 常规 outcome PG 的区别在于：

```text
那些方法主要在 reward 或 group advantage 上做文章；
EMPG 直接处理 policy gradient 的“熵-梯度耦合”问题。
```

所以它更像一种：

```text
entropy-aware credit assignment
```

## 一句话结论

EMPG 的核心贡献可以概括成：

```text
把长程 agent 的更新从“统一按 advantage 推一把”，
改成“根据 step entropy 重标定强度，并奖励更清晰的未来路径”。
```

## BibTeX

```bibtex
@misc{wang2025harnessinguncertaintyentropymodulatedpolicy,
  title={Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents},
  author={Wang, Jiawei and Liu, Jiacai and Fu, Yuqian and Li, Yingru and Wang, Xintao and Lin, Yuan and Yue, Yu and Zhang, Lin and Wang, Yang and Wang, Ke},
  year={2025},
  eprint={2509.09265},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.09265}
}
```
