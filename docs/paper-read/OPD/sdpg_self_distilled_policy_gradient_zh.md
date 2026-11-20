# SDPG：Self-Distilled Policy Gradient 论文解读

论文：

```text
Self-Distilled Policy Gradient
```

链接：

```text
https://arxiv.org/abs/2606.04036
```

项目页：

```text
https://lauyikfung.github.io/SDPG/
```

## 一句话总结

SDPG 的核心不是“再加一个蒸馏项”，而是：

```text
把 RLVR 的稀疏 verifier 信号
和“带特权上下文的自蒸馏”结合起来，
用全词表 on-policy reverse KL 给正确轨迹提供密集 token 级监督。
```

它解决的是这类问题：

```text
只有 outcome reward 太稀疏，训练慢、方差大；
只做自蒸馏又容易漂，尤其小模型更明显。
```

## 背景

SDPG 处在 reasoning post-training 这条线里，和 GRPO / RLVR / OPSD 都有关系，但它的特殊点是：

```text
teacher 不是外部大模型，
而是同一个模型在“特权上下文” c 下的 privileged branch。
```

训练时有两个视图：

```text
student: 只看题目 x
teacher: 额外看特权上下文 c + x
```

这里的 `c` 可以理解成：

```text
答案侧可见但推理时不可见的信息
```

因此它本质上是：

```text
self-distillation + RLVR + policy regularization
```

## 总目标

SDPG 的总损失写成：

```text
L_SDPG = L_out + β(k) L_OPD^+ + α L_K
```

其中：

```text
L_out   : verifier 驱动的 outcome policy gradient
L_OPD^+ : 只在正优势轨迹上启用的 full-vocab OPD
L_K     : reference-policy KL anchor
```

直觉是：

```text
L_out 负责“找对答案”；
L_OPD^+ 负责“把正确轨迹上的 token 级信号变密”；
L_K     负责“别跑飞”。
```

## 1. Outcome policy gradient

SDPG 保留标准 RLVR 的群组式 verifier 训练。

对同一个 prompt 采样一组回答 `y^(i)`，用 verifier 打分：

```text
R(x, y^(i))
```

再做组内标准化：

```text
A_out^(i) = (R(x, y^(i)) - μ_G) / (σ_G + ε_std)
```

这和 GRPO 一类方法的思路一致：

```text
好的 rollout 推高；
差的 rollout 压低。
```

## 2. Full-vocabulary OPD

这篇论文最关键的是这个项。

student 分布：

```text
p_t = π_θ(. | x, y_<t)
```

teacher 分布：

```text
q_t = π_θ(. | c, x, y_<t)
```

对采样到的 reasoning prefix，做全词表 reverse KL：

```text
ℓ_OPD_{i,t}
= Σ_{a in V} p_{i,t}(a) log [ p_{i,t}(a) / SG[q_{i,t}(a)] ]
```

这里的 `SG` 是 stop-gradient。

这表示：

```text
不是只蒸馏采样 token，
而是把 teacher 在整张 vocab 上的偏好都传给 student。
```

和普通 KD / 单 token 蒸馏相比，它更密、更稳定。

## 3. Positive-advantage gate

SDPG 不是对所有 rollout 都做 OPD，而是只对正优势样本启用：

```text
m_i = 1[A_out^(i) > 0]
L_OPD^+ = E[ Σ_{i,t} m_i ℓ_OPD_{i,t} ]
```

这个设计很重要，因为：

```text
特权上下文可能给出“看起来合理、但全局是错的”续写；
如果把这种轨迹也强行蒸馏进去，会把 student 往坏方向拉。
```

所以它的逻辑是：

```text
先让 verifier 选出“值得学”的 rollout，
再把这些 rollout 的 token 级信号蒸馏密一些。
```

## 4. Warmup-decay 权重

蒸馏系数不是常数，而是：

```text
β(k) = β_base min(1, k / T_warm) min(1, (T - k) / T_decay)
```

含义很直接：

```text
前期先 warm up，别太早相信 privileged teacher；
中期蒸馏最强；
后期 decay，让模型不要过度依赖训练时才看得到的特权信息。
```

这和很多 OPD 里“先对齐、再放松”的逻辑是一致的。

## 5. Reference-policy KL

SDPG 还加了一个参考策略锚点，防止策略漂移过大。

项目页里给了两种 surrogate：

```text
URKL: 反向 KL 形式
UFKL: 正向 KL 形式
```

它们都和 outcome objective 叠加使用，作用是：

```text
让密集蒸馏不要压过 reward 优化，
同时维持 reasoning policy 的稳定性。
```

## 训练流程

1. 采样 `(x, c)`。
2. 用 student policy `π_θ(.|x)` 生成一组回答。
3. 用 verifier 给每个回答打分，算 group-relative advantage。
4. 对正优势 rollout，在对应 prefix 上算 full-vocab OPD。
5. 加上 reference KL，一起更新。

所以它不是纯 SFT，也不是纯 RL：

```text
它是 RLVR 主干 + privileged self-distillation 辅助。
```

## 计算例子

假设某个题目只有两个候选 rollout：

```text
y^(1): 正确
y^(2): 错误
```

verifier 打分后得到：

```text
R(x, y^(1)) = 1
R(x, y^(2)) = 0
```

组内标准化后：

```text
A_out^(1) > 0
A_out^(2) < 0
```

那么：

```text
只对 y^(1) 计算 OPD^+
不对 y^(2) 蒸馏
```

这时 student 只会学“正确轨迹上的 teacher 偏好”，而不会把错误轨迹也一起模仿进去。

如果某个 prefix 下 teacher 偏好是：

```text
token A: 0.70
token B: 0.20
token C: 0.10
```

student 偏好是：

```text
token A: 0.40
token B: 0.30
token C: 0.30
```

那么 full-vocab OPD 会把整个分布拉向 teacher，而不是只处理采样到的单个 token。

## 实验设置

项目页给出的实验设置是：

```text
模型: Qwen3
训练步数: 400
训练集: DAPO-Math-17k
评测: AIME2024, AIME2025, AMC23
指标: pass@1 mean@32
```

## 主结果

Qwen3-4B 的结果里，SDPG 比 GRPO 和 RLSD 更稳，也更强：

| Method | AIME24 Last | AIME24 Best | AIME25 Last | AIME25 Best | AMC23 Last | AMC23 Best |
|---|---:|---:|---:|---:|---:|---:|
| GRPO | 0.280 | 0.316 | 0.242 | 0.279 | 0.714 | 0.739 |
| RLSD | 0.378 | 0.395 | 0.300 | 0.304 | 0.813 | 0.813 |
| SDPG-URKL | 0.380 | 0.401 | 0.307 | 0.308 | 0.863 | 0.863 |
| SDPG-UFKL | 0.380 | 0.408 | 0.327 | 0.335 | 0.858 | 0.870 |

项目页的结论很明确：

```text
SDPG 在 AIME24 / AIME25 / AMC23 上都优于 GRPO 和 RLSD；
UFKL 在部分指标上略强，URKL 也很稳。
```

## 训练动态

论文强调了三个现象：

```text
1. 早期就拉开差距，约前 50 step 已经能看到优势。
2. entropy 不会像 RLSD 那样迅速塌掉。
3. response length 更健康，不会极短化。
```

这说明它的收益不只是最终精度，更是训练过程的稳定性。

## 消融结论

项目页的消融结论是：

```text
去掉 OPD，会损失早期加速和准确率优势；
去掉 reference KL，会更容易出现长度变短、entropy 失控；
去掉蒸馏权重调度，稳定性会更差。
```

也就是说：

```text
OPD 提供密集监督，
KL anchor 提供稳定边界，
gate 和 schedule 防止 privileged signal 过强。
```

## 和 RLSD / OPSD 的关系

SDPG 可以看成是：

```text
RLVR 的 outcome 优化
+ OPSD 风格的 full-vocab self-distillation
+ KL regularization
```

但它和 OPSD 的不同点也很明显：

```text
OPSD 更像“teacher 自己的 on-policy 蒸馏”；
SDPG 更像“把 verifier 和 privileged self-distillation 绑在一起”。
```

换句话说：

```text
OPSD 重点是 on-policy 蒸馏本身；
SDPG 重点是蒸馏如何帮助稀疏 reward 的 RLVR 更稳、更快。
```

## 一句话结论

SDPG 的核心贡献可以概括成：

```text
用特权上下文构造自蒸馏 teacher，
只在正优势轨迹上施加全词表 reverse KL，
再用 reference KL 和 warmup-decay 把 RLVR 训练稳住。
```

## BibTeX

```bibtex
@article{liu2026self,
  title  = {Self-Distilled Policy Gradient},
  author = {Liu, Yifeng and Zhang, Shiyuan and Zhang, Yifan and Gu, Quanquan},
  journal= {arXiv preprint arXiv:2606.04036},
  year   = {2026}
}
```

