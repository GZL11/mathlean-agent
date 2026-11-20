# RLSD：Self-Distilled RLVR 技术解读

本文档解读论文：

```text
Self-Distilled RLVR
https://arxiv.org/abs/2604.03128
```

## 一句话总结

RLSD 解决的是 OPSD 的核心风险：

```text
如果 teacher 的优势来自 privileged information，
直接做 distribution matching 可能导致信息泄漏和长期训练退化。
```

RLSD 的核心改法是：

```text
用 RLVR / verifier reward 决定更新方向；
用 self-distillation teacher signal 只调节 token-level 更新幅度。
```

所以它不是把 GRPO 和 OPSD loss 简单相加，而是把 self-distillation 从“目标分布”改成“credit assignment 权重”。

## 背景：为什么 OPSD 会失败

OPSD 的 teacher 是：

```text
p_T(. | x, r)
```

其中 `r` 是 privileged information，例如：

```text
reference answer
verified reasoning trace
environment feedback
```

student 是：

```text
p_S(. | x)
```

它测试时看不到 `r`。

RLSD 论文指出：如果直接让 student 匹配 `p_T(. | x, r)`，目标是 ill-posed 的。

原因是：

```text
student 无法条件化到 r；
但 teacher distribution 强依赖 r。
```

论文把 OPSD objective 分解为：

```text
L_OPSD = L* + I(Y_t; R | X, Y_<t)
```

其中：

```text
I(Y_t; R | X, Y_<t)
```

是当前 token 和 privileged information 之间的条件互信息。

这个项与 student 参数无关，student 无法消除。

直觉：

```text
teacher 每个样本都拿着答案讲题；
student 测试时没有答案；
强行逐样本匹配会逼 student 在参数里编码“答案痕迹”。
```

这就是论文所指的 privileged information leakage：

```text
不是说 student 真的“看到了” privileged information，
而是说这种逐样本 matching 会把 r 相关的偏好注入更新轨迹，
从而造成梯度污染和训练退化。
```

## RLSD 的关键洞察

论文认为 self-distillation signal 不是没用，而是用错了地方。

OPSD 把 privileged teacher signal 用来决定：

```text
更新方向
```

即 teacher 喜欢哪个 token，就直接拉高哪个 token。

RLSD 改成：

```text
环境 reward 决定方向；
teacher signal 决定幅度。
```

换句话说：

```text
正确 / 错误由 verifier 说了算；
哪个 token 更值得大力更新，由 privileged teacher signal 提供细粒度证据。
```

## 和 GRPO 的关系

GRPO 对同一条 response 的所有 token 使用同一个 sequence-level advantage：

```text
A^(i) = (R(x, y^(i)) - mean_G) / std_G
```

问题是：

```text
同一条 response 里，不同 token 对最终正确性的贡献不同；
GRPO 却给所有 token 同一个 A。
```

RLSD 保留 GRPO 的 sequence-level direction，但把它改成 token-level advantage：

```text
A_hat_t^(i) = A^(i) * token_weight_t
```

这相当于给 GRPO 加了一个 token-level credit assignment。

## Token Weight 怎么算

对 student 采样出的第 `i` 条 response：

```text
y^(i) = (y_1, ..., y_T)
```

RLSD 先计算 privileged teacher 和 student 对 sampled token 的 logprob 差：

```text
Delta_t =
log pi_theta(y_t | x, r, y_<t)
- log pi_theta(y_t | x, y_<t)
```

这里：

```text
pi_theta(. | x, r, y_<t):
  同一个模型，额外看到 privileged information r

pi_theta(. | x, y_<t):
  同一个模型，不看 r
```

然后根据 response-level advantage 的符号构造权重：

```text
w_t = exp(sign(A) * Delta_t)
```

最后得到 token-level advantage：

```text
A_hat_t
= A * ((1 - lambda) + lambda * clip(w_t, 1 - eps_w, 1 + eps_w))
```

直觉：

```text
如果整条 response 是正样本：
  teacher 更支持的 token 权重大；

如果整条 response 是负样本：
  teacher 更支持的 token 权重反向处理，避免错误方向放大；

无论如何，方向仍由 A 的符号决定。
```

## 为什么不是 GRPO + OPSD

一个自然 baseline 是：

```text
L = L_GRPO + alpha * L_OPSD
```

论文认为这会有两个问题：

```text
1. GRPO reward 是 bounded scalar；
2. OPSD KL 可能高方差、尺度不稳定。
```

线性相加会出现 scale mismatch。

更关键的是，OPSD loss 仍然可能把 privileged teacher 的偏好注入 student 的更新轨迹。

RLSD 则把 teacher signal 变成 multiplicative modulation：

```text
只改变幅度，不改变 sign。
```

这就是它和 GRPO+OPSD 的本质区别：

```text
RLSD 不让 teacher 直接决定方向，
只让 teacher 调节 token-level 的更新幅度。
```

## 一个具体计算例子

假设同一个问题 `x` 下，student 采样了两条 response：

```text
y^(1) = "2 + 3 = 5"    reward = 1
y^(2) = "2 + 3 = 6"    reward = 0
```

在 GRPO 里，组内标准化后可以简化理解为：

```text
A^(1) = +1
A^(2) = -1
```

也就是：

```text
第一条整段 response 的所有 token 都被鼓励；
第二条整段 response 的所有 token 都被压低。
```

RLSD 进一步看每个 token。以第一条 response 的最后一个 token `"5"` 为例，student 和 privileged teacher 的 logprob 是：

```text
log pi(y_t="5" | x, y_<t)       = -1.20
log pi(y_t="5" | x, r, y_<t)    = -0.40
```

其中 `r` 是 privileged information，可以翻成：

```text
特权信息
```

例如参考答案、verified trace、人工/程序验证过的推理链。此时：

```text
Delta_t = -0.40 - (-1.20) = 0.80
```

因为这条 response 的 `A = +1`，所以：

```text
w_t = exp(sign(A) * Delta_t)
    = exp(0.80)
    = 2.23
```

假设 clipping 区间是 `[0.5, 1.5]`，则：

```text
clip(w_t) = 1.5
```

如果 `lambda = 0.5`：

```text
A_hat_t
= A * ((1 - lambda) + lambda * clip(w_t))
= 1 * (0.5 + 0.5 * 1.5)
= 1.25
```

解释：

```text
"5" 本来因为整条 response 正确而应该被鼓励；
privileged teacher 也明显更支持 "5"；
所以这个 token 的正向更新幅度从 1.0 放大到 1.25。
```

再看错误 response 里的最后一个 token `"6"`。假设：

```text
log pi("6" | x, y_<t)       = -0.60
log pi("6" | x, r, y_<t)    = -2.00
Delta_t = -2.00 - (-0.60) = -1.40
```

这条 response 的 `A = -1`，所以：

```text
w_t = exp(sign(A) * Delta_t)
    = exp((-1) * (-1.40))
    = exp(1.40)
    = 4.05
clip(w_t) = 1.5

A_hat_t = -1 * (0.5 + 0.5 * 1.5) = -1.25
```

解释：

```text
"6" 本来因为整条 response 错误而应该被压低；
privileged teacher 也不支持 "6"；
所以这个 token 的负向更新幅度从 -1.0 放大到 -1.25。
```

关键点：

```text
teacher signal 没有直接把 student 拉向 teacher distribution；
它只告诉 GRPO：这一步 token 应该更大力更新，还是轻一点更新。
```

## 训练流程

每个 training iteration：

```text
1. 对每个问题 x，student 采样 G 条 responses；
2. verifier 给每条 response 一个 binary reward；
3. 用 group reward 算 GRPO advantage A；
4. teacher branch 额外看到 privileged information r；
5. 对每个 sampled token 计算 Delta_t；
6. 用 Delta_t 构造 w_t；
7. 用 A_hat_t 替换 GRPO 原来的 uniform advantage；
8. 做标准 policy update。
```

论文强调：

```text
没有额外 distillation loss；
只是替换 GRPO 里的 advantage。
```

额外成本主要是：

```text
每条 response 多一次 teacher-context forward。
```

## 实验设置

论文主要做 multimodal reasoning：

```text
model:
  Qwen3-VL-8B-Instruct

training data:
  MMFineReason-123K

benchmarks:
  MMMU
  MathVista
  MathVision
  ZeroBench
  WeMath
```

对比方法包括：

```text
Base LLM
GRPO
OPSD
SDPO
GRPO+OPSD
RLSD
```

主表中 RLSD 平均分最高：

```text
Base LLM : 51.49
GRPO     : 53.86
OPSD     : 52.49
GRPO+OPSD: 52.91
RLSD     : 56.18
```

论文还观察到：

```text
OPSD 早期可能提升，但后期退化；
RLSD 继承 GRPO 稳定性，同时有更高收敛上限。
```

## 和 OPSD 的关系

OPSD：

```text
teacher signal = 训练方向
```

RLSD：

```text
verifier reward = 训练方向
teacher signal  = token-level 幅度调节
```

所以 RLSD 可以理解成：

```text
把 OPSD 从“privileged teacher imitation”
改造成 “RLVR credit assignment helper”。
```

论文自己的说法是：

```text
self-distillation as RLVR's wingman
```

可以翻成：

```text
自蒸馏作为 RLVR 的辅助僚机
```

## 价值

RLSD 在 OPD/OPSD 发展线里的意义很大：

```text
它明确指出：privileged self-distillation 的 dense signal 有用，
但不能直接决定 policy direction。
```

这影响了后续很多工作：

```text
1. teacher signal 要做 reliability / leakage 检查；
2. dense token signal 更适合作为 credit assignment；
3. outcome reward / verifier reward 仍然是方向锚点。
```

## 局限

RLSD 也有边界：

```text
1. 仍需要 verifier reward；
2. 需要 privileged information r；
3. token weight 的指数形式依赖 clipping，否则可能不稳定；
4. 它解决的是 verifiable reward 场景，不是所有开放式生成任务。
```

## 一句话判断

OPSD 是“让模型拿着答案教自己”。

RLSD 则进一步说：

```text
可以让模型拿着答案判断哪些 token 更关键，
但是否奖励 / 惩罚，必须由环境正确性决定。
```

这也是 RLSD 相比 OPSD 更稳的根本原因。

## BibTeX

```bibtex
@misc{yang2026selfdistilledrlvr,
  title={Self-Distilled RLVR},
  author={Chenxu Yang and Chuanyu Qin and Qingyi Si and Minghui Chen and Naibin Gu and Dingyu Yao and Zheng Lin and Weiping Wang and Jiaqi Wang and Nan Duan},
  year={2026},
  eprint={2604.03128},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2604.03128}
}
```
