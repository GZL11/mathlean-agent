# ReOPD：用 Prefix Replay 做多轮 On-Policy Distillation

本文档解读论文：

```text
Multi-Turn On-Policy Distillation with Prefix Replay
https://arxiv.org/abs/2607.04763
https://baohaoliao.github.io/ReOPD/
```

## 一句话总结

ReOPD 解决的是 **多轮 agent 场景里 on-policy distillation 太贵** 的问题。

它把训练时最贵的：

```text
在线环境 rollouts + tool calls + teacher 逐步查询
```

改成：

```text
重放 teacher 预先采集好的 prefix，
让 student 在这些 prefix 上生成当前 turn，
teacher 再提供逐 token 的 RKL 监督。
```

结果是：**student 训练阶段不再需要新的环境交互 / tool calls**，但仍保留 OPD 的局部 on-policy 结构。

## 论文想解决什么

在 agentic 任务里，普通 OPD 的训练流程很重：

```text
1. student 先和环境跑一条多轮轨迹；
2. teacher 在 student 访问到的 history 上给监督；
3. 重复这个过程做更新。
```

这会有两个问题：

```text
1. 环境交互成本高，尤其是 Python execution / search / retrieval 这类工具；
2. 多轮 history 越往后越偏，teacher 在 student 跑偏后的状态上可能不可靠。
```

ReOPD 的目标是把这件事做成一个 **离线可复用的数据问题**。

## 核心直觉

论文把多轮 OPD 里的难点概括成 **prefix trap**：

```text
prefix 越像 student 自己走出来的轨迹，越 relevant；
但 prefix 越偏离 teacher 熟悉的分布，teacher 的监督越可能不可靠。
```

所以多轮 OPD 不是简单的“越 on-policy 越好”，而是存在一个张力：

```text
student occupancy vs. teacher reliability
```

ReOPD 的做法是把这个张力转化成 **prefix 分布设计** 问题。

## ReOPD 怎么做

ReOPD 的训练数据来自 teacher 预先采集好的 trajectory pool。

先把一条 teacher trajectory 写清楚：

```text
τ_T = (x, A_1^T, O_2^T, A_2^T, O_3^T, ..., A_T^T)
```

其中：

```text
x      : 初始任务 prompt
A_t^T  : teacher 在第 t 个 turn 的输出 / action
O_t^T  : 环境返回的 observation，例如 Python 执行结果或 search 结果
```

第 `t` 个 step 前的 prefix 是：

```text
h_t = (A_1^T, O_2^T, ..., A_{t-1}^T, O_t^T)
```

也就是说，**prefix 是 teacher 轨迹里截到当前 step 之前的历史**。

训练时不是从环境重新 rollout，而是：

```text
1. 从 teacher trajectory pool 里采样一个 (trajectory, step t)；
2. 取出这个位置前的 teacher prefix h_t；
3. student 在 h_t 上生成当前 turn 的 action A_t^S；
4. teacher 在同一个 h_t 和 student 已生成 token 的上下文上给分布监督；
5. 更新 student；
6. 不执行 A_t^S，所以没有新的 environment observation。
```

关键点：

```text
不是 teacher 在 student prefix 上 rollout；
而是 student 在 teacher prefix 上 rollout one turn。
```

如果要训练下一个 turn，也不是接着 student 的 `A_t^S` 继续走环境，而是重新从 teacher pool 里取另一个 prefix。

## OPD Loss 具体怎么算

对一个被采样到的 teacher prefix `h_t`，student 生成当前 turn：

```text
A_t^S = (a_t^1, a_t^2, ..., a_t^{n_t})
```

这里的 `a_t^j` 只是 student 当前 turn 输出里的第 `j` 个 token，不是特殊 token。

ReOPD 在这段 student 输出的每个 token 位置上做 reverse KL：

```text
L_t =
sum_j KL(
  π_S(. | x, h_t, a_t^{<j})
  ||
  π_T(. | x, h_t, a_t^{<j})
)
```

其中：

```text
π_S : student distribution
π_T : teacher distribution
h_t : teacher prefix
a_t^{<j} : student 在当前 turn 已经生成的前 j-1 个 token
```

所以 ReOPD 不是只对最终答案算 loss，而是对 **当前 turn 的 student response/action tokens** 算 token-level RKL。

如果当前 turn 是工具调用，例如：

```text
Action: python("print(13*17)")
```

训练时也只是把这串文本当作 token 序列来算 KL，**不会真的执行 Python**。

## 一个具体计算例子

假设任务是：

```text
What is 2 + 3?
```

teacher 预先采过一条 trajectory：

```text
A_1^T: Action: python("print(2+3)")
O_2^T: 5
A_2^T: Final answer: 5
```

如果采样到 `t=2`，则 teacher prefix 是：

```text
h_2 =
用户问题: What is 2 + 3?
teacher A_1^T: Action: python("print(2+3)")
environment O_2^T: 5
```

student 在这个 prefix 上生成当前 turn：

```text
A_2^S = "Final answer: 5"
```

粗略切成 token：

```text
["Final", " answer", ": 5"]
```

第一个 token 位置的 context 是 `h_2`。假设只看三个候选 token：

```text
token      π_S    π_T
Final      0.50   0.80
Search     0.40   0.10
Think      0.10   0.10
```

这一位的 reverse KL 是：

```text
KL(π_S || π_T)
= 0.50 * log(0.50 / 0.80)
+ 0.40 * log(0.40 / 0.10)
+ 0.10 * log(0.10 / 0.10)
≈ 0.319
```

第二个 token 位置的 context 变成：

```text
h_2 + "Final"
```

再算一次 `KL(π_S || π_T)`。第三个 token 位置也是一样。

这个 turn 的 loss 就是：

```text
L_2 = KL_1 + KL_2 + KL_3
```

实现里可能会对 token 数做平均，但核心就是沿着 student 当前 turn 的生成路径逐 token 比较 student 和 teacher 分布。

## Step-Decay 如何应用

论文里的 step-decay 是：

```text
w_t = κ^t
```

直观的加权写法是：

```text
L = sum_t w_t * L_t
```

也就是每个 turn 的 OPD loss 乘一个随 step 递减的系数。

但实际实现更常用采样式：

```text
p_t ∝ κ^t
```

即：

```text
早期 prefix 被更高概率采到；
后期 prefix 被更低概率采到；
采到以后照常算当前 turn 的 RKL loss。
```

例如 `κ = 0.6`，一条 trajectory 有 5 个 turn：

```text
t=1: 0.6
t=2: 0.36
t=3: 0.216
t=4: 0.1296
t=5: 0.0778
```

归一化后大约是：

```text
t=1: 43%
t=2: 26%
t=3: 16%
t=4:  9%
t=5:  6%
```

所以 batch 里会更常出现早期 prefix。

## 为什么这不是普通 Offline Distillation

普通离线蒸馏通常是在固定数据上模仿 teacher。
ReOPD 不是这个味道，它保留了 on-policy 的一部分结构：

```text
student 仍然要在 prefix 上自己做决策，
只是这个 prefix 不再来自实时环境，而来自 replay。
```

因此它更像是：

```text
offline data reuse + on-policy style supervision
```

而不是纯粹的静态 imitation learning。

## Prefix 分布到底在选什么

论文说的“选择 prefix 分布”，不是简单从大量 teacher trajectory 里筛出最好的一整条轨迹。

更准确地说，采样单位是：

```text
(trajectory, step t) -> prefix h_t
```

所以它是在决定：

```text
teacher trajectory pool 里的哪些 step / prefix 应该被更多用来训练。
```

一条 teacher 轨迹有 8 个 turn，ReOPD 更可能采到 step 1/2/3 的 prefix，较少采 step 7/8 的 prefix。

理论上，论文把每个 step 的有效 prefix 分布写成 `ρ_t`，目标是同时靠近：

```text
1. student occupancy d_S^t
2. teacher occupancy d_T^t
```

它的桥接形式是：

```text
ρ_t^*(h_t | x)
∝ [d_S^t(h_t | x)]^{γ_t}
  [d_T^t(h_t | x)]^{1-γ_t}
```

其中：

```text
γ_t = 1 : 完全 student-on-policy
γ_t = 0 : 完全 teacher-supported
```

普通 online OPD 更接近 `γ_t = 1`，ReOPD 使用 teacher pool，更接近 teacher-supported roll-in，但通过 step-decay 把训练质量差异纳入进来。

## 什么叫 Teacher Reliability

这里容易误解。论文不是说 teacher 在自己的 trajectory 上也会偏移。

teacher prefix 本身当然来自 teacher trajectory，所以它在 `d_T` 下是 supported 的。

问题是：训练目标最终是 student。理想 OPD 想在 student 自己会访问的 history 上训练：

```text
d_S^t(h_t | x)
```

ReOPD 用的是 teacher prefix：

```text
d_T^t(h_t | x)
```

所以需要考虑：

```text
这个 teacher prefix 对 student 来说像不像它自己会走到的状态？
```

论文里的 teacher reliability error 定义为：

```text
ε_T,t(x, h_t)
= | loss(student, ideal target q_t^*)
   - loss(student, teacher π_T) |
```

如果 teacher 在这个 history 上给出的分布接近理想目标 `q_t^*`，误差小，就叫可靠。

但 `q_t^*` 不可观测，所以实际用 **teacher support** 代理：

```text
prefix 越接近 teacher 自己会访问的历史，teacher 越可靠；
prefix 越远离 teacher support，teacher 可靠性风险越高。
```

这解释了 online OPD 的问题：student 自己跑出来的 prefix 虽然更 on-policy，但可能把 teacher 带到它不熟悉的坏 history 上。

## Step Index 为什么能当 Proxy

论文衡量 teacher prefix 和 student occupancy 的偏移时，理论上看的是密度比：

```text
r_t(h_t) = d_S^t(h_t | x) / d_T^t(h_t | x)
```

这个整条历史概率不好算，于是用 teacher prefix 上每一步 action 的 likelihood ratio 近似：

```text
\hat r_t
= ∏_{s<t} π_S(a_s^T | h_s) / π_T(a_s^T | h_s)
```

其中 `a_s^T` 是 teacher 在第 `s` 步真实选择的 action。

如果 student 也认为 teacher 的前序动作很自然，那么每一项不会太小，`\hat r_t` 不会快速下降。

如果 student 不太可能做 teacher 的那些动作，那么这些比值通常小于 1，乘积会随着深度下降。

更严谨地看 log ratio：

```text
log \hat r_t
= sum_{s<t} [
    log π_S(a_s^T | h_s)
  - log π_T(a_s^T | h_s)
  ]
```

如果平均每步：

```text
E[log π_T(a_s^T | h_s) - log π_S(a_s^T | h_s)] > 0
```

那么 `log \hat r_t` 会随 step 近似线性下降。

所以 step index 不是可靠性本身，也不是严格证明。

它是一个低成本 proxy：

```text
越深的 teacher prefix，
累计 teacher-student action mismatch 通常越大，
也越不像 student 自己会自然走到的 history。
```

论文也承认这是 coarse 的：它只能区分“多深”，不能区分同一个 step 内哪个 prefix 更可靠。

## 实验结论

论文在 **数学推理（Python）** 和 **search** 环境里验证了 ReOPD。

结论可以压缩成三点：

```text
1. 准确率：ReOPD 基本能持平或超过 online OPD；
2. 成本：训练阶段 0 tool calls；
3. 速度：每个 rollout 至少快 4x，项目页给出的典型数值是 math 4.2x、search 9.1x。
```

项目页还展示了多环境联合蒸馏：单个 student 可以复用多个 teacher 的轨迹池，而不必把所有环境都在线挂着。

## 实验设置和 Baseline

模型设置：

```text
student:
  主要是 Qwen3-4B-Instruct-2507
  另外有 Qwen3-8B student 搭配更大 teacher 的设置

teacher:
  Qwen3-4B-Instruct-2507
  Qwen3-8B
  Qwen3-30B-A3B-Instruct-2507
```

训练协议：

```text
1. cold start: 用更强模型采集的 trajectory 做 SFT，让模型学会工具格式；
2. teacher GRPO: 对 teacher 做 RL；
3. student distillation: 用 OPD 或 ReOPD 从 teacher 蒸馏到 student。
```

teacher prefix pool 默认直接复用 teacher 做 GRPO 时产生的 agentic trajectories，因此不需要额外采集成本。

benchmarks：

```text
Math + Python tool:
  AIME24, AIME25, AMC23, Minerva Math, OlympiadBench, MATH500

Search / retrieval:
  NQ, TriviaQA, PopQA, HotpotQA, 2WikiMultiHopQA, MuSiQue, Bamboogle
```

baselines：

```text
Base:
  未训练模型。

Cold Start:
  只经过工具格式 / 基础轨迹 SFT 的模型。

SFT:
  在 teacher trajectory 上做普通 supervised finetuning。

OPD:
  fully student-on-policy distillation；
  student 在线跑环境产生 prefix，teacher 对这些 prefix 给逐步监督。

ReOPD:
  使用 teacher prefix replay 的 off-environment OPD。

GRPO:
  teacher 自己的 RL 结果，作为 teacher scale 的参考上界，不是同尺寸 student baseline。
```

## 多环境联合蒸馏

论文还做了 multi-environment setting。

这里不是一个 teacher 同时覆盖所有环境，而是：

```text
一个 single student
同时学习 math 和 search 两个环境
每个环境各有自己的 RL-trained teacher
```

训练时：

```text
1. math 的训练数据和 search 的训练数据 concatenate；
2. shuffle 后一起喂给 student；
3. 如果样本来自 math，就用 math teacher / math teacher prefix pool；
4. 如果样本来自 search，就用 search teacher / search teacher prefix pool；
5. student 最后在两个环境的所有 benchmark 上评估。
```

这说明 ReOPD 的工程意义不只是省单个环境的 tool calls，还可以避免在联合训练时同时部署多个在线环境。

## 我怎么理解它的价值

这篇工作的意义不在于“再造一个更强的 loss”，而在于：

```text
它把 OPD 的瓶颈从算法问题，转成了轨迹资源管理问题。
```

也就是说，真正贵的不是 teacher 本身，而是：

```text
每次训练都重新生成可监督的多轮历史。
```

ReOPD 把这部分历史先采集下来，之后重复利用，适合这类任务：

```text
1. 交互环境成本高；
2. 轨迹结构明确；
3. 可以用结果导向或步骤导向的监督做训练。
```

## 局限

这类方法的边界也很明确：

```text
1. 需要先有 teacher 轨迹池；
2. 如果真实测试状态和 replay prefix 差很多，收益会下降；
3. step-decay 是有效但偏启发式的设计，不是更深层的理论最优解。
```

所以它更像一个 **高性价比的工程化 OPD 改造**，而不是通用的多轮学习理论框架。

## 和相关工作的关系

和仓库里已有的几篇 OPD 工作比：

```text
TCOD / TurnOPD：
  重点是怎么在 live rollout 里安排时间课程或 turn 预算。

ReOPD：
  重点是直接把 rollout 这件事离线化，改成 prefix replay。
```

所以它解决的不是“多轮 OPD 该怎么调度”，而是“多轮 OPD 能不能别每次都重新跑环境”。

## BibTeX

```bibtex
@misc{liao2026reopd,
  title={Multi-Turn On-Policy Distillation with Prefix Replay},
  author={Liao, Baohao and Dong, Hanze and Monz, Christof and Xu, Xinxing and Dong, Li and Wei, Furu},
  year={2026},
  eprint={2607.04763},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2607.04763}
}
```
