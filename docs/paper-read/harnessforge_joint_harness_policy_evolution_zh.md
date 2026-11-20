# HarnessForge：联合进化 Harness 和 Policy 的 Agent 系统框架

本文档解读论文：

```text
HarnessForge: Joint Harness and Policy Evolution for Adaptive Agent Systems
https://arxiv.org/abs/2606.01779
```

## 一句话总结

这篇论文不是单纯的模型训练，也不是单纯的 workflow 搜索，而是做 **agent 系统级自适应**：

```text
把一个 agent 系统拆成 harness + policy 两部分，
再让这两部分一起进化，解决“结构和执行能力不匹配”的问题。
```

核心目标是：

```text
不是只改外部流程，也不是只训内部模型，
而是让 harness 和 policy 成为一对可协同优化的匹配系统。
```

## 系统设置

论文把 agent 系统写成：

```text
G = (H, R_δ)
```

其中：

```text
H = (P, A, M)
```

分别表示：

```text
P: planning
A: action
M: memory
```

也就是说，作者把 agent 的外部执行结构明确拆成：

```text
规划模块 + 动作模块 + 记忆模块
```

而 policy 是在这个 harness 接口里执行的 reasoner，通常以冻结底座模型 + 轻量 adapter 的形式出现。

## 这篇论文想解决什么

作者观察到，很多 agent 适配方法只做一边：

```text
1. 只改 harness：比如改 workflow、tool schema、memory 结构
2. 只训 policy：比如 SFT / RL / preference learning
```

但现实里问题常常是：

```text
harness 和 policy 不匹配
```

比如：

```text
harness 暴露了更复杂的动作或记忆结构，
但 policy 不会稳定执行；
或者 policy 变强了，但 harness 仍然限制了执行方式。
```

所以 HarnessForge 的核心判断是：

```text
要把 harness 和 policy 作为一个耦合对象一起优化。
```

## 方法主线

### 1. 先跑 rollout，找失败点

对当前的 harness-policy pair，先在任务 batch 上 rollout，收集轨迹和评估结果。

然后用一个 meta-agent 做 fault attribution：

```text
判断失败主要发生在 planning / action / memory 哪一块。
```

论文把这一步叫做 fault-guided harness tailoring。

### 2. 再改 harness

根据 fault report 和历史 archive，生成新的 harness 候选。

编辑空间不是无限开放的，而是受约束的：

```text
只在 planning、action、memory 等接口内做结构化修改，
这样可执行性更强，也更容易审计。
```

### 3. 再训练 policy

对筛选出来的 survivor harness，复用成功轨迹，训练对应的 harness-conditioned adapter。

默认做法是 SFT：

```text
用成功轨迹做监督微调，
让模型学会在这个 harness 下如何执行。
```

但论文强调，这个 policy alignment 本身是训练目标无关的，也可以换成 preference 或 RL。

### 4. 选择更好的 harness-policy 对

最后用多目标过滤选出更优的 pair，进入下一轮。

所以整体是一个循环：

```text
rollout -> fault diagnosis -> harness tailoring -> policy alignment -> selection -> next round
```

## 训练数据从哪来

数据不是静态标注集，而是执行过程中产生的轨迹：

```text
1. 在当前 harness 下 rollout
2. 保留成功轨迹
3. 把这些轨迹转成监督样本
4. 用来训练新 adapter
```

所以它更像是：

```text
用执行过程自动生成训练数据，再反过来改系统。
```

## 关键公式和对象

论文里最重要的是“配对进化”这个单位：

```text
G^(r) = {(H_i^(r), R_{δ_i}^(r))}
```

每一轮先对 pair rollout，得到：

```text
T_i^(r)
J(G_i^(r); B_r)
```

其中 `J` 是多目标评估向量，表示任务成功率、答案质量、执行效率等。

之后：

```text
1. 根据错误报告改 harness
2. 根据成功轨迹训 policy adapter
3. 重新组成新的 pair
```

## 剪枝和筛选

作者不是把所有候选都无脑保留，而是做 staged Pareto filtering。

大意是：

```text
在每个过滤阶段上，
根据多目标评价向量做 Pareto 选择，
保留非劣候选中的前半部分。
```

这有两个好处：

```text
1. 节省 rollout 预算
2. 保留性能和效率之间的折中
```

## 实验设置

主实验覆盖 5 个数据集，分成 4 个 benchmark family：

```text
ToolHop
RestBench-TMDB
API-Bank
SearchQA family:
  HotpotQA
  2WikiMultiHopQA
```

底座模型用的是：

```text
Qwen3-4B
Qwen3-8B
```

## 主要结果

论文报告：

```text
1. 平均比最强的 harness-only / policy-only baseline 高 3.56%
2. 最多提升 12.0%
3. 在多数 benchmark 上达到 SOTA
4. rollout-性能折中较好
```

最明显的收益出现在：

```text
TMDB
```

同时在 tool-use 和 retrieval-heavy 场景都有效。

## 消融结论

作者做了两个关键消融：

```text
1. 去掉 harness evolution，性能明显掉
2. 去掉 policy alignment，性能也掉
```

其中 harness tailoring 的影响更大，说明：

```text
先把执行结构改对，往往比只训 policy 更关键。
```

## 为什么不是只训 policy

论文的判断是：

```text
复杂 agent 任务里，失败不只来自“模型不会想”，
还来自“结构不给力”。
```

比如：

```text
动作 schema 不对
任务分解不对
记忆暴露不足
```

所以只做 policy training，常常补不齐系统层面的缺口。

## 为什么不是只搜 harness

反过来，只搜 harness 也不够，因为：

```text
新的 harness 只有在 policy 能稳定执行时才真正有用。
```

这就是论文强调的 compatibility gap。

## policy alignment 具体是什么

默认是 SFT：

```text
用过滤后的成功轨迹训练 harness-specific LoRA adapter，
base model 冻结。
```

但作者明确说，这个步骤可以替换成：

```text
SFT / preference optimization / RL
```

所以 HarnessForge 更像一个框架，而不是单一优化器。

## 这篇工作的本质

可以把它概括成：

```text
1. harness 负责“怎么做”
2. policy 负责“怎么想”
3. HarnessForge 负责让两者匹配
```

它关注的不是单点能力，而是：

```text
agent system 的可执行兼容性
```

## 局限性

论文自己也承认一些限制：

```text
1. rollout 成本高
2. 需要结构化的编辑空间
3. 依赖 meta-agent 做故障诊断
4. 当前主要在特定 benchmark 上验证
```

## 一句话结论

HarnessForge 的核心贡献是：

```text
把 agent 适配从“单独改 harness 或单独训 policy”
推进到“harness-policy 联合进化”。
```

如果把 agent 系统看成一个整体，它解决的是：

```text
结构和执行能力必须一起对齐，系统才会稳定变强。
```

