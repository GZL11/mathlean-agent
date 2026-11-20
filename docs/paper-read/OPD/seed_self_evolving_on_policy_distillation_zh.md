# SEED：自演化的 on-policy 蒸馏，用完成轨迹反哺 agent 学习

本文档解读论文：

```text
SEED: Self-Evolving On-Policy Distillation for Agentic Reinforcement Learning
https://arxiv.org/abs/2607.14777
```

## 一句话总结

这篇论文是一个 **agentic RL + on-policy distillation** 框架。

它的核心想法是：

```text
把当前 policy 自己 rollout 出来的完整轨迹，
变成训练时的 hindsight skills，
再把这些 skills 的行为效果蒸馏回 policy。
```

重点不在于“再找一个外部 teacher”，而在于：

```text
policy 既负责采样轨迹，也负责分析轨迹；
监督信号会随着 policy 一起演化。
```

## 论文想解决什么

在长时序 agent 任务里，outcome-based RL 有一个老问题：

```text
奖励通常只在 episode 结束时给一个结果，
中间每一步到底该怎么做，信号很稀疏。
```

这会带来一个 supervision gap：

```text
episode-level outcome 够粗，
token-level policy learning 够细，
两者之间缺一层可用的中间监督。
```

SEED 的目标就是补这层。

## 系统设置

论文面向的是 agentic 任务：

```text
多轮交互
工具使用
环境反馈
```

公开页面显示，它在这些场景上做实验：

```text
ALFWorld
WebShop
Search-based QA
Sokoban
EZPoints
```

底座模型包括：

```text
Qwen2.5-3B-Instruct
Qwen2.5-VL-3B-Instruct
```

## 方法主线

### Stage 1：先让 policy 学会分析完成轨迹

第一阶段是 **hindsight-skill supervised fine-tuning**。

做法是先把完成的轨迹交给当前 policy，让它产出自然语言 skills，比如：

```text
可复用的工作流
关键观察
失败规避规则
```

这一步的作用是给 policy 一个“会总结轨迹”的能力。

可以理解为：

```text
先训练一个轨迹分析器
```

## Stage 2：让 actor 和 analyzer 同步演化

第二阶段开始后，当前 policy 同时扮演两种角色：

```text
1. actor：在环境里 rollout
2. analyzer：分析自己刚完成的轨迹
```

这就是它“self-evolving”的含义：

```text
随着 policy 更新，
它采样到的轨迹会变，
它从轨迹里抽取的 hindsight skills 也会变。
```

所以 supervision 不是静态的，而是跟着当前策略分布走的。

## 什么叫 on-policy

这里的 on-policy 不是泛泛地说“训练时用当前策略”这么简单，而是：

```text
skill guidance 也是从当前 policy 自己 induced 出来的轨迹里抽取的。
```

也就是说：

```text
监督来源和采样分布是同一个 policy 产生的。
```

这比静态 skill library 更贴近当前行为和当前 failure mode。

## 什么叫 dense token-level supervision

Outcome RL 通常只看整条轨迹的成败。  
SEED 进一步把 hindsight skills 转成 token-level 信号。

它的核心操作是：

```text
对同一批 sampled actions，
分别在普通上下文和 skill-augmented 上下文下重新打分，
看概率变化有多大。
```

这个概率变化就是 distillation 信号的来源。

直观上：

```text
如果加上某个 hindsight skill 后，
某些 action token 的概率明显更高，
那就说明这些 token 真的被这个 skill 支持。
```

因此，trajectory-level hindsight 被展开成了更密的 token-level supervision。

## 这和普通蒸馏有什么不同

普通 skill distillation 常见问题是：

```text
skill 是静态的
skill prompt 在训练和推理都要显式给
```

SEED 的不同点是：

```text
1. skill 从当前 policy 的 on-policy trajectory 中生成
2. skill 会随着 policy 更新而更新
3. skill 只在训练时作为 privileged supervision
4. 推理时不需要额外 skill prompt
```

所以它更像：

```text
把“事后总结出的经验”内化进参数里
```

而不是：

```text
把一段外部提示词挂在输入端
```

## 和 outcome RL 怎么结合

SEED 不是只做蒸馏，它把蒸馏信号和 outcome-based RL 一起优化。

也就是：

```text
RL 负责 episode 级目标
on-policy distillation 负责 token 级中间指导
```

这能改善信用分配：

```text
奖励告诉模型“这次有没有成功”
蒸馏告诉模型“哪些中间动作是被成功轨迹支持的”
```

## 为什么要让 analyzer 跟着 policy 一起变

这是论文最关键的设计。

如果 hindsight skills 是静态的，问题会很明显：

```text
policy 变了，轨迹分布变了，
旧 skills 可能就不再适配新的 failure mode。
```

SEED 认为：

```text
经验怎么变，监督也应该怎么变。
```

所以它把 analyzer 和 actor 绑在一起更新。

## 实验结论

公开页面给出的结论是：

```text
1. 在 ALFWorld 和 WebShop 上平均表现更强
2. 样本效率更高
3. 对未见场景泛化更好
4. 在视觉 agent 任务上也有效
```

具体页面展示的几个亮点包括：

```text
ALFWorld 平均表现显著优于 GRPO
只用 60% 训练数据时，也能超过 full-data GRPO
ALFWorld Unseen 上提升明显
Sokoban 和 EZPoints 也有较大收益
```

## 消融说明了什么

公开页面的消融结论很清晰：

```text
1. 去掉 hindsight skill SFT，性能下降
2. 去掉 self-evolving OPD，性能下降
3. 去掉 on-policy skills，掉得最厉害
```

这说明：

```text
静态 skill 库不如当前 policy 自己诱导出来的 skills
```

也说明：

```text
一次性的 skill 监督不够，
持续从新轨迹里蒸馏才跟得上 policy 演化。
```

## 和 OPID 的关系

这篇和你前面看的 OPID 很接近，但有一个明显推进：

```text
OPID 更像“on-policy skill distillation”
SEED 更像“会自我更新的 on-policy skill distillation”
```

SEED 的关键增量是：

```text
skill extraction 的能力也由当前 policy 负责，并且随着 RL 一起同步更新
```

所以它更强调闭环：

```text
rollout -> analyze -> distill -> update -> 新 rollout
```

## 这篇工作的本质

可以把它理解成：

```text
把完成轨迹中的经验总结成 hindsight skills，
再把这些 skills 的行为效果转成 token-level 监督，
并且让这一整套监督机制随着 policy 一起演化。
```

## 一句话结论

SEED 的核心不是“再加一个 teacher”，而是：

```text
让 policy 自己从 on-policy trajectory 里提炼 hindsight skills，
再把这些 skills 变成 dense supervision，
和 outcome-based RL 一起训练。
```

