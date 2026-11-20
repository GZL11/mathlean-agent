# Edit-R2：面向多轮图像编辑的上下文感知强化学习

> 论文：**Edit-R2: Context-Aware Reinforcement Learning for Multi-Turn Image Editing**  
> arXiv：<https://arxiv.org/abs/2606.05950>  
> 代码：<https://github.com/yuxiaooye/Edit-R2>

## 1. 背景

Edit-R2 解决的是多轮 in-context image editing 问题。传统文本驱动图像编辑大多是单轮：输入一张图和一条指令，输出一张编辑后的图。但真实创作里，用户通常会连续发多条指令，并要求系统记住前文约束，比如先改主体、再改背景、再修正先前偏好。

论文指出两类核心失败模式：

1. long-context dilution：随着历史越来越长，稀疏的文本约束变得难以恢复；
2. state contamination：前一轮的错误会污染后续状态，导致后面的生成越来越差。

因此，它把多轮图像编辑看成一个 session-level RL 问题，而不是孤立的单轮编辑。

## 2. 技术细节

### 2.1 总体框架

Edit-R2 建立在统一多模态模型 BAGEL 之上，使用 LoRA 做后训练。每一轮先做 IC-CoT，再做图像生成，然后把当前生成结果和下一轮指令拼回状态，继续下一轮。

它的关键是两个模块：

- IC-CoT（In-Context Chain-of-Thought）：把整段历史压成一个显式的 session-intent rewrite；
- Flow-matching 图像生成：在连续 latent 空间里完成编辑。

### 2.2 IC-CoT

IC-CoT 不是普通 CoT，而是面向编辑会话的“意图重写器”。它读取完整的交错历史 $s_t$，生成增强指令 $\tilde{T}_t$，把散落的历史约束压缩成更短、更稳定的可执行状态表示。

这一步的目标不是单独把文字写对，而是让后续图像生成更容易抓住真正该保留的会话约束。

### 2.3 unified multi-turn policy optimization

Edit-R2 同时优化两种信号：

- 离散文本空间里的 IC-CoT 生成；
- 连续 latent 空间里的图像生成。

训练时每轮会得到 reward $r_t$，由三部分组成：

- IF：instruction following
- CC：content consistency
- GA：global awareness

默认权重为 `0.33 / 0.33 / 0.33`，并用 prefix-valid advantage refinement 过滤已经坏掉的 rollout。论文把这个阈值设为 $\delta = 0.5$：如果某轮 reward 低于阈值，就提前终止该 rollout，不让坏轨迹污染后续归一化和更新。

### 2.4 MICE-Bench

论文还构造了 MICE-Bench，一个 720 条样本、每条 3 轮的多轮 in-context editing benchmark，覆盖两类任务：

- Content Memory
- Content Understanding

它的特点是自动评测，并且显式要求保留 session-level constraints，而不是只看当前一轮。

## 3. 主实验结果

### 3.1 主结果

论文主张 Edit-R2 在 MICE-Bench 上明显优于 BAGEL 及其它基线，并在 EdiVal-Bench / GEdit-Bench-EN 上也保持竞争力。摘要页确认的关键结论是：Edit-R2 substantially improves multi-turn in-context editing and achieves competitive performance against strong baselines.

### 3.2 评测设置

MICE-Bench 关注三类指标：

- IF: instruction following
- CC: content consistency
- GA: global awareness

论文把它们用于多轮累计约束下的自动评测。

## 4. 消融实验

### 4.1 消融结论

摘要页已确认的消融结论是：

- IC-CoT 是核心贡献之一；
- trajectory filtering 对稳定训练很关键；
- unified objective 同时优化 reasoning 和 generation；
- MICE-Bench 上的提升主要来自多轮上下文建模，而不是单轮图像修饰能力。

### 4.2 奖励与评测

论文使用 IF / CC / GA 作为自动评测核心，并把它们组合成多轮累计约束下的 session reward。具体数值表建议直接参考原文 PDF 图表。

## 5. 结论

Edit-R2 的核心不是“更强的单轮编辑模型”，而是把多轮编辑里的 session intent recovery、reward shaping 和 trajectory filtering 统一到一个 RL 训练框架里。它最重要的贡献有三点：

1. 把多轮图像编辑明确建模成 session-level RL；
2. 用 IC-CoT 把长历史压成可执行的显式意图；
3. 用 prefix-valid refinement 把坏轨迹挡在优化之外。

对创作类 agent 来说，这篇论文的价值在于：它给出了一个很实用的多轮编辑训练闭环，而不是只做单轮图像修饰。
