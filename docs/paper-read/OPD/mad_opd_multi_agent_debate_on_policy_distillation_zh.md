# MAD-OPD：通过多智能体辩论突破 on-policy distillation 的单教师上限

本文档解读论文：

```text
MAD-OPD: Breaking the Ceiling in On-Policy Distillation via Multi-Agent Debate
https://arxiv.org/html/2605.01347v1
```

## 一句话总结

这篇论文不是在改 RL，也不是在做普通多教师蒸馏，而是在做 **on-policy distillation 的升级版**：

```text
把单教师 OPD 改成“多教师先辩论，再按辩论后的置信度加权”的监督源，
同时再加一个面向 agent 任务的 OPAD 变体，用 step-level 采样稳住长轨迹训练。
```

核心目标有三个：

```text
1. 打破单教师能力上限
2. 把 OPD 扩展到长时序 agent 任务
3. 给 agentic / code 两类任务选合适的 divergence
```

## 论文想解决什么

原始 OPD 的问题很直白：

```text
student 在自己的 on-policy trajectory 上学习，
但 teacher 还是单个模型。
```

这会带来两个瓶颈：

```text
1. teacher 本身会错，student 会继承 teacher 的错误
2. 单教师很难覆盖复杂任务里的多种正确路径
```

对 agent 任务来说，问题更严重：

```text
一步错，后面步步错。
```

所以作者的判断是：

```text
不能只靠一个 teacher，
要让多个 teacher 先辩论出更强的监督信号。
```

## 系统设置

论文的实验覆盖：

```text
agentic tasks:
  BFCL-v4
  Vita

code tasks:
  LiveCodeBench v6
  MBPP+
```

teacher / student 体系基于：

```text
Qwen3
Qwen3.5
```

六组 teacher-student 配对里，student 覆盖 1.7B–14B，teacher 覆盖 8B–32B。

## 方法主线

### 1. 多老师先辩论，再提供监督

MAD-OPD 的 teacher 不是各自独立出答案，而是：

```text
先对 student 当前状态进行多轮辩论，
再把辩论后的 collective signal 当成 privileged supervision。
```

这意味着 teacher 的监督不是静态平均，而是：

```text
“辩论后形成的共识”。
```

论文把这称作：

```text
debate-generated privileged information
```

也就是：

```text
老师之间先互相纠错，再教 student。
```

### 2. 每个 teacher 的贡献按辩论后的置信度加权

辩论结束后，每个 teacher 会输出一个 confidence score。

然后作者把这些分数 softmax 成权重：

```text
w_i = softmax(conf_i / T)
```

含义是：

```text
谁在辩论后更有把握，谁就更影响最终监督。
```

这一步很关键，因为它不是简单平均：

```text
不是所有 teacher 一票等权，
而是动态地信任更可靠的那个。
```

### 3. 在 student 的 on-policy trajectory 上做 token-level distillation

student 先自己 rollout 出一条轨迹。

然后 teacher 在“带辩论历史”的 privileged context 下，对这条轨迹 force decode，给出 token 分布。

训练目标就是让 student 在自己的轨迹上，靠近这些加权后的 teacher 分布。

所以它仍然是 OPD 的核心形式：

```text
student 在自己的轨迹上学
```

但 teacher 变强了：

```text
单个 teacher -> 多老师辩论后的共识
```

## OPAD 是什么

MAD-OPD 里还单独提出了 **OPAD**：

```text
On-Policy Agentic Distillation
```

它的目的很明确：

```text
把 OPD 做到多步 agent 任务上。
```

因为 agent 任务不是单步分类，而是：

```text
step 1 错了，step 2 的状态就偏了，step 3 更偏。
```

所以 OPAD 不是只在完整序列上监督，而是：

```text
每一步都采样、每一步都更新环境状态、
每一步都用当前 on-policy context 做蒸馏。
```

这就是它说的 **step-level sampling**。

## divergence 为什么分任务选

论文做了一个很重要的结论：

```text
agentic tasks 选 JSD
code generation 选 reverse KL
```

原因是：

```text
agentic 任务更需要稳定，不希望梯度太尖；
code 任务更需要集中在一条连贯的正确路径上。
```

所以：

```text
JSD = 有界、稳定，适合 agentic
reverse KL = mode-seeking，适合 code coherence
```

这篇论文把 divergence 选择从“拍脑袋调参”提升成了“任务结构相关的原则”。

## 实验结果怎么看

主实验表 1 里，MAD-OPD 在六组 teacher-student 配置上都排第一。

你可以抓住两个最重要的结论：

```text
1. 比单教师 OPD 更强
2. 比多教师但不辩论的 MT-OPD 更强
```

尤其在 14B + 8B -> 4B 这组上，MAD-OPD 的总体平均分最高。

### 表 3 的消融很说明问题

论文把组件逐步加上去：

```text
Base -> MT-SeqKD -> OPD -> MT-OPD -> MAD-OPD w/o Conf. Wt. -> MAD-OPD
```

结果是：

```text
1. 多教师有帮助
2. on-policy 比 off-policy 更好
3. 辩论有帮助
4. 辩论后的置信度加权还有额外收益
```

也就是说，最强结果不是某一个技巧单独带来的，而是：

```text
多教师 + on-policy + 辩论 + 动态加权
```

一起起作用。

## 这篇和普通 OPD 的区别

普通 OPD：

```text
一个 teacher
student 在自己的轨迹上学 teacher
```

MAD-OPD：

```text
多个 teacher 先 debate
再按辩论后的置信度形成监督
student 仍然在自己的轨迹上学
```

所以它解决的是：

```text
teacher ceiling + 多路径歧义 + agent 长轨迹不稳
```

## 一句话结论

MAD-OPD 的本质是：

```text
把单教师 on-policy distillation 改造成“多教师辩论式 on-policy distillation”，
再用 OPAD 把它扩展到多步 agent 任务，并用任务自适应 divergence 稳定训练。
```

