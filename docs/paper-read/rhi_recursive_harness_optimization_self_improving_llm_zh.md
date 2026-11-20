# RHI：用递归 Harness 优化让 LLM 自我改进

本文档解读论文：

```text
Recursive Harness-Instruction Optimization for Self-Improving LLM
https://arxiv.org/abs/2607.15524
```

## 一句话总结

RHI 不是训练模型参数，也不是 RL。它做的是：

```text
冻结 base LLM，
把“如何让模型解题”的外部 prompt / harness 当成可优化对象，
通过递归 self-comparison 和 LLM optimizer，不断改写 harness。
```

所以它优化的是：

```text
harness / instruction
```

不是：

```text
model weights
```

## 系统设置

每个任务实例都有一个自己的 harness：

```text
H_i^t
```

其中：

```text
i : 第 i 个任务
t : 第 t 轮优化
```

LLM 在第 t 轮用当前 harness 生成答案：

```text
R_i^t = M(H_i^t, P_i)
```

其中：

```text
M   : 冻结的 LLM
P_i : 原始问题 prompt
R_i^t : 当前答案
```

然后系统比较：

```text
R_i^t vs R_i^{t-1}
```

再基于比较结果更新 harness：

```text
H_i^{t+1} = Optimize(H_i^t, feedback)
```

## 论文想解决什么

传统 prompting / harness 设计通常是：

```text
人工写一个模板
或者对一批任务共享一个 prompt
```

问题是：

```text
1. 不同题目需要的推理策略不同
2. 单个静态 harness 很难适配所有题
3. 只采样更多答案不一定会让模型“学会怎么想”
```

RHI 的判断是：

```text
既然模型参数不能改，
那就让每个问题自己的 harness 在测试时递归改进。
```

## 什么是 harness

这里的 harness 可以理解成：

```text
包在原始问题外面的控制指令
```

它可能包含：

```text
1. 解题策略
2. 检查清单
3. 推理顺序
4. 反思要求
5. 输出格式约束
6. 针对上一轮失败的修正提示
```

例如原始问题是：

```text
Solve this math problem...
```

harness 可能变成：

```text
先识别变量；
列出约束；
不要跳过边界情况；
最后代回验证答案。
```

## 方法流程

每个任务递归执行：

```text
1. 用当前 harness H_i^t 让模型生成答案 R_i^t
2. 和上一轮答案 R_i^{t-1} 做 pairwise comparison
3. 生成 feedback，指出哪一个答案更好、原因是什么
4. 把当前 harness、历史输出、comparison feedback 喂给 optimizer LLM
5. optimizer LLM 生成下一轮 harness H_i^{t+1}
6. 重复多轮
```

最后不是输出所有轮的平均，而是：

```text
从历史 responses 里选择比较优的答案作为最终结果。
```

## Pairwise Self-Comparison

RHI 不直接问模型：

```text
你这轮答案对不对？
```

而是做相邻轮比较：

```text
R_i^t 是否比 R_i^{t-1} 更好？
```

比较通常考虑：

```text
正确性
完整性
逻辑一致性
是否满足题目要求
```

这样做的理由是：

```text
相对比较比绝对打分更稳定。
```

这和很多 preference / reward modeling 里的经验类似：让模型判断 A/B 哪个好，往往比直接给一个绝对分数可靠。

## Harness Optimizer

comparison 之后，RHI 会让 LLM optimizer 改写 harness。

优化器输入大致包括：

```text
1. 原始问题 P_i
2. 当前 harness H_i^t
3. 当前答案 R_i^t
4. 上一轮答案 R_i^{t-1}
5. pairwise comparison feedback
```

输出是：

```text
H_i^{t+1}
```

注意这里的 optimizer 不是梯度优化器，而是：

```text
一个 LLM prompt optimizer
```

它通过自然语言编辑 harness。

## 和 Best-of-N 的区别

Best-of-N：

```text
同一个 prompt 独立采样很多答案
最后挑一个
```

RHI：

```text
每一轮会根据上轮失败或不足修改 harness
下一轮的采样分布会因为 harness 改变而变
```

所以 RHI 更像：

```text
iterative prompt-level learning
```

而不是：

```text
纯采样搜索
```

## 和 TTT-Discover 的区别

TTT-Discover：

```text
测试时更新 LoRA / model weights
目标是找到单个最优 candidate solution
需要自动 reward / verifier
```

RHI：

```text
测试时不更新模型参数
只更新 harness / instruction
依赖 self-comparison / feedback 来改 prompt
```

两者都属于 test-time improvement，但粒度不同：

```text
TTT-Discover 改权重
RHI 改外部 instruction
```

## 和 HarnessForge 的区别

HarnessForge：

```text
针对 agent system，
搜索 planning / action / memory 结构，
还会训练对应 policy adapter。
```

RHI：

```text
更轻量，
主要递归优化每个任务的 prompt harness，
不做 policy adapter 训练。
```

所以 RHI 更像：

```text
per-instance harness optimization
```

而 HarnessForge 更像：

```text
system-level harness-policy co-evolution
```

## 实验任务

论文主要在三类任务上评估：

```text
数学推理
代码生成
instruction following
```

对比对象包括：

```text
standard prompting
chain-of-thought
self-refine / reflection 类方法
best-of-N / majority vote 类采样方法
其他 prompt optimization baselines
```

核心结果是：

```text
RHI 在多个 benchmark 上提升最终答案质量，
并且相对直接多采样更高效。
```

## 为什么它可能有效

RHI 的有效性来自两个假设：

```text
1. LLM 不一定一次性知道最适合当前题目的解题策略
2. 但 LLM 可以通过比较前后答案，逐步改写“该怎么解这道题”的 instruction
```

也就是说，它让模型在测试时做的是：

```text
不是直接改答案，
而是先改“生成答案的程序性提示”。
```

这比直接让模型反思答案有一个优势：

```text
harness 会显式积累到下一轮，形成可复用的局部策略。
```

## 局限性

RHI 的局限也比较明显：

```text
1. 仍然依赖 LLM 自己比较答案，self-comparison 可能错
2. 多轮 harness 优化会增加推理成本
3. 对需要外部验证器的任务，self-feedback 可能不如真实 verifier
4. prompt 变长后可能带来上下文污染
5. 每个任务单独优化，不能直接沉淀到模型参数
```

## 一句话结论

RHI 的核心贡献是：

```text
把 test-time improvement 从“多采样答案”提升到“递归优化每个任务自己的 harness”，
在不改模型权重的情况下，让 LLM 通过自比较反馈逐轮改进解题指令。
```

