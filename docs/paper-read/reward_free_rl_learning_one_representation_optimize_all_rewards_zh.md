# Learning One Representation to Optimize All Rewards：reward-free RL / FB 代表作解读

本文档解读论文：

```text
Learning One Representation to Optimize All Rewards
https://arxiv.org/abs/2103.07945
```

## 一句话总结

这篇论文提出了 **forward-backward (FB) representation**。  
它的目标是：

```text
先在没有固定 reward 的情况下学一个通用表示，
之后只要给出任意 reward，就能直接导出近似最优策略，
不需要重新规划，也不需要重新训练。
```

这就是后面很多论文说的 **reward-free RL** 的核心原型。

## reward-free RL 是什么

这里的 “reward-free” 不是说环境里永远没有 reward，  
而是说训练的前期不依赖某一个固定任务奖励。

更准确地说：

```text
先用无任务约束的交互数据学一组通用表示；
等测试时再给 reward / 目标状态 / 偏好，
从这个表示里直接取出对应策略。
```

所以它关注的是：

```text
“能不能学一个 representation，之后尽量不再为新 reward 重新优化？”
```

## 论文要解决什么

传统 RL 通常是：

```text
1. 给定一个 reward
2. 训练一个 policy
3. 换 reward 再重来
```

这很不通用。  
作者想做的是：

```text
先预训练一个和 reward 无关的控制表示，
让它后面能快速适配任意 reward。
```

## 核心对象

FB 不是只学一个 Q 函数，而是学两个表示：

```text
F(s, a, z)
B(s, a)
```

其中：

```text
F : forward representation
B : backward representation
z : reward / task descriptor 对应的 latent
```

直觉上：

```text
B 负责把状态-动作对投影到一个表示空间里；
F 负责在给定 z 时，预测从当前 state-action 出发能“朝哪个方向”到达。
```

## 策略怎么定义

FB 里，给定 latent `z` 后，策略直接写成：

```text
π_z(s) = argmax_a F(s, a, z)^T z
```

意思是：

```text
对每个动作 a，
看它和当前任务向量 z 的匹配程度，
选得分最高的动作。
```

这就是它“直接导出策略”的关键。

## 训练分两阶段

### 1. 无监督 / reward-free 预训练

第一阶段不先绑定具体任务 reward，而是利用 reward-free 交互数据和 replay buffer 来训练 `F` 和 `B`。

论文说这个阶段用的是：

```text
off-the-shelf deep learning methods + TD learning
```

也就是说，它本质上是一个带自举的表示学习过程。

### 2. 测试时根据 reward 构造 z_R

当你事后拿到一个 reward function `R` 时，论文会先把它变成一个 reward latent：

```text
z_R
```

这个 latent 可以由 reward 观测或者显式 reward 描述估计出来。  
之后直接用：

```text
π_{z_R}(s)
```

就能执行任务。

## 训练目标的直觉

FB 的 unsupervised loss 的目标是让表示满足类似 Bellman consistency 的关系。  
如果训练足够好，作者证明：

```text
对任意 reward，导出的策略都能近似最优。
```

如果训练不完美：

```text
性能退化和无监督表示误差成正比。
```

所以它不是“玄学表示学习”，而是有理论保证的 reward-free 控制表示。

## 为什么叫 forward-backward

因为它学的是两类互补表示：

```text
forward: 从当前状态动作出发，往未来推
backward: 从状态动作数据里总结可达结构
```

可以把它理解成：

```text
forward 负责规划方向感
backward 负责环境结构记忆
```

二者配合，就能在 reward 出现之后快速导出 policy。

## 和普通 model-based RL 的区别

FB 不是显式学环境模型再搜索。  
它更像是：

```text
直接学一个“可控制表示”
```

所以推理时不需要再做复杂 planning。

## 和 goal-conditioned RL 的区别

它和 goal-conditioned RL 有点像，但更一般：

```text
goal-conditioned RL 通常只适合显式目标状态
FB 试图覆盖更一般的 reward function
```

也就是说，它不是只学“去某个目标点”，而是学“对任意 reward 都能调起来”的表示。

## 这篇论文的作用

这篇是很多后续工作的基础。  
你前面问的 MORL 论文借的就是这个思想：

```text
把 reward-free 的 FB 作为通用表示底座，
再往多目标偏好里塞具体任务信号。
```

所以它在后续文献里的地位很像：

```text
一个通用的 reward-free pretraining backbone
```

## 局限性

这类方法的问题也很明确：

```text
1. 表示学不好时，后面所有 reward 都受影响
2. 训练稳定性和表示可辨识性比较敏感
3. 对连续高维环境，优化可能更难
4. 真正“任意 reward 都最优”在实践里做不到，只能近似
```

## 一句话结论

这篇论文的核心贡献是：

```text
在 reward-free 交互阶段学出 forward-backward 表示，
使得之后只要给任意 reward，
就能直接从表示里导出对应策略。
```

如果把后续 MORL 论文里的那套方法往回追，源头基本就是这篇。

