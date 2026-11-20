# Mechanistic World Models：从预测走向机制化科学发现

本文档解读论文：

```text
From Observation to Insight: Mechanistic World Models and the Quest for Autonomous Discovery
https://arxiv.org/abs/2607.12474
```

## 一句话总结

这篇论文不是 OPD / RL 训练算法，也不是一篇带 benchmark 主表的实验论文，而是一篇 **科学发现 AI 的概念框架论文**。

它的核心观点是：

```text
当前 AI for Science 主要擅长预测；
但科学发现真正需要的是解释，也就是找到可复用的生成机制。
```

作者提出：

```text
Mechanistic World Models, MWM
```

作为一种新的世界模型范式：

```text
不要只围绕 observation -> prediction 建模，
而要围绕 variables、mechanisms、structure 来组织知识。
```

## 论文想解决什么

现在的 foundation models 在科学里已经很强：

```text
蛋白质结构预测
天气预测
材料性质预测
生物数据建模
```

但作者认为它们大多还是：

```text
predictors, not explainers
```

也就是说，它们能回答：

```text
会发生什么？
```

但不一定能回答：

```text
为什么会这样？
背后的变量是什么？
哪个机制在起作用？
如果干预某个变量，会发生什么？
这个机制能否迁移到另一个系统？
```

论文的核心判断是：

```text
科学发现本质上不是更强预测，而是知识组织问题。
```

## Prediction 和 Insight 的区别

论文从科学哲学出发区分：

```text
description / prediction:
  描述现象如何发生，预测未来观测。

explanation / insight:
  揭示为什么这样发生，找到可复用的生成机制。
```

例子：

```text
预测哪些物体会浮起来，是 prediction；
发现浮力原理，是 explanation。
```

对应到机器学习：

```text
普通模型:
  学 x -> y 的预测映射

科学发现模型:
  学生成 y 的变量、机制和结构
```

## 为什么普通 World Model 还不够

普通 world model 通常做：

```text
observation_t -> latent state_t -> predicted observation_{t+1}
```

重点是：

```text
预测未来状态。
```

MWM 不是否定预测，而是说：

```text
预测是必要条件，但不是充分条件。
```

MWM 要把 latent dynamics 从一个整体黑盒函数改成：

```text
显式变量 + 可复用机制 + 机制绑定结构
```

这样模型不只知道未来会怎样，还能组织出：

```text
哪些变量存在
哪些机制作用在这些变量上
这些机制如何组合成当前系统的解释
```

## 机制是什么

论文先给一个概念定义：

```text
m = (V, f)
```

其中：

```text
V : entities / properties，也就是变量或实体属性
f : activities / transformations，也就是让这些变量变化的过程
```

直观理解：

```text
机制 = 一组变量 + 作用在这些变量上的变换规则
```

机制可以是：

```text
物理定律
生化通路
神经回路
经济系统里的反馈过程
机器人环境里的动力学规则
```

作者的观点是：

```text
科学定律可以看作解释范围特别广的机制。
```

所以 MWM 不把 law 和 mechanism 分开，而是统一成：

```text
reusable explanatory mechanisms
```

## MWM 需要哪些能力

论文把 MWM 的能力分成两类：

```text
1. knowledge organisation
2. active inquiry
```

### Knowledge organisation

也就是从观测中组织出可复用解释。

包括三个能力：

```text
variable discovery:
  发现哪些 latent variables 才是解释系统所需的变量。

mechanism discovery:
  发现变量之间如何演化和相互作用的可复用机制。

structure discovery:
  发现这些变量和机制如何绑定、组合，形成对某个具体系统的解释。
```

### Active inquiry

光组织知识还不够，科学发现还需要主动提出假设、设计实验。

包括：

```text
explanatory hypothesis generation:
  提出新的变量、机制、结构组合，解释未见过的现象。

experimental design:
  决定下一步应该观测什么、干预什么，才能区分竞争性解释。
```

## 设计原则

论文提出 4 个 design principles。

### 1. Joint predictive and explanatory modelling

预测和解释不能分开。

不是：

```text
先训练一个预测模型；
再事后解释它。
```

而是：

```text
训练时就让解释结构和预测能力一起形成。
```

作者特别强调：

```text
post-hoc interpretability 不够。
```

因为解释的是模型内部怎么预测，不一定等于世界真实机制。

### 2. Mechanisms as fundamental representation units

MWM 的基本表示单元不是普通 latent vector，而是：

```text
mechanism
```

也就是：

```text
带变量签名的可复用 transformation。
```

### 3. Mechanism reuse

科学理解的力量来自：

```text
少量机制解释大量现象。
```

所以 MWM 应该优先复用已有机制，而不是每个环境重学一个预测器。

这和 causal learning 里的思想对应：

```text
Independent Causal Mechanisms
Sparse Mechanism Shift
Invariance
```

### 4. Hierarchical organisation

复杂系统不是一层结构，而是层级化的：

```text
简单机制 -> 组合机制 -> 高层解释结构
```

所以 MWM 应该支持：

```text
机制的层级组织和组合。
```

## 两个核心归纳压力

论文认为 MWM 需要两个 inductive pressures。

### Parsimony

中文可以理解成：

```text
简约性 / 奥卡姆压力
```

好的解释应该：

```text
用尽量少的变量、机制和结构解释尽量多的现象。
```

这里 parsimony 不是简单的参数少，而是：

```text
解释知识组织要简洁。
```

### Compositionality

中文就是：

```text
组合性
```

科学知识不是每次推倒重来，而是：

```text
把已有机制重新组合，解释新系统。
```

所以 MWM 应该鼓励：

```text
机制库复用
机制组合
局部更新
跨环境泛化
```

## MWM 的形式化 anatomy

论文把 MWM 定义成四元组：

```text
W = (T, Z, M, S)
```

### T：semantic variable types

```text
T = {tau_1, tau_2, ...}
```

它不是人工预设标签，而是模型学出来的变量功能类型。

可以类比：

```text
position
velocity
mass
concentration
temperature
```

但论文强调，这些类型在模型里是 emergent functional roles。

### Z：typed latent variables

某个系统由一组 latent variables 表示：

```text
Z = {z_1, ..., z_n}
```

并有一个 typing function：

```text
T: Z -> T
```

表示每个变量属于哪个 semantic type。

### M：mechanism library

机制定义为：

```text
m = (Sigma_m, f_m)
```

其中：

```text
Sigma_m:
  typed variable signature
  指定这个机制需要什么类型的输入变量、输出变量

f_m:
  deterministic 或 probabilistic mapping
  也就是实际变换函数
```

关键点：

```text
机制的 signature 指向变量类型，而不是某个具体变量。
```

所以同一个机制可以在不同系统里复用。

### S：binding structure

structure 表示如何把机制的角色绑定到当前系统里的具体变量。

绑定写成：

```text
b: Sigma_m -> Z
```

并满足类型约束：

```text
T(b(tau)) = tau
```

也就是说：

```text
机制需要 position 类型变量，就必须绑定到当前系统里 position 类型的 latent variable。
```

不同环境不是靠换机制解释，而是靠：

```text
同一机制库 + 不同 binding structures
```

## 学习目标

论文给了一个概念化目标：

```text
W* = argmin_{T,Z,M,{S_e}}
[
  Omega_p(T, M, Z)
  + Omega_c({S_e})
  + sum_{e in E} L_pred(T, Z, M, S_e)
]
```

其中：

```text
L_pred:
  预测损失，保证模型还能预测观测。

Omega_p:
  parsimony penalty，约束变量、机制、类型不要太复杂。

Omega_c:
  compositionality penalty / pressure，鼓励不同环境复用和组合同一机制库。

S_e:
  第 e 个环境自己的 binding structure。
```

这就是 MWM 的核心：

```text
预测能力 + 简约变量/机制 + 跨环境机制复用
```

## 和现有研究方向的关系

论文把几个方向看作 MWM 的局部成分。

### Mechanistic interpretability

贡献：

```text
能从模型内部找可解释 feature / circuit。
```

问题：

```text
解释的是模型，不一定是世界。
```

作者认为：

```text
explanatory organisation 不能只靠 post-hoc extraction；
必须在训练过程中被鼓励出来。
```

### Causal representation learning

贡献：

```text
强调 causal variables、invariance、independent mechanisms。
```

问题：

```text
理论强，但常局限在合成 / 受限设定；
难扩到开放科学系统。
```

### Equation discovery

贡献：

```text
用简洁方程解释观测；
parsimony 很明确。
```

问题：

```text
常假设已经有合适变量；
更擅长发现单个 governing law；
不擅长开放环境里的机制库组合。
```

### MoE / modular architectures

贡献：

```text
证明模块化组织能提升扩展性和效率。
```

问题：

```text
模块通常按预测能力分工，
不一定对应世界里的可复用机制。
```

## 和 AI Scientist / Autonomous Discovery 的区别

这点很重要。

很多 AI Scientist 系统做：

```text
文献阅读
实验规划
代码执行
报告生成
```

但论文认为它们大多还是建立在：

```text
predictive foundation models
```

上面。

它们自动化的是：

```text
scientific workflow
```

而 MWM 关注的是：

```text
explanatory knowledge 本身如何表示和学习。
```

所以两者层级不同：

```text
AI Scientist:
  自动化科学家做的流程。

MWM:
  改变模型内部知识组织方式，让模型能形成机制化解释。
```

## 现有方法比较

论文表 1 比较了一些代表方法：

```text
VCD
COMET
SPARTAN
DIDS
C-JEPA
RIM
C-SWM
DreamCoder
NEO
PoE-World
```

比较维度：

```text
variable discovery
mechanism learning
structure learning
compositionality
parsimony
scalable architecture
```

结论是：

```text
现有方法各自覆盖一部分；
没有一个完整覆盖 MWM 所需全部能力。
```

MWM 的定位就是：

```text
把这些方向统一到机制中心的世界模型框架里。
```

## 关键挑战

论文最后指出几个未解决问题。

### 1. 联合学习很难

真正的 MWM 需要同时学：

```text
variables
mechanisms
binding structures
```

还要平衡：

```text
prediction
parsimony
compositionality
```

不是把现有模块简单拼起来就行。

### 2. 可识别性问题

变量和机制不是唯一的。

同样观测可能有很多解释：

```text
不同 latent variables
不同机制分解
不同绑定结构
```

什么时候能恢复真实机制，理论仍然不完整。

### 3. Active inquiry 还很弱

MWM 不应该只是被动学数据，还要知道：

```text
什么时候加新机制
什么时候合并机制
什么时候 refine 机制
下一步应该做什么实验
哪种干预最能区分两个解释
```

这些机制管理能力目前还很早期。

### 4. 如何融合已有科学知识

很多领域已经有大量机制知识。

MWM 不该完全从零开始，而应该能：

```text
初始化机制库
吸收已有理论
在新证据下修改或替换机制
```

### 5. 缺少合适评测

普通 prediction accuracy 不够。

MWM 需要评测：

```text
变量是否恢复正确
机制是否可解释
机制能否复用
结构绑定是否正确
干预预测是否可靠
能否在新环境里组合已有机制
能否持续学习而不遗忘
```

## 这篇论文的本质

这篇更像是在提出一个研究纲领：

```text
AI for Science 不能只继续堆预测模型；
如果目标是 autonomous discovery，
模型内部必须以机制为中心组织知识。
```

它不是给你一个马上可跑的算法，而是提出：

```text
未来科学发现模型应该具备什么结构、能力和训练压力。
```

## 一句话结论

Mechanistic World Models 的核心是：

```text
把世界模型从 observation-level forecasting，
推进到 mechanism-level explanation；
让模型学习变量、机制和绑定结构，
并通过简约性和组合性形成可复用的科学解释。
```

