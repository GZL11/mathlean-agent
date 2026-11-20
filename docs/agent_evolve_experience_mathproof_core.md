# Agent + Evolve + Experience + MathProof：核心逻辑整理

这份文档只回答一个问题：

- 如果我们要做一套 `agent + evolve + experience + mathproof` 框架，它的核心逻辑到底是什么？真正可能成立的创新点又是什么？

## 0. 先验知识：几个核心名词先解释清楚

为了让不熟悉 formal math / agent / evolution 方向的读者也能直接读下去，先把几个核心英文名词和它们之间的关系说明白。

### 0.1 `mathproof`

这里的 `mathproof` 指的是：

- 用形式化系统证明数学命题
- 不是只写自然语言证明
- 而是要把证明写成 Lean 这类 proof assistant 可以检查的形式

最关键的特点是：

- 证明是否正确，不由人主观判断
- 而由 Lean 内核 / verifier 客观判断

### 0.2 `verifier`

`verifier` 指的是证明校验器。  
在我们的场景里，它主要就是：

- Lean checker
- Lean 内核
- 以及围绕 Lean 的 diagnostics / goal state / build 检查工具

它的作用是回答：

- 这个 theorem 到底有没有被真正证明
- 当前 Lean 代码有没有错误
- 是否还残留 `sorry`

所以 verifier 是系统里最硬的“裁判”。

### 0.3 `agent`

`agent` 指的是一个能多步思考、调用工具、持续修改工作对象的智能执行体。

在这个方向里，agent 不是简单的一次模型补全，而是会做：

- 拆题
- 搜 theorem
- 看 proof state
- 写 Lean 草稿
- 根据错误继续修复

所以 agent 更像一个“局部工作流执行者”。

### 0.4 `evolve`

`evolve` 指的是演化式搜索 / 演化式优化。

它的典型操作包括：

- `selection`：从候选中选哪些继续扩展
- `mutation`：对候选做局部改写
- `recombination`：把不同候选的优点组合起来
- `archive update`：把有价值的候选存起来

所以 evolve 不是“再试一轮”，而是：

- 用一套明确的搜索控制逻辑去管理候选群体

### 0.5 `experience`

`experience` 指的是系统在证明过程中积累下来的可复用经验。

它不只是日志，而可以是：

- 成功的 lemma
- 有用的 proof sketch
- 失败修复模板
- theorem search 模式
- decomposition 模板

所以 experience 更像：

- 一个显式的外部记忆库

### 0.6 `proof object`

`proof object` 指的是证明过程中的“候选对象”。

它不一定非得是完整 Lean 证明，也可以是：

- proof sketch
- subgoal tree
- lemma set
- partial Lean draft
- repair trace

这个词非常关键，因为它决定了系统到底在“演化什么”。

### 0.7 数学证明领域里更关键的几个名词

下面这些词比 `agent / evolve / experience` 更偏数学证明本身，也是后面方法设计里最容易混淆的部分。

#### `theorem`

`theorem` 就是最终要证明的命题。

在 Lean 里，它通常会长成：

```lean
theorem real_abs_triangle_ineq (a b : ℝ) : |a + b| ≤ |a| + |b| := by
  ...
```

这里的 theorem 是整个任务的最终目标。

#### `lemma`

`lemma` 是辅助引理，也就是为了证明最终 theorem，中途额外提出并证明的中间命题。

例如为了证明最终结论，先证明一个更小的性质：

- 某个绝对值变形
- 某个不等式子结论
- 某个代数恒等式

lemma 的作用是：

- 把大证明拆成更小的可验证单元
- 降低单次 proof search 的难度
- 为后续步骤提供可复用结论

所以在复杂 formal proof 里，lemma 往往不是附属品，而是一级对象。

#### `proof sketch`

`proof sketch` 可以理解成“证明草图”或“证明大纲”。

它通常不是完整 Lean 代码，而是：

- 用自然语言描述证明主线
- 指出大致分几步
- 每步打算使用什么数学思想或什么定理

它比正式证明更抽象，但比随意讨论更结构化。

proof sketch 的价值在于：

- 帮系统先确定大方向
- 再把大方向逐步落成 formal proof

#### `subgoal`

`subgoal` 是子目标。

在证明过程中，原始 theorem 往往会被拆成多个更小的目标。  
Lean 在 tactic 执行时，经常会把当前目标转化成一个或多个 subgoal。

例如你本来要证明一个复杂命题，做了一次分解后，可能变成：

- 先证条件 A
- 再证条件 B

这两个新的目标就是 subgoal。

所以 `subgoal decomposition` 的本质就是：

- 把一个难目标分裂成多个更容易处理的小目标

#### `proof state`

`proof state` 指的是“当前证明到这一步时，系统手里到底掌握了什么、还缺什么”。

通常它包含：

- 当前还要证明的目标是什么
- 当前上下文里有哪些假设
- 当前有哪些已知变量

proof state 是 Lean proving 里最重要的中间信息之一，因为：

- 它告诉你当前到底卡在哪里
- 也决定了你接下来该搜什么 theorem、用什么 tactic

#### `premise`

`premise` 在这个领域里通常指“当前证明状态下可能有用的前提或现成定理”。

它有两种常见来源：

1. 当前上下文里已经有的假设
2. Mathlib 里现成的 theorem / lemma / definition

所以 `premise search` 的意思不是泛泛地搜资料，而是：

- 在当前 proof state 下，找最可能直接可用的前提

#### `tactic`

`tactic` 是 Lean 中推进证明的一种操作命令。

例如：

- 改写目标
- 分解目标
- 应用某个定理
- 自动化求解某类简单目标

你可以把 tactic 理解成：

- 形式化证明里的“局部动作”

它不是最终答案，而是证明过程中一步一步执行的动作。

#### `formalization`

`formalization` 指的是把自然语言数学题翻成 Lean 可检查的正式表述。

例如自然语言：

> 对任意实数 a, b，证明 |a+b| ≤ |a| + |b|

formalization 后会变成 Lean theorem statement。

这一步非常重要，因为：

- Lean 只能证明你写进系统里的命题
- 它不能自动保证你写下的命题和原题完全等价

所以 formalization 和 proving 是两个不同问题。

#### `blueprint`

`blueprint` 是比 proof sketch 更工程化的一种证明规划对象。

它通常不仅包含“证明思路”，还包含：

- 证明节点列表
- lemma 间的依赖关系
- 哪些节点已完成
- 哪些节点还待 formalize / 待证明
- 每个节点尝试过多少次

如果说：

- proof sketch 更像“证明大纲”
- blueprint 更像“证明工程计划表”

#### `dependency topology`

`dependency topology` 指的是 theorem / lemma 之间的依赖结构。

例如：

- 最终 theorem 依赖 lemma A 和 lemma B
- lemma B 又依赖 lemma C

那整个证明就不是一条线，而是一张依赖图。

这张图的作用是：

- 决定先做哪些节点
- 决定哪些节点已经具备可证明条件
- 帮系统把复杂证明按依赖顺序逐步推进

#### `repair`

`repair` 指的是“根据 verifier 暴露出的失败信息，对当前证明候选做修复”。

修复可能包括：

- 改 statement
- 换 theorem
- 补中间 lemma
- 调整 tactic 顺序
- 重写 proof sketch

所以 repair 不是完全重新来一次，而是：

- 以当前失败版本为基础的局部纠错

### 0.8 这些数学名词之间的关系

可以把它们之间的关系理解成下面这张图：

```text
Natural language problem
        |
        v
Formalization
        |
        v
Theorem
        |
        v
Proof sketch / Blueprint
        |
        v
Subgoal decomposition
        |
        v
Lemma generation
        |
        v
Proof state
        |
        +--> premise search
        +--> theorem search
        +--> tactic application
        +--> repair
        |
        v
Verified proof
```

如果用一句话概括这些词之间的层次关系：

- `theorem` 是最终目标
- `sketch / blueprint` 是高层计划
- `subgoal / lemma` 是中间拆解单元
- `proof state / premise / tactic` 是底层证明执行单元
- `repair` 是失败后的局部修复机制

### 0.9 `pass@k`

`pass@k` 是 theorem proving / code generation 里很常见的评测方式。

它的含义是：

- 对同一道题采样 `k` 次
- 只要有一次通过 verifier，就算成功

这个指标常见，但它的问题是：

- 候选之间通常彼此独立
- 没有经验共享
- 没有显式演化结构

### 0.10 `iterative repair`

`iterative repair` 指的是：

- 先生成一个候选
- 看到错误后继续修
- 再看错误，再修

它是单轨迹、多轮修复。  
它比 one-shot 强，但通常仍然没有：

- population
- recombination
- experience memory

### 0.11 这些概念之间的关系

可以把它们的关系理解成下面这张图：

```text
Math Problem
    |
    v
Agent
  - 负责局部推理、搜索、构造、修复
    |
    v
Proof Objects
  - sketch / lemma / partial proof / final proof
    |
    v
Verifier
  - 判断哪些对象真的有效、哪些只是中间状态
    |
    v
Evolve
  - 根据 verifier 反馈做 selection / mutation / recombination / archive
    |
    v
Experience
  - 把成功和失败转成可复用记忆
    |
    +--> 反过来指导下一轮 Agent
```

如果用一句话概括它们的分工：

- `agent` 负责做事
- `verifier` 负责裁判
- `evolve` 负责全局搜索
- `experience` 负责记忆和复用

理解了这四者的关系，后面的正文就比较容易读了。

## 1. 一句话结论

我对这套框架的判断是：

> 这套方法的核心，不是“让模型多试几次证明”，而是把数学证明重新表述成一种“带显式记忆的、可验证的、结构化对象演化搜索”。

如果写成更像论文里的方法表述，可以是：

> We formulate formal mathematical proving as verifier-grounded evolutionary search over structured proof objects, with agentic decomposition for local reasoning and explicit experience memory for cross-attempt and cross-problem reuse.

## 2. 四个词分别在系统里扮演什么角色

### 2.1 `mathproof`

这是任务本体。

它和普通代码优化的关键差别是：

- 有强 verifier，Lean 内核能做客观裁判
- 证明是长链条结构，不是单步答案
- 中间产物有价值，例如 lemma、proof sketch、subgoal decomposition
- reward 看起来稀疏，但过程信号其实很多

因此它天然适合：

- 结构化搜索
- 多阶段构造
- 过程级反馈优化

而不太适合只做一次性 end-to-end 文本生成。

### 2.2 `agent`

agent 不是“更大的模型”，而是局部工作流执行器。

比较合理的职责边界是：

- 拆题
- 生成 proof sketch
- 生成 / 修复 Lean draft
- 搜 theorem / premise
- 总结失败原因
- 生成下一步动作建议

也就是说：

- agent 负责局部 reasoning 和 action execution
- 它不应该单独承担整个系统的全局优化

### 2.3 `evolve`

evolve 是全局搜索控制器。

它不等于“再来一轮”，而是负责：

- population maintenance
- selection
- mutation
- recombination
- archive update
- diversity control
- budgeted search

所以：

- agent 是局部执行器
- evolve 是全局优化器

### 2.4 `experience`

这是最值得做出差异化的部分。

experience 不能只理解成日志或历史记录，而应该是：

- 可被检索的
- 可被复用的
- 可被组合的
- 显式存在的经验对象

例如：

- 成功的 lemma
- 失败的 proof trace
- 某类题目的 decomposition 模板
- 常见 Lean error -> repair pattern
- theorem search pattern
- 高价值 blueprint 片段

所以 experience 不是被动记录，而是主动参与下一轮搜索。

## 3. 这套框架的核心逻辑

我认为它的核心逻辑可以写成下面这条链：

```text
数学证明不是一次性生成最终 Lean proof
而是：
在 verifier 约束下，
围绕一组结构化 proof objects，
用 agent 做局部推理与构造，
用 evolve 做全局选择与变异，
用 experience 做跨轮次和跨题目的知识复用，
最终在固定预算内提升成功率和搜索效率。
```

这里最关键的变化是：

- 证明不再被看成一条单独的文本轨迹
- 而被看成一个带记忆、带结构、带选择压力的搜索生态

## 4. 真正可能成立的核心创新点

### 4.1 创新点一：把搜索对象从 final proof 扩展成 structured proof objects

不是只搜索最终 Lean 文件，而是搜索：

- proof sketch
- subgoal tree
- auxiliary lemma set
- partial Lean draft
- repair trace

这很关键，因为一旦只搜索 final proof，方法很容易退化成：

- pass@k
- independent sampling
- iterative repair

只有把中间对象也显式纳入搜索空间，你们才真正和现有 prover / agent 方法拉开差距。

### 4.2 创新点二：把 experience 做成显式外部记忆，而不是隐式历史

experience 至少可以分成三层：

1. `within-problem experience`
   - 同一道题内部，哪些尝试有用

2. `cross-attempt experience`
   - 同一道题不同轨迹之间，哪些 lemma / repair / decomposition 可复用

3. `cross-problem experience`
   - 不同题之间，共享什么模式、模板、失败修复策略

如果这一层做起来，你们的系统就不再只是“单题搜索更强”，而是：

- 系统会显式积累 proving knowledge

### 4.3 创新点三：在固定预算下，evolutionary controller 比简单多试更有效

要证明的不是：

- 多试几次当然更好

而是：

- 在相同 token budget / verifier budget / wall-clock budget 下
- 有 population + experience 的 evolutionary controller
- 比 independent sampling 和 iterative repair 更高效

这个才是方法学创新，而不是算力堆叠。

## 5. 更好的问题表述方式

我不建议把问题表述成：

- “我们提出一个新的 theorem prover”
- “我们做了一个更强的 math agent”

这两个说法都太泛，而且会直接撞上大量已有工作。

我更建议用下面这类表述：

### 版本 A

`Formal theorem proving as evolutionary search over proof objects`

### 版本 B

`Agentic formal reasoning with explicit experience memory`

### 版本 C

`Budgeted verifier-grounded evolution for long-horizon formal proofs`

如果要把四个词都带上，可以写成：

> We propose an agentic, experience-augmented evolutionary framework for formal mathematical proving under verifier feedback.

## 6. 理想的系统闭环

比较合理的系统闭环应该是：

```text
Problem
  -> agent builds sketch / subgoals / lemmas / Lean drafts
  -> verifier evaluates progress
  -> evolve selects and mutates proof objects
  -> experience stores useful artifacts and failure patterns
  -> retrieval from experience conditions the next agent actions
  -> repeat under fixed budget
```

这里最关键的是：

- verifier 不只是最后裁判
- experience 不只是日志
- evolve 不只是 retry
- agent 不只是写代码

四者必须形成闭环。

## 7. 最容易被质疑的点

### 7.1 质疑一：这不就是多轮 self-repair 吗？

你们需要明确回答：

- 不是单轨迹
- 有 population
- 有显式 selection / recombination / archive
- 有跨轨迹经验共享

### 7.2 质疑二：这不就是 pass@k 换个名字吗？

你们需要明确回答：

- independent samples 之间没有信息传递
- 我们的候选之间有经验共享、重组、选择压力
- 我们优化的是搜索效率，不只是样本数

### 7.3 质疑三：experience 不就是 RAG 吗？

你们需要明确回答：

- 不只是检索现成 theorem
- 还包括 system-generated lemma、repair trace、decomposition pattern
- 是 proof-process-level memory，不只是 corpus retrieval

### 7.4 质疑四：Lean verifier reward 太稀疏，evolve 未必能工作

你们需要明确回答：

- 不只使用 binary reward
- 要使用过程型 fitness，例如：
  - goal reduction
  - sorry-free progress
  - verified subgoals
  - reusable lemma utility
  - compile success

## 8. 最值得先想清楚的问题

如果只能选一个最核心的问题，我认为是：

> 你们的 population 里到底存什么？

这是整套方法的灵魂。

如果存的是：

- 完整 Lean 文件

那方法很容易退化成：

- 多个 proof drafts 并行搜索

这还不够新。

如果存的是：

- blueprint
- subgoal tree
- lemma candidates
- partial proofs
- repair traces

那方法立刻就不一样了。

所以方法设计的第一性问题是：

- `proof object design`

## 9. 我最认可的核心创新表述

如果现在就要给一个最像论文创新点的说法，我会定成：

### 核心创新

`显式经验驱动的 proof-object evolution`

展开来说就是：

- agent 负责构造和修复 proof objects
- verifier 提供过程级反馈
- evolve 在 proof objects 上做预算受限搜索
- experience 把成功与失败都转成可复用对象
- 系统在题内和题间都能积累 proving knowledge

这个表述比单说 `agent + evolve` 要强很多。

## 10. 后续最值得展开的三个问题

如果继续深入，我建议后续讨论按这三个问题推进：

1. `proof object` 到底定义成什么
   - final proof？
   - lemma？
   - subgoal tree？
   - blueprint？
   - 它们的组合？

2. `experience` 到底存什么
   - theorem retrieval history？
   - 失败修复模板？
   - 可复用 lemma？
   - decomposition template？
   - 哪些是一等对象？

3. `evolve` 的操作符是什么
   - mutation 怎么定义？
   - recombination 怎么定义？
   - selection 的 fitness 怎么定义？
   - diversity 怎么定义？

这三个问题定清楚，整套方法基本就成型了。

## 11. 一句话收束

如果最后只留一句最核心的话，我建议是：

> 这套 `agent + evolve + experience + mathproof` 框架的真正核心，不是让模型反复试 Lean，而是让系统围绕结构化 proof objects，在 verifier 约束下进行带显式经验记忆的预算受限演化搜索。
