# Agent + Evolve + Experience + MathProof：完整设计方案

本文档给出一版完整的方法设计草案。目标不是讨论已有工作，而是把我们自己的框架明确成一个可以实现、可以写论文、也可以做实验验证的系统方案。

## 修订说明

下面这份文档最初采用的是较早的一版口径，即：

- `一个 plan = 一个局部搜索岛屿`
- 内层做 `plan-local evolution`

结合后续讨论，我们认为更稳妥、也更适合论文表达的版本应该是：

- 外层维护 `plan population`，做 `plan-level evolve`
- 内层在固定 `plan` 下，围绕 `subgoals` 做多轮局部求解与修复
- 内层不再改 `plan` 的拓扑结构，而是逐步把 subgoal 变成 `verified`
- 全局长期记忆不只包含经验摘要，还要包含可复用的 `verified lemma library`

因此，本文后半部分保留早期表述作为背景，但应以本节和文末新增的“修订版实现方案”为准。

本文档采用的核心立场是：

- `proof object` 不是最终 Lean 文件，而是证明过程中的结构化对象
- `plan` 不是一句自然语言策略，而是一个可执行的证明计划实例
- `一个 plan = 一个局部搜索岛屿`
- 内层做 `plan-local evolution`
- 外层做 `plan-level generation / selection / experience reuse`

如果只用一句话概括整套方案：

> 我们把形式化数学证明看作一种在 verifier 约束下、围绕结构化 proof objects 进行的层级式演化搜索：外层负责生成、筛选和进化证明计划，内层在固定 plan 下进行有预算约束的 subgoal realization，并持续把成功/失败经验与已验证引理写入全局记忆。

## 1. 设计目标

我们希望解决的不是“怎么让模型再多试几次 Lean”，而是下面这几个问题：

1. 如何把一个复杂数学证明拆成可被局部验证的中间单元
2. 如何让不同证明路线以 plan 的形式并行竞争
3. 如何在失败时积累经验，而不是只丢弃轨迹
4. 如何在固定预算下，比 independent sampling / iterative repair 更高效

因此，这个框架的目标是：

- 支持长时程 formal proof
- 支持结构化 decomposition
- 支持中间 lemma 复用
- 支持 verifier-grounded PRM reward
- 支持题内和题间经验积累

## 2. 问题表述

给定一个形式化数学证明任务：

- 输入：一个 theorem，或一个可 formalize 成 theorem 的自然语言问题
- 输出：一个 verifier 接受的、无 `sorry` 的 Lean 证明

我们不把求解过程看成一次性生成，而看成一个分层搜索问题：

### 外层问题

生成和选择有潜力的 `plan`

### 内层问题

在固定 `plan` 的拓扑结构下，最大化该 plan 的证明推进度

也就是说，整个系统有两种搜索对象：

1. `plan`
2. `proof object`（plan 内部的中间对象）

## 3. 核心对象定义

## 3.1 PlanObject

我建议把种群中的主个体定义成 `PlanObject`。

一个 `PlanObject` 表示：

- 针对同一个 theorem 的一条完整证明路线
- 内部有自己的 subgoal DAG
- 有自己的 lemma 集合
- 有自己的 Lean 草稿状态
- 有自己的验证进度

一个最小的 `PlanObject` 可以包含下面这些字段：

```text
PlanObject
- plan_id
- theorem_id
- theorem_statement
- plan_summary
- strategy_type
- subgoal_graph
- lemma_pool
- lean_artifacts
- frontier_nodes
- verifier_feedback
- combined_score
- parent_plan_ids
- experience_refs
- status
```

### 字段解释

- `plan_id`
  - 计划唯一标识

- `theorem_id`
  - 当前 plan 对应哪个 theorem

- `theorem_statement`
  - 目标 theorem 的形式化表述

- `plan_summary`
  - 对该计划的自然语言摘要

- `strategy_type`
  - 该计划属于哪种大路线，例如 direct theorem application、induction、contradiction、decomposition-first 等

- `subgoal_graph`
  - 该计划内部的 subgoal DAG

- `lemma_pool`
  - 当前计划显式维护的辅助 lemma 集合

- `lean_artifacts`
  - 当前 Lean draft、step 文件、证明骨架等

- `frontier_nodes`
  - 当前 plan 里最值得继续推进的边界节点

- `verifier_feedback`
  - 当前计划聚合后的验证信息

- `combined_score`
  - 该计划当前的总评分

- `parent_plan_ids`
  - 该计划由哪些旧计划变异 / 组合而来

- `experience_refs`
  - 生成本计划时使用了哪些经验对象

- `status`
  - running / solved / stalled / exhausted / discarded

## 3.2 SubgoalNode

`PlanObject` 的核心是 `subgoal_graph`，所以必须定义节点对象。

```text
SubgoalNode
- node_id
- node_type
- statement_nl
- statement_formal
- dependencies
- dependents
- local_lean_draft
- local_status
- verifier_state
- node_score
- importance_weight
```

### 字段解释

- `node_type`
  - theorem / lemma / decomposition step / repair task

- `statement_nl`
  - 自然语言版本的节点描述

- `statement_formal`
  - Lean 中的正式目标

- `dependencies`
  - 该节点依赖哪些前置节点

- `dependents`
  - 哪些后续节点依赖它

- `local_lean_draft`
  - 针对这个节点的局部 Lean 草稿

- `local_status`
  - unattempted / drafted / compilable / partially_verified / verified / failed

- `verifier_state`
  - diagnostics、goal state、sorry count 等

- `node_score`
  - 该节点的局部得分

- `importance_weight`
  - 该节点对整体计划的拓扑重要性权重

## 3.3 LemmaObject

我建议 lemma 既可以是 `SubgoalNode` 的一种类型，也可以单独抽成经验对象。

最小字段：

```text
LemmaObject
- lemma_id
- statement_formal
- statement_nl
- proof_status
- reusable_score
- source_plan_id
- used_by
```

它的关键作用是：

- 作为计划内部的中间证明对象
- 作为跨计划、跨题目的经验载体

## 3.4 ExperienceObject

experience 不只是“历史记录”，要有显式结构。

我建议至少支持 4 类经验：

```text
ExperienceObject
- exp_id
- exp_type
- content
- source_problem
- source_plan
- verifier_outcome
- reuse_count
- quality_score
```

其中 `exp_type` 可以是：

- reusable_lemma
- decomposition_template
- repair_pattern
- theorem_search_pattern

## 4. 设计原则：一个 plan = 一个局部搜索岛屿

这是整个方案最核心的设计原则。

### 4.1 为什么这样定义

这样做是为了明确分开两类问题：

1. 这个 plan 本身好不好
2. 在这个 plan 下，局部 Lean 搜索做得好不好

如果不分开，系统会很难判断：

- 当前失败是 plan 的错
- 还是只是局部 repair 还不够

### 4.2 这个定义带来的好处

- plan-level 与 proof-repair-level 分层清楚
- PRM reward 可以自然定义在 plan DAG 上
- 每个 plan 可以在固定预算内被“充分探索”
- plan 结束后更容易沉淀经验

## 5. 整体系统流程

```text
Problem / Target Theorem
        |
        v
Retrieve relevant experience
        |
        v
Generate initial plan population
        |
        v
For each PlanObject:
    run plan-local island search under fixed budget
        |
        +--> agent expands / repairs plan internals
        +--> verifier evaluates progress
        +--> local evolution updates proof objects
        |
        v
    obtain plan score + experience artifacts
        |
        v
Outer loop:
    - select promising plans
    - discard exhausted plans
    - summarize experience
    - generate new plans by mutation / recombination
        |
        v
Repeat until budget exhausted or theorem solved
```

## 6. 内层：Plan-Local Island Search

内层的目标是：

- 在固定 plan 的拓扑结构下
- 尽可能推进这个 plan 的验证进度

### 6.1 固定什么，不固定什么

一个 plan 一旦进入局部 island，应该固定：

- target theorem
- 主 decomposition 路线
- subgoal DAG 的主体结构
- 主要 strategy type

在 island 内允许变化的包括：

- lemma 的具体表述
- 子目标推进顺序
- Lean tactic 实现
- theorem / premise 选择
- 局部 repair

如果这些局部变化已经开始推翻 DAG 主体结构，就不应继续视为同一个 plan，而应生成新 plan。

### 6.2 内层的基本循环

```text
Select frontier node
    |
    v
Agent proposes local action
  - prove node
  - refine statement
  - add lemma
  - search theorem/premise
  - repair Lean draft
    |
    v
Run verifier
    |
    +--> update node status
    +--> update local Lean state
    +--> update plan frontier
    |
    v
Update plan-local score
```

### 6.3 局部动作类型

局部动作我建议先限制在下面几类：

1. `prove_node`
2. `refine_subgoal`
3. `introduce_lemma`
4. `replace_theorem`
5. `repair_lean_draft`
6. `reorder_subgoals`

这样 action space 不会太散。

## 7. 外层：Plan-Level Evolution

外层的目标是：

- 维护多个 plan 之间的竞争
- 让系统不断生成更好的证明路线

### 7.1 外层的基本循环

```text
Finished / active plan pool
        |
        v
Select top / diverse plans
        |
        +--> mutate one plan into a new plan
        |
        +--> recombine two plans into a new plan
        |
        +--> generate fresh plan using experience retrieval
        |
        v
New plan enters next local island
```

### 7.2 外层 mutation 可以做什么

plan-level mutation 不应只是文字改写，而应作用于计划结构：

- 更换 decomposition 方式
- 替换关键 lemma 路线
- 合并 / 拆分某些 subgoal
- 改变 strategy type

### 7.3 外层 recombination 可以做什么

两个 plan 之间的交叉重点不在完整 Lean 代码，而在：

- 借用某一段 subgoal DAG
- 借用某一组 lemma
- 借用某个成功 repair 模式

所以 recombination 的对象更多是结构，而不是表面文本。

## 8. Reward 设计：从 ORM 到 PRM

这是整个系统最关键的一部分。

我建议不要只用最终 theorem 是否 solved 的 ORM 奖励，而是采用分层 PRM。

## 8.1 最终奖励

```text
R_final = 1.0 if theorem solved else 0.0
```

## 8.2 过程奖励

对一个 plan，令其 subgoal DAG 的节点集合为 `V`。

每个节点 `v` 的状态分数可以定义为：

```text
status_score(v):
  unattempted = 0.0
  drafted = 0.1
  compilable = 0.3
  partially_verified = 0.6
  verified = 1.0
  failed = 0.0
```

那么 plan 的归一化拓扑进度可以定义为：

```text
R_progress =
  sum_v w(v) * status_score(v)
  --------------------------------
         sum_v w(v)
```

其中 `w(v)` 是节点权重。

## 8.3 节点权重设计

我建议节点权重不要平均分，而要和拓扑位置有关。

例如：

```text
w(v) =
  α * dependency_importance(v)
  + β * frontier_importance(v)
```

其中：

- `dependency_importance(v)`
  - 下游多少节点依赖它

- `frontier_importance(v)`
  - 当前是否位于关键 frontier

这可以避免系统只喜欢刷“容易的小节点”。

## 8.4 质量奖励

除了 DAG 进度，plan 还应该有质量项：

```text
R_quality =
  a1 * compile_ratio
  + a2 * sorry_free_ratio
  + a3 * statement_consistency
```

## 8.5 经验复用奖励

如果某个 plan 产出了高复用价值的 lemma 或 repair 模式，应给予额外奖励：

```text
R_reuse = reusable_lemma_bonus + reusable_pattern_bonus
```

## 8.6 效率奖励

为了避免方法退化成“多算力更强”，建议显式加效率项：

```text
R_efficiency =
  progress_per_token
  + progress_per_verifier_call
```

## 8.7 最终 combined_score

一版实用的总分公式可以写成：

```text
combined_score =
  λ1 * R_final
  + λ2 * R_progress
  + λ3 * R_quality
  + λ4 * R_efficiency
  + λ5 * R_reuse
```

### MVP 版

第一版可以更简单：

```text
combined_score =
  0.5 * R_final
  + 0.4 * R_progress
  + 0.1 * R_reuse
```

## 9. 不同 plan 长度不同，如何公平比较

这是必须提前回答的问题。

如果：

- Plan A 有 3 个节点
- Plan B 有 10 个节点

直接按“完成节点数”打分会偏向拆得更细的 plan。

所以必须使用：

- 归一化拓扑进度

也就是：

```text
normalized_progress =
  weighted_verified_progress / total_weight
```

这样不同粒度的 plan 才能公平比较。

## 10. Population 如何维护

## 10.1 主种群：Plan Population

主种群里的个体是 `PlanObject`。

每个个体：

- 对应同一个 theorem 的一条证明路线
- 有自己的 subgoal DAG
- 有自己的 Lean artifacts

我建议主种群的保留策略是：

- 一部分保留高分 plan
- 一部分保留结构差异较大的 plan
- 一部分保留高潜力但未完成的 plan

不要只做纯 top-k。

## 10.2 共享经验池：Experience Memory

共享经验池不等于主种群。

它存的是跨 plan 可复用对象，例如：

- reusable lemma
- decomposition template
- theorem search pattern
- repair pattern

所以：

- 主种群存“路线”
- 经验池存“可复用片段”

## 11. Experience 机制

experience 我建议按三层组织。

### 11.1 题内经验

- 同一道题内部，不同 plan / 不同尝试之间共享经验

### 11.2 跨尝试经验

- 同一 theorem 的不同轨迹之间共享经验

### 11.3 跨题经验

- 不同 theorem 之间共享高价值模式

## 11.4 经验写回机制

当一个 plan 结束时，系统应自动总结：

- 哪些 lemma 有用
- 哪些 subgoal 经常卡住
- 哪些 theorem search 查询有效
- 哪些 repair 模式成功率高

这些总结要变成结构化 `ExperienceObject`，而不是只留自然语言日志。

## 12. 停止条件

### 12.1 plan-local island 停止条件

一个 plan 可以在以下条件下结束：

1. 达到 `k` 轮上限
2. 达到 token budget 上限
3. 达到 verifier budget 上限
4. 连续若干轮 PRM 无提升
5. frontier 长时间不变化

满足这些条件时，可以认为：

- 这个 plan 在当前预算下基本探索到极限了

### 12.2 全局停止条件

整个系统可以在以下条件下停止：

1. theorem solved
2. 达到总预算
3. 没有 plan 再能带来有效改进

## 13. 系统流程图

```text
Target theorem
    |
    v
Retrieve past experience
    |
    v
Generate initial plan population
    |
    v
For each plan:
    allocate local island budget
        |
        v
    run plan-local evolution
        |
        +--> select frontier node
        +--> propose local action
        +--> run verifier
        +--> update node / plan score
        |
        v
    summarize local experience
    return final plan score
    |
    v
Outer-loop plan evolution
    - keep best/diverse plans
    - mutate plans
    - recombine plans
    - inject retrieved experience
    |
    v
Next generation of plans
```

## 14. 为什么这个方案比“完全自由漂移”更好

如果不把一个 plan 当作局部岛屿，而允许证明轨迹在全局上自由漂移，会有几个问题：

- 很难判断当前失败是 plan 问题还是局部 repair 不足
- 很难总结“这条路线”为什么失败
- 很容易退化成复杂版 iterative repair

相反，plan-local island 的好处是：

- 结构更清楚
- 预算更公平
- 经验总结更稳定
- 更容易写成层级式方法论文

## 15. 最容易被质疑的点

### 15.1 “这不就是多轮 repair 吗？”

回答：

- 不是单轨迹
- 个体是 plan，不是单个 proof draft
- 有显式 DAG、population、experience、outer-loop evolution

### 15.2 “这不就是 pass@k 吗？”

回答：

- plan 之间不是独立样本
- 有共享经验
- 有重组和结构化 mutation
- 有 plan-local PRM

### 15.3 “reward 太复杂，会不会很难调？”

回答：

- 先用简单版
- 从 `R_final + R_progress + R_reuse` 开始
- 再逐步加质量和效率项

## 16. MVP 实现建议

如果做第一版系统，我建议尽量克制。

### 第一版只做：

1. `PlanObject`
2. `SubgoalNode`
3. `LemmaObject`
4. `plan-local normalized progress reward`
5. `reusable lemma bonus`
6. `outer-loop plan mutation`

### 第一版先不做：

- 太复杂的 recombination
- 太多类型的 experience
- 过于复杂的效率项

先把最核心的层级搜索闭环跑通更重要。

## 17. 最后一句话

如果最后只留一句方法总结，我建议是：

> 我们把每一个 proof plan 视为一个固定预算的局部搜索岛屿：内层演化在既定 subgoal 拓扑下最大化 verifier-grounded 过程奖励，外层演化则在经验记忆的支持下生成、筛选和重组新的 plan。

## 18. 修订版实现方案：固定 Plan 的内层求解 + 全局经验库

本节用于覆盖和细化前文中“内层做 plan-local evolution”的早期表述。后续实现与论文写法，应优先采用本节定义。

### 18.1 修订后的核心分层

新的分层应明确为：

- 外层：
  - 维护 `plan population`
  - 做 `plan-level mutation / crossover / selection`
  - 决定哪个 plan 值得继续分配 inner budget
  - 在一个 plan 的固定预算耗尽后，基于其结果进行经验总结，并决定是否重新规划新 plan

- 内层：
  - 接收一个固定的 `PlanObject`
  - 不修改该 plan 的 `subgoals` 集合、依赖边和 theorem statement
  - 只围绕 frontier / blocked subgoals 做多轮 Lean-guided repair
  - 逐步把 subgoal 从 `pending / attempted / proved` 推进到 `verified`
  - 把中间成功引理和失败轨迹写回全局 memory

这里最关键的约束是：

- 外层改 `plan`
- 内层不改 `plan`，只改 `plan` 上节点的证明状态

这能把两个问题清晰分开：

1. 这个 decomposition / strategy 设计得好不好
2. 在固定 decomposition 下，这个 plan 是否真的容易被实现出来

### 18.2 为什么内层不应继续改整个 plan

如果内层仍然反复重写整份 plan，会出现几个问题：

- 很难判断失败到底来自 plan 设计不好，还是局部 Lean 修复没做好
- 外层和内层的 credit assignment 会纠缠在一起
- 容易被质疑为“只是多轮 iterative repair，直到成功为止”
- 很难给每个 plan 设定公平的固定探索预算

因此，更合理的设计是：

- 外层负责探索不同 plan
- 内层负责在一个 plan 固定不变的前提下，测试这个 plan 的 realization quality

也就是说，外层比较的是：

- 哪个 plan 在固定内层预算下更容易被做出来

而不是：

- 哪个 plan 在无限次 repair 下最后终于能被做出来

### 18.3 修订后的完整流程

```text
Input theorem / problem
        |
        v
Retrieve theorem-level experience + global lemma memory
        |
        v
Generate / mutate a plan population
        |
        v
Select one plan for inner realization
        |
        v
Freeze plan topology
  - keep theorem statement fixed
  - keep subgoal set fixed
  - keep dependency graph fixed
        |
        v
Inner fixed-plan subgoal solving loop
  - select frontier / blocked subgoal
  - retrieve relevant lemma / experience
  - generate local sketch + local Lean draft
  - run Lean verifier
  - update subgoal status and verifier state
  - extract reusable lemma / failure experience
        |
        v
Return realization result
  - verified subgoal ratio
  - theorem solved or not
  - generated verified lemmas
  - failure patterns
  - inner search score
        |
        v
Outer plan-level evolution
  - keep high-quality / diverse plans
  - discard exhausted plans
  - summarize memory
  - generate new plans
        |
        v
Repeat until theorem solved or global budget exhausted
```

### 18.4 修订后的 Proof Object 分层

为了支持上面的分层，建议把 proof object 明确拆成三层。

#### 18.4.1 PlanObject

`PlanObject` 仍然是外层种群中的主个体，但它更强调“计划结构”，而不是“动态 repair 轨迹”。

关键字段：

- `plan_id`
- `theorem_statement`
- `plan_summary`
- `strategy_type`
- `subgoal_graph`
- `status`
- `outer_score`
- `memory_refs`

这里的 `subgoal_graph` 一旦送入内层，在该次 inner rollout 中不应再被改动。

#### 18.4.2 SubgoalState

建议把当前 `SubgoalNode` 的运行态含义进一步明确成 `SubgoalState`。它代表一个固定节点当前被做到什么程度。

关键字段：

- `node_id`
- `statement_formal`
- `dependencies`
- `proof_sketch`
- `local_lean_draft`
- `status`
- `attempt_count`
- `diagnostics`
- `retrieved_lemmas`
- `retrieved_experiences`
- `local_reward`

它是内层优化的直接对象。

#### 18.4.3 LemmaAsset

建议把“可复用引理”从 plan 内局部对象进一步升级为全局知识资产。

关键字段：

- `lemma_id`
- `hypotheses_formal`
- `statement_formal`
- `full_proof_lean`
- `proof_status`
- `source_problem`
- `source_plan_id`
- `source_subgoal_ids`
- `verification_metadata`
- `retrieval_text`
- `hash`

其中最重要的三项是：

- 前置条件
- 引理陈述本身
- 完整 Lean 证明

如果缺了 hypotheses，只记录结论，后续复用会不安全。

### 18.5 全局 Memory 不应只是一组摘要

当前直觉上最重要的修订，是把长期记忆拆成两个全局库，而不是继续把所有信息都挂在单个 plan 上。

#### 18.5.1 Experience Memory

这个库存的是“成功/失败经验”，偏策略知识。

每条经验建议至少包含：

- `exp_id`
- `problem_id`
- `source_plan_id`
- `scope`
  - theorem-level
  - subgoal-level
  - lemma-level
- `target_statement`
- `context_hypotheses`
- `attempt_type`
  - planning
  - subgoal_repair
  - lemma_proving
- `outcome`
  - success
  - fail
  - partial
- `failure_type`
  - parse_error
  - lean_compile_error
  - sorry_remaining
  - statement_drift
  - wrong_lemma
  - blocked_dependency
  - timeout
- `diagnostics`
- `repair_action`
- `reward_delta`
- `metadata`

这个库主要在两个地方使用：

- 外层 planning 前，检索类似 theorem / 类似 decomposition 的经验
- 内层 repair 前，检索类似 subgoal / 类似 diagnostics 的经验

#### 18.5.2 Verified Lemma Library

这个库存的是可直接复用的证明资产。

建议进入条件非常严格：

- Lean compile success = 1
- no `sorry` = 1
- statement match = 1
- hypotheses 和完整 proof 都可恢复

每条记录至少包含：

- `lemma_id`
- `hypotheses_formal`
- `statement_formal`
- `full_proof_lean`
- `imports`
- `open_namespaces`
- `proof_status`
- `source_problem`
- `source_plan_id`
- `source_subgoal_ids`
- `diagnostics`
- `metadata`
- `embedding_text`
- `hash`

这个库会直接影响后续所有 plan 和 subgoal 的求解。

#### 18.5.3 Failed Lemma Attempt Memory

除了 verified lemma，还应显式保存失败的 lemma 尝试，但其用途不是“直接复用证明”，而是“避免重复犯错”。

每条失败记录建议包含：

- 目标 lemma statement
- hypotheses
- 尝试过的 Lean 代码
- 失败 diagnostics
- 失败类别
- 修复建议
- 是否接近成功

因此：

- 成功 lemma 是知识库
- 失败 lemma 是经验库

二者都应进入长期记忆，但作用不同。

### 18.6 修订后的检索时机

为了让 memory 真正参与决策，检索应嵌到两层循环中。

#### 外层 planning 前检索

query 由这些组成：

- theorem statement
- task constraint
- 当前 problem scaffold
- 历史成功 decomposition 模式
- 历史高价值 lemma

返回：

- 类似题的成功 plan
- 类似题的失败模式
- 可提前纳入新 plan 的 verified lemmas
- 常见 decomposition 模板

#### 内层 subgoal repair 前检索

query 由这些组成：

- 当前 subgoal 的 formal statement
- 当前可用 hypotheses
- dependency 已完成节点的结论
- 最新 Lean diagnostics
- 当前 frontier context

返回：

- 同类 subgoal 的成功 repair case
- 相关 verified lemmas
- 类似 diagnostics 的失败经验
- 可能有效的 proof pattern / tactic pattern

### 18.7 修订后的 Reward：明确拆成两层

reward 应显式区分 inner ranking 和 outer ranking。

#### 18.7.1 Inner Reward

内层分数用于：

- 同一个固定 plan 内，排序不同的 subgoal realization state
- 决定 inner loop 是否还值得继续

建议由这些项组成：

- `subgoal_verified_ratio`
- `frontier_progress`
- `blocked_reduction`
- `verified_lemma_gain`
- `diagnostic_improvement`
- `final_theorem_bonus`

只有当最终 theorem compile 成功、无 `sorry`、statement 对齐时，`final_theorem_bonus` 才取 1。

#### 18.7.2 Outer Reward

外层分数用于：

- 比较不同 plan 的质量
- 进行 selection / evolution / budget allocation

建议由这些项组成：

- `success_score`
  - 只有 theorem solved 时才等于 1
- `realization_score`
  - 固定 inner budget 下达到的实现程度
- `memory_gain_bonus`
  - 是否产出了高价值 lemma / experience
- `cost_penalty`
- `plan_complexity_penalty`

一个可行表达是：

```text
outer_score =
  b1 * success_score
  + b2 * realization_score
  + b3 * memory_gain_bonus
  - b4 * cost_penalty
  - b5 * plan_complexity_penalty
```

这里非常关键的一点是：

- `success_score = 1.0` 等价于最终 theorem 真正证明成功
- 如果没有达到 1.0，说明只是“离成功更近”，还不能算 solved

### 18.8 当前代码库与修订版目标的关系

从实现上看，当前代码库更接近：

- 外层维护 population
- 内层在 local island 中做固定预算的局部搜索（outer mutation + fixed-plan subgoal repair）
- 每一步的 outer mutation 仍然会生成一份新的完整 plan，但 inner repair 会保持 plan 拓扑不变，只修一个选定 subgoal 的局部 Lean draft

这说明当前系统已经有了：

- plan object
- Lean verifier bridge
- 局部预算
- 经验抽取

为了避免“文档说没做、代码其实已经做了”的误解，这里用一个实现状态表来对齐当前版本。

已完成（代码与单测/真实运行均已覆盖）：

- Phase 1：冻结内层
  - 内层已实现 `fixed-plan subgoal repair`（保持 DAG 拓扑不变，只修 subgoal 局部 Lean draft）
- Phase 2：建设全局 Memory（基础能力）
  - 已支持 theorem-level 与 subgoal-level 的经验写入与检索
  - prompt 已注入 top-k memory（theorem memories + lemma memories）
- Phase 3：增强外层 Evolve（核心能力）
  - adaptive budget allocation
  - memory-guided replanning
  - plan crossover
  - island specialization

部分完成（能力存在，但还没完全达到论文版表述的“最终形态”）：

- 全局 verified lemma asset library
  - 目前已存在 lemma memory 的全局持久化与检索，但尚未形成更强的“跨题长期复用资产层”（例如更系统的质量评估、去重归并、依赖/前提归一化等）
- 真实 multi-island crossover 覆盖
  - 当前代码、单测以及单独真实 case 已覆盖 crossover 逻辑，但在最近一次 multi-island smoke run 中，由于 `inspiration_ids=[]`，`crossover_recombination` 岛没有真实触发 `crossover_with_inspirations`，因此这部分还缺一个更有代表性的真实回归场景

未完成（属于实验与论文交付层，不是核心系统可运行性的缺口）：

- benchmark 大规模跑分与统计
- baseline 对比与消融
- 结果分析与论文图表整理

因此，后续工作应主要聚焦在：把“部分完成/未完成”这两类补齐成可写论文、可复现实验的版本，而不是重复实现 Phase 1-3 的核心机制。

### 18.9 建议的实现顺序

#### Phase 1：冻结内层

目标：

- 保留外层 `plan mutation`
- 内层改为 `fixed-plan subgoal repair`
- local island 不再修改 DAG 拓扑
- verifier 回写 subgoal 状态

这是最关键的一步。

#### Phase 2：建设全局 Memory

目标：

- 把 plan 上的局部 experience / lemma 抽取为全局存储
- 支持 theorem-level 和 subgoal-level 检索
- prompt 中注入 top-k memory

#### Phase 3：增强外层 Evolve

目标：

- plan mutation
- plan crossover
- island specialization
- memory-guided replanning
- 成本约束下的 budget allocation

### 18.10 修订版的一句话总结

如果要用一句最适合论文的方法描述来收束本节，我建议是：

> 我们维护一个外层的 plan population，并在每个候选 plan 上分配固定预算的内层 subgoal-solving search。内层不再改动 plan 拓扑，而是围绕 frontier subgoals 进行多轮 Lean-guided repair，并持续把 verified lemmas 与成功/失败经验写入全局 memory。外层基于 fixed-budget 下的 plan realization quality，而不是无限制的 repair 轮数，来进化更优的证明分解策略。
