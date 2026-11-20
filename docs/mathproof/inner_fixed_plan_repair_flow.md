# 当前代码中的 Inner Fixed-Plan Repair 流程

这份文档只描述当前 `famou-lean` 代码库里已经实现的 `inner fixed-plan repair`。

不包含：

- 还没有实现的多次在线检索刷新
- 一个 inner step 内部再做多轮 Claude debug
- 任何外部 Numina / MCP 编排逻辑

## 1. 一句话概括

当前 `inner` 的本质是：

- 在一个已经生成出来的 `PlanObject` 上
- 固定 theorem 与 subgoal DAG 拓扑
- 每次只选择一个最值得修的 subgoal
- 检索相关的 `subgoal memory + lemma memory`
- 让 LLM 输出一个局部 `repair patch`
- 把 patch merge 回 parent plan
- 再走一次 Lean verifier
- 把这一步的 subgoal / lemma 经验写回全局 memory

所以当前 inner 的粒度是：

- `一次 inner rollout = 一次选点 + 一次 patch + 一次 Lean verify`

而不是：

- 一个 inner rollout 内部自己做很多轮 debug 直到修好

## 2. 它在整体流程中的位置

当前外层策略在：

- `famou/strategies/mathproof_plan_island.py`

运行方式是两段式 local island：

1. 第 0 步先跑一次 `outer_mutation`
2. 后续步数再跑 `inner_repair`

也就是说，inner 不是从 seed 直接开始，而是：

- 先有一个 outer mutation 产出的 child plan
- 然后在这个 child plan 上做固定拓扑的局部修复

## 3. 流程图

```text
+-----------------------------------------------------------+
| Outer mutation child plan                                 |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| select_subgoal_for_repair                                 |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Build repair prompt                                       |
| - selected subgoal                                        |
| - dependency context                                      |
| - local lemmas                                            |
| - parent experiences                                      |
| - global subgoal memories                                 |
| - global lemma memories                                   |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| LLM outputs JSON repair patch                             |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| merge_subgoal_repair_patch                                |
| - update one node only                                    |
| - optionally merge introduced lemmas                      |
| - check topology signature unchanged                      |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| EvaluateModule                                            |
| - verify subgoal Lean drafts                              |
| - verify lemma Lean drafts                                |
| - verify full theorem                                     |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| PlanJudge                                                 |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| extract_and_attach_experiences                            |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| persist_program_memory                                    |
| - theorem memory                                          |
| - subgoal step memory                                     |
| - lemma memory                                            |
+-----------------------------------------------------------+
                            |
                            v
+-----------------------------------------------------------+
| Next inner step or stop                                   |
+-----------------------------------------------------------+
```

## 4. Step by Step

## 4.1 进入 inner

入口在：

- `famou/controller/evolver.py`

关键函数：

- `_execute_local_island_rollout(...)`

行为是：

- 先复制当前 island population 到 `local_population`
- local step `0` 用 `self.current_rollout` 跑 `outer_mutation`
- local step `1..budget-1` 用 `inner_rollout` 跑 `inner_repair`

这里的 `inner_rollout` 来自：

- `famou/strategies/mathproof_plan_island.py`

其中 inner modules 是：

1. `TopKSelect`
2. `SubgoalRepairGenerate`
3. `EvaluateModule`
4. `PlanJudge`

## 4.2 选择一个 subgoal

选择逻辑在：

- `famou/utils/subgoal_selection.py`

函数：

- `select_subgoal_for_repair(plan)`

当前优先级是：

1. verifier feedback 里明确 blocked 的节点
2. frontier 上且依赖已满足的节点
3. 其他 unresolved 节点

排序时还会考虑：

- dependency importance
- diagnostic severity
- attempt count
- node status

所以当前 inner 是单点修复，不是多点并行修复。

## 4.3 检索 memory 并构造 prompt

逻辑在：

- `famou/modules/generate/subgoal_repair.py`

函数：

- `SubgoalRepairGenerate.build_prompt(...)`

这一步会构造两类上下文。

### 4.3.1 本地 plan 上下文

包括：

- 当前 `frozen_plan`
- `topology_signature`
- 被选中的 `selected_node`
- 它的 `statement / status / proof_sketch / local_lean_draft / last_diagnostic`
- `dependency_context`
- `local_lemmas`
- 当前 `frontier_nodes / blocked_nodes`
- `parent_experiences`

### 4.3.2 全局 memory 上下文

包括：

- `retrieve_subgoal_memories(...)`
- `retrieve_lemma_memories(...)`

当前 query 方式是：

- subgoal memory：
  - `selected node.statement_formal + node.last_diagnostic`
- lemma memory：
  - `theorem_statement + selected node.statement_formal`

这些 memory 会被格式化进 prompt：

- `Global subgoal memories`
- `Global lemma memories`

## 4.4 LLM 只输出 repair patch

当前 inner 不要求 LLM 重写整个 plan，而是只输出一个 JSON patch。

结构定义在：

- `famou/utils/subgoal_repair_runtime.py`

数据结构：

- `SubgoalRepairPatch`

允许修改的内容主要有：

- `updated_proof_sketch`
- `updated_local_lean_draft`
- `updated_status`
- `verifier_state_update`
- `introduced_lemmas`
- `retrieved_lemma_ids`
- `retrieved_experience_ids`
- `last_diagnostic`
- `repair_summary`

这意味着 inner 当前是一个严格的“局部编辑器”。

## 4.5 merge patch，但不允许改拓扑

merge 逻辑也在：

- `famou/utils/subgoal_repair_runtime.py`

函数：

- `merge_subgoal_repair_patch(...)`

它会：

1. deep copy parent plan
2. 找到被选中的 node
3. 只更新这个 node 的局部状态
4. 记录：
   - `attempt_count`
   - `attempt_history`
   - `retrieved_lemma_ids`
   - `retrieved_experience_ids`
   - `last_diagnostic`
   - `last_repair_summary`
5. 如果 patch 带了 `introduced_lemmas`，合并进 `lemma_pool`
6. 刷新 `frontier_nodes`
7. 重新计算 `fixed_plan_topology_signature`

随后它会调用：

- `assert_plan_topology_unchanged(...)`

这里强约束：

- theorem statement 不能改
- strategy type 不能改
- subgoal id 不能改
- dependencies 不能改
- subgoal 不能增删

所以当前 inner 的定义非常清楚：

- `fix topology, repair node-local proof state`

## 4.6 merge 后再走 Lean verifier

child plan merge 完之后，不会直接认为修复成功，而是继续经过：

- `EvaluateModule`

当前 evaluator 是：

- `famou/utils/plan_lean_evaluator.py`

它会做三类验证：

1. `subgoal_results`
   - 检查当前 plan 里 subgoal 的 Lean draft
2. `lemma_results`
   - 检查 lemma pool 里的 Lean draft
3. `full_theorem_result`
   - 尝试拼出最终 theorem 工件再验证

这一步会回写：

- subgoal status
- lemma status
- `theorem_solved`
- `compile_success`
- `no_sorry`
- `statement_match`
- `success_score`
- `search_score`

## 4.7 生成 judge feedback

Lean 验证之后会经过：

- `PlanJudge`

它负责做结构化反馈总结，例如：

- 哪些 frontier 还没推进
- 哪些 blocked node 还在
- 下一步建议关注什么

Judge 本身不改 plan，只负责把 verifier signal 变成更适合下一轮生成的反馈。

## 4.8 把这一步写回 memory

rollout 成功后，`Evolver` 会做两步后处理：

1. `extract_and_attach_experiences(...)`
2. `persist_program_memory(...)`

当前 inner 相关的 memory 写回包括：

### theorem-level

- 当前这次 rollout 的 theorem run memory

### subgoal-level

- 当前这次 repair step 的 subgoal attempt memory
- 包括：
  - `before_status / after_status`
  - `diagnostic_before / diagnostic_after`
  - `action_type`
  - `reward_delta`
  - `search_score_delta`
  - `trajectory_metadata`

### lemma-level

- 当前这一步里涉及的 lemma attempt memory
- 包括成功和失败的 lemma

所以 inner 不是一个“只消费上下文”的过程，它同时会持续生产长期记忆。

## 5. 当前 inner 已实现了什么

已经实现的能力：

- 固定 plan 拓扑下的单点 repair
- deterministic subgoal selection
- subgoal memory 检索
- lemma memory 检索
- JSON patch generation
- patch merge + topology invariant check
- Lean verifier grounded scoring
- subgoal / lemma / theorem memory 回写

## 6. 当前 inner 还没有实现什么

当前还没有实现：

- 一个 inner step 内部连续多轮 debug 同一个 node
- verifier 失败后的即时二次检索刷新
- 多个 subgoal 并行 repair
- 根据 memory 动态改 subgoal selection policy
- 基于 memory 的 patch reranking

这些都可以作为下一阶段增强项，但不属于当前已经落地的 inner。

## 7. 当前 inner 的准确定位

如果用一句更论文化的话来描述当前 inner：

> 当前 inner 是一个 verifier-grounded fixed-topology local repair operator。它在给定 proof plan 的前提下，选择单个 subgoal，检索相关经验与 lemma 记忆，生成局部 patch，并通过 Lean verifier 将这一步修复转化为 step-level reward 与长期记忆。

这也是为什么它天然适合后续做：

- PRM-style reward
- subgoal-level experience reuse
- plan outer loop + subgoal inner loop 的层级搜索
