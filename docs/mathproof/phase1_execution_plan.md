# MathProof Phase 1 执行方案

本文档只回答一件事：

- 如何把当前 `famou-lean` 的 mathproof 实现，从“内层整 plan mutation”改成“固定 plan 的内层 subgoal repair loop”

不包含：

- Phase 2 的全局 memory 详细实现
- Phase 3 的外层 crossover / budget allocation 细节

## 1. Phase 1 目标

Phase 1 的唯一核心目标是：

- 外层继续做 `plan-level mutation`
- 内层 local island 改为 `fixed-plan subgoal repair`
- 在不破坏当前 benchmark 与 Lean verifier 通路的前提下，让一个 plan 在固定 budget 内被“实现”，而不是被“重写”

更具体地说，Phase 1 完成后：

- 进入 inner rollout 的 `PlanObject` 拓扑被冻结
- inner steps 只允许更新：
  - `proof_sketch`
  - `local_lean_draft`
  - `status`
  - `verifier_state`
  - `lemma_pool` 中的新局部 lemma
- inner steps 不允许更新：
  - theorem statement
  - subgoal ids
  - dependency graph
  - strategy type

## 2. 当前代码中的切入点

### 2.1 外层 strategy

当前 strategy 组装位于：

- `famou/strategies/mathproof_plan_island.py`

它现在的核心问题是：

- `local_island_budget` 复用了同一条 rollout
- rollout 中的生成模块仍然是 `PlanMutationGenerate`
- 因而 inner loop 里的每一步都可能生成一份新的完整 plan

### 2.2 inner loop controller

当前 local island 循环位于：

- `famou/controller/evolver.py`

它已经具备以下优势：

- 可以维护临时 local population
- 可以按固定 budget 执行 inner steps
- 可以在 solved 时早停
- 可以把最优结果回传给外层

因此，Phase 1 不需要推翻这层控制器，而应改造它的“inner operator”。

### 2.3 Lean verifier

当前 Lean verifier 通路位于：

- `famou/utils/plan_lean_evaluator.py`

这一层基本可以保留，因为它已经支持：

- subgoal 级 artifact 物化
- lemma 级 artifact 物化
- final theorem 检查
- verifier 结果回写

Phase 1 需要做的更多是：

- 调整它与新 `SubgoalRepairGenerate` 的数据接口
- 让它更自然地支持“单个 subgoal 的局部修复”

## 3. Phase 1 的模块设计

## 3.1 新增 `SubgoalRepairGenerate`

建议新增：

- `famou/modules/generate/subgoal_repair.py`

职责：

- 输入一个固定 plan
- 选择一个待修复 subgoal
- 基于当前 subgoal 状态、依赖上下文、diagnostics 生成局部修复结果
- 输出仍然是一个 `Program`，但其中的 `plan_object` 必须保持同一拓扑，只更新局部字段

建议 prompt 输入：

- target theorem
- frozen plan summary
- selected subgoal formal statement
- dependency closure summary
- current subgoal draft
- current diagnostics
- available local lemmas
- selected frontier / blocked nodes

建议 prompt 输出：

- `node_id`
- `updated_proof_sketch`
- `updated_local_lean_draft`
- optional `introduced_lemmas`
- optional `repair_rationale`

注意：

- 这里不要求 LLM 返回整份 plan YAML
- 只返回局部 patch 信息，随后由 runtime 合并进原 plan

## 3.2 新增 `subgoal_selection.py`

建议新增：

- `famou/utils/subgoal_selection.py`

职责：

- 根据当前 plan state 选择本轮 inner step 要处理的 subgoal

第一版策略建议简单且可测：

1. 优先选 `blocked frontier`
2. 再选 `frontier`
3. 再按：
   - dependency importance
   - diagnostics severity
   - attempt_count
   - node status
   排序

输出：

- `selected_node_id`
- `selection_reason`
- `candidate_node_ids`

## 3.3 新增 `subgoal_repair_runtime.py`

建议新增：

- `famou/utils/subgoal_repair_runtime.py`

职责：

- 接收 parent plan 和 subgoal patch
- 校验 patch 只修改允许字段
- 合并 patch，生成 child plan
- 计算 topology hash，确认 graph 未变化

建议它显式执行三类校验：

1. theorem statement 未变化
2. subgoal id 集合未变化
3. dependencies 未变化

如果违反约束：

- 直接判定该 repair 结果无效
- 写入 error_info / repair failure

## 3.4 proof object 扩展

建议更新：

- `famou/core/proof_objects.py`

需要给 `SubgoalNode` 增加最小运行态字段：

- `attempt_count`
- `attempt_history`
- `last_diagnostic`
- `retrieved_lemma_ids`
- `retrieved_experience_ids`
- `last_repair_summary`

目标不是一步到位把 memory 全塞进去，而是先把 inner repair 所需运行态补齐。

## 4. Controller 改造方案

建议改：

- `famou/controller/evolver.py`

当前逻辑是：

- 对 local island 反复调用完整 rollout

Phase 1 改法建议分两种路线。

### 方案 A：复用现有 rollout，新增一条 repair strategy

新增一条 strategy，例如：

- `mathproof_fixed_plan_repair`

其 rollout 可以是：

- `TopKSelect`
- `SubgoalRepairGenerate`
- `EvaluateModule`
- `PlanJudge`

然后 local island 在 mathproof 场景下调用这条新 rollout。

优点：

- 改动边界清楚
- 不会污染当前 `mathproof_plan_island` baseline
- 方便 A/B 对比

这是我建议的方案。

### 方案 B：在现有 strategy 中用 config 切换 generate module

这也能做，但缺点是：

- 逻辑容易混在一起
- baseline 与新方案边界不清楚

因此，不建议作为第一步。

## 5. Judge 与 Evaluator 的调整

## 5.1 Judge 调整

建议更新：

- `famou/modules/judge/plan_judge.py`

当前 judge 偏重：

- plan delta
- frontier / blocked nodes

在 Phase 1 中，需要增加对 inner realization 的反馈：

- 当前 subgoal 是否推进
- blocked 是否减少
- diagnostics 是否改善
- 是否出现 verified lemma

Judge 不需要变成一个复杂 LLM agent，但要让反馈更贴近 inner repair。

## 5.2 Evaluator 调整

建议更新：

- `famou/utils/plan_lean_evaluator.py`

重点不是推翻，而是补以下能力：

- 支持“只针对 selected subgoal 做优先局部验证”
- 支持把本轮 selected subgoal 的结果回写成更明确的 node-level metrics
- 对 introduced lemmas 给出更细的 verifier feedback

建议保留最终 theorem 的 periodic check，因为：

- 某些 plan 在中途就可能已经足够 assemble 成 final theorem

## 6. 开发顺序

建议严格按下面顺序推进，不要并行改太多。

### Step 1：建立 topology freeze 机制

文件：

- `famou/utils/subgoal_repair_runtime.py`
- `famou/core/proof_objects.py`

产物：

- topology hash
- patch merge
- graph invariants check

验收：

- 给定 parent plan 和 patch，若 patch 改了 dependencies，必须被拒绝

### Step 2：实现 subgoal selection

文件：

- `famou/utils/subgoal_selection.py`

产物：

- frontier / blocked 节点选择器

验收：

- 对同一个 plan，selection 结果稳定可预测

### Step 3：实现 `SubgoalRepairGenerate`

文件：

- `famou/modules/generate/subgoal_repair.py`
- `famou/prompts/templates/generation/subgoal_repair.txt`

产物：

- 基于 selected subgoal 的局部 repair 生成器

验收：

- 返回的 patch 能合并进原 plan
- 合并后 topology 不变

### Step 4：接到新 strategy

文件：

- `famou/strategies/mathproof_fixed_plan_repair.py`
- strategy registry 相关文件

产物：

- 一条与 baseline 并行存在的新 strategy

验收：

- 能跑最小 example

### Step 5：让 controller 的 local island 使用新 repair rollout

文件：

- `famou/controller/evolver.py`

产物：

- local island 调用 fixed-plan inner operator

验收：

- inner step history 中记录 selected subgoal
- solved 时早停

### Step 6：补测试与 smoke run

建议新增测试：

- `tests/test_subgoal_selection.py`
- `tests/test_subgoal_repair_runtime.py`
- `tests/test_subgoal_repair_generate.py`
- `tests/test_mathproof_fixed_plan_repair.py`

最小真实跑法：

- `examples/triangle_inequality`
- miniF2F easy smoke set

## 7. Phase 1 验收标准

Phase 1 完成必须同时满足：

1. baseline strategy 仍然可运行
2. 新 fixed-plan strategy 可运行
3. 新 strategy 的 inner loop 中，plan topology hash 始终不变
4. `success_score = 1.0` 时仍然表示 theorem 真正 solved
5. 至少一个 example 和一组 miniF2F smoke 能完整跑通

## 8. 风险与控制

### 风险 1：局部 patch 格式不稳

控制：

- 先让 LLM 输出严格 JSON patch，而不是自由文本
- merge runtime 做强校验

### 风险 2：inner repair 没有足够上下文

控制：

- prompt 中显式加入 dependency closure、当前 diagnostics、已有 lemmas

### 风险 3：改动 controller 后 baseline 回归受影响

控制：

- 新 strategy 与旧 strategy 并存
- 先不替换 baseline，只新增

### 风险 4：verification 成本上升

控制：

- 先优先局部验证 selected subgoal
- final theorem check 保持为轻量周期性检查

## 9. 立即执行建议

如果按当前优先级往下做，我建议下一个开发循环就做这三件事：

1. 新增 `subgoal_repair_runtime.py`
2. 新增 `subgoal_selection.py`
3. 新增 `tests/test_subgoal_repair_runtime.py`

因为这三项不依赖 LLM 输出格式，可以先把“固定 plan”这条边界彻底立住。
