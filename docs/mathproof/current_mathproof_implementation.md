# 当前代码库的 MathProof 全流程实现

这份文档只描述一件事：

- 当前 `famou-lean` 代码库里，已经实现的 `mathproof` 流程到底是怎么跑起来的

不包含：

- 还没有落地的外部 agent 形态
- Numina-Lean-Agent 的外部工作流
- Claude Code 手工交互式 debug Lean 的流程

当前实现的核心范式是：

- `LLM plan mutation`
- `deterministic Lean verification`
- `feedback / experience driven next mutation`

## 1. 一句话概括

当前 `mathproof` 把一个数学证明候选表示成一个 `proof plan object`。

每一轮流程是：

1. 从当前种群里选一个 parent proof plan
2. 让 LLM 生成一个新的 child proof plan
3. 把 child 中的 subgoals / lemmas / final theorem 转成 Lean 工件
4. 用 Lean 真正执行检查
5. 计算成功分和搜索分
6. 提取结构化经验
7. 更新 archive / population
8. 进入下一轮

## 2. 核心对象

当前 mathproof 主要围绕四类对象运行：

### 2.1 Program

外层 Famou 的统一候选解对象。

在 mathproof 中：

- `Program.code`
  - 存放序列化后的 proof plan YAML
- `Program.meta["plan_object"]`
  - 存放解析后的结构化 `PlanObject`
- `Program.metrics`
  - 存放 Lean verifier 返回的评测指标
- `Program.combined_score`
  - 顶层成功分
  - 对 mathproof，`1.0` 表示 theorem 已证明成功
- `Program.search_score`
  - 搜索分
  - 用于 evolve 排序

### 2.2 PlanObject

mathproof 的真正个体。

它包含：

- theorem statement
- plan summary / sketch
- subgoal DAG
- lemma pool
- verifier feedback
- experience refs

### 2.3 ExperienceObject

附着在 plan 上的结构化经验，而不是原始对话。

当前主要包含：

- `success_summary`
- `failure_pattern`
- `lemma_memory`

### 2.4 Lean Runtime Artifacts

每轮 evaluator 会把当前 plan 物化成一组 Lean 文件，供 Lean 校验和复现：

- `definition.lean`
- `step_i.lean`
- `sg_xxx.lean`
- `lemma_xxx.lean`
- `final_theorem.lean`
- `artifact_manifest.json`

## 3. 全流程框图

### 3.1 总体流程图

```text
Example Directory
  - problem.lean
  - init.scratch
  - evaluator.py
  - config.yaml
  - run.sh
        |
        v
python -m famou / bash run.sh
        |
        v
Load initial Program
  - Program.code = init.scratch
  - Program.meta["plan_object"] = parsed proof plan
        |
        v
Initial Enrichment
  - EvaluateModule
  - PlanJudge
  - (mathproof 默认不启用 EmbeddingFeature；如果启用会在此阶段补 feature_vector)
        |
        v
Archive / Initial Population
        |
        v
================ Evolution Loop ================
        |
        v
TopKSelect
  - choose parent
  - choose inspirations
        |
        v
PlanMutationGenerate
  - build prompt from:
    - parent plan
    - parent metrics
    - repair agenda
    - parent experiences
    - inspiration experiences
  - LLM outputs child proof plan
        |
        v
EvaluateModule
  - parse child plan
  - materialize Lean artifacts
  - verify subgoals / lemmas / final theorem
  - compute:
    - combined_score = success score
    - search_score = evolution ranking score
    - theorem_solved
    - compile_success / no_sorry / statement_match / ...
        |
        v
PlanJudge
  - summarize plan delta
  - build structural feedback
  - output:
    - frontier_nodes
    - blocked_nodes
    - next_actions
    - plan_metrics
        |
        v
EmbeddingFeature (optional)
  - feature_vector for diversity / clustering compatibility (mathproof 默认关闭)
        |
        v
Extract Experiences
  - success_summary / failure_pattern / lemma_memory
  - attach back into program + plan
        |
        v
Update Archive + Population
  - population ranking uses:
    - solved first
    - then search_score
        |
        v
Stop?
  - if theorem_solved == 1.0: solved
  - else next iteration
===============================================
```

### 3.2 一次 rollout 的内部图

```text
parent Program
    |
    v
build mutation prompt
    |
    v
LLM outputs child proof plan
    |
    v
parse -> PlanObject
    |
    v
for each subgoal / lemma / final theorem:
  build Lean artifact
    |
    v
  run Lean verifier
    |
    +--> compile_success
    +--> no_sorry
    +--> statement_match
    +--> diagnostics
    +--> latency
    |
    v
aggregate metrics
    |
    v
success_score + search_score
    |
    v
judge feedback + experience extraction
    |
    v
child Program committed
```

## 4. 输入目录是怎么被解释的

以 `examples/triangle_inequality/` 或 `examples/lean_and_left/` 为例，一个任务目录当前最重要的文件是：

- `problem.lean`
  - 目标 theorem scaffold
  - evaluator 会把 child plan 的 Lean draft 填到这里的 theorem 上下文中
- `init.scratch`
  - 初始 proof plan
  - 不是最终 Lean 代码，而是结构化 proof object 的 YAML
- `evaluator.py`
  - 任务绑定的 mathproof evaluator
  - 通常通过 `create_mathproof_example_evaluator(...)` 创建
- `config.yaml`
  - rollout / population / LLM / infra 配置
- `run.sh`
  - 启动命令

## 5. proof plan 里到底有什么

当前 proof plan 的最小关键字段是：

```yaml
plan_id: ...
theorem_statement: ...
plan_summary: ...
strategy_type: ...
sketch: ...
subgoals:
  - id: sg_x
    statement: ...
    dependencies: [...]
    proof_sketch: ...
    local_lean_draft: |
      theorem ...
    status: pending | in_progress | verified | ...
lemmas: [...]
```

这意味着当前 LLM 不是直接输出“完整最终证明”，而是输出一个更细粒度的证明对象：

- 子目标怎么拆
- 子目标之间依赖关系是什么
- 每个子目标局部 Lean 草稿是什么
- 有没有可复用 lemma

## 6. Generate 阶段在做什么

当前生成模块不是直接让模型“写 Lean 文件”，而是让模型“变异 proof plan”。

输入上下文主要包括：

- parent plan
- parent metrics
- judge feedback
- repair agenda
- parent experiences
- inspiration plans
- inspiration experiences

输出是一个新的 child plan。

也就是说，当前 LLM 负责的是：

- 子目标设定
- 子目标依赖拓扑
- lemma 候选
- 每个局部节点的 `proof_sketch`
- 每个局部节点的 `local_lean_draft`

它不负责直接调用 Lean。

## 7. Lean 转换、执行、debug 是怎么做的

这部分当前是“程序化流水线”，不是交互式 agent。

### 7.1 Lean 转换

`PlanLeanEvaluator` 会把 child plan 中的每个 subgoal / lemma / final theorem 转成真实 Lean 工件：

- 如果 `local_lean_draft` 已经是 declaration，就尽量保留
- 如果只是 tactic body，就包到 theorem scaffold 中
- 如果 example 目录里有 `problem.lean`，就复用其中的：
  - imports
  - preamble
  - suffix
- 对 final theorem，如果 scaffold 里有 `sorry`，就优先把 terminal proof body 填进去
- 如果 scaffold 里原本已经有完整 proof、没有 `sorry`，现在也会按 scaffold 的 theorem header 重建一个新的 final theorem 工件，而不是直接复用旧证明

### 7.2 Lean 执行

然后 evaluator 调 verifier，对这些 Lean 文件逐个检查：

- subgoal verification
- lemma verification
- final theorem verification

返回的典型结果包括：

- `compile_success`
- `no_sorry`
- `statement_match`
- `proof_validity`
- `diagnostics`
- `latency_ms`

### 7.3 debug 机制

当前没有一个单独的 Claude Code agent 在 theorem 内循环 debug。

当前的“debug”机制是：

1. Lean verifier 返回 diagnostics
2. Judge 总结 frontier / blocked nodes / next actions
3. Experience extractor 总结 failure patterns / repair hints
4. 下一轮 mutation prompt 读取这些信息
5. LLM 再生成下一版 child plan

所以当前 debug 的本质是：

- `verifier-guided next mutation`

而不是：

- `interactive theorem proving agent`

## 8. Evaluate 阶段如何判定成功

### 8.1 成功信号

当前真正的题目成功信号是：

- `theorem_solved == 1.0`

顶层分数语义已经改成：

- `combined_score = success_score`
- `search_score = search/evolution score`

因此对 mathproof：

- `combined_score == 1.0`
  - 表示 theorem 已成功证明
- `combined_score == 0.0`
  - 表示 theorem 尚未成功证明

### 8.2 theorem_solved 的判定

优先看 final theorem verification：

- final theorem compile success
- final theorem has no `sorry`
- final theorem statement matches expected theorem

如果当前任务绑定了 theorem scaffold，例如 benchmark / example 的 `problem.lean` 场景：

- 必须有成功的 full theorem verification
- 不能只因为 terminal subgoal 验证通过就记为 solved

只有在没有 scaffold、也无法组装 full theorem 的更弱场景下，才会退化为：

- terminal subgoals 都验证通过
- theorem statement 没漂移

## 9. 两层分数是怎么用的

### 9.1 success score

也就是当前 `Program.combined_score`。

作用：

- 表示“这题是否已经证完”
- 直观展示给用户

### 9.2 search score

也就是当前 `Program.search_score`。

作用：

- 选择 parent
- population 排序
- archive 中 best candidate 排序
- 在未 solved 的候选里区分优劣

当前排序逻辑是：

- solved candidates 必然优先
- solved / unsolved 两个层级内部，再看 `search_score`

## 10. Judge 阶段输出什么

Judge 当前做的是结构反馈，不负责真正打最终成功分。

它主要输出：

- `change_summary`
- `frontier_nodes`
- `blocked_nodes`
- `next_actions`
- `score_breakdown`

其中 `score_breakdown` 会保存在：

- `program.meta["plan_metrics"]`

这部分是 plan 结构分，不再覆盖 evaluator 给出的顶层成功分。

## 11. Experience 是何时抽取的

经验抽取发生在：

- rollout 成功
- 即将把 child program 提交进 archive / population

时机上，它是在 evaluator 和 judge 之后。

当前不是每个失败中间态都单独存一条全局记忆，而是对“最终被提交的这个 child”做蒸馏。

## 12. Experience 里存了什么

当前 experience 不是原始推理轨迹，而是压缩后的结构化摘要。

主要内容包括：

- 这次 plan 最终 `solved` 还是 `unsolved`
- `success_score`
- `search_score`
- `compile_success`
- `no_sorry`
- `statement_match`
- failure patterns
- repair hints
- reusable lemmas

存储位置：

- `program.meta["experience_pool"]`
- `program.meta["experience_refs"]`
- `plan.metadata["experiences"]`

下一轮 mutation 时，会把 parent / inspirations 的 experience 序列化到 prompt 里。

## 13. local island 是什么

如果配置了 `local_island_budget > 1`，那么一次外层 iteration 不只是一轮单步 rollout，而是一个局部小搜索。

定义上：

- `inner rollout`
  - 一次 parent -> child 的单步局部搜索
  - 在当前 `mathproof_plan_island` 中：
    - local step `0` 是 `PlanMutationGenerate`
    - 后续 local steps 是 `SubgoalRepairGenerate`
  - 也就是说，inner 不再等价于“整 plan mutation”
- `outer iteration`
  - 一个 local island 搜索预算
  - 里面可能执行多个 inner rollouts

局部岛搜索会：

- 维护一个临时 local population
- 先做一次 outer mutation
- 再在固定拓扑上做多步 subgoal repair
- 如果某一步已经 solved，可提前停
- 最后只把选中的 best / last 结果提交回全局 archive

### 13.1 adaptive budget allocation

现在 `local island` 除了固定预算，还支持自适应预算分配。

相关配置项是：

- `local_island_budget`
  - 默认预算上限
- `local_island_budget_mode`
  - `fixed` 或 `adaptive`
- `local_island_budget_min`
  - 自适应模式下的最小预算
- `local_island_budget_max`
  - 自适应模式下的最大预算

当前自适应分配发生在每次外层 rollout 开始前。

控制器会从当前 population snapshot 里选一个 anchor plan，然后根据它的当前状态决定这一轮局部岛到底给多少步预算。

当前使用的信号包括：

- `success_score`
- `search_score`
- `progress_score`
- `cost_penalty`
- `frontier_size`
- `blocked_size`

直观上：

- 越接近成功
- frontier 越多、blocked 越少
- verifier / repair 成本越低

就越值得多给一些局部修复预算。

如果 anchor 已经 `theorem_solved == 1.0` 或 `success_score == 1.0`，那么预算会直接收缩到 `min_budget`，避免在已解题目上继续浪费局部搜索。

### 13.2 adaptive budget 的落盘位置

这部分不是只在运行时内存里存在，当前会写进 rollout 结果里，便于复盘：

- `result.stats["local_island_budget"]`
  - 这一轮真正分配到的 budget
- `result.stats["local_island_budget_summary"]`
  - 分配依据摘要
- `result.stats["local_island_history"]`
  - 这次局部岛实际执行了哪些 local steps
- `program.meta["local_island_summary"]`
  - 被提交回 archive 的那个最终候选对应的局部岛摘要

其中 `local_island_budget_summary` 目前会记录：

- 分配模式
- base / min / max budget
- allocated budget
- anchor program id
- 分配原因
- success / progress / cost / frontier / blocked 等关键指标

### 13.3 当前真实回归现象

对 `examples/lean_and_left/` 的真实 Lean 回归中，seed plan 的自适应预算被分配为 `3/4`，然后在 `local step 0` 的 outer mutation 就已经把 theorem 解出来，因此：

- `allocated_budget = 3`
- `steps_executed = 1`
- 提前停止

这说明当前逻辑是：

- 先做预算分配
- 再执行局部岛
- 一旦 theorem solved，就按 `stop_on_solved` 立即截断

## 14. 当前实现边界

当前代码库已经实现的是：

- plan-centric proof object
- LLM plan mutation
- deterministic Lean verification
- success score + search score
- judge structural feedback
- experience extraction and reuse
- local island inner search
- adaptive local island budget allocation
- fixed-plan subgoal repair
- scaffold-aware final theorem assembly
- benchmark / example scaffold 场景下更严格的 theorem success 判定
- theorem / subgoal / lemma memory 回写

当前还没有实现的是：

- 每个 subgoal 由独立 proving agent 显式接管
- 在线 theorem search agent / Lean MCP 多轮对话
- 全局跨题 experience vector database
- proof trace replay / tactic-level tree search

## 15. 用一句话总结

当前 `famou-lean` 的 mathproof 不是“交互式 Lean agent”，而是：

- 把数学证明表示成 proof plan population
- 让 LLM 持续变异这个 plan
- 用 Lean 作为确定性 verifier 给出真反馈
- 用 success score、search score、judge feedback、experience memory 驱动下一轮演化
