# 当前代码库工作总结与运行逻辑总览

这份文档只描述当前 `famou-lean` 代码库里已经落地、已经能运行、并且已经过实际验证的内容。

本文档重点回答四个问题：

1. 我们目前已经完成了哪些工作
2. 当前代码库里实际存在哪几条运行路径
3. 当前推荐的 mathproof 主流程到底是怎么跑起来的
4. 运行过程中会生成哪些中间产物和结果目录

不包含：

- 还没有落地的未来规划
- 论文视角下的方法扩展
- 外部交互式 agent 产品形态

## 1. 一句话总结

当前 `famou-lean` 已经具备一套可运行的 Lean 数学证明框架：

- 外层用 `Famou` 做 population / archive / iteration 管理
- 中间用 `PlanObject` 表示结构化证明计划
- 生成阶段让 LLM 变异 plan，而不是直接盲写完整 Lean
- 评测阶段把 plan 物化成 Lean 工件，再用真实 Lean 工程做校验
- 校验结果被转换成 `success_score + search_score + experience`
- 支持 benchmark 物化、批量运行、suite 汇总

同时，代码库里还保留了一条较早期的 `proof_runtime/evaluator_lean.py` 两阶段流程，用于简单示例和兼容老入口。

## 2. 当前已经完成的工作

截至当前仓库状态，已经完成并落地的内容可以分为 8 类。

### 2.1 目录与运行入口整理

已经整理出统一的 `examples/` 目录结构，当前可直接运行的入口包括：

- `examples/triangle_inequality/`
- `examples/lean_and_left/`
- `examples/lean_identity_namespace/`
- `examples/minif2f_test_smoke/`
- `examples/_template/`

这意味着：

- 单题 example 有统一入口
- benchmark 有统一入口
- 新题目可以直接从模板复制

### 2.2 当前推荐的 mathproof plan 流程

已经实现了一条基于 `PlanObject` 的 mathproof 主流程，核心包括：

- 结构化 proof plan 表示
- subgoal DAG
- local Lean draft
- Lean artifact 物化
- project-context Lean bridge 校验
- success / search 双层分数
- verifier-guided experience 提取

主要代码位置：

- [famou/utils/mathproof_example.py](../../famou/utils/mathproof_example.py)
- [famou/utils/plan_lean_evaluator.py](../../famou/utils/plan_lean_evaluator.py)
- [famou/utils/proof_runtime_bridge.py](../../famou/utils/proof_runtime_bridge.py)
- [famou/utils/plan_runtime.py](../../famou/utils/plan_runtime.py)

### 2.3 外层 plan evolve

已经实现：

- parent selection
- inspiration selection
- LLM-based plan mutation
- archive / population 更新
- solved 优先排序
- search-score 驱动的未解候选排序

主要代码位置：

- [famou/modules/generate/plan_mutation.py](../../famou/modules/generate/plan_mutation.py)
- [famou/controller/evolver.py](../../famou/controller/evolver.py)

### 2.4 内层 fixed-plan subgoal repair

已经实现一条“固定 plan 拓扑、只修 subgoal”的内层局部修复流程：

- 先做一次 outer plan mutation
- 再围绕 frontier subgoals 做多轮 repair
- 不在 inner loop 中重建整个 plan 拓扑
- 可以在 fixed-plan 下逐步把 subgoal 从 `pending / in_progress` 推进到更强状态

主要代码位置：

- [famou/modules/generate/subgoal_repair.py](../../famou/modules/generate/subgoal_repair.py)
- [famou/utils/subgoal_repair_runtime.py](../../famou/utils/subgoal_repair_runtime.py)

### 2.5 奖励与经验系统

当前已经实现：

- `success_score`
  - 用于“这题是否真正证明成功”
- `search_score`
  - 用于 evolve 排序
- `realization_score`
- `memory_gain_bonus`
- `plan_complexity_penalty`
- failure pattern / repair hint 抽取
- verified lemma 信号抽取

也就是说，当前系统已经不是只有一个粗糙的二值分数，而是：

- 一层对外成功语义
- 一层对内搜索排序语义

主要代码位置：

- [famou/utils/plan_runtime.py](../../famou/utils/plan_runtime.py)
- [famou/utils/proof_experience.py](../../famou/utils/proof_experience.py)

### 2.6 全局记忆与 lemma 复用

已经落地：

- theorem-level experience
- subgoal-level experience
- verified lemma library
- 基于 memory 的 replanning 支持

主要代码位置：

- [famou/utils/global_memory_store.py](../../famou/utils/global_memory_store.py)

### 2.7 多岛搜索、memory-guided replanning、plan crossover

已经实现并验证了：

- island specialization
- `memory_guided_replanning`
- `crossover_recombination`
- 基于 inspiration 的 plan crossover

主要代码位置：

- [famou/controller/evolver.py](../../famou/controller/evolver.py)
- [famou/modules/generate/plan_mutation.py](../../famou/modules/generate/plan_mutation.py)

### 2.8 benchmark 支持

已经接入：

- `miniF2F-test`
- `PutnamBench`

并且已经实现：

- benchmark theorem 枚举
- 单题物化为当前 example 风格目录
- 批量执行
- suite 级 JSON/CSV 汇总

主要代码位置：

- [famou/utils/mathproof_benchmarks.py](../../famou/utils/mathproof_benchmarks.py)
- [famou/utils/mathproof_benchmark_runner.py](../../famou/utils/mathproof_benchmark_runner.py)

## 3. 当前代码库里实际存在的两条运行路径

当前代码库里，Lean 数学证明相关的运行路径有两条。

### 3.1 路径 A：当前推荐的 plan-based mathproof 主路径

这是当前更重要、也是 benchmark 和新 mathproof 实验主要使用的路径。

特点：

- 输入是结构化 `init.scratch` proof plan
- evaluator 通过 `create_mathproof_example_evaluator(...)` 构造
- 会把 plan 物化成 `definition/step/subgoal/final_theorem` Lean 工件
- 会通过 Lean project bridge 做真实校验
- 支持 memory、repair、crossover、island specialization

典型入口：

- [examples/lean_and_left/run.sh](../../examples/lean_and_left/run.sh)
- [examples/lean_identity_namespace/run.sh](../../examples/lean_identity_namespace/run.sh)
- benchmark runner 生成的 task 目录

### 3.2 路径 B：legacy `evaluator_lean.py` 两阶段流程

这条路径是更早期的实现，当前仍然保留，主要用于：

- 简单最小案例
- 兼容原有 example 入口

特点：

- 输入是 `.scratch`
- evaluator 内部先生成 `step_*.lean`
- 再调用 Claude 去补全 `sorry`
- 然后再做验证

典型入口：

- [examples/triangle_inequality/run.sh](../../examples/triangle_inequality/run.sh)
- [proof_runtime/evaluator_lean.py](../../proof_runtime/evaluator_lean.py)

### 3.3 当前建议

如果是：

- 新的 mathproof 功能开发
- benchmark 实验
- plan / memory / repair / crossover 相关研究

应优先使用路径 A。

如果只是：

- 最快验证 Lean 环境和旧 evaluator 能不能跑

可以使用路径 B。

## 4. 完整程序框图

下面这张图把当前代码库里的主流程和 legacy 分支都放在一起。

```text
User Command
  |
  +--> bash examples/triangle_inequality/run.sh
  |         |
  |         v
  |     run_famou.py
  |         |
  |         v
  |     proof_runtime/evaluator_lean.py
  |         |
  |         v
  |     [Legacy two-stage path]
  |
  +--> bash examples/lean_and_left/run.sh
  |         |
  |         v
  |     python -m famou
  |         |
  |         v
  |     examples/.../evaluator.py
  |         |
  |         v
  |     create_mathproof_example_evaluator(...)
  |         |
  |         v
  |     [Current plan-based path]
  |
  +--> python -m famou.utils.mathproof_benchmark_runner ...
            |
            v
        materialize benchmark tasks
            |
            v
        generated task dir
          - problem.lean
          - init.scratch
          - config.yaml
          - evaluator.py
          - run.sh
            |
            v
        create_mathproof_example_evaluator(...)
            |
            v
        [Current plan-based path]


================ Current plan-based path ================

Load Program
  - Program.code = init.scratch
  - parse into PlanObject
  - attach theorem scaffold / task description
        |
        v
Initial Enrichment
  - EvaluateModule
  - PlanJudge
        |
        v
Archive / Population initialized
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
  - read parent plan
  - read plan metrics
  - read frontier / blocked nodes
  - read experience pool
  - read inspirations
  - read island specialization / crossover directive
        |
        v
LLM outputs child PlanObject
  - subgoals
  - dependencies
  - proof sketches
  - local Lean drafts
  - lemma candidates
        |
        v
EvaluateModule
  |
  +--> PlanLeanEvaluator
  |       |
  |       +--> materialize Lean artifacts under temp_plan_eval/
  |       |      - definition.lean
  |       |      - step_*.lean
  |       |      - sg_*.lean
  |       |      - lemma_*.lean
  |       |      - final_theorem.lean
  |       |      - artifact_manifest.json
  |       |
  |       +--> ProofRuntimeBridgeVerifier
  |              |
  |              +--> stage artifacts under temp/plan_bridge/
  |              +--> build wrapper files:
  |              |      - _verify_sg_xxx.lean
  |              |      - _verify_final_theorem.lean
  |              |
  |              +--> run lake lean in shared Lean project
  |              |
  |              +--> return:
  |                     compile_success
  |                     no_sorry
  |                     statement_match
  |                     proof_validity
  |                     diagnostics
  |
  +--> aggregate plan metrics
         - theorem_solved
         - success_score
         - search_score
         - realization_score
         - memory_gain_bonus
         - plan_complexity_penalty
        |
        v
PlanJudge
  - change_summary
  - frontier_nodes
  - blocked_nodes
  - next_actions
  - score_breakdown
        |
        v
Experience Extraction
  - success summary
  - failure pattern
  - repair hints
  - reusable lemmas
        |
        v
Global Memory Update
  - theorem experience
  - subgoal experience
  - verified lemma library
        |
        v
Population / Archive Update
  - solved first
  - otherwise sort by search_score
        |
        v
Stop?
  - if success_score == 1.0: solved
  - else next iteration / next island / next repair


================ Legacy two-stage path ================

Load .scratch
        |
        v
proof_runtime/evaluator_lean.py
        |
        +--> Stage 1: generate step files
        |      - definition.lean
        |      - step_1.lean
        |      - step_2.lean
        |
        +--> Stage 2: call Claude to fill sorry
        |
        +--> Stage 3: verify generated steps
        |
        v
Return legacy evaluator metrics
```

## 5. 当前推荐主路径的详细运行逻辑

以下描述以 `examples/lean_and_left/`、`examples/lean_identity_namespace/` 和 benchmark task 为准。

### 5.1 输入层

一个任务目录至少包含：

- `problem.lean`
- `init.scratch`
- `config.yaml`
- `evaluator.py`
- `run.sh`

它们的职责分别是：

- `problem.lean`
  - theorem scaffold
  - 定义最终 theorem 的上下文、imports、namespace、header
- `init.scratch`
  - 初始 proof plan
  - 不是完整 Lean 证明，而是结构化 plan
- `config.yaml`
  - rollout / island / llm / infra 配置
- `evaluator.py`
  - 把当前任务绑定到 plan-based Lean evaluator
- `run.sh`
  - 启动入口

### 5.2 初始 Program

系统读取 `init.scratch` 后，会把它作为初始候选：

- `Program.code`
  - 保存 plan YAML 文本
- `Program.meta["plan_object"]`
  - 保存解析后的 `PlanObject`

一个 `PlanObject` 当前至少会包含：

- theorem statement
- plan summary
- subgoal 列表
- 子目标依赖关系
- `proof_sketch`
- `local_lean_draft`
- frontier / blocked / metadata

### 5.3 Outer evolve：plan 级生成

外层每一轮先做：

- parent selection
- inspiration selection

然后 `PlanMutationGenerate` 根据以下信息构建 prompt：

- parent plan
- parent metrics
- frontier / blocked nodes
- judge feedback
- repair agenda
- parent experiences
- inspiration plans / experiences
- island specialization directive
- crossover directive

LLM 的输出不是最终 Lean 文件，而是新的 child plan：

- 怎么拆 subgoals
- subgoal 之间依赖关系是什么
- 哪些 lemma 候选值得引入
- 每个节点的局部 Lean draft 是什么

### 5.4 Inner evolve：fixed-plan subgoal repair

如果配置允许局部岛搜索，那么外层 plan mutation 之后，还会进入 fixed-plan 的 local repair。

它的特点是：

- 不再重做整个 plan 拓扑
- 只在当前 frontier subgoal 上继续修复
- 把 verifier 反馈聚焦到最值得修的局部节点

也就是说，当前系统的内层更像：

- `subgoal-solving agent`

而不是：

- 在每一步都重建整个证明计划

### 5.5 Lean 物化

一旦 child plan 生成，`PlanLeanEvaluator` 会把它物化为一组 Lean 文件。

典型输出：

- `definition.lean`
- `step_1.lean`, `step_2.lean`, ...
- `sg_xxx.lean`
- `lemma_xxx.lean`
- `final_theorem.lean`
- `artifact_manifest.json`
- `plan.yaml`

这些文件会先落在：

- `proof_runtime/project/temp_plan_eval/`

这里保存的是“当前这个 plan 被展开后的 Lean 产物”。

### 5.6 Lean bridge 校验

随后 `ProofRuntimeBridgeVerifier` 会把这些产物拷到 Lean 工程内部的桥接暂存区：

- `proof_runtime/project/temp/plan_bridge/`

并为每个待校验目标构造 wrapper：

- `_verify_sg_xxx.lean`
- `_verify_final_theorem.lean`

然后在共享 Lean 工程上下文里执行：

- `lake lean <wrapper file>`

返回的核心指标包括：

- `compile_success`
- `no_sorry`
- `statement_match`
- `proof_validity`
- `diagnostics`

### 5.7 分数计算

当前分数已经分成两层。

#### 成功分

- `success_score`
- 顶层暴露为 `Program.combined_score`

含义：

- `1.0` 表示 theorem 已真正证明成功
- `0.0` 表示 theorem 还没有成功证明

#### 搜索分

- `search_score`

用途：

- outer evolve 排序
- 选择 parent
- 比较未 solved 的 plan 优劣

此外还会计算更细的结构信号：

- `realization_score`
- `memory_gain_bonus`
- `plan_complexity_penalty`

### 5.8 Judge 与 Experience

Lean verifier 的结果不会直接结束在“通过/失败”。

后面还会进入：

- `PlanJudge`
- `proof_experience`
- `global_memory_store`

当前会提取和沉淀：

- 成功摘要
- 失败模式
- repair hints
- reusable lemmas
- theorem-level experience
- subgoal-level experience
- verified lemma library

这些信息下一轮会重新进入 mutation prompt，形成：

- verifier-guided next mutation
- memory-guided replanning

### 5.9 Stop 条件

当前题目真正的成功条件是：

- `success_score == 1.0`
  或
- `theorem_solved == 1.0`

只要命中这个条件：

- 该题就视为真正证明成功
- 之后不需要再继续做无意义演化

## 6. legacy 两阶段路径的逻辑

这条路径由 [proof_runtime/evaluator_lean.py](../../proof_runtime/evaluator_lean.py) 驱动。

它的运行逻辑是：

1. 从 `.scratch` 生成 `definition.lean + step_*.lean`
2. 做一次轻量 coherence check
3. 再让 Claude 去补全各 step 中的 `sorry`
4. 再做验证

它主要保留给：

- 旧 example
- 最小环境 smoke
- evaluator 兼容测试

这条路径和当前推荐主路径最大的区别是：

- 它更像“先拆 step，再补 step”
- 而不是“先维护 proof plan，再把 plan 物化成 Lean 工件”

## 7. benchmark 运行逻辑

当前 benchmark 主入口是：

- [famou/utils/mathproof_benchmark_runner.py](../../famou/utils/mathproof_benchmark_runner.py)

它的逻辑是：

1. 读取 benchmark 原始 Lean 文件
2. 枚举 theorem
3. 把 theorem 物化成当前 example 风格任务目录
4. 为每道题补 `config.yaml`、`evaluator.py`、`run.sh`
5. 顺序执行每道题
6. 汇总 suite 级结果

当前支持：

- `miniF2F-test`
- `PutnamBench`

### 7.1 miniF2F

主入口：

- [examples/minif2f_test_smoke/run.sh](../../examples/minif2f_test_smoke/run.sh)

### 7.2 PutnamBench

当前没有单独 example 目录，推荐直接调 benchmark runner CLI。

## 8. 运行时目录和产物含义

当前和 Lean 证明相关的几个重要目录如下。

### 8.1 `proof_runtime/project/temp_plan_eval/`

作用：

- plan evaluator 的 Lean 产物目录

里面放的是：

- `definition.lean`
- `step_*.lean`
- `sg_*.lean`
- `lemma_*.lean`
- `final_theorem.lean`
- `plan.yaml`
- `artifact_manifest.json`

这是“plan 被物化之后的中间产物”，不是源码目录。

### 8.2 `proof_runtime/project/temp/plan_bridge/`

作用：

- bridge verifier 的 staged 校验目录

里面放的是：

- 从 `temp_plan_eval/` 拷过来的产物
- `_verify_sg_xxx.lean`
- `_verify_final_theorem.lean`

这是“为了在 Lean project 里做真实校验而临时搭的桥接目录”。

### 8.3 `proof_runtime/project/temp/temp_*`

作用：

- legacy `evaluator_lean.py` 的临时目录

里面可能出现：

- `_generate_prompt.txt`
- `_construct_prompt.txt`
- `_verify_prompt.txt`
- `definition.lean`
- `step_*.lean`

### 8.4 `famou_data/` 或 benchmark `suite_data/`

作用：

- 正式实验输出目录

里面通常会有：

- `experiment.log`
- `experiment_checkpoint_*.json`
- `programs/*.json`
- `programs/*.lean`

这些更接近“实验结果”，通常比上面的 temp 目录更值得保留。

## 9. 当前已经验证过的状态

从当前仓库实际测试情况来看，下面这些事情已经被验证过。

### 9.1 单题 example 可运行

已经有可运行 example：

- `triangle_inequality`
- `lean_and_left`
- `lean_identity_namespace`

### 9.2 benchmark 链路可运行

已经真实跑通过：

- `miniF2F-test` pilot
- `PutnamBench` pilot

已确认：

- benchmark 物化正常
- evaluator 正常
- Lean bridge 校验正常
- suite 汇总正常

### 9.3 多岛 memory / crossover 机制已实际覆盖

已经完成过真实 smoke 和真实行为确认：

- memory-guided replanning
- plan crossover
- multi-island specialization

## 10. 当前代码库的现实边界

当前仓库已经是一个“能跑、能验证、能做 benchmark pilot”的系统，但仍然有一些现实边界需要明确。

### 10.1 已经解决的问题

- 代码库结构已经整理到可以直接运行
- plan-based mathproof 主路径已经贯通
- Lean project bridge 校验已经贯通
- benchmark runner 已接入并可执行
- 经验库、lemma 记忆、局部 repair、多岛 specialize 已接入主流程

### 10.2 还不是最终论文终态的问题

- 默认 budget 仍偏小
- benchmark pilot 已跑通，但默认配置下未必能快速收敛到 `success_score=1.0`
- temp 目录命名和运行时产物管理还可以进一步工程化
- legacy path 与 current path 仍并存，需要在后续进一步收口

## 11. 推荐阅读顺序

如果你想最快理解当前代码库，建议按下面顺序读。

1. [README.md](../../README.md)
2. [examples/lean_identity_namespace/problem.lean](../../examples/lean_identity_namespace/problem.lean)
3. [examples/lean_identity_namespace/init.scratch](../../examples/lean_identity_namespace/init.scratch)
4. [examples/lean_identity_namespace/evaluator.py](../../examples/lean_identity_namespace/evaluator.py)
5. [famou/utils/mathproof_example.py](../../famou/utils/mathproof_example.py)
6. [famou/utils/plan_lean_evaluator.py](../../famou/utils/plan_lean_evaluator.py)
7. [famou/utils/proof_runtime_bridge.py](../../famou/utils/proof_runtime_bridge.py)
8. [famou/modules/generate/plan_mutation.py](../../famou/modules/generate/plan_mutation.py)
9. [famou/modules/generate/subgoal_repair.py](../../famou/modules/generate/subgoal_repair.py)
10. [famou/utils/global_memory_store.py](../../famou/utils/global_memory_store.py)

