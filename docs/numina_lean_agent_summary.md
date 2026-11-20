# Numina-Lean-Agent 整理笔记

本文档基于仓库内的本地材料整理：

- `third_party/numina-lean-agent/README.md`
- `third_party/numina-lean-agent/tutorial/usage.md`
- `third_party/numina-lean-agent/scripts/run_claude.py`
- `third_party/numina-lean-agent/scripts/runner.py`
- `third_party/numina-lean-agent/scripts/lean_checker.py`
- `third_party/numina-lean-agent/prompts/docs/prompts/*.md`
- `third_party/numina-lean-agent/prompts/docs/technique/WORKFLOW_DIAGRAM.md`

说明：

- 仓库里带了 `NuminaLeanAgent.pdf`，但当前环境缺少 PDF 提取工具，因此本文主要依据仓库内 README、prompt 文档和执行脚本整理。
- 这份笔记的目标不是复述论文全文，而是帮助你快速判断这个系统的设计思想、执行方式，以及它和你当前 `Famou + Lean` 方案的关系。

## 1. 这项工作的定位

Numina-Lean-Agent 在仓库 README 里的定位非常明确：

- 它是一个 built on Claude Code 的 formal theorem proving agent
- 它针对的是 Lean theorem proving tasks
- 它强调的是 agentic reasoning system，而不是单纯的 tactic model 或 premise retriever

README 中给出的代表性结果：

- 证明了 Putnam 2025 的 12 个问题
- 完成了论文级 formalization：Effective Brascamp-Lieb inequalities

从方法风格上看，这个系统属于：

- 通用代码 agent 驱动
- 以 Lean MCP / Lean LSP 工具为主要环境接口
- 用多角色 agent 和 blueprint 文档组织复杂证明过程

它不是：

- 端到端训练出来的专用 theorem prover
- RL/MCTS 为主的搜索系统
- 单轮 prompt + 单文件补全工具

## 2. 核心思想

Numina-Lean-Agent 的关键思想不是“直接让一个大模型无结构地写 proof”，而是把证明过程拆成可管理的层级工作流。

从本地 prompt 文档和 workflow diagram 来看，它的核心结构是：

1. 用 `BLUEPRINT.md` 作为单一事实源，维护依赖拓扑、状态、尝试次数和证明说明
2. 用 Coordinator 做任务编排，而不是亲自证明
3. 把具体工作拆给不同子 agent
4. 让 Proof Agent 在 Lean 工具环境中多轮尝试、反复验证
5. 用 Lean 检查和 `sorry` 策略保证“随时可编译”

这套设计明显在解决两个问题：

- 长证明的上下文爆炸
- 大模型直接对复杂 Lean 代码端到端生成时稳定性不够

## 3. 系统架构

从 `coordinator.md`、`proof_agent.md`、`sketch_agent.md`、`blueprint_agent.md` 可以看出，它的角色分工很清楚。

### 3.1 Coordinator Agent

Coordinator 的职责是：

- 读取 `BLUEPRINT.md`
- 选择下一个可做的目标
- 判断目标复杂度
- 决定走哪条 agent 流程
- 更新 blueprint

它被明确禁止直接做 proof work。

这条规则的出发点是：

- proof 细节会迅速撑爆顶层 agent 的上下文
- orchestration 和 proving 分离更稳定

### 3.2 Blueprint Agent

Blueprint Agent 负责：

- 分析 lemma 是否过大、证明是否过于模糊
- 调用 Gemini 补详细 informal proof
- 必要时把复杂 lemma 拆成多个子 lemma
- 更新依赖拓扑

也就是说，它不是在写 Lean proof，而是在重写问题结构。

### 3.3 Sketch Agent

Sketch Agent 负责：

- 从 blueprint 中的 informal statement 生成 Lean statement
- 在目标文件中插入状态注释
- 留下 `sorry`
- 更新 blueprint 中的 file:line 信息

它只做“从 informal 到 formal statement 的结构映射”，不负责真正证明。

### 3.4 Proof Agent

Proof Agent 是真正做 Lean 证明的 agent。

它的职责包括：

- 在临时文件里尝试证明
- 必须优先查库和找定理
- 必须先咨询 Gemini 讨论策略
- 系统化尝试多个证明类别
- 保证退出时 Lean 代码可编译
- 如果做不完，只允许保留最小范围的 `sorry`

这说明它的目标不是一次生成最终答案，而是做深度、分阶段、带回退的证明探索。

## 4. 数据组织：Blueprint 驱动

Numina-Lean-Agent 的很大一个特色是 `BLUEPRINT.md`。

这不是简单的 TODO 列表，而是一个依赖图和项目状态文件。每个条目通常包含：

- label
- uses 依赖
- file
- status
- attempts
- informal statement
- informal proof

这相当于把一个复杂证明项目拆成很多可追踪的节点，并用依赖顺序来安排求解顺序。

这套设计有几个直接好处：

- 顶层 agent 不需要一次装下整份证明
- proof work 可以按依赖拓扑逐步推进
- 失败可以定位到具体 lemma，而不是整题失败
- 可以明确区分“需要 formalize”与“需要 prove”的工作

## 5. 执行流程

根据 `WORKFLOW_DIAGRAM.md` 和各 agent prompt，Numina-Lean-Agent 的标准流程大致如下。

### 5.1 Session 开始

1. 读 `BLUEPRINT.md`
2. 选择一个 dependencies 已满足的 TODO 或 partial 节点
3. 判断复杂度

复杂度分类：

- Simple：已经 formalized，直接证明
- Medium：先 formalize 再证明
- Complex：先重构 blueprint，再 formalize，再证明

### 5.2 三种主要工作流

#### Pattern A: Simple

Coordinator -> Proof Agent

适用于：

- statement 已经 formalized
- 目标比较清楚

#### Pattern B: Medium

Coordinator -> Sketch Agent -> Proof Agent

适用于：

- 有 informal statement
- 但还没进入 Lean 代码

#### Pattern C: Complex

Coordinator -> Blueprint Agent -> Sketch Agent -> Proof Agent

适用于：

- informal proof 很弱
- lemma 太大
- 之前 proof attempt 已经耗尽预算

这三条路径说明系统把“证明问题”拆成了结构化工作流，而不是统一塞给一个 agent。

## 6. Proof Agent 的方法论

`proof_agent.md` 是最重要的材料之一，它反映了 Numina 的 proving philosophy。

### 6.1 先保证可编译，再追求完全证明

它明确要求：

- 所有代码退出时必须编译
- 禁止使用 `axiom`
- 如果无法完成，只保留最小 stuck part 为 `sorry`

这是一种很工程化的策略：

- 保证 proof search 过程中的中间状态始终可检查
- 防止因为局部错误让整个文件坏掉

### 6.2 强制 Gemini 先给证明思路

Proof Agent 被要求在写任何 Lean 代码前，先咨询 Gemini：

- 高层证明策略
- 数学 insight
- 可能用到的 mathlib lemma
- 可能的坑

这说明它把“数学启发式”与“Lean 形式化动作”分离了：

- Gemini 更像数学讨论伙伴
- Claude Code 更像 Lean 工程执行者

### 6.3 明确反对 brute-force enumeration

它特别反对：

- `decide`
- `fin_cases`
- `interval_cases`
- 大量枚举式 case split

偏好：

- general proof
- induction
- 结构化 decomposition
- 利用已有 mathlib theorem

这说明系统追求的不是只把 benchmark 做出来，而是更可扩展、可复用的证明策略。

### 6.4 把证明过程分成五类尝试

文档里把尝试方法划成五类：

1. library search
2. direct tactics
3. structural approaches
4. term mode
5. decomposition

这相当于把 proof search 明确过程化了。不是“想到什么试什么”，而是强制覆盖多个搜索方向。

## 7. 运行方式

从 `tutorial/usage.md`、`scripts/run_claude.py`、`scripts/runner.py` 看，它的执行器本质是一个 Claude CLI 包装器。

入口支持三种模式：

- `run`：单任务
- `batch`：批量配置驱动
- `from-folder`：对一个目录里的所有 `.lean` 文件跑任务

命令入口：

```bash
python -m scripts.run_claude run <target> [options]
python -m scripts.run_claude batch <config_file> [options]
python -m scripts.run_claude from-folder <folder> [options]
```

这不是 Python SDK 风格，而是典型的：

- Python 负责 orchestration
- Claude CLI 负责执行 agent
- Lean checker 负责本地检查

## 8. Claude 会话控制方式

`runner.py` 展示了它的核心执行循环。

### 8.1 Claude 的调用方式

它实际是通过 subprocess 调用 `claude`：

- 新会话：`claude -p --verbose -d ... <prompt>`
- 继续会话：`claude -c -p continue`

也就是说：

- 它不是走 Python SDK
- 它依赖 Claude Code CLI
- 通过 END_REASON 控制多轮继续

### 8.2 多轮会话控制

它会识别模型输出中的：

- `END_REASON: LIMIT`
- `END_REASON: COMPLETE`
- `END_REASON: SELECTED_TARGET_COMPLETE`

然后决定：

- 是继续同一个 session
- 还是重置 session
- 还是执行验证回调

这点很重要，因为这说明系统不是单次调用，而是多轮 agent session。

### 8.3 文件级跟踪

执行器还会记录：

- statement tracker
- line count changes
- 每轮 round result
- MCP log stats

这让它具备比较强的实验分析能力。

## 9. Lean 验证机制

Numina 的“最终成功”不是纯文本判断，而是 Lean 检查。

`lean_checker.py` 的核心策略是：

- 找到 Lean project root
- 执行 `lake env lean <file>`
- 检查是否有 error
- 检查是否有 sorry warning

因此它的基本成功标准是：

- 没有编译错误
- 没有 sorry 警告

如果 `allow_sorry=False`，那么存在 `sorry` 也会被视为失败。

这比“只看 agent 说自己完成了没”要可靠得多。

## 10. MCP 和 Lean 工具的角色

Numina 不是只靠模型内推理。它大量依赖 Lean MCP 工具：

- search 工具
- hover/info
- local search
- goal / diagnostics
- build/check

从 prompt 的要求来看，MCP 工具相当于：

- 让 agent 能访问 Lean 当前语境
- 让 agent 能做 theorem search
- 让 agent 能用 proof assistant feedback 纠错

这类设计的本质是“tool-augmented theorem proving”，不是裸语言模型证明。

## 11. 这个系统的强项

基于本地材料，我认为 Numina-Lean-Agent 的强项主要有这些。

### 11.1 工程结构清晰

它把复杂证明任务拆成：

- 任务结构规划
- statement formalization
- proof search
- verification

这比单 agent 直冲更稳定。

### 11.2 适合长链条项目

`BLUEPRINT.md` 的依赖图特别适合大证明项目，因为：

- 能局部推进
- 能从 partial 状态恢复
- 能显式记录 attempts 和 blocker

### 11.3 充分利用通用 coding agent

它没有绑定专用 theorem prover model，而是直接把 Claude Code 作为 general agent，用 Lean 工具来补环境信息。

这使它：

- 更灵活
- 更容易接入新的 agent 平台
- 更接近“agent systems for formal math”而非“单模型证明器”

### 11.4 验证闭环做得比较扎实

和很多只看 pass@k 的生成式系统不同，Numina 明确把 Lean check 放进执行闭环中。

## 12. 这个系统的弱项和代价

### 12.1 系统复杂

它不是一个轻量方法。要跑起来，需要：

- Lean project
- Claude Code CLI
- MCP server
- blueprint 文件组织
- 多 agent prompt 体系

门槛明显高于单模型生成。

### 12.2 对 prompt engineering 和流程纪律依赖很强

大量行为不是写在算法里，而是写在 prompt 规则里：

- 先问 Gemini
- 不准 coordinator 自己证明
- 必须更新 blueprint
- 必须试五类方法

这类系统的稳定性部分取决于 agent 是否遵守流程。

### 12.3 不是显式的 population-based evolution

这是和你当前思路最重要的差别之一。

Numina：

- 更像 hierarchical agent orchestration
- 重点是 decomposition + tool use + multi-round proving

而不是：

- 同时维护一批 candidate proofs
- 对 candidate 做 selection / mutation / archive
- 用 evaluator 作为显式 fitness 做演化搜索

## 13. 它和你当前方案的关系

你现在的方向可以概括成：

- `agent + evolve`
- 维护 proof / Lean candidate 种群
- 用 formal checker 做 fitness
- 类似 AlphaEvolve 的迭代优化

Numina 和你的共同点：

- 都不是单次生成
- 都依赖 Lean 工具反馈
- 都把证明过程当成可迭代优化的过程
- 都承认长证明需要中间结构，而不是端到端一次写完

Numina 和你的不同点：

- Numina 维护的是 workflow state，不是 candidate population
- Numina 更偏“任务分解和逐 lemma 推进”
- 你的方向更偏“搜索多个候选证明/候选大纲并演化”

可以粗略理解为：

- Numina 更像：`Planner + Worker + Verifier`
- 你的方向更像：`Population Search + Formal Fitness`

这也意味着，Numina 对你最有价值的部分不一定是它的整体范式，而是它的两个局部设计：

1. `BLUEPRINT` 作为长期状态与依赖图
2. `Proof Agent + Lean tools + compile/no-sorry verification` 的内循环

你完全可以把这两部分塞进一个演化框架里。

## 14. 对你当前项目最值得借鉴的点

如果目标是把 `Famou + Lean` 做成更强的数学证明 agent/evolve 框架，我认为 Numina 最值得借鉴的是：

### 14.1 把“证明对象”拆层

可以区分：

- informal proof sketch
- sub-lemma blueprint
- Lean statement skeleton
- complete Lean proof

这样演化时不一定只演化最终 `.lean`，还可以演化：

- proof outline
- lemma decomposition
- auxiliary lemma sets

### 14.2 引入 Blueprint 作为结构化中间层

你现在的 `.scratch` 已经有点像 proof outline 了，但还缺一层显式依赖图。

如果引入 Numina 风格的 blueprint，你的演化对象可以不只是一段文本，而是：

- 目标 theorem
- proof plan
- 子目标 DAG
- 每个子目标的状态和 best candidate

这会让 evolve 更可控。

### 14.3 更严格的验证环节

Numina 的本地 Lean check 比你当前 evaluator 里“主要查有没有 sorry”的逻辑更扎实。

如果你后面继续做 Lean evaluator，至少应该吸收它的两个标准：

- `lake env lean` / diagnostics 无 error
- 无 sorry warning

### 14.4 多轮 session 控制

它的 `LIMIT -> continue -> verify -> retry` 这套会话控制值得借鉴，特别适合长证明问题。

## 15. 一句话评价

Numina-Lean-Agent 不是“proof population 演化系统”，而是一个高度工程化的、以 Claude Code 和 Lean MCP 为核心的层级式 formal math agent 框架。

它最突出的价值在于：

- 不是把 Lean 证明看成一次性生成任务
- 而是把它看成一个需要 decomposition、tool use、state tracking、verification 的长期 agent 过程

对你当前项目来说，它最像的是：

- 你未来系统里的 proving 内循环和任务状态层

而不是：

- 你整个 AlphaEvolve 风格种群搜索层

## 16. 我建议你怎么用这份材料

如果你后面要写方案或论文 related work，可以把 Numina 放在：

- `agentic formal theorem proving system`
- `tool-augmented Lean proving`
- `hierarchical orchestration for long-horizon proofs`

而把你自己的方案放在：

- `population-based agentic proof search`
- `AlphaEvolve-style formal proof evolution`
- `formal verifier as fitness over proof/plan populations`

这样区分会比较清楚。


