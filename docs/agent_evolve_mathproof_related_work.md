# Agent + Evolve + Math Proof 相关工作整理

本文聚焦与我们计划中的方向最接近的一组工作：`agent + evolve + formal math proof`。我们目前在规划中的框架，其核心假设是：

- 用一个通用 agent 负责题目理解、证明策略拆解、辅助引理构造、Lean 代码生成与修复。
- 用 Lean verifier 作为强约束评估器，不只判断最终 proof 是否通过，也评估中间步骤、子目标收缩、是否存在 `sorry`、辅助引理是否可复用。
- 在外层维护一个持续演化的种群，而不是只做单次采样。种群中的个体可以是 proof sketch、step-by-step plan、auxiliary lemmas、partial Lean drafts、proof traces。
- 通过 selection / mutation / recombination / archive / curriculum 让系统在题内和题间持续改进。

这里的几类个体分别指：`proof sketch` 是粗粒度证明思路，`step-by-step plan` 是更细的分步求证计划，`auxiliary lemmas` 是为主定理服务的中间引理集合，`partial Lean drafts` 是尚未完全通过校验的 Lean 证明草稿，`proof traces` 是完整记录尝试、报错、修复与推进过程的证明轨迹。

从这个角度看，现有工作大致分成四类：

- `formal theorem proving agent`：强调多模型、多工具编排。
- `verifier-guided self-improvement`：强调 Lean/compiler 反馈、RL、自纠错、搜索。
- `self-play / curriculum`：强调自动生成新题或新猜想来持续变强。
- `evolutionary search`：强调 evaluator-guided population search，但通常不在 formal theorem proving 场景中。

结论先行：截至 `2026-04-10`，结合我对近期论文与公开项目的补充检索，我仍然没有看到一篇工作把这四条线完整地统一成“Lean-verifier-grounded evolutionary agent for formal theorem proving”。目前最接近的仍然是 `STP + AlphaProof + AlphaEvolve` 的交叉区域，以及 `Prover Agent / Numina-Lean-Agent / Goedel-Prover-V2 / EvolProver` 这些分别覆盖 agent、verifier-guided repair、formal RL、problem evolution 中若干子方向的系统，但它们依然没有把 `Lean-grounded verifier`、`显式 proof-object population evolution`、`agentic orchestration`、`跨题经验记忆` 四点同时合到一个统一框架里。

## 1. 最相关工作总览

| 工作　　　　　　　 | 时间　　　| 与我们方向的相关性 | 一句话判断　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　|
| --------------------| -----------| --------------------| -------------------------------------------------------------------------------------|
| STP　　　　　　　　| 2025　　　| 很高　　　　　　　 | 最接近“自生成任务 + 持续改进”，但更偏 self-play training，不是 population evolution |
| AlphaProof　　　　 | 2025　　　| 很高　　　　　　　 | 最强的 RL + Lean + test-time adaptation 路线，但不是 evolve 框架　　　　　　　　　　|
| AlphaEvolve　　　　| 2025　　　| 很高　　　　　　　 | 方法论最像我们，核心是 evaluator-guided evolution，但任务不是 formal proof　　　　　|
| Prover Agent　　　 | 2025/2026 | 很高　　　　　　　 | 最像“agent orchestration + auxiliary lemma generation”的 formal math 系统　　　　　 |
| Numina-Lean-Agent　| 2026　　　| 很高　　　　　　　 | 最像“通用 coding agent + MCP + Lean”的开放式 formal math agent　　　　　　　　　　　|
| Goedel-Prover-V2　 | 2025　　　| 高　　　　　　　　 | verifier-guided self-correction 很强，但主体仍是 prover training pipeline　　　　　 |
| DeepSeek-Prover-V2 | 2025　　　| 高　　　　　　　　 | 子目标分解 + RL 很关键，接近“先拆再证”的路线　　　　　　　　　　　　　　　　　　　　|
| LeanAgent　　　　　| 2024/2025 | 中高　　　　　　　 | 跨仓库持续学习很强，接近“长期积累”，但不是 inference-time population　　　　　　　　|
| Lean-STaR　　　　　| 2024/2025 | 中　　　　　　　　 | 把 thought 和 proving 交织起来，适合借鉴“个体表示”设计　　　　　　　　　　　　　　　|
| EvolProver　　　　 | 2025　　　| 中高　　　　　　　 | 显式使用 evolve，但对象是 formalized problems/data，而不是 proof population　　　　 |
| Intern-S1-MO　　　 | 2025　　　| 中　　　　　　　　 | 很强的长时程数学推理 agent，但主要是自然语言奥赛解题，不是 Lean formal proof　　　　|

## 2. 逐篇整理

### 2.1 STP: Self-play LLM Theorem Provers with Iterative Conjecturing and Proving

- 时间与来源
  - ICML 2025 / PMLR
  - 论文页: https://proceedings.mlr.press/v267/dong25h.html
  - arXiv: https://arxiv.org/abs/2502.00212

- 核心思想
  - 不再只让 prover 被动证明已有题目，而是让系统同时扮演 `conjecturer` 和 `prover`。
  - conjecturer 生成“当前 prover 勉强可证”的新猜想；prover 再对这些新题做 expert iteration。
  - 这样系统能自动制造新的训练信号，缓解 formal proof 数据稀缺和 reward 稀疏问题。

- 创新点
  - 把 formal theorem proving 从“只做 proof search”扩展为“边出题边解题”的 self-play 过程。
  - conjecturer 的目标不是任意生成题，而是围绕当前 prover 能力边界生成任务，形成难度递进。
  - 在 Lean 和 Isabelle 上都验证了这种机制。

- 与我们的不同
  - STP 演化的是 `problem distribution / conjecture distribution`，不是 `proof-plan / Lean draft / lemma population`。
  - 它主要是训练时自博弈，不是 inference-time 的 evolutionary loop。
  - 它更像“让训练集自己长出来”，而我们更像“让解题过程里的候选证明个体自己长出来”。

- 对我们的启发
  - 如果我们后面做 evolve，不应只演化 proof，也可以演化“题的邻域”或“辅助子命题”。
  - “刚好可证”的 curriculum 信号值得直接借鉴到 proof population 的 selection 上。

### 2.2 Olympiad-level formal mathematical reasoning with reinforcement learning (AlphaProof)

- 时间与来源
  - Nature, published `2025-11-12`
  - 论文页: https://www.nature.com/articles/s41586-025-09833-y

- 核心思想
  - 把 Lean 看成可交互环境，用 AlphaZero 风格 RL 学 formal proof。
  - 通过大规模 autoformalized problems 训练主 agent。
  - 对最难问题，引入 `test-time RL`：围绕目标题生成大量相关变体，在推理时继续学，做 problem-specific adaptation。

- 创新点
  - 把 RL 真正落到高难 formal mathematics，而不是停留在小型 benchmark。
  - 提出了 test-time RL，使系统可以围绕单题做深度适配。
  - 在 2024 IMO 上达到银牌等效表现，说明 formal verifier + 大规模学习 + test-time adaptation 的上限很高。

- 与我们的不同
  - AlphaProof 的核心优化器是 RL，不是 evolutionary population search。
  - 它的 compute 和训练规模非常大，难以直接复现；而我们的路线更适合做“可复现的 agentic system + evolve loop”。
  - 它把 adaptation 放在问题级别；我们更可能把 adaptation 落在 `proof states / lemmas / plans / drafts` 级别。

- 对我们的启发
  - 题内持续改进是有效的，而且值得作为核心卖点。
  - “variant generation around the target problem” 可以被 reinterpret 成演化中的 mutation operators。

### 2.3 AlphaEvolve: A coding agent for scientific and algorithmic discovery

- 时间与来源
  - arXiv, submitted `2025-06-16`
  - arXiv: https://arxiv.org/abs/2506.13131

- 核心思想
  - 用 LLM 提出代码修改，再由 evaluator 自动打分，然后通过 evolutionary loop 迭代提升程序。
  - 关键不是“一次性写出最佳代码”，而是“持续提出候选、自动评估、保留优胜个体、继续搜索”。

- 创新点
  - 把 coding agent 和 evolutionary search 真正结合起来，并通过可执行 evaluator 保证改进有客观依据。
  - 在科学发现、算法优化、基础设施优化等多个任务上证明 evaluator-guided evolution 的泛化性。

- 与我们的不同
  - 它不是 theorem proving 工作，个体是程序，evaluator 是程序指标或 correctness check。
  - 我们要把这套机制迁移到 formal math，个体要从“代码程序”扩展成“proof plans / Lean proofs / auxiliary lemmas / proof-state edits”。
  - 我们的 evaluator 不应是单一 pass/fail，而应包含更多 Lean-specific signal，例如 goal reduction、step validity、sorry-free、lemma reuse 等。

- 对我们的启发
  - 这是我们方法论上最关键的参照系。
  - 如果后续写 NeurIPS，最自然的 framing 不是“another theorem prover”，而是“formal theorem proving as evaluator-grounded evolutionary search over proof objects”。

### 2.4 Prover Agent: An Agent-Based Framework for Formal Mathematical Proofs

- 时间与来源
  - arXiv, latest version `2026-02-17`
  - arXiv: https://arxiv.org/abs/2506.19923

- 核心思想
  - 用 agent 框架把 informal reasoning LLM、formal prover model 和 Lean feedback 组织在一起。
  - 系统不只试图直接证明定理，还会主动生成 auxiliary lemmas，包括特例、从假设推出的有用事实等。
  - 这些中间 lemma 帮助系统找到更可行的整体证明路线。

- 创新点
  - 显式把 `auxiliary lemma generation` 作为核心推理单元，而不是附属技巧。
  - 强调 informal reasoning 与 Lean feedback 的互补。
  - 以较小 sample budget 达到很强的 formal benchmark 成绩。

- 与我们的不同
  - Prover Agent 仍是单题上的 agent orchestration，不是 population-based evolve framework。
  - 它生成 lemma，但没有把 lemma / proof sketch / draft 看成一个会跨轮竞争与繁殖的种群。
  - 我们更进一步的点可以是：不仅生成中间 lemma，而且把这些 lemma 作为可复用个体进入 archive，并参与后续变异与重组。

- 对我们的启发
  - 辅助引理不是锦上添花，而应当成为系统的一级对象。
  - “证明失败但产生了高价值 lemma” 在我们的框架里也应被视为高适应度个体，而不是简单丢弃。

- 整体框架流程图

```text
Target Theorem in Lean
        |
        v
Agent Controller
  - 读取当前 theorem
  - 维护 sample budget
  - 决定先直接证明，还是先造辅助引理
        |
        +------------------------------+
        |                              |
        | Direct proof route           | Auxiliary-lemma route
        v                              v
Informal Reasoning LLM          Informal Reasoning LLM
  - 分析题意                       - 提出中间命题 / 特例 / 从假设推出的有用事实
  - 给出证明思路                   - 判断哪些 lemma 值得先证明
        |                              |
        +--------------+---------------+
                       |
                       v
                Formalizer (when needed)
                 - 把非形式化 lemma / step
                   转成 Lean theorem statement
                       |
                       v
                Formal Prover Model
                 - 证明主定理
                 - 或先证明辅助 lemma
                 - 生成 Lean proof candidate
                       |
                       v
                Lean Checker / Verifier
                 - 编译检查
                 - 返回报错 / 失败位置
                 - 确认哪些 lemma 已成功
                       |
         +-------------+-------------+
         |                           |
         | verified                  | failed / incomplete
         v                           v
   Reuse verified lemmas      Feed Lean feedback back to agent
   in main theorem proof       - 调整思路
                                - 更换 lemma
                                - 修正 formalization / proof
                                        |
                                        v
                               Next attempt under sample budget
```

- 流程说明
  - 根据论文摘要和官方 README，`Prover Agent` 的顶层不是单一 prover，而是一个 controller，把 `informal reasoning LLM`、`formal prover model`、`formalizer` 和 `Lean feedback` 串成一个 agent loop。
  - 它有两条典型路线：一条是直接让 prover 证明主定理；另一条是先生成并求证 `auxiliary lemmas`，再把已验证的 lemma 回灌到主证明里。
  - 这里的辅助引理并不只等于当前 Lean proof state 里的子目标，也可以是特例、显然有帮助的中间命题，或者从原假设中额外抽出的可用事实。
  - 当某个辅助结论还是非形式化描述时，系统会经过 `formalizer` 把它转成 Lean theorem statement；之后再交给 formal prover 真正去证。
  - Lean checker 的作用不是只在最后判一次对错，而是在每轮 candidate 之后提供强反馈，告诉系统当前证明是否编译通过、哪一步失败、哪些 lemma 已经可用。
  - 整个过程受 `sample budget` 约束，因此它本质上是 budgeted agent search，而不是无限制地修到成功为止。

- 关键细节补充
  - `informal reasoning LLM` 做的不是直接输出 Lean code，而是负责更高层的策略判断，例如：主定理难点在哪、哪些中间命题值得先证明、当前失败更像 statement 问题还是 proof 搜索问题。
  - `auxiliary lemmas` 是这篇工作的关键对象。论文明确强调这些 lemma 不局限于“主 proof 的局部子目标”，还可以是特殊情形、桥接步骤、或者从条件中导出的有用结论。
  - `formalizer` 的角色是把自然语言层面的中间想法压到 Lean 语句层。官方 README 还专门提到，这个模块在整个 workflow 中调用频率不高，因此可以和其他模型分开调度，以减少算力占用。
  - `formal prover model` 则是真正负责输出 Lean proof candidate 的模型，可以作用在主定理上，也可以作用在某个辅助 lemma 上。
  - `Lean feedback` 是系统闭环的核心。这里的反馈主要是编译/检查结果，而不是自然语言自评；因此系统的修正有很强的 verifier grounding。
  - 从方法学上看，`Prover Agent` 更像“lemma-centric single-problem agent orchestration”。
  - 从你们的视角看，它最值得借鉴的是：把“失败的主证明过程中产生的高价值 lemma”当成一等公民，而不是只盯最后那条完整 proof。

### 2.5 Numina-Lean-Agent: An Open and General Agentic Reasoning System for Formal Mathematics

- 时间与来源
  - arXiv, submitted `2026-01-20`
  - arXiv: https://arxiv.org/abs/2601.14027

- 核心思想
  - 直接把通用 coding agent 当作 formal math reasoner，而不是先训练一个专用 theorem prover。
  - 通过 MCP 让 agent 自主调用 Lean、定理检索、informal proving 和辅助工具。
  - 依靠底层通用模型能力和工具调用能力，而不是 task-specific pipeline。

- 创新点
  - 提出了“general coding agent can directly do formal mathematics”的范式。
  - 强调无训练、可替换底模、MCP 可扩展，复现成本更低。
  - 证明了通用 agent 路线在 formal math 上不是只能做玩具问题。

- 与我们的不同
  - Numina 的重点是 `agentic orchestration + tool use`，不是 `evolutionary optimization over proof populations`。
  - 它强调开放、通用、可替换模型；我们更强调如何让 verifier 信号驱动种群持续优化。
  - 它比较像“强 agent + 强工具”，我们则要做“强 agent + 强工具 + 长时程 evolutionary memory/search”。

- 对我们的启发
  - 通用 coding agent + Lean MCP 这条 implementation 路线是现实可行的。
  - 我们可以站在这类系统之上加一层 evolve controller，而不是从零发明 proof agent。

- 整体框架流程图

```text
Lean Project / Target Theorem
        |
        v
Coordinator reads BLUEPRINT.md
  - 读取依赖拓扑
  - 选择下一个可做节点
  - 判断目标复杂度
        |
        +-------------------------------+
        |                               |
        | Simple                        | Medium / Complex
        v                               v
   Proof Agent                 Sketch Agent / Blueprint Agent
   直接进入证明                  - 重写问题结构
                                 - 拆分复杂 lemma
                                 - 从 informal statement 生成 Lean statement
                                 - 更新 BLUEPRINT.md
        |                               |
        +---------------+---------------+
                        |
                        v
                 Proof Agent in Lean loop
                  - 先咨询 Gemini 获取证明策略
                  - 搜索 Mathlib / theorem / premise
                  - 在临时文件中尝试 Lean 证明
                  - 多轮修复与继续
                        |
                        v
                 Lean MCP / lean checker
                  - diagnostics
                  - goal state
                  - theorem search
                  - lake env lean
                        |
                        +--> proven
                        |
                        +--> partial / sorry
                        |
                        +--> budget exhausted
                        |
                        v
                 Update BLUEPRINT.md
                  - 标记状态
                  - 记录尝试次数
                  - 写回 file:line 信息
                        |
                        v
                 Coordinator selects next target
```

- 流程说明
  - Numina 的顶层不是直接证明，而是先由 `Coordinator` 依据 `BLUEPRINT.md` 做任务编排。
  - 当问题还没 formalize 好时，会先经过 `Blueprint Agent` 或 `Sketch Agent`，把问题改写成更容易推进的 Lean 子任务。
  - 真正的证明工作由 `Proof Agent` 完成，并且它不是一次性写完，而是在 Lean 工具环境中反复查库、查看 goal、修改代码、继续尝试。
  - 每轮结果都会回写到 `BLUEPRINT.md`，因此系统的“记忆”主要不是 population，而是 blueprint 驱动的项目状态。

- 关键细节补充
  - `BLUEPRINT.md` 大致包含：每个待证明节点的 `label`、`uses` 依赖、目标所在 `file`、当前 `status`、已尝试次数 `attempts`、`informal statement`、`informal proof`。它不是普通 TODO，而是一个证明项目的状态总表。
  - `读取依赖拓扑` 的意思是：Coordinator 先看每个 lemma / theorem 依赖哪些前置节点，再优先选那些依赖已经满足的节点。这个拓扑主要不是自动从 Lean AST 完整抽出来的，而是由 blueprint 中的 `uses` 字段和后续 agent 更新共同维护出来的。
  - `搜索 Mathlib / theorem / premise` 这一步其实包含三类不同查询：
    - `Mathlib search`：在整个 Mathlib 里找可能相关的现成定理、定义、结构和实例，目标是减少重复造轮子。
    - `theorem search`：更偏“我要找一个名字、结论形式或语义接近的定理”，例如搜索某个结论形状或某个关键词。
    - `premise search`：更偏“在当前 proof state 下，哪些已有前提/库定理最可能直接可用”，它和当前目标状态绑定得更强。
  - 这一步本质上是在做：把非形式化证明思路映射到 Lean / Mathlib 里真实存在的可调用对象，避免模型凭空编造 theorem 名称。
  - `多轮修复与继续` 的意思不是简单重复调用，而是基于上一轮 Lean 反馈继续改。主要修复依据包括：
    - Lean 报错信息
    - 当前 goal state
    - 搜到的新 theorem / premise
    - 上一轮失败暴露出的 tactic 或 statement 问题
    - blueprint 中记录的 blocker 和尝试历史
  - 常见修复动作包括：改 import、换 theorem、补中间 lemma、调整 statement、改 tactic 顺序、把一个过大的步骤再拆开。
  - `diagnostics`：文件级诊断信息，主要是 Lean 编译/类型检查产生的错误、warning、info，用来定位“哪里不对”。
  - `goal state`：当前证明位置下还剩下什么目标没证、上下文里有哪些假设，是最直接的 proof state 视图。
  - `theorem search`：根据关键词、结论形状、语义相似度或当前状态，去找可能能用的现成定理。
  - `lake env lean`：在目标 Lean project 的环境里真正运行 Lean 检查某个文件，这一步更接近最终裁判，回答“这个文件在当前工程里到底能不能过”。

### 2.5.1 Prover Agent 与 Numina-Lean-Agent 的并排对比

- `workflow` 差异
  - `Prover Agent` 更像一个围绕单个定理展开的 `lemma-centric proving loop`。
  - 它的核心流程是：分析主定理 -> 生成辅助引理 -> 必要时 formalize -> 用 prover 证明 -> 用 Lean 反馈继续修。
  - `Numina-Lean-Agent` 更像一个 `project-style orchestration loop`。
  - 它先由 `Coordinator` 读 `BLUEPRINT.md`，再按依赖拓扑派发节点，把复杂问题拆成一个个文件级或 lemma 级任务持续推进。

- `工具 / verifier` 差异
  - `Prover Agent` 的核心闭环是 `informal reasoning LLM + formalizer + formal prover + Lean checker`。
  - 它更强调模型角色分工，以及 Lean 编译反馈如何帮助主定理与辅助引理之间来回切换。
  - `Numina-Lean-Agent` 的核心闭环是 `general coding agent + MCP tools + Lean project runtime`。
  - 它更强调通用 agent 如何自主调用 `goal state`、`theorem search`、`premise search`、`lake env lean` 这类工具。

- `lemma` 使用方式差异
  - `Prover Agent` 把 `auxiliary lemmas` 放在方法正中心。
  - 这些 lemma 不只是“当前子目标”，而是为了找到可行证明路线而主动发明的中间命题，因此它的方法学更像“先长出桥，再过河”。
  - `Numina-Lean-Agent` 也会拆 lemma，但其重点不在“lemma generation 本身”，而在 blueprint 驱动的任务管理与工具交互。
  - 换句话说，`Prover Agent` 更像 `lemma-first`，`Numina` 更像 `workflow-first`。

- `状态管理` 差异
  - `Prover Agent` 的状态更集中在当前题目的尝试链路里：当前 theorem、当前辅助 lemma、当前 Lean 反馈、当前 sample budget。
  - `Numina-Lean-Agent` 的状态则被显式外化到 `BLUEPRINT.md`，包括依赖关系、状态、尝试次数、file:line 和 informal 描述。
  - 因此前者更像“单题内的推理闭环”，后者更像“可持续推进的 formalization project manager”。

- `方法定位` 差异
  - `Prover Agent` 的强项是：在固定 sample budget 下，用辅助引理把难题转化成更可证的若干中间问题。
  - `Numina-Lean-Agent` 的强项是：把通用 coding agent 真正落到 formal math 工程环境里，并展示多工具、多文件、多节点协同推进的可行性。

- 和我们方法的关系
  - 如果从你们的视角看，`Prover Agent` 更像是内层 `subgoal / lemma solving agent` 的重要参照。
  - `Numina-Lean-Agent` 更像是外层 `agent runtime / tool orchestration substrate` 的重要参照。
  - 你们的方法如果继续往前走，最自然的组合方式其实是：
    - 吸收 `Prover Agent` 的 `auxiliary-lemma-centric solving`
    - 吸收 `Numina` 的 `tool-rich Lean workflow`
    - 再在外层加上你们自己的 `proof-object population evolution + cross-problem experience memory`
  - 这也是为什么我前面一直说：现有工作已经分别覆盖了若干关键部件，但还没有把这三层真正合成一个统一系统。

### 2.6 Goedel-Prover-V2: Scaling Formal Theorem Proving with Scaffolded Data Synthesis and Self-Correction

- 时间与来源
  - arXiv, submitted `2025-08-05`
  - arXiv: https://arxiv.org/abs/2508.03613

- 核心思想
  - 以 expert iteration + RL 为主干，加入 `scaffolded data synthesis`、`verifier-guided self-correction` 和 `model averaging`。
  - 系统会利用 Lean compiler 的反馈对 proof 进行迭代修正。
  - 同时通过分层构造的 synthetic tasks 逐步提升模型对高难定理的掌握能力。

- 创新点
  - 自纠错明确依赖 verifier 反馈，而不是仅靠自然语言反思。
  - 数据合成不是简单扩充，而是难度有梯度的 scaffolded synthesis。
  - 在开源 prover 中表现很强。

- 与我们的不同
  - 它还是以训练更强 prover 为主，不是围绕每道题维护 evolving population。
  - 自纠错是单轨迹迭代修正；我们更可能做多轨迹并行竞争与重组。
  - 它的“知识增长”主要进入模型参数；我们的部分“知识增长”可能进入显式 archive，例如 reusable lemmas、successful rewrites、failure patterns。

- 对我们的启发
  - Lean compiler feedback 不只可用于 binary verification，也可作为 revision policy 的直接监督信号。
  - 显式 self-correction 很值得纳入 evolve loop 作为 mutation operator。

### 2.7 DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via Reinforcement Learning for Subgoal Decomposition

- 时间与来源
  - arXiv, latest version `2025-07-18`
  - arXiv: https://arxiv.org/abs/2504.21801

- 核心思想
  - 先用强模型把复杂问题分解成一系列 subgoals，再将解出的 subgoal proofs 合成为训练信号。
  - 把 informal step-by-step reasoning 与 formal proof synthesis 融合到统一模型中。
  - 然后在这个基础上继续做 RL。

- 创新点
  - 把 `subgoal decomposition` 提升为 formal theorem proving 的核心训练信号。
  - 冷启动不是直接喂 proof，而是通过“拆题-解子题-回填推理链”构造。
  - 形式推理和非形式推理的鸿沟被显式压缩。

- 与我们的不同
  - 它最终仍然在训练一个强 prover。
  - 它的 subgoal decomposition 更像训练数据构造与模型能力的一部分；我们可以把 subgoal tree 当成 evolve 个体本身。
  - 我们更关心 test-time 多样性与跨候选竞争，而不只是更强单模型。

- 对我们的启发
  - “拆题树”是很自然的 population 表示。
  - 适应度可以不只看 final proof，而看 subgoal tree 的可解性、覆盖度和重用率。

### 2.8 LeanAgent: Lifelong Learning for Formal Theorem Proving

- 时间与来源
  - arXiv, latest version `2025-03-06`
  - arXiv: https://arxiv.org/abs/2410.06209

- 核心思想
  - formal theorem proving 不应局限于静态 benchmark 或单一仓库，而应像数学家一样跨主题持续积累。
  - 系统通过 curriculum learning、dynamic database 和 progressive training，在不断扩大的 Lean repo 集合中持续成长。
  - 重点是跨域 generalization 且避免遗忘。

- 创新点
  - 把 theorem proving 问题从“单次 benchmark”扩展到“lifelong learning over repositories”。
  - 动态知识库和稳定性/回迁指标很适合衡量持续学习。
  - 在 advanced mathematics 场景上展示了明显优于 static baseline 的表现。
  - 不再把新 repo 当成彼此独立的数据源，而是显式设计“持续接触新仓库、持续吸收新知识、同时控制遗忘”的学习机制。
  - 把 `dynamic database` 引入 formal theorem proving，使系统可以随着训练过程持续更新可检索知识，而不是依赖一次性冻结的静态语料。
  - 把 `progressive training` 和 `curriculum learning` 结合起来，让系统先在较易、较熟悉的 theorem 上吸收能力，再逐步扩展到更复杂、更新的数学仓库。
  - 在评测层面不只看当前阶段证明了多少 theorem，还显式看 `stability / backward transfer`，也就是学了新仓库以后是否还保留旧能力。

- 整体框架流程图

```text
Stream of Lean Repositories / Theorem Sets
        |
        v
Repository Curriculum Scheduler
  - 决定先学哪些 repo
  - 控制难度与顺序
        |
        v
Current Prover + Dynamic Database
  - 当前参数化 prover
  - 持续更新的检索知识库
        |
        v
Attempt Theorems in Current Repository
  - 对当前 repo 中的 theorem 做证明
  - 记录 solved / unsolved / sorry theorems
        |
        +-------------------------------+
        |                               |
        | solved / useful traces        | unsolved / hard cases
        v                               v
Update Dynamic Database          Form next-stage training targets
  - 新 theorem / proof pattern     - 保留难题
  - 新检索条目                     - 进入后续 curriculum
  - repo-specific artifacts
        |                               |
        +---------------+---------------+
                        |
                        v
                Progressive Training
                 - 在累积数据上继续训练 prover
                 - 吸收新 repo 能力
                 - 尽量避免遗忘旧 repo
                        |
                        v
                Evaluate Lifelong Performance
                 - 当前 repo 表现
                 - 历史 repo 保持情况
                 - advanced math downstream proving
                        |
                        v
                Move to Next Repository / Stage
```

- 流程说明
  - `LeanAgent` 的核心不是单题上的 search loop，而是一个跨 repo 的持续学习循环。
  - 系统面对的是不断到来的 Lean repositories 或 theorem sets，而不是一个固定不变的数据集；因此第一步就需要决定“先学什么、后学什么”，这就是它的 `curriculum` 部分。
  - 在每一阶段，当前 prover 会在目标 repo 上尝试大量 theorem，并把成功证明、失败样例、可复用模式和 repo-specific 信息沉淀到 `dynamic database` 里。
  - `dynamic database` 的作用不是简单缓存 proof 文本，而是作为后续证明与训练都可使用的动态知识源，因此它和“静态预处理语料库”不一样。
  - 然后系统做 `progressive training`：在已经积累的多阶段数据上继续训练 prover，使其逐步吸收新仓库知识，同时尽量不丢掉旧仓库能力。
  - 最后评测也不是只看最新阶段，而是同时看新 repo 学得怎么样、旧 repo 保留得怎么样，以及在更高阶数学仓库上的 downstream proving 能力。

- 与我们的不同
  - LeanAgent 的“持续性”主要发生在训练周期与跨 repo 学习上。
  - 我们的“持续性”更想放在在线运行时，即每道题、每轮尝试、每批 lemma 的显式演化。
  - 它强调参数化 lifelong learning；我们更偏“参数模型 + 显式种群记忆”的混合系统。

- 对我们的启发
  - 如果后面要做长期系统，archive 不应只存 proof，还应存 domain-specific patterns、retrieval artifacts 和成功 decomposition。
  - “不会遗忘”的问题在 evolve 系统里也会出现，只是表现为 archive 退化或模式坍缩。

### 2.9 Lean-STaR: Learning to Interleave Thinking and Proving

- 时间与来源
  - arXiv, latest version `2025-03-15`
  - arXiv: https://arxiv.org/abs/2407.10040

- 核心思想
  - formal proof 里缺少人类思考过程，因此模型只学正式 proof 数据不够。
  - 该工作为每一步 tactic 生成先行的 informal thoughts，让模型在“想一想”之后再输出 formal action。
  - 再结合 expert iteration 进一步优化。

- 创新点
  - 把 thought trace 显式引入 Lean proof generation，而不是把思维链完全留在自然语言侧。
  - 通过 retrospective tactics 合成训练 thought，降低人工标注成本。
  - 不要求人工逐步标注“人类思考过程”，而是从已有 formal proof 反向构造与 tactic 对应的 thought-supervision，降低数据构建门槛。
  - 把“先想后证”做成模型内部的显式表示学习问题，而不是只在推理时临时加一段自然语言提示。
  - 再结合 expert iteration，让带有 thought augmentation 的 prover 持续吸收更高质量的证明轨迹。

- 整体框架流程图

```text
Existing Lean Proof Data
        |
        v
Retrospective Thought Synthesis
  - 根据 formal proof / tactic 序列
  - 反向构造对应的 informal thoughts
        |
        v
Thought-Augmented Training Data
  - 每一步包含:
    thought -> formal action / tactic
        |
        v
Train Lean-STaR Prover
  - 学会先生成 thought
  - 再生成 formal proving step
        |
        v
Proof Search / Theorem Proving
  - 对新 theorem 先想一想
  - 再输出 Lean action
        |
        v
Collect Improved Proof Traces
  - 成功证明
  - 更高质量的 step sequences
        |
        v
Expert Iteration
  - 用新 proof traces 继续训练
  - 进一步增强 thought + proving 能力
```

- 流程说明
  - `Lean-STaR` 的核心不是多 agent 协作，而是在单个 prover 内部显式插入 `thought` 这一中间层。
  - 它先从现有 Lean proof 数据出发，通过 `retrospective thought synthesis` 为每个 formal proving step 补出一个先行的思考描述。
  - 这样训练出来的数据不再只是“状态 -> tactic”，而是“状态 -> thought -> tactic / formal step”。
  - 推理时，模型会先生成一个简短 thought，帮助自己在当前 proof state 下整理该做什么，再输出正式的 Lean action。
  - 然后系统把新得到的更强 proof traces 继续回流到 `expert iteration` 中，形成一个“thought-augmented proving”的持续增强闭环。

- 与我们的不同
  - Lean-STaR 聚焦单模型表征增强，不是 agent 框架，也不是 population search。
  - 它的 thought 是辅助 token；我们可以把 thought / sketch / outline 提升成 evolve 中的显式个体。
  - 它更像“更会想的 prover”；我们要的是“会产生、筛选、重组多种想法的 proving ecosystem”。

- 对我们的启发
  - proof sketch、thought trace、subgoal rationale 都可以成为可变异对象，而不必局限于最终 Lean code。

### 2.10 EvolProver: Advancing Automated Theorem Proving by Evolving Formalized Problems via Symmetry and Difficulty

- 时间与来源
  - arXiv, submitted `2025-10-01`
  - arXiv: https://arxiv.org/abs/2510.00732

- 核心思想
  - formal theorem proving 的脆弱性之一是对题面变形不鲁棒，因此通过“演化题目”来增强模型。
  - 题目演化包括语法层对称、语义层跨领域变换，以及难度维度的可控扩展。
  - 最终用这些 evolved problems 训练 prover。

- 创新点
  - 把 evolve 显式带入 theorem proving，但对象是 `formalized problems`，而不是 proofs。
  - 同时处理 symmetry 与 difficulty 两种增长维度。
  - 不只是做普通数据增强，而是显式设计“哪些变换能保持有效性、哪些变换能提高难度”的 problem evolution 机制。
  - 把 theorem proving 的鲁棒性问题转写成“模型是否能适应系统性题面变换”的问题，因此更强调结构变换后的泛化能力。
  - 让 evolved problems 直接服务于后续 prover 训练，从而把 evolve 放在训练数据空间而不是求解轨迹空间。

- 整体框架流程图

```text
Seed Formalized Problems
        |
        v
Problem Evolution Engine
  - symmetry transformations
  - cross-domain transformations
  - difficulty-controlled transformations
        |
        v
Evolved Formalized Problems
  - 保持形式化有效性
  - 生成更多样、更难的新题
        |
        v
Filter / Validate Generated Problems
  - 检查是否仍然成立
  - 控制题目质量与难度
        |
        v
Augmented Training Set
        |
        v
Train / Fine-tune Prover
  - 在原题 + evolved problems 上训练
        |
        v
Evaluate on Formal Benchmarks
  - FormalMATH-Lite
  - MiniF2F-Test
  - Ineq-Comp-*
```

- 流程说明
  - `EvolProver` 的起点是已有的 `seed formalized problems`，而不是当前某道题的证明轨迹。
  - 系统先通过 `problem evolution engine` 对这些题目做结构化变换，变换来源包括对称性变化、跨领域变化，以及可控难度扩展。
  - 变换后的目标不是直接拿来当答案，而是形成一批新的 `formalized problems`，作为训练阶段的数据扩充与难度增强来源。
  - 这些新题经过过滤和有效性检查后，被并入训练集，再用于训练或微调 prover。
  - 最终评测看的也不是“演化过程中某题是否被解出”，而是“在 evolved problems 加持后，最终 prover 在正式 benchmark 上是否更强”。

- 与我们的不同
  - 它最接近我们名字里的 `evolve`，但 evolve 的对象不是解，而是题。
  - 它本质上还是 data evolution / augmentation pipeline。
  - 我们更想做的是 `proof evolution` 或 `reasoning-object evolution`。

- 对我们的启发
  - 演化对象可以是题、解、引理、变体、prompt、策略；不同对象决定不同研究问题。
  - 如果我们后续要扩展训练阶段，也完全可以把它的题目演化思想与 proof evolution 结合。

### 2.11 Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving

- 时间与来源
  - arXiv HTML version `v1`
  - 论文页: https://arxiv.org/html/2512.10739v1

- 核心思想
  - 这篇工作的重点不是 Lean formal proof，而是面向奥赛级数学题的长时程自然语言推理 agent。
  - 方法主体是 `Intern-S1-MO`：把 `reasoner agent`、`summarizer agent`、`verifier agent` 和 `lemma memory` 组织成多轮推理闭环。
  - 每一轮先尝试推进解题，再把有效进展压缩成中间 `lemmas` 写入记忆，供后续轮次继续使用。

- 创新点
  - 把长题求解从单轮 CoT 扩展成多轮 lemma accumulation。
  - 提出了 `OREAL-H` 训练框架，用 process verifier 和 lemma dependency graph 为多轮 agent 提供更细粒度 credit assignment。
  - 在多个真实奥赛 benchmark 上用统一 agent 框架取得很强结果。
  - 不只保留最终答案，而是把每轮有效推进显式压缩成 `lemma memory`，从而把“中间进展”变成后续轮次可消费的状态。
  - 用 `summarizer agent` 专门负责从长推理轨迹中提炼稳定中间结论，降低多轮推理时上下文不断膨胀的问题。
  - 用 `verifier agent + process verifier` 同时约束中间 lemma 和最终解答质量，使奖励不再只依赖最后是否答对。
  - 用 `lemma dependency graph` 做 credit assignment，把一次成功解答拆回到哪些中间 lemma 真正起了作用，从而更细粒度地训练长时程 agent。

- 整体框架流程图

```text
Olympiad-Level Math Problem
        |
        v
Reasoner Agent
  - 尝试当前轮解题
  - 提出思路、推导、候选 lemma
        |
        v
Summarizer Agent
  - 从当前轮长轨迹中提炼关键 progress
  - 压缩成可复用的 lemma memory items
        |
        v
Verifier Agent / Process Verifier
  - 检查中间 lemma 是否可靠
  - 评估最终解答质量
  - 给出过程级反馈
        |
        +------------------------------+
        |                              |
        | useful / verified progress   | weak / wrong progress
        v                              v
Update Lemma Memory              Discard / downweight bad traces
  - 存储中间结论
  - 维护依赖关系
        |
        v
Next Reasoning Round
  - 读取问题
  - 读取 lemma memory
  - 基于已有结论继续推进
        |
        v
Final Solution + Multi-round Training
  - 用最终结果与过程反馈训练 Intern-S1-MO
  - OREAL-H 用 lemma dependency graph 做 credit assignment
```

- 流程说明
  - `Intern-S1-MO` 的核心不是“单轮直接解出题”，而是把长题求解拆成多个 reasoning rounds。
  - 每一轮先由 `reasoner agent` 尝试推进证明或解答，然后由 `summarizer agent` 把这一轮真正有价值的中间结论压缩成较短、可复用的 `lemma memory`。
  - 接着 `verifier agent` 与过程级 verifier 对这些中间结论和当前最终答案候选进行评估，判断哪些推进值得保留，哪些是错误分支。
  - 已验证或高价值的中间结论会写入 `lemma memory`，下一轮 reasoner 会显式读取这份记忆，而不是每轮都从零开始。
  - 论文进一步用 `OREAL-H` 做训练，把多轮轨迹、过程级反馈和 `lemma dependency graph` 结合起来，学习怎样在长时程解题中更稳定地积累有用 lemma。

- 与我们的不同
  - 它不是 formal theorem proving，也不依赖 Lean verifier。
  - 它维护的是自然语言 `lemma memory`，不是可编译的 proof object population。
  - 它更像“多轮自然语言数学 agent + verifier + RL”，而我们要做的是“Lean-grounded agent + evolve + experience”。

- 对我们的启发
  - `lemma memory` 非常像我们设想中的经验库，但它的存储对象更偏自然语言引理与阶段性结论。
  - `lemma dependency graph` 对 reward / credit assignment 很有借鉴意义，可以直接对应我们后续的 `subgoal DAG + PRM reward` 设计。

## 3. 可补充引用但相对次一级的工作

### DeepSeek-Prover-V1.5

- 论文: https://arxiv.org/abs/2408.08152
- 价值
  - 很典型的 `proof assistant feedback + RL + Monte-Carlo tree search` 路线。
  - 说明 verifier feedback 和 search 的结合已经能带来明显提升。
- 与我们的差异
  - 更偏单模型 + 搜索，不是 agentic population framework。

### Goedel-Prover

- 论文: https://arxiv.org/abs/2502.07640
- 价值
  - 很强的数据合成与 iterative prover construction 路线。
  - 强调 formalization pipeline 与 prover 迭代共进化。
- 与我们的差异
  - 主要演化的是训练数据和模型能力，不是在线证明候选的种群。

### Formal Mathematics Statement Curriculum Learning

- 论文: https://openreview.net/forum?id=-P7G-8dmSh4
- 价值
  - 是 expert iteration / automatic curriculum 在 formal math 里的重要先驱。
  - 证明“越来越难的自动课程”是可行的。
- 与我们的差异
  - 更偏 training-time curriculum，而非 agent + evolve 的运行时系统。

## 4. 我们与现有工作的核心差异，可以怎样表述

如果后续论文真的按当前思路推进，我认为最值得强调的差异不是“我们也做 formal theorem proving”，而是下面这些点：

### 4.1 我们优化的对象不是单个最终 proof，而是一个显式种群

现有多数工作优化的是：

- 一个 prover 模型
- 一条 proof trajectory
- 一个搜索树
- 一个数据分布

而我们更适合优化：

- proof sketch population（证明思路种群，即多个粗粒度证明路线候选）
- auxiliary lemma population（辅助引理种群，即多组可供主证明调用的中间命题候选）
- Lean draft population（Lean 草稿种群，即多份部分完成、可继续修复的 Lean 证明草稿）
- subgoal decomposition population（子目标拆分种群，即多种把原题拆成若干可解子问题的方案）
- reusable tactic / rewrite pattern archive（可复用 tactic / 改写模式档案库，即跨题积累的证明操作模板与变换规则）

这会让系统从“求一个答案”转向“维护一个持续增殖、筛选、积累的证明生态”。

### 4.2 我们的 evaluator 不只是 pass/fail

Lean 天然给出比一般 coding evaluator 更细粒度的反馈，可以把 fitness 设计为多目标：

- 编译是否通过
- 是否完全消除 `sorry`
- 子目标数量是否下降
- 目标复杂度是否下降
- 新引理是否能被多处复用
- 证明草稿是否提升后续 mutation 成功率

这使得 formal theorem proving 可能成为比一般代码优化更适合做 evolutionary search 的场景。

### 4.3 我们的改进发生在 test time，也保留长期记忆

多数工作偏向：

- 训练更强模型
- 或对单题做搜索/自纠错

我们更像：

- 题内：围绕一个问题维护 evolving population
- 题间：把高价值个体进入 archive，迁移到后续问题

这在方法论上介于 `AlphaProof` 的题内适配、`LeanAgent` 的长期积累、`AlphaEvolve` 的显式演化之间。

### 4.4 我们天然适合 agent system，而不是纯 prover model

因为个体类型多样，系统更适合 agent 编排：

- 一个 agent 拆题
- 一个 agent 产 lemma
- 一个 agent 产 Lean step
- 一个 controller 负责 selection / mutation / recombination
- Lean verifier 负责客观评估

这让系统设计更接近 `Numina-Lean-Agent / Prover Agent`，但外层优化机制更接近 `AlphaEvolve`。

## 5. 当前最自然的论文定位

如果后续投稿 NeurIPS，我建议 related work 和 intro 里的定位不要写成：

- “我们提出一个新的 formal theorem proving agent”

更好的写法是：

- “We formulate formal theorem proving as verifier-grounded evolutionary search over structured proof objects.”
- “Unlike prior agentic theorem provers that rely on single-trajectory planning or verifier-guided repair, we maintain an explicit evolving population of proof plans, auxiliary lemmas, and Lean drafts.”
- “Our framework bridges agentic formal reasoning and evaluator-driven evolutionary optimization.”

## 6. 个人判断：哪些最值得优先对齐

如果只选最关键的 5 篇进入主文 related work，我建议是：

1. STP
2. AlphaProof
3. AlphaEvolve
4. Prover Agent
5. Numina-Lean-Agent

理由很直接：

- `STP` 对应 self-play / curriculum / self-improvement
- `AlphaProof` 对应 formal RL / test-time adaptation
- `AlphaEvolve` 对应 evaluator-guided evolution
- `Prover Agent` 对应 auxiliary lemma-centric agent framework
- `Numina-Lean-Agent` 对应 general coding agent + Lean MCP paradigm

如果扩展到 8 到 10 篇，则把 `Goedel-Prover-V2`、`DeepSeek-Prover-V2`、`LeanAgent`、`EvolProver` 加进来最合适。

## 7. 这些工作用了哪些模型 / benchmark / 指标 / baseline

这一节专门回答“他们怎么评测自己”。需要说明的是：

- 我这里优先依据论文摘要页、官方项目页、官方 README 和公开实验说明整理。
- 对少数工作，公开摘要会明确 benchmark，但不会把完整 baseline 表全部展开。我会明确标注“公开材料未完整列出”。
- 因为这些工作横跨 `formal theorem proving`、`agentic proving` 和 `evolutionary coding`，它们的评测口径并不完全一致。
- 对 `模型` 一列，我优先写公开论文或官方仓库里能够直接确认的主体模型；如果公开材料没有把单一底模写死，我会直接写成“公开材料未完整披露单一底模”。

| 工作 | 使用的模型 | 主要 benchmark / evaluation setting | 主指标 | 主要 baseline / comparison target |
| --- | --- | --- | --- | --- |
| STP | 公开论文重点是 `self-play conjecturer + prover` 训练框架；公开材料可确认主实验覆盖 `1.5B / 7B` 级 prover family，但摘要页没有把单一 branded base model 作为主卖点 | `LeanWorkbook` 中的大规模 theorem proving 任务；`miniF2F-test`；`ProofNet-test`；`PutnamBench`；并在 `Lean` 与 `Isabelle` 上都验证 | `pass@3200` solved rate；`LeanWorkbook` 上已证明 statement 比例 | 明确对比 `standard expert iteration`；同时与 prior whole-proof generation methods 对比 |
| AlphaProof | `AlphaProof` 是 DeepMind 自研的 AlphaZero-style formal reasoning system；公开论文强调 RL agent 和 test-time RL，而不是一个公开可下载的单一 LLM 名称 | `historical mathematics competition problems`；`2024 IMO` 非几何题；与 `AlphaGeometry 2` 组合看整体 IMO 成绩 | 历史竞赛题 solved count；IMO official scoring / medal-equivalent performance | 公开摘要强调显著超过 prior SOTA on historical competition problems，但摘要页没有逐项列出完整 baseline 名称 |
| AlphaEvolve | 公开材料写法是 `Gemini-powered coding agent`；官方 blog 强调 `Gemini` family，但没有把单一固定底模作为论文主表重点 | 不使用标准 theorem proving benchmark；评测任务包括 `50+ open math problems`、矩阵乘法算法发现、`Borg` 调度、`TPU` 电路验证、`Gemini` 训练 kernel、`FlashAttention` kernel | 相对已有最好解的目标函数提升、开放问题求解、系统性能增益 | `previous best known solutions`、现有生产 heuristics、已有 kernel / algorithm 或线上方案 |
| Prover Agent | 官方仓库 README 明确是三路模型协同：`informal reasoning LLM + formal prover model + formalizer model`；公开仓库配置可确认实验部署示例使用过 `DeepSeek-R1-0528-Qwen3-8B`、`Goedel-Prover-V2-8B`、`Goedel-Formalizer-V2-8B` | `MiniF2F`；`PutnamBench`；以及论文中的 theorem proving case studies | `MiniF2F` success rate vs sample budget；`PutnamBench` solved count | 公开材料可确认 baseline 包括 `Kimina-Prover-Preview-Distill`、`STP`、`Leanabell-Prover-GD-RL`、`Goedel-Prover-SFT`、`BFS-Prover`、`InternLM2.5-StepProver + BFS + CG` |
| Numina-Lean-Agent | 论文与仓库公开口径可确认主体是 `Claude Opus 4.5 + Claude Code + Numina-Lean-MCP` 的通用 agent 系统，而不是专用 prover model | 官方公开结果重点是 `Putnam 2025` 全 `12/12`；另有论文级 formalization case study `Effective Brascamp-Lieb inequalities`；官方仓库同时提供 `MiniF2F` 运行配置和示例 | solved count / solved ratio；公开口径主打 `Putnam 2025 12/12` | 官方 README / 摘要公开口径是“`12/12`, matching the best closed-source system”；公开材料没有完整展开 baseline 表 |
| Goedel-Prover-V2 | 官方公开提供 `8B` 与 `32B` 级 prover family；README / blog 明确区分 base model 与 self-correction setting | `MiniF2F`；`PutnamBench`；`MathOlympiadBench` | `MiniF2F pass@32`；`PutnamBench` solved count at `Pass@64 / Pass@184`；并报告 self-correction fixed rounds | 官方 blog / README 明确对比 `DeepSeek-Prover-V2-671B`、`Kimina-Prover-72B`；PutnamBench leaderboard 还出现 `DSP+`、`Self-play Theorem Prover`、`Goedel-Prover-V1` |
| DeepSeek-Prover-V2 | 官方 README 明确发布 `DeepSeek-Prover-V2-7B` 与 `DeepSeek-Prover-V2-671B`；其中 `671B` 建立在 `DeepSeek-V3-Base` 上，冷启动分解数据由 `DeepSeek-V3` 递归生成 | `MiniF2F-test`；`PutnamBench`；自建 `ProverBench`，其中包含 `15` 道 `AIME 2024/2025` 题和 `310` 道教材题 | `MiniF2F` pass ratio；`PutnamBench` solved count；`ProverBench` solved count | 官方 README / 摘要明确是对 prior neural theorem proving SOTA 做比较，但公开摘要页未完整展开 baseline 全名单 |
| LeanAgent | 公开论文主张是 `lifelong learning prover + dynamic database + progressive training`；公开摘要更强调训练范式与 repository-level setting，未把单一底模名放成主卖点 | `23` 个 Lean repositories 上的 lifelong setting；并在 advanced mathematics downstream proving 上验证 | `proved/total sorry theorems`、新证明 theorem 数、stability / backward transfer 等 lifelong metrics | 明确对比 `static baseline` / 非终身学习式 prover；公开摘要未完整列出全部模型名 |
| Lean-STaR | 公开项目页与论文强调在现有 Lean prover 上加入 `thought augmentation`；主实验是 `7B` 级 prover family 的同底模对比 | `miniF2F-test` | `Pass@64`；公开项目页直接给出 `43.4% -> 46.3% Pass@64` | 最主要 baseline 是相同 base model 在无 thought augmentation 时的版本；公开页面同时列出 `GPT-4-Turbo`、`DSP`、`COPRA`、`ReProver`、`Curriculum Learning`、`HTPS` 等公开结果 |
| EvolProver | 公开摘要强调 `evolve formalized problems` 的训练框架；公开摘要没有完整披露单一底模名，但明确与同规模 reasoning / non-reasoning models 比较 | `FormalMATH-Lite`；`MiniF2F-Test`；`Ineq-Comp-Seed`；`Ineq-Comp-Transformed` | `pass@32` | 公开摘要可确认其超越 `comparable-size models`，包括 `reasoning-based models` 与 `non-reasoning models`；公开摘要未逐一展开全部模型名 |
| Intern-S1-MO | 论文主体方法是 `Intern-S1-MO`，并报告较小版本 `Intern-S1-mini-MO`；是多 agent 长程自然语言数学推理系统，不是 Lean prover | `HMMT2025`；`AIME2025`；`CNMO2025`；`IMO2025` 非几何题；`CMO2025` 正式参赛结果 | 多数 benchmark 用 unbiased `pass@1`；`IMO2025` 报 `pass@4`；`CNMO2025 / IMO2025 / CMO2025` 还报 points / total score | 明确对比 `Gemini 2.5 Pro`、`o3-high`、`Grok 4`、`GPT-OSS-120B`、`DeepSeek-R1-0528`、`Qwen3-235B-A22B`；论文还包含自身 ablation：`Single-round with Agents`、`+ Multi-round Reasoning`、`+ Theorem Verifier`、`+ Process Verifier`、`+ OREAL-H` |

### 7.1 他们到底是 single-shot，还是 pass@k / fixed-budget

这是规划我们论文时最关键的问题之一。因为如果我们的方法看起来只是“比别人多试很多次”，reviewer 一定会质疑公平性。

现有工作实际上有三种主流评测协议：

- `single-shot / one-run`
  - 很少作为 formal theorem proving 的主指标
  - 因为 whole-proof generation 的方差很大，单次成功率容易低且不稳定

- `pass@k / sample-budget scaling`
  - 这是目前 formal theorem proving 里最主流的口径
  - 含义是：对同一道题采样 `k` 次，只要有一个 proof 被 Lean 验证通过，就算 solved
  - 代表工作：`STP`、`Lean-STaR`、`Goedel-Prover-V2`、`EvolProver`

- `fixed-budget agent evaluation`
  - 更适合 agent 系统
  - 给定固定的 `round budget / attempt budget / token budget / wall-clock budget / verifier budget`
  - 系统可以多轮交互、修复、生成引理，但不能无限迭代
  - 代表工作：`Numina-Lean-Agent`、`AlphaProof`，以及部分 agent 系统的实际运行方式

按这个视角看，现有工作并不是“都只跑一轮”。恰恰相反，formal theorem proving 领域大量工作本来就在做 `pass@k` 或 `budgeted search`。因此，我们的问题不在于“是否允许多轮”，而在于：

- 我们是否对所有方法施加了同等预算约束
- 我们的方法是否在相同预算下，比简单多采样或多轮修复更有效

下面是更细的归类。

| 工作 | 公开评测协议 | 主指标 | 对我们的启发 |
| --- | --- | --- | --- |
| STP | 明确使用 `pass@3200`；不是 single-shot | `miniF2F-test` / `ProofNet-test` / `PutnamBench` solved rate under `pass@3200`；`LeanWorkbook` 上看已证明 statement 比例 | 它已经证明 theorem proving 里大样本预算是常见设定，所以我们必须和同预算基线比较，而不是只和单轮方法比 |
| AlphaProof | 公开口径更像 `competition-style solved count` + `multi-day compute`；不是 single-shot | 历史竞赛题 solved count、IMO 最终得分；公开摘要没有用统一 `pass@k` 来表述 | 如果我们想走顶会路线，最好不要只学它“给很多算力”，而是学它“明确写清 compute 和 test-time adaptation” |
| AlphaEvolve | 不是 single-shot，也不是 pass@k；是 `bounded evaluator-guided iterative improvement` | 目标函数提升、开放问题求解、相对已有最佳方案的改进幅度 | 这更像我们想做的上层评测逻辑：不是无限试，而是在预算内看迭代优化曲线 |
| Prover Agent | 公开图表明确强调 `sample budget`；不是 single-shot | `MiniF2F` success rate 随 sample budget 变化；公开版本之间结果有更新：`2025-06-24` arXiv/ICML 口径是 `86.1%`，`2026-02-11` OpenReview 修订口径是 `88.1%`，并额外报告 `PutnamBench` solved count | 这篇说明 agent 论文也不一定只报单点结果，更合理的是报“预算-性能曲线” |
| Numina-Lean-Agent | 更像 `fixed-round / fixed-attempt budget` 的 agent 系统；公开展示不是标准 pass@k 表 | 公共材料主打 `Putnam 2025 12/12` 和大型 formalization case study；仓库中公开了 `max_rounds`、attempt budget 等运行约束 | 如果我们做 agent + evolve，Numina 是最接近的公平比较对象之一；比较时要把 round / attempt budget 对齐 |
| Goedel-Prover-V2 | 明确使用 `pass@32`；PutnamBench 还报告 `Pass@64` 和 `Pass@184`；并公开 self-correction 轮数 | `MiniF2F` `pass@32`；`PutnamBench` num-solved at fixed pass budgets；self-correction mode 固定两轮修复 | 这是最值得借鉴的评测模板之一：固定样本预算、固定修复轮数、画 scaling curve |
| DeepSeek-Prover-V2 | 公开 README 报 `pass ratio` 和 solved count；不是 single-shot，但 README 没把主结果统一写成某个固定 `k` | `MiniF2F-test` 的 pass ratio、`PutnamBench` solved count、`ProverBench` solved count | 它提醒我们：如果主指标写得太松，reviewer 会追问 sample budget，所以我们自己的协议必须更清楚 |
| LeanAgent | 不是 single-shot，也不是 pass@k；是 `repository-level lifelong evaluation` | `proved/total sorry theorems`、总共新证明 theorem 数、stability/backward transfer 等 lifelong metrics | 如果我们后续做跨题 archive，这篇的“长期记忆”指标很值得借鉴 |
| Lean-STaR | 明确使用 `Pass@64`；不是 single-shot | `miniF2F-test Pass@64` | 这是最标准的“同底模、同 benchmark、固定 k 对比”范式 |
| EvolProver | 明确使用 `pass@32`；不是 single-shot | `FormalMATH-Lite`、`MiniF2F-Test`、`Ineq-Comp-*` 的 `pass@32` | 它说明 evolve 类工作同样可以用标准 `pass@k` 说清楚，而不是只给模糊的最好结果 |
| Intern-S1-MO | 更像 `fixed-round agent evaluation + repeated rollout grading`；不是 Lean `pass@k` | 多数 benchmark 报 unbiased `pass@1`，`IMO2025` 报 `pass@4`，竞赛题还报 point-based grading | 它提醒我们 agent 论文即使不做 Lean formal proof，也会把 rollout 次数和评分协议写得很清楚，我们后续也必须把 budget 与 grading 说死 |

### 7.2 对我们最有用的 benchmark 启发

如果我们后续真的做 `agent + evolve + math proof`，我认为最值得借鉴的评测组合不是只跑一个基准，而是至少做三层：

1. `标准 formal benchmark`
   - 例如 `miniF2F-test`、`PutnamBench`
   - 作用是和 `DeepSeek / Goedel / STP / Prover Agent` 这些工作建立可比较性

2. `长时程 agent benchmark`
   - 例如更长、更依赖辅助引理和拆步的题，或者一个小型 theorem formalization project
   - 作用是证明 evolution 不是只对短题有效

3. `过程型指标`
   - 例如 compile rate、sorry-free rate、subgoal reduction、auxiliary lemma reuse、proof repair success rate、population diversity
   - 作用是说明 evolve 框架为什么有效，而不只是最终 pass@k 偶然更高

### 7.3 对 baseline 设计的直接建议

如果按论文写作标准来设计，我建议我们最终至少要对比四类 baseline：

1. `single-shot agent`
   - 没有 evolve，只给一次或少量重试

2. `iterative repair agent`
   - 有 verifier-guided repair，但没有 population

3. `search-based prover`
   - 类似 `DeepSeek-Prover / Goedel-Prover / STP` 代表的 search 或 self-improvement 路线

4. `ablation on evolution controller`
   - 去掉 selection
   - 去掉 archive
   - 去掉 lemma population
   - 去掉 recombination

这样才能把“agent 有用”与“evolve 有用”拆开证明。

### 7.4 这对我们的方法定义意味着什么

如果我们最后的方法叫 `agent + evolve + math proof`，我认为 `evolve` 绝对不应该被定义成：

- “一直迭代到 Lean 成功为止”

这种定义会直接被质疑成：

- 只是 verifier-guided retry
- 只是比别人用了更多 test-time compute
- 只是把 pass@k 换了个说法

更合理的定义应该是：

- 在固定预算内维护一个 `population of proof objects`
- 个体可以是 `proof sketch / subgoal tree / auxiliary lemma / Lean draft / repair trace`
- 每一轮不是简单重试，而是有显式的 `selection / mutation / recombination / archive update`
- 最终比较的是：在相同预算下，`evolutionary controller` 是否比 `independent sampling` 和 `iterative repair` 更高效

因此，我建议我们最终论文里把公平性写死为：

1. 固定 `token budget`
2. 固定 `Lean verifier call budget`
3. 固定 `wall-clock budget`
4. 固定 `max rounds / max generations`
5. 所有 baseline 共用相同底模和相同工具接口，尽量只改变 controller

这样 reviewer 就很难把我们的方法简单归结为“只是多跑几轮”。

## 8. 参考链接

- STP: https://arxiv.org/abs/2502.00212
- STP (ICML 2025 / PMLR): https://proceedings.mlr.press/v267/dong25h.html
- AlphaProof: https://www.nature.com/articles/s41586-025-09833-y
- AlphaEvolve: https://arxiv.org/abs/2506.13131
- Prover Agent: https://arxiv.org/abs/2506.19923
- Numina-Lean-Agent: https://arxiv.org/abs/2601.14027
- Goedel-Prover-V2: https://arxiv.org/abs/2508.03613
- DeepSeek-Prover-V2: https://arxiv.org/abs/2504.21801
- LeanAgent: https://arxiv.org/abs/2410.06209
- Lean-STaR: https://arxiv.org/abs/2407.10040
- EvolProver: https://arxiv.org/abs/2510.00732
- DeepSeek-Prover-V1.5: https://arxiv.org/abs/2408.08152
- Goedel-Prover: https://arxiv.org/abs/2502.07640
- Formal Mathematics Statement Curriculum Learning: https://openreview.net/forum?id=-P7G-8dmSh4
- AlphaEvolve (official blog): https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- Goedel-Prover-V2 (official blog): https://blog.goedel-prover.com/
- Goedel-Prover-V2 (official repo): https://github.com/Goedel-LM/Goedel-Prover-V2
- DeepSeek-Prover-V2 (official repo): https://github.com/deepseek-ai/DeepSeek-Prover-V2
- Lean-STaR (official project page): https://leanstar.github.io/
- EvolProver (OpenReview): https://openreview.net/forum?id=cBoYkLG3EJ
- Long-horizon Reasoning Agent for Olympiad-Level Mathematical Problem Solving: https://arxiv.org/html/2512.10739v1

## 9. 我们的方法里，`evolve` 到底指什么

这一节不是在总结已有论文，而是给我们自己的方法定义边界。之所以需要单独写，是因为 formal theorem proving 很容易把下面几件事混在一起：

- 多采样
- 多轮修复
- search
- population evolution

如果这几者不区分清楚，reviewer 很容易把我们的方法概括成：

- “只是 verifier-guided retry”
- “只是比别人用了更多 test-time compute”
- “只是把 pass@k 换了个新名字”

所以，`evolve` 必须被定义为一种**结构化、预算受限、显式维护种群的优化过程**，而不是“只要没成功就继续试”。

### 9.1 什么不叫 evolve

下面这些都不应该被我们称作 `evolve`：

1. `independent sampling`
   - 同一个 prompt 采样很多次
   - 候选之间没有信息传递
   - 最后只看有没有一个过 Lean

2. `iterative repair`
   - 上一版 proof 报错了
   - 模型根据 Lean error message 修一下
   - 继续单轨迹往前走

3. `unbounded retry`
   - 一直试到成功为止
   - 只是把预算藏起来，没有明确控制 token / round / verifier call

4. `plain tree search`
   - 只做搜索树展开
   - 但没有显式 archive、没有个体重组、没有跨候选知识积累

这些过程可能都很有用，但它们本身还不足以支持一篇 `agent + evolve + math proof` 的顶会论文。

### 9.2 我们这里的 evolve 应该怎么定义

我建议把 `evolve` 明确定义为：

> 在固定预算内，维护一个由结构化证明对象组成的显式种群；基于 Lean verifier 和过程信号对这些个体做 selection、mutation、recombination 和 archive update，从而逐轮提升最终证明成功率与中间推理质量。

这个定义里有四个不可缺的部分。

1. `显式种群`
   - 不是一条 proof trajectory
   - 而是一组并行存在、相互竞争的候选个体

2. `结构化个体`
   - 个体不一定只是完整 Lean proof
   - 也可以是：
     - proof sketch
     - subgoal decomposition tree
     - auxiliary lemma set
     - partial Lean draft
     - repair trace

3. `显式进化操作`
   - selection：保留哪些候选
   - mutation：如何局部改写
   - recombination：如何组合不同候选的优点
   - archive update：哪些中间成果值得长期保留

4. `固定预算`
   - 所有进化都必须发生在固定的 test-time 预算内
   - 不能用“直到成功”为 stopping criterion

### 9.3 为什么我们不能把成功定义成“迭代直到 Lean 通过”

因为那样方法学上会立刻塌成一个很弱的 claim。

如果 stopping rule 是：

- “直到成功”

那 reviewer 会自然追问：

- 你是不是只是比 baseline 调了更大的 `k`
- 你是不是只是给 agent 更多轮次
- 你是不是只是在做 repair loop
- 你报告的提升是不是完全来自更大的 compute

所以我们必须把 stopping rule 改成：

- 固定 `max generations`
- 固定 `token budget`
- 固定 `Lean verifier call budget`
- 固定 `wall-clock budget`

然后比较不同 controller 在相同预算下的表现。

### 9.4 我们的个体可以是什么

我不建议把个体只定义成“完整 proof 文件”，那样太窄，也不利于体现 evolve 的独特价值。

更合理的是让种群里允许多种粒度的对象：

1. `proof sketch individual`
   - 用自然语言或半形式化方式给出证明路线
   - 适合大跨度变异和重组

2. `subgoal tree individual`
   - 把目标拆成多个可验证的子目标
   - 适合评估 decomposition 的质量

3. `lemma set individual`
   - 一组辅助引理
   - 适合跨题迁移和 archive

4. `partial Lean draft individual`
   - 已经能部分编译的 Lean 代码
   - 适合做 repair 和局部替换

5. `proof trace individual`
   - 一连串 tactic / rewrite / failed attempts 的轨迹
   - 适合提取高价值操作模式

这样定义后，`evolve` 的对象就不再是“重复生成完整答案”，而是“在多个层级上优化证明过程”。

### 9.5 fitness 不该只看 final pass/fail

如果 fitness 只看最终是否证明成功，那么 evolve 会退化成普通的 sparse-reward search。

Lean 的优势就在于它能给出非常强的过程信号。比较合理的多目标 fitness 可以包括：

- final theorem 是否完全证明
- 是否 `sorry-free`
- 是否保持主 theorem statement 不变
- 子目标数量是否下降
- 目标复杂度是否下降
- 新引理是否可复用
- repair 后的 compile rate 是否上升
- 这个个体是否对后续 offspring 有贡献

这样 `evolve` 才不是“撞大运找到一个过的 proof”，而是“持续提升 proof ecosystem 的整体质量”。

### 9.6 一个更像顶会论文的方法表述

如果后续要写论文，我建议不要写：

- “We iteratively refine Lean proofs until one succeeds.”

这太像 repair loop。

更合适的写法是：

- “We maintain a budgeted population of structured proof objects and evolve them under Lean-verifier feedback.”
- “Our method goes beyond independent pass@k sampling by explicitly selecting, recombining, and archiving intermediate proof artifacts.”
- “The key question is not whether more attempts help, but whether an evolutionary controller yields better success under the same inference budget.”

### 9.7 对应到实验，应该如何公平

为了避免“只是多轮重试”的质疑，我建议后续实验必须同时满足：

1. `固定底模`
   - 与 internal baselines 共用同一个 base model

2. `固定工具接口`
   - 都能访问同样的 Lean checker / retrieval / MCP 工具

3. `固定预算`
   - token budget
   - verifier call budget
   - max generations / rounds
   - wall-clock budget

4. `明确对照组`
   - independent sampling
   - iterative repair
   - search-only baseline
   - ours with full evolution

5. `除了最终成功率，还要报过程指标`
   - compile rate over time
   - sorry-free rate
   - useful lemma discovery
   - archive reuse
   - diversity vs success

只有这样，`evolve` 才会被看作方法创新，而不是算力堆叠。

### 9.8 一句最简洁的定义

如果最后只留一句话，我建议用这句：

> In our setting, evolution is not unlimited retry until Lean succeeds; it is a budgeted optimization process over an explicit population of proof objects, driven by verifier-grounded selection, mutation, recombination, and memory.
