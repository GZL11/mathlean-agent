# 当前代码库中的 MathProof 流程

这份文档只回答一个问题：

- **当前代码库里，mathproof 是怎么跑的？**

整体上，当前代码库的 mathproof 由三部分串起来：

1. `Famou` 负责外层迭代与 archive / population 管理
2. `third_party/numina-lean-agent` 负责证明任务拆解与 Lean 代码生成
3. `Lean MCP tools + lean_checker` 负责 theorem search、proof state 查询和最终校验

## 1. 完整流程图

```text
Math Problem / Lean Task
        |
        v
Load task into current experiment
        |
        v
Famou creates Context
  - task_description
  - task_constraint
  - current population snapshot
  - iteration / diversity / island info
        |
        v
Select
  - 选当前 parent candidate
  - 选 inspirations
        |
        v
Generate
  - 调用 mathproof worker
  - 进入 Numina-style proving workflow
        |
        v
Coordinator chooses target
  - 读取 BLUEPRINT
  - 选择下一个 lemma / theorem / subgoal
  - 判断复杂度
        |
        +-------------------------------+
        |                               |
        | simple                        | medium / complex
        v                               v
   Proof Agent                  Blueprint Agent / Sketch Agent
   直接做证明                     - 补 informal proof
                                 - 拆 lemma
                                 - 生成 Lean statement / skeleton
        |                               |
        +---------------+---------------+
                        |
                        v
                Lean-guided proving loop
                  - informal prover 给出思路
                  - theorem / premise search
                  - 查看 proof state
                  - 修改 Lean draft
                  - 局部重试
                        |
                        v
                lean_checker / Lean MCP
                  - lake env lean <file>
                  - diagnostics
                  - goal state
                  - sorry 检查
                        |
                        v
Evaluate
  - 提取 verifier-grounded metrics
  - combined_score / validity / error_info / custom metrics
        |
        v
Judge
  - 总结当前版本变化
  - 归纳失败原因
  - 产出下一轮改进建议
        |
        v
Feature
  - 提取 feature_vector
        |
        v
Population / Archive Update
  - 新候选加入 archive
  - 当前 island population 更新
  - best program 更新
        |
        v
Next rollout / next iteration
```

## 2. 流程分解

### 2.1 输入阶段

输入可以是：

- 已经 formalize 好的 Lean theorem / `.lean` 文件
- 需要先形式化的自然语言数学题

无论哪种输入，都会先进入当前实验的 `Context`，作为这一轮 rollout 的任务描述和约束。

### 2.2 Select：从当前种群里选父代

当前代码库的外层控制逻辑和普通任务一致，先从当前 population 中选 parent candidate。

这一步做的事是：

- 从当前 island population 里选出当前要扩展的证明候选
- 额外选若干 inspirations 作为上下文参考

在 mathproof 里，这个 candidate 对应的是当前的 Lean 证明版本或证明草稿。

### 2.3 Generate：进入证明 worker

当前代码库里，mathproof 的生成阶段不是简单“生成一段代码”，而是进入一条更长的 proving workflow。

这条 proving workflow 由 `numina-lean-agent` 风格的多角色流程组成。

核心角色有三个：

- `Coordinator`
- `Blueprint Agent`
- `Sketch Agent`
- `Proof Agent`

### 2.4 Coordinator：决定当前先做什么

Coordinator 会读取 `BLUEPRINT.md`，然后：

- 找出依赖已满足的目标
- 判断当前目标是简单、中等还是复杂
- 决定后续走哪条 proving 路径

它的角色是编排，不直接写证明。

### 2.5 Blueprint Agent：重写问题结构

当目标较复杂时，Blueprint Agent 会先介入。

它负责：

- 补强 informal proof
- 把过大的 lemma 再拆开
- 更新依赖拓扑
- 把整题改写成更容易逐步推进的证明图

这一步的输出不是最终 Lean 证明，而是更清晰的证明结构。

### 2.6 Sketch Agent：把结构落成 Lean 草稿

Sketch Agent 把 blueprint 里的节点翻成 Lean 代码骨架。

它负责：

- 生成 theorem / lemma statement
- 把需要证明的节点落到具体文件和行号
- 生成初始 proof skeleton
- 在必要位置保留 `sorry`

这一步的输出是一个**可继续编辑、可被 Lean 检查的中间草稿**。

### 2.7 Proof Agent：围绕当前目标做真正的 Lean 证明

Proof Agent 是核心 proving worker。  
它会围绕当前目标进入一个局部闭环：

```text
当前目标 / 当前 Lean 草稿
    |
    v
拿到 informal 思路
    |
    v
搜索定理 / premise
    |
    v
查看当前 proof state
    |
    v
修改 Lean 代码
    |
    v
重新检查
    |
    +--> solved
    |
    +--> 继续下一个子目标
    |
    +--> stuck -> repair / add lemma / change strategy
```

它用到的核心工具都已经在当前代码库里整理好了，主要包括：

- `gemini_informal_prover`
- `gpt_informal_prover`
- `lean_goal`
- `lean_diagnostic_messages`
- `lean_local_search`
- `lean_leandex`
- `lean_loogle`
- `lean_leanfinder`
- `lean_state_search`
- `lean_hammer_premise`

### 2.8 Lean 校验：当前版本到底有没有过

每次 proof agent 修改完 Lean 草稿后，都会回到 Lean 校验环节。

当前代码库里这一步的核心标准非常明确：

- 运行 `lake env lean <file>`
- 看是否有编译错误
- 看是否还有 `sorry`

如果 `allow_sorry=False`，那么存在 `sorry` 也视为失败。

因此一个版本的 Lean 证明，会被判成下面几类状态：

1. `编译失败`
   - 语法、类型、tactic 或依赖错误

2. `编译通过但还有 sorry`
   - 说明只是中间状态

3. `编译通过且无 sorry`
   - 说明该文件里的目标已真正证明完成

这一步给出的不是语言模型主观判断，而是 Lean 的客观结果。

### 2.9 Evaluate：把 Lean 结果转成当前框架里的分数和指标

Lean 校验结果回到 Famou 的 `Evaluate` 阶段后，会被整理成当前框架能消费的指标，例如：

- `combined_score`
- `validity`
- `error_info`
- 其他自定义 metrics

在 mathproof 任务里，这些指标的来源不再是普通程序评测，而是 Lean verifier 的结果。

### 2.10 Judge：把这次失败或进展变成文本反馈

`Judge` 阶段不是再做一次真伪判断，而是把当前结果总结成更适合下一轮使用的反馈。

这一步通常会总结：

- 当前版本相比父代做了什么修改
- 为什么失败
- 哪一步卡住
- 下一轮更适合往哪个方向改

### 2.11 Population / Archive Update：把本轮结果放回系统

最后，当前版本的证明候选会进入：

- `archive`
- `population`

更新后的结果包括：

- 当前 island 是否保留这个候选
- archive 是否新增一条新证明轨迹
- best candidate 是否更新

然后外层进入下一轮 rollout。

## 3. 当前代码库里的单轮 MathProof 流程图

```text
Parent proof candidate
        |
        v
Coordinator reads BLUEPRINT
        |
        +--> need restructure? --> Blueprint Agent
        |
        +--> need formal statement? --> Sketch Agent
        |
        v
Proof Agent
  - informal reasoning
  - theorem search
  - premise search
  - Lean editing
        |
        v
lean_checker / diagnostics / goal tools
        |
        +--> compile error
        |
        +--> compile ok but sorry remains
        |
        +--> compile ok and no sorry
        |
        v
Evaluator converts result to score/metrics
        |
        v
Judge summarizes feedback
        |
        v
Child proof candidate written back to archive/population
```

## 4. 当前代码库里的关键判定标准

对 mathproof 来说，当前代码库里最关键的成功判定不是“模型说自己证完了”，而是：

```text
Lean checker accepts the file
AND
there is no sorry
```

也就是：

- 文件能通过 Lean 检查
- 没有 `sorry`

这时才把它视为真正成功的证明版本。

## 5. 一句话总结

当前代码库里的 mathproof 主链就是：

```text
Select 当前证明候选
-> 用 Numina-style proving worker 生成下一版 Lean 证明
-> 用 Lean MCP + lean_checker 校验
-> 把结果转成分数和反馈
-> 更新 archive / population
```

这就是当前代码库里 mathproof 的实际流程。
