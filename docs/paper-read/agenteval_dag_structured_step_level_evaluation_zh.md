# AgentEval：DAG-Structured Step-Level Evaluation 技术解读

本文档解读论文：

```text
AgentEval: DAG-Structured Step-Level Evaluation for Agentic Workflows with Error Propagation Tracking
https://arxiv.org/abs/2604.23581
```

## 一句话总结

AgentEval 不是训练算法，也不是 OPD / RL 方法，而是一个 **agent 工作流评测基础设施**。

它解决的问题是：

```text
只看最终答案是否正确，会掩盖中间步骤错误；
逐步看 trace，又很难判断哪个错误是 root cause，哪个只是上游错误传播。
```

AgentEval 的做法是：

```text
把 agent 执行轨迹建成 evaluation DAG；
每个节点按 step type 用 LLM-as-judge 评分；
如果下游节点失败且上游父节点也失败，就标记为 propagated failure；
如果没有失败父节点，就标记为 root cause。
```

所以它的核心不是“judge 更强”，而是：

```text
用 DAG dependency modeling 做错误传播和根因定位。
```

## 系统设置速览

论文评测的是三个生产 agent workflow：

```text
CS:
  customer service

DA:
  data analysis

DP:
  document processing
```

agent 模型：

```text
Claude 3.5 Sonnet
Llama 3 70B
```

judge 模型：

```text
GPT-4o
gpt-4o-2024-08-06
temperature = 0
```

作者特意让 agent LLM 和 judge LLM 来自不同模型族，降低 self-judging / circular bias。

## 为什么 end-to-end evaluation 不够

传统 agent 评测常见做法：

```text
1. 只看最终输出好不好。
2. 人手动检查 trace。
3. 在静态 benchmark 上跑成功率。
```

问题是长程 agent workflow 里，错误经常发生在中间：

```text
规划少了一步；
选错工具；
参数传错；
工具返回结果被误读；
最终 synthesis 幻觉。
```

如果只看 final answer，有两类信息会丢：

```text
1. 中间失败但最终被补救了。
2. 最终失败，但真正根因在更早步骤。
```

AgentEval 的观点是：

```text
agent evaluation 需要像软件故障定位一样，
不仅判断失败，还要定位失败传播链。
```

## Evaluation DAG 定义

论文把 agent 工作流表示为：

```text
G = (V, E, tau, M)
```

其中：

```text
V:
  evaluation nodes，每个节点对应一个 agent step。

E:
  有向依赖边。

tau:
  节点类型映射。

M:
  节点对应的 metric set。
```

节点类型包括 5 类：

```text
PLAN:
  规划步骤。

TOOLSEL:
  工具选择。

PARAMGEN:
  参数生成。

EXEC:
  工具执行。

SYNTH:
  综合/回答生成。
```

每个节点 `v_i` 有：

```text
c_i:
  来自父节点的上下文。

o_i:
  agent 当前 step 输出。

r_i:
  可选 reference。

q(v_i):
  当前节点的质量分数。
```

## step-level metrics

不同节点类型使用不同评价指标：

```text
PLAN:
  completeness
  feasibility

TOOLSEL:
  selection accuracy
  relevance

PARAMGEN:
  parameter correctness
  completeness

EXEC:
  success rate
  result validity

SYNTH:
  faithfulness
  completeness
  coherence
```

这些指标由 GPT-4o 根据结构化 rubric 打 1-5 分。

论文强调一个细节：

```text
PLAN 相对原始用户 query 评分；
后续步骤相对它收到的 local context 评分。
```

这很重要，因为下游步骤不应该因为上游给了错误输入而被无脑判错。

如果下游步骤很好地处理了错误上游输入：

```text
不标记为 local failure。
```

如果下游步骤进一步放大了上游错误：

```text
标记为 propagated failure。
```

## failure taxonomy

论文从 523 条 agent traces 中人工归纳了三层 failure taxonomy。

Level 1 包括：

```text
Planning
Execution
Integration
```

Level 2 包括 9 类：

```text
Goal misinterpretation
Missing steps
Incorrect ordering
Wrong tool selection
Parameter errors
API/tool failures
Context loss
Output hallucination
Premature termination
```

论文还给出 21 个 Level 3 子类。

一个关键统计：

```text
Context loss 的 downstream amplification factor = 3.2x
```

这说明上下文丢失这种错误特别容易向后传播，因此 DAG-based attribution 很有用。

## 核心算法

AgentEval 的流程：

```text
1. ParseDAG(trace, schema)
2. 对 DAG 做 topological sort
3. 对每个节点聚合 parent context
4. 用 LLM judge 给当前节点打分
5. 如果分数低于该类型阈值，标记 failure
6. 如果存在失败 parent:
     当前 failure 标为 propagated
     propagated from 最低分 parent
   否则:
     当前 failure 标为 root cause
7. 聚合 step scores、failure labels、root causes
```

root cause attribution 是一个简单 greedy heuristic：

```text
多个 parent 都失败时，选最低分 parent 作为传播源。
```

论文明确说这不是正式 causal inference，而是工程上可用的故障定位启发式。

## Non-DAG trace 怎么处理

论文内部 agent 大约：

```text
12% traces 是 non-DAG
```

主要包括：

```text
retry loops:
  8%

dynamic branching:
  4%
```

处理方式：

```text
retry loops:
  unroll 成顺序 attempts。

dynamic branches:
  用 timestamp 解析实际执行路径。

完全无法重建:
  fallback 到 flat step evaluation。
```

完全无法重建的比例约：

```text
0.8%
```

## 自动 regression suite

AgentEval 也不是单次离线评测，而是面向 CI/CD 的持续评测：

```text
test case 用 JSON 定义 expected DAG 和阈值；
每次 evaluation 记录 model / prompt / tool config；
使用 paired bootstrap test 检测 regression；
GitHub Actions 里阻断 critical regressions；
先跑 smoke test，再跑 full suite。
```

论文说 fast smoke tests：

```text
10 cases
< 5 minutes
```

full suite：

```text
100+ cases
< 1 hour
```

这种 progressive evaluation 在开发阶段降低约 80% 成本。

## 实验设置

数据分两套，完全 disjoint：

```text
taxonomy development set:
  523 traces
  2025-01 到 2025-03

evaluation dataset:
  450 test cases
  3 个 workflow，每个 150
  2025-04 到 2025-06
```

人工标注：

```text
5 位 domain experts 标注 ground truth；
human evaluation subset:
  150 traces
  987 step annotations

failing steps:
  195 个 step 被 human majority 标为 <= 2 分。
```

对比方法：

```text
E2E Only:
  只看最终输出。

Flat Step:
  每步独立评估，但不建 DAG。

Rule-Based:
  47 条手写规则。

AgentEval:
  DAG + LLM judge + taxonomy + RCA。
```

## 主要指标

论文用四个指标：

```text
FDRec:
  Failure Detection Recall，失败步骤召回率。

FPR:
  False Positive Rate。

HA:
  Human Agreement，Cohen's kappa。

RCA:
  Root Cause Accuracy。
```

## 主要结果

主表结果：

```text
Method        FDRec   FPR   HA(kappa)   RCA
E2E Only      0.41   0.08   0.52        N/A
Flat Step     0.67   0.15   0.71        0.38
Rule-Based    0.58   0.05   0.63        0.45
AgentEval     0.89   0.07   0.84        0.72
```

关键结论：

```text
AgentEval failure recall 是 E2E 的 2.17 倍。
比 Flat Step 高 22 个百分点。
root cause accuracy 达到 72%，接近 human ceiling 81%。
```

论文还指出：

```text
63% step-level failures 是 propagated failures，
不是 local root cause。
```

这解释了为什么仅仅 flat step evaluation 不够。

## 消融实验

消融结果：

```text
Full AgentEval:
  FDRec 0.89
  HA    0.84
  RCA   0.72

- DAG structure:
  FDRec 0.67
  HA    0.71
  RCA   0.38

- LLM judge, replaced with rules:
  FDRec 0.62
  HA    0.66
  RCA   0.51

- failure taxonomy:
  FDRec 0.82
  HA    0.79
  RCA   0.54

- calibration:
  FDRec 0.85
  HA    0.76
  RCA   0.68
```

最大贡献来自：

```text
DAG structure
```

去掉 DAG 后：

```text
FDRec -22 pp
RCA   -34 pp
```

## cross-system generalization

作者还测试了外部 benchmark：

```text
tau-bench:
  120 traces

SWE-bench:
  80 traces
```

不改 taxonomy / rubric 的情况下：

```text
tau-bench Retail:
  FDRec 0.83
  RCA   0.61

tau-bench Airline:
  FDRec 0.79
  RCA   0.55

SWE-bench Code editing:
  FDRec 0.78
  RCA   0.52
```

结论：

```text
failure detection 还能迁移；
root cause attribution 掉得更多，
因为新领域里会出现 taxonomy 没覆盖的新失败模式。
```

## 部署试点

4 个月 pilot：

```text
时间:
  2025-09 到 2026-01

参与:
  18 engineers

evaluated traces:
  12,847

unique evaluation runs:
  342

pre-release regressions detected:
  23

genuine / borderline / false:
  8 / 12 / 3
```

工程收益：

```text
root-cause time:
  4.2 hours -> 22 minutes

CS failure rate:
  31% -> 18%

DA failure rate:
  27% -> 15%
```

但论文也提醒：

```text
before / after 使用了不同测量方式，
改善幅度应谨慎解读。
```

## 成本和部署形态

AgentEval 作为 sidecar service：

```text
trace collection:
  OpenTelemetry-compatible

latency overhead:
  < 2%

step-level evaluation:
  about 2 seconds per step

cost per trace:
  about $0.02 with GPT-4o-mini
```

大规模场景：

```text
100K traces/day:
  about $2K/day with GPT-4o-mini
```

因此 release-gating 才建议多 judge aggregation，普通开发阶段用 progressive evaluation 控成本。

## 适用边界

AgentEval 最适合：

```text
顺序型 tool-calling workflow；
中等分支的生产 agent；
可从 trace 中重建依赖关系的系统。
```

不太适合：

```text
高度动态 multi-agent 系统；
长时间无界 reasoning loop；
没有显式 dependency declaration 的并行 agent handoff。
```

论文模拟显示：

```text
non-DAG trace rate <= 60%:
  DAG advantage 仍然明显。

non-DAG trace rate > 60%:
  DAG advantage 变得很小。
```

## 和训练类论文的关系

这篇和 Uni-OPD / SDAR / SIRI / DOPD 不在同一层。

```text
Uni-OPD / SDAR / SIRI / DOPD:
  训练方法，目标是提高模型或 agent 能力。

AgentEval:
  评测和诊断基础设施，目标是发现失败、定位根因、支撑 regression testing。
```

但它可以和训练方法结合：

```text
1. 用 AgentEval 标注失败步骤和根因。
2. 把 root cause steps 转成训练数据或 reward signal。
3. 对高频 failure taxonomy 做 targeted fine-tuning / RL。
```

论文结尾也提到未来方向之一是：

```text
把 evaluation signals 集成进 agent training loop。
```

## 我的理解

这篇论文的价值在于把 agent evaluation 从：

```text
最终答案打分
```

推进到：

```text
结构化流程诊断。
```

真正关键的是 DAG：

```text
没有 DAG，你只能知道哪个 step 看起来错了；
有 DAG，你才能判断这个 step 是自己错，还是被上游带错。
```

对于生产 agent，后一件事更重要，因为工程师要修的是 root cause，而不是所有 downstream symptom。

