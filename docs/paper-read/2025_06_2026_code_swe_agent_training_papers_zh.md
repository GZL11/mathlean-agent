# 2025-06 以来 Code / SWE Agent 训练论文汇总

本文只看 2025-06 到 2026-08 之间和 code / software engineering agent 训练直接相关的论文，重点放在：

- 这篇论文主要解决 **数据 / 算法 / 环境 / mid-training / test-time scaling** 哪一环；
- 用了什么模型或模型族；
- 训练信号是 SFT、RL/RLVR、DPO/Preference、OPD-like credit assignment，还是环境构造；
- 最后在 SWE-bench / code benchmark 上的指标如何。

## 1. 总览

| 论文 | 时间 | 侧重点 | 训练范式 | 模型 | 主指标 |
|---|---:|---|---|---|---|
| [SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling](https://arxiv.org/abs/2506.07636) | 2025-06 | 数据 + test synthesis + agent trajectories | SFT / inference scaling | open-source LLM，7B / 32B | SWE-bench Verified: 23.4 / 36.6 |
| [SWE-Factory](https://arxiv.org/abs/2506.10954) | 2025-06 | 自动构造 issue resolution 数据和可执行环境 | env / data factory | GPT-4.1 mini 用于 builder | 671 issues 中构造 337 valid instances，约 $0.047/instance |
| [Agent-RLVR](https://arxiv.org/abs/2506.11425) | 2025-06 | 稀疏 reward 下让 RLVR 能训 SWE agent | RLVR + guidance | Qwen2.5-72B-Instruct | SWE-bench Verified: 9.4 -> 22.4；RM TTS 到 27.8 |
| [Learning to Align Human Code Preferences](https://arxiv.org/abs/2507.20109) | 2025-07 | code preference 中 SFT / DPO 如何选 | SFT / DPO / APO | 多模型实验，摘要未强调单一 SWE 模型 | 6 类 code preference tasks，APO 持平或超过 SFT / SFT+DPO |
| [Long-Context Multi-Turn SWE Agents with RL](https://arxiv.org/abs/2508.03501) | 2025-08 | 长上下文多轮 agent RL pipeline | RFT + DAPO | Qwen2.5-72B-Instruct | SWE-bench Verified: 11 -> 39；RFT baseline 20 |
| [EntroPO](https://arxiv.org/abs/2509.12434) | 2025-09 | 多轮工具式 coding agent 的偏好优化 | SFT + entropy-enhanced DPO/KTO | 多模型，最大 106B；重点报告 30B | open-weight SWE-bench Lite 第 1，Verified 第 4 |
| [Kimi-Dev](https://arxiv.org/abs/2509.23045) | 2025-09 | agentless skill prior 迁移到 SWE-Agent | mid-training / RL / SFT adaptation | Kimi-Dev-72B，基于 Qwen2.5-72B | SWE-bench Verified 60.4；5k trajectories 后 agent pass@1 48.6 |
| [Agentic RL for Real-World Code Repair](https://arxiv.org/abs/2510.22075) | 2025-10 | 真实仓库 code repair 的可验证 RL | SFT + RL | Qwen3-32B，GPT-4.1 trajectories 蒸馏 | RL 在 matched train-test 下 +7-20% absolute |
| [DPO-F+](https://arxiv.org/abs/2511.01043) | 2025-11 | code repair feedback 对齐开发者偏好 | DPO + reward model | 摘要未强调 SWE patch 模型 | feedback accuracy / alignment 优于 baseline 和 standard DPO |
| [SWE-Lego](https://arxiv.org/abs/2601.01426) | 2026-01 | 把 SWE issue resolving 的 SFT 做到极限 | SFT-only + verifier TTS | Qwen3-8B / Qwen3-32B | Verified: 42.2 / 52.6；TTS@16: 49.6 / 58.8 |
| [daVinci-Dev](https://arxiv.org/abs/2601.18418) | 2026-01 | agent-native mid-training | mid-training + post-training | 32B / 72B | SWE-bench Verified: 56.1 / 58.5 |
| [SWE-World](https://arxiv.org/abs/2602.03419) | 2026-02 | Docker-free surrogate environment | surrogate env + SFT/RL/TTS | Qwen2.5-Coder-32B | Verified: 6.2 -> 52.0 SFT；55.0 RL；68.2 TTS |
| [daVinci-Env / OpenSWE](https://arxiv.org/abs/2603.13023) | 2026-03 | 大规模可执行 SWE 环境合成 | env + trajectory curation | OpenSWE-32B / 72B | 45,320 Docker envs；Verified: 62.4 / 66.0 |
| [Open-SWE-Traces](https://arxiv.org/abs/2606.16038) | 2026-06 | 多语言 agent 轨迹蒸馏数据 | SFT / distillation | Minimax-M2.5 / Qwen3.5-122B-A10B 生成轨迹 | 207,489 trajectories；Verified: 61.7；Multilingual: 57.1；SWE-bench Pro: 36.8 |
| [SWE-TRACE](https://arxiv.org/abs/2604.14820) | 2026-04 | 轨迹压缩 + PRM + TTS | SFT + rubric PRM + RL + TTS | 摘要未写明 backbone | 60K SFT corpus；强调 resolution/token/latency 同时优化 |
| [Function-Aware FIM Mid-Training](https://arxiv.org/abs/2607.12463) | 2026-07 | 给 coding agent 补 tool-return/continuation 归纳偏置 | self-supervised mid-training + existing post-training | Qwen2.5-Coder 7B/14B, Qwen3-8B | Verified +2.8/+3.0/+3.2；Lite +3.7/+4.0/+5.4 |
| [DiDPO](https://arxiv.org/abs/2608.07147) | 2026-08 | code diff 内部细粒度 credit assignment | RLVR / GRPO-like | Qwen2.5-Coder-7B 等 | 多个 code benchmark 上超过 GRPO/GiGPO；7B 提升超过 10% |

## 2. 主要趋势

### 2.1 数据和环境先行

SWE agent 的训练瓶颈首先不是 loss，而是可验证环境、测试、轨迹和失败反馈。`SWE-Gym`、`SWE-smith`、`SWE-Factory`、`SWE-World`、`daVinci-Env/OpenSWE`、`Open-SWE-Traces` 都在解决这个问题：

- `SWE-Gym` 提供最早一批真实可执行 SWE 训练环境；
- `SWE-smith` 把训练数据规模化到 50k 级别；
- `SWE-Factory` 自动构造 GitHub issue resolution 数据、测试补丁和可执行环境；
- `SWE-World` 用 learned surrogate 替代 Docker 物理执行，降低训练和评估成本；
- `OpenSWE` 直接规模化构造 45,320 个 Docker 环境，并开放 Dockerfiles、eval scripts 和基础设施。
- `Open-SWE-Traces` 则把多语言 agent trajectories 直接做成可蒸馏训练语料。

这说明 SWE agent 的 RL/SFT 能不能做起来，很大程度取决于有没有稳定、可复现、可批量运行的 environment。

### 2.2 SFT 仍然非常强

`SWE-Lego` 是这条线最明确的论文：它没有靠复杂 RL，而是用 32K 高质量 task instances、18K validated trajectories、error masking 和 difficulty curriculum，把 Qwen3-8B / 32B 推到 SWE-bench Verified 42.2 / 52.6。

这说明在 SWE 场景里，SFT 的作用不是简单教模型“写代码”，而是教它：

- 如何在 agent scaffold 中稳定使用工具；
- 如何定位文件、修改代码、运行测试；
- 如何控制轨迹长度和 action 格式；
- 如何避免把环境返回 token 当作 policy token 学。

### 2.3 RL 的核心是克服稀疏奖励

`Agent-RLVR` 和 `Long-Context Multi-Turn SWE Agents with RL` 都直接面对这个问题。SWE-bench 的最终 reward 很稀疏，普通 RLVR 很难从失败轨迹里学习。

两篇的处理方式不同：

- `Agent-RLVR` 用 guidance 帮 agent 走到更有学习信号的轨迹，再用 unit tests 给 reward；
- `Long-Context...` 先用 execution feedback 做 RFT，建立格式和工具使用能力，再用 DAPO 做同步 RL。

这和数学 RLVR 不一样：数学题通常最后 verifier 足够便宜，SWE 任务每条 rollout 都很长、环境很重、失败空间巨大。

### 2.4 Preference 优化开始转向多轮和多样性

普通 DPO/KTO 容易把 policy 压到少数 winning modes。对 SWE agent 来说，这会伤害 test-time scaling，因为多次采样出来的候选 patch 越来越相似。

`EntroPO` 的核心就是：把 preference optimization 扩到多轮 tool-use trajectory，并显式保留 policy entropy。它关心的不是单条 response 更像偏好答案，而是让 agent 在 TTS 时仍能产生多样候选轨迹。

`DPO-F+` 的关注点更窄：不是 patch 本身，而是 code repair feedback 是否符合不同开发者群体偏好。它对“人机协作式修复系统”有价值，但不是直接提升 SWE-bench patch resolve 的主线。

### 2.5 Mid-training 重新变重要

`Kimi-Dev`、`daVinci-Dev`、`Function-Aware FIM` 都说明：只靠 post-training 训 agent 不够，底座模型需要先获得更接近 agent workflow 的中间能力。

- `Kimi-Dev` 用 agentless workflow 学 localization、edit、self-reflection 等 skill prior；
- `daVinci-Dev` 强调 agent-native trajectories，包括 contextually-native 和 environmentally-native 两类；
- `Function-Aware FIM` 把函数调用结构类比成 agent 的 action-observation-continuation loop，用 FIM mid-training 补归纳偏置。

这条线的结论是：SWE agent 能力不只是 post-training 产物，mid-training 阶段就应该对齐 agent 的信息流。

### 2.6 OPD-like 方向目前更像 credit assignment，而不是纯蒸馏

严格意义上的 SWE-OPD 论文不多。`DiDPO` 虽然放在 OPD 目录下很自然，但它不是 teacher-student distillation；它是 RLVR 里的细粒度 credit assignment。

它的洞察很代码化：一次 agent action 可能同时改多个代码区域，最终测试通过/失败不能说明每个局部 diff 都好或坏。因此它把 diff 切成 sub-diff，在跨 rollout 相似 sub-diff 之间算局部 advantage，再投回 response tokens。

这和 OPD 的共同点是都在反对“整条 trajectory 一个 reward”；区别是 DiDPO 的信号来自 executable reward，不来自 teacher logits。

## 3. 逐篇解读

### 3.1 SWE-Dev: Building Software Engineering Agents with Training and Inference Scaling

论文：<https://arxiv.org/abs/2506.07636>

侧重点：**造训练数据 + 合成测试 + agent trajectory scaling**。

这篇的目标是构建 open SWE agent，而不是提出一个新 RL loss。它先做 robust test-case synthesis，用来评估 patch；再规模化生成 agent trajectories 作为训练数据。

模型与训练：

- 模型规模：7B / 32B open-source LLM；
- 训练方式：以 agent trajectories 为核心做 SWE agent 训练；
- 评测：SWE-bench Verified。

主结果：

| 模型 | SWE-bench Verified |
|---|---:|
| SWE-Dev 7B | 23.4 |
| SWE-Dev 32B | 36.6 |

阅读重点：这篇更像“如何把 SWE agent 的训练数据和测试信号做出来”，而不是算法论文。

### 3.2 SWE-Factory

论文：<https://arxiv.org/abs/2506.10954>

侧重点：**自动化 issue resolution 数据和 evaluation environment 工厂**。

它解决三个工程问题：

1. test patch 可能漏掉 binary file changes；
2. evaluation environment 手工构造成本高；
3. fail2pass validation 需要人工读日志、写解析脚本。

方法：

- 自动恢复缺失 binary test files；
- SWE-Builder：LLM-based multi-agent system，自动构造 eval environment；
- 用 exit-code-based log parsing 标准化 fail2pass validation。

指标：

| 项目 | 数值 |
|---|---:|
| GitHub issues | 671 |
| valid task instances | 337 |
| GPT-4.1 mini cost | $0.047 / instance |
| fail2pass validation F1 | 0.99 |

阅读重点：如果要自己做 SWE RL 环境，这篇比很多训练算法更基础。

### 3.3 Agent-RLVR

论文：<https://arxiv.org/abs/2506.11425>

侧重点：**让 RLVR 在稀疏、多步 agentic SWE 环境里可训练**。

普通 RLVR 在 SWE agent 上的问题是：大多数 rollout 失败，reward 太稀疏，模型很难知道怎么改。Agent-RLVR 引入 agent guidance：高层计划、错误反馈、环境交互线索等，帮助模型采到更接近成功的轨迹。

训练流程：

```text
agent first attempt
  -> unit test validation
  -> add guidance
  -> guided reattempt
  -> RLVR update on guided trajectory reward
```

模型与指标：

| 模型 | 设置 | SWE-bench Verified pass@1 |
|---|---|---:|
| Qwen2.5-72B-Instruct | base | 9.4 |
| Qwen2.5-72B-Instruct | Agent-RLVR | 22.4 |
| Qwen2.5-72B-Instruct | + test-time reward model | 27.8 |

阅读重点：这篇不是改 PPO/GRPO 公式，而是通过 guidance 改善探索和 reward density。

### 3.4 Learning to Align Human Code Preferences

论文：<https://arxiv.org/abs/2507.20109>

侧重点：**code preference 场景里什么时候用 SFT，什么时候用 DPO**。

它不是 SWE-bench agent 论文，而是更一般的 code preference alignment 论文。核心结论：

- 有客观可验证最优解时，SFT 通常更合适；
- 没有客观最优解、偏好更主观时，SFT + DPO 更能探索更优解；
- 提出 APO，把 preferred amplification、dispreferred suppression 和 exploration 动态结合。

训练与指标：

- 范式：SFT / DPO / APO；
- 评测：6 类 code preference tasks；
- 结论：APO 在这些任务上稳定持平或超过 SFT 与 SFT+DPO。

阅读重点：它给 code DPO 选型提供方法论，但不直接回答 SWE-bench agent 怎么训。

### 3.5 Training Long-Context, Multi-Turn SWE Agents with RL

论文：<https://arxiv.org/abs/2508.03501>

侧重点：**真正把 RL 跑到长上下文、多轮 SWE agent 上**。

方法流程：

```text
RFT with execution feedback
  -> teach instruction / format / tool-use stability
  -> synchronous RL with DAPO
  -> iterative improvement
```

模型与指标：

| 模型 | 设置 | SWE-bench Verified pass@1 |
|---|---|---:|
| Qwen2.5-72B-Instruct | initial | 11 |
| Qwen2.5-72B-Instruct | RFT baseline | 20 |
| Qwen2.5-72B-Instruct | RFT + DAPO RL | 39 |

额外评测：

| Benchmark | Pass@1 |
|---|---:|
| SWE-rebench May | 35 |
| SWE-rebench June | 31 |

阅读重点：这是最值得看的一篇 SWE RL pipeline 论文。它证明 open-weight 72B 通过 RFT+RL 可以接近更大模型的 SWE agent 表现。

### 3.6 EntroPO

论文：<https://arxiv.org/abs/2509.12434>

侧重点：**多轮工具式 coding agent 的 preference optimization，同时保留多样性**。

问题：

```text
DPO/KTO 能对齐偏好，
但容易降低生成多样性，
导致 TTS 多采样时候选轨迹重复。
```

方法：

- 把 preference optimization 从 single-turn response 扩到 multi-turn tool-use trajectory；
- 在目标里加入 entropy regularization，避免 policy collapse；
- 推理时用 hybrid best-trajectory selector：learned verifier + model-free methods。

模型与指标：

- fine-tune 多种模型族和规模，最大到 106B；
- 重点结果：30B EntroPO 模型在 open-weight SWE-bench leaderboard 上 Lite 第 1、Verified 第 4。

阅读重点：这篇适合看 DPO/KTO 怎么进入多轮 agent。它服务的是 test-time scaling，不只是单次 pass@1。

### 3.7 Kimi-Dev

论文：<https://arxiv.org/abs/2509.23045>

侧重点：**把 agentless workflow 的 skill prior 迁移给 SWE-Agent**。

Kimi-Dev 认为 Agentless 和 SWE-Agent 不是对立路线。Agentless 的 localization、edit、self-reflection 等能力，可以作为 SWE-Agent 的 skill prior。

模型与训练：

- 模型：Kimi-Dev-72B，模型卡显示 base model 为 Qwen/Qwen2.5-72B；
- 训练：agentless training recipe；后续用 5k publicly-available trajectories 做 SFT adaptation 给 SWE-Agent；
- 评测：SWE-bench Verified。

指标：

| 设置 | SWE-bench Verified |
|---|---:|
| Kimi-Dev workflow | 60.4 |
| Kimi-Dev + 5k trajectories into SWE-Agent | 48.6 pass@1 |

阅读重点：这篇的关键是“先训 agentless 能力，再适配 agentic scaffold”，不是一开始就完全靠多轮环境 RL。

### 3.8 Agentic Reinforcement Learning for Real-World Code Repair

论文：<https://arxiv.org/abs/2510.22075>

侧重点：**真实仓库 code repair 的可验证训练闭环**。

它强调真实仓库环境的问题：build 复杂、依赖漂移、evaluation 不稳定。因此先构建可验证 pipeline，通过 pin dependencies 和禁止自动升级来提升复现性。

模型与训练：

- SFT：Qwen3-32B，蒸馏 GPT-4.1 trajectories；
- RL：在 simplified environment 上接着训；
- reward：post-fix build validation。

指标：

- SFT 模型与 GPT-4.1 trajectories 表现相当，但模型小 56 倍；
- 在 train-test environment matched 的条件下，RL 带来 7-20% absolute gains；
- 跨环境泛化失败明显，说明环境匹配是 code repair RL 的关键。

阅读重点：它最重要的结论不是“RL 一定泛化”，而是“真实 code repair 的训练-测试环境 mismatch 会毁掉收益”。

### 3.9 DPO-F+

论文：<https://arxiv.org/abs/2511.01043>

侧重点：**对齐 code repair feedback，而不是直接优化 patch**。

它关注 human-AI teaming：模型给出的自然语言反馈能否帮助不同水平开发者理解和迭代修复。它定义七个固定维度的 feedback alignment metric，自动构造 pairwise preference data，再用 DPO + reward model 训练。

训练与指标：

- 范式：DPO augmented with reward model；
- 输出对象：code-repair feedback；
- 结果：在 feedback accuracy 和 overall alignment 上优于 baseline 和 standard DPO。

阅读重点：如果场景是 IDE assistant / code review assistant，这篇有价值；如果目标是 SWE-bench resolved rate，它不是主线。

### 3.10 SWE-Lego

论文：<https://arxiv.org/abs/2601.01426>

侧重点：**SFT-only recipe 的上限**。

组成：

1. 32K high-quality task instances；
2. 18K validated trajectories；
3. real + synthetic data 混合；
4. error masking；
5. difficulty-based curriculum；
6. verifier-based TTS。

模型与指标：

| 模型 | SFT | TTS@16 |
|---|---:|---:|
| SWE-Lego-Qwen3-8B | 42.2 | 49.6 |
| SWE-Lego-Qwen3-32B | 52.6 | 58.8 |

阅读重点：它提醒我们，很多 SWE agent 的“RL 收益”其实建立在 SFT foundation 上。没有好的 trajectory 数据和训练 mask，RL 很难稳定。

### 3.11 daVinci-Dev

论文：<https://arxiv.org/abs/2601.18418>

侧重点：**agent-native mid-training**。

它认为 post-training 已经是 SWE agent 的默认做法，但更可扩展的是 mid-training：在大规模数据上让模型提前适应 agentic workflow。

数据分两类：

- contextually-native trajectories：保留完整信息流，提供广覆盖；
- environmentally-native trajectories：来自可执行仓库和真实 tool/test observations，提供交互真实性。

模型与指标：

| 模型规模 | SWE-bench Verified |
|---|---:|
| 32B | 56.1 |
| 72B | 58.5 |

论文还强调：相对 Kimi-Dev，它用少于一半的 mid-training tokens，即 `73.1B`，在 aligned base model 和 agentic scaffold 下更强。

阅读重点：这是 mid-training 路线的代表，不是单纯 SFT 或 RL recipe。

### 3.12 SWE-World

论文：<https://arxiv.org/abs/2602.03419>

侧重点：**Docker-free surrogate environment**。

问题：

```text
真实 Docker 环境昂贵、依赖复杂、维护困难；
但 agent 训练又需要 execution feedback。
```

SWE-World 用 LLM-based learned surrogate 预测中间 execution outcomes 和最终 test feedback，保留 agent-environment loop，但减少物理容器成本。

模型与指标：

| 设置 | SWE-bench Verified |
|---|---:|
| Qwen2.5-Coder-32B base | 6.2 |
| Docker-free SFT | 52.0 |
| Docker-free RL | 55.0 |
| + TTS | 68.2 |

阅读重点：它是 env/simulator 方向，不是 loss 方向。对大规模 SWE RL 来说，surrogate environment 可能比换优化器更重要。

### 3.13 daVinci-Env / OpenSWE

论文：<https://arxiv.org/abs/2603.13023>

侧重点：**开源可执行 SWE 环境大规模合成**。

核心产物：

- 45,320 executable Docker environments；
- 覆盖 12.8K+ Python repos；
- 开源 Dockerfiles、evaluation scripts、infrastructure；
- multi-agent synthesis pipeline；
- 64-node distributed cluster；
- quality-centric filtering；
- 13,000 curated trajectories from roughly 9,000 quality-guaranteed environments。

成本与指标：

| 项目 | 数值 |
|---|---:|
| environment construction cost | $891K |
| trajectory sampling + curation | $576K |
| total investment | about $1.47M |
| OpenSWE-32B Verified | 62.4 |
| OpenSWE-72B Verified | 66.0 |

阅读重点：这篇本质是“工业级 SWE RL 环境怎么开放复现”。它说明 SWE agent 的护城河很大一部分在 environment synthesis。

### 3.14 SWE-TRACE

论文：<https://arxiv.org/abs/2604.14820>

侧重点：**trajectory reduction + rubric PRM + heuristic TTS**。

它把 SWE agent lifecycle 分成三段一起优化：

1. data curation：用 LLM multi-task cascading 和 stepwise oracle verification，蒸馏 60K token-efficient shortest-path SFT corpus；
2. RL：Memory-Augmented Agentic RL + Rubric-Based PRM，给 long-horizon steps 提供 dense heuristic feedback；
3. inference：把 PRM 复用到 test-time scaling，动态评估和剪枝 action candidates。

指标：

- 摘要强调在 standard SWE benchmarks 上推进 SOTA；
- 同时降低 token consumption 和 inference latency；
- arXiv 摘要页没有给出具体 resolved-rate 数值，精确表格建议读 PDF 主表。

阅读重点：这篇适合和 ECHO / ABSeeker 一起看，都是在解决长程 agent 的过程监督和信用分配，只是落到 SWE。

### 3.15 Function-Aware FIM Mid-Training

论文：<https://arxiv.org/abs/2607.12463>

侧重点：**用代码本身的函数结构给 coding agent 做 mid-training**。

核心类比：

```text
coding agent: action -> tool observation -> continuation
function call: caller binds args -> callee returns value -> downstream code consumes value
```

因此它用 function-aware FIM 做 self-supervised mid-training：通过 program dependency graph 和 complexity-inferability criterion 选择函数 mask，让模型根据周围代码重建函数体。

模型和数据：

- Qwen2.5-Coder-Instruct 7B / 14B；
- Qwen3-8B；
- 2.6B-token decontaminated corpus；
- 来自 968 GitHub repositories；
- 再接 R2E-Gym / SWE-Smith / SWE-Lego 等 agentic post-training pipeline。

指标：

| 模型 | SWE-Bench Verified 增益 | SWE-Bench Lite 增益 |
|---|---:|---:|
| Qwen2.5-Coder 7B | +2.8 | +3.7 |
| Qwen2.5-Coder 14B | +3.0 | +4.0 |
| Qwen3-8B | +3.2 | +5.4 |

额外结论：它还能缓解 agentic post-training 对 LiveCodeBench、tau-bench、BFCL 等能力的侵蚀。

阅读重点：这是很干净的 mid-training idea：不用先造 SWE trajectory，而是从大规模普通代码中挖 agent workflow 的结构同构。

### 3.16 DiDPO

论文：<https://arxiv.org/abs/2608.07147>  
本地详解：[DiDPO 文档](OPD/didpo_diff_in_diff_policy_optimization_coding_agent_training_zh.md)

侧重点：**coding agent RL 里的 diff-level credit assignment**。

普通 GRPO 对一条 rollout 给一个 advantage；但代码 action 往往一次改多个位置。DiDPO 把完整 diff 拆成 sub-diff，在跨 rollout 相似 sub-diff 之间构造 advantage group，再把局部 diff advantage 投回生成该 sub-diff 的 tokens。

训练和模型：

- 范式：critic-free RLVR / GRPO-like objective；
- 模型：Qwen2.5-Coder-7B 等；
- 不需要 teacher，不是 OPD loss；
- 训练时额外做 sub-diff matching，推理无额外成本。

评测：

- long-horizon coding and reasoning benchmarks；
- 本地文档里记录的代码 benchmark 包括 APPS、HumanEval、MBPP、LiveCodeBench、LeetCode、USACO、OJBench、ICPC；
- 摘要称在 Qwen2.5-Coder-7B 上超过 comparable methods over 10%。

阅读重点：它不是造数据，也不是 env。它是目前最像“代码 agent 专用 RL 算法”的论文。

## 4. 前置与扩展

### 4.1 前置工作：SWE-Gym / SWE-smith

这两篇严格说早于本文的时间窗，但它们是后续 SWE agent 训练路线的直接前置。

**SWE-Gym**

- 论文：<https://arxiv.org/abs/2412.21139>
- 侧重点：最早一批真实可执行 SWE 训练环境；
- 数据：2,438 个真实 Python 任务实例，每个都带 executable runtime、unit tests 和自然语言任务描述；
- 训练：基于 SWE-Gym 训练 SWE agents 与 verifiers；
- 指标：摘要报告在 SWE-bench Verified / Lite 上最高可达 19% absolute gains，结合 verifier 后达到 32.0 / 26.0。

**SWE-smith**

- 论文：<https://arxiv.org/abs/2504.21798>
- 侧重点：训练数据规模化；
- 数据：50k task instances，来自 128 个 GitHub repositories；
- 方法：自动为任意 Python codebase 构造 execution environment，并 synthesize 大量破坏性任务；
- 指标：训练出的 SWE-agent-LM-32B 在 SWE-bench Verified 上达到 40.2% Pass@1。

**Hybrid-Gym**

- 论文：<https://arxiv.org/abs/2602.16819>
- 侧重点：把 SWE 训练环境推广成可组合的通用 agent gym；
- 数据/环境：任务来自软件工程与相关交互式环境；
- 训练：更偏向训练环境与任务编排，不是单纯 loss 设计。

**SWE-Mirror**

- 论文：<https://arxiv.org/abs/2509.18059>
- 侧重点：把真实 agent traces 镜像成可复用训练语料；
- 数据：更强调日志、轨迹与失败案例的镜像化整理；
- 阅读重点：适合和 Open-SWE-Traces 一起看，都是“轨迹即数据”的路线。

这几篇基本把后续工作分成两条线：

1. 环境/轨迹规模化；
2. 在规模化数据上继续做 SFT / RL / distillation。

### 4.2 2026-06 的数据扩展：Open-SWE-Traces

论文：<https://arxiv.org/abs/2606.16038>  
数据卡：<https://huggingface.co/datasets/nvidia/Open-SWE-Traces>

侧重点：**多语言 agent trajectory 蒸馏数据**。

它是 NVIDIA 做的大规模轨迹数据集，目标不是造新环境，而是给 SWE agent 提供可直接用于 SFT / distillation 的多轮 agentic traces。数据特点：

- 207,489 条 agentic trajectories；
- 覆盖 9 种语言：Python、Go、TS、JS、Rust、Java、PHP、C、C++；
- 来源于 20,000 个真实 PR；
- 轨迹由 OpenHands 和 SWE-agent harness 采集；
- 使用 dual-mode synthesis：Minimax-M2.5 生成带 thinking 的轨迹，Qwen3.5-122B-A10B 生成 non-thinking traces。

验证结果：

| 模型 | SWE-bench Verified | SWE-bench Multilingual | SWE-bench Pro |
|---|---:|---:|---:|
| Qwen3-30B-A3B 系列最优模型 | 61.7 | 57.1 | 36.8 |

阅读重点：这篇是 data/distillation 方向的代表，和 SWE-smith 的关系很像，但更强调多语言、双模式轨迹蒸馏和 open-source agent distillation。

### SWE-Dev: Evaluating and Training Autonomous Feature-Driven Software Development

论文：<https://arxiv.org/abs/2505.16975>

这篇 v1 是 2025-05，严格说早于本文范围；但 v2/v3 落在 2025-06 之后，并且非常相关。它构造 14,000 training / 500 test samples，关注 feature-driven development，即在已有大型 codebase 中开发新功能，而不只是 bug fixing。

它提供 runnable environment 和 developer-authored executable unit tests，因此既支持 SFT，也支持 RL。摘要报告：best single-turn model 在 hard split 上 Pass@1 只有 22.51%，OpenHands agents 到 56.44%，说明 feature-driven SWE 仍然很难。

## 5. 怎么读这批论文

如果目标是训练真实 SWE agent，我建议阅读顺序是：

1. **环境/数据基础**：SWE-Gym -> SWE-smith -> SWE-Factory -> SWE-World -> daVinci-Env/OpenSWE -> Open-SWE-Traces；
2. **SFT 基线**：SWE-Dev -> SWE-Lego；
3. **RL 主线**：Agent-RLVR -> Long-Context Multi-Turn SWE RL -> Real-World Code Repair；
4. **偏好优化**：EntroPO -> DPO-F+ -> Learning to Align Human Code Preferences；
5. **mid-training**：Kimi-Dev -> daVinci-Dev -> Function-Aware FIM；
6. **细粒度 credit**：DiDPO -> SWE-TRACE。

一句话总结：

> 2025-06 以来 code/SWE agent 的核心趋势不是某一个新 loss 统治全场，而是 **环境可验证化、轨迹规模化、SFT 做强基础、RL 解决稀疏奖励、preference 保持多样性、mid-training 对齐 agent 信息流、credit assignment 进入 diff/step/token 内部**。
