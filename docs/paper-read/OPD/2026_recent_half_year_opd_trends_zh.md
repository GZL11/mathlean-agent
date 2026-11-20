# 2026 最近半年 OPD 热点综述

统计窗口：

```text
2026-02-02 到 2026-08-02
```

这份文档不追求穷举所有带 `OPD` 字样的论文，而是整理最近半年比较明显的研究主线：这些论文分别发现了 OPD 的什么问题，又把改进目标放在了哪里。

## 总结先行

最近半年 OPD 的研究重心已经从：

```text
student rollout + teacher token-level KL 是否有效
```

转向：

```text
1. teacher signal 是否可靠；
2. privileged information 是否造成不可学习的信息差；
3. OPD 如何和 RLVR / verifier reward 配合；
4. 多轮 agent / tool 场景里如何降低 rollout 和 tool-call 成本；
5. OPD 如何支持 black-box teacher、跨 tokenizer、跨模态、跨架构；
6. OPD 是否必须只在 output logits 空间做蒸馏。
```

换句话说，OPD 正在从一个简单的蒸馏目标，变成一种 post-training 框架：

```text
on-policy 数据生成
+ teacher / verifier / rubric / representation 多源监督
+ reliability-aware weighting
+ RL-style credit assignment
+ systems-level rollout acceleration
```

## 代表论文速览

| 时间 | 论文 | 主要发现的问题 | 改进目标 |
|---|---|---|---|
| 2026-03 | X-OPD: Cross-Modal On-Policy Distillation | Speech LLM 相比文本 LLM 有 modality gap，SFT/RL 难以完全补齐 | 让语音 student 在自己分布上 rollout，由文本 teacher 给 token-level feedback |
| 2026-03 | VLA-OPD | VLA 模型 offline SFT 有分布偏移和遗忘，online RL 奖励稀疏、样本效率低 | 用 expert teacher 在 student 自生成轨迹上提供 dense token-level supervision |
| 2026-04 | RLSD: Self-Distilled RLVR | OPSD 的 privileged teacher signal 直接当训练方向会泄漏信息、长期不稳定 | RLVR/verifier reward 决定方向，self-distillation 只做 token-level 幅度调节 |
| 2026-05 | ROPD: Rubric-based OPD | logit-based OPD 依赖 white-box teacher logits | 用 teacher-student contrast 生成 rubric，再用 rubric 给 student rollout 打分 |
| 2026-05/06 | TrOPD | teacher/student 分布差大时，student tokens 上的 teacher gradient 不可靠 | trust region + outlier 处理 + teacher-prefix guidance |
| 2026-06 | OPRD: On-Policy Representation Distillation | output-space OPD 有采样方差和 LM-head 信息瓶颈 | 在 hidden-state / representation 空间做 on-policy distillation |
| 2026-06 | Rubric-Conditioned Self-Distillation | 参考 CoT 昂贵且可能有噪声，RLVR scalar reward 太粗 | 用 rubrics 作为结构化细粒度反馈，指导 on-policy self-distillation |
| 2026-06 | AsyncOPD | synchronous OPD 中 rollout 生成占训练时间大头 | rollout 与 learner 异步化，研究 stale rollout / stale teacher score 的影响 |
| 2026-06 | MOPD | 多能力 RL teacher 难以合并，Mix-RL/Off-policy finetune 易低效或掉能力 | 每个领域独立训练 RL teacher，再通过 OPD 合入一个 student |
| 2026-06 | DOPD | privileged information 会制造 `privilege illusion` | token-level 动态路由 teacher distillation / student self-distillation |
| 2026-07 | ReOPD | multi-turn agent OPD 需要在线环境交互和 tool calls，成本高 | replay teacher prefix，student 只在当前 turn rollout，训练阶段 0 tool calls |
| 2026-07 | Direct-OPD | weak post-RL teacher 的最终 policy 混有小模型能力上限，直接模仿会拉低 strong student | 蒸馏 weak teacher 的 RL-induced policy shift，而不是最终 policy |
| 2026-07 | TurnOPD | long-horizon agent OPD 的 rollout 和 loss budget 都错配 | adaptive rollout-depth budgeting + turn-normalized loss budgeting |

## 主线 1：从构造 Teacher 到判断 Teacher 是否可靠

早期 OPD 的默认假设是：

```text
teacher 比 student 强，
所以 teacher 在 student rollout 上给出的 token-level distribution 就值得学。
```

最近半年这条假设被反复拆开。

### OPSD / RLSD：同一个模型能不能当 teacher

OPSD 的思路是：

```text
student:
  p_theta(. | x)

teacher:
  p_theta(. | x, r)
```

其中 `r` 是 privileged information，中文可以翻成：

```text
特权信息
```

例如参考答案、verified reasoning trace、环境反馈、rubric 等。teacher 不是更大的模型，而是同一个模型在训练时多看了信息。

这解决了一个问题：

```text
不需要外部大 teacher，也能得到 dense token-level supervision。
```

但 RLSD 指出这里有严重风险：

```text
teacher 的强可能来自 r；
student 测试时看不到 r；
直接匹配 teacher distribution 会造成 privileged information leakage。
```

RLSD 的改法是：

```text
verifier reward / RLVR:
  决定更新方向。

self-distillation teacher signal:
  只决定 token-level 更新幅度。
```

所以 RLSD 的贡献不是“重新构造 teacher”，而是定义了 self-distillation 在 RLVR 里的合理位置：

```text
self-distillation as credit assignment helper
```

可以翻成：

```text
自蒸馏作为 token 级信用分配辅助器。
```

### DOPD：privileged teacher 不一定真强

DOPD 进一步把问题命名为：

```text
privilege illusion
```

可以翻成：

```text
特权幻觉
```

意思是：

```text
teacher/student 的表现差距里，
一部分是可迁移的能力差距；
另一部分只是 teacher 多看了 privileged input 的信息差。
```

DOPD 的改进目标是：

```text
不要所有 token 都盲目学 teacher。
```

它按 token 判断 advantage gap 和相对概率，再动态路由：

```text
1. 有些 token 学 privileged teacher；
2. 有些 token 做 student self-distillation；
3. 有些 token 只给轻量约束；
4. 不同 token 使用不同 objective / strength / strategy。
```

### TrOPD：teacher 在 student prefix 上也会 OOD

TrOPD 处理的是另一种 reliability 问题：

```text
student 和 teacher 分布差太大时，
student 采样出来的 token / prefix 可能不是 teacher 熟悉的区域。
```

这时 teacher 的 token-level supervision 不一定可靠，尤其是 reverse KL estimator 可能产生很差的梯度。

TrOPD 的目标是：

```text
只在 trust region 内执行 OPD；
对 outlier region 做 clipping、masking 或 forward-KL estimation；
同时用 teacher-prefix guidance 把 student exploration 拉回可靠区域。
```

这条线的共同结论是：

```text
OPD 的核心不再是“有没有 teacher”，
而是“teacher 在这个 prefix、这个 token、这个条件下是否值得信”。
```

## 主线 2：OPD 和 RLVR 的关系被重新定义

RLVR 的优势是方向可靠：

```text
答案对 / 错由 verifier 或环境决定。
```

RLVR 的缺点是信号稀疏：

```text
一整段 reasoning 只得到一个 scalar reward。
```

OPD 的优势是信号 dense：

```text
每个 token 都能有 teacher distribution / logprob / rubric / representation supervision。
```

OPD 的缺点是方向可能不可靠：

```text
teacher 偏好不一定等于最终任务正确性。
```

所以最近半年很多工作不是让 OPD 替代 RL，而是让两者分工。

### RLSD：RL 管方向，OPD 管幅度

RLSD 的公式可以压缩成：

```text
A_hat_t = A * token_weight_t
```

其中：

```text
A:
  RLVR / GRPO 给出的 response-level advantage。

token_weight_t:
  privileged teacher 和 student 对 sampled token 的 logprob gap 构造出来的 token 权重。
```

关键边界：

```text
teacher signal 不改变 sign；
只改变 magnitude。
```

这让 self-distillation 变成 credit assignment，而不是 imitation target。

### Direct-OPD：迁移 RL 方向，而不是模仿 post-RL teacher

Direct-OPD 面对的是 weak-to-strong 问题：

```text
小模型跑 RL 便宜；
大模型跑 RL 贵；
但小模型 post-RL 后仍可能弱于大模型。
```

普通 OPD 直接模仿 weak post-RL teacher 会有风险：

```text
post-RL weak teacher final policy
= 有用的 RL improvement
+ 小模型自身能力上限 / 错误习惯。
```

Direct-OPD 改为蒸馏：

```text
Delta_T = log pi_T_postRL - log pi_T_preRL
```

也就是：

```text
RL 让 weak teacher 更倾向或更不倾向哪些 token。
```

这个 `Delta_T` 被视为 dense implicit reward：

```text
隐式稠密奖励
```

它作用在 strong student 自己 on-policy 访问到的 states 上。这样迁移的是 RL 学到的改进方向，而不是弱模型最终行为本身。

### VLA-OPD：在机器人里连接 SFT 和 RL

VLA-OPD 的背景是 vision-language-action 模型：

```text
offline SFT:
  分布偏移、灾难性遗忘。

online RL:
  奖励稀疏、样本效率低。
```

VLA-OPD 把 OPD 放在两者中间：

```text
student 自己生成 trajectory；
expert teacher 在这些 policy-induced states 上给 dense token-level supervision；
用 reverse KL 做稳定 policy learning。
```

它说明 OPD 不只是 LLM reasoning 的技巧，也可以作为具身 / action policy 的 post-training 框架。

## 主线 3：多轮 Agent 场景从“全量在线”走向“预算控制”

多轮 agent OPD 的基本形式是：

```text
student 和环境交互多轮；
teacher 在 student 访问到的 history 上给 token-level supervision。
```

问题是：

```text
1. 每次 rollout 都可能调用工具、搜索、执行代码；
2. 前面一步错了，后面 observation 会被带偏；
3. 后续 turn 的 teacher signal 可能更不可靠；
4. 普通 token 平均会让浅层 turn 吃掉大部分 loss。
```

### TCOD：先短后长 / 时间课程

TCOD 发现多轮 OPD 容易出现：

```text
Trajectory-Level KL Instability
```

可以翻成：

```text
轨迹级 KL 不稳定性。
```

根因是：

```text
student 前几轮一旦跑偏，
后续 history 就不是 teacher 熟悉的状态，
teacher supervision 的可靠性下降，
KL 可能上升，success rate 可能崩溃。
```

TCOD 的改进目标是：

```text
不要一开始就学完整长轨迹；
通过 temporal curriculum 逐渐增加 student 负责的时间范围。
```

### ReOPD：把环境交互变成可复用 prefix

ReOPD 的核心背景是：

```text
fully online multi-turn OPD 太贵。
```

它提出：

```text
prefix replay
```

可以翻成：

```text
前缀重放。
```

具体做法：

```text
1. 预先收集 teacher trajectory pool；
2. 从轨迹中采样某个 step 前的 teacher prefix；
3. student 在这个 prefix 上生成当前 turn；
4. teacher 在同一 prefix + student token 上给 RKL supervision；
5. 不执行 student action，不产生新的 tool call。
```

因此 ReOPD 不是全局意义上的 on-policy rollout，而是：

```text
当前 turn 的 response token 是 student on-policy；
历史 prefix 来自 teacher replay。
```

它解决的是：

```text
训练阶段环境交互成本 / tool-call 成本。
```

同时它把问题形式化成：

```text
student occupancy vs teacher reliability
```

可以翻成：

```text
学生访问分布与教师可靠性的张力。
```

### TurnOPD：turn 级 rollout 和 loss 预算

TurnOPD 发现两类错配：

```text
rollout 预算错配：
  固定跑完整 horizon，很多尾部 turn 低信号、低覆盖。

loss 预算错配：
  trajectory-level KL 按 token 平均，浅层 turn token 更多、KL 更大，
  深层关键决策 turn 训练不足。
```

它提出两个 controller：

```text
adaptive rollout-depth budgeting:
  自适应 rollout 深度预算。

progressive turn-normalized loss budgeting:
  渐进式 turn 归一化 loss 预算。
```

这条线的共同判断是：

```text
长程 OPD 的效率瓶颈不是只靠更快 forward 能解决，
而是要决定在哪些 turn、哪些 prefix、哪些 depth 上花训练预算。
```

## 主线 4：从 white-box logits 到 rubric / black-box teacher

传统 OPD 依赖：

```text
teacher full logits
```

这有几个限制：

```text
1. proprietary teacher 不一定给 logits；
2. full-vocab logits 存储和传输昂贵；
3. teacher/student tokenizer 不同时，KL 不好直接算；
4. token-level distribution 未必能表达高层 reasoning 标准。
```

### ROPD：rubric 作为 teacher logits 的替代

ROPD 的目标是：

```text
保留 OPD 的 on-policy 性质，
但不访问 teacher logits。
```

它的流程是：

```text
1. 比较 teacher response 和 student response；
2. 为每个 prompt 归纳 prompt-specific rubric；
3. 用 rubric 给 student rollout 打分；
4. 用这些分数做 on-policy optimization。
```

这里的 `rubric` 可以翻成：

```text
评分规约 / 评价准则。
```

ROPD 把 OPD 从：

```text
token distribution matching
```

推到：

```text
semantic criteria guided on-policy optimization
```

也就是：

```text
语义准则引导的 on-policy 优化。
```

### Rubric-Conditioned Self-Distillation：用 rubric 替代单条 reference rationale

这篇工作处理的是另一类问题：

```text
参考 CoT 昂贵，而且即使最终答案正确，rationale 本身也可能有噪声、不完整或局部错误。
```

它把 teacher 条件化到 criterion-level rubrics：

```text
criterion-level rubrics
```

可以翻成：

```text
准则级评分规约。
```

这样 teacher 给 student rollout 的 token-level guidance 不是围绕某一条 reference rationale，而是围绕“强回答应该满足哪些条件”。

它的目标不是替代 RLVR，而是缓解：

```text
RLVR scalar reward 太粗；
reference rationale 太窄 / 太噪。
```

## 主线 5：从 output-space OPD 到 representation-space OPD

普通 OPD 在 output logits 空间做：

```text
KL(teacher next-token distribution || student next-token distribution)
```

OPRD 指出这有两个问题：

```text
1. 大词表下 MC KL / sampled estimator 方差高；
2. 只看 LM head 输出，会丢掉 teacher hidden states 中的结构信息。
```

所以 OPRD 把 on-policy distillation 提升到：

```text
hidden-state / representation space
```

可以翻成：

```text
隐状态 / 表征空间。
```

它让 student 和 teacher 在相同 on-policy rollouts 上对齐选定层的 representations，并通过 bridge 支持 cross-architecture / cross-tokenizer 场景。

这条线说明一个重要趋势：

```text
OPD 不一定非要等价于 token-level KL。
只要数据分布来自 student on-policy rollout，
监督信号可以来自 logits、rubric、reward、hidden states，甚至多者组合。
```

## 主线 6：多 teacher / 多能力整合

MOPD 的背景是工业 post-training：

```text
不同能力通常需要不同 domain RL；
但把多个 domain ability 合进一个模型很难。
```

传统方案的问题：

```text
Off-policy finetune:
  容易 exposure bias，且 dense signal 不够贴合 student distribution。

Mix-RL:
  多 domain reward / rollout 耦合，训练效率低，能力之间互相干扰。

Param merge:
  合并不一定保留行为能力。
```

MOPD 的改进目标是：

```text
每个领域可以独立训练 RL teacher；
最终 student 在自己的 rollouts 上分别接受多 teacher 的 dense supervision；
把多个 domain teacher 的能力集成到一个模型里。
```

它代表的是 OPD 的工业化方向：

```text
OPD 不只是 teacher-student 压缩，
还可以是 multi-domain capability integration。
```

## 我对热点的排序

如果只按最近半年论文密度和问题重要性排序，我会这样看：

```text
第一梯队：
  teacher reliability / privileged leakage / RL-OPD 配合
  代表：RLSD、DOPD、TrOPD、Direct-OPD

第二梯队：
  multi-turn agent efficiency / tool-call reduction / turn-level budgeting
  代表：TCOD、ReOPD、TurnOPD、AsyncOPD

第三梯队：
  black-box / rubric / structured feedback
  代表：ROPD、Rubric-Conditioned Self-Distillation

第四梯队：
  heterogeneity extension
  代表：GOLD、OPRD、X-OPD、VLA-OPD

第五梯队：
  industrial multi-teacher capability integration
  代表：MOPD
```

这里的“梯队”不是说后面不重要，而是说对 OPD 核心范式的压力不同。

最直接影响 OPD 基础训练目标的是：

```text
RLSD / DOPD / TrOPD
```

最直接影响 agent 工程成本的是：

```text
ReOPD / TurnOPD / AsyncOPD
```

最直接影响可用 teacher 范围的是：

```text
ROPD / GOLD / OPRD
```

## 一个统一视角

可以把 OPD 的近期演化写成一个三层框架。

### 第 1 层：谁提供监督

```text
external teacher:
  普通 OPD、TrOPD、X-OPD、VLA-OPD

self teacher with privileged information:
  OPSD、RLSD、DOPD、Rubric-Conditioned Self-Distillation

multi teacher:
  MOPD、MAD-OPD

black-box semantic teacher:
  ROPD
```

### 第 2 层：监督信号是什么

```text
token logits / token distribution:
  vanilla OPD、TrOPD、DOPD、ReOPD、TurnOPD

policy shift / log-ratio:
  Direct-OPD

RLVR advantage reweighting:
  RLSD

rubric score:
  ROPD、Rubric-Conditioned Self-Distillation

hidden states / representations:
  OPRD
```

### 第 3 层：在哪些状态上训练

```text
student full rollout:
  vanilla OPD、TrOPD、RLSD、VLA-OPD

teacher prefix + student current turn:
  ReOPD

curriculum-selected trajectory segments:
  TCOD

turn-depth-budgeted rollout:
  TurnOPD

async stale rollout:
  AsyncOPD
```

这个框架能解释最近半年为什么出现这么多变体：

```text
大家都保留了“student on-policy distribution 很重要”这个核心，
但不再把 OPD 限定为“完整 student rollout 上做 teacher-student KL”。
```

## 论文链接

- X-OPD: <https://arxiv.org/abs/2603.24596>
- VLA-OPD: <https://arxiv.org/abs/2603.26666>
- RLSD: <https://arxiv.org/abs/2604.03128>
- ROPD: <https://arxiv.org/abs/2605.07396>
- TrOPD: <https://arxiv.org/abs/2606.01249>
- OPRD: <https://arxiv.org/abs/2606.06021>
- Rubric-Conditioned Self-Distillation: <https://arxiv.org/abs/2606.19327>
- AsyncOPD: <https://arxiv.org/abs/2606.24143>
- MOPD: <https://arxiv.org/abs/2606.30406>
- DOPD: <https://arxiv.org/abs/2606.30626>
- ReOPD: <https://arxiv.org/abs/2607.04763>
- Direct-OPD: <https://arxiv.org/abs/2607.05394>
- TurnOPD: <https://arxiv.org/abs/2607.05804>

## 最终判断

最近半年 OPD 最核心的变化是：

```text
从“模仿 teacher 的 token distribution”
走向“在 student 自己会访问的状态上，选择可靠、低成本、可迁移的监督信号”。
```

因此后续最值得关注的不是单个 KL 形式，而是三件事：

```text
1. reliability:
   怎么判断 teacher/rubric/verifier/representation signal 在当前 prefix 上可信；

2. credit assignment:
   dense supervision 如何和 sparse reward 分工，而不是互相覆盖；

3. systems efficiency:
   多轮 agent、tool、search、code execution 场景里，怎么减少无效 rollout 和 teacher query。
```

