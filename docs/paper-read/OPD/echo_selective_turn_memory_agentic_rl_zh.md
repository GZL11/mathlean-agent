# ECHO：面向 Agentic RL 的可追踪选择性 Turn Memory

> 论文：**ECHO: Prune to Act, Trace to Learn with Selective Turn Memory in Agentic RL**  
> arXiv：<https://arxiv.org/abs/2606.31650>

## 1. 背景

长程 agent 任务通常会遇到两个相互耦合的问题：

1. **上下文不够用**：搜索、浏览、工具调用、用户交互不断累积，完整轨迹很快超过 context window。
2. **信用分配不准**：如果只用最终成功/失败奖励，普通 GRPO 会把整条成功轨迹都强化，包括无关搜索、冗余步骤和错误中间结论。

已有 context management 方法，比如 rolling summary、context folding、semantic retrieval，主要解决“怎么继续 act”。但它们通常没有保留清晰的 provenance：后续答案到底依赖了原始哪几轮工具观察，训练时很难把 reward 精确打回去。

ECHO 的核心观点是：**memory 不只要压缩上下文，还要能追踪证据来源；否则长程 RL 的 credit assignment 会被 summary 和冗余历史污染。**

## 2. 核心方法

### 2.1 Selective Turn Memory

ECHO 把 agent 交互拆成 turn。每个 turn 包含：

- assistant 的 action / tool call；
- tool observation；
- 当前轮产生的 compact finding。

对第 `i` 个 turn，ECHO 保存一个带来源编号的 memory record：

$$
e_i = (i, \alpha_i, m_i),
$$

其中：

- `i` 是 source turn id；
- $\alpha_i$ 是该轮 action / tool call 的简要描述；
- $m_i$ 是模型从该轮 observation 中提炼出的 finding。

关键点是：ECHO 不把所有历史滚成一个不可分解的 summary，而是保留每条 memory 到原始 turn 的映射。

### 2.2 Prune to Act

当上下文快到预算边界时，ECHO 让模型选择哪些历史 turn 需要保留。选择动作可以写成：

```text
select: [3, 7, 12, 15]
```

下一段 agent context 由三部分组成：

```text
task prompt + selected source-indexed memories + recent local turns
```

论文设置中，ECHO 默认保留最近 `3` 个 turn，并最多额外选择 `8` 个历史 source-indexed memory。单次 policy call 仍使用 `32k` context，但通过最多 `5` 次 reconstruction，把有效长程交互预算扩展到约 `192k`。

### 2.3 Trace to Learn

ECHO 的训练仍然是 outcome-based RL，不单独训练 reward model。基础优势来自 GRPO 的 group-relative reward：

$$
A =
\frac{r - \operatorname{mean}(\{r_j\}_{j=1}^{G})}
{\operatorname{std}(\{r_j\}_{j=1}^{G})}.
$$

区别在于 reward 不再平均打给整条轨迹，而是通过 memory provenance 做 credit routing。

ECHO 只使用正优势：

$$
A_+ = \max(A, 0).
$$

当一条 rollout 成功并且优于组内平均时，正优势主要分配给：

1. 最后一个 trajectory segment；
2. 最终 context 中被选中的 source turns 对应的原始 action tokens；
3. 这些 source turns 的 memory finding tokens；
4. memory-selection action tokens。

也就是说，模型最终使用了哪些 memory，训练时就奖励这些 memory 的来源 turn 以及“选择这些 memory”的动作。失败或低于组内平均的轨迹不会反向惩罚 memory 选择，避免因为 verifier 噪声或探索失败把潜在有用证据误伤。

## 3. 与常见方法的区别

| 方法 | 解决什么 | 主要问题 |
|---|---|---|
| GRPO | 用 outcome reward 做在线 RL | 长程任务中 credit assignment 太粗 |
| Rolling summary / SUPO-style summary | 压缩历史，让 agent 继续跑 | summary 缺少原始 turn provenance |
| Semantic top-k retrieval | 从历史中按相似度取片段 | retrieval 不是 policy action，训练时很难奖励“选择行为” |
| **ECHO** | 同时做 context pruning 和 traceable credit assignment | 需要模型额外生成 memory 和 selection，训练/实现更复杂 |

ECHO 和 ABSeeker 都关注长程 credit assignment，但监督来源不同：

- ABSeeker：从 verified answer 反向恢复 clues，再给每步打 process reward；
- ECHO：从最终 context 中实际被选择和复用的 memory source 反推 credit。

因此 ABSeeker 更适合“答案可验证、能恢复证据链”的搜索 QA；ECHO 更适合“长历史必须压缩、且要把压缩选择纳入训练”的 agent 任务。

## 4. 实验设置

### 4.1 Benchmark

主实验使用 **BrowseComp-Plus**，是长程网页搜索 QA 任务。Agent 可以使用三类工具：

```text
search, open_page, finish
```

训练集约 `747` 条，held-out 约 `83` 条。每个 prompt 采样 `8` 条 rollout，工作上下文为 `32k`。

评测使用 DeepSeek V4-Flash 作为 LLM verifier，判断最终答案是否满足问题约束。

### 4.2 Baseline

主要 baseline 包括：

- **GRPO**：普通 bounded-context agentic RL；
- **SUPO-style rolling summarization**：用滚动 summary 做 context management；
- **ECHO**：source-indexed selective memory + traceable credit routing。

论文还在 MoE backbone 和 zero-shot generalization 任务上做了额外验证。

## 5. 主实验结果

### 5.1 BrowseComp-Plus

Qwen3-32B-Instruct / BrowseComp-Plus held-out accuracy：

| 方法 | Accuracy |
|---|---:|
| GRPO | 28.9 |
| SUPO-style summary | 36.1 |
| **ECHO** | **43.4** |

结论：

1. 普通 GRPO 在长程搜索里受限明显；
2. summary 能提升效果，但会带来更长轨迹和更多 reconstruction；
3. ECHO 同时提升准确率和交互效率。

### 5.2 轨迹效率

| 方法 | Final turn count | Split rate | Trajectories / rollout |
|---|---:|---:|---:|
| SUPO-style summary | 62.5 | 85.5% | 4.18 |
| **ECHO** | **45.3** | **57.8%** | **3.13** |

这说明 ECHO 不是靠“搜得更多”换准确率，而是通过更有选择性的 memory 使用减少无效延长。

### 5.3 MoE Backbone

在 Qwen3-30B-A3B-Instruct 上，论文报告普通 GRPO 约 `22.9%`，SUPO 后期下降到约 `13.3%`，ECHO 稳定在约 `35.0%`。这组结果强调的是训练稳定性：rolling summary 可能因为压缩误差和 credit assignment 混乱而在后期退化。

## 6. 消融实验

### 6.1 Zero-shot Generalization

论文在多个未见任务上评测：

- Multi-Objective QA：2/4/8/16 objectives；
- CodeGym；
- LoCoBench-Agent；
- GAIA；
- HLE；
- Frames。

Qwen3-32B-Instruct 平均分：

| 方法 | Avg |
|---|---:|
| GRPO | 33.6 |
| SUPO-style summary | 34.8 |
| ECHO w/ Top-K retrieval | 38.2 |
| ECHO w/o traceable CA | 35.6 |
| **ECHO** | **40.2** |

### 6.2 结论

消融主要说明两点：

1. **Learned source selection 优于静态 semantic retrieval**：历史片段该不该保留，本身是任务相关决策，应该被 policy 学习，而不是只靠 embedding 相似度。
2. **Traceable credit assignment 很关键**：去掉 credit routing 后，性能明显下降，并且轨迹更容易膨胀。

## 7. 对 Agentic RL 的启发

ECHO 的价值不在于提出一个新的外部 memory store，而在于把三个环节打通：

1. agent 执行时，把工具观察转成 source-indexed memory；
2. context 超预算时，让 policy 显式选择要保留的 memory；
3. outcome reward 回来后，把正优势打回被选择 memory 的源 turn 和选择动作。

这对长链路创作 agent 很有参考意义。比如图片/视频编辑、多轮文案创作、复杂素材检索与修改任务中，系统不能只维护一个不可追踪 summary，而应该记录：

- 哪次用户反馈被后续采用；
- 哪次工具调用产出了关键素材或约束；
- 哪次编辑操作成为最终结果的依据；
- 模型在压缩历史时选择保留了哪些来源。

这样后续无论做 GRPO/PPO、DPO，还是 process reward，都能把训练信号落到更具体的历史选择和操作上。

## 8. 局限

1. ECHO 依赖模型自己生成 memory finding 和 selection action；如果 memory 写错，后续 credit routing 也会跟着偏。
2. 论文主任务还是搜索 QA，迁移到多模态创作 agent 时，需要重新设计 memory record，比如图像 patch、编辑参数、用户偏好、素材来源等。
3. 评测依赖 LLM verifier，BrowseComp-Plus 的答案判定会受 verifier 稳定性影响。
4. ECHO 主要奖励正优势，没有充分讨论失败轨迹中“错误 memory 选择”是否应该被显式负向学习。

## 9. 总结

ECHO 可以概括为：

> **Prune to Act：用 source-indexed selective memory 让长程 agent 继续行动；Trace to Learn：用 memory provenance 把最终 reward 追溯到真正被复用的历史 turn。**

它最适合的场景是长上下文、多轮工具调用、最终结果可验证、并且中间历史需要被反复压缩的 agentic RL。对工业落地来说，ECHO 的关键启发是：context management 不能只作为推理工程模块存在，它也应该进入训练闭环，成为可学习、可评估、可回放、可归因的 policy 行为。
