# 2025-08 ~ 2026-08 MLLM + RL / OPD 数据与训练热点综述

统计窗口：

```text
2025-08-15 到 2026-08-15
```

这份文档主要看近一年里和多模态大模型有关的 RL / OPD / self-distillation / tool-use 训练工作，**重点放在数据、训练和 credit assignment**，benchmark 只作为补充背景，不把纯评测当主角。它想回答的不是“哪篇最好”，而是：

```text
最近一年，MLLM 的 post-training 到底在补什么洞？
```

## 数据 / 训练主线（重点）

### 1. 通用 MLLM 推理

如果只看通用 multimodal reasoning，这一年的主线不是再造一个更大的 benchmark，而是把监督颗粒度从 `answer-level` 往 `evidence / trajectory / token / latent` 下移。

| 论文 | 侧重点 | 数据 / 训练信号 | 值得看什么 |
|---|---|---|---|
| [TED: Training-Free Experience Distillation for Multimodal Reasoning](https://arxiv.org/abs/2603.26778) | 数据 / 经验蒸馏 | 把 teacher 的 experience 压缩进上下文，而不是反复更新参数 | 低成本蒸馏思路 |
| [Beyond Where to Look: Trajectory-Guided Reinforcement Learning for Multimodal RLVR](https://arxiv.org/abs/2603.26126) | RL | 用更强模型的 reasoning trajectory 去引导视觉证据进入推理链 | 让“看见”变成可学信号 |
| [Visual-Advantage On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2605.21924) | OPD | 用 visual advantage 区分高价值 token 和普通 token | token 级视觉监督 |
| [Visual-OPSD: Cross-Modal On-Policy Self-Distillation for Efficient Unified Multimodal Reasoning](https://arxiv.org/abs/2606.18974) | self-distillation | 把 privileged visual thoughts 压成 text-only student 的监督 | 训练免费/低成本蒸馏 |
| [OPLD: On-Policy Latent Distillation for Multimodal Reasoning](https://arxiv.org/abs/2607.28154) | latent distillation | 把显式 CoT / visual operation 监督搬到 latent space | 更紧凑的 reasoning 表示 |
| [Self-Boosting Vision-Language Models with Noisy Student On-Policy Self-Distillation](https://arxiv.org/abs/2607.23125) | self-distillation | 用 clean / corrupted prediction discrepancy 做自监督信号 | 去 teacher 化 |
| [APPO: Attention-guided Perception Policy Optimization for Video Reasoning](https://arxiv.org/abs/2602.23823) | RL | 用 token-level dense reward 强化关键帧感知 | 视频里先解决 perception |
| [Perception-Aware Policy Optimization for Multimodal Reasoning (PAPO)](https://arxiv.org/abs/2507.06448) | RL | 在 GRPO / DAPO 里显式注入 perception bias | 把视觉错误从源头压下去 |
| [CFPO: CounterFactual Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2606.23206) | RL | 用 counterfactual visual cues 约束因果一致性 | 抑制 language prior 漂移 |
| [Omni-Perception Policy Optimization for Multimodal Emotion Reasoning](https://arxiv.org/abs/2606.25325) | RL | 拆视觉 / 声学 / 情绪 cue 的细粒度奖励 | 多模态 cue 融合 |

这一组论文的共同方向很明确：

```text
从“最终答对没”
转向
“视觉证据有没有真正进入推理链、哪些 token 真值得学、监督该放在哪里”。
```

### 2. 医疗多模态

医疗这条线更偏“检索 + 工具 + 证据驱动推理”，而不是单纯堆更大的 VLM。

| 论文 | 侧重点 | 数据 / 训练信号 | 值得看什么 |
|---|---|---|---|
| [Proactive Reasoning-with-Retrieval Framework for Medical Multimodal Large Language Models](https://arxiv.org/abs/2510.18303) | 方法 | 两阶段 RL + confidence-driven re-retrieval | 让模型主动补证据 |
| [MEDVISTAGYM: A Scalable Training Environment for Thinking with Medical Images via Tool-Integrated Reinforcement Learning](https://arxiv.org/abs/2601.07107) | 环境 + RL | tool-integrated RL，教模型何时调用视觉/知识工具 | 医疗 agent 化训练 |

### 3. 金融文档 / 金融推理

金融方向近一年更多是 **agentic document reasoning**，方法侧的重点是 iterated retrieval、self-verification 和 multi-agent workflow。

| 论文 | 侧重点 | 数据 / 训练信号 | 值得看什么 |
|---|---|---|---|
| [Agentic Retrieval-Augmented Generation for Financial Document Question Answering](https://arxiv.org/abs/2605.05409) | 方法 | iterative retrieval + self-verification + program-of-thought | 文档问答的 agent 化 |
| [P1GPT: a multi-agent LLM workflow module for multi-modal financial information analysis](https://arxiv.org/abs/2510.23032) | 方法 | 多 agent workflow 融合技术面 / 基本面 / 新闻 | 交易/研判类工作流 |

### 4. 工具调用 / agentic multimodal

如果目标是“让 MLLM 真正会调工具”，这一年最值得看的是 tool-use reasoning 的训练闭环，而不是单纯 benchmark。

| 论文 | 侧重点 | 数据 / 训练信号 | 值得看什么 |
|---|---|---|---|
| [MATRIX: Multimodal Agent Tuning for Robust Tool-Use Reasoning](https://arxiv.org/abs/2510.08567) | 数据 + 偏好优化 | synthetic multimodal trajectories + step-wise preference pairs | 最像可直接训练的路线 |
| [Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs](https://arxiv.org/abs/2511.19773) | RL | multi-turn RL in VISTA-Gym | tool-integrated reasoning 的 RL 放大 |
| [Process-Supervised Reinforcement Learning for Interactive Multimodal Tool-Use Agents](https://arxiv.org/abs/2509.14480) | RL | turn-level adjudicated RL | 过程监督比只看最终答案更稳 |
| [ARM-Thinker: Reinforcing Multimodal Generative Reward Models with Agentic Tool Use and Visual Reasoning](https://arxiv.org/abs/2512.05111) | reward model + RL | multi-stage RL 教 reward model 调工具找证据 | reward model agentic 化 |

这组工作说明：

```text
工具调用能力的上限，
不在“会不会调用一次工具”，
而在“能不能多轮调用、调用后能不能把证据接回推理、训练时能不能把 credit 给对位置”。
```

### 5. 指令跟随

指令跟随这条线，最近一年真正值得看的不是“能不能做简单 IFEval”，而是：

```text
1. 有没有把 instruction 里的逻辑结构显式建模；
2. 有没有把多模态约束做成可训练数据；
3. 有没有把 SFT / DPO / RL 的监督对准 exact constraints。
```

| 论文 | 侧重点 | 数据 / 训练信号 | 值得看什么 |
|---|---|---|---|
| [MM-IFEngine: Towards Multimodal Instruction Following](https://arxiv.org/abs/2504.07957) | 数据 + SFT/DPO | 生成 `MM-IFInstruct-23k` 和 `MM-IFDPO-23k` | 多模态 IF 的数据工厂 |
| [LSRIF: Logic-Structured Reinforcement Learning for Instruction Following](https://arxiv.org/abs/2601.06431) | RL | `LSRInstruct` 按 parallel / sequential / conditional 构造，奖励按结构传播 | instruction 逻辑结构建模 |
| [Complex Logical Instruction Generation](https://arxiv.org/abs/2508.09125) | 数据 | `LogicIFGen` 自动生成可验证逻辑指令，构建 `LogicIFEval` | 复杂逻辑 IF 数据生成 |
| [MaXIFE: Multilingual and Cross-lingual Instruction Following Evaluation](https://arxiv.org/abs/2506.01776) | 评测 | 23 种语言、1,667 个任务 | 多语种 IF 的基础盘 |
| [MCIF: Multimodal Crosslingual Instruction-Following Benchmark from Scientific Talks](https://arxiv.org/abs/2507.19634) | 评测 | 语音 + 视觉 + 文本，多语言科学演讲 | 长上下文跨模态 IF |

这条线的核心变化是：

```text
指令跟随不再只是“按字面执行一句话”，
而是在训练时显式处理逻辑结构、跨模态约束和多语种表述差异。
```

## 先说结论

这一波工作大致收敛到四个方向：

```text
1. RL 不再只看最终答案，而是开始补视觉证据、轨迹和推理过程。
2. OPD 不再按所有 token 平均蒸馏，而是开始区分高价值 token、证据 token、失败 token。
3. teacher / privileged information 不再被默认可信，很多论文都在处理“信息差”。
4. 有些工作甚至把参数更新拿掉，直接做训练免费式经验蒸馏或上下文蒸馏。
```

换句话说，最近半年 MLLM 的 post-training 主题不是“再堆一个更大的 teacher”，而是：

```text
把监督放到更对的位置，
把 rollout 花在更值的位置，
把视觉证据真正接进推理过程。
```

## 评测侧补充

下面这些更偏 benchmark / evaluation，用来补齐能力边界，但不是这版的主角。

| 时间 | 论文 | 主要发现的问题 | 改进目标 |
|---|---|---|---|
| 2026-03 | [TED: Training-Free Experience Distillation for Multimodal Reasoning](https://arxiv.org/abs/2603.26778) | 传统蒸馏要反复训练，低资源场景成本太高 | 把“参数更新”改成“上下文经验”，并做经验压缩 |
| 2026-03 | [Beyond Where to Look: Trajectory-Guided Reinforcement Learning for Multimodal RLVR](https://arxiv.org/abs/2603.26126) | 只改最终答案，不一定能把视觉证据带进推理链 | 用更强模型的轨迹引导视觉证据到 reasoning 里 |
| 2026-05 | [Visual-Advantage On-Policy Distillation for Vision-Language Models](https://arxiv.org/abs/2605.21924) | 传统 OPD 会把视觉关键 token 淹没在大量语言 scaffolding 里 | 用 visual advantage 区分高价值 token 和普通 token |
| 2026-06 | [Visual-OPSD: Cross-Modal On-Policy Self-Distillation for Efficient Unified Multimodal Reasoning](https://arxiv.org/abs/2606.18974) | 生成视觉 thoughts 很贵，但直接删掉又不一定损失性能 | 把 privileged visual thoughts 压成 text-only student 的监督 |
| 2026-07 | [OPLD: On-Policy Latent Distillation for Multimodal Reasoning](https://arxiv.org/abs/2607.28154) | 显式 CoT / visual operation 约束太重，难以学到抽象视觉思维 | 把推理级监督搬到 latent space |
| 2026-07 | [Med-OPD: Improving Medical Vision-Language Models via Evidence-Aware On-Policy Distillation](https://arxiv.org/abs/2607.16303) | 医学 VLM 容易依赖语言模板，不是真看病灶区域 | 用 evidence-aware token / trajectory supervision 强化视觉证据依赖 |
| 2026-07 | [SOPD-SocialNav: Selective On-Policy Distillation for Vision-Language Social Navigation](https://arxiv.org/abs/2607.19850) | 社交导航里并非所有 token 都重要，平均蒸馏会稀释关键决策位 | 用 entropy 选出社会语义更强的决策 token 再蒸馏 |
| 2026-07 | [Self-Boosting Vision-Language Models with Noisy Student On-Policy Self-Distillation](https://arxiv.org/abs/2607.23125) | 想摆脱外部 teacher 和 ground-truth 依赖 | 用 clean/corrupted prediction discrepancy 做自监督式 OPD |

## RL 侧的补充

这半年不只有 OPD，RL 侧也在明显往“视觉证据 + 过程监督”靠。

| 时间 | 论文 | 主要发现的问题 | 改进目标 |
|---|---|---|---|
| 2026-02 | [APPO: Attention-guided Perception Policy Optimization for Video Reasoning](https://arxiv.org/abs/2602.23823) | 视频推理里 perception 比纯 reasoning 更关键，但普通 RL 不会主动强调关键帧 | 用 token-level dense reward 强化关键帧感知 |
| 2025-07 | [Perception-Aware Policy Optimization for Multimodal Reasoning (PAPO)](https://arxiv.org/abs/2507.06448) | 视觉推理的主要错误来自 perception，而不是纯 reasoning | 在 GRPO / DAPO 里注入隐式 perception loss 和双重熵正则 |
| 2026-03 | [Beyond Where to Look: Trajectory-Guided Reinforcement Learning for Multimodal RLVR](https://arxiv.org/abs/2603.26126) | 只管 answer 不够，视觉证据要进入 trajectory | 用轨迹去显式引导视觉证据进入 reasoning |
| 2026-06 | [CFPO: CounterFactual Policy Optimization for Multimodal Reasoning](https://arxiv.org/abs/2606.23206) | 模型容易偏语言先验，长 CoT 里会出现 grounding failure / hallucination drift | 用 counterfactual visual cues 约束因果一致性 |
| 2026-06 | [Omni-Perception Policy Optimization for Multimodal Emotion Reasoning](https://arxiv.org/abs/2606.25325) | 情绪推理里多模态 cue 利用不足，且容易跨模态幻觉 | 把视觉 / 声学 / 情绪 cue 拆成细粒度奖励和 evidence token 约束 |

## 这波论文主要在补什么洞

### 1. 从“答对没”转向“有没有真的看见”

最直接的变化是，RL 和 OPD 都开始把视觉证据当成核心对象。

`Beyond Where to Look` 的问题意识很明确：

```text
模型能关注相关区域，
但不一定能把视觉证据写进后续推理。
```

所以它不是只盯最终 answer，而是用更强模型的 reasoning trajectory 去拉动“看见 -> 用上 -> 推出来”这条链。

`Visual-Advantage OPD` 也在做类似的事，只不过方式更 token-level：

```text
不是所有 token 都一样值钱。
真正携带视觉监督信号的，只是一小部分 high-VA token。
```

这类工作共同说明：

```text
MLLM 的问题不是单纯的语言建模，
而是视觉证据和语言推理之间的对齐。
```

### 2. 从“所有 token 一视同仁”转向“高价值 token / evidence token / failure token”

这是最近半年 OPD 侧最明显的变化。

`Visual-Advantage OPD`：

```text
先算 teacher 在有无细粒度视觉信息时的概率差，
把差异大的 token 视为高价值 token。
```

`Med-OPD`：

```text
先找 diagnosis-critical evidence，
再看哪些 token 真依赖这部分医学视觉证据。
```

`SOPD-SocialNav`：

```text
用 teacher uncertainty 找 socially informative decision tokens，
压掉 trivial navigation states 上的梯度。
```

`Self-Boosting Vision-Language Models with Noisy Student On-Policy Self-Distillation`：

```text
clean input 和 corrupted input 的预测差异，
本质上就是一种 self-supervision signal。
```

这说明 OPD 在多模态里已经不再是“整段轨迹平均 KL”，而是：

```text
围绕哪些 token 值得学、为什么值钱、学多少，
重新定义蒸馏信号。
```

### 3. 从“外部 teacher 更强”转向“teacher 信号是否真的可靠”

多模态场景里，teacher 的问题比纯文本更尖锐，因为 teacher 可能强在两件不同的事：

```text
1. 真正更会推理；
2. 只是看到了 student 看不到的信息。
```

`Visual-OPSD` 把这个问题说得很直接：

```text
teacher 和 student 权重相同，
差别只在于 teacher 看到了 privileged visual thoughts。
```

这时候监督的价值不在“teacher 是否更大”，而在：

```text
这些 privileged thoughts 里到底有没有可迁移的 reasoning content。
```

`Med-OPD` 也是同一类问题：

```text
医学 VLM 不能只学语言模板，
必须知道哪些 token 真的依赖病灶证据。
```

这条线的核心变化是：

```text
OPD 不再默认 teacher 可信，
而是开始拆 teacher 的信息来源。
```

### 4. 从“在线 rollout + 反复训练”转向“更便宜的复用 / 上下文蒸馏 / latent 蒸馏”

`TED` 直接把蒸馏对象改掉了：

```text
不是更新参数，
而是把 teacher 总结出来的 experience 塞进 prompt。
```

并且它还处理了一个现实问题：

```text
经验会无限长，也会积噪声，
所以要做 experience compression。
```

`Visual-OPSD` 则是另一种省钱方式：

```text
把 expensive 的 visual thoughts 生成路径，
压缩成 text-only student 可学习的 on-policy signal。
```

`OPLD` 更进一步：

```text
连显式 CoT 也不执着了，
把多模态 reasoning 的监督放进 latent representation。
```

这说明 MLLM 的 post-training 已经在从：

```text
显式 token 监督
-> 轨迹监督
-> 上下文监督
-> latent 监督
```

一路往更紧凑的表示移动。

## 背景：这一波为什么会起来

如果把时间线往前拨一点，`Vision-R1` 这类工作已经证明：

```text
MLLM 可以做 RL，
但前提是有像样的 multimodal reasoning cold start。
```

而更早的 `Perception-Aware Policy Optimization (PAPO)` 说明：

```text
视觉推理里的大问题之一是 perception error，
只盯最终奖励不够。
```

所以最近半年这些新工作基本都在补同一件事：

```text
把“看见”变成可优化对象，
把“推理过程”变成可蒸馏对象，
把“高成本 rollout”变成可复用对象。
```

## 这一波热点词

```text
visual advantage
evidence-aware
trajectory-guided
privileged context
latent reasoning
experience compression
entropy-based selection
noisy student
corrupted input supervision
```

这些词背后其实都在指同一件事：

```text
多模态 post-training 不再只问“答案对不对”，
而是问“哪部分视觉证据被用上了、哪些 token 真值得学、监督该放在哪里”。
```

## 一句话总结

最近半年 MLLM + RL / OPD 的主线，不是再造一个更大的 teacher，而是把监督从“最终答案”细化到：

```text
视觉证据
token 重要性
轨迹差异
上下文经验
latent 表示
```

也就是说，这一波研究真正关心的是：

```text
怎么让多模态模型既看得见，
又学得到，
还学得起。
```
