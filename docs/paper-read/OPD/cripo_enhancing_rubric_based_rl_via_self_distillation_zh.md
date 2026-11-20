# CriPO：Enhancing Rubric-based RL via Self-Distillation 论文解读

论文：

```text
Enhancing Rubric-based RL via Self-Distillation
```

链接：

```text
https://arxiv.org/abs/2607.18082
```

## 一句话总结

CriPO 解决的是 rubric-based RL 里的两个问题：

```text
1. UC (Unexplored Criteria)：有些 rubric criterion(标准) 根本没被任何 rollout 覆盖到。
2. SC (Suppressed(被压制) Criteria)：有些 criterion 明明被满足了，但被 scalar reward aggregation(聚合) 压没了。
```

它的核心不是换一个更强 external teacher，而是：

```text
用当前 policy 自己构造两个 self-teacher，
一个补 UC，一个救 SC，
并把它们接到 GRPO 上。
```

所以 CriPO 可以理解成：

```text
GRPO + on-policy self-distillation
```

## 背景

Rubric-based RL 面向的是 open-ended 任务，比如：

```text
medical consultation
scientific QA
long-form writing
```

这类任务没有稳定的唯一 gold answer，因此常把输出拆成若干 rubric criteria 来打分。

但 rubric-based RL 有两个常见失败模式：

### UC：Unexplored Criteria

某些 criteria 在当前 rollout group 里一个都没被满足，所以：

```text
没有学习信号。
```

### SC：Suppressed Criteria

某些 criteria 在部分 rollout 里其实满足了，但最后被标量 reward 聚合成了非正 advantage，于是：

```text
有用的局部模式被优化过程压掉了。
```

论文的观察是：

```text
SC 很普遍，超过 57% 的样本都出现过，
平均每个样本约有 1.8 个 SC。
```

## 方法总览

CriPO 保留 GRPO 作为稳定 backbone：

```text
L_CriPO = L_GRPO + beta * L_OPSD
```

其中 `L_OPSD` 不是一个统一的单 loss，而是由两个 self-distillation 分支组成：

```text
1. behavior injection for UC
2. advantage flipping(翻转) for SC
```

这两个分支都用的是 **当前 policy 自己构造的 teacher**，不是外部 teacher。

## 具体例子：t model 是如何构造的

先强调一点：CriPO 里的 `t model / teacher` 不是另一个 checkpoint，也不是 reward model。

它更准确地说是：

```text
同一个 current policy pi_theta
+ 不同的 teacher prompt / criterion context
= 条件化出来的 self-teacher distribution
```

也就是：

```text
student distribution:
  pi_theta(. | 原始问题 x, student response prefix)

teacher distribution:
  pi_theta(. | 原始问题 x, rubric criterion context, teacher-style instruction, response prefix)
```

下面用一个医疗咨询例子说明。

### 原始问题和 rubric

用户问题：

```text
我最近胸口闷、偶尔胸痛，应该怎么办？
```

rubric criteria 假设有 4 条：

| Criterion | 含义 |
|---|---|
| C1 | 询问症状持续时间、诱因、疼痛性质 |
| C2 | 提醒胸痛可能是急症，出现严重症状应立即就医 |
| C3 | 询问年龄、既往心脏病、高血压等风险因素 |
| C4 | 不直接给出确定诊断，不替代医生 |

当前 policy 对这个问题采样 4 条 rollout，rubric scorer 得到：

| rollout | 满足的 criteria | scalar reward / advantage |
|---|---|---:|
| y1 | C1, C4 | 0.2 |
| y2 | C1 | -0.1 |
| y3 | C3, C4 | 0.0 |
| y4 | C1, C3, C4 | 0.5 |

这里可以看到：

```text
C2 没有任何 rollout 满足 -> UC, unexplored criterion
```

而 `y2` 虽然整体 advantage 是负的，但它满足了 C1，所以 C1 在 `y2` 上属于：

```text
SC, suppressed criterion
```

### UC teacher：criterion-injection self-teacher

对 UC，CriPO 先选 group 里最好的 rollout：

```text
best rollout = y4
满足 C1, C3, C4
缺失 C2
```

然后构造 teacher prompt。这个 prompt 不是普通问答 prompt，而是“修订 prompt”：

```text
原始问题:
我最近胸口闷、偶尔胸痛，应该怎么办？

当前回答 y4:
你可以先记录胸痛出现的时间、持续多久、是否和运动有关，
也需要说明年龄、是否有高血压或心脏病史。我的建议不能替代医生诊断。

未覆盖 criterion:
C2: 提醒胸痛可能是急症，出现严重症状应立即就医。

指令:
请在尽量少改动当前回答的前提下，补上未覆盖 criterion C2。
```

这时 teacher 不是新模型，而是同一个 `pi_theta` 在这个增强 prompt 下的输出分布：

```text
p_T,t = pi_theta(. | x, y4, C2, revise-instruction, y_<t)
```

student 仍然只看普通上下文：

```text
p_S,t = pi_theta(. | x, y_<t)
```

如果 teacher 在某个位置明显更倾向输出：

```text
如果胸痛加重、伴随呼吸困难、出汗或放射到左臂，应立即就医。
```

那么这些 token 对应的 KL contribution 会变大，CriPO-U 会把它们挑出来做局部 OPSD，让 student 在原始 prompt 下也更容易生成“胸痛急症提醒”这类内容。

所以 UC teacher 的构造逻辑是：

```text
从 best rollout 出发
+ 注入 unmet criterion
+ 要求最小修改
= criterion-injection self-teacher
```

### SC teacher：counterfactual self-teacher

再看 SC。假设负 advantage rollout `y2` 是：

```text
你需要告诉我胸痛持续多久、是否运动后加重、是否有压榨感。
```

它满足 C1，但整体 reward / advantage 是负的，因为它漏了 C2、C3、C4。如果直接用 scalar advantage 更新，C1 相关 token 也会被一起压低，这就是 suppressed criterion。

CriPO-S 的 teacher prompt 反过来构造：

```text
原始问题:
我最近胸口闷、偶尔胸痛，应该怎么办？

当前回答 y2:
你需要告诉我胸痛持续多久、是否运动后加重、是否有压榨感。

被满足但被压制的 criterion:
C1: 询问症状持续时间、诱因、疼痛性质。

指令:
请在尽量少改动当前回答的前提下，移除或削弱满足 C1 的内容。
```

同一个 `pi_theta` 在这个 counterfactual prompt 下会倾向生成一个“去掉 C1 证据”的反事实版本，例如：

```text
你可以先观察一下身体情况，必要时咨询医生。
```

于是 teacher distribution 是：

```text
p_T,t = pi_theta(. | x, y2, remove-C1-instruction, y_<t)
```

student distribution 仍是原始回答条件下：

```text
p_S,t = pi_theta(. | x, y_<t)
```

如果在 token `持续多久`、`运动后加重`、`压榨感` 这些位置上：

```text
student 对原 token 概率高；
counterfactual teacher 对这些 token 概率显著降低；
```

那么说明这些 token 正是 C1 的局部证据。CriPO-S 就会把这些 token 的非正 advantage 翻正，避免它们被 scalar reward 一起压掉。

所以 SC teacher 的构造逻辑是：

```text
从负 advantage rollout 出发
+ 指定已满足但被压制的 criterion
+ 要求最小删除 / 削弱该 criterion
= counterfactual self-teacher
```

### 这个例子里的 t/s model 总结

| 分支 | student 看什么 | teacher 看什么 | teacher 的作用 |
|---|---|---|---|
| UC | 原始问题和普通 prefix | 原始问题 + best rollout + unmet criterion + 最小修订指令 | 找到应该补进来的 criterion token |
| SC | 原始问题和普通 prefix | 原始问题 + negative rollout + suppressed criterion + 最小删除指令 | 定位不该被压掉的 criterion token |

一句话：

```text
CriPO 的 teacher 不是“更强模型”，而是同一个 policy 在 criterion-conditioned prompt 下形成的条件分布。
```

## Reward 从哪来

是的，这篇的 reward 本质上来自一个 judge model 的 rubric 打分，但严格说它不是传统意义上单独训练出来的 reward model，而是 **LLM-as-a-Judge / rubric judge**。

具体形式是：

```text
1. 先定义多个 rubric criteria c_j
2. judge model 对每个 rollout 在每个 criterion 上打分 r_ij
3. 用 criterion weight ω_j 聚合成 scalar reward R_i
```

论文里的公式是：

```text
R_i = sum_j ω_j r_ij / sum_j ω_j
```

也就是说，reward 不是来自一个额外训练的 RM head，而是来自 rubric judge 对每条回答的 criterion-wise scoring。

### 这个 judge model 是什么

论文明确写了：

```text
训练时用 Qwen3-32B 作为 rubric judge
```

并且：

```text
将其部署在 64 × 910B2 NPUs 上
```

评测时则使用：

```text
GPT-4o-mini 作为 evaluation judge
```

所以要区分两件事：

```text
training judge: Qwen3-32B
evaluation judge: GPT-4o-mini
```

### 这意味着什么

这篇的 reward 不是人工标注的单一分数，也不是一个单独训练的 preference reward model，而是：

```text
rubric criteria -> judge model 打分 -> 加权成 scalar reward
```

这也正是后面 UC / SC 能工作的基础：

```text
1. judge 能给出 criterion-wise score
2. 才能判断哪些 criterion 未被覆盖，哪些 criterion 被压制
```

## 完整流程总览

CriPO 的训练流程可以按下面理解：

### 1. 先跑一组 on-policy rollouts

当前 policy 先对同一个 rubric 任务采样一个 group of responses：

```text
x -> y1, y2, y3, y4
```

每条 response 都会被 rubric scorer 打分，得到：

```text
1. 哪些 criteria 被满足
2. 哪些 criteria 仍然没被覆盖
3. 哪些 criteria 虽然满足了，但最后被 scalar reward 压住了
```

### 2. 区分 UC 和 SC

然后把 rubric criteria 分成两类：

```text
UC: 一个 group 里完全没被覆盖到的 criteria
SC: 被某些 rollout 覆盖了，但被 scalar advantage 压掉的 criteria
```

### 3. 对 UC 构造 criterion-injection self-teacher

做法是：

```text
1. 选 group 里 advantage 最高的 best rollout
2. 把它 + unmet criteria + minimal-revision instruction 放进 teacher prompt
3. 用同一个 current policy 生成 revised response
4. 用 revised response 和原始 student response 做局部 forward KL
5. 只保留 KL contribution 最大的 token
```

注意：这里的 UC teacher 只围绕 best rollout 构造。group 里的其他 trajectories 主要用于两件事：

```text
1. 判断哪些 criteria 是 UC / SC
2. 为 SC 分支提供负 advantage rollout
```

它们不会一起直接变成 UC teacher 的 target trajectory。

### 4. 对 SC 构造 counterfactual self-teacher

做法是：

```text
1. 选负 advantage rollout
2. 找出其中被满足但被压掉的 criterion
3. 把该 criterion 作为反事实删除目标写进 prompt
4. 用同一个 current policy 生成 counterfactual response
5. 找出被 criterion 支撑的 token
6. 把这些 token 的非正 advantage 翻正
```

### 5. 和 GRPO 合并

最后把两个分支都挂回 GRPO：

```text
L_CriPO = L_GRPO + beta * L_OPSD
```

所以整体是：

```text
GRPO 负责 reward-grounded optimization
CriPO 负责补 criterion-level credit assignment
```

这比“只靠 GRPO”更细，也比“只靠 OPSD”更稳。

## 1. UC 分支：Criterion-injection self-teacher

对 UC，CriPO 先在一个 group 里找 **best rollout**：

```text
advantage 最高的那个 rollout
```

然后用当前 policy 重新 prompt 它，让它在 **尽量少改动原答案** 的前提下补上未覆盖的 criteria。

这个 prompt 的本质是：

```text
请在保持原回复大体结构不变的前提下，
只补上这些未满足的 rubric criteria。
```

这样生成出来的 revised response 就是：

```text
criterion-injection self-teacher
```

### 这里的 teacher 是怎么来的

不是新训练一个模型，而是：

```text
同一个 current policy π_θ
在“带 unmet(未满足的) criteria 的修订 prompt”下生成的 revised(修改) response。
```

也就是说，teacher 是 **on-policy self-teacher**。

### UC 的 loss

对 revised response 里和目标 criterion 相关的 token，计算局部 forward KL：

```text
L_OPSD-U = mean_t D_KL( sg(p_T,t) || p_S,t )
```

其中：

```text
p_S,t = π_θ(· | x, y_<t)
p_T,t = π_θ(· | x, C_u, y_<t)
```

这里 `C_u` 是未探索到的 criteria。

更直白地说：

```text
student 看普通 prompt；
teacher 看“补了缺失 criteria”的修订 prompt；
student 去贴 teacher 的 token 分布。
```

### Token filtering

CriPO 不对所有 token 都蒸馏，而是做 token 过滤：

```text
按 KL contribution 排序，只保留最有贡献的 token。
```

论文里用的是累计贡献阈值，重点保留真正体现“补进来的行为”的 token。

### 先纠正一个容易误解的点

UC 分支里的 teacher 不是：

```text
把 student rollout + revision prompt 当作一个普通 prefix，
再让 teacher 生成一整条 revised response，
然后把 revised response 整段当 target 学。
```

更准确是：

```text
1. 只选 group 里 highest-advantage rollout y*；
2. student 分布看普通上下文和 y*_<t；
3. teacher 分布看普通上下文 + unmet criteria / revision instruction 和同一个 y*_<t；
4. 在同一个 on-policy prefix y*_<t 上比较 next-token distribution。
```

形式上是：

```text
p_S,t = pi_theta(. | x, y*_<t)
p_T,t = pi_theta(. | x, C_u, y*_<t)
```

所以 revision prompt 的作用是给 teacher 注入 criterion-conditioned preference，而不是把 teacher rollout 出来的新答案全量拿来 SFT。

### 目标 criterion 相关 token 如何确定

UC 分支里没有人工标注“哪个 token 对应哪个 criterion”。论文用的是 Contribution-Guided Filtering，可以翻译成：

```text
贡献度引导的 token 过滤
```

具体做法是，对 revised response 的每个 token position t 计算 teacher 和 student 的 forward KL：

```text
c_t = D_KL(p_T,t || p_S,t)
```

其中：

```text
p_T,t = pi_theta(. | x, C_u, y_<t)
p_S,t = pi_theta(. | x, y_<t)
```

然后把所有 token 按 c_t 从大到小排序，选择累计贡献达到阈值的最小 token 集合：

```text
S_U = TopCum({c_t}, gamma)
```

论文默认：

```text
gamma = 0.95
```

也就是说，所谓“和目标 criterion 相关的 token”是这样定义的：

```text
在加入 unmet criterion C_u 后，
teacher 和 student 分布差异最大的那批 token。
```

直觉上，如果 revised response 里有一句：

```text
如果胸痛加重、伴随呼吸困难或出汗，应立即就医。
```

而这句话正是为了补 C2“胸痛急症提醒”，那么这些位置的 teacher/student KL 会很大，就会被 TopCum 选中。相反，像“你好”“建议如下”这种通用连接词，即使出现在 revised response 里，KL contribution 通常小，不会被重点蒸馏。

因此这里的 token 选择是：

```text
不是字符串匹配 criterion；
不是人工找 span；
而是看 criterion-conditioned teacher 让哪些 token 分布变化最大。
```

### 如何做到“尽量少改动原答案”

这个约束主要来自 criterion-injection prompt，不是一个额外的编辑距离 loss。

UC 分支的流程是：

```text
1. 在 group 里选 advantage 最高的 best rollout；
2. 把 best rollout、未覆盖 criteria C_u、原始问题一起放进 prompt；
3. 明确要求 current policy 做 minimal revision；
4. 生成 revised response；
5. 只对 high-contribution tokens 做 OPSD。
```

所以“尽量少改动”有两层含义：

```text
第一层：prompt 直接要求保留原答案，只补缺失 criterion；
第二层：loss 只作用在 KL contribution 最大的少量 token 上，不把整段 revised response 全量蒸馏过去。
```

这也解释了为什么 CriPO 不是普通 SFT：

```text
普通 SFT: 直接把 revised response 全部当 target 学；
CriPO-U: 只把 criterion injection 引起分布变化最大的 token 局部蒸馏给 student。
```

## 2. SC 分支：Counterfactual(反事实) self-teacher

对 SC，CriPO 先找 **负 advantage rollout**，再构造 counterfactual self-teacher：

```text
把这个 response 里“满足该 criterion 的部分”尽量删掉 / 改掉，
只保留最小必要改动。
```

这会形成一个对照版本：

```text
original response: 仍然包含某个 satisfied criterion
counterfactual teacher: 刻意削弱这个 criterion
```

然后比较：

```text
p_S = π_θ(· | x, y_<t)
p_T = π_θ(· | C_s, y_<t)
```

其中 `C_s` 是被削弱的 satisfied suppressed criteria。

### SC 的 token 选择

论文选 token 的条件有两个：

```text
1. sampled token 被删除 criterion 后明显降权
   Δ_i,t = log p_S(i,t) - log p_T(i,t) > 0

2. teacher 真的更偏向别的 token
   p_T(i,t) < α * max_v p_T(v)
```

默认 `α = 0.1`。

满足这两个条件，才认为这个位置是 criterion-relevant token。

### SC 的处理

对这些 token，CriPO 会把原来非正的 token-level advantage 翻正：

```text
flip advantage at localized tokens
```

直白理解就是：

```text
别让 scalar reward 把已经学到的好局部模式压没了。
```

## 为什么不是直接用 OPSD

论文明确说，单独用 OPSD 不稳定。

原因很直接：

```text
OPSD 只能补局部行为，
但没有 GRPO 这个 reward-grounded backbone 时，
整体优化方向不稳。
```

所以 CriPO 的设计是：

```text
GRPO 保方向
OPSD 补局部 criterion
```

## 训练设置

论文主要在：

```text
RaR-Medicine
RaR-Science
```

上训练，并在 medicine / science 的多个 benchmark 上评测。

模型 backbone：

```text
Qwen3-1.7B
Qwen3-4B
```

## 完整主结果

### GPT-4o-mini judge

平均分：

| Method | Qwen3-1.7B Avg | Qwen3-4B Avg |
|---|---:|---:|
| Base | 57.2 | 63.7 |
| GRPO | 59.2 | 68.2 |
| HeRL | 60.4 | 68.4 |
| OPSD | 38.2 | 43.6 |
| CriPO-U | 61.6 | 68.8 |
| CriPO-S | 60.4 | 69.6 |
| CriPO | 62.4 | 69.6 |

结论：

```text
1. CriPO 在两个 backbone 上都优于 GRPO、HeRL、OPSD。
2. OPSD 单独用会明显崩。
3. CriPO-U 和 CriPO-S 分别对应 UC / SC 的单分支修正。
4. 完整 CriPO 通常最好或并列最好。
```

### Qwen3-32B judge

平均分：

| Method | Qwen3-1.7B Avg | Qwen3-4B Avg |
|---|---:|---:|
| Base | 40.9 | 53.9 |
| GRPO | 43.7 | 59.4 |
| HeRL | 45.6 | 57.5 |
| OPSD | 21.7 | 30.4 |
| CriPO-U | 46.8 | 60.2 |
| CriPO-S | 44.7 | 60.6 |
| CriPO | 47.1 | 61.5 |

结论：

```text
换 judge 后趋势仍然一致，说明提升不是某个 evaluator 的偶然偏差。
```

## OOD 结果

论文还测了 instruction-following OOD，基本没有大幅伤害通用能力：

| Model | IFEval | IFBench | MulDimIF |
|---|---:|---:|---:|
| Qwen3-1.7B | 69.7 | 18.7 | 14.9 |
| + CriPO-RM | 69.5 | 19.0 | 14.8 |
| + CriPO-RS | 69.9 | 20.0 | 14.9 |
| Qwen3-4B | 81.1 | 27.3 | 17.6 |
| + CriPO-RM | 80.6 | 26.3 | 17.8 |
| + CriPO-RS | 80.4 | 28.3 | 17.2 |

含义：

```text
CriPO 的修正主要是 rubric 相关的局部行为，
不是把模型整体 instruction-following 能力训坏。
```

## 一句话结论

CriPO 的核心不是“更强 teacher”，而是：

```text
1. 对没覆盖到的 criterion，用当前 policy 生成的 criterion-injection self-teacher 补行为；
2. 对被 scalar reward 压掉的 criterion，用 counterfactual self-teacher 找回 token-level signal；
3. 两者都挂在 GRPO 上，避免单独 OPSD 的不稳定。
```

