# DiDPO: Diff-in-Diff Policy Optimization for Coding Agent Training

- 论文：DiDPO: Diff-in-Diff Policy Optimization for Coding Agent Training
- arXiv：<https://arxiv.org/abs/2608.07147>
- 版本：v1，2026-08-07，16 pages，work in progress
- 作者：Xucong Wang 等
- 关键词：coding agent、RLVR、credit assignment（信用分配）、diff、GRPO、GiGPO

## 一句话结论

DiDPO（Diff-in-Diff Policy Optimization，差分中的差分策略优化）不是 OPD 蒸馏，也不是构造 teacher model，而是一个面向 coding agent 的、无 critic 的强化学习算法。

它观察到：一次 coding action 往往同时修改多个不相干的代码区域。如果把整条 trajectory，甚至整次 diff，都用一个 outcome reward 监督，正确和错误的局部修改会被混在一起。DiDPO 将每次修改拆成可跨 rollout 对齐的 sub-diff（子差分），以相似的 sub-diff 作为 anchor（锚点），在对应 sub-diff 之间计算局部的 group-relative advantage（组相对优势），再把该优势投影回产生它的 response tokens。

核心形式是：

$$
\hat A_{i,l}=A^E(\tau^{(i)})+\lambda A^D(a_l^{(i)}),
$$

其中 $A^E$ 是整条 episode/trajectory 的优势，$A^D$ 是当前 token 所属代码子差分的局部优势。默认 $\lambda=1.2$。因此 DiDPO 保留 GRPO 的全局任务信号，同时增加代码 diff 内部的局部信用分配。

## 1. 背景：coding agent 的信用分配更细

RLVR（Reinforcement Learning with Verifiable Reward，带可验证奖励的强化学习）很适合代码任务：代码可以放进 sandbox（沙箱）执行，编译器和测试用例可以给出客观反馈。

但 coding agent 与普通 agent task 有一个重要区别：普通环境中，一步 action 通常作用于一个相对原子的 state；代码环境中的一次 `add` 或 `delete` action 可能同时改动多个函数、循环、边界条件和导入声明。也就是说，代码状态/代码 diff 是 divisible（可分解的），一次 action 不是一个不可再分的语义单元。

例如，某次 rollout 在同一个 response 中同时：

```text
1. 给循环增加边界判断，修复越界问题；
2. 改写函数签名，但调用方没有同步修改。
```

最终测试失败。若使用 GRPO 的整条 trajectory reward，整个 response 的 token 都收到负优势；模型会同时被惩罚“修复边界”的 token 和“破坏接口”的 token。反过来，如果最终测试通过，整条 response 的 token 都会获得正优势，但这也无法区分哪些修改真正贡献了成功。

DiDPO 的目标不是额外执行反事实环境，而是只利用已经采样的 rollout 和它们最终产生的代码 diff，从现有数据中获得更细的 credit。

## 2. 术语对照

| 英文 | 中文 | 在论文中的含义 |
|---|---|---|
| diff | 差分/代码修改 | 当前代码版本相对上一个版本的新增、删除内容 |
| sub-diff | 子差分 | 将一次完整 diff 切成的连续局部片段 |
| anchor | 锚点 | 一类跨 rollout 反复出现、语义相近的 sub-diff 模式 |
| occurrence | 出现实例 | 某个 anchor 在某条 rollout/某个 action 中的一次具体出现 |
| group mass | 组质量/组规模 | 一个 advantage group 中包含的 occurrence 数量 |
| groupability score, GS | 可成组性分数 | 同时衡量 sub-diff 语义规模和可聚合样本数的分数 |
| trajectory-level advantage | 轨迹级优势 | 对整条 rollout 使用的 outcome-reward 相对优势 |
| diff-level advantage | 差分级优势 | 在同一 anchor 对齐的 sub-diff 之间计算的局部优势 |
| critic-free | 无评论家/无价值 critic | 不训练额外 value model 来估计优势 |
| credit assignment | 信用分配 | 判断哪些 action/token 对最终成功有贡献 |
| cold start | 冷启动 | 先用高质量多轮轨迹做 SFT，使模型学会 thought-action 格式 |

## 3. 问题定义

对一个任务 prompt $x$，采样 $N$ 条 rollout：

$$
\tau^{(i)}=(s_1^{(i)},a_1^{(i)},r_1^{(i)},\ldots,s_{T_i}^{(i)},a_{T_i}^{(i)},r_{T_i}^{(i)}).
$$

$s_t^{(i)}$ 是代码环境状态，$a_t^{(i)}$ 是文本形式的 thought-action response，动作类型可以是 `add`、`del` 或 `none`，$T_i$ 是该 trajectory 的交互步数。对代码任务，论文主要使用 outcome reward $R^{(i)}$，由编译/测试执行结果得到。

### 3.1 轨迹级优势

DiDPO 首先计算与 GRPO 类似的 episode-level advantage：

$$
A^E(\tau^{(i)})=
\frac{R^{(i)}-\frac{1}{N}\sum_{j=1}^{N}R^{(j)}}
{\operatorname{std}(\{R^{(j)}\}_{j=1}^{N})}.
$$

这一步回答的是：“这条完整 rollout 相对同题其他 rollout 好不好？”它提供全局任务信号，但仍然把一整条 response 当作一个信用载体。

若使用每步奖励，也可以从当前步开始计算折扣回报：

$$
R_t^{(i)}=\sum_{k=t}^{T_i}\gamma^{T_i-k}r_k^{(i)},
$$

论文附录的默认设置是 $\gamma=0.95$。在 coding 主设置中，最终结果主要依赖执行反馈/结果奖励。

### 3.2 为什么整条 diff 不够

设 rollout 1 产生：

```text
Diff 1 = [修复边界判断] + [修改函数签名]
```

rollout 2 产生：

```text
Diff 2 = [修复边界判断] + [优化循环实现]
```

两条 diff 都包含“修复边界判断”，但整体文本不同。若按整个 diff 做相似度匹配，Diff 1 与 Diff 2 可能低于阈值，导致无法成组；拆出 sub-diff 后，两个“修复边界判断”可以被匹配到同一个 anchor。

## 4. DiDPO 的完整流程

DiDPO 的一次训练 batch 可以理解为以下步骤：

1. 对每个任务采样一组多轮 coding trajectories。
2. 对每个 action 解析出本次代码 diff，并去掉注释、空白等无关内容。
3. 枚举每个 diff 的多尺度连续 sub-diff。
4. 在不同 rollout/不同 action 的 sub-diff 之间做相似度匹配，只匹配相同 edit type。
5. 将相互匹配的 occurrence 聚合为 anchor。
6. 用 GS 选择一组互不重复、覆盖收益较大的 anchor。
7. 对每个 anchor 内的 occurrence 计算局部 diff-level advantage。
8. 将该局部优势映射到产生该 sub-diff 的 response token。
9. 将轨迹级优势和差分级优势相加，使用标准 clipped policy optimization 更新 policy。

整个过程不需要 teacher model，不需要额外 rollout，也不需要 value model/critic；新增成本主要是 batch 内的 sub-diff 相似度计算。

## 5. Sub-diff 如何构造

### 5.1 Diff 元数据

对每个 diff $s\in\mathcal S$，论文保存：

$$
\operatorname{Metadata}(s)=(u(s),v(s),w(s),q(s)).
$$

- $u(s)$：去掉 annotations（标注信息）和 blank lines（空行）后的 normalized text（规范化文本）；
- $v(s)$：该 diff 对应的 response-token span；
- $w(s)\in\{\texttt{add},\texttt{del},\texttt{none}\}$：动作的编辑类型；
- $q(s)$：产生该 diff 的 task instance。

$|u(s)|$ 表示规范化后的 diff 大小。

### 5.2 多尺度连续切片

对一个 diff $s$，枚举所有连续片段：

$$
\mathcal I(s)=\{\operatorname{Seg}(s,[p,q)):\;0\le p<q\le |u(s)|\}.
$$

工程上可以理解为：不是固定按一行切，而是同时考虑短片段、中等片段和覆盖完整功能块的长片段。这样可以让“增加边界检查”作为一个整体被匹配，而不是被拆成若干孤立 token 或空白行。

## 6. 跨 rollout 匹配与 anchor

对来自不同 rollout 或不同 action 的两个 diff $s,s'$，枚举其 sub-diff 集合，并建立相似度矩阵：

$$
\mathcal M_{s,s'}=\{(I,J,\sigma_{I,J})\mid I\in\mathcal I(s),J\in\mathcal I(s'),
\sigma_{I,J}=\operatorname{sim}(u(s_I),u(s'_J))\ge\eta\}.
$$

论文中的 `sim` 同时结合 token-level lexical matching（词法/token 匹配）和 embedding similarity（嵌入相似度）。此外必须满足：

$$
w(s)=w(s').
$$

也就是 `add` 只能和 `add` 匹配，`del` 只能和 `del` 匹配。默认相似度阈值为 $\eta=0.8$。

所有达到阈值的匹配实例聚合成一个 anchor $c\in\mathcal C$。记：

- $\mathcal O(c)$：支持该 anchor 的 occurrence 集合；
- $\bar L(c)$：anchor occurrence 的平均规范化长度；
- $n(c)$：该 anchor 的 occurrence 数，也就是 group mass 的基本计数。

### 6.1 Groupability Score

论文定义：

$$
\operatorname{GS}(c)=
\left(1-e^{-\bar L(c)}\right)
\left(1-e^{-(n(c)-1)}\right).
$$

它是两个饱和因子的乘积：

- 第一项随 sub-diff 长度增加而增加。太短的片段通常没有稳定的功能语义；
- 第二项随 occurrence 数增加而增加。只有一个 occurrence 的片段无法做有效的组内相对比较；
- 使用乘法而不是加法，要求“语义规模”和“跨 rollout 支持”同时足够好。某一项很低时，另一项不能完全补偿它；
- 两个因子都饱和，避免特别长或特别大的组无限支配选择。

例如：

| anchor | 平均长度 $\bar L$ | occurrence 数 $n$ | GS 直觉 |
|---|---:|---:|---|
| 单个空行 | 0.2 | 20 | 长度项很低，GS 低 |
| 一个完整边界检查 | 4 | 2 | 语义较完整，但支持少 |
| 完整循环修复 | 6 | 8 | 两项都高，GS 高 |

论文实现还加入长度尺度 $s_0=8.0$ 和支持尺度 $g_0=8.0$ 的饱和版本；正文 Eq. 9 的核心形式如上。最多允许 64 个 anchors。

### 6.2 避免重叠 anchor 重复占用样本

仅按 GS 排序仍可能选出大量互相重叠的 anchor。因此论文选择最多 $K$ 个 anchor，使所有候选 sub-diff 被覆盖的 GS 总收益最大：

$$
\mathcal C^*=\arg\max_{|\mathcal C|\le K}
\sum_{s\in\mathcal I(\mathcal S)}
\left\{\max_{c\in\mathcal C}
\mathbf 1[s\in\mathcal O(c)]\operatorname{GS}(c)\right\}.
$$

这是 cardinality-constrained facility-location（带基数约束的设施选址）形式的 submodular maximization（子模最大化）。论文使用 greedy selection（贪心选择）：每次加入能带来最大边际覆盖收益的 anchor，直到没有收益或达到 $K$。

## 7. Diff-level advantage 如何计算

选定 anchors 后，将每条完整 diff 切成 anchor occurrence 和其他 sub-diff。属于同一 anchor 的 action 组成局部 advantage group：

$$
G(\mathcal C^*)=
\{(a_{t,m}^{(i)},R_t^{(i)})\mid
\exists c\in\mathcal C^*:s_{t,m}^{(i)}\in\mathcal O(c)\}.
$$

其中 $a_{t,m}^{(i)}$ 表示 trajectory $i$ 在第 $t$ 步中作用于第 $m$ 个 sub-diff 的 action，$R_t^{(i)}$ 是从第 $t$ 步开始的折扣回报。

在一个 anchor group 内做标准 group-relative normalization：

$$
A^D(a_{t,m}^{(i)})=
\frac{R_t^{(i)}-operatorname{avg}\{R_t^{(j)}\mid
(a_{t,m}^{(j)},R_t^{(j)})\in G(\mathcal C^*)\}}
{F_{norm}\{R_t^{(j)}\mid
(a_{t,m}^{(j)},R_t^{(j)})\in G(\mathcal C^*)\}}.
$$

$F_{norm}$ 是论文使用的组内归一化因子，通常实现为标准差类 normalization；如果 group 太小或方差接近零，需要按实现做稳定化处理。

直观地说，$A^D$ 问的不是“这整条轨迹是否成功”，而是：

> 在做了同一种局部代码修改的 rollout 中，当前这个局部修改所在的后续回报，相对于同类修改的平均水平如何？

### 7.1 具体计算案例

假设同一个编程题采样 4 条 rollout。某个 anchor 对齐了“增加数组边界判断”这一 sub-diff：

| rollout | 该局部 sub-diff | 从该 action 起的回报 $R_t$ | 最终是否通过 |
|---|---|---:|---|
| 1 | 增加边界判断 | 1.0 | 是 |
| 2 | 增加边界判断 | 1.0 | 是 |
| 3 | 增加边界判断 | 0.0 | 否 |
| 4 | 增加边界判断 | 0.0 | 否 |

组内均值为 $0.5$，标准差按总体定义为 $0.5$，因此：

$$
A^D_1=A^D_2=(1.0-0.5)/0.5=+1,
$$

$$
A^D_3=A^D_4=(0.0-0.5)/0.5=-1.
$$

如果 rollout 1 的整条轨迹优势为 $A^E=0.4$，且 $\lambda=1.2$，那么属于该 sub-diff 的 token 得到：

$$
\hat A_{1,l}=0.4+1.2\times 1=1.6.
$$

如果 rollout 3 的整条轨迹优势为 $A^E=-0.6$，对应 token 得到：

$$
\hat A_{3,l}=-0.6+1.2\times(-1)=-1.8.
$$

若同一个 response 还包含“错误修改函数签名”的另一个 sub-diff，那么该签名修改对应的 token 会被分配另一个 anchor 的 $A^D$，不会机械地和边界修复 token 使用同一个 advantage。这就是 DiDPO 相对整条轨迹 broadcast（广播式信用分配）的关键。

## 8. 最终 policy optimization loss

DiDPO 并不使用 OPD loss。它使用 PPO/GRPO 风格的 clipped policy objective。

对 rollout $i$ 的 response token $y_{i,l}$，定义新旧 policy 概率比：

$$
\rho_{i,l}(\theta)=
\frac{\pi_\theta(y_{i,l}\mid y_{i,<l},x)}
{\pi_{\theta_{old}}(y_{i,l}\mid y_{i,<l},x)}.
$$

最终 advantage 为：

$$
\hat A_{i,l}=A^E(\tau^{(i)})+\lambda A^D(a_l^{(i)}).
$$

目标函数为：

$$
\mathcal J=\mathbb E_{x,i,l}\left[
\min\left(
\rho_{i,l}\hat A_{i,l},
\operatorname{clip}(\rho_{i,l},1-\epsilon,1+\epsilon)\hat A_{i,l}
\right)\right].
$$

训练时最大化 $\mathcal J$，或在代码实现中最小化其相反数。它的本质是“GRPO/PPO 的 token policy gradient + diff-level advantage”，不是 student 对 teacher 的 FKL/RKL/JSD 蒸馏。

## 9. 理论解释

论文给出两个结果解释为什么局部 group 有用。

### 9.1 Anchor 误差带来的 bias

令局部奖励贡献 $r(s)$ 关于论文定义的 sub-diff 距离 $\Delta$ 是 $L$-Lipschitz（利普希茨连续），若一个 group 中每个 sub-diff 与同一 anchor 的 correspondence error（对应误差）不超过 $\epsilon$，则用 anchor 匹配替代精确代码匹配时，局部 group contrast 的变化至多为：

$$
O(L\epsilon).
$$

含义是：相似度匹配越准确，局部 credit 的系统性偏差越小。

论文定义结构失真：

$$
\operatorname{dis}(\pi)=
\sup_{(a,b),(a',b')\in\pi}
|d_s(a,a')-d_{s'}(b,b')|,
$$

并用文本相似度和结构失真定义 sub-diff 距离：

$$
\Delta(s,s')=
\inf_{\pi\in\Pi(s,s')}
\left[\frac12\operatorname{dis}(\pi)+
\beta(1-\operatorname{sim}_\pi(s,s'))\right].
$$

这里 $\pi$ 是合法的 code-unit 对齐关系，$\beta$ 平衡结构一致性和文本一致性。

### 9.2 group mass 降低 variance

论文假设某次 step return 可以写成：

$$
R^{(i)}=r(s_i)+\xi_i,
$$

其中 $r(s_i)$ 是局部因果贡献，$\xi_i$ 是零均值、非因果的 trajectory noise（轨迹噪声），方差为 $\sigma_\xi^2$。若一个 DiDPO group 有 $m>1$ 个对应 sub-diff，且对应误差不超过 $\epsilon$，局部优势估计满足：

$$
\operatorname{MSE}(\hat A_i^D)
\le O(L^2\epsilon^2)+O(\sigma_\xi^2/m).
$$

第一项是匹配误差导致的 bias，第二项是组内平均降低的 variance。相比把整条 trajectory 的噪声广播给所有 token 的 episode-level estimator，局部 group 可以减少无关轨迹因素对某个代码片段的污染。

## 10. 训练 pipeline 与超参数

论文的训练不是直接从裸模型开始 RL，而是先做 cold start：

1. 从训练集挑选中等规模、中等难度任务，并用模板填充/改写扩增到约 7K。
2. 用 Qwen3.6-27B 生成最多 12-turn 的 thought-action rollout。
3. 经过 rejection sampling（拒绝采样）和 LLM-based evaluation（基于 LLM 的评估）保留约 3K 高质量轨迹。
4. 用这些轨迹进行 2 个 epoch 的 SFT。
5. 从 SFT checkpoint 继续执行 DiDPO 或其他 RL baseline。

主要设置：

| 项目 | 设置 |
|---|---|
| 主 backbone | Qwen2.5-Coder-7B；另测 Qwen3.5-4B |
| rollout group size | 32 |
| 最大交互步数 | 8 |
| DiDPO diff weight $\lambda$ | 1.2 |
| sub-diff 相似度阈值 $\eta$ | 0.8 |
| anchor 数上限 | 64 |
| groupability scale | $s_0=8.0$, $g_0=8.0$（实现中的额外尺度） |
| discount factor | $\gamma=0.95$ |
| learning rate | $1\times10^{-6}$ |
| PPO clip | 0.2 |
| KL coefficient | 0.01 |
| entropy coefficient | 0.001 |
| PPO 每 batch epoch | 1 |
| mini-batch size | 48 |
| response max length | 4096 |
| prompt max length | 8192 |
| 训练硬件 | 8 张 H20，tensor parallel size 2 |
| 总训练步数 | 120 |

DiDPO 的 grouping 只发生在训练阶段；推理时仍是普通 autoregressive decoding（自回归解码），不需要在线构造 diff anchor。

## 11. 实验 benchmark 与 baseline

评测包含 8 个代码生成 benchmark：

- APPS：从 introductory 到 competition 难度的编程题，并报告 All、Introductory、Interview、Competition；
- HumanEval：函数级代码合成；
- MBPP：基础 Python 代码合成；
- LiveCodeBench：持续更新、降低训练集污染风险的代码生成评测；
- LeetCode：面试风格算法推理；
- USACO：竞赛编程，报告 All/Bronze/Silver/Gold/Platinum；
- OJBench：在线评测编程，报告 All/Easy/Medium/Hard；
- ICPC：ICPC 风格竞赛编程。

Baseline 分三类：

| 类别 | 方法 | 作用 |
|---|---|---|
| 大模型参考 | Kimi-K2.6、Qwen3.6-27B、GLM-5.2、GPT-5.5、Skywork-OR1 | 给出规模/能力上限参考，不在本文 pipeline 中继续训练 |
| 推理/agent 格式 | CoT、CodeAct、Self-Planning | 检查仅靠 prompting 或 agent format 是否足够 |
| RL | SFT、CodeRL+、GRPO、GiGPO、DiDPO | 比较代码 RL 与 agentic credit assignment |

其中 GRPO 是最直接的 episode-level baseline；GiGPO 按重复环境 state 构造 step-level group；DiDPO 则进一步进入一次 code diff 内部，处理同一 action 同时修改多个代码区域的问题。

## 12. 完整主结果

下面保留论文 Table 1 的主要数值。指标均为通过率/准确率（%）；`Avg` 是论文表中报告的平均值，`-` 表示论文未报告。

### 12.1 标准代码生成 benchmark

| Method | APPS All | Intro | Interview | Competition | HumanEval | MBPP | LiveCodeBench | LeetCode | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kimi-K2.6 | 84.4 | 92.5 | 80.1 | 89.0 | 97.0 | 82.2 | 74.9 | 72.4 | 82.2 |
| Qwen3.6-27B | 87.6 | 91.6 | 87.7 | 83.4 | 96.9 | 79.1 | 72.2 | 62.3 | 79.6 |
| GLM-5.2 | 86.3 | 93.0 | 84.1 | 86.1 | 93.3 | 69.7 | 76.5 | 78.9 | 80.9 |
| GPT-5.5 | 89.5 | 95.5 | 88.6 | 86.4 | 100.0 | 83.7 | 100.0 | 85.2 | 91.7 |
| Skywork-OR1 | - | - | - | - | 87.2 | - | 33.8 | 60.0 | - |
| Qwen2.5-Coder-7B | 16.9 | 40.8 | 12.7 | 5.5 | 69.3 | 61.1 | 15.0 | 14.3 | 35.3 |
| + CoT | 25.9 | 52.6 | 22.0 | 11.0 | 70.5 | 67.4 | 11.9 | 12.0 | 37.5 |
| + CodeAct | 14.7 | 34.4 | 11.6 | 4.4 | 66.0 | 62.0 | 19.1 | 18.2 | 36.0 |
| + Self-Planning | 17.4 | 40.4 | 13.6 | 6.0 | 68.9 | 66.0 | 26.5 | 15.5 | 38.9 |
| + SFT | 16.0 | 38.3 | 11.0 | 8.8 | 66.2 | 64.2 | 19.1 | 21.5 | 37.4 |
| + CodeRL+ | 24.4 | 50.8 | 19.5 | 12.9 | 72.0 | 68.2 | 32.0 | 17.0 | 42.7 |
| + GRPO | 23.8 | 49.6 | 18.3 | 14.3 | 70.7 | 69.8 | 31.3 | 18.4 | 42.8 |
| + GiGPO | 25.1 | 53.8 | 18.0 | 17.7 | 67.0 | 71.7 | 35.0 | 22.1 | 44.2 |
| + DiDPO (7B) | **31.3** | **54.7** | **28.4** | 16.8 | **72.3** | **74.2** | **39.3** | **24.7** | **48.4** |
| Qwen3.5-4B | 43.8 | 72.1 | 44.9 | 12.0 | 83.7 | 68.9 | 31.3 | 20.5 | 49.6 |
| + CoT | 45.8 | 70.4 | 48.5 | 13.1 | 86.3 | 67.7 | 36.0 | 25.9 | 52.3 |
| + CodeAct | 40.7 | 70.5 | 39.1 | **16.0** | 84.2 | 61.5 | 33.0 | 25.7 | 49.0 |
| + Self-Planning | 45.1 | 74.0 | 46.4 | 12.0 | 85.4 | 54.9 | 26.7 | 28.1 | 48.1 |
| + SFT | 43.9 | 69.6 | 46.8 | 9.8 | 82.9 | 70.1 | 32.8 | 20.3 | 50.0 |
| + CodeRL+ | 47.1 | 74.1 | 50.1 | 11.0 | 85.0 | 70.7 | 35.3 | 27.7 | 53.2 |
| + GRPO | 45.0 | 76.0 | 46.2 | 10.4 | 85.2 | 67.4 | 30.2 | 25.4 | 50.6 |
| + GiGPO | 48.5 | 77.6 | 50.2 | 14.2 | 88.4 | 72.0 | 31.0 | 28.5 | 53.7 |
| + DiDPO (4B) | **51.3** | **82.2** | **53.0** | 15.5 | **91.8** | **76.9** | **40.4** | **32.8** | **58.6** |

注意：论文正文叙述把 DiDPO 在 7B 上的平均提升概括为相对 GRPO `+5.6`，相对 GiGPO `+4.2`；4B 上分别为 `+8.0` 和 `+4.9`。表中的分类列不一定每列都由 DiDPO 最高，例如 7B APPS Competition 和 4B APPS Competition 的最高者并非 DiDPO，这说明收益主要集中在整体与长程/多区域信用分配，而不是所有子集都单调提升。

### 12.2 竞赛级 benchmark

| Method | USACO All | Bronze | Silver | Gold | Platinum | OJ All | Easy | Medium | Hard | ICPC | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B | 6.5 | 9.8 | 5.0 | 4.8 | 0.0 | 3.8 | 25.0 | 1.9 | 0.0 | 0.9 | 3.7 |
| + CoT | 12.7 | 24.4 | 7.0 | 3.2 | 0.0 | 8.6 | 50.0 | 3.8 | 1.9 | 7.5 | 9.6 |
| + CodeAct | 7.1 | 15.4 | 2.0 | 1.6 | 0.0 | 4.4 | 25.0 | 3.8 | 0.0 | 0.0 | 3.9 |
| + Self-Planning | 5.2 | 8.9 | 5.0 | 0.0 | 0.0 | 3.8 | 25.0 | 1.9 | 0.0 | 2.8 | 3.9 |
| + SFT | 5.2 | 10.6 | 2.0 | 1.6 | 0.0 | 3.8 | 25.0 | 1.9 | 0.0 | 4.7 | 4.6 |
| + CodeRL+ | 13.7 | 26.8 | 5.0 | 6.3 | 0.0 | 7.3 | 40.0 | 3.8 | 1.9 | 8.5 | 9.8 |
| + GRPO | 6.8 | 12.2 | 5.0 | 1.6 | 0.0 | 5.7 | 40.0 | 1.9 | 0.0 | 4.7 | 5.7 |
| + GiGPO | 9.1 | 13.8 | 10.0 | 1.6 | 0.0 | 5.7 | 40.0 | 1.9 | 0.0 | 2.8 | 5.9 |
| + DiDPO (7B) | **15.6** | **27.6** | 10.0 | **6.3** | 0.0 | **8.0** | **45.0** | **3.8** | **1.9** | **8.5** | **10.7** |
| Qwen3.5-4B | 31.6 | 49.6 | 26.0 | 15.9 | 0.0 | 8.8 | 50.0 | 7.5 | 0.0 | 3.8 | 14.7 |
| + DiDPO (4B) | **43.0** | **60.9** | **44.0** | **20.6** | 0.0 | **13.9** | 60.0 | **17.0** | **1.2** | **9.4** | **22.1** |

论文 Table 2 还列出了 4B 下 CoT、CodeAct、Self-Planning、SFT、CodeRL+、GRPO、GiGPO 的逐列结果；其关键对比是 DiDPO 7B 在 USACO All 上 15.6 对 GRPO 6.8，超过一倍；4B 的 DiDPO 平均为 22.1，对应基础 4B 的 14.7。

## 13. 消融实验与结论

### 13.1 GS 设计消融

在 Qwen2.5-Coder-7B 上：

| GS 设计 | APPS | OJBench |
|---|---:|---:|
| $\bar L(c)\times n(c)$ | 16.5 | 1.9 |
| $\bar L(c)+n(c)$ | 24.4 | 5.7 |
| Qwen3.6-27B Judge | 21.5 | 1.9 |
| DiDPO Eq. 9：饱和乘积 | **31.3** | **8.0** |

结论：简单乘积的数值尺度容易被某一项主导；加法则允许“长度大但支持少”或“支持多但语义很碎”的 anchor 得高分。饱和乘积把两项限制在 `(0,1)`，要求两者都不能太差，同时避免大长度/大 group mass 无限增加分数。LLM judge 成本高，且倾向少数大组，格式错误时也不稳定。

### 13.2 DiDPO 组件消融

| 变体 | APPS | OJBench | 去掉了什么 |
|---|---:|---:|---|
| 完整 DiDPO | **31.3** | **8.0** | 无 |
| w/o $A^E$ | 10.4 | 3.8 | 去掉 trajectory-level advantage |
| w/o $A^D$ | 23.8 | 5.7 | 去掉 diff-level advantage，接近 GRPO |
| w/o sub-diff | 25.0 | 4.4 | 把完整 diff 当作原子分组单位 |

解释：

- 去掉 $A^E$ 后性能大幅跌到 10.4，说明局部相似并不等于全局任务成功，episode-level 信号仍是必要的全局约束；
- 去掉 $A^D$ 后退到 23.8，说明局部 credit 本身贡献约 7.5 个 APPS 点，并且性能接近只使用整条轨迹优势的 GRPO；
- 保留 diff-level grouping、但不做 sub-diff decomposition 时是 25.0：比 GRPO 仍好，因为完整 diff 之间仍然可以做一些局部化比较，但部分重叠的功能性修改无法对齐；
- 完整方法高于这些单项相加的直觉收益，说明全局约束、子差分拆分和局部优势存在组合效应。

### 13.3 lambda 敏感性

$\lambda$ 控制局部 diff advantage 相对 episode advantage 的强度。论文观察到性能随 $\lambda$ 呈倒 U 形：

- $\lambda<0.6$：$A^D$ 太弱，方法接近 GRPO；
- $\lambda\approx1.2$：全局成功信号与局部信用信号平衡最好；
- $\lambda>1.2$：可能过度拟合局部高回报模式，学到“像高分样本的局部修改”，但局部修改组合起来未必全局正确。

因此默认取 $\lambda=1.2$，且论文称 `0.6` 到 `1.2` 区间内有一定稳定性。

### 13.4 训练动态与效率

训练最初约 20 steps，各方法都因共享的 episode-level signal 而提升；约 step 40 后，DiDPO 继续上升，而 GiGPO 开始平台，说明模型输出变得更多样后，代码内部 sub-diff credit 的价值更明显。

DiDPO 训练时相对 GRPO 额外开销约 2.3%，主要来自跨 rollout 的 sub-diff similarity 计算；贪心 anchor 选择的开销很小。推理阶段没有额外开销，因为不需要 grouping。

## 14. 与 OPD 的关系

这篇论文放在 OPD 文档目录下，主要是因为它属于同一组“从粗粒度 credit 走向细粒度 credit assignment”的研究脉络，但方法本身不是 on-policy distillation：

| 对比项 | OPD | DiDPO |
|---|---|---|
| teacher | 通常有 teacher/reference policy | 没有 teacher model |
| 核心信号 | teacher/student logits 的 KL 或其他分布差异 | rollout outcome reward 的 group-relative advantage |
| 训练目标 | 蒸馏 loss，常见 FKL/RKL/JSD 变体 | PPO/GRPO 风格 clipped policy objective |
| 局部化对象 | token、turn、prefix 或 trajectory | 代码 diff 内的 sub-diff |
| 是否需要额外 rollout | 取决于 OPD 变体 | 不需要，仅处理已有 rollout |
| 主要问题 | teacher 信号如何可靠地作用于 student | 一次代码 action 内不同局部修改如何分配 credit |

所以 DiDPO 的“diff-in-diff”不是 teacher 在 student prefix 上继续 rollout，也不是把 teacher token 作为监督目标；它是“一个 diff 里面再做 diff-level 分组”的命名。

## 15. 一个端到端小例子

任务：实现 `safe_divide(a, b)`，要求处理 $b=0$。

同题采样三条 rollout：

```text
Rollout A:
  Diff A = [增加 if b == 0 的边界处理] + [返回 float(a / b)]
  outcome = pass

Rollout B:
  Diff B = [增加 if b == 0 的边界处理] + [错误地返回字符串]
  outcome = fail

Rollout C:
  Diff C = [不处理 b == 0] + [返回 float(a / b)]
  outcome = fail
```

经过 sub-diff 枚举与匹配，A/B 的“增加 b==0 边界处理”形成 anchor；A/C 的“返回 float(a/b)”也可能形成另一个 anchor。假设第一个 anchor 的局部回报为 A: 1、B: 0，则其组内 advantage 分别为正/负；这会告诉模型：边界处理这一功能性修改在成功 rollout 中出现，而同类修改出现在失败 rollout 时可能与后续组合有关。

如果 A 的完整 episode advantage 为 `+0.5`，边界 sub-diff 的 diff advantage 为 `+1.0`，则该 sub-diff 的 token 使用：

$$
\hat A=0.5+1.2\times1.0=1.7.
$$

而 A 中“返回 float”对应的 token 可能属于另一个 group，使用另一个 $A^D$。因此同一条成功 rollout 内不同代码片段也不必得到完全相同的更新强度。

## 16. 局限与阅读判断

1. sub-diff 的相似度依赖 lexical matching 和 embedding similarity；语义相似但实现差异较大的代码可能漏匹配，过宽的阈值又会引入错误对应。
2. GS 是启发式的选择分数。论文的理论结果说明匹配误差和组规模如何影响 bias/variance，但没有证明 Eq. 9 对所有代码任务都是最优。
3. 局部 diff advantage 仍然来自最终执行回报，并不是真正的反事实因果效应；它减少的是无关 trajectory noise，而不是完全解决 credit assignment。
4. 训练 pipeline 依赖约 3K 高质量 thought-action 轨迹做冷启动 SFT，弱模型若不具备稳定的多轮代码编辑格式，单纯从 base model 开始可能不能复现结果。
5. 论文标注为 work in progress；主结果中的大模型名称和部分 benchmark 设定应视为该版本报告，后续版本可能变化。

## 17. 总结

DiDPO 解决的是 coding agent 特有的“一个 action 包含多个功能修改”的信用分配问题。它的结构可以概括为：

```text
多条 rollout
  -> 每个 action 的代码 diff
  -> 多尺度 sub-diff
  -> 跨 rollout 相似匹配
  -> GS 选择非冗余 anchor
  -> 每个 anchor 内计算 A^D
  -> A^E + lambda * A^D
  -> PPO/GRPO clipped policy update
```

论文实验支持的最核心结论是：仅使用整条 trajectory advantage 不够；进入代码 diff 内部做 sub-diff 分解，能在不增加环境 rollout 的情况下显著改善长程代码任务的信用分配。它与 OPD 的共同点是都在寻找比“整条 response 一个信号”更细的训练信号，但 DiDPO 的信号来源仍然是 RL outcome reward，而不是 teacher policy 分布。

## 参考

- Wang et al., “DiDPO: Diff-in-Diff Policy Optimization for Coding Agent Training”, arXiv:2608.07147, 2026.
- 论文 HTML：<https://arxiv.org/html/2608.07147v1>
- 论文 PDF：<https://arxiv.org/pdf/2608.07147>
