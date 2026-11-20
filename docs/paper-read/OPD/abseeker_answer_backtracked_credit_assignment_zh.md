# ABSeeker：通过答案回溯式信用分配训练长程搜索 Agent

> 论文：**ABSeeker: Training Long-Horizon Search Agents via Answer-Backtracked Credit Assignment**  
> arXiv：<https://arxiv.org/abs/2608.05102>

## 1. 核心结论

ABSeeker 的核心不是 OPD，而是 **ABC（Answer-Backtracked Credit Assignment，答案回溯式信用分配）**：从已验证答案反向恢复中间证据线索，用线索逐步检查 Agent 的搜索行为，得到密集的步骤奖励，再用于步骤加权 SFT 和步骤级 GRPO。

它解决长程搜索中的 credit assignment（信用分配）问题：失败轨迹的前半段可能已经找到有用证据，成功轨迹中也可能有错误或冗余步骤，最终答案的 0/1 奖励无法区分这些情况。

## 2. 它是不是 OPD？

不是。ABSeeker 没有 teacher model 给 student model 提供 logits，也没有 `KL(student || teacher)`、OPD loss、RKL/FKL 或 teacher-prefix replay。

| 项目 | ABSeeker |
|---|---|
| teacher/student | 没有 teacher-student 蒸馏结构 |
| 监督来源 | verified answer（已验证答案）及 DeepSeek-V4-Flash 恢复、检查的 clues |
| 第一阶段 | ABC-SFT：按步骤奖励加权的 SFT |
| 第二阶段 | ABC-GRPO：把步骤奖励用于 GRPO 的步骤级优势 |
| 优化 | 加权 next-token CE，随后是标准 clipped GRPO（带 KL 项） |

用于线索恢复和步骤打分的外部 LLM 是评分器/恢复模型，不是 OPD 意义上的 teacher policy。

## 3. 为什么轨迹级奖励不够

搜索轨迹写成

$$
\tau=(s_1,s_2,\ldots,s_T,a),
$$

`s_t` 包括 Agent 推理、工具调用和工具返回，`a` 是最终答案。传统结果奖励为

$$
r_{\mathrm{ans}}(\tau)=
\begin{cases}1,&a=a^*,\\0,&a\ne a^* .\end{cases}
$$

这会造成两种错误归因：失败轨迹中的有用步骤全部被判零；成功轨迹中的错误或冗余步骤全部被奖励。ABC 改问：“这一步相对于可验证证据链做了什么？”

## 4. ABC 流程

### 4.1 答案回溯式线索恢复

Answer-Backtracked Clue Recovery（答案回溯式线索恢复）从正确答案反向寻找解题中间点。线索可以是 entity（实体）、fact（事实）、attribute（属性）、relation（关系）以及需要跨页面核验的中间结论。

恢复过程本身是主动的 ReAct loop（推理-行动循环）：恢复模型调用搜索工具、访问网页，把候选 clue 锚定到真实网页内容。论文围绕答案 `CeraVe` 的示例恢复了六条线索，包括 Ceramides（神经酰胺）、L'Oréal 的收购关系、Eugène Schueller 是创始人、Schueller 于 1904 年毕业等。

这些 clues 是训练时的证据锚点，不是部署时给 Agent 的额外上下文，也不是 teacher trajectory。

### 4.2 基于线索的步骤评分

对每个步骤 `s_t`，评分器接收当前 reasoning/tool call/tool response、原始问题 `q`、完整 clue 集合 `C`，输出标量 reward 和简短理由。每步从 `1.0` 开始：

| 行为 | 增量 `Delta` |
|---|---:|
| 发现或验证正确线索 | `+0.8` |
| 排除错误候选 | `+0.4` |
| 错误否定正确线索 | `-0.8` |
| 提交已验证答案 | `+1.0` |
| 提交错误答案 | `-1.0` |

令 `A_t` 为第 `t` 步检测到的行为实例集合：

$$
r_t=\operatorname{clip}\left(1.0+\sum_{j\in A_t}\Delta_j,0,2.0\right).
$$

多条线索的增量累加，最后裁剪到 `[0,2.0]`。`r_t=1` 是中性基线，不表示该步已经证明正确。

### 4.3 具体奖励案例

- Step 22 找到 `Ceramides` 并关联 `CeraVe`：`r_22=1.0+0.8=1.8`。
- Step 35 同时验证两条线索：`r_35=clip(1.0+0.8+0.8,0,2.0)=2.0`。
- Step 56 放弃正确证据并回到错误候选：`r_56=1.0-0.8=0.2`。
- Step 64 提交错误答案：`r_64=1.0-1.0=0`。

所以最终失败的轨迹仍能保留高质量步骤，不会因为最终答案错误而整条归零。

## 5. ABC-SFT

`x_{t,j}` 是第 `t` 步由 policy 生成的第 `j` 个 token。环境提供的工具返回被 mask，不计算 loss：

$$
\mathcal L_{\mathrm{SFT}}(\theta)=-\sum_{t=1}^{T}w(r_t)\sum_j
\log p_\theta(x_{t,j}\mid x_{t,<j}).
$$

正文给出一般形式 `w(r_t)=sigma(alpha(r_t-beta))`。附录给出实际设置：

$$
\boxed{w(r_t)=2\sigma(2(r_t-1))}.
$$

即 `alpha=2`、`beta=1`，外乘 2 使中性奖励 `r_t=1` 映射到权重 1。

| `r_t` | `w(r_t)` | 作用 |
|---:|---:|---|
| `0.0` | `0.2384` | 错误步骤明显降权，但不硬删除 |
| `0.2` | `0.3363` | 错误回退步骤低权重 |
| `1.0` | `1.0000` | 中性步骤保持原权重 |
| `1.8` | `1.6637` | 有用线索步骤加强 |
| `2.0` | `1.7616` | 高质量步骤平滑加强 |

若一步 3 个 token 的 NLL 为 `0.4,0.6,0.5`，原损失为 `1.5`。`r_t=1.8` 时贡献 `1.6637*1.5≈2.4956`；`r_t=0.2` 时贡献 `0.3363*1.5≈0.5045`。这是连续重加权，不是硬过滤；正确和错误的完整轨迹都保留。

## 6. ABC-GRPO

第二阶段从 ABC-SFT checkpoint 做 online RL。第 `i` 条 rollout 的第 `t` 步有

$$
R_{i,t}=r_{i,t}.
$$

在 rollout group 内归一化得到 `R_hat_{i,t}`，再计算折扣步骤优势：

$$
A_{i,t}=\sum_{k=t}^{T_i}\gamma^{k-t}\widehat R_{i,k},
\qquad \gamma=0.25.
$$

因此不是把最终 0/1 奖励复制给整条轨迹：每一步先有自己的 reward，未来 reward 再以 `0.25,0.25^2,...` 的权重影响更早步骤。

### 6.1 数值计算案例

假设同一问题有两条长度为 3 的 rollout，组内归一化奖励如下：

| rollout | `t=1` | `t=2` | `t=3` |
|---|---:|---:|---:|
| A：`R_hat` | `+1` | `-1` | `-1` |
| B：`R_hat` | `-1` | `+1` | `+1` |

取 `gamma=0.25`：

$$
A_{A,1}=1-0.25-0.25^2=0.6875,\quad
A_{A,2}=-1-0.25=-1.25,\quad A_{A,3}=-1;
$$

$$
A_{B,1}=-1+0.25+0.25^2=-0.6875,\quad
A_{B,2}=1+0.25=1.25,\quad A_{B,3}=1.
$$

该表只用于解释公式，论文没有把这个简化归一化例子作为固定算法。

`A_{i,t}` 分配给该步骤所有 policy-generated tokens，环境 tool-response tokens 被 mask，再代入标准 clipped GRPO：

$$
\mathcal L_{\mathrm{GRPO}}=-\mathbb E\left[
\min\left(\rho A_{i,t},\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A_{i,t}\right)
\right]+\mathcal L_{\mathrm{KL}},
$$

其中 `rho=pi_theta/pi_old`。RL 还加入 format penalty（格式惩罚），KL loss coefficient 为 `0.001`。所以 ABC-GRPO 是 GRPO policy-gradient 更新，不是 OPD loss。

## 7. 训练配置

### SFT

- OpenSeeker 轨迹 `8.5K` 条：`5.5K` 条最终正确、`3.0K` 条最终错误；
- Qwen3.5-4B，3 epochs，global batch size 64；
- Adam，学习率 `5e-5`，cosine decay，warmup ratio `0.1`，最小学习率 `1e-6`，weight decay `0.1`；
- tensor model parallel 2，context parallel 8，每条轨迹最多 200 个交互步骤；
- 工具返回不计算 loss；线索恢复和步骤评分使用 DeepSeek-V4-Flash。

### RL

- 从 ABC-SFT checkpoint 初始化；1,000 个问题，每题 8 条 rollout；batch size 16；
- 每条 rollout 最多 200 步；异步 rollout，16 个 agent-loop workers；
- temperature `1.0`，top-p `0.95`；`gamma=0.25`；actor learning rate `1e-6`；KL 系数 `0.001`；
- clue-anchored step reward 加 format penalty，并在 rollout group 内归一化。

## 8. Benchmark 与 baseline

| Benchmark | 中文解释 | 主要考察 |
|---|---|---|
| BrowseComp | 长程英文网页浏览与信息搜寻 | 多约束、多跳检索，跨网页拼接证据 |
| BrowseComp-ZH | BrowseComp 中文网页版本 | 中文网络检索与证据核验 |
| xbench-2505 / xbench-2510 | xbench 两个版本 | 深度研究、规划、推理、跨来源综合 |
| GAIA-text | GAIA 纯文本子集 | 通用助理、工具使用和多跳推理 |

Baseline 包括工具型 foundation models（Gemini-3.1-Pro、Seed-2.0-Pro、GLM-5、DeepSeek-V4-Pro-Max、GPT-5 High），约 30B 搜索 Agent（MiroThinker-1.7-mini、RedSearcher、DeepMiner、Tongyi-DeepResearch、OpenSeeker），以及 4B 搜索 Agent（QUEST-4B、DR-Venus、AgentCPM-Explore）。评测最多 200 次 tool calls，每项运行 3 次取平均。

## 9. 完整主结果表

数值为百分比。`*` 表示无 context management；ABSeeker 的前两项用“无上下文 / 有上下文”表示。

| 模型 | 参数量 | BrowseComp | BrowseComp-ZH | xbench-2505 | xbench-2510 | GAIA-text |
|---|---:|---:|---:|---:|---:|---:|
| Gemini-3.1-Pro | - | - | **85.9** | - | 53.0 | **80.6** |
| Seed-2.0-Pro | - | 77.3 | 82.4 | - | - | 78.6 |
| GLM-5 | 358B | 75.9 | 72.7 | - | - | - |
| DeepSeek-V4-Pro-Max | 1.6T | **83.4** | - | - | **80.0** | - |
| GPT-5 High | - | 54.9 | 63.0 | 77.9 | 75.0 | 76.4 |
| MiroThinker-1.7-mini | 30B | **67.9** | **72.3** | - | 57.2 | **80.3** |
| RedSearcher | 30B | 57.4 | 58.2 | - | - | 80.1 |
| DeepMiner | 32B | 33.5 | 40.1 | 62.0 | - | 58.7 |
| Tongyi-DeepResearch | 30B | 43.4* | 46.7* | **75.0** | - | 70.9 |
| OpenSeeker | 30B | 29.5* | 48.4* | 74.0 | - | - |
| QUEST-4B | 4B | 40.0 | - | - | - | 77.7 |
| DR-Venus | 4B | 29.1* | 37.7* | 74.7 | 40.7 | 64.4 |
| AgentCPM-Explore | 4B | 24.1* | 29.1* | 70.0 | 34.0 | 63.9 |
| **ABSeeker** | **4B** | **37.3* / 55.3** | **39.1* / 52.9** | **77.0** | **46.0** | **81.6** |

ABSeeker 在 4B Agent 类别中的五项有报告指标均为最高，但无上下文 BrowseComp 的 `37.3` 低于 QUEST-4B 的 `40.0`；加入上下文管理后是 `55.3`。它在 xbench-2505 和 GAIA-text 上超过表中已报告的 30B Agent。

## 10. 完整消融表与结论

以下均不使用 context management。

| 模型/训练阶段 | BrowseComp | BrowseComp-ZH | xbench-2505 | xbench-2510 | GAIA-text |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-4B + Standard SFT | 28.5 | 30.4 | 73.0 | 27.0 | 66.0 |
| Qwen3.5-4B + ABC-SFT | 30.8 | 31.8 | 72.0 | 35.0 | 72.8 |
| ABSeeker-4B-SFT + Standard GRPO | 33.5 | 36.3 | 75.0 | 41.0 | 77.7 |
| ABSeeker-4B-SFT + ABC-GRPO | **37.3** | **39.1** | **77.0** | **46.0** | **81.6** |

ABC-SFT 相对 Standard SFT 的变化为 `+2.3, +1.4, -1.0, +8.0, +6.8`；ABC-GRPO 相对 Standard GRPO 为 `+3.8, +2.8, +2.0, +5.0, +3.9` 个百分点。即 ABC-SFT 在 4/5 个指标提升，ABC-GRPO 在五项全部提升。

## 11. 额外实验观察

- 成功轨迹中约 `4%` 的步骤奖励低于 `1.0`，说明答对不代表每一步都好；
- 失败轨迹中接近 `10%` 的步骤奖励高于 `1.0`，说明答错不代表每一步都无用；
- 在 200 个 BrowseComp 验证问题上，ABC-GRPO 比轨迹级 GRPO 表现更强，并产生更长的搜索轨迹；
- 最大 256K context、最多五轮 discard-all strategy（全部丢弃策略）下，BrowseComp 从 `37.3` 到 `55.3`，BrowseComp-ZH 从 `39.1` 到 `52.9`。

最后一项是推理时上下文工程收益，不能全部归因于 ABC。论文图 4--6 未给出完整离散数据，本文不把图中估读值写成精确数字。

## 12. 优点、局限及与 OPD 的边界

优点是用可验证答案反向构造过程监督，保留失败轨迹中的有用步骤，并惩罚成功轨迹中的错误步骤。局限是线索恢复和评分依赖 DeepSeek-V4-Flash，成本和评分误差都不可忽略；任务还必须有可靠的 verified answer，clue set 不完整也会漏掉合理行为。

OPD/OPSD 通常在 student 的 on-policy token 上利用 teacher/snapshot logits 计算 KL；ABSeeker 没有 teacher logits、RKL/FKL 或 prefix replay。它属于 process-supervised RL（过程监督强化学习）/ reward shaping（奖励塑形），而不是 policy distillation（策略蒸馏）。

论文的核心命题是：对有可验证终点答案的长程搜索任务，可以把终点反向转成固定的中间证据锚点，据此给每个搜索步骤分配更合理的训练信用。
