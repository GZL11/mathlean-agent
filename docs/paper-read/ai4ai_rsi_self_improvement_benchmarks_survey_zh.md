# AI4AI / RSI 自我改进类 Benchmark 全景调研

本文档调研 "AI4AI"（用 AI 构建/训练/优化 AI）与 RSI（Recursive Self-Improvement，递归自我改进）方向下的评测体系，覆盖三个子方向：

```text
1. 模型自进化 / 自训练：AI agent 能否自主完成模型训练/后训练全流程
2. Memory 自进化：agent 能否自主积累、编辑、压缩、检索自己的记忆并因此变强
3. Harness 自进化：agent 能否自主设计/优化自己的脚手架（工具/prompt/编排/记忆接口）
```

调研时间：2026-09-01。方法：多个子 agent 并行联网检索（WebSearch 为主，部分域名如 arxiv.org / GitHub 在当前环境下被 WebFetch 拦截，只能通过搜索引擎摘要核实），信息可能有转述误差，关键数字建议对照原文复核。未能确认存在的名称均已明确标注"未找到"，不做编造。

---

## 目录

```text
一、模型自进化 / 自训练类 benchmark
二、Memory 自进化类 benchmark
三、Harness 自进化类 benchmark / 方法
四、跨方向共性与空白
五、链接速查表
```

---

## 一、模型自进化 / 自训练类 benchmark

### 1.1 PostTrainBench

- **论文**：*PostTrainBench: Can LLM Agents Automate LLM Post-Training?*，[arXiv:2603.08640](https://arxiv.org/abs/2603.08640)，ICML 2026 poster。GitHub：[aisa-group/PostTrainBench](https://github.com/aisa-group/PostTrainBench)
- **核心思路**：给 agent 一个 base LLM（Qwen3-1.7B/4B、SmolLM3-3B、Gemma-3-4B）+ 一个目标下游 benchmark（AIME2025/GSM8K/GPQA/HumanEval/BFCL/ArenaHard/HealthBench）+ 单张 H100、10 小时算力和网络访问，不提供任何起始代码/数据/超参，让 agent 从零构建完整后训练 pipeline（数据、方法、超参全自主选择）。
- **输入格式**：隔离 workspace，只读软链的 base 模型权重 + 训练数据入口 + 可写产出目录；agent 通过 CLI scaffold（Claude Code / Codex CLI / Gemini CLI / OpenCode）自由操作。
- **输出格式与评测方式**：agent 产出一个后训练完的模型 checkpoint，由自带的 grading server 读取该 checkpoint 在目标 benchmark 上跑推理评分。**LLM-judge 双重角色**：既做污染检测器（训练集混测试集、模型替换、蒸馏外部模型等作弊直接判 0 分改进）又做约束执行器。
- **评测指标**：4 个 base 模型 × 7 个下游 benchmark = 28 个配置，每配置跑 3 次估计方差；总分是 7×4 的**加权平均**（相对官方 instruct 模型的相对表现）。
- **关键结果**：最强 agent（Claude Opus 4.6）仅达到官方指令微调模型水平的 **23.2%**（相对值）。
- **公开 Leaderboard**：✅ 有，官网 [posttrainbench.com](https://posttrainbench.com/) 持续更新；第三方镜像 [llm-stats.com/benchmarks/posttrainbench](https://llm-stats.com/benchmarks/posttrainbench)、[epoch.ai/benchmarks/post-train-bench](https://epoch.ai/benchmarks/post-train-bench)；完整 agent 轨迹数据集见 [HuggingFace](https://huggingface.co/datasets/aisa-group/PostTrainBench-Trajectories)。

### 1.2 Agent² RL-Bench —— PostTrainBench 的延伸：从"能不能后训练"到"能不能做完整 RL pipeline 工程"

- **论文**：*Agent² RL-Bench: Can LLM Agents Engineer Agentic RL Post-Training?*，[arXiv:2604.10547](https://arxiv.org/abs/2604.10547)，Microsoft Research Asia / Soochow University / Peking University / Stony Brook University / University of Chicago
- **核心思路**：PostTrainBench 里 SFT 就能拿到不错的分数，静态、单轮任务测不出"agent 能不能做交互式 RL 工程"这件事。Agent² RL-Bench 专门把"agentic RL post-training"定义为一项长时程系统工程挑战——agent 要理解训练目标、选策略、搭数据 pipeline、实现环境交互和 rollout 收集、管理轨迹级 reward、诊断失败、迭代到模型真的变强，而不只是写一段能跑的代码。
- **资源/时间限制（论文 Appendix A.4 Table 7）**：**所有 6 个任务统一固定 12 小时单次 run**，硬件为**单卡 H200**（正文 HumanEval trace walkthrough 明确写"12-hour budget on a single H200 GPU"）。两组主实验配置：(1) 7B 主对比——Agents 为 OpenHands/OpenCode/Claude Code，driver 为 GPT-5.4(OH/OC)/Claude Opus 4.6(CC)，base 模型 Qwen2.5-7B-Instruct，每 system/task/mode 跑一次 12h；(2) 8B 受控 CLI 研究——Scaffold 为 Claude Code/Codex CLI/Gemini CLI，driver 覆盖 Opus 4.6/Sonnet 4.5/GPT-5.4/GPT-5.2/GPT-4o/Gemini 2.5-flash，base 模型 Qwen3-8B-Base（未对齐），每 system 跑一次 12h。另有一个**扩展算力对照实验**：针对最难的 DeepSearchQA，专门给 Claude Code(Opus 4.6) vs Codex(GPT-5.4) **12 小时 × 2×H200**，测试多给一张卡能否突破瓶颈——结果只从早期本地 sweep 小幅提升到 23.0/21.0，仍明显偏低，说明该任务困难是结构性的、不是算力不够。评测集规模：ALFWorld 134 episode、WebShop 100 条指令、DeepSearchQA 200 条 held-out 样本，judge 统一用 GPT-4o。
- **输入/输出格式**：统一的 agent 接口，形式化为 `(agent scaffold, task, base model, time budget)` 四元组；每次 run 从一个隔离 workspace 开始，内含 base 模型、任务数据、指令、一个 grading API；agent 在固定预算内反复训练模型、提交 artifact 评测。runtime 会记录每一次提交和代码修订，并自动生成结构化 run report，做训练行为的事后诊断。
- **任务设计**：6 个任务分 3 个难度层级，一层比一层多加一个结构性要求：
  - **L1 静态规则式后训练**：GSM8K(数学)、HumanEval(代码)——单轮输出、确定性规则 verifier，只测基础数据构造+监督训练能力。
  - **L2 静态 judge 式后训练**：AlpacaEval 2.0——仍是单轮非交互，但 reward 换成非确定性的 LLM-judge 偏好信号。
  - **L3 交互式 rollout 任务**（关键层级）：ALFWorld(稀疏二元reward,长时程)、WebShop(稠密连续reward,长时程)、DeepSearchQA(judge式)——要求完整环境交互、多步轨迹收集、闭环在线 RL。
- **评测指标**：区分 benchmark 原生输出和离线诊断两类指标，分开报表；process metric 由保存的 run artifact 确定性算出，AlpacaEval 2.0 用 Azure 上托管的 GPT-4o judge。
- **关键结果**：5 个 agent 系统 × 6 个 driver LLM 测试——ALFWorld 上纯 RL agent 通过 SFT 热身+GRPO 在线 rollout 把分数从 5.97 拉到 **93.28**；但 DeepSearchQA 上进步几乎在噪声范围内(+2.75)，即便是有完整网络访问的前沿 agent 官方 leaderboard 也只有 66% 全对，说明这个任务结构性地难；**同一个 scaffold 换 driver LLM,交互式任务提升幅度能从接近 0 摆动到 +78 个百分点**——driver 模型本身比 scaffold 设计更决定 RL post-training 能不能做成；且 6 个任务里有 5 个,agent 最先尝试 RL 方法但最终都退回 SFT-based pipeline,说明当前 agent 在资源/时间受限时更信任监督式流程。
- **公开 Leaderboard**：未找到独立持续更新的公开网页，代码见 Microsoft Research 发布页及关联的 [RD-Agent GitHub](https://github.com/microsoft/rd-agent)，目前是论文内实验结果。

### 1.3 RSIBench-Data

- **论文**：*RSIBench-Data: Benchmarking Data-Centric Research for Recursive Self-Improvement*，[arXiv:2607.25886](https://arxiv.org/abs/2607.25886)，Fanqing Meng et al.（Evolvent AI + NUS）。GitHub：[evolvent-ai/RSIBench-Data](https://github.com/evolvent-ai/RSIBench-Data)，项目页 [rsibench.co](https://rsibench.co/)
- **核心思路**：固定训练/推理 stack（Tinker 后端训练+服务、Harbor+E2B 沙箱评测），只让 agent 迭代式地为固定目标模型设计/修正**训练数据策略**，从而把"agent 的数据研究能力"从优化器/系统实现等因素中隔离出来。
- **资源/时间限制**：每个主 run **名义上限是 16 小时 wall-clock + $500 Tinker 预算**（不涉及具体 GPU 型号/数量，训练/serving 全部走 Tinker 托管服务）；论文报告的 24 个主 run 实际用量中位数是 **6.55 小时、$62.51**（Tinker 有效候选成本），但方差很大——耗时从 1.14 到 14.91 小时、成本从 $4.80 到 $363.77 不等，且**花费高不代表分数高**（例如 Claude Code Sonnet-5 在 Terminal-Bench 2.0 上花 8.87 小时/$156.93 只拿 5.62%，Codex gpt-5.6-sol 在 GPQA Diamond 上只花 2.42 小时/$10.37 就拿到 65.00%）。
- **下游 6 个 benchmark（精确名称）**：SWE-bench Verified、SWE-bench Multilingual、SWE-bench Pro（三个变体用 Mini-SWE-Agent 跑）、Terminal-Bench 2.0、GPQA Diamond、AIME 2026（后三个用 Terminus-2 跑）；SWE-bench 三变体和 GPQA Diamond 用固定 100 条子集，Terminal-Bench 2.0 和 AIME 2026 用全集（AIME 每题跑 4 次解码）。
- **输入格式**：固定 base 模型（Qwen3.5-35B-A3B-Base）+ 固定优化器/serving 路径/评测沙箱与打分规则，agent 只能动"训练数据策略"这一个变量。
- **输出格式与评测方式**：agent **提交训练产物（数据/数据生成配置），不提交训练代码**，也不能替换 stack 其它组件；Runner 校验后用同一套 Tinker-backed LoRA SFT 后端训练出 checkpoint。
- **评测指标**：不是单一 scalar，而是记录整个研究过程（假设、数据策略、基于反馈的修订、checkpoint 选择），核心统计量是轨迹层面的**改进率/退化率**。
- **关键结果**：4 个前沿 agent 在 6 个下游 benchmark（软件工程/终端使用/科学问答/数学等）上测试：**58.33%** 的设置中 agent 能基于反馈优于第一次有效尝试；但持续搜索后 **78.26%** 的 run 最终得分反而低于历史最高分——说明 agent 有"发现"能力但难以把反馈稳定转化为持续改进。
- **公开 Leaderboard**：官网 [rsibench.co/data](https://rsibench.co/data/) 是项目介绍页，非严格排行榜；另有一个从属关系未确认的第三方追踪站 [harness-rsibench.com](https://harness-rsibench.com/)，标注具体 harness 版本（Claude Code 2.1.87、Codex 0.147.0、Pi 0.82.1），需谨慎引用。

### 1.4 nanochat / NanoGPT Speedrun 生态（"给 agent 一个极简训练脚本，让它自己迭代改进"这一类）

这不是单一 benchmark，而是围绕同一套极简 LLM 训练代码衍生出的一整个生态，2026 年内又长出了好几个正式化的 benchmark。**各家给 agent 的算力/时间预算差异极大**，这一点本身就是理解这个生态的关键，逐一列出并标明资源限制：

#### Karpathy's Autoresearch（nanochat）

- Karpathy 在 X 上明确将 nanoGPT/nanochat 定位为 "recursive self-improvement benchmark"（[原帖](https://x.com/karpathy/status/1939709449956126910)），GitHub：[karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- **核心思路**：极简三文件仓库——`prepare.py`（固定数据预处理，agent 不可改）、`train.py`（**唯一允许 agent 编辑的文件**，含 GPT 模型/Muon+AdamW 优化器/训练循环）、`program.md`（人类基线说明）。设计哲学是"一块 GPU、一个文件、一个指标"。
- **资源/时间限制**：**单卡 GPU**，单轮训练预算固定为 **5 分钟**；agent/实验串行执行，吞吐量约 12 次实验/小时；Karpathy 原始跑法总耗时约 **2 天**（约 700 次实验），总成本量级是"几十美元"而非几万美元。
- **输入/输出**：agent 每轮对 `train.py` 打 diff 并触发训练；用 **git commit 作为状态机**——新分数优于历史最佳则保留该 commit，否则 `git reset` 回退，git 历史本身就是实验日志。
- **评测指标**：核心指标是 **val_bpb（bits-per-byte）**（逐 token 交叉熵之和除以目标文本总字节数再换算成 bit），使分数在分词器变化时仍可比较。
- **关键结果**：约 700 次实验，约 20 个改动最终迁移进代码库，达到 GPT-2 水平耗时从 2.02h 压缩到 1.80h；agent 发现了一个 QK-Norm 缺失 scalar 乘子的 bug，Karpathy 本人此前都没注意到。
- **后续扩展（SkyPilot）**：把同一套 loop 搬到 GPU 集群上并行跑，吞吐提到约 90 次实验/小时（单卡约 10 次/小时的 9 倍），**8 小时内提交约 910 次实验（约 700 次有效）**，val_bpb 从 1.003 降到 0.974——相当于把 Karpathy 单卡 2 天的量压缩到 8 小时。
- **公开 Leaderboard**：无，本质是开源仓库+一次性实验记录。

#### NanoGPT Speedrun（Keller Jordan, modded-nanogpt）—— 这条生态的源头，本身就有社区 leaderboard

- GitHub：[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
- **核心思路**：在**固定硬件**（单个 **8×H100 节点**）上，把语言模型训练到目标验证 loss 所需时间最短，社区持续提交新记录。两条赛道：small track（目标 loss 3.28，从 GPT-2-small 124M 起步）、medium track（目标 loss 2.92，从 GPT-2-med 350M 起步）。
- **资源/时间限制**：**没有单次提交的时间上限**——这是一个持续进行、没有截止日期的社区竞赛（自 2024-05 起），"训练到目标 loss 的 wall-clock 时间"本身就是被最小化的指标，而非受限资源。
- **关键结果**：2024-05 到 2026-03，36 位贡献者提交 77 条记录，训练时间从 45 分钟压缩到 1.43 分钟（**31 倍加速**）；产出的 Muon 优化器已进入 GLM-4.5、Kimi K2 等前沿模型。截至 2026-04，已有 4 条记录由 AI agent 贡献。
- **公开 Leaderboard**：✅ 有，[GitHub README 内嵌排行榜](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file)，持续接受社区提交（人类+agent 混合）。

#### NanoGPT-Bench（Intology）—— 在 NanoGPT Speedrun 上正式构建的 agent 评测

- GitHub：[IntologyAI/NanoGPT-Bench](https://github.com/IntologyAI/NanoGPT-Bench)
- **核心思路**：把 NanoGPT Speedrun 包装成对 agent 长时程前沿 ML 研究能力的正式评测。
- **资源/时间限制**：agent 拿到约 **2.7 天**（折合约 **512 H100-hour**）在**一个 8×H100 节点**上的算力，且**不联网**；起点是 2025-09-03 的人类世界记录训练脚本。
- **评测指标**：以"追平人类同期进步幅度"为标尺——统计 agent 相对 2025-09-03 至 2026-01-19 这 5 个月内人类新增加速幅度的**恢复比例**；并统计 agent 把预算花在超参调整 vs 算法性改动上的占比。
- **关键结果**：三个前沿 agent（Codex GPT-5.4 xhigh、Claude Code Opus 4.6 Max、Autoresearch 式 prompting 的 Claude Code）**全部只恢复了人类同期进步的不到 10%**；agent 把大部分算力花在超参搜索上，而人类同期 **77%** 的世界记录来自算法性改动。
- **公开 Leaderboard**：GitHub 仓库内维护结果，未确认是否有独立网页。

#### The Automated LLM Speedrunning Benchmark

- [arXiv:2506.22419](https://arxiv.org/abs/2506.22419)
- **核心思路**：同样基于 NanoGPT speedrun 社区，任务锚定在**固定硬件目标**（单个 8×H100 节点，验证 loss 目标 3.28）上；给 agent"上一条记录的训练脚本"+ 三种可选提示格式之一（从伪代码到类论文式的改进描述），测试前沿推理模型+scaffold 组合能否**复现**已知的历史突破，覆盖 21 个连续 speedrun 记录/任务。
- **资源/时间限制**：不是单次固定时间上限，而是"以上一条记录的训练时间为目标"逐任务复现；论文报告的**总实验算力**为约 **6,840 × 8-H100 节点-小时**（合约 **54,720 个 H100-GPU-小时**）的内部集群算力。
- **和其它 speedrun 类的区别**：不是要求 agent 发现全新改进，而是测"给定不同粒度的提示，agent 能否照着复现出该有的改进"，用来诊断 agent 在有/无提示下的能力差距。

#### Prime Intellect Autonomous Speedrunning

- 项目页/正式 speedrun 页：[app.primeintellect.ai/speedrun/nanogpt](https://app.primeintellect.ai/speedrun/nanogpt)，早期实验页：[primeintellect.ai/auto-nanogpt](https://www.primeintellect.ai/auto-nanogpt)，GitHub：[PrimeIntellect-ai/experiments-autonomous-speedrunning](https://github.com/PrimeIntellect-ai/experiments-autonomous-speedrunning)
- **核心思路**：把 agent（Claude Code/Opus、Codex/GPT）投入 Keller Jordan 的 modded-nanogpt **优化器赛道**（模型/数据/架构全部固定，agent 只能改优化器、调度、初始化等超参）打连续多"波"（wave）实验；引入**novelty-gated**（新颖性门控）变体，要求 agent 必须通过新颖性检查才能提交（不能只是重组公开 leaderboard/论文/历史 run 的内容）。
- **资源/时间限制**：用团队**空闲算力**，**两周**时间窗口，**约 1 万次 run，约 1.4 万 H200-hours，23.9B tokens**。
- **波次结构**：第一波目标追上 Muon 优化器基线（3500 步）；第二波 novelty-gated（每个想法须过新颖性检查）；第三波从 v1/novelty 最佳结果出发冲击 3000 步以下；第四波从 v2 前沿+社区公开 PR 出发冲击 2900 步以下。
- **关键结果**：novelty-gated 设置下 agent 一开始无法改进 baseline；给更多上下文做 continuation run 后逐步进步——v2 两个 agent 都突破到 3035 步；v3 中 Claude Opus 4.7 达到 **2930 步**、Codex GPT-5.5 达到 **2950 步**（**均超过人类基线 2990 步**）。
- **公开 Leaderboard**：Prime Intellect 官方 speedrun 页面本身即是一个持续榜单（人类+agent 混合提交）。

#### NanoGPT Speedrun Frontier —— Prime Intellect 后续研究，BenchmarkList 做第三方追踪展示

- Prime Intellect 原始研究页：[primeintellect.ai/research/nanogpt-speedrun](https://www.primeintellect.ai/research/nanogpt-speedrun)，博客：[Measuring Autonomous AI Research](https://www.primeintellect.ai/blog/measuring-autonomous-research)；第三方聚合展示：[benchmarklist.com/benchmarks/nanogpt_speedrun_frontier](https://benchmarklist.com/benchmarks/nanogpt_speedrun_frontier/)
- **核心思路**：把"自主 coding agent 独立完成 speedrun 任务"扩大规模，做成一个跨模型的标准化排行榜。
- **资源/时间限制**：**每次 run 最长 8 天，每次 run 独占 8×H200**；覆盖 153 次自主 run、18 个前沿模型（每模型多个随机种子），harness 包括 claude-code、codex、kimi-code、grok-cli、qwen-code 等。
- **评测指标**：每个模型的最佳已验证记录 + "**缺口消解比例**"——在 3290（baseline）到 2600（人类记录）这个区间里，agent 消除了多少百分比的差距。
- **公开 Leaderboard**：✅ 有，即 BenchmarkList 该站点。

#### METR "Expenditure Horizon" 研究 —— 专门衡量"AI 相对人类的成本效益临界点"

- [metr.org/blog/2026-07-21-expenditure-horizon](https://metr.org/blog/2026-07-21-expenditure-horizon/)
- **核心思路**：不满足于单纯比较分数，而是把 token 成本、实验算力成本、人类人力成本都换算成美元，找到"预算超过多少时人类比 AI 更划算"的 **expenditure horizon（花费临界点）**。
- **资源/时间限制**：从 speedrun 第 78 条记录（2026-03，85.56 秒训练时间）出发，**6 个 agent**，每个 **wall-clock 上限 5 天、预算上限 $10K（含 GPU 成本）**，硬件固定为 **8×H100**。
- **关键结果**：估算人类每提升 1% 的边际成本约 **$2500**（约 16 小时人力，时薪假设 $150）；agent 花费超过 $10K 后，估计的 expenditure horizon 落在 **$0-3K** 区间——也就是说在这个预算量级，人类已经比 AI 更划算；speedrun 维护者评价 AI 生成的想法约 70% 原则上可整合进项目，但大多不算真正原创。

#### AutoScientists（Harvard）

- [arXiv:2605.28655](https://arxiv.org/abs/2605.28655) · GitHub：[mims-harvard/AutoScientists](https://github.com/mims-harvard/AutoScientists)
- **核心思路**：在 nanochat 训练优化任务上做多 agent 协作（团队自组织、噪声感知晋升门、死胡同登记），验证跨域可迁移性；系统本身是 Claude Code 子 agent 通过本地 ClawInstitute server 协调，orchestrator 只负责调度和收集结果，自己不训练任何东西。
- **资源/时间限制**：与 Autoresearch 同源——**单次实验＝单卡 H100 上 5 分钟 GPT 训练 run**；AutoScientists 默认配置用 **9 个 agent**（6 个实验+3 个分析），完整对照跑了 **71 次实验、11 次 accepted、总耗时约 6.4 小时**；Autoresearch 单 agent 基线在同代码库/同模型后端下跑了 **83 次实验、15 次 accepted**。
- **关键结果**：val_bpb 从起点 0.998 降到 0.9777（AutoScientists）；接受的改进分散在吞吐量/容量/优化器质量三个不同维度，而非集中在单一方向；团队规模扫描显示 n=9 综合效果最好，n=14 反而变差（0.9821）。

**跨实验室资源预算对比一览**（同一个 nanoGPT speedrun 生态下，各团队给 agent 的算力/时间差异极大）：

| 团队/工作 | 硬件 | 时间/预算上限 | 备注 |
|---|---|---|---|
| Anthropic（自动化 AI R&D evals） | 仅 CPU node | — | 未使用 GPU |
| OpenAI（nanoGPT track 1） | 1×H100 | 不到 1 天 | |
| Karpathy Autoresearch 原版 | 单卡 GPU | 约 2 天（约 700 次 5 分钟实验） | 成本"几十美元"量级 |
| AutoScientists | 单卡 H100 | 约 6.4 小时（71 次 5 分钟实验，9 agent） | |
| Intology NanoGPT-Bench | 8×H100 节点 | 约 2.7 天（约 512 H100-hour），不联网 | |
| METR Expenditure Horizon | 8×H100 | 5 天 / $10K 上限（含 GPU 成本），6 个 agent | |
| Prime Intellect（单模型 waves） | — | 两周窗口，约 1 万次 run，约 1.4 万 H200-hours | 用团队空闲算力 |
| Prime Intellect（NanoGPT Speedrun Frontier） | 每次 run 独占 8×H200 | 每次 run 最长 8 天，153 次 run×18 模型 | |
| Automated LLM Speedrunning Benchmark | 单个 8×H100 节点（任务目标） | 总实验算力约 6840×8-H100 节点-小时 | 21 个连续复现任务 |

#### AI4AI-Bench 对该生态的批评（重要的方法论提醒）

AI4AI-Bench 论文（见 1.5）直接点名批评 Karpathy 的 autoresearch：一次 5 分钟的单脚本运行算不上一个研究仓库；用经典优化器（CMA-ES、TPE）做对照，agent 的编辑行为更像超参搜索而非算法设计；论文核心论点是"编辑代码"不等于"设计算法"——区别不在改动大小，而在改动的对象：超参是训练算法认为给定的一个数字，而算法性改动是重写算法本身。这提醒我们：nanochat 系生态里的"进步数字"未必都对应真正的算法创新，需要结合 NanoGPT-Bench 的"算法改动占比 77% vs agent 的超参搜索为主"这一发现一起看。

- **该生态整体的公开 Leaderboard 情况**：NanoGPT Speedrun（GitHub README）、Prime Intellect Speedrun 页、NanoGPT Speedrun Frontier（BenchmarkList）**三个是持续更新、可比较的公开榜单**；Karpathy 原版 autoresearch、NanoGPT-Bench、Automated LLM Speedrunning Benchmark、METR Expenditure Horizon、AutoScientists 则仍停留在论文/仓库内一次性实验记录。

### 1.5 AI4AI-Bench

- **论文**：*AI4AI-Bench: Benchmarking LLM Agents in Algorithmic Design for Recursive Self-Improvement*，[arXiv:2608.20318](https://arxiv.org/abs/2608.20318)（2026-08-24），Yizhe Chi 等，Navers Lab / Einsia.AI / 清华。GitHub：[Einsia/AI4AI-Bench](https://github.com/Einsia/AI4AI-Bench)，项目页：[lab.einsia.ai/ai4ai](https://lab.einsia.ai/ai4ai)
- **核心思路**：与 PostTrainBench/RSIBench-Data 不同，专门隔离出"**算法设计**"这一层——不是考数据策略、不是考超参搜索，而是要求 agent 重写训练算法本身（目标函数/更新规则）。论文明确点名批评 Karpathy 的 autoresearch："一次 5 分钟的单脚本运行算不上一个研究仓库；用经典优化器（CMA-ES、TPE）做对照，agent 的编辑行为更像超参搜索而非算法设计"；也提到 **MLS-Bench**（140 个任务，agent 改 ML 系统某个组件）是除 autoresearch 外最接近的相关工作，但其组件边界是人为划定的，把执行层面的改进和学习算法层面的改进混在一起打分。核心论点：区别不在改动大小，而在改动的对象——超参是训练算法认为给定的一个数字，算法性改动是重写算法本身（损失函数、更新规则）。
- **资源/时间限制**：单卡 **B300**；探索阶段 `T_e = 4 小时`（agent 可反复查询 proxy metric，但看不到最终 metric）；提交后代码在全新容器里从零重跑，验证阶段 `T_v = 12 小时`；agent 提交的**只是源代码**（重写后的 repo），不能提交训练好的权重或缓存状态，提交后就不能再修改/交互。

**完整的 10 个任务**（论文 Appendix A + 正文 2.2 节）：

| 任务 | 算法家族 | 起始 base 模型 | Proxy metric（探索期可查） | 最终 metric（agent 看不到） |
|---|---|---|---|---|
| **OpenR1** | Supervised Fine-Tuning | Qwen2.5-Coder-1.5B-Instruct（8005 行去污染 Python CodeForces 数据，completion-only SFT） | `livecodebench_public_pass_at_1` | LiveCodeBench v6 全量 175 题，10 samples/题，temp 0.2，top-p 0.95，截断 2048 token |
| **RAGEN** | Multi-turn Agentic RL | Qwen2.5-3B-Instruct（Sokoban，multi-turn on-policy GRPO，在线生成棋盘/轨迹） | 4 组棋盘的 solve rate | held-out 512 棋盘的 solve rate（不同 seed 协议） |
| **OPD** | On-Policy Distillation | 1.5B 学生模型，从挂载的 teacher 做 sampled-token on-policy distillation | MATH-500（4 samples/题，12288 token 上限） | AIME 2024+2025 共 60 题，32 samples，31744 token 上限（proxy 和最终指标是**完全不同的两个 benchmark**） |
| **BTRM** | Bradley-Terry Reward Modeling | 固定 Mistral-7B 起点，去污染 UltraFeedback 偏好对 | 2985 对中可见的 512 对 | 剩余 2473 对（留出）；和 RewardBench 有重叠则该 run 直接判无效 |
| **DPO** | Preference Optimization | 一个 merged 的 Zephyr/Mistral-7B 模型 | — | IFEval prompt-level strict accuracy，413 条 held-out prompt |
| **DDPO** | Diffusion RL | Stable Diffusion v1.5 + LoRA adapter，on-policy DDPO 对齐一个冻结的 CLIP 美学 reward | 64 张生成图打分 | 256 张图打分（另附报 CLIP alignment、图像间平均距离） |
| **NPO** | Machine Unlearning | Llama-3.2-1B-Instruct，TOFU forget10 协议，官方 OpenUnlearning NPO 配方 | 只挂载已发布 anchor + train-role 数据 | extraction strength（越低越好）与 model utility（越高越好）的调和平均（"balanced score"），retain-90 anchor 只在评分阶段出现 |
| **DiGress** | Discrete Graph Diffusion | QM9（无氢）上训练的离散图扩散模型 | validity×uniqueness×novelty 乘积（越高越好） | 上游测试集负对数似然（越低越好） |
| **Model Soup** | Weight Averaging（模型融合） | 72 个固定 CLIP ViT-B/32 checkpoint（默认均匀平均） | 2000 张 ImageNet-V2 | 完整 10000 张 ImageNet-V2 |
| **OWL** | One-shot Pruning | 稠密 OPT-6.7B，单次 activation-aware OWL/Wanda 剪枝（不做微调） | WikiText-2 验证集困惑度（硬性 gate：decoder 稀疏度须落在 [0.699, 0.701]） | WikiText-2 测试集困惑度 |

其中 **Model Soup 和 OWL 两个任务原生"不训练"**（一个组合已有 checkpoint，一个对已发布模型做单次剪枝），但约束并不禁止 agent 加训练——实际有提交在剪枝流程里加了蒸馏微调、在融合里加了梯度搜索。

- **"只能改算法"具体是什么意思：base 模型和最终评测数据锁死，但训练数据的用法可以算作算法设计的一部分**：容易被误解成"数据、模型都完全一样、agent 只能碰算法代码"，实际按任务有细微差别，真正一以贯之的只有两条红线：
  1. **起始 base 模型身份不能变**——不能偷偷替换成更强的模型，这是全部 10 个任务共同的硬约束，和 PostTrainBench 的反作弊逻辑一致。
  2. **最终评分用的 held-out 数据/资产全程锁死、只在打分阶段才挂载**（proxy metric 之外的那部分），agent 在 4 小时探索阶段永远看不到，防止过拟合。

  但"训练阶段怎么用数据"这件事本身，在不少任务里被明确划进"算法设计"的范畴，允许 agent 动：
  - **OpenR1**：论文原文明确写 agent 可以对给定的 8005 行 CodeForces 数据做**筛选/重加权/打包/变换/甚至合成新训练信号**，也可以改 masking 方式和目标函数。
  - **RAGEN**：因为是在线 RL，训练数据本身就是 agent 自己跑出来的——棋盘构造、课程设计、rollout 收集方式（决定训练数据长什么样）都是**开放**的，只有打分引擎、动作解码、最终 seed 锁死。
  - **DDPO**：prompt 构造和采样方式也是开放的，只有 reward 用的资产（CLIP 打分器）和最终评测用的 prompt/latent 流固定、只在打分阶段挂载。
  - **NPO / DiGress / Model Soup / OWL**：这几个任务的训练/校准数据（TOFU 协议数据、QM9、72 个 CLIP checkpoint、C4 校准 shard）基本给定不动，agent 主要在算法/流程层面做文章（剪枝准则、融合系数、遗忘目标函数）。

  换句话说，AI4AI-Bench 测的边界不是"数据能不能碰"，而是论文反复强调的那句话：**超参是训练算法认为给定的一个数字，算法性改动是重写算法本身**——如果对数据的处理触及了"训练信号怎么构造"这个层面，也算在被测的能力范围内，不算作弊。

- **评测指标（精确公式）**：每个任务把原始 metric 通过单调函数 `φ` 映射到统一进度坐标，用三个锚点——`φ⊥`（完全没学到东西）、`φb`（仓库自带算法的水平）、`φ*`（任务理论最优）——分段线性打分：

```
σ(x) = 0.1 · (φ(x)-φ⊥)/(φb-φ⊥)         当 φ(x) ≤ φb（未达到 baseline）
     = 0.1 + 0.9 · (φ(x)-φb)/(φ*-φb)   当 φ(x) > φb（超过 baseline）
```

两段在 `σ=0.1` 处拼接，裁剪到 [0,1]，未返回可训练模型直接记 0。

- **关键结果（Table 2 完整数据）**：评测 **6 个系统**——GPT-5.6 三个变体（Sol/Terra/Luna，用 Codex，各测 6 档 reasoning effort）、Claude 5 两个变体（Opus 5/Sonnet 5，用 Claude Code，各测 5 档）、Kimi K3（用 Claude Code，仅最高档），共 29 configs × 10 任务 = 290 个格子。
  - 系统均分：**Opus 5 0.250** > GPT-5.6 Sol 0.191 > Kimi K3 0.174 > Sonnet 5 0.145 > GPT-5.6 Terra 0.135 > GPT-5.6 Luna 0.117
  - 单格最高分：**Claude Opus 5 @ medium effort = 0.288**
  - 290 格里有 **124 格（超过 40%）低于 0.1**——agent 改完比仓库原装算法还差
  - 花费和分数不挂钩：GPT-5.6 Sol 中位数花费 $434、Luna 只要 $48，但 Opus 5 中位数只花 $181 却是分数最高的系统
  - 多数提交根本没有改变模型的学习方式，改了的少数提交平均分 0.226 vs 未改的 0.126；更高 reasoning effort 主要买到的是"愿不愿意去改算法"的意愿（这一比例从 8% 提升到 64%）
- **公开 Leaderboard**：✅ 有，第三方站点 [benchlm.ai/benchmarks/ai4ai-bench](https://benchlm.ai/benchmarks/ai4ai-bench)。

### 1.6 InferenceBench —— 把"自我改进"延伸到推理服务这一环

- **论文**：*InferenceBench: A Benchmark for Open-Ended LLM Inference Optimization by AI Agents*，[arXiv:2607.20468](https://arxiv.org/abs/2607.20468)，Jehyeok Yeon、Ben Rank、Maksym Andriushchenko（与 PostTrainBench 同一批作者）。GitHub：[aisa-group/InferenceBench](https://github.com/aisa-group/InferenceBench)
- **核心思路**：训练之外，"部署并优化自己的推理服务"同样是 AI 系统能不能自我改进的一环——论文的出发点是很多"开放式"任务其实靠检索一个知名 recipe+调几个超参就能解决，不清楚强结果是真优化还是背答案，所以特意设计成**组合爆炸式的工程决策空间**而非单一已知配方。
- **输入/输出格式**：每次 run 给 agent 一个 base LLM + 一块 H100 + 一个优化场景 + 2 小时 wall-clock 预算；agent 必须交付一个**跑起来的、OpenAI 兼容的推理 server**，暴露 inference framework、attention backend、量化格式、KV-cache 布局、调度器调优等真实的组合选择（不是简单调超参——组合错了会直接在启动时崩溃，而不是优雅降级）。
- **评测指标**：4 个优化场景——3 个分别隔离 prefill 延迟、decode 延迟、并发请求吞吐这三个不同瓶颈，第 4 个要求三者兼顾；主指标是相对朴素 PyTorch baseline 的**加速比**，同时过"质量门"和"完整性门"两道 gate 才算有效提交。与 PostTrainBench（自动化后训练）、ISO-Bench（vLLM/SGLang 内的局部 patch 级优化）定位不同，InferenceBench 测的是**端到端部署+优化一整个推理服务**。
- **关键结果**：15 个前沿 agent 配置测试，相对朴素 baseline 最高加速 **8.08×**，常能匹配甚至超过默认配置的成熟 serving 引擎；但行为分析显示 **93.9% 的 run 最终都用 vLLM 收尾**（即便 prompt 里明确提到 SGLang/TGI/TensorRT-LLM），中位数一次 run 全程只尝试了一个非默认 vLLM 配置——说明 agent 的探索广度其实很有限。180 条记录的 run 里，65.0% 同时通过质量门和完整性门，18.9% 卡在质量门，6.1% 被标记诚信问题，10.0% server 起不来或运行时报错。
- **公开 Leaderboard**：✅ 有，第三方站点 [benchlm.ai/benchmarks/inferencebench](https://benchlm.ai/benchmarks/inferencebench) 镜像了官方榜单（2026-05-20 快照，18 个前沿 agent 配置 × 4 个场景 × 单卡 H100 80GB，2 小时预算），截至该快照 **Claude Opus 4.8 (xHigh)** 以 7.34× 综合加速比领先。

### 1.7 同类方法（非 benchmark，供参考）

- **Absolute Zero Reasoner (AZR)**（[arXiv:2505.03335](https://arxiv.org/abs/2505.03335)，NeurIPS 2025）：单模型身兼出题者与解题者，靠代码执行器验证自生成任务/答案正确性，纯 RL 自我改进，数学/代码推理达 SOTA。属训练算法层面自我改进的方法，不是 benchmark。
- **STaR / Self-Rewarding Language Models**（STaR: [arXiv:2203.14465](https://arxiv.org/abs/2203.14465)）：经典"自举式自训练"方法脉络，衍生 V-STaR、Quiet-STaR。
- **Agent0**（[arXiv:2511.16043](https://arxiv.org/abs/2511.16043)，ICLR 2026 Workshop RSI Oral，UNC-Chapel Hill/Salesforce Research/Stanford）：两个从**同一 base LLM** 初始化的 agent 自我博弈——curriculum agent 用 RL 不断出更难的前沿任务（reward 来自 executor 的不确定性和工具调用频率），executor agent 学着用工具解题；两者互相施压，形成自我强化的课程生成循环，**完全不需要外部人工数据**。Qwen3-8B-Base 上数学推理 +18%、通用推理 +24%。和 AZR 是同一类"自我博弈式训练"方法，区别在于 Agent0 显式拆成两个角色并引入工具调用。

### 1.8 小结

这个子方向的核心是一批"**极简闭环 speedrun**"式测试场：给 agent 一份可运行的参考代码（LLM 预训练脚本、训练算法仓库、推理服务栈）+ 一个明确、可自动执行的打分函数，让它反复编辑、跑分、保留/回退，逼近或超越人类记录。nanoGPT/nanochat 生态是这条线里发展最完整的一支，已经从 Karpathy 一次性的个人实验，长成了一整套正式化评测体系（NanoGPT-Bench、Automated LLM Speedrunning Benchmark、NanoGPT Speedrun Frontier），且已有三个持续更新的公开榜单。往上一层是更"较真"的控制变量式研究基准——PostTrainBench、Agent² RL-Bench、RSIBench-Data、AI4AI-Bench、InferenceBench，分别隔离 post-training 全流程/交互式 RL pipeline 工程/数据策略/训练算法本身/推理服务部署优化，共同结论都是前沿 agent 明显落后人类专家或探索广度有限（PostTrainBench 23.2%、AI4AI-Bench 最好系统均分 0.250·单格最高 0.288、NanoGPT-Bench 恢复不到人类同期进步的 10%、Agent² RL-Bench 里 5/6 任务 agent 最终放弃 RL 退回 SFT、InferenceBench 里 93.9% 的 run 无论 prompt 怎么提示都收敛到同一个 vLLM 方案）。**共同的关键发现**：agent 普遍擅长超参搜索/方法堆叠/收敛到已知配方，但难以提出真正新颖的算法性改动或做大胆的组合空间探索（NanoGPT-Bench 测得人类记录 77% 来自算法改动，agent 却把大部分算力花在调参上；AI4AI-Bench 测得 290 个配置里超过 40% 反而比仓库原装算法更差；Agent² RL-Bench 发现同一 scaffold 换 driver LLM，交互式任务提升幅度能从接近 0 摆动到 +78pp，说明底层模型比 scaffold 设计更决定成败）。**明显空白**：所有工作局限于小模型（≤4B）/单一 GPU/短时间窗口（数小时到十几天），尚无覆盖预训练规模或多轮递归自我改进直到收敛的长时程评测；"agent 训出的模型再去训练下一代 agent"这一真正闭环的 RSI 场景仍是空白。

---

## 二、Memory 自进化类 benchmark

### 2.1 专门评测"记忆自我进化"的核心 benchmark

#### EvoMemBench

- [arXiv:2605.18421](https://arxiv.org/abs/2605.18421) · [GitHub](https://github.com/DSAIL-Memory/EvoMemBench)
- **核心思路**：沿 memory scope（episode 内 vs 跨 episode）× memory content（知识型 vs 执行型）两个轴组织，构成 4 种设置。
- **输入/输出**：知识型——增量注入的多轮 chunk（继承自 MemoryAgentBench 子集）+ 问题，输出答案字符串；执行型（工具使用/网页搜索/具身交互）——任务目标+环境/工具 API，输出动作轨迹，由任务专属 evaluator 判定终态。
- **评测指标**：`Accuracy = (1/N)ΣCorrect(i)`（0/1，知识型）；`Success Rate = (1/N)ΣSuccess(i)`（0/1，执行型）；`Progress Score`：单任务 `PS_i = (1/T_i)ΣPass(i,t)`（多轮 checkpoint），总体取平均；另报告 Average Steps 与 token 用量。
- **关键发现**：长上下文 baseline 依然极具竞争力，没有一种记忆形式能在所有设置下一致占优。
- **资源/时间限制**：记忆无关基线模型为 Gemini-3-Flash、GPT-5-mini、DeepSeek-V3.2，记忆增强方法统一用 **DeepSeek-V3.2** 作为骨干模型；四种设置合计约 **5,554 条**评测样本（InEp-Know 2800、InEp-Exec 800、CrossEp-Know 884、CrossEp-Tool 800、CrossEp-Web 270、CrossEp-Emb 200）；未报告总 API 调用次数/美元成本上限，仅以 token 用量作为效率指标逐方法逐数据集列出。
- **公开 Leaderboard**：无独立网页榜单，仅论文内 15 种方法对比表。

#### Evo-Memory（Google DeepMind）

- [arXiv:2511.20857](https://arxiv.org/abs/2511.20857) · [GitHub](https://github.com/zhaosnw/evo_mem)
- **核心思路**：把独立任务改造成"任务流（streaming）"，要求 agent 在流式任务序列中持续检索、整合、更新记忆；明确区分"对话记忆回忆"与"经验复用"。
- **输入/输出**：单轮数据集（MMLU-Pro/GPQA/AIME/ToolBench）——问题→答案字符串；多轮数据集（AlfWorld/BabyAI/PDDL/ScienceWorld）——观测/目标→动作序列。方法基线 ReMem 建模为 (F 基座, U 记忆更新, R 检索, C 上下文构造) 四元组，循环 Think→Act→Memory Refine。
- **评测指标**：单轮用 Answer Accuracy（Exact Match，示例 REMEM 平均 EM=0.65）；多轮用 Success（0/1）与 Progress（部分完成度）+ Step 数；报告提升与"任务相似度"的 Pearson 相关系数（Gemini-2.5-Flash 上 r≈0.72）。
- **资源/时间限制**：评测骨干覆盖 **Gemini-2.5 Flash/Flash-Lite/Pro** 及 **Claude 3.5 Haiku、Claude 3.7 Sonnet**，未使用 GPT 系列模型；各数据集规模——MMLU-Pro（Engineering 969/Economics 844/Philosophy 499 题）、GPQA-Diamond 198 题、AIME-24/25 各 30 题、ToolBench 750 任务、AlfWorld 134 任务、BabyAI 112 任务、ScienceWorld 90 任务、PDDL 60 任务；论文采用"固定 API 预算"保证方法间公平比较，总算力成本量级为"数万美元"（未给更精确数字）。
- **公开 Leaderboard**：无独立网页排行榜，GitHub 提供评测 pipeline + 论文内对比表。

#### EvoAgentBench

- [arXiv:2607.05202](https://arxiv.org/abs/2607.05202) · [GitHub](https://github.com/EverMind-AI/EvoAgentBench) · **Leaderboard**：[evermind-ai.github.io/EvoAgentBench](https://evermind-ai.github.io/EvoAgentBench/)
- **核心思路**：核心构造是 "Ability Graph"——节点为任务，边为从执行轨迹抽取的可复用 Ability（搜索策略/debug 流程/验证 workflow），专门隔离"经验能否变成可迁移操作程序"这一能力（区别于纯信息保留型记忆 benchmark）。
- **输入/输出**：训练阶段方法只能拿训练侧证据（任务/verifier 结果/元数据/轨迹），输出一个 "evolution state"（技能库/案例库/workflow 集合/演化后的 prompt）；测试阶段在固定任务/工具/评分契约下接入 agent，输出动作轨迹由固定 verifier 打分。
- **评测指标**：`Average Transfer Gain = mean_x[r_m(x) − r₀(x)]`（有/无演化状态的分数**差值**），覆盖 528/267 train/test 任务、2 个 scaffold、3 个 backbone。
- **资源/时间限制**：构建（trace collection）阶段用三个 backbone——Kimi-K2.5、GLM-5.1、DeepSeek-V3.2（同时用于 ability 卡片的 LLM 评判）；评测阶段的 3 个 backbone 具体为 **Qwen3.5-27B、Qwen3.5-397B、Gemma-4-31B**；2 个 scaffold 为 **OpenClaw** 与 **Nanobot**；每个任务执行统一 `timeout = 1800 秒`，未见总 API 调用次数/美元成本上限。
- **公开 Leaderboard**：✅ 有，可交互榜单支持按 agent（OpenClaw、Nanabot）、领域（信息检索/推理/软件工程/代码实现/知识工作）筛选，覆盖 5 种方法（EverOS、EvoSkill、Memento、OpenSpace、ReasoningBank）。

#### MemoryArena（ICML 2026）

- [arXiv:2602.16313](https://arxiv.org/abs/2602.16313) · [项目页](https://memoryarena.github.io/)
- **核心思路**：把"记忆"和"行动"耦合在 Memory-Agent-Environment 循环里评测，任务显式跨 session 相互依赖（如团体旅行规划中后来者约束依赖前面旅行者的行程）。
- **输入/输出**：四环境——Bundled Web Shopping、Group Travel Planner（270条目/1869问题）、Progressive Search（221条目，最多15步递进线索）、Formal Reasoning；输入任务目标+逐步 subtask 描述，输出每个 subtask 的答案/动作。
- **评测指标**：Task Success Rate（完全解决，含全部约束满足）；Task Progress Score（依赖链中已正确完成的前缀比例）。
- **关键发现**：**在 LoCoMo 上接近满分的模型在这个"记忆驱动决策"设置下表现很差**（长上下文 agent 整体 SR 约 0.12–0.19）——暴露纯回忆型评测的盲区。
- **资源/时间限制**：Long Context Agent 基线覆盖 **GPT-5.1-mini、GPT-4.1-mini、Gemini-3-Flash、Claude-Sonnet-4.5** 四个模型；外部记忆系统（Letta/Mem0/Mirix/Mem0-g/ReasoningBank 等）统一以 **GPT-5.1-mini** 作为 task agent backbone；Group Travel Planner 环境设置工具调用步数上限 `max_steps=30`；未找到总体 token/美元成本预算数字。
- **公开 Leaderboard**：项目页展示论文内结果表，更接近"配套结果展示页"而非开放提交榜单。

#### MemoryAgentBench（ICLR 2026）

- [arXiv:2507.05257](https://arxiv.org/abs/2507.05257) · [GitHub](https://github.com/HUST-AI-HYZ/MemoryAgentBench)（README 内持续更新结果表）
- **核心思路**：四大能力维度——Accurate Retrieval、**Test-Time Learning**（部署中学新规则）、Long-Range Understanding、**Conflict Resolution/Selective Forgetting**（矛盾证据下改写/覆盖/删除旧记忆）。
- **输入/输出**："Memory Construction → Query Execution" 两阶段：长文本切 chunk 逐个喂入做增量记忆构建/更新（"inject once, query multiple times"），随后提问，agent 基于已建记忆输出答案，按 accuracy/Recall@5/F1 打分。上下文深度 103K–1.44M token，2071 个问题。
- **评测指标**：四维度独立打分；FactConsolidation 任务要求答案反映"最新"事实而非旧事实。示例：GPT-4.1-mini 长上下文 agent 总分 71.8 vs GPT-4o-mini 49.2；FactConsolidation 单跳上 HippoRAG-v2 54.0%、Mem0/Contriever 仅 18.0%。
- **公开 Leaderboard**：GitHub README 表格式榜单，22 个系统 × 5 backbone，持续更新（如 2026-05 新增 GPT-5-Mini）。

#### LifelongAgentBench

- [arXiv:2505.11942](https://arxiv.org/abs/2505.11942)
- **核心思路**：Docker 化交互环境（DB/OS/KG 三域），构建有依赖关系的技能相关任务序列，量化 catastrophic forgetting、transfer efficiency、retention。
- **输入/输出**：形式化为 goal-conditioned POMDP，agent 策略把观测映射为动作（自然语言/Bash/SQL/SPARQL 风格），任务结束一次性 0/1 奖励。
- **评测指标**（设 J_{t,i} 为完成 t 个任务后在任务 i 上的表现）：
  - `AP_t = (1/t)Σ J_{t,i}`；`AIP = (1/T)Σ AP_t`
  - **FGT（遗忘率）** `= (1/(t-1))Σ[max_{j≥i}(J_{j,i}) − J_{t,i}]`（越低越好）
  - **BWT** `= (1/(t-1))Σ(J_{t,i} − J_{i,i})`（正=促进/负=遗忘）
  - **FWT** `= (1/(t-1))Σ(J_{i,i} − J̃_i)`（J̃_i 为无先验经验 baseline 表现）
- **资源/时间限制**：评测骨干均为开源模型——**Qwen2.5-7B/32B-Instruct、Llama-3.1-8B/70B-Instruct、DeepSeek-R1-Distill-Llama-8B、DeepSeek-R1-Distill-Qwen-7B、QwQ-32B**；三域任务总量 **1,396 个**（DB 500/OS 500/KG 396）；未给出统一的上下文窗口 token 上限数值，仅定性讨论 context overflow 导致的失败案例（如单条 `find` 命令输出超 400 万字符引发溢出）。
- **公开 Leaderboard**：未找到独立网页榜单，为论文内实验结果；后续 [PATH-Bench](https://arxiv.org/html/2608.01149v1) 复用其环境研究"路径依赖性"（同一 benchmark 换任务顺序排名会变）。

#### Continual Learning Bench (CL-Bench)

- [arXiv:2606.05661](https://arxiv.org/abs/2606.05661)（UC Berkeley Sky Computing Lab）· [GitHub](https://github.com/pgasawa/continual-learning-bench)
- **核心思路**：6 个真实有状态领域（软件工程/信号处理/疫情预测/数据库查询/策略博弈/需求预测），每个任务设计"可被有状态系统在线发现、无状态系统发现不了"的潜在结构。
- **输入/输出**：CLI 用法 `clbench run exploitable_poker --schedule quick_test --system icl`；输入领域观测/反馈，输出领域动作（代码 patch/预测数值/博弈动作），由领域专属 reward 打分。
- **评测指标**：**Normalized Reward**（减去无状态 GPT-5.4 baseline 后按任务专属最大奖励缩放再平均）；**Normalized Gain**（每个系统减去"自己的"无状态 baseline 后归一化平均）；同时报告 Cost 及 95% 置信区间。
- **关键发现（负面结果）**：naive in-context learning（历史直接塞进上下文）在多数任务上跑赢专门设计的记忆架构（如 ACE），最好系统 gain 也只 25.4%；专用记忆系统成本高（ACE 每次运行 $62.8）与收益不成正比。
- **资源/时间限制**：评测模型明确为 **Claude Opus 4.7、Claude Sonnet 4.6、Gemini 3.1 Pro、Gemini 3 Flash、GPT-5.4**，评测系统含 ICL、ICL Notepad、Mem0(k=10)、ACE、Claude Code、Codex（遵循 TerminalBench 的 headless harness）；六域合计约 **206 个任务实例**（Blind Spectrum Monitoring 90 次 scan/Codebase Adaptation 19 个 SWE-bench 问题/Cohort Studies 5 个研究/Database Exploration 40 问题/Exploitable Poker 40 手每手最多 15 次查询/Sales Prediction 12 实例）；所有系统按每任务 **5 次 rollout** 取平均（Codex 因资源限制仅 1 次，故其置信区间缺失）。
- **公开 Leaderboard**：✅ 有，[snorkel.ai/leaderboard/continual-learning-bench](https://snorkel.ai/leaderboard/continual-learning-bench/)，展示 normalized reward/gain、成本，是该方向少数持续更新的第三方托管榜单之一。

#### SkillLearnBench（COLM 2026）

- [arXiv:2604.20087](https://arxiv.org/abs/2604.20087) · [项目页](https://cxcscmu.github.io/SkillLearnBench/)
- **核心思路**：20 个来自真实技能分类体系、有依赖关系的任务（15 子域），评测 "generate-store-reuse" 式技能库持续学习（One-Shot/Self-Feedback/Teacher-Feedback/Skill Creator 四种方法）。
- **输入/输出**：方法输入任务规范+（可选）oracle 轨迹/教师反馈，输出技能文档（文本），agent 执行时调用该文档解题，由确定性 verifier 判定结果。
- **评测指标**（三层）：Level 1 技能文档质量——Coverage（覆盖人工关键点比例）/Executability/Safety；Level 2 执行行为——Trajectory Alignment/Skill Usage Rate；Level 3 任务结果——Accuracy + #Tokens。LLM-judge 统一用 GPT-5-mini。
- **关键发现（负面结果）**：所有方法都优于无技能 baseline，但**纯自我反馈（无外部信息）会导致 "recursive drift"**——覆盖率不变、对齐度持续下降、准确率先升后跌；只有引入外部（teacher）反馈才能持续稳定提升。
- **资源/时间限制**：核心受控实验固定用 **Claude Sonnet 4.6**（temperature=0）作为统一 solver 以隔离"技能生成 LLM"与"解题 LLM"的影响；另测试 Claude Opus 4.6、Claude Haiku 4.5、Gemini 3.1 Pro、Gemini 3 Flash、Gemini 3.1 Flash Lite 作为技能生成 LLM 做敏感性分析；每次 trial 限制**最多 100 turn + 1800 秒 wall-clock timeout**，超时即终止并按已完成状态由 verifier 打分；未见总 API 调用次数/美元成本上限。
- **公开 Leaderboard**：未见独立托管榜单，项目页展示论文内对比表。

#### SkillEvolBench —— 专门诊断"经验能不能真的变成可复用技能"

- [arXiv:2605.24117](https://arxiv.org/abs/2605.24117)，Yingtie Lei 等（Ohio State University / Michigan State University 等）
- **核心思路**：LLM agent 在解任务过程中会积累大量 episodic 轨迹，但这些"一次性经验"到底能不能被蒸馏成可复用的**程序性技能（procedural skill）**，此前没有专门的诊断性 benchmark 去分离这个问题。SkillEvolBench 就是干这个的。
- **输入/输出**：180 个任务，覆盖 6 个真实 agent 环境，按"角色条件化任务族"组织（同一族内任务共享潜在的执行程序）。Agent 先在 acquisition 任务上学习，用压缩后的轨迹+verifier 反馈更新一个外部技能库；随后在**冻结**的 deployment 任务上测试，这些任务专门考验 context shift（情境变化）、adversarial shortcuts（对抗性捷径）、composition（技能组合）。
- **评测设计**：把"自生成技能演化"和"人工预置起点技能演化"分别与"无技能 baseline"、"直接复用原始轨迹（不做技能抽象）"两个对照组比较，从而把**程序性抽象能力**和**基座能力/人工先验知识/直接经验复用**这三者分离开——这是它区别于其他技能类 benchmark 的关键设计。
- **关键结果（重要负面发现）**：10 个模型配置 × 3 个 agent harness 测试，发现当前 agent **经常只能做局部适应，很少能形成真正稳健、可复用的技能**；技能相关的设置有时能提升 acquisition 或 replay 阶段的表现，个别模型在特定 deployment 维度上有提升，但这些提升在冻结部署下**不稳定**；更关键的是——**直接复用原始轨迹（不做任何技能抽象）经常比蒸馏出的技能表现更好**，说明现在的抽象方法会丢掉一些对未来任务仍然有用的上下文/程序性线索。容量分析还发现"多写技能"或"扩大技能库"本身不够——额外的更新会在提升覆盖率的同时引入 episode 特定的漂移和冗余。
- **资源/时间限制**：3 个 agent harness 为 **Claude Code、Codex CLI、Gemini CLI**；10 个模型配置具体为 Claude Code 下的 Opus 4.6/Opus 4.5/Sonnet 4.6/Sonnet 4.5、Codex CLI 下的 GPT-5.4/GPT-5.3-Codex/GPT-5.2-Codex、Gemini CLI 下的 Gemini 3.1 Pro/Gemini 3 Flash/Gemini 2.5 Pro；论文含"Cost–Success Analysis"（按任务尝试的美元成本做 log-scale 分析），但未给出总预算上限数字。
- **公开 Leaderboard**：无公开 leaderboard，仅论文内实验结果。
- **相关工作**：**SkillFlow**（[arXiv:2604.17308](https://arxiv.org/abs/2604.17308)）是同期另一个"技能演化"benchmark，166 个任务、20 个任务族，用统一的 Domain-Agnostic Execution Flow 组织任务，测试"Agentic Lifelong Learning"协议（agent 从无技能开始，在同一族内顺序解任务，通过轨迹/rubric 驱动的技能补丁externalize经验，并把更新后的技能库带到下一个任务）。结果比 SkillEvolBench 更乐观但也分化明显：Claude Opus 4.6 上终身技能演化把任务成功率从 62.65% 提到 71.08%（+8.43），但 Kimi K2.5 尽管技能使用率高达 66.87% 却只提升 0.60 分，Qwen-Coder-Next 更是在启用技能后**反而比不用技能时更差**——"高技能使用率"和"高技能收益"之间没有必然联系。这两个 benchmark 合起来看，说明"技能自进化"这条路径目前证据是**混合偏负面**的，和本节其他负面结果（CL-Bench、SkillLearnBench 的 recursive drift、"Useful Memories Become Faulty"）指向同一个结论。

#### 其他诊断性研究（重要负面证据）

- **"Useful Memories Become Faulty When Continuously Updated by LLMs"**（[arXiv:2605.12978](https://arxiv.org/abs/2605.12978)）：在 ARC-AGI Stream（提供 Retain/Delete/Consolidate 三种记忆操作）及 ALFWorld/ScienceWorld/WebShop/AppWorld/Mind2Web 上评测 CLIN、AWM、Dynamic Cheatsheet、ACE。**记忆效用随"整合"次数先升后降，最终跌破无记忆 baseline**——GPT-5.4 从 ground-truth 整理记忆后，对此前无记忆就能解出的 54% ARC-AGI 题目反而失败；问题出在"整合"步骤本身。
- **AgentMemoryBench**（ICLR 2026 Workshop，[链接](https://iclr.cc/virtual/2026/10012519)）：完整标题为《Benchmarking Continual Agent Memory for Online Learning, Transfer, and Forgetting》（"Lifelong Agents" workshop poster，2026-04）。统一评测 system memory 与 personal memory，五种模式量化 improvement/retention/forgetting/generalization/knowledge conflict resolution，覆盖 code-centric tool use、embodied tasks、web interaction、long-horizon dialogue 四类场景。**资源/时间限制：未找到**——目前无独立 arXiv 号、无代码仓库，评测模型型号/任务规模/预算数字均未公开，标记"细节待补充"。
- **MemEvolve**（ICML 2026 poster，[arXiv:2512.18746](https://arxiv.org/abs/2512.18746)）：不是纯 benchmark，而是方法+评测框架——"元演化"记忆系统本身（Encode/Store/Retrieve/Manage 四模块的"架构基因型"也被演化），4 个 agentic benchmark 上相对 SmolAgent/Flash-Searcher 提升最高 17.06%。**资源/时间限制**：主实验 backbone 为 **GPT-5-mini**，另用 DeepSeek V3.2、Kimi K2 测试跨模型泛化；4 个 benchmark 为 GAIA（165 任务）、WebWalkerQA（采样 170 题子集）、xBench-DeepSearch（100 任务）、TaskCraft（120 条查询用于 3 轮 meta-evolution，每轮 40 条）；单任务级 API 成本约 GAIA $0.085 vs 无记忆基线 $0.086，单任务耗时约 693-773 秒；未找到总体成本汇总数字。

### 2.2 静态检索型 benchmark（对照组，用于对比"回忆"与"自进化"的区别）

#### LoCoMo

- Maharana et al., ACL 2024，[arXiv:2402.17753](https://arxiv.org/abs/2402.17753) · [GitHub](https://github.com/snap-research/locomo)
- **输入/输出**：数据存于 `data/locomo10.json`，10 段多 session 对话（19-32 session/~9K token），附五类 QA（单跳/多跳/时序/开放式/对抗式）；输入完整/部分对话历史+问题，输出答案字符串/摘要文本。
- **资源/时间限制**：原论文基线模型覆盖 **Mistral-7B-Instruct、Llama-2-70B-Chat、gpt-3.5-turbo、gpt-4-turbo**（长上下文/RAG 设置支持 4K/8K/12K/16K 不同上下文窗口）；RAG 检索模型为 **DRAGON**；多模态对话生成任务用 **MiniGPT-5**；QA 问题总数 **7,512 个**（单跳 2,705/多跳 1,104/时序 1,547/开放域 285/对抗性 1,871）；未找到 API 调用次数/成本/wall-clock 限制信息。
- **评测指标**：官方主指标 **token-level F1（Porter 词干化）**——预测答案与参考答案先做 Porter 词干化再算 token 重叠 F1；不少第三方复现改用 LLM-as-Judge（如 GPT-4.1）；adversarial 类问题常被过滤不参与主指标。
- **公开 Leaderboard**：无官方独立排行榜；多家厂商（Mem0、MemoryLake、Backboard）自评自报"排名第一"，**均为厂商自报数据，非独立第三方维护统一榜单**，需谨慎对待。
- **和自进化的关系**：记忆内容由人工/规则流水线预先生成，agent 本身不在部署过程中自主编辑/压缩记忆，是纯回忆型评测；MemoryArena 已证明在此接近满分的模型在真正需要"记忆驱动决策"时会失败。

#### LongMemEval / LongMemEval-V2

- [GitHub](https://github.com/xiaowu0162/longmemeval)（ICLR 2025）· [LongMemEval-V2 GitHub](https://github.com/xiaowu0162/LongMemEval-V2)
- **输入/输出**：原版——长对话历史（`longmemeval_oracle.json`，s 版本限制 115000 token）+ 问题，输出存为 jsonl（`question_id`+`hypothesis`），用 `evaluate_qa.py` 以 GPT-4o 作 judge；V2——完整 agent 交互轨迹历史，记忆系统需输出"压缩后的证据"供下游 QA，同时评测准确率和查询延迟。
- **评测指标**：原版 **QA Accuracy**（GPT-4o judge 二元判定后取比例），参考分数 Oracle GPT-4o≈82.4%，多数系统 60-95%；V2 用 **LAFS Gain**（对数尺度延迟预算上平均可达准确率，相对已发布 baseline+AgentRunbook 构成的"准确率-延迟前沿"的提升幅度）。
- **公开 Leaderboard**：✅ LongMemEval-V2 有正式提交流程和公开榜单（[leaderboard README](https://github.com/xiaowu0162/LongMemEval-V2/blob/main/leaderboard/README.md)），分 Web/Enterprise 两领域、Small/Medium 两档；原版无正式在线榜单。

#### GoodAI LTM Benchmark

- [GitHub](https://github.com/GoodAI/goodai-ltm-benchmark) · [官方介绍](https://www.goodai.com/introducing-goodai-ltm-benchmark/)
- **输入/输出**：多个子测试（Instruction Recall/Trigger Response/SallyAnne/Prospective Memory 等），`run_benchmark.py` 传入配置跑对话交互流程，产出 HTML 报告逐条展示测试/回复/判定结果。
- **评测指标**：按测试类型各自定义通过/失败判据，汇总通过率；已知结果"8K 上下文 LTM agent 在放大 10 倍记忆需求的 benchmark 上比 128K 上下文 GPT-4-turbo 高 13%，成本不到 16%"，具体打分函数未找到完整公式（标记未找到）。
- **公开 Leaderboard**：无独立托管榜单，结果通过官方博客+GitHub HTML 报告分享，可 PR 提交但非维护中的排行榜网站。

### 2.3 经验/技能库自我积累框架（案例研究类）

| 名称 | 链接 | 输入/输出 | 核心指标 | Leaderboard |
|---|---|---|---|---|
| **Voyager** | [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) [GitHub](https://github.com/minedojo/voyager) | Minecraft 环境状态(Mineflayer API)+课程目标 → 可执行 JS 代码，验证后存入技能库 | 独特物品数/探索距离/科技树里程碑解锁速度，相对 SOTA 分别 3.3×/2.3×/15.3× 更快 | 无，论文内一次性对比；具体公式未找到，部分未确认 |
| **Agent Workflow Memory (AWM)** | [arXiv:2409.07429](https://arxiv.org/abs/2409.07429) [GitHub](https://github.com/zorazrw/agent-workflow-memory) | 网页指令+DOM状态 → 逐步操作动作序列；offline批量归纳/online在线归纳workflow | Mind2Web用Step Success Rate；WebArena用Task Success Rate（程序化checker判定），WebArena上 23.6%→35.6%（+12.0分/相对+51.1%） | 无，论文内结果 |
| **ExpeL** | [arXiv:2308.10144](https://arxiv.org/abs/2308.10144) [GitHub](https://github.com/LeapLabTHU/ExpeL) | 训练阶段试错积累经验池→蒸馏insight；评测阶段单次尝试注入insight+相似成功轨迹 | HotpotQA用EM；ALFWorld用6类任务Success Rate；WebShop用Success Rate(reward=1.0比例)+Task Score(平均reward) | 无，论文内结果 |
| **Reflexion** | [arXiv:2303.11366](https://arxiv.org/abs/2303.11366) [GitHub](https://github.com/noahshinn/reflexion) | 失败轨迹→言语反思生成self-hint存入episodic memory，下次尝试同任务重新输入 | AlfWorld Success Rate(+22%绝对提升)；HotpotQA EM(+20%)；HumanEval pass@1(91.0%) | 无，GitHub提供完整实验日志复现 |
| **ReasoningBank** | [arXiv:2509.25140](https://arxiv.org/abs/2509.25140) [GitHub](https://github.com/google-research/reasoning-bank) | 轨迹→压缩为(title,description,content)三段式记忆项，embedding检索注入system prompt；MaTTS并行/序列自我对比精炼 | Success Rate：WebArena 40.5%→48.8%(+8.3)；SWE-Bench-Verified +4.6；k=1检索优于k=4 | 无，论文内结果 |
| **StreamBench** | [arXiv:2406.08747](https://arxiv.org/abs/2406.08747)（NeurIPS 2024） [GitHub](https://github.com/stream-bench/stream-bench) | agent建模为(θ,p,r,M)，接收x_t输出ŷ_t再收反馈更新组件；5个任务(Text-to-SQL/Python/医疗诊断/QA/工具使用) | 累积准确率；Text-to-SQL用执行结果匹配率；工具使用需API名精确匹配+参数正确 | 数据在[HuggingFace](https://huggingface.co/datasets/appier-ai-research/StreamBench)，项目页提及"提交指南(建设中)"，尚未建成 |

**上表各框架的资源/时间限制补充**（原表未列出，按框架逐条说明）：

- **Voyager**：主 LLM 为 **GPT-4（gpt-4-0314）**（课程生成/代码生成/自我验证），仅在辅助性标准 NLP 任务因预算考虑改用 GPT-3.5；主探索实验预算为 **160 次 prompting iteration**（发现 63 种独特物品，3.3× 于 baseline），其他任务实验预算上限为 **50 次 prompting iteration**；未给出美元成本/wall-clock 数字，仅定性提到"GPT-4 API 成本比 GPT-3.5 贵 15×"。
- **AWM**：backbone 主要为 **GPT-4（gpt-4-0613）**（WebArena 与 Mind2Web 均使用），另有 GPT-3.5-turbo 用于消融；WebArena 规模 812 个网页导航任务（5 个站点），Mind2Web 论文正文笼统写"两个 benchmark 合计覆盖 1000+ 任务、200+ 领域"；未找到 API 成本/token 用量数字。
- **ExpeL**：policy/actor 及评测阶段统一用 **gpt-3.5-turbo-0613**，Insight Extraction 阶段专用 **gpt-4-0613**；四折交叉验证下任务规模为 HotpotQA 100 个、ALFWorld 134 个、WebShop 100 个；每任务最大环境步数 H=7(HotpotQA/FEVER)/15(WebShop)/20(ALFWorld)；论文只给出硬件规格（单台 i9-9900K+RTX 2080 Ti 桌面机），未报告 API 成本数字。
- **Reflexion**：所用 LLM 因任务而异——AlfWorld 用 **GPT-3**（沿用 ReAct 的 few-shot 设置）；代码任务（HumanEval/MBPP/Leetcode）主要用 **GPT-4**；HotpotQA 消融覆盖 text-davinci-003/gpt-3.5-turbo/gpt-4；AlfWorld 允许最多 **12 次连续 trial**（134 个环境）；HotpotQA 采样 100 题，反复重试直到"连续 3 次失败"才停止（无固定次数上限）；未找到明确的 API 成本数字。
- **ReasoningBank**：backbone 覆盖 **Gemini-2.5-Flash、Gemini-2.5-Pro、Claude-3.7-Sonnet**（经 Vertex AI API）；WebArena 规模 684 个任务，SWE-Bench-Verified 用全部 500 条测试实例；未找到明确的美元成本/token 用量数字。
- **StreamBench**：主实验评测模型为 **gpt-3.5-turbo-0125、gemini-1.0-pro-001、claude-3-haiku-20240307**（5 组随机打乱顺序跑），另有 gpt-4o-2024-08-06、gemini-1.5-flash-001 因成本原因仅在子集上测试；五任务数据规模——Text-to-SQL(Spider 2147/CoSQL 1007/BIRD 1534)、Python编程(DS-1000 1000)、工具使用(ToolBench 750)、医疗诊断(DDXPlus 1764)、QA(HotpotQA采样1500)；论文报告了明确的 **token 用量**（按方法/任务的百万级输入输出 token 数），是这批经典框架论文中唯一有量化 token 预算数据的一篇。

### 2.4 重要 Survey

- **"From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms"**（ACL 2026 Findings + ICLR 2026 Workshop），[arXiv:2605.06716](https://arxiv.org/abs/2605.06716)，持续更新 140+ 论文/40+ benchmark（[GitHub 列表](https://github.com/FeishuLuo/Evolving-LLM-Agent-Memory-Survey)）。提出存储层→反思层→经验层三层框架。
- **"A Survey of Self-Evolving Agents"**，[arXiv:2507.21046](https://arxiv.org/abs/2507.21046)
- **"Rethinking Memory Mechanisms of Foundation Agents in the Second Half"**，[arXiv:2602.06052](https://arxiv.org/pdf/2602.06052)
- **"Always-On Agents: A Survey of Persistent Memory, State, and Governance"**，[arXiv:2606.30306](https://arxiv.org/pdf/2606.30306)——汇总了 CL-Bench 等负面结果，是很好的批判性综述入口。

### 2.5 小结

2025-2026 年，"agent memory 自进化"已从早期定性案例（Voyager、Reflexion、ExpeL）发展出一批标准化 benchmark（EvoMemBench、Evo-Memory、EvoAgentBench、MemoryArena、MemoryAgentBench、LifelongAgentBench、CL-Bench、SkillLearnBench、SkillEvolBench、SkillFlow），共同范式是把任务组织成任务流/多 session/跨 episode 结构，用"有演化记忆 vs 无记忆/stateless baseline"的差值（transfer gain、normalized gain）作为核心指标，并开始把"记忆自我改写/冲突消解/遗忘"作为独立打分维度。但**2026 年出现了一批重要的负面/质疑性证据**：CL-Bench（专用记忆架构未必优于朴素历史堆叠）、"Useful Memories Become Faulty"（记忆整合反而让效用跌破无记忆 baseline）、SkillLearnBench（纯自我反馈导致 recursive drift）、SkillEvolBench（**直接复用原始轨迹经常比蒸馏出的技能表现更好**，说明当前的技能抽象方法可能在丢弃有用信息）——共同说明"记忆/技能自进化"这件事本身尚未被证明稳定有效。**明显空白**：评测环境仍偏网页/代码/具身小样本任务，缺少工业级长周期场景；各 benchmark 指标口径不统一，无类似 SWE-bench 地位的"事实标准"；同时测"检索准确率"与"记忆自我改写质量"两个维度的工作仍是少数（MemoryAgentBench、MemoryArena 做得最好）。

---

## 三、Harness 自进化类 benchmark / 方法

### 3.1 经典 / 奠基性框架（2024-2025）

#### ADAS / Meta Agent Search

- [arXiv:2408.08435](https://arxiv.org/abs/2408.08435) · [项目页](https://www.shengranhu.com/ADAS/)
- **核心思路**：meta-agent 用代码（图灵完备）迭代编程出新 agent，维护"发现存档"不断产生更强设计。
- **输入/输出**：meta-agent（GPT-4）读取不断增长的发现存档（历史 agent 代码+分数），输出**新 Python 代码**（`AgentSystem.forward(...)` 实现，完整可执行 agent 逻辑而非 diff）；种子存档初始化为 5 个人工 baseline（CoT/CoT-SC/Self-Refine/LLM-Debate/Quality-Diversity）。
- **评测指标**：候选在验证集上跑 5 次取均值降方差，用 bootstrap 置信区间形式的 fitness；搜索 25 轮，meta-agent 用 GPT-4，候选评测用 GPT-3.5（省算力）。
- **资源/时间限制**：meta-agent 具体为 gpt-4o-2024-05-13，候选评测用 gpt-3.5-turbo-0125；ARC 域搜索 **25 轮**，推理/问题求解域搜索 **30 轮**（两个域轮数不同）；据论文 Appendix "Cost of Experiments"，单次 ARC 域完整搜索+评测约耗费 **$500**，推理/问题求解域单次约耗费 **$300**。
- **Leaderboard**：无公开 leaderboard，仅论文内跨 coding/science/math 多域对比 5 个人工 baseline 的结果。

#### AgentSquare

- [arXiv:2410.06153](https://arxiv.org/abs/2410.06153)，ICLR 2025 · [GitHub](https://github.com/tsinghua-fib-lab/AgentSquare)
- **核心思路**：把 agent 设计抽象成 Planning/Reasoning/Tool Use/Memory 四模块设计空间，用模块演化+重组两种算子搜索，用 in-context 性能预测器跳过低 promise 候选。
- **输入/输出**：输入四模块统一 IO 接口下的模块库，输出一组模块组合（完整 agent 配置）。
- **评测指标**：六个 benchmark 各自用不同指标（ALFWorld/M3ToolEval 用 success rate；Webshop 用 episode 平均 reward；SciWorld/PDDL 用 progress rate；TravelPlanner 用 micro pass rate）；综合提升为相对最佳已知人工设计平均 **+17.2%**；性能预测器成本仅为 GPT-4o 完整评测成本的约 0.025%（ALFWorld 上）。
- **资源/时间限制**：搜索 driver LLM 分别用 GPT-3.5-turbo-0125 和 GPT-4o 两组独立跑；连续 5 轮无性能提升即终止搜索；六个任务上 GPT-3.5 驱动的搜索实际跑 **8-23 轮**、每任务平均成本 **$1.84-$4.25**，GPT-4o 驱动的搜索跑 **8-18 轮**、每任务平均成本 **$10.51-$42.14**（成本随驱动模型强度显著上升）；模块池具体总规模论文未给出——未找到。
- **Leaderboard**：无独立公开榜单，仅论文内 12 个人工 baseline 对比表。

#### Darwin Gödel Machine (DGM)

- [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)，Sakana AI + Jeff Clune 组，ICLR 2026 · [GitHub](https://github.com/jennyzzt/dgm)
- **核心思路**：达尔文式进化，维护不断增长的 agent 变体存档，agent 反复编辑自己代码库（patch validation、更好的文件查看/编辑工具、多解生成排序、失败历史记忆），经验证提升即保留（放弃原版 Gödel Machine 需数学证明改进的不现实假设）。
- **输入/输出**：agent 读取自身当前 Python 代码库，输出一次代码 diff（自我修改）。
- **评测指标**：主指标 **Pass@1**。三级递进评测省算力：10 个任务健全性检查（0/10 淘汰）→通过后 50 个任务评测→表现好的加入存档后在 200 个任务（140 新+60 旧）上确认真实能力。用 **SWE-bench Verified** 子集。
- **资源/时间限制**：共跑 **80 轮迭代**（每轮生成一个新 agent），SWE-bench 上 2 个迭代并行、Polyglot 上 4 个迭代并行；自我修改与 SWE-bench 评测均由 **Claude 3.5 Sonnet (New)** 驱动，Polyglot 评测改用 **o3-mini**（避免速率限制、加快迭代）；据论文 Appendix，SWE-bench 上单次完整运行**耗时约 2 周、总成本约 $22,000**（两个消融 baseline 各约 $10,000）。
- **关键结果**：SWE-bench 20.0%→50.0%，Polyglot 14.2%→30.7%；给的计算越多自我提升越多。
- **Leaderboard**：**未在 [swebench.com](https://www.swebench.com/) 官方榜单上找到 DGM 独立条目**——DGM 自行报告结果，未确认是否提交至官方榜单，标记"未找到"。

#### STOP (Self-Taught Optimizer)

- [arXiv:2310.02304](https://arxiv.org/abs/2310.02304)，Microsoft · [GitHub](https://github.com/microsoft/stop)
- **核心思路**：用"improver"脚手架程序反复调用 LM 改进任意程序，再把该 improver 应用于自身来自我改进；LM 自己提出了 beam search、遗传算法、模拟退火等自改进策略。
- **输入/输出**：输入种子 improver 程序（生成候选改进并按 utility function 选最优的脚手架代码）+ utility function，输出新版本的 improver 程序。
- **评测指标**："meta-utility"——held-out 测试实例上评估 T 轮自我改进后 improver 在下游任务 "learning parity with noise (LPN)" 上的表现。CoT baseline 无报错 57.7%±3.0%（有报错 49.6%±3.5%），贪心迭代 improver 64.2%±0.9%。
- **资源/时间限制**：主实验固定跑 **T=4 轮**自我改进（对 GPT-4/GPT-3.5/Mixtral 三种底层模型分别测试）；每轮每次 run 约消耗 **3000 次 GPT-4 API 调用**；附录另有一个 T=10 轮的定性案例展示，非主线量化结果。
- **关键发现**：GPT-4 下单调提升，但换成 GPT-3.5/Mixtral 等弱模型时**性能反而下降**——说明基座模型能力是自我改进的前提。
- **Leaderboard**：无公开 leaderboard，仅论文内小规模 LPN 任务结果。

#### SICA (Self-Improving Coding Agent)

- [arXiv:2504.15228](https://arxiv.org/abs/2504.15228)，ICLR 2025 workshop
- **核心思路**：agent 自主编辑自己的 Python 代码库以提升编码任务表现，维护改进存档（类似 DGM），区别是 meta-agent 也从存档中被选择。
- **输入/输出**：meta-agent（从存档选出当前最佳 agent）读取历史 agent 及其 benchmark 结果，输出对自身代码库的编辑。
- **评测指标**：显式效用函数选出"最佳"agent：`U = 0.5·p_score + 0.25·(1−min(1,p_cost/$10)) + 0.25·(1−min(1,p_time/300s))`，p_score 为归一化 benchmark 分数∈[0,1]，单题成本上限 $10，超时 300 秒。
- **资源/时间限制**：meta-agent 基座模型为 **Claude 3.5 Sonnet**，搭配一个用 **o3-mini** 的 reasoning 子智能体；SWE-bench Verified 主实验共跑 **15 轮自我改进迭代**，总 API 成本约 **$7,000**；存档规模无显式容量上限，随迭代逐轮累积。
- **关键结果**：SWE-bench Verified 随机子集从 17%→53%，平均耗时略降。论文明确指出"纯脚手架系统存在能力上限"。
- **Leaderboard**：无公开 leaderboard，仅论文内实验结果。

#### Gödel Agent

- [arXiv:2410.04444](https://arxiv.org/abs/2410.04444)，ACL 2025 · [GitHub](https://github.com/Arvid-pku/Godel_Agent)
- **核心思路**：受理论 Gödel Machine 启发，agent 通过 monkey patching **在运行时**读取并修改自己的代码（包括修改"修改逻辑"本身），仅用高层目标 prompt 引导。
- **输入/输出**：agent 通过 sensor 模块读取自己的运行时源码（逻辑/prompt/决策流程），输出通过 monkey patching 对自身方法的**运行时动态重写**（立即生效，非离线 diff）。
- **评测指标**：test accuracy 及 95% bootstrap 置信区间；设 "Gödel-base"（受限）和 "Gödel-free"（无约束）两变体。示例：Game of 24 任务从 4%→78%。
- **资源/时间限制**：每个任务跑 **6 次独立自我改进周期**，每个周期最多 **30 轮**递归自我修改迭代；驱动模型为 gpt-4o-2024-05-13，被优化的策略/baseline 模型用 gpt-3.5-turbo-0125；一次完整 30 轮迭代跑完 DROP/MGSM/MMLU/GPQA 四数据集总 API 成本约 **$15**（对照 ADAS 同等设置下需 $300，成本低一个数量级）。
- **Leaderboard**：无公开 leaderboard，仅论文内多领域实验（4 个推理 benchmark + Game of 24）。

#### AFlow

- [arXiv:2410.10762](https://arxiv.org/abs/2410.10762)，ICLR 2025 Oral · [GitHub](https://github.com/FoundationAgents/AFlow)
- **核心思路**：把 agentic workflow 优化建模为代码表示工作流的搜索问题，用 MCTS + 代码修改 + 树状经验 + 执行反馈演化工作流。
- **输入/输出**：输入任务描述+（可选）工作流示例，输出**代码表示的工作流图**（节点为 LLM 调用/预定义算子如 Generate/Format/Review&Revise/Ensemble/Test/Programmer，代码边连接）；自定义 benchmark 需继承 `BaseBenchmark` 实现 `evaluate_problem`/`calculate_score`/`get_result_columns`。
- **评测指标**：HumanEval/MBPP 用 **pass@1**；GSM8K/MATH(lv5) 用 Solve Rate(%)；HotpotQA/DROP 用 **F1**；同时统计 token 成本画性能-成本 Pareto 前沿。
- **资源/时间限制**：优化器固定用 **Claude-3.5-Sonnet** 生成/修改 workflow 代码，执行器分别用 DeepSeek-V2.5/GPT-4o-mini-0718/Claude-3.5-Sonnet-0620/GPT-4o-0513 四种模型测试；MCTS 搜索**固定跑 20 轮**（对照 baseline ADAS 用 Claude-3.5-Sonnet+GPT-4o-mini 跑 30 轮）；搜索阶段具体 API 调用次数/总成本未给出——未找到（论文提到的"4.55%推理成本"指最终 workflow 的推理阶段成本，不是搜索过程花费）。
- **关键结果**：6 数据集平均比人工设计高 5.7%，比同类自动化方法高 19.5%，平均分 80.3%，在 MATH lv5/MBPP 上比 ADAS 高 57%（相对提升）。
- **Leaderboard**：**无 AFlow 专属公开榜单**（HumanEval 等本身有各自公开生态，但 AFlow 未提交至统一"AFlow leaderboard"）。

#### EvoAgent

- [arXiv:2406.14228](https://arxiv.org/abs/2406.14228)，NAACL 2025 · [GitHub](https://github.com/siyuyuan/evoagent)
- **核心思路**：把已有专家 agent 视为"初代个体"，用遗传算法（变异/交叉/选择）自动把单 agent 框架（如 MetaGPT）扩展为多样化多智能体系统。
- **输入/输出**：交叉算子输入两父代 prompt 生成融合新 prompt；变异算子输入单父代 prompt 生成不同但仍有效的新配置（须含 `system_prompt` 字段）；输出扩展后的多智能体配置集合。
- **评测指标**：**未见形式化的百分比提升公式**，评测是跨知识问答/多模态推理/交互式科学解题/复杂规划等任务的定性效果展示。
- **资源/时间限制**：记为 EvoAgent(N,T)，N=每轮种群规模、T=迭代轮数；主实验对齐 Self-Refine 基线设置 **N=1、T=3**，部分任务为求效率设为 **N=1、T=1**；消融额外测试 (1,5)/(2,3)/(3,3) 等组合；变异/交叉算子直接复用被扩展的初始 agent 所用骨干模型（覆盖 GPT-4/GPT-3.5/GPT-4V/Gemini-Pro/LLama2-13B-Chat/Mistral-7B），未引入独立于骨干模型之外的专门 driver 模型。
- **Leaderboard**：无公开 leaderboard，仅论文内案例实验。

#### GEPA

- [arXiv:2507.19457](https://arxiv.org/abs/2507.19457)，ICLR 2026 Oral · [GitHub](https://github.com/gepa-ai/gepa)
- **核心思路**：反思式 prompt 演化，用自然语言反思执行轨迹（而非把结果压缩成标量 reward）来变异 prompt，维护 Pareto frontier 避免局部最优贪心选择。
- **输入/输出**：输入训练集+当前候选 prompt 的执行轨迹（含错误信息/profiler 输出/推理日志），输出新候选 prompt（也可扩展到完整 DSPy 程序）。
- **评测指标**：三段式划分（train/validation/test，test 全程隔离）；维护 Pareto frontier——prompt 只要在训练集某子集上不被严格支配即进入前沿；标量 fitness 如 HotpotQA 用 F1。
- **资源/时间限制**：不依赖外部更强的"driver"模型做反思变异——prompt 反思由被优化的目标模型自身完成（实验分别用开源 **Qwen3-8B** 和闭源 **GPT-4.1 Mini** 作为目标兼反思模型）；6 个任务总优化预算（rollout 数）分别为 HotpotQA 6871、IFBench 3593、HoVer 7051、PUPA 2426、AIME-2025 1839、LiveBench-Math 1839，对照组 GRPO 统一固定 24000 次 rollout；GPT-4.1 Mini 实验总成本低于 $500（GEPA 本身 $86，GEPA+Merge $67）。
- **关键结果**：HotpotQA 42%基线用 **6871** 次 rollout 达 62%（原文档"6438"为转述误差，已按论文原文修正；GRPO 需 24000 次才到 43%）；相比 GRPO 最多提升 20% 且 rollout 少 35 倍。**注**：此前版本文档提到的"MATH 67%→93%"、"AIME 2025 46.6%→56.6%"两个数字在当前 arXiv v2 全文中未找到对应表述（论文六个任务不含独立"MATH"任务），疑似转述自旧版本论文或与其他工作混淆，暂不确认，标记待核实。
- **Leaderboard**：**无 GEPA 专属独立榜单**——是应用于既有 benchmark（HotpotQA/AIME/AppWorld/Spider2）的优化方法，如 "RLM-GEPA on AppWorld 超过已有 leaderboard 0.804 SGC 基准" 说明 **AppWorld 本身有独立公开榜单**，GEPA 是在其上刷新而非自建榜单。

### 3.2 2026 年最新：直接针对"harness 自进化"的方法

这是 2026 年才真正独立成型的研究线，多篇论文明确使用 "harness self-improvement/self-evolution" 措辞并互相引用形成谱系。

#### Meta-Harness

- [arXiv:2603.28052](https://arxiv.org/abs/2603.28052)，Stanford/MIT/KRAFTON · [复现代码](https://github.com/stanford-iris-lab/meta-harness-tbench2-artifact)
- **核心思路**：单个 coding-agent proposer（Claude Code + Opus 4.6）读取历史全部诊断日志，对 harness 做**算法级重写**（检索/记忆/prompt 构造逻辑直至整程序重写），而非模板填空或预定义变异算子。指出"换 harness 在同一 benchmark 上可造成 6 倍性能差距"。
- **输入/输出**：outer-loop proposer 读取文件系统中此前所有候选的源码/执行轨迹/分数（每轮中位数读 82 个文件，41%为历史 harness 源码、40%为执行轨迹、6%为分数摘要、13%为其他），输出新的 harness 代码（单文件 Python，可局部改动也可整体重写）。
- **评测指标**：形式化目标 `H* = argmax_H E[r(τ,x)]`。提出 **TerminalBench-2**：89 个 Docker 化任务，二元通过/失败，每题 5 次独立试验；候选按"通过率+辅助成本（如 context token 用量）"构成 Pareto 前沿选择。
- **资源/时间限制**：一次典型搜索 run 耗时约"几小时"wall-clock；单次候选评测可产生最多约 **1000 万 token** 的诊断信息（比此前文本优化器的反馈预算高约三个数量级）；"每轮中位数读 82 个文件"这一统计来自一次 **10 轮迭代**的 TerminalBench-2 子实验（定性分析 proposer 行为），与主实验"20 轮迭代、评测约 60 个 harness"是不同规模的 run；另有数学检索领域的独立实验跑了 **40 轮迭代、产生 109 个候选 harness**；论文未报告具体美元成本——未找到。
- **关键结果**：Opus 4.6 上发现的 harness 达 76.4% pass rate（超过 Terminus-KIRA 基线 74.7%）；Haiku 4.5 上 37.6%（超过基线 35.5%）。
- **Leaderboard**：**TerminalBench-2 有独立官方 leaderboard**：[tbench.ai/leaderboard/terminal-bench/2.0](https://www.tbench.ai/leaderboard/terminal-bench/2.0)。但需注意：查证时官方榜单快照上，Meta-Harness 的 76.4% 并未反超榜首（当时榜首 ForgeCode+Opus 4.6 的 79.8%），即其"领先"声明相对论文内自建基线，而非官方榜单最高分。

#### Self-Harness

- [arXiv:2606.09498](https://arxiv.org/abs/2606.09498)，上海AI Lab
- **核心思路**：区分三种范式——人类工程/Meta-Harness（更强外部 agent 改进较弱 agent）/**Self-Harness**（agent 内化改进自己的 harness，无需人类或更强外部 agent）。
- **输入/输出**：输入模型在任务上产生的执行轨迹，聚类为"验证器基失败签名" `φ(rᵢ)=(cᵢ,qᵢ,mᵢ)`（终端验证器级原因、因果状态、暴露的 agent 机制三元组）；输出 proposer 角色生成的一组多样且最小化的候选 harness 修改（每个只针对一个具体失败机制）。
- **评测指标**（精确公式）：设 D_in（held-in，衡量是否解决观测到的失败）和 D_ho（held-out，回归测试防过拟合）；`Δ_in^(j) = P_in(h_t^(j)) − P_in(h_t)`，`Δ_ho^(j) = P_ho(h_t^(j)) − P_ho(h_t)`。**接受规则**：仅当 `Δ_in≥0` 且 `Δ_ho≥0` 且 `max(Δ_in,Δ_ho)>0` 时接受该候选合并进下一版 harness，否则记录但不改变当前活跃 harness。
- **资源/时间限制**：不依赖外部更强的 proposer 模型——**被优化的基座模型本身同时承担任务执行和 harness 修改提案两个角色**；三个测试模型为 **MiniMax M2.5、Qwen3.5-35B-A3B、GLM-5**，其中 Qwen3.5-35B-A3B 本地部署在 **4×H200**（SGLang 推理），MiniMax/GLM-5 通过官方 API 调用；论文未报告固定迭代轮数 T 或 API 成本——未找到，仅展示收敛轨迹图。评测规模：Terminal-Bench-2.0 固定 64 任务子集、SWE-bench Verified 固定 100 任务子集（67 held-in/33 held-out）、AppWorld 180 任务（90/90 对半分）。
- **关键结果**：SWE-bench 等三个核心 benchmark 上验证。
- **Leaderboard**：无公开 leaderboard，仅论文内实验结果。

#### Recursive Harness Self-Improvement (RHI)

- [arXiv:2607.15524](https://arxiv.org/abs/2607.15524)，Sakana AI + UC Berkeley。**本仓库已有详细解读文档**：[rhi_recursive_harness_optimization_self_improving_llm_zh.md](./rhi_recursive_harness_optimization_self_improving_llm_zh.md)
- **核心思路**：把 harness 表示为 **prompt 级别**（非代码级）的 agent-loop 规范——角色分配/指令/agent 间通信契约(contracts)/控制流跳转(hops)；每轮把当前 harness 与上一轮做**成对比较**（trajectory-local self-comparison），不搜索整个种群，只做局部轨迹对比。关键论点：harness 的执行轨迹还是未来模型训练的数据来源（"model-harness co-evolution"）。
- **评测指标**：隐式 Bradley-Terry 式偏好目标 `f_x(H) = E_{H'~μ}[σ(u_x(H) − u_x(H'))]`（修订历史分布 μ 上，H 相对比较对象 H' 的效用差 sigmoid 期望）。评测 30 个合成机器学习研究任务（量化金融/机器人/制药）。
- **资源/时间限制**：三个被优化的 base coding agent 为 **Claude Sonnet 4.6、Claude Opus 4.7、Claude Opus 4.8**（默认 reasoning effort=high）；pairwise 评估用 **gpt-5.5**（max reasoning）和 **opus-4.7/4.8**（xhigh reasoning）两种 evaluator 配置交叉验证，各跑 3 个随机种子；具体迭代轮数——Sonnet-4.6 实验跑 **4 轮迭代**（i=0…4）、Opus-4.8 实验跑 **2 轮迭代**（i=0…2），论文明确"效果在 2~4 轮迭代内趋于平台期"；每次 pairwise 对比覆盖 30 个任务。
- **关键结果**：少数迭代内（2-4 轮）使低推理强度设置超过原本最高推理强度设置，同时推理成本最多降 60%（成本降低主要来自缓存读写用量下降，而非输出 token 减少）。
- **Leaderboard**：无公开 leaderboard，仅论文内合成任务实验结果。

#### MOSS

- [arXiv:2605.22794](https://arxiv.org/abs/2605.22794)，USTC/HKUST/HKBU
- **核心思路**：主张此前"self-evolving agent"只改文本可变工件（skill 文件/prompt 配置/记忆 schema/工作流图），未触及路由/hook 顺序/状态不变量等只存在于代码中的结构性故障；用外部 coding CLI 做**源码级**自重写。
- **输入/输出**：自动汇总生产失败证据（日志/错误轨迹/评分失败任务）作为输入，通过七阶段流水线（故障定位→修复规划→实现→代码审查→任务评估→裁决）委托可插拔的外部 coding-agent CLI 产出源码级修改；经"影子试验 worker"重放失败批次验证后，`moss evo apply` 触发容器热替换部署。
- **评测指标**：部署安全机制——网关写 swap-request，宿主守护进程重启新镜像，90 秒探测窗口每 5 秒心跳/状态探测，连续 3 次通过才提交 swap，否则自动回滚。效果指标是"四任务平均 grader 分数"。
- **资源/时间限制**：案例研究中"被演化的 agent"本体使用 **DeepSeek V3.2** 作为底层模型；负责代码编辑的外部 coding-agent CLI 为**可插拔组件**，内置四种可选实现（**Claude Code、OpenAI Codex CLI、OpenCode、DeepSeek-TUI**）由配置在启动时选定；论文未报告七阶段流水线单次运行的总耗时或整个案例研究消耗的 API 调用数/token 成本——未找到，仅明确"90 秒探测窗口、每 5 秒心跳"这一项部署安全时间参数。
- **关键结果**：OpenClaw 上从 0.25 提升到 0.61（单周期内无人工干预）。
- **Leaderboard**：无公开 leaderboard，仅论文内 OpenClaw 基准实验结果。

#### AutoSaddler（Microsoft）

- [arXiv:2608.23041](https://arxiv.org/abs/2608.23041) · [GitHub](https://github.com/microsoft/AutoSaddler) · [项目页](https://autosaddler-projectpage.github.io/)
- **核心思路**：把 harness 优化建模为对执行轨迹的**离线学习**问题，结构化补丁生成（prompt/工具/中间件）+ 基于 EvoDAG 的泛化感知更新选择，产出"持久性"改进而非针对单条轨迹的修复。
- **输入/输出**：输入失败的长程执行轨迹+harness 代码库，通过三个 session 处理——诊断-补丁 session（生成"Capability补丁"[代码/基础设施]和"Steering补丁"[文本行为改变]）、反思 session（对比补丁前后轨迹）、进化 session（结合 EvoDAG 综合历史谱系）；输出通过验证的结构化 harness 补丁。
- **评测指标**：development split 上做候选验证与门控选择，预算耗尽后返回该集上排名最高候选。
- **资源/时间限制**：三个 session 均基于 **Claude Agent SDK**，统一用 **Claude Opus 4.6** 作为底层驱动模型；优化预算按 epoch 设定：GAIA2/SWE-Bench Pro 各 **2 epochs**，Terminal-Bench 2.0 因训练集更小翻倍为 **4 epochs**（GAIA2 上对应约 **50 次迭代**，51 个候选中 21 个被接受进入 dev 评估）；GAIA2 上实测每生成一个候选 patch 耗时 **533 秒、成本 $14.56、76.1 次 LLM 调用、约 2.1 万输出 token**；达到最佳性能仅消耗 **147 条执行轨迹**（对比 Meta-Harness 需 1,400 条、GEPA 类方法需约 2,800 次任务执行）；单次下游 task-agent rollout 本身平均消耗 20.2 次 LLM 调用、约 55 万输入 token、204 秒。
- **关键结果**：相比对应基础 harness，GAIA2/SWE-Bench Pro/Terminal-Bench 2.0 上分别提升 **9.0/9.6/10.0** 个百分点，比最强自动化基线再高 **7.4/4.4/6.7** 个百分点；GAIA2 默认基础 harness 起点 47.7%，两轮内提升到 72.3%。
- **Leaderboard**：无 AutoSaddler 专属公开榜单，仅项目页/论文自报结果表；GAIA2/SWE-Bench Pro/Terminal-Bench 2.0 本身各有独立公开 leaderboard，但未确认 AutoSaddler 结果被提交至官方榜单。

#### Hyperagents / DGM-Hyperagents (DGM-H)

- [arXiv:2603.19461](https://arxiv.org/abs/2603.19461)，UBC/Meta FAIR，ICLR 2026
- **核心思路**：在 DGM 基础上更进一步——不仅任务 agent 可编辑，连"**元层面的自我改进机制本身**"也可编辑，实现"改进如何改进"的元认知自我修改。
- **输入/输出**：输入存档中历史成功 agent 变体（"stepping stones"），选择脚本计算所有合法父代的跨域平均分，用**随机化（而非贪心）**选择维持多样性；输出新一代任务 agent **及**新一代元改进机制（统一为单一可编辑程序）。
- **评测指标**：核心指标 **improvement@50 (imp@50)**——固定元 agent 在新领域 50 次修改步骤后取得的性能提升量。评测用 **IMO-GradingBench**（IMO 数学解题评分任务）。
- **资源/时间限制**：驱动模型按任务域分开配置——Polyglot 编程域自我修改用 **Claude 3.5 Sonnet (New)**、评测用 **o3-mini**；Paper review 域自我修改 **Claude 4.5 Sonnet**、评测 **GPT-4o**；Robotics reward design 域自我修改与评测均为 **Claude 4.5 Sonnet**；IMO 数学评分域自我修改 **Claude 4.5 Sonnet**、评测 **o4-mini**。主实验（imp@50 相关结果）默认跑 **50 轮迭代**，每个方法重复 5 次取中位数+95% bootstrap CI；"200 轮迭代后达 0.640"specifically 是**迁移实验**（把 paper-review/robotics 域训出的 meta agent 迁移到 IMO 新域后续跑 200 轮），并非主实验默认预算；论文估算跑 **100 轮迭代**仅自我修改阶段约消耗 **3300 万 token**，加任务评测后总成本约 **8860 万 token**。
- **关键结果**：迁移后 HyperAgent 元机制 imp@50 达 **0.630**，手工 DGM 基线在同一迁移任务上为 **0.0**；完整 IMO-GradingBench 上 DGM-H 把 ProofAutoGrader 准确率从 0.561 提到 0.601（MAE 0.178→0.175）；200 轮迁移迭代后达 0.640（CI:0.550-0.720），若从 ProofAutoGrader+迁移初始化则达 0.700。
- **Leaderboard**：无公开 leaderboard，仅论文内实验结果。

#### 其他相关方法（部分涉及权重更新，超出纯 harness 范围但可对照）

- **SIA**（Hexo Labs，[arXiv:2605.27276](https://arxiv.org/abs/2605.27276)）：Feedback-Agent 决定做 harness 重写还是 LoRA 权重更新，三项任务上 harness+weight 联合更新均优于 harness-only。
- **Harness-Aware Self-Evolving (HASE)**（[arXiv:2607.03935](https://arxiv.org/abs/2607.03935)）及 **Co-Harness**（[arXiv:2607.22688](https://arxiv.org/abs/2607.22688)）：单模型既生成任务解也编辑 harness 组件，权重+harness 联合演化。
- **RoboPhD**（[arXiv:2604.04347](https://arxiv.org/abs/2604.04347) [GitHub](https://github.com/andborth/RoboPhD)）：首个对"Elo 锦标赛选择 vs Pareto 选择(GEPA) vs 贪心爬山(Autoresearch)"三种进化优化范式做同预算头对头比较；ARC-AGI 上把 22 行种子 agent 演化成 1013 行多策略系统，准确率从 27.8% 提升到 65.8%（Gemini 3.1 Flash Lite 做 solver），榜单最高达 86.2%。

### 3.3 专门评测"harness/scaffold 设计能力"本身的 benchmark

这类是用户最关心的——不是"某个固定 harness 在下游任务多强"，而是"agent 会不会设计/优化 harness"这件事本身。

#### Evo-Bench

- [arXiv:2608.09096](https://arxiv.org/abs/2608.09096)，RUCAIBox/BOSS直聘 · [GitHub](https://github.com/RUCAIBox/Evo-Bench) · **Leaderboard**：[evobench.org](https://evobench.org/) · [HuggingFace数据集](https://huggingface.co/datasets/RUC-AIBOX/Evo-Bench)
- **核心思路**：第一个明确定位为评测"模型内在 harness 进化能力"的 benchmark：用 harness 引导构造框架挑出对框架改进敏感的任务（避免与基座模型强度混淆、避免任务特化过拟合），涵盖 Search/Office/General 三域共 608 个任务。
- **输入/输出**：输入**最小 CodeAct 种子 harness**（固定初始 policy model+固定 benchmark 协议+固定资源预算：20 轮迭代、1000 步、48 小时），被测模型（evolver）自主诊断/重写/改进这个可执行 agent harness，输出重写后的 harness 代码。构造方法：先用 320 个辅助任务（不与最终 5 个源 benchmark 重叠）跑 4 个独立演化（GLM-5.2/Opus 4.8/Sonnet 5/GPT-5.6 Sol）产生 73 个 harness 变体，去重筛选保留 12 个，据此给候选任务打敏感度分数。
- **评测指标**：敏感度=任务分数与每个 harness "留一法"整体质量的相关性；过滤 `Sens(x)≤0` 的任务，按难度分层/敏感度排序划入验证集/评测集。最终得分为 "Overall Score"（608 个跨域任务综合分数）。
- **关键结果**：种子 CodeAct harness 基线 29.7 分，最佳自主结果 GPT-5.6-Sol 达 46.3 分（+16.6），Claude Opus-4.8 达 45.8 分（+16.1），人工工程基线 47.5 分；Search/General 域自主进化甚至超过人工基线，Office 域（高度 specific 流程）不及人工。

#### HarnessOpt-Bench

- [arXiv:2608.06301](https://arxiv.org/abs/2608.06301)，Scale Labs
- **核心思路**：专门评测"LLM 做 harness 优化"这个能力：优化器（LLM+编码 harness）拿到种子 harness+分级反馈+固定预算，编辑并提交候选，在从未可见的 held-out 测试集上打分。
- **输入/输出**：优化器接收目标 agent 的种子 harness+分级评测反馈+固定目标评测预算，输出编辑后的 harness 及最终提交候选。
- **评测指标**：候选按"相对种子的**归一化增益**"在 **held-out 测试分区**（全程对优化过程不可见）打分。三前提：目标模型/环境/验证器全程固定；最终评测全程隔离（衡量泛化而非拟合可见分数）；"可信执行边界"强制预算、阻止访问 held-out 状态、保留每个候选供审计。
- **规模**：5 个前沿 LLM 优化器×共享编码 harness/各自原生 harness×4 个下游任务，共 111 次评分运行；结果因模型/任务剧烈波动。
- **资源/时间限制**：5 个优化器模型具体为 **claude-opus-5、claude-sonnet-5、gpt-5.6-sol、gpt-5.6-terra、kimi-k3**（各配自身原生 harness：claude-code/codex/kimi-cli，另有 opencode 作为跨模型统一 harness）；评测预算固定为**每个数据分区 100 次评估调用 + development/validation 分区各 4 次全量 case pass**，target-model token 总量有硬上限但 optimizer 自身推理开销只记录不设上限；每个候选打分取 **3 次独立 rollout 均值**（K=3）；论文未报告具体 wall-clock 时间或美元成本上限（刻意用"评估调用次数"计量以避免不同预算规模的 optimizer 天然占优）。
- **Leaderboard**：无公开 leaderboard，仅论文内 111 次运行的实验结果表。

#### EvoAgentBench（跨方向，见第二部分 2.1）

- 用"Ability Graph"衡量 agent 自演化的可复用程序性经验迁移能力，与 harness 自进化高度相关但更偏"经验/技能层面"而非"harness 代码层面"，已催生 Evo-Bench、GDPevo、PACE-Bench 等后续工作。

#### Harness-Bench

- [arXiv:2605.27922](https://arxiv.org/abs/2605.27922)，Qihoo 360 · **Leaderboard**：[harness-bench.ai](https://www.harness-bench.ai/) · [GitHub](https://github.com/Qihoo360/harness-bench)
- **核心思路**：**度量 harness 效应**的诊断性 benchmark（而非"harness 自我优化"benchmark），专门隔离测量"harness 配置层面"（工具/权限/追踪/恢复等执行层）对性能的独立影响。
- **输入/输出**：106 个沙盒化任务×8 个模型后端×6 种 harness 配置组合，每个配置在共享任务环境/预算/评测协议下运行；输出/评测对象是模型-harness 配对产生的完整执行轨迹（共 5194 条）。
- **评测指标**：多指标同时报告——Score/Completion/Security/Tool Use/Consistency/Robustness（越高越好），Tokens/Turns（越低越好）。失败模式归因：36.4% 为"契约/格式违反"，24.6% 为"工具报错无恢复"，14.6% 为"证据未关联到结论"。
- **资源/时间限制**：8 个 model backend 具体为 **claude-opus-4.6、claude-sonnet-4.6、gemini-3.1-pro-preview、qwen3.6-plus、glm-5.1、kimi-k2.5、gpt-5.4、deepseek-v4-flash**；6 种可配置 harness × 8 个 backend × 106 任务 = 5088 条轨迹，另加 Codex（用 gpt-5.4，作为不可换模型的"model-bound coding agent"单独跑一遍 106 任务）共 5194 条；每个 task-model-harness 组合**只跑一次**（非多次取平均）；每任务的 timeout 按复杂度分级固定（简单办公任务 600 秒、数据库迁移 1200 秒、多步骤客服路由 3600 秒），而非全局统一预算；未报告统一的 token 用量上限——未找到。

#### "Harness Updating Is Not Harness Benefit"

- [arXiv:2605.30621](https://arxiv.org/abs/2605.30621)，Penn State/UCSC/Amazon
- **核心思路**：方法论贡献——把"harness 自进化"效果解耦成两个独立能力：(i) harness-updating（从执行证据产出有用更新的能力）和 (ii) harness-benefit（任务 agent 能否受益于更新后 harness 的能力）。
- **输入/输出**：设任务集 𝒳；`H_T^(f,e)` 表示以 agent f、evolver e 演化 T 步后的最终 harness（初始为 H₀）。输入是"agent-evolver 配对"，输出该配对下最终演化出的 harness 及其在任务集上的表现。
- **评测指标（精确公式）**：
  - `M_base(f)` = 初始 harness H₀ 下 f 的任务表现（评分函数 J）
  - **成对演化增益** `Δ(f,e) = J(f, H_T^(f,e)) − M_base(f)`
  - **harness-updating 能力** `Δ_update(e)` = 固定一组"锚定 agent"，取 Δ(f,e) 在这些 agent 上的**均值**
  - **harness-benefit 能力** `Δ_benefit(f)` = 固定一组"锚定 evolver"，取 Δ(f,e) 在这些 evolver 上的**最大值**
- **资源/时间限制**：7 个 LLM 具体为 **Claude Opus 4.6、Claude Sonnet 4.6、Claude Haiku 4.5**（闭源）+ **Qwen3-235B-A22B、Qwen3-32B、Qwen3.5-9B、GPT-OSS-120B**（开源）；agent 侧分析用其中 6 个（不含 Qwen3.5-9B），evolver 侧分析用全部 7 个，evolver 与被评测 agent 是**同一批模型**，通过角色互换解耦两种能力；全部实验走官方 API/推理端点调用，不涉及权重更新；三个评测 benchmark 为 SWE-bench Verified（500 任务）、MCP-Atlas（500 任务，36 个 MCP server/220 个工具）、SkillsBench（86 任务，每任务 5 次 trial 取平均）；论文形式化定义了演化步数 T、演化预算 β 和 per-task turn limit 为跨实验固定/共享的量，但**未在正文或附录给出这三者的具体数字**——这是论文本身的报告缺口，标记"未找到"。
- **关键发现**：三个 agentic benchmark×7 个 LLM 受控分析发现——harness-updating 能力在不同能力层级模型间基本持平（弱模型 Qwen3.5-9B 产出的更新增益接近 Claude Opus 4.6）；harness-benefit 呈**非单调关系**（弱模型获益少，中等模型获益最多，强模型不如中等模型）——直接挑战"更强模型=更强 harness 进化者"的直觉。
- **Leaderboard**：无公开 leaderboard，仅论文内受控实验结果表。

#### AI4AI-Bench（跨方向参考，见第一部分 1.4）

侧重"训练算法本身设计"而非 harness/scaffold，与本节核心对象有交叉但不完全等同，仅作参照。

### 3.4 聚合资源 / Survey

- **Lilian Weng, "Harness Engineering for Self-Improvement"**（2026-07-04），[博客](https://lilianweng.github.io/posts/2026-07-04-harness/)。综合约 35 篇论文，核心论点：近期 RSI 的现实路径不是改模型权重，而是改 harness；系统梳理 ACE、Meta-Harness 等代表工作，列出评测、记忆、负面结果、多样性坍缩、reward hacking、长期目标等开放问题。**是本方向最好的地图式入口**。
- **Awesome-Self-Improving-Agents**（[GitHub](https://github.com/selfimproving-agent/awesome-Self-Improving-Agents)，配套 "Self-Improvements in Modern Agentic Systems: A Survey"，[网站](https://selfimproving-agent.github.io/)）：明确把 Foundation Model Improvement（权重层）与 Scaffolding Improvement（harness层）分两条主线，并单列评测与 benchmark 分支。另有 `XMUDeepLIT/Awesome-Self-Evolving-Agents` 和 `EvoAgentX/Awesome-Self-Evolving-Agents`（配 "A Comprehensive Survey of Self-Evolving AI Agents"）。
- **HarnessForge**（[arXiv:2606.01779](https://arxiv.org/abs/2606.01779)）：本仓库已有解读文档 [harnessforge_joint_harness_policy_evolution_zh.md](./harnessforge_joint_harness_policy_evolution_zh.md)——把 agent 系统拆成 harness + policy 两部分一起进化，解决"结构和执行能力不匹配"问题，可作为 harness 与模型联合演化的代表性参照。

### 3.5 小结

2026 年之前，"agent 自动设计 agent"更多是通用 AutoML 式架构搜索（ADAS、AgentSquare、AFlow、EvoAgent）或纯代码自改写的 RSI 雏形（DGM、SICA、Gödel Agent、STOP），评测方式基本是"整套系统在 SWE-bench/GAIA 等下游任务上的分数曲线"。2026 年最显著的变化是"harness"被单独确立为一等研究对象：Meta-Harness、Self-Harness、RHI、MOSS、AutoSaddler、Hyperagents 等论文明确把"改 prompt/工具/编排/元改进规则本身"和"改模型权重"分开讨论，并催生了专门度量"harness 设计/优化能力本身"的 benchmark（Evo-Bench、HarnessOpt-Bench、EvoAgentBench、Harness-Bench）。方法论收敛为三类：(1) 存档式进化/遗传搜索（DGM、EvoAgent、RoboPhD）；(2) coding-agent 直接读取失败轨迹重写 harness 源码（Meta-Harness、Self-Harness、MOSS、AutoSaddler）；(3) 元层面可编辑的递归改进规则（Gödel Agent、Hyperagents）。**评测方式的共性问题**也被这批论文自己指出：容易过拟合已知 benchmark（PAST-Bench 明确警告分数可能只反映对素材的熟悉而非可迁移能力）；harness-updating 能力和 harness-benefit 能力常被端到端分数混为一谈（"Harness Updating Is Not Harness Benefit"）；不同优化器/任务组合下结果方差极大；可复现性一般，多数系统依赖特定商业 coding agent 作 proposer，成本高、随机性大。**明显空白**：缺少长期(multi-month)、跨任务族、抗 reward-hacking 的标准化评测协议，以及对"harness 自进化是否会趋同/多样性坍缩"的系统研究。

---

## 四、跨方向共性与空白

```text
1. 三个方向都在 2026 年从"零散案例展示"转向"标准化benchmark + 带负面结果的严谨评测"。
2. 三个方向都独立得出类似结论：自我改进/自进化在受控评测下往往没有想象中稳定或有效
   （模型侧：PostTrainBench 23.2%、AI4AI-Bench 0.288 上限；
    记忆侧：CL-Bench/SkillLearnBench/"Useful Memories..." 系列负面结果；
    harness侧："Harness Updating Is Not Harness Benefit" 挑战"越强模型越会自我改进"）。
3. 评测方法论共同趋势：引入"控制变量"（隔离数据/算法/harness/记忆各自的贡献）、
   引入 held-out/隐藏评测器防止过拟合已知任务、
   用"有/无该能力"的差值(gain)而非绝对分数衡量真实提升。
4. 公开 leaderboard 的成熟度参差不齐：
   - 有正式持续榜单的：PostTrainBench、AI4AI-Bench、NanoGPT Speedrun 系列(GitHub/Prime Intellect/BenchmarkList)、
     EvoAgentBench、CL-Bench、LongMemEval-V2、Evo-Bench、Harness-Bench、TerminalBench-2
   - 仅论文内一次性结果的：占大多数（RSIBench-Data、METR Expenditure Horizon、
     绝大部分 harness 自进化方法论文、绝大部分 memory 案例研究框架）
5. 明显的共同空白：
   - 缺少"预训练规模"或长时程(multi-month)的递归自我改进评测
   - "agent训出的模型/记忆/harness再去改进下一代agent"的真正闭环 RSI 场景，三个方向都还没做到
   - 缺少统一的、跨这三个子方向的综合评测协议
```

---

## 五、链接速查表

### 模型自训练类

| Benchmark | 论文 | Leaderboard |
|---|---|---|
| PostTrainBench | [2603.08640](https://arxiv.org/abs/2603.08640) | [posttrainbench.com](https://posttrainbench.com/) |
| Agent² RL-Bench | [2604.10547](https://arxiv.org/abs/2604.10547) | 无 |
| RSIBench-Data | [2607.25886](https://arxiv.org/abs/2607.25886) | [rsibench.co](https://rsibench.co/data/)（非严格榜单） |
| nanochat/Autoresearch | [GitHub](https://github.com/karpathy/autoresearch) | 无 |
| NanoGPT Speedrun | [GitHub](https://github.com/KellerJordan/modded-nanogpt) | ✅ [GitHub README](https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file) |
| NanoGPT-Bench | [GitHub](https://github.com/IntologyAI/NanoGPT-Bench) | 无独立网页 |
| NanoGPT Speedrun Frontier | — | ✅ [benchmarklist.com](https://benchmarklist.com/benchmarks/nanogpt_speedrun_frontier/) |
| Prime Intellect Speedrun | [GitHub](https://github.com/PrimeIntellect-ai/experiments-autonomous-speedrunning) | ✅ [app.primeintellect.ai](https://app.primeintellect.ai/speedrun/nanogpt) |
| AI4AI-Bench | [2608.20318](https://arxiv.org/abs/2608.20318) | [benchlm.ai](https://benchlm.ai/benchmarks/ai4ai-bench) |
| InferenceBench | [2607.20468](https://arxiv.org/abs/2607.20468) | ✅ [benchlm.ai](https://benchlm.ai/benchmarks/inferencebench) |

### Memory 自进化类

| Benchmark | 论文 | Leaderboard |
|---|---|---|
| EvoMemBench | [2605.18421](https://arxiv.org/abs/2605.18421) | 无 |
| Evo-Memory | [2511.20857](https://arxiv.org/abs/2511.20857) | 无 |
| EvoAgentBench | [2607.05202](https://arxiv.org/abs/2607.05202) | [evermind-ai.github.io](https://evermind-ai.github.io/EvoAgentBench/) |
| MemoryArena | [2602.16313](https://arxiv.org/abs/2602.16313) | 无（配套结果页） |
| MemoryAgentBench | [2507.05257](https://arxiv.org/abs/2507.05257) | GitHub README 表 |
| LifelongAgentBench | [2505.11942](https://arxiv.org/abs/2505.11942) | 无 |
| CL-Bench | [2606.05661](https://arxiv.org/abs/2606.05661) | [snorkel.ai](https://snorkel.ai/leaderboard/continual-learning-bench/) |
| SkillLearnBench | [2604.20087](https://arxiv.org/abs/2604.20087) | 无 |
| SkillEvolBench | [2605.24117](https://arxiv.org/abs/2605.24117) | 无 |
| SkillFlow | [2604.17308](https://arxiv.org/abs/2604.17308) | 无 |
| LoCoMo | [2402.17753](https://arxiv.org/abs/2402.17753) | 无（厂商自报不可信） |
| LongMemEval-V2 | — | [GitHub leaderboard](https://github.com/xiaowu0162/LongMemEval-V2/blob/main/leaderboard/README.md) |

### Harness 自进化类

| Benchmark/方法 | 论文 | Leaderboard |
|---|---|---|
| DGM | [2505.22954](https://arxiv.org/abs/2505.22954) | 未确认收录进 swebench.com |
| Meta-Harness | [2603.28052](https://arxiv.org/abs/2603.28052) | [tbench.ai](https://www.tbench.ai/leaderboard/terminal-bench/2.0)(TerminalBench-2通用榜) |
| Evo-Bench | [2608.09096](https://arxiv.org/abs/2608.09096) | [evobench.org](https://evobench.org/) |
| Harness-Bench | [2605.27922](https://arxiv.org/abs/2605.27922) | [harness-bench.ai](https://www.harness-bench.ai/) |
| HarnessOpt-Bench | [2608.06301](https://arxiv.org/abs/2608.06301) | 无 |
| Hyperagents/DGM-H | [2603.19461](https://arxiv.org/abs/2603.19461) | 无 |
| "Harness Updating≠Benefit" | [2605.30621](https://arxiv.org/abs/2605.30621) | 无 |

---

## 免责声明

本文档由多个联网检索 agent 并行调研汇总而成，受限于当前环境网络策略（arxiv.org / GitHub 部分域名的 WebFetch 被拦截），大量信息来自搜索引擎摘要转述，未逐一核对原文全文。关键数字、公式、leaderboard 当前状态（榜单会持续更新）建议在正式引用前对照原论文/官网复核一次。
