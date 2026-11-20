# Agentic-DPO：从模仿到面向专家轨迹的 Agentic Policy Optimization

> 论文：**Agentic-DPO: From Imitation to Agentic Policy Optimization on Expert Trajectories**  
> arXiv：<https://arxiv.org/abs/2607.10601>  
> 代码：<https://github.com/Schuture/Agentic-DPO>

## 1. 背景

LLM agent 最常见的训练方式是直接对专家轨迹做 SFT。问题是，这种做法本质上只是“学会复现专家动作序列”，并没有训练模型在同一状态下区分：

- 哪个 action 是对的；
- 哪些看起来合理但其实是错的 action 更可能被当前 student 选中。

而在线 RL / preference learning 虽然能补这个问题，但通常要付出高昂的环境 rollout 和 reward model 成本。

Agentic-DPO 的目标就是：**把专家轨迹转换成 state-conditioned preference supervision，同时保持离线、低成本**。

## 2. 技术细节

### 2.1 问题设定

输入是专家轨迹数据集：

$$
\mathcal{D}_{\mathrm{exp}}=\{\tau_i\}_{i=1}^N,\qquad
\tau_i=\{(s_{i,t},u_{i,t}^{+})\}_{t=1}^{T_i}.
$$

其中：

- $s_{i,t}$ 是 agent state，包含 system instruction、history、tool/action 空间、观察等；
- $u_{i,t}^{+}$ 是专家在该 state 采取的 latent action。

### 2.2 schema view 与 SFT

Agentic-DPO 强调一个现实问题：同一个 latent action 可以有不同 textual schema 表达，比如 ReAct 风格、JSON function call 风格等。

因此它定义 schema view $\phi$，把 latent action 映射成 rendered action：

$$
s_t^\phi=g_\phi(s_t),\qquad a_t^\phi=r_\phi(u_t).
$$

SFT 只做正样本模仿：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=-\mathbb{E}_{(s_t,u_t^+),\phi}\left[\log \pi_\theta(r_\phi(u_t^+)\mid g_\phi(s_t))\right].
$$

### 2.3 Agentic-DPO

核心区别在于：它不拿整条 trajectory 做 preference，而是**在 expert state 上做 one-step preference**。

流程是：

1. 在 expert state $s_t$ 上，从当前 student 采样 one-step negative action $u_t^-$；
2. 把 expert action $u_t^+$ 和 negative action $u_t^-$ 组成偏好对；
3. 用 DPO-style objective 学习“在这个 state 下，expert action 比 plausible wrong action 更好”。

这一步仍然是离线的：不执行环境，不做在线 rollout，也不需要 reward model。

### 2.4 PPA

Agentic-DPO 还引入 Policy-Preserving Augmentation (PPA)。

原因是 agent action 不只是 policy 问题，也有 schema 问题：模型可能学到的是某种输出模板，而不是 latent decision 本身。

PPA 的做法是对同一个 latent trajectory 采样多个 schema view，让 policy decision 保持不变，只改变表面形式，从而抑制“把格式学成偏好”的问题。

### 2.5 SFT anchor 和 refresh rounds

训练不是纯 DPO，而是：

$$
\mathcal{L}_{\mathrm{train}}(\theta)=\mathcal{L}_{\mathrm{ADPO}}(\theta)+\lambda\mathcal{L}_{\mathrm{SFT}}(\theta).
$$

也就是保留一个 SFT anchor，防止 action schema 崩掉。

另外，论文会反复 refresh negatives：用当前 student 重新采样 one-step negatives，再继续训练。这样能把 negative 持续拉到更贴近当前 policy 的“决策边界”上。

## 3. 主实验结果

### 3.1 Bench 与指标

论文在三个 benchmark 上测评：

- StableToolBench
- tau-bench retail
- Mind2Web

其中 StableToolBench 报 Canonical / Perturbed，tau-bench retail 报 Success Rate，Mind2Web 报 Avg. Step Success。

### 3.2 主表

#### Qwen3.5-2B

| Method | StableToolBench Canon | StableToolBench Pert | tau-bench | Mind2Web | Avg |
|---|---:|---:|---:|---:|---:|
| Base | 45.2 | 42.1 | 12.5 | 25.6 | 31.4 |
| SFT | 57.1 | 52.3 | 18.0 | 35.4 | 40.7 |
| PPA+SFT | 83.4 | 81.1 | 22.1 | 42.1 | 57.2 |
| DFT | 85.2 | 79.6 | 25.4 | 49.2 | 59.9 |
| ETO | 85.3 | 81.3 | 28.5 | 48.5 | 60.9 |
| GRPO | 85.0 | 79.8 | 32.1 | - | - |
| **Agentic-DPO** | **90.9** | **85.5** | **35.2** | **52.3** | **66.0** |

#### Qwen3.5-4B

| Method | Canon | Pert | tau-bench | Mind2Web | Avg |
|---|---:|---:|---:|---:|---:|
| Base | 50.5 | 48.2 | 14.5 | 28.4 | 35.4 |
| SFT | 65.2 | 60.5 | 20.4 | 38.6 | 46.2 |
| PPA+SFT | 82.0 | 77.0 | 26.5 | 46.1 | 57.9 |
| DFT | 85.4 | 81.2 | 29.8 | 49.0 | 61.4 |
| ETO | 88.5 | 85.1 | 32.6 | 50.5 | 64.2 |
| GRPO | 86.4 | 82.1 | 35.4 | - | - |
| **Agentic-DPO** | **91.3** | **89.4** | **38.5** | **54.5** | **68.4** |

#### Qwen3.5-9B

| Method | Canon | Pert | tau-bench | Mind2Web | Avg |
|---|---:|---:|---:|---:|---:|
| Base | 65.2 | 62.1 | 16.5 | 35.2 | 44.8 |
| SFT | 78.5 | 74.2 | 21.7 | 45.6 | 55.0 |
| PPA+SFT | 92.2 | 90.1 | 23.8 | 52.4 | 64.6 |
| DFT | 93.0 | 90.7 | 33.0 | 55.6 | 67.9 |
| ETO | 92.5 | 91.1 | 20.9 | 56.8 | 65.5 |
| GRPO | 88.6 | 84.9 | 40.0 | - | - |
| **Agentic-DPO** | **94.1** | **92.0** | **41.4** | **64.4** | **73.0** |

论文的关键结论是：

- Agentic-DPO 在三个规模上都优于纯 SFT、PPA+SFT、DFT 和 ETO；
- tau-bench retail 上，9B 模型从 `21.7%`（SFT）提升到 `41.4%`；
- 在同一 backbone 下，它基本可以和 online GRPO 持平或更好，但训练时不需要在线环境交互。

### 3.3 鲁棒性与 OOD

论文还做了两组额外评测：

| Setting | SFT | PPA+SFT | Agentic-DPO |
|---|---:|---:|---:|
| StableToolBench-Perturbed In-class | 53.2 | 61.7 | 89.6 |
| StableToolBench-Perturbed Held-out | 52.3 | 61.2 | 86.9 |

| ToolACE trained, BFCL-v3 eval | Base | SFT | Agentic-DPO |
|---|---:|---:|---:|
| Non-Live | 82.66 | 81.87 | 84.17 |
| Live | 56.11 | 51.54 | 57.18 |

这说明它不只是把训练分布背下来，而是在一定程度上提升了 perturbation robustness 和 OOD tool-use generalization。

## 4. 消融实验

StableToolBench 上的消融表说明了四个稳定组件都很关键：

| Method | Qwen3.5-2B Canon | Pert | Gap | Gemma3-4B Canon | Pert | Gap |
|---|---:|---:|---:|---:|---:|---:|
| SFT | 57.1 | 52.3 | 4.8 | 65.4 | 50.1 | 15.3 |
| PPA+SFT | 83.4 | 81.1 | 2.3 | 72.4 | 70.1 | 2.3 |
| w/o SFT warm-up | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| w/o SFT anchor | 72.5 | 68.2 | 4.3 | 62.5 | 58.1 | 4.4 |
| w/o action-rendering PPA | 88.5 | 75.3 | 13.2 | 75.8 | 61.2 | 14.6 |
| w/o recovery-context PPA | 89.2 | 78.5 | 10.7 | 76.1 | 65.3 | 10.8 |
| w/o PPA | 88.0 | 76.6 | 11.4 | 74.4 | 64.0 | 10.4 |
| **Full Agentic-DPO** | **90.9** | **85.5** | **5.4** | **76.5** | **72.8** | **3.7** |

对应结论很直接：

- SFT warm-up 是地基，没它连 action schema 都容易跑飞；
- SFT anchor 负责稳定 preference training；
- 两种 PPA 都在压缩 canonical / perturbation gap；
- action-rendering PPA 对扰动鲁棒性略更关键。

## 5. 训练尺度与动态

论文还发现：

- one-step negative 的数量从 `K=1` 增到 `K=2~4` 提升明显，但再加收益趋于饱和；
- 只用 `25%` 的 expert trajectories，Agentic-DPO 就能接近甚至匹配 `100%` 数据上的 PPA+SFT；
- refresh rounds 对小模型更重要，9B 模型通常一轮 refresh 就接近最好。

这说明它不是靠暴力 rollout，而是靠把“expert state + plausible mistake”构造成更高密度的学习信号。

## 6. 结论

Agentic-DPO 的核心不是把 RL 替换成另一个 SFT，而是把专家轨迹改写成**状态级偏好学习**：

1. 在 expert state 上采样当前 student 可能犯的错；
2. 用 DPO-style 目标把 expert action 和 plausible wrong action 拉开；
3. 用 SFT anchor 和 PPA 抑制 schema 偏差；
4. 完全离线完成训练，不依赖在线环境回放。

它很适合 agentic tool use 场景，因为这类任务最贵的通常不是梯度，而是环境交互。

