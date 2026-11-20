# TurnOPD：面向长程 Agent 的 turn-aware on-policy distillation

本文档解读论文：

```text
TurnOPD: Making On-Policy Distillation Turn-Aware for Efficient Long-Horizon Agent Training
https://arxiv.org/abs/2607.05804
```

## 一句话总结

TurnOPD 是一个 **长程 agent 任务里的 OPD 效率优化方法**。

它不是换 teacher，也不是换 RL 算法，而是指出 vanilla OPD 在多轮 agent 场景里有两个预算错配：

```text
1. rollout 预算错配：
   固定跑完整条 trajectory，很多尾部 turn 低信号、低覆盖，浪费 wall-clock。

2. loss 预算错配：
   普通 token-level / trajectory-level KL 平均会让浅层 turn 占掉大部分 KL loss，
   深层关键决策 turn 训练不足。
```

TurnOPD 的解决方案是两个 controller：

```text
1. adaptive rollout-depth budgeting：
   动态决定每次 rollout 到第几轮就停。

2. progressive turn-normalized loss budgeting：
   训练早期按 token 权重，后期逐渐转向按 turn 均衡权重。
```

## Teacher / Student 设置

论文实验里使用任务专用 GRPO teacher：

```text
ALFWorld:
  student: Qwen3-1.7B / Qwen3-4B
  teacher: Qwen3-8B-GRPO

WebShop:
  student: Qwen3-1.7B
  teacher: Qwen3-8B-GRPO

Multi-Hop Search:
  student: Qwen3.5-2B
  teacher: Qwen3.5-9B-GRPO
```

所有方法共用 OPD reverse-KL 训练栈。

## Vanilla Agent OPD 怎么做

多轮 agent rollout 形式是：

```text
τ = (x, o1, r1, o2, r2, ..., oT, rT)
```

其中：

```text
x  : 初始任务 prompt
ot : 第 t 轮环境 observation
rt : 第 t 轮 student response，可能包含 reasoning 和 tool call
```

student 自己采样 trajectory：

```text
τ ~ πθ(· | x)
```

teacher 不生成完整轨迹，而是在 student 已访问到的 prefix 上重新打分：

```text
πT(· | si)
```

OPD loss 是 token-level reverse KL：

```text
L_OPD = E[ 1 / sum_i m_i * sum_i m_i * KL(πθ(· | s_i) || πT(· | s_i)) ]
```

这里 `s_i` 是第 `i` 个 supervised token 之前的完整上下文，包括之前的 student 输出和环境 observation。

## 发现的问题 1：尾部 turn 的 KL 信号不可靠

论文按 turn 统计 KL 后发现：

```text
浅层 turn 的 KL 大，训练早期下降快；
深层 turn 的 KL 往往更低、更噪。
```

这不一定说明深层 turn 已经学好了。作者提出一个解释：

```text
污染压缩效应（contamination-compression）
```

意思是：

```text
越到后面，上下文越多来自 student 自己；
很多后续 token 会被格式、重复、工具返回内容、已生成文本强约束；
teacher 和 student 在表面 token 上看起来 KL 小，
但真实决策分歧可能被压缩了。
```

因此，固定跑到最大 horizon 不一定划算。

## 发现的问题 2：普通 KL 平均让浅层 turn 吃掉 loss

如果一条轨迹有 `T` 个 turn，第 `t` 个 turn 有 `n_t` 个 supervised tokens，普通 trajectory-level loss 是：

```text
L_traj = 1 / N * sum_t sum_i l_{t,i}
N = sum_t n_t
```

这等价于：

```text
每个 token 权重一样。
```

但 agent 任务里，深层 turn 本来就更少被访问到，而且通常每个 turn 的 token 数也更少。结果是：

```text
浅层 turn 占了大部分 KL loss budget。
```

论文诊断里报告：

```text
ALFWorld:
  最深 1/3 turn 只拿到 3.6%–4.5% KL loss budget

Multi-Hop Search:
  最深 1/3 turn 只拿到 11%–13% KL loss budget
```

## 方法一：Adaptive Rollout-Depth Budgeting

这个 controller 动态决定 rollout 到多深：

```text
H_ctrl = max(H_eff, H_cov)
```

### H_eff：效率侧深度

先用 periodic full-depth probe rollout 统计每个 turn 的：

```text
K_t     : 第 t 个 turn 的平均 reverse-KL
n_t/n_0 : 第 t 个 turn 的存活概率（survivor probability）
```

然后定义 turn t 的有效蒸馏质量：

```text
m_t = [K_t]_+ * n_t / n_0
q_t = m_t / (sum_j m_j + eps)
```

再取这个分布的重心：

```text
H_eff = round(sum_t t * q_t)
```

直觉：

```text
KL 高、还能被很多 trajectory 到达的 turn，才值得花 rollout 预算。
```

### H_cov：覆盖侧深度

只看成功 trajectory 的完成长度 `L_succ`，取一个分位数：

```text
H_cov = min{ H : F_succ(H) >= p }
p = 0.80
```

直觉：

```text
不能为了省时间截得太短；
至少要覆盖 80% 成功轨迹真正完成任务所需的 turn。
```

### EMA 更新最终 rollout cap

每次 probe 后：

```text
H_ctrl,k = max(H_eff,k, H_cov,k)
H_bar,k = (1 - alpha_ema) * H_bar,k-1 + alpha_ema * H_ctrl,k
H_hat,k+1 = clip(round(H_bar,k) + 1, H_min, H_max)
```

论文默认：

```text
alpha_ema = 0.30
probe_interval = 8
warmup_steps = 3
H_min / H_max = 2 / 50
```

注意：周期性 probe 必须是 full-depth，因为如果只用已经截断的 rollout 估计深层统计，会产生截断偏差。

## 方法二：Progressive Turn-Normalized Loss Budgeting

TurnOPD 不直接从 token-level loss 硬切到 turn-level loss，而是线性混合。

普通 trajectory-level turn 权重：

```text
q_t^traj = n_t / sum_j n_j
```

严格 turn-level 权重：

```text
q_t^turn = 1 / T
```

混合权重：

```text
q_t^blend = (1 - alpha) * q_t^traj + alpha * q_t^turn
```

其中：

```text
alpha = clip((progress - s) / (e - s), 0, 1)
progress = k / K
```

默认 `s=0, e=1`，所以训练过程里 `alpha` 从 0 线性涨到 1。

直觉：

```text
早期先让模型学浅层基本交互，保持稳定；
后期逐渐把 loss budget 分给更深的 turn，补深层决策。
```

## 训练算法

每一步大致是：

```text
1. 根据 H_hat_k 决定当前 rollout cap
2. 如果是 warmup 或 probe step，则 rollout 到 H_max
3. 否则 rollout 到 H_hat_k
4. student 在环境里 on-policy 生成 trajectories
5. teacher 在 student 访问过的 supervised tokens 上计算 top-K reverse-KL
6. probe step 更新 H_eff / H_cov / H_hat
7. 根据训练进度构造 token-level 和 turn-level 混合权重
8. 用加权 RKL loss 更新 student
```

## 数据是如何构造的

这是这篇论文里很重要的一点：

```text
训练数据不是固定 teacher trajectory。
```

每个任务只给 OPD pipeline 一个 prompt parquet。里面的“样本”是任务实例：

```text
ALFWorld:
  一个 household goal

Multi-Hop Search:
  一个 QA question

WebShop:
  一个 shopping goal id
```

真正的 trajectory 是训练时在线产生的：

```text
student policy -> live environment -> observation -> student 下一轮 action -> ...
```

teacher 只在这些 student 生成的轨迹上做 token-level supervision。

### ALFWorld 数据构造

ALFWorld 的任务实例是程序化合成的：

```text
1. 均匀采样 6 种任务类型
2. 从固定 object / receptacle vocabulary 里采 object-receptacle pair
3. 使用 seeded RNG，保证 train / validation 可复现且不重叠
4. training route 到 train games
5. evaluation route 到 eval_out_of_distribution / valid-unseen games
```

规模：

```text
train: ≈ 5000 instances
eval : ≈ 300 instances
step cap: 80
```

六种任务类型：

```text
pick_and_place
pick_clean_then_place
pick_heat_then_place
pick_cool_then_place
look_at_obj_in_light
pick_two_obj_and_place
```

动作工具：

```text
alfworld_step(action: str)
```

动作模板包括：

```text
go to {loc}
take {obj} from {loc}
put {obj} in/on {loc}
open/close {loc}
use {loc}
clean/heat/cool {obj} with {loc}
examine {x}
inventory
look
```

环境 reward：

```text
binary reward:
  完成目标 = 1
  否则 = 0
```

### Multi-Hop Search 数据构造

训练集来自外部准备的 Search-R1 格式 QA 混合数据：

```text
train: ≈ 30000 question-answer instances
eval : ≈ 400 held-out questions
```

eval 来自四个 QA 源：

```text
PopQA
Natural Questions
2WikiMultiHopQA
HotpotQA
```

prompt 会打开 thinking：

```text
<think>...</think>
<search> query </search>
<information> retrieved docs </information>
<answer>...</answer>
```

工具：

```text
search_r1/search(query: str)
```

环境：

```text
固定 Wikipedia corpus + dense retrieval server
```

reward：

```text
最终 <answer> 和 gold answer 做 exact match
```

step cap：

```text
50
```

### WebShop 数据构造

WebShop 的实例不是把 instruction 文本提前复制进 prompt，而是：

```text
只用 goal id 启动 environment session；
instruction 作为环境初始 observation 返回给 agent。
```

split 方式：

```text
eval 使用 goal ids: [0, N_eval)
train 使用 goal ids: [N_eval, N_eval + N_train)
```

因此 train / eval goal id 严格不重叠。

规模：

```text
train: ≈ 3000 goals
eval : ≈ 200 goals
step cap: 30
```

动作工具：

```text
webshop_step(action: str)
```

动作格式：

```text
search[<keywords>]
click[<button>]
```

reward：

```text
终止时给 [0, 1] dense score，
按商品类型、属性、选项、价格等匹配程度给部分分。
只有 Buy Now 终止动作携带 reward。
```

## 需要注意的数据点

这篇里的 “OPD data” 不是：

```text
提前存好的 teacher demonstrations
```

而是：

```text
任务实例集合 + 在线 student rollout + teacher force scoring
```

所以它和普通 SFT 数据构造完全不同。

训练数据动态变化，因为：

```text
student 每次更新后，on-policy trajectories 也会变。
```

## 主实验结果

主结果 Table 2：

| Task | Student | Teacher | Method | Least-Time Avg@4 | Same-Step Avg@4 | Wall h | Speedup |
|---|---|---|---|---:|---:|---:|---:|
| ALFWorld | Qwen3-1.7B | Qwen3-8B-GRPO | Vanilla OPD | 73.52±1.84 | 83.00±1.69 | 4.42 | 1.00x |
| ALFWorld | Qwen3-1.7B | Qwen3-8B-GRPO | TCOD-F2B | 80.06±1.43 | 80.06±1.43 | 1.87 | 2.37x |
| ALFWorld | Qwen3-1.7B | Qwen3-8B-GRPO | TurnOPD | 85.60±0.95 | 86.29±0.48 | 1.93 | 2.29x |
| ALFWorld | Qwen3-4B | Qwen3-8B-GRPO | Vanilla OPD | 90.79±0.61 | 91.81±0.83 | 2.86 | 1.00x |
| ALFWorld | Qwen3-4B | Qwen3-8B-GRPO | TCOD-F2B | 86.50±1.88 | 86.50±1.88 | 1.89 | 1.51x |
| ALFWorld | Qwen3-4B | Qwen3-8B-GRPO | TurnOPD | 91.73±1.41 | 92.21±0.42 | 2.16 | 1.33x |
| Multi-Hop Search | Qwen3.5-2B | Qwen3.5-9B-GRPO | Vanilla OPD | 45.77±0.48 | 47.82±0.92 | 4.45 | 1.00x |
| Multi-Hop Search | Qwen3.5-2B | Qwen3.5-9B-GRPO | TCOD-F2B | 45.64±1.06 | 47.77±1.21 | 3.80 | 1.17x |
| Multi-Hop Search | Qwen3.5-2B | Qwen3.5-9B-GRPO | TurnOPD | 47.24±1.03 | 47.24±1.03 | 2.94 | 1.51x |
| WebShop | Qwen3-1.7B | Qwen3-8B-GRPO | Vanilla OPD | 76.98±0.79 | 81.65±1.47 | 1.57 | 1.00x |
| WebShop | Qwen3-1.7B | Qwen3-8B-GRPO | TCOD-F2B | 80.45±1.57 | 81.66±0.89 | 1.33 | 1.18x |
| WebShop | Qwen3-1.7B | Qwen3-8B-GRPO | TurnOPD | 82.80±0.75 | 82.80±0.75 | 1.24 | 1.26x |

这四列的定义是：

```text
Least-Time Avg@4
  在“最短同墙钟时间”协议下报告的 Avg@4。
  Least-Time 的意思是：先找到所有方法里达到 100 个 training steps 所需的最短 wall-clock 时间，
  然后用这个共同时间截点比较各方法在该时间前的表现。

Same-Step Avg@4
  在“同训练步数”协议下报告的 Avg@4。
  Same-Step 的意思是：所有方法都训练到恰好 100 个 optimizer steps，
  再比较它们的表现。

Avg@4
  每个 evaluation checkpoint 上的验证准确率指标。
  表中的数值再对对应 cutoff 前最后 4 个 evaluation checkpoints 取平均；
  ± 是这 4 个 checkpoints 上 accuracy 的标准差。

Wall h
  到 100 个 optimizer steps 为止的累计训练 wall-clock 时间，单位是小时。

Speedup
  相对同一 task-student 组里的 Vanilla OPD 的训练加速比，
  一般可理解为 vanilla OPD 的 Wall h 除以该方法的 Wall h。
```

## 消融实验

ALFWorld-1.7B 上的组件消融：

| Config | Adaptive Depth | Linear Blend | Same-Step Avg@4 | Wall h |
|---|---|---|---:|---:|
| Vanilla OPD | no | no | 83.0 | 4.42 |
| + Adaptive Depth | yes | no | 82.8 | 1.96 |
| + Linear blend norm | no | yes | 85.1 | 2.59 |
| TurnOPD | yes | yes | 86.3 | 1.93 |

结论：

```text
adaptive depth 主要省时间；
linear blend 主要提优化效果；
两个合起来得到最好的 accuracy-time tradeoff。
```

这四行可以这样理解：

```text
+ Adaptive Depth:
  主要作用是少跑低收益的后缀 turn。
  它直接降低 Wall h，但因为部分深层 trajectory 被提前截断，
  Same-Step Avg@4 可能略掉或持平。

+ Linear blend norm:
  主要作用是修正 trajectory-level token 平均带来的浅层 turn 偏置。
  它不直接减少 rollout 成本，所以 Wall h 仍较高；
  但它把更多 loss budget 分给中后段 turn，因此更容易涨分。

TurnOPD = Adaptive Depth + Linear blend norm:
  一个管“rollout 到多深”，一个管“loss 分给哪些 turn”。
  二者不是同一个机制，但有互补关系：
  adaptive depth 节省交互成本，linear blend 补偿截断后更稀缺的深层监督。
```

所以消融表里出现：

```text
Adaptive Depth 单独用：更快，但略掉分；
Linear blend 单独用：更准，但省时有限；
两者一起用：速度接近 Adaptive Depth，分数超过 Linear blend。
```

KL normalization 消融：

| KL normalizer | Same-Step Avg@4 | Deep budget E/M/L | alpha E/M/L |
|---|---:|---|---|
| Trajectory-level KL | 83.0 | 3.2/0.7/1.2% | 0/0/0 |
| Turn-level KL hard switch | 85.0 | 29.9/32.0/31.9% | 1/1/1 |
| Linear blend | 85.1 | 12.8/21.6/27.7% | 0.17/0.50/0.83 |

结论：

```text
纯 trajectory-level 太偏浅层；
纯 turn-level 又太激进；
linear blend 更平滑。
```

## 和 TCOD / SOD 的区别

TCOD：

```text
更像 trajectory length curriculum。
```

SOD：

```text
更像 step-wise / tool-step reweighting。
```

TurnOPD：

```text
同时控制两个预算：
1. 采样到多深
2. loss 分给哪些 turn
```

## 一句话结论

TurnOPD 的核心不是“更强 teacher”，而是：

```text
在长程 agent OPD 中，把训练单位从 flat token 序列提升到 turn-aware 预算分配：
该少 rollout 的地方少 rollout，该补深层决策的地方补 loss。
```
