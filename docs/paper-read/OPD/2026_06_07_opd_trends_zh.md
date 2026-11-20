# 2026 年 6-7 月 OPD 热点汇总

这份汇总只看我们已经整理进来的 6-7 月 OPD 主线论文，目标不是逐篇复述，而是抽出这一波研究到底在补什么洞。

## 这一波的主问题

6-7 月的 OPD 工作，核心都在问三类问题：

```text
1. teacher 给的监督到底靠不靠谱？
2. 多轮 / 长程 / 工具任务里，怎么别把 rollout 和 loss 预算浪费在低收益位置？
3. 怎么把 OPD 从“单纯模仿”推进到“迁移改进方向 / 自演化 / 跨环境复用”？
```

## 代表论文速览

| 月份 | 论文 | 主要发现的问题 | 主要解决方式 |
|---|---|---|---|
| 6 月 | TrOPD | teacher/student 分布差太大时，token-level 监督会出现离群和不稳定 | trust-region 式约束 + outlier 处理 |
| 6 月 | DOPD | privileged information 会制造“teacher 更强只是因为看见更多信息”的错觉 | dual on-policy distillation，先拆 privilege illusion |
| 7 月 | Direct-OPD | 把弱模型学到的 RL 改进转给强模型太贵 | 直接蒸馏“改进方向”，不是只学最终 policy |
| 7 月 | ReOPD | 在线 multi-turn OPD 的环境交互和 tool calls 太贵 | teacher prefix replay，把 rollout 离线化 |
| 7 月 | TurnOPD | 长程 agent 里 rollout 和 loss 都在浅层 turn 上浪费 | rollout-depth budgeting + turn-normalized loss budgeting |
| 7 月 | ShortOPD | 剪枝后模型自由生成会崩，长输出尤其差 | short-to-long curriculum 式 OPD 恢复生成能力 |
| 7 月 | SEED | 纯 outcome RL 的中间监督太稀疏 | 把自己 rollout 的轨迹再反哺给自己，形成自演化监督 |

## 这一波的热点主线

### 1. 从“teacher 更强”转向“teacher 信号是否可靠”

TrOPD、DOPD 都在处理这个问题。

```text
TrOPD:
  teacher / student 分布差太大时，
  teacher 在 student 采样 token 上的监督会很容易失真，
  所以要加 trust-region 和 outlier 过滤。

DOPD:
  teacher 看起来更强，不一定真是能力更强，
  也可能只是 privileged information 更占优，
  所以要先拆 privilege illusion。
```

这是 6 月最明显的一条线：**OPD 不再默认 teacher 信号天然正确，而是要先做可靠性诊断。**

### 2. 从“完整在线 rollout”转向“更便宜的轨迹复用”

ReOPD、TurnOPD、ShortOPD 都在减少无效交互。

```text
ReOPD:
  直接复用 teacher prefixes，
  训练阶段 0 tool calls。

TurnOPD:
  不必每次都跑满 horizon，
  该少跑的 turn 少跑。

ShortOPD:
  先短后长，
  不在一开始把预算砸给重复、低信息后缀。
```

这说明 7 月的 OPD 很像在做一件事：

```text
把“跑更多轨迹”改成“只跑更值得跑的部分”。
```

### 3. 从“按 token 平均”转向“按 turn / depth / prefix 重新分配预算”

TurnOPD 和 ReOPD 都体现了这一点。

```text
TurnOPD:
  发现浅层 turn 吃掉了大部分 KL budget，
  所以引入 turn-aware loss 和 rollout 深度控制。

ReOPD:
  发现 teacher prefix 里，越深越容易偏离 student occupancy，
  所以用 step-decay 重新分配 prefix 采样概率。
```

这类工作都在说同一句话：

```text
OPD 的关键不是“每个 token 一样重要”，
而是“不同 turn、不同 prefix、不同深度的重要性不一样”。
```

### 4. 从“模仿最终 policy”转向“迁移改进方向 / 演化出来的监督”

Direct-OPD 和 SEED 是这条线。

```text
Direct-OPD:
  不是复制 post-RL teacher 的最终 policy，
  而是把 weak-to-strong 的改进方向直接蒸馏过去。

SEED:
  不再依赖外部 teacher，
  而是让当前 policy 自己 rollout、自己总结、自己再学习，
  形成自演化监督闭环。
```

这意味着 OPD 已经从“跟着 teacher 学”扩展到：

```text
1. 学 teacher 的改进方向
2. 学自己产生的 hindsight skills
3. 把 supervision 变成一种可迭代资源
```

## 如果只看“热点词”

这一波最常出现的词其实是：

```text
reliability
prefix
turn
horizon
curriculum
budgeting
trust region
privilege
replay
evolving supervision
```

它们背后的共同意思是：

```text
OPD 现在关心的，不只是“teacher 说了什么”，
而是“在什么位置说、对谁说、说得是否可靠、要花多少预算去听”。
```

## 一句话总结

6-7 月的 OPD 热点可以概括成：

```text
先确认 teacher 信号是否可靠，
再决定在哪些 turn / prefix / depth 上花预算，
最后把 OPD 从静态模仿推进到方向迁移、轨迹复用和自演化监督。
```
