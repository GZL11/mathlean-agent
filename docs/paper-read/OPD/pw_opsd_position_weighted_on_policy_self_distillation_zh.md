# PW-OPSD: Position-Weighted On-Policy Self-Distillation 论文解读

论文：

```text
When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning
```

来源：

```text
arXiv:2605.21606
```

## 一句话总结

PW-OPSD 解决的是 OPSD 的一个隐含假设：

```text
把 teacher 给出的所有 token 一视同仁，是否合理？
```

论文的答案是：

```text
不一定。
```

它发现 reasoning 里的 teacher-token 可靠性不是“每个位置差不多”，而是强烈依赖轨迹位置。于是 PW-OPSD 不是改 teacher，也不是加外部监督，而是：

```text
保留 OPSD 的 student rollout + privileged teacher pass + clipped forward-KL，
只把均匀 token 权重改成 position-weighted token 权重。
```

## 背景：为什么要改 OPSD

原始 OPSD 的问题是：

```text
student 自己 rollout；
teacher 在 student prefix 上做 privileged forward pass；
所有 token 位置的监督强度相同。
```

但 reasoning 里 teacher entropy 高并不总代表“这个位置不可靠”：

```text
有时是实在不确定；
有时只是存在多个合理解法分支。
```

所以单用 entropy 去做 token weighting 会有歧义。

PW-OPSD 先做了一个诊断：

```text
teacher token 的可靠性到底由什么预测？
```

## 诊断：branch-viability

论文提出 `branch-viability diagnostic`。

这里有两个术语需要先说清。

### next-token alternatives

可以翻成：

```text
下一 token 备选项
```

指的是在 teacher 分布里、某个位置上除了主答案 token 之外的其他候选 token，通常就是 teacher 认为也“有可能接得下去”的那些备选分支。它不是 student 真正采样出来的 token，而是 teacher 侧提出来的候选替代。

更具体一点，它是**单个位置**上的替换候选，不是把整条 response 都换成 teacher 轨迹。

### student 的 on-policy spine prefix

可以翻成：

```text
student 的 on-policy 主干前缀
```

这里的 `spine` 就是“主干 / 骨架”的意思，指 student 按自己的 on-policy rollout 生成出来的那条主路径 prefix。branch-viability 不是拿 teacher 自己的整条轨迹来测，而是把 teacher 给出的备选 token 接到这条 student 主干前缀上，看后续还能不能回到正确解。

直觉是：

```text
在 teacher prompt 里找 next-token alternatives，
把这些备选 token 强行接到 student 的 on-policy spine prefix 上，
看后续 student-template continuation 能不能恢复正确答案。
```

这个实验可以拆成 4 步：

```text
1. 先用 student 生成一条 on-policy rollout；
2. 在这条 rollout 的某个位置 t，取 teacher 的 next-token alternatives；
3. 用这些 alternatives 逐个替换掉 student 在位置 t 的 token；
4. 让 student 从这个新 prefix 继续续写到结束，检查最后是否还能答对。
```

所以它测的是：

```text
如果只改这一处 token，后续还能不能把整条轨迹救回正确答案。
```

这就回答了你问的第二个问题：

```text
会不会把原来好的轨迹变坏？
```

会，有可能会。因为这个实验本来就是在测“局部替换之后的分支是否仍然 viable”：

```text
原来好的轨迹
  + 替换成 teacher alternative
  -> 可能仍然好，也可能变坏。
```

论文把“替换后还能回到正确答案”记成 branch-viable；不能回到正确答案就记成 non-viable。它不是在证明 teacher token 总能改好轨迹，而是在统计：

```text
在不同位置上，teacher 的备选 token 有多大概率还能保住正确解。
```

如果一个位置的备选 token 接上去后还能走回正确答案，说明这个位置的 teacher token 可靠性更高。

论文在 `Qwen3-4B` 上发现：

```text
1. 序列内部位置 score 是最强预测信号；
2. AUROC = 0.83；
3. 局部 entropy 之类的信号最高只有 0.57 左右。
```

这说明：

```text
teacher token 的可靠性是 trajectory-structured 的，
不是纯局部 entropy 能完全解释的。
```

## PW-OPSD 怎么做

PW-OPSD 保留 OPSD 的主骨架：

```text
1. student rollout 来自 on-policy sampling；
2. teacher 仍然是 privileged self-teacher；
3. target 仍然是 clipped forward-KL；
4. 只改 token 权重。
```

核心变化是：

```text
对序列更后面的 token 给更高的位置权重。
```

论文的具体设计是一个 sigmoid 型递增权重：

```text
w_{i,t} = w_min + (1 - w_min) * sigma((r_{i,t} - tau) / s)
```

其中：

```text
r_{i,t}:
  第 i 条 response 里第 t 个 token 的归一化位置

w_min:
  早期 token 的最低权重

tau:
  权重开始明显抬升的位置阈值

s:
  sigmoid 过渡的平滑系数
```

论文默认使用：

```text
(w_min, tau, s) = (0.25, 0.30, 0.10)
```

所以它不是“前面全不管”，而是：

```text
前段保底监督；
中后段逐渐加重；
后段接近 full weight。
```

论文的直觉是：

```text
越往后，student prefix 越偏离 teacher 可验证区域，
teacher token 的 reliability 越需要重新标定。
```

## 为什么是位置加权

因为 teacher-token reliability 不是纯局部量，而是有序列结构。

论文的结论可以压成一句话：

```text
位置越靠后，teacher supervision 的价值越高。
```

于是 PW-OPSD 选择递增权重，而不是简单用 entropy 拉平所有位置。

这也解释了它和 EGRSD 的区别：

```text
PW-OPSD: 主要看位置结构。
EGRSD:   主要看 teacher uncertainty。
```

## 实验结果

论文给出的核心结论是：

```text
1. 在 AIME 2024 和 AIME 2025 上，PW-OPSD 相对 OPSD 的 Avg@12 分别提升 +1.0 和 +1.1。
2. 在 Qwen3-4B 上，位置 score 比 entropy 更能预测 token 可靠性。
3. 在 DeepSeek-R1-Distill-Llama-8B 和 Olmo-3-7B-Think 上也有一致的 aggregate Avg@12 提升。
```

论文还报告了不同 schedule：

```text
Moderate schedule:
  提升 AIME 2024 / AIME 2025 Avg@12
  但整体略弱于更激进的 schedule。

Aggressive schedule:
  在 HMMT 2025 上进一步提升，
  同时保持对 AIME 任务的正收益。
```

最关键的结论是：

```text
不需要额外 teacher compute，
只要把 token 权重改对，OPSD 就能继续涨。
```

## 具体 loss 是什么

PW-OPSD 不是另起一个新损失，而是：

```text
OPSD 的 clipped forward-KL + position weight
```

实现上，它沿用 OPSD 的那套 teacher/student full-vocab matching，只是在 token 级别乘上 `w_{i,t}`。

更具体地说：

```text
1. 计算 student 和 teacher 在 distillation temperature 下的分布；
2. 构造 per-token clipped forward-KL；
3. 对词表维度求和；
4. 乘以 position weight w_{i,t}；
5. 对有效 token 做平均；
6. 对 batch 再平均。
```

所以答案是：

```text
是，主损失还是 OPD / OPSD 那类 forward-KL；
没有额外的 PG loss，也没有单独的新 auxiliary loss。
```

## 和 OPSD 的关系

PW-OPSD 不是新范式，而是 OPSD 的轻量升级：

```text
student rollout:
  不变

privileged teacher pass:
  不变

clipped forward-KL target:
  不变

token weighting:
  从 uniform -> position-weighted
```

所以它最像：

```text
OPSD-v1.1
```

## 我怎么理解它

PW-OPSD 的价值在于它把一个直觉变成了可测量事实：

```text
teacher token 不是处处等价，
可靠性是有轨迹结构的。
```

这对后续 OPD 很重要，因为它说明：

```text
“给谁更大权重”可能比“用什么 KL”更关键。
```
