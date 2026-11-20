# EGRSD: Entropy-Guided Reinforced Self-Distillation 论文解读

论文：

```text
Respecting Self-Uncertainty in On-Policy Self-Distillation for Efficient LLM Reasoning
```

来源：

```text
arXiv:2605.13255
```

## 一句话总结

EGRSD 解决的是 OPSD 的另一个问题：

```text
teacher 不是每个位置都该同样强地教。
```

它认为 reasoning 过程里 teacher entropy 本身就携带了一个重要信号：

```text
高 entropy token 往往更不适合强行模仿。
```

所以 EGRSD 不是简单地“减小 OPSD 强度”，而是把 token 更新拆成三部分：

```text
1. reward-grounded direction
2. teacher-student likelihood-ratio magnitude
3. teacher-entropy confidence gate
```

## 背景：为什么需要 EGRSD

原始 OPSD 常把整条 CoT 的 token supervision 统一加权：

```text
每个 token 都差不多重。
```

但 reasoning 里 teacher entropy 高不一定是坏事：

```text
1. 可能是模型真的不确定；
2. 也可能是存在多个合理解法分支；
3. 还有可能只是中间转折点。
```

因此，单纯用 uniform weight 会把真正的高不确定位置和“合理分叉位置”混在一起。

## EGRSD 怎么做

EGRSD 的核心是一个 `teacher-entropy confidence gate`：

```text
高 entropy token 降权，
但不把权重压到 0。
```

也就是说，它不是 hard mask，而是保留一个 nonzero lower bound。

论文还提出了一个改进版：

```text
CL-EGRSD
```

可以翻成：

```text
因果前瞻式 EGRSD
```

它试图区分：

```text
1. 持续性的高 entropy span
2. 短暂的高 entropy pivot
```

后者如果后续上下文很快变得清晰，就不该被一刀切地强降权。

## 直观理解

可以把 EGRSD 看成：

```text
OPSD + teacher uncertainty routing
```

它不是像 PW-OPSD 那样主要看位置，也不是像 DASD 那样改方向，而是：

```text
看 teacher 自己有多不确定，再决定这一步该教多重。
```

## 实验结果

论文在：

```text
Qwen3-4B
Qwen3-8B
thinking mode
```

上做实验。

核心结论是：

```text
EGRSD 和 CL-EGRSD 在可训练方法里推进了 accuracy-length frontier。
```

也就是说，在同样或者相近的生成长度预算下，它们把准确率推高了；或者在同样准确率下，它们可以用更短的推理长度达成。

论文主张的重点不是“单点最强”，而是：

```text
更好的 accuracy-length tradeoff。
```

## 和 OPSD 的关系

EGRSD 仍然属于 OPSD 家族：

```text
student rollout 不变；
privileged teacher pass 不变；
token-level supervision 仍然是主干。
```

但它试图回答的是 OPSD 的一个更细的问题：

```text
teacher 在这个位置到底有多值得信？
```

和 PW-OPSD 相比：

```text
PW-OPSD 用位置解释 reliability；
EGRSD 用 teacher uncertainty 解释 reliability。
```

和 DASD 相比：

```text
EGRSD 主要调强度；
DASD 会调方向。
```

## 我怎么理解它

EGRSD 的价值在于它把“自蒸馏里的不确定性”正式引进了 token routing：

```text
不是所有高 entropy 都该被同样惩罚，
也不是所有低 entropy 都该被同样信任。
```

它比原始 OPSD 更像一个：

```text
uncertainty-aware token weighting
```

而不是简单的自蒸馏放大器。

