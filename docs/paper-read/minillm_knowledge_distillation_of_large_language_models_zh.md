# MiniLLM: On-Policy Distillation of Large Language Models 论文解读

论文：

```text
MiniLLM: Knowledge Distillation of Large Language Models
```

链接：

```text
https://arxiv.org/abs/2306.08543
```

## 一句话总结

MiniLLM 的核心不是“再做一次普通 KD”，而是：

```text
把生成式 LLM 的 KD 目标从 forward KL 改成 reverse KL，
并把它改写成一个可稳定优化的 policy-gradient 形式。
```

它的直觉很直接：

```text
forward KL 会逼 student 覆盖 teacher 的所有 mode，
但对生成式 LLM 来说，teacher 分布太复杂、长尾太多，
student 很容易被迫去学一些低概率、低质量区域。
```

MiniLLM 想做的是：

```text
让 student 更像 teacher 的 major modes，
而不是把 teacher 的整个分布都抄下来。
```

## 背景：为什么标准 KD 不适合生成式 LLM

传统 KD 里，常见目标是：

```text
KL[p || q]
```

也就是 teacher-to-student 的 forward KL。

这在分类任务里通常没问题，因为输出空间小、mode 少；但对 LLM 的开放式生成任务，输出空间巨大，teacher 分布会有很多模式。forward KL 的效果是：

```text
student 被迫覆盖 teacher 的所有概率质量，
包括很多它其实不该学的长尾区域。
```

MiniLLM 的判断是：

```text
对生成式 LLM，forward KL 会把概率推到 teacher 的 void regions，
导致 student 在 free-run 时更容易生成低质量文本。
```

所以它转而使用：

```text
reverse KL
```

即：

```text
KL[q || p]
```

这会让 student 更倾向于 teacher 的 major modes，避免去覆盖那些不必要的尾部区域。

## MiniLLM 的主目标

设 teacher 分布为 `p(y|x)`，student 分布为 `q_theta(y|x)`，MiniLLM 优化：

```text
L(theta)
= KL[q_theta || p]
= E_{x ~ p_x, y ~ q_theta(.|x)}
  [ log q_theta(y|x) - log p(y|x) ]
```

也可以写成论文里的等价形式：

```text
L(theta)
= - E_{x ~ p_x, y ~ q_theta(.|x)}
  [ log p(y|x) / q_theta(y|x) ]
```

这里的 `log p(y|x) / q_theta(y|x)` 应理解为：

```text
log ( p(y|x) / q_theta(y|x) )
```

所以：

```text
- log (p / q) = log (q / p)
```

也就是说，它仍然是标准的：

```text
KL[q_theta || p]
= sum_y q_theta(y|x) log q_theta(y|x) / p(y|x)
```

`q_theta` 已经作为采样分布出现在期望里，不需要再额外写一个：

```text
q_theta * KL[q_theta || p]
```

更具体地，token 级别可以写成：

```text
log q_theta(y|x) - log p(y|x)
= sum_t [
    log q_theta(y_t | x, y_<t)
    - log p(y_t | x, y_<t)
  ]
```

这和标准 KD 的差别在于：

```text
标准 KD 想让 student 覆盖 teacher；
MiniLLM 想让 student 只抓住 teacher 的主要可行分布。
```

## 为什么要把它写成 Policy Gradient

直接最小化 `KL[q || p]` 不容易优化，因为目标里有 `y ~ q_theta`，要对 student 自己的采样分布求梯度。

MiniLLM 用 policy gradient 把它变成可优化形式：

```text
∇L(theta)
= - E_{x,y ~ q_theta} [ sum_t (R_t - 1) ∇ log q_theta(y_t | y_<t, x) ]
```

这里的 `R_t` 是从当前位置往后的累计 teacher-student log-ratio 信号。

直观上：

```text
teacher 更偏好的 token -> 正向推动；
student 自己采出来但 teacher 不太喜欢的 token -> 压低。
```

## 三个稳定化设计

MiniLLM 不是只改 KL 方向，还专门加了三项稳定化。

### 1. Single-Step Decomposition

这一步的目标是降方差。

问题是：

```text
长句子的前面 token 会把后面所有误差都串起来，
导致整条句子的训练信号方差很大。
```

所以 MiniLLM 把总梯度分解成两部分：

```text
single-step quality term
long-range term
```

设：

```text
r_t = log p(y_t | x, y_<t) / q_theta(y_t | x, y_<t)
```

这里同样表示：

```text
r_t = log ( p(y_t | x, y_<t) / q_theta(y_t | x, y_<t) )
```

从第 `t` 步往后的累计信号是：

```text
R_t = r_t + r_{t+1} + ... + r_T
```

原始 policy-gradient 形式会让第 `t` 步的 token 吃到整段未来的随机性：

```text
grad_t ~= - (R_t - 1) grad log q_theta(y_t | x, y_<t)
```

因为：

```text
R_t = r_t + R_{t+1}
```

MiniLLM 把它拆成：

```text
当前这一步的 token-level 信号：r_t
后续所有 token 的 long-term 信号：R_{t+1}
```

其中 single-step 部分不再只靠本次 rollout 采到的一个 token，而是直接计算整个词表上的期望：

```text
E_{y_t ~ q_theta(.|x,y_<t)} [ r_t ]
```

而不是只靠 Monte-Carlo 采样。

注意：

```text
E_{y_t ~ q_theta} [
  log p(y_t | prefix) / q_theta(y_t | prefix)
]
= - KL[
    q_theta(. | prefix) || p(. | prefix)
  ]
```

所以 single-step 项本质上是在当前 prefix 下做一个精确的 next-token reverse-KL 对齐。

后续项 `R_{t+1}` 仍然保留，用来表示：

```text
当前 token 对后续生成质量的影响
```

因此 Single-Step Decomposition 不是“只 rollout 一步”，而是：

```text
当前 token 的分布级 RKL 精确算；
未来 token 的序列级影响仍然用 rollout 估计。
```

这能更精确地估计局部 token 的质量，训练更稳。

#### 一个具体计算例子

假设某个 prefix 是：

```text
Question: 法国首都是哪里？
Answer:
```

当前下一 token 只有三个候选，为了方便计算：

| token | student `q_theta` | teacher `p` |
|---|---:|---:|
| 巴黎 | 0.50 | 0.90 |
| 伦敦 | 0.30 | 0.05 |
| 柏林 | 0.20 | 0.05 |

对每个 token 计算：

```text
r(token) = log (p(token) / q_theta(token))
```

得到：

```text
r(巴黎) = log(0.90 / 0.50) = 0.588
r(伦敦) = log(0.05 / 0.30) = -1.792
r(柏林) = log(0.05 / 0.20) = -1.386
```

如果不用 single-step，只靠一次 student rollout，假设这次刚好采到 `巴黎`，那当前步只看到：

```text
r(巴黎) = 0.588
```

下一次如果采到 `伦敦`，当前步信号又会变成：

```text
r(伦敦) = -1.792
```

这会带来很大的采样方差。

Single-Step Decomposition 直接算当前 student 分布下的完整期望：

```text
E_q[r]
= 0.50 * 0.588
  + 0.30 * (-1.792)
  + 0.20 * (-1.386)
= -0.521
```

这等价于：

```text
- KL(q_theta || p)
```

训练时最大化这个 single-step quality term，就会推动：

```text
巴黎：teacher 概率更高，student 应该上调
伦敦：teacher 概率更低，student 应该下调
柏林：teacher 概率更低，student 应该下调
```

同时，假设 student 这条完整 rollout 是：

```text
巴黎 / 是 / 法国首都
```

并且后面两个 token 的信号是：

```text
r_2 = 0.1
r_3 = 0.2
```

原始第 1 步会用：

```text
R_1 = r_1 + r_2 + r_3
```

Single-Step 拆成：

```text
当前步 r_1：
  用全词表精确 RKL 算

未来项 R_2 = r_2 + r_3 = 0.3：
  仍然乘到 grad log q_theta(y_1 | prefix_1)，
  表示第 1 个 token 对后续生成质量的影响
```

所以它降低的是“当前 token 质量估计”的方差，而不是删除长程 credit assignment。

### 2. Teacher-Mixed Sampling

这一步是为了防止 reward hacking。

如果只按 student 自己采样，模型很容易学出一些：

```text
重复短句
空话
teacher 高分但实际退化的字符串
```

所以 MiniLLM 在采样时混合 teacher 和 student 分布：

```text
tilde p(y_t | y_<t, x)
= alpha * p(y_t | y_<t, x) + (1 - alpha) * q_theta(y_t | y_<t, x)
```

默认：

```text
alpha = 0.2
```

直观上就是：

```text
让 teacher 帮 student 兜底，
避免 student 在早期采到太多坏轨迹。
```

### 3. Length Normalization

MiniLLM 发现长序列往往会让 `R_{t+1}` 很小，模型会倾向于生成过短答案。

所以它对 long-term reward 做长度归一化：

```text
R^{Norm}_{t+1}
= 1 / (T - t - 1) * sum_{t' > t} log p(y_{t'}|x,y_<t') / q_theta(y_{t'}|x,y_<t')
```

目的就是：

```text
不要让长答案天然吃亏。
```

## 训练流程

MiniLLM 不是“只做一层 reverse KL 的普通 KD”，而是一个带 on-policy 优化的蒸馏流程：

```text
Phase 1:
  先做普通 SFT warm-up，得到一个起点。

Phase 2:
  从 Phase 1 checkpoint 继续做 MiniLLM on-policy 蒸馏训练，
  同时保留一个预训练语料上的语言模型损失 L_PT。
```

论文里的常见设置是：

```text
teacher-mix-in strength alpha = 0.2
clip epsilon = 0.2
max length = 512
mini-batch size = 64
```

最终更新并不是只靠 reverse KL 的梯度，而是：

```text
on-policy 蒸馏梯度 + 预训练语言模型损失
```

## 实验设置

### 模型 family

```text
GPT-2: 120M / 340M / 760M
OPT:   1.3B / 2.7B / 6.7B
LLaMA: 7B
```

对应 teacher 分别是：

```text
GPT-2-1.5B
OPT-13B
LLaMA-13B
```

### 数据

训练数据来自：

```text
Databricks Dolly 15K
```

测试时评测 5 个 instruction-following 数据集：

```text
DollyEval
SelfInst
VicunaEval
S-NI
UnNI
```

### 指标

```text
Rouge-L
GPT-4 feedback
Human evaluation
```

## 主结果

下面给出论文主表的核心结果。指标顺序是：

```text
DollyEval GPT4 / R-L
SelfInst  GPT4 / R-L
VicunaEval GPT4 / R-L
S-NI R-L
UnNI R-L
```

| Model | Method | DollyEval GPT4 | DollyEval R-L | SelfInst GPT4 | SelfInst R-L | VicunaEval GPT4 | VicunaEval R-L | S-NI R-L | UnNI R-L |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2 1.5B | Teacher | 58.4 | 27.6 | 42.9 | 14.3 | 48.6 | 16.3 | 27.6 | 34.9 |
| GPT-2 120M | SFT w/o KD | 38.6 | 23.3 | 26.3 | 10.0 | 32.8 | 14.7 | 16.3 | 21.4 |
| GPT-2 120M | KD | 40.3 | 22.8 | 27.8 | 10.8 | 31.9 | 13.4 | 19.7 | 24.8 |
| GPT-2 120M | SeqKD | 41.2 | 22.7 | 26.2 | 10.1 | 31.0 | 14.3 | 16.4 | 21.0 |
| GPT-2 120M | MiniLLM | 44.7 | 24.6 | 29.2 | 13.2 | 34.1 | 16.9 | 25.3 | 30.1 |
| GPT-2 340M | SFT w/o KD | 51.9 | 25.5 | 39.6 | 13.0 | 42.3 | 16.0 | 25.1 | 32.0 |
| GPT-2 340M | KD | 51.6 | 25.0 | 39.2 | 12.0 | 42.8 | 15.4 | 23.7 | 31.0 |
| GPT-2 340M | SeqKD | 50.5 | 25.3 | 39.0 | 12.6 | 43.0 | 16.9 | 22.9 | 30.2 |
| GPT-2 340M | MiniLLM | 52.2 | 25.4 | 40.5 | 15.6 | 42.6 | 17.7 | 27.4 | 34.5 |
| GPT-2 760M | SFT w/o KD | 50.7 | 25.4 | 38.3 | 12.4 | 43.1 | 16.1 | 21.5 | 27.1 |
| GPT-2 760M | KD | 53.4 | 25.9 | 40.4 | 13.4 | 43.4 | 16.9 | 25.3 | 31.7 |
| GPT-2 760M | SeqKD | 52.0 | 25.6 | 38.9 | 14.0 | 42.4 | 15.9 | 26.1 | 32.9 |
| GPT-2 760M | MiniLLM | 54.7 | 26.4 | 44.6 | 15.9 | 45.7 | 18.3 | 29.3 | 37.7 |
| OPT 13B | Teacher | 70.3 | 29.2 | 56.1 | 18.4 | 58.0 | 17.8 | 30.4 | 36.1 |
| OPT 1.3B | SFT w/o KD | 52.6 | 26.0 | 37.7 | 11.4 | 40.5 | 15.6 | 23.1 | 28.4 |
| OPT 1.3B | KD | 52.7 | 25.4 | 36.0 | 12.2 | 40.8 | 14.9 | 21.9 | 27.0 |
| OPT 1.3B | SeqKD | 51.0 | 26.1 | 36.6 | 12.7 | 42.6 | 16.6 | 21.4 | 28.2 |
| OPT 1.3B | MiniLLM | 60.7 | 26.7 | 47.0 | 14.8 | 50.6 | 17.9 | 28.6 | 33.4 |
| OPT 2.7B | SFT w/o KD | 55.4 | 27.1 | 38.9 | 13.9 | 44.8 | 16.6 | 24.9 | 32.3 |
| OPT 2.7B | KD | 60.5 | 25.9 | 48.6 | 13.8 | 51.3 | 16.7 | 26.3 | 30.2 |
| OPT 2.7B | SeqKD | 57.6 | 27.5 | 40.5 | 13.3 | 44.5 | 16.5 | 25.3 | 32.3 |
| OPT 2.7B | MiniLLM | 63.2 | 27.4 | 52.7 | 17.2 | 55.9 | 19.1 | 30.7 | 35.1 |
| OPT 6.7B | SFT w/o KD | 67.9 | 27.6 | 56.4 | 16.4 | 57.3 | 17.8 | 30.3 | 28.6 |
| OPT 6.7B | KD | 68.6 | 28.3 | 58.0 | 17.0 | 57.0 | 17.5 | 30.7 | 26.7 |
| OPT 6.7B | SeqKD | 69.6 | 28.5 | 54.0 | 17.0 | 57.6 | 17.9 | 30.4 | 28.2 |
| OPT 6.7B | MiniLLM | 70.8 | 29.0 | 58.5 | 17.5 | 60.1 | 18.7 | 32.5 | 36.7 |
| LLaMA 13B | Teacher | 79.0 | 29.7 | 75.5 | 23.4 | 65.1 | 19.4 | 35.8 | 38.5 |
| LLaMA 7B | SFT w/o KD | 73.0 | 26.3 | 69.2 | 20.8 | 61.6 | 17.5 | 32.4 | 35.8 |
| LLaMA 7B | KD | 73.7 | 27.4 | 70.5 | 20.2 | 62.7 | 18.4 | 33.7 | 37.9 |
| LLaMA 7B | SeqKD | 73.6 | 27.5 | 71.5 | 20.8 | 62.6 | 18.1 | 33.7 | 37.6 |
| LLaMA 7B | MiniLLM | 76.4 | 29.0 | 73.1 | 23.2 | 64.1 | 20.7 | 35.5 | 40.2 |

## 论文的几个关键观察

### 1. MiniLLM 普遍优于 KD 和 SeqKD

论文的总体结论是：

```text
MiniLLM 在几乎所有模型 family 和评测集上都优于标准 KD / SeqKD。
```

尤其在较小 student 上更明显。

### 2. MiniLLM 在非 Dolly 数据上更强

作者观察到：

```text
MiniLLM 在 Dolly 之外的数据集上通常收益更大。
```

这说明它不只是记住训练集，而是在 instruction-following 上学到了更稳的生成分布。

### 3. 长 response 上优势更大

对于短 response（`<= 5` tokens），MiniLLM 和 standard KD 差异不大；
但对更长的 response，MiniLLM 优势明显。

原因是：

```text
长输出空间更复杂，teacher 的多个 mode 更难被 student 完整覆盖；
reverse KL 更适合抓 major modes。
```

### 4. 生成多样性损失很小

论文统计了 Dist-4 和 LM Loss，结果显示：

```text
MiniLLM 保留了生成多样性，
并没有因为 mode-seeking 而明显塌缩。
```

表里的一个代表性结论是：

```text
MiniLLM 的 distinct 4-grams 和 baseline 接近，
说明它没有把输出变得单调。
```

## 消融实验

MiniLLM 的三个优化技巧都很重要。

| Variant | Valid Rouge-L | Dolly Rouge-L |
|---|---:|---:|
| MiniLLM | 27.4 | 24.6 |
| w/o Length Norm. | 17.4 | 14.7 |
| w/o Teacher-Mixed | 22.3 | 20.4 |
| w/o Single-Step | 27.0 | 23.7 |

这说明：

```text
1. Length normalization 最关键，去掉后性能掉得最多。
2. Teacher-mixed sampling 也很重要，否则容易 reward hacking。
3. Single-step decomposition 主要贡献在稳定训练和降方差。
```

## MiniLLM 解决了什么

这篇论文真正解决的不是“KD 能不能做”，而是：

```text
生成式 LLM 的 KD 应该更像 mode-seeking 的分布匹配，
而不是标准的 mode-covering imitation。
```

它的核心收益可以总结成四点：

```text
1. 更高的生成精度
2. 更低的 exposure bias
3. 更好的 calibration
4. 更强的长文本生成能力
```

## 和 OPD / RL 系列的关系

MiniLLM 和后来的 OPD / RLSD / RLVR 不同，它不是 on-policy 蒸馏，而是更接近：

```text
teacher distribution -> student policy optimization
```

但它和后面的 work 有一个共同主题：

```text
不要把监督分布当成绝对真理，
要考虑 student 容量、生成分布形状和长尾 mode 的影响。
```

MiniLLM 提前把这件事说得很清楚：

```text
对生成式模型，reverse KL 往往比 forward KL 更合适。
```

## 局限

### 1. 需要白盒 teacher

MiniLLM 依赖 teacher 分布可见，这不是黑盒 API KD。

### 2. 仍然需要采样和策略优化

虽然它是 KD，但优化过程已经很像 RL，工程复杂度比普通 SFT 高。

### 3. mode-seeking 不保证全局最优

reverse KL 有利于抓 major modes，但在一些需要覆盖多样答案的任务里，过强 mode-seeking 仍可能压掉有用多样性。

## 一句话结论

MiniLLM 的贡献可以概括成：

```text
把 LLM 的 KD 从 forward KL 的“全覆盖模仿”，
推进到 reverse KL 的“主要模式蒸馏”，
并用一套稳定化技巧把它做成可训练、可扩展、有效的方法。
```

## BibTeX

```bibtex
@misc{gu2023minillm,
  title={MiniLLM: Knowledge Distillation of Large Language Models},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  year={2023},
  eprint={2306.08543},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2306.08543}
}
```
