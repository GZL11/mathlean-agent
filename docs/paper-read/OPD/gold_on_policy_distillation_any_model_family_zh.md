# GOLD：任意模型族 On-Policy Distillation 技术解读

本文档解读 Hugging Face H4 的 research article：

```text
Unlocking On-Policy Distillation for Any Model Family
https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
published: Oct. 29, 2025
```

作者提出 **GOLD: General On-Policy Logit Distillation**，目标是把 on-policy distillation 从“teacher 和 student 必须共享 tokenizer”的限制中解放出来。

## Student / Teacher 模型速览

GOLD 关注的是不同 tokenizer / 不同模型族之间的 on-policy logit distillation。文章实验里的核心 teacher：

```text
Teacher:
  Qwen/Qwen2.5-7B-Instruct
  Qwen/Qwen3-4B-Instruct-2507
```

主要 student：

```text
Student:
  Qwen/Qwen2.5-1.5B-Instruct
  meta-llama/Llama-3.2-1B-Instruct
  google/gemma-3-1b-it
```

最重要的跨 tokenizer 设置是：

```text
Qwen3-4B-Instruct-2507 teacher
  -> Qwen2.5-1.5B student   同族/近似同 tokenizer
  -> Llama-3.2-1B student    跨模型族/不同 tokenizer
  -> Gemma-3-1B student      跨模型族/不同 tokenizer
```

所以 GOLD 的重点不是 teacher 多强，而是 teacher/student tokenizer 不同时如何对齐序列位置和词表概率。

## 一句话总结

GOLD 解决的是：

```text
teacher 和 student 来自不同模型族、tokenizer 不同，
但仍希望做 on-policy logit distillation。
```

它的核心方法是：

```text
1. 让 student 自己生成 completion。
2. teacher 对 student completion 给出 token distribution。
3. 如果两边 tokenizer 不同，先做 sequence alignment。
4. 再做 vocabulary alignment。
5. 最后用 logit / probability distribution loss 蒸馏 teacher。
```

和 Uni-OPD 的一句话区别：

```text
GOLD:
  重点是跨 tokenizer 的 full-logit on-policy distillation。

Uni-OPD:
  重点是用 outcome reward 校准 sampled-token OPD advantage。
```

## 背景：为什么普通 OPD 需要相同 tokenizer

普通 on-policy distillation / GKD 的训练流程是：

```text
1. student 根据 prompt 生成 response。
2. teacher 在同一个 response 上计算 token-level logits / probabilities。
3. student 用 teacher distribution 做 KL / JSD / CE 类蒸馏。
```

这里的 KL / JSD / CE 都是在每个 token position 上比较两个词表分布。记：

```text
p_T(v):
  teacher 在当前 context 下对 vocab token v 的概率。

p_S(v):
  student 在当前 context 下对 vocab token v 的概率。

V:
  teacher/student 已对齐后的词表空间。
```

序列级 loss 通常是对所有 response token position 求和或平均。

### KL Distillation

常见 teacher-to-student KL：

```text
D_KL(p_T || p_S)
  = sum_{v in V} p_T(v) * log(p_T(v) / p_S(v))
```

训练时 `p_T` 不反传，优化 student。去掉和 student 无关的 teacher entropy 项后，等价于最小化：

```text
- sum_{v in V} p_T(v) * log p_S(v)
```

含义：

```text
让 student 在 teacher 认为高概率的 token 上也给高概率。
```

方向很重要：

```text
D_KL(p_T || p_S):
  teacher-covering。
  student 要覆盖 teacher 的概率质量，不能漏掉 teacher 认为可能的 token。

D_KL(p_S || p_T):
  mode-seeking。
  student 更倾向集中到 teacher 分布的高概率模式上。
```

GKD 文中常通过 `beta` 参数控制 generalized JSD 的方向；当 `beta=0` 时，文中称为 forward KL setting。

### CE Distillation

Cross Entropy 蒸馏：

```text
H(p_T, p_S)
  = - sum_{v in V} p_T(v) * log p_S(v)
```

它和 `D_KL(p_T || p_S)` 的关系是：

```text
H(p_T, p_S)
  = D_KL(p_T || p_S) + H(p_T)
```

其中：

```text
H(p_T) = -sum_v p_T(v) * log p_T(v)
```

对 student 来说，`H(p_T)` 是常数，所以最小化 CE 和最小化 teacher-to-student KL 的梯度等价。

含义：

```text
把 teacher soft label 当作训练目标。
相比 hard-label SFT，它不仅告诉 student 正确 token，
还告诉 student teacher 对其他候选 token 的相对偏好。
```

### JSD Distillation

Jensen-Shannon Divergence 是对称、平滑的分布距离。标准形式：

```text
m = 0.5 * (p_T + p_S)

JSD(p_T, p_S)
  = 0.5 * D_KL(p_T || m)
    + 0.5 * D_KL(p_S || m)
```

GKD 使用 generalized JSD：

```text
pi_beta = beta * p_S + (1 - beta) * p_T

D_JSD(beta)(p_S, p_T)
  = beta * D_KL(p_S || pi_beta)
    + (1 - beta) * D_KL(p_T || pi_beta)
```

含义：

```text
不是直接把 student 拉到 teacher，
而是把二者都和一个混合分布 pi_beta 对齐。
```

`beta` 的作用：

```text
beta 越接近 0:
  更接近 teacher-to-student / forward-KL 风格。

beta 越接近 1:
  更强调 student-to-teacher / reverse-KL 风格。

beta = 0.5:
  更接近标准对称 JSD。
```

直觉对比：

```text
CE / forward KL:
  更像“完整模仿 teacher 的 soft distribution”。

reverse KL:
  更像“选择 teacher 最认可的高概率模式”。

JSD:
  在两者之间折中，训练更平滑，数值上也更稳定。
```

如果 teacher 和 student tokenizer 相同，这很直接：

```text
response token 序列长度一致
token id 对应同一个词表元素
teacher logits.shape == student logits.shape
```

但跨模型族时问题很大：

```text
Llama tokenizer
Qwen tokenizer
Gemma tokenizer
SmolLM tokenizer
```

这些 tokenizer 对同一段文本的切分、token id、词表大小都可能不同。于是有两个核心 mismatch：

```text
1. Sequence mismatch
   同一文本被切成不同长度的 token 序列。

2. Vocabulary mismatch
   同一个字符串 token 在两个 tokenizer 中 id 不同；
   甚至某些 token 只存在于一个词表里。
```

GOLD 就是在处理这两个 mismatch。

## Off-Policy 和 On-Policy Distillation

文中把蒸馏分成两类：

```text
Off-policy distillation:
  student 在固定数据上训练。
  数据通常来自 teacher 预先生成的 completion 或 teacher logits。

On-policy distillation:
  student 自己生成 completion。
  teacher 对 student 生成的 completion 提供 token-level feedback。
```

On-policy 的优势：

```text
1. context alignment
   训练数据来自 student 自己会遇到的状态分布。

2. positive feedback loop
   student 越强，生成的数据越接近目标任务分布。

3. token-level dense signal
   比 RL 的 sparse outcome reward 更细。
```

GKD 用 `lambda` 控制 off-policy / on-policy 混合：

```text
L_GKD = (1 - lambda) * L_SD + lambda * L_OD
```

其中：

```text
lambda = 0:
  完全 off-policy / supervised distillation。

lambda = 1:
  完全 on-policy distillation。
```

文中实验发现：

**lambda 越高通常越好；完全 on-policy 效果最好。**

## GKD、ULD、GOLD 三者关系

对应论文：

```text
GKD:
  On-Policy Distillation of Language Models:
  Learning from Self-Generated Mistakes
  Rishabh Agarwal et al.
  arXiv:2306.13649
  https://arxiv.org/abs/2306.13649

ULD:
  Towards Cross-Tokenizer Distillation:
  the Universal Logit Distillation Loss for LLMs
  Nicolas Boizard et al.
  arXiv:2402.12030
  https://arxiv.org/abs/2402.12030

GOLD:
  Unlocking On-Policy Distillation for Any Model Family
  Hugging Face H4
  https://huggingface.co/spaces/HuggingFaceH4/on-policy-distillation
```

### GKD

GKD 是 Generalized Knowledge Distillation。

特点：

```text
1. 支持 off-policy 和 on-policy 混合。
2. 使用 teacher / student distribution 的 divergence loss。
3. 默认假设 teacher 和 student tokenizer 一致。
```

核心限制：

```text
如果 tokenizer 不同，teacher logits 和 student logits 没法直接按 token 位置和 vocab id 对齐。
```

### GKD 为什么支持 off-policy 和 on-policy 混合

GKD 的关键是把训练样本来源拆成两类：

```text
1. off-policy / supervised distillation 数据
   来自固定数据集，通常是 teacher 预先生成的 completion，
   或者人工/静态数据上的 teacher logits。

2. on-policy distillation 数据
   来自 student 当前 policy 自己生成的 completion，
   然后 teacher 对这些 student completion 提供 logits / distribution feedback。
```

于是可以用 `lambda` 线性混合两个 loss：

```text
L_GKD
  = (1 - lambda) * L_SD
    + lambda * L_OD
```

其中：

```text
L_SD:
  supervised distillation loss。
  数据分布来自固定 teacher/fixed completions。

L_OD:
  on-policy distillation loss。
  数据分布来自 student 自己生成的 response。
```

不同 `lambda` 的含义：

```text
lambda = 0:
  完全 off-policy。
  只在 teacher/fixed completions 上训练。

lambda = 1:
  完全 on-policy。
  只在 student 自己生成的数据上训练。

0 < lambda < 1:
  off-policy + on-policy 混合。
```

为什么这种混合合理：

```text
off-policy teacher 数据:
  优点是质量高、训练稳定，适合 student 很弱的早期。
  缺点是 context mismatch，student 推理时不会总处在 teacher-generated context 下。

on-policy student 数据:
  优点是 context alignment 好，teacher 可以纠正 student 自己会犯的错误。
  缺点是 student 初期生成质量差，训练信号更噪。
```

因此 GKD 的 `lambda` 可以看成一个数据分布层面的 knob：

```text
低 lambda:
  更像 SFT / teacher forcing，稳定。

高 lambda:
  更贴近 student 推理分布，on-policy 性更强。
```

GOLD 文章里的实验也观察到：

```text
lambda 越高通常越好；
fully on-policy 效果最好。
```

### ULD

ULD 是 Universal Logit Distillation。

目标：

```text
支持不同 tokenizer 的 logit distillation。
```

但它主要是 offline setting，并且对齐方式比较粗：

```text
Sequence alignment:
  直接截断到较短序列长度。

Vocabulary alignment:
  pad 到相同 vocab size 后，对 softmax vector 排序。
```

### ULD 中 student token 不在 teacher vocab 时怎么办

这是 ULD 和 sampled-token OPD 最大的区别。

Uni-OPD / sampled-token OPD 需要计算：

```text
teacher_logprob(student_sampled_token | context)
```

如果 student token 不在 teacher vocab 中，这个值就没有直接定义。例如：

```text
student tokenizer:
  "123" 是一个 token

teacher tokenizer:
  没有 "123"，只能切成 "1", "2", "3"
```

那么普通 OPD 想算：

```text
log pi_T("123" | context)
```

会卡住，因为 teacher 的动作空间里没有 `"123"` 这个 token。

ULD 不这样算。ULD 不要求 teacher vocab 中存在 student rollout token。它比较的是两个完整 probability vector 的“形状”：

```text
p_S:
  student 当前 token position 上的 vocab distribution
  shape = [|V_S|]

p_T:
  teacher 当前 token position 上的 vocab distribution
  shape = [|V_T|]
```

如果 vocab size 不同：

```text
1. 将较短的 probability vector padding 到相同长度。
2. 分别对 p_S 和 p_T 降序排序。
3. 按排序后的 index 比较概率质量。
```

可以近似写成：

```text
L_ULD_vocab
  = sum_i | sort(p_S)_i - sort(p_T)_i |
```

### ULD 计算例子

假设当前同一个文本位置上，student 和 teacher 的 tokenizer 不同。

student vocab：

```text
V_S = ["123", "1", "2", "+", "</s>"]
```

teacher vocab：

```text
V_T = ["1", "2", "3", "12", "+", "=", "</s>"]
```

注意：

```text
student 有 token "123"；
teacher 没有 token "123"。
```

普通 sampled-token OPD 如果 student 当前 rollout token 是 `"123"`，会想算：

```text
log pi_T("123" | context)
```

但 teacher vocab 中没有 `"123"`，所以不能直接算。

ULD 的做法是取两边当前 token position 的完整概率分布。假设 student 给出：

```text
p_S over V_S:

"123" : 0.40
"1"   : 0.25
"2"   : 0.15
"+"   : 0.12
"</s>": 0.08
```

写成向量：

```text
p_S = [0.40, 0.25, 0.15, 0.12, 0.08]
```

teacher 给出：

```text
p_T over V_T:

"1"   : 0.30
"2"   : 0.22
"3"   : 0.18
"12"  : 0.10
"+"   : 0.09
"="   : 0.07
"</s>": 0.04
```

写成向量：

```text
p_T = [0.30, 0.22, 0.18, 0.10, 0.09, 0.07, 0.04]
```

两边 vocab size 不同：

```text
|V_S| = 5
|V_T| = 7
```

先把较短的 `p_S` padding 到长度 7：

```text
p_S_pad = [0.40, 0.25, 0.15, 0.12, 0.08, 0.00, 0.00]
```

然后分别降序排序：

```text
sort(p_S_pad)
  = [0.40, 0.25, 0.15, 0.12, 0.08, 0.00, 0.00]

sort(p_T)
  = [0.30, 0.22, 0.18, 0.10, 0.09, 0.07, 0.04]
```

逐项做 L1 差：

```text
|0.40 - 0.30| = 0.10
|0.25 - 0.22| = 0.03
|0.15 - 0.18| = 0.03
|0.12 - 0.10| = 0.02
|0.08 - 0.09| = 0.01
|0.00 - 0.07| = 0.07
|0.00 - 0.04| = 0.04
```

因此这个位置上的 ULD vocab loss 近似为：

```text
L_ULD_vocab
  = 0.10 + 0.03 + 0.03 + 0.02 + 0.01 + 0.07 + 0.04
  = 0.30
```

这个 loss 的含义不是：

```text
teacher 认为 student token "123" 的概率是多少。
```

而是：

```text
student 当前分布的最大概率质量是 0.40，
teacher 当前分布的最大概率质量是 0.30；
student 第二大概率是 0.25，
teacher 第二大概率是 0.22；
...

ULD 希望两边排序后的 probability profile 更接近。
```

因此，即使 `"123"` 不在 teacher vocab 里，ULD 仍然可以计算 loss，因为它完全绕开了 token string 对齐问题。

代价是语义很弱：

```text
它并不知道:
  student 的 "123" 应该对应 teacher 的 "1" + "2" + "3"。

它只知道:
  student 的概率分布形状应该接近 teacher 的概率分布形状。
```

如果把这个位置放到一个 response 序列里，ULD 会对每个对齐后的 position 都算一次类似的分布距离，然后对所有位置求和或平均：

```text
L_ULD_sequence
  = mean_t L_ULD_vocab(t)
```

### 为什么降序排序能作为一种分布距离

这里要特别注意：

```text
排序后的 ULD 距离不是在衡量 token 语义是否相同。
它衡量的是两个概率分布的 probability mass profile 是否相似。
```

也就是说，它关心：

```text
teacher 是否也很 confident？
teacher 的 top-1 概率有多大？
teacher 的 top-k probability tail 衰减是否和 student 类似？
teacher 分布是尖锐还是平坦？
```

而不是关心：

```text
student top-1 token 和 teacher top-1 token 是不是同一个字符串。
```

理论直觉来自一个“未知排列”的问题。

如果两个 tokenizer 的 vocab id 完全不对应，那么我们并不知道：

```text
student vocab 第 i 个 token 应该对应 teacher vocab 第几个 token。
```

这时可以把 teacher distribution 看成 student distribution 的某个未知排列：

```text
p_T_perm = permute(p_T)
```

理想情况下，如果两个 tokenizer 只是换了 token id，那么存在一个 permutation 使得：

```text
p_S ≈ permute(p_T)
```

但是这个 permutation 不知道。ULD 的排序做法等价于选择一个“最乐观”的排列：

```text
把最大概率对最大概率，
把第二大概率对第二大概率，
...
把最小概率对最小概率。
```

对于 L1 / Lp 这类只比较数值差的距离，同向排序可以给出在所有排列中很小、通常也是最优的匹配距离。直觉上：

```text
如果把最大概率和最小概率硬配在一起，差值会被人为放大；
如果把大配大、小配小，得到的是最合理的 unknown-permutation matching。
```

这和一维 optimal transport / rearrangement inequality 的思想接近：

```text
当只知道两组 probability masses，
不知道元素语义对应关系时，
按排序后的 quantile / rank 做匹配，
是一个自然的 permutation-invariant 距离。
```

因此，ULD 的排序距离具备一个性质：

```text
permutation-invariant:
  不依赖具体 vocab id。
```

这正是跨 tokenizer 时需要的，因为：

```text
student token id = 10
teacher token id = 10
```

通常没有任何语义关系。

但这也是它的根本局限：

```text
排序距离丢掉了 token identity。
```

例如下面两个 teacher 分布，排序后可能完全一样：

```text
teacher A:
  "1": 0.50, "2": 0.30, "3": 0.20

teacher B:
  "cat": 0.50, "dog": 0.30, "tree": 0.20
```

排序后都是：

```text
[0.50, 0.30, 0.20]
```

ULD 会认为它们在 probability profile 上一样，但语义完全不同。

所以 ULD 的理论依据可以概括为：

```text
在没有 tokenizer mapping 时，
用排序后的 probability profile 做一个 permutation-invariant 的分布形状匹配。
```

它成立的前提偏弱：

```text
假设 teacher 和 student 在同一 context 下应该有相似的置信度结构，
即 top-k 概率形状相似。
```

它不保证：

```text
top-k token 的语义真的对应。
```

这也是 GOLD 要进一步做 matched-token mapping 的原因：

```text
能用 token string 精确匹配的部分，
就不要只靠排序形状；

实在无法匹配的 unmatched 部分，
才退回 ULD 排序近似。
```

更直观地说：

```text
ULD 不问:
  teacher 对 student token "123" 给多少概率？

ULD 问:
  student 的整词表概率分布形状，
  是否接近 teacher 的整词表概率分布形状？
```

所以当 student rollout token 不在 teacher vocab 时：

```text
不会单独为这个 token 计算 teacher_logprob；
它只通过整分布排序后的概率质量差异，给 student 一个间接的分布级监督。
```

这也是 ULD 的局限：

```text
优点:
  不需要相同 tokenizer，也不需要 token string 能 1:1 对齐。

缺点:
  语义监督很弱。
  它不知道 student 的 "123" 应该对应 teacher 的 "1", "2", "3"。
  它只让两个 distribution 的 rank/probability profile 更像。
```

GOLD 的改进就是：

```text
能找到 token string 1:1 mapping 的部分:
  用 GKD / KL 类精确蒸馏。

找不到 mapping 的部分:
  才 fallback 到 ULD 的排序 / padding 近似。
```

问题：

```text
1. 截断会丢掉尾部信息。
2. 序列早期一旦错位，后续 token 语义全部错位。
3. 排序假设两个分布只是 token id permutation，但没有利用真实共享 token。
```

### GOLD

GOLD 是本文提出的方法。

它可以理解为：

```text
GOLD = on-policy ULD + 更好的 sequence alignment + 更好的 vocabulary alignment
```

核心公式：

```text
L_GOLD(x, y) = w1 * L_GKD + w2 * L_ULD
```

直觉：

```text
能做 1:1 token 映射的词表部分:
  用 GKD loss。

不能做 1:1 映射的词表部分:
  fallback 到 ULD 的排序 / padding loss。
```

## GOLD 训练流程

一轮训练可以写成：

```text
1. 给定 prompt x。

2. student 生成 response y。

3. 用 student tokenizer 得到:
   student token sequence:
     s_1, s_2, ..., s_m
   student logits:
     P_S(. | x, prefix)

4. 用 teacher tokenizer 对同一文本 response y 重新编码。
   得到:
     teacher token sequence:
       t_1, t_2, ..., t_n
     teacher logits:
       P_T(. | x, prefix)

5. 做 sequence alignment:
   把 student token group 和 teacher token group 对齐到同一文本 span。

6. 做 vocabulary alignment:
   匹配共享 token；
   未匹配部分用 ULD-style sorting/padding。

7. 计算 distribution distillation loss。
```

这和 sampled-token OPD 不同。GOLD 关注的是 distribution-level logit distillation：

```text
teacher 给的是整词表分布。
student 学的是整词表分布。
```

## Sequence Alignment

### ULD 的做法

ULD 简单做：

```text
min_len = min(student_seq_len, teacher_seq_len)
截断两边 token 序列到 min_len
```

问题：

```text
同一个 index 的 token 可能不是同一个语义片段。
```

例如：

```text
Tokenizer A:
  ["Hugging Face", " is", " awesome!"]

Tokenizer B:
  ["Hugging", " Face", " is", " awesome!"]
```

如果按 index 对齐：

```text
A[0] = "Hugging Face"
B[0] = "Hugging"
```

语义已经错位。后续再截断，会持续污染 loss。

### GOLD 的做法

GOLD 不直接截断，而是先找出 token merge。

核心思想：

```text
把覆盖同一段原始文本的多个 token 合并成一个 aligned group。
```

例如：

```text
student:
  ["Hugging", " Face"]

teacher:
  ["Hugging Face"]
```

合并后：

```text
student group:
  ["Hugging", " Face"]

teacher group:
  ["Hugging Face"]

二者覆盖同一段文本 span。
```

合并概率时利用自回归链式法则：

```text
P("Hugging Face" | context)
  = P("Hugging" | context)
    * P(" Face" | context, "Hugging")
```

在 logprob 空间就是：

```text
log P_merged = sum(log P_each_token)
```

文中说原始实现用 decoded string buffer 判断两边何时相等；2026 年 6 月更新后改成 **UTF-8 byte offset walker**：

```text
1. 用 fast tokenizer 得到每个 token 在原文本中的 byte span。
2. 两边按 byte end 前进。
3. 当 student 和 teacher 当前 group 的 byte end 相同时，关闭一个对齐 group。
```

好处：

```text
1. 不依赖 decode 后字符串。
2. 能处理 BOS token 长度不对称。
3. 对 ByteLevel BPE 更稳。
4. 对齐条件更简单：byte_end 相同。
```

例如 Llama 有 `<|begin_of_text|>`，Qwen 没有对应 BOS。如果用字符串 buffer，第一步就会错位；byte span 里 BOS 可以是 zero-width span，walker 能自然跳过。

## Vocabulary Alignment

### ULD 的做法

ULD 假设没有可靠 token mapping，于是：

```text
1. 将两个 vocab distribution pad 到同一长度。
2. 对 softmax vector 排序。
3. 在排序后的 probability vector 上计算 distillation loss。
```

这个方法的隐含假设：

```text
两个模型的概率分布形状相似，只是 vocab id permutation 不同。
```

缺点：

```text
没有利用“同一个字符串 token 在两个词表中都存在”这个事实。
```

### GOLD 的做法

GOLD 做 hybrid vocabulary alignment：

```text
1. 找出两个 tokenizer 中字符串完全匹配的 token。
2. 对这些 token 建立 1:1 mapping。
3. 对 matched tokens 使用 GKD loss。
4. 对 unmatched tokens 使用 ULD fallback。
```

最终：

```text
L_GOLD = w1 * L_GKD_matched + w2 * L_ULD_unmatched
```

### GOLD 计算例子

继续使用前面的 toy vocab。

student vocab：

```text
V_S = ["123", "1", "2", "+", "</s>"]
```

teacher vocab：

```text
V_T = ["1", "2", "3", "12", "+", "=", "</s>"]
```

当前 token position 上的概率分布：

```text
p_S:
  "123" : 0.40
  "1"   : 0.25
  "2"   : 0.15
  "+"   : 0.12
  "</s>": 0.08

p_T:
  "1"   : 0.30
  "2"   : 0.22
  "3"   : 0.18
  "12"  : 0.10
  "+"   : 0.09
  "="   : 0.07
  "</s>": 0.04
```

### Step 1：找 matched tokens

两个 vocab 中字符串完全相同的 token 是：

```text
matched = ["1", "2", "+", "</s>"]
```

这些 token 虽然在两个 tokenizer 里的 token id 可能不同，但 token string 一样，所以可以建立 1:1 mapping：

```text
"1"    <-> "1"
"2"    <-> "2"
"+"    <-> "+"
"</s>" <-> "</s>"
```

无法匹配的 token：

```text
student unmatched:
  ["123"]

teacher unmatched:
  ["3", "12", "="]
```

### Step 2：matched 部分算 GKD / KL / CE

先取 matched tokens 上的概率：

```text
p_S_matched:
  "1"   : 0.25
  "2"   : 0.15
  "+"   : 0.12
  "</s>": 0.08

p_T_matched:
  "1"   : 0.30
  "2"   : 0.22
  "+"   : 0.09
  "</s>": 0.04
```

实际实现里可以在 matched 子空间上做 KL / JSD / CE。为了手算简单，这里用 teacher-to-student CE：

```text
L_GKD_matched
  = - sum_{v in matched} p_T(v) * log p_S(v)
```

代入自然对数：

```text
= -[
    0.30 * log(0.25)
  + 0.22 * log(0.15)
  + 0.09 * log(0.12)
  + 0.04 * log(0.08)
  ]
```

近似：

```text
log(0.25) ≈ -1.386
log(0.15) ≈ -1.897
log(0.12) ≈ -2.120
log(0.08) ≈ -2.526
```

所以：

```text
L_GKD_matched
  ≈ -[
      0.30 * (-1.386)
    + 0.22 * (-1.897)
    + 0.09 * (-2.120)
    + 0.04 * (-2.526)
    ]

  ≈ 1.125
```

这个 matched loss 有明确语义：

```text
teacher 认为 "1" 概率是 0.30，
student 对 "1" 给 0.25；

teacher 认为 "2" 概率是 0.22，
student 对 "2" 给 0.15；

这些 token 是真实对齐的，所以可以直接蒸馏。
```

### Step 3：unmatched 部分算 ULD fallback

unmatched 部分没有 1:1 token string mapping。

student unmatched probability：

```text
p_S_unmatched:
  "123": 0.40
```

teacher unmatched probability：

```text
p_T_unmatched:
  "3" : 0.18
  "12": 0.10
  "=" : 0.07
```

写成向量：

```text
p_S_unmatched = [0.40]
p_T_unmatched = [0.18, 0.10, 0.07]
```

padding 到同样长度：

```text
p_S_unmatched_pad = [0.40, 0.00, 0.00]
p_T_unmatched     = [0.18, 0.10, 0.07]
```

排序后仍然是：

```text
sort(p_S_unmatched_pad)
  = [0.40, 0.00, 0.00]

sort(p_T_unmatched)
  = [0.18, 0.10, 0.07]
```

做 L1 距离：

```text
L_ULD_unmatched
  = |0.40 - 0.18|
  + |0.00 - 0.10|
  + |0.00 - 0.07|

  = 0.22 + 0.10 + 0.07
  = 0.39
```

这部分没有明确语义对齐：

```text
GOLD 不知道 student 的 "123" 应该对应 teacher 的 "3"、"12" 还是 "="。
```

它只用 ULD fallback 约束：

```text
unmatched probability mass 的形状不要和 teacher 差太多。
```

### Step 4：组合成 GOLD loss

假设权重：

```text
w1 = 1.0
w2 = 1.0
```

则：

```text
L_GOLD
  = w1 * L_GKD_matched
    + w2 * L_ULD_unmatched

  = 1.0 * 1.125
    + 1.0 * 0.39

  = 1.515
```

如果希望更信任精确 token mapping、降低 unmatched 近似噪声，可以调小 `w2`：

```text
w1 = 1.0
w2 = 0.2

L_GOLD
  = 1.125 + 0.2 * 0.39
  = 1.203
```

这个例子体现 GOLD 的核心：

```text
matched 部分:
  用真实 token string 对齐，做精确 distribution distillation。

unmatched 部分:
  没有语义对齐，退回 ULD 的排序分布形状匹配。

最终:
  两部分加权求和。
```

直觉：

```text
能准确对齐的 token，就不要用排序近似；
不能准确对齐的部分，再退回 ULD。
```

## Tokenizer Similarity

文中用 Jaccard index 衡量 tokenizer similarity：

```text
similarity = |matched tokens| / |union tokens|
```

它比较了两种匹配：

```text
Strict Matching:
  token id 必须相同。

Token Mapping:
  token string 相同即可，即使 token id 不同也算 matched。
```

实验观察：

```text
Llama-3.2-1B vs Qwen teacher:
  strict similarity = 0
  token mapping similarity = 0.64

Gemma-3-1B vs Qwen teacher:
  strict similarity = 0
  token mapping similarity = 0.063

Qwen2.5-1.5B vs Qwen3-4B:
  similarity ≈ 0.999974
```

Qwen2.5 和 Qwen3 tokenizer 几乎相同，差异主要是 Qwen3 多了：

```text
<think>
<tool_response>
</tool_response>
</think>
```

因此 Qwen2.5 student 和 Qwen3 teacher 基本可以看成 same-tokenizer case。

## 实验设置

### 任务：Countdown

Countdown 是一个算术游戏：

```text
给一组数字和目标值，
模型必须用 + - * / 以及每个数字最多一次，
构造一个等于目标值的表达式。
```

正确性要求：

```text
1. 每个数字只能使用一次。
2. 表达式结果等于目标。
3. 答案必须放在 <answer>...</answer> 标签内。
```

这类任务适合比较 KD 和 RL：

```text
1. 有可验证 reward。
2. 又需要多步 reasoning。
3. 格式、规则、答案正确性可以分开评分。
```

### 数据

prompt 来源：

```text
Jiayi-Pan/Countdown-Tasks-3to4
```

数据规模：

```text
80k train prompts
10k test prompts
```

teacher completion 过滤：

```text
只保留 teacher 能正确回答的 prompt。
```

公开数据：

```text
Qwen2.5-7B-Instruct teacher:
  30.4k prompts

Qwen3-4B-Instruct-2507 teacher:
  27.7k prompts
```

on-policy 实验使用的是 prompts，而不是固定 teacher completion，因为 response 来自 student 自己生成。

### 模型

Student：

```text
meta-llama/Llama-3.2-1B-Instruct
  baseline Countdown: 0.016

Qwen/Qwen2.5-1.5B-Instruct
  baseline Countdown: 0.076

google/gemma-3-1b-it
  baseline Countdown: 0.023
```

Teacher：

```text
Qwen/Qwen2.5-7B-Instruct
  Countdown: 0.3555

Qwen/Qwen3-4B-Instruct-2507
  Countdown: 0.7145
```

## 主要实验结论

### 1. Same-tokenizer 下 fully on-policy 最好

实验：

```text
student:
  Qwen/Qwen2.5-1.5B-Instruct

teacher:
  Qwen/Qwen3-4B-Instruct-2507
```

改变 `lambda`：

```text
lambda = 0:
  SFT / offline distillation

lambda = 1:
  fully on-policy distillation
```

结论：

```text
只要加入 on-policy，通常就优于纯 SFT；
lambda 越高效果越好；
fully on-policy 最强。
```

### 2. GKD 能稳定蒸馏 teacher 80% 以上能力

文中说在多个 teacher 设置下，GKD 可以稳定恢复 teacher 在 Countdown 上超过 80% 的性能。

含义：

```text
student 最终能力上限很大程度取决于 teacher。
teacher 越强，student 可蒸馏出来的最终性能越高。
```

### 3. GOLD 支持不同 tokenizer

跨 tokenizer 设置：

```text
student:
  meta-llama/Llama-3.2-1B-Instruct

teacher:
  Qwen/Qwen3-4B-Instruct-2507
```

结果：

```text
GOLD:
  student 初始性能提升 25%
  恢复 teacher 约 60% 性能

ULD:
  student 初始性能提升 5%
  只恢复 teacher 约 10% 性能
```

原因：

```text
GOLD 的 sequence alignment + vocabulary hybrid alignment 显著降低了 tokenization mismatch 噪声。
```

### 4. Tokenizer similarity 越高，效果越好

文中表格：

```text
Qwen/Qwen2.5-1.5B-Instruct:
  Countdown 0.6515
  similarity with Qwen3 teacher: 0.999974

Llama-3.2-1B-Instruct:
  Countdown 0.4235
  similarity: 0.64

Gemma-3-1B-it:
  Countdown 0.0305
  similarity: 0.063
```

说明：

```text
GOLD 不是魔法。
tokenizer 越不相似，对齐噪声越大，蒸馏效果越差。
```

### 5. GOLD / GKD 优于 GRPO

文中用 Countdown 任务比较 GRPO：

GRPO reward 包含：

```text
1. Format:
   正确使用 <answer></answer> 标签。

2. Following Rules:
   只使用 prompt 给出的数字，且每个数字只用一次。

3. Correct Equation:
   表达式结果正确。
```

结论：

```text
same-tokenizer:
  KD 约为 GRPO 2x performance。

cross-tokenizer:
  GOLD 仍比 GRPO 高约 20%。
```

解释：

```text
RL reward 是 sparse outcome / rule reward；
GOLD/GKD 给 teacher token distribution，
每个 token 都有 dense learning signal。
```

## Domain Distillation：领域微调后的能力恢复

文中还复现了 Thinking Machines 的 domain personalization 思路。

问题：

```text
领域 SFT 能提升目标领域能力，
但会损害通用 instruction following。
```

实验：

```text
base:
  Qwen/Qwen3-4B-Instruct-2507

domain SFT data:
  open-r1/codeforces

domain eval:
  livecodebench

general ability eval:
  IFEval
```

SFT 后：

```text
LiveCodeBench:
  35.1% -> 40.3%

IFEval:
  83.4% -> 79.5%
```

然后用 GKD 在：

```text
allenai/tulu-3-sft-mixture
```

上蒸馏初始 Qwen3-4B teacher。

结果：

```text
IFEval:
  79.5% -> 82.8%

LiveCodeBench:
  约 39.8%，基本保持领域收益。
```

这说明 on-policy distillation 不只用于 math reasoning，也可以用于：

```text
领域微调后恢复 chat / instruction-following 能力。
```

## 和 Uni-OPD 的对比

### 共同点

```text
1. 都是 on-policy distillation。
2. 都让 student 自己生成 trajectory。
3. teacher 都对 student trajectory 提供 token-level 信号。
4. 都比单纯 RL 的 sparse reward 更 dense。
5. 都能用于小模型吸收强 teacher 的能力。
```

### 关键差异

```text
GOLD:
  目标是跨 tokenizer / 跨模型族 logit distillation。
  teacher 监督是 full vocabulary distribution。
  核心难点是 sequence alignment 和 vocabulary alignment。

Uni-OPD:
  目标是让 OPD token reward 与 outcome correctness 一致。
  teacher 监督主要是 sampled token 的 teacher_logprob - student_logprob。
  核心难点是 teacher token preference 可能和最终答案正确性冲突。
```

换句话说：

```text
GOLD 解决“teacher 和 student 怎么对齐”；
Uni-OPD 解决“对齐后的 teacher 信号是否可信”。
```

### Loss 形式差别

GOLD 更像：

```text
L = divergence(P_teacher_aligned_vocab, P_student_aligned_vocab)
```

Uni-OPD 更像：

```text
adv_t = log pi_T(a_t | s_t) - log pi_S(a_t | s_t)
pg_loss = PPO / GRPO style policy loss with adv_t
```

其中 Uni-OPD 还会做：

```text
1. outcome-guided margin shift
2. greedy margin mask
3. online correctness balancing
4. teacher-student logprob clip
```

而 GOLD 的重点是：

```text
1. UTF-8 byte-offset sequence alignment
2. matched-token GKD loss
3. unmatched-token ULD fallback
4. lambda 控制 on-policy/off-policy mixture
```

## 和 Nemotron-Cascade 2 MOPD 的对比

Nemotron-Cascade 2 的 MOPD：

```text
1. 使用多个 domain teacher。
2. 针对 Cascade RL 后不同能力域做能力再平衡。
3. 仍然更接近 sampled-token OPD / token logprob advantage。
```

GOLD：

```text
1. 主要解决跨 tokenizer。
2. 不是多 domain teacher selection。
3. 核心是 distribution alignment。
```

如果要结合二者，合理方向是：

```text
Multi-domain GOLD:
  对每个 domain 选择 teacher；
  如果 teacher / student tokenizer 不同，就用 GOLD 对齐；
  再做 domain-level on-policy distillation。
```

## 和本仓库 Uni-OPD 代码的关系

本仓库当前 OPD 相关实现主要假设：

```text
teacher_log_probs:
  teacher 对 student sampled tokens 的 logprob。

student_log_probs:
  student 对同一 sampled tokens 的 logprob。
```

也就是它不需要 full vocabulary logits 对齐，只需要：

```text
同一 token 序列上的 teacher sampled-token logprob。
```

如果 teacher 和 student tokenizer 不同，本仓库这套 sampled-token OPD 会遇到问题：

```text
student 生成的 token id 对 teacher tokenizer 没有直接对应关系。
teacher 无法直接给每个 student token 一个同 vocab 下的 logprob。
```

要把 GOLD 思路接入 Uni-OPD，至少需要：

```text
1. 保存 student response 的原始文本。
2. teacher 用自己的 tokenizer 重新 tokenize。
3. student / teacher token 序列按 UTF-8 byte span 对齐。
4. 对 aligned group 计算 group-level teacher logprob 和 student logprob。
5. 再构造 OPD advantage 或 full-logit distillation loss。
```

如果继续走 Uni-OPD 的 margin calibration，还需要：

```text
6. 将 group-level OPD return 聚合到 trajectory-level G_OPD。
7. 用 outcome reward 做 margin shift / margin mask。
```

也就是说：

```text
GOLD 可以补 Uni-OPD 的跨 tokenizer 对齐能力；
Uni-OPD 可以补 GOLD 的 outcome-guided reliability calibration。
```

## 方法局限

### 1. 需要 teacher full logits

GOLD 是 logit distillation，因此比 sampled-token OPD 更重：

```text
teacher 需要输出 full vocabulary distribution。
```

如果 teacher 是黑盒 API，只返回 sampled token logprob 或不返回 logits，GOLD 很难直接用。

### 2. 跨 tokenizer 对齐有噪声

即使使用 byte span 对齐，vocabulary alignment 仍然不是完美的：

```text
matched tokens:
  相对可靠。

unmatched tokens:
  仍然依赖 ULD 的排序/padding 近似。
```

Gemma 和 Qwen 的 similarity 只有 0.063 时，效果明显很差。

### 3. 显存和计算开销更高

full-logit distillation 通常需要：

```text
teacher forward logits
student forward logits
vocab-level divergence
alignment buffer
```

对于大 vocab / 长序列 / 大 teacher，成本明显高于 sampled-token logprob difference。

### 4. 对 teacher 上限敏感

文中也强调 student 性能受 teacher 能力上限影响。teacher 在 Countdown 上越强，student 最终效果越好。

## 复现要点

### Same-tokenizer GKD

关键设置：

```text
teacher:
  Qwen/Qwen3-4B-Instruct-2507

student:
  Qwen/Qwen2.5-1.5B-Instruct

lambda:
  越接近 1 越 on-policy

beta:
  0.0，即 forward KL setting

student generation temperature:
  1.0
```

### Cross-tokenizer GOLD

关键设置：

```text
student:
  meta-llama/Llama-3.2-1B-Instruct

teacher:
  Qwen/Qwen3-4B-Instruct-2507

sequence alignment:
  byte-offset / token merge

vocab alignment:
  matched-token GKD + unmatched-token ULD
```

### Domain distillation recipe

文中给出的 TRL recipe 大意：

```text
1. 用 SFTTrainer 在 Codeforces 数据上微调 Qwen3-4B。
2. 用 GOLD / GKD 在 Tulu-3-SFT-Mixture 上蒸馏原始 Qwen3-4B teacher。
3. 目标是恢复 IFEval，同时尽量保留 LiveCodeBench 收益。
```

关键参数示例：

```text
lmbda:
  0.25

beta:
  0.0

use_uld_loss:
  true

use_extended_uld:
  true

uld_use_hybrid_loss:
  true

uld_distillation_weight:
  1.0

teacher_model:
  Qwen/Qwen3-4B-Instruct-2507
```

## 最终理解

GOLD 的核心贡献可以压缩成两句话：

```text
1. 把 ULD 从 offline distillation 扩展到 on-policy distillation。
2. 用更可靠的序列对齐和词表对齐，让不同 tokenizer 的 teacher/student 也能做 logit-level KD。
```

它不是在解决 outcome reward 和 teacher token preference 的冲突，而是在解决：

```text
不同模型族之间，teacher 的 token distribution 如何投影到 student 可学习的坐标系里。
```

因此，放到我们之前看的几篇论文谱系里：

```text
GKD:
  same-tokenizer on/off-policy distillation 框架。
  论文: arXiv:2306.13649

ULD:
  cross-tokenizer offline logit distillation。
  论文: arXiv:2402.12030

GOLD:
  cross-tokenizer on-policy logit distillation。
  文章: Hugging Face H4 research article

Uni-OPD:
  outcome-calibrated sampled-token OPD。

Nemotron MOPD:
  multi-domain teacher pool + on-policy distillation for cascade post-training。
```
