# OPSD：Self-Distilled Reasoner 技术解读

本文档解读论文：

```text
Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models
https://arxiv.org/abs/2601.18734
```

## 一句话总结

OPSD 是一个 **不需要外部 teacher 的 on-policy self-distillation 方法**。

它让同一个模型扮演两个角色：

```text
student:
  只看问题 x，和测试时一致。

teacher:
  看问题 x + 参考解答 / verified reasoning trace y*。
```

student 先自己 rollout，然后 teacher 不重新生成答案，只沿着 student 的同一条轨迹做 forward，给逐 token 分布监督。

## 背景：为什么需要 OPSD

论文把常见 reasoning post-training 方法分成几类：

```text
SFT:
  用参考答案做监督，但训练在 off-policy trajectory 上，有 exposure bias。

RLVR / GRPO:
  student 自己采样，on-policy；
  但 reward 通常只有最终正确 / 错误，token-level 监督太稀疏。

外部 teacher OPD:
  student 自己采样，teacher 给 dense token-level supervision；
  但需要单独的大 teacher，成本高。
```

OPSD 想解决的是：

```text
能不能不用外部 teacher，
直接让同一个模型借助 ground-truth solution 做自己的 teacher？
```

论文的直觉是：

```text
生成正确解法很难；
但看到参考解法后，理解 / rationalize 它往往更容易。
```

## Student / Teacher 怎么构造

给定一个带参考解答的数据：

```text
(x, y*)
```

OPSD 构造两个分布：

```text
student policy:
  p_S(. | x)

teacher policy:
  p_T(. | x, y*)
```

注意：

```text
p_S 和 p_T 来自同一个 LLM；
参数共享，只是 conditioning context 不同。
```

teacher 不是更大的模型，它的优势来自 privileged information：

```text
参考答案
参考 CoT
verified reasoning trace
```

## 训练流程

OPSD 的每一步是：

```text
1. 采样问题 x；
2. student 只看 x，生成 on-policy response：
   y_hat ~ p_S(. | x)

3. 对 y_hat 的每个 token 位置 n，
   student 计算：
   p_S(. | x, y_hat_<n)

   teacher 计算：
   p_T(. | x, y*, y_hat_<n)

4. 对两个 next-token distribution 做 token-level divergence；
5. 只通过 student branch 反传，teacher branch stop-gradient。
```

关键边界：

```text
teacher 不生成另一条标准答案；
训练轨迹只由 student 生成一次；
teacher 只是拿着 privileged information 沿 student prefix 打分。
```

这就是 OPSD 的 on-policy 性质。

## 目标函数

对 student 生成的 response：

```text
y_hat = (y_hat_1, ..., y_hat_T)
```

OPSD 定义轨迹平均 token divergence：

```text
D(p_T || p_S)(y_hat | x)
= 1 / |y_hat| * sum_n
  D(
    p_T(. | x, y*, y_hat_<n)
    ||
    p_S(. | x, y_hat_<n)
  )
```

整体目标：

```text
L_OPSD(theta)
= E_(x,y*) E_{y_hat ~ p_S(.|x)}
  [ D(p_T || p_S)(y_hat | x) ]
```

论文里 `D` 可以是：

```text
forward KL
reverse KL
JSD_beta
```

实际重点是 full-vocabulary distribution matching，而不是只对 sampled token 做单点 reward。

## OPSD 是否还需要 PG Loss

不需要。OPSD 主版本不是：

```text
L = L_PG + alpha * L_OPSD
```

而是直接用 full-vocabulary distribution matching 作为训练目标：

```text
L = L_OPSD
```

这里的 `on-policy` 指：

```text
训练 prefix / trajectory 来自 student 当前 policy 自己生成。
```

它不等于：

```text
一定要使用 PPO / GRPO 风格的 policy-gradient loss。
```

论文也讨论过 sampled-token policy-gradient 形式。可以把 sampled token `y_n` 的 teacher-student logprob gap 写成：

```text
A_n = log p_T(y_n | x, y*, y_hat_<n)
    - log p_S(y_n | x, y_hat_<n)
```

然后用类似 policy gradient 的目标：

```text
L_PG-style = - A_n * log p_S(y_n | x, y_hat_<n)
```

但这是 sampled-token 近似 / 替代变体，不是 OPSD 最终主实验使用的主目标。论文的结论是：

```text
full-vocabulary logit distillation
  > sampled-token PG-style objective
```

原因很直接：

```text
1. sampled-token PG 只看 student 已经采出来的那个 token；
2. full-vocab KL 会同时告诉 student：
   哪些 token 应该升高，哪些 token 应该降低；
3. reasoning 任务里正确 token 很稀疏，full distribution signal 更稳定。
```

所以 OPSD 的关键边界是：

```text
on-policy rollout，
但不是 RLVR / GRPO / PPO。
```

## OPSD 默认用 FKL 还是 RKL

OPSD 主版本默认用：

```text
Forward KL:
  KL(p_T || p_S)
```

也就是：

```text
teacher-to-student KL
```

可以展开为：

```text
KL(p_T || p_S)
= sum_v p_T(v) * log( p_T(v) / p_S(v) )
```

这里的期望分布来自 teacher，所以它会重点要求：

```text
teacher 认为重要的 token，
student 也要给足概率。
```

这和很多 OPD 论文常用的 RKL 不一样。RKL 是：

```text
Reverse KL:
  KL(p_S || p_T)

= sum_v p_S(v) * log( p_S(v) / p_T(v) )
```

它的期望分布来自 student，所以更像：

```text
student 自己偏好的 token，
如果 teacher 不认可，就压下去。
```

## 为什么 OPSD 里 FKL 更好

OPD 常用 RKL，主要是因为很多 OPD 设置里只想在 student on-policy token 上训练：

```text
student rollout 出 token y_n；
teacher 在同一个 prefix 上给这个 token 的 logprob；
用 sampled-token 估计 KL / reward。
```

这时 RKL 很自然，因为采样分布来自 student：

```text
y_n ~ p_S
```

用 RKL 可以避免 teacher-forcing 到 teacher 自己会生成、但 student 当前根本不会访问的区域。它更 on-policy、更 mode-seeking，也更贴近 RL-style policy update。

但 OPSD 的情况不一样。

### 1. OPSD 的 teacher 不是外部大模型，而是 privileged self-teacher

OPSD teacher 是：

```text
p_T(. | x, y*)
```

它和 student 参数相同，只是多看了参考解答 / verified trace。也就是说 teacher 的优势不是：

```text
模型规模更大；
分布完全不同；
tokenizer / style 完全不同。
```

而是：

```text
同一个模型在多看 privileged information 后，
对下一步 token 的判断更有依据。
```

这种情况下，teacher distribution 的 full support 本身更值得保留。用 FKL 可以更完整地把 teacher 的解题分布迁移给 student。

### 2. FKL 更适合“教会 student 覆盖 teacher 认为可能的正确 token”

FKL 是 mode-covering：

```text
teacher 给高概率的多个 token，
student 都需要覆盖。
```

在 reasoning 中，这很重要。因为某一步正确推理可能有多个等价表达：

```text
"therefore"
"so"
"hence"
"we get"
```

或者多个合理中间变量写法。FKL 会鼓励 student 覆盖 teacher 认为合理的候选集合。

RKL 更 mode-seeking：

```text
student 倾向于选择 teacher 分布里的一个高概率 mode，
压掉其他 mode。
```

这在代码生成 / 精确格式 / 避免发散时可能有利，但在 OPSD 的 self-distillation 里，过早 mode-seeking 可能会丢掉 teacher 给出的丰富 reasoning 分布。

### 3. OPSD 要修正 student 错误 token，而不只是惩罚 sampled token

假设 student 在某个 prefix 下分布是：

```text
p_S("5") = 0.30
p_S("6") = 0.40
p_S("4") = 0.10
```

teacher 看到参考解答后：

```text
p_T("5") = 0.70
p_T("6") = 0.05
p_T("4") = 0.05
```

如果用 sampled-token RKL / PG，且 student 这次 sampled token 是 `"6"`，训练信号主要围绕 `"6"`：

```text
teacher 不认可 "6"，
所以压低 "6"。
```

但 FKL 会同时做两件事：

```text
提高 "5"；
压低 "6"；
并按 teacher 分布重新塑形整个 next-token distribution。
```

这就是 full-vocab FKL 在 OPSD 里更稳的原因。

### 4. OPSD 的目标更像 distillation，不是 RL policy improvement

普通 OPD 很多时候把 teacher logprob 变成 token reward，再走 policy-gradient 风格：

```text
teacher signal -> token reward / advantage -> PG loss
```

而 OPSD 更直接：

```text
teacher signal -> full distribution target -> KL distillation
```

所以它天然更适合用标准 teacher-to-student FKL。

## 为什么 OPD 常用 RKL，但 OPSD 用 FKL 不矛盾

两者的差异来自训练目标不同。

| 方法类型 | 轨迹来源 | 监督对象 | 常见 KL | 原因 |
|---|---|---|---|---|
| 普通 OPD / ReOPD / TurnOPD | student rollout | student 访问过的 token / prefix | RKL `KL(p_S || p_T)` | 更 on-policy，惩罚 student 自己偏离 teacher 的 mode |
| TrOPD outlier 区域 | student prefix + teacher top-k | teacher 认为可靠的 top-k token | FKL `KL(p_T || p_S)` | 避免 outlier sampled-token RKL 极端梯度 |
| OPSD | student rollout prefix | privileged self-teacher 的 full distribution | FKL `KL(p_T || p_S)` | teacher 与 student 同源，full distribution 更有教学价值 |

所以更准确的说法不是：

```text
OPD 一定用 RKL。
```

而是：

```text
当监督只围绕 student sampled token / student occupancy 时，RKL 更自然；
当 teacher full distribution 本身可靠且要被覆盖时，FKL 更自然。
```

## Student Action Token 是什么

在 OPSD / OPD 语境里，`action token` 可以翻成：

```text
动作 token
```

含义很具体：

```text
student rollout 时真实采样出来、写进 response 的那个 token。
```

例如 student 生成：

```text
2 + 3 = 6
```

则 token 序列里的 `6` 就是某一步的 student action token。训练时不是让 teacher 重新采样一个 token 来替换它，而是：

```text
在 student 已经生成的 prefix 上，
同时计算 student 和 teacher 对“下一个 token 分布”的看法。
```

因此 OPSD 的轨迹来自 student，监督分布来自 teacher。

## 一个具体计算例子

假设问题是：

```text
x = "2 + 3 等于多少？"
y* = "2 + 3 = 5，所以答案是 5。"
```

student 只看 `x`，采样出 response：

```text
y_hat = "答案是 6"
```

看最后一个 token 位置，prefix 是：

```text
x + "答案是"
```

student distribution 可能是：

```text
p_S("5" | x, "答案是") = 0.30
p_S("6" | x, "答案是") = 0.40
p_S("4" | x, "答案是") = 0.10
p_S("其他") = 0.20
```

teacher 看到 `x + y* + "答案是"`，因此更倾向于正确答案：

```text
p_T("5" | x, y*, "答案是") = 0.70
p_T("6" | x, y*, "答案是") = 0.05
p_T("4" | x, y*, "答案是") = 0.05
p_T("其他") = 0.20
```

如果用 forward KL：

```text
D_KL(p_T || p_S)
= sum_v p_T(v) * log(p_T(v) / p_S(v))
```

只看这个简化词表，贡献大致是：

```text
"5": 0.70 * log(0.70 / 0.30) =  0.593
"6": 0.05 * log(0.05 / 0.40) = -0.104
"4": 0.05 * log(0.05 / 0.10) = -0.035
"其他": 0.20 * log(0.20 / 0.20) = 0
合计约 0.454
```

反传时 student 会被推向：

```text
提高 "5" 的概率；
降低 "6" 的概率；
整体接近 teacher 的 next-token distribution。
```

注意这个例子里 student 最终生成了错误 token `6`，但 OPSD 不是只在 `6` 这个 sampled token 上给一个负分，而是在同一个 prefix 上对完整 next-token distribution 做蒸馏。

## 和普通 OPD 的区别

普通 OPD：

```text
student:
  p_S(. | x)

teacher:
  外部大模型 p_T(. | x)
```

OPSD：

```text
student:
  p_theta(. | x)

teacher:
  p_theta(. | x, y*)
```

所以 OPSD 的 teacher advantage 不是来自模型规模，而是来自：

```text
privileged context
```

## 和 RLVR / GRPO 的区别

GRPO 给一条 response 的所有 token 同一个 advantage：

```text
A_i = normalize(r_i)
```

如果一组 sample 全错或全对，advantage 可能消失。

OPSD 则每个 token 都有 dense distribution supervision：

```text
token n:
  teacher distribution vs student distribution
```

所以它比 RLVR 更 token-efficient。

## Per-token Pointwise Divergence Clipping

论文更新版里还加入了一个稳定化技巧：

```text
per-token pointwise KL clipping
```

动机是：

```text
一些 style token 的 KL 可能远高于数学相关 token，
导致训练信号被风格词主导。
```

做法是对 vocabulary-level divergence contribution 做截断：

```text
D_clip = sum_n sum_v min(ell_{n,v}, tau)
```

直觉：

```text
不要让少数高 KL 风格 token 吃掉全部训练信号。
```

这和后续 OPD 系列里反复出现的主题一致：**token-level teacher signal 并不天然可靠，需要稳定化处理。**

## 实验结论

论文主要在数学 reasoning benchmarks 上比较 OPSD、SFT、GRPO 和不同蒸馏目标。完整结果和消融可以分成五块看。

### 1. 主结果表

评测数据集：`AIME24 / AIME25 / HMMT25`

| Model | AIME24 | AIME25 | HMMT25 | Avg |
|---|---:|---:|---:|---:|
| Qwen3-8B Base | 75.8 | 65.6 | 43.9 | 61.8 |
| Qwen3-8B + SFT | 72.3 | 64.2 | 42.9 | 59.8 |
| Qwen3-8B + GRPO | 76.4 | 68.9 | 46.7 | 64.0 |
| Qwen3-8B + OPSD | 77.8 | 70.8 | 45.8 | 64.8 |
| Qwen3-4B Base | 74.9 | 66.4 | 42.2 | 61.2 |
| Qwen3-4B + SFT | 70.2 | 62.3 | 43.4 | 58.6 |
| Qwen3-4B + GRPO | 75.6 | 68.1 | 44.4 | 62.7 |
| Qwen3-4B + OPSD | 76.4 | 68.3 | 46.1 | 63.6 |
| Qwen3-1.7B Base | 51.5 | 36.7 | 23.1 | 37.1 |
| Qwen3-1.7B + SFT | 48.4 | 36.3 | 22.7 | 35.8 |
| Qwen3-1.7B + GRPO | 51.1 | 38.3 | 23.7 | 37.7 |
| Qwen3-1.7B + OPSD | 57.2 | 43.9 | 29.2 | 43.4 |

### 2. divergence objective 消融

论文原文这张表是一个**单点消融**：

```text
dataset:
  AIME25

model:
  Qwen3-1.7B

metric:
  Avg@12

training checkpoints:
  Base / Step 50 / Step 100
```

也就是说，它不是跨多个 benchmark 的汇总表，而是专门看 divergence objective 在同一设置下的训练曲线表现。

| Method | Base | Step 50 | Step 100 |
|---|---:|---:|---:|
| FKL `KL(p_T || p_S)` | 36.7 | 43.9 | 41.1 |
| RKL `KL(p_S || p_T)` | 36.7 | 37.5 | 35.0 |
| JSD (`beta = 0.5`) | 36.7 | 36.9 | 39.0 |

读法是：

```text
FKL 在 step 50 提升最大，step 100 仍高于 base；
RKL 和 JSD 只能带来很有限甚至负向的收益。
```

### 3. generation style / token category 统计

这张表不是最终 accuracy，而是不同 student/teacher generation style 下，各 token 类别的平均 KL。

| Student | Teacher | Qwen3-1.7B Style | Math | Other | Qwen3-4B Style | Math | Other | Qwen3-8B Style | Math | Other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TM-off | TM-off | 0.68 | 0.12 | 0.11 | 0.61 | 0.06 | 0.10 | 0.56 | 0.05 | 0.11 |
| TM-on | TM-off | 0.51 | 0.10 | 0.17 | 0.41 | 0.05 | 0.18 | 0.33 | 0.05 | 0.15 |
| TM-on | TM-on | 0.51 | 0.09 | 0.08 | 0.50 | 0.04 | 0.09 | 0.42 | 0.04 | 0.08 |
| TM-off | TM-on | 0.85 | 0.14 | 0.25 | 0.92 | 0.10 | 0.29 | 0.79 | 0.06 | 0.25 |

### 4. full-vocab vs sampled-token objective

| Variant | AIME25 | HMMT25 |
|---|---:|---:|
| Full-vocabulary logit distillation | 84.1 | 60.0 |
| Sampled-token distillation / PG-style | 82.1 | 57.3 |

### 5. training / eval 配置

`GRPO` 和 `OPSD` 训练配置：

| Parameter | GRPO | OPSD |
|---|---:|---:|
| Learning Rate | 5e-6 | 5e-6 |
| Effective Batch Size | 32 | 32 |
| LoRA Rank | 64 | 64 |
| LoRA Alpha | 128 | 128 |
| Max Completion Length | 16,000 | 1,024 |
| Number of Generations per Prompt | 8 | 1 |
| Sampling Temperature | 1.2 | 1.1 |
| KL Coefficient | 0.0 | - |
| Training Steps | 500 | 100 |

SFT 训练配置：

| Parameter | Value |
|---|---:|
| Learning Rate | 5e-6 |
| Effective Batch Size | 32 |
| LoRA Rank | 64 |
| LoRA Alpha | 128 |
| Max Sequence Length | 16,000 |
| Number of Training Step | 100 |

评测配置：

| Parameter | Value |
|---|---:|
| Max New Tokens | 38,912 |
| Thinking Mode | Enabled |
| Top-p | 0.95 |
| Top-k | -1 |
| Min-p | 0.0 |
| Presence Penalty | 0.0 |
| Samples per Prompt | 12 |
| Temperature | 1.0 |

### 6. 消融结论

核心结论：

```text
1. OPSD 比 off-policy distillation 更能缓解 exposure bias。
2. OPSD 比 RLVR / GRPO 更 token-efficient。
3. 不需要外部大 teacher，只需要训练集里有 reference solution / reasoning trace。
4. FKL 明显优于 RKL 和 JSD。
5. TM-off student / TM-on teacher 是最有效的 generation style。
6. pointwise KL clipping 能防止风格 token 主导训练。
7. 更长 generation length 不是单调更好，1024 和 4096 没有稳定赢家。
8. full-vocabulary logit distillation 明显优于 sampled-token PG-style objective。
```

更具体一点：

```text
1. Qwen3-1.7B 上 OPSD 相对 base 提升最大，说明小模型更吃 dense supervision。
2. Qwen3-8B / 4B 上 OPSD 也稳定优于 SFT，并接近或超过 GRPO。
3. GRPO 在该数据集上容易出现 reward diversity collapse，token budget 也更大。
4. OPSD 的收益主要来自 token-level teacher distribution，而不是单纯多采样。
```

项目页和论文报告都指向同一个结论：

```text
OPSD 在同训练 token / steps 下能显著优于 GRPO，
尤其在 Qwen3-1.7B 这类小模型上效果最明显。
```

### 7. 作者最终结论

论文自己的总结可以压成三句：

```text
1. 一个足够强的 reasoning LLM 可以借助 privileged answer information 教自己。
2. on-policy self-distillation 能同时兼顾 token efficiency 和性能。
3. 只要有高质量 reference solution / reasoning trace，就不一定需要外部大 teacher。
```

## 局限

OPSD 的关键风险是：

```text
teacher 的优势来自 privileged information，
但 student 测试时看不到这些信息。
```

如果 privileged information 是每个样本特定的参考解答，student 可能学到一种训练时依赖 PI 的行为模式。

后续 RLSD 和 Many Faces of OPD 这类工作就指出：

```text
OPSD 可能出现 privileged information leakage；
性能可能早期上升，后期退化。
```

所以 OPSD 是一篇经典起点论文，但后续工作通常会问：

```text
如何保留 OPSD 的 dense token signal，
同时避免 privileged information 泄漏？
```

## 后继工作 / 近亲变体

截至目前我没看到一个官方命名的 `OPSD-v2`。但 2026 年已经出现了一批很像“OPSD 后继版”的工作，思路大致分成三类。

### 1. 直接改 OPSD 的权重或方向

- [PW-OPSD: Position-Weighted On-Policy Self-Distillation for Reasoning](https://arxiv.org/abs/2605.21606)  
  最像“OPSD 直接升级版”的一篇。它保留 student rollout、privileged teacher pass、clipped forward-KL target，但认为 teacher token 的可靠性沿序列位置变化，于是用递增 position weight 替代 uniform weighting。

- [EGRSD: Respecting Self-Uncertainty in On-Policy Self-Distillation for Efficient LLM Reasoning](https://arxiv.org/abs/2605.13255)  
  它保留 OPSD 的 self-distillation 框架，但不再把所有 token 一视同仁，而是加入 teacher entropy gate，专门压低高不确定 token 的权重。

- [DASD: Tailoring Teaching to Aptitude](https://arxiv.org/abs/2605.22263)  
  进一步把“同向蒸馏”改成“方向自适应蒸馏”：高 entropy token 反而要偏离 teacher，以保留探索；低 entropy token 再贴近 teacher。

### 2. 说明 OPSD 在强 thinking model 上有失效风险

- [Rethinking On-Policy Self-Distillation for Thinking Models](https://arxiv.org/abs/2607.05184)  
  这是非常重要的反思论文。它表明 privileged-context OPSD 会让强 thinking model 在长 reasoning trace 上退化，尤其会压掉 self-correction、backtracking、hedging 这些有用行为。

这类结果说明：

```text
OPSD 不是“越强越好”的通用版本，
privileged context 过强反而可能伤害探索。
```

### 3. 把 OPSD 改造成更具体的场景方法

- [EDGE-OPD: Internalizing Privileged Context with Evidence Guided On-Policy Distillation](https://arxiv.org/abs/2605.23493)  
  它把 OPSD 改成“evidence-guided”版本，强调 guided rollout + evidence mask，解决 rare-token / identity 类任务里 naive OPSD 学不到目标行为的问题。

- [Self-Distilled Agentic Reinforcement Learning](https://arxiv.org/abs/2605.15155)  
  它把 OPSD 作为 gated auxiliary objective 放进多轮 agent RL 里，说明在长程交互场景里，OPSD 更适合做辅助，而不是主损失。

### 4. 旁支但不是 OPSD 继承线

- [On-Policy Self-Distillation for Reasoning Compression](https://arxiv.org/abs/2603.05433)  
  这篇更像 sibling method，不是 OPSD 后继。它用“be concise”指令让模型自己蒸馏自己的简洁行为，目标是压缩 reasoning，而不是 privileged-trace teacher。

## 我怎么判断谁更像 OPSD 的后继

如果按“和原始 OPSD 的技术距离”排：

```text
PW-OPSD / EGRSD / DASD
  > EDGE-OPD / SDAR
  > Rethinking OPSD for Thinking Models
```

其中最像“OPSD-v2”的，实际上是：

```text
PW-OPSD
```

因为它几乎不改主框架，只是把 uniform token weight 改成 position weight。

## 一句话判断

OPSD 的贡献是把 OPD 从“外部 teacher 蒸馏”推进到：

```text
同一个模型 + privileged context + on-policy rollout
```

它证明了 self-distillation 可以产生强 dense supervision，但也暴露了后续 OPD/OPSD 系列最核心的问题之一：

```text
privileged teacher signal 是否会泄漏到 student 的不可观测状态里？
```

## BibTeX

```bibtex
@misc{zhao2026selfdistilledreasoner,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Siyan Zhao and Zhihui Xie and Mengchen Liu and Jing Huang and Guan Pang and Feiyu Chen and Aditya Grover},
  year={2026},
  eprint={2601.18734},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2601.18734}
}
```
