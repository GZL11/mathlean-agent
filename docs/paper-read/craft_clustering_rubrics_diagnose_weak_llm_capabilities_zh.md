# CRAFT：用 rubric 聚类诊断 LLM 弱能力，并生成定向微调数据

本文档解读论文：

```text
CRAFT: Clustering Rubrics to Diagnose Weak LLM Capabilities and Generate Targeted Fine-Tuning Data
https://arxiv.org/abs/2607.16122
```

## 一句话总结

这篇论文不是在做新的评测指标，而是在做一个 **“评测 -> 诊断 -> 造数据”** 的闭环。

它的核心思路是：

```text
把任何 rubric-based evaluation 数据集，
变成针对某个模型的能力诊断图谱，
再据此生成更有针对性的 fine-tuning 数据。
```

## 论文想解决什么

普通评测通常只能告诉你：

```text
模型在这些样本上失败了
```

但很少告诉你：

```text
它到底是“推理能力弱”“格式理解弱”“知识覆盖弱”，还是别的能力弱。
```

CRAFT 的判断是：

```text
评测不该只指出失败位置，还应该指出失败原因。
```

## 核心对象

论文输入的是一个 rubric-based evaluation dataset。

这里的 rubric 可以理解成：

```text
一组评分标准 / 评测准则 / 评测条目
```

比如一个样本会有：

```text
prompt + rubric criteria + target model answer + judge score
```

CRAFT 把每个 rubric criterion 当成一个 **capability probe**：

```text
“这个评分条目到底在测模型的哪种能力？”
```

## 方法主线

### 1. 从 prompt-rubric 对里抽能力描述

对每个 `prompt + rubric` 组合，CRAFT 会抽取一个 capability description。

直观上就是：

```text
这个 rubric criterion 具体在测什么能力？
```

比如可能被映射成：

```text
事实核对
多步推理
格式遵循
局部证据提取
```

### 2. 把能力描述聚成层次树

接下来不是简单按标签平铺，而是把这些能力描述做层次聚类，形成一个 **hierarchical capability tree**。

这样就能把能力组织成：

```text
总能力
  ├─ 子能力 A
  │   ├─ 更细的失败模式 A1
  │   └─ 更细的失败模式 A2
  └─ 子能力 B
      ├─ 更细的失败模式 B1
      └─ 更细的失败模式 B2
```

这样做的意义是：

```text
同一个“失败”，可以在不同粒度上被解释。
```

### 3. 在树的每个节点上给目标模型打分

CRAFT 会在树的每一层、每个节点上评估目标模型的表现。

这样就不只是得到“整体差”，而是得到：

```text
哪一类能力弱
弱到什么层级
```

### 4. 动态挑出低分节点

论文的关键是动态选择：

```text
在树的不同层上，选那些表现最差、且失败最清晰的节点。
```

这比只看顶层弱点更有用，因为：

```text
顶层太粗
叶子太碎
中间层往往最适合生成可操作的训练数据
```

## 为什么要做树

如果只做平铺分类，容易出现两个问题：

```text
1. 诊断太粗，看不出为什么差
2. 诊断太细，难以组织成可训练的数据计划
```

树结构的好处是：

```text
1. 可以从粗到细定位弱点
2. 可以按层级决定先修哪一块
3. 可以把诊断结果直接映射到 targeted data generation
```

## 训练数据怎么生成

CRAFT 不是只停在诊断层，而是把诊断结果接到后续数据生成。

逻辑是：

```text
1. 找到弱能力节点
2. 围绕这个节点构造更有针对性的训练样本
3. 用这些样本做 fine-tuning
```

所以它本质上是一个：

```text
diagnosis-guided data generation pipeline
```

## 数据构造细节

CRAFT 里的数据构造要分成两类：

```text
1. 诊断数据：用来发现模型弱点
2. 微调数据：根据弱点重新生成，用来 SFT
```

### 1. 诊断数据来自 PRBench 的 rubric 数据

论文实验使用 PRBench 的两个专业领域子集：

```text
Finance:
  629 prompts
  10,806 rubric criteria

Legal:
  532 prompts
  9,637 rubric criteria
```

这些数据只用于：

```text
1. 构建 capability tree
2. 对目标模型做 rubric-level scoring
3. 选择 weak capability nodes
```

它们不是最终报告指标的 held-out benchmark，也不是直接拿来做 SFT 的训练集。

### 2. prompt-rubric pair 是最小诊断单位

一个 prompt 可能有多个 rubric criterion。CRAFT 会把它展开成多条：

```text
(prompt, rubric criterion)
```

每一条都被视为一个 capability probe。

例如一个金融问题可能有三个 criterion：

```text
criterion 1: 是否识别利率上升会提高融资成本
criterion 2: 是否区分 maintenance capex 和 growth capex
criterion 3: 是否说明 payout policy 的约束
```

那么它会变成三个独立的诊断叶子：

```text
(prompt, criterion 1)
(prompt, criterion 2)
(prompt, criterion 3)
```

所以 CRAFT 的树不是按“题目”建的，而是按“题目里的评分要求”建的。

### 3. 从每个 prompt-rubric pair 抽 capability description

对每个 pair，作者用一个 LLM 生成短的、去上下文化的能力描述：

```text
input:
  prompt + rubric criterion

output:
  capability description
```

例子：

```text
prompt:
  How would a 250 bps rate shock affect capex and payout policy...

rubric criterion:
  Higher policy rates raise financing costs, WACC, or hurdle rates.

capability description:
  Analyzing policy transmission
```

这个 capability description 才是后面聚类的文本对象。

### 4. 用 embedding + LLM 层次聚类建树

CRAFT 先把 capability description 做 embedding，再用 k-means 形成最细粒度的 base clusters。

之后逐层往上建父节点：

```text
1. node summarization:
   LLM 读取一组相近 child clusters，提出更泛化的 parent capability labels

2. candidate consolidation:
   LLM 合并重复、重叠、过细的 parent candidates

3. cluster routing:
   LLM 把每个 child cluster 分配给最合适的 parent

4. parent refinement:
   LLM 根据最终 children 重新描述 parent node
```

如果原数据有 metadata，例如 domain 或 rubric subcategory，论文会把它们固定成树的顶部层级：

```text
L0: domain
L1: rubric subcategory
L2-L5: learned capability hierarchy
```

实验里最终得到的是：

```text
6-level tree
20,443 prompt-rubric pairs
```

使用的模型 / 工具：

```text
Gemini 2.5 Pro:
  node summarization
  candidate consolidation
  cluster routing
  parent refinement

Gemini 2.5 Flash:
  routing stability checks

text-embedding-3-small:
  capability embedding
```

### 5. 对目标模型打分，选择 weak nodes

对每个目标模型，CRAFT 会先让它回答诊断 prompt，然后用 LLM-as-a-judge 按 rubric criterion 打 pass/fail。

于是每个叶子都有一个二值结果：

```text
pass = 1
fail = 0
```

内部节点的分数就是该子树里所有 prompt-rubric pairs 的平均通过率：

```text
node pass rate = mean(pass/fail over descendant criteria)
```

选择 weak node 时有几个规则：

```text
1. 节点必须有足够 support
   主实验里要求 > 30 个 rubric criteria

2. 从严格 pass-rate threshold 开始找低分节点
   如果数量不够，就逐步放宽阈值

3. 一旦某个节点被选中
   它的 ancestors 和 descendants 都会被排除
```

这样可以避免同时选到：

```text
一个大类弱点
和它下面的子类弱点
```

最终输出是一组互不重叠的 weak capability nodes。

### 6. 每个 weak node 生成 40 个 prompts

进入微调数据生成阶段后，CRAFT 对每个选中的 weak node 生成：

```text
40 prompts
```

生成 prompt 里包含：

```text
1. 当前 weak node 的 capability description
2. 该 node 中的示例 prompt-rubric pairs
3. tree context，例如 parent descriptions 和 sibling descriptions
```

这里 parent / sibling context 很重要，因为它告诉生成器：

```text
要围绕当前弱能力生成，
但不要和相邻能力混在一起。
```

### 7. 轻量过滤

生成 prompt 后，作者做轻量过滤：

```text
1. 用 word-overlap Jaccard similarity 去近重复
2. 对重复 prompt pattern 做 diversity constraints
```

也就是过滤掉：

```text
1. 文字高度重合的样本
2. 模板痕迹太强、模式重复的样本
```

### 8. teacher 生成 response，得到 SFT 样本

最后用 teacher model 给这些 prompts 生成 responses。

最终训练样本是普通 SFT 格式：

```text
(synthetic prompt, teacher response)
```

所有方法共享同一个下游设置：

```text
1,000-example target budget
same teacher response-style recipe
same SFT hyperparameters
same evaluation protocol
```

所以 CRAFT、EvalTree、Random 的区别不在于：

```text
teacher 更强
训练步数更多
数据更多
```

而在于：

```text
这 1,000 条 synthetic SFT 数据到底围绕什么目标生成。
```

CRAFT 是围绕 rubric-level weak capability nodes 生成；
EvalTree 是围绕 prompt-level weak clusters 生成；
Random 是从领域池里随机采样生成。

## 这篇论文的本质

可以把它理解成三步：

```text
评测 rubric
  -> 能力树
  -> 定向数据
```

它不是单纯在做 benchmark，也不是单纯在做合成数据，而是：

```text
用评测结构本身来驱动后训练数据生成。
```

## 和普通评测的区别

普通评测告诉你：

```text
模型在任务 X 上分数低
```

CRAFT 想告诉你：

```text
模型在能力 A / A1 / A2 上弱，
所以应该优先补这类数据。
```

这比“找低分样本再人工看”更系统。

## 和后训练的关系

这篇很适合放在后训练链路里看：

```text
1. 先评测
2. 再诊断弱能力
3. 再生成 targeted fine-tuning data
4. 再回训模型
```

所以它不是 evaluation-only，而是：

```text
evaluation for training.
```

## 一句话结论

CRAFT 的核心贡献是：

```text
把 rubric-based evaluation 变成 hierarchical capability diagnosis，
再把诊断结果变成定向微调数据生成的依据。
```
