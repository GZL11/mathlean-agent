# Uni-OPD 论文技术解读

本文档从工程实现角度解读论文 **Uni-OPD: Unifying On-Policy Distillation with a Dual-Perspective Recipe**，重点说明它要解决什么问题、核心方法如何工作、训练链路如何落到本仓库代码中，以及复现时需要注意哪些细节。

## Student / Teacher 模型速览

Uni-OPD 不是单一 teacher-student 配置，而是在文本、多教师和多模态蒸馏里都做了验证。最重要的是先看 teacher：

```text
文本单教师 / 多教师:
  Student:
    Qwen3-4B
    Qwen3-1.7B

  Teacher:
    Qwen3-4B-Math-RL
    Qwen3-4B-Code-RL

  含义:
    teacher 是 Qwen3-4B backbone 经过数学 / 代码领域 RL 得到的专家模型。

大文本 teacher:
  Student:
    Qwen3-4B
    Qwen3-1.7B

  Teacher:
    Qwen3-30B-A3B-Instruct-2507

视觉语言多教师:
  Student:
    Qwen3-VL-4B-Instruct
    Qwen3-VL-2B-Instruct

  Teacher:
    Qwen3-VL-4B-Instruct-Math-RL
    Qwen3-VL-4B-Instruct-Logic-RL
    Qwen3-VL-4B-Instruct-Document-RL
    Qwen3-VL-4B-Instruct-Code-RL
```

所以 Uni-OPD 的 teacher 主要是 **RL 后的领域专家 teacher**，论文核心就是用 outcome reward 校准这些 teacher 的 token 级监督信号。

## 一句话总结

Uni-OPD 的本质是：用 **outcome reward** 作为轨迹级锚点，校准 **teacher token-level supervision** 的优化方向。

标准 OPD 直接使用：

```text
teacher_logprob - student_logprob
```

作为 token 级训练信号。但 teacher 的 token-level 偏好不一定和最终答案正确性一致。Uni-OPD 将两类信号结合：

```text
teacher logprob:
  提供细粒度 token-level credit assignment

outcome reward:
  提供粗粒度但更可靠的 trajectory-level ordering constraint
```

当 teacher token reward 聚合后符合 outcome 顺序时不处理；当它把错误轨迹排到正确轨迹前面时，通过 margin shift 或 greedy mask 校准。

## 背景：OPD 在做什么

On-Policy Distillation 中，学生模型先基于当前策略生成轨迹，然后 teacher 对这些学生生成的轨迹进行 token-level 打分。对于某条 response token `o_t`，OPD token reward 是：

```text
r_t^OPD = log pi_T(o_t | q, o_<t) - log pi_theta(o_t | q, o_<t)
```

其中：

```text
pi_T:
  teacher model

pi_theta:
  student / current policy model

q:
  prompt

o_<t:
  response prefix
```

如果 `r_t^OPD > 0`，说明 teacher 比 student 更支持这个 token，训练会鼓励学生提高该 token 概率。如果 `r_t^OPD < 0`，则会压低该 token 概率。

代码位置：

- `miles/miles/backends/training_utils/loss.py`
  - `compute_advantages_and_returns`
  - `adv = teacher_logprob - student_logprob`

## 核心问题

Uni-OPD 认为标准 OPD 有两个主要瓶颈。

### 1. 学生探索不足

如果 prompt 太简单，学生 rollout 可能全对；如果太难，学生 rollout 可能全错。这两种情况都缺少足够有信息量的状态和正负对比。

论文提出两种数据均衡：

```text
offline difficulty-aware data balancing:
  训练前重塑 prompt 难度分布

online correctness-aware data balancing:
  训练时平衡 correct / incorrect rollout 组成
```

### 2. teacher supervision 不可靠

teacher 的 token-level logprob 不一定等价于最终答案正确性。

可能出现：

```text
错误轨迹:
  写法流畅、格式像 teacher，G_OPD 高

正确轨迹:
  解法不那么 teacher-like，但答案正确，G_OPD 低
```

如果直接训练，学生会被推向错误轨迹。这就是论文里的 order inconsistency。

## Student / Teacher 模型设置

Uni-OPD 不是单一 teacher-student 配置，而是在多个蒸馏场景中验证方法。核心模型族是 Qwen3 / Qwen3-VL。

### LLM 单教师 / 多教师蒸馏

文本任务覆盖数学和代码。

```text
Student:
  Qwen3-4B
  Qwen3-1.7B

Teacher:
  Qwen3-4B-Math-RL
  Qwen3-4B-Code-RL
```

这些 teacher 是从同一个 Qwen3-4B backbone 经过 domain-specific RL 得到的专家模型。

单教师设置下，学生分别从单个 math/code teacher 蒸馏；多教师设置下，数学数据路由到 math teacher，代码数据路由到 code teacher。

仓库对应配置：

```text
exps/scripts/OPD/multi_teacher/
exps/scripts/OPD/multi_teacher/configs/
```

### Strong-to-Weak 蒸馏

该设置用于验证更强 teacher 到更小 student 的迁移。

```text
Student:
  Qwen3-4B
  Qwen3-1.7B

Teacher:
  Qwen3-30B-A3B-Instruct-2507
```

仓库对应配置：

```text
exps/scripts/OPD/strong_to_weak/
exps/scripts/OPD/strong_to_weak/configs/
```

### MLLM 蒸馏

多模态任务覆盖数学、逻辑和文档理解。

```text
Student:
  Qwen3-VL-4B-Instruct
  Qwen3-VL-2B-Instruct

Teacher:
  Qwen3-VL-4B-Instruct-Math-RL
  Qwen3-VL-4B-Instruct-Logic-RL
  Qwen3-VL-4B-Instruct-Document-RL
```

这些 teacher 同样是通过 domain-specific RL 得到的专家模型。

### Cross-Modal 蒸馏

该设置验证文本能力和多模态能力能否统一到一个学生策略中。

```text
Student:
  Qwen3-VL-4B-Instruct

Teacher:
  Qwen3-VL-4B-Instruct-Code-RL
  Qwen3-VL-4B-Instruct-Math-RL
  Qwen3-VL-4B-Instruct-Logic-RL（附录扩展设置）
```

### Teacher 路由

代码里每条训练样本可带 `teacher_model_name`，训练时根据该字段路由到对应 teacher server。

相关文件：

```text
miles/miles/utils/data.py
miles/Uni_OPD_utils/OPD_reward/reward_manager.py
miles/Uni_OPD_utils/OPD_reward/teacher_server_map.json
miles/Uni_OPD_utils/OPD_reward/teacher_server_list.json
```

## 离线难度感知数据均衡

离线阶段先用学生模型对整个训练集做一次 rollout。

论文附录描述的过程：

```text
1. 对每个 prompt 用 student 采样 N=8 条 response
2. 用任务 verifier 判断每条 response 是否正确
3. 统计正确数 k
4. 用 k / 8 作为 prompt 难度 proxy
```

难度桶：

```text
k = 0:
  全错，最难

k = 8:
  全对，最易

1 <= k <= 7:
  中等难度，有对有错
```

论文区分两种分布：

```text
U-shaped:
  两端多，中间少
  上采样 1-7 correct out of 8 的中等难度样本

mirrored-J-shaped:
  附录文字描述为 easy instances dominate
  上采样 1-8 correct out of 8 的 non-trivial 样本
```

目标是 flatten difficulty distribution，而不是过滤掉全对或全错样本。

需要注意：论文和当前仓库都没有公开精确的离线上采样实现细节，例如：

```text
每个难度桶最终采到多少
target 是最大桶、均值、分位数还是手动指定
是否有 replacement
是否限制最大重复次数
如何自动判定 U-shaped / mirrored-J-shaped
```

当前仓库训练配置直接消费已生成的 `offline_data_balanced.jsonl`。

代码相关：

- `miles/miles/utils/data.py`
  - 支持从 YAML 读取多个数据源
  - 支持给样本绑定 `teacher_model_name`
- `exps/scripts/OPD/**/configs/*.yaml`
  - 指向 `offline_data_balanced.jsonl`

## 在线正确性感知数据均衡

在线均衡发生在 rollout 之后、训练之前。

每个 prompt 会采样多条 response，每条 response 通过 outcome verifier 得到：

```text
response_correct = True / False
```

如果一个 batch 中 correct / incorrect 严重失衡，margin calibration 很难工作。例如：

```text
全对:
  没有错误轨迹，S_-(q) 为空

全错:
  没有正确轨迹，S_+(q) 为空
```

这时 prompt-level margin 没法定义。即使不是全空，极端不平衡也会让多数类轨迹主导梯度。

在线均衡的做法是：随机 mask 掉过多的一侧，使 correct / incorrect 接近目标比例，默认约为 `1:1`。

代码位置：

- `miles/Uni_OPD_utils/online_balance/rollout_sample_balance.py`

核心行为：

```text
correct 过多:
  随机 mask 一部分 correct samples

incorrect 过多:
  随机 mask 一部分 incorrect samples

mask 方式:
  sample.remove_sample = True
```

随后在：

- `miles/miles/ray/rollout.py`

被 mask 的样本会变成：

```text
loss_mask = [0] * response_length
```

也就是这些 token 不参与 loss。

这确实会丢弃一部分具备梯度信号的轨迹。它的取舍是：牺牲一部分样本利用率，换取 batch 内更稳定的正负对比结构和更可靠的优化方向。

## 轨迹级 OPD Return

论文定义：

```text
G_OPD(q, tau)
  = 1 / |tau| * sum_t log(pi_T(o_t | q, o_<t) / pi_theta(o_t | q, o_<t))
  = mean_t r_t^OPD
```

代码中没有始终显式保存一个叫 `G_OPD` 的标量，而是先保存 token-level advantage：

```text
advantages[i].shape = [R_i]
advantages[i][t] = teacher_logprob[t] - student_logprob[t]
```

在 margin calibration 中临时聚合：

```python
mean_advs = torch.stack([adv.mean() for adv in advantages])
```

其中：

```text
mean_advs[i] = G_OPD(q, tau_i)
```

代码位置：

- `miles/miles/backends/training_utils/loss.py`
- `miles/Uni_OPD_utils/margin_calibration/margin_shift.py`

## Outcome-Guided Margin Calibration

这是论文最核心的 teacher-perspective 方法。

对于同一个 prompt `q` 的 rollout group，按 outcome reward 分成：

```text
S_+(q):
  正确轨迹集合

S_-(q):
  错误轨迹集合
```

prompt-level margin 定义为：

```text
m(q) = min_{tau in S_+(q)} G_OPD(q, tau)
       - max_{tau in S_-(q)} G_OPD(q, tau)
```

如果 `m(q) >= 0`，说明最差的正确轨迹仍高于最好的错误轨迹，teacher token reward 的轨迹级排序和 outcome reward 一致。

如果 `m(q) < 0`，说明 teacher token reward 聚合后更偏向错误轨迹，需要校准。

代码支持两种 mode：

```text
minmax:
  correct_stat = min(G_OPD of correct)
  incorrect_stat = max(G_OPD of incorrect)

mean:
  correct_stat = mean(G_OPD of correct)
  incorrect_stat = mean(G_OPD of incorrect)
```

对应 gap：

```text
gap = correct_stat - incorrect_stat
```

目标：

```text
gap >= delta
```

## Margin Shift

Margin Shift 不删除样本，而是平移轨迹上的 token-level advantage。

如果：

```text
gap < delta
```

计算：

```text
shift = delta - gap
      = incorrect_stat - correct_stat + delta
```

然后按方向处理：

```text
correct_up / Lift:
  正确轨迹所有 token advantage += shift

incorrect_down / Suppress:
  错误轨迹所有 token advantage -= shift

both / Spread:
  正确轨迹所有 token advantage += shift / 2
  错误轨迹所有 token advantage -= shift / 2
```

这样对应轨迹的 `G_OPD = mean(adv)` 也会整体上移或下移，从而恢复正确轨迹高于错误轨迹的顺序约束。

代码位置：

- `miles/Uni_OPD_utils/margin_calibration/margin_shift.py`

代码支持两种 scope：

```text
global:
  整个 batch 内所有正确/错误 response 放在一起比较

group:
  按 sample_group_index 分组，只在同一个 prompt 的 N 条 rollout 内比较
```

论文语义更接近 group-level，因为 prompt-level margin 是对同一道题定义的。

## Greedy Margin Mask

Greedy Margin Mask 是 Margin Shift 的替代方案，不是通常一起使用的后处理步骤。

区别：

```text
Margin Shift:
  保留样本
  修改 advantage 数值

Greedy Margin Mask:
  不平移剩余样本的 advantage
  贪心 mask 掉破坏 margin 的轨迹
```

如果已经经过 `use_opd_margin_shift`，理论上对应 scope/mode 下应满足 margin，因此再做 greedy mask 通常没有必要。脚本中也有互斥检查。

代码位置：

- `miles/Uni_OPD_utils/margin_calibration/margin_mask_greedy.py`

## 从 Advantage 到 PG Loss

`compute_advantages_and_returns()` 产生：

```text
batch["advantages"]
```

随后 `policy_loss_function()` 中重新 forward 当前 actor，得到当前 logprob：

```text
log_probs = log pi_current
old_log_probs = batch["log_probs"] 或 batch["rollout_log_probs"]
```

PPO ratio：

```text
ratio = exp(log pi_current - log pi_old)
```

代码中：

```python
ppo_kl = old_log_probs - log_probs
ratio = exp(-ppo_kl)
```

policy loss：

```text
pg_loss = - min(ratio * A, clipped_ratio * A)
```

代码中用负号形式实现为：

```python
pg_losses1 = -ratio * advantages
pg_losses2 = -clip(ratio) * advantages
pg_loss = max(pg_losses1, pg_losses2)
```

代码位置：

- `miles/miles/backends/training_utils/loss.py`
  - `policy_loss_function`
- `miles/miles/utils/ppo_utils.py`
  - `compute_policy_loss`

## Token-Level 过滤与训练稳定技巧

除了论文核心 margin calibration，代码还有一些训练稳定性处理。

### Teacher 失败 token mask

teacher server 请求失败或 logprob 解析失败时，teacher logprob 会被填成：

```text
TEACHER_LOGP_FAILED_SENTINEL = -100.0
```

随后：

```python
teacher_valid_mask = (t_logps != -100).float()
adv = (t_logps - s_logps) * teacher_valid_mask
```

失败 token 的 advantage 变成 0，不参与梯度。

### Teacher-student logprob clip

如果启用：

```bash
--use-teacher-student-logprob-clip
```

代码会过滤 teacher 和 student logprob 差异过大的 token：

```python
opd_clip_mask = (abs(log_pi_student - log_pi_teacher) < threshold)
pg_loss = pg_loss * opd_clip_mask
```

这属于 token-level clip，不是 trajectory-level calibration。

### TIS / ICEPOP

`rollout_log_probs` 来自 rollout engine，例如 SGLang/vLLM 生成时记录的 token logprob。

`train_log_probs` 是当前训练 actor 对同一批 rollout trajectory 重新 forward 得到的 logprob。

TIS 处理的是训推不一致：

```text
ratio = pi_train / pi_rollout
      = exp(train_logprob - rollout_logprob)
```

`vanilla_tis_function` 会 clip ratio 后继续使用 token。

`icepop_function` 则更激进：ratio 超出范围直接把 token 权重置 0。

这些不是 Uni-OPD 的核心创新，而是 Miles 训练框架中的 off-policy correction 工具。

## 训练链路总览

整体流程：

```text
1. 数据加载
   - 从 YAML / jsonl 读取 prompt
   - 每条样本可绑定 teacher_model_name

2. 学生 rollout
   - SGLang 生成 response
   - 保存 rollout tokens / rollout_log_probs

3. reward 获取
   - verifier 给 response_correct
   - teacher server 给 input_token_logprobs

4. reward post-process
   - 截取 response 部分 teacher_log_probs
   - 写入 sample.teacher_log_probs
   - 写入 sample.response_correct

5. 在线正确性均衡
   - 根据 response_correct mask 掉过多一侧

6. 计算 OPD advantage
   - adv = teacher_logprob - student_logprob

7. margin calibration
   - margin shift 或 greedy margin mask

8. policy loss
   - 用 PPO-style objective 从 adv 得到 pg_loss
   - 可选 teacher-student clip / TIS

9. 更新 student
```

## 关键代码索引

```text
数据读取:
  miles/miles/utils/data.py

Sample 数据结构:
  miles/miles/utils/types.py

rollout:
  miles/miles/rollout/sglang_rollout.py

teacher reward:
  miles/Uni_OPD_utils/OPD_reward/get_reward.py
  miles/Uni_OPD_utils/OPD_reward/reward_manager.py
  miles/Uni_OPD_utils/OPD_reward/post_process_rewards.py

在线均衡:
  miles/Uni_OPD_utils/online_balance/rollout_sample_balance.py

OPD advantage 与 policy loss:
  miles/miles/backends/training_utils/loss.py

PPO loss:
  miles/miles/utils/ppo_utils.py

Margin Shift:
  miles/Uni_OPD_utils/margin_calibration/margin_shift.py

Greedy Margin Mask:
  miles/Uni_OPD_utils/margin_calibration/margin_mask_greedy.py

实验脚本:
  exps/scripts/OPD/
```

## 复现注意点

1. 离线 difficulty balancing 的精确生成脚本未公开。

   当前仓库只引用 `offline_data_balanced.jsonl`，没有给出完整生成代码。

2. `group` scope 更贴近论文公式。

   `global` scope 可以用，但会跨 prompt 比较 `G_OPD`，语义更粗。

3. Online balance 会丢样本。

   它不是无代价增强，而是在样本利用率和正负对比稳定性之间做取舍。

4. Margin Shift 和 Greedy Mask 是替代方案。

   一般不要同时启用。

5. teacher-student logprob clip 是 token-level 稳定技巧。

   它和 margin calibration 不是一回事。

6. TIS / ICEPOP 处理的是 rollout policy 和 train policy 不一致。

   它们不是 Uni-OPD 的核心方法，但可能影响训练稳定性。

## 总结

Uni-OPD 可以理解为一个双视角 recipe：

```text
学生视角:
  离线难度均衡 + 在线正确性均衡
  目的：让学生持续探索有信息量的状态和轨迹

教师视角:
  outcome-guided margin calibration
  目的：用最终正确性校准 teacher token-level supervision 的轨迹级排序
```

最核心的思想是：

```text
teacher token reward 很细，但不一定方向可靠；
outcome reward 很粗，但最终方向可靠。

Uni-OPD 用 outcome reward 约束 teacher token reward 的轨迹级排序，
从而让 OPD 的 token-level 梯度更符合最终任务目标。
```
