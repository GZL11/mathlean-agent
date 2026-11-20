# StepOPSD：Step-Aware Online Preference Distillation for Agent RL 论文解读

论文：

```text
StepOPSD: Step-Aware Online Preference Distillation for Agent Reinforcement Learning
```

链接：

```text
https://arxiv.org/abs/2605.27140
https://arxiv.org/pdf/2605.27140
```

作者单位：

```text
Independent Researcher
Tencent
DeepWisdom
```

## 一句话总结

StepOPSD 解决的是 agent RL 中的 credit assignment mismatch：

```text
环境 reward 是 trajectory-level / sparse，
但真正导致成败的往往是某一个 action step。
```

它的核心不是让 student 模仿 teacher 的完整分布，也不是额外加一个普通 OPD loss，而是：

```text
把完成 rollout 后的轨迹切成 action-centered step spans，
用 hindsight-enriched teacher context 重新打分这些 span，
把 teacher-student logprob gap 转成 token/step 级 advantage multiplier，
再用这个 multiplier 重塑 GRPO advantage。
```

所以 StepOPSD 可以概括为：

```text
GRPO + post-rollout step-aware advantage shaping from hindsight teacher gaps
```

关键点：

```text
1. online rollout 本身不改；
2. teacher 不负责生成新轨迹，只负责 post-rollout rescoring；
3. 没有单独 distillation imitation objective；
4. RL advantage 的正负方向仍由环境 reward / GRPO 决定；
5. OPD gap 只决定每个 token/step 更新强弱。
```

## 术语表

| 英文术语 | 建议中文 | 含义 |
|---|---|---|
| StepOPSD | 步级在线偏好蒸馏 | 以 agent step 为单位做 post-rollout credit shaping |
| Online Preference Distillation | 在线偏好蒸馏 | 不做分布模仿，而是用 teacher-student gap 形成局部偏好权重 |
| action-centered step segment | 动作为中心的步骤片段 | 从轨迹中抽取由模型控制的 action / reasoning-action span |
| causal action span | 因果动作片段 | 真正会影响环境状态或检索结果的模型输出片段 |
| hindsight-privileged teacher | 后见特权教师 | teacher context 中额外加入同组成功轨迹作为 hindsight |
| peer-trajectory hindsight | 同组轨迹后见信息 | 从同一个 GRPO group 中找成功轨迹，给失败轨迹 teacher branch 参考 |
| stale reference policy | 滞后参考策略 | 用若干训练步之前的 policy 参数作为 teacher，避免 teacher 随 student 同步漂移 |
| local trust region | 局部信赖域 | 用 `alpha_clip` 限制每个 token 的 OPD 权重偏离 1 的幅度 |
| equal_step_mean_abs | 步级等均值绝对预算 | 每个 step 的平均绝对 shaping budget 相同，避免长 step 占更多权重 |
| trajectory-level reward | 轨迹级奖励 | 整条 rollout 结束后才给一个 reward |
| sign-preserving shaping | 保符号优势重塑 | 只改变更新幅度，不改变成功/失败方向 |

## 背景：为什么需要 step-aware

标准 GRPO 对一个 task 采样 `G` 条 trajectories，拿 terminal reward 算 group-relative advantage：

```text
A_seq = (R - mean(R_group)) / (std(R_group) + epsilon)
```

然后把同一个 `A_seq` 广播给整条 trajectory 的所有 response tokens。

问题是 agent trajectory 很异质：

```text
observation: 环境返回，模型不可控；
reasoning: 模型内部推理；
action/tool call: 真正改变环境或检索结果；
final answer: 最后输出。
```

如果一次失败只是因为某一步 action 错了，比如：

```text
ALFWorld: 忘了 heat / clean / cool；
Search-QA: 第一次 query 写错，导致后续证据全错；
```

GRPO 会惩罚整条轨迹，普通 OPD 又常把整条 response 当一个 monolithic string。StepOPSD 的观点是：

```text
credit 不该平均撒到整条轨迹，
也不该无视 observation/action/reasoning 的结构，
而应该落到 action-centered step span 上。
```

## 方法总流程

StepOPSD 的 pipeline：

```text
rollout
-> reward
-> step extraction
-> teacher-student rescoring
-> advantage shaping
-> GRPO policy update
```

也就是说：

```text
在线 rollout 阶段：仍然按普通 policy 和环境交互。
离线训练阶段：对已完成 rollout 做 step 解析和 teacher rescoring。
```

## 1. Step extraction：抽取哪些 token 做 distillation signal

论文不对完整 trajectory 做 OPD，而是先抽取 step segments。

ALFWorld：

```text
action_only
```

含义：只抽取严格 action 命令，因为 embodied task 中真正驱动环境的是 action。

Search-QA：

```text
clean_step_no_observation
```

含义：包含 agent 的内部 reasoning / query / answer step，但 mask 掉 retrieved external knowledge / observation。

原因：

```text
observation 是环境给的，不是 policy 生成的可控行为；
如果把 observation 也蒸馏，会把监督浪费在不可控 token 上。
```

## 2. Teacher / Student 是什么

对第 `k` 个 extracted step：

```text
student context: c_k^S
teacher context: c_k^T = c_k^S concat h_k
```

其中：

```text
h_k = hindsight information
```

论文实际使用的是：

```text
peer-trajectory hindsight
```

具体做法：

```text
同一个 GRPO group 内，如果同时有成功和失败 trajectories，
对失败 trajectory 的 teacher context，
注入该 group 中第一条成功 peer trajectory 作为 hindsight。
```

这点很重要：

```text
teacher 不是外部 oracle demonstration；
teacher 也不是额外大模型生成的新答案；
而是用同组成功轨迹作为训练期 hindsight context。
```

teacher policy 本身：

```text
pi_T = stale_ref_policy
```

即若干训练步之前的 policy 参数，论文写的是例如每 10 steps refresh 一次。这样做是为了避免 teacher 和 student 同步更新导致 moving-target instability。

student policy：

```text
pi_S = 当前训练 policy
```

所以 teacher/student 的差异有两层：

```text
1. teacher context 多了 hindsight h_k；
2. teacher 参数是 stale reference policy，不是当前 student 参数。
```

## 3. Token-level teacher-student gap

对 step `k` 内第 `j` 个 token `z_{k,j}`：

```text
Delta_{k,j}
= log pi_T(z_{k,j} | c_k^T, z_{k,<j})
  - log pi_S(z_{k,j} | c_k^S, z_{k,<j})
```

含义：

```text
Delta > 0:
  加入 hindsight 后，teacher 更支持 student 已经生成的这个 token。

Delta < 0:
  hindsight teacher 不支持这个 token，说明该 token 可能和成功路径不一致。
```

注意：这是对 student rollout 中已经生成的 token 重新打 logprob，不是让 teacher 继续 rollout 新动作。

## 4. 从 gap 到 weight

StepOPSD 先把 gap 变成 raw multiplicative weight：

```text
w_raw_l = 2 * sigmoid(sign(A_l) * Delta_l)
```

这里 `A_l` 是 base token-level advantage，来自 GRPO broadcast。

为什么要乘 `sign(A_l)`？

```text
成功 trajectory: A_l > 0
  Delta_l > 0 表示 teacher 支持该 token，应该放大正向更新；
  Delta_l < 0 表示 teacher 不支持该 token，应该减小正向更新。

失败 trajectory: A_l < 0
  如果 Delta_l > 0，teacher 支持失败轨迹里的这个 token，说明它可能不是坏 token，应该减小惩罚；
  如果 Delta_l < 0，teacher 也不支持这个 token，应该放大负向更新。
```

`2 * sigmoid(x)` 的范围是：

```text
(0, 2)
```

并且：

```text
x = 0 -> w_raw = 1
x > 0 -> w_raw > 1
x < 0 -> w_raw < 1
```

然后做 local trust-region clipping：

```text
w_l = clip(w_raw_l, 1-alpha_clip, 1+alpha_clip)
```

默认：

```text
alpha_clip = 0.2
```

因此：

```text
w_l in [0.8, 1.2]
```

消融中还试了：

```text
alpha_clip = 0.05
w_l in [0.95, 1.05]
```

## 5. Advantage shaping

最后 shaped advantage：

```text
A_tilde_l
= (1-lambda_mix) A_l + lambda_mix (w_l A_l)
```

等价写法：

```text
A_tilde_l = A_l * [1 - lambda_mix + lambda_mix*w_l]
```

其中：

```text
Psi_l = 1 - lambda_mix + lambda_mix*w_l
```

是最终 multiplier。

默认实现：

```text
lambda_mix 初始为 0.2，并在前 50 steps 线性衰减到 0。
```

论文还比较了：

```text
lambda_mix = 0.05
lambda_mix = 0.2
```

这个公式和很多 OPD/RL 混合方法一样，本质是：

```text
原始 RL advantage + lambda_mix 控制的 OPD correction
```

目的：

```text
1. 保留 GRPO 的 reward 方向；
2. 允许 lambda_mix=0 时退化为 GRPO；
3. 防止 teacher gap 直接接管 policy update；
4. 用一个 bounded residual correction 调整 token/step 更新强弱。
```

## 6. Step normalization：为什么还需要 equal_step_mean_abs

如果只做 token-wise shaping，会有一个问题：

```text
长 step 的 token 更多，累计的 Delta / shaping mass 更大；
短 action step 的 token 少，影响更小。
```

这会让 verbose reasoning step 在数学上占据更多 credit budget。

论文用：

```text
equal_step_mean_abs
```

含义：

```text
把每个 extracted step 内的 shaping weights 重新缩放，
让每个 step 的 mean absolute modification budget 相同。
```

直觉：

```text
在没有 dense sub-step rewards 的情况下，
先假设每个 step 是同等重要的因果链环节；
Delta 只负责在这个 step 内部分配哪些 token 更该放大或缩小。
```

这也是 StepOPSD 和简单 token-level OPD/RL reweight 的一个区别。

## 7. 最终 loss 是什么

论文没有给一个完整展开的 GRPO clipped loss，但明确 StepOPSD 是把 `A_l` 替换为 `A_tilde_l` 后继续做 GRPO policy update。

可以写成：

```text
L_StepOPSD(theta)
= - E_l [ min(
      r_l(theta) A_tilde_l,
      clip(r_l(theta), 1-eps, 1+eps) A_tilde_l
    ) ]
  + beta_KL L_KL
```

其中：

```text
r_l(theta) = pi_theta(z_l | context_l) / pi_old(z_l | context_l)
```

关键点：

```text
StepOPSD 不是 L_GRPO + beta * L_OPD；
而是用 teacher-student gap 改写 advantage，再进入 GRPO。
```

## 8. 理论性质

### 8.1 Sign preservation：符号保持

由于：

```text
A_tilde_t = Psi_t * A_t
Psi_t = 1 - lambda_mix + lambda_mix*w_t
w_t > 0
lambda_mix in [0, 1)
```

所以：

```text
Psi_t > 0
sign(A_tilde_t) = sign(A_t)
```

含义：

```text
StepOPSD 不会把成功轨迹 token 改成负更新，
也不会把失败轨迹 token 改成正更新。
```

### 8.2 Directional consistency：方向一致性

论文声称在 multiplier 有界且为正时，StepOPSD 的期望梯度和原 RL 梯度保持非负 cosine similarity：

```text
g_RL = E[A_t grad log pi]
g_StepOPSD = E[A_tilde_t grad log pi]
```

因为每个 token gradient 只是被正数缩放，所以整体方向仍处在 RL 梯度的同一半空间。这个说法的直觉是成立的，但严格地说，逐坐标/逐样本正缩放不必然保证所有复杂情况下的全局 cosine 非负；论文把它作为 bounded positive shaping 的性质使用。

### 8.3 Variance reduction：方差降低

论文假设：

```text
A_t = A*_t + epsilon_t
```

其中：

```text
A*_t = oracle local advantage
epsilon_t = 未来 rollout 不确定性带来的噪声
```

如果 teacher gap `Delta_t` 能无偏反映 `A*_t`，并且当 trajectory-level `A_t` 和 local oracle direction 冲突时让 `Psi_t < 1`，那么 StepOPSD 会压低这些高噪声 token 的更新幅度，从而降低 gradient variance。

直觉：

```text
普通 GRPO：失败轨迹里所有 token 都被惩罚，哪怕某个 step 是对的。
StepOPSD：如果 hindsight teacher 认为这个 step/token 是合理的，就降低惩罚。
```

## 具体计算案例

假设某个 token 来自成功 trajectory：

```text
A_l = +1.0
Delta_l = +0.7
lambda_mix = 0.2
alpha_clip = 0.2
```

先算 raw weight：

```text
w_raw = 2 * sigmoid(sign(A_l) * Delta_l)
      = 2 * sigmoid(+0.7)
      = 2 * 0.668
      = 1.336
```

clip：

```text
w = clip(1.336, 0.8, 1.2) = 1.2
```

shaped advantage：

```text
A_tilde = A_l * [1 - 0.2 + 0.2*1.2]
        = 1.0 * [0.8 + 0.24]
        = 1.04
```

解释：

```text
成功轨迹里的这个 token 被 hindsight teacher 支持，所以正向更新增强 4%。
```

再看失败 trajectory 中一个 teacher 支持的 token：

```text
A_l = -1.0
Delta_l = +0.7
```

```text
sign(A_l) * Delta_l = -0.7
w_raw = 2 * sigmoid(-0.7) = 0.664
w = clip(0.664, 0.8, 1.2) = 0.8
A_tilde = -1.0 * [0.8 + 0.2*0.8]
        = -0.96
```

解释：

```text
虽然整条轨迹失败，但 teacher 认为这个 token/step 还不错，
所以负向惩罚从 -1.0 减弱到 -0.96。
```

如果失败轨迹中 teacher 也不支持这个 token：

```text
A_l = -1.0
Delta_l = -0.7
sign(A_l)*Delta_l = +0.7
w = 1.2
A_tilde = -1.04
```

解释：

```text
trajectory 失败且 hindsight teacher 也不支持，说明这个局部 token 更可能有问题，负向更新增强。
```

## 实验设置

Benchmarks：

```text
ALFWorld
Search-QA
```

ALFWorld 六类：

```text
Pick, Look, Clean, Heat, Cool, Pick2
```

Search-QA 七个子集：

```text
single-hop: NQ, TriviaQA, PopQA
multi-hop: HotpotQA, 2Wiki, MuSiQue, Bamboogle
```

Models：

```text
Qwen3-1.7B
Qwen2.5-3B-Instruct
```

训练配置：

| 项目 | ALFWorld | Search-QA |
|---|---:|---:|
| steps | 150 | 150 |
| GPUs | 4 A800 | 4 A800 |
| batch tasks | 16 | 128 |
| rollouts per prompt | 8 | 论文未单独列出，按 GRPO group 训练 |
| max prompt length | 2048 | 4096 |
| max response length | 512 | 未单独列出 |
| retriever | - | E5 |
| train split | standard splits | NQ + HotpotQA |
| OOD eval | standard splits | TriviaQA, PopQA, 2Wiki, MuSiQue, Bamboogle |

其他实现超参：

| Hyperparameter | Value | 含义 |
|---|---:|---|
| stale ref refresh | 10 steps | teacher 每 10 步刷新一次 |
| `alpha_clip` default | 0.2 | `w in [0.8, 1.2]` |
| reduced `alpha_clip` | 0.05 | `w in [0.95, 1.05]` |
| `lambda_mix` initial | 0.2 | 初始 shaping 强度 |
| decay schedule | first 50 steps to 0 | 前 50 步线性衰减到 0 |
| GRPO KL penalty | 0.01 | reference KL 系数 |
| invalid action penalty | 0.1 | invalid action 惩罚系数 |
| ALFWorld span extraction | action_only | 只抽 action |
| Search-QA span extraction | clean_step_no_observation | 去掉 observation 的 step |
| step budget | equal_step_mean_abs | 步级等预算 |

注意：

```text
step 50 之后 lambda_mix 已经衰减到 0，
所以后面的 teacher-student gap 曲线主要是 stale teacher 与 student drift 的诊断，
不是说明 StepOPSD shaping 仍在持续强作用。
```

## Baselines

论文对比：

| 类型 | 方法 | 含义 |
|---|---|---|
| Training-free | Vanilla | base model |
| Training-free | Skill-Prompt* | 推理时 prepend retrieved skills |
| Post-training | OPSD | vanilla OPSD |
| Post-training | GRPO | 标准 GRPO |
| Post-training | Skill-GRPO | 训练 prompt 注入 skills，测试不注入 |
| Post-training | Skill-GRPO* | 训练和测试都注入 skills |
| Hybrid | GRPO+OPSD | GRPO + naive OPSD auxiliary loss |
| Hybrid | Skill-SD | skill-conditioned distillation |
| Hybrid | RLSD | teacher-student gap 缩放 RL update |
| Hybrid | SDAR | GRPO + gated auxiliary self-distillation loss |

## 完整主结果

### Qwen3-1.7B

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | ALF Avg | NQ | Triv | Pop | Hotp | 2Wk | MuS | Bam | Search Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 25.0 | 22.2 | 3.1 | 0.0 | 21.4 | 4.2 | 12.5 | 29.4 | 46.9 | 37.0 | 23.5 | 19.6 | 6.4 | 10.5 | 24.8 |
| Skill-Prompt* | 10.3 | 50.0 | 16.1 | 0.0 | 5.0 | 9.4 | 14.1 | 29.4 | 46.5 | 36.2 | 22.9 | 20.8 | 4.3 | 10.1 | 24.3 |
| OPSD | 26.3 | 33.3 | 9.1 | 0.0 | 4.5 | 5.3 | 14.1 | 4.2 | 8.3 | 4.6 | 6.6 | 15.3 | 0.7 | 1.2 | 5.8 |
| GRPO | 71.1 | 41.7 | 36.4 | 40.0 | 31.8 | 31.6 | 42.1 | 40.0 | 58.9 | 43.5 | 35.4 | 30.3 | 12.0 | 65.7 | 40.8 |
| Skill-GRPO | 27.6 | 54.5 | 22.7 | 27.3 | 0.0 | 19.2 | 21.1 | 39.2 | 58.6 | 43.9 | 35.2 | 28.2 | 11.5 | 66.1 | 40.4 |
| Skill-GRPO* | 31.4 | 42.9 | 51.9 | 8.3 | 11.5 | 7.1 | 28.1 | 38.0 | 58.4 | 43.9 | 36.3 | 29.0 | 12.5 | 66.9 | 40.7 |
| GRPO+OPSD | 38.2 | 50.0 | 30.8 | 28.6 | 30.0 | 21.1 | 32.0 | 40.7 | 58.9 | 45.0 | 37.0 | 34.6 | 13.3 | 65.7 | 42.2 |
| Skill-SD | 52.9 | 37.5 | 69.2 | 42.9 | 60.0 | 36.8 | 52.3 | 39.1 | 57.5 | 45.4 | 34.8 | 34.1 | 10.7 | 64.1 | 40.8 |
| RLSD | 50.0 | 37.5 | 61.5 | 19.0 | 50.0 | 21.1 | 42.2 | 38.6 | 57.3 | 43.0 | 34.5 | 34.1 | 11.5 | 65.3 | 40.6 |
| SDAR | 73.5 | 25.0 | 76.9 | 33.3 | 40.0 | 36.8 | 53.9 | 39.7 | 58.9 | 45.3 | 35.9 | 35.5 | 12.6 | 65.3 | 41.9 |
| StepOPSD, lambda=0.05 | 64.7 | 44.4 | 56.5 | 60.9 | 42.1 | 55.0 | 56.3 | 40.5 | 58.6 | 45.6 | 34.9 | 29.8 | 10.8 | 64.5 | 41.4 |
| StepOPSD, lambda=0.2 | 58.8 | 44.4 | 52.2 | 26.1 | 52.6 | 20.0 | 43.8 | 40.6 | 59.4 | 44.4 | 37.1 | 32.0 | 11.6 | 64.1 | 42.0 |

### Qwen2.5-3B-Instruct

| Method | Pick | Look | Clean | Heat | Cool | Pick2 | ALF Avg | NQ | Triv | Pop | Hotp | 2Wk | MuS | Bam | Search Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | 44.4 | 11.1 | 6.2 | 15.4 | 28.6 | 12.5 | 21.9 | 24.6 | 48.1 | 31.0 | 26.3 | 25.3 | 7.2 | 59.7 | 31.7 |
| Skill-Prompt* | 51.7 | 66.7 | 48.4 | 0.0 | 4.3 | 10.0 | 28.9 | 23.7 | 46.2 | 30.6 | 24.4 | 22.1 | 7.5 | 12.5 | 23.9 |
| OPSD | 48.8 | 41.7 | 16.7 | 0.0 | 15.8 | 16.7 | 23.1 | 0.1 | 0.1 | 0.1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| GRPO | 91.2 | 62.5 | 96.2 | 61.9 | 65.0 | 47.4 | 70.0 | 43.5 | 58.8 | 43.0 | 36.8 | 32.2 | 11.7 | 72.5 | 42.6 |
| Skill-GRPO | 98.9 | 71.4 | 58.8 | 70.6 | 40.7 | 29.2 | 60.2 | 44.3 | 59.6 | 44.3 | 39.0 | 36.1 | 14.5 | 66.4 | 43.5 |
| Skill-GRPO* | 94.3 | 57.1 | 100.0 | 66.7 | 73.1 | 57.1 | 80.5 | 44.3 | 59.6 | 44.3 | 39.0 | 36.1 | 14.5 | 14.9 | 36.1 |
| GRPO+OPSD | 100.0 | 82.4 | 85.7 | 75.0 | 70.0 | 60.0 | 81.2 | 44.9 | 61.2 | 45.2 | 40.4 | 38.5 | 16.0 | 66.1 | 44.6 |
| Skill-SD | 88.2 | 50.0 | 96.2 | 52.4 | 65.0 | 57.9 | 73.4 | 44.4 | 60.4 | 44.0 | 39.5 | 40.4 | 15.4 | 64.9 | 44.1 |
| RLSD | 87.9 | 75.0 | 90.9 | 75.0 | 73.1 | 68.4 | 79.7 | 41.5 | 58.6 | 42.3 | 40.4 | 40.2 | 16.8 | 66.9 | 43.8 |
| SDAR | 97.1 | 62.5 | 100.0 | 61.9 | 75.0 | 84.2 | 84.4 | 44.8 | 58.1 | 44.3 | 38.6 | 36.2 | 15.7 | 66.1 | 43.4 |
| StepOPSD, lambda=0.05 | 82.4 | 66.7 | 82.6 | 52.2 | 73.7 | 75.0 | 73.4 | 43.6 | 61.2 | 43.8 | 39.2 | 38.1 | 15.8 | 64.5 | 43.7 |
| StepOPSD, lambda=0.05, alpha=0.05 | -- | -- | -- | -- | -- | -- | -- | 45.0 | 61.6 | 46.2 | 39.5 | 39.5 | 14.4 | 65.3 | 45.7 |
| StepOPSD, lambda=0.2 | 45.0 | 55.6 | 87.0 | 65.2 | 57.9 | 85.3 | 69.5 | 44.4 | 61.4 | 46.8 | 40.4 | 38.1 | 14.1 | 65.0 | 44.3 |
| StepOPSD, lambda=0.2, alpha=0.05 | 97.1 | 66.7 | 87.0 | 79.1 | 78.9 | 95.0 | 83.6 | -- | -- | -- | -- | -- | -- | -- | -- |

## 主结果如何读

论文的结果不是“StepOPSD 在所有平均分上统一碾压”。更准确的结论是：

```text
StepOPSD 在那些 sparse trajectory reward 与局部 causal action 最不对齐的子任务上最有优势。
```

典型例子：

```text
ALFWorld Heat:
  1.7B: StepOPSD lambda=0.05 达到 60.9，超过 GRPO 40.0 / SDAR 33.3。
  3B: StepOPSD lambda=0.2, alpha=0.05 达到 79.1，是最高。

ALFWorld Pick2:
  1.7B: StepOPSD lambda=0.05 达到 55.0，是最高。
  3B: StepOPSD lambda=0.2, alpha=0.05 达到 95.0，是最高。

Search-QA TriviaQA:
  3B: StepOPSD lambda=0.05, alpha=0.05 达到 61.6，是最高。

Search-QA HotpotQA:
  1.7B: StepOPSD lambda=0.2 达到 37.1，是最高。
  3B: StepOPSD lambda=0.2 达到 40.4，并列最高。
```

但是：

```text
3B ALFWorld Avg:
  SDAR 84.4 仍高于 StepOPSD 最好 follow-up 83.6。

3B Search Avg:
  StepOPSD alpha=0.05 follow-up 45.7 高于 GRPO+OPSD 44.6 / SDAR 43.4。
```

所以这篇论文更像是在强调：

```text
step-aware shaping 的收益是 failure-mode dependent，
不是一个统一提高所有 aggregate average 的方法。
```

## 消融实验与诊断

论文重点消融两个 knob：

```text
lambda_mix: 全局 shaping 强度
alpha_clip: 局部 token correction 的最大偏移
```

### 1. lambda_mix

比较：

```text
lambda_mix = 0.05
lambda_mix = 0.2
```

发现：

```text
ALFWorld 更偏好较弱 shaping，尤其 1.7B：
  lambda=0.05 Avg 56.3
  lambda=0.2  Avg 43.8

Search-QA 更能接受较强 shaping：
  1.7B Search Avg:
    lambda=0.05: 41.4
    lambda=0.2:  42.0
  3B Search Avg:
    lambda=0.05: 43.7
    lambda=0.2:  44.3
```

论文解释：

```text
Search-QA reward 更稀疏，搜索链更依赖 teacher 对关键 query / evidence step 的约束；
ALFWorld 学会基础动作后，过强 hindsight shaping 会和 RL exploration 拉扯。
```

### 2. alpha_clip

比较：

```text
alpha_clip = 0.2  -> w in [0.8, 1.2]
alpha_clip = 0.05 -> w in [0.95, 1.05]
```

结论：

```text
更小 alpha_clip 是更稳定的 local trust region。
```

3B ALFWorld：

```text
lambda=0.2, alpha=0.2 Avg = 69.5
lambda=0.2, alpha=0.05 Avg = 83.6
```

3B Search-QA：

```text
lambda=0.05, alpha=0.2 Avg = 43.7
lambda=0.05, alpha=0.05 Avg = 45.7
```

论文的 two-knob law：

```text
alpha_clip 越小，通常越稳定；
lambda_mix 最优值随任务变化：
  embodied control 更偏弱 shaping；
  retrieval QA 更偏强或更受 clipping 影响的 shaping。
```

### 3. teacher-student gap 方差诊断

论文跟踪：

```text
Std(Delta)
```

也就是 teacher-student gap 的标准差。

Search-QA：

```text
step 50 前: Std(Delta) 大约 0.40--0.49
step 50 后:
  lambda=0.05 跳到 0.61
  lambda=0.2  稳在约 0.44
```

解释：

```text
Search-QA 中 reward 稀疏，student 更依赖 teacher；
弱 shaping 不够，student 容易 drift。
```

ALFWorld：

```text
heavy shaping lambda=0.2:
  Std(Delta) 从 0.49 到 0.61
milder shaping / tighter clipping:
  Std(Delta) 从 0.46 到 0.26
```

解释：

```text
ALFWorld 中 policy 学会基础控制后，过强 hindsight 会和 RL 探索冲突；
弱 shaping 或 tight clipping 更稳。
```

### 4. late-stage clipping 诊断

3B ALFWorld：

```text
heavy baseline lambda=0.2, alpha=0.2:
  late-stage Std(Delta) = 0.770

reduced alpha lambda=0.2, alpha=0.05:
  late-stage Std(Delta) = 0.252
```

论文还观察到：

```text
ALFWorld mature policy 动作数从 29.2 缩短到 24.3，
valid-action ratio 接近不变。
```

Search-QA：

```text
alpha_clip 收紧后：
  tool use 从 1.45 提升到 1.59；
  response length 从 98.9 降到 86.4 tokens。
```

含义：

```text
收益不是来自格式更正确，
而是把行为重新分配到更有效的 search/tool use 上。
```

## 和 AgentOPSD 的区别

StepOPSD 和 AgentOPSD 都不是简单 `GRPO + OPD loss`，而是用 OPD/self-teacher signal 做 advantage shaping。但二者差异很大。

| 维度 | StepOPSD | AgentOPSD |
|---|---|---|
| 单位 | extracted step / action span | environment turn |
| teacher context | 同组成功 peer trajectory hindsight | retrieved skill `c+` |
| teacher 参数 | stale reference policy | same policy + privileged skill context |
| gap 用法 | 直接变成 token multiplier | 先聚合成 turn evidence，再递归 belief update |
| 是否递归 | 否 | 是 |
| 是否有 belief state | 否 | 有 `B_k` |
| normalization | equal_step_mean_abs | trajectory 内 `q_k` 标准化 |
| 核心主张 | step span 比整条轨迹更合理 | local gap 不是 sequential credit |

可以理解为：

```text
StepOPSD: 把 OPD signal 对齐到 action step，并用 clipping / step budget 稳住。
AgentOPSD: 进一步问每个 turn 在历史 evidence 下是不是 pivotal，用递归 belief revision 打 credit。
```

## 局限性

论文自己也承认：

```text
1. lambda_mix 最优值随环境类型变化，需要调参；
2. 当前只做 post-rollout advantage shaping，不在 generation 阶段动态干预 decoding。
```

我额外补充一点：

```text
peer-trajectory hindsight 依赖同组内有成功样本。
如果 group 全错或成功样本质量低，teacher context 的信息量会明显下降。
```

另外，论文理论里的 variance reduction 依赖一个较强假设：

```text
teacher gap Delta_t 能无偏反映 oracle local advantage A*_t。
```

这在真实 agent 任务中未必总成立，因此它更像解释性理论，而不是严格保证。

## 一句话结论

StepOPSD 的核心不是“更强蒸馏”，而是：

```text
agent 任务的 OPD/RL 信号必须对齐到可控 action step，
再用 bounded sign-preserving advantage shaping 重新分配 GRPO 更新强度。
```

它最适合：

```text
失败由少数局部 action/query 决定，
而 trajectory-level reward 无法指出是哪一步错了的任务。
```
