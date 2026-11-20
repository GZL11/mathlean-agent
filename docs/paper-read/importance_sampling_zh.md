# Uni-OPD / MILES 中的重要性采样技术解读

本文档解释本仓库里容易混淆的两类 importance sampling：

```text
1. PPO / GRPO policy loss 中的 old-policy importance ratio
2. TIS / ICEPOP 中用于处理训推不一致的 train-rollout importance ratio
```

二者都叫 importance ratio，但服务的对象不同。理解这两个 ratio，是读懂 `rollout_log_probs`、`batch["log_probs"]`、`old_log_probs` 和当前 `log_probs` 的关键。

## 一句话总结

在本仓库中：

```text
PPO ratio:
  exp(current_log_probs - old_log_probs)
  约束当前正在更新的 policy 不要偏离采样/old policy 太远。

TIS / ICEPOP ratio:
  exp(train_log_probs - rollout_log_probs)
  衡量训练框架重新 forward 的 logprob 和 rollout engine 生成时记录的 logprob 是否一致。
```

所以重要性采样既可以用于 policy model 和 old policy 之间的校正，也可以用于训推不一致校正。本仓库里二者都出现了。

## 关键变量表

### `rollout_log_probs`

来源：

```text
vLLM / SGLang / endpoint rollout engine
```

含义：

```text
生成 response token 当时，rollout engine 记录的 token logprob。
```

形状：

```text
rollout_log_probs: list[Tensor]
len = B
rollout_log_probs[i].shape = [R_i]
```

其中：

```text
B:
  当前 rollout batch 中的 response 数量，通常等于 prompt 数 * 每个 prompt 采样数。

R_i:
  第 i 条 response 的 response token 长度。
```

代码入口：

```text
miles/rollout/sglang_rollout.py
miles/router/middleware_hub/radix_tree_middleware.py
miles/utils/types.py
```

### `batch["log_probs"]` / `train_log_probs`

来源：

```text
train engine 在训练前，对同一批 rollout trajectory 重新 forward 一遍 actor / old_actor。
```

含义：

```text
训练框架视角下，这批已生成 token 的 logprob。
```

在 TIS / ICEPOP 中，它被显式命名为：

```python
train_log_probs = batch["log_probs"]
```

形状：

```text
batch["log_probs"]: list[Tensor]
len = B
batch["log_probs"][i].shape = [R_i]
```

代码入口：

```text
miles/backends/fsdp_utils/actor.py
  actor_results = self._compute_log_prob("actor", ...)

miles/backends/megatron_utils/actor.py
  self.compute_log_prob(...)
```

注意：Megatron 后端在 `keep_old_actor=True` 时会显式切到 `old_actor` 计算这份 logprob；FSDP 后端这里使用 actor 计算，语义上仍然是训练开始前的 snapshot logprob。

### `old_log_probs`

来源：

```python
old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
```

代码位置：

```text
miles/backends/training_utils/loss.py
  policy_loss_function
```

含义：

```text
PPO / GRPO policy loss 中作为 old policy baseline 的 logprob。
```

它不是一个固定字段，而是一个局部变量：

```text
args.use_rollout_logprobs=True:
  old_log_probs = rollout_log_probs

args.use_rollout_logprobs=False:
  old_log_probs = batch["log_probs"] = train_log_probs
```

所以 `train_log_probs` 和 `old_log_probs` 是否相同，取决于 `args.use_rollout_logprobs`。

### 当前 `log_probs` / `current_log_probs`

来源：

```text
policy loss 反向传播时，当前正在训练的 actor 对同一批 token 再 forward 一遍。
```

代码位置：

```text
miles/backends/training_utils/loss.py
  policy_loss_function
    log_probs_and_entropy = get_log_probs_and_entropy(...)
    log_probs = log_probs_and_entropy["log_probs"]
```

含义：

```text
当前 policy π_θ 对 rollout trajectory 中 token 的 logprob。
```

这份 `log_probs` 是有梯度的，会参与反向传播；前面的 `rollout_log_probs` 和 `batch["log_probs"]` 都是无梯度 baseline / 统计量。

## 一轮训练的时序

完整链路可以写成：

```text
1. actor 权重 θ_k 同步到 rollout engine

2. rollout engine 使用 θ_k 生成 trajectories
   得到:
     - tokens
     - rewards / outcome correctness
     - rollout_log_probs = log π_rollout(a_t | s_t)

3. train engine 收到 rollout batch

4. train engine 在训练前重新 forward 这批 tokens
   得到:
     - batch["log_probs"] = train_log_probs = log π_train_snapshot(a_t | s_t)
     - 可选 ref_log_probs
     - 可选 values

5. 计算 advantages / returns
   对 Uni-OPD:
     - student_log_probs 来自 `batch["log_probs"]` 或 `rollout_log_probs`
     - teacher_log_probs 来自 teacher model
     - OPD advantage = teacher_log_probs - student_log_probs
     - 可选 margin shift / margin mask

6. 开始 actor 训练
   每个 optimizer step 中，当前 actor 重新 forward:
     - current_log_probs = log π_θ(a_t | s_t)

7. PPO / GRPO policy loss:
     ratio = exp(current_log_probs - old_log_probs)

8. 如果开启 TIS / ICEPOP:
     tis_ratio = exp(train_log_probs - rollout_log_probs)
     对 pg_loss 额外加权或 mask

9. 一个 rollout batch 训练完后，actor 从 θ_k 更新到 θ_{k+1}

10. θ_{k+1} 再同步到 rollout engine
    下一轮 rollout 使用新 policy 生成数据
```

## PPO / GRPO 中的 importance ratio

代码位置：

```text
miles/backends/training_utils/loss.py
  policy_loss_function

miles/utils/ppo_utils.py
  compute_policy_loss
```

代码逻辑：

```python
ppo_kl = old_log_probs - log_probs
ratio = (-ppo_kl).exp()
```

等价于：

```text
ratio = exp(current_log_probs - old_log_probs)
```

从概率角度看：

```text
ratio_t = π_θ(a_t | s_t) / π_old(a_t | s_t)
```

它衡量的是：

```text
当前正在优化的 policy，和生成/记录这批训练数据时的 old policy，相差多大。
```

PPO clipped loss：

```python
pg_losses1 = -ratio * advantages
pg_losses2 = -clip(ratio, 1 - eps_clip, 1 + eps_clip_high) * advantages
pg_loss = max(pg_losses1, pg_losses2)
```

含义：

```text
如果当前 policy 相比 old policy 变化太大，就通过 clip 限制更新收益，避免一步把 policy 推太远。
```

这里的 `old_log_probs` 有两种来源：

```text
args.use_rollout_logprobs=True:
  old_log_probs = rollout_log_probs
  PPO ratio 直接对齐 rollout engine 采样分布。

args.use_rollout_logprobs=False:
  old_log_probs = batch["log_probs"]
  PPO ratio 对齐 train engine 训练前 snapshot 分布。
```

## TIS / ICEPOP 中的 importance ratio

代码位置：

```text
miles/backends/training_utils/loss.py
  vanilla_tis_function
  icepop_function
```

TIS 公式：

```python
tis = torch.exp(train_log_probs - rollout_log_probs)
```

等价于：

```text
tis_ratio_t = π_train_snapshot(a_t | s_t) / π_rollout_engine(a_t | s_t)
```

它衡量的是：

```text
训练框架里重新 forward 的 actor，和推理框架里实际生成 token 的 actor，是否一致。
```

理想情况下：

```text
train_log_probs ≈ rollout_log_probs
tis_ratio ≈ 1
```

如果偏离很大，说明可能存在训推不一致：

```text
1. rollout engine 权重同步滞后
2. 异步训练中生成时权重版本和训练时权重版本不一致
3. vLLM / SGLang 与训练框架 kernel 数值差异
4. 量化、并行切分、cache 或采样路径带来的 logprob 差异
```

### Vanilla TIS

代码：

```python
tis = torch.exp(train_log_probs - rollout_log_probs)
tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
pg_loss = pg_loss * tis_weights
```

含义：

```text
ratio 在范围内:
  使用原始 ratio 加权。

ratio 低于下界:
  提升到下界。

ratio 高于上界:
  压到上界。
```

Vanilla TIS 不会直接丢弃 token，只是对 token-level `pg_loss` 加权。

### ICEPOP

代码：

```python
ice_ratio = torch.exp(train_log_probs - rollout_log_probs)
ice_weight = torch.where(
    (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip),
    ice_ratio,
    torch.zeros_like(ice_ratio),
)
pg_loss = pg_loss * ice_weight
```

含义：

```text
ratio 在 [tis_clip_low, tis_clip] 内:
  pg_loss 乘以 ice_ratio。

ratio 超出范围:
  pg_loss 乘以 0，即该 token 不参与梯度。
```

ICEPOP 更像 rejection / masking：

```text
训推差异可接受:
  保留 token，并按 ratio 加权。

训推差异过大:
  认为该 token 太 off-policy 或不可信，直接 mask。
```

注意，不是“范围内填 0、范围外用 ratio”。代码实际是相反的：范围内用 `ice_ratio`，范围外用 0。

## 两个 ratio 同时开启时怎么叠加

在 `policy_loss_function` 里，顺序是：

```text
1. 先算 PPO / GRPO clipped pg_loss
2. 再应用 OPD teacher-student clip / sentinel mask
3. 再应用 TIS / ICEPOP
4. 最后 reduce 成 scalar loss
```

可以写成近似形式：

```text
pg_loss_token
  = PPOClipLoss(current_log_probs, old_log_probs, advantage)
    * OPD_mask
    * TIS_or_ICEPOP_weight
```

其中：

```text
PPOClipLoss:
  处理 current policy vs old policy 的更新幅度。

OPD_mask:
  处理 teacher token-level supervision 是否可信。

TIS_or_ICEPOP_weight:
  处理 train engine vs rollout engine 的 logprob mismatch。
```

## `train_log_probs` 和 `old_log_probs` 到底是不是一个

结论：

```text
不一定。
```

代码中：

```python
old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
```

而：

```python
train_log_probs = batch["log_probs"]
```

所以：

```text
args.use_rollout_logprobs=False:
  old_log_probs == train_log_probs

args.use_rollout_logprobs=True:
  old_log_probs == rollout_log_probs
  old_log_probs != train_log_probs
```

这也是最容易混淆的地方。

## policy 更新粒度

更准确的说法是：

```text
一个 rollout batch 会被切成多个 train step / mini-batch / micro-batch。
micro-batch 用于梯度累积。
optimizer.step() 才真正更新一次 policy。
```

在一个 rollout batch 内：

```text
old_log_probs:
  固定不变。

advantages:
  通常固定不变。

current_log_probs:
  每次当前模型 forward 都会重新计算。

policy 参数:
  每次 optimizer.step() 后更新。
```

因此，随着同一个 rollout batch 被训练多步：

```text
current_log_probs 会逐渐偏离 old_log_probs，
PPO ratio 会偏离 1，
clip 的作用会越来越明显。
```

## 和 Uni-OPD 的关系

Uni-OPD 的核心不是 importance sampling，而是：

```text
用 outcome reward 校准 teacher token-level supervision。
```

也就是：

```text
OPD advantage:
  teacher_log_probs - student_log_probs

trajectory-level G_OPD:
  mean(OPD advantage)

margin shift / margin mask:
  如果错误轨迹的 G_OPD 高于正确轨迹，则修正 teacher supervision 的方向。
```

importance sampling 在这里属于训练稳定性组件：

```text
PPO ratio:
  控制 policy update 不要偏离 old policy 太远。

TIS / ICEPOP:
  控制 train engine 和 rollout engine 不一致带来的 off-policy 噪声。
```

可以理解为：

```text
Uni-OPD 决定“advantage 应该怎么构造和校准”；
PPO ratio 决定“按这个 advantage 更新时步子不能太大”；
TIS / ICEPOP 决定“这批 token 的训推一致性是否足够可信”。
```

## 代码索引

### PPO / GRPO ratio

```text
miles/backends/training_utils/loss.py
  policy_loss_function
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    ppo_kl = old_log_probs - log_probs
    pg_loss, pg_clipfrac = compute_policy_loss(...)

miles/utils/ppo_utils.py
  compute_policy_loss
    ratio = (-ppo_kl).exp()
```

### TIS / ICEPOP

```text
miles/backends/training_utils/loss.py
  vanilla_tis_function
    tis = exp(train_log_probs - rollout_log_probs)
    pg_loss *= clamp(tis)

  icepop_function
    ice_ratio = exp(train_log_probs - rollout_log_probs)
    pg_loss *= ice_ratio if in range else 0
```

### train logprob 计算

```text
miles/backends/fsdp_utils/actor.py
  _compute_log_prob("actor", ...)
  rollout_data.update(actor_results)

miles/backends/megatron_utils/actor.py
  compute_log_prob(...)
```

### rollout logprob 产生

```text
miles/rollout/sglang_rollout.py
  sample.rollout_log_probs += new_response_log_probs

miles/router/middleware_hub/radix_tree_middleware.py
  sample.rollout_log_probs = retrieve_output["rollout_logp"][-sample.response_length:]
```

## 常见误区

### 误区 1：`old_log_probs` 一定等于 `train_log_probs`

不一定。只有 `args.use_rollout_logprobs=False` 时才相等。

### 误区 2：TIS 是 PPO ratio 的另一个名字

不是。PPO ratio 比较的是：

```text
current policy vs old policy
```

TIS 比较的是：

```text
train engine snapshot vs rollout engine
```

### 误区 3：ICEPOP 是把正常 token 置零

不是。ICEPOP 是：

```text
ratio 在阈值内:
  保留并乘以 ratio

ratio 超出阈值:
  置零
```

### 误区 4：一个 mini-batch 更新一次 policy

严格说是 optimizer step 更新一次。一个 optimizer step 可以由多个 micro-batch 梯度累积组成。

### 误区 5：rollout engine 下一次一定立刻用最新 policy

同步取决于训练框架的权重同步时机。同步完成后，下一轮 rollout 才会使用更新后的 policy。如果是异步训练，rollout engine 可能短时间落后于 train engine，这也是 TIS / ICEPOP 有意义的原因。

## 最终理解

可以把本仓库中的三类 logprob 放到一个公式里：

```text
rollout_log_probs:
  π_rollout，推理引擎实际生成 token 时的分布。

train_log_probs = batch["log_probs"]:
  π_train_snapshot，训练框架在训练前重放这批 token 的分布。

current_log_probs:
  π_θ，当前正在反向传播更新的分布。
```

然后：

```text
PPO ratio:
  current_log_probs / old_log_probs

TIS / ICEPOP ratio:
  train_log_probs / rollout_log_probs
```

其中 `old_log_probs` 可以选：

```text
batch["log_probs"]
或
batch["rollout_log_probs"]
```

取决于 `args.use_rollout_logprobs`。

