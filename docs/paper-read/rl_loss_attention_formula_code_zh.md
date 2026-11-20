# 损失函数与注意力公式 / 代码速查

这份文档把常见的 `cross-entropy loss`、`KL`、`entropy loss`、`DPO / PPO / GRPO / GSPO / GDPO` 的 `adv_compute` 和 `pg_loss`，以及 `softmax / MHA / GQA` 统一整理成一份参考实现。

说明：

```text
1. DPO 不是严格意义上的 policy gradient，但这里也给出它的 pairwise loss 代码。
2. GSPO / GDPO 在不同代码库里有实现变体；下面写的是和主流论文 / TRL / verl 约定一致的版本。
3. 代码是 reference implementation，重点是公式、张量形状和符号约定一致。
```

## 0. 统一记号

```text
B  = batch size
T  = sequence length
V  = vocab size
H  = attention heads
Hq = query heads
Hkv = key/value heads
D  = head_dim
G  = num_generations / group_size
R  = number of reward dimensions
```

常用张量形状：

```text
logits:           [B, T, V]
labels:           [B, T]
token_log_probs:   [B, T]
mask:             [B, T]   # 1 表示有效 token，0 表示 padding / ignore
rewards:          [B]      # 每条 completion 一个标量 reward
rewards_per_func:  [B, R]   # 每条 completion 的多维 reward
values:           [B, T+1]  # PPO / GAE 常见写法，最后一个是 bootstrap value
```

---

## 1. Softmax / Cross-Entropy / Entropy / KL

### 1.1 Softmax

公式：

$$
\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

数值稳定写法：

```python
import torch
import torch.nn.functional as F


def stable_softmax(
    logits: torch.Tensor, 
    dim: int = -1
) -> torch.Tensor:
    """
    避免数值溢出的稳定写法
    Args:
        logits: 任意 shape 的未归一化分数，通常是 [B, T, V] 或 [B, V]
        dim:   softmax 的维度，通常是最后一维 V

    Returns:
        probs: 与 logits 同 shape 的概率分布，沿 dim 归一化后和为 1
    """
    logits = logits - logits.max(dim=dim, keepdim=True).values
    exp_logits = logits.exp()
    return exp_logits / exp_logits.sum(dim=dim, keepdim=True)
```

### 1.2 Cross-Entropy Loss

token-level 交叉熵：

$$
\mathcal{L}_{CE}
=
-\frac{1}{N}\sum_{n=1}^N \log p_\theta(y_n \mid x_n)
$$

对 causal LM 常用的 next-token 版本：

$$
\mathcal{L}_{CE}
=
-\frac{1}{\sum m}
\sum_{b,t} m_{b,t}\log \pi_\theta(y_{b,t}\mid x_b, y_{b,<t})
$$

代码：

```python
def token_logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        logits: [B, T, V]
            已经和 labels 对齐的 logits；对 causal LM 一般是 shift 后的 next-token logits。
        labels: [B, T]
            目标 token id；ignore_index 位置会被 mask 掉。
            每个位置对应的目标 token 在词表里的整数 id（一个标量），值域是 [0, V)
        ignore_index: 需要忽略的位置，通常是 -100

    Returns:
        token_log_probs: [B, T]
        mask:            [B, T]，有效位置为 1，ignore 位置为 0
    """
    # 对 vocab 维度做 log_softmax，得到每个 token 位置上的 log probability。
    # shape: [B, T, V]
    log_probs = F.log_softmax(logits, dim=-1)

    # 构造有效 token 的 mask。
    # labels != ignore_index 的位置参与 loss，labels == ignore_index 的位置被忽略。
    # shape: [B, T]
    mask = labels.ne(ignore_index)

    # gather 不接受 -100 这样的非法 token id，所以先把 ignore 位置临时替换成 0。
    # 这些位置后面会乘 mask 清零，因此替换成 0 不影响最终结果。
    # shape: [B, T]
    safe_labels = labels.masked_fill(~mask, 0)

    # 从 log_probs 里取出目标 token 对应的 log probability。
    # safe_labels.unsqueeze(-1): [B, T, 1]
    # gather 后: [B, T, 1]
    # squeeze 后: [B, T]
    token_log_probs = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    # 把 ignore 位置的 logprob 清零，避免它们参与后续求和。
    # 注意这里返回的是 logprob，不是 loss；真正的 CE loss 会再取负号。
    # shape: [B, T]
    token_log_probs = token_log_probs * mask

    # 返回每个位置的目标 token logprob，以及有效位置 mask。
    return token_log_probs, mask


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Args:
        logits: [B, T, V]
        labels: [B, T]
        ignore_index: padding / ignore 的 label id

    Returns:
        scalar CE loss
    """
    token_log_probs, mask = token_logprobs_from_logits(logits, labels, ignore_index)
    loss = -(token_log_probs.sum() / mask.sum().clamp_min(1))
    return loss
```

### 1.3 KL Divergence

#### 精确 KL

当你同时有两个完整 logits 分布时，精确 KL 为：

$$
\mathrm{KL}(p\|q)
=
\sum_i p_i(\log p_i - \log q_i)
$$

代码：

```python
def exact_kl_from_logits(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
    dim: int = -1,
) -> torch.Tensor:
    """
    Args:
        p_logits: [B, T, V] 或 [B, V]，KL 里的 p
        q_logits: [B, T, V] 或 [B, V]，KL 里的 q
        dim:      vocab 维度

    Returns:
        kl: 与输入 batch shape 对齐、去掉 vocab 维度后的 KL
            例如输入 [B, T, V]，输出 [B, T]
    """
    p_log_probs = F.log_softmax(p_logits, dim=dim)
    q_log_probs = F.log_softmax(q_logits, dim=dim)
    p_probs = p_log_probs.exp()
    return (p_probs * (p_log_probs - q_log_probs)).sum(dim=dim)
```

#### 三种常见 KL 估计器

设：

$$
\Delta = \log \pi_\theta - \log \pi_{\text{ref}}
$$

则常见估计器是：

$$
k_1 = \Delta,\qquad
k_2 = \frac{1}{2}\Delta^2,\qquad
k_3 = e^{-\Delta} + \Delta - 1
$$

其中 `k3` 是 Schulman blog 里常见的 non-negative / low-variance 估计器。代码：

```python
def approx_kl_from_logprobs(
    log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    estimator: str = "k3",
) -> torch.Tensor:
    """
    Args:
        log_probs:      [B, T] 或 [B]，当前策略 logprob
        ref_log_probs:  [B, T] 或 [B]，参考策略 logprob
        estimator:      "k1" / "k2" / "k3" / "low_var_kl"

    Returns:
        kl_estimate: 与 log_probs 同 shape
    """
    log_ratio = log_probs - ref_log_probs

    if estimator in ("k1", "kl"):
        kl = log_ratio
    elif estimator == "k2":
        kl = 0.5 * log_ratio.square()
    elif estimator in ("k3", "low_var_kl"):
        kl = torch.exp(-log_ratio) + log_ratio - 1.0
    else:
        raise ValueError(f"unknown estimator: {estimator}")

    return kl
```

#### 方向提醒

有些代码库写成 `log_ref - log_cur`，那 `k3` 会写成：

$$
e^{\log \pi_{\text{ref}} - \log \pi_\theta}
-1
\;-\;(\log \pi_{\text{ref}} - \log \pi_\theta)
$$

本质是同一个式子，只是符号方向相反。

### 1.4 Entropy

定义：

$$
H(p) = -\sum_i p_i \log p_i
$$

如果要最小化 `-entropy`，就是鼓励更高熵：

```python
def entropy_from_logits(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Args:
        logits: [B, T, V] 或 [B, V]
        dim:    vocab 维度

    Returns:
        entropy: 去掉 vocab 维度后的 entropy，shape = logits.shape 去掉 dim
    """
    log_probs = F.log_softmax(logits, dim=dim)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


def entropy_loss(logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Args:
        logits: [B, T, V]
        mask:   [B, T]，可选

    Returns:
        scalar loss；这里返回的是 -entropy.mean()，即最大化 entropy
    """
    ent = entropy_from_logits(logits, dim=-1)
    if mask is None:
        return -ent.mean()
    return -(ent * mask).sum() / mask.sum().clamp_min(1)
```

---

## 2. DPO / PPO / GRPO / GSPO / GDPO

### 2.1 DPO

DPO 不是严格的 policy gradient。它的核心是 pairwise preference loss：

$$
\ell_{\text{DPO}}
=
-\log \sigma\Big(
\beta\big[
(\log \pi_\theta(y^+|x)-\log \pi_{\text{ref}}(y^+|x))
-
(\log \pi_\theta(y^-|x)-\log \pi_{\text{ref}}(y^-|x))
\big]
\Big)
$$

其中 `y+` 是 chosen，`y-` 是 rejected。

代码：

```python
def sequence_logprob(token_log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        token_log_probs: [B, T]
        mask:           [B, T]

    Returns:
        seq_logprob: [B]，每条序列的 token logprob 之和
    """
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(
    chosen_log_probs: torch.Tensor,
    rejected_log_probs: torch.Tensor,
    chosen_ref_log_probs: torch.Tensor,
    rejected_ref_log_probs: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        chosen_log_probs:       [B]，当前策略下 chosen 序列 logprob
        rejected_log_probs:     [B]，当前策略下 rejected 序列 logprob
        chosen_ref_log_probs:    [B]，参考策略下 chosen 序列 logprob
        rejected_ref_log_probs:  [B]，参考策略下 rejected 序列 logprob
        beta:                   DPO 温度 / scaling

    Returns:
        loss:   scalar
        margin: [B]，DPO 的 pairwise margin
    """
    margin = beta * (
        (chosen_log_probs - chosen_ref_log_probs)
        - (rejected_log_probs - rejected_ref_log_probs)
    )
    loss = -F.logsigmoid(margin).mean()
    return loss, margin
```

`DPO` 没有单独的 `adv_compute`，上面的 `margin` 就是它真正起作用的训练信号。

### 2.2 PPO

PPO 的 clipped policy loss：

$$
\mathcal{L}_{\text{PPO}}
=
-\mathbb{E}\Big[
\min(r_t A_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t)
\Big]
$$

其中：

$$
r_t = \exp(\log \pi_\theta - \log \pi_{\text{old}})
$$

经典 PPO 的 advantage 通常用 GAE：

$$
\delta_t = r_t + \gamma V_{t+1} - V_t
$$

$$
A_t = \delta_t + \gamma\lambda A_{t+1}
$$

代码：

```python
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        rewards: [B, T]，每步奖励
        values:  [B, T+1]，最后一列是 bootstrap value
        dones:   [B, T]，终止位置为 1，否则为 0

    Returns:
        advantages: [B, T]
        returns:     [B, T]
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[:, t]
        delta = rewards[:, t] + gamma * values[:, t + 1] * next_non_terminal - values[:, t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[:, t] = last_gae

    returns = advantages + values[:, :-1]
    return advantages, returns


def ppo_pg_loss(
    token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    Args:
        token_log_probs:     [B, T]
        old_token_log_probs: [B, T]
        advantages:          [B, T]
        mask:                [B, T]
        clip_range:          PPO clip 范围

    Returns:
        scalar policy loss
    """
    ratio = torch.exp(token_log_probs - old_token_log_probs)
    unclipped = -ratio * advantages
    clipped = -(ratio.clamp(1.0 - clip_range, 1.0 + clip_range)) * advantages
    loss = torch.maximum(unclipped, clipped)
    return (loss * mask).sum() / mask.sum().clamp_min(1)
```

### 2.3 GRPO

GRPO 的优势通常来自同一 prompt 下的 group reward：

$$
A_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}
$$

其中 `\mu_g` / `\sigma_g` 是该 prompt 的 group 内均值和标准差。

代码：

```python
def grpo_advantages(
    rewards: torch.Tensor,
    group_size: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Args:
        rewards:    [B]，B = num_prompts * group_size
        group_size: 每个 prompt 采样的 completion 数
        eps:        数值稳定项

    Returns:
        seq_adv:    [B]，每条 completion 一个标量 advantage
    """
    B = rewards.shape[0]
    assert B % group_size == 0

    group_rewards = rewards.view(-1, group_size)   # [num_prompts, group_size]
    group_mean = group_rewards.mean(dim=1, keepdim=True)
    group_std = group_rewards.std(dim=1, keepdim=True)
    seq_adv = (group_rewards - group_mean) / (group_std + eps)
    return seq_adv.reshape(B)


def grpo_pg_loss(
    token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    seq_advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    Args:
        token_log_probs:     [B, T]
        old_token_log_probs: [B, T]
        seq_advantages:      [B]，每条 completion 一个标量 advantage
        mask:                [B, T]
        clip_range:          PPO-style clip 范围

    Returns:
        scalar policy loss
    """
    advantages = seq_advantages[:, None] * mask
    ratio = torch.exp(token_log_probs - old_token_log_probs)
    unclipped = -ratio * advantages
    clipped = -(ratio.clamp(1.0 - clip_range, 1.0 + clip_range)) * advantages
    loss = torch.maximum(unclipped, clipped)
    return (loss * mask).sum() / mask.sum().clamp_min(1)
```

### 2.4 GSPO

GSPO 的关键区别不是 advantage，而是 **sequence-level importance ratio**。

序列级 ratio：

$$
r_{\text{seq}}
=
\exp\Big(
\sum_t \log \pi_\theta(a_t|s_t)
-
\sum_t \log \pi_{\text{old}}(a_t|s_t)
\Big)
$$

GSPO 可以看成：

```text
advantage 仍可来自 GRPO / reward normalization
ratio 改为 sequence-level
```

代码：

```python
def gspo_pg_loss(
    token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    seq_advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    Args:
        token_log_probs:     [B, T]
        old_token_log_probs: [B, T]
        seq_advantages:      [B]
        mask:                [B, T]
        clip_range:          sequence-level clip range

    Returns:
        scalar policy loss
    """
    seq_log_prob = (token_log_probs * mask).sum(dim=-1)       # [B]
    seq_old_log_prob = (old_token_log_probs * mask).sum(dim=-1)
    ratio = torch.exp(seq_log_prob - seq_old_log_prob)        # [B]

    unclipped = -ratio * seq_advantages
    clipped = -(ratio.clamp(1.0 - clip_range, 1.0 + clip_range)) * seq_advantages
    loss = torch.maximum(unclipped, clipped)
    return loss.mean()
```

### 2.5 GDPO

GDPO = **Group reward-Decoupled Normalization Policy Optimization**。

它的关键是：**多 reward 时，不要先把 reward 加总再做 group normalization，而是先对每个 reward 维度分别做 group normalization，再聚合。**

设多维 reward 为：

$$
\mathbf{r}_i = (r_i^{(1)}, \dots, r_i^{(R)})
$$

GDPO 的做法是先对每个 `r^{(j)}` 单独做 group norm：

$$
\hat{r}_i^{(j)}
=
\frac{r_i^{(j)} - \mu_g^{(j)}}{\sigma_g^{(j)} + \epsilon}
$$

再加权聚合：

$$
A_i^{\text{pre-bn}}
=
\sum_{j=1}^R w_j \hat{r}_i^{(j)}
$$

最后做一次 batch norm：

$$
A_i
=
\frac{A_i^{\text{pre-bn}} - \mu്ബ}{\sigma_b + \epsilon}
$$

代码：

```python
def gdpo_advantages(
    rewards_per_func: torch.Tensor,
    group_size: int,
    reward_weights: torch.Tensor,
    scale_rewards: bool = True,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Args:
        rewards_per_func: [B, R]，每条 completion 的多维 reward
        group_size:       每个 prompt 的 completion 数
        reward_weights:   [R]，每个 reward 的聚合权重
        scale_rewards:    是否在最后做 batch-wise normalization
        eps:              数值稳定项

    Returns:
        advantages:       [B]
    """
    B, R = rewards_per_func.shape
    assert B % group_size == 0
    assert reward_weights.shape[0] == R

    rewards_per_func = torch.nan_to_num(rewards_per_func)
    all_reward_advantages = []

    for j in range(R):
        reward_j = rewards_per_func[:, j]                  # [B]
        grouped = reward_j.view(-1, group_size)            # [num_prompts, G]
        mean_g = grouped.mean(dim=1, keepdim=True)
        std_g = grouped.std(dim=1, keepdim=True)
        adv_j = (grouped - mean_g) / (std_g + eps)
        all_reward_advantages.append(adv_j.reshape(B))     # [B]

    stacked = torch.stack(all_reward_advantages, dim=1)    # [B, R]
    pre_bn_adv = (stacked * reward_weights.to(stacked.device).unsqueeze(0)).sum(dim=1)

    if scale_rewards:
        pre_bn_adv = (pre_bn_adv - pre_bn_adv.mean()) / (pre_bn_adv.std() + eps)

    return pre_bn_adv


def gdpo_pg_loss(
    token_log_probs: torch.Tensor,
    old_token_log_probs: torch.Tensor,
    seq_advantages: torch.Tensor,
    mask: torch.Tensor,
    clip_range: float = 0.2,
) -> torch.Tensor:
    """
    GDPO 本身是 advantage 侧的改进，policy loss 通常仍然复用 GRPO / PPO 的 clipped ratio loss。

    Args:
        token_log_probs:     [B, T]
        old_token_log_probs: [B, T]
        seq_advantages:      [B]
        mask:                [B, T]
        clip_range:          PPO-style clip 范围

    Returns:
        scalar policy loss
    """
    advantages = seq_advantages[:, None] * mask
    ratio = torch.exp(token_log_probs - old_token_log_probs)
    unclipped = -ratio * advantages
    clipped = -(ratio.clamp(1.0 - clip_range, 1.0 + clip_range)) * advantages
    loss = torch.maximum(unclipped, clipped)
    return (loss * mask).sum() / mask.sum().clamp_min(1)
```

---

## 3. Attention：MHA / GQA

### 3.1 Multi-Head Attention

公式：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
\mathrm{Attn}(Q,K,V)
=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{D}} + M\right)V
$$

$$
\mathrm{MHA}(X)
=
\mathrm{Concat}(h_1,\dots,h_H)W_O
$$

其中每个 head：

$$
h_i = \mathrm{Attn}(Q_i, K_i, V_i)
$$

代码：

```python
def mha_forward(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_heads: int,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        x:          [B, T, D_model]
        W_q:        [D_model, H * D_head]
        W_k:        [D_model, H * D_head]
        W_v:        [D_model, H * D_head]
        W_o:        [H * D_head, D_model]
        num_heads:  H
        attn_mask:  可选，形状 [B, 1, T, T] 或 [B, T, T]，
                    True 表示可见，False 表示屏蔽

    Returns:
        out: [B, T, D_model]
    """
    B, T, D_model = x.shape
    head_dim = W_q.shape[-1] // num_heads
    scale = head_dim ** -0.5

    q = x @ W_q   # [B, T, H*D]
    k = x @ W_k   # [B, T, H*D]
    v = x @ W_v   # [B, T, H*D]

    q = q.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, D]
    k = k.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, D]
    v = v.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, D]

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale   # [B, H, T, T]

    if attn_mask is not None:
        if attn_mask.dim() == 3:
            attn_mask = attn_mask[:, None, :, :]
        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

    attn = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, v)                             # [B, H, T, D]
    ctx = ctx.transpose(1, 2).contiguous().view(B, T, -1)   # [B, T, H*D]
    out = ctx @ W_o                                         # [B, T, D_model]
    return out
```

### 3.2 Grouped-Query Attention

GQA 的区别是：

```text
query heads 多，key/value heads 少
多个 query head 共享同一组 key/value head
```

设：

$$
H_q = g \cdot H_{kv}
$$

则每个 kv head 服务 `g` 个 query heads。

公式和 MHA 相同，只是 `K, V` 的 head 数更少，随后复制到 query head 数目。

代码：

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Args:
        x:    [B, Hkv, T, D]
        n_rep: 每个 kv head 要复制几次，满足 Hq = Hkv * n_rep

    Returns:
        [B, Hkv * n_rep, T, D]
    """
    return x.repeat_interleave(n_rep, dim=1)


def gqa_forward(
    x: torch.Tensor,
    W_q: torch.Tensor,
    W_k: torch.Tensor,
    W_v: torch.Tensor,
    W_o: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Args:
        x:            [B, T, D_model]
        W_q:          [D_model, Hq * D_head]
        W_k:          [D_model, Hkv * D_head]
        W_v:          [D_model, Hkv * D_head]
        W_o:          [Hq * D_head, D_model]
        num_q_heads:  Hq
        num_kv_heads: Hkv
        attn_mask:    [B, 1, T, T] 或 [B, T, T]

    Returns:
        out: [B, T, D_model]
    """
    assert num_q_heads % num_kv_heads == 0
    n_rep = num_q_heads // num_kv_heads

    B, T, D_model = x.shape
    head_dim = W_q.shape[-1] // num_q_heads
    scale = head_dim ** -0.5

    q = x @ W_q   # [B, T, Hq*D]
    k = x @ W_k   # [B, T, Hkv*D]
    v = x @ W_v   # [B, T, Hkv*D]

    q = q.view(B, T, num_q_heads, head_dim).transpose(1, 2)      # [B, Hq, T, D]
    k = k.view(B, T, num_kv_heads, head_dim).transpose(1, 2)     # [B, Hkv, T, D]
    v = v.view(B, T, num_kv_heads, head_dim).transpose(1, 2)     # [B, Hkv, T, D]

    k = repeat_kv(k, n_rep)                                       # [B, Hq, T, D]
    v = repeat_kv(v, n_rep)                                       # [B, Hq, T, D]

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale         # [B, Hq, T, T]

    if attn_mask is not None:
        if attn_mask.dim() == 3:
            attn_mask = attn_mask[:, None, :, :]
        scores = scores.masked_fill(~attn_mask, torch.finfo(scores.dtype).min)

    attn = torch.softmax(scores, dim=-1)
    ctx = torch.matmul(attn, v)                                   # [B, Hq, T, D]
    ctx = ctx.transpose(1, 2).contiguous().view(B, T, -1)         # [B, T, Hq*D]
    out = ctx @ W_o                                               # [B, T, D_model]
    return out
```

---

## 4. 一个统一的“怎么用”视角

```text
1. 先从 logits 算 token_log_probs / entropy / KL
2. 再根据算法算 advantage
3. 再把 log_probs 和 advantage 合成 pg_loss
4. DPO 例外，它直接做 pairwise loss，没有显式 advantage
```

如果你要的是工程里的最小闭环，通常只需要这几个函数：

```text
cross_entropy_loss
exact_kl_from_logits / approx_kl_from_logprobs
entropy_loss
compute_gae
grpo_advantages
ppo_pg_loss / grpo_pg_loss / gspo_pg_loss / gdpo_pg_loss
dpo_loss
mha_forward / gqa_forward
```

---

## 5. 参考来源

```text
1. DPO: Direct Preference Optimization
   https://arxiv.org/abs/2305.18290

2. PPO: Proximal Policy Optimization Algorithms
   https://arxiv.org/abs/1707.06347

3. GRPO / GSPO 相关实现参考
   https://huggingface.co/docs/trl
   以及本仓库 miles/miles/utils/ppo_utils.py

4. GDPO: Group reward-Decoupled Normalization Policy Optimization
   https://nvlabs.github.io/GDPO/
   https://arxiv.org/abs/2601.05242

5. GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
   https://arxiv.org/abs/2305.13245
```
