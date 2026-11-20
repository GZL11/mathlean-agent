# Qwen-VLA：统一多任务、多环境、多机器人本体的 Vision-Language-Action 模型

本文档解读论文：

```text
Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
https://arxiv.org/abs/2605.30280
```

## 一句话总结

Qwen-VLA 不是 OPD / RLHF 论文，而是一篇 **机器人 VLA 基座模型** 论文。

它的核心思路是：

```text
用 Qwen3.5-4B 作为视觉语言骨干，
再接一个 DiT / flow-matching action decoder，
把机器人 manipulation、视觉语言导航、人体第一视角动作、轨迹预测统一成 action / trajectory prediction。
```

最重要的设计是：

```text
不为每个机器人单独设计 policy head；
而是用 embodiment-aware prompt 描述当前机器人本体和控制约定，
再用统一 action tensor + mask 处理不同动作维度。
```

## 模型输入输出

统一形式：

```text
p_theta(y_{t:t+H-1} | o_t, x, e, z)
```

其中：

```text
o_t : 当前视觉观测，可以是单帧、多帧、视频或历史窗口
x   : 自然语言任务指令
e   : embodiment description，描述机器人平台和控制方式
z   : 可选 task identifier
y   : 未来 H 步 action / trajectory
```

不同任务里的 `y` 含义不同：

```text
manipulation:
  末端执行器位置 / 旋转 / 关节角 / gripper / dexterous hand joints

navigation:
  waypoint 序列，通常是 (delta x, delta y, delta theta)

egocentric human data:
  人手 / 手腕 / 身体未来运动轨迹

autonomous-driving / trajectory-centric:
  未来空间轨迹
```

## 架构

Qwen-VLA 分两部分：

```text
1. Qwen3.5-4B vision-language backbone
2. DiT-style flow-matching action expert
```

### VLM backbone

骨干是 Qwen3.5：

```text
视觉 token 和文本 token 早期融合；
支持 image / video / text；
提供空间理解、指代 grounding、多语言指令跟随和推理能力。
```

### Action expert

action expert 是一个单流 DiT：

```text
输入:
  VLM hidden states
  noisy action chunk
  diffusion / flow timestep

输出:
  clean action chunk
```

结构：

```text
16 个 DiT blocks
约 1.15B 参数
flow-matching objective
AdaLN timestep conditioning
multi-section RoPE
```

推理时：

```text
从噪声 action chunk 出发，
用少量 Euler integration steps 去噪，
得到未来动作序列。
```

## Embodiment-Aware Prompt

每个样本前面加一段机器人本体描述：

```text
The robot is {robot_tag} with {single arm / dual arms}[, waist][, and mobile base].
The control frequency is {FPS} Hz.
Please predict the next {chunk_size} control actions to execute the following task:
{ori_instruction}.
```

这段 prompt 告诉模型：

```text
1. 当前是什么机器人
2. 单臂还是双臂
3. 有没有腰部 / 移动底盘
4. 控制频率
5. action chunk 长度
6. 控制约定
```

所以跨机器人泛化不是靠换 head，而是靠：

```text
同一个模型 + 不同 embodiment prompt
```

## 统一动作表示

论文不是把所有机器人的动作强行映射成同一个物理语义空间。

更准确地说：

```text
保留每个数据集自己的动作语义和控制约定；
只统一 tensor 接口。
```

每个样本有：

```text
Y ∈ R^{H x K}
M ∈ {0,1}^{H x K}
```

其中：

```text
H : 最大预测 horizon
K : 统一 channel 维度
M : mask，标记哪些 step / channel 是有效动作
```

如果某个 embodiment 只用 `c <= K` 个 action channels：

```text
前 c 维放真实动作；
剩余 K-c 维 zero padding；
loss 只在 mask=1 的位置计算。
```

优点：

```text
1. 不需要 per-embodiment output head
2. 不同动作维度可以混合训练
3. padding 不参与梯度
4. 控制语义由 prompt + dataset normalization 指定
```

动作归一化用每个数据集每个维度的 1% / 99% quantile：

```text
tilde a_d = 2 * (a_d - q_01^k) / (q_99^k - q_01^k) - 1
```

然后 clip 到：

```text
[-1, 1]
```

## 训练目标

总 loss：

```text
L = lambda_act * L_act + lambda_vl * L_vl
```

### Action loss

对连续动作使用 conditional flow matching。

给定 clean action：

```text
Y_0
```

采样噪声：

```text
Y_1 ~ N(0, I)
```

构造插值：

```text
Y_tau = (1 - tau) Y_0 + tau Y_1
```

训练 DiT 预测速度场：

```text
v_theta(Y_tau, tau | observations, instruction, embodiment)
≈ Y_1 - Y_0
```

loss 按 active channel 和 valid timestep 两级平均，避免 padding 或高维 embodiment 主导梯度。

### Vision-language loss

对普通 VQA、grounding、caption、action caption 等文本输出继续做 next-token prediction：

```text
L_vl = - sum_i log p_theta(w_i | w_<i, o)
```

作用：

```text
防止大规模动作训练破坏原本 VLM 的视觉理解和语言 grounding。
```

## 四阶段训练 recipe

### Stage I：T2A，text-to-action DiT pretraining

先冻结 VLM，只训练 DiT action decoder。

输入：

```text
language instruction + embodiment prompt
```

刻意不输入图像。

目标：

```text
让随机初始化的 DiT 先学会“从文本和本体描述解压出动作分布”。
```

论文把这看成一个 decompression problem：

```text
文本任务描述很短；
动作轨迹很长、连续、高维；
T2A 先学 language-conditioned action prior。
```

### Stage II：CPT，continued pretraining

T2A 后解冻 VLM 和 DiT，一起在大规模混合数据上训练。

目标：

```text
把 T2A 学到的 action prior ground 到真实视觉观测上。
```

输出 checkpoint：

```text
Qwen-VLA-Base
```

### Stage III：SFT

从 CPT checkpoint 分两条 SFT track：

```text
1. multi-task SFT:
   VQA + spatial grounding + manipulation + navigation

2. real-robot SFT:
   用 ALOHA in-house teleoperation 数据做真实机器人部署验证
```

SFT loss 权重：

```text
vision-language next-token: 0.1
manipulation action: 1.0
navigation action: 1.0
```

### Stage IV：RL

从 multi-task SFT checkpoint 开始，用 PPO 在 SimplerEnv 里做 RL。

最终模型：

```text
Qwen-VLA-Instruct
```

RL reward：

```text
sparse binary success reward
R = 1 if task success else 0
```

没有 learned reward model。

## PPO 如何用于 flow-matching action decoder

这部分很关键，因为 DiT / flow-matching policy 不像语言模型 softmax 那样天然给 token logprob。

论文做法是：

```text
把 deterministic probability-flow ODE 转成带噪声的 SDE；
每个 Euler denoising step 变成显式 Gaussian transition；
这样可以解析计算 log probability。
```

rollout 时保存：

```text
intermediate denoising states
```

PPO 更新时：

```text
用当前参数重新计算 velocity field；
重算 Gaussian logprob；
得到 ratio r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t)。
```

默认：

```text
随机选一个 denoising step 估计 logprob
```

这样只需要额外一次 DiT forward。

PPO 配置：

```text
GAE gamma = 0.99
lambda = 0.95
clip epsilon = 0.2
value loss coefficient = 1
4 optimization epochs per rollout batch
actor lr = 5e-6
value head lr = 1e-4
```

value head：

```text
mean-pool VLM hidden states -> linear scalar value
```

并且：

```text
value head 输入处 stop-gradient；
value loss 不回传到 VLM backbone。
```

rollout：

```text
128 parallel envs
每个 iteration 收集 8 rollout epochs * 128 env steps
action chunk H = 16
每轮约 8192 transition chunks
rollout temperature = 1.0
eval temperature = 0.6
```

## Pretraining 数据

预训练混合比例：

| Data Source | Proportion |
|---|---:|
| Robot Manipulation Trajectories | 74.2% |
| Human Egocentric Trajectories | 6.0% |
| Navigation Trajectories | 7.5% |
| Synthetic Simulation Trajectories | 3.7% |
| General Vision-Language Data | 3.4% |
| Spatial Grounding 2D | 2.5% |
| Autonomous Driving VQA | 2.4% |
| Fine-Grained Embodied Action Caption | 0.2% |

机器人数据包括公开数据和自有数据：

```text
公开 real robot:
  RobotSet, Galaxea, AgiBot World, RoboCOIN, RoboMIND,
  RDT-1B, DROID, BridgeData V2, RH20T, RT-1, BC-Z

simulation:
  InternData-A1, GR00T-X-Embodiment-Sim

自有 real robot:
  1000+ 小时

自有 synthetic simulation:
  8M+ trajectories
```

代表机器人本体包括：

```text
WidowX
Google Robot
Franka Panda
ARX5
Fourier GR-1
Mobile ALOHA
AgiBot A2-D
Galaxea R1
AIRBOT MMK2
TienKung
Real Human
```

动作类型包括：

```text
delta EEF + gripper
absolute joint + gripper
absolute joint + dexterous hand
human wrist / hand trajectory
```

## Synthetic simulation 数据怎么造

论文自建两类 synthetic simulation：

```text
1. vision-language-action data
2. language-action data
```

### Vision-language-action synthetic data

使用内部早期版本 ROBOINF：

```text
1. 构造 manipulation scenes
2. 生成 scene-conditioned tasks
3. 合成 executable success checks
4. 用 simulator feedback 生成 robot motion programs
5. rollout 成功轨迹
6. 做 domain randomization
```

规模：

```text
20 tabletop scenes
每个 scene 10 个 object initial-pose configurations
共 200 base scene configurations
450 manipulation tasks
每个 task 300 successful trajectories
最终 359,848 full successful trajectories including subtask segments
```

随机化：

```text
lighting
camera poses
backgrounds
table textures
robot initial states
object initial poses
controller stiffness / damping
```

背景和纹理池：

```text
3K backgrounds
1K table textures
```

轨迹还会切成 subtask segments：

```text
完整任务教长程指令跟随；
子任务片段教原子动作和中间目标。
```

### Language-action synthetic data

这部分没有图像，只用语言和动作。

六类 task templates：

```text
pick-and-place
linear pushing
linear pulling
rotation with repositioning
rotation toward a viewpoint direction
swapping two objects
```

六种单臂机器人：

```text
Franka Panda
UR10e
UR5e
Kinova Gen3
TM12
xArm7
```

每个 robot-task pair：

```text
约 200K trajectories
```

总规模：

```text
约 7.2M trajectories
超过 14,000 小时 simulated robot trajectory data
```

这部分主要用于 T2A：

```text
先让 DiT 学 language-action prior，
再进入视觉 grounding。
```

## 主实验：仿真 manipulation

| Method | Type | LIBERO | RoboCasa-GR1 | Simpler-WidowX | RoboTwin-Easy | RoboTwin-Hard |
|---|---|---:|---:|---:|---:|---:|
| pi0 | Specialist | 94.4 | - | - | 65.9 | 58.4 |
| StarVLA-OFT | Specialist | 96.6 | 48.8 | 64.6 | 50.4 | - |
| GR00T N1.6 | Specialist | 97.2 | 49.9 | 63.2 | 47.6 | - |
| pi0.5 | Specialist | 97.6 | 37.0 | 46.9 | 82.7 | 76.8 |
| ABot-M0 | Specialist | 98.6 | 58.3 | - | 86.0 | 85.0 |
| Being-H0.5 | Specialist | 97.6 | 53.3 | - | - | - |
| Qwen-VLA-Base | Generalist | 90.8 | 40.4 | 64.3 | 64.3 | 66.4 |
| Qwen-VLA-Instruct | Generalist | 97.9 | 56.7 | 73.7 | 86.1 | 87.2 |

结论：

```text
Qwen-VLA-Instruct 是单一 generalist policy，
但在多个 benchmark 上达到或超过 specialist baselines。
```

## 主实验：真实 ALOHA

真实平台：

```text
ALOHA dual-arm robot
两个 6-DoF arms + grippers
3 个 RGB cameras：两个 wrist + 一个 FPV
```

比较：

```text
Qwen-VLA-aloha w/o pretrain:
  同架构，从 scratch 训练

Qwen-VLA-aloha w/ pretrain:
  从 Qwen-VLA-Base fine-tune
```

### In-domain

| Model | Pick and Place | Table Cleaning | Bowl Stacking | Bowl Pick & Place | Towel Folding | Fine-grained Manipulation | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| GR00T N1.6 | 30.8 | 38.5 | 53.8 | 19.2 | 19.2 | 10.3 | 28.6 |
| pi0.5 | 73.1 | 84.6 | 88.5 | 69.2 | 80.8 | 33.3 | 71.6 |
| Qwen-VLA-aloha w/o pretrain | 30.8 | 53.8 | 61.5 | 64.1 | 50.0 | 30.8 | 48.5 |
| Qwen-VLA-aloha w/ pretrain | 96.2 | 92.3 | 98.7 | 87.2 | 65.4 | 61.5 | 83.6 |

### OOD

| Model | Color | Instance | Position | Background | Instruction | Avg |
|---|---:|---:|---:|---:|---:|---:|
| GR00T N1.6 | 46.2 | 38.5 | 3.8 | 19.2 | 19.2 | 25.4 |
| pi0.5 | 57.7 | 61.5 | 19.2 | 26.9 | 42.3 | 41.5 |
| Qwen-VLA-aloha w/o pretrain | 42.3 | 30.8 | 34.6 | 30.8 | 42.3 | 36.2 |
| Qwen-VLA-aloha w/ pretrain | 88.5 | 76.9 | 53.8 | 80.8 | 84.6 | 76.9 |

结论：

```text
预训练是关键来源；
不是架构本身就自动带来真实机器人泛化。
```

## 主实验：VLN 导航

| Method | R2R NE ↓ | R2R OS ↑ | R2R SR ↑ | R2R SPL ↑ | RxR NE ↓ | RxR SR ↑ | RxR SPL ↑ | RxR nDTW ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NaVid | 5.7 | 49.2 | 41.9 | 36.5 | 5.7 | 45.7 | 38.2 | - |
| Uni-NaVid | 5.6 | 53.3 | 47.0 | 42.7 | 6.2 | 48.7 | 40.9 | - |
| NaVILA | 5.2 | 62.5 | 54.0 | 49.0 | 6.8 | 49.3 | 44.0 | 58.8 |
| StreamVLN | 5.0 | 64.2 | 56.9 | 51.9 | 6.2 | 52.9 | 46.0 | 61.9 |
| Qwen-VLA-Base | 5.2 | 61.7 | 53.8 | 49.4 | 6.4 | 55.1 | 45.8 | 56.2 |
| Qwen-VLA-Instruct | 5.1 | 69.0 | 57.5 | 51.2 | 5.8 | 59.6 | 47.8 | 57.1 |

## OOD 静态 manipulation：SimplerEnv-OOD

| Method | MoveAway | MoveRight | PlaceNear | PlaceRight | PutFront | StackYellow | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| pi0.5 | 26.1 | 0.0 | 0.0 | 32.1 | 13.0 | 4.2 | 12.6 |
| Qwen-VLA-Base | 31.3 | 31.6 | 16.7 | 47.1 | 6.3 | 18.8 | 25.3 |
| Qwen-VLA-Instruct | 43.8 | 33.3 | 39.6 | 47.9 | 4.2 | 22.9 | 32.0 |

## OOD 动态 manipulation：DOMINO

| Method | SR ↑ | MS ↑ |
|---|---:|---:|
| OpenVLA | 1.5 | 6.1 |
| RDT-1B | 5.3 | 17.7 |
| pi0 | 8.2 | 24.0 |
| pi0.5 | 9.6 | 26.2 |
| InternVLA-M1 | 5.4 | 27.6 |
| VLA-Adapter | 4.4 | 24.3 |
| pi0-FAST | 3.5 | 20.9 |
| OpenVLA-OFT | 9.1 | 24.1 |
| StarVLA-OFT | 10.9 | 30.5 |
| PUMA | 17.2 | 35.0 |
| OpenVLA-OFT zero-shot | 6.7 | 20.0 |
| pi0.5 zero-shot | 7.5 | 20.4 |
| LingBot-VLA w/ depth | 11.8 | 26.7 |
| LingBot-VA | 24.1 | 36.1 |
| Qwen-VLA-Base | 21.1 | 37.4 |
| Qwen-VLA-Instruct | 26.6 | 39.5 |

论文强调：

```text
Qwen-VLA-Instruct 没有用 dynamic manipulation fine-tuning，
但在 DOMINO 上 zero-shot 超过了许多专门 fine-tuned baseline。
```

## 消融 1：T2A 是否有用

T2A 主要结论：

```text
1. T2A 明显优于 no-T2A
2. 20% synthetic + 80% real 最好
3. full-sequence prediction 优于 chunk prediction
4. T2A 不应该输入图像
5. T2A 用 Sigmoid-Normal timestep distribution 更好
6. 训练 2000 steps 最好，太久会过拟合
```

关键数字：

```text
no T2A:
  60.9%

best T2A:
  71.1%
  20% synthetic + 80% real
  full-sequence
  no image
```

T2A duration：

```text
2000 steps: 71.1%
4000 steps: 67.5%
10000 steps: 67.2%
40000 steps: 60.4%
```

## 消融 2：VL 数据是否干扰动作学习

结论：

```text
VL + VLA co-training 不会明显伤害简单 manipulation，
并且会提升更依赖识别和组合指令的任务。
```

数字：

```text
RoboCasa-GR1:
  VLA-only 51.1%
  VL+VLA 56.0%
  +4.9 pp

RoboTwin-2.0:
  VLA-only 81.8%
  VL+VLA 86.4%
  +4.6 pp
```

## 消融 3：多本体 action projection

三种设计：

```text
Multi-MLP:
  每个 embodiment 一个私有 encoder / decoder

Concat:
  所有 embodiment action 拼成大向量，每个占固定 segment

Zero-Padding:
  短 action vector 右侧 zero-pad 到 d_max，共享 encoder / decoder
```

结果：

| Setting | Bridge | Robocasa |
|---|---:|---:|
| Bridge only | 62.8 | - |
| Robocasa only | - | 53.4 |
| Multi-MLP | 63.3 | 52.1 |
| Concat | 63.0 | 52.8 |
| Zero-Pad | 63.0 | 53.2 |

最终选 Zero-Padding，因为：

```text
性能差不多，但参数最少、架构最简单。
```

## 消融 4：RL post-training

| Stage | Simpler | RoboCasa | RoboTwin-E | RoboTwin-H | LIBERO | SimplerOOD | DOMINO SR | DOMINO MS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CPT | 64.3 | 40.4 | 64.3 | 66.4 | 90.8 | 25.3 | 21.1 | 37.4 |
| + SFT | 70.8 | 56.0 | 86.3 | 87.1 | 97.8 | 31.6 | 25.7 | 39.1 |
| + RL | 73.7 | 56.7 | 86.1 | 87.2 | 97.9 | 32.0 | 26.6 | 39.5 |

结论：

```text
SFT 贡献最大；
RL 在 SimplerEnv 上带来额外 +2.9 pp，
并且没有明显破坏其他 benchmark。
```

## 消融 5：是否需要 proprioceptive state

| Conditioning | RoboTwin-Easy | RoboTwin-Hard |
|---|---:|---:|
| No State | 88.7 | 87.4 |
| State in VLM Prompt | 89.3 | 88.7 |
| State in DiT | 89.4 | 88.3 |

结论：

```text
显式 joint state 只带来很小收益；
默认不加入 proprioceptive state，
保持 embodiment prompt 作为唯一平台特定输入。
```

论文解释：

```text
1. 多视角图像已经能看到机器人当前构型
2. action decoder 预测 relative displacement，不太依赖绝对关节状态
3. 加 joint state 会引入不同机器人的 state interface 复杂性
```

## 这篇论文真正强调的点

Qwen-VLA 的重点不是某一个单独 benchmark，而是：

```text
把 manipulation、navigation、human egocentric motion、trajectory prediction
都放进统一的 action-and-trajectory prediction 框架。
```

它的技术路线是：

```text
强 VLM backbone
+ flow-matching action expert
+ embodiment-aware prompt
+ 大规模异构数据联合训练
+ T2A -> CPT -> SFT -> RL staged recipe
```

## 局限性

论文承认的限制包括：

```text
1. embodied action data 仍远少于普通 VL 数据
2. joint training 会带来目标权衡，可能轻微影响纯视觉语言或导航能力
3. 当前评测仍偏短程和 benchmark-driven
4. 长程真实部署、失败恢复、记忆、world modeling 仍需加强
5. 力觉、触觉、proprioception 等物理反馈还没充分利用
```

## 一句话结论

Qwen-VLA 的核心贡献是：

```text
把 Qwen3.5 的视觉语言能力扩展成可执行动作的统一 VLA 模型，
通过 embodiment-aware prompt 和统一 action tensor，
在不换架构的情况下覆盖 manipulation、navigation、多机器人本体和轨迹预测。
```

