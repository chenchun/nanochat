# NanoChat 三阶段训练脚本对比

NanoChat 的三个训练脚本不是相互替代的方案，而是顺序衔接的三阶段训练流程：

```text
scripts/base_train.py
    从零预训练基础模型
          ↓ base checkpoint
scripts/chat_sft.py
    使用对话与指令数据进行监督微调
          ↓ SFT checkpoint
scripts/chat_rl.py
    在 GSM8K 上通过在线采样和奖励进行强化学习
```

## 总览

| 对比项 | `base_train` | `chat_sft` | `chat_rl` |
|---|---|---|---|
| 训练阶段 | 基础预训练 | 监督微调（SFT） | 强化学习（RL） |
| 模型来源 | 随机初始化，或恢复 base checkpoint | 加载 base checkpoint | 加载 SFT checkpoint |
| 训练数据 | Parquet 普通文本文档 | SmolTalk、MMLU、GSM8K | GSM8K |
| 监督信号 | 原始文本的下一个 token | 标准对话中的 assistant token | 模型回答是否正确的奖励 |
| 训练目标 | 全 token 交叉熵 | assistant-only 交叉熵 | 策略梯度目标 |
| 数据生成方式 | 离线数据 | 离线数据 | 当前模型在线生成回答 |
| Prompt 是否计算 loss | 是，无角色区分 | 否 | 否 |
| 默认训练长度 | 根据 step、FLOPs 或 token/参数比计算 | 一轮混合数据集，或指定 step | 指定轮数的 GSM8K 数据 |
| 主要评估 | Validation BPB、CORE | Validation BPB、ChatCORE | GSM8K Pass@k |
| Checkpoint 目录 | `base_checkpoints` | `chatsft_checkpoints` | `chatrl_checkpoints` |
| 是否保存优化器 | 是 | 是 | 否 |
| 完整断点恢复 | 支持 | 不支持精确恢复数据位置 | 不支持 |
| FP8 | 可选支持 | 无对应参数 | 无对应参数 |

## 1. `scripts/base_train.py`：基础模型预训练

### 目标

从随机权重开始训练通用语言模型，让模型学习语言、知识和文本续写能力。

模型结构由以下参数控制：

- `depth`
- `aspect-ratio`
- `head-dim`
- `max-seq-len`
- `window-pattern`

模型维度和注意力头数大致为：

```text
model_dim ≈ depth × aspect_ratio
num_heads = model_dim / head_dim
```

### 数据与打包

- 从预先准备好的 Parquet 文本文档读取数据。
- 前面的 Parquet 文件作为训练集，最后一个作为验证集。
- 每篇文档前加入 BOS token。
- 使用 best-fit 算法将多篇文档装入固定长度序列。
- 剩余空间无法容纳完整文档时，裁剪文档来填满序列。
- 不使用 padding，序列中的所有目标 token 都参与训练。

### 损失函数

使用标准自回归语言模型交叉熵：

```math
L_{base} = -mean(log p(x_t | x_{<t}))
```

它不区分 user、assistant 或工具等角色，训练序列中的每个位置都需要预测下一个 token。

### 训练规模与优化

训练步数按以下优先级确定：

1. `--num-iterations`
2. `--target-flops`
3. `--target-param-data-ratio`

默认会根据模型规模和 scaling law 自动计算或缩放：

- 训练 token 数
- 总 token batch size
- 学习率
- weight decay
- Muon momentum

优化器是 Muon 和 AdamW 的组合：Transformer 矩阵参数主要使用 Muon，embedding、lm head 和标量参数使用 AdamW。

脚本还支持：

- DDP
- gradient accumulation
- `torch.compile`
- BF16/FP16 和 FP16 GradScaler
- 可选 FP8
- 学习率 warmup、恒定阶段和 warmdown
- Muon momentum 调度
- weight decay 余弦衰减

### 评估与保存

评估内容包括：

- Validation BPB
- CORE metric
- 固定 prompt 的生成样例

Checkpoint 保存模型、优化器、DataLoader 位置和训练循环状态，因此能够较完整地恢复训练。

## 2. `scripts/chat_sft.py`：监督式对话微调

### 目标

加载 base checkpoint，让基础模型学习对话格式、指令遵循、选择题、数学推理和工具调用。

脚本会从 base checkpoint 继承默认训练配置，包括：

- 最大序列长度
- device batch size
- total batch size
- embedding、unembedding 和 matrix 学习率

用户显式传入参数时可以覆盖这些值。

默认还会加载 base 优化器状态，以继承 momentum buffer；加载后会重新设置 SFT 学习率，避免沿用预训练结束时已经衰减到很低的学习率。

### 数据组成

训练数据是以下任务的混合：

- SmolTalk：通用对话
- MMLU：选择题和知识能力
- GSM8K：数学推理和工具使用

默认比例通过重复任务数据实现：

```text
SmolTalk × 1
MMLU × 3
GSM8K × 4
```

### 对话打包与 loss mask

`tokenizer.render_conversation()` 将结构化对话转换成 token，并同时产生 loss mask。

SFT 采用 best-fit-pad 打包：

- 一个序列中可以打包多段完整对话。
- 不裁剪放不下的对话，避免截断问题或标准答案。
- 没有合适对话可填充剩余空间时使用 BOS padding。
- padding target 设置为 `-1`，由交叉熵的 `ignore_index` 忽略。

Loss mask 的语义是：

```text
user prompt          mask=0
BOS/特殊 token        mask=0
tool output          mask=0
assistant completion mask=1
padding              mask=0
```

### 损失函数

SFT 仍使用交叉熵，但只模仿 assistant 应生成的 token：

```math
L_{SFT} = -mean(log p(y_t | conversation, y_{<t})), t 属于 assistant 输出
```

因此，Base 训练学习“所有文本如何继续”，SFT 学习“面对用户输入时 assistant 应该如何回答”。

### 训练调度

- 默认 `--num-iterations=-1`，消费完一轮训练混合数据后停止。
- 也可以通过 `--num-iterations` 限制优化步数。
- 学习率按整体训练进度 `0 → 1` 调度，而不是必须依赖预先确定的总 step。
- weight decay 固定为 0。
- 初始学习率默认为基础学习率的 0.8 倍。
- Muon momentum 从 0.85 逐渐增加到 0.95。

### 评估与保存

评估内容包括：

- Validation BPB
- ChatCORE

ChatCORE 覆盖：

- ARC-Easy
- ARC-Challenge
- MMLU
- GSM8K
- HumanEval

训练结束时保存模型和优化器，但不保存 DataLoader 位置及完整循环状态，因此不能像 `base_train` 一样精确恢复训练进度。

## 3. `scripts/chat_rl.py`：GSM8K 强化学习

### 目标

加载 SFT checkpoint，根据模型回答 GSM8K 问题的正确性继续优化模型。

训练数据只有 GSM8K，但与 SFT 不同，标准答案不直接作为模型模仿目标，而只用于计算奖励。

### 在线 rollout

对每道 GSM8K 问题执行以下流程：

1. 保留 user 问题和 assistant 起始 token。
2. 删除数据集中的标准 assistant 回答。
3. 使用当前模型为同一道题生成 `num_samples` 个回答。
4. 从每个回答的 `#### 数字` 中提取最终答案。
5. 与标准答案进行比较。
6. 正确回答奖励 1，错误回答奖励 0。

默认配置为：

```text
examples_per_step = 16
num_samples = 16
device_batch_size = 8
```

因此，每个优化 step 默认涉及 16 道题、共 256 条生成序列。`device_batch_size` 只限制单次生成或 forward 的序列数。

### Reward 与 advantage

奖励为二值正确性奖励：

```text
回答正确：reward = 1
回答错误：reward = 0
```

同一道题的多个样本使用去均值 advantage：

```math
advantage_i = reward_i - mean(rewards)
```

例如奖励为：

```text
rewards    = [1, 0, 0, 1]
advantages = [0.5, -0.5, -0.5, 0.5]
```

正确回答的生成概率会提高，错误回答的生成概率会降低。如果同一道题的样本全部正确或全部错误，所有 advantage 都为 0，该题不会产生梯度。

### 损失函数

模型首先调用交叉熵并设置 `loss_reduction='none'`，得到每个位置的负对数概率：

```math
\operatorname{NLL}_{i,t}
= -\log \pi_\theta\!\left(y_{i,t}\mid x_i,y_{i,<t}\right)
```

脚本再将其取负，得到对应目标 token 的 log probability：

```math
\log p_{i,t}=-\operatorname{NLL}_{i,t}
```

设 $m_{i,t}\in\{0,1\}$ 为有效 token mask。只有模型自由生成的 token 对应 $m_{i,t}=1$；`targets=-1` 的 prompt、padding 和工具强制输出位置，由交叉熵的 `ignore_index=-1` 处理，其 NLL 和 log probability 都为 0。

对于一次 forward pass，代码计算的策略梯度目标为：

```math
J_p(\theta)
=\frac{1}{N_pPE}
\sum_{i\in p} A_i
\sum_t m_{i,t}
\log \pi_\theta\!\left(y_{i,t}\mid x_i,y_{i,<t}\right)
```

最终最小化的 loss 是其相反数：

```math
L_p(\theta)=-J_p(\theta)
```

其中：

- $A_i=r_i-\bar r$，是在同一道题的全部采样回答之间计算的 advantage。
- $N_p=\sum_{i\in p,t}m_{i,t}$，是当前 forward pass 中的有效生成 token 数。
- $P$ 是一道题的回答被拆成的 forward pass 数，即 `num_passes`。
- $E$ 是当前 rank 在一个优化 step 中处理的题目数，即 `examples_per_rank`。

每道题的 $P$ 次 forward 和每个 rank 的 $E$ 道题都会依次调用 `backward()` 来累积梯度，随后只执行一次 `optimizer.step()`。因此，实际实现是对“每个 forward pass 内按有效 token 数归一化”的目标，再按 pass 数和题目数取平均；它不完全等价于把整个优化 step 的所有有效 token 合并后做一次全局 token 平均。

由于 $A_i$ 在整条回答上保持不变，同一回答中的所有有效生成 token 都乘以相同的 advantage：正 advantage 会提高整条回答的生成概率，负 advantage 会降低整条回答的生成概率。

loss 只覆盖模型自由生成的 token：

- prompt token 不参与训练
- padding token 不参与训练
- 工具强制输出 token 不参与训练
- 模型生成 token 参与策略梯度

### 与标准 GRPO/PPO 的区别

虽然脚本注释称其为 GRPO，但实现经过了明显简化：

- 没有 reference model
- 没有 KL penalty
- 没有 trust region
- 没有 PPO ratio 和 clipping
- 使用当前模型在线采样
- advantage 只减均值，不除以标准差
- loss 按有效生成 token 数归一化

因此，更准确地说，它是基于同题多样本相对奖励的 on-policy REINFORCE。

### 优化、评估与保存

- 使用与其他阶段相同的 Muon + AdamW 优化器构造方式。
- 不加载 SFT 优化器状态。
- 初始学习率默认为配置学习率的 0.05 倍。
- 学习率在训练过程中线性衰减到 0。
- 没有 token batch 的 gradient accumulation、GradScaler 或 `torch.compile` 逻辑。

评估指标是 GSM8K Pass@k：前 k 个生成结果中只要有一个正确，该题就视为 Pass@k 成功。

RL checkpoint 会定期保存模型参数，但不保存优化器状态，因此不能完整恢复 RL 优化过程。

## 三种训练目标的直观对比

假设训练样本是：

```text
User: 2 + 3 等于多少？
Assistant: 5
```

### Base Train

把整个样本视为普通文本，对每个位置预测下一个 token，所有 token 都参与 loss。

### Chat SFT

将 user 输入作为条件，只对 assistant 的答案 `5` 计算监督交叉熵，直接模仿标准答案。

### Chat RL

不向模型提供标准回答，而是让模型生成多个答案：

```text
样本 1：#### 5 → reward=1
样本 2：#### 6 → reward=0
样本 3：#### 5 → reward=1
样本 4：#### 4 → reward=0
```

然后提高正确生成轨迹的概率，降低错误生成轨迹的概率。

## 总结

三个训练阶段分别解决不同问题：

- `base_train`：学习语言、知识和通用文本规律，使模型具备基础续写能力。
- `chat_sft`：学习对话格式、指令遵循和标准解题方式，使模型知道应该如何回答用户。
- `chat_rl`：根据答案正确性优化模型自身的生成分布，提高高奖励答案的生成概率。

整体监督信号的演变是：

```text
全 token 文本模仿
    → 只模仿 assistant 标准答案
    → 不模仿固定答案，直接优化在线生成结果的奖励
```
