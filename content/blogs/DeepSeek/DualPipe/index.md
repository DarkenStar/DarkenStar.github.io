---
title: "DualPipe"
date: 2025-06-21T22:00:13+08:00
lastmod: 2025-06-21T22:00:13+08:00
author: ["WITHER"]

categories:
- Source Code Reading

tags:
- DeepSeek

keywords:
- DeepSeek
- DualPipe

description: "Source code reading of DualPipe" # 文章描述，与搜索优化相关
summary: "Source code reading of DualPipe" # 文章简单描述，会展示在主页
weight: # 输入1可以顶置文章，用来给文章展示排序，不填就默认按时间排序
slug: ""
draft: false # 是否为草稿
comments: true
showToc: true # 显示目录
TocOpen: true # 自动展开目录
autonumbering: true # 目录自动编号
hidemeta: false # 是否隐藏文章的元信息，如发布日期、作者等
disableShare: true # 底部不显示分享栏
searchHidden: false # 该页面可以被搜索到
showbreadcrumbs: true #顶部显示当前路径
mermaid: true
cover:
    image: ""
    caption: ""
    alt: ""
    relative: false
---

# Preliminary

本节先回顾流水线并行以及 DeepSeek-V3 中作为 baseline 的 [PipeDream](https://arxiv.org/pdf/1806.03377) 论文中的 1F1B 和 [ZeroBubble](https://openreview.net/pdf?id=tuzTN0eIO5) 论文中的 ZB1P (ZB-H1 的自动搜索结果).

## PipeDream 1F1B 

1F1B (One-Forward-One-Backward) 的工作流程如图所示，想象一条工厂流水线，用于组装一个复杂的设备。这个设备需要经过多个工位（GPU），每个工位负责一部分装配任务（模型的不同层）。当第一个产品的第一个部件在工位1上加工时，其他所有工位都在闲置等待。当它被传递到工位2时，工位1开始加工第二个产品，但工位3、4…依然在等待。这种在流水线启动和结束阶段产生的设备空闲时间，就是流水线气泡 (Pipeline Bubble). 在大模型训练中，这意味着 GPU 算力被浪费，直接导致训练时间延长和成本增加。

![1F1B pipeline Schedule](https://share.note.youdao.com/yws/api/personal/file/WEB787301b994e9c90258f8ad84fd1f8b67?method=download&shareKey=db772d656fe8be439988e887fd6910a3 "1F1B pipeline Schedule")

后续批次的后向传播永远在前一批次的后向传播全部启动后才开始，为了防止激活占用内存过多，图中 1F1B 的 bs=8，流水线并行过程中最多保存 4 个 batch 的激活，当 batch1 反向传播结束后再进行 batch5 的正向传播。为了减少激活占用，1F1B 中进行反向传播的优先级高于正向传播。

## ZeroBubble ZB1P

ZeroBubble 减少气泡的关键是将反向传播中对于权重和输入的梯度计算分开进行。传统上，一个层的反向传播包含两个核心任务:
- B Pass: 计算关于输入梯度并将其传递给前一层，这是误差反向传播链的一部分。
- W Pass: 计算该层自身权重的梯度，用于后续的参数更新。

如图所示第 i-1 层的 B Pass 依赖于第 i 层的 B Pass. 但第 i 层的 W Pass，只要在其 B Pass 完成之后，可以被灵活地安排在任何时间点执行。

![Computation Graph for MLP](https://share.note.youdao.com/yws/api/personal/file/WEBeef090dd44e296a4a4e985200c62e4c7?method=download&shareKey=2d9e68a50dd5b0b3c46f079499ed2bec "Computation Graph for MLP")
--- 

**Handcrafted Pipeline Schedule**

基于这个思想，文中提出了两个手工设计的调度方案作为概念验证:
- ZB-H1 (Memory Efficient Schedule): 在维持与 1F1B 相似峰值内存消耗的情况下，通过将 W Pass 推迟执行，填充了流水线末尾的 cooldown 气泡，成功将气泡大小减少到 1F1B 的三分之一。
- ZB-H2 (Zero Bubble Schedule): 当内存预算更宽松时，在流水线 warm-up 安排更多的 F Pass，并巧妙地重排 W Pass，将整个流水线的执行过程从一个梯形变成了一个平行四边形，从而在理论上完全消除了气泡。

![Handcrafted pipeline schedules ZB-H1 (top)  & ZB-H2 (bottom)](https://share.note.youdao.com/yws/api/personal/file/WEB9f702f88f4b48c048d602ddfe7b69ffb?method=download&shareKey=010eea5b8230b6175d7777444e4dcc64 "Handcrafted pipeline schedules ZB-H1 (top)  & ZB-H2 (bottom)")

文中基于一个标准的 Transformer架构，其中 FFN 的中间层维度是模型隐藏维度 `h` 的4倍。给出了 F, B, W 各自的计算量和激活占用。其中计算量只统计占据主要部分的矩阵乘法的浮点运算次数。

* `b`: microbatch size 
* `s`: sequence length 
* `h`: hidden dimension size 
* `a`: number of attention heads 

![Transformer Architecture](https://share.note.youdao.com/yws/api/personal/file/WEBd9f810c1f506d383eb59a0c1186e602b?method=download&shareKey=ee9e8a521ed880701b05e9b25b1ae001 "Transformer Architecture")
---

*Table1: FLOPs and activations memory required per transformer layer for each pass*
| Pass | FLOPs | Activations Memory Required |
| :---: | :---: | :---: |
| F | $sbh(24h+4s)$ | 0 |
| B | $sbh(24h+8s)$ | $sb(34h+5as)$ |
| W | $sbh(24h)$ | $32sbh$ |

---
前向传播 $T_F \approx (8bsh^2 + 4bs^2h) + 16bsh^2 = 24bsh^2 + 4bs^2h = sbh(24h + 4s)$. 反向传播关于权重的计算量等于 Linear 层的 GEMM. 
* **Self-Attention**: $6bsh^2 + 2bs^2h + 2bs^2h + 2bsh^2 = 8bsh^2 + 4bs^2h$
    * **Q, K, V Projection**：输入 `(b, s, h)` 通过与权重矩阵 `(h, h)` 相乘，生成Q, K, V。这涉及到3次矩阵乘法。$\text{FLOPs} \approx 2 \times b \times s \times h \times 3h = 6bsh^2$
    * **Attention Score**:`Q` `(b, a, s, h/a)` 与 `K^T` `(b, a, h/a, s)` 相乘。$\text{FLOPs} \approx 2 \times b \times a \times s \times (h/a) \times s = 2bshs$.
    * **Score@V**：注意力分数 `(b, a, s, s)` 与 `V` `(b, a, s, h/a)` 相乘。$\text{FLOPs} \approx 2 \times b \times a \times s \times s \times (h/a) = 2bshs$.
    * **O Projecyion**：结果与输出权重矩阵 `(h, h)` 相乘。$\text{FLOPs} \approx 2 \times b \times s \times h \times h = 2bsh^2$.

* **FFN FLOPs**: $8bsh^2 + 8bsh^2 = 16bsh^2$
    * **Up Projection**：输入 `(b, s, h)` 与权重矩阵 `(h, 4h)` 相乘。$\text{FLOPs} \approx 2 \times b \times s \times h \times 4h = 8bsh^2$.
    * **Down Projection**：中间结果 `(b, s, 4h)` 与权重矩阵 `(4h, h)` 相乘。$\text{FLOPs} \approx 2 \times b \times s \times 4h \times h = 8bsh^2$.

---

激活占用方面除了 Dropout Mask 是 INT8 类型以外，假设 activations 均以 16-bit float 类型保存。表中的 activation memory 均以字节为单位进行统计。和权重梯度无关的部分只有 dropout 相关的以及 Softmax output.

| Category | Item | Original | TP |
| :---: | :---: | :---: | :---: |
| **Attention** | **Total** | **$11sbh + 5as^2b$** | **$3sbh + \frac{8sbh}{t} + \frac{5as^2b}{t}$** |
| | QKV input | $2sbh$ | $2sbh$ |
| | QK^T | $4sbh$ | $\frac{4sbh}{t}$ |
| | Softmax output | $2as^2b$ | $\frac{2as^2b}{t}$ |
| | Dropout mask | $as^2b$ | $\frac{as^2b}{t}$ |
| | Dropout output | $2as^2b$ | $\frac{2as^2b}{t}$ |
| | V | $2sbh$ | $\frac{2sbh}{t}$ |
| | Linear projection input | $2sbh$ | $\frac{2sbh}{t}$ |
| | Attention dropout mask | $sbh$ | $sbh$ |
| **MLP** | **Total** | **$19sbh$** | **$3sbh + \frac{16sbh}{t}$** |
| | Linear1 input | $2sbh$ | $2sbh$ |
| | GeLU input | $8sbh$ | $\frac{8sbh}{t}$ |
| | Linear2 input | $8sbh$ | $\frac{8sbh}{t}$ |
| | Dropout mask | $sbh$ | $sbh$ |
| **LayerNorm** | **Total** | **$4sbh$** | **$4sbh$** |
| | LayerNorm1 input | $2sbh$ | $2sbh$ |
| | LayerNorm2 input | $2sbh$ | $2sbh$ |

---
在没有 $T_F = T_B = T_W$ 假设的情况下，ZB-H1 和 ZB-H2 的峰值激活内存和气泡大小如 Table 2 所示。值得注意的是，对于设备 *i*，其在 ZB-H1 方案下的激活内存为 $(p-i+1)M_B + (i-1)M_W$，在 ZB-H2 方案下的激活内存为 $(2p - 2i + 1)M_B + (2i - 2)M_W$。如 Table 1 所示，*W* 所需的激活内存小于 *B* 所需的激活内存。因此，ZB-H1 和 ZB-H2 的峰值激活内存分别为 $pM_B$ 和 $(2p-1)M_B$。

*Table 2: Comparison between 1F1B and our handcrafted schedules.*
| Schedule | Bubble size | Peak activations memory |
| :---: | :---: | :---: |
| 1F1B | $(p-1)(T_{F}+T_{B}+T_{W})$ | $pM_{B}$ |
| ZB-H1 | $(p-1)(T_{F}+T_{B}-T_{W})$ | $pM_{B}$ |
| ZB-H2 | $(p-1)(T_{F}+T_{B}-2T_{W})$ | $(2p-1)M_{B}$ |

**Automatic Pipeline Scheduling**

手工调度依赖于 F、B、W 的执行时间相等的理想情况。为了应对真实世界中复杂的执行时间和通信延迟，该文开发了一个自动化流水线调度算法。该算法通过一系列启发式策略，在一个给定的内存限制下，自动地为流水线生成一个高效的调度方案。核

1. **Warm-up**：
    * 在内存限制的范围内，算法会尽可能多地调度 F pass ，以最小化在第一个 B pass 开始前产生的气泡。
    * 此阶段使用一个超参数来控制是否要调度一个可能会延迟后续B Pass的额外F Pass。

2. **Steady State**：
    * 热身阶段结束后，调度进入一个迭代模式，轮流调度一个F Pass和一个B Pass。
    * 为了填充气泡，算法会伺机插入 W pass. 插入策略是：
        * 当出现一个大于 $T_W$ (W Pass 执行时间) 的气泡时，直接插入一个W Pass.
        * 当出现一个小于 $T_W$ 的气泡时，如果这个气泡会导致当前阶段的累计气泡时间成为所有阶段中最长的，那么仍然会插入一个W Pass.
        * 当内存达到上限时，也会插入 W Pass 以回收和释放部分内存。
    * 通常这个启发式策略在稳态阶段会形成一个 1F-1B-1W 的调度模式。

3. **Global Schedule**：
    * 在整个调度过程中，算法始终保证在 F Pass 用完之前，第 i 阶段调度的 F Pass 数量至少比第 i+1 阶段多一个。
    * 当这个数量差超过一时，会使用另一个超参数来决定在不产生额外气泡的前提下，是否要跳过第 i 阶段的一次F Pass调度。
    * 算法会通过 grid search 来寻找这些超参数的最佳组合。

4. **Final**：当某个阶段的 F Pass 和 B Pass 都执行完毕后，算法会一次性逐个调度完所有剩余的 W Pass.

---

**Bypassing Optimizer Synchronization**

要实现完美的平行四边形调度，还需要解决优化器同步（Optimizer Synchronization）. 在分布式训练中，通常需要在更新模型参数前，在所有 GPU 间进行一次 All-Reduce，以进行梯度裁剪（Gradient Clipping）或检查数值稳定性 (NaN/INF). 这个同步点会强制所有设备等待，从而破坏平行四边形，重新引入气泡。

论文提出了 Bypassing Optimizer Synchronization，每个 GPU 在执行优化器更新步骤时，不再等待全局同步，而是基于从前一个 GPU 传来的部分 reduce 的信息进行推测性更新。该 micro-batch 完整的全局状态会在下一个迭代的 warp 阶段异步地传回。每个 GPU 在收到最终的全局状态后，会验证自己上一步的更新是否合法。如果发现不一致（例如，全局梯度范数超出了裁剪阈值），它会执行一次原地回滚（In-place Rollback），然后使用正确的全局状态重新执行优化器步骤。

![The Post-validation Strategy to Replace Optimizer Synchronization](https://share.note.youdao.com/yws/api/personal/file/WEB69834a68b841e1af30873b5a95a2fc90?method=download&shareKey=0544362bccaefd3cad59bb0be406a145 "The Post-validation Strategy to Replace Optimizer Synchronization")

# DualPipe
DualPipe 是一种创新的双向流水线并行算法。它的核心思想是在一组设备上同时处理两个方向的数据流：一个前向流水线和一个反向流水线。使得计算和通信能够更充分地重叠，从而减少流水线气泡（即 GPU 空闲时间）.

与传统的 GPipe（1F1B）只有一个数据流方向不同，DualPipe 将设备对折，形成两条对称的流水线。例如，在一个有 8 个 PP ranks (GPU) 的设置中：
- 前向流水线 (Forward Pipeline): 数据从 rank 0 -> 1 -> 2 -> 3.
- 反向流水线 (Backward Pipeline): 同时有另一组数据从 rank 7 -> 6 -> 5 -> 4.
Rank 3 和 Rank 4 成为两条流水线的中间节点，它们之间会交换数据。每个设备实际上会处理两个流水线阶段的模型块，一个用于前向流水线，另一个用于反向流水线。

## Initialization
- modules: 每个 DualPipe 实例接收一个元组，包含两个 nn.Module. `modules[0]` 用于处理前向->反向的计算，`modules[1]` 用于处理反向->前向的计算。
- Rank 角色判断: 代码会根据当前 rank 的 ID 判断其在整个流水线中的位置（是否是第一个、最后一个、是否在后半部分、是否是中间节点）. 这个角色判断对于后续的通信和计算调度至关重要。例如 `is_in_second_half` 决定了该 rank 的 phase 0 和 phase 1 究竟对应前向流水线还是反向流水线。

```python{linenos=true}
class DualPipe(nn.Module):
    def __init__(
        self,
        modules: Tuple[nn.Module, nn.Module],
        # ...
    ) -> None:
        super().__init__()

        # 每个 rank 持有两个模型模块
        self.module = nn.ModuleList(modules)
        # ...
        self.group = process_group or dist.distributed_c10d._get_default_group()
        self.num_ranks = self.group.size()

        # ...
        # 计算当前 rank 在流水线中的角色
        self.rank = rank_mapping[self.group.rank()]
        self.is_first_rank = self.rank == 0
        self.is_last_rank = self.rank == self.num_ranks - 1
        # 判断 rank 是否在对折后的后半部分
        self.is_in_second_half = self.rank >= self.num_ranks // 2
        # 判断是否为中间的 rank
        self.is_middle_rank = (self.rank == self.num_ranks // 2 - 1) or (self.rank == self.num_ranks // 2)
```
## Core Function: step
[step](https://github.com/deepseek-ai/DualPipe/blob/main/dualpipe/dualpipe.py#L294) 方法是 `DualPipe` 的核心，它协调了所有 micro-batches 的计算和通信。整个过程被划分为 8 个阶段，以实现最大程度的计算-通信重叠。

输入处理: 只有 rank 0 和 rank N-1 会接收外部输入数据 `inputs` 和 `labels`. 这些数据被 `scatter` (`dualpipe/utils.py`) 切分成 `half_num_chunk` 个 micro-batch 。Rank 0 的输入用于前向流水线，Rank N-1 的输入用于反向流水线。
```python
def step(
        self,
        *inputs: Optional[torch.Tensor],
        num_chunks: int = 0,
        # ...
    ) -> Tuple[Optional[torch.Tensor], Optional[Union[torch.Tensor, Tuple[torch.Tensor]]]]:
        # ...
        # 重置状态
        self._reset_states()

        # 将输入数据切分成 micro-batch
        inputs = scatter(inputs, half_num_chunks, self.batch_dim)
        labels = scatter(labels, half_num_chunks, self.batch_dim)
        if self.is_first_rank:
            self.input_chunks = (inputs, [])
            self.labels = ([], labels)
        elif self.is_last_rank:
            self.input_chunks = ([], inputs)
            self.labels = (labels, [])
        # ...
```
接下来是 8 个核心调度阶段的，在此之前会进行一些准备工作：
- 状态重置: `_reset_states()` 清空上一轮迭代的缓存，如输入/输出块、梯度、损失等。
- rank 确定: 计算 `num_half_ranks`（流水线对折后的一半设备数）和 `half_rank`（当前秩在对折流水线中的位置. 这些变量将决定每个阶段的循环次数。
- 数据分发: `scatter` 函数将输入数据 inputs 和 labels 切分成 half_num_chunks 个 micro-batch 。根据 is_first_rank 或 is_last_rank，将这些 micro-batch 存放到 self.input_chunks 和 self.labels 中。

调度示意图如下图所示，红色线分隔了每个步骤

![DualPipe Schedule](https://share.note.youdao.com/yws/api/personal/file/WEB6ab2cfcee4bad0937e6f0c4c0d225598?method=download&shareKey=754ad9e0092a41faf2c692912283f5ff "DualPipe Schedule")

**Step 1: Warm-up Forward nF0**

这是一个纯前向计算阶段，用于填满流水线。距离流水线中点越远的 rank（half_rank 越小）执行的预热步骤越多。 `_forward_chunk(0)` 被调用，在此函数内部:
  1. `_recv_forward(0)`: 尝试接收前一个 rank 的数据。对于 rank 0 来说，它直接使用 self.input_chunks 的数据，不接收。
  2. `_commit_and_wait_comm()`: 等待数据接收完成。
  3. `_forward_compute_chunk(0)`: 执行 `self.module[0]` 的前向计算。
  4. `_send_forward(0)`: 将计算结果异步地发送给下一个 rank.
```python{linenos=true}
step_1 = (num_half_ranks - half_rank - 1) * 2
for i in range(step_1):
    self._forward_chunk(0)
```
---

**Step 2: Dual Forward nF0F1**

两条流水线都开始执行前向计算。两条流水线都开始工作。当前 rank 不仅继续处理 phase 0 的前向计算，也开始处理从另一端（phase 1）传来的数据的前向计算。
- `_forward_chunk(0, recv=False, ...)` 处理一个 phase 0 的 micro-batch ，但不立即接收下一个，因为前面已经调用了 `_recv_forward(0).`
- `_forward_chunk(1, ...)`: 处理一个 phase 1 的 micro-batch 。

```python{linenos=true}
# Step 2: nF0F1
step_2 = half_rank + 1
self._recv_forward(0)
for i in range(step_2):
    self._forward_chunk(0, recv=False, send=self.is_middle_rank)
    self._recv_forward(0)
    self._forward_chunk(1, send=(not self.is_middle_rank) or (i < step_2 - 1))
    if not self.is_middle_rank:
        self._send_forward(0)
```
--- 

**Step 3: 前向-后向-权重混合阶段 (Zero Bubble) nB1W1F1**

这是 DualPipe 提高效率的关键。当一条流水线开始进行反向计算时，另一条流水线仍在进行前向计算。
- `_backward_chunk(1, enable_zb=True)`: 执行反向计算，并启用 Zero Bubble (ZB) 优化。ZB 通过 `WeightGradStore` 将权重梯度（weight gradients）的计算（通常在反向传播中阻塞）缓存起来，推迟执行，从而让路给其他计算或通信。
- `_weight_chunk()`: 执行被推迟的权重梯度计算。
- `_forward_chunk(1)`: 同时执行另一个方向的前向计算。
```python
# Step 3: nB1W1F1 (Use zero bubble)
step_3 = num_half_ranks - half_rank - 1
for i in range(step_3):
    self._backward_chunk(1, enable_zb=True)
    self._recv_forward(1)
    self._weight_chunk()
    self._forward_chunk(1, recv=False)
```
---

**Step 4: Main Steady State nF0B1F1B0**

这是流水线完全填满后的主循环。在一个循环迭代中，一个 rank 会执行两次计算和通信的重叠操作：一次是（前向计算 + 反向计算），另一次也是（前向计算 + 反向计算）. 这里调用 `_forward_backward_chunk(0, 1)` 和 `_forward_backward_chunk(1, 0)`. 这个函数尝试将一个方向的前向计算（F）与另一个方向的反向计算（B）打包在一起执行，实现 F&amp;B 重叠。

```python
# Step 4 (Main step): nF0B1F1B0
step_4 = half_num_chunks - num_ranks + half_rank + 1
for i in range(step_4):
    # ...
    self._forward_backward_chunk(0, 1)  # i != 0
    self._forward_backward_chunk(1, 0)
```
---

**Step 5 & 6: 后向-后向混合阶段 (Cooldown Backward) nB1F1B0 和 nB1B0**

当前向数据流耗尽后，流水线进入收尾阶段。这个阶段主要执行剩余的反向计算。同样，ZB 优化在后半段被启用，以减少气泡。

```python
# Step 5: nB1F1B0
step_5 = num_half_ranks - half_rank - 1
for i in range(step_5):
    self._backward_chunk(1)
    self._forward_backward_chunk(1, 0)

# Step 6: nB1B0 (The second half of the chunks use zero bubble)
step_6 = half_rank + 1
enable_zb = False
for i in range(step_6):
    if i == step_6 // 2 and half_rank % 2 == 1:
        enable_zb = True
    self._backward_chunk(1, enable_zb=enable_zb)
    if i == step_6 // 2 and half_rank % 2 == 0:
        enable_zb = True
    self._backward_chunk(0, enable_zb=enable_zb)
```
---

**Step 7 & 8: 权重更新收尾阶段 nWB0 和 nW**

- Step 7 将最后的后向计算与权重计算重叠。
- Step 8 是纯粹的权重计算阶段，循环调用 _weight_chunk() 直到 WeightGradStore.funcs_queue 队列为空，确保所有梯度都已计算完毕。

```python
# Step 7: nWB0 (Use zero bubble)
step_7 = num_half_ranks - half_rank - 1
for i in range(step_7):
    self._weight_chunk()
    self._backward_chunk(0, enable_zb=True)

# Step 8: nW
step_8 = half_rank + 1
for i in range(step_8):
    self._weight_chunk()
assert WeightGradStore.funcs_queue.empty()
```

## Computation-Communication Overlap
`_forward_backward_compute_chunk` 函数是实现计算重叠的关键。在理想情况下（如果模型结构支持），它可以将一个 micro-batch 的前向计算和另一个 micro-batch 的反向计算在同一个函数调用中完成。该函数在 step4 使用的`_forward_backward_chunk` 函数中被调用。


```python{linenos=true}
def _forward_backward_compute_chunk(self, phase0: int, phase1: int) -> None:
    # ...
    if not self.overlapped_forward_backward:
        self._forward_compute_chunk(phase0)
        self._backward_compute_chunk(phase1)
        return
    # ...
    # forward & backward
    outputs0, loss0 = type(module0).overlapped_forward_backward(
        module0, inputs0, criterion0, labels0,
        module1, loss1, outputs1, output_grads1,
    )
    # ...
```

如果模型定义了一个 `overlapped_forward_backward` (@classmethod)，DualPipe 就会调用它。在这个方法里，开发者可以自定义前向和后向计算的交错执行顺序，以达到最佳的重叠效果。DeepSeek-v3 的重叠方法在技术报告里已经讲解。

# Real Case
通过 `examples/example_dualpipe.py `中的 main 函数来详细讲解一个完整的 DualPipe 流程。

1. 环境初始化和配置

- 分布式设置: main 函数首先初始化 PyTorch 的分布式通信组（init_process_group），并为每个进程（rank）分配一个 GPU.
- 参数配置: 定义了 micro-batch 数量 (num_chunks)、每个 micro-batch 的大小 (micro_batch_size) 等超参数。
- P2P通信设置: 在执行 DualPipe 的 step 方法前，必须调用 `set_p2p_tensor_shapes` 和 `set_p2p_tensor_dtype` 来告知 DualPipe 在流水线中传递的张量的形状和数据类型。这是因为 DualPipe 需要预先分配内存来接收来自其他 rank 的数据。

```python{linenos=true}
def main(rank, pp_size):
    # 判断当前进程的角色
    is_first_rank = rank == 0
    is_last_rank = rank == pp_size - 1

    # 初始化分布式环境
    dist.init_process_group(backend='nccl', init_method="env://", world_size=pp_size, rank=rank)
    torch.cuda.set_device(rank)
    torch.set_default_device(f"cuda:{rank}")
    torch.manual_seed(233)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # 定义流水线参数
    num_chunks = 20
    micro_batch_size = 3
    seq_len = 256
    hidden_size = 512
    if is_first_rank:
        print(f"{pp_size=}, {num_chunks=}, {seq_len=}, {hidden_size=}", flush=True)

    # 设置P2P通信的Tensor形状和类型
    set_p2p_tensor_shapes([(micro_batch_size, seq_len, hidden_size)])
    set_p2p_tensor_dtype(torch.float32)
```

2. 模型和参考基准的创建

```python{linenos=true}
# 创建一个完整的、未分割的模型
full_modules = nn.Sequential(*[PipelineStage(hidden_size) for _ in range(pp_size)])

# 创建完整的输入数据
full_x = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size)
full_l = torch.randn(num_chunks * micro_batch_size, seq_len, hidden_size)

# 参考步骤：在一个GPU上，用标准的数据并行方式运行完整模型，得到基准结果
loss_ref, output_ref = ref_step(full_x, full_l, full_modules, num_chunks)
```
- 创建模型: 代码首先创建了一个完整的 `nn.Sequential` 模型 (full_modules)，它包含了流水线所有的阶段。
- 参考步骤 (ref_step): 为了验证 DualPipe 的正确性，`ref_step` 函数模拟了标准的、非流水线并行的训练过程。它将数据分块，依次通过完整模型计算损失和输出。`loss_ref` 和 `output_ref` 将作为后续比较的正确答案。

3. DualPipe模型的创建和输入准备
- 模型分割: 每个 rank r 会持有两个 PipelineStage: 一个是 `full_modules[r]`，另一个是 `full_modules[pp_size - 1 - r]`. 这就是 Dual (双向) 的体现。例如，在一个 4-GPU 的设置中：
    - Rank 0 持有 stage 0 和 stage 3 的模型。
    - Rank 1 持有 stage 1 和 stage 2 的模型。
    - Rank 2 持有 stage 2 和 stage 1 的模型。
    - Rank 3 持有 stage 3 和 stage 0 的模型。
- 输入数据分割: DualPipe 有两个数据入口点：rank 0 和最后一个 rank.
    - rank 0 接收前半部分的输入 (`full_x.chunk(2)[0]`) 和 后半部分 的标签 (`full_l.chunk(2)[1]`).
    - last rank 接收后半部分的输入 (`full_x.chunk(2)[1]`) 和 前半部分 的标签 (`full_l.chunk(2)[0]`).

一共有两个数据流: 一个从 rank 0 开始，其对应的标签在最后一个 rank；另一个从最后一个 rank 开始，其对应的标签在 rank 0.

```python{linenos=true}
# DualPipe 模型创建
# 每个 rank 获取两个处于对称位置的模型块
local_full_modules = nn.Sequential(full_modules[rank], full_modules[pp_size - 1 - rank])
local_modules = nn.Sequential(PipelineStage(hidden_size), PipelineStage(hidden_size))
# ... 加载权重 ...
dualpipe_model = DualPipe(local_modules)

# DualPipe输入数据准备
if is_first_rank:
    x = full_x.chunk(2)[0]
    l = full_l.chunk(2)[1]
elif is_last_rank:
    x = full_x.chunk(2)[1]
    l = full_l.chunk(2)[0]
else:
    x = None
    l = None
```

4. 执行训练步骤

调用 `dualpipe_model.step`，触发了前面讲解中提到的复杂的8阶段调度流程。

```python
loss, outputs = dualpipe_model.step(x, num_chunks=num_chunks, criterion=criterion, labels=(l,), return_outputs=False)
```

5. 结果验证

检查损失

```python
if is_first_rank:
    assert torch.equal(loss, loss_ref.chunk(2)[1])
elif is_last_rank:
    assert torch.equal(loss, loss_ref.chunk(2)[0])
else:
    assert loss is None
```
训练步骤完成后，step 方法会返回计算出的损失。
- rank0 计算出的 loss 对应的是从 last rank 输入的数据流，等于参考损失的后半部分 (`loss_ref.chunk(2)[1]`).
- 同理，last rank 计算出的 loss 对应的是从 rank0 输入的数据流，等于参考损失的前半部分 (`loss_ref.chunk(2)[0]`).
- 中间的 ranks 不计算最终损失，返回 None.

检查梯度

```python{linenos=true}
for (p0, p1) in zip(local_modules[0].parameters(), local_modules[1].parameters()):
    # ...
    dist.all_gather_into_tensor(p0all, p0.grad)
    dist.all_gather_into_tensor(p1all, p1.grad)
    # 手动聚合对称rank的梯度
    p0.grad += p1all[pp_size - 1 - rank]
    p1.grad += p0all[pp_size - 1 - rank]
for ((n, p), p_ref) in zip(local_modules.named_parameters(), local_full_modules.parameters()):
    assert cal_diff(p.grad, p_ref.grad) < 1e-13
```
由于每个 rank r 持有 r 和 `pp_size - 1 - r` 两个阶段的模型，如果这两个阶段在逻辑上是同一个权重（例如，在Encoder-Decoder结构中共享权重），那么它们的梯度需要手动聚合。示例通过 `dist.all_gather_into_tensor` 收集所有 rank 上对称模块的梯度，然后手动将它们相加。最后，将聚合后的梯度与 ref_step 中计算出的参考梯度进行比较，验证反向传播的正确性。