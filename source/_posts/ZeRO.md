---
title: ZeRO, ZeRO-Offload, ZeRO-Infinity
date: 2024/12/11 17:25:42
categories: Paper Reading
tags: Distributed Training
excerpt: Paper reading of ZeRO.
mathjax: true
katex: true
---
# ZeRO

Zero 用于优化内存，极大地提高了训练速度，同时增加了可以训练的模型大小。ZeRO 消除了数据和模型并行训练中的内存冗余，同时保持了低通信量和高计算粒度，能够以持续的高效率按设备数量等比例扩展可训练模型的大小。

## Introduction 

ZeRO 首先总结了下当前并行方法存在的问题
- Basic DP: 没有减少每个设备的内存，在 32GB 内存的 GPU 上训练超过 1.4B 参数的模型便会 OOM.
- Model Parallelsim (MP): 切分了每一层的计算和激活到每个设备上，但引入了大量的通信 (前向和反向都需要 2xAll-Reduce)，因此扩展性差，通常只在一个节点内的高带宽连接的 GPU 中进行。在 DGX-2 节点训练 40B 参数的模型每个 V100 GPU 仅能达到硬件峰值的 5% 算力 (5T flops).

模型状态通常占据了训练时的大部分内存，但 DP 在所有数据并行进程中保存一份模型状态，导致冗余内存消耗；而 MP 对这些状态进行切分以获得高内存效率，但通常会导致过于细粒度的计算和昂贵的通信，扩展效率较低。此外，这些方法都静态地维护整个训练过程所需的**整个模型状态**。

[ZeRO-DP](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) 通过在数据并行过程中划分模型状态 (参数、梯度和优化器状态) 消除了数据并行过程中的内存冗余。

结论：如下图所示 ZeRO-DP 有三个主要的优化阶段，它们对应于优化器状态、梯度和参数的划分。
对于使用 FP16 的模型，内存占用包括参数 (FP16)、梯度 (FP16)、Adam 优化器状态 (动量 (FP32)，方差 (FP32) 以及更新后的参数 (FP32), 因此 K=12).
1. 优化器状态划分 (Pos) —— 内存减少 4 倍，需要对梯度进行 reduce-scatter，用各自的优化器状态更新梯度后进行 All-gather 使所有设备都有最新的梯度，通信量与数据并行性相同 (对 Loss 进行一次 All-reduce).
2. 添加梯度划分  (Pos+g) -- 内存减少 8 倍，每个设备需要将自己的梯度 scatter 到负责更新那部分参数的设备上，然后使用 Gather 将其他设备更新后的模型参数同步到自己上面，通信量与数据并行性相同。
3. 添加参数划分 (Pos+g+p) -- 内存减少与数据并行度 Nd 呈线性关系。通信量增加了50%，因为在前向/反向传播中需要每个设备需要额外广播自己存储的模型参数 `2*(N-1)/N*P`，反向传播时需要对发送梯度到对应的设备上 `(N-1)/N*P`.


![Memory Savings and Communication Volume for the 3-stage of ZeRO](https://note.youdao.com/yws/api/personal/file/WEBcfab82173b0f76eb5b3c8396e81e238a?method=download&shareKey=1b8bb86256be5b15bec039beecee062b "Memory Savings and Communication Volume for the 3-stage of ZeRO")

激活、临时缓冲区和不可用的内存片段会成为次要内存瓶颈。作者开发了 ZeRO-R 优化了这三个因素分别消耗的剩余内存。
1. 对于激活 (在前向传播中存储，反向传播中使用)，仅仅使用激活检查点是不够的。ZeRO-R 通过激活划分识别和删除现有 MP 方法中重复存储的激活，并且在适当时候将激活存储在 CPU 中。
2. ZeRO-R 定义了适当大小的临时缓冲区，以实现内存和计算效率的平衡。
3. 由于不同张量的寿命存在差异，ZeRO-R 根据张量的不同生命周期主动管理内存，防止内存碎片。

在某些情况下，MP 仍可以和 ZeRO 一起使用：i）当与 ZeRO-R 一起使用时，MP 可以减少超大模型的激活内存占用。ii）对于较小模型，当单独使用 DP 的 batchsize 太大而无法实现良好的收敛时，MP 也可以带来好处。

## Where Did All the Memory Go?

在模型训练期间，大部分内存被模型状态消耗 (优化器状态、梯度和参数). 除了这些模型状态之外，剩余的内存被激活、临时缓冲区和碎片内存所消耗，称之为剩余状态。

### Model States: Optimizer States, Gradients and Parameters

Adam 优化器需要存储两个优化器状态：时间平均动量和梯度方差来计算更新后的参数。此外，还需要有足够的内存来存储梯度和权重本身。

**混合精度训练 (Mixed-Precision Training)** 中参数和激活以 fp16 格式存储并且在前向和反向传播中也使用 fp16 格式的权重和激活。Adam 优化器存储 fp32 格式的参数副本、动量和方差以保证更新的精度。

假设模型参数量为 ψ，模型参数需要占用 2ψ 字节的内存，反向传播中产生的 fp16 梯度需要占用 2ψ 字节的内存。Adam 优化器存储 fp32 格式的参数副本、动量和方差每个都需要占用 4ψ 字节的内存。因此训练时总共需要 16ψ 字节的内存，为存储模型参数的 8x.

### Residual Memory Consumption

在训练过程中，**激活**会占用大量的内存。基于 transformer 的模型的激活内存占用与层数×隐藏维度×序列长度×批大小成正比。对于类似 GPT-2的结构，总激活约为 12×隐藏亮度×批大小×序列长度×变层数 (`QKV(h*3h) + O(h*h) + MLP(h*4h+4h*h)=12h*h`，没有考虑 mask). 激活重计算可以以 33% 的额外计算开销 (之前是一次前向，一次反向，反向因为需要对输入和参数都进行求导所以计算量是前向的两倍，现在多了一次前向) 换取接近原先激活大小平方级别的内存占用。

对于大型模型，用于存储中间结果的**临时缓冲区**会消耗大量内存。对梯度进行 All-Reduce 或梯度归一化计算等操作倾向于在操作之前将所有梯度融合到单个扁平缓冲区中，以提高吞吐量。

**碎片化内存**会导致即使有足够的内存但没有足够大的连续块进行分配时的 OOM，作者观察到极端情况下在有 30% 剩余内存时也会产生 OOM.

## ZeRO: Insight and Overview

ZeRO有两组优化：ZeRO-DP 旨在减少模型状态的内存占用；ZeRO-R 旨在减少剩余内存消耗。

ZeRO-DP 基于三个关键见解：
1. DP 比 MP 具有更好的扩展效率，因为 MP 减少了计算的粒度，同时也增加了通信开销。
2. DP 内存效率低下，因为模型状态被在所有数据并行进程中都存有一份。
3. DP 和 MP 都保留了整个训练过程中所需的所有模型状态，但并非所有状态在整个训练期间都需要。

ZeRO-DP 划分模型状态，并使用动态通信调度利用模型状态的内在的暂时性，同时最小化通信量。

ZeRO-R 基于两个关键见解：
1. MP 对模型状态进行切分，但通常需要重复存储激活。
2. 对于GPT-2或更大的模型，算术强度 (每次迭代计算量与激活检查点数量的比值) 非常大 (≥10K)，并且随着隐藏维数的增加而线性增加，即使在带宽较低的情况下，也可以隐藏激活检查点的数据移动成本。

ZeRO 通过跨 GPU 划分激活检查点来消除 MP 中的内存冗余，并根据需要使用 All-Gather 来重建；使用恒定大小的缓冲区来避免临时缓冲区随着模型大小的增加而爆炸；通过将激活检查点和梯度移动到预分配的连续内存缓冲区来执行动态内存碎片整理。

## Deep Dive into ZeRO-DP

下表显示了逐渐切分 (1) 优化器状态、(2) 梯度和 (3) 参数冗余后的内存占用。称为ZeRO-DP的三个优化阶段：Pos， Pg和Pp，将在下面详细说明。

<table border="1" cellspacing="0" cellpadding="5">
  <thead>
    <tr>
      <th rowspan="2">DP</th>
      <th colspan="3">7.5B Model (GB)</th>
      <th colspan="3">128B Model (GB)</th>
      <th colspan="3">1T Model (GB)</th>
    </tr>
    <tr>
      <th>Pos</th>
      <th>Pos+g</th>
      <th>Pos+g+p</th>
      <th>Pos</th>
      <th>Pos+g</th>
      <th>Pos+g+p</th>
      <th>Pos</th>
      <th>Pos+g</th>
      <th>Pos+g+p</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>120</td>
      <td>120</td>
      <td>120</td>
      <td>2048</td>
      <td>2048</td>
      <td>2048</td>
      <td>16000</td>
      <td>16000</td>
      <td>16000</td>
    </tr>
    <tr>
      <td>4</td>
      <td>52.5</td>
      <td>41.3</td>
      <td><b>30</b></td>
      <td>896</td>
      <td>704</td>
      <td>512</td>
      <td>7000</td>
      <td>5500</td>
      <td>4000</td>
    </tr>
    <tr>
      <td>16</td>
      <td>35.6</td>
      <td><b>21.6</b></td>
      <td>7.5</td>
      <td>608</td>
      <td>368</td>
      <td>128</td>
      <td>4750</td>
      <td>2875</td>
      <td>1000</td>
    </tr>
    <tr>
      <td>64</td>
      <td><b>31.4</b></td>
      <td>16.6</td>
      <td>1.88</td>
      <td>536</td>
      <td>284</td>
      <td><b>32</b></td>
      <td>4187</td>
      <td>2218</td>
      <td>250</td>
    </tr>
    <tr>
      <td>256</td>
      <td>30.4</td>
      <td>15.4</td>
      <td>0.47</td>
      <td>518</td>
      <td>263</td>
      <td>8</td>
      <td>4046</td>
      <td>2054</td>
      <td>62.5</td>
    </tr>
    <tr>
      <td>1024</td>
      <td>30.1</td>
      <td>15.1</td>
      <td>0.12</td>
      <td>513</td>
      <td>257</td>
      <td>2</td>
      <td>4011</td>
      <td>2013</td>
      <td><b>15.6</b></td>
    </tr>
  </tbody>
</table>

### Pos: Optimizer State Partitioning

设 DP 并行度为 Nd, 每个数据并行进程只需要存储和更新总优化器状态的 1/Nd，然后只更新参数的 1/Nd. 在每个训练步骤结束时，在数据并行进程中执行一次 All-Gather，以获得所有数据并行过程中完全更新的参数。这使得每个设备上保存模型状态需要的内存从 4ψ+Kψ 变成 4ψ+Kψ/Nd，当使用 Adam 优化器 (K=12) 并且 Nd 很大时，内存需求可以降低接近 4x.

### Pg: Gradient Partitioning

由于每个数据并行进程只用更新自己被分配的参数，因此他也只需要那部分参数 reduce 后的梯度。只在负责更新相应参数的数据并行过程中进行 reduce. 完成后它们的内存可以被释放。这使得了梯度所需的内存占用从 2Ψ 字节减少到 2Ψ/Nd. 更新后的参数再被 scatter 到其他进程。

通常为了效率，将需要 reduce 的梯度按照参数的分区划分成多个 buckets，每个 bucket 对应特定的一组参数，对每个 bucket 进行整体 reduce 操作，而不是对单个梯度进行操作。进一步划分梯度后，每个设备上保存模型状态需要的内存进一步减少到 2ψ+(K+2)ψ/Nd

蓝色箭头串起来的白色长方形代表的是 Transformer Block，蓝色的第一行代表 FP16 参数；橙色的第二行代表 FP16 梯度，反向传播时将用于更新参数；绿色的行代表优化器状态 (FP32 的梯度，动量，方差，以及更新后的参数)，其中在计算完 FP16 梯度以后不再需要保存 FP32 参数。同时也需要 buffer 来保存部分 transformer block 的输出激活。

### Pp: Parameter Partitioning

更进一步我们可以将模型参数也进行划分，当设备所没有的参数需要进行向前和向后传播时，通过广播从其他的的数据并行进程接收。通过前面的分析可知这使得通信量变为原来的 1.5x， 但使得所有的模型参数都被划分到每个设备上，只需要 (4+K)/Nd 字节的内存。

### Execution Steps of ZeRO3

![Overview of Memory Consumption](https://note.youdao.com/yws/api/personal/file/WEB1fd41e8b92bcd256a910ce757d4eea21?method=download&shareKey=d82f49f0d59309d987c164a100966895 "Overview of Memory Consumption")

每个 GPU 只需要保存自己部分的 Pos+g+p. 前向传播时保存对应模型参数的 GPU 需要把参数广播到其他 GPU 中，其他 GPU 用自己部分的数据完成前向传播后就可以删除这部分参数 (最后一部分除外). `(N-1)/N*P`

![Broadcast of Model Parameters](https://note.youdao.com/yws/api/personal/file/WEB82b66a8a7b40a2fdf9512545189cc37a?method=download&shareKey=37a4614d85a77d2ca00e6d5a4769e2f9 "Broadcast of Model Parameters")

前向传播完成后，第一次反向传播可以利用最后一次正向传播已经广播了的模型参数，每个 GPU 计算自己部分的梯度，然后 Reduce 到存储对应模型参数的 GPU 中。之后和前向传播一样，每个 GPU 都需要广播自己的参数，然后其他 GPU 用自己的数据完成梯度计算以后 Reduce 到自己的梯度。`(N-1)/N*P + 1/N*G*(N-1)`

![Gradient Accumulation](https://note.youdao.com/yws/api/personal/file/WEB575b3869e59814ae1449351cf1b18d01?method=download&shareKey=10f29390f47227ca8eefbcc00f4fca6e "Gradient Accumulation")

反向传播结束以后，每个 GPU 使用优化器更新自己的 FP32 模型参数后转换成 FP16 格式。

![Update Parameters Locally](https://note.youdao.com/yws/api/personal/file/WEB19cb21dfa63ab76437a2246ff52b00aa?method=download&shareKey=ddb63d57f3bf976b1dce4596e77a2009 "Update Parameters Locally")

## Deep Dive into ZeRO-R

### Pa: Partitioned Activation Checkpointing

一旦计算了模型的一层的前向传播，输入激活将在所有模型并行过程中进行划分，直到在反向传播期间再次需要输入激活。此时，ZeRO 使用一个 All-Gather 操作来重新实现激活的复制副本。称这个优化为 Pa. 将 Pa 与激活检查点结合，只存储分区的激活检查点，这样使得激活占用空间的减少与 MP 并行度成正比。

### CB: Constant Size Buffers

通信的效率不仅仅与数据量相关，还受到固定启动开销和带宽利用率的影响。较大的输入更容易充分利用硬件的带宽和优化机制，因而能显著提高 All-Reduce 操作的效率。因此经常将需要进行通信的数据合并到一个缓冲器。然而，合并缓冲区的内存开销与模型大小成正比，模型过大时容易 OOM. 为了解决这个问题，**当模型很大时，简单地使用一个性能高效的固定大小的合并缓冲区**。

### MD: Memory Defragmentation  

前向传播中只需要保存检查点的激活而丢弃其他激活会产生碎片化内存。同样的反向传播中只需要保存参数的梯度而丢弃激活的梯度也会产生碎片化内存。内存碎片导致两个问题: (1) 即使有足够的可用内存，由于缺乏连续内存导致 OOM. (2) 由于内存分配器花费大量时间搜索连续内存块以满足内存请求而导致效率低下。ZeRO 通过**为激活检查点和梯度预分配连续内存块，并在它们产生时将它们复制到预分配的内存中**，从而实时地进行内存碎片整理。

## Communication Analysis of ZeRO-DP

使用 Pos 和 Pg 时，ZeRO-DP 不会产生额外的通信，同时可以减少高达 8 倍的内存。使用 Pos+g+p 时，ZeRO-DP 最多会产生 1.5 倍的通信，同时减少内存占用为原来的 1/Nd.

在数据并行训练过程中，在计算下一步的更新之前，在反向传播结束时对所有数据并行进程的梯度使用 All-Reduce 进行平均，因此通信量为 2ψ. 使用 Pos+g 时每个设备需要将自己的梯度 scatter 到负责更新那部分参数的设备上，然后使用 Gather 将其他设备更新后的模型参数同步到自己上面，总通信量仍为 2ψ，与数据并行相同。使用 Pos+g+p 时负责该分区的数据并行进程将权重 brocast 给所有数据并行进程 (前向反向各一次)，最后仍需要 Gather 其他进程上更新好的参数，因此总通信量为 3ψ.

## Communication Analysis of ZeRO-R

在使用激活检查点的 Megatron-LM 中，每个 transformer block 在前向传播中执行 2 次大小为 批大小×序列长度×隐藏维度的 All-Reduce 操作，反向传播中执行 2 次同样大小的 All-Reduce 操作，同时激活重计算也需要 2 次同样大小的 All-Reduce 操作。因此每个块的总通信量为 12×序列长度×隐藏维度。

当使用 ZeRO-R 划分激活检查点时，在对每个激活检查点上的反向传播进行前向重新计算之前，需要进行额外的一次 All-Gather 操作。因此，Pa的总通信开销相对于原先 MP 通信量增加了 1/12，但是使得激活内存占用减小到原来的 1/MP_degree.

如果使用了 Pa+cpu，则分区激活检查点将被存储到 CPU，对激活内存需求减少到几乎为零，而代价是与 Pa 相比，需要从 CPU 和内存之间的数据移动增加了 2 倍。

# ZeRO-Offload 

ZeRO-Offload 通过将数据和计算下放到 CPU 来实现大型模型训练。为了保持计算效率，它尽可能减少数据在 GPU 和 CPU 之间的移动，同时最大限度地减少 CPU 的计算时间，并最大限度地节省 GPU 上的内存。

## Introduction

PP, MP 和 ZeRO 等并行技术都需要有足够的 GPU 设备，使得它们的内存之和能够容纳训练所需的模型状态的存储。目前基于注意力的大模型训练的主要内存瓶颈是模型状态，而不是激活。现有的异构训练在两个主要方面受到限制：(i) 几乎所有的训练都利用 CPU 内存，而不是 CPU算力。(ii) 它们主要是为单个 GPU 设计和评估的。

ZeRO-Offload 为了提高计算效率采取的设计原则有三条：(i) 它需要的 CPU 计算量与 GPU 相比减少了几个数量级。(ii) 它最小化了 CPU 和 GPU 之间的通信量，防止了通信成为瓶颈。(iii) 可以证明在实现最小通信量的同时最大限度地节省了 GPU 的内存。

ZeRO-Offload 将梯度，优化器状态和优化器计算卸载到 CPU，而将参数和前向和反向计算保留在 GPU上。这样 CPU 上的计算量为 O(M)，而 GPU 上的计算量则为 O(MB)，其中 M 和 B 分别为模型大小和 batchsize. 因为 CPU 只处理模型参数的更新，而不参与与 batch size 相关的梯度求平均的操作。在大多数情况下，batchsize 较大，因此 CPU 计算不是瓶颈。但是对于较小的 batchsize，CPU 计算可能会成为瓶颈。

## Unique Optimal Offload Strategy

为了确定最佳的卸载策略，ZeRO-Offload 将 DL 训练建模为如下图所示的数据流，并有效地在 CPU 和 GPU 设备之间进行划分。GPU 和 CPU 之间的卸载策略可以使用该图的二分图来表示，这样一个分区中的计算节点将在拥有该分区的设备上执行，分区中的数据节点也存储在拥有该分区的设备上。

![The Dataflow of Fully Connected Neural Networks](https://note.youdao.com/yws/api/personal/file/WEBc55ea0bd058b2e603052658a6cb25aa6?method=download&shareKey=e2474b8904eff6869bcdce3b09e545f6 "The Dataflow of Fully Connected Neural Networks")

由于 CPU 的算力远远低于 GPU，所以前向传播和反向传播 (它们的计算复杂度都是 O(MB)) 必须在 GPU上完成，而其余复杂度为 O(M) 的计算 (如归一化计算、权重更新等) 会被卸载到 CPU 上。

CPU 内存带宽 (100xGB) 至少比 CPU 和 GPU 之间的 PCIe 带宽 (10xGB) 快一个数量级，而 GPU 内存带宽比 CPU 内存带宽 (TB) 快另一个数量级。数据流中的每个节点都是环的一部分。因此，对该图进行任何划分都需要切割至少两条边，每条边的权值至少为 2M，从而总通信量至少 4M (通过仅卸载部分模型状态，可以进一步减少通信量). 因此，为了实现最小的通信量，所有卸载策略必须使得关于 fp32 模型状态操作的生产者和消费者相同。fp16 参数节点必须和 FWD-BWD 节点在一个子图中，因为这两个节点之间的边权值是 4M.

下表显示了最小化通信量情况下的所有有效分区策略所节省的内存。通过将 fp16 梯度和 Update Super 节点放到 CPU 可以实现 8x 的最大内存节省。

| FWD-BWD | p16 | g16 | Update | Memory | Reduction       |
|---------|------|------|--------|--------|-----------------|
| gpu     | gpu  | gpu  | gpu    | 16M    | 1x (baseline)   |
| gpu     | gpu  | cpu  | gpu    | 14M    | 1.14x           |
| gpu     | gpu  | cpu  | cpu    | 4M     | 4x              |
| gpu     | cpu  | cpu  | cpu    | 2M     | 8x              |


综上所述 ZeRO-Offload 在 CPU 上存储所有 fp32 模型状态以及 fp16 梯度，并且还在 CPU 上计算更新后的参数。fp16 的参数保存在 GPU 上，前向和反向计算也在GPU上完成。

## ZeRO-Offload Schedule

在训练过程中，首先通过前向传播计算损失。由于 fp16 参数已经存放在GPU上，因此这部分计算不需要与 CPU 通信。在损失的反向传播过程中，不同设备计算不同参数的梯度。ZeRO-Offload 可以在计算完每个参数后，将这些梯度单独或分组传输到 CPU 内存中。由于梯度是逐层传输的，因此 GPU 上只需要很小的缓冲区来存放每一层的梯度。在反向传播之后，ZeRO-Offload 直接在 CPU 上更新 fp32 参数和优化器状态），并将更新后的 fp32 参数从 CPU 内存复制到 GPU 内存上的 fp16 参数。

![ZeRO-Offload Training Process on a Single GPU](https://note.youdao.com/yws/api/personal/file/WEB810cb8d722c2c9e8e140b80084c47cbe?method=download&shareKey=6ab97f8c087ad948e03121afde1c266a "ZeRO-Offload Training Process on a Single GPU")

在卸载之前进行如上一节所述的划分的主要好处是，对于具有超过 1 个 GPU 的系统，每个数据并行进程只负责更新参数的一个子集。所有数据并行进程的 GPU 到 CPU 的通信量总和保持不变，CPU 资源可以并行使用，共同计算单个权重更新。ZeRO-Offload 在不同 GPU 之间划分梯度和优化器状态，每个 GPU 将其拥有的部分卸载到 CPU 内存中，并在整个训练过程中将其一直保存在那里。在反向传播过程中，在 GPU上使用 reduce-scatter 计算普遍复核一遍梯度，每个 GPU 只将属于其那一部分的平均梯度卸载到 CPU 内存中。然后优化器状态将由每个数据并行进程直接在 CPU 上并行更新。更新后，参数被移回 GPU，然后在 GPU 上执行类似于 ZeRO-2 的 All-Gather 操作来获取所有更新后的参数。

![ZeRO-Offload Data Placement with Multiple GPUs](https://note.youdao.com/yws/api/personal/file/WEB6e4ed2b9f7f37bf8a5f8f9326bb98971?method=download&shareKey=0bf3ca60c8edb9a602a3ae93262c0967 "ZeRO-Offload Data Placement with Multiple GPUs")

## Optimized CPU Execution

1. 作者使用高性能计算技术实现了一个加速版的 CPU Adam 优化器
2. 开发了一个一步延迟参数更新计划，将 CPU 参数更新计算与 GPU 上的前向和反向计算重叠，隐藏了 CPU 执行时间。

### Implementing the CPU Optimizer

作者使用三级并行性来提高 CPU 优化器的性能。
1. SIMD 矢量指令，充分利用 CPU 架构的硬件并行性。
2. 循环展开，一种提高指令级并行性的有效技术，能更好地利用内存带宽。
3. OMP 多线程，可以有效地并行利用 CPU 上的多个内核和线程。

算法的输入为 β₁(动量系数), β₂(RMSProp 的平方梯度衰减系数), α(学习率)，以及梯度，动量，方差和 fp32 参数作为输入。我们还使用了一些特定于实现的参数，如 simd_width 和 unroll_width. Adam 优化器分别发送更新的方差、动量和参数的 fp16 和 fp32 格式到 GPU 和 CPU. 首先将数据读入矢量寄存器。然后，主循环中使用 Fused Multiplication Add 矢量操作。其他操作，如乘法、除法和平方根，也在矢量模式下运行。为了获得最佳性能，使用 AVX512 simd 指令集和基于自动调优结果的 unroll_width=8. 除了 CPU-Adam 优化器之外，还以分块的方式实现了 CPU 到 GPU 的 fp16 参数复制。通过并行化 Adam 计算并将参数复制到 GPU 来重叠 CPU 和 GPU 的执行。**当在 CPU 上处理当前数据块的 Adam 计算时，将先前处理过的数据块的参数写回 GPU.**

![CPU-ADAM Optimizer](https://note.youdao.com/yws/api/personal/file/WEB7e25392bc19670dfd7b80cc9a84a5d73?method=download&shareKey=ac89809b9f51717d9a5429d1cd5d9865 "CPU-ADAM Optimizer")

### One-Step Delayed Parameter Update

下图展示了 Delayed Parameter Update(DPU) 的 ZeRO-Offload 训练的工作流程。
1. 前 N−1 步不使用 DPU 进行训练，避免在梯度变化迅速的早期阶段破坏训练的稳定性。
2. 在第 N 步中，从 GPU 获取梯度，但跳过 CPU 优化步骤，也不更新 GPU 上的 fp16 参数。
3. 在第 N+1 步中，我们使用第 N 步的梯度计算 CPU 上的参数更新，同时使用第 N-1 步更新的参数并行计算 GPU 上的前向和反向。

![Delayed Parameter Update During the Training Process](https://note.youdao.com/yws/api/personal/file/WEBc56f9c814071e1766b9f33833024d829?method=download&shareKey=d5be7d41c0247d8243bddb3452eba75b "Delayed Parameter Update During the Training Process")

# ZeRO-Infinity

ZeRO-Infinity 是一种新的异构系统技术，它利用 GPU, CPU 和 NVMe 内存，在有限的资源上实现前所未有的模型扩展，并且不需要模型代码重构。

目前大型模型训练技术中最先进的是三维并行 (3D parallelism)，它将模型（张量切片）和流水线并行与数据并行相结合。但是 GPU 内存跟不上模型大小的增长。

ZeRO-Infinity 的优势如下
1. 通过同时利用 CPU 和 NVMe 内存，在有限的 GPU 资源上支持大模型训练。
2. 引入了一种称为 *memory-centric tiling* 的 GPU 内存优化技术，以应对 GPU 内存无法一次放下的超大 block 情况。
3. 引入了一种称作 *bandwidth-centric partitioning* 的数据分区策略，用于利用所有设备上的内存带宽，并将其与重叠通信与计算的技术结合。

## MEMORY REQUIREMENTS

**Memory for Model States:** 基于 Transformer 的模型中的参数总数主要取决于隐藏维度 (hd) 和 Transformer 层数 (nl). Transformer block 中的几乎所有参数都来自四个线性层，大小分别为：QKV_Linear(nd,3nd), O_Linear(hd, hd),MLP(hd, 4hd)+(4hd, hd). 因此一个 Transformer block 的参数量约为 **12 x nl x (hd)²**，因此占用的内存大小为 192 x nl x (hd)² 字节。

**Memory for Residual States:** 剩余状态主要由激活内存组成，它取决于模型架构、批处理大小 (bsz) 和序列长度 (seq). 存储激活检查点所需的内存估计为 **2×bsz×seq×hd×nl/ci**，其中 ci(checkpoint interval) 是两个激活检查点之间的 Transformer block 的数量。

**Model State Working Memory (MSWM)** 是在所有模型状态被卸载到 CPU 或 NVMe 之后，在模型中最大的单个算子上执行前向或反向传播所需的最小 GPU 内存。对于基于 Transformer 的模型，最大的算子是将隐藏维度从 hd 转换为 4hd 的线性层，因此 fp32 格式下需要 **4xhdx4hd** 字节的内存。

**Activation Working Memory (AWM):** 是在执行实际的反向传播之前重新计算激活所需的内存，即两个连续激活检查点之间的激活大小 bsz × seq × ci × (16 × hd + 2 × attn_heads × seq) 字节。

## BANDWIDTH REQUIREMENTS

假设没有任何计算和通信重叠的工作负载执行，我们可以使用峰值计算吞吐量 (peaktp)，数据移动带宽 (bw) 及其算术强度 (ait) 来估计训练效率。需要注意 peaktp 不是理论上的硬件峰值，而是在没有任何通信瓶颈的情况下可以达到的峰值。

算术强度 (AIT) 是总计算量与计算所需数据量之比。它描述了每次数据移动的计算量。

1. compute_time = total_computation / peaktp
2. ait = total_computation / total_data_movement
3. communication_time = total_data_movement / bw = total_computation / (ait × bw)
4. efficiency = compute_time / (compute_time + communication_time) = ait x bw / (ait x bw + peaktp)

### Quantifying AIT in DL training

Transformer block 中一次前向传播中的计算量可以近似为输入乘以参数大小 2 × bsz × seq × params. 反向传播则为其 2 倍。如果使用激活检查点则还需要一次额外的前向传播，因此每次迭代的总计算量为 computation_per_iter = 2 × 4 × bsz × seq × parameters = 2 × 4 × 12 × bsz × seq × nl × (hd)² 

**AIT w.r.t. Parameters and Gradients:** 前向和反向过程中模型参数必须从存储位置位置加载到 GPU 寄存器各次。在使用激活检查点的情况下，还需要加载一次，以便在反向传播期间重新计算。此外，梯度必须从 GPU 寄存器存储到其最终位置至少一次。因此总共要移动模型参数 4 次，总计 2 x 4 x parameters 字节。因此关于参数和梯度的计算强度为 seq x bsz.

**AIT w.r.t. Optimizer States:** 优化器状态必须至少读取和写入一次。所以总的数据移动是 2 × optimizer_states，总计 2 × 16 × parameters 字节。因此关于优化器状态的计算强度为 seq x bsz / 4.

**AIT w.r.t. Activation Checkpoints:** 前向传播时必须将激活检查点保存到它们的最终位置，然后在反向传播期间加载激活检查点。因此总数据移动量为 4 × nl/ci × hd × seq × bsz 字节。因此关于激活检查点的计算强度为 24 × hd × ci.

### Bandwidth Requirements

通过前面的分析可知模型状态的计算强度仅取决于批大小和序列长度，激活检查点的计算强度仅取决于存储间隔和模型的隐藏维度大小。下图 a 说明当传输参数和梯度的带宽超过 70 GB/s 时，即使是最小的批处理大小，也可以实现超过 50% 的效率。图 b 说明，传输优化器状态需要近 4 倍的带宽才能达到 50% 的效率。并且**优化器状态更新需要等待所有前向和反向传播结束，不能与计算重叠**。图 c 说明，启用激活检查点后，即使隐藏大小为2K，2 GB/s 的带宽也能够保持 50% 以上的效率。

![Impact of Bandwidth on Efficiency with 70 TFlops of single GPU Peak Throughput](https://note.youdao.com/yws/api/personal/file/WEBfc095fefa5837b449dcafbd0b9441d63?method=download&shareKey=42ab0064e0986bfde564af6e879f5cd6 "Impact of Bandwidth on Efficiency with 70 TFlops of single GPU Peak Throughput")

## ZERO-INFINITY DESIGN OVERVIEW

GPU 集群采用异构内存存储，除了 GPU 内存还拥有 CPU 内存以及比 GPU 内存大 50x, 比 CPU 内存大近 20x 的大规模 NVMe 存储。下图为 ZeRO-Infinity 架构，描述了第一层的反向传递的通信。将划分后的参数从慢速内存移动到 GPU，然后 All-Gather 以形成完整的层。在计算梯度之后，参数被聚合和重新划分，然后卸载到慢速内存中。层用下标表示，DP rank 用上标表示。

![A Snapshot of ZeRO-Infinity Training a Model with 2 Layers on 4 DP Ranks](https://note.youdao.com/yws/api/personal/file/WEB07d94538fa97fe4b478f44d44eb19e79?method=download&shareKey=82f65d6660819491e1bec3688d2715ae "A Snapshot of ZeRO-Infinity Training a Model with 2 Layers on 4 DP Ranks")

**Efficiency w.r.t Parameter and Gradients:** 现有的异构解决方案 (例如 ZeRO-Offload) 要求先将参数从 CPU 移动到拥有这些参数的 GPU，然后再进行广播。这种方式需要在每个 GPU 上使用足够大的 batchsize，以确保通信能被计算掩盖。但这带来了两个问题：
1. 对于超大规模模型，激活的内存占用会过大，甚至超过 CPU 的内存容量。
2. 当扩展到数百甚至上千个 GPU 时，为了实现有效的收敛，实际的 batchsize 会变得过大。

**Efficiency w.r.t Optimizer States:** 与在前向和反向传播期间参数和梯度的产生有先后顺序不同，优化器状态可以同时更新。ZeRO-Infinity 建立在 ZeRO-3 之上，因此在将优化器状态卸载到 CPU 内存时，它还可以利用所有的 GPU 和 CPU 内存带宽以及所有 CPU 算力用于优化器状态更新。然而，使用 NVMe 卸载，需要将数据从 NVMe 传入到 CPU 内存中，再从 CPU 内存返回。由于 CPU 内存有限，必须将数据分块从 NVMe 加载到 CPU 内存，进行计算后再写回 NVMe.

**Efficiency w.r.t Activations:** 在一台 DGX-2 节点上，每个 GPU 可以通过 PCIe 接口以大约 3 GB/s 的速度并行读写数据到 CPU 内存。这使得在隐藏层大小为 8K 或更大时，可以将激活检查点卸载到 CPU 内存的同时保持超过 80% 的效率。

## EFFICIENCY OPTIMIZATIONS

### Bandwidth-Centric Partitioning

在 ZeRO-3 和 ZeRO-Offload 中每层的参数为单个数据并行进程拥有，在需要时将它们广播给其他进程，ZeRO-Infinity 在所有数据并行进程中划分单个参数，并在需要参数时使用 All-Gather. 相较于广播只用到了单个 PCIe 链路将参数从存储位置加载到 GPU，All-Gather 同时使用所有的 PCIe 链路，每条链路传输 1/dp 的参数。

### Overlap Centric Design

访问 NVMe 内存需要三个步骤：(i) 从 NVMe 读取数据到CPU内存 (nc-transfer). (ii) 将数据从 CPU 内存复制到 GPU 内存 (cg-transfer). (iii) 执行 All-Gather 以在所有 GPU 上获得完整参数 (gg-transfer).

ZeRO-Infinity 的通信重叠有两个组件
1. 一个 dynamic prefetcher，在每次迭代期间，跟踪其在算子序列中的位置，并预取未来算子所需的参数。在执行第 i 个操作符之前，prefetcher 可以分别对第 i+3，第 i+2 和第 i+1 个算子所需的参数调用 nc, cg 和 gg-transfer.
2. 一个通信和卸载重叠机制，用于并行执行梯度所需的数据移动和反向计算。将第 i+1 个算子中参数梯度的 Reduce-Scatter 与第 i 个算子的计算重叠，同时将第 i+2 个算子 Reduce-Scatter 划分的梯度传输给 CPU 或 NVMe.