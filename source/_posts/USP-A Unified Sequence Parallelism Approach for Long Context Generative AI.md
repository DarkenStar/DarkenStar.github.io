---
title: USP-A Unified Sequence Parallelism Approach for Long Context Generative AI
date: 2024/11/14 12:07:33
categories: Paper Reading
tags: Distributed Training
excerpt: Paper reading of Pyramid Attention Broadcast.
mathjax: true
katex: true
---
# Abstract

我们结合 DeepSpeed-Ulysses 和 Ring-Attention 提出了一种统一的 SP 方法，该方法对 Transformer 结构和网络硬件拓扑具有更强的鲁棒性。我们使用 SP 对序列长度为 208K 的LLAMA3-8B 模型进行训练，在两个8x A800 节点上实现了 47% 的MFU.

# Introduction

| Model          | # token      |
| -------------- | ------------ |
| Claude         | 100,000      |
| GPT-4          | 128,000      |
| Gemini 1.5 Pro | 10,000,000   |
| OpenAI Sora    | ≥ 1,000,000 |

Sequence Parallelsim (SP) 是一种切分序列维度的方法。当序列长度和计算设备成比例增加时，DeepSpeed-Ulysses 保持恒定的通信量，而 Ring-Attention 通过计算和通信的重叠掩盖 SP 的 P2P 通信成本。DeepSpeed-Ulysses 的 SP 并行度被限制在注意力头数以内，Ring-Attention 由于矩阵乘法的分解降低计算效率。

根据以上内容我们
- 提出了一种融合了 DeepSpeed-Ulysses 和 Ring-Attention 的统一序列并行方法。
- 分析 SP 与 TP, ZeRO & PP 作为四维并行的应用。

# Sequence Parallelism Approaches

SP 的难点在于 在 softmax 之后的矩阵乘法中，序列维度作为一个公共维度。这使得在对序列维度进行切片后，难以对张量进行划分，也难以将计算分布在多个节点上。

## Megatron-SP

Megatron-SP 将原先 Megatron-LM 复制张量上的 AllReduce 操作 (左图) 替换为对切分数据数据上的等效 allgather 和 reduce-scatter 操作 (右图). 然而，如果没有 TP，Megatron-SP 不能独立使用，并且通信量与并行度无关。

![The Principle of Megatron-LM Sequence Parallelism](https://note.youdao.com/yws/api/personal/file/WEBb3988f66e1d6bf026852541435998b42?method=download&shareKey=685d1f41d3f138895b095f213b784c7c "The Principle of Megatron-LM Sequence Parallelism")

## Ring-Attention

Ring-Attention 可以看作 FlashAttention 的一个分布式版本。如下图所示，SP-Ring 采用嵌套的两级以环形方式组织 P2P 通信，其中每个设备同时发送和接收 KV，允许通信重叠计算。

![Ring-Attention](https://note.youdao.com/yws/api/personal/file/WEB0453cc60526b9cd252e451142c4e8c21?method=download&shareKey=ee97a4d72e213996beaa7986e35d1f00 "Ring-Attention")

## DeepSpeed-Ulysses

如下图所示， DeepSpeed-Ulysses 对沿序列为度切分后的 QKVO 进行 All2All 通信，于是这四个张量的切分由序列维 L 变为注意力头维 hc. 因此，每个注意力头的 softmax(QK.T)V 的计算是完整的，并且可以使用 FlashAttention 来计算。

![DeepSpeed-Ulysses](https://note.youdao.com/yws/api/personal/file/WEB26e98b9cbb56a7a352893171376f0977?method=download&shareKey=dfd14921c261b92697b6029e35a57f97 "DeepSpeed-Ulysses")

# Unified Ulysses-Ring Sequence Parallelism

前文已经说过 DS-Ulysses 的最大并行度不能超过注意力头的数目，并且不适合 GQA (Group Query Attention) 和 MQA (Multi-Query Attention). 另外由于 TP 也需要切分注意力头维度，使得 DS-Ulysses 不能与 TP 一起使用。

而 Ring-Attention 由于矩阵乘法的分解降低计算效率。并且如果是因果注意力计算，其面临着负载不均衡的问题。解决方案是沿着序列维度重新排序输入序列，如下图所示。将序列为度划分成 2*ring_degree 份，并且按着 0 -> 1 -> ...-> ring_degree-1 -> ring_degree-1 -> ring_degree-2 -> ... -> 0 的顺序进行分配。

![Load Balance Sequence Segment](https://note.youdao.com/yws/api/personal/file/WEB18be587d0ab7f5b05c4254ea42258986?method=download&shareKey=073ac22ba8509a0d8b099484807648ef "Load Balance Sequence Segment")

![Loading Balancing for SP-Ring](https://note.youdao.com/yws/api/personal/file/WEB06a9ddd1da328e47b4d874deeddb8108?method=download&shareKey=65a76075f6b18f36beaf8ab48ac3ce89 "Loading Balancing for SP-Ring")

我们将 SP-Ring 和 SP-Ulysses 以一种称为 **USP-Attention** 的混合并行方式组织在一起，以划分序列维度。SP 进程组可以看作是一个 2D mesh，SP-Ring 在网格的每一列上运行，SP-Ulysses 在每一行上运行。

如下图所示前向传播时，scatter_idx=1，gather_idx=2. AllToAll4D 合并 QKV 的维度 L 并划分维度 hc，输出张量的形状为 (hc/N, bs， L, d). 对于逐一计算后的输出 O 沿着 L 维划分并合并 hc维。在反向传播过程中，scatter_idx=2，gather_idx=1.

![Unified Sequence Parallelism Attention Implementation](https://note.youdao.com/yws/api/personal/file/WEB31cd7f99897382ca3dfec21e769dff6a?method=download&shareKey=3b650182d6be62fdd449af4eb27dc890 "Unified Sequence Parallelism Attention Implementation")

USP-Attention 的通信模式特别适合于异构通信网络。如下图所示，可以允许 All2All 操作在高带宽互连中运行，而异步 P2P 通信在低带宽部分运行。

![The Unified-SP is more robust to network hardware topology](https://note.youdao.com/yws/api/personal/file/WEB8dcb2a56c474818171428b41a1382147?method=download&shareKey=61251486eb0b70c612b74889ddc7e5e9 "The Unified-SP is more robust to network hardware topology")

{% note primary %}
**TIP 1:** 建议使用 Unified-SP 来代替 SP-Ring 和 SP-Ulysses，因为它包含了两者的功能，同时提供了额外的优势。
{% endnote %}
# SP in 4D Parallelism

本节将分析 SP 与 DP, TP 和 PP 之间的关系，并讨论涉及 SQ 的 4D 并行设计的最佳方案。下表分析了不同并行度下标准 Transformer block 的通信和内存成本。Communications 的 Params columns 列的表示对 Transformer block 的参数和梯度进行集合通信操作。它包括 self-attention 中 4 个线性层的权重和偏置的参数/梯度，以及 FFN 层的 2 个线性层，在GPT-2 模型中总计 12×O(d²) 个元素。

该表是使用 fp16 (bf16) 的混合精度训练。模型参数和梯度的内存需求分别为 P 字节和 G 字节。在使用 Adam 优化器进行训练时，优化器状态 (Optimizer States, OS) 占用的显存大约是模型参数 (fp16) 显存的 6 倍。这是因为 Adam 优化器在计算梯度时，需要额外存储一些中间变量，如参数本身的 fp32 副本、动量和方差。峰值激活的内存需求是 A 字节,并行度为 N.


<table border="1">
  <tr>
    <th rowspan="2"></th>
    <th colspan="4" style="text-align: center;">Communication (FWD+BWD)</th>
    <th rowspan="2">Split Dim</th>
    <th colspan="3" style="text-align: center;">Memory</th>
  </tr>
  <tr>
    <th>Param</th>
    <th>Cost</th>
    <th>Act</th>
    <th>Cost</th>
    <th>P/G</th>
    <th>OS</th>
    <th>Act</th>
  </tr>
  <tr>
    <td>SP-Ulysses</td>
    <td>allreduce</td>
    <td>12O(d²)</td>
    <td>8*all2all</td>
    <td>(8/N)O(bs*L*d)</td>
    <td>hc/L</td>
    <td>P+G</td>
    <td>6P</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>SP-Ring</td>
    <td>allreduce</td>
    <td>12O(d²)</td>
    <td>P2P</td>
    <td>4O(bs*L*d)</td>
    <td>L/L</td>
    <td>P+G</td>
    <td>6P</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>DP</td>
    <td>allreduce</td>
    <td>12O(d²)</td>
    <td>0</td>
    <td>0</td>
    <td>bs / bs</td>
    <td>P+G</td>
    <td>6P</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>ZeRO1</td>
    <td>allgather + reducescatter</td>
    <td>12O(d²)</td>
    <td>0</td>
    <td>0</td>
    <td>hc/L</td>
    <td>P+G</td>
    <td>6P/N </td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>SP-Unified + ZeRO1</td>
    <td>allgather + reducescatter</td>
    <td>12O(d²)</td>
    <td>P2P + 8*all2all</td>
    <td>≤ 4O(bs*L*d)</td>
    <td>hc/L</td>
    <td>P+G</td>
    <td>6P/N</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>SP-Unified + ZeRO2</td>
    <td>allgather + reducescatter</td>
    <td>12O(d²)</td>
    <td>P2P + 8*all2all</td>
    <td>≤ 4O(bs*L*d)</td>
    <td>hc/L</td>
    <td>P+(G/N)</td>
    <td>6P/N</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>SP-Unified + ZeRO3</td>
    <td>2*allgather + reducescatter</td>
    <td>18O(d²)</td>
    <td>P2P + 8*all2all</td>
    <td>≤ 4O(bs*L*d)</td>
    <td>hc/L</td>
    <td>(P+G)/N</td>
    <td>6P/N</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>TP</td>
    <td>0</td>
    <td>0</td>
    <td>4*allreduce</td>
    <td>8O(bs*L*d)</td>
    <td>hc/d</td>
    <td>(P+G)/N</td>
    <td>6P/N</td>
    <td>αA</td>
  </tr>
  <tr>
    <td>TP-sp</td>
    <td>0</td>
    <td>0</td>
    <td>6*allgather + 4*reducescatter</td>
    <td>10O(bs*L*d)</td>
    <td>hc/d</td>
    <td>(P+G)/N</td>
    <td>6P/N</td>
    <td>A/N</td>
  </tr>
</table>


**Data Pallelsim (DP):** SP 和 DP 在反向传播过程中都需要对梯度进行 allreduce 操作。但 SP 为激活引入了额外的通信开销。当 ulysses_degree 大于 1 时，all2all 操作使得 SP 的通信开销将大于 DP. 当使用Ring-Attention 时，尽管额外的 P2P 通信能被计算掩盖的，但它引入了额外的性能问题。最理想的情况是在计算 attention 时不需要额外的通信，性能与 DP 相同。内存方面，SP 和 DP 都可以将激活占用空间减少为原来的 1/N.

{% note primary %}
**TIP 2:** 建议优先使用 DP 而不是 SP. 只有当 batch size 不足以进行划分时，才应该考虑是否使用 SP.
{% endnote %}

**ZeRO:** ZeRO 是一种分布式参数管理方法，通过对多个设备上的优化器状态 (ZeRO-1)，梯度 (ZeRO-2) 和参数 (ZeRO-3) 进行切分，减少了每个计算设备的存储空间需求到原先的 1/N. 它也可以在 SP 进程组中操作，因为沿着批处理维度 (bs) 或序列维度 (L) 进行划分与 ZeRO 的方法相同。

{% note primary %}
**TIP 3: 建议在使用 SP 时与 ZeRO-1/2 结合使用。**也可以考虑使用 ZeRO-3 和 Offload技术来权衡通信成本以节省内存。
{% endnote %}

**Tensor Pallelsim (TP):** Megatron-LM 提出的 TP 方法将模型的参数在计算设备之间进行切分。并非所有的激活张量都被划分并分布在多个计算设备上。因此，激活的内存成本在表中用 αA 表示。Megatron-SP 用一个 allgather 和一个 reducescatter 取代了原先 TP 中的一个allreduce，将激活内存成本降低到 A/N. 通信量并不会随着并行度改变。另外使用 GQA/MQA 可以降低 SP 的通信成本，而 Megatron-SP 的通信成本保持不变。

{% note primary %}
**TIP 4:** SP 在大规模通信成本方面比 Megatron-SP 有优势。GQA 可以进一步降低 SP 的通信成本。
{% endnote %}

在内存占用方面， 即使 SP 采与 ZeRO-1/2 结合使用，Megatron-SP 占用量也更小。但当 SP 采用 ZeRO3 时，其内存占用与 Megatron-SP 相似。SP-Ulysses+ZeRO3 正是 DS-Ulysses 作者所采用的扩展序列长度的策略。

但 SP 仍能以较高的并行度扩展序列长度。当序列很长时，参数通信量占总通信量的比例相对较小。因此，ZeRO 引入的 allgather 操作的额外通信开销影响有限。

{% note primary %}
**TIP 5:** 单纯在训练中将 Megatron-SP 切换为 SP 并不能增加序列长度。但 SP+ZeRO3 可以达到与 Megatron-SP 近似的训练长度。
{% endnote %}

{% note primary %}
**TIP 6:** 建议使用更高的 SP 并行度，当注意力头数量有限时，可能需要设置较大的 ring_degree，以便在更多的计算设备上训练长序列，这是 Megatron-SP 方法无法实现。
{% endnote %}

**Pipeline Parallelism (PP):** PP 将模型所有的 Trasformer blocks 划分成多个部分，SP 在每个 Trasformer block 内部划分张量。因此，SP 和 PP 是完全兼容的。

{% note primary %}
**TIP 7:** 在四维混合并行中，进程组维度从小到大依次为 TP, SP-Ulysses, SP-Ring, ZeRO-DP, PP.
{% endnote %}

# Experiments

## Performance of Unified SP

在 L20 PCIe GPU 集群上进行 llama3-8B 的推理以评估 SP-Unified 性能，指标为 iter/sec. 满足 ulysses_degree * ring_degree = 8. Load Balacing Ring (lb-ring) 的性能优势随着序列长度增加而增加。

| group_num | bs | seqlen | head_num | head_size | ulysses_degree | basic-ring | lb-ring |
|-----------|----|--------|----------|-----------|----------------|------------|---------|
| 4         | 1  | 8K     | 32       | 128       | 8              | 57.346     | 57.098  |
| 4         | 1  | 8K     | 32       | 128       | 4              | **153.134** | **152.189** |
| 4         | 1  | 8K     | 32       | 128       | 2              | 415.5      | 454.93  |
| 4         | 1  | 8K     | 32       | 128       | 1              | 358.595    | 361.969 |
| 4         | 1  | 32K    | 32       | 128       | 8              | 15.229     | 14.262  |
| 4         | 1  | 32K    | 32       | 128       | 4              | 28.584     | **32.818** |
| 4         | 1  | 32K    | 32       | 128       | 2              | 44.348     | 62.754  |
| 4         | 1  | 32K    | 32       | 128       | 1              | 40.478     | 58.377  |
| 4         | 1  | 128K   | 32       | 128       | 8              | 2.563      | 2.586   |
| 4         | 1  | 128K   | 32       | 128       | 4              | 3.217      | **4.235** |
| 4         | 1  | 128K   | 32       | 128       | 2              | 3.399      | 5.476   |
| 4         | 1  | 128K   | 32       | 128       | 1              | 3.131      | 5.186   |

当在 8x A100-SXM4 NVLink 节点上进行上面设置的实验时，当 ulysses-degree=8 时，32K 和 128K 序列长度的吞吐量都达到最高。这也验证了尽管 SP-Ring 通信开销可以通过与计算的重叠来隐藏，但它会导致计算效率的降低。

| group_num | bs | seqlen | head_num | head_size | ulysses_degree | basic-ring | lb-ring  |
|-----------|----|--------|----------|-----------|----------------|------------|----------|
| 4         | 1  | 32K    | 32       | 128       | 8              | 135.569    | **136.375** |
| 4         | 1  | 32K    | 32       | 128       | 4              | 103.525    | 132.979 |
| 4         | 1  | 32K    | 32       | 128       | 2              | 91.365     | 132.979 |
| 4         | 1  | 32K    | 32       | 128       | 1              | 81.985     | 113.79  |
| 4         | 1  | 128K   | 32       | 128       | 8              | 2.782      | **2.785**  |
| 4         | 1  | 128K   | 32       | 128       | 4              | 2.024      | 2.771   |
| 4         | 1  | 128K   | 32       | 128       | 2              | 1.73       | 2.89    |
| 4         | 1  | 128K   | 32       | 128       | 1              | 1.628      | 2.91    |

## End-to-End SP Performance in Megatron-LM

我们在 Megatron 我们对 DP 和 SP 都使用 ZeRO-1，并对张 Megatron-SP 应用 Sp 优化。不使用梯度累积或激活重计算。实验设置为两个 GPU 节点，每个节点配备 8x A800 GPU，通过 400GB/s NVLink 连接，通过 1.6 Tbps RDMA 进行节点间通信。我们考虑因果注意力中下三角的 MFU.

## SP vs. DP

我们已经将 SP-Unified 方法整合到 Megatron-LM 中。目前 Megatron-LM 只有 SP-Ring 的初步版本，缺乏 SP-Ulysses 的实现。

我们比较了在相同 LLAMA2-7B 工作负载下，在单个 A800 GPU 节点上 SP 和 DP 的性能。batch size 为 8. SP-Unified 在单个节点中通常在 ulysses-degree=8 时有最佳性能。如下图所示，DP 在各种输入序列长度上优于 SP-Unified，这证实了我们在提示2中的结论。

## Hybrid SP and TP

下表给出了 llama2-7B 在 8xA800 GPU 80GB 单节点上的性能。当 seqlen=64K 时，SP-only 会出现 OOM 问题，这验证了 SP 的内存效率低于 Megatron-SP. 

当 global-bs=16, sequence=30K 时，SP-ulysses 的性能最优，明显优于其他 SP 和 TP 混合策略。在吞吐量方面，它也比 TP-sp-only 好 26%. 这表明，尽管 SP-Ulysses 和 Megatron-SP 的通信成本相似 (**???明明不一样**)，但在 NVLINk 连接的情况下，SP-Ulysses 的实际通信效率高于 Megatron-SP.

| seqlen | global-bs | tp-degree | ulysses-degree | ring-degree | FLOPS/GPU | MFU  |
|--------|-----------|-----------|----------------|-------------|-----------|------|
| 64K    | 1         | 4         | 2              | 1           | **154.49** | **0.50** |
| 64K    | 1         | 4         | 1              | 2           | 151.40    | 0.49 |
| 64K    | 1         | 8         | 1              | 1           | 141.85    | 0.45 |
| 30K    | 16        | 2         | 4              | 1           | 155.98    | 0.50 |
| 30K    | 16        | 2         | 1              | 4           | 147.77    | 0.47 |
| 30K    | 16        | 4         | 2              | 1           | 150.05    | 0.48 |
| 30K    | 16        | 1         | 8              | 1           | **163.42** | **0.52** |
| 30K    | 16        | 1         | 1              | 8           | 142.16    | 0.46 |
| 30K    | 16        | 8         | 1              | 1           | 129.12    | 0.41 |

我们采用 TP 和 SP-Unified 的混合并行策略对 LLAMA3-8B 跨两个节点的训练吞吐量进行测试，因为 LLAMA3-8B 只有 8 个头，所以 ulysses-degree 和 TP-degree 的最大乘积为 8.

序列长度为 64K 和 80K，TP-degree=8 时，可以增加 TP+SP 的 global-bs. 然而，SP-only 在使用 global-bs=2 时总是存在 OOM 问题。将 global-bs 增加到 2，在序列长度为 64K 时，TP+SP 的吞吐量提高 2.7%. 但在序列长度为 80K 时，global-bs=2 的 TP+SP 的吞吐量仍然比 global-bs=1  的 SP-only 差。当序列长度达到 120K 时 两个TP+SP 混合设置的 MFU 为 0.47 和 0.48，非常接近最佳值。

| seqlen | global-bs | tp-degree | ulysses-degree | ring-degree | FLOPS/GPU | MFU  |
|--------|-----------|-----------|----------------|-------------|-----------|------|
| 64K    | 1         | 1         | 8              | 2           | 136.31    | 0.44 |
| 64K    | 1         | 1         | 4              | 2           | **137.48** | **0.44** |
| 64K    | 1         | 2         | 4              | 2           | 129.44    | 0.41 |
| 64K    | 1         | 1         | 2              | 8           | 121.83    | 0.39 |
| 64K    | 1         | 8         | 1              | 2           | 129.75    | 0.42 |
| 64K    | 1         | 4         | 2              | 2           | 122.45    | 0.39 |
| 64K    | 1         | 2         | 4              | 2           | 87.67     | 0.28 |
| 64K    | 1         | 4         | 1              | 4           | 89.35     | 0.29 |
| 64K    | 1         | 2         | 2              | 4           | 122.57    | 0.39 |
| 64K    | 1         | 2         | 1              | 8           | 101.35    | 0.32 |
| 64K    | 2         | 8         | 1              | 1           | 141.20    | 0.45 |
| 80K    | 1         | 1         | 8              | 2           | 147.46    | 0.47 |
| 80K    | 1         | 1         | 4              | 4           | **148.90** | **0.48** |
| 80K    | 1         | 1         | 2              | 8           | 140.13    | 0.45 |
| 80K    | 1         | 1         | 1              | 16          | 132.86    | 0.43 |
| 80K    | 1         | 8         | 1              | 2           | 136.16    | 0.44 |
| 80K    | 1         | 4         | 2              | 2           | **137.49** | **0.44** |
| 80K    | 1         | 2         | 4              | 2           | 111.05    | 0.36 |
| 80K    | 1         | 4         | 1              | 4           | 110.81    | 0.36 |
| 80K    | 1         | 2         | 2              | 4           | 130.27    | 0.42 |
| 80K    | 1         | 2         | 1              | 8           | 121.14    | 0.39 |
| 80K    | 2         | 8         | 1              | 1           | 144.40    | 0.46 |
| 120K   | 1         | 4         | 2              | 2           | **152.51** | **0.49** |
| 120K   | 1         | 2         | 4              | 2           | 136.63    | 0.44 |
| 120K   | 1         | 8         | 1              | 2           | 145.92    | 0.47 |
| 120K   | 1         | 4         | 1              | 4           | 150.96    | 0.48 |


我们在 2 个 8xA800 NVLink 节点上探索了序列长度的上界，结果如下表所示。由于 Megatron-SP 具有更高的内存效率，因此对于训练最长的序列，并行度限制为 8 个注意力头数的情况下应全部分配给 Megatron-SP.

<table border="1">
  <tr>
    <th>seqlen</th>
    <th>global-bs</th>
    <th>tp-degree</th>
    <th>ulysses-degree</th>
    <th>ring-degree</th>
    <th>FLOPS/GPU</th>
    <th>MFU</th>
  </tr>
  <tr style="background-color: #D3D3D3;">
    <td>160K</td>
    <td>1</td>
    <td>4</td>
    <td>2</td>
    <td>2</td>
    <td>158.64</td>
    <td>0.51</td>
  </tr>
  <tr>
    <td>160K</td>
    <td>1</td>
    <td>8</td>
    <td>1</td>
    <td>2</td>
    <td>156.63</td>
    <td>0.50</td>
  </tr>
  <tr>
    <td><b>208K</b></td>
    <td>1</td>
    <td>8</td>
    <td>1</td>
    <td>2</td>
    <td>147.26</td>
    <td>0.47</td>
  </tr>
  <tr style="background-color: #D3D3D3;">
    <td>160K</td>
    <td>1</td>
    <td>4</td>
    <td>1</td>
    <td>4</td>
    <td>159.37</td>
    <td>0.51</td>
  </tr>
  <tr style="background-color: #D3D3D3;"> 
    <td>190K</td>
    <td>1</td>
    <td>4</td>
    <td>1</td>
    <td>4</td>
    <td><b>157.08</b></td>
    <td><b>0.50</b></td>
  </tr>
</table>

 

## Convergence 

我们使用相同的数据集比较了 USP 和 DP 的收敛性差异，在 4 个 GPU 上测试了 10K 次迭代的损失函数曲线。我们发现 USP 和 DP 的曲线完全重合。