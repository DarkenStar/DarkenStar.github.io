---
title: DSP-Dynamic Sequence Parallelism For Multidimensional Transformers
date: 2024/11/13 00:12:13
categories: Paper Reading
tags: Diffusion Model Inference
excerpt: Paper reading of DSP.
mathjax: true
katex: true
---
# Abstract

克服多维 Transformer 缩放成长序列的大内存需求和慢推理速度的缺点需要序列并行。所有现有的方法都属于嵌入式序列并行的范畴，这些方法仅限于沿着单个序列维进行分片，引入了大量的通信开销。动态序列并行 (DSP) 是一种新的序列并行，采用高效的重分片策略，根据计算阶段在各序列之间动态切换并行维度。

![Comparison of Embedded and Dynamic Sequence Parallelism](https://note.youdao.com/yws/api/personal/file/WEB8c7e10437124cadc4cdb12571cdb5597?method=download&shareKey=48b5a09ed5fa36fe19aa02984915f224 "Comparison of Embedded and Dynamic Sequence Parallelism")
# Introduction

在视频生成模型中，时空注意力比自注意力更受青睐。与嵌入式序列并行相比，DSP 有如下优势：

1. 高效通信：由于其简化的通信模式和减少的交换频率，DSP 的通信成本显着降低。
2. 适应性：DSP 可以无缝适应大多数模块，无需进行特殊修改，使用限制很少。
3. 易用性：DSP 非常容易实现。

DSP 相比于 SOTA 的嵌入式序列并行方法 实现了从 32.2% 到 10 倍的端到端吞吐量改进，并将通信量减少了至少 75%.

# Background and Related Work

文中使用符号及其含义。

| Symbol | Description            | Symbol           | Description                                  |
| ------ | ---------------------- | ---------------- | -------------------------------------------- |
| B      | batch sizes 大小       | N                | GPU 个数                                     |
| C      | 隐藏层维度             | n                | The*n*-th GPU                              |
| Sᵢ    | 第 i 个序列维度        | **X**      | 一个多维序列张量                             |
| sᵢ    | 序列沿着*Sᵢ* 维切分 | **Xₚ,ₙ** | **X** 被分配到第 n 个 GPU 上的部分 |
| ŝ     | 表示序列没有被切分     | M                | 序列张量的大小                               |

## Background

Transformer 每一层都由多头注意 (Multi-Head Attention) 和前馈网络 (Feed-Forward Network) 组成。MHA 包括 H 个独立参数化的注意头，其公式为

{% mathjax %}
\text{MHA}(x)=\text{Concat}(\text{head}_1,\ldots,\text{head}_H)\mathbf{W}^O,\quad\text{head}_i=\text{Att}(\mathbf{Q}_i,\mathbf{K}_i,\mathbf{V}_i),
{% endmathjax %}

{% mathjax %}
\operatorname{Att}(\mathbf{Q},\mathbf{K},\mathbf{V})=\operatorname{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{\prime}}{\sqrt{d_{k}}}\right)\mathbf{V},\quad x_{\mathrm{MHA}}=\operatorname{Laver}\text{Norm}(x+\operatorname{MHA}(x)),
{% endmathjax %}

{% mathjax %}
\text{FFN}(x)=\max(0,x\mathbf{W}_1+\mathbf{b}_1)\mathbf{W}_2+\mathbf{b}_2,\quad x_{\mathrm{out}}=\text{Layer}\mathrm{Norm}(x_{\mathrm{MHA}}+\text{FFN}(x_{\mathrm{MHA}})),
{% endmathjax %}

多维 Transformer 在多个序列维度上都进行自注意力计算。下图展示了 2D-Transformer 的一个例子。

![llustration of Multi-Dimensional (2D) Transformer](https://note.youdao.com/yws/api/personal/file/WEBd5c543ff3af63af24c1604b97dd50496?method=download&shareKey=36eeb916dc881932021a4ee8c021f99c "llustration of Multi-Dimensional (2D) Transformer")

对于输入 {% mathjax %} \mathbf{X}\in\mathbb{R}^{[B,S_{1},S_{2},\ldots,S_{K},C]} {% endmathjax %}多维 Transformer 公式如下：

{% mathjax %}
\mathbf{X}_{\mathrm{reshape}}=\mathrm{Reshape}(\mathbf{X},[B\times\prod_{j\neq i}S_{j},S_{i},C]).
{% endmathjax %}

然后沿着第 i 个序列维度进行 Transformer 运算：

{% mathjax %}
\mathbf{X_{\mathrm{out}}}=\text{transformer block}(\mathbf{X_{\mathrm{reshape}}}).

{% endmathjax %}

在对所有 K 个维度进行 Transformer 运算后，最终输出张量与输入具有相同的形状。

## Related Work

- Ring Attention 使用环状 P2P 通信划分序列维度，跨 GPU 传输 Key 和 Value.
- Megatron-SP 为了在 Transformer block 中的张量并行性和序列并行性之间转换，引入了额外的 all-gather 和 reduce-scatter 操作。
- DeepSpeed-Ulysse 在进行 attention 之前沿着隐藏层维度进行 QKV 的 all-to-all 通信，计算完成后沿着序列维度进行 O 的 all-to-all 通信。

Ring Attention 对 P2P 通信的依赖在高延迟环境中效率较低。后两者并行维度受到注意力头数量的限制。此外，这些序列并行方法都是针对单个序列维的并行性而设计的。

# Dynamical Sequence Parallelism

## Overview

如下图所示，DSP 的关键在于在计算阶段的交叉处动态切换并行维度。通过仅在计算阶段之间而不是在它们内部动态地重新切分允许 DSP 保持独立于模块内的计算逻辑。对于涉及所有序列维度的操作，包括模型的开始和结束，DSP通过split 和 gather 操作来处理它们。

![System Overview of Dynamic Sequence Parallelism](https://note.youdao.com/yws/api/personal/file/WEB80642cf0cf346f619037ffffcbc7663f?method=download&shareKey=d9edf3d3de6f5f919d652827940249a5 "System Overview of Dynamic Sequence Parallelism")
## Problem Definition

序列并行是跨多个 GPU 分配激活计算，以减少长序列造成的内存开销。目标是在优化 GPU 资源利用率的同时实现最小化总体计算开销的平衡。然而，这种方法会在 GPU 之间产生额外的通信成本。我们的目标是在多维转换环境中优化这种权衡。

{% mathjax %}
\min_p\sum_{n=1}^N\text{CommCost}(\mathbf{X}_{p,n})\:s.t.\:\text{Memory}(\mathbf{X}_{p,n})<Capacity.
{% endmathjax %}

## Dynamic Primitives

DSP 的主要的动态切换原语如下表所示。

| Source Shard | Target Shard | Primitives | Comm Operation | Comm Volume | Freq |
|--------------|--------------|------------|----------------|-------------|------|
| sᵢ          | sᵢ          | /          | /              | /           | High |
| sᵢ          | sⱼ          | Switch     | all-to-all     | M/N         | Low  |
| ŝ            | sᵢ          | Split      | /              | 0           | Low  |
| sᵢ          | ŝ            | Gather     | all-gather     | M           | Low  |

将切分维度从 i 切换到 j 的操作可以表示为

{% mathjax %}
\mathbf{Y}=\text{DynamicSwitch}(\mathbf{X},i,j),
{% endmathjax %}

![Illustration of Dynamic Switch](https://note.youdao.com/yws/api/personal/file/WEBc14df3fefbe8b92ecfc846e817584212?method=download&shareKey=d50e06dddb8b01c47c09a0d0c31808da "Illustration of Dynamic Switch")

Split 和 Gather 操作用于切分和不切分状态之间的转换。尽管它们涉及更多的通信，但因为主要用于大多数网络的开始和结束，以及在非常罕见的情况下用于一些全局操作，其成本可以忽略不计。

## Adaptability and Flexibility

由于 DSP 与模块计算解耦，使其能与各种 Transformer 变体兼容。此外，DSP 也能和传统的数据并行以及更复杂的方法，如 ZeRO 和流水线并行一起使用。

# Technology Analysis

模型选择的是 2D-Transformer 的 OpenSora 变体，删除了其特定的交叉注意力模块。因此，在每一层中，只有两个 Transformer block 分别处理两序列的两个维度为时间 t 和空间 s. Baselien 选择 DeepSpeed-Ulysses Megatron-SP 和 RingAttention.

## Communication Analysis

设激活大小为 M，序列并行大小为N.
- Megatron-SP 使用 2 个 all-gather 操作来聚合整个序列，并使用 2 个 reduce-scatter 操作来将结果分发到一个 Transformer 块的注意力层和MLP层。共有 8 个集合通信操作，总通信量为8M.
![Megatron-SP](https://note.youdao.com/yws/api/personal/file/WEB478ff6b36530710ba4405652e75ca78c?method=download&shareKey=f431c9c53db9c0e920da1de9b34c5485 "Megatron-SP")

- DeepSpeed-Ulysses 在 Temporal block 中为 QKVO 的转换转换引入 4 个通信操作。因此，对于大小为 M 的跨 N 个 GPU 的 all-to-all 通信，每个设备通信量为 4M/N.
![DS-Ulysses](https://note.youdao.com/yws/api/personal/file/WEB19bf064915e4efa2ba3c19d0cdc80205?method=download&shareKey=b5c0b3343723fd427cb5d36862f4678f "DS-Ulysses")

- Ring-Attention 还需要在 Temporal block 中 传递整个 Key 和 Value，总通信量为 2M.
![Ring-Attention](https://note.youdao.com/yws/api/personal/file/WEBb81935a636493d5c7045a5b938147205?method=download&shareKey=78c32f3e10471c3e634a1eefd2ea8da6 "Ring-Attention")

- DSP 通过在每层总共两个 block 中只使用两个 all-to-all 操作，每个设备通信量为 2M/N.
![DSP](https://note.youdao.com/yws/api/personal/file/WEB19a74ed76244be5d9adb709f58869a86?method=download&shareKey=f7de1ca7c736a7ceb319283010c2d8d3 "DSP")

## Memory Analysis

在实践中，DSP 需要更少的 Reshape 和通信开销，与其他方法相比，进一步减少中间激活内存。另一方面，MegatronSP 需要在 all-gather 操作之后保存整个激活，从而导致所需的内存更高。

# Experiments

实验在 128 个 NVIDIA H100 GPU 上进行，节点内通过 NVLink 连接，节点间通过 InfiniBand 连接。实验采用 720M 和 3B 大小的 Transformer-2D 模型。

| Model Name | Layers | Hidden States | Attention Heads | Patch Size |
|------------|--------|---------------|-----------------|------------|
| 720M       | 28     | 1152          | 16              | (1, 2, 2)  |
| 3B         | 36     | 2038          | 32              | (1, 2, 2)  |


## End2End Performance

将序列并行性和数据并行性结合使用，实验设置如下表所示。不同方法里的括号表示 (序列并行大小，数据并行大小)

| Model Size | Sequence Length | Temporal Sequence | Spatial Sequence | DeepSpeed Ulysses | Megatron SP | Ring Attention | DSP     |
|------------|-----------------|-------------------|------------------|-------------------|-------------|----------------|---------|
| 720M       | 0.5M            | 128               | 4096            | (2, 64)   | (2,64)      | (2, 46)        | (2, 64) |
|            | 1M              | 256               | 4096            | (4, 32)   | (4,32)      | (4, 32)        | (4, 32) |
|            | 2M              | 512               | 4096            | (8, 16)   | (16,8)      | (8, 16)        | (8, 16) |
|            | 4M              | 1024              | 4096            | (16, 8)   | /           | (16, 8)        | (16, 8) |
| 3B         | 0.5M            | 128               | 4096            | (4, 32)   | (4, 32)     | (4, 32)        | (4, 32) |
|            | 1M              | 256               | 4096            | (8, 16)   | (16,8)      | (8, 16)        | (8, 16) |
|            | 2M              | 512               | 4096            | (16, 8)   | /           | (16, 8)        | (16, 8) |
|            | 4M              | 1024              | 4096            | (32, 4)   | /           | (32, 4)        | (32, 4) |

端到端性能如下图所示。随着 token 的序列长度从 50 万增加到 400万 ，DSP 的 FLOPS/GPU 最多下降 23%，而其他方法至少下降 40%.

![End-to-end Performance Comparison](https://note.youdao.com/yws/api/personal/file/WEB5a9d55ea669c31ea3847b72d3cee66c4?method=download&shareKey=9170227c7e5864374d320fd0dba6c37b "End-to-end Performance Comparison")

## Scaling Ability

弱可伸缩性是指每个设备的计算工作负载保持不变，而设备数量却在逐渐增加的情况。batch size 与 GPU 数量成正比线性增加，序列长度固定。实验设置如下表所示。

| Model Size | Type       | GPU Number | Batch Size | Temporal | Spatial |
|------------|------------|------------|------------|----------|---------|
| 720M       | Intra-Node | 1          | 1          | 64       | 4096    |
|            |            | 2          | 2          | 64       | 4096    |
|            |            | 4          | 4          | 64       | 4096    |
|            |            | 8          | 8          | 64       | 4096    |
|            | Inter-Node | 8          | 1          | 256      | 4096    |
|            |            | 16         | 2          | 256      | 4096    |
|            |            | 32         | 4          | 256      | 4096    |
| 3B         | Intra-Node | 1          | 1          | 16       | 4096    |
|            |            | 2          | 2          | 16       | 4096    |
|            |            | 4          | 4          | 16       | 4096    |
|            |            | 8          | 8          | 16       | 4096    |
|            | Inter-Node | 8          | 1          | 128      | 4096    |
|            |            | 16         | 2          | 128      | 4096    |
|            |            | 32         | 4          | 128      | 4096    |
|            |            | 64         | 8          | 128      | 4096    |


实验结果如下图所示。DSP 的性能明显优于其他方法 80.7% 以上。此外，DSP 可以扩展到 64 个 GPU，不像 DeepSpeed-Ulysses 和 Megatron-SP 受注意力头数量的限制。DSP的吞吐量几乎保持线性增长，从8 到 64 个 GPU 的性能损失仅为 15%.

![Weak Scaling Ability Evaluation](https://note.youdao.com/yws/api/personal/file/WEBb94f9351e3b5c5aeb1f8cb2cd466feb3?method=download&shareKey=f41e3f0680ddd9d44d392b01466666d3 "Weak Scaling Ability Evaluation")

强可伸缩性更具挑战性，因为它需要在增加设备数量的同时保持总计算工作负载不变。batch size 和序列长度都是固定的。实验设置如下表所示。

| Model Size | Type        | Batch Size | Temporal | Spatial |
|------------|-------------|------------|----------|---------|
| 720M       | Intra-Node  | 1          | 64       | 4096    |
|            | Inter-Node  | 1          | 256      | 4096    |
| 3B         | Intra-Node  | 1          | 16       | 4096    |
|            | Inter-Node  | 1          | 128      | 4096    |


实验结果如下图所示。720M 大小模型下性能可以线性扩展到 8 个 GPU， 3B 大小可扩展到 4 个 GPU. 为了评估 DSP 的极限性能，进一步扩展到 64 个 GPU，每个设备的工作负载很少。

![Strong Scaling Ability Evaluation](https://note.youdao.com/yws/api/personal/file/WEB5e407bd4c9cfbe584ed3644d3d2bad8f?method=download&shareKey=6cdfa291ec43d9fa671dbd913e88fe78 "Strong Scaling Ability Evaluation")

还比较了所有 baseline 在相同工作负载下的推理延迟。实验结果如下图所示，与 baseliens 相比，DSP 可以提高 29% 到 63% 的速度。

![Inference Latency Comparison](https://note.youdao.com/yws/api/personal/file/WEB5887952d3d5596b8d642614e9b8a360c?method=download&shareKey=1bb0531c0e4f50530d9105dbf7b1f889 "Inference Latency Comparison")

## Memory Consumption

下图展示了在弱扩展性设置中不同 baseline 的内存消耗比较。半透明表示缓存内存，实心示已分配的内存，总内存使用量是它们的总和。此外，DSP 没有过多的缓存内存膨胀。

![Memory  Consumption Comparison](https://note.youdao.com/yws/api/personal/file/WEBec2e5d141492af4ccc1ec6d95e69a136?method=download&shareKey=11b30ed241cc958ce08bec402463dd31 "Memory Consumption Comparison")

## Conclusion and Discussion

DSP 的限制是它是专门为多维 Transformer 设计的，可能不能很好地适应像语言模型这样的一维 Transformer. 此外，对于涉及所有序列维度的全局操作，DSP 可能无法达到最佳效率。在未来，DSP可以将其范围从 Transformer 架构扩展到包括卷积、循环和图神经网络在内的架构，以利用其在各种任务中的潜力。此外，自动化优化技术可以使 DSP 基于网络分析动态自主地确定最有效的切换策略，从而优化整个系统的效率和效能。