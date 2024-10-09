---
title: Efficient Large-Scale Language Model Training on GPU Clusters
date: 2024/10/05 10:09:35
categories: Distributed Training
tags: Paper Reading
excerpt: Paper reading about Efficient Large-Scale Language Model Training on GPU Clusters.
mathjax: true
katex: true
---
# Abstract

本文展示了如何将张量、流水线和数据并行性组合起来以扩展到数千个gpu。我们提出了一种新的交错流水线调度，可以在内存占用与现有方法相当的同时将吞吐量提高 10%.

![Trend of Sizes of SOTA NLP Models](https://note.youdao.com/yws/api/personal/file/WEBb0e290b7248b1bb11233e55661bbb736?method=download&shareKey=c1cd9d017d6f99b06b1fc2635f48fc19 "Trend of Sizes of SOTA NLP Models")

# Introduction

张量（层内）模型并行对于较大的模型会崩溃。较大的模型在多个多 GPU 服务器上进行切分会导致两个问题：

1. 张量并行所需的 all-reduce 通信需要通过服务器间链路进行，这比多 GPU 服务器内可用的高带宽 NVLink 要慢
2. 高度模型并行会产生小规模的矩阵乘法（GEMM），从而可能降低 GPU 利用率。

流水线模型并行化是指模型的各层在多个 GPU 上进行条带化处理。batch 被拆分成更小的 microbatch ，并在这些 microbatch 之间流水线执行。无论进度如何，为了保持严格的优化器语义，优化器步骤需要跨设备同步，从而在每个 batch 结束时进行流水线刷新 (*pipeline flush*)，允许 microbatch 完成执行 (不再注入新的 microbatch). microbatch 数量与流水线级数的比例越大，流水线刷新所花费的时间就越少。

我们展示了如何结合流水线、张量和数据并行性，我们称之为PTD-P. 配置分布式训练的指导原则如下:

- 不同形式的并行性以不同的方式相互作用: 并行策略对通信量、执行内核的计算效率以及由于流水线冲洗 (流水线气泡) 而花费的等待计算的空闲时间有影响。
- 用于流水线并行性的调度对通信量、流水线气泡大小和用于存储激活的内存有影响。
- 超参数 (如 microbatch 大小) 的值会影响内存占用、在工作线程上执行的内核的算术效率和流水线气泡大小。
- 随着规模扩展分布式训练是通信密集型的。使用较慢的节点间互连或更密集的通信分区会影响扩展性能。

# Model Parallelism

本节中将讨论有助于不适合单个 GPU 内存的大模型的并行训练方法。我们将流水线模型并行和张量模型并行 (如图 2 所示的组合) 与数据并行结合起来，简称为PTD-P.

![Combination of Tensor and Pipeline Model Parallelism (MP)](https://note.youdao.com/yws/api/personal/file/WEB09f440d2c0b5bd3b923d2d09dbf47eb5?method=download&shareKey=59701400f477d8dd03c4ea43138c933f "Combination of Tensor and Pipeline Model Parallelism (MP)")

## Data Parallelism

使用数据并行时，每个 worker 都有一个完整模型的副本，输入数据集被分片， worker 定期汇总他们的梯度，以确保所有 worker 看到一个一致的权重版本。

## Pipeline Parallelism

通过流水线并行，模型的层被分散到多个设备上。一个 batch 被分成更小的 microbatch. 在 microbatch 之间进行流水线执行。为了准确地保持优化器语义，我们引入了定期的流水线刷新，以便在设备之间同步优化器步骤。在每个 batch 处理的开始和结束时，设备都是空闲的。我们称这段空闲时间为流水线气泡 (*pipeline bubble*).

### Default Schedule

GPipe 提出了一个调度方案，如图 3 所示 (假设**反向传播的时间是前向传播的两倍**，管道调度的效率并不取决于这个因素)，首先执行一个 batch 中所有 microbatch 的前向传播，然后执行所有 microbatch 的反向传播。设 GPipe 流水线气泡的大小为 t_pb，microbatch 的数量为 m，流水线阶段数量 (用于流水线并行的设备数量) 表示为 p，每次迭代的理想时间表示为 t_id (假设理想缩放)，执行单个 microbatch 的向前和反向传播的时间表示为 t_f 和 t_b. 在该调度中，流水线气泡由批处理开始时的 p−1 个前向传播和 p−1 个反向传播组成。则流水线气泡总时间为 t_pb=(p−1)·(t_f+t_b). batch 的理想执行时间为 t_id=m·(t_f+t_b)。因此，在流水线气泡中花费与理想计算时间的比例为:

流水线气泡占比 = t_pb / t_id = (p−1) / m.

为了使流水线气泡占比小，我们需要 m 远大于 p. 然而 m 非常大时这种方法的内存占用很高，因为它需要在训练一次迭代时间内为所有 m 个 microbatch 保存中间激活.

![GPipe Pipeline Schedule](https://note.youdao.com/yws/api/personal/file/WEB04e4031b886573061f614e73854d1f43?method=download&shareKey=97391411ed3d06fc2c6de5de5f20d1d0 "GPipe Pipeline Schedule")

### Schedule with Interleaved Stages

为了缩小流水线气泡的大小，每个设备都可以对多个层的子集（称为模型块）进行计算，流水线中的每个设备都被分配了多个流水线阶段（与之前相比，每个流水线阶段的计算量更少），而不是单个连续的层。

{% fold info@An Example %}
例如，如果每个设备之前被分配 4 层 (即设备 1 有 1 - 4 层，设备 2 有 5 - 8层...)，我们可以让每个设备为两个模型块执行计算 (每个模型块被分配 2 层)，即设备 1 有 1、2、9、10 层; 设备 2 具有第3、4、11、12层...
{% endfold %}

和上一小节一样，我们可以执行完所有 microbatch 的前向传播然后执行所有反向传播 (all-forward, all-backward)，但这将占用大量内存 (与 m 成正比). 因此如图 4 所示，我们设计了一个适配于之前的内存高效 1F1B 的交错调度。它要求 **microbatch 数量是流水线并行度 (流水线中的设备数量) 的整数倍**。

如果每个设备都有 v 个阶段 (模型块)，那么每个阶段 microbatch 的前向和反向传播的时间分别为 t_f/v 和 t_b/v. 流水线气泡时间因此减少到 𝑡^int_pb=(p−1)·(tf+tb)/v，

流水线气泡占比为 𝑡^int_pb / t_id = (p−1) / (m·v).

这意味着该调度减少气泡时间到原先的 1/v，但该计划需要额外的通信，因此通信量也为原来的 v 倍。

![Default and Interleaved 1F1B Pipeline Schedules](https://note.youdao.com/yws/api/personal/file/WEBb74cfaaf752e14cf2e44f4abd7e3e7bf?method=download&shareKey=4558481494c1d4b22574e739743b123d "Default and Interleaved 1F1B Pipeline Schedules")

## Tensor Model Parallelism

详情见 Megatron-LM.

![Blocks of Transformer Model Partitioned with Tensor Model Parallelsim](https://note.youdao.com/yws/api/personal/file/WEB02b1b15f4d736bfee41738f3c3ee72b3?method=download&shareKey=b1a7e4a6171585f4f0a8a39fb4b2d8b3 "Blocks of Transformer Model Partitioned with Tensor Model Parallelsim")

# Performance Analysis of Parallelization Configurations

首先定义下符号含义

- (p,t,d): 并行化维度。p 表示流水线模型并行大小，t 表示张量模型并行大小，d 表示数据并行大小。
- n: GPU 数量，要求 ptd=n.
- B: 全局批大小 (作为输入提供)
- b: microbatch 大小。
- m = B/(db): 一个 batch 中每个流水线中的 microbatch 的数量。

## Tensor and Pipeline Model Parallelism

如前所述，使用带有周期性冲洗的流水线并行会产生大小为 (p−1)/m 的流水线气泡. 固定 d=1，则 tp=n，气泡大小可以用 t 表示为

(p−1)/m=(n/t-1)/m.

GPU 之间的通信量也受 p 和 t 大小的影响。流水线模型并行的特点是更便宜的点对点通信，每个 microbatch 的每对连续设备之间 (前向或后向传递) 需要执行的通信总量为 bsh. 张量模型并行则使用 all-reduce 通信，总大小为 bsh 的张量需要在每层的前向和后向传递中，在 t 个模型副本之间进行两次 all-reduce，因此每个 microbatch 每层每个设备的总通信量为 4bsh(t-1)/t. 每个设备通常有多个层，则每个设备上每个 microbatch 的张量并行通信总量为 l^stage4bsh(t-1)/t, 其中 l^stage 为流水线阶段的层数。

{% note primary %}
启示 1: 当 t 大于单个节点中的 GPU 数量时，在较慢的节点间链路上执行张量模型并行的开销非常大。在考虑不同形式的模型并行时，使用 g-GPU 服务器时张量模型并行度一般为 g (all-reduce 通信量大，NVLink 带宽高)，然后可以使用流水线模型并行来扩展到跨服务器的更大模型 (P2P 通信量小，PCIe 带宽低).
{% endnote %}

## Data and Model Parallelism

管道模型并行性。设 t=1，每个管道的 microbatches 数量 m=𝐵/(db)=b'/d, b':=B/b. 设 GPU 总数为 n ，流水线阶段数为 p=n/d，气泡大小为

(p−1)/m=(n/d-1)/(b'/d)=(n-d)/b'

管道气泡随着 d 变大而变小。如果数据并行所需的 all-reduce 通信不会随着 d 的变大而急剧增加，那么总体吞吐量将会增加，因为基于环的实现的通信时间随着 d 的变化为 (d−1)/d=1−1/d.同样对于给定的并行配置，随着批量大小的增加，b' = B/b 增加，因此吞吐量上升。同时数据并行所需的 all-reduce 通信频率也下降，进一步提高了吞吐量。

![Fraction of Time Spent Idling due to Pipeline Flush](https://note.youdao.com/yws/api/personal/file/WEB27b82e198a75ad564ac917d6d560dec1?method=download&shareKey=cfef60f8f216007803bdafe8b5a5e64c "Fraction of Time Spent Idling due to Pipeline Flush")

在张量模型并行下，每个 microbatch 都需要进行 all-reduce 通信，这在多 GPU 服务器上开销很大；而数据并行只需要在每个 batch 中执行一次的 all-reduce通信。此外，使用张量模型并行，每个设备计算每层的一部分，因此对于不够大的层， GPU 可能无法以峰值效率执行这些子矩阵计算。

{% note primary %}
启示 2：在使用数据和模型并行时，应使用 M=tp 的总模型并行大小，以便模型参数和中间数据满足 GPU 内存限制；数据并行可用于将训练扩展到更多 GPU.
{% endnote %}

## Microbatch Size

给定函数 t_f(b) 和 t_b(b)，将 microbatch 大小映射为单个 microbatch 的前向和反向计算时间，计算一个 batch 所花费的总时间 (忽略通信成本) 为

(b'/b+p-1)·(t_f(b)+t_b(b)).

microbatch 大小因此既影响运算的算术强度，也影响管道气泡大小。

![Per-GPU Throughput versus Microbatch Size for a GPT Model](https://note.youdao.com/yws/api/personal/file/WEBe3b16b74c3ca1aedb0f81939501da9de?method=download&shareKey=cd1c360e2597b37f705bd5f4906d64b3 "Per-GPU Throughput versus Microbatch Size for a GPT Model")

![](https://note.youdao.com/yws/api/personal/file/WEB3de398053a7040a405722f3b8c929bf1?method=download&shareKey=7014b2696bffc97d5646b4b2614bd3fb "Behavior of Throughput for the same GPT Model")

{% note primary %}
启示 3：最佳 microbatch 大小 b 取决于模型的吞吐量和内存占用特征，以及流水线深度 p、数据并行大小 d 和批量大小 B.
{% endnote %}

## Activation Recomputation

激活重计算通过在向后传递之前运行第二次正向传播 (并且仅存储给定流水线阶段的输入激活)，来权衡所执行的计算操作数量的增加对额外内存占用的影响。设 A^input 为一层的输入激活的大小，A^intermediate 为每层的中间激活的大小，一个模型阶段有 l 层， 激活保存点的数量为 c，那么总内存占用为 c·A^input + l/c·A^intermediate. 因此取 c = \sqrt(l·A^input·A^intermediate) 时内存占用最小。

# Implementation

## Communication Optimizations

使用流水线并行时，我们希望在正向和反向并行发送和接收张量。每台 DGX A100 都配备了 8 个 InfiniBand（IB）网卡。然而发送和接收都是点对点的，只发生在两台服务器上的一对 GPU 之间，因此很难充分利用所有网卡。对于流水线内的单次通信，每个 transformer 层的输出都会在张量并行的设备中复制。为了减少这种冗余，我们可以在发送端将张量分割成大小相等的块，然后使用每个 rank 自己的 InfiniBand 发送. 在接收端通过比 InfiniBand 互连快得多的 NVLink 执行 all-gather，重新组装整个张量。通过 scatter-gather 通信优化，将每对连续流水线阶段之间需要执行的通信总量减少为 bsh/t.

## Computation Optimizations

将数据布局从 (b,s,a,h) 更改为 (s,b,a,h). 其次，使用 PyTorch JIT 为一系列元素操作 (bias+GeLU 和 bias+dropout+add) 生成融合算子。

# Evaluation

在 Selene 超级计算机上以混合精度运行。每个集群节点有 
- 8 个 NVIDIA 80GB A100 GPU，通过 NVLink 和 NVSwitch 互连。
- 8 个 NVIDIA Mellanox 200Gbps HDR Infiniband HCA 用于应用程序通信
- 额外有 2 个 HCA 用于专用存储。
节点以三级 (leaf, spine, core) 胖树拓扑结构连接，一共有 850个交换机。集群使用 all-NVME 共享并行文件系统进行高性能数据访问和存储。16 位精度的 A100 GPU 的峰值设备吞吐量为 312 teraFLOP/s.

QKV 变换的线性层权重参数量均为 h^2, attention 后的线性层权重参数量为 h^2, 两层前馈网络每个线性层的权重参数量为 4h^2，因此每一个 transformer block 的所有线性层的参数量为 12h^2. 词嵌入的参数量为 Vh，位置编码的参数量为 sh.

一个 {% mathjax %} A_{m\times k}\times X_{k\times n} {% endmathjax %} 矩阵乘法需要 2mkn FLOPs( 2 是因为乘法和加法). transformer block 包含一个注意力块和一个两层前馈网络组成。对于注意力块，主要的 FLOP 来源于 QKV 转换 (6Bsh^2 次操作)、注意力矩阵计算 (2Bhs^2 次操作)、注意力乘 Value (2Bhs^2 次操作) 和 attention 后的线性层 (2Bsh^2 次操作). 前馈网络将隐藏维度放大到 4h，然后再减小到 1h，需要 16Bsh^2 次操作。将这些加在一起，每个 transformer block 一共有 24Bsh^2+4Bhs^2 FLOPs. 反向传播需要两倍的计算量，因为需要计算关于输入张量和权重张量的梯度。此外，使用激活重计算需要在反向传播之前进行额外的正向传播。因此，每一层的总计算量为 FLOPs 为 4*(24Bsh^2+4Bhs^2).

计算量另一方面来源于 head 的 logit 层，它将维度的特征 h 转换为词汇表维度的特征 V. 该操作所需的计算量为正向传播的 2BshV 和反向传播的 4BshV，总共 6BshV FLOPs.  

## Result

Pipeline-parallel 并行度增加降低 GPU 的计算效率，因为 bubble 变多了。
Batchsize 的增大可以减少 pipeline-parallel 并行度大小带来的影响。

Batch size增加有助于提高GPU的计算效率。
Interleaved schedules 能显著提高GPU的计算效率。

不使用激活重计算的话单位时间内的训练的吞吐是要高于使用重计算的，因为重计算在反向传播中引入额外的计算量。
由于重计算可以节省显存，batchsize 可以相应提高不少。由于 batchsize 的提高，训练吞吐量也得到了提高，从而达到了优化的效果。