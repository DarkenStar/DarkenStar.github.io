---
title: Comparsion of Parallelsim Metods in ViT
date: 2024/11/13 16:05:23
categories: Paper Reading
tags: Distributed Training
excerpt: Paper reading of Pyramid Attention Broadcast.
mathjax: true
katex: true
---
| Symbol | Description             | Symbol | Description |
| ------ | ----------------------- | ------ | ----------- |
| a      | 注意力头数              | n      | 并行度大小  |
| b      | batchsize               | s      | 序列长度    |
| h      | 隐藏层维度              | t      | 张量并行度  |
| L      | tranformer layer 层数数 | v      | 词汇表大小  |

transformer block 的输入是形状为 bsh 的三维张量，其中 b 为 batchsize. 每个变压器层由一个具有注意头的自注意块组成，随后是一个具有两层的 MLP，第一层将隐藏维度增加到 4h，然第二层将其减少到 h. 每个变压器层的输入和输出具有相同的形状 (bsh).

# Model Parameters

QKVO Linear 的权重形状均为 `h*h`, 偏置形状均为 `h*1`；MLP 两个 Linear 的权重形分别为 `h*4h` 和 `4h*h`，偏置形状分别为 `4h*1` 和 `h*1`. 因此每个模型的参数量为 `(12hh+13h)L`，占用大小还要 `x2`.

{% note info %}
在传统的 LLM 中最后还需要经过 logits layer，将隐藏层维度 `h` 转换成词汇表大小 `v`，参数量还要加上 `hv`.
{% endnote %}

# FLOPs Calculation

对于浮点数计算量 (FLOPs)，只考虑占主要部分的通用矩阵乘法 (GEMMs). 对于 Attention 部分，QKV Linear 的计算量为 `6bshh`，attention matrix (Q@K.T) 的计算量为 `2bssh`, attention@V 的计算量为 `2bssh`, O Linear 的计算量为 `2bshh`. MLP 的两个线性层的每一个计算量都为 `8shh`. 相加后得到正向传播中总计算量为 `(24bshh + 4bssh)L` bytes.

{% note info %}
在传统的 LLM 中最后还需要经过 logits layer，将隐藏层维度 `h` 转换成词汇表大小 `v`，其计算量为 `2bshv`.
{% endnote %}

反向传播因为要计算输入和权重的梯度，其计算量为正向传播的两倍，因此整个模型的计算量为 72BLshh(1+s/(6h)).

# Activation Memory

激活的定义为在前向传播中产生并且需要在反向传播中进行梯度计算的张量，即不包括模型参数和优化器状态。并且不考虑相对非常小的激活。例如 LayerNorm 层的输入还需要张量每个通道的均值和方差 (大小均为 bs)，由于 h 大小通常超过 1k，因此只考虑输入张量所占激活的大小 bsh，忽略掉 2bs. 假设数据格式为 fp16/bf16，即每个数据占用 2 bytes 的存储空间，需要特殊处理的是 dropout 层的 mak，每个元素均为 unsigned int，只占用 1 byte.

Attention 部分激活占用如下 (共计 11bsh + 5bssa)

- QKV Linear: 三个线性层需要的输入相同，占用 2bsh bytes.
- Q@K.T: 需要存储 Q 和 K，占用 4bsh bytes.
- Softmax: 需要存储大小为 2bssa bytes 的输入
- Softmax droppot: 需要存储一个大小为 bssa bytes 的 mask.
- attention@V: 需要存储 dropout 的输出和 V，分别占用 2bssa 和 2bsh bytes.
- O Linear: 需要存储注意力的输出，占用 2bsh bytes.
- O dropout 需要存储一个大小为 bsh bytes 的 mask;

MLP (共计 18bsh): 第一层和第二层的输入分别占用 2bsh 和 8bsh bytes. GeLU 层需要第二层的输入用于反向传播，占用大小为 8bsh bytes. dropout 需要一个大小为 bsh bytes 的 mask.

LayerNorm (共计 4bsh): 需要存储该层的输入，占用 2bsh bytes. 一共有两个 LayerNorm.

加起来就可以得到每个 transformer block 需要激活大小为 bsh(34+5sa/h) bytes.

# Tensor Parallelsim

Megatron 张量并行的思想是将输入进行连续的两个矩阵乘法的第一个按列切分成 t 份，第二个按行切分成 t 份. 在 Transformer block 中体现为利用多头注意力本身的并行性将 Attention 计算中的 QKV 按列进行切分，O Linear 的权重按行进行切分；MLP 中第一个线性层的权重按列进行切分，第二个权重按行进行切分。

在这种并行方式下，前向传播和反向传播均需要进行 2 次 All-Reduce 通信，由于每次 All-Reduce 通信可以看作 Reduce-Scatter + All-Gather, 因此每次每个设备的通信量为 8αbsh bytes，其中 α=(n-1)/n.

对于激活，2*LayerNorm, QKV Linear 的输入, O dropout mask，MLP 第一层的输入和 MLP dropout 不会被切分，因此每个设备每个 block 要占用的激活为 bsh(10+24/t+5as/(ht))

# Megatron Sequence Parallelsim

Megatron 张量并行中 LayerNorm 以及 O Linear 和 MLP 之后的 dropouts 在每个设备中都有一个副本。这些模块不需要大量的计算，但需要占用 10bsh bytes 大小的激活内存。Megatron-SP 沿着序列维度划分这些模块来减少激活内存，但需要配合 TP 一起使用，本质上是将 TP 中的 All-Reduce 拆成了在 TP 前进行 All-Gather 和在 TP 后进行 Reduce-Scatter. 但除去第一个 LayerNorm 外的每一个模块的激活都得到了切分。Megatron-SP 这里选择每个设备存储自己的部分并在 反向传播中插入一次额外的 All-Gather 通信。因此通信量为 10bsh, 每个设备每个 block 需要占用的激活为 bsh/t*(34+5as/h)

# Pipeline Parallelsim

流水线张量并行仅仅将 L 个 Transformer block 平均分到 p 个设备上，并没有划分激活所要占用的内存。在考虑 1F1B 策略下 batchsize 进一步被划分成 p 个 micro batch. 第一个 stage 必须存储 p 个 micro batch 的激活。每个 stage 包含 L/p 层，所以无论流水线并行大小 p 如何，第一个 stage 必须存储 p × L/p = L 层的激活值。在 Megatron-LM 中的 interleaving schedule 需要存储 L(1 + (p−1)/(pm)) 层的激活，其中 m 是 interleaving 的数量。

{% note info %}
在使用 output-tensor-deallocation 优化 (输出传到下一个 stage 后就释放) 的情况下，可以为为每个设备节省 bshr 内存，其中 r 是每个设备正在运行的 micro batch 的数量，在第一个 stage r=p 时达到峰值。
{% endnote %}

# Deepseed-Ulysses Sequence Parallel

DS-SP 也是利用多头注意力的并行性，首先将输入按序列维度切分到每个设备上 (a 要能整除 n)，每个设备占有的输入形状为 b*(s/n)*h. 在计算 Attention 之前对 QKV 进行 All-to-All 通信变成按隐藏层维度切分，通信量为 6αbsh/n bytes. 计算完 attention@v 之后再进行一次 All-to-All 通信，通信量为 2αbsh/n bytes，总计通信量为 8αbsh/n bytes. 激活占用上 Attention 中 Softmax 及其 dropout mask 和 attention 没有被切分，激活占用量为 bsh(34/n+5sa/h).

# Ring-Attention Sequence Parallel

Ring-SP 实际上为环状的 FlashAttention，将输入沿着序列维度切分到每个设备上，在 Attention 计算过程中每个设备向相邻设备通信 KV 并更新自己的 Softmax 矩阵，通信量为 4bsh bytes. 激活占用和 DS-SP 一样为 bsh(34/n+5sa/h).


