---
title: "PipeFusion"
date: 2025-06-13T11:39:32+08:00
lastmod: 2025-06-13T11:39:32+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- Diffusion

keywords:
- Diffusion

description: "Paper Reading of PipeFusion" # 文章描述，与搜索优化相关
summary: "Paper Reading of PipeFusion" # 文章简单描述，会展示在主页
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

# Abstract

PipeFusion 是一种利用多 GPU 并行来进行 DiT 模型推理的方法。

- 将图像分割成 patch，并将 Transformer Blocks 分布在多个设备上。
- 通过重用上一步 (*one-step stale*) 的特征图作为当前步的上下文，消除了流水线中的等待时间。

# Introduction

由于 Attention 的计算特性，计算时间与序列长度的平方成正比，使得 DiT 模型生成高分辨率图形 (长视觉序列) 的推理延迟非常高。
[DistriFusion](https://arxiv.org/abs/2402.19481) 观察到在相邻的扩散时间步中输入和激活存在高度相似性，我们将这种现象称为输入时间冗余 (*input temporal redundancy*). 它保留所有层 KV 的完整形状。内存开销不会随着计算设备数量的增加而减少，在可扩展性方面表现不佳。

如下图所示，DistriFusion 在 2 个设备上都保存一份 DiT 参数。它将图像分成 2 个小块，并对每一层的激活使用异步 allgather. PipeFusion 将 DiT 参数切分到 2 个设备上，将图像分成 4 个 patch ，两个设备之间采用异步 P2P 通信来传输激活。它只在每个设备上传输初始层的输入激活和最终层的输出激活

![Comparsion Between DistriFusion & PipeFusion](https://note.youdao.com/yws/api/personal/file/WEB0b7d56775c290d039aba1c7aab220319?method=download&shareKey=6330f70503af01609e9fe13a35826aab "Comparsion Between DistriFusion & PipeFusion")

# Background & Related Works

扩散模型通常使用 DNN 预测噪声。给定有噪声的图像 xt，模型 ϵθ 将 xt、去噪时间步 t 和附加条件 c (例如文本、图像) 作为输入，以预测 xt 中相应的噪声ϵt.

扩散模型具有较长的序列长度和较小的模型大小，但在推理过程中通信开销仍然很大。DistriFusion 为 U-Net 为主干的扩散模型引入了位移 patch 并行(displacement patch parallelism)，将模型的输入划分为多个 patch，便于激活的异步通信并且使得通信与计算重叠。然而，当将该方法应用于 DiT 时，内存缓冲区的开销将导致巨大的内存开销。

# Methods

不同并行策略下 DiT 单步扩散过程的比较如下表所示。

- p: 生成的序列长度 (即隐空间下的像素数量).
- hs: 模型的隐藏通道大小。
- N: 设备数量。
- M: 图像切分的 patch 数量。
- L: Transformer Blocks 层数。
- P: 模型总参数量。
- A: Attention 的过程中的激活大小 (Q, K, V, O 大小一样)

名称后面的 * 表示采用异步通信，通信开销可以通过计算隐藏。

|                      | attn-KV | communication cost | param | QO Activations       | KV Activations         |
| -------------------- | ------- | ------------------ | ----- | -------------------- | ---------------------- |
| Tensor Parallel      | fresh   | 4O(p × hs)L       | P/N   | (2/N) A = (1/N) QO | (2/N) A = (1/N) KV     |
| DistriFusion*        | stale   | 2O(p × hs)L       | P     | A                    | 2AL = (KV)L            |
| Ring Seq Parallel*   | fresh   | NA                 | P     | A                    | A                      |
| Ulysses Seq Parallel | fresh   | 4O(p × hs)L       | P     | (2/N) A = (1/N) QO | (2/N) A = (1/N) KV     |
| PipeFusion*          | stale-  | 2O(p × hs)        | P/N   | (2/M) A = (1/M) QO | (2L/N) A = (1/N) (KV)L |

## Sequence Parallelism & Tensor Parallelism

针对 LLM 提出的张量并行 (tensor parallelism, TP) 和序列并行 (sequence parallelism, SP) 可以应用于 DiT 推理。因为他们的主干都是 Transformer.
在 TP[^1] 中，权重矩阵按列被切分为 *N* 份，这样矩阵乘法后激活值也被切分成 *N* 份，使得每个设备的参数量和激活量均为原来的 1/N. 在 attention 计算和 FFN 层之后都需要进行两次同步 all-reduce 操作，因此每一层通信量为 4O(p × hs).

在 SP 中，可以将输入图像分割成 patch，DiT 中的多头注意模块可以采用 Ring-Attention[^2]，DeepSpeed-Ulysses[^3]，或者两者的组合。Ulysses SP 并行需要 4 次 all-to-all 操作，因此每一层通信量为 4O(p × hs), 和 TP 相同。

> TP 和 SP 可以在 DiT 推理中一起使用。

## Displaced Patch Parallelism

输入时间冗余意味着给定层中激活 patch 的计算并不完全取决于其他 patch 的最新激活。在前一个扩散步骤中加入稍微过时的激活是可行的。该方法将输入图像划分为 N 个patch，每个设备计算其各自 patch 的输出结果。 如下图所示 attention 模块需要具有完整形状的 KV 激活。它采用异步 all-gather 收集上一步扩散步骤的 KV 激活，并用其进行当前步的 attention 计算。

DistriFusion[^4] 可以看作是异步 SQ 的一种形式。它通过正向计算扩散步骤来隐藏 KV 通信，但代价是消耗更多内存。DistriFusion 利用 N-1/N 的 T+1 步的 KV 激活和 T 步的 1/N 的局部 KV 激活相结合。与 Ring-Attention 相比，DistriFusion 可以更有效地隐藏通信开销，因为它允许 KV 通信与扩散步骤的整个前向计算重叠，而 Ring-Attention 只允许通信在注意模块内部重叠。

![DistriFusion vs. Ring-Attention SQ for an Attention Module](https://note.youdao.com/yws/api/personal/file/WEB9a64239185e8b3604db9a46098203d05?method=download&shareKey=eab8f5ec3cff754ab7711e87333e8797 "DistriFusion vs. Ring-Attention SQ for an Attention Module")

在 Ring-Attention中，其通信缓冲区 c × hs 可由图中块大小 c 控制，其值小于 p/N. DistriFusion要求每个计算设备始终保持 KV 的完整形状的通信缓冲区，因此通信开销总共是 AL.

## Displaced Patch Pipeline Parallelism

PipeFusion 相比于 DistriFusion 有着更高的内存效率和更低的通信成本。它将输入图像划分为 M 个不重叠的 patch，DiT Blocks 被划分为 N 个阶段，每个阶段按顺序分配给 N 个计算设备。每个设备在其被分配的阶段以流水线方式处理一个 patch 的计算任务。

> DiT 模型中因有许多相同的 transformer block，很容易将去噪网络的工作负载均匀地划分为 N 个部分。然而，U-Net 扩散模型没有这种重复结构。

一个 M=4, N=4 的 PipeFusion 例子如下图所示，利用输入时间冗余，设备不需要等待接收到当前步骤的完整形状激活，利用上一步的激活就可以开始自己所处阶段的计算。考虑流水线气泡，流水线的有效计算比为 MS/MS+N−1，其中 S 为扩散步长数。

![The Workflow of Displaced Patch Pipeline Parallelism](https://note.youdao.com/yws/api/personal/file/WEB157240ab4733f1ca4cca87a2389a7b08?method=download&shareKey=b0593903312b6da2796d081015a30baa "The Workflow of Displaced Patch Pipeline Parallelism")

PipeFusion 在计算设备之间仅传输属于一个阶段的 (连续 transformerl blocks) 的输入和输出的激活，因此通信开销为 2O(p × hs). PipeFusion 通过异步 P2P 传输前一步 Patch 数据和接收后一步骤 Patch 数据来与当前 Patch 计算重叠，从而将通信隐藏在计算中。PipeFusion 中的每个设备仅存储与其特定阶段相关的 1/N 份参数。由于使用陈旧 KV 进行注意力计算，要求每个设备保持其阶段对应的 L/N 层的完整 KV.

PipeDiffusion 理论上优于 DistriFusion，因为它利用了更多的新激活。如图所示，在单个扩散步骤内，PipeDiffusion 中最新激活的占比随着流水线执行而增加。而 DistriFusion 中最新激活的占比一直都是 1/N.

> 尽管 DiT 没有采用 GroupNorm 层，但在 PipeFusion 中，U-Net 中 DistriFusion 对 GroupNorm 层的精度保持设计，特别是校正异步群归一化 (Corrected Asynchronous GroupNorm)，可以无缝地应用于 PipeFusion.

![The Fresh Part of Activations](https://note.youdao.com/yws/api/personal/file/WEBf19f26f13cfdf79d2e13e8b012b2954b?method=download&shareKey=9983249909c804bca818835ce2f953ce "The Fresh Part of Activations")

由于使用输入时间冗余需要一个预热期，DistriFusion 使用了几次同步 path 并行的预热步骤作为预备阶段。为了优化预热开销，可以将预热步骤与其余步骤分开，并将其分配给不同的计算资源。

# Experiments

我们在 Pixart-α 上进行实验 (0.6B)，它支持分辨率 1024px 的高分辨率图像合成，采用标准的 DiT，并结合交叉注意模块注入文本条件。使用了三个 GPU 集群，包括 4xGPU A100 80GB (PCIe) 集群，8xGPU A100 80GB (NVLink) 集群和 8xGPU L20 40GB (PCIe) 集群。测试的 GPU P2P 带宽分别为23 GB/s、268 GB/s 和 26 GB/s. 切分的 patch 数目 M 从 2,4,8,16,32 中搜索来确定最佳延迟性能。

- TP: 参考 Megatron-LM实 现了一个 TP DiT.
- SP: 采用了两种不同的序列并行，DeepSpeed-Ulysses 和 Ring-Attention.
- DistriFusion: 将 U-Net 扩散模型中的官方 DistriFusion 应用于DiT.
- Original:在单个 GPU 上的串行实现。

{{< notice note>}}
在 VAE 中由于卷积算子的临时内存使用会产生内存峰值，因此 VAE 比 DiT 层需要更多的内存。为了缓解这个问题，我们将卷积层的输入图像分成几个块，将单个卷积操作转换为按顺序执行的多个操作的序列。
{{< /notice >}}

## Quality Results

使用 20 步 DPM-Solver，预热步骤为 4 步。当 patch 数为 1 时，PipeFusion 的精度与 DistriFusion 相当。当 patch 数超过 1 时，其精度在理论上比 PipeFusion 更接近原始版本。PipeFusion 在 FID 方面略优于 DistriFusion.

## Latency and Memory

20 步 DPM-Solver，预热步骤为 1 步。

- 4xA100 (PCIe)集群上: 对于 8192px 的情况，在，DistriFusion 和 SQ 都会遇到内存不足 (OOM) 问题。
- 8xL20 (PCIe)集群上: 生成 4096px 分辨率的图像时，DistriFusion 和 SQ 都会遇到 OOM 问题。
- 8xA100 (NVLink) 集群上: 使用异步通信的 SQ (Ulysses) 的延迟与异步 DistriFusion 非常相似，并且优于 Ring 版本。此外，PixArt-α 在跨 8 个设备部署时面临限制，因为28个DiT层不能在均分，从而导致额外的开销。

### 4x A100 (PCIe) 集群

| **Latency** | **PipeFusion** | **Tensor Parallel** | **DistriFusion** | **Seq Parallel (Ulysses)** | **Seq Parallel (Ring)** | Single-GPU |
| ----------------- | -------------------- | ------------------------- | ---------------------- | -------------------------------- | ----------------------------- | ---------- |
| **1024px**  | **1.00x**      | 2.41x                     | 2.69x                  | 2.01x                            | 3.04x                         | 2.4x       |
| **2048px**  | **1.00x**      | 3.02x                     | 1.79x                  | 1.48x                            | 2.06x                         | 3.02x      |
| **4096px**  | 1.02x                | 1.77x                     | 1.16x                  | **1.00x**                  | 1.12x                         | 3.05x      |
| **8192px**  | **1.00x**      | 1.10x                     | OOM                    | OOM                              | OOM                           | 3.1x       |

![Overall Latency on a 4×A100-80GB (PCIe)](https://note.youdao.com/yws/api/personal/file/WEB7384a580336eb8b92972343922c549b6?method=download&shareKey=a203914b94020ad72b87708703fd829f "Overall Latency on a 4×A100-80GB (PCIe)")

内存效率方面，PipeFusion优于除了张量并行的其他方法。虽然张量并行的内存占用最低，但与其他并行化策略相比，由于通信量大会导致更高的延迟。

| **Max Memory** | **PipeFusion (Baseline)** | **Original** | **Tensor Parallel** | **DistriFusion** | **Seq Parallel (Ulysses)** |
| -------------------- | ------------------------------- | ------------------ | ------------------------- | ---------------------- | -------------------------------- |
| **1024px**     | 1.00x                           | 1.04x              | 0.98x                     | 1.21x                  | 1.21x                            |
| **2048px**  | 1.00x                           | 0.98x              | 0.90x                     | 1.54x                  | 1.33x                            |
| **4096px**  | 1.00x                           | 1.18x              | 0.69x                     | 2.35x                  | 1.63x                            |
| **8192px**  | 1.00x                           | 1.41x              | 0.71x                     | 2.34x                  | OOM                              |

![Overall GPU Memory on a 4×A100-80GB (PCIe)](https://note.youdao.com/yws/api/personal/file/WEB27159a6d3fb6cafa4dfc7fbc5883a211?method=download&shareKey=90098eb080c078296e3b0fa0fd260ee6 "Overall GPU Memory on a 4×A100-80GB (PCIe)")

### 8x L20 (PCIe) 集群

| **Latency** | **PipeFusion** | **Tensor Parallel** | **DistriFusion** | **Seq Parallel (Ulysses)** | **Seq Parallel (Ring)** | Single-GPU |
| ----------------- | -------------------- | ------------------------- | ---------------------- | -------------------------------- | ----------------------------- | ---------- |
| **1024px**  | **1.00x**      | 2.46x                     | 3.26x                  | 1.48x                            | 4.42x                         | 2.46x      |
| **2048px**  | 0.99x                | 2.26x                     | **1.00x**        | 1.58x                            | 1.09x                         | 4.16x      |
| **4096px**  | **1.00x**      | 1.16x                     | OOM                    | 1.31x                            | 4.40x                         | 4.30x      |

![Overall latency on a 8×L20 (PCIe)](https://note.youdao.com/yws/api/personal/file/WEB883bda408bc38824e7bda7425ae4fb51?method=download&shareKey=403004516bb9ff8d9271e0f8ef88a693 "Overall latency on a 8×L20 (PCIe)")


### 8x A100 (NVLink) 集群

| **Latency** | **PipeFusion** | **Tensor Parallel** | **DistriFusion** | **Seq Parallel (Ulysses)** | **Seq Parallel (Ring)** | Single-GPU |
| ----------------- | -------------------- | ------------------------- | ---------------------- | -------------------------------- | ----------------------------- | ---------- |
| **1024px**  | 1.26x                | 1.59x                     | **1.00x**        | 1.79x                            | 3.38x                         | 2.15x      |
| **2048px**  | 1.64x                | 2.85x                     | **1.00x**        | **1.00x**                  | 1.43x                         | 3.99x      |
| **4096px**  | 1.08x                | 1.56x                     | **1.00x**        | 1.18x                            | 1.93x                         | 7.28x      |
| **8192px**  | 1.35x                | **1.00x**           | OOM                    | OOM                              | OOM                           | 5.98x      |

![Overall latency on a 8×A100 (NVLink)](https://note.youdao.com/yws/api/personal/file/WEBf294c0c56206b35fb2bd495b65caa1f8?method=download&shareKey=a6780d54a1429bba9fa05174f1ab44e8 "Overall latency on a 8×A100 (NVLink)")


### Scalability

PipeFusion 在 NVLink 和 PCIe 上的时延相似，PCIe 甚至在表现出了轻微的优势。在 PCIe 集群上，对于相同的任务，PipeFusion 总是比 DistriFusion 快。说明 PipeFusion 的通信带宽要求非常低，因此不需要使用 NVLink 等高带宽网络。

![Scalability of PipeFusion and DistriFusion on A100 PCIe vs. NVLink cluster](https://note.youdao.com/yws/api/personal/file/WEBbd3f234475f8a4dd0831cb0de02c3023?method=download&shareKey=186d86f87e9eaadc6df309ff516b2b1c "Scalability of PipeFusion and DistriFusion on A100 PCIe vs. NVLink cluster")

## Ablation Studies

随着 patch 数目 M 的增加，内存消耗减少，并且对通信没有影响。但在实践中，M 不应该设置得太高。生成 1024px 和 2048px 图像时，当 M 超过一定阈值时，整体延迟增加。然而，这种现象很少出现在高分辨率图像 4K×4K 的情况下。这是因为过于细粒度的计算分区会导致 GPU 的理论吞吐量下降。

![Latency of PipeFusion with various patch numbers M](https://note.youdao.com/yws/api/personal/file/WEBeff3ad6b39db3ce27ce5019245deecbc?method=download&shareKey=65604fe4530f3c7236a57f0497d163c4 "Latency of PipeFusion with various patch numbers M")

绝大多数差异可以忽略不计或接近零，即扩散过程中连续步骤输入之间的高度相似性。

有一些方法可以减轻由预热步骤引起的性能损失: 增加采样步骤，在单独的设备上执行，利用序列或张量并行。

# Summary

我们的方法是先用 Pipeline Parallel 将模型的 transformer block 切分成多个 stage, 再用 Tensor Parallel (Megatron: 切分前一个权重的列，后一个权重的行, Two-dimenson: 切分输入的列，切分权重的行和列)，每一层的 KV 结果需要进行 all-reduce 或者 all-gather + reduce-scatter. 不同 stage 之间是 P2P 通信.

PipeFusion 行为更像单纯的 Pipeline Parallel，利用上一步的 KV 完成当前步的计算，P2P 通信的是自己所处 stage 的激活 (与切分的 patch 数成反比)，与 transformer block 的层数无关。

[xDiT的分析中](https://darkenstar.github.io/2024/09/27/xDiT/#Construct-Parallel-Groups)提到过将并行维度从小到大可以分为 TP-SP-PP-CFG-DP，其中 CFG 和 DP 实际上是对 数据的 batchsize 维度进行切分，PP 的大小取决于划分的 patch 数，每个 stage 的 transformer block 计算的时候可以进一步再进行 SP 和 TP.


# References

[^1]: https://darkenstar.github.io/blogs/MegatronLM/
[^2]: https://darkenstar.github.io/blogs/ringattention/
[^3]: https://darkenstar.github.io/blogs/deepspeedulysses/
[^4]: https://darkenstar.github.io/blogs/distrifusion/