---
title: "HASTILY"
date: 2025-10-07T09:33:58+08:00
lastmod: 2025-10-07T09:33:58+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- CIM

keywords:
- CIM

description: "Paper reading of HASTILY." # 文章描述，与搜索优化相关
summary: "Paper reading of HASTILY." # 文章简单描述，会展示在主页
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

本文在利用内存计算 (CIM) 架构的同时，通过硬件软件协同设计方法解决了与加速注意力相关的挑战。HASTILY 旨在加速softmax计算，注意力的积分运算，并最大限度地减少其高片上内存需求，随着输入序列长度的二次增长。我们的体系结构由称为统一计算和查找模块 (Unified Compute and Lookup Modules, UCLMs) 的全新CIM单元组成，这些单元在相同的SRAM阵列中集成了查找和乘法累加功能，比标准CIM阵列产生最小的面积开销。UCLMs采用台积电65nm工艺设计，可同时执行指数和矩阵向量乘法运算。为了补充所提出的体系结构，HASTILY 采用了细粒度的流水线策略来调度注意层和前馈层，以减少对序列长度的二次依赖到线性依赖。此外，对于涉及计算指数值的最大值和和的快速softmax计算，这些操作使用reduce和gather策略在多个核心上并行化。

# 1. Introduction

加速Transformer面临两个独特的挑战:
1. 对复杂的Softmax运算的依赖: 输入序列变长时，Softmax 的计算时间不成比例地疯涨，逐渐成为与矩阵乘法同样致命的性能瓶颈 
2. 中间矩阵带来的巨大内存占用: 注意力计算过程中会生成一个巨大的中间矩阵 $QK^T$. 这个矩阵的大小与输入序列长度 l 的平方成正比，对硬件的片上内存容量提出高要求。

CIM 通过在内存中直接计算，能够有效突破内存墙，并且在加速CNN等权重固定的网络时表现优异。然而，Transformer中的注意力矩阵是动态生成的，这使得CIM的权重固定优势大打折扣，因为需要频繁地将新生成的矩阵写入内存，带来了额外的延迟和能耗开销，之前的工作未能解决这个问题。

![Fig. 2. (a) Comparison of our work, HASTILY, with other works addressing the challenges associated with softmax computation. (b) Distinguishing the fine-grained pipelining technique proposed in this work compared to a prior work, ReTransformer.](https://share.note.youdao.com/yws/api/personal/file/WEB42529a3a28f40350014b301f83c84e9c?method=download&shareKey=885949443b736e66a6a14a881c756f19 "Fig. 2. (a) Comparison of our work, HASTILY, with other works addressing the challenges associated with softmax computation. (b) Distinguishing the fine-grained pipelining technique proposed in this work compared to a prior work, ReTransformer.")

HASTILY 是一个软硬件协同设计框架。
- 硬件层面: 设计了新的 UCLMs，它基于一种双功能8T-SRAM阵列，可以在同一个硬件单元内同时执行计算 (矩阵乘法) 和查表 (用于Softmax中的指数运算)，且几乎没有额外的面积开销。

- 软件层面: 针对Softmax中的求最大值和求和操作，设计了跨多核并行的规约和收集策略，以减少计算延迟。为了解决内存瓶颈，提出了一种**细粒度流水线 (fine-grained pipelining)** 策略。该策略将计算从“矩阵级”拆解为“向量级”，让数据在不同计算单元之间像流水一样处理，从而将内存需求从二次方复杂度降低到线性复杂度。

# 2. Preliminaries and Challenges

为了保证数值稳定性 (避免因指数运算导致数值溢出)，Softmax的计算通常遵循以下公式

$$
\mathrm{softmax}(M_{i,j})=\frac{\exp(M_{i,j}-max_j)}{\sum_j\exp(M_{i,j}-max_j)}\tag{1}
$$

要完成Softmax，需要执行一连串复杂的操作: 
1. Maxima: 找到一个向量中的最大值。
2. Subtraction: 向量中的每个元素都减去这个最大值。
3. Exponent: 对减法后的每个元素计算 $e^x$.
4. Reduction: 将所有指数运算结果相加，得到分母。
5. Division: 每个指数结果除以总和。

指数运算对于硬件来说非常不友好。论文在表 I 中对比了几种实现指数运算的方法: 纯软件计算延迟高；使用片上查找表 (LUT) 会占用宝贵的芯片面积；运行时从片外加载查找表又会引入巨大的延迟和功耗。

CIM 通过改造内存 (如SRAM) 的外围电路，使其能够在内存阵列内部直接执行矩阵向量乘法，从而避免了数据在处理器和内存之间的频繁搬运。模拟CIM通过同时激活多行 (Wordline) ，在列 (Bitline) 上以模拟电流/电压累加的方式完成乘加运算，最后通过 ADC 将结果转换为数字信号。

一个经典的、以PUMA架构为代表的CIM加速器层级结构由以下三层组成:
- 芯片级 (Chip level): 由多个Tile和一个**全局缓冲 (Global Buffer)**构成。
- Tile级 (Tile level): 由多个Core和一个**共享内存 (Shared Memory)**构成。
- 核心级 (Core level): 由多个 MVM单元 (MVMUs)、一个寄存器堆和一个**向量功能单元 (Vector Function Unit, VFU)** 构成。VFU是一个可编程单元，负责执行标准的向量化算术运算和非线性函数。

# 3. Proposed CIM Core Architectures

![Fig. 3. (a) The hierarchical spatial architecture of the proposed CIM accelerator, (b) hardware architecture of each core in HASTILY, (c) micro-architecture details of each UCLM consisting of multiple SRAM arrays, (d) physical layout of each dual-functionality 8T-SRAM array implemented in TSMC65nm and (e) depiction of the two operations, compute on the left and lookup on the right, in UCLM and correspondingly in the SRAM.](https://share.note.youdao.com/yws/api/personal/file/WEB2c90407fed5e61c62c81b9f26bdf2c6e?method=download&shareKey=73b342d87744869f31b83615e6083908 "Fig. 3. (a) The hierarchical spatial architecture of the proposed CIM accelerator, (b) hardware architecture of each core in HASTILY, (c) micro-architecture details of each UCLM consisting of multiple SRAM arrays, (d) physical layout of each dual-functionality 8T-SRAM array implemented in TSMC65nm and (e) depiction of the two operations, compute on the left and lookup on the right, in UCLM and correspondingly in the SRAM.")

HASTILY 加速器总体架构是一个分层的空间加速器，由多个Tile构成，每个Tile又包含多个Core (如图3(a)所示). HASTILY Core的架构如图3(b)所示，它包含一个向量功能单元 (VFU) 、寄存器堆 (RF) 以及多个 UCLMs.

UCLMs 的基础是双功能8T-SRAM阵列，这种SRAM阵列被设计成可以执行两种完全不同的功能: 计算 (Compute) 和查找 (Lookup). 它通过在标准的8T-SRAM单元的每一行增加一条额外的控制线——查找线 (Lookup Line, LKL) ，就实现了全新的查找功能。存储在SRAM中的查找表数据，是通过在制造时选择将存储单元的源极线 (Sourceline) 连接到LKL或接地 (GND) 来硬编码的。作者在TSMC 65nm工艺下的物理版图实现 (图3(d))，增加这条LKL线并不会带来任何额外的芯片面积开销。

这种双功能SRAM阵列有三种工作模式: 
1. *Read/Write operation:* LKL线始终接地 (保持低电平) ，SRAM单元的行为与普通SRAM完全相同。
2. *Lookup operation:* 用于快速获取预存的 Softmax 指数函数值。该操作需要4个时钟周期完成，过程如下: 
    1. 保存原始数据: 将当前SRAM行中存储的原始权重数据读出，并暂存到一个缓冲区中，以防丢失。
    2. 预充电: 将该行的所有SRAM单元都写入 1.
    3. 执行查找: 将LKL线设置为高电平，并激活该行的读字线 (RWL) . 此时，如果某个单元的源极线连接到了LKL，其对应的读位线 (RBL) 会输出 1；如果连接到地，则输出 0.
    4. 恢复原始数据: 将缓冲区中保存的原始权重数据写回到SRAM行中。
3. *Compute operation*: 在此模式下，LKL线也保持接地。通过同时激活多个 RWL，输入的向量电压作用于字线，与存储在SRAM单元中的权重矩阵进行点积运算。运算结果体现为位线 (BL) 上的模拟电压或电流变化，该模拟结果经过采样保持 (S&H) 、模数转换器 (ADC) 和移位累加 (S&A) 单元处理后，得到最终的数字输出结果。

HASTILY进一步通过软硬件协同的方式，将Softmax计算的各个步骤进行分解和优化。
1. 使用UCLMs并行加速指数运算。HASTILY采用了一种高效的近似计算方法，将指数运算分解为查表、位移和乘法的组合。

$$
e^x=2^n\times2^{d/K}\times e^r
$$
- $K$: 一个预先确定的数，代表查找表有多少项。
- $2^{d/K}$: 通过 UCLMs 的查找功能从预存的表中获得，$d=\lfloor(x/log(2)-n)K\rfloor$.
- $2^n$: 对应移位操作，$n = \left\lfloor x/log(2)\right\rfloor $.
- $e^r$: 是一个很小的残差项，可以直接近似为 1 或者 1+r, $r=x-nlog(2)/K$.

![Fig. 4. (a) Each SRAM array stores a 128-entry lookup table for 2^k/128, 1 ≤ k ≤ 128 (b) Illustration of parallel exponent operations execution in multiple UCLMs in a core of HASTILY architecture.](https://share.note.youdao.com/yws/api/personal/file/WEB8fe078d06eca046f44db218e69aefc3f?method=download&shareKey=40923e6827c7dd93094eae8ab03d04e0 "Fig. 4. (a) Each SRAM array stores a 128-entry lookup table for 2^k/128, 1 ≤ k ≤ 128 (b) Illustration of parallel exponent operations execution in multiple UCLMs in a core of HASTILY architecture.")

考虑到128项指数表中每个值的位精度为16位，我们将指数表组织在大小为64 × 64的双功能SRAM数组中，如图4(a)所示。执行流程 (如图4(b)所示):
1. VFU计算地址: VFU首先并行计算出整个向量中每个元素对应的查表地址d和位移值 n.
2. UCLMs并行查找: 多个UCLMs中的SRAM阵列同时切换到查找模式，并行地查出所有 $2^{d/K}$.
3. VFU整合结果: VFU最后对查找到的值进行位移操作，完成整个向量的指数运算。

这个流程将高延迟的指数运算转化为了硬件极其擅长的并行查表和位移操作，实现了巨大加速。

2. 多核协同优化规约操作

![Fig. 5. Computing softmax by gathering all required vectors in a single core versus parallelizing the compute across multiple cores and gathering them in a tree-like fashion.](https://share.note.youdao.com/yws/api/personal/file/WEBaa26e8e1e08875cb3107ce3c5b552949?method=download&shareKey=9e9aa9ffc268b4fd40a92d77488f6a9c "Fig. 5. Computing softmax by gathering all required vectors in a single core versus parallelizing the compute across multiple cores and gathering them in a tree-like fashion.")

Softmax中还包含 max 和求和 sum 这类规约操作。HASTILY在软件层面采用了一种**树形规约 (Tree-based Reduction)** 策略 (如图5(b)所示).
1. 局部计算: 每个Core先独立计算自己所拥有的那部分数据的局部最大值或局部和。例如，Core 0计算max(A1,A2)，Core 1计算max(A3,A4).
2. 分层聚合: 这些局部结果像二叉树一样，通过片上网络进行层层聚合，最终得到全局的结果。
3. 结果分发: 全局结果再被分发回各个Core，用于后续的计算。

# 5. Fine-Grained Pipeline

关键思想是将每个计算阶段划分为输入矩阵的行数，允许每个阶段在包含多个 tile 和核心的空间架构上与其他阶段并行运行。Matmul可以分解成多个 MVM 执行，每个MVM计算输入的一行与权重矩阵之间的点积，产生相应的输出行。

1. *Q,K,V Projection:* 输入矩阵被同时送入三个计算单元，生成Q, K, V三个矩阵。因为在计算注意力分数 $QK^T$ 时，必须拥有完整的K矩阵，才能让Q矩阵中的每一个向量都与之作用。因此，必须等待K和V矩阵完全生成后，才能开始下一阶段的计算。K和V生成后按头划分作为下一个核心的权重。
2. *Pipelining Attention:* $Q_i^T$ 的一行 被送入UCLM，与已存为权重的 $K_i^T$ ​矩阵进行MVM运算，生成一个结果向量。该结果向量立刻在同一个UCLM中进行Scale和Softmax运算。Softmax的输出向量又被立即送往下一个UCLM，与 $V_i$ 矩阵进行MVM运算，生成最终注意力输出 P 的一个向量。计算$P_i$的同时，下一个 Q 向量的 $QK^T$ 和 Softmax 计算已经并行开始了，实现了完美的流水作业。
3. *Pipelining within Encoder Layer:* 注意力计算的输出向量P生成后，会立刻与输入进行残差连接和层归一化，这些也都是以向量为单位流水进行的。紧接着的两个 FFN 层，同样可以无缝地接入这个细粒度流水线。
4. *Pipelining between Encoder Layers:* 这种流水线可以跨越多个Encoder层。第一个Encoder层输出的第一个向量，可以立即作为第二个Encoder层的输入，启动其Q,K,V的计算。