---
title: "Oltron"
date: 2025-10-02T20:56:51+08:00
lastmod: 2025-10-02T20:56:51+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- Quantization

keywords:
- Quantization

description: "Paper Reading of Oltron." # 文章描述，与搜索优化相关
summary: "Paper Reading of Oltron." # 文章简单描述，会展示在主页
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

研究人员提出了几种混合精度量化技术来管理这些激活异常值。这些方法采用了 value-wise 异常值粒度，在平衡模型准确性和硬件效率方面面临挑战。为了解决这个问题，作者发现 LLM 的激活异常值通常聚集在特定通道内。因此，作者发明了 Oltron，这是一种全面的软件/硬件协同设计策略，用于具有层间/层内适应的 LLM 异常值感知量化。方法包括三个关键创新：
1. 一种新的量化算法，可以识别不同层之间的异常值和一层内通道组的最佳比例；
2. 一个可重构的架构，适应层间和层内分布；
3. 一个基于tile的数据流优化器，它可以复杂地安排混合精度张量的复杂计算和内存访问。

Oltron优于最先进的异常值感知加速器OliVe，实现了1.9倍的性能提升和1.6倍的能源效率，同时还提高了模型精度。

# 1. Introduction

目前，LLM 的权重可以量化到 4-bit，然而由于异常值问题激活量化仍然具有挑战性。一小部分激活值具有极大的幅度，对模型精度的主要影响，使得很难在不牺牲模型精度的情况下应用 4-bit 激活量化。为了保持模型的准确性，以前的工作已经提出了具有专用硬件支持的异常值值感知量化，例如OLAccel 和 GOBO. 这些方法采用混合精度量化，其中低比特 4-bit 量化应用于大多数正态值，而高精度 (16比特) 保留给异常值。这种方法实现了几乎平均的 4-bit 量化，精度损失可以忽略不计。然而，它们将异常值视为稀疏矩阵，需要额外的码本来处理稀疏格式和复杂的控制逻辑来处理不规则计算和内存访问。另一方面，OliVe 引入了具有局部异常值存储的异常值-受害者对 (OVP) 编码，以最小的控制开销确保对齐的内存访问。然而，局部异常值点编码方案在扩展异常值点位宽度方面存在局限性，导致llm的精度不足。此外，为了局部处理随机出现的异常值值，OliVe 增加了所有计算单元处理异常值值的能力，这损害了硬件效率。

论文的创新在于观察到LLMs激活异常值倾向于聚类在特定通道中，而不是完全随机分布。这启发了作者提出Oltron框架，一个软硬件协同设计的 **异常值感知量化 (Outlier-Aware Quantization) 方案，强调层间 (Inter-Layer) 和层内 (Intra-Layer)** 适应性。Oltron的核心组件包括：

- Tile-wise Outlier-Balanced Encoding (TOBE)：一种新型编码格式，将显著通道 (salient channels) 以高精度编码，同时通过通道重排序确保 Tile 大小均衡，实现规律的内存访问和计算数据流。
- 自适应量化算法：自动优化每层的异常通道比例和精度，平衡存储预算与模型精度。
- 数据流优化：通过编译时优化 (如内核融合和静态重排序) 消除运行时重排序开销。
- 可重配置架构：混合**处理单元 (Processing Elements, PEs) **设计，包括高效PE和灵活PE，支持不同精度计算。

# 2. Background & Motivation

**显著通道**指那些数值幅度远大于大多数其他通道的特殊通道。其显著性体现在两个方面:
1. 在相同的位宽表示下显著通道的量化误差大得多。
2. 如果把显著通道和正常通道混在一起量化
    - 截断显著通道: 为了适应正常通道的小范围，整个张量的量化尺度会设得很小，导致显著通道的大值被强行挤压到上限丢失信息。
    - 舍入误差: 反之，如果尺度调大适应显著通道，正常通道的小值在低比特下容易被四舍五入成 0，造成过度舍入。

**层间非均匀性**是指不同层间显著不同的凸道组成比例。例如，与图1(a)所示的注意输入映射层相比，图1(b)所示的MLP收缩层包含的异常通道数量要多得多。因此，我们将前者归类为异常值丰富层，后者归类为异常值稀缺层。

**层内非均匀性**是指显著通道在一层内随机分布，导致显著通道在通道组间的比例不同。在图1 (c-f) 中，在两个选定的层中四个不同的通道组之间存在明显的显著分布差异，每个通道组由128个通道组成。有些组别 (如图1(c)) 完全没有异常值，而另一些组别 (如图1(f)) 则显示出较大比例的突出通道，其余组别介于两者之间。

![Figure 1. Input activation distribution of linear layers from LLaMA65B. For illustration, channels of > 2× median magnitude are deemed salient. (a) and (b) are high-level views resized with maxpooling. (c-d) / (e-f) are zoom-in views of (a) / (b), respectively.](https://share.note.youdao.com/yws/api/personal/file/WEB48cdc68c8fe078f10eded5e4da630a17?method=download&shareKey=8c0ffb2d3484b51d767a080a2036a08e "Figure 1. Input activation distribution of linear layers from LLaMA65B. For illustration, channels of > 2× median magnitude are deemed salient. (a) and (b) are high-level views resized with maxpooling. (c-d) / (e-f) are zoom-in views of (a) / (b), respectively.")

![Figure 2. Oltron’s framework overview.](https://share.note.youdao.com/yws/api/personal/file/WEBe52996f5528afa040e56ae3b49b29e3c?method=download&shareKey=ab0c738bc4a5d1385da3cc098bb3946e "Figure 2. Oltron’s framework overview.")

# 3. Software Design 

## 3.1 Tile-wise Outlier-Balanced Encoding (TOBE)

对混合精度张量进行编码的方法如下
1. Original: 将异常值存储在其原始位置 (图3(a)). 由于异常值需要额外的编码位宽度，这种方法会导致不规则的张量存储和不一致的数据块大小。
2. Split: 通过分离稀疏的异常值编码来解决内存对齐的挑战，以确保正常 Tile 的均匀性 (图3(b))，但变长异常值编码仍然存在问题。它还引入了复杂的控制开销来管理单独存储和计算的正常/异常值。
3. OVP: 修剪异常值附近的受害值，并分配保存的位宽度用于扩展异常值表示 (图3(c)). 然而，这种方法只能为异常点提供有限的位宽扩展，导致异常点表示的精度受到约束，最终影响模型的精度。
4. TOBE: 通过在通道粒度上调整精度来实现规律的内存访问模式，并对高精度通道进行重新排序，以确保平铺子块的大小平衡，从而大大降低了稀疏异常值所需的复杂控制逻辑 (图3(d)). 与以前的编码方案单独处理异常值不同，TOBE以高精度编码具有严重异常值问题的显著通道。为了实现 tile 大小的平衡，TOBE 对通道排列重新排序，将显著通道均匀地分布到 tile 中。由于片上缓冲区以数据块为单位访问片外存储器，因此TOBE的统一数据块大小确保了对片外存储器的常规访问。此外，在每个 tile 内，指定的突出通道被放置在最前面。tile 的确定和一致的数据布局确保片上存储器的访问和计算可以以规则的方式进行。

![Figure 3. Comparisons between different encoding format.](https://share.note.youdao.com/yws/api/personal/file/WEBa73b70caf76917b6fc6bcbcfb22f3988?method=download&shareKey=901dd0ba411d69b998b6e603c46ad83e "Figure 3. Comparisons between different encoding format.")

Oltron 主要将 TOBE 应用于表现出明显异常值问题的线性层的输入激活张量。为了保证矩阵乘法结果的正确性，给定激活张量的重排序通道排列，我们还需要对权张量的输入通道进行相应的重排序，如图4(a)所示。由于突出通道是根据其在校准数据上的极值来选择的，因此通道排列是预先确定的。因此，权重张量的行可以在部署前进行静态重排列。

## 3.2 Dataflow Optimization

![Figure 4. TOBE-based computation workflow. (a) Linear layers adopt TOBE-based matrix multiplication. (b) The encoding overhead of explicit run-time reordering is mitigated with graph-level compilation-time optimization.](https://share.note.youdao.com/yws/api/personal/file/WEB266b7ca7517308c46443f0818022d098?method=download&shareKey=d9dbae87818e68deed0a47532c164773 "Figure 4. TOBE-based computation workflow. (a) Linear layers adopt TOBE-based matrix multiplication. (b) The encoding overhead of explicit run-time reordering is mitigated with graph-level compilation-time optimization.")

TOBE 在内存中按块重新排序张量布局，以提高层内计算效率。然而，这种编码需要显式的重排操作来准备数据。因为前一个算子的输出不一定是以 TOBE 格式编码的。

1. 利用矩阵乘法固有的交换特性: 在 M × N × K 矩阵乘法中，重新排序操作可以直接应用于前一层的输出。例如，重新排序 M 维中的行或 N 维中的列不会影响最终结果。此外，如果两个矩阵乘法层之间的层也表现出交换特性，例如 element-wise (激活层) ，则这种重新排序策略可以传播。因此将 $O_{proj}$ 和 $FC_2$ 的重排操作转换成前一个权重矩阵列重排。

2. 使用核融合方法将重排序操作与前一层合并： 对于 $O_{proj}$ / $O_{proj}$ / $O_{proj}$ 和 $FC_1$，将重排融合到前一步的的LayerNorm 中，其中每个输出通道被重定向到片外存储器中的不同地址，以匹配下一层输出的 TOBE 格式。

在框架的编译过程中实现这个数据流优化。遍历整个计算图，检查每一对线性层 (例如，FC1和FC2). 作者额外分析了它们之间直接连接的操作符 (例如，激活层)，并应用上述两种策略来消除由显式重新排序操作引起的任何额外的内存访问。

## 3.3 Adaptive Quantization Algorithm

算法在给定目标平均位宽 B 下自动确定两个因子: 显著通道所占比例以及显著通道使用的位宽 (fp8/12/16). 在不超过预算时基于MSE 优化，迭代调整阈值 τ 和类型 t.

# 4. Architecture Design

![Figure 5. Overview of Oltron architecture.](https://share.note.youdao.com/yws/api/personal/file/WEBd952fb0a88d9753a4d9bf3c1d8458bca?method=download&shareKey=49cf209d63e55d13cc8c152cb06489a7 "Figure 5. Overview of Oltron architecture.")

Oltron 架构采用了类似于谷歌 TPU 的 4 级流水线:
1. 读取片外内存：权重/输入缓冲区从片外内存加载 TOBE 编码后的数据块。
2. 预加载：根据输入 tile 的异常通道配置，对权重 tile 进行相应的预加载，控制单元将控制信号发送给解码器。
3. 矩阵乘法：输入/输出通道的数据分别在对应的 PE 阵列的列/行中流动。正常值和异常值的 MAC 操作可以在芯片上并发地无缝执行，因为正常值和异常值的输入值都使用相同的量化尺度，并且它们的乘法结果以统一的整数格式存储。
4. 写片外内存：最后，输出缓冲区将累积的输出块写回片外内存。

![Figure 6. Oltron PE design.](https://share.note.youdao.com/yws/api/personal/file/WEBad564c1c3e4e22a4ab817ffc9f86136c?method=download&shareKey=48e8577eb45b6dac329c71ccc8aa179d "Figure 6. Oltron PE design.")

Oltron 的架构采用混合 PE 设计，大多数 PE 采用简化设计以提高效率，少数 PE 采用灵活设计以提供适应性。
- Efficient PE: 只能支持有限累积位宽 (16位) 的有符号 int4×int4 乘法，从而获得更好的能效和更小的面积。
- Flexible PE:  在 Efficient PE 设计的基础上进行了改进，以获得更好的灵活性。首先，原始的 4 位输入整数寄存器扩展了一个额外的符号位以支持有符号和无符号输入操作数。其次，为了支持 int×fp 乘法，添加了一个由 5 位输入指数控制的移位器。最后，我们将累积位宽度扩展到32，以捕获 fp 输入的高动态范围。通过以上修改，每个 Flexbile PE 可以支持 int4×fp8 (E5M2)，并且可以组成多个 PE来支持更多尾数位的 fp 输入。

![Figure 7. Oltron Decoder Design.](https://share.note.youdao.com/yws/api/personal/file/WEBc7fc2e512e2b34724993095ae35f677a?method=download&shareKey=ec4dd3ce78a14a7c484ea2661cc06abd "Figure 7. Oltron Decoder Design.")

Oltron 采用可重构解码器设计来支持混合数据类型的处理，包括 int4，fp8和fp12。目前的解码器设计支持高达12位的fp精度，因为fp12可以在测试的llm上实现与fp16相当的精度结果。但是，如果需要的话，扩展当前的设计以支持更高的fp精度是可行的。使用指定的数据类型，解码器相应地访问片上缓冲区。当访问片上缓冲区时，它加载1字节 (两个int4) 或2字节 (两个fp8或一个填充4位的fp12)  (图7(a)) 。解码器还将获取的数据转换为统一的指数整数对格式，以简化后续计算。对于int4，解码器用零填充指数字段，用输入值填充整数字段 (图7(b)) 。对于 fp 的解码公式如下

$$
sign × (1 << mb + mantissa) << (exponent − bias)  \tag{1}
$$

- sign：符号位 (1-bit) ，0表示正，1表示负。负值时，整个结果取反 (或在MAC时处理) 。
- mb：尾数位宽 (Mantissa Bit-Width) ，即 mantissa 字段的比特数。
    - 对于fp8 (E5M2) ：mb=2 (5-bit指数 + 2-bit尾数，隐含1位) 。
    - 对于fp12 (E5M6) ：mb=6 (5-bit指数 + 6-bit尾数，隐含1位) 。
- mantissa：尾数值，无符号整数 ($0 ~ 2^{mb}-1$) ，代表小数部分 (实际值 = $1.mantissa / 2^{mb}$，隐含前导1) 。
- exponent：原始指数字段 (无符号) 。
- bias：偏置常数，固定值，用于将无符号指数转为有符号 (避免负指数) 。论文强调“constant bias is properly selected to avoid any fractions in integer fields”——bias选为2^{mb-1} (e.g., fp8 bias=15 for 5-bit exp) ，确保移位后整数无小数。

当 mantissa 最大时，$integer_max = 2^{mb} + (2^{mb} - 1) = 2^{mb+1} - 1$. 所以 fp8/12 最大值为 7/127，考虑到符号位
分别需要 4/8 位。因此 fp8 只需要一个 PE，而 fp12 需要两个 PE 协作完成 MAC.