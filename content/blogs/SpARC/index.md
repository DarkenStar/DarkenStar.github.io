---
title: "SpARC"
date: 2025-10-03T17:48:20+08:00
lastmod: 2025-10-03T17:48:20+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- Accelerator

keywords:
- Accelerator

description: "Paper Reading of SpARC." # 文章描述，与搜索优化相关
summary: "Paper Reading of SpARC." # 文章简单描述，会展示在主页
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

作者提出了SpARC，一种稀疏注意力转换器加速器，通过降低自注意力机制的计算复杂度来提高吞吐量和能量效率。利用 Transformer 注意图中固有的行级冗余来减少整体的自注意计算。通过采用逐行聚类，每个类只计算一次注意力分数，而不会严重影响准确性。为了利用所提出的聚类近似注意力的高并行性，我们开发了一个具有专用内存层次结构的全流水线加速器。实验结果表明，SpARC 的注意图稀疏度达到 85-90%，精度损失可以忽略不计。与先前的稀疏注意力转换器加速器相比，SpARC实现了高达4倍的核心注意力加速和6倍的能效提高。

# 1. Introduction

稀疏注意力旨在通过选择性地跳过对低相关性 token pairs 的计算，减轻密集注意力的计算负担，有效消除整个自注意过程中潜在的冗余。SpARC，一种新的软硬件协同设计框架，通过逐行结构化稀疏注意加速注意力计算，主要贡献可以概括如下: 

- 提出了一种计算高效的 token 聚类近似注意力，它分析 token 之间的相似性，将它们分组到簇中以实现行级稀疏。这种方法只需要计算每个簇的质心，从而降低了计算复杂度。所提出的注意力行聚类近似注意力在平衡延迟和性能方面提供了灵活性。
- 提出了SpARC硬件架构，通过高度流水线化的专用加速器有效地利用了所提出的聚类近似注意力固有的数据并行性。此外，实现了一个专用的内存层次结构，称为集群质心缓存，以显着减少与不规则访问模式相关的DRAM访问流量和延迟开销。
- SpARC的高性能在三种不同的模型上进行了演示: GPT-2， BERT， ViT和五种不同的基准: SQuAD v1.1， MNLI， CLOTH, Wikitext-2, ImageNet-1K。实施结果表明，SpARC可以持续实现 0.85-0.9 的平均稀疏度，精度损失可以忽略不计，核心注意力加速提高了6倍，与之前最先进的加速器相比，能效提高了4倍。

# 2. Background

![Figure 1: (a) Transformer architecture [1] (b) Prior sparse attention approach](https://share.note.youdao.com/yws/api/personal/file/WEB9899a2221888ceaaf0209689d974ac98?method=download&shareKey=9b4e350648338f9d3d2a10fa30d1e51e "Figure 1: (a) Transformer architecture [1] (b) Prior sparse attention approach")

稀虽然注意力机制旨在捕获输入序列中标记之间的关系，但并非所有标记都表现出强的关系。例如，"a" 或 "the" 等标记可能不会显着影响序列处理，Softmax 函数输出接近零。如图1 (b) 所示，先前的工作利用这一观察结果，通过省略逐元素稀疏注意力图的近零值的计算来降低计算复杂性。然而，由于运行时稀疏性模式的动态性质，这种元素方面的稀疏性给高效的硬件利用带来了挑战。非常需要硬件友好的结构化稀疏注意力机制来充分利用稀疏注意力的计算优势。

# 3. Proposed Row-Cluter Approximate Attention

![Figure 2: Motivation of the row-clustering approximate attention (a) Sample dense attention (b) Row-clustered attention](https://share.note.youdao.com/yws/api/personal/file/WEB4ef8ad77bd375d4ba09116fcae4a2242?method=download&shareKey=7df7a780e4c9b804e6d20203c15e1477 "Figure 2: Motivation of the row-clustering approximate attention (a) Sample dense attention (b) Row-clustered attention")

在图2 (a)中，可以明显看出密集注意图的行{0,3,7}、{1,2,5}和{4,6}显示出相似的模式。这些相似的行模式使得作者想到了对行进行聚类，其中具有相似模式的行被分组在一起，如图2 (b)所示。核心思想是通过计算每个代表性类别行即质心的注意力得分来近似整个注意力地图，而不是逐行计算。此外，这种基于逐行聚类的方法通过调整类别的数量，在平衡计算减少和性能方面提供了灵活性。

![Figure 3: Proposed row-clustering approximate attention](https://share.note.youdao.com/yws/api/personal/file/WEB2ef203f4cd24f9d097b28dfcf176c82e?method=download&shareKey=83f4205fa4d205b6714e98a4ce61fb5d "Figure 3: Proposed row-clustering approximate attention")

1. Low-rank Extraction: 使用线性判别分析 (LDA，Linear Discriminant Analysis) 离线选 $K^T$ 的特定列 (论文用15%) ，生成低秩近似 $\hat{K}^T$，然后 $Q \times \hat{K}_{ij}^T$ 得到低秩注意力预矩阵。
2. Centroid Initialization: 初始化为每一类所负责的所有行 $A_c$ 的平均值。
$$
Centroid_c=\frac{\sum_{j\in A_c}Q\hat{K}_{ij}^T}{|A_c|},
$$
3. Similarity Compute: 计算每行ˆQKᵀ与所有中心的L1距离。选择L1因无乘法，低开销。生成的相似度矩阵可以表示为
$$
Sim_{ij}=||\hat{QK_i^T}-Centroid_j||_1.
$$
4. Cluster Allocation: 为每行分配最近中心的簇，然后列扫描 Sim，选簇内最相似行作为新中心索引。
5. Centroid Retrieval: 从Q中检索每个簇的代表行 $Q_c$ 进行注意力计算。
6. Norm & Softmax: 用 $Q_c$ 计算 $Q_c K^T$，缩放、Softmax 得 $A_c$，然后 $Z_c = A_c V$.
7. 将 $Z_c\in\mathbb{R}^{C\times\frac Dh}$ 扩展回原尺寸 $hat{Z}\in\mathbb{R}^{C\times\frac Dh}$. 扩展规则为用相应的簇质心替换属于对应簇的所有行。

通过考虑准确性和计算复杂性之间的权衡来预先确定每个变压器块的簇数。为了衡量聚类带来的误差，采用 KL 散度作为指向性度量来比较聚类前后的注意图。我们考虑原注意矩阵与克隆A后聚类注意矩阵之间的误差如下: 
$$
error=\sum_{i=0}^NKL(A_{i},\hat{A_{i}}). 
$$

通过为每个 transformer block 建立一个全局错误阈值 t，可以确定满足指定错误阈值 ≤ t 的最小集群数量。这种全局阈值方法允许不同 transformer block 的聚类数量不同。

# 4. SpARC Hardware Architecture

虽然该方法显著降低了自关注机制的计算复杂度，但现有的通用计算平台并不适合加速所提出的聚类近似关注。这种限制源于两个主要因素。
1. 一般计算平台没有针对行聚类近似注意力中涉及的特定操作 (如绝对距离和最小距离定位计算) 进行优化。
2. 行聚类近似关注算法在利用数据并行性方面面临着挑战。聚类过程中固有的数据依赖性限制了跨多个处理单元并行计算的能力，从而限制了在这些平台上可实现的性能增益。

![Figure 4: SpARC hardware architecture](https://share.note.youdao.com/yws/api/personal/file/WEBd0b0a00dad06f44d34d4d9c475ea20c5?method=download&shareKey=c4e171f81d17e5af53cb6a7720154801 "Figure 4: SpARC hardware architecture")

硬件设计上，SpARC是全流水线加速器 (图4)，核心是矩阵乘法单元 (MMU，Matrix Multiply Unit): 16个向量处理器，每处理器64 MAC (乘加) 线，支持INT16，配可重构加法网络 (reconfigurable adder network) 适应数据流。子处理器包括: 除法单元 (Div. Unit，基于倒数单除法器，64 路并行乘法) ；指数单元 (Exp. Unit，Taylor级数近似64并行exp)；距离单元 (Dist. Unit，64 路并行绝对值)；聚类单元 (Clust. Unit，48 个比较器，支持 16 路并行3输入min查找).

![Figure 5: SpARC computation pipeline](https://share.note.youdao.com/yws/api/personal/file/WEB96279d27988d2116fe240d24ca057a19?method=download&shareKey=d76ddd9772ac62aa0d267ddb6ae892e9 "Figure 5: SpARC computation pipeline")

流水线设计上，利用不同行之间的计算独立性，一旦获得了 Q 和 K 矩阵，MMU 就开始逐行计算 $Q \hat{K}^T$. 计算完每一行时，将其转发给加法器网络和除法器单元进行插值质心初始化。类似地，每个计算出的质心随后传递给除法器单元和绝对距离单元，用于归一化和L1距离计算。在完成相似性矩阵的每一行后，聚类单元内的索引搜索器开始遍历矩阵，将 $Q \hat{K}^T$ 的每一行分配给一个聚类，并更新相应的质心。

在计算L1距离和 cluster allocation 时，MMU同时计算V矩阵，通过消除MMU空闲时间来确保高吞吐量。生成V和 cluster allocation 完成后，收集每个质心对应的行进行低成本的 $Q_cK^T$，然后进行 Softmax、$Z_c = A_cV$ 和线性投影计算。

![Figure 6: (a) Centroid cache architecture (b) Normalized DRAM access versus different cache sizes for multiple replacement policies and attention map density](https://share.note.youdao.com/yws/api/personal/file/WEB3109ea826aec6c129044928e7dcc2502?method=download&shareKey=4f3534f71b0b6c47840589585578a5a2 "Figure 6: (a) Centroid cache architecture (b) Normalized DRAM access versus different cache sizes for multiple replacement policies and attention map density")

clone 过程将每个集群中的行替换为其相应的质心。虽然所提出的算法和硬件架构显著提高了能效和速度，但聚类过程的动态性和向集群分配行的不规则性由于不可预测的访问模式而引入了主内存开销。在每次请求时从主存加载质心会导致碎片化而不是连续的内存访问，从而导致过多的数据移动、内存带宽利用率降低和内存访问能耗增加。为了解决这个问题增加了一个质心缓存 (32KB LRU)，当请求一个质心进行计算时，请求被发送到内存获取器。获取器检查缓存以确定请求的质心是否存在。如果发生缓存命中 (请求1)，则获取器从缓存中检索质心，从而消除了访问主内存的需要。在缓存丢失 (请求2) 的情况下，获取器从主内存加载质心，将其存储在质心缓存中以备将来使用。