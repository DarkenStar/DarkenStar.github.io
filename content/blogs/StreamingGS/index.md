---
title: "StreamingGS"
date: 2025-10-05T19:45:54+08:00
lastmod: 2025-10-05T19:45:54+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- 3DGS

keywords:
- 3DGS

description: "Paper reading of StreamingGS." # 文章描述，与搜索优化相关
summary: "Paper reading of StreamingGS." # 文章简单描述，会展示在主页
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

三维高斯溅射 (3DGS) 在资源有限的移动设备上，很难满足 90 fps 的实时要求，只能达到 2-9 fps. 现有的加速器注重计算效率，而忽视内存效率，导致冗余的DRAM流量。我们引入了 STREAMINGGS，这是一种完全流式3DGS算法架构协同设计，实现了细粒度的流水线，并通过从以 tile 为中心的渲染转换为以内存为中心的渲染来减少DRAM流量。结果表明，与移动 Ampere GPU 相比，实现了高达 45.7x 的加速和 62.9x 的节能。

# Introduction

![Fig. 1. Tile-centric rendering (a) vs. memory-centric rendering (b). The computation pattern is highlighted in green. Tile-centric rendering computes the complete pixel values tile-by-tile. Memory-centric rendering computes the partial pixel values voxel-by-voxel and regularizes the memory accesses.](https://share.note.youdao.com/yws/api/personal/file/WEBca85c2ce34379f145f1f6db3bcaaefeb?method=download&shareKey=fc800135baee78e033dad19d7b45cb3c "Fig. 1. Tile-centric rendering (a) vs. memory-centric rendering (b). The computation pattern is highlighted in green. Tile-centric rendering computes the complete pixel values tile-by-tile. Memory-centric rendering computes the partial pixel values voxel-by-voxel and regularizes the memory accesses.")

虽然有一些研究提出了用于3DGS 的专用加速器，但它们的设计主要是优化计算效率，并且在渲染过程中经常忽略片外内存流量，这使得它们难以满足现实场景中的实时性要求。在算法上，现有的方法侧重于一种以点为中心的呈现范式，这种范式需要在阶段之间进行过多的中间数据通信，导致图1(a)中频繁的片外流量。

我们的表征表明，在 90 FPS 的实时要求下，当前渲染范式的DRAM流量将超过当今现实场景中移动设备的带宽限制 (第II-B节). 传统的以 tile 为中心渲染中的高数据通信来自于对每个单独像素的顺序处理射线高斯交叉点，导致冗余的片外流量和内存不规则。为了解决带宽限制，我们主张将从以 tile 为中心转变为以内存为中心的方法。

作者为3DGS引入了一种流式算法来解决内存访问不规则性。关键思想是通过将场景划分为体素并在体素基础上执行来正则化稀疏高斯表示，如图1(b)所示。由于体素内的所有点都连续存储在DRAM中，允许将体素内的所有点流式传输到片上缓冲区，并仅将最终像素值写入片外，从而完全消除了各阶段之间的中间片外流量。

虽然以内存为中心的方法消除了中间的片外流量，但可能会在体素流中引入不必要的片外流量，因为**体素内的所有点都是作为一个整体加载到芯片上的**。为了减少不必要的片外流量，我们提出分层滤波来逐步排除不相关的高斯信号 (第III-B节). 具体来说，我们将每个高斯特征分成两部分，并有意保持前一半的轻量级。然后，我们执行粗略但轻量级的计算，根据前半部分 (即位置和半径) 过滤掉不相关的点。接下来，我们执行精确的滤波，通过加载后半部分来进一步去除不相关的高斯函数。为了进一步减少特征加载的开销，我们使用向量量化压缩后半部分。通过在芯片上存储一个紧凑的码本，我们只需要从DRAM中获取它们的轻量级索引。通过这种方式，我们大大减少了体素流的DRAM流量。

我们的贡献如下：
- 引入了一个以内存为中心的渲染范例，可以实现完全流式的3DGS渲染，完全避免了中间的片外流量。
- 提出了一种分层过滤方案，以进一步减少体素流的数据流量。
- 我们的架构支持我们的优化，与移动 Ampere GPU 相比实现了数量级的加速和节能。与GSCore相比，我们还实现了 2.1x 的加速和 2.3x 的节能。

# 2. Background and Characterization

![Fig. 2. The overall rendering pipeline of 3DGS. The percentage numbers represent the DRAM traffic proportion across stages.](https://share.note.youdao.com/yws/api/personal/file/WEBd972bd4f39af828b20e3b43c28a8288e?method=download&shareKey=d8565113aace8a2471ac02056558f05a "Fig. 2. The overall rendering pipeline of 3DGS. The percentage numbers represent the DRAM traffic proportion across stages.")

3DGS 使用高斯椭球体以逐 tile 的方式渲染场景。如图2所示，该过程可分为三个阶段：
- Projection: 如图2左侧所示，此阶段将每个高斯椭球体投影到 2D 屏幕上，并确定其相交的 tile. 彩色虚线箭头所示，每个高斯分布可能与多个图块相交。同时，这个阶段为后面的阶段计算如颜色和深度之类的高斯属性。
- Sorting: 然后，每个 tile 根据高斯深度对其相交的高斯分布进行排序，以确保从最近到最远的正确渲染顺序。
- Rendering: Tile 中的每个像素遍历相同的高斯列表，并通过alpha混合将高斯颜色累加到自己身上。例如，我们显示B块需要混合两个高斯函数，一个蓝色的和一个红色的。

![The DRAM bandwidth requirement for 90 FPS under different scenes. The red dashed line highlights the bandwidth limit of Orin NX.](https://share.note.youdao.com/yws/api/personal/file/WEB05653c6fe64a3188379c5a5bcc8dca7b?method=download&shareKey=ef9d1adc7abb1f8482cc319abf67cee0 "The DRAM bandwidth requirement for 90 FPS under different scenes. The red dashed line highlights the bandwidth limit of Orin NX.")

对于现实场景，DRAM 带宽需求超过了最近发布的移动设备 Nvidia Orin NX 的带宽限制。这意味着单靠数据通信的要求已经无法实现实时性。我们进一步剖析了不同场景下的DRAM流量贡献。如图4所示，投影和排序在整个片外数据通信中占主导地位，共占DRAM总流量的90%. 其原因为：
- 在投影中，每个高斯函数需要将59个参数加载到片上缓冲区中，并将处理后的特征和交集元数据写回DRAM，占 DRAM 流量的 41%.
- 在排序中，渐进式高斯排序的重复读写操作导致重复的DRAM访问，占 DRAM 流量的 49% 以上.

# 3. Algorithm Design

原始 3DGS 流水线的主要缺点是它逐块渲染帧，每个块依次执行三个阶段。每个阶段的中间数据太大，片上无法容纳，因此在执行期间引入频繁的片外流量。

**Idea.** 为了消除中间阶段频繁的芯片外通信，我们的想法是将整个3D场景划分为小体素，并逐体素增量渲染整个场景。通过一次处理一个体素，我们的算法保证了阶段之间的中间数据可以完全在片上保存，因此不会发生片外流量。

![Fig. 5. The overview of our fully-streaming algorithm.](https://share.note.youdao.com/yws/api/personal/file/WEB7cb55392351270e66e1250ad1a1a77ab?method=download&shareKey=f5604651b5e2ad06ae45c6378fac5dda "Fig. 5. The overview of our fully-streaming algorithm.")

图5给出了我们算法的概述。我们的算法将一组像素呈现在一起，例如，R0到R3在同一组中。
1. 对于每一组，首先确定哪些体素与这些像素射线相交，并将结果存储在体素渲染表中。一旦确定了每个像素的相交体素，我们就建立它们的全局渲染顺序。
2. 对每个体素依次进行分层过滤、排序和渲染，得到该 tile 的部分像素值。

最终的像素值是通过累加所有体素的部分结果得到的。在应用我们的算法时存在三个挑战

## Inter-Voxel Order

单个组内的像素经常与多个体素相交，如图5所示。第一个挑战是确保所有相交体素的的渲染顺序是从最近的体素到最远的体素全局顺序。

为了在每个 tile 中建立正确的体素顺序，我们首先在图5的体素排序表中收集单个像素的渲染顺序。基于单独的渲染顺序，我们构建了一个 DAG，其中每个体素被表示为一个节点，边缘表示渲染依赖关系。然后，**在 DAG 上应用拓扑排序**，以保证图中没有循环，并且所有呈现依赖项都满足。在剩下的流程中使用排序后的顺序。

## Cross-Voxel Gaussian Order

![Fig. 6. An example of incorrect rendering order among Gaussians and the effect of boundary-aware fine-tuning.](https://share.note.youdao.com/yws/api/personal/file/WEBc10cd3517186a80afea1d9f1eee7877c?method=download&shareKey=39961f99058db0db13ed1465949a1a0e "Fig. 6. An example of incorrect rendering order among Gaussians and the effect of boundary-aware fine-tuning.")

在3DGS中，每个像素都需要从最近到最远来渲染高斯分布，但是正确的体素级排序并不总是保证在高斯级粒度下渲染顺序的正确性。例如，在图6的左侧，一个高斯可能伸出 voxel边界，部分在voxel1，部分在voxel2. voxel2整体远，但其中的某个高斯 (Gaussian 2) 实际比 voxel1 里的 Gaussian 1 更近。

我们观察到，只有当高斯分布跨越多个体素时，才会出现错误的顺序。为了缓解这个问题，我们引入了 *boundary-aware finetuning* 来调整跨体素高斯分布。保持每个高斯位置固定，以保留场景几何形状，同时微调其剩余如模，方向等的可训练参数。这样减少了高斯分布的大小和方向，最小化了它们跨越其他体素的可能性。在我们的微调中，我们提出了一个新的损失函数来惩罚跨边界高斯函数，

$$
\mathcal{L}=\mathcal{L}_{origin}+\beta\mathcal{L}_{CBP},\tag{1}
$$
- $\mathcal{L}_{origin}$: 原始 3DGS 的损失。
- $\mathcal{L}_{CBP}$: cross-boundary penalty loss 用于惩罚那些跨越体素边界的高斯。
- $\beta$: 用于调整的超参数。


$$
\begin{aligned}\mathcal{L}_{CBP}&=\frac{1}{N}\sum_{i=0}^{N-1}S_iT_i,\\\mathrm{where~}T_i&=\begin{cases}1,&\text{if depth}_i<\max_{j\in[0,i)}\{\text{depth}_j\},&\text{(2)}\\0,&\text{otherwise,}&\end{cases}\end{aligned} \tag{2}
$$

如果当前高斯 i 的深度小于之前最远深度 (即i比某个前面的更近，但按顺序后渲染) ，则$T_i=1$，触发惩罚。这捕捉“跨序”情况，尤其跨体素的。

## Redundant Gaussians in Voxels

每个体素包含近距离的高斯分布；然而，加载整个体素可能无意中在芯片上带来不相关的高斯分布。如果不小心忽略这些冗余的高斯分布，将会引入过多的DRAM流量。

在投影过程中，每个高斯分布需要59个参数。为了避免不相交的高斯，我们提出了一种两阶段分层滤波来逐步获取高斯参数，如图5所示。第一阶段用作轻量级但粗粒度的过滤器，它近似于高斯函数是否与图像块相交。在这个阶段，每个高斯函数只从片外存储器请求 4个参数 (3D坐标和最大尺度) 来计算相对于图像块的投影中心和最大投影半径。如果高斯函数没有通过粗粒度过滤器，我们跳过剩余的计算，这样避免加载剩余的55个参数。例如，图5中的高斯0不与黄色图像块相交，可以安全地将其从后续计算中删除。

在第二个细粒度滤波中，通过计算精确投影来识别与图像块相交的实际高斯分布。例如，在获得形状和方向后，我们可以知道图5中的高斯3实际上并不与图像tile相交，因此可以跳过后续的排序和渲染。为了进一步减少片外流量，我们提出了一种新的数据压缩方案，将剩余的参数编码成一个紧凑的码本。

![Fig. 8. The DRAM Data layout.](https://share.note.youdao.com/yws/api/personal/file/WEBade474fe5fc206630614aa1486e43c9c?method=download&shareKey=c49e775debc5fd15c84924ecf8f8e4e4 "Fig. 8. The DRAM Data layout.")

DRAM中，按voxel (Voxel 0, Voxel 1, ... Voxel N) 连续存储高斯数据。每高斯特征分成两半，分开存，用不同过滤阶段：
- Uncompressed Data for Coarse-Grained Filter: 仅4个参数——3D坐标 (x, y, z) 和最大尺度 (maximum scale, s) . 专供粗过滤。
DRAM 中直接存，专供粗过滤。
- Compressed Data for Fine-Grained Filter: 剩余55参数，包括旋转矩阵 (rotation matrices), 球谐系数 (SH coefficients), DC (直流分量，颜色基调) 等。用 vector quantization (VQ) 压缩成代码本索引。codebook 全部放在 SRAM (250 KB)，DRAM只存索引。

# Architectural Support (VSU)

![Fig. 9. Overview of STREAMINGGS accelerator, consisting of four components: a voxel sorting unit (VSU), a hierarchical filtering unit (HFU), a sorting unit, and a rendering unit.](https://share.note.youdao.com/yws/api/personal/file/WEBca4fb179342781a9d4d3caf1e6c8fc28?method=download&shareKey=e1abecd1039a3c1a8e5639e7cd093841 "Fig. 9. Overview of STREAMINGGS accelerator, consisting of four components: a voxel sorting unit (VSU), a hierarchical filtering unit (HFU), a sorting unit, and a rendering unit.")

我们的加速器架构由四个组件组成：体素排序单元 (VSU) 、分层过滤单元 (HFU) 、排序单元和渲染单元。给定一组射线方向，VSU首先识别并分类相交体素。然后，对于每个相交体素，HFU应用分层滤波器来识别相交的高斯分布。HFU获得过滤后的高斯分布后，排序和渲染单元分别从各自的缓冲区执行排序和渲染阶段。排序和渲染设计均采用GSCore. 我们通过采用 GSCore 的 bitonic 排序单元来简化排序单元，因为基于体素的渲染只需要在一个体素内建立高斯的渲染顺序。

## Voxel Sorting Unit

![Fig. 10. The design of voxel sorting unit (VSU).](https://share.note.youdao.com/yws/api/personal/file/WEB020cb69e5080339f9f778b6c085ba3e7?method=download&shareKey=ca7a899e29cef6fb9270a536502dd61a "Fig. 10. The design of voxel sorting unit (VSU).")

VSU的架构如图10所示。对于组内的像素，VSU沿着每个像素射线进行采样以识别相交的体素，并执行拓扑排序以建立这些体素的全局渲染顺序。
1. 对于tile内每个像素，沿射线采样多点，每个点坐标通过离线的voxel网格直接算VID (简单哈希).
2. 用预建表 (离线扫描场景) 过滤空VID (无高斯voxel). 然后重命名 VID → VIDr. 输出每个像素的renamed VIDr list.
3. 迭代所有像素的VIDr list，收集依赖，构建邻接表。每个entry有tag (源VIDr) + value (目标VIDr列表). 例如，对于像素P的list [A, B] (A近B远) ，加边 A → B (A源，B目标).
4. 用邻接表初始化 in-degree table：每个VIDr的in-degree = 其作为目标的边数。然后排序循环：
    - 找in-degree=0的VIDr (无依赖，可先渲染)，输出到Voxel Queue.
    - 对于每个输出VIDr，从邻接表取其目标，减目标in-degree (--1).
    - 重复直到in-degree table全 0 (无环保证).

## Hierarchical Filtering Unit (HFU)

一旦VSU生成了交叉体素的渲染顺序，HFU开始一次加载一个体素，并处理该体素内的高斯分布。

**Coarse-Grained Filter.** HFU只读取粗粒度滤波器中的每个高斯坐标和最大尺度，计算其投影位置和粗半径。这些结果被发送到相交测试单元，以确定高斯图像是否与当前图像块相交。如果相交，HFU向码本发出信号，对码本缓冲区中的剩余参数进行解码，如图9所示。解码后的参数，以及未压缩的参数，然后被送入FIFO进行细粒度过滤。

**Fine-Grained Filter.** 这一步计算精确的半径计算，并进行相交测试。同时，细粒度滤波器还根据解码后的尺度、旋转等参数计算RGB值和圆锥矩阵。基于相交结果，如果该高斯通过相交测试，HFU输出高斯RGB和圆锥结果到排序缓冲区。

与细粒度过滤器相比，粗粒度过滤器大大减少了计算量 (从427个mac减少到55个). 此外，粗粒度滤波器避免了HFU中对FIFO的解码和读取不必要的高斯信号。