---
title: "MixQ"
date: 2025-10-07T07:26:56+08:00
lastmod: 2025-10-07T07:26:56+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- Quantization

keywords:
- Quantization

description: "Paper reading of MixQ." # 文章描述，与搜索优化相关
summary: "Paper reading of MixQ." # 文章简单描述，会展示在主页
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

MixQ 是一个高效的混合精度量化系统。通过对离群分布的深入分析，我们引入了一种基于位置的离群预测算法，该算法可以预测95.8%的 token 的所有离群值。基于这种准确的预测，我们提出了一种量化超前检测 (Quantization Ahead of Detection, QAD) 技术来验证预测的正确性。

# 1. Introduction

在计算过程中，中间结果 (也称为激活) 被量化为4位或8位整数等低精度格式，使用高吞吐量低精度计算单元进行处理，并在层归一化和非线性激活层层等后续层中去量化为高精度格式。量化方法通常表示为 "WxAy"，其中x和y分别表示权重和激活的位宽。

一些非常大的值 (也称为异常值) 对模型精度有巨大的影响，但有时它们被基本量化方法截断为小值。为了缓解这种精度问题，引入了混合精度量化，它以高精度格式 (FP16) 存储异常值以保持模型精度，而以低精度格式存储其他非异常值以提高效率。这些技术表示为 "WxAyOz"，其中 Oz 表示Outliers的位宽。

![Fig. 1: Performance of quantization methods compared with non-quantized baseline (FP16). The kernel computes C = A × W with shape (A) = (batch size, 8192), shape (W ) = (8192, 28672). The activation A and weight W are taken from LLaMA-2-70B. Perplexity (PPL) evaluates the accuracy of the quantization methods, lower is better.](https://share.note.youdao.com/yws/api/personal/file/WEB1bd60f3348ca97ddcc1c48cce5031b4c?method=download&shareKey=db92253eac8bbb59d6a27afe41779b9a "Fig. 1: Performance of quantization methods compared with non-quantized baseline (FP16). The kernel computes C = A × W with shape (A) = (batch size, 8192), shape (W ) = (8192, 28672). The activation A and weight W are taken from LLaMA-2-70B. Perplexity (PPL) evaluates the accuracy of the quantization methods, lower is better.")

**异常值通常只占激活张量的不到 1%，但会显著降低计算速度**。图1 显示一个纯粹的 W8A8 量化 (不处理离群值) 性能高达 395 TFLOPs. 而加入了离群值处理的 W8A8O16 量化，尽管离群值仅占 0.97%，其性能却暴跌了 70%，降至 233 TFLOPs.

MixQ的创新基于对离群值分布的深入分析，其关键思想和技术包括：
- 基于局部性的离群值预测: 作者发现离群值的分布存在局部性规律，并基于此提出了一种预测算法，该算法能够成功预测95.8%的token中的所有离群值。
- 量化先行检测 (QAD): 基于精准的预测，作者设计了一种名为 Quantization Ahead of Detection 的新技术，它可以在量化过程中顺带验证预测的正确性，避免了传统方法中昂贵的检测开销。
- 高效的数据结构: 为了高效处理离群值，论文还提出了一种新的数据结构。

# 2. Background and Challenge

仅权重量化 (*Weight-only quantization*)，将权重量化为低精度格式，以减少内存使用并提高内存受限设置 (如小批量解码) 的吞吐量。然而，由于这些方法仍然执行高精度的计算，因此它们的吞吐量受到限制。

联合权重激活量化 (*Joint weight-activation quantization*)，进一步将激活量化为低精度格式，因此它们可以使用低精度的计算单元来实现更高的吞吐量。然而，这些方法在量化为低精度格式 (如INT8或INT4) 时面临精度问题。

为了同时实现高吞吐量和高精度，BitsandBytes 提出了混合精度量化 (*mixed-precision quantization*). 它将激活区分为正常值和异常值，对正常值进行高效的低精度矩阵乘法，对异常值进行高精度矩阵乘法。与不进行量化的W16A16计算相比，该方法可以保持99%以上的精度。

**Notation.** 每层的输入 $A_{FP16}$ 是一个 m × h 张量，其中 m 为输入 token 的个数，h 为隐藏层大小。我们识别激活 $A_{FP16}$ 的异常通道 (带有异常列) O，并将其余部分量化为低精度格式，如INT8或INT4.

离群通道 O 的计算公式为：

每层的输入 $A_{FP16}$ 是一个 m × h 张量，其中 m 为输入 tokrn 的个数，h 为隐藏层大小。进行量化的步骤为:
1. 离群值通道识别 (Outlier Identification): 首先，需要识别出哪些列 (channels) 包含了离群值。一个通道被定义为离群值通道的条件是，该列中存在任何一个元素的绝对值超过了预设的阈值 θ (通常设为6) . 所有离群值通道的集合 O 由 公式(1) 定义。

$$
\mathcal{O}=\{j\mid\max_i|(A_{FP16})_{ij}|>\theta,\quad j\in[0,h)\}\tag{1}
$$

2. 量化 (Quantization): 对于不包含离群值的列，可以安全地进行量化。量化过程如 公式(2) 所示，它将FP16数值通过一个动态生成的缩放因子 S 转换到INT8范围。

$$
(A_{INT8})_{ij}=round\left(\frac{(A_{FP16})_{ij}}{S_{i}}\right),j\notin\mathcal{O} \tag{2}
$$

这个缩放因子 S 是根据每行非离群值中的最大绝对值动态计算的，如 公式(3) 所示。

$$
S_i=\frac{\max_{j\notin\mathcal{O}}\left|(A_{FP16})_{ij}\right|}{2^{bit-1}-1} \tag{3}
$$


计算 (Computation): 最终的矩阵乘法被分解为两个部分的总和，如 公式(4) 所示：
- 低精度计算: 对非离群值通道中的量化激活值 AINT8和量化权重 WINT8进行整数矩阵乘法，然后乘以缩放因子。
- 高精度计算: 对离群值通道中的原始激活值 AFP16 和权重 WFP16 进行浮点矩阵乘法。

$$
\begin{aligned}C_{i,j}^{FP16}\approx&S_i^AS_j^W\cdot\sum_{k\notin\mathcal{O}}(A_{INT8})_{i,k}(W_{INT8})_{j,k}\\&+\sum_{k\in\mathcal{O}}(A_{FP16})_{i,k}(W_{FP16})_{j,k}\end{aligned} \tag{4}
$$

![Fig. 2: Workflow of mixed-precision quantization.](https://share.note.youdao.com/yws/api/personal/file/WEBcac15e4a4443bb6cf100a2c7ab56a541?method=download&shareKey=f930579d928e812f66c444503e332bd8 "Fig. 2: Workflow of mixed-precision quantization.")

混合精度量化的标准工作流：(1) 离群值检测 -> (2) 量化 -> (3) 计算 -> (4) 反量化 。性能问题主要源于以下两点：
1. 离群值的动态性导致高昂的检测开销 (Outlier dynamism leads to high detection overhead): 离群值的位置在不同输入、不同token之间是动态变化的，无法预先确定。因此，必须在运行时 (runtime) 进行检测。如 图3(a) 所示，这个检测过程非常繁重，需要在GPU上执行大量的 **原子操作 (atomic operations)** 来统计离群值的位置和数量，这会引发线程间的同步和竞争，远无法充分利用硬件性能。同时这样也会导致精度问题。

![Fig. 3: Different outlier detection methods.](https://share.note.youdao.com/yws/api/personal/file/WEB65b8bd051f91088222142cdd4e9a451a?method=download&shareKey=0684597d45b6b0faa827b626fc1aac93 "Fig. 3: Different outlier detection methods.")

2. 离群值的稀疏性导致计算效率低下 (Outlier sparsity makes computation inefficient): 
    - 内存访问低效: 由于离群值在整个激活张量中是稀疏分布的，它们在内存中的存储位置是不连续的 。GPU架构为了效率，总是以一个固定大小的块 (如32字节) 来读取内存。如 图4 所示，当GPU为了读取一个2字节的FP16离群值而执行一次32字节的内存事务时，其中有30字节的数据被浪费掉了。

    - 计算核心低效: 当前最先进的方法 (如BitsandBytes) 在处理离群值的高精度计算时，因为数据是稀疏的，只能采用 SpMM 的通用计算方式。这种计算只能利用GPU上相对较慢的通用 CUDA Cores，而无法利用 Tensor Cores.

![Fig. 4: Outlier sparsity makes the memory access inefficient.](https://share.note.youdao.com/yws/api/personal/file/WEBf974f9d09572f7337a761238713ecc35?method=download&shareKey=bd3dc77a3f118f691cfe688fe652c3ef "Fig. 4: Outlier sparsity makes the memory access inefficient.")

# 3. Motivation: Predictable Outliers

![Fig. 6: Outlier channels of the activation matrix of the 31st layer of LLaMA-7B model. The input data comes from WikiText2. The orange columns denote the activation which contains new-come outliers.](https://share.note.youdao.com/yws/api/personal/file/WEBfb2ac56334e5ad49d0dfd07cb42748ef?method=download&shareKey=d733c6354b55f982d67aa5f151acd8f4 "Fig. 6: Outlier channels of the activation matrix of the 31st layer of LLaMA-7B model. The input data comes from WikiText2. The orange columns denote the activation which contains new-come outliers.")

作者通过分析发现离群值具有 **句子级的局部性 (sentence-level locality)**. 在解码一个句子的过程中，离群值倾向于集中在少数固定的通道中。因此可以大胆地预测，其离群值通道的集合就是所有先前token中出现过的离群值通道的并集。只要没有新的离群值通道出现，这个预测就是100%准确的。

![Fig. 7: Percentage of outlier channels in different layers. The LLaMA-7B model contains four types of projection layers: the QKV, the Dense, the Up-gate, and the Down projection layers.](https://share.note.youdao.com/yws/api/personal/file/WEBac0cff4d9df482603d58a2186a920066?method=download&shareKey=fd6a7809ce201712750032489f167e01 "Fig. 7: Percentage of outlier channels in different layers. The LLaMA-7B model contains four types of projection layers: the QKV, the Dense, the Up-gate, and the Down projection layers.")

图7 进一步从量化的角度证实了离群值的两个关键特征: 
1. 稀疏性: 在LLaMA-7B模型的大多数投影层 (如QKV, Dense, Up-gate projections) 中，离群值通道的占比都非常低，低于1% 。即使是在离群值相对较多的Down projection层，其占比也只在解码后期达到4%左右，大部分时候远低于此。
2. 局部性: 离群值通道的百分比在解码最初的少数几个token时快速增长。

作者将这种基于局部性的预测方法公式化。对于第 l 层网络：
- 初始状态 (解码第0个token时) ，预测的离群值通道集合 $\mathcal{O}_l^0$ 为空集。
- 当解码第 n 个token时，其真实的离群值通道集合为 $o_l^0$
 那么，用于预测下一个 (第 n+1 个) token的离群值通道集合，就是历史集合与当前集合的并集：$\mathcal{O}_l^{n+1}=\mathcal{O}_l^n\cup o_l^n$.

# 4. Design of MixQ

![Fig. 8: Workflow of MixQ. The gray and blue paths show the workflow when outlier prediction fails and succeeds, respectively.](https://share.note.youdao.com/yws/api/personal/file/WEB54e793c79a413afb7a4a9ee8b1aa4e78?method=download&shareKey=3ab9e468fa9f777e5b354b70b9c8526c "Fig. 8: Workflow of MixQ. The gray and blue paths show the workflow when outlier prediction fails and succeeds, respectively.")

对于一个输入的激活张量，MixQ 直接采用预测的离群值通道集合 $\mathcal{O}_l$，并基于这个预测来执行量化操作，生成量化后的激活值 $A_{INT8}$ 和缩放因子 S. 然后MixQ会使用一个极其轻量级的验证算法 QAD 来检查预测是否完全正确。成功则直接进行混合精度乘法，否则就重新检测然后量化计算。

![Fig. 9: Comparison of the workflow between the classical mixed-precision method and MixQ.](https://share.note.youdao.com/yws/api/personal/file/WEB4a499f26ffdec1dcca48c61488222877?method=download&shareKey=f1b709ab54cf2c4e25dd9600023c35aa "Fig. 9: Comparison of the workflow between the classical mixed-precision method and MixQ.")

QAD算法包含三个步骤:
1. Quantization: MixQ首先使用预测的离群值通道集合 $\mathcal{O}_l$ 来计算缩放因子 S 和量化激活矩阵。
2. Verification: 验证所有不在预测集合 $\mathcal{O}_l$ 中的值是否都不是离群值。
    - 数学上，这等价于检查 $max(|(A_{FP16})_{i,j}|)<\theta,\quad\forall j\notin\mathbf{O},i$. 
    - 这个检查可以通过量化步骤中已经计算出的缩放因子 $S^A$ 来完成。上述条件可以转化为: $|S_i^A|\times(2^{bit-1}-1)<\theta,\quad\forall i$. 
    - 最终，验证过程被简化为一个极其轻量级的操作: $\max_i(|S_i^A|)<\frac\theta{(2^{\boldsymbol{b}it-1}-1)}$.
3. Re-quantization: 如果验证失败，MixQ需要检测出新的离群值通道并重新量化。但之前的预测和量化结果仍然能加速这个过程。如 图10 所示，因为预测的离群值通道在第一遍量化时已经被处理，系统在检测新离群值时，只需要扫描剩余的、未被预测的通道。

![Fig. 10: Detect new outliers based on quantization result.](https://share.note.youdao.com/yws/api/personal/file/WEB6156fca3c4c1e5d7fa84dffa5386e5f2?method=download&shareKey=d90577d514d6935d587adadda5908a61 "Fig. 10: Detect new outliers based on quantization result.")

为了解决离群值稀疏分布导致的内存访问和计算低效问题，MixQ提出了一种新的数据结构。图 11 对比了现有工作和 MixQ 的区别。

![Fig. 11: Different approaches of organizing outliers. Constructing an order-permuted structure requires global memory movement while constructing an order-reserved structure is more lightly.](https://share.note.youdao.com/yws/api/personal/file/WEBf88855f3aa3be33b1fc172fcda06028b?method=download&shareKey=bbe7ded980d31e4efb80f10033fb7655 "Fig. 11: Different approaches of organizing outliers. Constructing an order-permuted structure requires global memory movement while constructing an order-reserved structure is more lightly.")

现有工作是保序重排数据结构 (order-permuted data structure): 通过复杂的全局内存移动，将离群值和非离群值分别移动到不同的连续内存区域。这个过程需要移动大量的非离群值数据开销很大。

MixQ：保序预留数据结构 (order-reserved data structure): 保持原始张量的顺序和布局不变，会额外开辟一小块预留空间，将预测的离群值复制到这个空间中，形成一个紧凑连续的数组。同时将原始张量中离群值的位置清零。

缓存权重切片 (Cached weight slice): 当验证失败时，MixQ会从权重矩阵中提取出对应离群值通道的列，并将这部分权重缓存起来。在后续的解码步骤中，只要离群值集合保持不变，系统就可以直接使用缓存好的权重切片。