---
title: "AwesomeQuantizationMethod"
date: 2025-10-13T11:32:59+08:00
lastmod: 2025-10-13T11:32:59+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- Quantization

keywords:
- Quantization

description: "GPTQ, AWQ, SmmothQuant & QuaRot." # 文章描述，与搜索优化相关
summary: "" # 文章简单描述，会展示在主页
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

# Quantization

量化是指在推理性能损失最小的情况下，减少模型参数中的比特数 (即精度) 的过程。对模型进行量化的好处有
1. 减小模型体积。
2. 提升推理速度。
    - 在模型推理时，需要不断地从内存中读取模型权重和中间激活值。量化后就能在相同时间内读取更多数据。
    - 处理器执行整数运算的速度远快于浮点运算，如果硬件有专门的低精度计算单元可以显著增加运算速度。

## Uniform Quantization

![Comparison between uniform quantization (left) and non-uniform quantization (right).](https://share.note.youdao.com/yws/api/personal/file/WEB3abff78b989e5c64417f06cbc4007a52?method=download&shareKey=e80a0ce0e7d2d101b7a71181b4636d09 "Comparison between uniform quantization (left) and non-uniform quantization (right).")

均匀量化 (Uniform Quantization) 是一种将浮点实数值映射到一组有限的、均匀间隔的数值 (量化级别) 的方法。其量化算子 Q 定义如下 : 

$$
Q(r)=\text{Int}\big(r/S\big)-Z
$$

- r 是实数输入值 (权重或激活)
- Z 是整数零点
- Int 函数将一个实数通过舍入操作 (四舍五入或截断) 映射成一个整数

可以通过反量化 (Dequantization) 操作从量化值 $Q(r)$ 中恢复实数值 $\tilde{r}$，但由于舍入操作，恢复的值与原始值不会完全相等。

$$
\tilde{r} = S(Q(r) + Z)
$$

## Symmetric and Asymmetric Quantization

![Illustration of symmetric quantization and asymmetric quantization. Symmetric quantization with restricted range maps real values to [-127, 127], and full range maps to [-128, 127] for 8-bit quantization.](https://share.note.youdao.com/yws/api/personal/file/WEB4f16239911a203b40406a788196a2c07?method=download&shareKey=c62b1d49fece8811b4ae8e0c21362fcc "Illustration of symmetric quantization and asymmetric quantization. Symmetric quantization with restricted range maps real values to [-127, 127], and full range maps to [-128, 127] for 8-bit quantization.")

这两种方法的区别在于如何选择缩放因子 (scaling factor) S 和剪裁范围 (clipping range) $[\alpha, \beta]$. 缩放因子 S 的计算公式为: 

$$
S = \frac{\beta - \alpha}{2^b - 1}
$$

其中 $b$ 是量化位宽。

选择剪裁范围的过程叫做校准 (Calibration).
- 非对称量化 (Asymmetric Quantization): 直接使用信号的最小值和最大值作为剪裁范围，即 $\alpha = r_{min}$ 和 $\beta = r_{max}$. 
    - 由于剪裁范围不一定关于原点对称 (即 $-\alpha \neq \beta$) ，因此被称为非对称量化。
    - 当权重或激活值的分布不均衡时 (例如，经过ReLU激活函数后总是非负) ，这种方法能提供更紧密的剪裁范围，从而更有效。
- 对称量化 (Symmetric Quantization): 选择一个关于原点对称的剪裁范围，即 $\alpha = -\beta$.
    - 通常选择 $-\alpha = \beta = max(|r_{max}|, |r_{min}|)$. 这种方法可以简化量化函数，因为零点 $Z$ 变为 0. 
    - 由于零点为 0 可以减少推理过程中的计算成本，对称量化在实践中被广泛用于权重量化 。
    - 对于对称量化，存在全范围 (full range) 和限制范围 (restricted range) 两种模式，其中全范围模式 $S=floor(\frac{2max(|r|)}{2^n-1})$ 利用了全部的整数范围 (INT8 情况下为 [-128, 127]). 而限制范围模式选择 $S=floor(\frac{max(|r|)}{2^{n-1}-1})$，INT8 情况下只用到了 [-127, 127].

利用信号的最小/最大值进行对称和非对称量化是一种常用的方法。然而，这种方法容易受到激活中的异常数据的影响。可能会降低量化的分辨率。解决这个问题的一种方法是使用百分位数而不是信号的最小/最大值。也就是说，不是使用最大/最小值，而是使用第 i 个最大/最小值作为 β/α. 另一种方法是选择 α 和 β，以最小化实值和量化值之间的KL散度 (即信息损失) .

## Range Calibration Algorithms: Static vs Dynamic Quantization

确定剪裁范围的时间点对于激活值的量化尤为重要，因为激活值会随着每个输入样本的不同而变化。

- 动态量化 (Dynamic Quantization): 在运行时为每个激活图动态计算其剪裁范围。由于是为每个输入精确计算范围，通常能获得更高的准确性。缺点是需要实时计算信号统计数据 (如 min/max)，这会带来非常高的计算开销。
- 静态量化 (Static Quantization): 剪裁范围在推理前预先计算好，并在整个推理过程中保持不变。这种方法不会增加额外的计算开销。通常的做法是使用一系列校准输入来计算激活值的典型范围。预计算范围的常用指标是最小化原始张量和量化后张量之间的均方误差 (MSE).

## Quantization Granularity

![Definition of per-tensor, per-token, and perchannel quantization.](https://share.note.youdao.com/yws/api/personal/file/WEBc40c87c6b19a06a37246a87116dcdde7?method=download&shareKey=2cec82518cc52c2eb8ae5657f13b7e67 "Definition of per-tensor, per-token, and perchannel quantization.")

量化粒度指的是为权重计算剪裁范围 $[\alpha, \beta]$ 的精细程度。主要分为以下几种: 
- **per-tensor**: 对一个权重使用同一个剪裁范围。这种方法实现简单，但如果权重数值范围差异很大，则可能导致次优的准确性。
- **per-channel**: 为每个通道独立计算一个剪裁范围和缩放因子。可以确保更好的量化分辨率，并通常带来更高的准确性，而且开销可以忽略不计。但需考虑硬件不友好的问题。
- **group-wise**: 将一个张量的多个通道分组，为每个组计算剪裁范围。在量化分辨率和计算开销之间提供了一个很好的折衷。
- **per-token**: 为了有效地利用 INT8 GEMM 核的矢量量化，我们只能使用外部维度 (即 token 维度 T 和输出通道维度 $C_o$) 的缩放因子，而不能使用输入通道维度 $C_i$. 

## Non-Uniform Quantization

与量化级别均匀分布的均匀量化不同，非均匀量化的量化级别和量化步长 (阈值) 都可以是不均匀的。

定义: 如果一个实数值 $r$ 落在量化步长 $[\Delta_i, \Delta_{i+1})$ 之间，量化器 $Q$ 会将其映射到对应的量化级别 $X_i$ 4.

通过将更多的量化级别分配给重要的数值区域，非均匀量化可以更好地捕捉数据分布，从而在固定的位宽下实现更高的准确性。这对于权重和激活值常见的钟形分布特别有效。

常用方法有: 
- 基于规则: 例如使用对数分布，其中量化步长和级别呈指数增长。
- 基于优化: 将非均匀量化表述为一个优化问题，通过最小化原始张量与量化后张量之间的差异来调整量化步长/级别。
- 基于聚类: 使用 k-means 等聚类算法来确定量化步长和级别。

尽管有潜在的准确性优势，但非均匀量化方案通常难以在通用计算硬件 (如 GPU 和 CPU) 上高效部署。因此，均匀量化因其简单性和硬件映射效率而成为实际的标准方法。

## Fine-tuning Methods

![Comparison between Quantization-Aware Training (QAT, Left) and Post-Training Quantization (PTQ, Right).](https://share.note.youdao.com/yws/api/personal/file/WEB1da377ad24e39be92b15f07bdf3bb226?method=download&shareKey=d14ed5cad447c3b2602e0d7a15059b76 "Comparison between Quantization-Aware Training (QAT, Left) and Post-Training Quantization (PTQ, Right).")

量化后通常需要调整神经网络的参数以恢复准确性。主要有两种方法: 量化感知训练 (QAT) 和训练后量化 (PTQ).

- 量化感知训练 (Quantization-Aware Training, QAT): 在再训练过程中模拟量化效应。在前向和后向传播过程中，模型参数以浮点形式处理，但在每次梯度更新后都会被量化。于量化算子是不可微的，通常使用**直通估计器 (Straight Through Estimator, STE)** 来近似其梯度。STE 在反向传播时忽略舍入操作，并将其近似为一个恒等函数。QAT 的主要缺点是重新训练模型所需的巨大计算成本。

![Illustration of Quantization-Aware Training procedure.](https://share.note.youdao.com/yws/api/personal/file/WEB43a139b71ab45993c6009d363d858437?method=download&shareKey=06c620a4aed1c19b69b3fa0b988813f8 "Illustration of Quantization-Aware Training procedure.")

- 训练后量化 (Post-Training Quantization, PTQ): 在没有任何微调或再训练的情况下对权重进行量化和调整。PTQ 的开销非常低，并且在训练数据受限或无标签的情况下也能应用。与 QAT 相比，PTQ 通常会导致准确性下降，尤其是在低精度量化时。为了缓解准确性下降，研究者提出了多种方法，如偏差校正、均衡不同层或通道间的权重范围等。

## Simulated and Integer-only Quantization

![Comparison between full-precision inference (Left), inference with simulated quantization (Middle), and inference with integer-only quantization (Right).](https://share.note.youdao.com/yws/api/personal/file/WEB33b9d1d62c2141b229033bf4f439f610?method=download&shareKey=7c8682f5973c54a8bf98a2c7414c2b8d "Comparison between full-precision inference (Left), inference with simulated quantization (Middle), and inference with integer-only quantization (Right).")

模拟量化与纯整数量化区分了两种部署量化神经网络模型的常见方法: 
- 模拟量化 (Simulated Quantization): 也称为伪量化 (fake quantization). 模型参数虽然以低精度形式存储，但在执行计算 (如矩阵乘法) 之前，会先将它们反量化回浮点数。这意味着实际的运算仍然是使用浮点算术完成的。因此，这种方法无法完全利用低精度整数逻辑带来的速度和能效优势。不过，对于瓶颈在于内存带宽而非计算的任务 (例如推荐系统)，模拟量化可能是有益的，因为其主要优势在于减少内存占用和加载参数的成本。
- 纯整数量化 (Integer-only Quantization): 也称为定点量化 (fixed-point quantization)，所有运算都使用低精度整数算术执行 。整个推理过程可以在高效的整数算术下完成，无需任何浮点反量化操作。这种方法更为理想，因为它能够充分利用低精度逻辑在延迟、功耗和芯片面积上的优势。例如，在 45nm 工艺下，一个 INT8 加法比 FP32 加法在能效上高30倍，面积效率上高 116 倍。
- Dyadic Quantization 是纯整数化的一种，它的所有缩放因子都是2的幂，从而可以用高效的位移操作替代整数除法。

![(Left) Comparison between peak throughput for different bit-precision logic on Titan RTX and A100 GPU. (Right) Comparison of the corresponding energy cost and relative area cost for different precision for 45nm technology.](https://share.note.youdao.com/yws/api/personal/file/WEB01d90cbde3e398ebb0c2beccf89c4132?method=download&shareKey=b7b9f5118198008fc1e8f46df4c09cfchttps://share.note.youdao.com/yws/api/personal/file/WEB01d90cbde3e398ebb0c2beccf89c4132?method=download&shareKey=b7b9f5118198008fc1e8f46df4c09cfc "(Left) Comparison between peak throughput for different bit-precision logic on Titan RTX and A100 GPU. (Right) Comparison of the corresponding energy cost and relative area cost for different precision for 45nm technology.")

# GPTQ (2023 ICLR) W3/4A16

GPTQ 是一种**一次性的训练后量化 (One-shot Post-Training Quantization, PTQ)** 方法。所谓一次性，是指它不需要对模型进行成本极高的重新训练或微调，仅需使用少量校准数据，在几个小时内就能完成整个压缩过程。

## OBQ

GPTQ 的理论基础是早前的一个名为最优大脑量化 (Optimal Brain Quantization, OBQ) 的方法。OBQ 是一种非常精细的逐层量化方法，其目标是最小化量化前后每一层输出的误差。具体来说，它在一个权重矩阵中，不是一次性把所有权重都量化掉，而是一次只量化一个权重。为了决定先量化哪个权重，它会贪心地选择那个引入误差最小的权重。更关键的是，在量化完这个权重后，OBQ 会利用二阶信息 (Hessian矩阵) 来更新所有其他尚未被量化的权重，以补偿刚刚引入的误差。

传统的量化方法，如简单的四舍五入 (RTN)，对待所有权重都一视同仁，独立地将它们量化到最近的离散值上。这种方法的缺点是显而易见的: 它完全忽略了权重之间的相互作用。一个微小的权重变动，可能会因为网络层的复杂运算而被放大，从而导致巨大的精度损失。

OBQ 则完全不同，它认为: 
1. 量化是有先后顺序的: 不应该一次性量化所有权重，而应该一个一个来。
2. 误差是可以补偿的: 每当一个权重被量化后，必然会产生一个误差。这个误差不应该被忽略，而应该通过微调其他尚未被量化的权重来进行补偿。

OBQ 采用**逐层 (Layer-by-Layer)** 的方式进行量化。对于网络中的某一个权重矩阵 $W$，OBQ 的目标是找到一个量化后的矩阵 $\hat{W}$，使得该层的输出误差最小化，即最小化 $||WX-\hat{W}X||_2^2$.

1. **初始化**: 在开始时，该层的所有权重都是全精度 (未量化) 的。
2. **计算二阶信息 (Hessian)**: OBQ 利用了二阶信息，它会根据一小部分校准数据 (输入为 $X$) 计算出该层权重的 Hessian 矩阵的逆矩阵 $H^{-1}$. 这个 Hessian 矩阵 $H_F$ 大致可以表示为 $2X_fX_f^T$. F 指的是全精度 (Full-precision).
3. 贪心选择下一个量化的权重: OBQ 会遍历所有尚未被量化的权重，并计算如果对其中某一个权重 $w_q$ 进行量化，会产生多大的误差。它选择的不是误差绝对值最小的权重，而是经过其他权重补偿后，最终带来的净误差最小的那个权重。这个选择标准由以下公式决定: 

$$
w_q=\arg\min_{w_q}\frac{(\text{quant}(w_q)-w_q)^2}{[H_F^{-1}]_{qq}}
$$

- $\text{quant}(w_q)-w_q$ quant 为四舍五入量化，该表达式意为化单个权重产生的直接误差。
- $[H_F^{-1}]_{qq}$ 是 Hessian 逆矩阵对角线上的元素，它代表了权重 $w_q$ 的敏感度或重要性。分母越大，说明这个权重越敏感，量化它带来的影响越大。
- 整个公式的含义是，寻找一个**自身量化误差**相对于其**重要性**比值最小的权重进行量化。
4. **量化并更新补偿**: 选定了权重 $w_q$ 后，就将其量化为最接近的离散值 $\text{quant}(w_q)$. 然后，OBQ 利用 Hessian 逆矩阵计算出一个精确的更新量 $\delta_F$ 并将其应用到所有其他尚未被量化的权重上，以补偿 $w_q$ 被量化所带来的误差。更新公式为: 

$$
\delta_F=-\frac{w_q-\mathrm{quant}(w_q)}{[H_F^{-1}]_{qq}}\cdot(H_F^{-1})_{:q}
$$
这里的 $(H_F^{-1})_{:q}$ 是 Hessian 逆矩阵的第 q 列，它描述了如何调整其他权重以最小化因为 $w_q$ 带来的整体误差。

5. **更新 Hessian**: 在量化完一个权重并更新了其他权重后，这个权重就相当于被“固定”住了。因此，需要将它从待优化的集合中移除，并更新 Hessian 逆矩阵，为下一步的选择做准备。

$$
\mathbf{H}_{-q}^{-1}=\left(\mathbf{H}^{-1}-\frac{1}{[\mathbf{H}^{-1}]_{qq}}\mathbf{H}_{:q}^{-1}\mathbf{H}_{q:}^{-1}\right)_{-p}
$$

可以向量化的同时更新 W 的多行。

6. 重复 3-5 直到该层所有权重全都被量化完毕。

OBQ 的优点是由于其精密的贪心选择和误差补偿机制，OBQ 在后训练量化方法中能达到顶尖的精度水平，尤其是在较低比特 (如3位、4位) 的量化上，显著优于简单方法。

OBQ 的主要瓶颈在于其巨大的计算复杂度。对于一个 $d_{row}\times d_{col}$ 的权重矩阵，其算法复杂度为 $O(d_{row}\cdot d_{col}^3)$. 因为矩阵中的每个权重都需要计算一次选择标准，并且每次量化后都需要进行 Hessian 逆矩阵更新。

## The GPTQ Algorithm

![GPTQ Quantization Procedure.](https://share.note.youdao.com/yws/api/personal/file/WEB32b1f7fb1260da0fe6fdc7faf3f95591?method=download&shareKey=525e62223790584164a6629ddfc5b4ad "GPTQ Quantization Procedure.")

以下是GPTQ三个优化的详细介绍。

---
**Arbitrary Order Insight**: GPTQ 放弃了复杂的贪心选择，改为对权重矩阵 W 中的每一行都采用完全相同的顺序 (即按列顺序) 进行量化。这样在处理第 j 列时，所有行中未被量化的权重集合 都是相同的。由于 Hessian 矩阵仅依赖于输入数据 X 而非权重本身，这意味着 Hessian逆矩阵的更新计算只需要在处理每一列时执行一次，而不是像 OBQ 那样需要为矩阵中的每一个权重都执行一次。算法的整体复杂度从 $O(d_{row}\cdot d_{col}^3)$ 降低为 $O(\max\{d_{row}\cdot d_{col}^2,d_{col}^3\})$.

---
**Lazy Batch-Updates**: 因为每量化一列，就需要更新剩余的所有权重和整个Hessian逆矩阵，计算访存比很低。**在量化第 i 列时，做出的舍入决策只受到对第 i 列本身更新的影响**，而此时对后面 (i+1, i+2, ...) 列的更新是无关紧要的。因此 GPTQ 采用以下更新。

1. 分块处理: GPTQ将列分成一个个 block，例如一次只处理 B=128 列。
2. 局部更新与全局更新分离: 在一个 block 内部进行量化时，所有的权重更新和 Hessian 更新都仅限于 block 内部。
3. 延迟批量应用: 只有当整个 block (例如128列) 全部处理完毕后，才将累积的更新效果一次性地应用到矩阵中所有剩余的权重和整个Hessian逆矩阵上。

---
**Cholesky Reformulation**: 反复地、近似地计算逆矩阵会累积浮点运算误差，导致Hessian逆矩阵 H 失去其数学特性 (例如，不再是正定矩阵). 一旦发生这种情况，算法可能会计算出错误的更新方向。**算法在量化权重 q 时，实际上并不需要完整的Hessian逆矩阵，而仅仅需要该矩阵的第 q 行信息**。因此，GPTQ不再迭代地更新逆矩阵，而是在算法开始时，就利用高效且稳定的 Cholesky 核函数，一次性地预计算出后续所有步骤所需要的Hessian 信息。

> we observed that the probability of this happening **increases with model size**: concretely, it almost certainly occurs for at least a few layers on models that are larger than a few billion parameters.

![GPTQ Algorithm.](https://share.note.youdao.com/yws/api/personal/file/WEB9a6f0cee8e891703232b22fd34c9f11e?method=download&shareKey=66f136b72e3e0d9114f72c68a43e0fdb "GPTQ Algorithm.")

- 第 3 行左侧的 $H^{-1}$ 指的是 $H^{-1}$ 进行 Choles可以 分解后得到的 $L^T$.

{{< details title="Cholesky Decomposition" >}}

Cholesky分解是一种将一个对称正定矩阵 (Symmetric Positive-Definite Matrix) 分解为一个下三角矩阵与它的共轭转置乘积的方法。如果矩阵是实数矩阵 (在机器学习中通常是这种情况)，那么分解形式就是该下三角矩阵与它的转置的乘积。

- 对称矩阵: 矩阵 A 等于其转置 $A^T$.
- 正定矩阵: 对于任何非零向量 x，都有 $x^T
 Ax>0$. 像 Hessian 矩阵这样的二阶信息矩阵，在经过适当处理 (例如GPTQ中提到的dampening，即对角线加一个小的正数) 后，通常能满足这个条件。

对于一个对称正定矩阵 A，Cholesky分解的目标是找到一个唯一的下三角矩阵 L (对角线元素为正)，使得: 

$$
A=LL^T
$$

其中 $L^T$ 是 L 的转置 (一个上三角矩阵).

举个例子，一个3x3的矩阵分解形式如下: 

$$
\begin{pmatrix}A_{11}&A_{12}&A_{13}\\A_{21}&A_{22}&A_{23}\\A_{31}&A_{32}&A_{33}\end{pmatrix}=\begin{pmatrix}L_{11}&0&0\\L_{21}&L_{22}&0\\L_{31}&L_{32}&L_{33}\end{pmatrix}\begin{pmatrix}L_{11}&L_{21}&L_{31}\\0&L_{22}&L_{32}\\0&0&L_{33}\end{pmatrix}
$$

我们可以通过逐个元素求解的方式来计算矩阵 L 的所有元素。计算顺序通常是从左到右，从上到下，即先计算第一列，再计算第二列，以此类推。假设我们正在计算 L 的第 j 列，而已知前 j−1 列。

1. 对角线元素计算

对于第 j 列的对角线元素 $L_{jj}$，我们考察 $A=LL^T$ 中 $A_{jj}$ 的来源: 

$$
A_{jj}=\sum_{k=1}^nL_{jk}(L^T)_{kj}=\sum_{k=1}^nL_{jk}L_{jk}=\sum_{k=1}^jL_{jk}^2=\left(\sum_{k=1}^{j-1}L_{jk}^2\right)+L_{jj}^2
$$

移项后，我们可以得到 $L_{jj}$ 的计算公式: 

$$
L_{jj}=\sqrt{A_{jj}-\sum_{k=1}^{j-1}L_{jk}^2}
$$

这个公式表明，$L_{jj}$ 的值依赖于 $A_{jj}$ 以及 L 矩阵第 j 行中前面已经计算出的元素。这也是为什么 A 必须是正定的，以确保开方根号内的值始终为正。

2. 非对角线元素计算

对于第 j 列的对角线下方元素 $L_{ij}$ (其中 i > j)，我们考察 $A=LL^T$ 中 $A_{ij}$ 的来源: 

$$
A_{ij}=\sum_{k=1}^nL_{ik}(L^T)_{kj}=\sum_{k=1}^nL_{ik}L_{jk}=\sum_{k=1}^jL_{ik}L_{jk}=\left(\sum_{k=1}^{j-1}L_{ik}L_{jk}\right)+L_{ij}L_{jj}
$$

移项后，我们可以得到 $L_{ij}$ 的计算公式: 

$$
L_{ij}=\frac1{L_{jj}}\left(A_{ij}-\sum_{k=1}^{j-1}L_{ik}L_{jk}\right)
$$

这个公式表明，$L_{ij}$ 的值依赖于 $A_{ij}$ 、刚刚计算出的对角线元素 $L_{jj}$，以及 L 矩阵中已经计算出的相关元素。

{{< /details >}}

## Experimental Validation

作者只用一张 A100 80G 就完成了所有 BLOOM 和 OPT 系列所有模型的量化。

| 实验类别 | 实验设置 (Setup) | 对比基线 (Baselines) | 核心发现与关键数据 (Key Findings & Data) |
| :--- | :--- | :--- | :--- |
| **与SOTA精度对比 (小模型)** | - **模型:** ResNet18/50, BERT-base, OPT-125M  <br>- **目的:** 验证GPTQ在小模型上与其它高精度但昂贵的PTQ方法的竞争力 | - AdaRound  <br>- BRECQ  <br>- OBQ  <br>- AdaQuant  | - **精度相当:** 在4-bit量化上，GPTQ与SOTA方法 (如OBQ, BRECQ)表现持平 ；在3-bit上略逊于最精确的方法<br>- **速度优势巨大:** GPTQ量化小模型耗时**少于1分钟**，而其他SOTA方法需要约**1小时** |
| **量化时间 (Runtime)** | - **模型:** 完整的OPT和BLOOM模型家族，最大至1760亿参数<br>- **硬件:** 单张NVIDIA A100 GPU | - ZeroQuant-LKD (作为参考)  | - **扩展性良好:** GPTQ量化10-30亿参数的模型只需几分钟，量化1750亿参数模型只需几小时<br>- **具体数据:** 量化BLOOM-176B耗时**3.8小时**；量化OPT-175B耗时**4.2小时** |
| **语言生成任务 (Perplexity)** | - **模型:** 完整的OPT和BLOOM模型家族<br>- **数据集:** WikiText2, PTB, C4 | - FP16 (原始精度)  <br>- RTN (四舍五入)  | - **显著优于RTN:** GPTQ在3-bit和4-bit上都远超RTN ，尤其是在3-bit时，RTN的性能会彻底崩溃，而GPTQ依然稳健<br>- **关键数据 (OPT-175B, WikiText2):** FP16困惑度为**8.34**，4-bit GPTQ为**8.37** (几乎无损)，而4-bit RTN劣化至**10.54** |
| **实际应用价值 (大模型)** | - **模型:** OPT-175B<br>- **任务:** 推理内存占用和生成任务的端到端延迟<br>- **硬件:** NVIDIA A100和A6000 GPU | - FP16多GPU部署  <br>- LLM.int8()  | - **GPU需求大幅降低:** 3-bit量化后的OPT-175B模型可装入**单张A100-80GB GPU**运行 ，而FP16需要**5张** ，LLM.int8()需要**3张**<br>- **端到端显著加速:** 通过定制GPU算子，推理速度获得巨大提升在A100上加速约**3.25倍**，在成本更低的A6000上加速约**4.5倍** |
| **零样本任务 (Zero-Shot)** | - **模型:** 完整的OPT和BLOOM模型家族<br>- **数据集:** LAMBADA, PIQA, ARC等 | - FP16  <br>- RTN  | - **结论一致:** 结果模式与困惑度任务相似在4-bit时，RTN表现尚可，但在3-bit时性能崩溃，而GPTQ在所有情况下仍能保持良好的准确率 |
| **极限量化 (Extreme Quantization)** | - **模型:** OPT-175B, BLOOM-176B<br>- **方法:** 结合分组 (Grouping)技术 | - FP16  <br>- 3-bit GPTQ  | - **2-bit量化可行:** 结合分组技术，GPTQ可以在2-bit量化下取得合理的性能 ，在约2.6 bit (group-size 32)时，困惑度仅比FP16高0.6-0.7个点<br>- **三元量化:** 甚至可以实现三元 (-1, 0, +1)量化，在OPT-175B上困惑度仅下降不到1个点 |

## Source Code

[GPTQ 源代码](https://github.com/ist-daslab/gptq)中初始化会获取权重 row 和 columns，并初始化 Hessian 矩阵 H.
```python
class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        # ... for conv2d & conv1d
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        # Init Hessian with 0.
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
```

`add_batch()` 喂入一批校准数据，用于计算和累积 Hessian 矩阵 H. 

```python
def add_batch(self, inp, out):
        # ...
        # inp: [B, N, C]
        tmp = inp.shape[0]
        inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()  # [C, BN]
        # ... for conv2d
        # update H, div (numele)^2
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # 2XX^T / nsamples
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())
```

`fasterquant()` 函数包含了文中提到的所有优化
- Hessian的稳定化处理: 首先对 H 的对角线元素进行阻尼 (dampening) 处理 (H[diag, diag] += damp)，这是为了增加数值稳定性，对应论文优化3中提到的 mild dampening.
- Cholesky分解: 接着通过 `torch.linalg.cholesky `和 `torch.cholesky_inverse` 稳定地计算出Hessian逆矩阵的Cholesky分解形式 Hinv。这完全对应了论文的优化3: Cholesky重构，避免了直接求逆可能带来的数值不稳定问题。- 
- Lazy Batch-Updates: `for i1 in range(0, self.columns, blocksize)` 这个外层循环，将权重矩阵的列分成多个 block 进行处理。对应论文的优化2: 延迟批量更新，以提高GPU的实际执行效率。
- 固定顺序量化: 内层循环 `for i in range(count)`: 按照固定的列顺序对权重进行量化。对应了论文的优化1: 任意顺序量化洞察，放弃了OBQ昂贵的贪心搜索，从而大幅降低了理论复杂度。
- 误差补偿更新: `W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))` 根据当前列 w 的量化误差 err1，利用预先算好的 Hinv 信息，精确地更新区块内所有尚未被量化的权重，以补偿误差。
- 应用块外更新: 在一个块处理完毕后，`W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])` 会将块内累积的总误差 Err1 的影响，一次性地更新到所有剩余的、位于该块之外的权重上。

```python
def fasterquant(
    self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
):
    W = self.layer.weight.data.clone()
    if isinstance(self.layer, nn.Conv2d):
        W = W.flatten(1)
    if isinstance(self.layer, transformers.Conv1D):
        W = W.t()
    W = W.float()

    tick = time.time()

    if not self.quantizer.ready():  # 计算量化参数 (scale, zero)
        self.quantizer.find_params(W, weight=True)

    H = self.H
    del self.H
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0

    if static_groups:
        import copy
        groups = []
        for i in range(0, self.columns, groupsize):
            quantizer = copy.deepcopy(self.quantizer)
            quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
            groups.append(quantizer)

    if actorder:  # 按照权重要性重排权重 OBQ
        perm = torch.argsort(torch.diag(H), descending=True)
        W = W[:, perm]
        H = H[perm][:, perm]
        invperm = torch.argsort(perm)

    Losses = torch.zeros_like(W)
    Q = torch.zeros_like(W)
    # --- 对应论文优化3: 通过Cholesky分解稳定地计算Hessian逆矩阵 ---
    # 1. 添加阻尼项(damp)以增加数值稳定性
    damp = percdamp * torch.mean(torch.diag(H))
    diag = torch.arange(self.columns, device=self.dev)
    H[diag, diag] += damp
    # 2. 稳定地计算 H 的逆矩阵 H_inv，并对其进行Cholesky分解
    H = torch.linalg.cholesky(H)  # L
    H = torch.cholesky_inverse(H)  # H^-1
    H = torch.linalg.cholesky(H, upper=True)
    Hinv = H  # H^-1 分解后的上三角
    
    # --- 核心量化循环 ---
    # 对应论文优化2: Lazy Batch-Updates，以blocksize为单位处理
    for i1 in range(0, self.columns, blocksize):
        i2 = min(i1 + blocksize, self.columns)
        count = i2 - i1

        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Losses1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        
        # 对应论文优化1: 按固定顺序 (列序) 量化，而非 OBQ 的贪心顺序
        for i in range(count):
            w = W1[:, i]  # 进行量化的列
            d = Hinv1[i, i]
            if groupsize != -1:   # 启用了分组量化(groupsize)，则为每个组动态计算量化参数
                if not static_groups:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                else:
                    idx = i1 + i
                    if actorder:
                        idx = perm[idx]
                    self.quantizer = groups[idx // groupsize]

            q = quantize(  # 对当前列进行量化
                w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
            ).flatten()
            Q1[:, i] = q
            Losses1[:, i] = (w - q) ** 2 / d ** 2
            
            # --- 核心步骤: 更新剩余未量化权重，以补偿当前权重w的量化误差 ---
            err1 = (w - q) / d
            # 更新块内(block)所有后续的权重
            W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
            Err1[:, i] = err1
        # 量化后的权重写入
        Q[:, i1:i2] = Q1
        Losses[:, i1:i2] = Losses1 / 2

        # --- 应用延迟更新 ---
        # 将当前块产生的总误差，一次性地更新到所有后续的权重上
        W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

    torch.cuda.synchronize()
    if actorder:
        Q = Q[:, invperm]

    # 将最终的量化权重Q写回到原始的layer中
    self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
```

`quant_cuda_kernel.cu` 实现了一个高效的、3-bit量化的矩阵与向量的乘法操作 `VecQuant3MatMul`. 文件中包含了两个版本的 Kernel: 
- VecQuant3MatMulKernel: 功能正确、逻辑清晰的通用版本。
- VecQuant3MatMulKernelFaster: 使用half2 (半精度浮点数向量) 和硬件内置函数 (intrinsics) 进行深度优化的版本，性能更高。

PS: 怀疑 `while (k < BLOCKWIDTH)` 写错，应该为 `while (k < BLOCKHEIGHT)`，原本意思应该是一个线程负责一列输出通道的计算，一次处理 3 个 int32 打包后权重，需要 BLOCKHEIGHT / 3 个循环完成部分和计算。

{{< details title= "Details of VecQuant3MatMulKernel">}}
```cpp
#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// 声明两个CUDA Kernel函数
// 模板化的通用版本
template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
);

// 使用half2优化的更快版本
__global__ void VecQuant3MatMulKernelFaster(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width
);

// 定义线程块的大小
const int BLOCKWIDTH  = 256; // 线程块的宽度，即每个块处理256个输出特征
const int BLOCKHEIGHT =  24; // 线程块的高度

// ===================================================================================
// C++ Wrapper 函数 - 作为从Python到CUDA的桥梁
// ===================================================================================

// 通用版本的 C++ Wrapper
void vecquant3matmul_cuda(
  torch::Tensor vec, // 输入向量 (激活值)
  torch::Tensor mat, // 3-bit量化权重矩阵 (以int32形式存储)
  torch::Tensor mul, // 输出向量
  torch::Tensor scales, // 量化参数 scale
  torch::Tensor zeros  // 量化参数 zero
) {
  int height = mat.size(0);
  int width = mat.size(1);

  // 计算Grid和Block的维度
  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  // 根据输入vec的类型 (float或half) ，分发并启动对应的模板化Kernel
  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant3matmul_cuda", ([&] {
      VecQuant3MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(),
        height, width
      );
    })
  );
}

// 更快版本的 C++ Wrapper
void vecquant3matmul_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT - 1) / BLOCKHEIGHT,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  // 直接启动 half2 优化的 Kernel
  VecQuant3MatMulKernelFaster<<<blocks, threads>>>(
    (half2*) vec.data_ptr(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    scales.data_ptr<float>(),
    zeros.data_ptr<float>(),
    height, width
  );
}

// 将int重新解释为unsigned int，用于位运算
__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

// ===================================================================================
// Kernel 1: 通用实现 VecQuant3MatMulKernel y = x*w
// ===================================================================================
template <typename scalar_t>
__global__ void VecQuant3MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
) {
  // 每个线程负责计算输出向量中的一个元素的部分和
  int row = BLOCKHEIGHT * blockIdx.x; // 线程块负责的起始行
  int col =  BLOCKWIDTH * blockIdx.y + threadIdx.x; // 线程负责的列 (输出特征) 

  // 优化: 使用共享内存(Shared Memory)缓存输入向量vec的一部分，减少全局内存读取
  __shared__ scalar_t blockvec[BLOCKWIDTH];
  blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * BLOCKWIDTH + threadIdx.x];
  __syncthreads(); // 确保所有线程都已将数据加载到共享内存

  // 获取当前列对应的scale和zero
  scalar_t scale = scales[col];
  scalar_t zero = zeros[col];

  scalar_t res = 0; // 用于累加结果的寄存器
  int i = width * row + col; // 计算在mat中的线性索引
  int k = 0; // 追踪在vec中的索引

  unsigned int tmp1, tmp2, tmp;

  // --- 核心解包与计算循环 ---
  // 循环处理一个线程块负责的所有输入特征
  // 每一次迭代，都是在为它自己负责的那一个输出特征 col，累加由32个输入特征产生的计算结果 (一个“部分和”) 
  while (k < BLOCKWIDTH) {
    // 读取一个32-bit整数，它包含了10个3-bit权重和另一个权重的2个bit
    tmp1 = as_unsigned(mat[i]);
    // 逐个解包、解量化并与vec元素相乘累加
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0]; // `& 0x7` (二进制111) 用于提取3-bit
    res += (scale * scalar_t((tmp1 >>  3) & 0x7) - zero) * blockvec[k + 1];
    res += (scale * scalar_t((tmp1 >>  6) & 0x7) - zero) * blockvec[k + 2];
    res += (scale * scalar_t((tmp1 >>  9) & 0x7) - zero) * blockvec[k + 3];
    res += (scale * scalar_t((tmp1 >> 12) & 0x7) - zero) * blockvec[k + 4];
    res += (scale * scalar_t((tmp1 >> 15) & 0x7) - zero) * blockvec[k + 5];
    res += (scale * scalar_t((tmp1 >> 18) & 0x7) - zero) * blockvec[k + 6];
    res += (scale * scalar_t((tmp1 >> 21) & 0x7) - zero) * blockvec[k + 7];
    res += (scale * scalar_t((tmp1 >> 24) & 0x7) - zero) * blockvec[k + 8];
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    
    i += width; // 移动到权重矩阵的下一行
    // 读取下一个32-bit整数
    tmp2 = as_unsigned(mat[i]);
    
    // --- 跨整数边界的权重处理 ---
    // 第11个权重分散在tmp1的高2位和tmp2的低1位，需要将它们拼接起来
    tmp = (tmp1 >> 30) | ((tmp2 << 2) & 0x4);
    tmp2 >>= 1;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;

    // ... 后续代码块重复这个解包逻辑 ...
    // 这个复杂的解包模式是由 quant.py 中的 pack 函数决定的，两者必须严格对应
    res += (scale * scalar_t((tmp2 >>  0) & 0x7) - zero) * blockvec[k + 0];
    // ...
    res += (scale * scalar_t((tmp2 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    tmp1 = as_unsigned(mat[i]);
    tmp = (tmp2 >> 30) | ((tmp1 << 1) & 0x6);
    tmp1 >>= 2;
    res += (scale * scalar_t(tmp) - zero) * blockvec[k + 10];
    k += 11;
    res += (scale * scalar_t((tmp1 >>  0) & 0x7) - zero) * blockvec[k + 0];
    // ...
    res += (scale * scalar_t((tmp1 >> 27) & 0x7) - zero) * blockvec[k + 9];
    i += width;
    k += 10;
  }

  // 使用原子加法将每个线程计算的局部结果安全地累加到全局内存的最终结果中
  atomicAdd(&mul[col], res);
}

// ===================================================================================
// Kernel 2: 优化版本 VecQuant3MatMulKernelFaster
// ===================================================================================
__global__ void VecQuant3MatMulKernelFaster(
    const  half2* __restrict__ vec, // 输入向量使用half2，一次处理2个半精度浮点数
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ scales,
    const  float* __restrict__ zeros,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;

  int row = BLOCKHEIGHT * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  // 优化: 共享内存缓存vec，使用half2类型，大小减半
  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[(row / BLOCKHEIGHT) * blockwidth2 + threadIdx.x];

  // 优化: 创建一个解量化查找表(Lookup Table)在共享内存中
  // deq2[v] 存储了量化值 v 和 v+1 解量化后的 half2 向量
  // 避免在循环中重复计算 scale * val - zero
  __shared__ half2 deq2[64][32];
  int val = threadIdx.x / 32;
  int off = threadIdx.x % 32;
  for (; val < 64; val += BLOCKWIDTH / 32) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0x7), __int2half_rn(val >> 3)
    );
  }

  // 同样将scale和zero转为half2向量
  half2 scale = __float2half2_rn(scales[col]);
  half2 zero = __float2half2_rn(-zeros[col]); // 注意这里是-zeros，因为FMA是a*b+c，对应 a*b-(-c)

  int i = width * row + col;
  int k = 0;

  float res = 0;
  half2 res2;

  unsigned int tmp1, tmp2, tmp;

  __syncthreads(); // 确保vec和查找表deq2都已加载完毕

  while (k < blockwidth2) {
    res2 = {}; // 重置累加器
    tmp1 = as_unsigned(mat[i]);
    // 优化: 使用__hfma2 (fused multiply-add for half2) 内置函数
    // 这个函数执行 a*b+c，一条指令完成两次乘法和两次加法，效率极高
    // 逻辑: hfma2(lookup_value, scale, zero) -> 解量化
    //        hfma2(dequantized_value, vec_value, accumulator) -> 乘加
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  0) & 0x3f][off], scale, zero), blockvec[k + 0], res2);
    // ... 解包逻辑与通用版本类似，但为适配half2和6-bit的查找表(0x3f)做了调整 ...
    res2 = __hfma2(__hfma2(deq2[(tmp1 >>  6) & 0x3f][off], scale, zero), blockvec[k + 1], res2);
    // ...
    i += width;
    k += 5;
    // 将half2累加器的两个half元素转为float并累加到最终结果
    res += __half2float(res2.x) + __half2float(res2.y);
  }

  atomicAdd(&mul[col], res);
}
```
{{< /details >}}

# AWQ (2024 MlSys): W4A16

![AWQ Observation.](https://share.note.youdao.com/yws/api/personal/file/WEB5b9ecd07af9c4d9f148a0bd71784683c?method=download&shareKey=f5af44a6df47f2b7c7fa3b1b120c3607 "AWQ Observation.")

Activation-aware Weight Quantization (AWQ) 的核心洞察是*模型中的所有权重并非同等重要*。作者们通过实验发现，模型中存在大约 0.1% 到 1% 的显著权重 (salient weights). 只要保持这些极少数的关键权重的精度，就能极大地维持模型的性能。传统的思路是看权重本身的数值大小或者 L2 范数。但作者发现权重的重要性不应由其自身的大小决定，而应由其处理的激活的强度来决定。

虽然上述方法效果很好，但它产生了一种**混合精度(Mixed-Precision)** 格式，即模型中同时存在FP16和低精度整数两种数据类型。这种混合精度格式对硬件实现非常不友好，会使得系统实现变得困难和低效。AWQ 设计了一种等价的缩放变换，通过在量化前放大重要权重、在计算时缩小对应激活值的方式，在不引入额外硬件开销的前提下，显著降低了关键权重的相对量化误差。

与 GPTQ 等需要通过复杂的逐块重构来补偿量化误差的方法不同 ，AWQ的缩放因子是通过一个简单的、数据驱动的**网格搜索 (grid search)** 确定的。这使得 AWQ 仅需少量校准数据即可确定最佳缩放策略。并且由于不依赖于在特定校准集上进行复杂的优化重构，AWQ不易对校准数据过拟合，能够更好地保持 LLM 在不同领域和任务上的通用知识。

## Protecting Salient Weights by Activation-aware Scaling

考虑一个线性运算 $y=wx$，量化后为 $y=Q(w)x$. 量化函数可定义为:

$$
Q(\mathbf{w})=\Delta\cdot\text{Round}(\frac{\mathbf{w}}{\Delta}),\quad\Delta=\frac{\max(|\mathbf{w}|)}{2^{N\boldsymbol{-}1}}
$$

- N 是量化后格式的位宽。
- $\Delta$ 是量化缩放系数。

> 这里 Q(w) 感觉是模拟量化，用于在浮点数域模拟量化操作带来的精度损失。并且这里采用的是对称量化的公式，实际后面 Source Code 部分可以发现 `else: # we actually never used this`，明确指出对称量化的分支在实践中并未使用。

AWQ 提出可以将运算等价变换为 $Q(w\cdot s)\cdot(x/s)$，其中 s 是一个大于 1 的缩放因子。即在量化前将权重乘以 s，在计算时将对应的激活值除以 s.

$$
Q(w\cdot s)\cdot\frac{x}{s}=\Delta^{'}\cdot\text{Round}(\frac{ws}{\Delta'})\cdot x\cdot\frac{1}{s}
$$

其中 $\Delta'$ 是权重乘以 s 后新的量化缩放系数。量化误差主要来源于取整操作 Round()，将一个浮点数四舍五入的量化误差是一个 [0, 0.5] 的均匀分布，误差期望为 0.25. 因此新旧量化误差可以表示为: 

$$
\begin{aligned}\operatorname{Err}(Q(w)x)&=\Delta\cdot\text{RoundErr}(\frac{w}{\Delta})\cdot x\\\text{Err}(Q(w\cdot s)(\frac{x}{s}))&=\Delta^{'}\cdot\text{RoundErr}(\frac{ws}{\Delta'})\cdot x\cdot\frac{1}{s}\end{aligned}
$$

当只对一个权重内的部分显著权重进行放大时，通常不会改变这个组的最大绝对值，因此量化缩放系数可以认为几乎不变，即 $\Delta^{\prime}\approx\Delta$. 因此新误差与旧误差的比率约为 1/s，这意味着**显著权重的相对量化误差被缩小了**。

**Searching to scale.** 过度缩放 (s 过大) 可能会增大 $\Delta$，反而会放大非显著权重的量化误差，损害模型整体性能。因此需要寻找一个最优的缩放因子。AWQ 将此问题建模为一个优化目标: 

$$
\begin{aligned}\mathbf{s}^*&=\arg\min\mathcal{L}(\mathbf{s})\\\mathcal{L}(\mathbf{s})&=\|Q(\mathbf{W}\cdot\mathrm{diag}(\mathbf{s}))(\mathrm{diag}(\mathbf{s})^{-1}\cdot\mathbf{X})-\mathbf{W}\mathbf{X}\|\end{aligned}
$$

- Q 是权重量化函数
- W 是 FP16 精度下的权重
- $\mathrm{diag}(\mathbf{s})^{-1}\cdot\mathbf{X}$ 可以被融合进前一个算子。

即寻找一个逐输入通道的缩放向量 s，使得量化后的输出与原始FP16输出的差距最小化。由于量化函数不可导，无法使用反向传播来直接求解。AWQ 采用了一种简单高效的数据驱动搜索策略。它定义了一个搜索空间: 

$$
\mathbf{s}=\mathbf{s}_\mathbf{X}^\alpha,\quad\alpha^*=\arg\min_{\alpha}\mathcal{L}(\mathbf{s}_\mathbf{X}^\alpha)
$$

- $\mathbf{s}_\mathbf{X}$ 是激活的每个通道的平均值向量。
- $\alpha \in [0,1]$ 用于平衡对显著通道和非显著通道的保护程度。

搜索问题就简化为寻找一个单一的超参数，通过在一个小范围内进行快速的**网格搜索 (grid search)** 即可找到最佳的值。同时使用了梯度裁切 (weight clipping) 来最小化 MSE 误差。

## Experiments

| **实验类别** | **实验设置与目标** | **核心发现与结果** | **关键数据/图表引用** |
| :--- | :--- | :--- | :--- |
| **语言模型性能 (LLaMA/Llama-2)** | 评估AWQ在不同尺寸的LLaMA和Llama-2模型上的语言建模能力 (Perplexity, PPL). | AWQ在各种模型规模和精度下，其PPL始终优于RTN和GPTQ (包括GPTQ-R). | **Table 4**: 在Llama-2-70B INT4量化下，AWQ PPL为3.41，优于GPTQ的3.42和RTN的3.46. |
| **新架构模型性能 (Mistral/Mixtral)** | 评估AWQ在包含GQA和MoE等新架构的Mistral和Mixtral模型上的量化性能。 | 实验结果表明AWQ在Mistral和Mixtral模型上均取得了出色的量化性能，证明了其对不同模型架构的有效性. | **Table 5**: Mixtral-8x7B在INT4量化下PPL为6.05，非常接近FP16的5.94. 
| **指令遵循模型泛化性** | 在指令微调模型Vicuna上评估量化后的模型性能，使用GPT-4进行打分。 | AWQ量化后的Vicuna模型在与FP16版本的对比中，“获胜”的案例数一致性地超过了RTN和GPTQ，展示了其对指令微调模型的良好泛化能力. | **Figure 5**: 在Vicuna-13B上，AWQ量化模型获胜75次，远超GPTQ的57次. |
| **多模态模型泛化性** | 在视觉语言模型(VLM) OpenFlamingo-9B和VILA上评估量化性能，测试COCO图像描述等任务。 | AWQ在零样本和少样本设置下均优于现有方法，证明了其对多模态和上下文学习任务的泛化性。AWQ实现了近乎无损的量化性能。 | **Table 6**: 在OpenFlamingo-9B INT4量化下，AWQ将32-shot的性能下降从4.57 (RTN) 减少到仅1.17. <br>**Table 7**: VILA模型在11个视觉语言基准上表现出无损性能. |
| **特定领域任务泛化性** | 在编程(MBPP)和数学推理(GSM8K)等复杂生成任务上评估量化性能。 | AWQ在编程和数学数据集上均优于现有方法，证明了其对复杂推理任务的泛化能力。AWQ INT4量化后的性能与FP16相当. | **Table 8**: CodeLlama-7B在MBPP (pass@1)上的准确率为40.64，优于GPTQ的31.97，接近FP16的38.53. |
| **数据效率与鲁棒性** | 比较AWQ和GPTQ对校准集大小和分布的敏感度。 | 1. **数据效率**: AWQ达到优异性能所需的校准集比GPTQ小10倍以上。<br>2. **鲁棒性**: 当校准集与评估集分布不同时，AWQ的性能下降远小于GPTQ. | **Figure 8(a)**: AWQ使用16个序列校准的效果优于GPTQ使用192个序列。<br>**Figure 8(b)**: 跨域校准时，GPTQ PPL增加2.3-4.9，而AWQ仅增加0.5-0.6. |
| **系统加速评测** | 使用TinyChat系统在桌面(RTX 4090)、移动(Jetson Orin)和笔记本(RTX 4070) GPU上测量实际推理速度。 | 1. **显著加速**: TinyChat将AWQ的理论内存节省转化为实际速度提升，比Huggingface FP16实现平均快3.2-3.3倍.<br>2. **赋能低资源设备**: 使大模型 (如Llama-2-13B)能在内存有限的笔记本GPU上运行. | **Figure 9**: 在RTX 4090上，TinyChat(AWQ)对Llama-2-7B的推理速度达到194 tokens/s，远超Huggingface FP16的52 tokens/s. |

## Source Code

[AWQ 源代码中](https://github.com/mit-han-lab/llm-awq) 的量化过程可以分为两个核心阶段: 

1. 分析与参数搜索阶段 (Analysis & Search): 这个阶段并不进行真正的量化。它的目标是为模型中的每一个线性层找到最优的缩放因子 (scales) 和 裁剪范围 (clipping ranges)。这个过程由 pre_quant.py 中的 run_awq 函数驱动，核心算法在 auto_scale.py 和 auto_clip.py 中实现。

2. 真实量化与模型替换阶段 (Real Quantization & Module Replacement): 这个阶段使用第一阶段计算出的参数，对原始的 FP16 权重进行真实的低比特量化 (例如，转换为 INT4 整数) ，并将原始的 nn.Linear 模块替换为专门为高效推理设计的 WQLinear 模块。这个过程由 quantizer.py 中的 real_quantize_model_weight 函数和 qmodule.py 中的 WQLinear 类实现。

源代码 `quantize/auto_scale.py` 里的 `_search_module_scale()` 函数是论文中 3.2 节确定缩放参数的实现。文中说的 grid search 其实就是 `[0, 1)` 区间内均匀 0.05 步长去搜索，选择量化后 MSE 最小的那一个。

```python
# 文件: quantize/auto_scale.py

def auto_scale_block(module, module_kwargs, w_bit, q_config, input_feat):
    # ... (此函数根据模型结构，定义了哪些层之间需要进行缩放，并调用下面的搜索函数) ...
    # 例如，对于Llama模型，它会定义(input_layernorm -> q_proj, k_proj, v_proj)之间的缩放关系

    # 核心搜索函数
    def _search_module_scale(block, linears2scale: list, x, kwargs={}):
        # 获取激活值的统计量，对应论文中的 |X|
        x_max = get_act_scale(x)

        best_error = float("inf")
        best_ratio = -1
        best_scales = None

        n_grid = 20  # 网格搜索的步数
        org_sd = {k: v.cpu() for k, v in block.state_dict().items()}

        # 步骤 1: 网格搜索最优的缩放超参数 alpha (代码中称为 ratio)
        # 对应论文公式 (5): s = |X|^α
        for ratio in range(n_grid):
            ratio = ratio * 1 / n_grid
            
            # 计算当前 ratio 对应的缩放因子 scales
            scales = x_max.pow(ratio).clamp(min=1e-4).view(-1)
            scales = scales / (scales.max() * scales.min()).sqrt() # 归一化

            # 步骤 2: 模拟应用缩放和量化，并计算误差
            # 临时对权重应用缩放因子
            for fc in linears2scale:
                fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
                # 伪量化后再反向缩放，以模拟量化误差
                fc.weight.data = w_quantize_func(fc.weight.data) / (scales.view(1, -1))
            
            # 计算经过模拟量化后的模块输出
            out = block(x, **kwargs)
            if isinstance(out, tuple):
                out = out[0]

            # 计算与原始FP16输出的均方误差 (MSE)
            loss = ((org_out - out).float().pow(2).mean().item())
            
            # 记录误差最小的 ratio 和 scales
            is_best = loss < best_error
            if is_best:
                best_error = loss
                best_ratio = ratio
                best_scales = scales
            
            # 恢复原始权重以进行下一次循环
            block.load_state_dict(org_sd)
        
        # 返回找到的最佳缩放因子
        return best_scales.detach()
    # ...
```

在应用上一步搜索出的 s 向量后，源代码 `quantize/auto_clip.py` 里的 `auto_clip_layer()` 函数实现了论文中 3.2 节末尾提到的 weight clip. 它通过搜索一个最佳的裁剪范围来进一步减少由极端权重值 (outliers) 引起的量化误差。

```python
@torch.no_grad()
def auto_clip_layer(
    w, input_feat, n_bit, q_config, n_grid=20, max_shrink=0.5, n_sample_token=512
):
    # ... (准备工作，对权重和激活值进行分组和采样) ...

    org_max_val = w.abs().amax(dim=-1, keepdim=True) # 获取原始每一列的最大绝对值
    best_max_val = org_max_val.clone()
    min_errs = torch.ones_like(org_max_val) * 1e9
    org_out = (input_feat * w).sum(dim=-1) # 计算原始FP16输出

    # 搜索最佳裁剪范围
    # 从原始最大值开始，逐步缩小裁剪范围 (max_val)
    for i_s in range(int(max_shrink * n_grid)):
        max_val = org_max_val * (1 - i_s / n_grid)
        cur_w = torch.clamp(w, -max_val, max_val) # 裁剪权重
        q_w = pseudo_quantize_tensor(cur_w, n_bit=n_bit, **q_config)  # 对裁剪后的权重进行伪量化
        cur_out = (input_feat * q_w).sum(dim=-1)
        err = (cur_out - org_out).pow(2).mean(dim=1).view(min_errs.shape)
        # 记录误差最小的裁剪范围
        cur_best_idx = err < min_errs
        min_errs[cur_best_idx] = err[cur_best_idx]
        best_max_val[cur_best_idx] = max_val[cur_best_idx]
        
    # 返回找到的最佳裁剪范围
    return best_max_val.squeeze(1)
```

在获得最优的 s 向量和 clip 参数后，`quantizer.py` 里的 `real_quantize_model_weight()` 将模型转换为真正可用于高效推理的量化版本，它遍历所有线性层，将它们替换为 `WQLinear`.

```python
@torch.no_grad()
def real_quantize_model_weight(model, w_bit, q_config, init_only=False):
    from .qmodule import WQLinear
    # ...
    
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="real weight quantization..."):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        # ...

        for name, module in named_linears.items():
            if not init_only:
                module.cuda()
                
                # 步骤 1: 再次进行伪量化，但这次是为了获取真实的 scales 和 zeros
                # 此时传入的 module.weight.data 已经是经过 apply_scale 和 apply_clip 修改过的
                module.weight.data, scales, zeros = pseudo_quantize_tensor(
                    module.weight.data, n_bit=w_bit, get_scale_zp=True, **q_config
                )
                
                # 步骤 2: 从原始 nn.Linear 创建 WQLinear 模块
                # 这个 from_linear 方法会执行权重量化、打包等所有转换操作
                q_linear = WQLinear.from_linear(
                    module, w_bit, q_config["q_group_size"], False, scales, zeros
                )
                
                module.cpu()
                q_linear.to(next(layer.parameters()).device)
                
                # 步骤 3: 替换原始模块
                # 使用新的 WQLinear 模块替换掉原来的 nn.Linear
                set_op_by_name(layer, name, q_linear)
                # ... (清理内存) ...
    # ...
```

`WQLinear` 是最终用来替代 `nn.Linear` 的模块。它内部不存储FP16权重，而是存储打包后的INT4权重、量化尺度和零点，并拥有一个使用 CUDA Kernels 实现的高效 forward 方法。对应了论文 Section 4 中对 TinyChat 推理系统的描述。

```python
class WQLinear(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, dtype=torch.float16):
        super().__init__()
        # ...
        # 初始化用于存储量化参数的 buffers
        # self.qweight: 存储打包后的INT4权重
        # self.scales: 存储量化缩放因子
        # self.scaled_zeros: 存储 (scales * zeros)，用于加速计算
        # ...

    @classmethod
    def from_linear(
        cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None
    ):
        awq_linear = cls(...) # 初始化空的 WQLinear
        if init_only:
            return awq_linear

        # ... (准备 scales 和 zeros) ...
        awq_linear.scales = qscales.transpose(1, 0).contiguous()
        if linear.bias is not not None:
            awq_linear.bias = linear.bias.clone().to(dtype)

        # 核心: 将FP16权重转换为INT4整数
        intweight = []
        for idx in range(awq_linear.in_features):
            # 这是标准的仿射量化公式: Round((X - zero) / scale)
            # 这里为了计算效率，转换为: Round((X / scale) + zero)
            # 更进一步，为了融合 X 和 zero*scale, 转换为 Round((X + zero*scale) / scale)
            intweight.append(
                torch.round(
                    (linear.weight.data[:, idx] + scale_zeros[:, idx // group_size])
                    / qscales[:, idx // group_size]
                ).to(torch.int)[:, None]
            )
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.to(dtype=torch.int32)
        
        # 核心: 权重打包 (Weight Packing)
        # 将计算出的INT4整数权重打包成INT16格式，以实现更高效的内存访问
        # 这对应论文 Section 4.2 的 "SIMD-aware weight packing"
        awq_linear.qweight = pack_intweight(
            intweight.contiguous(), interleave=4, kstride=64
        )
        
        # ... (准备 scaled_zeros) ...
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        # 调用定制的 CUDA kernel 进行矩阵乘法
        # 对应论文 Section 4.2 的 "On-the-fly weight dequantization"
        # CUDA kernel 会在计算时即时解包和反量化权重，避免了将完整的FP16权重写回内存
        if inputs.numel() / inputs.shape[-1] < 8:
            # 针对 batch size 较小的情况，使用 GEMV (矩阵-向量乘法) kernel
            out = awq_inference_engine.gemv_forward_cuda_new(...)
        else:
            # 针对 batch size 较大的情况，使用 GEMM (矩阵-矩阵乘法) kernel
            out = awq_inference_engine.gemm_forward_cuda_new(...)
        
        out = out + self.bias if self.bias is not None else out
        return out
```

# SmoothQuant (2023 ICML) W8A8

![SmoothQuant's Intuition.](https://share.note.youdao.com/yws/api/personal/file/WEBc9f24d14b20c3ff076bf04a0423b4475?method=download&shareKey=c6a513fd01f87c239f38339bfca28672 "SmoothQuant's Intuition.")

由于 LLM 激活值中会出现很大的离群点 (Outliers)，会导致缩放因子非常大，导致大部分正常值的量化精度极低 (论文中称为 low effective bits)，从而引起精度下降。作者观察到一个关键现象: 虽然激活值因为离群点而难以量化，但权重的数值分布却非常均匀平滑，极易量化。同时，激活值的离群点往往固定出现在某些特定的通道上。

通道级 (per-channel) 激活量化虽然能有效保持精度，但与硬件加速器不兼容；而硬件友好的 Token级 (per-token) 激活量化又无法解决离群点问题，导致精度下降。

SmoothQuant 的出发点不是直接量化原始的激活值 X，而是先对其进行一个平滑处理这个处理通过一个按通道计算的平滑因子 s (一个向量，其长度等于输入通道数 $C_i$) 来实现。因此 SmoothQuant 是一个独立的预处理模块，**它与后续采用哪种具体的量化方案是相互独立的、互不干扰的**。

$$
Y=(X\cdot\mathrm{diag}(s)^{-1})\cdot(\mathrm{diag}(s)\cdot W)=\hat{X}\cdot\hat{W}
$$

这个变换可以完全离线完成。平滑因子 s 对 X 的影响，可以被融合到产生 X 的前一个线性层或归一化层的参数中，因此在实际推理时，模型直接产生的就是平滑后的 $\hat{W}$，不需要任何额外的除法运算。

为了减小量化误差，应该为所有通道增加其有效量化位数。最直接的方式是让平滑后的激活值 $\hat{X}$ 在所有通道上的数值范围都完全一样。可以通过设置 $s_j = \max(|X_j|)$ (即 s 的每个元素是对应激活通道的最大绝对值) 来实现。这样做之后，$\hat{X}$ 的每个通道的最大绝对值都变成了1，变得非常容易量化。然而，这相当于将所有的离群点信息 (即量化难度) 完全推给了权重 $\hat{W}$. 调整后的权重 $\hat{W}$ 变得极难量化，会导致严重的精度下降。

> The total effective quantization bits would be largest when all the channels have the same maximum magnitude.

![Main Idea of SmoothQuant When α is 0.5.](https://share.note.youdao.com/yws/api/personal/file/WEBe773b468bfeeac32fa75ec648991eaae?method=download&shareKey=f60b25a1dd6b961bb3907653216892a4 "Main Idea of SmoothQuant When α is 0.5.")

为此，SmoothQuant 引入了一个超参数 α，称之为迁移强度 (migration strength) ，来控制困难迁移的程度。平滑因子 s 的最终计算公式如下: 

$$
s_j=\frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}
$$

其中 $\alpha \in [0,1]$，越靠近 0，对激活的平滑作用越小；反之对激活的平滑作用很强。激活值的动态范围是变化的，因此 $max(|X_j|)$ 是通过在少量校准样本上运行模型来统计估算的。

![SmoothQuant's precision mapping for a Transformer block. All compute-intensive operators like linear layers and batched matmul (BMMs) use INT8 arithmetic.](https://share.note.youdao.com/yws/api/personal/file/WEB1f55c961faf22cd88c6fd98723f4d946?method=download&shareKey=7d19e947ace6c2b8d84442e3e4044934 "SmoothQuant's precision mapping for a Transformer block. All compute-intensive operators like linear layers and batched matmul (BMMs) use INT8 arithmetic.")

SmoothQuant 主要应用于计算最密集的 self-attention 和 FFN 这两个子模块的输入激活值上。所有计算密集型算子，如线性层和批量矩阵乘法 (BMM) ，其权重和激活值都进行 INT8 量化 (W8A8). 一些计算量较小的、对数值精度敏感的算子，如LayerNorm、Softmax 和 ReLU，则保留为 FP16 精度。


## Experiments

| 实验方面 | 核心内容 | 关键结果与数据 |
| :--- | :--- | :--- |
| **实验设置 (Setups)** | **- 基线方法**: 与FP16、朴素W8A8量化、ZeroQuant 、LLM.int8() 、Outlier Suppression  进行对比。<br>**- 测试模型**: 覆盖OPT、BLOOM、GLM-130B、LLaMA-1/2、Falcon、Mistral、Mixtral 等多种主流LLM。<br>**- 校准方法**: 使用预训练数据集(Pile)中的512个随机句子进行一次性离线校准。<br>**- 实现后端**: 基于PyTorch和高性能的FasterTransformer进行验证。 | **- 迁移强度 `α` 设置**: 对OPT和BLOOM模型，`α=0.5` 是一个通用平衡点；对GLM-130B，`α=0.75` 效果更佳。 |
| **精度验证 (Accuracy)** | 在多种LLM架构、不同参数规模的模型上验证SmoothQuant W8A8量化的准确性，包括基础模型、指令微调模型和混合专家(MoE)模型。 | **- OPT-175B**: SmoothQuant-O3平均准确率(71.1%)与FP16(71.6%)几乎持平，而ZeroQuant(31.7%)等基线方法精度完全崩溃。<br>**- 跨模型通用性**: 成功将BLOOM-176B, GLM-130B, Llama-2-70B, Mixtral-8x7B  等模型量化到W8A8，且性能损失极小。<br>**- 尺寸一致性**: 对从1.3B到175B的所有尺寸的OPT模型均能保持与FP16相当的精度 。 |
| **性能测试 (Speedup & Memory)** | 在PyTorch和FasterTransformer后端上，测量并对比上下文阶段(Context-stage)和解码阶段(Decoding-stage)的端到端推理延迟和峰值GPU显存占用。 | **- 加速效果**: 在FasterTransformer上，相比FP16实现，最高可达 **1.56倍** 加速。LLM.int8() 在多数情况下比FP16更慢。<br>**- 显存节省**: 显存占用接近减半(约 **2倍** 节省). <br>**- 硬件成本降低**: 对于OPT-175B，SmoothQuant仅需 **4个GPU** 即可达到FP16在 **8个GPU** 上的相似延迟。 |
| **超大规模模型验证 (Scaling Up)** | 将SmoothQuant方法扩展应用于超过5000亿参数的MT-NLG 530B模型，验证其在极大模型上的有效性和可扩展性。 | **- 精度**: 成功将MT-NLG 530B量化到W8A8，且在多个基准测试上平均准确率与FP16持平(均为73.1%). <br>**- 部署**: 使MT-NLG 530B能在一个**单节点(8x A100 GPU)** 内完成部署和服务，而FP16需要16个GPU. |
| **消融研究 (Ablation Study)** | **1. 量化方案影响**: 对比SmoothQuant三种不同效率等级(O1, O2, O3)的延迟差异。<br>**2. 迁移强度 `α` 影响**: 分析超参数 `α` 对模型量化后精度的影响，以验证其“平衡点”的假设。 | **- 量化方案**: 量化粒度越粗(从per-token动态到per-tensor静态，O1->O3)，延迟越低。<br>**- 迁移强度 `α`**: 存在一个“甜蜜点(sweet spot)”区间(如OPT-175B的0.4-0.6). `α` 过小，激活难量化；`α` 过大，权重难量化，都会导致精度急剧下降。 |

## Source Code

`smoothquant/calibration.py` 通过给模型的前向传播挂上hooks，在不改变模型结构的情况下，捕获每一个 `nn.Linear` 层输入的激活值，并统计出每个通道的最大绝对值。

```python
def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    # act_scales是一个字典，用于存储每个线性层的输入激活尺度
    # key是层的名字 (e.g., 'model.decoder.layers.0.self_attn.q_proj')
    # value是该层每个输入通道的最大绝对值
    act_scales = {}

    # 定义一个函数，用于统计张量的尺度
    def stat_tensor(name, tensor):
        hidden_dim = tensor.shape[-1]  # 获取特征维度/通道数
        # 将张量 reshape 成 (token数, 通道数)，取绝对值，并在token维度上取最大值
        # 得到一个长度为 hidden_dim 的向量，代表每个通道的最大值
        comming_max = torch.max(tensor.view(-1, hidden_dim).abs().detach(), dim=0)[0].float().cpu()

        if name in act_scales:  # 持续更新每个通道的最大值
            act_scales[name] = torch.max(act_scales[name], comming_max)
        else:
            act_scales[name] = comming_max

    # 定义前向钩子函数
    def stat_input_hook(m, x, y, name):
        # x是线性层的输入，也就是我们关心的激活值
        if isinstance(x, tuple):
            x = x[0]
        stat_tensor(name, x)

    hooks = []
    for name, m in model.named_modules():  # 遍历模型中所有的模块
        if isinstance(m, nn.Linear):  # 如果模块是线性层 (nn.Linear)，就给它注册一个前向钩子
            hooks.append(
                m.register_forward_hook(functools.partial(stat_input_hook, name=name))
            )

    # 加载校准数据集
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    # 用少量样本迭代校准过程
    for i in tqdm(range(num_samples)):
        input_ids = tokenizer(
            dataset[i]["text"], return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        # 运行模型的前向传播，触发所有注册的钩子函数
        model(input_ids)

    for h in hooks:  # 移除所有钩子，清理现场
        h.remove()

    return act_scales
```

`smoothquant/smooth.py` 根据校准阶段得到的激活尺度 act_scales 和当前模型的权重尺度，计算出平滑因子 s，然后用 s 来调整 LayerNorm 层和 Linear 层的参数，完成迁移。

```python
@torch.no_grad()
def smooth_ln_fcs(ln, fcs, act_scales, alpha=0.5):
    # ln: LayerNorm层
    # fcs: 一个或多个Linear层 (例如q, k, v投影)
    # act_scales: 对应fcs输入的激活尺度，即max(|X_j|)
    # alpha: 迁移强度

    # ... (省略断言和设备、类型获取) ...

    # 1. 获取权重的尺度 max(|W_j|)
    #    对每个fc层，计算其权重在输出维度上的最大绝对值，然后拼接起来
    weight_scales = torch.cat(
        [fc.weight.abs().max(dim=0, keepdim=True)[0] for fc in fcs], dim=0
    )
    # 如果有多个fc层，取它们在对应输入通道上的最大值
    weight_scales = weight_scales.max(dim=0)[0].clamp(min=1e-5)

    # 2. 计算平滑因子 s
    #    对应论文的公式(4)
    scales = (
        (act_scales.pow(alpha) / weight_scales.pow(1 - alpha))
        .clamp(min=1e-5)
        .to(device)
        .to(dtype)
    )

    # 3. 执行平滑变换 (数学等价变换)
    #    Y = X * W  =>  Y = (X / s) * (s * W)
    #    这里的X是由LayerNorm输出的，所以对LayerNorm的参数进行修改
    #    等效于对X除以s
    ln.weight.div_(scales)
    ln.bias.div_(scales)

    #    对后续的Linear层的权重乘以s
    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

# 主函数，遍历模型中所有需要平滑的层
@torch.no_grad()
def smooth_lm(model, scales, alpha=0.5):
    for name, module in model.named_modules():  # 遍历模型的所有模块
        # 根据模型结构 (如OPT, BLOOM, Llama等) 找到对应的LayerNorm和Linear层组合
        if isinstance(module, OPTDecoderLayer):
            # 对Attention部分的Q,K,V线性层进行平滑
            attn_ln = module.self_attn_layer_norm
            qkv = [
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ]
            qkv_input_scales = scales[name + ".self_attn.q_proj"]
            smooth_ln_fcs(attn_ln, qkv, qkv_input_scales, alpha)

            # 对FFN部分的全连接层进行平滑
            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + ".fc1"]
            smooth_ln_fcs(ffn_ln, fc1, fc1_input_scales, alpha)
        # ... (对Llama等其他模型的处理逻辑类似) ...
```

`smoothquant/opt.py` 为 OPT 模型定义了专门的INT8版本 (`Int8OPTForCausalLM`)，其中所有的线性层和BMM操作都被替换为torch-int库提供的、能调用底层INT8硬件核心的算子。

```python
# 文件: smoothquant/opt.py

# 引入torch-int库中真正的INT8线性层
from torch_int.nn.linear import W8A8BFP32OFP32Linear, W8A8B8O8Linear
from torch_int.nn.bmm import BMM_S8T_S8N_S8T

# 定义INT8版本的Attention模块
class Int8OPTAttention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ...
        # 使用torch-int提供的INT8 BMM算子
        self.qk_bmm = BMM_S8T_S8N_F32T(1.0)
        self.pv_bmm = BMM_S8T_S8N_S8T(1.0)

        # 使用torch-int提供的INT8 线性层算子
        self.k_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.v_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.q_proj = W8A8B8O8Linear(embed_dim, embed_dim)
        self.out_proj = W8A8BFP32OFP32Linear(embed_dim, embed_dim)

    @staticmethod
    @torch.no_grad()
    def from_float(module: OPTAttention, input_scale: float, ...):
        # 这个函数负责将一个平滑后的FP16模块转换为INT8模块
        # 它会加载权重、偏置，并设置好量化尺度
        int8_module = Int8OPTAttention(module.embed_dim, module.num_heads)
        # ...
        # 调用W8A8B8O8Linear.from_float，这个函数会进行真实的权重INT8量化
        int8_module.q_proj = W8A8B8O8Linear.from_float(
            module.q_proj, input_scale, q_output_scale
        )
        # ... (转换所有线性层)
        return int8_module

# ... (类似地定义Int8OPTDecoderLayer, Int8OPTModel, Int8OPTForCausalLM)

# 最终，Int8OPTForCausalLM.from_float会将整个FP16模型转换为真正的INT8模型
# 这个过程在 examples/export_int8_model.py 中被调用
```

# QuaRot (2024 NeurIPS) W4A4

QuaRot 的核心思想基于一个名为计算不变性 (Computational Invariance) 的理论。该理论指出，对于 Transformer 模型中的某些层，我们可以用一个正交矩阵 (Orthogonal Matrix) 同时旋转输入激活值和反向旋转该层的权重矩阵，而模型的最终输出保持不变。正交变换只改变向量的方向，不改变其长度 (范数) ，因此不会改变像 RMSNorm 这类依赖于范数的归一化操作的结果。

> As we target consumer-type GPUs, we evaluate all the performance experiments on NVIDIA RTX 3090 GPUs.

## Preliminary: Computational Invariance Theorem

在 Transformer 模型中，我们可以用一个正交矩阵 (Orthogonal Matrix) $Q$ 来同时变换相邻两个模块的权重以及它们之间传递的激活值，而模型的最终输出函数保持不变。

该定理的实现依赖于对模型中成对出现的权重矩阵进行协同修改。具体流程如下: 

1. 选择变换矩阵: 首先选择一个正交矩阵 $Q$. 在 QuaRot 中，这是一个 Hadamard 矩阵。

2. 修改成对权重: 对于一个 Transformer 模块的输出权重矩阵 (例如 FFN 块中的 $W_{down}$) ，将其进行右乘修改，变为 $W_{down} \cdot Q^{T}$. 相应地，对于下一个 Transformer 模块的输入权重矩阵 (例如 FFN 块中的 $W_{up}$ 和 $W_{gate}$)，将其进行左乘修改，变为 $Q \cdot W_{up}$ 和 $Q \cdot W_{gate}$.
3. 激活值的隐式变换: 当权重被修改后，第一个模块的输出激活值 (原本是 $X$) 现在变成了 $X' = X \cdot Q^{T}$. 在下一个模块中，它与被修改过的输入权重 $Q \cdot W_{up}$ 相乘。在计算过程中，$Q^{\top}$ 和 $Q$ 相遇并因为 $Q^{\top}Q = I$ (单位矩阵) 而相互抵消，从而保证了计算的数学等价性。

RMSNorm 通过将输入向量除以其自身的范数来进行归一化。正交变换的一个核心数学性质是它不改变向量的范数。RMSNorm 和旋转操作满足: 

$$
RMSNorm(X)=RMSNorm(XQ^T)Q
$$

该定理成立有一个前提，即 RMSNorm 层中的线性缩放部分 (论文中的 $diag(\alpha)$) 需要预先被吸收到相邻的权重矩阵中。

## Method

QuaRot 分为两个阶段。在第一阶段，对模型权重进行修改 (全精度) ，并在模型的前向传递中插入两个额外的 Hadamard操作。在第二阶段，使用一些现有的方法 (默认 GPTQ) 对权重进行量化，并将量化操作添加到前向传递中，以启用激活 (和缓存) 的在线量化。

![QuaRot Applied to a LLaMa-style FFN.](https://share.note.youdao.com/yws/api/personal/file/WEB6d64dfb3f8b1748af38f269f9ff4653f?method=download&shareKey=4880a25540fbb9233c1ebeb2eda612a1 "QuaRot Applied to a LLaMa-style FFN.")

**Stage 1a: Weight Modification.** 首先，将RMSNorm层中的可学习缩放参数 $diag(\alpha)$ 融合到相邻的权重矩阵中。选择一个与模型隐藏层维度相匹配的随机 Hadmard 矩阵 $Q$. 根据计算不变性理论，对模型中成对出现的输入和输出权重矩阵进行修改，以 $W_k$ 为例: 

$$
\mathbf{W}_k\leftarrow\mathbf{Q}^\top\mathrm{diag}(\boldsymbol{\alpha})\mathbf{W}_k
$$

对于输出侧的权重矩阵 (如FFN中的$W_{down}$，或Attention中的$W_{out}$) ，用 $Q$ 进行右乘。模型各模块之间传递的激活值$X$ 被隐式地变为了 $XQ$，这样可以消除异常值。

**Stage 1b: Rotate FFN activations.** 在 FFN中，紧邻 $W_{down}$ 之前，插入一个在线的 Hadamard 变换。在每次前向传播时，都执行一次这个快速变换。为了抵消这个在线变换，同时将另一个 Hadamard 矩阵 $H$ 融合到 $W_{down}$ 中，即$W_{down} \leftarrow HW_{down}$.

结合阶段 1a 的全局旋转矩阵 $Q$，最终FFN的下采样权重矩阵被修改为$HW_{down}Q$. 通过这种“一个在线变换 + 一个融合变换”的配对，移除了FFN内部激活值的异常值，而计算开销增加极小。

> 作者的意思应该是全精度被量化之前都需要进行一次正交变换以消除异常值。

![QuaRot Applied to an Attention Component.](https://share.note.youdao.com/yws/api/personal/file/WEBbff9f406e6d2317fa0a32144a3a6aaf2?method=download&shareKey=b555e26515be4df7465dc7b1474eb8e1 "QuaRot Applied to an Attention Component.")

**Stage 1c: Attention Value Projection.** 在注意力计算的最终环节，值投影矩阵 $W_v$ 和输出投影矩阵 $W_{out}$  实际上是隐式地相乘的。利用这一点，对每个注意力头的 $W_v^{(h)}$ 和 $W_{out}^{(h)}$ 分别进行右乘和左乘一个与头维度 $d_h$ 匹配的 Hadamard 矩阵 $H_{d_h}$. 

$$
\mathbf{W}_v^{(h)}\leftarrow\mathbf{W}_v^{(h)}\mathbf{H}_{d_h},\quad\mathbf{W}_\mathrm{out}^{(h)}\leftarrow\mathbf{H}_{d_h}\mathbf{W}_\mathrm{out}^{(h)}.
$$

这两个变换在计算流中完美抵消，不改变结果。因此权重可以用 Kronecker 乘积表示为

$$
\mathbf{W}_v\leftarrow\mathbf{W}_v(\mathbf{I}\otimes\mathbf{H}_{d_h}),\quad\mathbf{W}_\mathrm{out}\leftarrow(\mathbf{I}\otimes\mathbf{H}_{d_h})\mathbf{W}_\mathrm{out}\:.
$$

为了实现更彻底的旋转，作者进一步利用 Hadamard 矩阵的性质 $\mathbf{H}_{n_h\times d_h}=(\mathbf{I}\otimes\mathbf{H}_{d_h})(\mathbf{H}_{n_h}\otimes\mathbf{I})$，在输出投影前增加了一个在线的、跨注意力头的 Hadamard 变换。

**Stage 1d: Key Rotation.** 与 V 向量不同，K 和 Q 在与$W_k$ 和 $W_q$ 相乘后，还需要经过 RoPE. 这个非线性操作的存在，使得无法像之前那样直接将哈达玛矩阵融合进 $W_k$ 和$W_q$ 中。因此，作者采用了纯**在线变换**的策略。在 RoPE 操作之后，对 K 和 Q 都进行一次逐头的哈达玛在线旋转。

由于在计算注意力分数时，是计算 $QK^{\top}$，而$(QH)(KH)^{\top} = QHH^{\top}K^{\top} = QK^{\top}$. 因此，同时旋转Q和K保证了最终的注意力分数矩阵保持不变，KV Cache 都能保存为量化后的结果。

$$
\mathbf{Q}\leftarrow\mathrm{~Pos}(\mathbf{X}\mathbf{W}_q)(\mathbf{I}\otimes\mathbf{H}_{d_h})=\mathrm{concat}[\mathrm{Pos}(\mathbf{Q}_1)\mathbf{H}_{d_h},\ldots,\mathrm{Pos}(\mathbf{Q}_{n_h})\mathbf{H}_{d_h}]
$$
$$
\mathbf{K}\leftarrow\mathrm{~Pos}(\mathbf{X}\mathbf{W}_k)(\mathbf{I}\otimes\mathbf{H}_{d_h})=\mathrm{concat}[\mathrm{Pos}(\mathbf{K}_1)\mathbf{H}_{d_h},\ldots,\mathrm{Pos}(\mathbf{K}_{n_h})\mathbf{H}_{d_h}].
$$

**Stage 2a: Weight Quantization.** 改造后的权重可以直接使用任何现有的量化方法进行量化。论文默认使用 GPTQ，也展示了使用更简单的四舍五入方法同样能取得很好的效果。

**Stage 2b: Online Quantization Operations.** 于模型前向传播过程中的激活值，采用在线 (on-the-fly) 量化。具体方案是逐-Token对称量化 (per-token symmetric quantization). 对输入矩阵的每一行 (代表一个Token) ，计算其绝对值的最大值，并以此为基准计算一个缩放因子，然后将该行的所有数值量化到INT4范围。

**Stage 2c: Quantized Attention.** 在实际计算时，Q 保持 FP16 精度。计算注意力时，从内存中加载量化后的 KV Cache，将其在线反量化回 FP16，然后执行点积等注意力计算。

## Experimental Validation

使用 Hugging Face 和 PyTorch 框架实现。评估主要基于 LLaMA-2 系列模型。**没有对比100B 以上模型 不知道是否是因为速度？**

量化方案:
- **输入激活值**: per-Token 对称量化。
- **KV Cache**: 每128通道为一组值的非对称量化。
- **权重**: per-channel 对称量化，使用 GPTQ  或 RTN (Round-to-Nearest). GPTQ 使用来自 WikiText-2 的128个样本进行校准。

性能评估在消费级 GPU **NVIDIA RTX 3090** 上进行  

| 实验类别 | 评估指标/内容 | 关键发现与数据 |
| :--- | :--- | :--- |
| **准确率结果 (Accuracy Results)** | **语言生成任务 (Perplexity on WikiText-2)** | - QuaRot 性能优于所有先前工作，如 SmoothQuant 和 OmniQuant. <br>- 4-bit 量化的 LLaMA2-70B 模型困惑度损失最多为 **0.47**. <br>- 在 LLaMA2-7B 模型上，性能优于 Atom. |
| | **零样本任务 (Zero-Shot Tasks)** | - LLaMA-2 家族的平均准确率损失最多为4.18%. <br>- LLaMA2-70B 模型保留了 **99%** 的零样本任务性能。 |
| **性能分析 (Performance Analysis)** | **预填充阶段加速 (Prefill Stage Speedup)** | - 所有性能实验均在单个 Transformer Block 上评估。<br>- LLaMA2-7B 获得 **1.97x-2.16x** 的速度提升。<br>- LLaMA2-70B 获得高达 **3.33x** 的速度提升。 |
| | **解码阶段内存节省 (Decoding Stage Memory Saving)** | - LLaMA2-7B 实现了高达 **3.75x** 的峰值内存节省。<br>- LLaMA2-70B 在几乎所有情况下都实现了 **3.89x** 的峰值内存节省 。 |
| **消融研究 (Ablation Studies)** | **RTN权重 量化** | - 在8-bit下，QuaRot 结合简单的 RTN 即可完全保持 FP16 的模型精度，且无需任何校准数据。<br>- 在4-bit下，模型越大，QuaRot-RTN 和 QuaRot-GPTQ 的差距越小。 |
| | **分组量化 (Group-wise Quantization)** | - 结果显示了准确率和分组大小之间的明确权衡: 更小的分组能带来更高的准确率。 |

## Source Code

QuaRot的流程可以分为两大核心部分: 
1. 离线处理: 加载预训练的FP16模型，对其权重进行旋转和量化，最后保存为QuaRot格式的检查点。
2. 在线推理: 加载QuaRot检查点，执行高效的 4-bit 前向传播，其中包含在线的激活值量化和哈达玛变换。

在对权重进行修改前，首先需要生成论文中提到的全局旋转矩阵 Q 和其他哈达玛矩阵。

```python
# e2e/checkpoint_utils/rotation_utils.py

import torch
from quarot.functional.hadamard import random_hadamard_matrix

def get_orthogonal_matrix(n, device, seed=42):
    """
    生成一个随机的正交矩阵，论文中主要使用的是哈达玛矩阵。
    对应论文 3.1 节 "Randomized Hadamard Matrices"。
    """
    return random_hadamard_matrix(n, device=device, seed=seed)

def fuse_layer_norm(layer, Ws):
    """
    对应论文 Stage 1a 的第一步: 融合 RMSNorm 的可学习缩放参数 alpha.
    将 RMSNorm 的 weight (即 alpha) 乘到后续的线性层权重上。
    """
    ...

def rotate_layer(layer, Q):
    """
    对单个 Transformer 层的权重应用全局旋转 Q.
    对应论文 Stage 1a 的操作。
    """
    # 融合 RMSNorm 的缩放参数
    fuse_layer_norm(layer.input_layernorm, [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj])
    # ... 对 MLP 的 RMSNorm 也进行融合

    # 对输入侧的权重进行左乘 Q.T
    # 公式: W_in' = Q.T @ W_in
    layer.self_attn.q_proj.weight.data = torch.matmul(Q.T, layer.self_attn.q_proj.weight.data.to(Q.dtype)).to(layer.self_attn.q_proj.weight.data.dtype)
    layer.self_attn.k_proj.weight.data = torch.matmul(Q.T, layer.self_attn.k_proj.weight.data.to(Q.dtype)).to(layer.self_attn.k_proj.weight.data.dtype)
    # ... 对 v_proj, gate_proj, up_proj 执行同样操作

    # 对输出侧的权重进行右乘 Q
    # 公式: W_out' = W_out @ Q
    layer.self_attn.o_proj.weight.data = torch.matmul(layer.self_attn.o_proj.weight.data.to(Q.dtype), Q).to(layer.self_attn.o_proj.weight.data.dtype)
    layer.mlp.down_proj.weight.data = torch.matmul(layer.mlp.down_proj.weight.data.to(Q.dtype), Q).to(layer.mlp.down_proj.weight.data.dtype)
```

对 Transformer block 中的模块的先行曾权重进行旋转并进行 GPTQ 量化。

```python
# e2e/checkpoint_utils/quantize_llama_checkpoint.py

# ... import statements ...
from .gptq_utils import gptq_quantize
from .rotation_utils import get_orthogonal_matrix, rotate_layer, rotate_attention, rotate_mlp

# ...

def process_and_quantize_model(model, args):
    # 1. 生成全局旋转矩阵 Q
    # 对应论文 Stage 1a
    Q = get_orthogonal_matrix(model.config.hidden_size, device, args.seed)

    for i, layer in enumerate(model.model.layers):
        # 2. 应用全局旋转 Q
        # 这一步修改了所有输入和输出侧的权重
        rotate_layer(layer, Q)

        # 3. 应用 Attention 内部的旋转
        # 对应论文 Stage 1c 和 1d
        # 这会修改 W_v 和 W_out 以处理 Value 向量
        rotate_attention(layer.self_attn, Q)

        # 4. 应用 FFN 内部的旋转
        # 对应论文 Stage 1b
        # 这会修改 W_down 以处理 FFN 的内部激活值
        rotate_mlp(layer.mlp, Q)

    # ...

    # 5. 使用 GPTQ 对修改后的权重进行量化
    # 对应论文 Stage 2a (Weight Quantization)
    gptq_quantize(model, args.wbits, args.abits, args.group_size, args.num_samples)

    return model
```

离线处理完成后，模型权重已经是旋转并量化过的了。在线推理时，模型需要执行在线的激活值量化和哈达玛变换。这部分逻辑主要在 `e2e/quantized_llama/modeling_llama.py` 中。

LlamaAttention 的 forward 方法
```python
# e2e/quantized_llama/modeling_llama.py

# ... inside LlamaAttention class ...
def forward(self, hidden_states, ...):
    # 输入 hidden_states 已经是被前一层旋转过的 X * Q

    # 1. Q, K, V 投影
    # W_q', W_k', W_v' 都是离线修改过的权重
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    # 此时 value_states 已经是被 H_head 旋转过的 V * H_head

    # ...

    # 2. 应用 RoPE
    query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

    # 3. 在线旋转 Q 和 K
    # 对应论文 Stage 1d 的在线部分
    # 公式: Q_rot = Q * H_head, K_rot = K * H_head
    query_states = quarot.hadamard_transform(query_states, self.head_dim, in_place=True)
    key_states = quarot.hadamard_transform(key_states, self.head_dim, in_place=True)

    # 4. KV 缓存
    # 存入缓存的是旋转后的 K 和 V
    # 对应论文 Stage 2c (Quantized Attention)
    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

    # ...

    # 5. 计算注意力
    attn_output = self.attn_op(query_states, key_states, value_states, attention_mask, ...)

    # 6. 在线跨头旋转
    # 对应论文 Stage 1c 的在线部分
    attn_output = quarot.hadamard_heads_transform(attn_output, self.num_heads, in_place=True)

    # 7. 输出投影
    # W_out' 是离线修改过的权重, 它会“解开”所有的旋转
    attn_output = self.o_proj(attn_output)

    return attn_output, ...
```

LlamaMLP 的 forward 方法

```python
# e2e/quantized_llama/modeling_llama.py

# ... inside LlamaMLP class ...
def forward(self, x):
    # 1. 上采样和 gate 投影
    # W_gate' 和 W_up' 是离线修改过的
    gate = self.gate_proj(x)
    up = self.up_proj(x)

    # 2. gate 激活
    # 对应论文图3中的 σ 和 ⊗ 符号
    gated_hidden = self.act_fn(gate) * up

    # 3. 在线 FFN 哈达玛变换
    # 对应论文 Stage 1b 的在线部分
    # 公式: H_rot = H_gated * H_ffn
    gated_hidden = quarot.hadamard_transform(gated_hidden)

    # 4. 下采样投影
    # W_down' 是离线修改过的, 它会“解开”旋转
    down = self.down_proj(gated_hidden)

    return down
```

上面代码中调用的 `self.q_proj`, `self.down_proj` 等都是 quarot.nn.W4A16Linear 的实例。它的 forward 方法处理了在线的输入量化。

```python
# quarot/nn/linear.py

from ..functional.quantization import quantize_per_token_absmax, dequantize_per_token_absmax, pack_int4
from ..functional.hadamard import hadamard_transform
# ...

class W4A16Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=False, hadamard=False):
        # ...
        self.hadamard = hadamard

    def forward(self, x):
        # 1. (可选) 在线哈达玛变换
        # 这个标志位被 LlamaMLP 的 down_proj 和 Attention 的 o_proj 设为 True
        if self.hadamard:
            x = hadamard_transform(x)

        # 2. 在线输入激活值量化
        # 对应论文 Stage 2b (Online Quantization Operations)
        # 采用每-Token对称量化(absmax)
        x_q, scales = quantize_per_token_absmax(x, self.wbits)

        # 3. 打包为 INT4 格式
        # 两个4-bit整数打包成一个8-bit整数
        x_packed = pack_int4(x_q)

        # 4. 调用 CUDA Core 执行 INT4 GEMM (矩阵乘法)
        # quarot_gemm(A, B, C) -> C = A @ B
        # A 是量化的输入, B 是量化的权重
        y = quarot_gemm(x_packed, self.qweight, self.scales, scales, self.qzeros)

        # ...
        return y
```

# ATOM W4A4

## Introduction

- Weight-only Quantization: 量化在推理的时候仍需要将权重反量化成 FP16 格式与 activation 相乘。
- W8A8: 仍无法完全利用计算速度更快的 INT4 Tensor Core.

ATOM 的三个关键算法设计
1. 使用 mixed-precision, 保留一部分显著的 act 和 weight 为 INT8 格式以维持准确性。
2. 对 act 和 weight 使用 group quantization 以减小量化误差。
3. 动态量化 act 以更好适应每个输入的分布。

ATOM 的硬件友好设计
1. 对 act 和 weight 进行 reorder 以获得更好的内存访问。
2. 融合 reorder 和量化算子。
3. 将 KV Cache 也量化成 INT4 以减少数据搬运量。

![Overview of the fused GEMM operator.](https://share.note.youdao.com/yws/api/personal/file/WEB8225b3babb285e59dbd6b9016f501b1c?method=download&shareKey=6e4192de8c876d5bfb976abfcf284f44 "Overview of the fused GEMM operator.")

## Performance Analysis of Low-bit LLM Serving

Decode 阶段，每次推理只输入一个 token 并生成下一个 token，这主要依赖于 GEMV，需要加载庞大的权重矩阵，但执行的乘法运算相对较少，导致计算强度低、GPU 利用率不足。为了缓解这一问题，通常采用批处理技术合并多个请求，将 GEMV 转化为 GEMM，从而提高计算强度和 GPU 利用率。

测量了不同批处理大小下 Llama-7b 推理的时间分布，发现主要瓶颈集中在两类层：
- Dense Layers:包括 K/Q/V/O projection和 MLP. 在大批处理规模下，会从内存受限转变为计算受限。
- Self-attention Layers: 由于不同请求不共享上下文历史，自注意力层无法像稠密层那样从批处理中获得同等的重用收益。即使经过优化，它仍然受限于 KV Cache 的巨大内存移动开销。

结论：低比特 weight-activation 量化的能同时加速 compute-bound 的 Dense Layers 和 memory-bound 的 Self-attention 层。

![Runtime breakdown of Llama-7b inference with different batch sizes.](https://share.note.youdao.com/yws/api/personal/file/WEB2334a264d21a71e0bfc6a8975c3976b6?method=download&shareKey=10b4d3610f32d6cdabcb4ca6acf7f2c8 "Runtime breakdown of Llama-7b inference with different batch sizes.")

## 4 Design

**Mixed-precision quantization**: 为了应对 act 中存在的离群值，Atom 采用混合精度方法，将离群值保持在 INT8，将占绝大多数的普通数值量化为 INT4.

Channel Reordering: 传统的混合精度计算会导致内存访问不连续，降低硬件效率。Atom 将零散分布的离群值通道动态地移动到矩阵末尾。

![Sampled value of an activation matrix from Llama-7b.](https://share.note.youdao.com/yws/api/personal/file/WEB012eab6d94165d9848f5604d72573af3?method=download&shareKey=71c1e920b1606495b577ea812f85fa4c "Sampled value of an activation matrix from Llama-7b.")

为了保证计算正确性，权重矩阵会根据离线校准得到的索引进行静态 reorder，操作被融合到前一个算子中，使得这一步骤的运行时间开销低于总时间的 0.5%.

![Atom dynamically reorders activation (A) to move the outlier channels to the end of the matrix, with the reorder indices determined in offline calibration.](https://share.note.youdao.com/yws/api/personal/file/WEB23698e7fd71831ee280556845dffd4f6?method=download&shareKey=6d2a75a171b6a66fbc3235647cf968ae "Atom dynamically reorders activation (A) to move the outlier channels to the end of the matrix, with the reorder indices determined in offline calibration.")

---
**Fine-grained Group Quantization**: 为了进一步提升 INT4 计算精确度，Atom 采用了分组量化 (Group Size=128). 所有这些步骤都融合在单个 MMA 管道中执行，避免了额外的显存读写开销。
1. 利用 INT4 Tensor Core 计算各组的中间乘加结果。
2. 使用 CUDA Cores 将这些中间结果反量化为 FP16 格式。
3. 将反量化后的结果进行累加得到最终输出。

![Overview of the fused GEMM operator.](https://share.note.youdao.com/yws/api/personal/file/WEB97d17b3230d236613150aec91e6bc56a?method=download&shareKey=22fdf7bf71c6dc71702089cd7d19f80b "Overview of the fused GEMM operator.")

---
**Dynamic Activation Quantization**: 
act 的分布随输入的变化而剧烈波动，静态预计算的量化参数往往无法提供最优精度。
Atom 在推理过程中实时统计当前激活矩阵的最大值和最小值，从而动态计算量化参数。

为了避免非对称量化带来的减掉 zero-point 计算负担 ，Atom 使用对称量化，并通过Grid Search 确定最佳的截断阈值 (act=0.9, weight=0.85)，以过滤噪声。

对于静态的 weight，Atom 整合了 GPTQ 算法，在离线阶段利用二阶导数信息进行量化。

---
**KV-cache Quantization**:
Atom 对 KV Cache 采用 INT4 非对称量化。由于这是内存受限场景，非对称量化带来的略微计算增加在显著的内存带宽节省面前是完全值得的。Atom 发现 KV 缓存对量化更不敏感：
- Value 相比普通激活值，其分布更均匀，离群值极少。
- Key 由于后续会经过 Softmax 操作，量化带来的微小误差会被归一化过程平滑掉。

---
Atom 将量化操作（包括 reorder、量化和反量化）整合到现有的操作中。对于计算密集型的 Dense layers，Atom 利用低精度单元来提高吞吐量。对于 memory-bound 的 Attention layers，Atom 将反量化和 FlashInfer 进行融合，从而只加载 KV Cache 中的较低位。Atom 还引入了 PageAttention 来提高内存使用效率，以实现更大的批处理规模。

![Overview of Atom workflow on Llama model family.](https://share.note.youdao.com/yws/api/personal/file/WEB756689a7198ad5c81534bc8e1af92494?method=download&shareKey=622edf1f8aa98dd275df7e62863b3f73 "Overview of Atom workflow on Llama model family.")
## 5 Evaluation

1. Set UP:
    - 量化配置：权重与激活采用对称量化。KV Cache 采用非对称量化。使用 group size=128 进行细粒度量化。
    - 异常值处理：从 WikiText2 数据集中随机采样 128 个句子作为校准数据。选择平方和最高的 128 个通道作为异常值通道，并以 INT8 格式保留。
    - 参数优化：通过网格搜索确定截断因子 clipping factors：激活量化为 0.9，权重量化为 0.85. 权重量化过程中采用了 GPTQ 算法。
    - 量化预处理在 RTX Ada 6000 上运行。对于最大的 Llama-65B 模型，整个量化过程大约需要 4 小时。

2. Accuracy Evaluation
    - Benchmark：评估了 Llama 全系列模型（7B-65B）在 W4A4 和 W3A3 配置下的表现。
    - Zero-shot Acc (Table 1): 与 FP16 相比，Atom 在 7B、13B、30B 和 65B 模型上的平均准确率下降分别仅为 2.3%、1.7%、0.4% 和 1.4%. 相比之下，之前的量化方法 (SmoothQuant、OmniQuant) 在相同设置下准确率下降高达 9.6% 至 23.8%.
    - Perplexity (Table 2): 在 Llama-65B 的 W4A4 量化中，三个数据集 (WikiText2, PTB, C4) 增加均低于 0.4. 在 W3A3 设置下，Atom Llama-65B 的平均困惑度仅增加 2.3，而现有方案往往无法达到可接受的精度。

3. Efficiency Evaluation 在 RTX 4090 (24GB) GPU 上进行。
    - GEMM: 在 BatchSize=512 时，Atom 的内核相比 FP16 实现了 3.4 倍的加速，相比 INT8 实现了 1.9 倍的加速。
    - Self Attention: 在 BatchSize=128 时，由于 KV Cache 量化减少了显存压力，比 INT8 快 1.8 倍、比 FP16 快 3.5 倍。
    - E2E: 在相同延迟目标下，相比 FP16 基准吞吐量提升高达 7.73 倍，相比 INT8 量化提升 2.53 倍。
    - Decoding Latency: 在最高性能的 BatchSize=64 下，Atom 的延迟甚至低于 FP16 在 BatchSize=8 时的表现。即使在 BatchSize=256 时，延迟仍保持在 100ms 以下。

4. 消融实验分析 (Ablation Study)
    - 精度贡献：将异常值通道保持在 INT8 仅比保持在 FP16 增加极微小的困惑度（+0.05），证明了 INT8 混合精度处理异常值的有效性 。
    - 吞吐量开销：纯 INT4 GEMM 吞吐量约为 980 TOPS. 融合混合精度（处理 128 个 INT8 通道）后吞吐量为 900 TOPS，开销约 8%. 融合分组反量化后性能降至 770 TOPS，但仍比 INT8 的理论极限高出约 18%.

# QServe W4A8KV4

## 1 Introduction

尽管低比特量化理论上能加速推理，但现有的 INT4 量化方法在云端大批量推理场景中往往无法带来实际增益，甚至产生 20%~90% 的运行时开销。研究团队发现其根本原因在于：现有的反量化操作 (W4A16 将权重, W4A4 为了保持性能使用 per-group 量化需要将 GEMM 过程中的部分和转换回浮点数) 必须在低吞吐量的 CUDA Core 上执行，而在 A100 等硬件上，CUDA Core 的操作成本比 INT4 Tensor Cores 高出约 50 倍。

针对这一问题，论文提出了 QoQ (W4A8KV4) 及其配套的 QServe 系统，其核心思路是
1. Progressive Group Quantization: 使用带有 *protective range* ([−119, 119]) 的 per-channel 量化确保所有 GEMM 计算都能在 INT8 Tensor Core 上运行
2. SmoothAttention: 将 Key 中的异常值的量化难度丢给 Query (类似 SmoothQuant) 以保持精度
3. Qserve: INT4->INT8 反量化中使用 register-level 并行，compute-aware weight reordering 来减小 W4A8 GEMM 运算中的指针计算开销。

## 3 Motivation

Decoding 阶段，计算强度主要由 Batch Size ($m$) 决定 2。
- $m < 78$: 系统属于 Memory-bound，此时 W4A16 因为权重更小、搬运带宽需求低，性能优于 W8A8.
- $m > 78$: 系统进入 Compute-bound. 此时 W8A8 能够利用吞吐量更高的 INT8 Tensor Core，性能反超 W4A16 (后者受限于计算效率较低的 FP16 Tensor Core).

W4A8 结合了两者的优点——它既拥有 INT4 权重的低显存占用，又能利用高效的 INT8 Tensor Core 进行计算。

![A100 roofline for LLM serving](https://share.note.youdao.com/yws/api/personal/file/WEB0ef17cf29c6ec0865d6d7d7b1a44efae?method=download&shareKey=5251dc902d81746b9c7b86b7875912fe "A100 roofline for LLM serving")

KV Cache 量化的必要性：Attention 本质上是 Memory-bound，其内存流量主要由 KV Cache 访问主导。采用 KV4 可以将有效显存带宽提升 2 倍，在 Batch Size 为 64 时，由于 Attention 占据了超过 50% 的运行时间，这种优化至关重要。

理论上 W4A4 比 W4A8 快一倍，但论文揭示由于 GEMM 主循环开销，W4A4 在现有 GPU（Ampere/Hopper）上反而更慢 。

核心瓶颈在于：
- CUDA Core 的效率鸿沟：W4A4 为了保证精度，通常需要进行 Per-group 量化。导致每一轮计算产生的 INT32 部分和必须在主循环内转换回浮点数，这一反量化过程只能在慢速的 CUDA Core 上执行。
    > 在 A100 上，FP32 CUDA 核心的性能仅为 INT4 Tensor Core 的 2%，处理一个转换操作相当于浪费了 50 次张量核心的 MAC 运算。
- 寄存器压力：W4A4 需要同时维护两套寄存器：一套存储 INT32 部分和，另一套存储 FP32 部分和。寄存器消耗限制了 SM 上能够同时运行的 Warp数量。GPU 无法通过低成本的上下文切换来隐藏延迟，进一步加剧了主循环的开销。

![Quantized GEMM on GPUs](https://share.note.youdao.com/yws/api/personal/file/WEB0e292c4c288550747f2b26b1f1883983?method=download&shareKey=f971f3302a131259f07299097da2977c "Quantized GEMM on GPUs")

## 4 QoQ

**Progressive Group Quantization**: 为了保证精度，通常需要使用Per-group Quantization，但传统的实现会在 GPU 计算的主循环中产生巨大的反量化开销。 QoQ 的两级量化流程
1. Channel-wise: 对 $W$ 使用 INT8 堆成量化。$$\hat{W} = Q_{W_{s8}}^{(0)} \cdot s_{fp16}^{(0)}$$ 
2. Group-wise: 对上述中间值再进行非对称的 INT4 分组量化。$$Q_{W_{s8}}^{(0)} = (Q_{W_{u4}} - z_{u4}) \cdot s_{u8}^{(1)}$$ 

为了确保 INT4 反量化回 INT8 时不会发生溢出（超出 [-128, 127] 范围），QoQ 将第一级的 INT8 范围收缩到了 [-119, 119]. 这样可以保证反量化后的中间值完美落在 8-bit 整数范围内，从而直接调用高效的 INT8 Tensor Cores 进行计算。

![Progressive Group Quantization](https://share.note.youdao.com/yws/api/personal/file/WEB31c7c55203e1343d831262a2e3199a4b?method=download&shareKey=4b6ee2707d4918de20a5d40414b37c96 "Progressive Group Quantization")

**SmoothAttention:** 针对 INT4 KV Cache 导致的精度大幅下降，论文提出了 SmoothAttention. 作者观察到 Key 在每个 head 中都存在固定 Outlier channels，这些通道的值比其他通道大约 10 倍，而 Value则没有这种模式。受 SmoothQuant 启发，SmoothAttention 通过一个缩放因子 $\lambda$ 来平滑 Key 缓存中的 outliers. $$\lambda_{i} = \max(|K_{i}|)^{\alpha}$$ 

> 在 KV Cache 量化中，由于 Query 并没有被压缩到 4-bit，所以算法可以把所有的“平滑压力”都推给 Key，从而简化了缩放因子的选择。

由于使用码 RoPE，改变通道值会破坏其数学特性。QoQ 施加了一个硬性约束，要求成对的通道（$i$ 与 $i + D/2$）共享相同的缩放因子，从而保证了变换的交换性。这些缩放因子最终被融合进前一个线性层的权重中（$W_Q$ 和 $W_K$），因此在推理阶段不会增加额外的算子调用开销。

除了上述两项核心创新，QoQ 还整合了四种通用优化手段来进一步压低误差:
- Block Input Module Rotation: 使用 Hadamard 矩阵旋转激活值，抑制激活中的离群点，将量化难度从激活转移到权重上。
![Rotate the block input activations to suppress the outliers](https://share.note.youdao.com/yws/api/personal/file/WEBd1724a1345feae6e52f078f89e72d110?method=download&shareKey=1c0ccf8b58ac25cf820bf50fc3501321 "Rotate the block input activations to suppress the outliers")
- Block Output Module Smoothing: 对产生块输出的层（如 OProj 和 FFNDown）进行平滑处理，与传统的 SmoothQuant 侧重点有所不同，主要由权重决定平滑因子。
![Smooth the block intermediate activations, migrating the quantization difficulty](https://share.note.youdao.com/yws/api/personal/file/WEB6248600255e362e47ef24e65d3141f4d?method=download&shareKey=5bb88302af4da79d3d6227f97ead4969 "Smooth the block intermediate activations, migrating the quantization difficulty")
- Activation-Aware Channel Reordering: 根据输入激活的幅度确定通道的 Salience，并对通道进行 reordering，使得 Salience 相近的通道被划分到同一个 group.
![Reorder weight input channels based on their salience in group quantization.](https://share.note.youdao.com/yws/api/personal/file/WEBccedc402d6e8a497c3667f2cc9182944?method=download&shareKey=8cf04d91a257042451c4276b79813e29 "Reorder weight input channels based on their salience in group quantization.")
- Weight Clipping: 通过网格搜索剪裁比例 $\alpha$，除 q_proj 和 k_proj 以外最小化层输出 MSE. q_proj 和 k_proj 的优化目标是最小化整个 Transformer Block 的最终输出 MSE.

## 5 QServe Serving System

**QServe System Runtime:** QServe 采用了一种混合精度映射方案，确保每一层都在最适合的硬件单元上运行 ：
- GEMM: 输入为 W4A8，利用 INT8 Tensor Cores 进行计算，输出为 FP16.
- Attention: 在 CUDA 核心 上以 FP16 精度执行。

![QServe's precision mapping](https://share.note.youdao.com/yws/api/personal/file/WEBc0ea2e23dc5d8e16d4084e4b24e7cdf6?method=download&shareKey=174f83dfcc580fcf0e42ff3286212d39 "QServe's precision mapping")

为了消除额外算子开销，QServe 将激活量化过程融合到了前一个 LayerNorm 或激活函数算子中。

借鉴了 vLLM 的 Paged KV Cache 思想，但实现了 Per-head dynamic quantization. Scale 和 Zero 会随 quantized 特征一起存储在 Page 中，支持在线更新。

---
**Compute-Aware Weight Reorder**: Tensor Core 在进行矩阵运算时，要求每个线程持有的数据必须遵循特定的步长布局。对于标准的 INT8 运算，GPU 提供了 ldmatrix 指令来自动处理这种分布。但在 W4A8 场景下，由于存储格式（4位）与计算格式（8位）不同，ldmatrix 指令会按字节而非元素分配数据，导致线程拿到的数据是错误的。

如果不使用 ldmatrix，程序必须在运行时手动计算每个数据的地址。这种指针运算在慢速的 CUDA 核心上执行，其吞吐量远低于张量核心，严重拖慢了主循环速度。 QServe 在离线阶段直接改变权重的物理存储顺序，使其完全匹配张量核心在计算过程中需要的逻辑顺序。

![QServe applies compute-aware weight reoder to minimize the pointer arithmetics](https://share.note.youdao.com/yws/api/personal/file/WEBee209e1a9f244bd240686544d685b923?method=download&shareKey=9b9e3a0a9fcb24e6a1896230f9cfda3a "QServe applies compute-aware weight reoder to minimize the pointer arithmetics")

Fast Dequantization: 优化 INT4 到 INT8/FP16 的转换逻辑，降低了反量化在寄存器和计算资源上的压力。包含两个关键技术点：
- Subtraction after Multiplication:  通常的反量化公式是 $(Q_W - Z_W) \cdot S_W$，这需要在主循环中频繁执行减法操作。QServe 将公式重写为 $Q_W \cdot S_W - Z_W \cdot S_W$。在 Per-channel 量化中，零点偏移项 ($Z_W \cdot S_W$) 可以通过数学变换，从主循环中完全剥离，并融合到 GEMM 的最后 Epilogue 中执行。这样主循环就只需要处理纯粹的乘法。
- Register-Level Parallelism, RLP: 对于 Per-group 必须在循环内处理的减法，QServe 利用了 NVIDIA GPU 的 vadd4 指令。该指令可以在单个 INT32 ALU 周期内同时完成 4 路 INT8 的加/减法运算。这一步能成功的关键在于 QoQ 算法限制的 INT8 量化范围 [-119, 119]. 由于算法保证了第一步乘法后的中间结果不会超过 INT8 的表示范围，因此可以安全地利用并行指令进行减法，而不会发生数据溢出导致的错误。

---
KV4 Attention in Qserve: Attention 算子主要由 CUDA Core 执行 2。在常规实现中，反量化一个 INT4 数值需要 5 个 ALU 操作，包括：掩码（Mask）、移位（Shift）、类型转换（从整数到浮点）、浮点乘法（Mul）以及减法（Sub）. A100 的 FP32 CUDA Core Roofline 拐点仅为 9.8 Ops/Byte. 这意味着单纯反量化操作本身产生的计算强度就已经超出了硬件的处理能力，导致注意力算子从 memory-bound 变成了 compute-bound. 2

为了消除这一瓶颈，QServe 采用了双向优化的思路:
- 引入位算力技巧（Bit Tricks）减少计算强度类似于 5.2.2 节中对权重的解 Unpacking 优化，QServe 在处理 KV 将反量化的计算密度从原本的 5 Ops 降低到每元素 2 Ops. 
- 精度平移：QServe 将原始内核中的 FP32 操作替换为 FP16 操作。

QServe 还配套了以下手段：
- Asynchronous Prefetching: 在注意力内核开始时，异步加载量化参数（缩放因子和零点），有效地隐藏了内存访问延迟。
- Control Flow Simplification: 简化内核内部的地址计算和控制逻辑，进一步释放 CUDA Core 的算力资源用于核心计算。