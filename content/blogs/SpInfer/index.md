---
title: "SpInfer"
date: 2025-10-01T09:57:17+08:00
lastmod: 2025-10-01T09:57:17+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- Purning
keywords:
- Purning

description: "Paper reading of SpInfer." # 文章描述，与搜索优化相关
summary: "Paper reading of SpInfer." # 文章简单描述，会展示在主页
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

非结构化剪枝 (Unstructuredd Purning) 在 LLM 推理中实现的难点:
1. 非零元素索引的存储开销。
2. 在低稀疏度级别 (大约50%) 下稀疏矩阵乘法 (SpMM) 内核的低效。

提出 GPU 上的 LLM 剪枝推理框架 SpInfer:
1. Tensor-Core-Aware Bitmap Encoding (TCA-BME) 利用位图来减小获取索引开销。
2. 带有 Shared Memory Bitmap Decoding (SMBD) 的 SpMM 内核和异步流水线设计增加计算效率。

# 1. Introduction

权重剪枝 (weight pruning) 方法主要分为三种:
1. 结构化剪枝 (Structured Pruning): 移除神经网络中的整个组件，例如整个神经元、通道或滤波器，从而保持模型的结构完整性。它通常需要昂贵的后训练 (post-training) 过程来恢复性能，因为移除的组件会显著改变模型架构。尽管如此，它在硬件加速方面更友好，但可能导致准确率下降较多。
2. 半结构化剪枝 (Semi-Structured Pruning): 在灵活性和效率之间取得平衡，通过控制稀疏度来实现部分结构化移除。例如，N:M表示在每组N个权重中保留M个最重要权重。它结合了结构化方法的硬件友好性和非结构化方法的灵活性，适用于实际部署场景。
3. 非结构化剪枝 (Unstructured Pruning): 自由移除单个权重，而不考虑整体结构，提供最大的灵活性。它通常在后训练性能上表现更好，准确率一般优于结构化方法，因为它能更精细地保留重要连接。但缺点是稀疏性不规则，可能需要特殊硬件支持来加速推理。

LLM 推理中实现低稀疏度的非结构化修剪有两个关键挑战: 
1. 存储非零元素索引的开销会和减少的权重相抵消。
2. GPU 上 SpMM 内核速度难以超越 CuBLAS.

![Figure 1. Execution time comparison of unstructured SpMM implementations against cuBLAS on Nvidia RTX4090 (M/K/N=28K/8K/16, typical in LLM inference).](https://share.note.youdao.com/yws/api/personal/file/WEB44e11414cda33a9edef922bf700d795b?method=download&shareKey=b28a426a8ef246b11b920eff7e4b69d4 "Figure 1.  Execution time comparison of unstructured SpMM implementations against cuBLAS on Nvidia RTX4090 (M/K/N=28K/8K/16, typical in LLM inference).")

TCA-BME 减小索引开销，SpMM 内核加速计算，30%-70% 稀疏度上内核和框架级别推理速度提升。

# 2. Related Work

GPU具有多个流多处理器 (SMs) ，具有CUDA核心，张量核心 (TCs) 和分层内存结构。线程块被调度到SMs上，在一个warp中有32个线程在SIMT模式下同时执行指令。

内存层次结构包括所有线程都可以访问的高延迟全局内存，每个SM中用于线程块访问的更快的共享内存，以及每个线程私有的快速但有限的寄存器。缓存系统包括每个SM一个L1缓存 (可配置共享内存) 和一个统一的L2缓存 (可优化处理核心和全局内存之间的带宽和延迟).

TensorCore 是加速密集矩阵乘法的组件。在我们的实现中，我们利用PTX级的低级 mma 指令，这在管理寄存器方面提供了更大的灵活性。对于FP16精度，mma指令要求矩阵形状为16 × 16 × 8 或 16 × 8 × 8.

# 3. Gaps and Opportunities

在 2xRTX4090 上使用 FasterTransformer 框架进行 OPT-13B 运行时间和内存分解，批量大小为16，输出长度为 256. 模型权重存储占用了 87.6% 的内存，而相关的 GEMM 消耗了 61.6% 的执行时间，构成了主要瓶颈。**但 LLM 剪枝的低稀疏性限制了当前方法在 GPU 上的实际有效性。**

![Figure 2. Breakdown of OPT-13B on 2 RTX4090 GPUs.](https://share.note.youdao.com/yws/api/personal/file/WEBe11bbb469dc950c67d07ae04649afc0b?method=download&shareKey=8d8aad537b6a25704bac4bebb169ec5a "Figure 2. Breakdown of OPT-13B on 2 RTX4090 GPUs.")

压缩率 (Compression Ratio, CR) 定义为单位压缩格式能存储的原始矩阵元素个数。

$$
CR=\frac{2B\times M\times K}{Stor_{Format}} \tag{1}
$$

![Figure 3. Compression Ratio (CR) across varying sparsity levels for different sparse matrix formats.](https://share.note.youdao.com/yws/api/personal/file/WEB85783922d6804de7f20e58c8ad337d53?method=download&shareKey=a66e2ded8c915c8199ee5397c40128c9 "Figure 3. Compression Ratio (CR) across varying sparsity levels for different sparse matrix formats.")

Tiled-CSL 使用两个矩阵以 tile 来存储非零元素: NonZeros (16-bit x 2 代表权重和位置) 和 TileOffsets (每个 tile 的起始索引)。

$$
Stor_{Tiled-CSL}=4B\times NT+4B\times NNZ,\tag (2)
$$

其中 NT 代表 Tile 个数，NNZ 代表非零元素个数。可以看到用于存储索引的开销和非零元素相当。

CSR 使用三个一维数组来表示一个 m × n 的稀疏矩阵: 
- values (val): 存储所有非零元素的数值，按行优先顺序排列。长度为 NNZ.
- col_indices (col): 存储每个非零元素的列索引，与 values 对应，长度为 NNZ.
- row_ptr (row): 行指针数组，长度为 m+1. row[i] 表示第 i 行的非零元素在 values 中的起始索引.

$$
Stor_{CSR}=(2B+4B)\times NNZ+4B\times(M+1).\tag(3)
$$

CSR 用 32-bit 存储行指针导致开销很大。

SparTA 将矩阵划分为多个 2x2. 使用 2-bit 索引来存储非零元素 ≤ 2 的块中的元素索引；使用 CSR 存储非零元素 > 2 的块中的元素索引。
整个张量中，需要用 CSR 格式存储的预期非零元素数量。整个张量中，需要用 CSR 格式存储的非零元素数量的期望为

$$
E_{CSR_{nnz}} = \left( \frac{M \times K}{4} \right) \times \left[ 4 \times (1 - s)^3 \times s + 2 \times (1 - s)^4 \right] \tag{4}
$$
其中 s 为稀疏度 (零元素的比例). 假设元素独立，块内 nnz 服从二项分布 Binomial(4, 1-s).

总块数为 $ \frac{M \times K}{4} $. SparTA 的 CSR 只处理 nnz >2 的情况: 
- nnz = 3 的概率: $ \binom{4}{3} (1-s)^3 s = 4 (1-s)^3 s $. 对于 nnz=3，多余 nnz = 1 (只存第 3 个) 。
- nnz = 4 的概率: $ \binom{4}{4} (1-s)^4 s^0 = (1-s)^4 $. 对于 nnz=4，多余 nnz = 2 (存第 3 和第 4 个) 。

SparTA 整个张量的总存储开销

$$
Stor_{SparTA} = \left(2B + \frac{B}{4}\right) \times \frac{M \times K}{2} + Stor_{CSR}(E_{CSR_{nnz}})
$$

简单块索引那部分没太看懂。

计算强度定义为每次访问内存能执行的浮点运算次数。对于 GEMM

$$
CI_{\text{GEMM}}=\frac{2M\times N \times K}{MK+NK}=\frac{M\times N}{M+N}.
$$

对于 SpMM，计算强度受压缩率影响，反应为权重存储的大小。

$$
CI_{\mathrm{SpMM}}=\frac{M\times N}{\frac M{\mathrm{CR}}+N}. \tag{7}
$$

如果能完全消除索引开销，理想情况下 SpMM 的计算强度可达到

$$
CI_{\mathrm{Optimal}}=\frac{M\times N}{M\times(1-s)+N}, \tag{8}
$$

图 4 的 roofline 模型反映出 GeMM 和 SpMM 都是内存受限的运算。因此压缩率越高，计算强度越大。

![Figure 4. Roofline comparison of various SpMM implementations against GEMM at varying sparsities and batch sizes.](https://share.note.youdao.com/yws/api/personal/file/WEB50ac12d655fafb8ee91ad34298c34af1?method=download&shareKey=efdfc438c216505b96dc27400bb68f2c "Figure 4. Roofline comparison of various SpMM implementations against GEMM at varying sparsities and batch sizes.")

# 4. Design of SpInfer

![Figure 5. System overview of SpInfer.](https://share.note.youdao.com/yws/api/personal/file/WEB86b666c1b5a37c7568c0a40525931d18?method=download&shareKey=4493412b65ba00ace73bd829ffea75b2 "Figure 5. System overview of SpInfer.")

TCA-BME 采用多级 tile 设计，将权重矩阵划分为不同粒度的 tile，以适应不同层次的硬件。如图6所示，该设计包含三个关键抽象级别: 
- BitmapTile (BT): $BT_H × BT_W$ 设置成 8 × 8. 对应于 Tensor Core 中最小的计算单元，即一个 8 x 8 的矩阵块。维度设置成这个大小的另外一个原因是可以利用 CUDA 原生的 uint_64 格式作为 64-bit 位图。
- TCTile (TT): $BT_H × BT_W$ 设置成 2 × 2. 对应于 Tensor core PTX 级别 mma 指令的一个 16 x 16 的矩阵块。FP16 精度有 2 个相关的 PTX 指令 mma.m16n8k8 and mma.m16n8k16. 由于实验表明大矩阵有着更高的吞吐，因此优化针对于 mma.m16n8k16 指令。在TCTiles 中，2×2 Bitmap Tile 以列主序排列，与 mma 指令中四个Ra寄存器的顺序一致。具体来说，左上，左下，右上和右下方的 BitmapTile 分别对应 Ra0-Ra3.
- GroupTile (GT): 维度为 $GTH × GTW$，包含多个 TCTile，对应于线程块级别。以行主序存储。

![Figure 6. Tensor-Core-Aware Bitmap Encoding. BitmapTile is actually 8×8, shown as 4×4 for illustration.](https://share.note.youdao.com/yws/api/personal/file/WEBc4e659adf88d9471549b35a8062b7064?method=download&shareKey=8d42e47d2b4be3ef4656acbcb031ec40 "Figure 6. Tensor-Core-Aware Bitmap Encoding. BitmapTile is actually 8×8, shown as 4×4 for illustration.")

TCA-BME 格式采用三个数组有效地表示稀疏权矩阵。
- GTileOffset: 记录稀疏矩阵中每个 GroupTile 的起始偏移位置。使用 32 位整数 (4B) 表示偏移量，其大小为 $4B × (NGT + 1)$，包括标记最后一个 GroupTile 结束的附加元素。
- Values: 存储所有非零元素，以 GroupTile, TCTiles 和 BitmapTile 的嵌套顺序排列。使用 FP16 格式存储每个值，大小为 $2B × NNZ$.
- Bitmap: 包含所有 BitmapTile 的位图值，每个 BitmapTile 由 64 位整数表示，其中每个位表示相应的元素是否为非零。元素个数 $NBT = (M/BT_H)\times(N/BT_W)$，大小为 $8B \times NBT$. 

因此 TCA-BME 格式的总开销为
$$
Stor_{TCA-BME}=4B\times(NGT+1)+8B\times NBT+2B\times NNZ.\tag{9}
$$

SpMM 内核工作流程如图 7 所示:
1. GTile 加载: 块内的线程协作将全局内存中的 GTile 加载到共享内存中的 WTile 中。使用 LDGSTS.128 异步向量化访存指令来提高内存带宽利用率，因此 GTile 中 Value 数组的起始地址被填充到 8Byte 对齐。
2. WTile 解码: 通过一种称为共享内存位图解码 (SBMD) 的技术将 WTile 从共享内存解码到寄存器。这一步将稀疏矩阵的紧凑位图表示转换为寄存器文件中的正确分布，为张量核心计算做好准备，所有这些都在高速寄存器文件中。使用原始的 LDS 指令。
3. XTile 加载: 从全局内存加载密集输入矩阵 XT 的对应 XTile 到共享内存中。
4. XTile 寄存器传输: 使用 LDSM.M88 将 XTile 数据从共享内存传输到寄存器，并安排 TC 计算。
5. Tensor Core 计算: 执行解码的稀疏 WTile 和密集 XTile 矩阵乘法，两者现在都驻留在寄存器中。

![Figure 7. Data movement and instruction pipeline.](https://share.note.youdao.com/yws/api/personal/file/WEB3b4cc1bbc6e6a2d20e49802640fc25aa?method=download&shareKey=cd3914b639bf52abfa562220af39c5d9 "Figure 7. Data movement and instruction pipeline.")

一个 wrap 每个线程在每个 32 位寄存器 (.f16x2) 中存储两个半精度值。在每个线程中需要四个这样的寄存器 (Ra0、Ra1、Ra2和Ra3) 来存储整个片段。这些寄存器是通过位图解码填充的，位图解码从压缩格式中提取非零值。

Shared Memory Bitmap Decoding (SMBD) 使用的是一个二阶段解码算法。为了避免内存浪费，没有为每个位置预存显式偏移，而是通过动态计算偏移来加载正确的值，依赖两个位计数操作: 
- PopCount: 利用 GPU 的内置指令 __popcll 计算整个 64-bit 位图中 1 的个数 (即非零位置数).
- Masked PopCount: 每个线程计算其 lane 之前的 1 的个数。

通过累加 PopCount 得到 TCTile 级别起始偏移，通过 Masked PopCount 得到线程级 (lane内) 偏移。

第一阶段，每个线程在其32位寄存器中解码第一个半精度值 (a0). ID为i的线程检查位图的第 2i 位。如果此位设置为1，线程使用MaskedPopCount来计算在其位置之前存在多少非零值，并从压缩值数组中加载相应的值。如果该位为0，则线程将一个零值加载到寄存器中。

第二阶段，每个线程从同一个32位寄存器解码第二个半精度值 (a1). ID为i的线程检查位图的第 2i+1 位，以确定该位置是否存在非零值。然而，在第二阶段不需要额外的MaskedPopCount。阶段1的结果被重用。具体来说，如果第一个值 (a0) 非零，则偏移量增加1以加载第二个值 (a1).

![Figure 8. Shared Memory Bitmap Decoding. (a) Register distribution of Tensor Core mma instruction. (b) PopCount and online offset calculation. (c) The two-phase bitmap decoding process.](https://share.note.youdao.com/yws/api/personal/file/WEB20adb63f053f493d1cb80bb22897ffe1?method=download&shareKey=838cb244151f050b0b844e2c86699276 "Figure 8. Shared Memory Bitmap Decoding. (a) Register distribution of Tensor Core mma instruction. (b) PopCount and online offset calculation. (c) The two-phase bitmap decoding process.")

算法流程

1. 计算 Tile 索引 (Batch ID, TileY, TileX):  
    - BatchID = blockIdx.y / (M/TILE_M): 计算批次 ID (沿行维 M 分块) 。
    - TileY = blockIdx.y % (M/TILE_M): 当前 Tile 在 Y 方向的索引。
    - TileX = blockIdx.x: 当前 Tile 在 X 方向的索引。
    这些用于确定当前线程块 (block) 负责计算的子矩阵块。

2. 计算迭代次数 (NumIter): `NumIter = CalculateBuffer(max(nnz per tile, Split_K))`: 基于稀疏矩阵每 Tile 的非零元素数 (nnz) 和分裂 K 计算总迭代次数。K 维被分成多个迭代处理。

3. 分配共享内存缓冲区 (标记为 > Sparse buffer 和 > Double buffer): 使用双缓冲。一个缓冲用于当前计算，另一个用于预取下一个迭代的数据。
    - `shared ValueBuffer[2][TILE_NN * TILE_K]`: 双缓冲的稀疏值缓冲 (存储 W 的非零值). 
    - `shared BitMapBuffer[2][TILE_M * TILE_K]`: 双缓冲的位图缓冲 (存储 W 的稀疏模式位图). 
    - `shared XTileBuffer[2][TILE_M * TILE_K]`: 双缓冲的密集 X Tile 缓冲。

4. 前循环初始化 (Pre-loop Initialization) (标记为 > Commit for dense): 
    - `LoadBitmapAndSparse(BitmapBuffer, ValueBuffer, W)`: 从全局内存加载位图和稀疏值到共享内存 (初始迭代).
    - `cp.async.commit()`: 异步提交这些加载操作。
    - `LoadDenseToShared(XTileBuffer, X + BatchID * TILE_K)`: 异步加载密集 X 数据到共享内存。
    - `cp.async.commit()`: 提交异步加载。
    - `Wfrag = SharedMemoryBitmapDecoding(ValueBuffer, BitmapBuffer)`: 使用共享内存解码位图，生成当前迭代的稀疏 W 片段 (Wfrag).

5. 主计算循环 (Main Loop) (for k = 0 to NumIter - 2 step 2，标记为 > K-dim iterations): 
这个循环处理 K 维的前 (NumIter - 2) 个迭代 (步长 2，确保双缓冲对齐) 。每步 2 个迭代是为了充分利用双缓冲 (一个用于奇数迭代，一个用于偶数).
    1. 预取下一个迭代的数据 (隐藏延迟): 
    - `LoadBitmapAndSparse(BitmapBuffer, ValueBuffer, W + offset)`: 异步加载下一个迭代的位图和稀疏值 (offset 基于 k).
    - `cp.async.commit()` (> Commit for bitmap/sparse): 提交位图/稀疏加载。
    - `LoadDenseToShared(XTileBuffer, X + offset)`: 异步加载下一个密集 X Tile。
    - `cp.async.commit() (> Commit for dense)`: 提交密集加载。
    2. 当前迭代计算: 
    - `Xfrag = LoadDenseToRegisters(XTileBuffer)`: 从共享内存加载当前 X Tile 到寄存器 (快速访问).
    `Yfrag = TensorCoreCompute(Wfrag, Xfrag, Yfrag)`: 使用 Tensor Core 硬件加速计算稀疏矩阵乘法片段 (Yfrag 是累加结果).

    3. 准备下一个迭代的稀疏数据: 
    - `cp.async.wait_group(1)` (> Wait for bitmap/sparse): 等待位图/稀疏加载完成。
    - `Wfrag = SharedMemoryBitmapDecoding(ValueBuffer, BitmapBuffer)`: 解码下一个 W 片段。
    - `cp.async.wait_group(0)` (> Wait for dense): 等待密集 X 加载完成。
    - `__syncthreads()`: 线程同步，确保所有线程完成当前步骤。

6. 结尾处理 (Epilogue) (标记为 > Epilogue): 处理最后 1-2 个迭代 (主循环跳过了最后的部分，以避免越界) 。
    - `Xfrag = LoadDenseToRegisters(XTileBuffer)`: 加载最终 X Tile 到寄存器。
    - `Yfrag = TensorCoreCompute(Wfrag, Xfrag, Yfrag)`: 执行最终 Tensor Core 计算。
    - `StoreResults(ReductionWorkspace, Yfrag)`: 将结果 Yfrag 存储到 Reduction Workspace (用于跨 Tile 累加).

![Algorithm 1 SpInfer-SpMM kernel pseudo code](https://share.note.youdao.com/yws/api/personal/file/WEB42e168285a9846bf90799dbc6a78b395?method=download&shareKey=0474e8a183b328f03e526d91f4116228 "Algorithm 1 SpInfer-SpMM kernel pseudo code")