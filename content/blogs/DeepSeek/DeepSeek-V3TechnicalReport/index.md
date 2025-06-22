---
title: "DeepSeek-V3 Technical Report"
date: 2025-06-20T16:37:06+08:00
lastmod: 2025-06-20T16:37:06+08:00
author: ["WITHER"]

categories:
- Paper Reading

tags:
- DeepSeek

keywords:
- DeepSeek

description: "Paper Reading of DeepSeek-V3 Technical Report" # 文章描述，与搜索优化相关
summary: "Paper Reading of DeepSeek-V3 Technical Report" # 文章简单描述，会展示在主页
weight: # 输入1可以顶置文章，用来给文章展示排序，不填就默认按时间排序
slug: ""
draft: false # 是否为草稿
comments: true
showToc: true # 显示目录
TocOpen: True # 自动展开目录
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

DeepSeek-V3 (671B) 是 MoE 模型，每个 token 会激活 37B 的参数。采用 Multi-head Latent Attention (MLA) 和自创的 DeepSeek MoE 结构，在这两篇文章中已经做过讲解。同时采用了 auxiliary-loss-free 策略来实现专家负载平衡并且使用了 Multi-token Prediction (MTP) 来加速训练。整个预训练数据集一共有 14.8T tokens，通过 Suprvised Fine-Tuning (SFT) 和 强化学习来加强性能。训练时长为 2.788M H800 GPU 小时。

# 1. Introduction

**模型架构创新:**
- Multi-head Latent Attention (MLA): 加速推理。
- DeepSeek MoE: 减少训练开销。

**增强模型能力的策略:**
- auxiliary-loss-free: 实现负载平衡。
- Multi-token Prediction (MTP): 增强模型表现。

**提高训练效率的方法:**
- FP8 混合精度训练: 加速训练和减少 GPU 内存使用。
- DualPipe 流水线并行算法: 减少气泡并且在训练时候通过计算掩盖了大部分通信。
- 新的节点间 All-to-all 通信算子: 更好地利用 InfiniBand (IB) and NVLink 带宽。
- 优化了内存后 DeepSeek-V3 训练没有使用 TP.

**训练过程:**
- pre-training: 在 14.8T tokens 上进行。
- stage 1: 扩展最大上下文长度到 32K.
- stage 2: 扩展最大上下文长度到 128K.
- post-training: 使用 Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) 并且蒸馏了 DeepSeek-R1 系列模型来获得推理能力。

DeepSeek-V3 上训练每 1T token 只需要180K H800 GPU小时，即在 2048 个 H800 GPU 的集群上需要 3.7 天。

# 2. Architecture

![Illustration of the Basic Architecture of DeepSeek-V3](https://share.note.youdao.com/yws/api/personal/file/WEB4af4bca65c0e55a9290c2e4d808cb6b2?method=download&shareKey=d8c9ce5c9b545f9d954d769f4e520fed "Illustration of the Basic Architecture of DeepSeek-V3")

## 2.1 MLA

在相关文章中已经介绍。

## 2.2 DeepSeekMoE with Auxiliary-Loss-Free Load Balancing

$$
\begin{align}
\mathbf{h}'_t &= \mathbf{u}_t + \sum_{i=1}^{N_s} \text{FFN}_i^{(s)} (\mathbf{u}_t) + \sum_{i=1}^{N_r} g_{i,t} \text{FFN}_i^{(r)} (\mathbf{u}_t),  \\
g_{i,t} &= \frac{g'_{i,t}}{\sum_{j=1}^{N_r} g'_{j,t}},  \\
g'_{i,t} &= \begin{cases} s_{i,t}, & s_{i,t} \in \text{Topk}(\{s_{j,t}|1 \le j \le N_r\}, K_r), \\ 0, & \text{otherwise}, \end{cases} \\
s_{i,t} &= \text{Sigmoid}(\mathbf{u}_t^T \mathbf{e}_i),
\end{align}
$$

**Basic Architecture of DeepSeekMoE.** 在相关文章中已经介绍，V3 和 V2 不同之处在于使用sigmoid函数来计算亲和分数，并在归一化所有选定的亲和分数之间来产生门控制值。

**Auxiliary-Loss-Free Load Balancing.** 为每个专家引入一个偏差项 $b_i$，并将其与相应的亲和力分数$s_{i,t}$ 相加，以确定 top-K 路由

$$
g'_{i,t} = \begin{cases} s_{i,t}, & s_{i,t} + b_i \in \text{Topk}(\{s_{j,t} + b_j | 1 \le j \le N_r\}, K_r), \\ 0, & \text{otherwise}. \end{cases} \tag{16}
$$

**这个偏置项仅用于路由**，用于和 FFN 输出相乘的门控值还是来自于原先的原先的 $s_{i,t}$. 在每一步结束时，如果其对应的专家过载， DeepSeek-V3 将偏差项减少 $\gamma$ (一个超参数，被称作 bias update speed)，如果其对应的专家负载不足， DeepSeek-V3 将偏差项增加 $\gamma$.

V3 使用 sequence-wise balance loss，类似于 V2 中 Expert-Level Balance Loss。 不同之处在于使用归一化的亲和分数。

$$
\begin{align}
\mathcal{L}_{\text{Bal}} &= \alpha \sum_{i=1}^{N_r} f_i P_i, \\
f_i &= \frac{N_r}{K_r T} \sum_{t=1}^{T} 1 (s_{i,t} \in \text{Topk}(\{s_{j,t}|1 \le j \le N_r\}, K_r)),  \\
s'_{i,t} &= \frac{s_{i,t}}{\sum_{j=1}^{N_r} s_{j,t}},  \\
P_i &= \frac{1}{T} \sum_{t=1}^{T} s'_{i,t} 
\end{align}
$$

**Node-Limited Routing.** 对于每个 token 计算每个节点计算亲和度分数前 $\frac {K_r}M$ 的专家求和，选取前 $M$ 个作为路由节点。

**No Token-Dropping.** 训练和推理中均不采用。Token Dropping指的是在 MoE 当路由到某个专家的 Token 数量超过了该专家的处理容量时，系统会故意丢跳过那些超出容量的 token，不让它们被这个专家处理。这些 token 通常会通过一个残差连接，将其输入时的状态直接传递到下一层。

## 2.3 Multi-Token Prediction

![Illustration of Multi-Token Prediction (MTP) implementation](https://share.note.youdao.com/yws/api/personal/file/WEBcc8ba55fb1b39401e8287ae38a50d829?method=download&shareKey=3fd271dd7007a3fc5b6b0939e165869f "Illustration of Multi-Token Prediction (MTP) implementation")

与 Gloeckle 等人使用独立的输出头并行预测 D 个额外 token 不同， DeepSeek-V3 顺序预测额外 token 并在每个预测深度保持完整的因果链。

**MTP Modules.** DeepSeek-V3 使用 D 个顺序模块来预测 D 个额外 token。第 k 个 MTP 模块由一个共享嵌入层 Emb(·)、一个共享输出头 OutHead(·)、一个 Transformer 块 TRM_k(·) 和一个投影矩阵 $M_k \in \mathbb{R}^{d \times 2d}$ 组成。对于第 *i* 个输入 token  $t_i$，在第 *k* 个预测深度， DeepSeek-V3 首先通过线性映射**结合第 *i* 个 token 在第 (k−1) 个深度上的表示 $h_i^{k-1} \in \mathbb{R}^d$ 和第 (i+k) 个 token 的嵌入 $\text{Emb}(t_{i+k}) \in \mathbb{R}^d$**

$$
h_t^k = M_k[\text{RMSNorm}(h_t^{k-1}); \text{RMSNorm}(\text{Emb}(t_{i+k}))], \tag{21}
$$

其中 [·;·] 表示拼接操作。特别地，当 k = 1 时，$h_t^{k-1}$ 指的是由主模型给出的表示。请注意，对于每个 MTP 模块，其嵌入层与主模型共享。合并后的 $h_t^k$ 作为第 k 个深度上 Transformer 块的输入，以在当前深度生成输出表示 $h_t^k$:
$$
h_{1:T-k}^k = \text{TRM}_k(h_{1:T-k}^k), \tag{22}
$$
其中 T 代表输入序列长度，而 $_{1:T-k}$ 表示切片操作 (包含左右边界)。最后，将 $h_T^k$ 作为输入，共享输出头将计算第 k 个额外预测 token 的概率分布 $p_{t+k+1}^k \in \mathbb{R}^V$，其中 $V$ 是词汇表的大小:
$$
p_{t+k+1}^k = \text{OutHead}(h_T^k). \tag{23}
$$
输出头 OutHead(·) 将表示线性映射到 logits，随后应用 Softmax 函数来计算第 k 个额外 token 的预测概率。此外，对于每个 MTP 模块，其输出头与主模型共享。DeepSeek-V3 维持预测因果链的原则与 EAGLE (Li et al., 2024b) 的原则相似，但其主要目标是推测解码 (Leviathan et al., 2023; Xia et al., 2023)，DeepSeek-V3 而 利用 MTP 来改进训练。

**MTP Training Objective.** 对于每个预测深度，计算一个交叉熵损失 $\mathcal{L}_{\text{MTP}}^k$：

$$
\mathcal{L}_{\text{MTP}}^k = \text{CrossEntropy}(P_{2+k:T+1}^k, t_{2+k:T+1}) = -\frac{1}{T}\sum_{i=2+k}^{T+1} \log p_i^k[t_i], \tag{24}
$$

- $T$: 输入序列长度
- $t_i$: 第 i 个位置的真实 (ground-truth) token
- $p_i^k[t_i]$: 代表第 $k$ 个 MTP 模块对于第 $i$ 个位置的预测中，赋给真实正确 token $t_i$** 的概率。

最后计算所有深度的 MTP 损失的平均值，并乘以一个加权因子 $\lambda$ 来获得总体的 MTP 损失 $\mathcal{L}_{\text{MTP}}$，作为 DeepSeek-V3 的一个额外训练目标：

$$
\mathcal{L}_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{\text{MTP}}^k. \tag{25}
$$

**MTP in Inference.** MTP 策略主要旨在提升主模型的性能，因此在推理过程中可以直接丢弃 MTP 模块，主模型可以独立且正常地运作。此外，也可以重新利用这些 MTP 模块进行推测解码 (speculative decoding) ，以进一步改善生成延迟。

# 3. Infrastructure
## 3.1 Compute Clusters

DeepSeek-V3 在 2048 NVIDIA H800 GPU 组成的集群上训练。每个节点有 8 张通过 NVLink 和 NVSwitch 连接的 H800. 节点之间通过 InfiniBand (IB) 连接。

## 3.2 Training Framework

DeepSeek-V3 使用 16-way Pipeline Parallelism (PP), 横跨 8 个节点间的 64-way Expert Parallelism (EP) 以及 ZeRO-1 Data Parallelism (DP). 训练期间不使用 Tensor Parallelism (TP).

### 3.2.1  DualPipe and Computation-Communication Overlap

DeepSeek-V3 中专家并行导致的跨节点 All-to-all 通信所对应的计算通信比接近 1:1，效率很低。

DualPipe 的核心思想是在一对独立的 forward & backword chunk 内部重叠计算和通信。具体来说，将每个 chunk 分为四个部分: attention, all-to-all dispatch， MLP 和 all-to-all combine. 特别地，对于 backword chunk, attention 和 MLP 都像在 ZeroBubble (Qi et al., 2023b) 中一样，被进一步拆分为两个部分：针对输入的反向传播和针对权重的反向传播。此外，还有一个流水线并行通信组件。如下图所示，对于一对 forward & backword chunk，重排这些组件，并手动调整专用于通信与计算的 GPU SM 的比例。通过这种重叠策略，可以确保 all-to-all 和 PP 通信在执行期间都能够被完全隐藏。

![Overlapping Strategy for a Pair of Individual Forward and Backward Chunks](https://share.note.youdao.com/yws/api/personal/file/WEBf348c55ccc87c5e4e388f2df2f18fb76?method=download&shareKey=849010c74a6772b18fff8ca8d5550e8c "Overlapping Strategy for a Pair of Individual Forward and Backward Chunks")

基于这种高效的重叠策略，完整的 DualPipe 调度方案如下图所示。它采用了一种双向流水线调度，即同时从流水线的两端送入 micro-batches，从而使得一大部分通信可以被完全重叠。这种重叠还确保随着模型规模的进一步扩大，只要保持恒定的计算与通信比率，仍然可以在节点间使用细粒度的专家 (fine-grained experts)，同时实现接近于零的all-to-all通信开销。具体的分析见相关文章。

![DualPipe Schedule](https://share.note.youdao.com/yws/api/personal/file/WEB6ab2cfcee4bad0937e6f0c4c0d225598?method=download&shareKey=754ad9e0092a41faf2c692912283f5ff "DualPipe Schedule")



### 3.2.2 Efficient Implementation of Cross-Node All-to-All Communication

DeepSeek-V3 定制了高效的跨节点 All-to-all 通信内核，以节省专用于通信的 SM 数量。内核的实现与MoE门控算法和 DeepSeek-V3 集群的网络拓扑共同设计。集群中跨节点 GPU 通过 IB(50 GB/s) 全连接，节点内通信通过 NVLink(160GB/s) 处理。为了有效地利用 IB 和 NVLink 的不同带宽，每个 token 限制最多被 dispatch 到 4 个节点以减少 IB 流量。

经过测试每个 token 在每个节点平均选择 3.2 个专家的同时不会产生额外的 NVLink 通信开销。意味着虽然 DeepSeek-V3 虽然实际上只选择 8 个路由专家，但它可以在保持相同通信成本的情况下最多选择 13 个专家 (4 节点x 3.2 专家/节点). 在这种通信策略下，仅 20 个 SMs 就足以充分利用 IB 和 NVLink 的带宽。详细地说，DeepSeek-V3 采用了 warp specialization 技术，并将 20 个 SMs 划分为 10 个通信通道。在 dispatch 过程中的通信链路为 (1)IB发送，(2) IB-to-NVLink 转发，(3) NVLink 接收由各自的 warp 处理。combine 过程则是相反的通信链路。

### 3.2.3 Extremely Memory Saving with Minimal Overhead

DeepSeek-V3 采取了如下技术来减少训练过程中的内存占用。

- 重计算 RMSNorm 和 MLA 升维投影。
- Exponential Moving Average (EMA) 参数被存放在 CPU 中并且异步更新。
- MTP 的 Embedding 和输出头在 PP rank 相同的设备上是共享的。

## 3.3 FP8 Training

低精度计算在累加过程中容易出现的问题有:
1. 溢出 (Overflow): 当许多数字相加时，它们的和很容易会超出 FP8 格式所能表示的最大值。
2. 精度损失 (Precision Loss/Underflow): 在累加过程中，如果一个很大的中间和与一个很小的乘积相加，这个很小的乘积可能会因为精度限制而被吞掉，直接变成零，对最终结果毫无贡献。

DeepSeek-V3 引入了一种细粒度的量化策略: $1\times N_c$ 元素的 tile 分组或 $N_cN_c\times N_c$ 元素的 block 分组。并且在其设计的高精度累加过程过程中，相关的反量化开销在很大程度上得到了缓解。此外，为了进一步减少 MoE 训练中的内存和通信开销，DeepSeek-V3 用 FP8 格式缓存和 dispatch 激活，以 BF16 格式存储低精度优化器状态。相较于 BF16 baseline, FP8 训练的相对误差低于 0.25%.

### 3.3.1 Mixed Precision Framework

如图中所示 Fprop(forward pass), Dgrad(activation backward pass) 以及 Wgrad(weight backward pass) GEMM 操作的输入是 FP8 格式，输出为 BF16 或者 FP32 格式。以 FP8 格式进行 Wgrad 允许激活也以 FP8 格式进行存储，减少了内存占用。

![The Overall Mixed Precision Framework with FP8 Data Format](https://share.note.youdao.com/yws/api/personal/file/WEBa301c0983af1cc29da1165ca160c0d3e?method=download&shareKey=92f1e462ac7dc2e7fd40cb3dff511153 "The Overall Mixed Precision Framework with FP8 Data Format")

一些低开销的算子可以使用更高精度并且对训练开销的影响可以忽略不计。DeepSeek-V3 对这些模块使用原格式进行运算：Embedding，输出头，MoE 门控，归一化操作以及 Attention 操作。同时为了数值稳定性，以更高精度存储 master weights(FP32), weight gradients(FP32) & optimizer states(BF16). 这些高精度部分带来的内存开销可以被 DP 减轻。

### 3.3.2 Improved Precision from Quantization and Multiplication

DeepSeek-V3 使用了如下技术来提高低精度训练的准确性:

> As a standard practice, the input distribution is aligned to the representable range of the FP8 format by scaling the maximum absolute value of the input tensor to the maximum representable value of FP8 (Narang et al., 2017). This method makes lowprecision training highly sensitive to activation outliers, which can heavily degrade quantization accuracy.

**Fine-Grained Quantization.** 如下图 所示 DeepSeek-V3 采取更细粒度的方式对输入进行缩放到 FP8 的表示范围: (1) 对于激活以 1x128 tile 进行分组和缩放 (每个 token 的 128 通道为一组); (2) 对于权重以 128x128 进行分组和缩放 (每 128 个输入和输出通道为一组). 虽然原生的 FP8 GEMM 不支持对 reduction 维度进行按组缩放，但可以和下面介绍的 FP32 累加策略进行配合使用。

![Fine-grained Quantization](https://share.note.youdao.com/yws/api/personal/file/WEB9db831a1951c8d26e446ee1588f8f55b?method=download&shareKey=fab1571fc5238ec904095a783f855ef3 "Fine-grained Quantization")

**Increasing Accumulation Precision.**  NVIDIA H800 GPU 上的 FP8 GEMM 累加精度被限制在 14 bits (远低于 FP32). 为在低精度计算中确保最终的数值精度，DeepSeek-V3 采用了一种结合 Tensor Cores 与 CUDA Cores 的混合计算流程。首先利用 Tensor Cores 的高吞吐量特性来执行 MMA (Matrix Multiply-Accumulate) 运算，中间结果在硬件原生的有限位宽累加器中进行阶段性累加。当累加操作进行 $N_c$ 次后，所产生的部分和将被立即复制到 CUDA Cores 上的 FP32 寄存器中，并与各自对应的细粒度量化缩放因子相乘，从而在执行全精度 FP32 最终累加的同时，高效地完成了反量化操作。这样能将反量化开销无缝融入到高精度累加步骤中，从而以最小的性能代价保证了最终结果的精确性。

![Increasing Accumulation Precision](https://share.note.youdao.com/yws/api/personal/file/WEB123812d18b874758b234db511dc40e22?method=download&shareKey=982d72c6d03c9bf72d5461d643ad4c65 "Increasing Accumulation Precision")

在 H800 架构上，典型的情况是两个 WGMMA 同时存在，当一个 warp group 执行 promotion 到 CUDA Core 操作时，另一个 warp group 能够执行 MMA 操作。实验中取 $N_c=128$，对应于 4 个 WGMMA.

**Mantissa over Exponents.** 对所有高精度的张量使用 4EM3 格式。

{{< details title="How to Compute Float Point Value">}}
一个常规浮点数 (即非零、非无穷大等特殊值) 的计算公式为：

$$
\text{Value} = (-1)^S \times (1.M)_{\text{binary}} \times 2^{(E_{\text{decimal}} - \text{Bias})}
$$

要理解这个公式， DeepSeek-V3 需要拆解里面的三个关键部分：符号、尾数和指数。

---
1. 确定符号 (Sign)
    * `S = 0`，则数值为正，$(-1)^0 = 1$
    * `S = 1`，则数值为负，$(-1)^1 = -1$

2. 计算尾数的值 (Mantissa)

不能直接使用 M 的二进制值。在常规浮点数中，标准规定尾数部分永远以 `1.` 开头。这样，这个 `1` 就不需要实际存储，从而可以节省一个比特位来提高精度。因此，尾数的实际值是 `(1.M)` 的二进制形式。

假设尾数位是 $m_1 m_2 m_3 \dots$，其代表的小数值为 

$$
m_1 \times 2^{-1} + m_2 \times 2^{-2} + m_3 \times 2^{-3} + \dots
$$. 

最终尾数项的值为 $1 + (m_1 \times 2^{-1} + m_2 \times 2^{-2} + m_3 \times 2^{-3} + \dots)$.

3. 计算指数的值 (Exponent)

指数部分也不能直接使用。为了能表示正、负指数，引入了偏置值 (Bias). 首先从 E 的二进制值计算出其十进制值 $E_{\text{decimal}}$ 后减去 Bias($2^{k-1} - 1$). 其中 k 是指数位的比特数。
* 对于 **E4M3** (k=4)，Bias = $2^{(4-1)} - 1 = 2^3 - 1 = 7$.
* 对于 **E5M2** (k=5)，Bias = $2^{(5-1)} - 1 = 2^4 - 1 = 15$.

4. 特殊值说明
当指数 E 全为 0或全为 1 时，代表的是一些特殊值，计算规则也不同:
* E 全为 0:
    * 如果 M 也全为 0，代表零 (Zero).
    * 如果 M 不为 0，代表**非规格化数 (Subnormal Numbers)**，计算公式变为 $(-1)^S \times (0.M) \times 2^{(1 - \text{Bias})}$，此时没有隐含的 1.
* E 全为 1:
    * 如果 M 全为0，代表**无穷大 (Infinity)**。
    * 如果 M 不为0，代表**NaN(Not a Number)**.
{{< /details >}}

**Online Quantization.** 采用 online 方式计算每个 1x128 激活 tile 和 128x128 权重 block 的最大绝对值。

### 3.3.3 Low-Precision Storage and Communication

**Low-Precision Optimizer States.** 用 BF16 格式存储 AdamW 优化器的一阶和二阶动量。优化器存储的 master weights 和 betch 的累加梯度仍以 FP32 格式存储。

**Low-Precision Activation.** 大部分激活以 FP8 格式存储，但以下这些是例外。
  - **Inputs of the Linear after the attention operator.** 这些激活会在反向传播过程中作为 attention 的输入，对精度比较敏感，因此采用 E5M6 格式存储。量化过程的缩放因子被限制为 2 的整数次幂。
  - **Inputs of the SwiGLU operator in MoE.** 以 FP8 格式存储 SwiGLU 的输入然后再反向传播中重计算。

**Low-Precision Communication.** 在 dispatch 之前对  MoE up-projections 的输入进行 FP8 量化。专家接收到 FP8 数据后，可以直接进行兼容的 FP8 前向传播。量化过程的缩放因子被限制为 2 的整数次幂。在反向传播进入 MoE down-projections 之前同样使用该策略。前向传播和反向传播 combine 后的结果以 FP16 格式存储。

## 3.4 Inference and Deployment

为了同时保证 Service-Level Objective (SLO) 和高吞吐量, *prefilling* 和 *decoding* 阶段采用了不同的部署策略。

prefilling 阶段的部署单元为 4 个节点 (32 GPUs). 并行策略如下
- attention part: 采用带有 Sequence Parallel (SP) 的 4-way Tensor Parallel (TP4)，并且和 8-way Data Parallelism (DP8) 一起使用。
- MoE part: 采用 32-way Expert Parallelism (EP32), shallow layer 不使用 TP.

其他部署细节:

- *redundant experts*: 部署 32 高负载的专家 (每十分钟统计一次进行调整) 副本。每个 GPU 除了有自己的 8 个专家之外还有 1 个高负载专家。
- 同时处理两个计算量差不多的 micro-batches，来掩盖 All-to-all 和 TP 的通信。即将一个 micro-batch 的 attention+MoE 和另一个 batch 的 dispatch+combine 重叠。
- *dynamic redundancy*: 每个 GPU 上放置 16 个专家，但每次只有 9 个被激活。

decoding 阶段的部署单元为 40 个节点 (320 GPUs). 并行策略如下
- attention part: 采用带有 SP 的 TP4，并且和 DP80 一起使用。
- MoE part: 采用 EP320. 256 GPU 被用来放置路由专家，64 GPU 被用来放置共享专家和冗余专家。

All-to-all 通过 IB 进行点对点直接传输。同时利用 IBGDA 技术让网卡直接读写 GPU 内存。系统会根据流量统计周期性地判断哪些常规路由专家是当前最热门的，然后动态地让那 64 个GPU去扮演这些热门专家的副本。因为每个 GPU 只被放置一个专家，所以当需要更改冗余策略时系统只需要改变路由逻辑，不需要在物理上移动或重新加载模型权重。

在 decoding 过程中 attention 会耗费更多时间。因此将一个 micro-batch 的 attention 和另一个的 dispatch+MoE+combine 重叠。decoding 阶段每个 GPU 只需要加载一个专家的参数，因此可以分配更多的 SM 给 attention 部分来加速其计算。

## 3.5 Suggestions on Hardware Design

基于 All-to-all 实现和 FP8 训练框架，DeepSeek-V3 对 AI 硬件厂商提出了一些建议。

### 3.5.1 Communication Hardware

当前通信算子的实现依赖于 SM，DeepSeek-V3 使用了 20 个 H800 SMs (一共 132 个) 用于通信，但使用 SM 进行通信会导致 tensor core 利用率很低。

当前 SM 主要在 All-to-all 通信中执行以下任务:

- IB 和 NVLink 域之间的数据转发，将目的地为同一节点内多个不同 GPU 的流量，首先汇聚到单个代理GPU上。
- 在 RDMA 缓冲区 (已注册的 GPU 内存区域) 与模型的输入/输出缓冲区之间进行数据搬运。
- 为 All-to-all 通信的 combine 阶段执行 reduce 操作。
- 在需要跨越 IB 和 NVLink 网络域、向多个不同专家进行分块数据传输{{< sidenote >}}在一个 GPU上 的tokens，其中一些可能要去当前节点内的专家 (通过NVLink)，另一些则要去其他节点上的专家 (通过IB). 在发送之前，GPU必须在自己的内存里进行一次数据重排，把所有目的地是专家 A 的 tokens 打包成一个连续的内存块，所有去专家 B 的 tokens 打包成另一个内存块。{{< /sidenote >}}的过程中，管理细粒度的内存布局。

### 3.5.2 Compute Hardware

**Higher FP8 GEMM Accumulation Precision in Tensor Cores.** 在目前 NVIDIA Hopper 架构的 Tensor Core 实现中，FP8 GEMM 的累积精度有限。在根据最大指数右移对齐 32 个尾数乘积后，Tensor Core 只使用每个尾数乘积的最高 14 位进行加法，并截断超过此范围的位。将加法结果累加到寄存器中也采用 14 位精度。

**Support for Tile- and Block-Wise Quantization.** 目前的 GPU 只支持逐张量量化，缺乏对细粒度量化的原生支持，比如 DeepSeek 的 tile 量化和 block 量化。在当前的实现中，当累加 $N_c$ 次时，部分结果将从 Tensor Core 复制到 CUDA Core，乘以缩放因子，并累加到 CUDA Core 上的FP32 寄存器。尽管与精确的 FP32 累加策略相结合，反量化开销显着减轻，但 Tensor Core 和 CUDA Core 之间频繁的数据移动仍然限制了计算效率。

**Support for Online Quantization.** 当前情况下需要从 HBM 中读取 128 个 BF16 激活值 (之前计算的输出) 进行量化，然后将量化后的 FP8 值写回 HBM，然后再次读取以进行 MMA.

**Support for Transposed GEMM Operations.** 在当前工作流程中，前向传播的激活被量化为 1x128 FP8 tile 并存储。在反向传播中，矩阵需要被读出、反量化、转置、重新量化为 128x1 tile，并存储在 HBM 中。

DeepSeek-V3 的预训练阶段围绕着高质量的数据构建、精心设计的超参数、长上下文扩展以及全面的性能评测展开。

# 4. Pretraining
## 4.1 Data Construction

* **训练语料**:
    *  模型在一个包含 **14.8T** 高质量、多样化 token 的语料库上进行预训练。
    *  与 DeepSeek-V2 相比，新语料提升了数学和编程相关样本的比例，并扩展了除中英文之外的多语言覆盖范围。
    *  数据处理流程经过优化，旨在最小化冗余，同时保持语料的多样性。
* **FIM 策略**:
    *  模型训练中采用了 FIM (Fill-in-Middle) 策略，该策略被证明在不损害常规“下一词预测”能力的同时，赋予了模型根据上下文准确预测中间文本的能力。
    *  FIM 策略在文档层面以 10% 的应用率实施，并采用 Prefix-Suffix-Middle (PSM) 框架构建数据格式。
* **分词器**:
    * 分词器采用 Byte-level BPE，词汇表大小扩展至 **128K**.
    * 为了优化多语言压缩效率，对预分词器和训练数据进行了修改。
    * 为了解决因合并标点和换行符可能导致的 token边界偏差，训练中会随机拆分一部分这类组合 token.

## 4.2 Hyper-Parameters

* **模型结构超参数**:
    * 总共有 61 层 Transformer，隐藏层维度为 7168.
    * MLA 注意力头数 $n_h$ 为128，每个头的维度为 128. KV 压缩维度 $d_c$ 为 512，Query 压缩维度 $d_c^{'}$ 为1536.
    *  除了前三层，其余所有 FFN 都被替换为 MoE 层。
    * 每个 MoE 层包含 1个共享专家 和 256个路由专家。每个 token 会激活其中的 8个 路由专。
    * 采用 MTP 策略，预测深度为 1，即除了下一个词，还会额外再预测一个词。
    * 最终模型总参数量为 671B，每个 token 的激活参数量为 37B.
* **训练超参数**:
    * 优化器采用 AdamW，其中 $\beta_{1}=0.9, \beta_{2}=0.95$，权重衰减为 0.1.
    * 预训练阶段的最大序列长度为 4K.
    * 学习率调度：先在 2K 步内线性增长至 $2.2\times10^{-4}$，保持该速率直到消耗10T token，然后在 4.3T token 内余弦衰减至 $2.2\times10^{-5}$，最后在 500B token 的训练中进一步调整。
    *  采用了批次大小调度策略，从 3072 逐步增加到15360.
    * 路由机制被限制为每个 token 最多发送到 4 个节点，以平衡负载。
    * 负载均衡策略主要采用 auxiliary-loss-free，偏置更新速率 $\gamma$ 在前 14.3 Token 时为 0.001，后 500B token 时为 0.0. 
    * 对于序列级平衡损失 $\alpha=0.00001$，以防止单一样本内的极端不平衡。
    * MTP loss 权重 $\lambda$ 对于前 10T token 为 0.3，对于后 4.8T token 为 0.1.

## 4.3 Long Context Extension

* **扩展方法**: 采用与 DeepSeek-V2 类似的方法，在预训练后应用 **YaRN** 技术进行上下文扩展。
* **扩展阶段**: 分为两个阶段，分别将上下文窗口从 4K 扩展到 32K，再进一步扩展到 128K.
* **效果验证**: 通过大海捞针 (Needle In A Haystack) 测试表明，模型在高达 128K 的完整上下文长度内均表现出色且稳定。

## 4.4 Evaluations

* **评测范围**: 主要在中英文基准测试以及一个多语言基准上进行评测，与当前最先进的开源基础模型进行比较，如 DeepSeek-V2-Base, Qwen2.5 72B Base, 和 LLaMA-3.1 405B Base.
* **评测结果**:
    * DeepSeek-V3-Base 全面超越了 DeepSeek-V2-Base 和 Qwen2.5 72B Base，并在绝大多数基准上超过了 LLaMA-3.1 405B Base，成为当前最强的开源模型。
    * 与拥有 11 倍激活参数量的 LLaMA-3.1 405B Base 相比，DeepSeek-V3-Base 在多语言、代码和数学基准上表现出好得多的性能。
    * 在英语和中文语言基准上，DeepSeek-V3-Base 也展现出有竞争力或更好的性能。
* **训练效率**: 得益于高效的架构和工程优化，DeepSeek-V3 的训练效率极高。每训练 1T token 仅需 180K H800 GPU 小时，远比训练 72B 或 405B 的密集模型便宜。

## 4.5 Discussion

本节通过一系列消融实验，深入探讨了模型采用的两个关键新策略的有效性，并对负载均衡的不同实现方式进行了对比分析。

### 4.5.1 Ablation Studies for Multi-Token Prediction

* **实验设置**:
    * 在两个不同规模 (一个15.7B，一个228.7B) baseline MoE模型上进行验证。
    * 对比模型在 baseline 模型的基础上增加了一个预测深度为 1 的MTP模块，其他设置 (如训练数据、架构) 保持不变。
    * 为了保证公平比较，在推理阶段会丢弃MTP模块，因此对比模型的推理成本完全相同。
* **实验结论**:
    * 实验结果 (Table 4) 表明，MTP策略在绝大多数评测基准上都能稳定地提升模型性能。
    * 例如，在大型模型上，HumanEval (代码生成) 和 GSM8K (数学推理) 等任务的性能得到了显著提升。

### 4.5.2 Ablation Studies for the Auxiliary-Loss-Free Balancing Strategy

* **实验设置**:
    * 同样在两个不同规模 (一个小型15.7B，一个大型228.7B) baseline MoE 模型上进行验证。
    *  baseline 模型完全依赖传统的辅助损失函数来促进专家负载均衡。
    * 对比模型则移除了所有辅助损失，并引入了 Auxiliary-Loss-Free 的均衡策略，其他设置保持一致。
* **实验结论**:
    * 实验结果 (Table 5) 显示，Auxiliary-Loss-Free 策略在绝大多数评测基准上都取得了比纯辅助损失方法更好的模型性能。
    * 在代码和数学等任务上，性能提升尤为明显。

### 4.5.3 Batch-Wise Load Balance VS. Sequence-Wise Load Balance

* **核心区别**: Auxiliary-Loss-Free 策略是在整个训练批次 (batch-wise) 上实现均衡，而传统的辅助损失则是在每个序列 (sequence-wise) 内部强制实现均衡。
* **理论优势**: 批次级的均衡约束更为灵活，它不强制每个序列内部的专家使用频率都一样，从而允许专家更好地专精于特定领域 (如代码、数学等).
* **实验验证**:
    * 通过分析模型在不同领域数据上的专家负载，观测到 Auxiliary-Loss-Free 模型展现出了更强的专家特化模式。
    * 进一步的实验表明，只要能实现相似水平的批次级负载均衡，无论是使用 Auxiliary-Loss-Free 方法还是新设计的批次级 Auxiliary-Loss-Free 方法，都能达到相似的优异模型性能，且均优于序列级辅助损失方法。
* **潜在挑战与解决方案**:
    * 批次级均衡可能面临两个挑战：单个序列或小批次内的负载不均，以及推理时因领域切换导致的负载不均。
    * 第一个挑战通过使用大规模的专家并行和数据并行 (确保了每个微批次的规模足够大) 得以自然解决。
    * 第二个挑战则通过在推理部署中采用冗余专家策略来克服。

# 5. Post-Training

后训练阶段旨在将预训练好的基础模型与人类偏好对齐，并进一步解锁其潜力。该阶段主要包括监督微调 (SFT) 和强化学习 (RL)，并涉及从 DeepSeek-R1 系列模型中蒸馏推理能力。

## 5.1 Supervised Fine-Tuning, SFT

* **数据集构建**: SFT 数据集包含 150 万个实例，涵盖多个领域。
    * **推理数据**: 对于数学、代码、逻辑等推理任务，利用内部的 DeepSeek-R1 模型生成数据。虽然 R1 生成的数据准确性高，但存在过度思考、格式不佳和长度过长等问题。为了平衡准确性与简洁性，SFT 训练中会混合使用原始应答和经过精心设计的系统提示词引导下的 R1 应答。
    * **非推理数据**: 对于创意写作、角色扮演等任务，使用 DeepSeek-V2.5 生成应答，并由人类标注员进行验证。
* **SFT 设置**:
    * 模型在 SFT 数据集上微调了 2 个 epoch.
    * 学习率采用余弦衰减策略，从 $5\times10^{-6}$ 降至 $1\times10^{-6}$.
    * 训练序列由多个样本打包而成，但采用样本掩码策略确保样本间相互隔离。

## 5.2 Reinforcement Learning

### 5.2.1 Reward Model
RL 过程采用了 Rule-Based 模型和 Model-Based 的奖励模型。
* **Rule-Based RM**: 用于有明确验证规则的问题，如数学题的确定性答案或代码题的单元测试结果。这种方式可靠性高，不易被模型钻空子。
* **Model-Based RM**: 用于答案更开放、没有确定性对错的问题。该奖励模型由 DeepSeek-V3 的 SFT 版本训练而来，并通过包含思维链的偏好数据进行训练，以降低 reward hacking 的风险。

### 5.2.2 GRPO
* 采用了 **GRPO (Group Relative Policy Optimization)** 算法进行强化学习。
* GRPO 的一个特点是它不需要一个与策略模型同等大小的 critic 模型，而是从一组采样输出的分数中估计 baseline.
* RL 过程融合了来自编码、数学、写作、角色扮演等不同领域的提示词，这不仅使模型与人类偏好更对齐，也提升了在 SFT 数据有限场景下的基准测试性能。

## 5.3 Evaluations

* **评测设置**:
    * 除了基础模型评测用的基准外，进一步在 IFEval, GPQA, LongBench v2, SWE-Bench Verified, Aider, Codeforces, AIME 2024 等更具挑战性的基准上进行评估。
    * 对比的 baseline 模型包括其他强大的开源和闭源模型，如 Qwen2.5-72B-Inst, LLaMA-3.1-405B-Inst, Claude-3.5-Sonnet, 和 GPT-4o-0513。
* **Standard Evaluation**:
    * 评测结果 (Table 6) 显示，DeepSeek-V3 是表现最好的开源聊天模型。
    * 在知识基准 (MMLU, MMLU-Pro, GPQA-Diamond) 上，其性能与顶级的闭源模型相当或相近。
    * 在长上下文理解基准 (DROP, LongBench v2, FRAMES) 上，表现出顶级水平，例如在 DROP 上取得了 91.6 的 F1 分数，超越了所有其他模型。
    * 在代码和数学基准上表现卓越，尤其是在 AIME, MATH-500 等高难度数学竞赛基准上，绝对得分领先第二名约 10%，优势巨大。
    * 在中文基准上，如 C-SimpleQA，其表现也超越了包括 Qwen2.5 在内的其他模型。
* **Open-Ended Evaluation**:
    * 在 Arena-Hard 基准测试中，DeepSeek-V3 取得了超过 85% 的胜率，与顶级的 Claude-3.5-Sonnet-1022 表现持平，成为首个在该基准上突破 85% 的开源模型。
    * 在 AlpacaEval 2.0 上，其表现同样出色，超越了所有对比的开源和闭源模型。
* **作为奖励模型的能力**:
    * 在 RewardBench 基准上评测其作为奖励模型的判断能力，结果显示 DeepSeek-V3 与最新版本的 GPT-4o 和 Claude-3.5-Sonnet 表现相当。

## 5.4 Discussion

* **从 DeepSeek-R1 蒸馏知识**:
    * 消融实验 (Table 9) 证明，从长思维链 (long-CoT) 模型 DeepSeek-R1 中蒸馏知识的策略非常有效，显著提升了模型在 LiveCodeBench 和 MATH-500 上的性能。
    * 实验也揭示了一个权衡：蒸馏带来了性能提升，但也显著增加了回应的平均长度。因此，在 DeepSeek-V3 的开发中对蒸馏设置进行了仔细选择以求平衡。
* **Self-Rewarding**:
    * 在缺乏明确验证规则的通用场景中，模型开发采用了 constitutional AI 的方法，即**使用 DeepSeek-V3 自身的投票评估结果作为反馈源**来进行优化。
    * 这种自奖励的范式产生了显著的对齐效果，并被认为是实现LLM自我改进的重要方向。
* **MTP 评测**:
    * 模型采用的 MTP 技术可以预测第 2 个token.
    * 评测显示，这个额外预测的 token 的接受率在 85%-90%之间。
    * 结合 speculative decoding 框架，这个高接受率使得模型的解码速度 (TPS) 提升了1.8倍.