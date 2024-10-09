---
title: Ring Attention Principle
date: 2024/9/26 22:59:35
categories: Distributed Training
tags: Essay
excerpt: This is a brief introduction to the Ring Attention Principle.
mathjax: true
katex: true
---
# Background

&emsp;&emsp;如今 LLM 的 token 长度显著增加，从 GPT-3.5 的 16k 到 Claude 2 的 200k，现在 Gemini 1.5 Pro 甚至有 1M 的 token 长度。如此长的 token 在计算 attention 时对显存的需求非常大。[Ring Attention](https://arxiv.org/abs/2310.01889) 便是为了并行计算 attention 而提出的一种方法[^1]。

> Ring Attention 和 Flash Attention 可以同时使用。

# Attention and Memory

&emsp;&emsp;要计算 attention， 我们需要三个大小为 (s, d) 的矩阵：Q (query)、K (key)、V (value)，其中 s 为序列长度，d 为模型维度。attention 的计算公式为

{% mathjax %}
Attention(Q, K, V) = softmax(QK^T / \sqrt{d})V
{% endmathjax %}

&emsp;&emsp;忽略 sqrt(d) 项，我们记 Score Matrix 为 S = QK^T / \sqrt{d}，然后对 S 进行 softmax 归一化，得到 Attention Matrix. 可以发现它们占用显存大小是 O(s*s) 数量级。即使使用 [Flash Attention](https://arxiv.org/abs/2205.14135)，显存占用量也是 O(s) 数量级。

![Attention Compute Process](https://note.youdao.com/yws/api/personal/file/WEBe66e94e161b89a4ba25d05b67a47e393?method=download&shareKey=742185dd412edbdb3266fa16ab91d787 "Attention Compute Process")

&emsp;&emsp;我们希望如果在 N 个设备上并行计算 attention，每个设备的显存占用量为整个的 1/N, 因此就需要对 Q、K、V 的 sequence 长度进行切分。但是如果得到的最终 attention 矩阵需要在设备间进行集合通信组装每个的计算结果，通信量也和 sequence 长度成正比。Ring Attention 提出了一个巧妙的解决方案：在设备之间进行轮转，并行化所有计算而且完全隐藏通信的开销。

> We will rotate between devices to parallelize all computation and hide the communication overhead completely.

# Splitting the Query

&emsp;&emsp;假设我们有 N 个设备，我们将 Q 沿着 sequence 维度切分为 N 份，每份大小为 (s/N, d). 由于计算 Score 和 Attention 需要完整的 K 和 V，这样它们也被切分成 N 份，每份大小为 (s/N, d). 计算示意图如下。

![Split Q](https://note.youdao.com/yws/api/personal/file/WEB170087e68345309f813b8edee9487b92?method=download&shareKey=f848ff8adb5676443347921c65a3b104 "Split Q")

# Splitting the Key and Value

&emsp;&emsp;对 K 和 V 的切分并不能像 Q 那样直接。因为 softmax 的计算公式如下，要得到分母的值意味着我们需要对每一行进行计算。
{% mathjax %}
softmax(s_i) = \frac{\exp(s_i)}{\sum_{j=i}^d{\exp(s_j)}}
{% endmathjax %}

&emsp;&emsp;如果我们能对 K 和 V 进行切分并正确计算 softmax，那么计算过程可以由下图所示的那样完成 (忽略 softmax). 外循环遍历 Q 的所有分块，内循环遍历 K 和 V 的所有分块，一次计算一部分的 attention. Ring Attention 示意图如下所示，顾名思义所有设备组成一个环状，每个设备存储 Q 的一部分，每次迭代过程会传递 K 和 V 到下一个设备，最终每个设备将得到计算自己 Q 部分的 attention 矩阵所需要的 K 和 V. 每个设备被分配 Q 的一部分 (即一个外层循环索引)，并迭代计算每个 K 和 V 的分块 (内循环)。每个设备只需要跟踪形状为 (s/N, s/N) 的累积和 A_j。

![Attention Parallel Computation](https://note.youdao.com/yws/api/personal/file/WEBbc9ef7d01431fe639ecf44842bce0e1a?method=download&shareKey=03d587a38ca574ed1547f2594a45ab4c "Attention Parallel Computation")

# Online Softmax

&emsp;&emsp;在内循环的每次迭代中我们可以更新部分和为 {% mathjax %} l^j = l^{j-1} + \sum_{k_t\in K_j}{\exp(Q_ik_t^T)} {% endmathjax %}. 在内循环结束后我们就可以获得每一行的指数和。归一化和与 V 的相乘顺序不会影响结果，我们可以先累加总和，并在所有其他计算完成后再执行实际的归一化操作。

&emsp;&emsp;因此，设备 i 除了计算当前的累计和 {% mathjax %} A^j = A^{j-1} + \exp(Q_i K_j^T) V_j {% endmathjax %} 外，还需要在内循环每次迭代中更新部分和 {% mathjax %} l^j \in \mathbb{R}^{B_q} {% endmathjax %}，其中 {% mathjax %} B_q {% endmathjax %} 为 Q 的分块大小。

# Safe softmax

&emsp;&emsp;由于指数运算经常容易出现溢出，我们通常减去 max(s_i) 后进行指数运算，公式如下，这样并不会影响结果。

{% mathjax %}
\mathrm{softmax}(s_{1:N})=\frac{\exp(s_{1:N})}{\sum_i\exp(s_i)}\cdot\frac{\exp(-s_{max})}{\exp(-s_{max})}=\frac{\exp(s_{1:N}-s_{max})}{\sum_i\exp(s_i-s_{max})}
{% endmathjax %}

&emsp;&emsp;所以我们在内循环每次迭代中需要先更新当前的最大值 {% mathjax %} m^{j+1}=\max(m^j,\max(Q_iK_{j+1}^T)) {% endmathjax %}，然后更新之前迭代的计算结果 A_j 和 部分和 l_j. 最后再计算本次迭代的结果。

{% mathjax %}
A^{j+1}=A^j\cdot\exp(m^j-m^{j+1})+\exp(Q_iK_{j+1}^T-m^{j+1})\cdot V_j
{% endmathjax %}

更新部分和

{% mathjax %}
l^{j+1}=l^j\cdot\exp(m^j-m^{j+1})+\exp(Q_iK_{j+1}^T-m^{j+1})
{% endmathjax %}

# Putting it Together

Ring Attention 计算步骤如下：

1. 沿着 Q 的 sequence 长度拆分为一个独立的外循环。
2. 应用 Online Safe Softmax，以便沿着 K 和 V 的sequence 长度拆分，从而在内层循环中累积计算注意力。

&emsp;&emsp;这种并行化的方式是通过将每个设备分配一个 Q_i 块来实现的。因此，我们需要将 Q 拆分为 N 个相等的部分 (B_Q=N). 每个设备将分别计算它的输出块 {% mathjax %} \text{Output}(Qi,K,V)= \text{softmax}(Q_i K^T)V {% endmathjax %}，通过在 K 和 V 块上执行内循环来迭代计算。难点挑战在于设备无法一次存储完整的 K 和 V 矩阵。

&emsp;&emsp;如果我们有 4 个 GPU，那么我们将把每个设备的 Q 按序列维度分成 4 个块，K 和 V 被分割成 B_K=B_Q=N 个块，并对设备进行初始化，使每个设备都持有一个 Qi 块、 一个 Kj 块和 一个 Vj 块。为简单起见，我们可以假设设备 i 在开始时持有 Qi, Ki 和 Vj 块。在设备计算完与其当前 vj kj 相对应的一个内循环步骤后，每个设备都需要接收下一个 Key 和 Value 块，以继续内循环。 我们将 N 个设备围成一个环，其中设备 i 可以向设备 i+1 以此类推，如图所示：

![KV-overlap](https://note.youdao.com/yws/api/personal/file/WEB5d0930a41cedf1d4e46af9baa5071f78?method=download&shareKey=0e898be310f92f54f0b065a38771eb5f "KV-overlap")

&emsp;&emsp;如果在设备 i 上计算内循环的一个步骤 Qi,Vj,Kj 的这段时间内，设备 i 还能向设备 i+1 发送其当前 Kj Vj，并同时从设备 i-1 接收 V_j-1,K_j-1，那么只要发送和接收密钥和值块的时间低于计算时间，那么发送和接收 Key 和 Value 块的延迟就会隐藏在执行实际计算时间之内。一个例子如下图所示。

![KV-rotate](https://note.youdao.com/yws/api/personal/file/WEB935e1d2c0eb43c35c5c828abe8a44612?method=download&shareKey=9ec41a1c178534620d9f7274ff2ce9d0 "KV-rotate")

# Memory and Arithmetic Complexity

&emsp;&emsp;以深度学习中常用的 bfloat16 数据类型为例。GPU 或 TPU 等并行处理加速器通常以 FLOP:=F 来衡量，即设备理论上每秒可执行的浮点运算次数。我们假设硬件被完全利用。此外，我们设不同设备之间的连接带宽为:=B (Bytes/sec).

&emsp;&emsp;内存复杂度: 为了同时进行接收发送和计算，我们需要有用于接收新 KV 块的寄存器器。存储当前 KV  值块需要 2dc 浮点数或 4dc 字节。用于接收新的 KV 块的内存大小也是 2dc 浮点数或 4dc 字节。假设计算本身不需要更多内存 (利用 Flash Attention 或 Blockwise Attention)，计算当前步骤的输出需要 dc 个浮点数或 2dc 字节。此外，每个设备还需要存储其 Qi 块，这也需要 dc 个浮点数或 2dc 字节。总共需要 6dc 个浮点或 12dc 字节。

{% note info %}
&emsp;&emsp;Ring Attention 与 Flash Attention 是正交的，可以一起使用 (Flash Attention 实际上用于 Ring Attention 的内循环). Flash Attention 目标是不将整个 Score Matrix 加载到全局内存中，从而在序列长度上获得线性内存复杂度。Ring Attention 将 原始注意力方法和 Flash Attention 的内存复杂度至少降低了 N 倍，使用 N 个设备的内存复杂度至少降低 N 倍，因为它将所有矩阵都拆分为至少 N 个或更多部分 (将 QKV 分别分成 N 份，并将 Score Matrix 分成 N^2 分). 无论内存复杂度是由 QKV，还是由 Score Matrix 主导，Ring Attention 都能将内存成本降低至少 N 倍。
{% endnote %}

&emsp;&emsp;通信开销: 在内循环每一步中，每个设备需要通过带宽为 B 的信道向下一个设备发送 2⋅c_Q⋅d 浮点数。每个 bf16 大小为 2字节，因此，所需的时间约为 4⋅c⋅d/B.

&emsp;&emsp;运算强度： 一个内循环步骤，计算局部注意力需要 2⋅d⋅c^2 次浮点计算，计算 softmax，归一化向量和最大值向量需要 2⋅c⋅d 次浮点计算，计算局部注意力与 Vj 块的乘积需 2⋅d⋅c^2 次浮点计算。因此，总计算所需时间≈4⋅d⋅c^2/F.

&emsp;&emsp;为了重叠通信和计算 (隐藏通信开销)，我们需要 KV 块的传输时间小于等于计算本地 QKV 所需的时间：
{% mathjax %} 
4\cdot c\cdot d/B\leq4\cdot d\cdot c^2/F\iff B\geq F/c\iff s/N\geq F/B 
{% endmathjax %}

# Futher Optimization

&emsp;&emsp;Ring Attention 的一个应用是用于因果 Transformal 模型时，加上三角形掩码用于注意力计算。这意味着有些 GPU 不需要对整个序列进行计算，导致它们大部分时间处于闲置状态。作为 Ring Attention 的扩展，[Stripe Attention](https://arxiv.org/pdf/2311.09431.pdf) 解决了这一问题，并提供了一种分配计算更均匀的方案，从而使 Ring Attention 的计算速度更快。

&emsp;&emsp;除了 Ring Attention 和 Flash Attention 等使标准 Transformer 架构能有更长的上下文长度的技术外，人们还尝试使用 [Mamba](https://arxiv.org/abs/2312.00752) 等具有线性注意力的状态空间模型（SSM）等模型架构。

# References
[^1]: https://coconut-mode.com/posts/ring-attention/