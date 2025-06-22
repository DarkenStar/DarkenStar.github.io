---
title: "DeepSeekMLA"
date: 2025-06-19T11:24:45+08:00
lastmod: 2025-06-19T11:24:45+08:00
author: ["WITHER"]

categories:
- Paper Reading

tags:
- DeepSeek

keywords:
- MLA

description: "Principle of DeepSeekV3 MLA" # 文章描述，与搜索优化相关
summary: "Principle of DeepSeekV3 MLA" # 文章简单描述，会展示在主页
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
# Preliminary: What is RoPE
## Introduction
旋转位置编码 (RoPE) 是一种新颖的、基于相对位置的编码方法，它被设计用于提高 Transformer 模型处理长序列的能力，同时保持计算效率。与传统的绝对位置编码 (如正弦/余弦位置编码) 或直接的相对位置编码 (如 T5 中使用的相对偏置) 不同，RoPE 将位置信息集成到自注意力机制的 Q 和 K 的表示中，使得 Q 和 K 的点积自然地编码了**相对位置信息**。

RoPE 的核心思想是，通过对查询和键向量应用特定的旋转操作，使得两个向量的点积结果只依赖于它们之间的相对距离，而不是它们的绝对位置。这使得模型能够更好地泛化到更长的序列，并且在处理位置信息时更加高效。

**RoPE 的主要优点包括：**
* **编码相对位置信息：** 自然地将相对位置信息融入到注意力分数中。
* **长序列外推能力：** 提高了模型在训练时未见过的更长序列上的性能。
* **与自注意力机制的兼容性：** 无缝集成到 QKV 点积注意力中。
* **简单且高效：** 实现相对简单，且不会显著增加计算复杂度。

## Formular

RoPE 的主要思想是通过对查询 $q$ 和键 $k$ 应用一个旋转矩阵 $R_t$ (取决于其绝对位置 $t$) ，使得点积 $q_m^T k_n$ 能够通过某种方式转化为只依赖于相对位置 $m-n$ 的函数。

对于一个向量 $x \in \mathbb{R}^d$ 在位置 $m$ 处，RoPE 的变换函数 $f(x, m)$ 可以定义如下：

如果向量维度是偶数 $d$，我们可以将其分成 $d/2$ 对，每对执行一个二维旋转。
对于向量 $x = [x_0, x_1, \ldots, x_{d-1}]^T$，RoPE 对其每个维度对 $(x_{2i}, x_{2i+1})$ 应用旋转：

$$
f(x, m)_{2i} = x_{2i} \cos(m\theta_i) - x_{2i+1} \sin(m\theta_i) \\
f(x, m)_{2i+1} = x_{2i} \sin(m\theta_i) + x_{2i+1} \cos(m\theta_i)
$$

其中 $\theta_i$ 是预设的频率，通常定义为 $\theta_i = 10000^{-2i/d}$. $i=0, \dots, d/2 - 1$ 是维度对的索引。

**用矩阵形式表示：**
我们可以将这种旋转操作表示为一个稀疏的块对角矩阵 $R_m^d$，其形式为：
$$R_m^d = \begin{pmatrix}
\cos(m\theta_0) & -\sin(m\theta_0) & 0 & 0 & \cdots \\
\sin(m\theta_0) & \cos(m\theta_0) & 0 & 0 & \cdots \\
0 & 0 & \cos(m\theta_1) & -\sin(m\theta_1) & \cdots \\
0 & 0 & \sin(m\theta_1) & \cos(m\theta_1) & \cdots \\
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}$$
那么，经过 RoPE 编码的查询和键可以表示为：
$$\mathbf{q}_m = R_m^d \mathbf{q}$$
$$\mathbf{k}_n = R_n^d \mathbf{k}$$
其中 $\mathbf{q}$ 和 $\mathbf{k}$ 是原始的查询和键向量 (不含位置信息) ，$\mathbf{q}_m$ 和 $\mathbf{k}_n$ 是经过 RoPE 处理后的查询和键向量。

**RoPE 的关键特性：点积与相对位置**
经过 RoPE 变换后，注意力机制中的点积可以分解为：
$$\mathbf{q}_m^T \mathbf{k}_n = (R_m^d \mathbf{q})^T (R_n^d \mathbf{k})$$
由于 $R_m^d$ 是正交矩阵，其逆矩阵等于其转置，即 $(R_m^d)^T = (R_m^d)^{-1} = R_{-m}^d$. 因此有
$$\mathbf{q}_m^T \mathbf{k}_n = \mathbf{q}^T (R_m^d)^T R_n^d \mathbf{k} = \mathbf{q}^T R_{-m}^d R_n^d \mathbf{k} = \mathbf{q}^T R_{n-m}^d \mathbf{k}$$
这个最终结果 $\mathbf{q}^T R_{n-m}^d \mathbf{k}$ 表明，两个向量的点积只依赖于它们的**相对位置差 $n-m$**，而与它们的绝对位置 $n$ 和 $m$ 无关。这就是 RoPE 能够编码相对位置信息的数学基础。

# Workflow
## Notation

- $d$: embedding 维度
- $d_h$: 每个注意力头的维度
- $\mathbf{h}_t\in\mathbb{R}^d$: 某个 attention 层第 t 个 token 的输入。

## KV Compression

$$
\textcolor{red}{c_t^{KV}} = W^{DKV}h_t  \tag{1}
$$
$$
[k_{t,1}^{C}, k_{t,2}^{C}, \ldots, k_{t,n_h}^{C}] = k_t^C = W^{UK}c_t^{KV}  \tag{2}
$$
$$
\textcolor{red}{k_t^R} = \text{RoPE}(W^{KR}h_t)  \tag{3}
$$
$$
k_{t,i} = [k_{t,i}^C, k_{t}^R] \tag{4}
$$
$$
[v_{t,1}^C, v_{t,2}^C, \ldots, v_{t,n_h}^C] = v_t^C = W^{UV}c_t^{KV} \tag{5}
$$

- $c_t^{KV} \in \mathbb{R}^{d_c}$: 压缩后的 KV 潜在向量。
- $d_c (\ll d_h n_h)$: KV 压缩到的维度。
- $W^{DKV} \in \mathbb{R}^{d_c \times d}$: KV 降维投影矩阵。
- $W^{UK}, W^{UV} \in \mathbb{R}^{d_h n_h \times d_c}$ 分别是 K & V 的升维投影矩阵。
- $W^{KR} \in \mathbb{R}^{d_h^R \times d}$: 用于生成携带 RoPE 的解耦键的矩阵 (Su et al., 2024) 

红色的是需要缓存的向量，后续说明原因。注意到对 K 进行 RoPE 之前是对输入向量乘以了个投影再进行的。而且 K 的每个注意力头被拼接的都是同一个 $k_{t}^R$，有点类似于 MQA.

## Q Compression

$$c_t^Q = W^{DQ}h_t \tag{6}$$
$$[q_{t,1}^C, q_{t,2}^C, \ldots, q_{t,n_h}^C] = q_t^C = W^{UQ}c_t^Q \tag{7}$$
$$[q_{t,1}^R, q_{t,2}^R, \ldots, q_{t,n_h}^R] = q_t^R = \text{RoPE}(W^{QR}q_t^C) \tag{8}$$
$$q_{t,i} = [q_{t,i}^C, q_{t,i}^R] \tag{9}$$

- $c_t^Q \in \mathbb{R}^{d_c'}$: Q 压缩后的潜在向量。
- $d_c'(\ll d_h n_h)$ 表示 Q 压缩后的维度。
- $W^{DQ} \in \mathbb{R}^{d_c' \times d}, W^{UQ} \in \mathbb{R}^{d_h n_h \times d_c'}$: 分别是 Q 的降维和升维矩阵。
- $W^{QR} \in \mathbb{R}^{d_h^R n_h \times d_c'}$ 是用于生成携带 RoPE 的解耦 Q 的矩阵。

注意到对 Q 的 RoPE 是在压缩后进行的，即为每个注意力头都生成了一个位置编码信息后进行拼接。

## Attention Computation
最终 $q_{t,i}$, $k_{j,i}$, $v_{j,i}^C$ 被组合起来以生成最终的注意力输出 $u_t$

$$\mathbf{o}_{t,i} = \sum_{j=1}^{t} \text{Softmax}\left(\frac{q_{t,i}^T \mathbf{k}_{j,i}}{\sqrt{d_h + d_R}}\right)v_{j,i}^C \tag{10}$$
$$\mathbf{u}_t = W^O[\mathbf{o}_{t,1}, \mathbf{o}_{t,2}, \ldots, \mathbf{o}_{t,n_h}] \tag{11}$$

- $W^O \in \mathbb{R}^{d \times d_h n_h}$: 输出投影矩阵。

# Why Decoupled RoPE

假设不加 RoPE 的情况下进行 $q_{t,i}$, $k_{j,i}$ 的内积则有

$$
q_{t,i}^{T}\times k_{j,i}=(W_{(i)}^{UQ}c_{t}^{Q})^{T}\times W_{(i)}^{UK}c_{j}^{KV}=(c_{t}^{Q})^{T}\times(W_{(i)}^{UQ})^{T}W_{(i)}^{UK}\times c_{j}^{KV} \tag{12}
$$

RoPE 通过对向量应用一个**位置依赖的旋转变换**来注入相对位置信息。对于一个向量 $X$ 在位置 $t$，RoPE 可以被表示为一个旋转矩阵 $R_t$ 乘以 $X$：
$$\text{RoPE}(X, t) = R_t X$$
这里的 $R_t$ 是一个正交旋转矩阵，它取决于位置 $t$.

如果直接对压缩后 $k_t^C$ 的 使用 RoPE 那么情况会变成

$$
\begin{aligned}
q_{t,i}^{T}\times k_{j,i}&=(\mathcal{R}_{t}W_{(i)}^{UQ}c_{t}^{Q})^{T}\times\mathcal{R}_{j}W_{(i)}^{UK}c_{j}^{KV} \\
&=(c_{t}^{Q})^{T}\times(W_{(i)}^{UQ})^{T}\mathcal{R}_{t}^{T}\mathcal{R}_{j}W_{(i)}^{UK}\times c_{j}^{KV}\\
&=(c_{t}^{Q})^{T}\times(W_{(i)}^{UQ})^{T}\mathcal{R}_{t-j}W_{(i)}^{UK}\times c_{j}^{KV}
\end{aligned} \tag{13}
$$

中间的矩阵与相对位置有关，无法提前计算出来。因此文中就是对所有头都使用同一个 k 和计算 RoPE. 拼接后的向量再计算时

$$
q_{t,i}^T\times k_{j,i}=[q_{t,i}^C;q_{t,i}^R]^T\times[k_{j,i}^C;k_t^R]=(q_{t,i}^C,k_{j,i}^C)+(q_{t,i}^R,k_t^R) \tag{14}
$$

前一部分按照公式 (12) 进行计算，后一部分按照 MQA 方式计算。因此只用缓存 $c_t^{KV}$ 和 $k_t^R$.

# Source Code

[DeepSeek-V3 MLA](https://github.com/deepseek-ai/DeepSeek-V3/blob/f6e34dd26772dd4a216be94a8899276c5dca9e43/inference/model.py#L393-L494) 对应的源码位置

```python{linenos=true}
class MLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) Layer.

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.dim = args.dim
        self.n_heads = args.n_heads
        # 计算当前进程（卡）负责的注意力头数量，用于模型并行
        self.n_local_heads = args.n_heads // world_size 
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        # QK 头总维度 = 非 RoPE 部分 + RoPE 部分
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim

        # 查询投影 (wq) 的 LoRA 实现
        if self.q_lora_rank == 0:
            # 如果 q_lora_rank 为 0，表示不使用 LoRA，直接进行全秩投影
            # 将 dim 维度的输入投影到 n_heads * qk_head_dim 维度
            self.wq = ColumnParallelLinear(self.dim, self.n_heads * self.qk_head_dim)
        else:
            # 如果 q_lora_rank > 0，使用 LoRA 结构进行低秩投影
            # wq_a: dim -> q_lora_rank (低秩投影的第一步)
            self.wq_a = Linear(self.dim, self.q_lora_rank)
            # q_norm: RMSNorm 应用于低秩维度
            self.q_norm = RMSNorm(self.q_lora_rank)
            # wq_b: q_lora_rank -> n_heads * qk_head_dim (低秩投影的第二步)
            self.wq_b = ColumnParallelLinear(self.q_lora_rank, self.n_heads * self.qk_head_dim)
        
        # 键值投影 (wkv) 的 LoRA 实现
        # wkv_a: dim -> kv_lora_rank + qk_rope_head_dim
        # 对应图中的 W^{DKV} 投影到低秩 KV 潜在空间 (kv_lora_rank) 和解耦的 RoPE 键 (qk_rope_head_dim)
        # 这里的 kv_lora_rank 对应公式中的 d_c
        # 这里的 qk_rope_head_dim 对应公式中的 d_h
        self.wkv_a = Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim)
        # kv_norm: RMSNorm 应用于低秩维度
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        # wkv_b: kv_lora_rank -> n_heads * (qk_nope_head_dim + v_head_dim)
        # 对应图中的 W^{UK} 和 W^{UV} 的组合
        # 它将压缩后的 KV 潜在向量 (kv_lora_rank) 投影回非 RoPE 键 (qk_nope_head_dim) 和值 (v_head_dim) 的高维度空间
        self.wkv_b = ColumnParallelLinear(self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        
        # 输出投影 (wo)
        self.wo = RowParallelLinear(self.n_heads * self.v_head_dim, self.dim)
        
        # Softmax 缩放因子，用于注意力分数的缩放，防止内积过大
        self.softmax_scale = self.qk_head_dim ** -0.5
        
        # 如果序列长度超过原始训练长度，根据 RopeFactor 进行额外缩放，用于处理长序列外推问题
        if args.max_seq_len > args.original_seq_len:
            mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        # 根据注意力实现方式（naive 或 optimized）选择不同的 KV 缓存结构
        if attn_impl == "naive":
            # naive 实现直接缓存完整键 K 和值 V
            # k_cache: (max_batch_size, max_seq_len, n_local_heads, qk_head_dim)
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            # v_cache: (max_batch_size, max_seq_len, n_local_heads, v_head_dim)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            # optimized 实现缓存压缩后的 KV 潜在向量和解耦的 RoPE 键
            # kv_cache: (max_batch_size, max_seq_len, kv_lora_rank) - 对应论文中的 c_t^{KV}
            self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            # pe_cache: (max_batch_size, max_seq_len, qk_rope_head_dim) - 对应论文中的 k_t^R
            self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # 1. 查询 (Q) 的生成
        if self.q_lora_rank == 0:
            # 全秩投影
            q = self.wq(x)
        else:
            # LoRA 投影：x -> wq_a -> q_norm -> wq_b
            q = self.wq_b(self.q_norm(self.wq_a(x)))
        
        # reshape Q
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        
        # 分离 Q 的非 RoPE 部分和 RoPE 部分
        # q_nope 对应论文中的 q_{t,i}^C (非位置信息查询)
        # q_pe 对应论文中的 q_{t,i}^R (携带 RoPE 的解耦查询)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # 对 Q 的 RoPE 部分应用旋转位置编码
        # 对应论文中的 q_t^R = RoPE(W^{QR}c_t^Q) 的 RoPE 部分
        q_pe = apply_rotary_emb(q_pe, freqs_cis)
        
        # 2. 键值 (KV) 的生成
        # 将输入 x 投影到低秩 KV 潜在空间和解耦的 RoPE 键
        # 对应论文中的 c_t^{KV} 和 k_t^R
        kv = self.wkv_a(x)
        
        # 分离出 KV 潜在向量和解耦的 RoPE 键
        # kv 对应论文中的 c_t^{KV}
        # k_pe 对应论文中的 k_t^R (RoPE 解耦键)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        
        # 对 K 的 RoPE 部分应用旋转位置编码
        # 注意 k_pe.unsqueeze(2) 是因为 apply_rotary_emb 期望 (..., seq_len, head_dim) 结构
        # 这里的 k_pe 可能是 (bsz, seqlen, qk_rope_head_dim)，需要添加一个 head 维度
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)

        # 3. 注意力计算：根据实现方式 (naive 或 optimized)
        if attn_impl == "naive":
            # Naive 实现直接拼接 Q 的 RoPE 和非 RoPE 部分
            q = torch.cat([q_nope, q_pe], dim=-1) # Q 恢复为 (bsz, seqlen, n_local_heads, qk_head_dim)

            # 对 KV 潜在向量应用归一化，并进行第二阶段投影
            # 对应论文中将 c_t^{KV} 投影到非 RoPE 键和值的部分 (k_t^C 和 v_t^C)
            kv = self.wkv_b(self.kv_norm(kv))
            
            # 将 KV 结果重塑为 (batch_size, seq_len, n_local_heads, qk_nope_head_dim + v_head_dim)
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            
            # 分离出非 RoPE 键和值
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            
            # 拼接非 RoPE 键和 RoPE 键，组成完整的键 K
            # k_pe 之前是 (bsz, seqlen, 1, qk_rope_head_dim)，需要 expand 到 n_local_heads
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            
            # 更新 K 和 V 缓存 (在推理时用于自回归生成)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v

            # 计算注意力分数 (Q @ K^T)
            # scores: (batch_size, q_seq_len, n_local_heads, k_seq_len)
            # 使用整个缓存中的键进行计算
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else: # optimized 实现
            # 获取 wkv_b 权重，如果使用了量化则进行反量化
            wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size) 
            # 将 wkv_b 重塑为 (n_local_heads, head_dim, kv_lora_rank) 以便进行逐头的操作
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank) # (n_heads, (qk_nope+v), kv_rank)

            # 计算 Q_nope 与 K_nope 的点积 (通过 kv 缓存)
            # q_nope: (bsz, seqlen, n_local_heads, qk_nope_head_dim)
            # wkv_b[:, :self.qk_nope_head_dim] 是 W^{UK} 的部分
            # 这对应论文中的 Softmax(q_{t,i}^C @ c_{j,i}^{KV}) 的第一项
            q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])
            
            # 更新 KV 缓存 (kv_cache 对应论文中的 c_t^{KV})
            self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)
            # 更新 PE 缓存 (pe_cache 对应论文中的 k_t^R)
            # k_pe 之前是 (bsz, seqlen, 1, qk_rope_head_dim)，squeeze 掉那个 1 维度
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

            # 计算注意力分数
            # 第一项: 非 RoPE 查询 q_nope 与缓存的 kv_cache (压缩键) 的点积
            # 对应论文中的 Softmax(q_{t,i}^C @ c_{j,i}^{KV}) 的第一部分
            scores = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) + \
                      # 第二项: RoPE 查询 q_pe 与缓存的 pe_cache (解耦 RoPE 键) 的点积
                      # 对应论文中的 Softmax(q_{t,i}^R @ k_{j,i}^R) 的第二部分
                      torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            scores *= self.softmax_scale # 应用缩放因子

        # 应用注意力掩码 (如因果掩码，防止看到未来信息)
        if mask is not None:
            scores += mask.unsqueeze(1) # unsqueeze(1) 广播到 heads 维度

        # 对分数应用 Softmax 得到注意力权重
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        # 4. 值 (V) 的加权求和
        if attn_impl == "naive":
            # Naive 实现直接与 V 缓存进行点积
            # 对应论文中的 sum(Softmax(...) * v_{j,i}^C)
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else: # optimized 实现
            # optimized 实现通过 wkv_b 的值部分将加权后的压缩 KV 还原为 V
            # 第一步: 将注意力权重与缓存的 kv_cache (压缩值) 进行点积
            # 对应论文中的 Softmax(...) * c_{j,i}^{KV} 的第一部分
            x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
            # 第二步: 将加权后的压缩值通过 wkv_b 的值投影部分还原为最终的值向量
            # wkv_b[:, -self.v_head_dim:] 是 W^{UV} 的部分
            # 对应论文中的 Softmax(...) * v_{j,i}^C 的第二部分
            x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        
        # 将所有头的结果展平并进行最终的输出投影
        x = self.wo(x.flatten(2)) # x.flatten(2) 将 (bsz, seqlen, n_local_heads, v_head_dim) 展平为 (bsz, seqlen, n_local_heads * v_head_dim)
        return x
```