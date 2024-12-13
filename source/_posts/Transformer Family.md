---
title: Transformer Family
date: 2024/12/13 10:40:00
categories: Paper Reading
tags: blog
excerpt: Introduction of Transformer Family
mathjax: true
katex: true
---

# Origin of Transformer

Transformer 由谷歌研于 2017 年在一篇名为 [Attention is All You Need](https://arxiv.org/abs/1706.03762) 的论文中提出。与 RNN 的输入仅为一个 token 不同，Transformer 一次性可以输入一整个完整的序列。总体结构如下图所示，包含一个 Encoder 和一个 Decoder.

![Transformers Architecture](https://note.youdao.com/yws/api/personal/file/WEBd293bc1a46904e1af31ce993b83c68f1?method=download&shareKey=47cf357e488e7da5483a1b98f3257ab1 "Transformers Architecture")

## Embedding

Embedding 是一种将离散的、稀疏的输入 (如词语、字符、类别标签...) 转换为连续的、密集的向量表示的技术，核心是通过一个映射函数将离散的输入符号 (如单词) 映射到一个低维向量空间中。假设我们有一个包含 V 个单词的 Vocabulary，维度为 d，那么 Embedding Matrix 将是一个大小为 V×d 的矩阵，其中每一行是一个单词的向量表示。通过嵌入层，输入的词索引 (通常是整数) 就会被映射到该矩阵的对应行，从而得到词的向量表示。常见的预训练词嵌入方法包括：
- Word2Vec：通过上下文预测词语的方式学习词向量。
- GloVe：通过统计词共现信息来学习词向量。
- FastText：考虑了子词信息的词嵌入方法，能更好地处理词形变化。

在 PyTorch 和 TensorFlow 等框架中，通常有专门的 Embedding 层，Hugging Face 也有 tokenizer 将句子划分成单词并转换成对应的索引：

## Positional Encoding

Positional Encoding 作用是为输入的序列中的每个元素提供位置信息。由于 Transformer 架构并没有使用递归或卷积结构，本身无法捕捉输入序列中元素的相对位置关系，因此需要通过位置编码来显式地引入这些位置信息。

{% note info %}
Transformer 的主要优势是通过 Self-Attention 并行处理序列中的每个元素，但是这也意味着它没有自带顺序感知能力，它并不会自动知道一个单词是在句子的开头还是结尾，因此需要额外的机制来编码每个元素在序列中的位置。

位置编码 通过将每个单词的位置信息 (即它在序列中的位置) 编码为一个向量，并将该向量添加到单词的嵌入表示中，从而让模型能够感知每个元素的相对或绝对位置。
{%endnote%}

经典的 Transformer 位置编码使用 正弦和余弦函数的组合，为每个位置生成的向量在不同维度上具有不同的周期性，这能够捕捉到不同级别的相对位置关系。假设输入的序列中有 N 个单词，每个单词的嵌入维度为 d，那么 Positional Encodin(PE) 的计算公式如下:

{% mathjax %}
\begin{align}
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{\frac{2i}d}}\right) \\

PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{\frac{2i}d}}\right)
\end{align}
{% endmathjax %}

其中：
- pos 是单词在序列中的位置索引 (位置从 0 开始).
- i 是位置编码的维度索引，表示该位置编码向量中的第 i 个元素。
- d 是 Embedding 的维度

这些位置编码与单词的词嵌入 (Word Embedding) 相加，最终形成输入模型的向量表示。

## (Masked) Multi-Head Attention

Multi-Head Attention (MHA) 的目的是通过并行地计算多个注意力头 (Attention Head)，从多个子空间中学习输入序列的不同表示。经过 Word Embedding 后的输入 X 形状为 Nxd. 计算步骤如下

1. 通过学习的变换矩阵将 X 映射到查询 (Q)、键 (K) 和值 (V) 空间。
{% mathjax %}
\begin{aligned}&Q=XW^{Q}\\&K=XW^{K}\\&V=XW^{V}\end{aligned}
{% endmathjax %}
其中 {% mathjax %}W^{Q},W^{K}\in\mathbb{R}d_{model}\times d_{k},W^{Q},W^{V}\in\mathbb{R}d_{model}\times d_{v}{% endmathjax %}

2. 根据 QKV 计算 Attention
每个查询向量会与所有键向量进行相似度计算 (一般采用 scaled inner product)，从而获得权重，然后利用这些权重对所有值向量进行加权求和。

{% mathjax %}
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
{% endmathjax %}

<br>在多头注意力中，为了增加模型的表达能力，通常将 Q、K 和 V 通过多个不同的线性变换矩阵进行多次计算，得到多个注意力头 (Attention Heads). 每个头的计算是独立的，但它们的结果会在最后进行拼接并经过线性变换。最终的 Multi-Head Attention 公式为：

{% mathjax %}
\text{MultiHead}(Q,K,V)=\text{Concat}(head_1,head_2,\ldots,head_h)W^O
{% endmathjax %}

<br>每个头 {% mathjax %} head_i {% endmathjax %} 计算公式为

{% mathjax %}
\text{MultiHead}(Q,K,V)=\text{Concat}(head_1,head_2,\ldots,head_h)W^O
{% endmathjax %}

<br>这里的 {% mathjax %} W^{Q}_{i},W^{K}_{i},W^{V}_{i} {% endmathjax %} 是为每个头学习到的不同权重矩阵，{% mathjax %} W^O {% endmathjax %} 是输出的线性变换矩阵。

![Multi-Head Attention](https://note.youdao.com/yws/api/personal/file/WEB85e0bf86b5d9f2c649bbc3f08c03d203?method=download&shareKey=b5e662d324237709f786beb08c27b774 "Multi-Head Attention")

Decoder 中的 Masked MHA 确保模型只能在解码序列的当前位置及其之前的位置上操作，而不能 “看到” 将要生成的未来信息。与标准的 MHA 相同，注意力分数 {% mathjax %} \mathrm{Attention Scores}=\frac{QK^T}{\sqrt{d_k}} {% endmathjax %} 是通过 Q 和 K 的点积计算得到的。计算完成后我们给其加上一个下三角元素 (包含主对角线) 为 0，上三角元素为 —∞ 的 mask，这样未来的信息经过 Softmax 后的权重为 0，被完全屏蔽。

## Grouped Query Attention（GQA）& Multi-query Attention (MQA)

[GQA](https://arxiv.org/pdf/2305.13245) 将多个 Q 分成若干组，每一组共享相同的权重矩阵。这使得每组查询可以共同处理同一个 K 和 V，降低了计算量和内存需求。在 MHA 中，所有的头共享相同的输入 X，但使用不同的投影矩阵来生成 K 和 V. GQA 中 K 和 V 通常是对输入 X 进行一次性线性变换，并在所有同一分组中的 Q 共享。MQA 更为极端，所有的 Q 共享一个 K 和 V.

![Overview of MHA, GQA & MQA](https://note.youdao.com/yws/api/personal/file/WEB3c7dc003db55abf4b8a1ebeb4aabd667?method=download&shareKey=f1570d975432b38d6f74742e9bb4cf6e "Overview of MHA, GQA & MQA")

## Multi-Head Cross Attention 

Multi-Head Cross Attention 是 Transformer Decoder 中的一个核心组件。与 Self-Attention 不同，Cross Attention 负责将解码器的隐藏状态与编码器的输出上下文信息进行交互，允许解码器的每一个解码时间步的状态 **查看整个编码器的输出**。每个解码的时间步 t，Decoder 的隐藏状态作为 Q，Encoder 的输出作为 K 和 V，计算过程与 标准的 Self-Attention 相同。

# Evolution Tree of Transformer 

后续的研究逐渐把 Encoder 和 Decoder 分离开来，形成 Encoder-Only 和 Decoder-Only 的模型。如下图所示

![Transformer Evolution Tree](https://note.youdao.com/yws/api/personal/file/WEBa2db49ee75b563db2d846dab14947060?method=download&shareKey=12514a3314f3bb4c5e30936c2d634650 "Transformer Evolution Tree")

## Feed Forward Network

FFN 是一个两层的前馈全连接网络，中间有一个非线性激活函数。第一层全连接将 {% mathjax %} d_model {% endmathjax %} 映射到 {% mathjax %} 4d_model {% endmathjax %}，经过非线性激活函数后，第二层全连接再重新映射回 {% mathjax %} d_model {% endmathjax %}.


# Decoder-Only Transformer

Decoder-Only 删除了原先 Transformer Encoder 的部分以及 Encoder 和 Decoder 进行 Cross Attention 的部分。它具有三个必要的特征:
1. 在给定编码器输入作为上下文的情况下基于迄今为止生成的 token 自动回归预测下一个。
2. 在评估对输入序列的 Q 时看不到未来值。这就是为什么仅解码器的模型通常被称为 Casual Language Model (CLM).
3. 训练模型以在给定当前输入序列的情况下预测下一个 token. 这种训练方法与回归相结合，允许模型自回归生成任意长 (最高达输入序列的最大长度) 的序列。

![Decoder-only (left) and Encoder-only (right) Transformer Architectures](https://note.youdao.com/yws/api/personal/file/WEBa6c37075488053053efa01808163d0ba?method=download&shareKey=5542015805dbda24ff7ab5dbf44a368b "Decoder-only (left) and Encoder-only (right) Transformer Architectures")

# LLaMA Transformer Architecture

LLaMA Transformer 结构如下，主要有以下变化
1. 使用 RoPE (Rotary Position Embedding) 替代传统的位置编码。
2. RMSNorm 替代 LayerNorm
3. 引入 Gated Linear Unit (GLU)

## Rotary Position Embedding

传统的 Transformer 模型使用可学习的绝对位置编码 (如 sinusoidal position embedding)，但 RoPE 采用了旋转矩阵的思想，将位置编码与输入的 token 表示直接结合，而不依赖于额外的可学习参数。

输入向量的旋转角度为 {% mathjax %} \theta(p,i)=p\cdot10000^{-2i/d} {% endmathjax %}. p 表示位置索引，i 表示维度索引，d 为向量的总维度。对于输入的 token 向量 x 中的每一对偶数和奇数维度 {% mathjax %} (x_{2i},x_{2i+1}) {% endmathjax %}，旋转操作可以用 2D 旋转矩阵表示为 

{% mathjax %} \begin{bmatrix}x_{2i}^{\prime}\\x_{2i+1}^{\prime}\end{bmatrix}=\begin{bmatrix}\cos(\theta)&-\sin(\theta)\\\sin(\theta)&\cos(\theta)\end{bmatrix}\cdot\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix} {% endmathjax %}

<br>对于输入的 token 向量 {% mathjax %} \mathbf{x}\left[x_{0},x_{1},x_{2},x_{3},\cdots,x_{d-1}\right] {% endmathjax %}, RoPE 将其两两一组配对，每一组都会与位置相关的旋转角度 θ 对应地应用旋转操作。这个过程的本质是对输入 token 的表示做了旋转变换，使得这些特征不仅依赖于输入的特征，还隐含了该 token 在序列中的位置。

![RoPE](https://note.youdao.com/yws/api/personal/file/WEBf24aca24d7ff8bc2901ca4983cbf6c47?method=download&shareKey=9ac054d415fe2e172bb8a719935d4793 "RoPE")

## RMSNorm

RMSNorm 相对于 LayerNorm 去掉了均值计算，仅基于输入的均方根进行归一化 {% mathjax %} \mathrm{RMSNorm}(\mathbf{x})=\frac{\mathbf{x}}{\mathrm{RMS}(\mathbf{x})+\epsilon}\cdot\gamma {% endmathjax %}

{% mathjax %} 
\begin{aligned}&\text{其中:}\\&\bullet\mathrm{RMS}(\mathbf{x})=\sqrt{\frac1d\sum_{i=1}^dx_i^2}\colon\text{ 输入的均方根。}\\&\bullet\:\gamma{:}\text{ 可学习的缩放参数。}\\&\bullet\:\epsilon{:}\text{ 防止除零的小常数。}\end{aligned}
{% endmathjax %}

## SiLU

SiLU (Sigmoid Linear Unit) 是一种激活函数，也称为 Swish，其定义为输入 x 和 Sigmoid 函数输出的乘积。其定义为 {% mathjax %} 
\mathrm{SiLU}(x)=x\cdot\sigma(x)
{% endmathjax %} 其中 {% mathjax %}  \sigma(x)=\frac1{1+e^{-x}} {% endmathjax %}

![SiLU](https://note.youdao.com/yws/api/personal/file/WEB552a846c520bf2b5194c621e7b8e224e?method=download&shareKey=519f3a1e4cce59da1895fa7bc2bcc842 "SiLU")