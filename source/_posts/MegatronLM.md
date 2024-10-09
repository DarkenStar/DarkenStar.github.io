---
title: Megatron-LM
date: 2024/10/02 15:51:50
categories: Distributed Training
tags: Paper Reading
excerpt: Paper reading about Megatron-LM.
mathjax: true
katex: true
---

# Abstract

&emsp;&emsp;我们的方法不需要新的编译器或更改库，与流水线模型并行 (*pipeline model parallelism*) 正交互补，并且可以通过在原生 PyTorch 中插入一些通信操作来实现。为了阐述我们的方法，使用 512 个 GPU 将基于 transformer 的模型扩展到 83 亿个参数。与可保持 39 TeraFLOPs (峰值 FLOPs 的 30%) 的强大单 GPU 基准相比，我们在整个应用中保持了 15.1 PetaFLOPs，扩展效率高达 76%.

# Introduction

&emsp;&emsp;随着 LLM 变得越来越大，它们会超出现代处理器的内存限制，并需要如激活检查点 (activation checkpoint) 等额外的内存管理技术。广泛使用的优化算法 (如ADAM) 需要每个参数额外的内存来存储动量和其他优化器状态。这减少了可以有效训练的模型的大小。模型并行性的几种方法克服了这一限制，它们对模型进行分区，使权重及其相关的优化器状态不需要并发地驻留在处理器上。

{% fold info@Activation Checkpoint %}
&emsp;&emsp;在深度学习模型的训练过程中，前向传播会计算并存储每一层的激活值，这些激活值在后向传播时被用来计算梯度。然而，对于深度很大的模型因为需要存储大量的激活值，可能会导致内存溢出。激活检查点技术通过在前向传播过程中只存储一部分的激活值来解决内存占用问题，如果在后向传播过程中需要没有存储的激活值就进行重新计算。
{% endfold %}

&emsp;&emsp;为了证明方法的可扩展性，通过在单个英伟达 V100 32GB GPU 上训练一个包含 12 亿个参数的模型来建立基准。训练该模型可维持 39 TeraFLOPs 的算力，是在 DGX-2H 服务器中配置的单个 GPU 理论峰值 FLOPS 的 30%. 在 512 个 GPU 上将模型扩展到 83 亿个参数，并采用 8 路模型并行，在整个应用中实现了高达 15.1 PetaFLOPs 的持续运行速度。与单 GPU 情况相比，扩展效率提高了 76%. 下图展示了更详细的扩展结果。

![Model (blue) and model+data (green) parallel FLOPS](https://note.youdao.com/yws/api/personal/file/WEB64c800c2db5cda251cb35df9208d8f94?method=download&shareKey=3c2e4f94cd1ca9520e5d9f49a7dfb620 "Model (blue) and model+data (green) parallel FLOPS")

# Background & Chanllenges

## Neural Language Model Pretraining
&emsp;&emsp;早期的预训练和传递语言神经表示的例子表明，与从头开始学习的词嵌入表相比，预训练的词嵌入表改善了下游任务的结果。目前的技术水平已经从传输单词嵌入表发展到传输整个数十亿参数的语言模型。这种方法的进步要求硬件、系统技术和框架能够高效地大规模运行。

## Transformer Language Models and Multi-Head Attention

&emsp;&emsp;下图展示了使用的 transformer 模型的示意图。最近利用 transformer 进行语言建模的工作，如 BERT 和 GPT-2 根据需要分别只使用编码器和解码器。
> GPT-2 和 BERT 都对多头注意和 FFN 的输入使用 GeLU 非线性和层归一化，而原始 transformer 使用 ReLU 非线性并对输出进行层归一化。

![Transformer Architecture](https://note.youdao.com/yws/api/personal/file/WEB1fcc47c83c934bf20c33fa9a88bfc34e?method=download&shareKey=2132c1442224cafae4ca86d6fd01720d "Transformer Architecture")

## Data and Model Parallelism in Deep Learning

将深度神经网络训练扩展到多硬件加速器有两种范式:
- Data Parallelism (DP): 将 batch 拆分到多个 worker
- Model Parallelism (MP): 将模型的内存使用和计算分布在多个 worker 中。
    - Pipeline Parallelism (PP): 一组操作在一个设备上执行，然后将输出传递到流水线中的下一个设备执行另一组操作。
    - Distributed Tensor Computation: 将张量运算分割到多个设备上，以加速计算或增加模型大小。

&emsp;&emsp;然而，这些技术有一个基本的限制: 模型权重必须能加载进 worker. 我们的方法是利用模型并行性在多个加速器之间分割模型。

# Model Parallel Transformers

&emsp;&emsp;我们利用 transformer 网络的结构 (self-attention 和 FFN (2*MLP) 组成)，通过添加一些同步原语，创建了一个简单的并行计算模型。下面分别阐述对 FFN 和 self-attention 的并行化。

&emsp;&emsp;FFN 第一个 MLP 由一个 GEMM，后跟一个 GeLU 非线性组成:

{% mathjax %}
Y=\text{GeLU}(XA)
{% endmathjax %}

&emsp;&emsp;并行化 GEMM 的一种选择是将权重矩阵 A 沿着行切分，并将 X 沿着其列切分:

{% mathjax %}
X=[X_1,X_2], A=\begin{bmatrix}A_1\\A_2\end{bmatrix}
{% endmathjax %}

![Row Split of Weight](https://note.youdao.com/yws/api/personal/file/WEBb1a0688321b545061bd3652261e6bf71?method=download&shareKey=2cc0ee6c275925d756b0b877c961e682 "Row Split of Weight")

&emsp;&emsp;可以得出 {% mathjax %} Y = X_1A_1+X_2A_2 {% endmathjax %}. 由于 GeLU 是非线性函数，因此这种方法需要在 GeLU 函数之前进行同步。

&emsp;&emsp;另一个选择是沿着列切分 {% mathjax %} A=\begin{bmatrix}A_1,A_2\end{bmatrix} {% endmathjax %}. 这样可以让 GeLU 独立地应用于每个 GEMM 的输出 

{% mathjax %} [Y_1, Y_2]=\begin{bmatrix}\text{GeLU}(XA_1),\text{GeLU}(XA_2)\end{bmatrix} {% endmathjax %}.

![Column Split of Weight](https://note.youdao.com/yws/api/personal/file/WEB5732c68a72330d13b288ab3d1828a6d2?method=download&shareKey=01e733100161309cbb283548474f22f7 "Column Split of Weight")

&emsp;&emsp;这种切分方式的优点是不需要进行同步操作。

&emsp;&emsp;如下图所示，以列并行方式切分第一个 GEMM，并沿着其行切分第二个GEMM。然后，在将输出传递给 dropout 层之前，第二个GEMM 的输出在 GPU 之间进行 all-reduce 操作。这种方法将 FFN 中的两个 GEMM 拆分到多个 GPU 上执行，并且只需要在正向传播 (g 操作符) 和反向传播 (f 操作符) 中分别执行一次 all-reduce 操作。

![Parallelism of MLP](https://note.youdao.com/yws/api/personal/file/WEBc1b8e9f509879d5984e2b85db312760f?method=download&shareKey=39a5f19a9a477a8d9c3a80b4c5c3bd0f "Parallelism of MLP")

&emsp;&emsp;如下图所示，利用多头注意力操作中本身存在的并行性，以列并行的方式划分与 QKV 相关的 GEMM，以便每个注意力头对应的矩阵乘法在一个 GPU 上独立完成。输出线性层的 GEMM 沿着其行并行化，并直接获取并行 attention 的输出。

![Parallelism of Self-Attention](https://note.youdao.com/yws/api/personal/file/WEBbd9aaf24f7a7a275f026d617e5d49da7?method=download&shareKey=f927c189a40038f613ca4f917effa454 "Parallelism of Self-Attention")

&emsp;&emsp;如下图所示，这使能够仅在正向传播和反向传播中分别中使用两个 all-reduce 操作执行 transformer 中所有的 GEMM.

![Parallelism of Transformer Layer](https://note.youdao.com/yws/api/personal/file/WEB471b6d468483ddff2c68c46e71ec70ee?method=download&shareKey=7e29126eacd44c304a8f9b2b25b4bbc3 "Parallelism of Transformer Layer")

&emsp;&emsp;基于 transformer 的语言模型的输出嵌入维度为隐藏层大小 (H) 乘以词汇表大小 (v). 我们沿着词汇表维度 {% mathjax %} E = \begin{bmatrix}E_1,E_2\end{bmatrix} {% endmathjax %} 并行化权重矩阵。每一块现在只包含嵌入表的一部分，输入嵌入后需要一个 all-reduce (g 算子).

&emsp;&emsp;对于输出嵌入，一种方法是通过并行 {% mathjax %} \mathrm{GEMM} [Y_{1},Y_{2}]=[XE_{1},XE_{2}] {% endmathjax %} 来获得 logits，并对结果 all-gather 后送入交叉熵损失函数。在这种情况下，all-gather 通信量为 bsv 个元素 (b 是批处理大小，s 是序列长度). 为了减小通信规模，我们将输出与交叉熵损失融合，这样通信量降为 bs.

&emsp;&emsp;我们在每个 GPU 上维护层归一化参数的副本，并在将这些张量作为输入送到下一个模型并行区域之前，在本地输出上进行 dropout 和残差连接。为了优化模型，我们允许每个模型并行 worker 优化自己的一组参数。因为所有的值要么是本地的，要么是在 GPU上 重复的，所以在这个公式中不需要通信更新的参数值。