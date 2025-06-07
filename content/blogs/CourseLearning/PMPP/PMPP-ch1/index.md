---
title: PMPP Learning-Chapter 1 Introduction
date: 2024-09-03T21:20:12+08:00
lastmod: 2024-09-03T21:20:12+08:00
draft: false
author: ["WITHER"]
keywords: 
    - CUDA
categories:
    - CUDA
tags:
    - PMPP learning
description: Personal notebook 1 of Programming Massively Parallel 
summary: Personal notebook 1 of Programming Massively Parallel  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# 1 Introduction

基于单个中央处理器 (Central Processor Unit, CPU) 的微处理器外部看起来是按顺序执行指令，例如英特尔和 AMD 的 x86 处理器，随着时钟频率和硬件资源的快速增长，在20世纪80年代和90年代推动了计算机应用程序的性能快速提高和成本降低。可以给桌面应用提供 GFLOPS 级别的浮点运算，给数据中心提供 TFLOPS 级别的浮点运算。然而，由于能源消耗和散热问题，这种趋势从2003年开始放缓。这些问题限制了时钟频率的增加和保持按顺序步骤执行指令的同时在单个 CPU 上每个时钟周期内可以执行的行动。
之后几乎所有的微处理器供应商都转向了在每个芯片上使用多个物理 CPU (称为处理器核心) 来提高处理能力。在这个模型中，传统的CPU可以看作是一个单核CPU。这样就要求必须有多个指令序列并且可以同时在这些处理器核心上执行 (无论是来自相同的应用程序还是来自不同的应用程序)。为了使一个特定的应用程序受益于多个处理器核心，它的工作必须分成多个指令序列，这些指令序列可以同时在这些处理器核心上执行。这种从单个CPU按顺序执行指令到多个内核并行执行多个指令序列的转变造就了并行计算的需求。

## 1.1 Heterogeneous parallel computing

半导体行业确定了设计微处理器的两条主要路线

- *Multicore* Trajectory: 寻求在转变到多个核时保持顺序程序的执行速度。
- *Many-thread* Trajectory: 更多地关注并行应用程序的执行吞吐量。

自2003年以来，多线程处理器尤其是 GPU，一直在浮点计算性能上表现优异。多核和多线程之间在峰值性能上的如此大的差距促使许多应用程序开发人员将其软件的计算密集型部分转移到gpu上执行。

|                         | 64-bit double-precision | 32-bit single-precision |
| ----------------------- | ----------------------- | ----------------------- |
| Tesla A100 GPU          | 9.7 TFLOPS              | 156 TFLOPS              |
| Intel 24-core Processor | 0.33 TLOPS              | 0.66 TLOPS              |

CPU 的设计为面向延迟的 (*latency-oriented*) 设计。针对顺序代码性能进行了优化。计算单元和操作数传输逻辑的设计是为了最小化计算的有效延迟，代价是增加芯片面积和单位功率的使用。采用复杂的分支预测逻辑和执行控制逻辑来减少条件分支指令的延迟使得每个线程的执行延迟降低。然而，低延迟计算单元、复杂的操作数传递逻辑、大缓存存储器和控制逻辑消耗了芯片面积和功率，否则可以用来提供更多的算术执行单元和内存访问通道。
GPU 的设计是面向吞吐量 (*throught-put oriented*)的设计。寻求在有限的芯片面积和功耗预算下最大化浮点计算和内存访问吞吐量。许多图形应用程序的速度受到数据从内存系统传输到处理器的速率的限制，必须能够将大量数据加载和存储到 DRAM 中的图形帧缓冲区。
游戏应用程序普遍接受的宽松内存模型(各种系统软件，应用程序和I/O设备期望其内存访问工作的方式)也使 GPU 更容易支持访问内存的大规模并行性。通用处理器必须满足遗留操作系统、应用程序和I/O设备的要求，这些要求对支持并行内存访问提出了更多挑战，从而使提高内存访问的吞吐量 (通常称为内存带宽 *memory bandwidth*) 变得更加困难。
就功耗和芯片面积而言，减少延迟比增加吞吐量要昂贵得多[^1]。因此，GPU 的主流解决方案是针对大量线程的执行吞吐量进行优化，**而不是减少单个线程的延迟**。这种设计方法允许分级存储层次和计算具有较长的延迟，从而节省了芯片面积和功耗。

![CPU and GPU Design Philosophies](https://note.youdao.com/yws/api/personal/file/WEB0619836cbd0c830367d16469ab356a2e?method=download&shareKey=f86f3077eb42bd1e9ca6ed4c31c18a65 "CPU and GPU Design Philosophies")

## 1.2 Why More Speed or Parallelism

基于人工神经网络的深度学习是通过大幅提高计算吞吐量而实现的新应用。虽然自 20 世纪 70 年代以来，神经网络得到了积极的关注，但由于需要太多的标记数据和太多的计算来训练这些网络，它们在实际应用中一直效果不佳。互联网的兴起提供了大量有标签的图片，而 GPU 的兴起则带来了计算吞吐量的激增。因此，自2012年以来，基于神经网络的应用在计算机视觉和自然语言处理方面得到了快速的采用。这种采用彻底改变了计算机视觉和自然语言处理应用，并引发了自动驾驶汽车和家庭辅助设备的快速发展。

## 1.3 Speeding up real applications

并行计算系统相对于串行计算系统所能实现的加速的一个重要因素是可以并行化的应用程序部分，另一个重要因素是从内存访问数据和向内存写入数据的速度有多快。下图展示了顺序和并行应用程序部分的覆盖率。顺序部分和传统的(单核)CPU覆盖部分相互重叠。以前的GPGPU技术对数据并行部分的覆盖非常有限，因为它仅限于可以表示为绘制像素的计算。障碍是指难以扩展单核cpu以覆盖更多数据并行部分的功率限制。

![Coverage of Application Portions](https://note.youdao.com/yws/api/personal/file/WEBfc0b86a42c4ed9223a9b6539c92712fc?method=download&shareKey=796ebc8414ada67e650c087e44aa66a9 "Coverage of Application Portions")

## 1.4 Challenges in parallel programming

1. 设计具有与顺序算法相同的算法(计算)复杂度的并行算法可能具有挑战性。
2. 许多应用程序的执行速度受到内存访问延迟和/或吞吐量的限制。
3. 与顺序程序相比，并行程序的执行速度通常对输入数据特征更为敏感。
4. 有些应用程序可以并行化，而不需要跨不同线程的协作 (*embarrassingly parallel*)。其他应用程序需要使用同步操作 (*synchronization operations*) 使得线程能相互协作。

---

[^1]: 例如，可以通过将计算单元的数量翻倍来使吞吐量翻倍，但代价是芯片面积和功耗翻倍。然而，将算术延迟减少一半可能需要电流翻倍，代价是使用的芯片面积增加一倍以上，功耗变为四倍。
