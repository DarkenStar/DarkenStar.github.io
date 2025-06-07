---
title: PMPP Learning-Chapter 4 Compute Architecture and Scheduling
date: 2024-09-05T09:18:11+08:00
lastmod: 2024-09-05T09:18:11+08:00
draft: false
author: ["WITHER"]
keywords: 
    - CUDA
categories:
    - CUDA
tags:
    - PMPP learning
description: Personal notebook 3 of Programming Massively Parallel 
summary: Personal notebook 3 of Programming Massively Parallel  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# Compute Architecture and Scheduling

本章介绍 GPU 计算架构，并说明灵活资源分配、块调度和占用的概念。然后将深入讨论线程调度、延迟容忍、控制发散和同步。

## 4.1 Architecture of a modern GPU

下图展示了 CUDA GPU 架构，它被组织成一个流式多处理器 (*Streaming Multiprocessors, SMs*) 数组。每个 SM 都有几个处理单元，称为流处理器或 CUDA core (简称为 *core*)，如图中 SMs 内部的小块所示，它们共享控制逻辑和内存资源。

SMs 还带有不同的片上存储结构，统称为内存。GPU 还带有千兆字节的片外设备内存，称为全局内存 (*global memory*).
> 虽然旧的GPU使用 DDR DRAM，但从 NVIDIA 的 Pascal 架构开始 GPU 可能使用HBM (High-Bandwidth Memory) 或 HBM2，它们由 DRAM 模块组成，与GPU紧密集成在同一个封装中。

![Architecture of a CUDA-capable GPU](https://note.youdao.com/yws/api/personal/file/WEB4312b496c54bae36f2978ad5ef0fbe56?method=download&shareKey=6caf263b9392411f7d50e7f4d5bcaf80 "Architecture of a CUDA-capable GPU")

## 4.2 Block Scheduling

当调用内核时，CUDA runtime 系统启动执行内核代码的线程网格，**块中的所有线程同时分配给同一个的 SM**. 下图中每个 SM 分配了三个块，但是块需要占用硬件资源来执行，因此同时只能将有限数量的块分配给给定的 SM. 为了确保网格中的所有块都得到执行，runtime 系统维护一个需要执行的块列表，并在先前分配的块完成执行后再将新块分配给 SMs. 以块为基本单元将线程分配给 SMs 保证了**同一块中的线程在同一SM上同时被调度**。

![Thread Block Assignment to SMs](https://note.youdao.com/yws/api/personal/file/WEBba45a5209304777991608711b3734d55?method=download&shareKey=59a8744be11db1fad3afad00c6b06363 "Thread Block Assignment to SMs")

## 4.3 Synchronization and Transparent Scalability

CUDA 允许同一块中的线程使用 barrier 同步函数 `__syncthreads()` 来协调其行动。下图展示了屏障同步的执行情况，箭头表示线程各自执行运行的时间。弯曲线标记了每个线程开始执行 ` __syncthreads()` 的时间。弯曲线右侧的空白区域表示每个线程等待所有线程完成所需的时间。竖线标志着最后一个线程执行 ` __syncthreads()` 的时间，之后所有线程都被允许继续执行 ` __syncthreads()` 之后的代码。

不要在分支语句中使用 `__syncthreads()`
- 放在 if 语句中时，块中的所有线程要么全执行包含 `__syncthreads()` 的路径，要么都不执行。
- if-else 语句中的两个分支都存在，块中的所有线程要么全执行 if 情况下的 `__syncthreads()` 的路径，要么全执行 else 下的路径。

![A Example Execution of Barrier Synchronization](https://note.youdao.com/yws/api/personal/file/WEB973934d16ec550ef1e8998134754ea69?method=download&shareKey=eb626fd61b25664d6884d1c701e58756 "A Example Execution of Barrier Synchronization")

系统需要确保所有参与 barrier 同步的线程都能访问足够资源以到达 barrier. 否则，那些到达不了线程可能会导致死锁。因此只有当 runtime 系统确保了块中所有线程有完成执行所需的所有资源时，一个块才能开始执行。
通过禁止不同块中的线程一起执行 barrier 同步，CUDA runtime 系统可以以任何顺序执行块。如下图所示，在只有少量执行资源的系统中，一次执行两个块。反之，可以同时执行多个块。这种在不同硬件上使用不同数量的执行资源执行相同的代码的能力被称为透明可扩展性 (*transparent scalability*)

![Transparent Scalability of CUDA Programs](https://note.youdao.com/yws/api/personal/file/WEB415e750cfd1c8bd730783cf2aadeafa0?method=download&shareKey=1a0f812fee9a129ac6972abb6a59a12d "Transparent Scalability of CUDA Programs")

## 4.4 Warps and SIMD Hardware

当一个块被分配给一个 SM 时，它会被进一步划分为 32 个线程为一组的单元，称为 *warp*. 在 SMs 中，warp 是线程调度的单位。下图展示了一个划分的例子。

![Blocks are Partitioned into Warps for Thread Scheduling](https://note.youdao.com/yws/api/personal/file/WEBbc426e6de3199b6cd4706becd8760ec5?method=download&shareKey=1c78a595dc3474b5fe3314455b89f2cc "Blocks are Partitioned into Warps for Thread Scheduling")

由多维度的线程组成的块，将被投影到线性化的行主布局中来划分。线性布局是以 (z, y, x) 坐标升序的方式排列。下图展示了一个大小为 4*4 块的线性化视图。前 4 个线程的 `threadIdx.y` 为 0，它们以 `threadIdx.x` 升序的方式排列。

![Linear Layout of 2D Threads](https://note.youdao.com/yws/api/personal/file/WEBd0a03a116716e7f5420af4be591a86ad?method=download&shareKey=1d455651d0780cc68f3bfa1138a4b705 "Linear Layout of 2D Threads")

SM 是单指令多数据 (SIMD) 模型，按顺序执行所有线程，**warp 中的所有线程同时执行一条指令**。下图展示了 SM 中的内核如何被分组为处理块，其中每 8 个内核构成一个处理块 (*processing block*) 并共享一个指令获取/调度单元。同一 warp 中的线程被分配到相同的处理块，该处理块获取指令并让 warp 中的所有线程对各自负责数据的部分执行该指令。这种设计允许较小比例的硬件专注于控制，而较大比例的硬件专注于提高计算吞吐量。

![Processing Blocks Organization](https://note.youdao.com/yws/api/personal/file/WEB9402f58e22b5fbc96784b8fddd078fa6?method=download&shareKey=cad59438c3ce64bf22e7f18cd0d9591c "Processing Blocks Organization")

## 4.5 Control divergence

当同一 warp 中的线程执行不同的路径时，这些线程的行为被称作控制发散 (*control divergence*). 下图展示了一个 warp 在遇到分支语句时的执行方式，即通过两次 pass (执行代码的阶段) 来分别执行 then-path 和 else-path，最终实现所有线程的汇合。


- Pascal 及之前架构中，warp 需要顺序执行两个 pass，一个 pass 执行完才能开始下一个 pass。
  - Pass 1： 线程 0-23 执行 then-path 的代码 A，线程 24-31 处于 inactive 状态。
  - Pass 2： 线程 24-31 执行 else-path 的代码 B，线程 0-23 处于 inactive 状态。
  - Pass 3： 所有线程汇合，执行后续代码 C。
- Volta 及之后架构中，warp 可以同时执行两个 pass，不同的线程可以交错执行不同的代码路径。
  - Pass 1： 线程 0-23 开始执行 A 的第一个指令，线程 24-31 开始执行 B 的第一个指令。
  - Pass 2： 线程 0-23 执行 A 的第二个指令，线程 24-31 执行 B 的第二个指令。
  - ...
  - Pass N： 线程 0-23 执行完 A 的所有指令，线程 24-31 执行完 B 的所有指令。
  - Pass N+1： 所有线程汇合，执行后续代码 C。


![Example of a Warp Diverging at an if-else Statement](https://note.youdao.com/yws/api/personal/file/WEB2991b223f66252dc4c44389e5eb3fa54?method=download&shareKey=cb1261dd30f5d7573db9be0049648223 "Example of a Warp Diverging at an if-else Statement")

发散也可能出现在其他控制流中。下图展示了 warp 如何执行发散 for 循环。通常来说如果判断条件基于 `threadIdx` 的值，那么控制语句可能会导致线程发散。由于线程总数需要是线程块大小的倍数，而数据大小可以是任意的，因此具有线程控制发散的控制流程很常见。由以上两个例子可以看出不能假设 warp 中的所有线程都具有相同的执行时间。如果 warp 中的所有线程都必须完成执行的一个阶段，然后才能继续前进，则必须使用 barrier 同步机制 (如 `__syncwarp()` )来确保正确性。

控制发散对性能的影响随着被处理向量大小的增加而减小。例如对于长度为 100 的向量，4个 warp 中有 1 个将会控制发散 (25%)；对于大小为1000的矢量，32 个 warp 中只有 1 个将会控制发散 (3.125%).

![Example of a Warp Diverging at a for-loop](https://note.youdao.com/yws/api/personal/file/WEB46f65b52c565fcb503d299083c33932e?method=download&shareKey=56a12c21eb2f91c9ac6b3e6cefc6a6df "Example of a Warp Diverging at a for-loop")

## 4.6 Warp scheduling and latency tolerance

当将线程分配给 SMs 时，分配给 SM 的线程通常比 SM 中 core 的个数还要多，导致每个 SM 只能同时执行分配给它的所有线程的一部分。当要由 warp 执行的指令需要等待先前启动的操作的结果时，不会选择该 warp 执行。而是选择执行另一个不用等待先前指令结果的 warp。这种用其他线程的工作填充某些线程操作延迟时间的机制通常称为延迟容忍 (*latency tolerance*) 或者延迟隐藏 (*latency hiding*). 而选择准备执行的 warp 不会在执行时间线中引入任何空闲或浪费的时间的策略被称为零开销线程调度 (*zero-overhead thread scheduling*). 这种容忍长操作延迟的能力是 GPU 不像 CPU 那样为缓存和分支预测机制分配那么多芯片面积的主要原因，因此可以更专注于浮点数计算和内存读取。

{{< details title="Threads, Context-switching, and Zero-overhead Scheduling" >}}

之前介绍过线程由程序的代码、正在执行的代码中的指令、变量的值和数据结构组成。在基于冯·诺伊曼模型的计算机中，程序的代码存储在存储器中。PC (Program Counter) 跟踪正在执行的程序指令的地址。IR (Instruction Register) 保存正在执行的指令。寄存器和内存保存变量和数据结构的值。
现代处理器的设计允许上下文切换 (*Context-switching*)，多个线程可以通过轮流执行的方式分时复用一个处理器。通过保存和恢复 PC 值以及寄存器和内存的内容，可以暂停线程的执行，并在稍后正确恢复线程的执行。不过保存和恢复寄存器内容可能会增加大量执行时间。
传统的 CPU 从一个线程切换到另一个线程需要将执行状态 (例如被切换线程的寄存器内容) 保存到<font color="red;">**内存**</font>中，稍后再从内存中加载，这样会产生空闲周期。GPU SMs 通过在硬件<font color="red;">**寄存器**</font>中保存指定 warp 的所有执行状态来实现零开销调度，因此不需要保存和恢复状态。
{{< /details >}}

## 4.7 Resource partitioning and occupancy

给 SM 分配其所支持的最大 warp 数并不总是可行。分配给 SM 的 warp 数量与其支持的 warp 数量之比称为占用率 (*occupancy*). 例如，Ampere A100 GPU 每个 SM 最多支持 32 个 block，每个 SM 最多支持 64 个 warp (2048 个线程)，每个 block 最多支持 1024 个线程。意味着块大小可以从 64~1024 不等，每个 SM 分别可以有 32~2 个块。在这些情况下，分配给SM的线程总数为2048，这使占用率最大化。
SM 中的执行资源包括寄存器、共享内存线程块槽 (每个 SM 最大能被分配的线程块数量) 和线程槽 (每个线程块最大能被分配的线程数量)，这些资源在线程之间动态分配。资源的动态分配可能导致他们之间相互制约，使得资源利用不足。
- 硬件资源支持的影响。当每个块有32个线程时。Ampere A100 GPU 会将 2048 个线程槽分配给 64 个块。然而 Volta SM 只支持 32 个线程块槽，导致占用率只有 50%.
- 当每个块的最大线程数不能整除块大小时。当块大小为 768，SM 将只能容纳 2 个线程块 (1536个线程)，剩下512个线程槽未使用，占用率为 75%.
- 寄存器资源限制对占用率的影响。Ampere A100 GPU 允许每个 SM 最多占有 65,536个寄存器。为了达到满占用率每个线程不应该使用超过 32 个寄存器。
这种限制导致资源使用的轻微增加可能导致并行性和性能的显著降低，称为 performance cliff.