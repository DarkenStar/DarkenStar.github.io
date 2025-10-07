---
title: "APTMOE"
date: 2025-10-06T08:40:16+08:00
lastmod: 2025-10-06T08:40:16+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- MoE

keywords:
- MoE

description: "Paper reading of APTMOE." # 文章描述，与搜索优化相关
summary: "Paper reading of APTMOE." # 文章简单描述，会展示在主页
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

APTMoE 采用亲和力感知 (affinity-aware) 流水线并行性在带宽受限的GPU节点上微调MoE模型。我们提出了一种亲和力感知的卸载技术，该技术可以提高计算效率和模型大小的流水线并行性，并受益于分层加载策略和需求优先级调度策略。为了提高计算效率和减少数据移动量，分层加载策略设计了三个加载阶段，并利用expert popularity (专家流行度，即输入 token 偏好某些专家的 skewed distribution，偏斜分布) 和 computation affinity (计算亲和力，即不同专家对CPU/GPU的计算效率差异) 在 GPU 和 CPU 之间分配计算。需求优先调度策略以减轻三个加载阶段之间的相互干扰和最大限度地提高带宽利用率为目标，主动、动态地协调加载执行顺序。实验表明，在大多数情况下，APTMoE优于现有的方法。特别是，APTMoE在4个Nvidia A800 GPU (40GB) 上成功微调了612B MoE模型，与SOTA方法相比，吞吐量提高了33%。

# Introduction

![Fig. 1. Forward process with pipeline parallelism and offloading technique.](https://share.note.youdao.com/yws/api/personal/file/WEB5c3fcc657964f67a83e8f1674417dbca?method=download&shareKey=96cb4a230b6c96f4366081cb7fba7646 "Fig. 1. Forward process with pipeline parallelism and offloading technique.")

为了克服通信带宽和内存容量的固有硬件限制，以前的方法通过采用流水线并行和卸载技术在可负担的设备上微调大规模模型。然而，数据量与计算量之比的增加使得流水线并行和卸载技术的结合效率降低。图1说明了现有的组合。在每次微调迭代中，每个阶段都是从主机内存加载的。对于MoE架构，需要传输的数据量明显增加，而计算需求保持相对稳定。由于这种不平衡，加载过程的重叠程度大大降低，阻塞了计算。反向传播进程以类似的方式执行。

APTMoE 是一个在带宽受限的GPU节点上用于MoE模型的亲和感知流水线微调系统。APTMoE 提出了亲和力感知的卸载技术来增强管道并行性微调。其核心思想是*基于亲和力在 GPU 和 CPU 之间分配计算，从而提高计算效率，更好地跨异构内存管理数据*。

分层加载策略旨在确定计算在 GPU 和 CPU 之间的分配，并管理不同的加载决策。由于真正的专家人气要等到门操作完成后才能确定，往往会错过重叠的机会。为了克服这一问题，分层加载策略设计了三个加载阶段，
- Inter-stage Loading: 为了在不同流水线阶段之间重叠计算和加载，利用历史专家流行度贪婪地分配计算到最高亲和力的 GPU 上。其他一些加载决策被推迟，直到获得更准确的专家知名度。
- Inter-layer Loading: 为了提前利用专家的流行度，我们使用了一个预测器来预测后续层的流行度分布。利用预测的专家人气，使同一管道阶段的层间加载和计算重叠。根据预测，我们可以为具有高激活密度的专家做出加载决策，并在 GPU 上处理它们，而那些具有低激活密度的专家则留在 CPU 上执行。
- Inter-expert Loading: 重叠同一层不同专家的加载和计算，依靠实时专家人气。通过这三个加载阶段，亲和性感知卸载技术可以更好地在 GPU 和 CPU 之间分配计算。

虽然三个加载阶段中的每一个都识别不同的重叠空间，但它们都依赖于相同的PCIe带宽，从而导致相互干扰。此外，在同一方向上传输数据的内存复制内核不能并发执行，因此这些加载阶段是顺序运行的，可能会相互阻塞。因此，需求优先调度策略通过动态协调这些加载阶段的顺序来解决上述问题。程序定期向GPU查询加载进程状态，并在运行时动态确定加载顺序。

本文贡献如下:
- 在对MoE架构应用现有的经济有效的微调方法时，确定了由于数据与计算的比例增加而引起的计算阻塞问题。
- 提出APTMoE，一种针对带宽受限的GPU节点的MoE模型的亲和力感知管道微调系统，其关键思想是将部分亲和性计算转移到CPU上，以便更好地跨异构内存管理数据。APTMoE结合了分层加载策略和需求优先级调度策略。
- 提出了分层加载策略。利用专家知名度和计算亲和力的先验知识，设计了三个加载阶段，以最高的亲和力贪婪地分配计算量，使数据移动量最小化。
- 提出需求优先调度策略，通过动态协调加载顺序，缓解加载阶段之间的相互干扰，实现带宽利用率最大化。

# Background and Motivation

![Fig. 3. GPU nodes with different connections.](https://share.note.youdao.com/yws/api/personal/file/WEB86a444ed932cae708bba09cc33fa1c56?method=download&shareKey=50826efbcbf8e87281ce1c45e85a74b1 "Fig. 3. GPU nodes with different connections.")

预训练阶段通常在大规模集群上进行，通常利用数百个通过高速网络结构连接的GPU节点，如图3(a)中的NVLink。进入微调阶段，大多数人可以使用的硬件通常由单个节点或配备多个 GPU 的多个节点组成。如图3(b)所示，这些节点通常缺乏高速互连，GPU 间通信依赖于传统的PCIe总线，带宽受到限制。

![Fig. 4. Existing pipeline parallelism approaches. Fi,j , Bi,j denote the i-th stage’s forward/backward execution on the j-th micro-batch respectively.](https://share.note.youdao.com/yws/api/personal/file/WEB8e5cc8a4f021385ee74cb9c4a0b5aced?method=download&shareKey=54a66c2a3f1374e8cd261a8f5b3e37ac "Fig. 4. Existing pipeline parallelism approaches. Fi,j , Bi,j denote the i-th stage’s forward/backward execution on the j-th micro-batch respectively.")

Mobius不把模型简单分成一个stage一个GPU，而是将整个模型切分成更多stage，每个stage包含多个连续层。这些stages初始存储在主机DRAM，而非全塞进GPU内存。相邻stages映射到不同GPU避免同一GPU或同一CPU root complex 下的争用。通过计算 ontention degree 来优化映射，减少PCIe带宽瓶颈。

- Increased Ratio of Data to Computation: MoE 中每个输入 token 将被路由到k个专家。通常，k远小于该层中专家的数量。因此，与密集模型相比，MoE模型中的数据与计算的比率显著增加。在将流水线并行和卸载技术应用于大规模模型微调时，数据加载过程可能会阻塞现有方法中的计算。

- Expert popularity: 与密集模型相比，MoE模型的一个显著特征是专家受欢迎程度，输入 token将被路由到不同的专家，并且分布通常是倾斜的。在MoE微调期间，大多数输入 token 将选择一小部分专家，特别是对特定领域数据集的微调。此外，之前的研究也发现了少数专家在一段时间内总是被高强度激活，我们称之为历史专家的受欢迎程度。

![The time for loading, CPU computing, and GPU computing of a single expert, conducted on Intel Xeon Gold 6348 CPU with 28 Cores and Nvidia A800 GPU. d and h represent dimensions of linear layers in the expert.](https://share.note.youdao.com/yws/api/personal/file/WEB70d3e2cd4e6c87a5abf1c5c48bf123d3?method=download&shareKey=bae6ccf851d1e26438ec90f0c70d20ff "The time for loading, CPU computing, and GPU computing of a single expert, conducted on Intel Xeon Gold 6348 CPU with 28 Cores and Nvidia A800 GPU. d and h represent dimensions of linear layers in the expert.")

- Computation Affinity: 利用专家之间不平衡的受欢迎程度，不同专家的激活发生不均匀，导致计算强度的显着变化。如图6所示，当输入 token 数量较大时，GPU的性能明显优于CPU. 然而，当输入 token 的数量较少时，计算变得不那么受计算限制，并且CPU 更适合处理这种工作负载。因此，我们可以将高需求的专家分配给 GPU 并利用它们的并行处理能力，同时将低强度的专家分配给 CPU 并且这样不需要将其加载到GPU内存中，从而减少了数据移动量。

# The Design of APTMoE

![Fig. 7. The workflow of APTMoE system. The static part runs the profiler and generate the corresponding memory and time results, and the runtime part takes the affinity-aware offloading.](https://share.note.youdao.com/yws/api/personal/file/WEBd1602e6ef7eab8045172d8c055d722f9?method=download&shareKey=40fb816e4eaf517c134fcc0d183adc63 "Fig. 7. The workflow of APTMoE system. The static part runs the profiler and generate the corresponding memory and time results, and the runtime part takes the affinity-aware offloading.")

如图7所示，APTMoE系统可以分为两个部分。**静态部分**: 在给定MoE模型、参数设置和目标GPU节点的情况下，分析器在CPU和GPU上执行单个MoE层的微调。由于输入序列长度和批大小通常是固定的，因此 MHA 的工作负载在整个微调阶段保持不变。相反，由于gate操作动态地决定了 token 到专家的路由，因此分析器需要遍历单个专家，并将所有可能数量的 token 作为输入。只运行一个层来执行微调步骤，分析时间不会很长。此外，静态部分脱机运行，因此不会产生运行时开销。将计算分配给CPU或GPU也受到数据移动时间的影响，我们分析了单个MHA、单个 gate 操作和单个专家的数据移动时间。存储这些分析结果以确定计算关联。**运行时部分**: APTMoE采用亲和力感知的卸载策略，包括分层加载策略和需求优先级调度策略，详细说明如下。

## Affinity-aware Offloading on Pipeline Parallelism

![Fig. 5. Different offloading techniques.](https://share.note.youdao.com/yws/api/personal/file/WEB2fe4b4f459aeb8945e88b88855fb13f8?method=download&shareKey=fdb801f328398c0e2abbfb06ec93c28b "Fig. 5. Different offloading techniques.")

如图5(b)所示，亲和力感知的卸载可以从跨 GPU 和 CPU 分配计算中获益。对于给定的输入，利用专家流行度和计算亲和力，亲和力感知卸载将决定哪一部分应该在CPU上进行，哪一部分应该在GPU上进行。相应地，亲和力感知卸载相应地调度数据移动。其中，亲和力感知卸载采用分层加载策略来确定计算分配给设备并管理加载决策。

与图5(a)中的基本卸载技术相比，除了通过利用 CPU 减少 GPU 的负担外，它还可以加载和卸载更少的参数、激活和梯度。亲和力感知卸载采用需求优先级调度策略，减轻分层加载策略之间的相互干扰，提高主机内存和GPU内存之间的带宽利用率。

## Hierarchical Loading Strategy

![Fig. 8. Three loading phases of the hierarchical loading strategy.](https://share.note.youdao.com/yws/api/personal/file/WEBc1fba784fdf9ab5a02d83eaa0125a964?method=download&shareKey=d4c49b51b2ebe6a5cd9edfde350dea82 "Fig. 8. Three loading phases of the hierarchical loading strategy.")

加载策略分为前向和反向进程。对于前向过程，真正的专家人气在门控操作执行之前是未知的。相反，反向过程中所有真正的专家人气都是已知，从而允许预分配。

**Loading Decision Management:** 分层加载策略在模型块 (MHA、gate 操作和专家) 的粒度上管理加载决策。每一个阶段都使用了三个队列来管理加载决策，并为它们分配了从低到高的不同优先级。此外，这些队列将用于需求优先级调度策略，以协调加载执行。

**Inter-stage Loading:** 确定当前阶段的计算与后续阶段的加载之间的重叠空间。根据计算需求贪婪地加载部分数据。倾向于优先考虑极有可能表现出高计算强度的模型块。因此，MHA 和 gate 操作需要处理所有输入数据，在阶段间阶段具有固有的优先级。历史专家受欢迎程度表明，一些专家总是在时间段内被高强度激活。阶段间阶段以受欢迎程度从高到低加载在前一个迭代中高度活跃的专家。对于在此阶段未加载的 MHA 和 gate 操作和专家，将其加载决策推迟到后续阶段。

**Inter-layer Loading:** 同一stage内不同layers间加载的重叠空间。由于 gate 操作位于专家之前，真正的专家受欢迎程度在 gate 操作完成之前仍然是未知的。为了解决前向过程中的这一限制，我们引入了一个额外的专家人气预测器，预测器独立运行，不改变原始的 MoE 模型。

![Fig. 9. Design of the popularity predictor.](https://share.note.youdao.com/yws/api/personal/file/WEB31a8b85a83f2509c088ad2fb6a8e72c9?method=download&shareKey=299f074b6db499ffa298411ab9ac8573 "Fig. 9. Design of the popularity predictor.")

预测器采用与门操作相同的结构，首先使用对应 gate 操作的权重，然后进行训练以获得更好的预测。中间结果分别进入预测器和 gate，生成预测的专家人气和实际的专家人气。根据预测的专家流行度， 基于公式1在 GPU 和 CPU 之间预分配计算。

$$
R=\frac{\sum_{low}^{high}Comp_{cpu}}{Load_{MHA}+Load_{Gate}+\sum_{high}^{low}Load_{expert}}\tag{1}
$$

R 的分子为计算强度由低到高排列的所有执行微批次中CPU执行专家的预测时间之和。R 的分母为 MHA 加载时间、gate 操作和专家累计加载时间，计算强度由高到低排列。停止加载的阈值 R = 1. 一旦产生新的微批次的专家流行度，将重新调度。一旦当前微批中出现新的高需求专家，我们通过添加或删除层间队列的名称来修改层间加载决策以满足新的计算需求。

**Inter-expert Loading:** 同一layer内不同experts间的重叠空间。仍用公式 1 来判断是否加载专家，只不过这里使用的是 gate 操作后得到的实时 CPU 执行时间。

**Backward Loading Strategy:** 反向传播过程中所有准确的专家流行度都已经已知，可以在反向传播之前预先确定阶段间阶段所有专家的分配，而不需要层间和专家间加载阶段。

## Demand-priority Scheduling Strategy

![Fig. 10. The proactive loading coordination.](https://share.note.youdao.com/yws/api/personal/file/WEB385a98923b91577415459700462bb6a7?method=download&shareKey=bc9db05e2f1f4f6044c2d295d9a8952f "Fig. 10. The proactive loading coordination.")

图10说明了需求优先级调度策略，程序定期向GPU查询加载进程状态，并在运行时动态确定加载顺序。在执行期间，三个加载阶段动态地将模型块的名称添加或删除到三个队列中。需求优先级调度策略将优先级最高的阶段分配给专家间阶段，其次是层间阶段，最后是阶段间阶段。

一旦管道阶段切换，阶段间阶段就会对多个模型块进行加载决策。层间阶段只对下一层的几个模型块进行加载决策，而专家间阶段对当前层的较少专家进行加载决策。因此，加载决策时刻和加载执行时刻是不一致的。在某些情况下，这些加载执行应该是交错的。因此，实时动态调度这些阶段的加载执行变得至关重要。

由于中断和恢复内核非常困难，因此我们选择在数据移动的内核启动之前对其进行调度，简化了整个调度过程。如图10所示，一开始从加载阶段的队列中选择预先确定数量的数据移动操作，并将它们启动到GPU中的加载流中。

为了查询GPU加载状态，我们还合并了一个cuda事件，并利用CPU-GPU同步来查询事件是否被触发。cuda事件被插入到最后一个动作之前的位置。确保当CUDA事件被触发时，数据加载内核仍在执行，有效地隐藏了内核启动开销。此外，需求优先调度策略还负责确保数据依赖关系的正确性。为了保证在计算发生时数据已经移动到GPU内存中，在连续的数据加载内核后插入一个CUDA事件，并使用 inter-stream 同步来通知相应的计算内核数据已经准备好。