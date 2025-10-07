---
title: "HybridMoE"
date: 2025-10-04T17:18:08+08:00
lastmod: 2025-10-04T17:18:08+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- MoE

keywords:
- MoE

description: "Paper reading of HybridMoE." # 文章描述，与搜索优化相关
summary: "Paper reading of HybridMoE." # 文章简单描述，会展示在主页
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

CPU-GPU 混合推理利用CPU计算减少专家负载开销，但面临的主要挑战有
1. MoE模型的专家激活模式高度不稳定，使得现有工作中的固定映射策略效率低下。
2. 由于不同的专家规模、结构、不均匀的工作负载分布等，MoE的混合CPU-GPU调度本身就很复杂。

为了解决这些挑战，在本文中，作者提出了HybriMoE，通过一种新的CPU-GPU调度和缓存管理系统来提高资源利用率。HybriMoE引入了
1. 一种动态的层内调度策略来平衡 CPU 和 GPU 之间的工作负载。
2. 一种影响驱动的层间预取算法。
3. 一种基于分数的缓存算法来减轻专家激活的不稳定性。

# 1. Introduction

MoE 引入了大量内存需求，这对内存资源有限的边缘设备上的部署构成了特别的挑战。为了缓解这个问题，专家卸载技术将专家权重存储在辅助存储器中，例如CPU内存或SSD，并根据需要通过PCIE将它们加载到GPU内存中。在这种卸载场景中，主要的瓶颈是由庞大的通信规模和有限的带宽驱动的与按需加载相关的开销。

![Fig. 1. Execution timeline of three scenarios. Expert computation time on the GPU remains constant, while CPU execution time increases linearly with workload. The balanced scheduling in (c) achieves improved utilization and reduces overall execution time.](https://share.note.youdao.com/yws/api/personal/file/WEBa8c0cc47afb3a87ed286eca150ad47dd?method=download&shareKey=cdbe739301169c904c235aa8b725291a "Fig. 1. Execution timeline of three scenarios. Expert computation time on the GPU remains constant, while CPU execution time increases linearly with workload. The balanced scheduling in (c) achieves improved utilization and reduces overall execution time.")

以前在其他卸载场景中的工作已经进一步探索了利用CPU计算来降低内存传输的频率。Fiddler 和kTransformers，如图1所示，当发生缓存未命中时，CPU处理相应的专家计算，而不是将该层传递给GPU，从而减少了数据传输开销。

虽然CPU计算对于传统的推理任务是有效的，但 MoE 模型中的专家激活通常不太倾斜，并且在迭代中表现出显著的可变性，这使得很难预测哪些专家将被激活。这种动态行为使CPU和GPU之间的工作负载平衡变得复杂，因为静态任务分配策略无法适应工作负载分布的实时变化。然而，现有的解决方案依赖于基于历史激活频率的固定映射策略，忽略了MoE推理的动态性和不可预测性。如图1(b)和(c)所示，这些限制导致资源利用率不够理想，并增加了推理延迟。

Hybrid MoE的主要贡献如下：
- **混合 MoE CPU-GPU调度 (Hybrid MoE CPU-GPU Scheduling)**: 一种高效的MoE推理混合调度算法，动态平衡 GPU 和 CPU 之间的工作负载，优化资源利用率，通过优先任务执行和数据传输管理最小化延迟。
- **影响驱动的预取 (Impact-driven Prefetching)**: 利用残差连接的隐藏状态相似性预测并优先预取高影响专家.
- **MoE 专用缓存管理 (MoE-specialized Cache Management)**: 引入Minus Recent Score (MRS) 策略，根据路由分数更新专家优先级，避免LFU/LRU的局限。
- 系统实现: 在ktransformer框架之上实现了HybriMoE. 在三个流行的基于MoE的LLM和各种平台上评估了HybriMoE. 与现有的混合调度方法相比，HybriMoE在预填充和解码阶段分别实现了1.33倍和1.70倍的加速。

# 2. Background

![Fig. 2. An example of MoE architecture with shared and routed experts.](https://share.note.youdao.com/yws/api/personal/file/WEB911172ec63d56469e48cccc150fade49?method=download&shareKey=6f847b384031b013bc105027f1f86979 "Fig. 2. An example of MoE architecture with shared and routed experts.")

为了适应 MoE 模型中的稀疏激活模式，引入了几种专门的技术，包括高级门控、预取和量化策略。这些方法的目标是最小化按需加载开销，减少不必要的内存传输并提高整体性能。

# 3. Motivation

![Fig. 3. (a) Cumulative activation frequency(CDF) for neurons and experts, (b) Reuse probability of experts by score, suggesting cache optimization opportunities, (c) Expert workload distribution of DeepSeek in a prefill forward, (d) Latency of prefill 128 tokens for Qwen2(Q), Mixtral(M) and decode 10 tokens for Mixtral with three existing methods, (e) CPU vs. GPU time for varying numbers of experts at fixed load, with CPU benefiting from overlapping computations. (f) CPU and GPU time across workload sizes.](https://share.note.youdao.com/yws/api/personal/file/WEB2ebd4355ac5b1669ca007ee4f38c0003?method=download&shareKey=19d3a29b8a4d856a57be47a806b9fa32 "Fig. 3. (a) Cumulative activation frequency(CDF) for neurons and experts, (b) Reuse probability of experts by score, suggesting cache optimization opportunities, (c) Expert workload distribution of DeepSeek in a prefill forward, (d) Latency of prefill 128 tokens for Qwen2(Q), Mixtral(M) and decode 10 tokens for Mixtral with three existing methods, (e) CPU vs. GPU time for varying numbers of experts at fixed load, with CPU benefiting from overlapping computations. (f) CPU and GPU time across workload sizes.")

挑战 1: MoE 模型具有不可预测的激活模式，专家以动态和频繁变化的方式被激活。如图3(a)所示，与神经元级的稀疏度相比，MoE的激活频率分布更加均匀，这使得预测未来的专家使用情况变得更加困难。

尽管MoE激活不稳定，但专家激活的时间相关性为缓存优化提供了基础：如图3(b)所示，激活分数较高的专家更有可能在下一次迭代中被重用，这表明在缓存中保留高分专家可以减少访问延迟。此外，MoE模型通常在相邻层之间表现出高度的激活相似性，这可以用于预取。

挑战 2: MoE结构的复杂性与动态调度。现有的固定映射方法经常导致负载不平衡和资源未充分利用。随着共享专家的使用、专家的大小和数量以及运行时缓存行为的变化，MoE模型的不同结构进一步增加了调度复杂性。此外，预填充阶段的负载分布不均和执行顺序多变，使得高效调度更具挑战性，如图3(c)所示。考虑到需要逐层调整，静态最优解决方案是不切实际的，这使得实时调度成为一个重大挑战。

尽管调度问题具有NPHard性质，但在CPU-GPU系统上的MoE推理下，几个关键的现象可以指导高效调度规则的设计。
1. 专家迁移时间保持相对稳定，简化了决策。
2. GPU的计算时间与激活的专家数量呈线性增长，而CPU的计算则受益于内存访问和计算的重叠，因为它的缓存更大。在图3(e)中，CPU上的第一个专家计算较慢，但后续任务的处理速度更快，缓存利用率更高。同样，图3(f)显示GPU时间随着工作负载的增加保持稳定，而CPU时间随着工作负载的增加呈线性增长。利用这些模式，预定义的调度规则可以帮助实现MoE模型的高效工作负载平衡。

# 4. Hybrid Design

![Fig. 4. Overview of HybriMoE.](https://share.note.youdao.com/yws/api/personal/file/WEBd670330b691a41e00634644bf2d046e4?method=download&shareKey=31ba7662dddc4d7942f8f889cbe5243d "Fig. 4. Overview of HybriMoE.")

系统从预热阶段开始，收集基本的性能指标，如CPU和GPU处理速度和数据传输延迟。在推理过程中，HybriMoE利用这些信息来实现CPU-GPU混合调度、分数感知缓存和影响驱动的预取，从而确保在整个推理过程中高效地执行任务并优化资源。

![Fig. 5. An example of hybrid scheduling. The CPU computes the cached expert E while the GPU computes the uncached expert C to achieve better hardware utilization.](https://share.note.youdao.com/yws/api/personal/file/WEBf48f14a1b7b7e72a02261b9fddfdbf8a?method=download&shareKey=b813671542af6fa9cefa129a720aba87 "Fig. 5. An example of hybrid scheduling. The CPU computes the cached expert E while the GPU computes the uncached expert C to achieve better hardware utilization.")

由于专家激活的动态性和跨异构资源平衡工作负载的需要，HybriMoE提出了一种混合调度策略，通过引入三个关键优先级规则来简化任务到硬件的映射：
- GPU 优先级：GPU首先执行高负载缓存专家的计算。
- CPU 优先级：CPU 优先考虑未缓存专家的计算，专注于低负载任务的高效执行。此外，CPU可以在空闲时按照从低到高的负载顺序处理缓存的专家。
- 传输优先级：优先考虑从 CPU 移动高负载未缓存专家到 GPU，以最小化计算延迟。

基于这些优先级规则，HybriMoE将所有激活的专家划分为GPU队列和CPU队列。
- GPU队列包含GPU上的缓存专家，按负载降序排序。
- CPU队列包含CPU上未缓存的专家，按负载升序排序。

## B. Hybrid Scheduling Strategy

**在实际执行之前，HybriMoE执行一个模拟阶段来评估调度策略**。此模拟通过迭代地填充CPU计算、GPU计算和数据传输时间线来近似执行过程，使系统能够确定调度配置，从而在平衡异构硬件之间的资源利用率的同时最小化总体延迟。在模拟的每一步中，系统都会选择完成时间最早的时间线，执行相应的操作——CPU或GPU上的计算任务，或者通过PCIE传输数据。

如果将专家从 CPU 转移到 GPU，则按照负载降序插入到GPU队列中，确保高负载任务优先用于GPU计算。这种迭代模拟一直持续到计算出所有专家为止。调度过程通过图5中的一个示例加以说明。假设GPU的计算时间是恒定的，CPU的计算时间与专家的负载成正比，传输时间固定为3个单位。HybriMoE中的调度算法确定了一种最优策略，即CPU计算缓存的专家E，GPU处理未缓存的专家 C，有效地提高了硬件利用率。

## C. Impact-driven prefetching

![Fig. 6. Impact-driven prefetch workflow.](https://share.note.youdao.com/yws/api/personal/file/WEB5ef0109eb67cfe0f08c131e1e37537b0?method=download&shareKey=5417382ea15885308f544d7aa2155db3 "Fig. 6. Impact-driven prefetch workflow.")

HybriMoE通过重用来自这些层的门控信息来预测接下来三层的专家激活，如图6所示。该预测指导预取机制，使系统能够有效地预加载在后续计算中可能被激活的专家。

## D. Score-aware Caching

LFU 和 LRU 无法解释在MoE模型中观察到的特定激活模式，在MoE模型中，专家分数为未来的激活提供了有价值的预测信号。正如第三部分所讨论的，不仅当前激活的专家可能在未来被重用，而且未激活的高分专家在随后的迭代中被选中的概率也更高。

利用这一见解，我们提出了分数感知缓存，作者引入了 Minus Recent Score (MRS) 替换策略，该策略根据路由分数优先保留专家。定义 s 为当前迭代中所有专家的路由分数，S 为估计优先级分数，$\alpha$ 为平均系数，则估计优先级的更新可表示为:

$$
S=\alpha\times TopP(s)+(1-\alpha)\times S
$$

在这里，只会累积前 p 名的专家分数。这是由图3(b)的观察得出的，得分较低的专家的重用概率没有显著差异。通常，我们将p设置为激活专家数量的两倍。