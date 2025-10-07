---
title: "SpecInfer"
date: 2025-10-06T13:03:29+08:00
lastmod: 2025-10-06T13:03:29+08:00
author: ["WITHER"]

categories:
- PaperReading

tags:
- LLM
keywords:
- LLM

description: "Paper reading of SpecInfer." # 文章描述，与搜索优化相关
summary: "Paper reading of SpecInfer." # 文章简单描述，会展示在主页
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

SpecInfer 是一个基于树的推测推理和验证的加速 LLM 服务的系统。利用小型投机模型来预测 LLM 输出；这些预测被组织成一个 token 树，每个节点代表一个候选 token 序列。使用一种新颖的基于树的并行解码机制，在LLM上并行验证由 token 树表示的所有候选 token 序列的正确性。SpecInfer使用LLM作为 token 树验证器，而不是增量解码器，大大减少了服务生成LLM的端到端延迟和计算需求，同时保持模型质量。

# 1. Introduction

![Figure 1. Comparing the incremental decoding approach used by existing LLM serving systems, the sequence-based speculative  inference approach, and the tree-based speculative inference approach used by SpecInfer.](https://share.note.youdao.com/yws/api/personal/file/WEB8a393cf1de330c6c2237da2d579e5cef?method=download&shareKey=2577de8d0603f472fe6db57e9480bc14 "Figure 1. Comparing the incremental decoding approach used by existing LLM serving systems, the sequence-based speculative  inference approach, and the tree-based speculative inference approach used by SpecInfer.")

LLM生成文本的方式是逐个 token 生成的。每个新生成的 token 都依赖于输入的 prompt和所有先前已生成的 token. 之前的工作使用一个模型参数量远小于LLM的小型投机模型 (Small Speculative Model, SSM)来预先生成一个 token 序列，然后让LLM并行的验证这个序列的正确性。这种方法只考虑了由单个SSM生成的一个 token 序列。由于SSM和LLM之间存在巨大的模型能力差距 (SSM通常比LLM小100-1000倍)，SSM的预测序列很难与LLM的真实输出完全对齐，导致投机成功率不高。

SpecInfer 同时考虑多种投机候选，以最大化投机性能。这些候选标记被组织成一个标记树，每个节点代表一系列推测的标记。所有候选 token 序列的正确性都是针对LLM并行验证的，这使得SpecInfer可以在LLM解码步骤中显著增加生成 token 的数量。实现这个技术有两个难点。
1. 巨大的搜索空间: 要最大化投机性能，SpecInfer必须在一个极其庞大的候选序列空间中进行探索。这主要是因为LLM的词汇表非常大，并且需要预测未来的多个 token. SpecInfer不是使用单个SSM进行基于序列的推测，而是通过同时考虑为给定输入提示以树结构组织的各种 token 序列来最大化推测性能。SpecInfer引入了一种基于扩展和合并的机制，分别通过利用单个SSM和多个SSM中的多样性来构建 token 树。
2. 保证验证的正确性: 许多LLM应用采用随机解码 (stochastic decoding)，即从一个概率分布中采样下一个 token ，而非总是选择概率最高的那个。SpecInfer必须确保其复杂的树状验证机制最终生成的 token，遵循与原始LLM完全相同的概率分布。为此，论文提出了新的多步投机采样 (multi-step speculative sampling)并结合基于树的并行解码机制，以最低的验证成本，最大化可被验证的 token 数量，同时保证了数学上的等价性。

# 2. SpecInfer’s Overview

![Figure 2. An overview of SpecInfer’s tree-based speculative inference and verification mechanism.](https://share.note.youdao.com/yws/api/personal/file/WEB6c348041d63e1526d32c688cfb222ab5?method=download&shareKey=68f589b270d01908acb60124a89d6c79 "Figure 2. An overview of SpecInfer’s tree-based speculative inference and verification mechanism.")

SpecInfer的系统设计可以清晰地划分为两个主要部分: 

- 学习型投机器 (Learning-based Speculator): 负责猜测，即生成一个包含多种可能性预测的 token 树。SSM 通常使用与目标LLM同系列但尺寸小得多的预训练模型，因为他们是在同一数据集上训练的。为了最大化投机树能够覆盖LLM真实输出的概率，SpecInfer提出了两种构建方法 (在Figure 2顶部展示): 
    - 基于扩展的构建 (Expansion-based): 利用单个SSM在某些步骤生成多个候选 token ，形成树的分支。
    - 基于合并的构建 (Merge-based): 并行使用多个不同的SSM，然后将它们各自的预测结果合并成一棵更多样化的树。

-  token 树验证器 (Token Tree Verifier): 负责验证，即利用 LLM 高效地判断投机器给出的猜测哪些是正确的。在传统系统中，LLM是一个增量解码器，一次只预测下一个 token  。在SpecInfer中，LLM的角色转变为一个 token 树验证器，负责一次性地对整个投机树进行评估。为了验证树中某个特定 token 的正确性，系统会将其所有祖先节点构成的序列作为它的上下文。

SpecInfer相较于现有LLM服务系统实现了两大关键优势: 
1. 减少对LLM参数的内存访问: LLM推理的性能瓶颈主要在于从GPU内存中读取庞大的模型参数，而非计算本身 。在传统的增量解码中，每生成一个 token ，都需要完整地访问一次LLM的所有参数。只要投机树与LLM的实际输出存在重叠 (即猜对了至少一个 token )，SpecInfer就能显著减少访问LLM参数的次数。
2. 降低端到端推理延迟: LLM服务面临着很长的端到端延迟 。这主要是因为增量解码中， token 的生成是串行依赖的，系统必须等待前一个 token 生成完毕才能开始计算下一个。Spec 将 LLM作为验证器，一次性接收整个投机树作为输入，并能同时检查树中的所有 token 。

# 3. Learning-based Speculator

现有的投机解码方法只预测一个单一的 token 序列。这种方法的成功概率会随着期望对齐的序列长度呈指数级衰减。SpecInfer的目标是通过在每一步提供更多样化的候选 token 来提升匹配成功的概率 。它将这些由一个或多个SSM预测出的多样化 token 组织成一个 token 树结构，以最大化投机性能，同时保持较低的内存和延迟开销。

定义 3.1 (Token Tree):  token 树 N 是一个树形结构，其中每个节点 $u\in N$ 都被标记为一个 token  $t_u$. 对于每个节点 u，它所代表的 token 序列 $S_u$ 由其父节点所代表的序列 $S_{p_u}$ 与其自身的 token  ${t_u}$ 拼接而成。

**Expansion-based token tree construction.** 该方法基于一个非常重要的观察: 当SSM的预测与LLM不一致时 (即它们的top-1选择不同)，LLM实际选择的 token 通常也存在于SSM预测的 top-k token 列表中，且k值通常很小。

![Figure 3. Illustration of token tree expansion.](https://share.note.youdao.com/yws/api/personal/file/WEB22e5614e252b4acc978b3c7e5402d89e?method=download&shareKey=d2e6a529438f51a24efcf241a4509403 "Figure 3. Illustration of token tree expansion.")

如果在每一步都进行top-k扩展，会导致序列数量呈指数级爆炸，带来巨大的延迟和内存开销。因此，SpecInfer采用了一种静态策略，即遵循一个预设的扩展配置。这个配置是一个整数向量 $<k_1, k_1,\cdots , k_m>$，其中 m 是最大投机步数，$k_i$ 表示在第i步为每个节点扩展出的分支数量。作者也承认，如何动态地、自适应地决定扩展策略是一个开放的研究问题，并将其作为未来的工作。

**Merge-based token tree construction.** 该方法利用多个SSM来共同预测LLM的输出，以获得更强的多样性。借鉴了机器学习中的集成学习思想。通过组合多个弱分类器 (SSMs)，来形成一个更强大的强分类器，使其联合输出能更好地逼近LLM. 在一个通用文本语料库上，首先用LLM为每个prompt生成标准答案。然后，逐个训练SSM。第一个 SSM 被充分微调后，标记出所有它能猜对LLM输出的样本。接着，用剩下的、第一个 SSM 猜错的样本来训练第二个SSM，以此类推。通过这个过程得到了一组多样化的SSM，每个SSM可能擅长处理不同类型的文本模式，它们的聚合输出能够最大程度地覆盖LLM的输出。

当使用多个SSM时，每个SSM都会生成自己的 token 树 (或序列)，SpecInfer会执行 **token 树合并 (Token Tree Merge)** 操作，将所有这些树聚合成一个单一的、更大的树结构。

定义 3.2 (Token Tree Merge): 合并多个 token 树会产生一棵新树，这棵新树包含了原始所有树中的全部 token 序列。

# 4. Token Tree Verifier

通过一次对 LLM 参数的访问，并行地验证 token 树中的所有候选序列 。为了实现这一目标，作者设计了一整套精巧的机制，可以分解为以下三个关键步骤: 
1. 树注意力 (Tree Attention): 将传统作用于线性序列的注意力机制，从概念上推广到树形结构。
2. 基于树的并行解码 (Tree-based Parallel Decoding): 实现“树注意力”的高效计算方法，这是本章的技术核心。
3.  token 验证 (Token Verification): 在计算完成后，根据贪心或随机解码规则，最终确定哪些投机 token 被接受。

定义 4.1 (Tree Attention): 对于 token 树 N 中的任意一个节点 u，它的树注意力被定义为: 对该节点所代表的 token 序列 $S_u$ (即从树根到节点u的路径)进行一次标准的 Transformer 序列注意力计算。

![Figure 4. Comparing SpecInfer’s tree-based parallel decoding with existing sequence-based decoding.](https://share.note.youdao.com/yws/api/personal/file/WEB2031f42615314179578d295ff370e156?method=download&shareKey=64e964672ec8b033255bc4ca763eff1f "Figure 4. Comparing SpecInfer’s tree-based parallel decoding with existing sequence-based decoding.")

如果对树中的每条路径都独立计算一次，效率会非常低下。在树状结构中，不同的分支在同一深度 (位置) 上会有不同的 token ，这意味着它们的 KV 缓存是相互冲突的。SpecInfer 引入了两项关键技术来解决上述问题，如 Figure 4 右侧所示: 
1. 深度优先搜索 (Depth-first search) 来更新共享的 KV 缓存: SpecInfer 并不为每个序列创建独立的缓存，而是让所有序列共享同一个 KV 缓存区。系统按照深度优先的顺序遍历 token 树，并依次更新这个共享的缓存。这种遍历方式保证了在计算任何一个节点的注意力时，其所有祖先节点的 KV 值都已经被正确计算并存储在缓存中。
2. 拓扑感知的因果掩码 (Topology-aware causal mask): 为了避免为树中每个节点单独启动计算核，SpecInfer 将树中的所有 token 打包成一个批次，送入一个单一的、融合的计算核中进行处理。SpecInfer 对标准的因果掩码进行了巧妙的修改，使其能够感知树的拓扑结构。这个特殊的掩码会精确地屏蔽掉所有不具有祖先-后代关系的 token 对之间的注意力计算。

在通过并行解码计算出树中每个节点的 LLM 输出 (即每个位置最可能的下一个 token )后，最后一步就是根据具体的解码策略来决定接受哪些投机 token 。SpecInfer 同时支持贪心解码和随机解码
- 贪心解码 (Greedy decoding): 系统从 token 树的根节点开始，沿着一条路径向下遍历 。在每个节点 u，它会检查 u 的子节点中，是否有某个子节点 v 的 token  $t_v$ 与 LLM 在 u 位置预测出的最优 token  O(u) 相匹配。如果匹配成功，就将 $t_v$ 视为已验证的 token ，并继续从节点 v 向下验证。如果所有子节点都不匹配，则验证终止，系统接受 LLM 预测的 token  O(u) 作为最后一个正确的 token ，并将这条已验证的路径附加到最终的生成结果中。
- 随机解码 (Stochastic decoding): 在投机式推理的框架下，随机解码面临一个严峻的挑战: 如何确保经过复杂的“猜测-验证”流程后，最终输出的文本仍然严格遵循原始 LLM 的概率分布？作者提出的 多步投机采样 (MSS) 流程如下:

![Figure 5. Illustrating the multi-step speculative sampling  mechanism for verifying LLMs with stochastic sampling.](https://share.note.youdao.com/yws/api/personal/file/WEB51170f321c43c0e7dcc626edf8ec25d7?method=download&shareKey=90254bc50d0860d1e9e8cc9307ca9658 "Figure 5. Illustrating the multi-step speculative sampling  mechanism for verifying LLMs with stochastic sampling.")

假设我们当前已经验证到了 token 树的节点 u，需要决定下一个 token 是什么。u 的子节点集合为 H，代表了所有投机的候选。
1. 进入验证循环: 只要候选集合 H 不为空，就持续进行尝试。
2. 随机挑选候选: 从 H 中随机挑选一个候选子节点 s，其对应的 token 为 $x_s$.
3. 进行概率接受测试: 系统会以一定的概率接受 token  $x_s$. 这个接受概率 $P_accept$ 是基于 LLM 和 SSM 对该 token 的预测概率之比计算

$$
P_{accept}=\min\left(1,\frac{P(x_s|u,\Theta_{LLM})}{P(x_s|u,\Theta_{SSM})}\right)
$$

4. 根据测试结果进行分支处理: 
    - 如果接受: 将 token  $x_s$ 添加到已验证序列 V 中。将当前节点更新为 $s (u = s)$，然后跳出当前节点的验证循环，回到步骤1，对新的 u 的子节点进行验证。
    - 如果拒绝: 系统会从 LLM 的原始概率分布中减去刚刚被拒绝的 SSM 候选所对应的概率质量，然后对剩余的分布进行归一化，形成一个新的、修正后的残差分布 (Normalized residual distribution).
5. 所有候选均被拒绝: 系统会从最终修正后的 LLM 概率分布中进行一次标准的采样加入到已验证序列中，结束整个过程。

# 5. System Design and Implementation

![Figure 6. SpecInfer’s workflow for one iteration of speculative inference and verification. SpecInfer uses data parallelism to serve SSMs, and combine tensor model parallelism  and pipeline model parallelism for serving an LLM.](https://share.note.youdao.com/yws/api/personal/file/WEBe5a9612b1a8a22e66ff804a95137e07e?method=download&shareKey=e77cf425ddb0c666317991d4ed0cd059 "Figure 6. SpecInfer’s workflow for one iteration of speculative inference and verification. SpecInfer uses data parallelism to serve SSMs, and combine tensor model parallelism  and pipeline model parallelism for serving an LLM.")

请求管理器 (Request Manager) 通常在 CPU 上运行，主要负责以下功能:
1. 请求调度 (Request Scheduling): 接收用户的 LLM 服务请求，并决定处理哪些请求。
2. token 树合并 (Token Tree Merge): 从多个 SSM 收集生成的 token ，并将它们合并成每条请求对应的投机树。
3. token 树验证 (Token Tree Verification): 从 LLM 处获取验证结果，并执行最终的验证逻辑 (贪心或 MSS 算法). 

工作流程如下: 
1. 请求管理器将待处理的请求分发给多个并行的 SSM 实例。
2. 使用数据并行的方式来服务 SSM. 因为 SSM 模型较小，可以在单张 GPU 上运行，因此系统会将不同的请求分配到不同的 GPU 上，让多个 SSM 并行地生成候选 token . 
3. SSM 生成的候选 token 被送回请求管理器，管理器为每条请求构建一棵投机树，然后将这些树分发给 LLM 进行验证。
4. 使用混合并行策略，所有参与 LLM 计算的 GPU 都会执行基于树的并行解码来计算树注意力分数。
5. LLM 生成的验证结果被送回请求管理器，完成最后的 token 接受/拒绝判断 。

请求管理器和 GPU 工作节点之间只传输轻量级的 token  ID，而非高维度的向量表示，因此通信开销可以忽略不计。SpecInfer 采用了 Orca 系统中提出的连续批处理技术。

SpecInfer 在 FlexFlow 之上构建。为了避免直接调用 cuBLAS 和 cuDNN 库计算注意力时产生的高昂的核函数启动开销，SpecInfer 采用了基于 FasterTransformer 的定制化 CUDA 核函数。每个 GPU 线程块负责计算单个请求的单个注意力头。通过利用 GPU 的共享内存来缓存 query tensor 并进行中间结果的广播。

论文提出了一个关键的观点，即传统的增量解码模式在很多时候未能充分利用 GPU 的计算资源，导致 GPU 处于半闲置状态。SpecInfer 利用了这些闲置的计算资源 (under-utilized resources) 来执行树的并行验证，因此，尽管总计算量增加了，但并不会显著增加每次迭代的实际运行时间，引入的运行时开销可以忽略不计。

最后，论文指出了 SpecInfer 的技术能够显著受益的两个实际应用场景。

1. 分布式 LLM 推理 (Distributed LLM inference): SpecInfer 通过一次性验证多个 token ，增加了通信的粒度，减少了通信的频率。虽然不能减少单次通信的数据量，但通过显著减少总的解码步骤 (和通信次数)，从而降低了端到端延迟 。
2. 基于卸载的 LLM 推理 (Offloading-based LLM inference): SpecInfer 的优势在这里被进一步放大。通过机会性地一次验证多个 token ，它直接**减少了 LLM 解码的总步数，从而显著减少了参数在 CPU 和 GPU 之间的传输次数**。