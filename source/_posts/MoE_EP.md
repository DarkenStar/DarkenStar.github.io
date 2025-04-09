---
title: MoE & Expert Parallel
date: 2025/3/25 16:42:23
categories: Paper Reading
tags: blog
excerpt: Mixture of Experts and Expert Parallel
mathjax: true
katex: true
---
参数
-  D = 输入和输出维度。
-  I = 专家网络的中间隐蔵淮度。
-  N = 总专家数量。
-  K =  每个 token 激活的专家数量（Top－K 的  K  ）。
-  G =  专家分组数量。
-  G_topk =  每个 token 路由到的组数量。
-  W =  world＿size：设备数量。
-  R =  rank：当前设备编号。
-  T  ：token 数量（展平后的 token 总数）。
# What is Mixture of Experts(MoE)
在 MoE-Transformer 中，FFN 层被替换为 MoE 模块。MoE 通过一个门控网络 (Gating Network) 动态选择哪些专家来处理特定的输入 token. MoE 中每个输入 token 只激活一小部分专家 (通常通过 Top-K 路由选择 K 个专家，K≪N，其中 N 是专家总数)

# Workflow
假设经过 Attention 后将 bathsize 维度展平后张量的维度是 (T, D)， 其中 T 为所有 sequence 的 token 数量，D 为隐藏层维度。第一步会通过过 Gating Network(权重为 (D, E) 的线性层，其中 E 是专家数目)，再使用 Softmax 后得到每个 token 被路由到各个专家的权重分数 scores，如果对专家进行了分组，就将 scores reshape 成 (T, G, E/G)，不对 scores 加 bias 时对最后一维使用 argmax 得到 group_scores (T, G)，即每组中最大的专家权重。再通过 TopK-Groups 选取分数最大的 G_topk 组。再在这 G_topk 组的专家中选取 Top-K 专家对应的索引 indices (T, K). 再根据选择的专家对 scores 进行归一化得到最终的 weights (T, K). 最后就是遍历当前设备负责的所有专家，对每个 token 根据 indices 计算被选中的专家输出，并根据 weights 加权组合。

# MoE Parallel Training

Expert Parallel(EP) 指的是让每个 GPU 加载一部分专家的权重，可以与模型并行 (MP, TP) 和 数据并行 (DP) 一起使用。情况如下图所示。4x4 的网格代表 16 个 GPU，相同颜色的方块代表完全相同的数据或者权重，方块覆盖的网格数代表权重或数据被均分到几个 GPU 上。

## Non-MoE(TP+DP) MoE(EP+DP)
下图展示了 MoE 部分采取 EP+DP，Non-MoE 部分采取 TP+DP 的情况，其中 TP_degree = 2, EP_degree = 4, 一共 16 块 GPU，因为每个 TP group 内的 GPU 输入数据需要一致才能保证和单卡训练效果相同，因此 DP_gree = 8. 图中相同颜色的三角形代表相同的数据，相同颜色的矩形代表 Non-Moe 部分相同的权重，e_i 代表第 i 部分的专家。



前向过程中，经过 Non-MoE 后会进行一次 All-Reduce，此时每个 TP group(该组内的所有 GPU 合起来存储了完整的权重) 内的 GPU 的数据是相同的，再经过 Gating 获得 token 选择的专家后，每个 EP Group(该组内的所有 GPU 合起来存储了所有的专家) 会进行 All2All 将 token 送往对应的专家所处的 GPU. 经过 MoE 后每个 EP Group 再进行一次 All2All 将 token 送回原先的 GPU.

反向过程中，经过 MoE 部分时存储了相同专家的设备之间 (0,4,8,12;...) 各自计算梯度之后需要进行 AllReduce 获取梯度和，然后到了 Non-MoE 部分，每个存储了相同权重的设备 (0,2,...,14;...) 各自计算梯度之后需要进行 AllReduce 获取梯度和.

## EP All2All

前向过程中每个 GPU 需要按要发送到的 GPU 对 token 进行分组后进行 All2All，由于一个 TP group 在经过 Non-MoE 部分后进行 AllReduce，因此拥有的数据是相同的，此时进行标准的 All2All 会发送冗余的数据。因此我们让每个 TP 组内的 GPU 只保存要发往其他存储了相同 Non-MoE 权重的 (TP rank 相同) GPU 的数据，先在 TP 组内进行  All2All，再在 TP 组之间进行 All2All.

## MoE(TP+EP+DP)
如果对专家也做 TP，最好让 MoE 的 TP 组设置与 Non-MoE 部分相同。这时候 Non-MoE 的输出在 EP group(合起来存储了所有专家的相同位置部分权重) 中进行 All2All 后进行前向，输出结果需要在 TP group 之间再进行一次 AllReduce. 同样的反向时保存了相同 MoE 权重部分的 (e.g. 0,8 都保存了第一部分专家的一部分权重) GPU 在各自计算完梯度之后要进行 AllReduce.
