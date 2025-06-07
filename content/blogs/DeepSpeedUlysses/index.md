---
title: "DeepSpeedUlysses"
date: 2024-10-21T11:09:12+08:00
lastmod: 2024-10-21T11:09:12+08:00
author: ["WITHER"]

categories:
- Paper Reading

tags:
- Distributed Training

keywords:
- Parallel
description: "Paper reading of Deepseed Ulysses." # 文章描述，与搜索优化相关
summary: "Paper reading of Deepseed Ulysses." # 文章简单描述，会展示在主页
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
# DeepSpeed-Ulysses Core Design

## System Design 

原理如下图所示，假设设备数 P 等于多头注意力的头数 hc. 输入 `x[N,d]` 被切分到每个设备上 `[N/p, d]`，之后进行 QKV Projection，随后将 K 进行转置后进行一次 all-to-all 通信，这样每个设备上就有 `Q[N, d/P], K[d/P, N], V[N, d/P]`, 再执行标准的 attention 计算 $Outputcontext=Softmax((QK^T)/\sqrt{d})V$. 再进行一次 all-to-all 通信使得每个设备上有 `[N, d/P]` 结果再进行后续操作。

![DeepSpeed Sequence Parallelism (DeepSpeed-Ulysses) Design](https://note.youdao.com/yws/api/personal/file/WEB06300727bd2f239239db47091e81223c?method=download&shareKey=6abbc645b3fa3a039b464dd405f96d4a "DeepSpeed Sequence Parallelism (DeepSpeed-Ulysses) Design")

## Communication Analysis

在采用节点内 NVSwitch 互连和节点间 fat tree IB 拓扑的集群中，对于总消息大小为 M 的 all-to-all 通信，每条链路通过 P 个 gpu 传输的通信量为 M/P。对于隐藏层大小为 h、序列长度为 N、并行度为 P 的 transform 模型，DS-Sequence 对注意力计算前总消息大小为 3Nh 的 QKV Projection 执行 all-to-all 通信，对每个 transformer block 的输出执行 all-to-all 通信，大小为 Nh. 因此，DeepSpeed 序列下每条链路的总通信量为 4Nh/P (或复杂度为 O(N/P)). 也就是说当 N 和 P 按比例增加时，该通信量是恒定的。

## Comparison of Other Works

![Comparison of DS-Ulysses to Other Sequence Parallelism Methods](https://note.youdao.com/yws/api/personal/file/WEBff8d584feabe45900c3a57eea94a78a0?method=download&shareKey=7bae2e87b18707dabcd5e5ae7976e644 "Comparison of DS-Ulysses to Other Sequence Parallelism Methods")
- ColAI-SP 发明了 Ring-Attention，Q 存储在本地 而 KV 以环形方式传输以计算全局注意力，导致通信复杂度与消息大小 M 呈线性关系。
- Megatron-LM 序列并行方法与 Megatron 张量并行紧密集成。Megatron-LM 沿着序列维度划分序列，并应用 all gather 和 reduce scatter 来聚合 QKV 注意力计算的投影。并行通信量随消息大小 M 线性增加。
- DeepSpeed-Ulysses 通过增加与序列长度成比例的设备数来保持通信量恒定。同时将 Zero3 扩展到数据并行和序列并行的组合。ZeRO 跨序列和数据并行组划分模型状态，并在需要时使用 allgather 收集每个 rank 的部分。

## General and Attention Agnostic Solution

DeepSpeed-Ulysses 的优势在于一种以注意力为中心的序列并行设计。在注意力计算是 N/P 划分的序列并行之前，注意力计算是头并行，每个头的注意力都是完整的，但只有较少的头，因此注意力计算可以被任何类型的注意机制所取代，例如 dense attention 和各种形式的 sparse attention.