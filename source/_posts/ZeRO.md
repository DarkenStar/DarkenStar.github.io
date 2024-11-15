---
title: ZeRO
date: 2024/11/14 21:34:42
categories: Paper Reading
tags: Distributed Training
excerpt: Paper reading of ZeRO.
mathjax: true
katex: true
---

[ZeRO](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/) 通过在数据并行过程中划分模型状态 (参数、梯度和优化器状态) 消除了数据并行过程中的内存冗余。

如下图所示 ZeRO 有三个主要的优化阶段，它们对应于优化器状态、梯度和参数的划分。
对于使用 FP16 的模型，内存占用包括参数 (FP16)、梯度 (FP16)、Adam 优化器状态 (动量 (FP32)，方差 (FP32) 以及更新后的参数 (FP32), 因此 K=12).
1. 优化器状态划分 (Pos) —— 内存减少 4 倍，需要对梯度进行 reduce-scatter，用各自的优化器状态更新梯度后进行 All-gather 使所有设备都有最新的梯度，通信量与数据并行性相同 (对 Loss 进行一次 All-reduce).
2. 添加梯度划分  (Pos+g) -- 内存减少 8 倍，每个设备需要将自己的梯度 Broadcast 到其他设备，然后使用 Gather 将其他设备更新后的模型参数同步到自己上面，通信量与数据并行性相同。
3. 添加参数划分 (Pos+g+p) -- 内存减少与数据并行度 Nd 呈线性关系。通信量增加了50%，因为在前向/反向传播中需要每个设备需要额外广播自己存储的模型参数 `2*(N-1)/N*P`，反向传播时需要对发送梯度到对应的设备上 `(N-1)/N*P`.

![Memory Savings and Communication Volume for the 3-stage of ZeRO](https://note.youdao.com/yws/api/personal/file/WEBcfab82173b0f76eb5b3c8396e81e238a?method=download&shareKey=1b8bb86256be5b15bec039beecee062b "Memory Savings and Communication Volume for the 3-stage of ZeRO")

蓝色箭头串起来的白色长方形代表的是 Transformer Block，蓝色的第一行代表 FP16 参数；橙色的第二行代表 FP16 梯度，反向传播时将用于更新参数；绿色的行代表优化器状态 (FP32 的梯度，动量，方差，以及更新后的参数)，其中在计算完 FP16 梯度以后不再需要保存 FP32 参数。同时也需要 buffer 来保存部分 transformer block 的输出激活。

![Overview of Memory Consumption](https://note.youdao.com/yws/api/personal/file/WEB1fd41e8b92bcd256a910ce757d4eea21?method=download&shareKey=d82f49f0d59309d987c164a100966895 "Overview of Memory Consumption")

每个 GPU 只需要保存自己部分的 Pos+g+p. 前向传播时保存对应模型参数的 GPU 需要把参数广播到其他 GPU 中，其他 GPU 用自己部分的数据完成前向传播后就可以删除这部分参数 (最后一部分除外). `(N-1)/N*P`

![Broadcast of Model Parameters](https://note.youdao.com/yws/api/personal/file/WEB82b66a8a7b40a2fdf9512545189cc37a?method=download&shareKey=37a4614d85a77d2ca00e6d5a4769e2f9 "Broadcast of Model Parameters")

前向传播完成后，第一次反向传播可以利用最后一次正向传播已经广播了的模型参数，每个 GPU 计算自己部分的梯度，然后 Reduce 到存储对应模型参数的 GPU 中。之后和前向传播一样，每个 GPU 都需要广播自己的参数，然后其他 GPU 用自己的数据完成梯度计算以后 Reduce 到自己的梯度。`(N-1)/N*P + 1/N*G*(N-1)`

![Gradient Accumulation](https://note.youdao.com/yws/api/personal/file/WEB575b3869e59814ae1449351cf1b18d01?method=download&shareKey=10f29390f47227ca8eefbcc00f4fca6e "Gradient Accumulation")

反向传播结束以后，每个 GPU 使用优化器更新自己的 FP32 模型参数后转换成 FP16 格式。

![Update Parameters Locally](https://note.youdao.com/yws/api/personal/file/WEB19cb21dfa63ab76437a2246ff52b00aa?method=download&shareKey=ddb63d57f3bf976b1dce4596e77a2009 "Update Parameters Locally")