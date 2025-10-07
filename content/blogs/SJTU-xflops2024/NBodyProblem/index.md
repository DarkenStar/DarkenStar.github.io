---
title: "N-Body Problem"
date: 2025-09-21T09:33:37+08:00
lastmod: 2025-09-21T09:33:37+08:00
author: ["WITHER"]

categories:
- HPC

tags:
- HPC

keywords:
- 

description: "Solution of SJTU-xflops2024 N-Body Problem." # 文章描述，与搜索优化相关
summary: "Solution of SJTU-xflops2024 N-Body Problem." # 文章简单描述，会展示在主页
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


题目已经说明使用 OpenMP 和 MPI 对现有的串行代码进行并行化优化。

# OpenMP Optimization

OpenMP 是一种用于共享内存并行编程的 API，适用于多核 CPU 的并行化。它通过在 C/C++ 代码中添加编译指令 (pragmas) 来指导编译器将某些代码段分配给多个线程执行，而无需显式管理线程。

`#pragma omp parallel` 是 OpenMP 的核心指令，用于创建一个并行区域。进入这个区域的代码会由多个线程同时执行。每个线程都有自己的私有变量，但可以访问共享变量。

对于 OpenMP 主要优化有
- 在 compute_forces， update_particles， compute_total_momentum 和compute_total_energy中 使用 `#pragma omp parallel for` 并行化外部循环。

```cpp
// compute_forces
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    fx[i] = fy[i] = fz[i] = 0.0;
    for (int j = 0; j < N; j++) {  // 内层循环遍历所有粒子以计算引力，保持串行
        if (i != j) {
            // 计算粒子 i 受到粒子 j 的引力
            ...
        }
    }
}

// update_particles
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    particles[i].vx += fx[i] / particles[i].mass * DT;
    particles[i].vy += fy[i] / particles[i].mass * DT;
    particles[i].vz += fz[i] / particles[i].mass * DT;
    particles[i].x += particles[i].vx * DT;
    particles[i].y += particles[i].vy * DT;
    particles[i].z += particles[i].vz * DT;
}

```

- 在必要的地方使用 reduction 子句来安全地积累动量和动能。

```cpp
// compute_total_momentum
#pragma omp parallel for reduction(+:x,y,z)
for (int i = 0; i < N; i++) {
    x += particles[i].mass * particles[i].vx;
    y += particles[i].mass * particles[i].vy;
    z += particles[i].mass * particles[i].vz;
}
*px = x;
*py = y;
*pz = z;

// compute_total_energy
#pragma omp parallel for reduction(+:total_kinetic)
for (int i = 0; i < N; i++) {
    double v2 = particles[i].vx * particles[i].vx + particles[i].vy * particles[i].vy + particles[i].vz * particles[i].vz;
    total_kinetic += 0.5 * particles[i].mass * v2;
}
```

# MPI Optimization

MPI 是一种分布式内存并行编程标准，适用于多进程 (可能运行在不同节点上) 的并行计算。在你的代码中，MPI 用于将 N 体问题的计算任务分配到多个进程，每个进程负责一部分粒子 (`local_n = N / size`)，并通过通信同步数据和归约结果。

对于 MPI 的补充有

- 为 Particle 添加了 MPI 数据类型，因为它是一个包含 7 个双精度浮点数的结构体。
```cpp
// Create MPI datatype for Particle (added: since Particle is 7 contiguous doubles)
    MPI_Datatype MPI_PARTICLE;
    MPI_Type_contiguous(7, MPI_DOUBLE, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);
```

以下是三个 TODO 的通信实现.

1. 用 MPI_Bcast 实现初始广播
```cpp
// Broadcast initial particles to all processes so everyone has the full global array
    MPI_Bcast(particles, N, MPI_PARTICLE, 0, MPI_COMM_WORLD);
```

MPI_Bcast 参数:

- `particles`: 要广播的数据缓冲区 (Particle 数组).
- `N`: 数据元素数量 (N=4096 个粒子).
- `MPI_PARTICLE`: 数据类型 (每个元素是 Particle 结构体).
- `0`: 根进程的 rank (数据从 rank == 0 广播).
- `MPI_COMM_WORLD`: 通信组。

2. 用 MPI_Gather + MPI_Bcast 实现同步
```cpp
// Gather updated local_particles to rank 0's global particles, then broadcast the updated global to all processes
// This ensures all processes have the latest positions for the next iteration and for consistent energy/momentum calculations
MPI_Gather(local_particles, local_n, MPI_PARTICLE, particles, local_n, MPI_PARTICLE, 0, MPI_COMM_WORLD);
MPI_Bcast(particles, N, MPI_PARTICLE, 0, MPI_COMM_WORLD);
```

MPI_Gather 参数含义:
- `local_particles`: 每个进程的发送缓冲区 (local_n 个粒子).
- `local_n`: 每个进程发送的元素数量 (N / size).
- `MPI_PARTICLE`: 发送数据类型。
- `particles`: 根进程的接收缓冲区 (N 个粒子).
- `local_n`: 根进程从每个进程接收的元素数量。
- `MPI_PARTICLE`: 接收数据类型。
- `0`: 根进程的 rank.
- `MPI_COMM_WORLD`: 通信组。

3. 用 4 个 MPI_Reduce 实现归约操作

```cpp
// Reduce the 3 momentum components and the energy to rank 0 using sum (4 separate reductions as specified)
MPI_Reduce(&local_momentum_x, &global_momentum_x, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&local_momentum_y, &global_momentum_y, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&local_momentum_z, &global_momentum_z, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
MPI_Reduce(&local_energy, &global_energy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
```

MPI_Reduce 参数含义:
- `&local_momentum_x`: 每个进程的输入数据 (局部 x 动量).
- `&global_momentum_x`: 根进程的输出缓冲区 (全局 x 动量，仅在 rank == 0 有意义).
- `1`: 数据元素数量 (单个 double).
- `MPI_DOUBLE`: 数据类型。
- `MPI_SUM`: 归约操作 (求和).
- `0`: 根进程的 rank.
- `MPI_COMM_WORLD`: 通信组。