---
title: PMPP Learning-Chapter 8 Stencil
date: 2024-09-09T10:27:42+08:00
lastmod: 2024-09-09T10:27:42+08:00
author: ["WITHER"]

categories:
- CUDA

tags:
- PMPP learning

keywords:
- CUDA

description: "Personal notebook 8 of Programming Massively Parallel Processors." # 文章描述，与搜索优化相关
summary: "Personal notebook 8 of Programming Massively Parallel Processors." # 文章简单描述，会展示在主页
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
# 8 Stencil

在流体动力学、热传导、燃烧、天气预报、气候模拟和电磁学等应用领域，模板是求解偏微分方程的数值方法的基础。模板方法的基本思想是，将偏微分方程的求解转化为求解一个局部的线性方程组，然后在该局部进行迭代求解，最后得到全局解。由于求解微分问题时对数值精度的要求，模板处理的数据往往是高精度的浮动数据，对于 tiling 技术来说，这需要消耗更多的片上内存。

## Backgroud

用计算机数值计算和求解函数、模型、变量和方程的第一步是将它们转换成离散的表示形式。表示的保真度或这些近似插值技术的函数值的准确性取一方面决于网格点之间的间距:间距越小，近似越准确。离散表示的保真度还取决于所使用数字的精度。本章中将重点关注计算模式，其中模板应用于所有相关的输入网格点以生成所有网格点的输出值，这将被称为模板扫描 (*stencil sweep*).

![One-dimensional Stencil Example](https://note.youdao.com/yws/api/personal/file/WEB148a72c6a7b7556806a321b46ad917b7?method=download&shareKey=72f1615fc5c62f3e23a68a3a7738feac "One-dimensional Stencil Example")

![Two-dimensional & Three-dimensional Stencil Example](https://note.youdao.com/yws/api/personal/file/WEB71159c58a69685bed08948350c22dfcf?method=download&shareKey=20517947036fd220df2744f0d4fbad08 "Two-dimensional & Three-dimensional Stencil Example")

## 8.2 Parallel stencil: A Basic Algorithm

2D 情况下输出网格的 tiling 如下图所示，其中每个线程块负责一个 `4*4` 大小的输出 tile. 一个基本的 3D stencil 内核函数如下，其中每个线程块负责计算一个输出 tile 的值，每个线程用于计算一个元素。每个线程执行13次浮点操作 (7 次乘法和 6 次加法)，并加载 7 个输入元素 (每个 4 字节)。因此，这个内核的浮点对计算访存比是 13 / (7*4) = 0.46 OP/B.

![2D 5-point Stencil Tiling for Output Grid](https://note.youdao.com/yws/api/personal/file/WEBb784ac30b171fdcf2a3ec27e6b4351dd?method=download&shareKey=518a1c8267ed726cce9d05ebbe087bce "2D 5-point Stencil Tiling for Output Grid")

```cpp {linenos=true}
__global__ 
void stencil_kernel(float* in, float* out, unsigned int N) {
	unsigned int i = blockIdx.z*blockDim.z+threadIdx.z;
	unsigned int j = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x;

	if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
		out[i * N * N + j * N + k] = c0 * in[i * N * N + j * N + k] +
                                    c1 * in[i * N * N + j * N + k - 1] + c2 * in[i * N * N + j * N + k + 1] +
                                    c3 * in[i * N * N + (j - 1) * N + k] + c4 * in[i * N * N + (j + 1) * N + k] +
                                    c5 * in[(i - 1) * N * N + j * N + k] + c6 * in[(i + 1) * N * N + j * N + k];
	}
}
```

## 8.3 Shared Memory Tiling for Stencil Sweep

下图展示了二维五点模板的输入和输出 tile，可以发现五点模板的输入 tile 不包括四个角落的元素。因为每个输出网格点值只使用输入 tile 的 5 个元素，而 `3*3` 卷积使用 9 个元素。而 3D 情况下七点模板相对于 `3*3*3` 卷积从将输入网格点加载到共享内存中能获得的收益更低。由于为卷积加载输入 tile 的所有策略都直接应用于模板扫描，下面给出了一个加载到共享内存版本的内核函数，线程块的大小与输入 tile 相同，在计算输出 tile 点值时没有使用部分线程。每个表达式中减去的值1是因为内核假设一个3D七点模板，每边有一个网格点

![Input and Output Tiles for a 2D 5-point Stencil](https://note.youdao.com/yws/api/personal/file/WEB248ae00a16c8ed3da3dc8832ced6ebc0?method=download&shareKey=40a18c41ed91dacafbde0c5beac0aaf6 "Input and Output Tiles for a 2D 5-point Stencil")

```cpp {linenos=true}
#define IN_TILE_DIM 16
__global__
void stencil_shared_mem_tiling_kernel(float* in, float* out, unsigned int N) {
    // upper left corner of input tile
	unsigned int i = blockIdx.z*blockDim.z+threadIdx.z - 1;
	unsigned int j = blockIdx.y*blockDim.y+threadIdx.y - 1;
	unsigned int k = blockIdx.x*blockDim.x+threadIdx.x - 1;

	__shared__ float in_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
	if (i >= 1 && i < IN_TILE_DIM && j >= 1 && j < IN_TILE_DIM && k >= 1 && k < IN_TILE_DIM) {
		in_s[threadIdx.z][threadIdx.y][threadIdx.x] = in[i * N * N + j * N + k];
	}
	__syncthreads();

	if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
		if (threadIdx.x >=1 && threadIdx.x < IN_TILE_DIM-1 && 
            threadIdx.y >=1 && threadIdx.y < IN_TILE_DIM-1 && 
            threadIdx.z >=1 && threadIdx.z < IN_TILE_DIM-1) {  // 7 point template

        out[i * N * N + j * N + k] = c0 * in_s[threadIdx.z][threadIdx.y][threadIdx.x] +
                                    c1 * in_s[threadIdx.z][threadIdx.y][threadIdx.x - 1] + c2 * in_s[threadIdx.z][threadIdx.y][threadIdx.x + 1] +
                                    c3 * in_s[threadIdx.z][threadIdx.y - 1][threadIdx.x] + c4 * in_s[threadIdx.z][threadIdx.y + 1][threadIdx.x] +
                                    c5 * in_s[threadIdx.z - 1][threadIdx.y][threadIdx.x] + c6 * in_s[threadIdx.z + 1][threadIdx.y][threadIdx.x];
	}
}
```

硬件限制每个块最大为 1024 ，因此 tile 通常比较小。一般 tile 的边长为8，每个块的大小为 512 个线程。相反，卷积通常用于处理二维图像，可以使用更大的 tile 尺寸 (32x32).
第一个缺点是由于 halo cell 的开销，重用率随着 tile 大小的降低而降低。第二个缺点是它对内存合并有不利影响。对于一个 8x8x8 tile，每 warp 的线程将访问全局内存中至少四行 (8*8*8*4 bytes, 32 threads, 64 bits/DRAM = 4)

## 8.4 Thread Coarsening

下图假设每个输入 tile 由 6x6x6 个网格点组成。为了使输入 tile的内部可见，块的前、左和上面没有画出。假设每个输出 tile 由 4x4x4个网格点组成。分配给处理该 tile 的线程块由与输入 tile 的一个x-y平面 (即 6x6) 相同数量的线程组成。程序一开始，每个块需要将包含计算输出块平面值所需的所有点的三个输入块平面加载到共享内存中。在每次迭代期间，块中的所有线程将处理输出 tile 与迭代值相同的 z 索引对应的 x-y 平面。

![Mapping of Shared Memory Array after First Iteration](https://note.youdao.com/yws/api/personal/file/WEBec24cfc7edeb84a1a010b315f5ac49e0?method=download&shareKey=edc686e7586e2c3b7f55e32af7bbd83d "Mapping of Shared Memory Array after First Iteration")

```cpp {linenos=true}
#define OUT_TILE_DIM IN_TILE_DIM - 2
__global__
void stencil_thread_coarsening_kernel(float* in, float* out, unsigned int N) {
	int iStart = blockIdx.z * OUT_TILE_DIM;
	int j = blockIdx.y * blockDim.y + threadIdx.y - 1;
	int k = blockIdx.x * blockDim.x + threadIdx.x - 1;

	__shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
	__shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
	__shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

	if (iStart >= 1 && iStart < N - 1 && j >= 0 && j < N && k >= 0 && k < N) {
		inPrev_s[threadIdx.y][threadIdx.x] = in[(iStart - 1) * N * N + j * N + k];
	}
	if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0) {
		inCurr_s[threadIdx.y][threadIdx.x] = in[iStart * N * N + j * N + k];
	}

	for (int i = 0; i < OUT_TILE_DIM; i++) {
		i += iStart;
		if (i >= -1 && i < N - 1 && j >= 0 && j < N && k >= 0 && k < N) {
			inNext_s[threadIdx.y][threadIdx.x] = in[(i + 1) * N * N + j * N + k];
		}
		__syncthreads();

		if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 &&
			threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
			threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {

			out[i * N * N + j * N + k] = c0 * inCurr_s[threadIdx.y][threadIdx.x] +
				c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] + c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] +
				c3 * inCurr_s[threadIdx.y - 1][threadIdx.x] + c4 * inCurr_s[threadIdx.y + 1][threadIdx.x] +
				c5 * inPrev_s[threadIdx.y][threadIdx.x] + c6 * inNext_s[threadIdx.y][threadIdx.x];
			}
		
	}
	inPrev_s[threadIdx.y][threadIdx.x] = inCurr_s[threadIdx.y][threadIdx.x];
	inCurr_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
}
```

线程粗化内核的优点是，它不要求输入 tile 的所有平面都出现在共享内存中。在任意时刻，只有三层输入 tile 需要在共享内存中。

##  8.5 Register Tiling

根据计算过程可以发现每个 `inPrev_s` 和 `inNext_s` 的元素仅由一个线程在计算具有相同 x-y 索引的输出 tile 网格点时使用。只有 inCurr_s 的元素被多个线程访问，真正需要位于共享内存中。因此我们可以修改内涵函数如下，寄存器变量 `inPrev` 和 `inNext` 分别替换共享内存数组 `inPrev_s` 和 `inNext_s`. 保留了 `inCurr_s` 以允许在线程之间共享 x-y 平面相邻网格点值。这样这个内核使用的共享内存量减少到原来的 1/3.

```cpp {linenos=true}
void stencil_register_tiling_coarsening_kernel(float* in, float* out, unsigned int N) {
	int iStart = blockIdx.z * OUT_TILE_DIM;
	int j = blockIdx.y * blockDim.y + threadIdx.y - 1;
	int k = blockIdx.x * blockDim.x + threadIdx.x - 1;

	float inPrev;
	float inCurr;
	float inNext; 
	__shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];

	if (iStart >= 1 && iStart < N - 1 && j >= 0 && j < N && k >= 0 && k < N) {
		inPrev = in[(iStart - 1) * N * N + j * N + k];
	}
	if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0) {
		inCurr = in[iStart * N * N + j * N + k];
		inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
	}

	for (int i = 0; i < OUT_TILE_DIM; i++) {
		i += iStart;
		if (i >= -1 && i < N - 1 && j >= 0 && j < N && k >= 0 && k < N) {
			inNext = in[(i + 1) * N * N + j * N + k];
		}
		__syncthreads();

		if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1 &&
			threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
			threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {

			out[i * N * N + j * N + k] = c0 * inCurr +
				c1 * inCurr_s[threadIdx.y][threadIdx.x - 1] + c2 * inCurr_s[threadIdx.y][threadIdx.x + 1] +
				c3 * inCurr_s[threadIdx.y - 1][threadIdx.x] + c4 * inCurr_s[threadIdx.y + 1][threadIdx.x] +
				c5 * inPrev + c6 * inNext;
		}
	}
	__syncthreads();
	inPrev = inCurr;
	inCurr = inNext;
	inCurr_s[threadIdx.y][threadIdx.x] = inNext;
}
```

首先，许多对共享内存的读写现在被转移到寄存器中。其次，每个块只消耗三分之一的共享内存。当然，这是以每个线程多使用 3 个寄存器为代价实现的。需要注意**全局内存访问的数量没有改变**。