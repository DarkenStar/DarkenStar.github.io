---
title: PMPP Learning-Chapter 5 Memory Architecture and Data Locality
date: 2024/9/05 17:39:00
categories: CUDA
tags: PMPP learning
excerpt: Personal notebook 5 of Programming Massively Parallel Processors.
mathjax: true
katex: true
---
# 5 Memory Architecture and Data Locality

&emsp;&emsp;之前章节所写的 CUDA 内核只能达到底层硬件峰值算里的一小部分。因为全局内存 (通常使用片外 DRAM 实现) 往往具有较长的访问延迟 (数百个时钟周期) 和有限的访问带宽。

## 5.1 Importance of  Memory Access Efficiency

&emsp;&emsp;在之前矩阵乘法的内核函数中，每次迭代里执行一次浮点乘法和一次浮点加法需要访问全局内存两次。因此，从全局内存访问的浮点操作次数 (FLOP) 与字节数 (B) 的比率为 2 FLOP-to-8 B，即 0.25FLOP/B. 计算访存比 (*compute to global memory access ratio*) 定义为在程序的一个区域内对全局内存访问的单位字节执行的 FLOPS 数。
计算访存比对 CUDA 内核的性能有重大影响。A100 GPU 的全局内存带宽峰值为 1555 GB/s，矩阵乘法内核计算访存比为 0.25 OP/B，因此内核可以执行的单精度 FLOPs 的吞吐量为 389 GFLOPS，仅为 A100 GPU 峰值单精度运算吞吐量 (19,500 GFLOPS) 的 2%. 我们把执行速度受内存带宽限制的程序称为内存瓶颈 (*memory bound*) 程序。

{% fold info @The Roofline Model %}
&emsp;&emsp;Rooline 模型用于评估应用程序相在其所运行的硬件的限制上达到的性能。如下图所示，x 轴表示算术或计算强度 (*computational intensity*)，单位为 FLOP/B. y 轴表示以 GFLOPS 为单位的计算吞吐量。横线表示硬件可以提供的峰值计算吞吐量。
&emsp;&emsp;硬件通常关注两个指标:

- 算力 π：也称为计算平台的性能上限，指的是一个计算平台倾尽全力每秒钟所能完成的浮点运算数。单位是 FLOP/s。
- 带宽 ß：即计算平台的带宽上限，指的是一个计算平台倾尽全力每秒所能完成的内存交换量。单位是Byte/s。
  &emsp;&emsp;两个指标相除即可得到计算平台的计算强度上限 I_max = π / ß，它描述的是在这个计算平台上，单位内存交换最多用来进行多少次计算。

![Roofline Model](https://note.youdao.com/yws/api/personal/file/WEB6d519969c36c2ceb8f94fda0644c984a?method=download&shareKey=f85b96fc7fa07712421487c9b01f7e1b "Roofline Model")

&emsp;&emsp;从图中可以看出算力决定“屋顶”的高度（绿色线段），带宽决定“房檐”的斜率（红色线段）。

- Memory-Bound: 当模型的计算强度 I 小于硬件的计算强度上限 I_max 时，由于此时模型位于“房檐”区间，因此模型理论性能 P 的大小完全由硬件的带宽上限 ß （房檐的斜率）以及模型自身的计算强度 I 所决定，因此这时候就称模型处于 Memory-Bound 状态。
- Compute-Bound: 不管模型的计算强度 I 有多大，它的理论性能 P 最大只能等于硬件的算力 π 。当模型的计算强度 I 大于硬件的计算强度上限 I_max 时，模型在当前硬件处于 Compute-Bound 状态
  {% endfold %}
  &emsp;&emsp;为了让内核具有更高的性能，需要通过减少内核执行的全局内存访问次数来增加计算访存比。

## 5.2 CUDA memory types

&emsp;&emsp;下图展示了 CUDA 设备的内存。全局内存和常量内存这两种类型的内存都可以被主机写入 (W) 和读取 (R) 。全局内存也可以被设备读写，而常量内存只支持设备对其读取。
&emsp;&emsp;另一种类型的内存是本地内存，也可以被读写。**本地内存实际上放在全局内存中**，具有相似的访问延迟，但它不是跨线程共享的。每个线程都有自己的全局内存部分，将其用作自己的私有本地内存，存放私有但不能在寄存器中分配的数据。
&emsp;&emsp;寄存器 (*register*) 和共享内存 (*shared memory*) 是片上内存。存储在这些类型内存中的变量可以以高度并行的方式以高速访问。其中每个线程只能访问自己的寄存器。

![Overview of CUDA Memory Model](https://note.youdao.com/yws/api/personal/file/WEB6b7cbeee7a8279269c480cc3dd307c92?method=download&shareKey=c9a05598ffe57e76def11f1ea20593fa "Overview of CUDA Memory Model")

&emsp;&emsp;与基于冯·诺伊曼模型的计算机类比，CUDA 设备中的全局内存对应于内存框，寄存器对应于寄存器堆。与访问全局内存相比，每次访问寄存器所涉及的指令更少。当算术指令的操作数在寄存器中时，不需要额外的指令使算术逻辑单元(ALU)可以使用该操作数的值。如果操作数值在全局内存中，处理器需要执行内存加载操作让 ALU 能使用操作数。并且从寄存器堆访问所消耗的能量至少比从全局内存访问低一个数量级。

![Memory vs. Registers in a Modern Computer Based on the von Neumann Model](https://note.youdao.com/yws/api/personal/file/WEB976785b7c8c819c9e53543306299645d?method=download&shareKey=6af41bb3bef56687bcbe90e6d21f1204 "Memory vs. Registers in a Modern Computer Based on the von Neumann Model")

&emsp;&emsp;下图展示了 CUDA 设备中的共享内存和寄存器。共享内存实际上是一种暂存存储器 (*scratchpad memory*)，作为片上内存的一部分。当处理器访问存储在共享内存中的数据时，需要执行内存加载操作。CUDA 中共享内存和寄存器之间的一个重要区别是，存储在共享内存中的变量可以被块中的所有线程访问，而寄存器数据是线程私有的。

![Shared Memory vs. Registers in a CUDA Device SM](https://note.youdao.com/yws/api/personal/file/WEB9683f5c9b3fdd69aa62d8c8751be45f7?method=download&shareKey=62797256af4f18ce4ebf652d411de315 "Shared Memory vs. Registers in a CUDA Device SM")

&emsp;&emsp;下表给出了将程序变量声明为各种内存类型的 CUDA 语法。

- 所有在内核和设备函数中声明的 automatic scalar variables 都被放入寄存器中。
- Automatic array variables 存储在线程的本地内存中。如果所有访问都使用常量索引值，编译器可能决定将将其存储到寄存器中。
- 块中的所有线程都看到 shared variable 的相同版本。内核执行期间每个块会创建和使用一个私有版本。通常使用共享变量来保存在内核执行阶段经常使用和重用的全局内存数据部分。
- Constant variables 通常用于向核函数提供输入。内核函数不能修改常量变量的值。
- Global variables 通常用于将信息从一个内核调用传递到另一个内核调用。

| Variable Declaration                          | Memory   | Scope  | Lifetime    |
| --------------------------------------------- | -------- | ------ | ----------- |
| Automatic variables other than arrays         | Register | Thread | Kernel      |
| Automatic  array variables                   | Local    | Thread | Kernel      |
| `__device__ __shared__ int SharedVar;`     | Shared   | Block  | Kernel      |
| `__device__ int GlobalVar;`                | Global   | Grid   | Application |
| `__device__ __constant__ int ConstantVar;` | Constant | Grid   | Application |

&emsp;&emsp;在 CUDA 中，指针可以用来指向全局内存中的数据对象，通常有以下两种情况会使用

- 对象由主机函数分配，指向对象的指针由内存分配函数 (如 `cudaMalloc`) 初始化，作为参数传递给内核函数。
- 将在全局内存中声明的变量的地址赋给指针变量。

## 5.3 Tiling for Reduced Memory Traffic

&emsp;&emsp;一种常见的策略是将数据划分为称为 *tile* 的子集，以便每个 tile 都适合共享内存。能进行划分的一个重要的标准是这些 tile 上的内核计算可以彼此独立地完成。
&emsp;&emsp;下图展示了 block(0,0) 的四个线程所完成的计算。这四个线程计算P(0,0), P(0,1), P(1,0) 和 P(1,1). 每个线程在执行过程中访问 M 的 4 个元素和 N 的 4 个元素，可以看出有明显重复的部分。将每个块需要访问的数据先加载到共享内存，这样可以避免每个线程从全局内存里加载重复的数据。全局内存流量的减少与块的维度成正比。每个块大小为 Width*Width 时，全局内存流量将减少为原来的 1/Width.

![A Small Example of Matrix Multiplication](https://note.youdao.com/yws/api/personal/file/WEB3ed7adc0960b337ea099f2d34a5474db?method=download&shareKey=18100aa957fb01bf5f29920b908fd02f "A Small Example of Matrix Multiplication")

&emsp;&emsp;按 tile 进行矩阵乘法的基本思想是让线程在各自使用元素来进行内积计算之前，将 M 和 N 元素的子集加载到共享内存中。如下图所示把 M 和 N 分成大小为 `2*2` 的块。每个线程执行的内积计算现在被划分为几个阶段。在每个阶段，一个块中的所有线程协作将对应的 M 和 N 的 tile 加载到共享内存中。这样每个阶段关注的是输入矩阵元素的一个小子集。这种集中的访问行为称为局部性 (*locality*).

![Tiling M and N to Utilize Shared Memory](https://note.youdao.com/yws/api/personal/file/WEBae4e4638b9bd98256ba51c1f67530364?method=download&shareKey=d9458ef5fa7b14ad19e1f0e72defa1b7 "Tiling M and N to Utilize Shared Memory")

## 5.4 A Tiled Matrix Multiplication Kernel

&emsp;&emsp;按照上述方法编写的内核函数如下。如下图所示，x 轴方向上坐标为 bx 和 tx 的线程应该负责计算 P 中索引为 `bx * tile_width + tx` 元素。类似地，y 轴方向上线程要处理的 P 中索引为 `by * tile_width + ty`. 外循环的每次迭代对应于计算的一个阶段。两次调用 `__syncthreads()` 的原因不同，第一次被称为写后读 (*read-after-write*) 依赖关系，因为线程在尝试读取数据之前必须等待其他线程将数据写入正确的位置。第二种被称为读后写 (*write-after-read*) 依赖，因为线程必须等待所有需要它的线程读取数据，然后才能覆盖它。

{% note info %}
&emsp;&emsp;写后读依赖是一种真正依赖 (*true dependence*)，因为读线程确实需要写线程提供的数据，所以它别无选择，只能等待。读后写依赖关系是伪依赖 (*false dependence*) 关系，因为写线程不需要来自读线程的任何数据。这种依赖性是因为它们访问相同的内存地址，如果它们访问不同的地址，则不存在这种依赖性。
{% endnote %}

```cpp
__global__
void TilingMatrixMulKernel(float* M, float* N, float* P, int width) {

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;
	// Loop over the M and N tiles required to compute the P elemrnt
	for (int ph = 0; ph < width/TILE_WIDTH; ph++) {

		// Collaborative loading of M and N tiles into shared memory
		Mds[ty][tx] = M[Row * width + ph * TILE_WIDTH + tx];
		Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + Col];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; k++) {
			Pvalue += Mds[ty][k] * Nds[k][tx];
		}
		__syncthreads();
		P[Row * width + Col] = Pvalue;
	}
}
```

&emsp;&emsp;Tiling 技术并不是 GPU 上才能实现。CPU 上的 tiling 依赖缓存来将重用的数据保留在芯片上，而 GPU 上的 tiling 则直接地使用共享内存来存储片上数据。CPU 核心通常只运行一个或两个线程，因此线程可以依赖于缓存来保存最近使用的数据。相反，GPU SM 同时运行多个线程以隐藏延迟，些线程会竞争缓存槽，使得 GPU 缓存不太可靠。

## 5.5 Boundary Checks

&emsp;&emsp;我们需要扩展 tiling 矩阵乘法内核使其处理任意大小的矩阵。下图展示了 block(0,0) 在 phase 1 的内存访问模式。在不进行边界检查时 thead(0,1) 试图访问 M(0,3) 时实际上获得的是 M(1,0). 同样在 Block(1,1) 在 phase 0 访问时也会出现类似的问题。因此在加载所需的 M 和 N 的 tile 时边界条件为两个索引都小于 Width: `Row < Width && (ph * TILE_WIDT + tx) < Width`，否则将 0.0f 存入对应位置。

![Memory Access of Block(0,0) in Phase 1](https://note.youdao.com/yws/api/personal/file/WEB3510a9da44c7c085a52a178e1a9f1381?method=download&shareKey=ba4ad85a9a2fa48d75e5911e34f0342e "Memory Access of Block(0,0) in Phase 1")

&emsp;&emsp;扩展为一般的矩阵乘法内核是很容易的。将 Width 参数替换为三个无符号整数参数: m, k, n; 将用于指代 M 的行数/列数和 P 的行数/列数的 Width 替换为 m/n；将用于指代 M 的列数和 P 的行数的 Width 替换为 k. 修改后代码如下

![Calculation of the Matrix Indexes in Tiled Multiplication](https://note.youdao.com/yws/api/personal/file/WEB4072b51d6111b15f740c4efb83237a0f?method=download&shareKey=629c82b8ccd78e137a597b5a3e364c4b "Calculation of the Matrix Indexes in Tiled Multiplication")

```cpp
__global__
void GEMMKernel(float* M, float* N, float* P, int m, int n, int k) {

	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x;  int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the P element to work on
	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;

	float Pvalue = 0;
	// Loop over the M and N tiles required to compute the P element
	for (int ph = 0; ph < (k + TILE_WIDTH - 1) / TILE_WIDTH; ph++) {

		// Collaborative loading of M and N tiles into shared memory
		if (Row < m && ph * TILE_WIDTH + tx < k) {
			Mds[ty][tx] = M[Row * k + ph * TILE_WIDTH + tx];
		} else {
			Mds[ty][tx] = 0.0f;
		}

		if (ph * TILE_WIDTH + ty < k && Col < n) {
			Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * n + Col];
		} else {
			Nds[ty][tx] = 0.0f;
		}

		__syncthreads();

		for (int i = 0; i < TILE_WIDTH; i++) {
			Pvalue += Mds[ty][i] * Nds[i][tx];
		}

		__syncthreads();
	}

	if (Row < m && Col < n) {
		P[Row * n + Col] = Pvalue;
	}
}
```

## 5.6 Impact of Memory Usage on Occupancy

&emsp;&emsp;CUDA 设备提供有限的资源限制了可以同时在给定程序的 SM 中分配的线程数量。上面代码不支持主机代码对共享内存使用情况的任何动态调整，因为共享内存使用的大小是一个常量。
&emsp;&emsp;解决的方法是共享内存声明前添加一个 `extern` 关键字，并在声明中省略数组的大小。当调用内核时，可以根据设备查询结果动态配置每个块要使用的共享内存量，并将其作为第三个执行配置参数提供给内核调用。然后将数组中每个部分的大小作为参数传递给内核函数。

```cpp
size = ...;
matrixMulKernel<<<dimGrid,dimBlock,size>>>(Md，Nd，Pd, Width，size/2，size/2);

__global__
void matrixMulKernel(float* M, float* N,float* P,int width, unsigned Mdz_sz, unsigned Nds_sz) {
    extern __shared__ char float Mds_Nds[];
    float *Mds = (float *) Mds_Nds;
    float *Nds = (float*) Mds_Nds + Mds_sz;
}
```
