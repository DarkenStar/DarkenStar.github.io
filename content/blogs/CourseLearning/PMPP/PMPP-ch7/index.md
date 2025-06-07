---
title: PMPP Learning-Chapter 7 Convolution-An Introduction to Constant Memory and Caching
date: 2024-09-06T10:50:42+08:00
lastmod: 2024-09-06T10:50:42+08:00
author: ["WITHER"]

categories:
- CUDA

tags:
- PMPP learning

keywords:
- CUDA

description: "Personal notebook 7 of Programming Massively Parallel Processors." # 文章描述，与搜索优化相关
summary: "Personal notebook 7 of Programming Massively Parallel Processors." # 文章简单描述，会展示在主页
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
# 7 Convolution-An Introduction to Constant Memory and Caching

卷积的每个输出数据元素可以相互独立地计算，这是并行计算的理想特性。另一方面，在处理具有边界条件的输出数据元素时，有大量的输入数据共享。这使得卷积可以实现复杂的 tiling 方法和输入数据分段方法。

## 7.1 Background

输入数据向量 $[x_0, x_1, \cdots, x_{n-1}]$ 和包含 2r+1 个元素的 filter 数组 $[f_0, f_1, \cdots, f_{2r}]$， 1D卷积计算公式为
 $$y_i=\sum_{j=-r}^rf_{i+j}\times x_i$$ 
同样对于 `n*n` 大小的二维输入，和 `r*r` 大小的 filter，2D 卷积计算公式为
$$P_{y,x}=\sum_{j=-r_y}^{r_y}\sum_{k=-r_x}^{r_x}f_{y+j,x+k}\times N_{y,x}$$

## 7.2 Parallel Convolution: a Basic Algorithm

假设二维卷积内核接收五个参数: 输入数组 N 的指针; 滤波器 F 的指针; 输出数组 P 的指针; 方形滤波器的半径 r; 输入输出数组的宽度; 输入和输出数组的高度。如下图所示，一个简单的并行方式是网格中的每个线程计算与自身坐标相同的输出像素。对应的内核函数代码如下，浮点计算与全局内存访问的比仅为 0.25 OP/B (每加载 8 字节执行 2 次运算)

![Parallelization and Thread Organization for 2D Convolution](https://note.youdao.com/yws/api/personal/file/WEBb705cd006867704636e9e5261467570f?method=download&shareKey=d5710b7dee0a3c91d67011d92b623557 "Parallelization and Thread Organization for 2D Convolution")

```cpp {linenos=true}
__global__
void convolution_2D_basic_kernel (float *N, float *F, float *P, 
									int r, int width, int height) 
{
	int outCol = blockIdx.x * blockDim.x + threadIdx.x;
	int outRow = blockIdx.y * blockDim.y + threadIdx.y;
	int Pvalue = 0.0f;
	for (int fRow = 0; fRow < 2*r+1; fRow++) {
		for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
			int inRow = outRow - r + fRow;
			int inCol = outCol - r + fCol;
			if (inRow > 0 && inRow < height &&
				inCol > 0 && inCol < width) {
				Pvalue += P[inRow * width + inCol] * F[fRow * r + fCol];
			}
		}
	}
	P[outRow * width + outCol] = Pvalue;
}
```

## 7.3 Constant Memory and Caching

可以发现卷积核 F 通常很小，在整个卷积内核的执行过程中不会改变，所有线程都以相同的顺序访问其元素。因此我们可以考虑将其存储在常量内存里，之前说过它和全局内存的区别是线程不能修改常量内存变量的值并且常量内存非常小，目前为 64 KB. 
假设已经在主机代码里分配好 F_h 的内存，可以通过 `cudaMemcpyToSymbol()` 将其从主机内存传输到设备常量内存中。内核函数以全局变量的形式访问常量内存变量。因此，它们的指针不需要作为参数传递给内核函数。


**如果主机代码和内核代码位于不同的文件中，内核代码文件必须包含相关的外部声明的头文件，以确保声明对内核可见。**


CUDA runtime 知道常量内存变量在内核执行期间不会被修改，因此会让硬件在内核执行期间直接缓存常量内存变量。在不需要支持写的情况下，可以在减小芯片面积和降低功耗的情况下设计用于常量内存变量的专用缓存，被称为常量缓存 (`constant caching`).

## 7.4 Tiled Convolution with Halo Cells

我们定义输出 tile 为每个块处理的输出元素，输入 tile 为计算输出 tile 中元素所需的输入元素的集合。下图给出了一个例子，可以看到输入 tile 大小和输出 tile 大小之间的差异使 tile 卷积核的设计变得复杂。有两种线程组织可以处理这种差异。
- 启动与输入 tile 具有相同维度的线程块。这样因为每个线程只需要加载一个输入元素。但由于输入 tile 比对应的输出 tile 大，在计算输出元素时需要禁用一些线程，降低了资源利用率。
- 启动与输出 tile 具有相同维度的线程块。这样线程需要迭代以确保加载所有输入 tile 元素。但简化了输出元素的计算。

![Input Tile vs. Output Tile in 2D Convolution](https://note.youdao.com/yws/api/personal/file/WEBda4dfd50e011362c0cc68caaf130a16d?method=download&shareKey=b534279430a6d88b51d9523c3cdf486b "Input Tile vs. Output Tile in 2D Convolution")

第一种线程组织方式的内核如下。现在每个块中的线程共同执行 `OUT_TILE_DIM^2*(2*FILTER_RADIUS+1)` 次浮点运算。分配给输入 tile 元素的每个线程加载一个4字节的输入值。因此每个block加载 `IN_TILE_DIM^2*4=(OUT_TILE_DIM+2*FILTER_RADIUS)^2*4`
```cpp {linenos=true}
#define IN_TILE_DIM 32
#define FILTER_RADIUS 5
#define OUT_TILE_DIM (IN_TILE_DIM - 2*(FILTER_RADIUS))

__constant__ float F_c[2 * FILTER_RADIUS + 1][FILTER_RADIUS + 1];

__global__
void convolution_tiled_2D_constant_mem_kernel_1(
	float* N, float* P, int width, int height)
{
	// Upper left input tile coord
	int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
	int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

	// Loading input tile
	__shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
	if (row >= 0 && row < height && col >= 0 && col < width) {
		N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
	} else {
		N_s[threadIdx.y][threadIdx.x] = 0.0f;
	}
	__syncthreads();

	// Calculate output elements
	int tileCol = threadIdx.x - FILTER_RADIUS;
	int tileRow = threadIdx.y - FILTER_RADIUS;
	if (row >= 0 && row < height && col >= 0 && col < width &&
		tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {

		float Pvalue = 0.0f;
		for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
			for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
				Pvalue += F_c[fRow][fCol] * N_s[tileRow + fRow][tileCol + fCol];
			}
		}
		P[row * width + col] = Pvalue;
	}
}
```

第二种线程组织方式的内核如下，每个线程现在可能需要加载多个输入 tile 的元素。
```cpp {linenos=true}
__global__
void convolution_tiled_2D_constant_mem_kernel_2(  // OUT_TILE_DIM^2 threads per block
	float* N, float* P, int width, int height) {

	// Upper left output tile coord
	int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
	int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;

	// Each thread may need to load multiple elements into shared memory
	__shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
	for (int i = threadIdx.y; i < IN_TILE_DIM; i += OUT_TILE_DIM) {
		for (int j = threadIdx.x; j < IN_TILE_DIM; j += OUT_TILE_DIM) {
			int in_col = blockIdx.x * OUT_TILE_DIM + j - FILTER_RADIUS;
			int in_row = blockIdx.y * OUT_TILE_DIM + i - FILTER_RADIUS;

			if (in_row >= 0 && in_row < height && in_col >= 0 && in_col < width) {
				N_s[i][j] = N[in_row * width + in_col];
			} else {
				N_s[i][j] = 0.0f;
			}
		}
	}
	__syncthreads();

	// Calculate output elements
	if (threadIdx.x < OUT_TILE_DIM && threadIdx.y < OUT_TILE_DIM && row < height && col < width) {
		float Pvalue = 0.0f;
		for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
			for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
				Pvalue += F_c[fRow][fCol] * N_s[threadIdx.y + fRow][threadIdx.x + fCol];
			}
		}
		P[row * width + col] = Pvalue;
	}
}
```

## 7.5 Tiled Convolution Using Caches for Halo Cells

当一个块需要它的 halo cell 时，由于相邻块的访问，它们已经在二级缓存中了。因此，对这些  halo cell 的内存访问可以从 L2 缓存提供，而不会造成额外的 DRAM 流量。我们可以对原来的 N 进行这些 halo cell 的访问，而不是将它们加载到 `N_ds` 中。代码如下，加载 N_s 变得更简单，因为每个线程可以简单地加载与其分配的输出元素具有相同坐标的输入元素。然而，计算P个元素的循环体变得更加复杂。它需要添加条件来检查 helo cell 和 ghost cell.

```cpp {linenos=true}
__global__
void convolution_tiled_cached_2D_shared_mem_kernel(  // OUT_TILE_DIM^2 threads per block
    float* N, float* P, int width, int height) {

    int col =blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    int row =blockIdx.y * OUT_TILE_DIM + threadIdx.y;

    // loading input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row < height && col < width) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Calculate output elements
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        // turning off the threads at the edge of the block
        for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                if (threadIdx.x + fCol - FILTER_RADIUS >= 0 &&
                    threadIdx.x + fCol - FILTER_RADIUS < IN_TILE_DIM &&
                    threadIdx.x + fRow - FILTER_RADIUS >= 0 &&
                    threadIdx.x + fRow - FILTER_RADIUS < IN_TILE_DIM) {

                    Pvalue += F_c[fRow][fCol] * N_s[threadIdx.y + fRow][threadIdx.x + fCol];
                } else {
                    if (row - FILTER_RADIUS + fRow >= 0 &&
                        row - FILTER_RADIUS + fRow < height &&
                        col - FILTER_RADIUS + fCol >= 0 &&
                        col - FILTER_RADIUS + fCol < width) {

                        Pvalue += F_c[fRow][fCol] * N[(row - FILTER_RADIUS + fRow) * width + (col - FILTER_RADIUS + fCol)];
                    }
                }
            }
        }
        N[row * width + col] = Pvalue;
    }
}
```

- Halo Cell: 实际计算区域周围添加的一圈额外的单元格。本质上是 "虚拟" 单元格，存在于不直接关注的区域之外。
- Ghost Cell: 存储来自相邻 tile 的数据副本，使得 block 在无需直接访问彼此的内存的情况下访问相邻的必要数据。
