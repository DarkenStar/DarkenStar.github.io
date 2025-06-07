---
title: PMPP Learning-Chapter 9 Parallel Histogram-An Introduction to Atomic Operations and Privatization
date: 2024-09-09T10:27:42+08:00
lastmod: 2024-09-09T10:27:42+08:00
author: ["WITHER"]

categories:
- CUDA

tags:
- PMPP learning

keywords:
- CUDA

description: "Personal notebook 9 of Programming Massively Parallel Processors." # 文章描述，与搜索优化相关
summary: "Personal notebook 9 of Programming Massively Parallel Processors." # 文章简单描述，会展示在主页
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
# 9 Parallel Histogram-An Introduction to Atomic Operations and Privatization

本章介绍并行直方图计算模式，其中每个输出元素都可以由任何线程更新。因此，当线程更新输出元素时必须注意线程之间的协调，避免任何可能破坏最终结果的干扰。

## 9.1 Background

直方图是数据集中数据值出现的数量计数或百分比的显示。在最常见的直方图形式中，间隔区间沿水平轴绘制，每个间隔中的数据值计数表示为从水平轴上升的矩形或条形的高度。
许多应用领域依赖于直方图来总结数据集进行数据分析。其中一个领域就是计算机视觉。图像子区域直方图的计算过程是计算机视觉中特征 (图像中感兴趣的模式) 提取的重要方法。

![A Histogram Representation of  “programming massively parallel processors”](https://note.youdao.com/yws/api/personal/file/WEBf1b2ad27e9184b6ecf628e068c42e7e4?method=download&shareKey=a44ec6cce3fffb07208a7a67ee8005a5 "A Histogram Representation of  “programming massively parallel processors”")

## 9.2 Atomic Operations and A Basic Histogram Kernel

如下图所示，并行化直方图计算的最直接的方法是启动数据一样多的线程，让每个线程处理一个元素。每个线程读取其分配的输入元素，并增加对应的隔计数器的值。

![Basic Parallelization of a Histogram](https://note.youdao.com/yws/api/personal/file/WEBd162f505e52265f5421f0fa883e5d19b?method=download&shareKey=4d8e366384fcc19aebe518ad8fecdc7d "Basic Parallelization of a Histogram")

histo 数组中间隔计数器的增加是对内存位置的更新或 read-modify-write 操作。该操作包括读取内存位置(读)，在原始值上加 1(修改)，并将新值写回内存位置 (写)。在实际过程中会出现读-修改-写竞争条件 (*read-modify-write race condition*)，在这种情况下，两个或多个同步更新操作的结果会根据所涉及的操作的相对时间而变化。
下图 A 中线程 1 在时间段 1~3 期间完成了其读-修改-写序列的所有三个部分，然后线程 2 在时间段 4 开始，最后结果正确。在图 B 中，两个线程的读-修改-写顺序重叠。线程 1 在时间段 4 时将新值写入 `histo[x]`。当线程 2 在时间段 3 读取 `histo[x]`时，它的值仍然是 0，因此最后的写入的值是 1.

![Race Condition in Updating a histo Array Element](https://note.youdao.com/yws/api/personal/file/WEBfcb1b1249b8c3eaa4b079cc3c6211f60?method=download&shareKey=6c860292730da34f80c4f3020f5e709c "Race Condition in Updating a histo Array Element")

原子操作 (*atomic operation*) 的读、修改和写部分构成一个不可分割的单元，因此称为原子操作。对该位置的其他读-修改-写序列不能与其在时间上有重叠。需要注意*原子操作在线程之间不强制任何特定的执行顺序*，比如线程 1 可以在线程 2 之前或之后运行。CUDA内核可以通过函数调用对内存位置执行原子加法操作:

```cpp
int atomicAdd(int* address, int val);
```

`atomicAdd` 是一个内建函数 (intrinsic function)，它被编译成一个硬件原子操作指令。该指令读取全局或共享内存中 `address` 参数所指向的32位字，将 `val` 加上旧值中并写入结果回相同地址的内存中。该函数返回地址处的旧值。

{{< details title="Intrinsic Functions">}}
现代处理器通常提供特殊指令，这些指令要么执行关键功能 (如原子操作)，要么大幅提高性能 (如矢量指令)。这些指令通常作为内建函数暴露给程序员，从程序员的角度来看，这些是库函数。然而，它们被编译器以一种特殊的方式处理。每个这样的调用都被翻译成相应的特殊指令。在最终代码中没有函数调用，只有与用户代码一致的特殊指令。
{{< /details >}}

```cpp
__global__ 
void histo_kernel(char* data, unsigned int length, unsigned int* histo) {
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < length) {
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			atomicAdd(&histo[alphabet_position / 4], 1);
		}
	}
}
```

## 9.3 Latency and Throughput of Atomic Operations

高内存访问吞吐量的关键是同时进行许多 DRAM 访问。然而，当许多原子操作更新相同的内存位置时，一个后面线程的读-修改-写序列在前一个线程的写操作结束之前不能开始，即如下图所示，同时只能有一个线程在同一内存位置执行原子操作。更新这些间隔的大量争用流量会使得吞吐量降低。

![The Execution of Atomic Operations at the Same Location](https://note.youdao.com/yws/api/personal/file/WEBe7de6cb4432d570997a5d51a354269df?method=download&shareKey=244ed14ff414262053d92e3b884321b4 "The Execution of Atomic Operations at the Same Location")

提高原子操作吞吐量的一种方法是减少对竞争严重的位置的访问延迟。现代 GPU 允许在被所有 SM 共享的最后一级缓存中执行原子操作。由于对最后一级缓存的访问时间是几十个周期而不是几百个周期，因此原子操作的吞吐量与早期GPU相比至少提高了一个数量级。

## 9.4 Privatization

提高原子操作吞吐量的另一种方法是通过引导流量远离竞争严重的位置。这可以通过一种称为私有化 (*privatization*) 的技术来实现。其思想是将高度竞争的输出数据结构复制到私有副本中，以便线程的每个子集都可以更新其私有副本。
下图展示了如何将私有化应用于直方图统计。每个线程块由 8 个线程组成，争用只会在同一块中的线程之间以及在最后合并私有副本时发生，而不是更新相同直方图 bin 的所有线程之间发生争用。

![Reduce Contention of Atomic Operations by Private Copies of Histogram](https://note.youdao.com/yws/api/personal/file/WEBb7aa2df247e5dc432476efa8b601df20?method=download&shareKey=82f3b5074429b6bf29ca79a092a0044d "Reduce Contention of Atomic Operations by Private Copies of Histogram")

一个私有化版本的代码如下，为 histo 数组分配足够的设备内存 (`gridDim.x*NUM_BINS*4` bytes) 来保存直方图的所有私有副本。在执行结束时，每个线程块将把私有副本中的值提交到 块 0 的部分。

```cpp
#define NUM_BINS 7  // # histo bins 
__global__
void histo_private_kernel(char* data, unsigned int length, unsigned int* histo) {

	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < length) {
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			atomicAdd(&histo[blockIdx.x * 7 + alphabet_position / 4], 1);
		}
	}
	if (blockIdx.x > 0) {
		__syncthreads();
		//
		for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) { 
			unsigned int binValue = histo[blockIdx * NUM_BINS + bin];
			atomicAdd(&histo[bin], binValue);
		}
	}
}
```

在每个线程块的基础上创建直方图的私有副本的一个好处是线程可以在提交自己的统计结果之前使用 `__syncthreads()` 来等待彼此。另一个好处是，如果直方图中的 bin 数量足够小，则可以在共享内存中声明直方图的私有副本 (每个线程块一个)。下面代码直方图在共享内存中分配私有副本 `histo_s` 数组，并由块的线程并行初始化为 0.

```cpp
__global__
void histo_shared_private_kernel(char* data, unsigned int length, unsigned int* histo) {

	// Initializing private bins
	__shared__ unsigned int histo_s[NUM_BINS];
	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
		histo_s[bin] = 0;
	}
	__syncthreads();

	// Histogram
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i < length) {
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			atomicAdd(&histo_s[alphabet_position / 4], 1);
		}
	}
	__syncthreads();

	// Commit to global memory
	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
		unsigned binValue = histo_s[bin];
		if (binValue > 0) {
			atomicAdd(&histo[bin], binValue);
		}
	}
}
```

## 9.5 Coarsening

私有化的开销是需要将私有副本提交到公共副本。每个线程块都会执行一次提交操作，因此，使用的线程块越多，这个开销就越大。如下图所示，我们可以通过减少块的数量来减少私有副本的数量，从而减少提交到公共副本的次数，让每个线程处理多个输入元素。

![Contiguous Partition of Input Elements](https://note.youdao.com/yws/api/personal/file/WEB685390a635126cba569f8c85254bcfc5?method=download&shareKey=bf894427e4a2c0fdce32f9bf00094c52 "Contiguous Partition of Input Elements")

下面代码是一个连续分区 (*contiguous partition*) 策略的示例，输入被连续划分成多个段，每个段被分配给一个线程，每个线程从 `tid*CFACTOR` 迭代到 `(tid+1)*CFACTOR` 进行所负责部分的统计。

```cpp
#define CFACTOR 3
__global__
void histo_shared_private_contiguous_kernel(char* data, unsigned int length, unsigned int* histo) {
{

	// Initializing private bins
	__shared__ unsigned int histo_s[NUM_BINS];
	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
		histo_s[bin] = 0;
	}
	__syncthreads();

	// Histogram
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned int i = tid * CFACTOR; i < (tid + 1)*CFACTOR && i < length; i++) {
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			atomicAdd(&histo_s[alphabet_position / 4], 1);
		}
	}
	__syncthreads();

	// Commit to global memory
	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
		unsigned binValue = histo_s[bin];
		if (binValue > 0) {
			atomicAdd(&histo[bin], binValue);
		}
	}
}
```

上述在 GPU 上连续分区的思路会导致内存不友好的访问模式，因为 threadIdx 相同的线程访问的不是一块连续的内存区域。因此我们要采用交错分区 (*interleaved partition*)，如下图所示，即不同线程要处理的分区彼此交错。实际应用中每个线程在每次迭代中应该处理 4 个 char (一个 32 位字)，以充分利用缓存和 SMs 之间的互连带宽。

![Interleaved Partition of Input Elements](https://note.youdao.com/yws/api/personal/file/WEB166097c6dec3a7d7eed2c82d5706bf55?method=download&shareKey=79fb19c70cf16d76fdc9113eeefd12e8 "Interleaved Partition of Input Elements")

下面代码是一个交错分区的示例。在循环的第一次迭代中，每个线程使用其全局线程索引访问数据数组:线程 0 访问元素 0，线程 1 访问元素 1，线程 2 访问元素 2...所有线程共同处理输入的第一个 `blockDim.x*gridDim.x` 元素。

```cpp
__global__
void histo_shared_private_interleaved_kernel(char* data, unsigned int length, unsigned int* histo) {
{

	// Initializing private bins
	__shared__ unsigned int histo_s[NUM_BINS];
	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
		histo_s[bin] = 0;
	}
	__syncthreads();

	// Histogram
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			atomicAdd(&histo_s[alphabet_position / 4], 1);
		}
	}
	__syncthreads();

	// Commit to global memory
	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
		unsigned binValue = histo_s[bin];
		if (binValue > 0) {
			atomicAdd(&histo[bin], binValue);
		}
	}
}
```

## 9.6 Aggregation

一些数据集在局部区域有大量相同的数据值。如此高度集中的相同值会导致严重的争用，并降低并行直方图计算的吞吐量。一个简单而有效的优化是，如果每个线程正在更新直方图的相同元素，则将连续的更新聚合为单个更新。下面的代码展示了聚合的直方图计算。

```cpp
__global__
void histo_shared_private_interleaved_aggregated_kernel(char* data, unsigned int length, unsigned int* histo) {

	// Initializing private bins
	__shared__ unsigned int histo_s[NUM_BINS];
	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
		histo_s[bin] = 0;
	}
	__syncthreads();

	// Histogram
	unsigned int accumulator = 0;
	int prevBinIdx = -1;
	unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
	for (unsigned int i = tid; i < length; i += blockDim.x * gridDim.x) {
		int alphabet_position = data[i] - 'a';
		if (alphabet_position >= 0 && alphabet_position < 26) {
			int currBinIdx = alphabet_position / 4;
			if (currBinIdx != prevBinIdx) {  // Update previous statistics
				if (accumulator > 0) {
					atomicAdd(&histo_s[prevBinIdx], accumulator);
				}
				accumulator = 1;
				prevBinIdx = currBinIdx;
			} else {  // Accumulate statistics
				accumulator++;
			}
		}
	}
	if (accumulator > 0) {  // Update last bin
		atomicAdd(&histo_s[prevBinIdx], accumulator);
	}
	__syncthreads();

	// Commit to global memory
	for (unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
		unsigned binValue = histo_s[bin];
		if (binValue > 0) {
			atomicAdd(&histo[bin], binValue);
		}
	}
}
```

可以看出聚合内核需要更多的语句和变量。添加的 if 语句可能会出现控制发散。然而，如果没有争用或存在严重的争用，就很少有控制发散，因为线程要么都在增加累加器值，要么都在连续刷新。