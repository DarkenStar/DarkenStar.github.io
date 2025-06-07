---
title: PMPP Learning-Chapter 10 Reduction and Minimizing Divergence
date: 2024-09-10T21:07:12+08:00
lastmod: 2024-09-10T21:07:12+08:00
draft: false
author: ["WITHER"]
keywords: 
    - CUDA
categories:
    - CUDA
tags:
    - PMPP learning
description: Personal notebook 10 of Programming Massively Parallel 
summary: Personal notebook 10 of Programming Massively Parallel  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# 10 Reduction and Minimizing Divergence

归约 (*Reduction*) 是从输入数组计算出一个数的运算。

## 10.1 Background

归约是从输入数组计算出一个数的运算，通常是通过对数组中的元素进行某种二元运算来实现的。如果二元操作符具有定义良好的恒等值 (例如加法中的 0，乘法中的 1)，则可以为基于该操作符进行运算的一个数组中的值定义归约操作。可以通过顺序遍历数组的每个元素来进行归约。下面伪代码为运算符的一般归约形式，它被定义为接受两个输入并返回一个值的函数。
```cpp {linenos=true}
acc = IDENTITY;
for (i = 0; i < n; i++) {
    acc = Operator(acc, input[i]);
}
```

## 10.2 Reduction Trees

并行归约的基本思想如下图所示，时间竖直向下增加，水平方向为线程在每个时间点并行执行的活动。并行约简假定输出不随着输入值进行运算的顺序而改变 (即具有交换律)。

![A Parallel Max Reduction Tree](https://note.youdao.com/yws/api/personal/file/WEB3cdc79f24938b60e8ec694cf1efa2314?method=download&shareKey=81f05475b6bb20cc0a5a2f767e1c4d51 "A Parallel Max Reduction Tree")

上图中的并行归约模式被称为归约树 (*reduction tree*)，因为它看起来像一棵叶子是原始输入元素，根是最终结果的树。归约树的边是无实际意义，只是反映了从一个时间步执行的操作到下一个时间步执行的操作的信息流。执行的操作总数是一个几何级数 $\frac{1}{2}N + \frac{1}{2^2}N + \cdots + \frac{1}{N}N = N-1$. 归约树需要 $log_{2}{N}$ 步骤来完成。完成计算所需的资源数量随着时间步的增加而迅速减少，每个时间步的并行度与所需的执行单元数量相同。并行度和资源消耗随着时间步长的剧烈变化让归约树成为一种具有挑战性的并行模式。

## 10.3 A Simple Reduction Kernel

从实现一个**在单个线程块内**执行求和归约树的内核开始。其并行执行的情况如下图所示，假设输入数组位于全局内存中，并且在调用内核函数时将其指针作为输入参数传入。每个线程被分配到索引`2*threadIdx.x` 处，每一步归约的结果也会被写入此处。

![Threads Arrangment of the Input Array in the Simple Kernel](https://note.youdao.com/yws/api/personal/file/WEB1a32016b07d2b257c2c1aa0bdd008b25?method=download&shareKey=618b57308d448ccbd4dcb5e12b391ac0 "Threads Arrangment of the Input Array in the Simple Kernel")

对应的内核代码如下所示，for 循环中的 __syncthreads() 确保任何一个线程开始下一次迭代之前，所有线程都已经完成了上一次迭代的计算。

```cpp {linenos=true}
__global__
void SimpleReductionKernel(float* input, float* output) {  // launch single block with  1/2 #elements threads
	unsigned int i = threadIdx.x * 2;
	for (unsigned int stride = 1; stride = blockDim.x; stride *= 2) {
		if (threadIdx.x % 2 == 0) {
			input[i] += input[i + stride];
		}
		__syncthreads();  // Ensure partial sums have been written to the destinition.
	}
	if (threadIdx.x == 0) {
		*output = input[0];
	}
}
```

## 10.4 Minimizing Control Divergence

上面代码在每次迭代中对活动和非活动线程的管理导致了控制发散。只有那些线程的 `threadIdx.x` 为偶数的线程在第二次迭代中执行加法操作。由于控制发散造成的执行资源浪费随着迭代次数的增加而增加，第二次迭代中每个 warp 只有一半的线程执行加法操作，但消耗的计算资源却是相同的。如果输入数组的大小大于32，整个 warp 将在第五次迭代后不再执行加法操作。消耗的执行资源的总数与所有迭代中活动 warp 的总数成正比，计算方式如下。

$$\text{active warps} = (5+\frac{1}{2}+\frac{1}{4}+\cdots+1)*\frac{N}{64}*32$$

其中 N/64 代表启动的 warp 总数。每个 warp 在前五次迭代中都处于活动状态，之后每次迭代都只有上次一半的线程在活动状态，直到只剩最后一个。
每次迭代中活动线程计算出的结果个数等于活动线程的总数

$$\text{active threads} = \frac{N}{64}*(32+16+8+4+2+1)+\frac{N}{64}*\frac{1}{2}*1+\frac{N}{64}*\frac{1}{4}*1+\cdots+1$$

每个 warp 在前五次迭代中处于活动状态的线程数减半，之后每次迭代中每个处于活动状态的 warp 只有一个线程处于活动状态。这个结果应该非常直观的，因为其正等于完成归约所需的操作总数。
由此我们可以得出当输入大小为 256 时，执行资源利用率为 255/736 = 0.35. 如下图所示，为了减少控制分散应该安排线程和它们计算的位置使得能够随着时间的推移而彼此靠近。也就是说，我们希望步幅随着时间的推移而减少，而不是增加。修改后的内核函数如下，每次迭代中执行加法操作的线程数是相同的，但直到同时进行加法的线程数小于 32 之前，一个 warp 的线程数所走的分支相同。

![Arrangement with Less Control Divergence](https://note.youdao.com/yws/api/personal/file/WEB284d0ca9febf8b2fdd3644aee218b475?method=download&shareKey=a97e91212ca491a84532186a3398b9a4 "Arrangement with Less Control Divergence")

```cpp {linenos=true}
__global__
void ConvergentSumReductionKernel(float* input, float* output) {
	unsigned int i = threadIdx.x;
	for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {  // Decrease stride to reduce control divergence
		if (threadIdx.x < stride) {
			input[i] += input[i + stride];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		*output = input[0];
	}
}
```

这种情况下的进行规约操作消耗的计算资源总数为
$$(\frac{N}{64}*1 + \frac{N}{64}*\frac{1}{2}*1 + \frac{N}{64}*\frac{1}{4}*1 + \cdots + 1 + 5*1) * 32 $$
5*1 代表最后的五次迭代，只有一个活动的warp，并且它的所有32个线程都消耗执行资源，即使只有一小部分线程是活动状态。执行资源的利用率为 255/384 = 0.66.

## 10.5 Minimizing Memory Divergence

上面的内核还有内存分散的问题。在每次迭代中，每个线程对全局内存执行 2 次读取和 1 次写入。第一次从自己的位置读取，第二次从离自己 stride 的位置读取，相加后写入到自己的位置。
[10.3](#10-3-A-Simple-Reduction-Kernel) 节的内核代码中，第一次迭代每个 warp 中的相邻线程间隔 2 个元素，因此要访问 2 个内存位置，此后每次迭代 stride 都增加，直到第六次迭代时，每个 warp 都只有一个线程处于活动状态，只用访问 1 个位置。因此进行内存访问的总次数为
 $$(5*\frac{N}{64}*2+\frac{N}{64}*1+\frac{N}{64}*\frac{1}{2}*1+\cdots+1)*3$$ 
[10.4](#10-4-Minimizing-Control-Divergence) 节的内核代码中，每个 warp 在任何读或写时只进行一个全局内存请求，直到该 warp 中的所有线程都处于非活动状态。最后五次迭代的线程都位于一个 warp 中，因此进行内存访问的总次数为
$$((\frac{N}{64}*1+\frac{N}{64}*\frac{1}{2}*1+\frac{N}{64}*\frac{1}{4}*1+\cdots+1)+5)*3$$
对于长度为 2048 的输入，前者和后者全局内存请求的总数分别为 1149 和 204. 后者在使用 DRAM 带宽方面也具有更高的效率。

## 10.6 Minimizing Global Memory Accesses
通过使用共享内存，可以进一步改进 [10.4](#10-4-Minimizing-Control-Divergence) 节的内核。在每次迭代中，线程将它们的部分和结果值写入全局内存，这些值在下一次迭代中由相同的线程和其他线程重新读取。如下图所示，通过将部分和结果保存在共享内存中，可以进一步提高执行速度。

![Use Shared Memory to Reduce Accesses from the Global Memory](https://note.youdao.com/yws/api/personal/file/WEB73d407923a5daa0e9e69860be0089c98?method=download&shareKey=b6e4d5ed19de5a3712f1ed1eed127674 "Use Shared Memory to Reduce Accesses from the Global Memory")
对应的代码如下，每个线程从全局内存加载并 2 个输入元素并将部分和写入共享内存。剩下的所有迭代中的计算都在共享内存中进行。

```cpp {linenos=true}
#define BLOCK_DIM 512
__global__
void SharedMemoryReductionKernel(float* input) {
	__shared__ float input_s[BLOCK_DIM];
	unsigned int i = threadIdx.x;
	input_s[i] = input[i] + input[i + blockDim.x];  // Partial sum of first iteration
	for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		__syncthreads();  // Ensure all partial sums have been written to shared memory
		if (threadIdx.x < stride) {
			input_s[i] += input_s[i + stride];  // Partial sum of subsequent iterations
		}
	}
	if (threadIdx.x == 0) {
		input[0] = input_s[0];  // Write final sum to output
	}
}
```

全局内存访问的次数减少到初始加载输入数组和最终写入 `input[0]`，总共只有 (N/32) + 1 个全局内存请求。

## 10.7 Hierarchical Reduction for Arbitrary Input Length

由于 `__syncthreads()` 只对同一块中的线程有效，因此无法在不同块之间同步。下图展示了如何使用分级归约来解决这个问题，其思想是将输入数组划分为多个适合于线程块大小的段。然后，所有块都独立地执行归约树，并使用原子加法操作将它们的结果累积到最终输出。

![Segmented Multiblock Reduction Using Atomic Operations](https://note.youdao.com/yws/api/personal/file/WEB857e94f9f4cbf28c1f9a7980fdd83536?method=download&shareKey=29a4f000e0073fa0ae1ee6e4b8e08852 "Segmented Multiblock Reduction Using Atomic Operations")
对应的内核代码如下。每个线程块处理 `2*blockDim.x` 个元素。在每个线程块内，我们通过线程所属块的段起始位置加上 `threadIdx.x` 为每个线程分配其输入元素的位置。

```cpp {linenos=true}
__global__
void SegmentedSumReductionKernel(float* input, float* output) {
	__shared__ float input_s[BLOCK_DIM];
	unsigned int segment = blockIdx.x * blockDim.x * 2;  // Each block processes 2*blockDim.x elements
	unsigned int i = segment + threadIdx.x;
	unsigned int t = threadIdx.x;

	input_s[t] = input[t + blockDim.x];  // Partial sum of first iteration of each block
	for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		__syncthreads();  // Ensure all partial sums have been written to shared memory
		if (t < stride) {
			input_s[t] += input_s[t + stride];  // Partial sum of subsequent iterations
		}
	}
	if (t == 0) {
		atomicAdd(&output, input_s[0]);  // Write final sum to output
	}
}
```

## 10.8 Thread Coarsening for Reduced Overhead

到目前为止，我们使用过的归约内核都试图通过使用尽可能多的线程来最大化并行性。若线程块大小为 1024 个线程，则需要启动的线程块数量为 N/2048. 
下图展示了如何将线程粗化。线程独立地添加它们负责的四个元素，它们不需要同步，直到将所有的四个元素相加之后才能将部分和结果写入共享内存。剩下的步骤与 10.7 节后续相同。

![Thread Coarsening in Reduction](https://note.youdao.com/yws/api/personal/file/WEB4358648a96bbd27f045c7beb2f736e73?method=download&shareKey=770c7cecefe849a2acfe2353f538504a "Thread Coarsening in Reduction")
对应的内核如下，我们乘以 `COARSE_FACTOR` 来表示每个线程块的负责的段的大小是原来的 `COARSE_FACTOR` 倍。部分和累加到局部变量 `sum` 中，并且因为线程是独立运行的，在循环中不会调用 `__syncthreads()`.

```cpp {linenos=true}
#define COARSE_FACTOR 2
__global__
void CoarsenedSumReductionKernel(float* input, float* output) {
	__shared__ float input_s[BLOCK_DIM];
	unsigned int segment = blockIdx.x * COARSE_FACTOR * blockDim.x * 2;
	unsigned int i = segment + threadIdx.x;
	unsigned int t = threadIdx.x;
	float sum = input[i];
	for (int tile = 1;  tile < COARSE_FACTOR; tile++) {  // Partitial sum is accumulated independently
		sum += input[i + tile * blockDim.x];
	}
	input_s[t] = sum;
	for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
		__syncthreads();
		if (t < stride) {
			input_s[t] += input_s[t + stride];
		}
	}
	if (t == 0) {
		atomicAdd(&output, input_s[0]);
	}
}
```

下图比较了两个原始线程块在没有进行线程粗化下被硬件顺序执行情况，图 A 当第一个线程块完成后，硬件调度第二个线程块，在不同的数据段上执行相同的步骤。图 B 的这个线程块开始需要三个步骤，其中每个线程对它负责的四个元素求和。剩下的三个步骤执行归约树，每个步骤中有一半的线程退出活动状态。相比图 A，图 B 只需要6个步骤 (而不是 8 个)，其中 3 个步骤 (而不是 2 个) 充分利用了硬件。
当我们粗化线程时，并行完成的工作就会减少。因此，增加粗化因子将减少硬件正在利用的数据并行性的数量。

![Comparing Parallel Reduction with and without Thread Coarsening](https://note.youdao.com/yws/api/personal/file/WEBa50e84d0dc2bb278fdf643ea81cfd694?method=download&shareKey=c8899f1ad2ad8059daf269e68a9dac2c "Comparing Parallel Reduction with and without Thread Coarsening")