---
title: PMPP Learning-Chapter 12 Merge-An Introduction to Dynamic Input Data Identification
date: 2024-09-13T22:58:03+08:00
lastmod: 2024-09-13T22:58:03+08:00
draft: false
author: ["WITHER"]
keywords: 
    - CUDA
categories:
    - CUDA
tags:
    - PMPP learning
description: Personal notebook 12 of Programming Massively Parallel 
summary: Personal notebook 12 of Programming Massively Parallel  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# 12 Merge-An Introduction to Dynamic Input Data Identification

有序归并操作接受两个有序列表并生成一个合并后的有序列表。

## 12.1 Background

假设数组中的每个元素都有一个键并且键定义了一个用 ≤ 表示的顺序关系。下图展示了基于数字排序关系的简单归并函数的操作。一般来说，如果键值相等的元素在输出中的顺序与其在输入中的顺序相同，则称排序操作是稳定的。

![Example of a Merge Operation](https://note.youdao.com/yws/api/personal/file/WEB412d47f606b473f210d022c42a5bfb5a?method=download&shareKey=615ec66182adab314c2bd9f9dbe85bc1 "Example of a Merge Operation")

## 12.2 A Sequential Merge Algorithm

归并操作可以用如下一个简单的顺序算法来实现。顺序归并函数访问 A 和 B 的每个输入元素一次，并向 C 中每个位置写入一次。其算法复杂度为 O(m+n).

```cpp {linenos=true}
void merge_sequential(int* A, int* B, int* C, int m, int n) {
	int i = 0, j = 0, k = 0;  // Indices for A, B, and C
	while (i < m && j < n) {
		if (A[i] < B[j]) {
			C[k++] = A[i++];
		} else {
			C[k++] = B[j++];
		}
		if (i == m) {  // Done with A[], handling remaining B
			while (j < n) {
				C[k++] = B[j++];
			}
		}
		else {  // Done with B[], handling remaining A
			while (i < m) {
				C[k++] = A[i++];
			}
		}
	}
}
```

## 12.3 A Parallelization Approach

每个线程首先确定它将要负责的输出位置范围，并使用该输出范围作为 `co-rank` 函数的输入，以确定所负责 C 输出范围的对应的 A 和 B 输入范围。这样每个线程在它们的子数组上执行顺序合并函数，从而并行地进行合并。

![Examples of Observations](https://note.youdao.com/yws/api/personal/file/WEBcd07196d1e74562f5e44926357511c97?method=download&shareKey=cdae7d4e37b42cde506885c860e9fb5b "Examples of Observations")

- **Observation 1**：子数组 `C[0]-C[k-1]` (k 个元素) 是 `A[0]-A[i-1]` (i 个元素) 和 `B[0]-B[k-i-1]` (k-i 个元素) 的归并结果。
- **Observation 2**：对于任意满足 0≤k≤m+n 的 k，我们可以找到唯一的 i 和 j 使得 k=i+j, 0≤i≤m, 0≤j≤n，并且子数组 `C[0]-C[k-1]` 是子数组 `A[0]-A[i-1]` 和子数组 `B[0]-B[j-1]` 合并的结果。唯一的索引 i 和 j 被称 `C[k]` 的 co-rank.

我们可以通过将输出数组划分为子数组，并让每个线程负责一个子数组的生成来划分工作。由于并行归并算法中每个线程使用的输入元素的范围取决于实际的输入值使得我们需要辅助函数来完成。

## 12.4 Co-rank Function Implementation

将 co-rank 函数定义为接受输出数组 C 中元素的位置 k 和两个输入数组 A 和 B的信息，并返回输入数组 A 对应的 co-rank 值 i.
以下图为例，假设线程 1 的 co-rank 函数的目标是为其秩 k1=4 确定 co-rank值 i1=3 和 j1=1. 也就是说，从 `C[4]` 开始的子数组将由从 `A[3]` 和  `B[1]` 开始的子数组合并生成。我们可以发现线程 t 使用的输入子数组由线程 t 和线程 t+1 的 co-rank 确定。

![Example of co-rank Function Execution](https://note.youdao.com/yws/api/personal/file/WEBc24a7460cb3237cd2404a21d2f9984f9?method=download&shareKey=09ff2b105b2d3b266478ce1b840d92bb "Example of co-rank Function Execution")

目标是找到使得 `A[i - 1] <= B[j]` 并且 `B[j - 1] <= A[i]` 的索引。
- 如果 `A[i-1] > B[j]`，说明 `A[i]` 太大，需要减少 i，并增加 j。
- 如果 `B[j-1] > A[i]`，说明 `B[j]` 太大，需要减少 j，并增加 i。
每次调整时，i 和 j 都按照二分方式调整，即调整的步长是 delta / 2. i 和 i_low 确定了当前正在搜索的数组 A 的范围。

```cpp {linenos=true}
int co_rank(int k, int* A, int m, int* B, int n) {  // C[k] comes from A[i] of B[j]
	// k = i + j
	int i = k < m ? k : m;  // max starting search value for A, i.e. A[k-1] < B[0]
	int i_low = 0 > (k - n) ? 0 : k - n;  // when B is done, min starting search value for A is k-n
	int j = k - i;
	int j_low = 0 > (k - m) ? 0 : (k - m);
	int delta;
	bool active = true;
	while (active) {  // Binary search for C[k]
		if (i > 0 && j < n && A[i - 1] > B[j]) {
			delta = (i - i_low + 1) >> 1;
			j_low = j;
			j += delta;
			i -= delta;
		} else if (j > 0 && i < m && B[j - 1] > A[i]) {
			delta = (j - j_low + 1) >> 1;
			i_low = i;
			i += delta;
			j -= delta;
		} else {  // Found the correct position for C[k]
			active = false;
		}
		return i;
	}
}
```

## 12.5 A Basic Parallel Merge Kernel

在剩下的小节里，我们假设输入数组 A 和 B 存储在全局内存中，一个内核被启动用来合并两个输入数组，输出一个同样位于全局内存中的数组 C.
下面内核是并行归并的直接实现。它首先通过计算当前线程 (`k_curr`) 和下一个线程 (`k_next`) 产生的输出子数组的起点来确定负责输出的范围。然后分别调用自己和后一个线程的 co_rank 函数来确定对应的 A 和 B 输入子数组的范围。最后调用顺序合并函数来合并两个输入子数组，并将结果写入输出子数组。

```cpp {linenos=true}
__global__
void mergre_basic_kernel(int* A, int* B, int* C, int m, int n) {  // Each thread handles a section of C
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int elementsPerThread = ceil(m + n) / (blockDim.x * gridDim.x);
	int start = tid * elementsPerThread;
	int end = std::min(start + elementsPerThread, m + n);
	// Determin the range of A and B to be merged for this thread
	int i_curr = co_rank(start, A, m, B, n);
	int i_next = co_rank(end, A, m, B, n);
	int j_curr = start - i_curr;
	int j_next = end - i_next;
	merge_sequential(A + i_curr, B + j_curr, C + start, i_next - i_curr, j_next - j_curr);
}
```

上面的基本归并内核有 2 个问题：
1. warp 中的相邻线程在读写输入和输出子数组元素时不会访问相邻的内存位置。
2. 线程在执行 co-rank 函数时还需要从全局内存访问 A 和 B 的元素。

## 12.6 A Tiled Merge Kernel to Improve Coalescing

注意到相邻线程使用的 A 和 B 子数组在内存中彼此相邻。我们可以为为每个块调用 co-rank 函数来获得其 A 和 B 子数组的起始和结束位置。
{{< notice info >}}
回忆一下改进内核内存合并的主要策略有三种:
- 重新组织线程到数据的映射。
- 重新组织数据本身。
- 以合并的方式在全局内存和共享内存之间传输数据，并在共享内存中执行不规则访问。
{{< /notice >}}

下图展示了分段合并内核的块级别设计。A_S 和 B_S 可能无法覆盖块的整个输入子数组，因此在每次迭代期间，块中的所有线程将协作从块的 A 和 B 子数组中加载 x 个元素。这样每个块有足够的输入元素来生成至少 x 个输出数组元素 (**在最坏的情况下，当前输出部分的所有元素可能都来自 A 或 B 的子数组**)。假设每个块负责 y 个输出元素，则需要进行 y/x 次归并。每个块中的线程将在每次迭代中使用 A_S 的一部分和 B_S 的一部分 (深灰色部分)

![Design of a Tiled Merge Kernel](https://note.youdao.com/yws/api/personal/file/WEB176c6f865fd366df31615935a842b2ab?method=download&shareKey=bc8caa12fb14b0e5829a1fd57c3b8a14 "Design of a Tiled Merge Kernel")

下面是分段合并内核的实现的第一部分。本质上是线程级基本合并内核的块级版本的代码。每个块的第一个线程负责计算当前块和下一个块的开始输出索引的位置以及他们的 co-rank. 结果被放入共享内存中，以便块中的所有线程都可以看到。

```cpp {linenos=true}
__global__
void merge_tiled_kernel(int* A, int* B, int* C, int m, int n, int tile_size) {
	/* Part 1: Identify block-level output & input subarrays */
	// Use extern keywords to determine 
	// the shared memory size at runtime rather than compilation
	extern __shared__ int shared_AB[];
	int* A_s = &shared_AB[0];  // Start index of ShareA
	int* B_s = &shared_AB[tile_size];  // Start index of ShareB
	int C_curr = blockIdx.x * ceil((m + n) / gridDim.x);  // Start index of C for this block
	int C_next = std::min(C_curr + int(ceil((m + n) / gridDim.x)), m + n);  // End index of C for this block

	if (threadIdx.x == 0) {
		A_s[0] = co_rank(C_curr, A, m, B, n);  // Make block level co-rank values visible
		A_s[1] = co_rank(C_next, A, m, B, n);  // Next threads co-rank values in the block
	}
	__synctyhreads();
	int A_curr = A_s[0];
	int A_next = A_s[1];
	int B_curr = C_curr - A_curr;
	int B_next = C_next - A_next;
```

第二部分线程使用它们的 `threadIdx.x` 的值来确定要加载的元素，因此连续的线程加载连续的元素，内存访问是合并的。每次迭代从 A 和 B 数组中加载当前tile的起始点取决于块的所有线程在之前的迭代中消耗的 A 和 B 元素的总数。下图说明了 while 循环第二次迭代的索引计算。每个块在第一次迭代中消耗的 A 元素部分 为 A 子数组开头的白色小部分 (用竖条标记)。if 语句确保线程只加载 A 子数组剩余部分中的元素。

```cpp {linenos=true}
/* Part 2: Loading A & B elements into the shared memory */
	int counter = 0;
	int lenC = C_next - C_curr;
	int lenA = A_next - A_curr;
	int lenB = B_next - B_curr;
	int num_iterations = ceil(lenC / tile_size);
	// index of completed merge in 
	int C_completed = 0;
	int A_completed = 0;
	int B_completed = 0;
	while (counter < num_iterations) {  // Each iter threads in a block will generate tile_size C elements
		// Loading tile_size A and B elements into shared memory
		for (int i = 0; i < tile_size; i += blockDim.x) {  // Coalecsing loading from global memory
			if (i + threadIdx.x < lenA - A_completed) {
				A_s[i + threadIdx.x] = A[i + threadIdx.x + A_curr + A_completed];
			}
			if (i + threadIdx.x < lenB - B_completed) {
				B_s[i + threadIdx.x] = B[i + threadIdx.x + B_curr + B_completed];
			}
		}
		__syncthreads();
```

第三部分则是每个块的线程对共享内存的数组进行归并。在更新索引的部分中最后一次迭代中 A_s 和 B_s 可能没有 tile_size 个元素，调用 co-rank 可能会得到错误结果。但是，由于 while 循环不会进一步迭代，因此不会使用结果，因此不会造成任何影响。

```cpp {linenos=true}
		/* Part 3: All threads merge their subarrays in prallel */
		int c_curr = threadIdx.x * (tile_size / blockDim.x);  // Output index in shared memory
		int c_next = c_curr + (tile_size / blockDim.x);
		c_curr = (c_curr <= lenC - C_completed) ? c_curr : lenC - C_completed;
		c_next = (c_next <= lenC - C_completed) ? c_next : lenC - C_completed;
		// find co-rank for c_curr and c_next
		int a_curr = co_rank(c_curr, A_s, std::min(tile_size, lenA - A_completed), 
								B_s, std::min(tile_size, lenB - B_completed));
		int b_curr = c_curr - a_curr;
		int a_next = co_rank(c_next, A_s, std::min(tile_size, lenA - A_completed), 
								B_s, std::min(tile_size, lenB - B_completed));
		int b_next = c_next - a_next;
		// merge the subarrays
		merge_sequential(A_s + a_curr, B_s + b_curr, C + C_curr + C_completed + c_curr, a_next - a_curr, b_next - b_curr);
		// Update completed indices
		C_completed += tile_size;
		A_completed += co_rank(tile_size, A_s, tile_size, B_s, tile_size);  // Idx of A_s to generate tile_size Idx of merged A_s and B_s
		B_completed += tile_size - A_completed;
	}
}
```

## 12.7 A Circular Buffer Merge Kernel

上一节的内核不是那么高效因为下一次迭代 tile 的一部分已经被加载到共享内存中，但是我们每次迭代从全局内存中重新加载整个块，并覆盖掉前一次迭代中的这些元素。下图展示了 merge_circular_buffer_kernel 的主要思想，添加了两个额外的变量 A_S_start 和B_S_start，使得 while 循环的每次迭代动态确定从 A 和 B 的哪个位置开始加载，这样可以利用前一次迭代中剩余的 A_s 和 B_s 元素。修改后每个 for 循环都只加载 A_S_consumed 表示的填充 tile 所需的元素数量。因此，线程在第 i 次 for 循环迭代中加载的A 元素是 `A[A_curr+A_S_consumed+i+threadIdx.x]`. 取模(%) 操作检查索引值是否大于或等于 tile_size.

!A Circular Buffer Scheme for Managing the Shared Memory Tiles[](https://note.youdao.com/yws/api/personal/file/WEB5fab19bd6ec3f60e15cd82672ed06008?method=download&shareKey=8ad20192216748affc4e2f15a1b01b8d "A Circular Buffer Scheme for Managing the Shared Memory Tiles")

## 12.8 Thread Coarsening for Merge

多个线程并行执行归并的代价是每个线程必须执行自己的二进制搜索操作来识别其输出索引的 co-rank. 本章中介绍的所有内核都已经应用了线程粗化，因为它们都是为每个线程处理多个元素而设计的。在完全未粗化的内核中，每个线程将负责单个输出元素。