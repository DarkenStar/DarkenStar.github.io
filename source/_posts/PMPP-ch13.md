---
title: PMPP Learning-Chapter 13 Sorting
date: 2024/9/14 13:36:46
categories: CUDA
tags: PMPP learning
excerpt: Personal notebook 13 of Programming Massively Parallel Processors.
mathjax: true
katex: true
--- 
# 13 Sorting

&emsp;&emsp;排序算法将列表中的数据元素按一定的顺序排列。

## 13.1 Background

任何排序算法都必须满足以下两个条件:
- 输出是非递减顺序或非递增顺序。
- 输出是输入的一种排列 (permutation).

&emsp;&emsp;排序算法可以分为稳定算法和不稳定算法。当两个元素具有相同的键值时，稳定的排序算法保留了原始的出现顺序。
&emsp;&emsp;排序算法也可以分为基于比较的算法和非基于比较的算法。基于比较的排序算法无法达到比 O(NlogN) 更好的复杂度，因为它们必须在元素之间执行最少次数的比较。

## 13.2 Radix Sort

&emsp;&emsp;基数排序是一种基于非比较的排序算法，其工作原理是根据基数值将要排序的键分布到桶 (bucket) 中。如果键由多个数字组成，则重复对每个数字重复分配桶，直到覆盖所有数字。
下图展示了如何使用 1 位基数对 4 位整数列表进行基数排序。

![A Radix Sort Example](https://note.youdao.com/yws/api/personal/file/WEB25b04d30be5a63b12bfdbb3093994f44?method=download&shareKey=92a8bec7bdc022c628a42bd27a54086b "A Radix Sort Example")

## 13.3 Parallel Radix Sort

&emsp;&emsp;基数排序的每次迭代都依赖于前一次迭代的整个结果。因此，迭代是相对于彼此顺序执行的。我们将重点关注执行单个基数排序迭代的内核的实现，并假设主机代码每次迭代调用该内核一次。
&emsp;&emsp;在 GPU 上并行化基数排序迭代的一种直接方法是让每个线程负责输入列表中的一个键。线程必须确定键在输出列表中的位置，然后将键存储到该位置。
&emsp;&emsp;下图展示了这种并行化方法第一次迭代的执行情况。对于映射到 0 桶的键，目标索引可以通过如下公式计算：
{% katex %} \text{destination of a zero} =  \text{\#zeros before}
=\text{\#keys before} - \text{\#ones before} 
=\text{key index}-\text{\#ones before}{% endkatex %}
对于映射到 1 桶的键，目标索引如下所示:
{% katex %}\text{destination of a one}=\text{\#zeros in total}+\text{\#ones before} \\
=(\text{\#keys in total}-\text{\#ones in total})+\text{\#ones before} \\
=\text{input size}-\text{\#ones in total}+\text{\#ones before}{% endkatex %}

![Parallelizing a Radix Sort Iteration by Assigning One Input Key to Each Thread](https://note.youdao.com/yws/api/personal/file/WEBa8ef27e147df1b0c4de80ee951d9f79d?method=download&shareKey=c3797389c4f43d2d3ec67d06eef347f3 "Parallelizing a Radix Sort Iteration by Assigning One Input Key to Each Thread")

&emsp;&emsp;下图展示了每个线程查找其键的目标索引所执行的操作。

![Finding the Destination of Each Input Key](https://note.youdao.com/yws/api/personal/file/WEBb99f77f6611178972cecd41b56175600?method=download&shareKey=3c54c366645f7a1dadda9f884843f70a "Finding the Destination of Each Input Key")

&emsp;&emsp;对应的内核代码如下所示。在每个线程确定自己的索引并提取出对应的 bit 后，因为这些位不是 0 就是 1，所以排除扫描的结果就等于索引前面 1 的个数。

```cpp
__global__ 
void exclusiveScan(unsigned int* bits, int N) {
    extern __shared__ unsigned int temp[];

    int thid = threadIdx.x;
    int offset = 1;

    // Load input into shared memory
    temp[2 * thid] = (2 * thid < N) ? bits[2 * thid] : 0;
    temp[2 * thid + 1] = (2 * thid + 1 < N) ? bits[2 * thid + 1] : 0;

    // Build sum in place up the tree
    for (int d = N >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    // Clear the last element
    if (thid == 0) {
        temp[N - 1] = 0;
    }

    // Traverse down the tree
    for (int d = 1; d < N; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = offset * (2 * thid + 1) - 1;  // left child index of the thread
            int bi = offset * (2 * thid + 2) - 1;  // right
            unsigned int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }

    // Write results to output array
    __syncthreads();
    if (2 * thid < N) bits[2 * thid] = temp[2 * thid];
    if (2 * thid + 1 < N) bits[2 * thid + 1] = temp[2 * thid + 1];
}

__global__
void radix_sort_iter(unsigned int* input, unsigned int* output, unsigned int* bits, int N, unsigned int iter) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int key, bit;
	if (i < N) {
		key = input[i];
		bit = (key >> iter) & 1;
		bits[i] = bit;
	}
	exclusiveScan(bits, N);  // # ones before
	if (i < N) {
		unsigned int numberOnesBefore = bits[i];
		unsigned int numberOnesTotal = bits[N];
		unsigned int dst = (bit == 0) ? (i - numberOnesBefore) : (N - numberOnesTotal - numberOnesBefore);
		output[dst] = key;
	}
}
```

## 13.4 Optimizing for Memory Coalescing

&emsp;&emsp;上面方法效率低下的一个主要原因是，对输出数组的写入显示出不能以内存合并的模式访问。改进后的算法如下图所示，每个块中的线程将首先执行块级别的局部排序，以分离共享内存中映射到 0 bucket 的键和映射到 1 bucket 的键。此优化中的主要挑战是每个线程块在全局 bucket 中确定其位置。线程块的 0 桶的位置在前面线程块的所有 0 桶之后。另一方面，线程块的 1 桶的位置在所有线程块的 0 桶和之前线程块的所有 1 桶之后。

![Optimizing for Memory Coalescing by Sorting Locally in Shared Memory](https://note.youdao.com/yws/api/personal/file/WEBf574d6d0c92edf4dc69026e429cc9f87?method=download&shareKey=37e73d2ec0d91af6e330c10db179d93a "Optimizing for Memory Coalescing by Sorting Locally in Shared Memory")

&emsp;&emsp;下图展示了如何使用排除扫描来查找每个线程块的本地桶的位置的。在完成局部基数排序之后，每个线程块标识其每个自己桶中键的数量。然后每个块将结果记录在如图中所示的表中，该表按行主顺序存储，对线性化的表执行排除扫描，结果表示线程块的本地 bucket 的起始位置。

![Finding the Destination of Each Thread Block's Local Buckets](https://note.youdao.com/yws/api/personal/file/WEBea305ebe970559e5fae76374c79ee2f7?method=download&shareKey=1a6535297a6a2fbccbeed23f7a25eeb6 "Finding the Destination of Each Thread Block's Local Buckets")

```cpp
#define SECTION_SIZE 32
__global__
void memory_coalescing_radix_sort(unsigned int* input, unsigned int* output, unsigned int* bits, unsigned int* table, int N, int iter) {
    __shared__ unsigned int input_s[SECTION_SIZE];
    __shared__ unsigned int output_s[SECTION_SIZE];
    // Load input into shared memory
    unsigned int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (globalIdx < N) {
        input_s[threadIdx.x] = input[globalIdx];
    }
    __syncthreads();

    // Sort each section
    radix_sort_iter(input_s, output_s, bits + blockIdx.x * SECTION_SIZE, SECTION_SIZE, iter);
    __syncthreads();

    // Store local bucket num
    if (threadIdx.x == 0) {
        unsigned int numberOnesTotal = 0;
        unsigned int numberZerosTotal = 0;
        for (int i = 0; i < SECTION_SIZE; ++i) {
            numberOnesTotal += bits[blockIdx.x * SECTION_SIZE + i];
        }
        numberZerosTotal = SECTION_SIZE - numberOnesTotal;
        table[blockIdx.x] = numberZerosTotal;
        table[blockIdx.x + gridDim.x] = numberOnesTotal;
    }
    __syncthreads();

    // Exclusive prefix sum to determine output index
    exclusiveScan(table, 2 * gridDim.x);

    // Write results to output array
    if (globalIdx < N) {
        int zeroOffset = table[blockIdx.x];
        int oneOffset = table[blockIdx.x + gridDim.x];
        unsigned int bit = bits[blockIdx.x * SECTION_SIZE + threadIdx.x];
        unsigned int dst = (bit == 0) ? (globalIdx - zeroOffset) : (N - oneOffset);
        output[dst] = input[globalIdx];
    }
}
```

## 13.5 Choice of Radix Value

&emsp;&emsp;使用 2 bit 的基数时，如下图所示，每次迭代使用两个比特将键分发到存储桶。因此，两次迭代就可以对 4 bit 键进行完全排序。

![Radix Sort Example with 2-bit Radix](https://note.youdao.com/yws/api/personal/file/WEBaa1313e289f139eb041b02b86260874b?method=download&shareKey=6a9fbd43b0fe2c1926459a7a6ba87d92 "Radix Sort Example with 2-bit Radix")

&emsp;&emsp;为了内存合并访问，如下图所示，每个线程块可以在共享内存中对其键进行本地排序，然后将每个本地桶中的键的数量写入表中。和 13.4 节一样，对于 r 位基数，对具有 2^r 行的表执行排除扫描操作。最后以合并的方式将本地 bucket 写入全局内存。

![Optimizing 2-bit Radix Sorting for Memory Coalescing Using the Shared Memory](https://note.youdao.com/yws/api/personal/file/WEB7943d691aa80d8cf1d4cc9d9055dad8c?method=download&shareKey=b1c724f7996b46a5b15f81c91d6a88d3 "Optimizing 2-bit Radix Sorting for Memory Coalescing Using the Shared Memory")

使用更大的基数也有缺点
1. 每个线程块有更多的本地桶，每个桶有更少的键。这样就会向多个全局内存块进行写入，但每一部分写入的数据变少，不利于内存合并。
2. 进行排除扫描的表会随着基数的增大而变大，扫描的开销随着基数的增加而增加。

![Finding the Destination of Each Block's Local Buckets for a 2-bit Radix](https://note.youdao.com/yws/api/personal/file/WEB3df7c81b854884404a6b583c1fbb99fa?method=download&shareKey=82b507cb68c828d45eb81bf4b8be6e5a "Finding the Destination of Each Block's Local Buckets for a 2-bit Radix")

## 13.6 Thread Coarsening to Improve Coalescing

&emsp;&emsp;跨多个线程块并行化基数排序的一个代价是对全局内存的写的访问合并很差。每个线程块都有自己的本地桶，并将其写入全局内存。拥有更多的线程块意味着每个线程块拥有更少的键，这意味着本地存储桶将更小，从而在将它们写入全局内存时合并机会更少。另一个代价是执行全局排除扫描以识别每个线程块的本地桶的存储位置的开销。通过应用线程粗化，可以减少块的数量，从而减少表的大小和排除扫描操作的开销。
&emsp;&emsp;下图展示了如何将线程粗化应用于 2 位基数排序。每个线程被分配给输入列表中的多个键。

![Radix Sort for a 2-bit Radix with Thread Coarsening](https://note.youdao.com/yws/api/personal/file/WEBc83cef230dd49ea34fafa2da625e2181?method=download&shareKey=551310570976f664cdbd714653b982b2 "Radix Sort for a 2-bit Radix with Thread Coarsening")

## 13.7 Parallel Merge Sort

