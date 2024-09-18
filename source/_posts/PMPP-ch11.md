---
title: PMPP Learning-Chapter 11 Prefix sum (scan)-An Introduction to Work Efficiency in Parallel Algorithms
date: 2024/9/11 23:03:27
categories: CUDA
tags: PMPP learning
excerpt: Personal notebook 11 of Programming Massively Parallel Processors.
mathjax: true
katex: true
---
# 11 Prefix sum (scan)-An Introduction to Work Efficiency in Parallel Algorithms

&emsp;&emsp;一般来说，如果计算本质上可以被描述为数学递归，即序列中的每一项都是根据前一项定义的，那么它可能被并行化为并行扫描 (*parallel scan*) 运算。

## 11.1 Background

&emsp;&emsp;包含扫描 (*inclusive scan*) 操作接收一个二元可交换运算符 {% katex %} \oplus {% endkatex %} 和一个包含 n 个元素的输入数组 {% katex %} [x_0,x_1,\ldots,x_{n-1}] {% endkatex %}，输出数组 {% katex %} [x_0,(x_0\oplus x_1),\ldots,(x_0\oplus x_1\oplus\ldots\oplus x_{n-1})] {% endkatex %}. 包含扫描的名称体现在输出数组每个位置的结果都有对应输入元素参与。考虑包含扫描的一种直观方式是，接收一组所需香肠的长度的订单，并一次性得出所有所有订单对应的切割点。
&emsp;&emsp;排除扫描操作类似于包含扫描操作，只是输出数组的排列略有不同: {% katex %} [i,x_0,(x_0\oplus x_1),\ldots,(x_0\oplus x_1\oplus\ldots\oplus x_{n-2})] {% endkatex %}. 每个输出元素的计算都与相应输入元素无关。
用包含扫描函数计算排除扫描的结果时，只需将所有元素向右移动，并为第 0 个元素填充恒等值。反之，只需要将所有元素向左移动，并用排除扫描结果的最后一个元素 {% katex %} \oplus {% endkatex %} 最后一个输入元素来填充最后一个元素。

## 11.2 Parallel Scan with the Kogge-Stone Algorithm

&emsp;&emsp;计算位置 i 的输出元素 需要进行 i 次加法运算，因此除非找到一种方法来共享不同输出元素的归约树的部分和，否则这种方法计算复杂度为 {% katex %} O(N^2) {% endkatex %}.
Kogge-Stone 算法最初是为了设计快速加法器电路而发明的，如下图所示，它是一种就地扫描算法，它对最初包含输入元素的数组 XY 进行操作。经过 k 次迭代后，`XY[i]` 将包含在该位置及之前的最多 `2^k` 个输入元素的和。

![A Parallel Inclusive Scan Algorithm Based on Kogge-Stone Adder Design](https://note.youdao.com/yws/api/personal/file/WEBabaa1819ea6455c00659abbdd350e12e?method=download&shareKey=804e364d2be685b4e8a02798a814eb2a "A Parallel Inclusive Scan Algorithm Based on Kogge-Stone Adder Design")

&emsp;&emsp;对应的内核函数如下，假设输入最初位于全局内存数组 X 中。让每个线程计算其全局数据索引，即其负责计算输出数组的位置。每个个活动线程首先将其位置的部分和存储到一个临时变量中(在寄存器中)。当步幅值大于 threadIdx.x 时，意味着线程分配的 XY 位置已经累加了所有所需的输入值，退出活动状态。需要额外的 `temp` 和 `__syncthreads()` 因为更新中存在读后写数据依赖竞争关系。

```cpp
#define SECTION_SIZE 32
__global__ 
void Kogge_Stone_Scan_Kernel(int* X, int* Y, unsigned int N) {
	__shared__ float XY[SECTION_SIZE];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	/* Exclusive kernel initilization
	if (i < N && threadIdx.x != 0) {
		XY[threadIdx.x] = X[i];
	} else {
		XY[threadIdx.x] = 0.0f;
	}
	*/
	if (i < N) {
		XY[threadIdx.x] = X[i];
	} else {
		XY[threadIdx.x] = 0.0f;
	}
	for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
		__syncthreads();
		float temp;
		if (threadIdx.x >= stride) {
			temp = XY[threadIdx.x] + XY[threadIdx.x - stride];
		}
		__syncthreads(); // write-after-read dependence
		if (threadIdx.x >= stride) {  // Only N - stride threads are active
			XY[threadIdx.x] = temp;
		}
	}
	if (i < N) {
		Y[i] = XY[threadIdx.x];
	}
}
```

&emsp;&emsp;Kogge-Stone 算法重用了横跨归约树的部分和来降低计算复杂度。在上一章的归约内核中，活动线程在迭代中写入的元素不会在同一迭代中被任何其他活动线程读取，因此不存在读后写竞争条件。如果希望避免在每次迭代中都有 barrier 同步，那么克服竞争条件的另一种方法是为输入和输出使用单独的数组。这种方法需要两个共享内存缓冲区。交替变化不能输入/输出缓冲区的角色，直到迭代完成。这种优化称为双缓冲 (*double buffering*).

```cpp
#define SECTION_SIZE 32
__global__
void DF_Kogge_Stone_Scan_Kernel(int* X, int* Y, unsigned int N) {
	__shared__ float XY_in[SECTION_SIZE];
	__shared__ float XY_out[SECTION_SIZE];

	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

	// Initialization
	if (i < N) {
		XY_in[threadIdx.x] = X[i];
	}
	else {
		XY_in[threadIdx.x] = 0.0f;
	}

	bool read_in = true;  // Alternating ther role of XY_in and XY_out
	for (unsigned stride = 1; stride < blockDim.x; stride *= 2) {
		if (read_in) {
			if (threadIdx.x >= stride) {
				XY_out[threadIdx.x] = XY_in[threadIdx.x] + XY_in[threadIdx.x - stride];
			} else {
				XY_out[threadIdx.x] = XY_in[threadIdx.x];
			}
		}
		else {
			if (threadIdx.x >= stride) {
				XY_in[threadIdx.x] = XY_out[threadIdx.x] + XY_out[threadIdx.x - stride];
			} else {
				XY_in[threadIdx.x] = XY_out[threadIdx.x];
			}
		}
		read_in = !read_in;  // 切换数组
	}

	// 将结果写回全局内存
	if (i < N) {
		if (read_in) {
			Y[i] = XY_in[threadIdx.x];
		} else {
			Y[i] = XY_out[threadIdx.x];
		}
	}
}
```

## 11.3 Speed and Work Efficiency Consideration

&emsp;&emsp;算法的工作效率（*work efficiency*）是指算法所完成的工作接近于计算所需的最小工作量的程度。在每次迭代中，非活动线程的数量等于步长。因此我们可以计算出工作量为
{% katex %} \sum_{stride}(N-\mathrm{stride}), \text{for strides} 1, 2, 4, \ldots N/2(\mathrm{log}_2N \text{terms}) = N\log_2N - (N-1){% endkatex %}
因此，Kogge-Stone 算法的计算复杂度为 {% katex %} O(N\log_2N) {% endkatex %}.

&emsp;&emsp;使用计算步数 (compute steps) 的概念作为比较扫描算法的近似指标。顺序扫描用 N-1 步来处理 N 个输入元素；若 CUDA 设备有 P 个执行单元，Kogge-Stone 内核执行需要步数为 {% katex %} O(N\log_2N)/P {% endkatex %}. Kogge-Stone 内核相比串行代码所做的额外工作有两个问题。首先，使用硬件执行并行内核的效率要低得多。第二，所有额外的工作消耗额外的能量，不利于移动应用等场景。Kogge-Stone 内核的强大之处在于，当有足够的硬件资源时，它可以达到非常好的执行速度。

## 11.4 Parallel Scan with the Brent-Kung Algorithm

&emsp;&emsp;对一组值进行求和最快的方法是使用归约树，如果有足够的执行单元，就可以在 {% katex %} O(N\log_2N) {% endkatex %} 时间内计算 N 个值的求和结果。该树还可以生成几个子序列的和，它们可用于计算某些扫描输出值。
下图展示了基于 Brent-Kung 加法器设计的并行包含扫描算法的步骤。图中上半部分，花 4 步计算所有 16 个元素的和。下半部分是使用反向树将部分和分配到可以使用部分和的位置，以计算这些位置的结果。约简树中的求和总是在对一个连续的范围内的输入元素进行。因此，求和累积到 XY 的每个位置的值总是可以表示为输入元素的一个 xi…xj 的范围，其中 xi 是开始位置， xj 是结束位置 (包括)。

![A Parallel Inclusive Scan Algorithm Based on the Brent–Kung Adder Design](https://note.youdao.com/yws/api/personal/file/WEBda8f1aa90d4fcd75e9b2a976aec9f4c3?method=download&shareKey=3887f9dfcd0ffaacbd1a9ce6a0554d38 "A Parallel Inclusive Scan Algorithm Based on the Brent–Kung Adder Design")

下图展示了反向树中每个位置 (列) 的状态，包括已经累积到该位置的值以及在反向树的每级 (行) 上需要的额外输入元素值 (浅灰色表示 2，深灰色表示 1，黑色表示 0).

![Progression of Values in XY After Each Level of Additions in the Reverse Tree.](https://note.youdao.com/yws/api/personal/file/WEBff0a3e454e8d276a662e5c76b155939b?method=download&shareKey=70741f95168876e0d97cae8bd475d933 "Progression of Values in XY After Each Level of Additions in the Reverse Tree.")

上半部分归约树的内核代码如下，和第十章不同的是

1. 我们把求和结果写到最大索引的位置。
2. 我们将线程索引组织成 {% katex %} 2^n-1 {% endkatex %} 的形式 (n 为树的高度)。

```cpp
for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
	__syncthreads();
	if ((threadIdx.x + 1) % (2 * stride) == 0) {
		XY[threadIdx.x] += XY[threadIdx.x - stride];
	}
}
```

&emsp;&emsp;这种归约方式的一个缺点是存在控制发散问题。因此需要将线程的连续部分映射到索引为 {% katex %} k*2^n-1 {% endkatex %} 形式的 XY 位置。

```cpp
// Mapping a continous section of threads to the XY positions
for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
	__syncthreads();
	unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;  // index of the left child
	if (index < SECTION_SIZE) {
		XY[index] += XY[index - stride];
	}
}
```

&emsp;&emsp;反向树的实现要复杂一些。步长从 `SECTION_SIZE/4` 减小到 1. 在每次迭代中，我们需要将 XY 元素索引值从步长减去 1 后的两倍的位置向右推到距离其一个步长的位置。

```cpp
// Reverse tree stride value decreases from SECTION_SIZE / 4 to 1
for (unsigned int stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
	__syncthreads();
	unsigned int index = (threadIdx.x + 1) * 2 * stride - 1;  // index of the left child
	if (index + stride < SECTION_SIZE) {
		XY[index + stride] += XY[index];
	}
}
```

&emsp;&emsp;我们可以看到 Brent-Kung 算法无论在归约阶段还是分发阶段，都不需要超过 `SECTION_SIZE/2` 的线程。并行扫描中的运算总数，包括归约树 (N-1 次) 和反向树 ({% katex %} N-1-log_2N {% endkatex %} 次) 阶段，总共 {% katex %} 2N-2-log_2N {% endkatex %} 次。当输入长度变大时，Brent-Kung 算法执行的操作数量永远不会超过顺序算法执行的操作数量的 2 倍。

&emsp;&emsp;Brent-Kung 算法的活动线程的数量通过归约树比 Kogge-Stone 算法下降得快得多。然而，一些非活动线程可能仍然会消耗 CUDA 设备中的执行资源，因为它们通过 SIMD 绑定到其他活动线程。这使得在 CUDA 设备上前者工作效率上的优势不那么明显。在有充足执行资源的情况下，由于需要额外的步骤来执行反向树阶段，Brent-Kung 的时间是 Kogge-Stone 的两倍。

## 11.5 Coarsening for Even More Work Efficiency

&emsp;&emsp;如下图所示，粗化扫描分为三个阶段。在第一阶段，我们让每个线程对其相邻的子线程执行串行扫描。需要注意如果每个线程通过访问全局内存的输入直接执行扫描，则它们的访问不会合并。所以我们以合并的方式在共享内存和全局内存之间传输数据，并在共享内存中执行不是那么好的内存访问模式。在第二阶段，每个块中的所有线程协作并对由每个部分的最后一个元素组成的逻辑数组执行扫描操作。在第三阶段，每个线程将其前一个部分的最后一个元素的新值与自身部分除最后一个的所有元素相加。对应的内核代码如下。

![A Three-phase Parallel Scan for Higher Work Efficiency](https://note.youdao.com/yws/api/personal/file/WEB169e018f54e6c49c493de068d8a5f3f6?method=download&shareKey=a54b5ab82631a6c403d2709f702f48c0 "A Three-phase Parallel Scan for Higher Work Efficiency")

```cpp
#define CORASE_FACTOR 4
#define SUBSECTION_SIZE (SECTION_SIZE / CORASE_FACTOR)
__global__
void Corasened_Scan_Kernel(int* X, int* Y, unsigned int N) {  // Partition X into blockDim.x subsections

	// Load X into shared memory in coalesced fashion
	__shared__ float XY[SECTION_SIZE];
	__shared__ float subXY[SUBSECTION_SIZE];
	for (int i = 0; i < SECTION_SIZE; i+= blockDim.x) {
		XY[threadIdx.x + i] = X[threadIdx.x + i];
	}
	__syncthreads();

	// Part 1: Compute prefix sum of each subsection in sequenial 
	for (int i = 1; i < SUBSECTION_SIZE; i++) {
		XY[threadIdx.x * SUBSECTION_SIZE + i] += XY[threadIdx.x * SUBSECTION_SIZE + i - 1];
	}
	__syncthreads();

	// Part 2: Compute prefix sum of the last element of each subsection in parallel
	unsigned int lastElemId = (blockIdx.x + 1) * blockDim.x * CORASE_FACTOR - 1;
	subXY[threadIdx.x] = XY[(threadIdx.x + 1) * SUBSECTION_SIZE - 1];
	float temp = 0.0f;
	for (int stride = 1; stride < SUBSECTION_SIZE; stride *= 2) {
		__syncthreads();
		if (threadIdx.x >= stride) {
			temp = subXY[threadIdx.x] + subXY[threadIdx.x - stride];
		}
		__syncthreads();
		if (threadIdx.x >= stride) {
			subXY[threadIdx.x] = temp;
		}
	}
	__syncthreads();

	// Part 3: Add the reduction sum of the previous subsection to the current subsection (except the last element)
	for (int i = 1; i < SUBSECTION_SIZE - 1; i++) {
		XY[threadIdx.x * SUBSECTION_SIZE + i] += subXY[threadIdx.x];
	}
	__syncthreads();

	// Store back to Y
	for (int i = 0; i < SECTION_SIZE; i+= blockDim.x) {
		Y[threadIdx.x + i] = XY[threadIdx.x + i];
	}
}
```

## 11.6 Segmented Parallel Scan for Arbitrary-length Inputs

&emsp;&emsp;对于长度很大的输入数据，我们首先将其划分为多个部分，以便每个部分都可以放入流多处理器的共享内存中，并由单个块进行处理。如下图所示，第一步在每个块内部先进行扫描，完成后每个扫描块的最后一个输出元素为该扫描块的所有输入元素的和。第二步将每个扫描块的最后一个结果元素收集到一个数组中，并对这些输出元素执行扫描。第三步将第二步扫描输出值与其对应扫描块的值相加。

![A Hierarchical Scan for Arbitrary Length Inputs](https://note.youdao.com/yws/api/personal/file/WEB22da8ba64a41795b1047cae1caf6f6d3?method=download&shareKey=99c2a7dcbbdeaf1be80feb644be643c0 "A Hierarchical Scan for Arbitrary Length Inputs")

&emsp;&emsp;我们可以用三个内核实现分段扫描。第一个内核与 11.5 节的内核基本相同，第二个内核只是单个线程块的并行扫描内核，第三个内核将 S 数组和 Y 数组作为输入，并将其输出写回 Y.

## 11.7 Single-pass Scan for Memory Access Efficiency
