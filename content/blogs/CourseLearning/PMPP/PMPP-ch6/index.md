---
title: PMPP Learning-Chapter 6 Performance Considerations
date: 2024-09-05T22:22:12+08:00
lastmod: 2024-09-05T22:14:12+08:00
draft: false
author: ["WITHER"]
keywords: 
    - CUDA
categories:
    - CUDA
tags:
    - PMPP learning
description: Personal notebook 6 of Programming Massively Parallel 
summary: Personal notebook 6 of Programming Massively Parallel  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

# 6 Performance Considerations

并行程序的执行速度根据程序的资源需求和硬件的资源约束之间的相互制约会有很大的变化。管理并行代码和硬件资源约束之间的交互对于在几乎所有并行编程模型中实现高性能非常重要。

## 6.1 Memory Coalescing

影响 CUDA 内核性能最重要的因素之一是访问全局内存中的数据，有限的带宽可能成为瓶颈。CUDA 设备的全局内存是用 DRAM 实现的。数据存储在DRAM单元中，访问时间通常是纳秒级别，相对于亚纳秒级别的时钟周期来说很慢。现代 DRAM 通过并行化设计来提高数据访问速率，通常称为内存访问吞吐量 (*memory access throughput*).

{{< details title="Why Are DRAMs So Slow" >}}
DRAM 通过一个个 CMOS 晶体管 (称为 `cell`) 来存储 0/1. 当给晶体管最上面的一端 (称作栅极) 加上电压或是取消电压，晶体管两端就可以流过电流。cell 中的小电容是存储信息的关键，小电容可以存储电荷，当电容存有电荷，cell 存储 1；当电容不存电荷，存储 0.
当要读取 cell 的存储值，首先打开晶体），然后根据导通后的电容是否会进行充放电信息获得存储值。如果 cell 存储 1，即电容存有电荷，那么当打开开关时电容就会放电；反之则不会。
一个 cell 只能存储 1 比特信息，为了存储大量信息，需要构建起如图所示的 cell 阵列。可以看到每行 cell 的晶体管的栅极都是连在一起的，即都连在字线 (*word line*) 上，这意味着给字线施加电压，字线对应的一行cell都会被打开。当一行 cell 被打开，cell 电容就会向位线 (*bit line*) 充放电，一行中的每个 cell 都与一条位线直接相连，读取位线的电压变化，即可知道 cell 的存储信息。
- 字线：用来控制读取哪一个字，一个字由 4字节组成。之所以叫字线，是因为给这根线通电，一行 cell 都会被打开.多个 cell 组合起来就是多个字，因为这根线可以打开多个字，所以叫字线
- 位线：在读取信息时，每一根线上的电压波动都代表一位比特信息，所以叫做位线。
cell 的读取依靠小电容充放电，电容充放电导致位线产生电压波动，通过读取位线电压波动即可获取信息。小电容充放电所产生的电压波动是很微弱的，充放电所造成的电压波动的时间也是很短的，因此很难直接读取充放电信息，为此 cell 阵列的读取使用到了 sense amplifier，即读出放大器。读出放大器可以捕捉到微弱的电压波动，并根据电压波动的情况在本地还原出 cell 的电容电压，而且放大器内还有锁存器，可以把还原出来的电容电压值保存起来，这样一来 cell 保存的信息就从 cell 电容转移到了放大器本地。
每条位线都要接到一个放大器中。在读取 cell 行前，需要把每根位线都预充电 (precharge) 到电容电压/供电电压最大值的一半。在 DRAM 芯片中，读出放大器把 cell 阵列分成了两半，因为其采用的是差分放大器，需要同时接入两根位线。放大信号波动时需要用一个基准和待测线作比较，接到放大器上的两条位线的其中一条就作为基准。在读出数据之后，根据放大器锁存的值，把各条位线拉到供电电压或接到地，然后 cell 电容就会根据位线电压进行充电或放电，当 cell 电容充放电结束，就可以断开字线，宣告本次 DRAM 读取结束。
简单来说读取一个比特的总体流程是：获得行号，译码行号，开启单元行，放大位线电压波动并暂存数据到放大器，获得列号并根据列号选择一位进行输出，写回数据，关闭字线，重新预充电。而写一个比特的总体流程是：获得行号，译码行号，开启单元行，放大位线电压波动并暂存数据到放大器，获得列号并输入写入数据，根据列号把写入数据送到放大器并改写暂存值，写回数据，关闭字线，重新预充电。
其中花费时间最久的两项是开启单元行和放大电压波动并暂存数据。开启单元行时行地址译码器需要拉高一条字线，然后用这一条字线拉高单元行上所有晶体管的栅极电压，相当于给一个很大的电容充电，非常花费时间。放大器大部分是模拟电路，工作速度不快，因此放大电压波动并暂存数据也很花费时间。

![DRAM Cell Array](https://note.youdao.com/yws/api/personal/file/WEB6667e871e68ba1c132af4f6531083e10?method=download&shareKey=3a67fa6fac6a2e83fa7f0e5f0bf2c01c "DRAM Cell Array")
{{< /details >}}
    
由于读取非常耗时，DRAM 每次读取数据都会存储在放大器本地缓存 (*row buffer* / *cache line*). 缓存行内的各个字在内存上是相邻的，每当读取 cell 阵列中的一个比特会把其所在缓存行的所有比特都送到输出缓存，这种读取方式叫做突发 (*burst*). 当 warp 中的所有线程访问连续的全局内存位置时，硬件将所有这些访问合并 (colaesce) 为对连续 DRAM 位置的访问 (即行地址)。
有各种优化策略来实现内存合并。
- 重新排列线程到数据的映射。
- 重新排列数据本身的布局。
- *corner turning*: 以合并的方式在全局内存和共享内存之间传输数据，并在共享内存中执行不利的访问模式。共享内存是用SRAM技术实现的，不需要合并，因此不是连续的地址访问带来的影响不大。
内存合并的主要优点是，能通过将多个内存访问合并为单个访问来减少全局内存流量。

## 6.2 Hiding memory latency

一个 cell 阵列一次可以提供一个比特，那么 8 个 cell 阵列就可以一次提供 8 个比特，他们共享一组行地址和列地址，被称作一个 *bank*. 处理器包含一个或多个通道 (*channel*). 每个通道都是一个带有总线的内存控制器，该总线将一组 DRAM 组连接到处理器。
如下图所示当两个 bank 连接到通道总线时，当第一个 bank 为另一个访问提供服务时，可以在第二个 bank 发起访问。一般来说，如果 cell 阵列访问延迟与数据传输时间之比为 R，则充分利用信道总线的数据传输带宽至少需要 R+1 个 bank 。更多的 bank 减少了针对同一 bank 的多个同时访问的概率，这种现象称为 bank 冲突 (*bank conflict*). 由于每个 bank 一次只能译码一行字线，因此这些冲突访问的单元阵列访问延迟不能再重叠。拥有更多数量的 bank 会增加这些访问分散到多个 bank 的可能性。第二个原因是每个 cell 阵列的大小限制了每个 bank 可以提供的比特数。因此第四章所说的最大化占用率还有一个额外的好处，那就是确保发出足够的内存访问请求来隐藏 DRAM 访问延迟。

![Banking Improves the Utilization of Data Transfer Bandwidth of a Channel](https://note.youdao.com/yws/api/personal/file/WEB1ca208a23c106f7778f72d2d9a329c34?method=download&shareKey=705ee6d9699bf36549f5740c33688ce0 "Banking Improves the Utilization of Data Transfer Bandwidth of a Channel")

分布方案存储如下图所示，通常称为交错数据分布 (*interleaved data distribution*). 对于一个 4*4 的矩阵，每输出矩阵的每个元素计算将对通道 0 中的两个 bank 以及通道 2 中的两个 bank 进行合并访问。

![An Example of Interleaved Data Distribution](https://note.youdao.com/yws/api/personal/file/WEB0ff74f917a82000ff47b38ea6ca53b82?method=download&shareKey=def9f25f25bee24133ae10ac5eee4696 "An Example of Interleaved Data Distribution")

## 6.3 Thread Coarsening

以最细粒度并行化工作的缺点在于，并行化工作需要付出代价，例如不同线程块对数据的重复加载、冗余工作、同步开销等。如果硬件最由于资源不足而顺序执行，那么这个代价是不必要的。部分序列化工作，减少为并行性付出的代价。因此可以通过为每个线程分配多个最细粒度的工作来解决，通常被称为线程粗化 (*thread coarsening*).
如下图所示，在之前的 tiled 矩阵乘法里，由于共享内存内容不能跨块共享，每个块必须加载矩阵 M 的 tile 副本。因此可以让块中的每个线程处理两个输出元素。这样，粗化的线程块将加载 M 的 tile 一次，并将它们用于计算为多个输出 tile.

![Thread Coarsening for Tiled Matrix Multiplication](https://note.youdao.com/yws/api/personal/file/WEBd0127fed6f7a89006a4338bcd85b6c84?method=download&shareKey=701002f69c07f74fa723fbe036467ff9 "Thread Coarsening for Tiled Matrix Multiplication")

下面的代码展示了线程粗化的矩阵乘法内核函数，在 `width/TILE_WIDTH` 的每次迭代中，一个线程计算原来 `COARSE_FACTOR` 个线程对应位置的输出。


使用线程粗化时要注意：
- 不要在不必要的时候使用，当并行化的代价可以通过粗化来降低时，粗化是有益的。
- 不要使用过多的粗化，以免硬件资源得不到充分利用。
- 避免将资源消耗增加到损害占用的程度。根据内核的不同，线程粗化可能需要每个线程使用更多的寄存器或每个线程块使用更多的共享内存。


```cpp {linenos=true}
__global__ 
void CoarsingMatrixMulKernel(float* M, float* N, float* P, int width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	// Identify the row and column of the P element to work on
	int row = by * TILE_WIDTH + ty;
	int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

	// Initialize Pvalue for all output elements
	float Pvalue[COARSE_FACTOR];
	for (int i = 0; i < COARSE_FACTOR; i++) {
		Pvalue[i] = 0;
	}

	// Loop over the M and N tiles required to compute P element
	for (int ph = 0; ph < width/TILE_WIDTH; ph++) {
		// the COARSE_FACTOR tiles of N needs the same tile of M
		Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];

		for (int c = 0; c < COARSE_FACTOR; c++) {
			int col = colStart + c * TILE_WIDTH;  // Value to be computed in the c th tile
			// Collaborative loading of N tile into shared memory
			Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
			__syncthreads();

			for (int k = 0; k < TILE_WIDTH; k++) {
				Pvalue[c] += Mds[ty][k] * Nds[k][tx];
			}
			__syncthreads();
		}

		for (int c = 0; c < COARSE_FACTOR; c++) {
			int col = colStart + c * TILE_WIDTH;
			P[row * width + col] = Pvalue[c];
		}
	}
}
```