---
title: PMPP Learning-Chapter 14 Sparse Matrix Computation
date: 2024-09-18T11:43:12+08:00
lastmod: 2024-09-18T11:43:12+08:00
draft: false
author: ["WITHER"]
keywords: 
    - CUDA
categories:
    - CUDA
tags:
    - PMPP learning
description: Personal notebook 14 of Programming Massively Parallel 
summary: Personal notebook 14 of Programming Massively Parallel  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# 14 Sparse Matrix Computation

在稀疏矩阵中，大多数元素是零。存储和处理这些零元素在内存容量、内存带宽、时间和能量方面是浪费的。

## 14.1 Background

矩阵常用于求解 N 个未知数 N 个方程的线性系统，其形式为 AX+Y = 0，其中A是一个 NxN 矩阵，X 是一个 N 维的未知数向量，Y 是一个 N 维的常数向量。求解线性方程组的迭代方法中最耗时的部分是对计算 AX+Y，这是一个稀疏矩阵向量的乘法和累加。
删除所有的零元素不仅节省了存储空间，而且消除了从内存中获取这些零元素并对它们执行无用的乘法或加法操作的冗余步骤。
以下是一些在稀疏矩阵存储格式的结构中的关键考虑因素如下:

- 空间效率 (*Space efficiency*): 使用存储格式表示矩阵所需的内存容量。
- 灵活性 (*Flexibility*): 通过添加或删除非零来修改矩阵的存储格式的方便程度•
- 可访问性 (*Accessibility*): 存储格式是否易于访问数据。
- 内存访问效率 (*Memory access efficiency*): 存储格式在多大程度上为特定计算实现了有效的内存访问模式 (正则化的一个方面).
- 负载平衡 (*Load balancing*): 存储格式在多大程度上为特定计算在不同线程之间平衡负载 (正则化的另一个方面).

## 14.2 A simple SpMV kernel with the COO format

如下图所示， COO (*COOrdinate*) 格式是一种稀疏矩阵的存储格式，其中矩阵元素以三元组的形式存储，即 `(i, j, a_ij)`. 、

![Example of the Coordinate List (COO) Format](https://note.youdao.com/yws/api/personal/file/WEBef2abaed055d77396d3fb9ef77660515?method=download&shareKey=7c73fa2fae1de40aff87876e3c37e6f6 "Example of the Coordinate List (COO) Format")

使用以 COO 格式表示的稀疏矩阵并行执行 SpMV (*Sparse Matrix Vector Multiplication*) 的一种方法是为矩阵中的每个非零元素分配一个线程，下图是其示意图。

![Example of Parallelizing SpMV with the COO Format](https://note.youdao.com/yws/api/personal/file/WEBba4140a13706e68918e8a9fc953e764a?method=download&shareKey=21a3ec827ee9d185f7c0c0e124b9379b "Example of Parallelizing SpMV with the COO Format")

对应的内核代码如下所示，它在列索引对应的位置查找输入向量值，将其乘以非零值，然后将结果累加到对应的行索引处的输出值。

```cpp
struct COOMATRIX {
	int* rowIdx;
	int* colIdx;
	float* val;
	int numNonZeros;
};

__global__ 
void spmv_coo_kernel(COOMATRIX m, float* x, float* y) {  // Assign a thread to each nonzero element
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < m.numNonZeros) {
		int row = m.rowIdx[i];
		int col = m.colIdx[i];
		int val = m.val[i];
		atomicAdd(&y[row], val * x[col]);  // Perform the matrix-vector multiplication
	}
}
```

下面来分析 COO 格式在几个性能指标上的表现。

- 空间效率：COO 需要三个数组，`rowIdx`, `colIdx` 和 `value`，每个数组的元素数量与非零元素的数量相同。
- 灵活性：只要以相同的方式重新排序 `rowIdx`, `colIdx` 和 `value` 数组，就可以在不丢失任何信息的情况下任意地以 COO 格式重新排序元素。
- 可访问性方面：COO 不易访问某一行或某一列中的所有非零元素。
- 内存访问效率：相邻线程访问 COO 格式的每个数组中的相邻元素。因此，通过 SpMV/COO 对矩阵的访问是内存合并的。
- 负载平衡：由于每个线程负责计算一个非零元素，所有线程负责相同数量的工作。
  *SpMV/COO 的主要缺点是需要使用原子操作*，非常耗时。

## 14.3 Grouping Row Nonzeros with the CSR Format

如果将同一行中的所有非零都分配给同一个线程，那么该线程将是唯一更新相应输出值的线程，则可以避免原子操作。这种可访问性可以通过 CSR (Compressed Sparse Row ) 存储格式实现。下图说明了如何使用 CSR 格式存储 14.1 节中的矩阵。CSR 也将非零值存储在一维数组中，但这些非零值是按行分组的。COO 格式和 CSR 格式之间的关键区别在于，CSR 格式用 rowPtrs 数组替换了 rowIdx 数组，rowPtrs 数组存储了 colIdx 和 value 数组中每行非零的起始偏移量，每行中的非零元素不一定按列索引排序。

![Example of Compressed Sparse Row (CSR) Format](https://note.youdao.com/yws/api/personal/file/WEB58c586fae078693686b726fe92eca4d5?method=download&shareKey=011f87e0d08ea4e961006f353bf06fa7 "Example of Compressed Sparse Row (CSR) Format")

如下图所示，要使用以 CSR 格式表示的稀疏矩阵并行执行 SpMV，可以为矩阵的每一行分配一个线程。由于一个线程遍历一行，所以每个线程将输出写入不同的内存位置。

![Example of Parallelizing SpMV with the CSR Format](https://note.youdao.com/yws/api/personal/file/WEB5af8c1b2e91b7e460d54f44a9fa3baaf?method=download&shareKey=434cc8c983d44ceaa2c52d237d6e3c1c "Example of Parallelizing SpMV with the CSR Format")

对应的内核代码如下，每个线程确定它负责的行，循环遍历该行的非零元素来执行点积。线程在 `rowPtrs` 数组中确定它们的起始索引 (`rowPtrs[row]`)和通过下一行非零的起始索引 (`rowPtrs[row+1]`) 来确定结束位置。

```cpp
struct CSRMatrix {
	int* rowPtrs;
	int* colIdx;
	float* val;
	int numRows;
};
__global__
void spmv_csr_kernel(CSRMatrix m, float* x, float* y) {  // Assign a thread to each row
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < m.numRows) {
		int start = m.rowPtrs[row];
		int end = m.rowPtrs[row + 1];
		float sum = 0.0f;
		for (int i = start; i < end; i++) {
			int col = m.colIdx[i];
			float val = m.val[i];
			sum += val * x[col];
		}
		y[row] = sum;  // Perform the matrix-vector multiplication
	}
}
```

下面来分析 CSR 格式在几个性能指标上的表现。

- 空间效率：CSR 需要三个数组，其中 `colIdx` 和 `value` 的维度和非零元素的数量一样。`rowPtrs` 维度等于行数加 1.
- 灵活性：CSR 格式中要添加的非零必须添加到它所属的特定行中。这意味着后面行的非零元素都需要移动，后面行的行指针都需要相应增加。
- 可访问性：CSR 可以很容易地访问给定行的非零元素，允许在 SpMV/CSR 中跨行并行。
- 内存访问效率：CSR 访问模式使得连相邻程访问的数据相距很远，并不能进行内存合并。
- 负载平衡：线程在点积循环中进行的迭代次数取决于分配给线程的行中非零元素的数量，因此大多数甚至所有线程中都存在控制发散。

## 14.4 Improving Memory Coalescing with the ELL Format

ELL 存储格式通过对稀疏矩阵数据进行填充和转置，可以解决非合并内存访问的问题。它的名字来源于 ELLPACK 中的稀疏矩阵包，一个用于求解椭圆边值问题的包。
一个用 ELL 格式存储的例子如下图所示。从按行分组非零的 CSR 格式中确定具有最大非零元素数量的行。然后在所有其他行的非零元素之后的添加填充元素，使它们与最大行长度相同。最后按列主元素顺序存储填充矩阵。

![Example of ELL Storage Format](https://note.youdao.com/yws/api/personal/file/WEB8edc4d051905f48396eed329afa0c448?method=download&shareKey=88bf3affb57e04c78c662c0670721e0c "Example of ELL Storage Format")

下图使用 ELL 格式并行化 SpMV。与 CSR 一样，每个线程被分配到矩阵的不同行。

![Example of Parallelizing SpMV with the ELL Format](https://note.youdao.com/yws/api/personal/file/WEBacb970fc57de19c7e45ab63353ecfdce?method=download&shareKey=8b7e2e9723776c11dbf40de54b6a6075 "Example of Parallelizing SpMV with the ELL Format")

对应的内核代码如下，点积循环遍历每行的非零元素。SpMV/ELL 内核假设输入矩阵有一个向量 `ellMatrix.nnzPerRow` 记录每行中非零的数量，每个线程只迭代其分配的行中的非零元素。

```cpp
struct ELLMATRIX {
	int* nnzPerRow;  // Number of nonzeros per row
	int* colIdx;  // Column indices of nonzeros
	float* val;  // Nonzero values
	int numRows;  // Number of rows
};
__global__
void spmv_ell_kernel(ELLMATRIX m, float* x, float* y) {
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < m.numRows) {
		float sum = 0.0f;
		for (unsigned int i = 0; i < m.nnzPerRow[row]; i++) {
			// ell matrix stores values in column-major order
			unsigned int col = m.colIdx[i * m.numRows + row];
			float val = m.val[i * m.numRows + row];
			sum += val * x[col];
		}
		y[row] = sum;  // Perform the matrix-vector multiplication
	}
}
```

下面来分析 CSR 格式在几个性能指标上的表现。

- 空间效率：由于填充元素的空间开销，ELL格式的空间效率低于CSR格式。
- 灵活性：ELL 格式的比 CSR 格式有更高的灵活性。只要一行没有达到矩阵中非零的最大数目，就可以通过简单地用实际值替换填充元素来向该行添加非零。
- 可访问性：ELL 可以访问某一行的非零元素。ELL 还允许在给定非零元素的索引后得到该元素的行和列索引，因为 `i = col*m.numRows + row`, 通过 `i % m.numRows` 就可以得到所在的行。
- 内存访问效率：由于元素按列主序排列，所有相邻的线程现在都访问相邻的内存位置。
- 负载平衡：SpMV/ELL 仍然和 SpMV/CSR 具有相同的负载不平衡问题，因为每个线程循环次数仍取决它负责的行中的非零元素数量。

## 14.5 Regulating Padding with the Hybrid ELL-COO Format

在 ELL 格式中，当一行或少数行具有非常多的非零元素时，空间效率低和控制发散的问题最为明显。COO 格式可用于限制 ELL 格式中的行长度。在将稀疏矩阵转换为 ELL 之前，我们可以从具有大量非零元素的行中取出一些元素，并将这些元素用单独的 COO 格式存储。
下图展示了如何使用混合 ELL-COO 格式存储图中矩阵。从 ELL 格式中删除第二行的最后 3 个非零元素和第六行的最后 2 个非零元素，并将它们移动到单独的 COO 格式中。

![Hybrid ELL-COO Example](https://note.youdao.com/yws/api/personal/file/WEBa993b596c97e6f2c8d465a7d2eefee9a?method=download&shareKey=558b288f71efe76be4032f0848e44ebd "Hybrid ELL-COO Example")

对应的内核代码如下，点积将被划分为两部分处理，一部分负责处理 ELL 格式的非零元素，另一部分负责处理 COO 格式中 rowIdx 与 row 相同的非零元素。

```cpp
__global__
void spmv_hybrid_ell_coo_kernel(ELLMATRIX ell, COOMATRIX coo, float* x, float* y) {
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	// ELL part
	if (row < ell.numRows) {
		for (int i = 0; i < ell.nnzPerRow[row]; i++) {
			unsigned int col = ell.colIdx[i * ell.numRows + row];
			float val = ell.val[col];
			sum += val * x[col];
		}
	}
	y[row] = sum;  // Perform the matrix-vector multiplication

	// COO part
	for (int i = 0; i < coo.numNonZeros; i++) {
		int col = coo.colIdx[i];
		float val = coo.val[i];
		sum += val * x[col];
		atomicAdd(&y[row], val * x[col]);
	}
}
```

下面来分析混合 ELL-COO 格式在几个性能指标上的表现。

- 空间效率：因为减少了填充元素，混合 ELL-COO 格式比单独使用 ELL 格式的空间效率更高。
- 灵活性：混合 COO-ELL 既可以通过替换填充元素来添加非零。如果该行没有任何可以在 ELL 部分中替换的填充元素，也可以在格式的 COO 部分添加。
- 可访问性：访问给定行中所有的非零元素只能用于适合用 ELL 格式存储的部分行。
- 内存访问效率：SpMV/ELL 和 SpMV/COO 都能对稀疏矩阵进行合并内存访问。因此，它们的组合也将是合并访问模式。
- 负载平衡：从ELL 格式部分移除一些非零元素可以减少 SpMV/ELL 内核的控制发散。这些非零元素被放在 COO 格式部分，不会出现控制发散。

## 14.6 Reducing Control Divergence with the JDS Format

根据矩阵中行的非零元素夺少进行降序排序之后矩阵在很大程度上看起来像三角形矩阵，因此这种格式通常被称为 JDS (*Jagged Diagonal Storage*) 格式。
下图展示了如何使用 JDS 格式存储矩阵。首先，与 CSR 和 ELL 格式一样将非零元素按行分组。接下来，按每行中非零的个数从大到小排序。`value` 数组中的非零值及其存储其对应列索引的 `colIdx` 数组按列主元素顺序存储。在每次迭代中添加一个 `iterPtr` 数组来跟踪非零元素的开始位置。并且维护一个保留原始行索引的 `rowIdx` 数组。

![Example of JDS Storage Format](https://note.youdao.com/yws/api/personal/file/WEBb894f6329a43fbdd14f93fe7572ffaa5?method=download&shareKey=c451117b781a978aa9f937f2bb65f097 "Example of JDS Storage Format")

对应的内核代码如下，我们一共要迭代 `maxNumNonZerosPerRow` 次，每次迭代中每个线程判断自己负责的行是否还存在非零元素。

```cpp
struct JDSMATRIX {
	int* iterPtr;  // Pointer to the start of each row in the JDS format
	int* colIdx;  // Column indices of nonzeros
	float* val;  // Nonzero values
	int* rowIdx; // Original row indices
	int numRows;
	int maxNumNonZerosPerRow;
};
__global__
void spmv_jds_kernel(JDSMATRIX m, float* x, float* y) {
	unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < m.numRows) {
		float sum = 0.0f;
		for (int i = 0; i < m.maxNumNonZerosPerRow + 1; i++) {
			int start = m.iterPtr[i];
			int end = m.iterPtr[i + 1];
			if (row + i * blockDim.x >= end) {
				break;
			} else {
				sum += m.val[row + i * blockDim.x];
			}
		}
		y[m.rowIdx[row]] = sum;  // Perform the matrix-vector multiplication
	}
}
```

下面来分析 JDS 格式在几个性能指标上的表现。

- 空间效率：因为避免了填充 JDS 格式比 ELL 格式效率更高。
- 灵活性：JDS 格式的灵活性较差，因为添加非零会改变行大小，这可能需要重新对行进行排序。
- 可访问性：JDS 格式类似于CSR格式，允许在给定行索引的情况下访问该行的非零元素。
- 内存访问效率：JDS 格式的内存访问效率比 ELL 格式高，因为它可以对稀疏矩阵进行合并访问。
- 负载平衡：JDS 格式对矩阵的行进行排序，使得相邻线程遍长度接近的行。因此，JDS 格式能减少控制发散。
