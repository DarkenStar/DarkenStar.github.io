---
title: PMPP Learning-Chapter 2 Heterogeneous Data Parallel Computing
date: 2024/9/03 22:47:03
categories: CUDA
tags: PMPP learning
excerpt: Personal notebook 2 of Programming Massively Parallel Processors.
mathjax: true
katex: true
---
# 2 Heterogeneous Data Parallel Computing

&emsp;&emsp;数据并行 (*Data Parallel*) 是指在数据集的不同部分上执行的计算工作可以彼此独立地完成，从而可以并行执行的现象。

## 2.1 Data Parallel

&emsp;&emsp;在图像处理中，将彩色像素转换为灰度只需要该像素的数据。模糊图像将每个像素的颜色与附近像素的颜色平均，只需要像素的小邻域的数据。即使是一个看似全局的操作，比如找到图像中所有像素的平均亮度，也可以分解成许多可以独立执行的较小的计算。这种对不同数据块的独立计算是数据并行性的基础。
&emsp;&emsp;为了将彩色图像转换为灰度图像，我们通过以下加权和公式计算每个像素的亮度值L. 这些逐像素计算都不依赖于彼此，都可以独立执行。显然，彩色图到灰度图的转换具有大量的数据并行性。
{% katex %}  L=0.21r+0.72g+0.07b {% endkatex %}

{% fold info  @Task Parallelism vs. Data Parallelism %}
&emsp;&emsp;数据并行并不是并行编程中使用的唯一类型的并行。任务并行 (*Task Parallelism*) 在并行编程中也得到了广泛的应用。任务并行性通常通过应用程序的任务分解来暴露。例如，一个简单的应用程序可能需要做一个向量加法和一个矩阵-向量乘法。每个都是一个任务。如果两个任务可以独立完成，则存在任务并行性。I/O和数据传输也是常见的任务。
{% endfold %}

![Data Parallelsim in Image2Grayscale Conversion](https://note.youdao.com/yws/api/personal/file/WEB56d71b169d207ac51adc718f79fb006c?method=download&shareKey=d97c1f60eb44182ac8bb99f8a81035fe "Data Parallelsim in Image2Grayscale Conversion")

## 2.2 CUDA C Program Structure

&emsp;&emsp;CUDA C 用最少的新语法和库函数扩展了流行的 ANSI C 语言。CUDA C 程序的结构反映了计算机中主机 (CPU) 和一个或多个设备 (GPU) 的共存。每个 CUDA C 源文件可以同时包含主机 (*host*) 代码和设备 (*device*) 代码。
&emsp;&emsp;CUDA程序的执行流程如下图所示。执行从主机代码 (CPU 串行代码) 开始，当调用内核函数 (*kernel function*) 时，会在设备上启动大量线程[^1]来执行内核。由内核调用启动的所有线程统称为网格 (grid)。这些线程是 CUDA 并行执行的主要载体。

![Execution of a CUDA Program](https://note.youdao.com/yws/api/personal/file/WEBcfe42671ed5897d29371195eb557fa00?method=download&shareKey=2d501a2775f21e11292189f68d89e39a "Execution of a CUDA Program")

## 2.3 A vector addition kernel

&emsp;&emsp;使用向量加法来展示 CUDA C 程序结构。下面展示了一个简单的传统 C 程序，它由一个主函数和一个向量加法函数组成。

{% note info %}
当需要区分主机和设备数据时，我们都会在主机使用的变量名后面加上 “`_h`”，而在设备使用的变量名后面加上 “`_d`”.
{% endnote %}

```cpp
// Compute vector sum h_C = h_A+h_B
void vecAdd(float* h_A, float* h_B, float* h_C, int n)
{
    for (int i = 0; i < n; i++) h_C[i] = h_A[i] + h_B[i];
}

int main()
{
    // Memory allocation for h_A, h_B, and h_C
    // I/O to read h_A and h_B, N elements each
    // …
    vecAdd(h_A, h_B, h_C, N);
}
```

&emsp;&emsp;并行执行向量加法的一种直接方法是修改 `vecAdd` 函数并将其计算移到设备上。修改后的结构如下所示。

![Structure of the Modified VecAdd](https://note.youdao.com/yws/api/personal/file/WEB1aadc269025d2f33b2bd42b7838c7cc3?method=download&shareKey=04dc8aeb81948f9dd66f6dccb54e8bd5 "Structure of the Modified VecAdd")

```cpp
#include <cuda_runtime.h>

// …

void vecAdd(float* A, float* B, float* C, int n)
{
    int size = n* sizeof(float);
    float *d_A *d_B, *d_C;
    /*
    …
    1. // Allocate device memory for A, B, and C
       // copy A and B to device memory
    2. // Kernel launch code – to have the device
       // to perform the actual vector addition
    3. // copy C from the device memory
       // Free device vectors
    */
}
```

## 2.4 Device Global Memory and Data Transfer

&emsp;&emsp;在当前的CUDA系统中，设备通常是带有自己的 DRAM 的硬件卡，称为 (设备)全局内存 (*device global memory*). 对于向量加法内核，在调用内核之前，程序员需要在设备全局内存中分配空间，并将数据从主机内存传输到设备全局内存中分配的空间。这对应于 1. 部分。类似地，在设备执行之后，程序员需要将结果数据从设备全局内存传输回主机内存，并释放设备全局内存中不再需要的已分配空间。这对应于 3. 部分。
&emsp;&emsp;`cudaMalloc` 函数可以从主机代码中调用，为对象分配一块设备全局内存。第一个参数是指针变量的地址，该变量将被设置为指向分配的对象。指针变量的地址应强制转换为 `void**`，这样可以允许 `cudaMalloc` 函数将分配内存的地址写入所提供的指针变量中，而不考虑其类型[^2]。

```cpp
cudaError_t cudaMalloc(void** devPtr, size_t size);
```

- `devPtr`：指向指向设备内存的指针的指针。
- `size`：要分配的内存大小（以字节为单位）。

---

&emsp;&emsp;cudaFree 函数通过释放设备内存并将其返回到可用内存池来管理设备内存资源。它只需要 A_d 的值来识别要释放的内存区域，而不需要改变 A_d 指针本身的地址。
{% note warning%}
&emsp;&emsp;在主机代码中对设备全局内存指针进行解引用引用可能导致异常或其他类型的运行错误。
{% endnote %}

&emsp;&emsp;cudaMemcpy 函数是 CUDA 中用于在主机内存和设备内存之间传输数据的核心函数。它允许将数据从主机内存复制到设备内存，或从设备内存复制到主机内存。

```cpp
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind);
```

* `dst`：目标内存地址，可以是主机内存地址或设备内存地址。
* `src`： 源内存地址，可以是主机内存地址或设备内存地址。
* `count`： 要复制的数据大小（以字节为单位）。
* `kind`： 复制方向，可以使用[以下枚举值](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g18fa99055ee694244a270e4d5101e95b)：
  * `cudaMemcpyHostToDevice`：主机内存->设备内存。
  * `cudaMemcpyDeviceToHost`：设备内存->主机内存。
  * `cudaMemcpyDeviceToDevice`：设备内存->设备内存。
  * `cudaMemcpyHostToHost`：主机内存->主机内存

&emsp;&emsp;了解完这些后，可以更新代码的框架如下
{% fold info @Error Checking and Handling in CUDA %}
CUDA API 函数返回一个 `cudaError_t` 类型的标志，指示当它们处理请求时是否发生错误。
在 CUDA 运行时库的头文件 cuda_runtime.h 中，cudaError_t 被定义为一个 int 类型的别名

```cpp
typedef int cudaError_t;
```

&emsp;&emsp;一个例子如下

```cpp
// ...
float *d_a;
cudaError_t err = cudaMalloc(&d_a, 1024 * sizeof(float));

if (err != cudaSuccess) {
    printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
    return 1;
}
```

{% endfold %}

```cpp
void vecAdd(float* A, float* B, float* C, int n)
{
    int size = n* sizeof(float);
    float *d_A *d_B, *d_C;
  
    cudaMalloc((void **) %d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc((void **) %d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaMalloc((void **) %d_C, size);

    // Kernel invocation code - to be shown later
    // ...

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory for A, B, C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

## 2.5 Kernel functions and threading

&emsp;&emsp;内核函数指所有线程在并行阶段执行的代码，**网格中的所有线程执行相同的内核代码**。。当程序的主机代码调用内核时，CUDA runtime 系统启动一个线程网格，这些线程被组织成一个两级层次结构。每个网格都被组织为线程块 (*thread block*, 简称为块) 数组。网格的所有块都是相同的大小。在调用内核时，每个线程块中的线程总数由主机代码指定。
同一个内核可以在主机代码的不同部分用不同数量的线程调用。对于给定的线程网格，一个块中的线程数可以在名为 `blockDim` 的内置变量中获得，它是一个具有三个无符号整数字段 `(x, y, z)` 的结构体。
&emsp;&emsp;下图给出了一个示例，其中每个块由256个线程组成。每个线程都用一个箭头表示，标有线程在块中的索引号的方框。由于数据是一维向量，因此每个线程块被组织为一维线程数组。`blockDim.x` 的值表示每个块中的线程总数。`threadaIdx` 变量表示每个线程在块中的坐标。全局索引 i 的计算公式为 `i = blockIdx.x * blockDim.x + threadIdx.x`

{% note info %}
&emsp;&emsp;许多编程语言都有内置变量。这些变量具有特殊的含义和目的。这些变量的值通常由运行时系统预先初始化，并且在程序中通常是只读的。
{% endnote %}

![Hierarchical Organization in CUDA](https://note.youdao.com/yws/api/personal/file/WEB67f8186e554926a97be5d005a8c86056?method=download&shareKey=55b9fec27854c9abf7e06eaff5c5a612 "Hierarchical Organization in CUDA")

&emsp;&emsp;向量加法的核函数定义如下。网格中的每个线程对应于原始循环的一次迭代，这被称为循环并行 (*loop parallel*)，意为原始顺序代码的迭代由线程并行执行。`addVecKernel` 中有一个 `if (i < n)` 语句，因为并非所有的向量长度都可以表示为块大小的倍数。

```cpp
__global__
void vecAddKernel(float* A, float* B, float* C, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < n) C[i] = A[i] + B[i];
}
```

CUDA C 使用了三个可以在函数声明中使用的限定字。下表展示了这些关键词的意义。
- `__host__ ` 就是在主机上执行的传统 C 函数，只能从另一个主机函数调用。
- `__global__` 表示被声明的函数是 CUDA C 内核函数。内核函数在设备上执行，并且可以从主机上调用。
- `__device__` 函数在 CUDA 设备上执行，只能从内核函数或其他设备函数调用。

{% note info %}
&emsp;&emsp;可以在函数声明中同时使用 `__host__`  和 `__device__`. 编译系统会为同一个函数生成两个版本的目标代码。
{% endnote %}

| Qualifier<br />Keyword       | Callable<br />From | Executed<br />on | Executed<br />by          |
| ---------------------------- | ------------------ | ---------------- | ------------------------- |
| `__host__ `<br />(default) | Host               | Host             | Caller host thread        |
| `__global__`               | Host/Device        | Device           | New grid of device thread |
| `__device__`               | Device             | Device           | Caller device thread      |

## 2.6 Calling kernel functions

&emsp;&emsp;实现内核函数之后，剩下的步骤是从主机代码调用该函数来启动网格。当主机代码调用内核时，它通过执行配置参数 (*execution configuration parameters*) 设置网格和线程块大小配置参数在在传统的C函数参数之前由 `<<<...>>>` 之间给出。第一个配置参数给出网格中的块数量。第二个参数指定每个块中的线程数。

```cpp
int vectAdd(float* A, float* B, float* C, int n)
{
    // d_A, d_B, d_C allocations and copies omitted
    // ...
    // Run ceil(n/256) (or by (n + 256 - 1) / 256) blocks of 256 threads each 
    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);
}
```

&emsp;&emsp;下面展示了 `vecAdd` 函数中的最终主机代码。所有的线程块操作向量的不同部分。它们可以按任意顺序执行。

{% note info %}
&emsp;&emsp;实际上，分配设备内存、从主机到设备的输入数据传输、从设备到主机的输出数据传输以及释放设备内存的开销可能会使生成的代码比原始顺序代码慢，这是因为内核完成的计算量相对于处理或传输的数据量来说很小。
{% endnote %}

```cpp
void vecAdd(float* A, float* B, float* C, int n)
{
    int size = n * sizeof(float);
    float *d_A *d_B, *d_C;
    
    cudaMalloc(&d_A, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, size);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaMalloc(&d_C, size);

    vecAddKernel<<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory for A, B, C
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

## 2.7 Compilation

NVCC (NVIDIA C Compiler) 处理一个C处理一个CUDA C程序，使用 CUDA 关键字来分离主机代码和设备代码。
- 主机代码是就是普通的ANSI C代码，使用 C/C++ 编译器进行编译，并作为传统的 CPU 进程运行。
- 设备代码及其相关辅助函数和数据结构的CUDA关键字，由NVCC编译成称为 PTX (Parallel Thread Execution) 文件的虚拟二进制文件, 由 NVCC runtime 组件进一步编译成目标文件，并在支持 cuda 的 GPU 设备上执行。

![Overview of the Compilation Process of a CUDA C Program](https://note.youdao.com/yws/api/personal/file/WEB0a32f3aa7a8ffb0fbf51b81c298fcc26?method=download&shareKey=9eb69ac9f65a39b57002dcb02da3a39c "Overview of the Compilation Process of a CUDA C Program")

---
[^1]: 线程由程序的代码、正在执行的代码中的位置以及它的变量和数据结构的值组成。
    
[^2]: `cudaMalloc` 与 C 语言 `malloc` 函数的格式不同。前者接受两个参数，指针变量其地址作为第一个参数给出。后者只接受一个参数来指定分配对象的大小，返回一个指向分配对象的指针。
