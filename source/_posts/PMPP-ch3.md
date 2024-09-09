---
title: PMPP Learning-Chapter 3 Multidimensional Grids and Data
date: 2024/9/04 21:57:11
categories: CUDA
tags: PMPP learning
excerpt: Personal notebook 3 of Programming Massively Parallel Processors.
mathjax: true
katex: true
---
# 3 Multidimensional Grids and Data

&emsp;&emsp;本章将更广泛地介绍线程是如何组织的和如何使用线程和块来处理多维数组。

## 3.1 Multidimensional Grid Organization

&emsp;&emsp;再次强调**网格中的所有线程执行相同的内核函数**，它们依赖于线程索引来区分彼此，并确定各自要处理的数据的部分。这些线程被组织成两级结构: 一个网格由一个或多个块组成，每个块由一个或多个线程组成。调用内核函数时需要指定执行配置参数 `gridDim` 和 `blockDim`，`gridDim` 是一个三维块数组，`blockDim` 是一个三维线程数组。他们的类型都是 `dim3`，是包含三个元素 x, y 和 z 的整数向量类型，分别指定了每个维度上的块个数和线程个数。使用少于 3 个维度时可以将未使用的维度大小设置为 1。网格中的所有块都具有相同的维度和大小。**一旦网格启动，网格和块的尺寸将保持不变，直到整个网格完成执行。**

{% note primary %}
当前CUDA系统中，每个块的总大小限制为1024个线程。只要线程总数不超过1024，这些线程就可以以任何方式分布在三个维度上。
{% endnote %}

```cpp
function_name<<<gridDim, blockDim>>>(...);
```

&emsp;&emsp;一个例子如下，dimBlock和dimGrid是由程序员定义的主机代码变量。

```cpp
dim3 dimGrid(32, 1, 1);
dim3 dimBlock(128, 1, 1);
vecAddKernel<<<dimGrid, dimBlock>>>(...);
```

&emsp;&emsp;下图展示了 `gridDim(2,2,1)` 和 `blockDim (4,2,2)` 情况下线程组织的情况。

![A Multidimensional Example of CUDA Grid Organization](https://note.youdao.com/yws/api/personal/file/WEBcbb93fc419f4d6121d02e091d5666989?method=download&shareKey=e6560d2a922f6a2c322706cb282ac70f "A Multidimensional Example of CUDA Grid Organization")

## 3.2 Mapping threads to multidimensional data

&emsp;&emsp;选择 1D、2D 或 3D 的线程组织通常基于数据的性质。例如图像是一个二维像素数组。使用由 2D 块组成的 2D 网格可以方便地处理图像中的像素。下图展示了处理大小为 `62*76` 1F1F 的图片 P 的一种组织方式。假设使用 `16*16` 大小的块，那么在 y 方向上需要 4 个块，在 x 方向上需要 5 个块。横纵坐标的计算方式为

```plaintext
row coordinate = blockIdx.y * blockDim.y + threadIdx.y
col coordinate = blockIdx.x * blockDim.x + threadIdx.x
```

 {% note primary %}
 &emsp;&emsp;我们将按维度的降序 `(z, y, x)` 表示多维数据。这种顺序与 `gridDim` 和 ` blockDim` 维度中数据维度的顺序相反！！！
 {% endnote %}

&emsp;&emsp;实际上，由于现代计算机中使用二维存储空间，C 语言中的所有多维数组都是线性化的。虽然可以使用如 `Pin_d[j][i]` 这样的多维数组语法访问多维数组的元素，但编译器将这些访问转换为指向数组开始元素的基指针，以及从这些多维索引计算出的一维偏移量。
&emsp;&emsp;至少有两种方法可以对二维数组进行线性化。将同一行/列的所有元素放置到连续的位置。然后将行/列一个接一个地放入内存空间中。这种排列称为行/列主序布局 (*row/column-major layout*). **CUDA C 使用行主序布局。**

![Row-major Layout for a 2D C Array](https://note.youdao.com/yws/api/personal/file/WEB7aab5f499364badca84a8cf76a1793fb?method=download&shareKey=7df12b5fb5eae00c962e1a3ff98dabec "Row-major Layout for a 2D C Array")

&emsp;&emsp;下面内核代码将每个颜色像素转换为对应的灰度像素。我们计算坐标为 `(row, col)` 的像素对应的 1D 索引 `row * width + col`. 这个 1D 索引 `grayOffset` 就是 `Pout` 的像素索引，因为输出灰度图像中的每个像素都是 1字节 (unsigned char)。每个彩色像素用三个元素(r, g, b)存储，每个元素为1字节。因此 `rgbOffset` 给出了 `Pin` 数组中颜色像素的起始位置。从 `Pin` 数组的三个连续字节位置读取每个通道对应的值，执行灰度像素值的计算，并使用 ` grayOffset` 将该值写入 `Pout` 数组。

```cpp
// we have 3 channels corresponding to RGB
// The input image is encoded as unsigned characters [0, 255]
__global__
void colorToGreyscaleConversion(unsigned char * Pout, 
                                unsigned char * Pin,
                                int width, int height) 
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height) {

        // get 1D coordinate for the grayscale image
        int greyOffset = Row*width + Col;

        // one can think of the RGB image having
        // CHANNEL times columns than the grayscale image
        int rgbOffset = greyOffset*CHANNELS;
        unsigned char r = Pin[rgbOffset + 0]; // red value for pixel
        unsigned char g = Pin[rgbOffset + 1]; // green value for pixel
        unsigned char b = Pin[rgbOffset + 2]; // blue value for pixel
      
        // perform the rescaling and store it
        // We multiply by floating point constants
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}
```

## 3.3 Image blur: a more complex kernel

&emsp;&emsp;图像模糊函数将输出图像像素的值计算为相邻像素 (包括输入图像中像素) 的加权和。简便起见，我们使用相邻像素的平均值来计算结果，对应的代码如下。与 `colorToGrayscaleConversion` 中使用的策略类似，对每个输出像素使用 1 个线程来计算。`col`和 `row` 表示输入像素 patch 的中心像素位置。嵌套的 `for` 循环遍历 patch 中的所有像素。`if` 语句的 `curRow < 0` 和 `curCol < 0` 条件用于跳过执行超出图像范围的部分。

```cpp
__global__
void blurKernel(unsigned char *in, unsigned char *out, int width, int height)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;
    if (Col < width && Row < height) {
      
        int pixVal = 0;
        int pixels = 0;

        // Get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; blurRow++) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; blurCol++) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;
              
                // If the pixel is within the image, add its value to the sum
                if(curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
                    pixVal += in[curRow*width + curCol];
                    pixels++; // Keep track of the number of pixels in the avg
                }
            }
        }
        // Write our new pixel value out
        out[Row*width + Col] = (unsigned char)(pixVal / pixels);
    }
}
```

## 3.4 Matrix multiplication

矩阵乘法是 Basic Linear Algebra Subprograms (BLAS) 的重要组成部分。

- Level 1 形如 {% katex %} y = \alpha x + y {% endkatex %} 的向量运算。
- Level 2 形如 {% katex %} y = \alpha Ax + \beta y {% endkatex %} 的矩阵-向量运算。
- Level 3 形如 {% katex %} y = \alpha AB + \beta C {% endkatex %} 的矩阵-矩阵运算。

&emsp;&emsp;为了用 CUDA 实现矩阵乘法，我们可以采取与 colorToGrayscaleConversion 相同的方法将网格中的线程映射到输出矩阵 P 的元素，即每个线程负责计算 P 中的一个元素。

```cpp
// Assuming square matrices of size Width x Width
__global__ 
void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    // Calculate the row index of the P element and M
    int Row = blockIdx.y*blockDim.y+threadIdx.y;
    // Calculate the column index of P and N
    int Col = blockIdx.x*blockDim.x+threadIdx.x;

    if ((Row >= Width) || (Col >= Width)) return;

    float Pvalue = 0;
    // each thread computes one element of the block sub-matrix
    for (int k = 0; k < Width; ++k)
        Pvalue += M[Row*Width+k]*N[k*Width+Col];

    P[Row*Width+Col] = Pvalue;
}
```

![Matrix Multiplication by Tiling P](https://note.youdao.com/yws/api/personal/file/WEBb16353405dede29b1f85c4c05008cda6?method=download&shareKey=6ffa3c49a5c8db350dd70df1f42dd2de "Matrix Multiplication by Tiling P")
