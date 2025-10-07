---
title: "Clever_Clang"
date: 2025-09-19T13:37:36+08:00
lastmod: 2025-09-19T13:37:36+08:00
author: ["WITHER"]

categories:
- HPC

tags:
- HPC

keywords:
- 

description: "Solution of SJTU-xflops2024 Clever Clang." # 文章描述，与搜索优化相关
summary: "Solution of SJTU-xflops2024 Clever Clang." # 文章简单描述，会展示在主页
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

# Problem Analysis

题目已经说了使用向量化指令进行优化，这里使用的是 基于AVX2 SIMD指令集。并使用 Intel intrinsics _mm256_系列函数来实现向量化。具体优化的手段有:

1. 交换内外循环: 原代码的外循环是 N 个点，内循环是 M 次迭代。由于内循环每次更新依赖前一次，难以直接向量化内循环。因此，将循环交换为先迭代 M 次，在每次迭代中向量化处理 N 个点。
2. 内循环中使用 AVX2 指令：同时处理 8 个点的 x 值，计算 x², x³, 梯度，并进行更新。先处理 8 的倍数部分，然后用标量循环处理剩余点。

完整代码如下，经过测试优化后为满分。

```cpp{linenos=true}
void gradient_descent(float *points, uint32_t N, uint32_t M, float eta, const PolyParams* params) {
    // 预先计算常量向量
    __m256 a4 = _mm256_set1_ps(4.0f * params->a);
    __m256 b3 = _mm256_set1_ps(3.0f * params->b);
    __m256 c2 = _mm256_set1_ps(2.0 * params->c);
    __m256 d_vec = _mm256_set1_ps(params->d);
    __m256 eta_vec = _mm256_set1_ps(eta);

    for (uint32_t j = 0; j < M; ++j) {  // AVX2 一次处理 8 个 float (256b)
        uint32_t i = 0;
        for (; i + 8 <= N; i += 8) {
            __m256 x = _mm256_loadu_ps(&points[i]);

            // compute x^2, x^3
            __m256 x2 = _mm256_mul_ps(x, x);
            __m256 x3 = _mm256_mul_ps(x2, x);

            // compute gradient
            __m256 grad = _mm256_mul_ps(a4, x3);
            grad = _mm256_add_ps(grad, _mm256_mul_ps(b3, x2));
            grad = _mm256_add_ps(grad, _mm256_mul_ps(c2, x));
            grad = _mm256_add_ps(grad, d_vec);

            // update x -= eta * grad
            __m256 delta = _mm256_mul_ps(eta_vec, grad);
            x = _mm256_sub_ps(x, delta);

            // restore to points
            _mm256_storeu_ps(&points[i], x);
        }

        for (; i < N; ++i) {  // 剩余点
            float x = points[i];
            float grad = poly_gradient(x, params);
            x -= eta * grad;
            points[i] = x;
        }
    }
}
```

```
Using cores: 0,1,2,3
Running case 0/5
cost: 0.0ms
cost: 0.0ms
cost: 0.0ms
cost: 0.0ms
cost: 0.0ms
cost: 0.0ms
Running case 1/5
cost: 30.0ms
cost: 30.0ms
cost: 30.0ms
cost: 30.0ms
cost: 30.0ms
cost: 31.0ms
Running case 2/5
cost: 30.0ms
cost: 31.0ms
cost: 30.0ms
cost: 30.0ms
cost: 30.0ms
cost: 30.0ms
Running case 3/5
cost: 1537.0ms
cost: 1515.0ms
cost: 1568.0ms
cost: 1520.0ms
cost: 1558.0ms
cost: 1529.0ms
Running case 4/5
cost: 517.0ms
cost: 519.0ms
cost: 516.0ms
cost: 521.0ms
cost: 521.0ms
cost: 513.0ms
Case 1 score: 100.00/100         avg_time: 30.2
Case 2 score: 100.00/100         avg_time: 30.2
Case 3 score: 100.00/100         avg_time: 1538.0
Case 4 score: 100.00/100         avg_time: 518.0
Your score: 100.00/100
```