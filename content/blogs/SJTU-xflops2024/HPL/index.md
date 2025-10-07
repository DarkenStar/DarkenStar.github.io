---
title: "HPL"
date: 2025-09-20T11:42:56+08:00
lastmod: 2025-09-20T11:42:56+08:00
author: ["WITHER"]

categories:
- HPC

tags:
- HPC

keywords:
- 

description: "Solution of SJTU-xflops2024 HPL." # 文章描述，与搜索优化相关
summary: "Solution of SJTU-xflops2024 HPL." # 文章简单描述，会展示在主页
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

题目已经说了以从以下两个方面提高你的HPL性能:
- 修改 Make.xlops 文件，尝试使用更为快速库或编译器优化提升性能。
- 修改 HPL.dat 文件，尝试通过调整参数得到更高的性能。

```
HPLinpack benchmark input file
Innovative Computing Laboratory, University of Tennessee
HPL.out      output file name (if any)
6            device out (6=stdout,7=stderr,file)
4            # of problems sizes (N)
29 30 34 35  Ns
4            # of NBs
1 2 3 4      NBs
0            PMAP process mapping (0=Row-,1=Column-major)
3            # of process grids (P x Q)
2 1 4        Ps
2 4 1        Qs
16.0         threshold
3            # of panel fact
0 1 2        PFACTs (0=left, 1=Crout, 2=Right)
2            # of recursive stopping criterium
2 4          NBMINs (>= 1)
1            # of panels in recursion
2            NDIVs
3            # of recursive panel fact.
0 1 2        RFACTs (0=left, 1=Crout, 2=Right)
1            # of broadcast
0            BCASTs (0=1rg,1=1rM,2=2rg,3=2rM,4=Lng,5=LnM)
1            # of lookahead depth
0            DEPTHs (>=0)
2            SWAP (0=bin-exch,1=long,2=mix)
64           swapping threshold
0            L1 in (0=transposed,1=no-transposed) form
0            U  in (0=transposed,1=no-transposed) form
1            Equilibration (0=no,1=yes)
8            memory alignment in double (> 0)
```

矩阵大小
```
4            # of problems sizes (N)
29 30 34 35  Ns
```

- `4`：表示将测试 4 种不同的矩阵大小.
- `29 30 34 35`：以 K 为单位的矩阵大小的具体值。实际矩阵维度为 29000, 30000, 34000, 35000.

矩阵分块大小
```
4            # of NBs
1 2 3 4      NBs
```
- `4`：测试 4 种不同的块大小。
- `1 2 3 4`：块大小的具体值。

进程映射方式

```
0            PMAP process mapping (0=Row-,1=Column-major)
```

定义 MPI 进程如何映射到矩阵的行和列。

进程网格数量和配置

```
3            # of process grids (P x Q)
2 1 4        Ps
2 4 1        Qs
```

- `3`：测试 3 种不同的进程网格（P × Q）。
- `Ps` 和 `Qs`：定义进程网格的行数（P）和列数（Q），分别为 (2×2), (1×4), (4×1).

总进程数 = `P × Q`，例如 `(2×2)=4, (1×4)=4, (4×1)=4`.

