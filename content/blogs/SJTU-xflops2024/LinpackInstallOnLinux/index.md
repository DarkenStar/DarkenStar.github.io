---
title: "Linpack Install on Linux"
date: 2025-09-20T10:53:21+08:00
lastmod: 2025-09-20T10:53:21+08:00
author: ["WITHER"]

categories:
- category 1
- category 2

tags:
- tag 1
- tag 2

keywords:
- word 1
- word 2

description: "Linpack install record." # 文章描述，与搜索优化相关
summary: "Linpack install record." # 文章简单描述，会展示在主页
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

Linpack 基准测试通常指 High-Performance Linpack (HPL)，这是一个用于评估高性能计算系统浮点性能的工具，常用于 Top500 超级计算机排名。它依赖于 BLAS 库 (如 OpenBLAS)、MPI (如 OpenMPI) 和 HPL 源代码。

# Install Basic Dependency

运行以下命令安装依赖:

```bash
sudo apt update
sudo apt install -y build-essential gfortran gcc g++ make wget git
```

- `build-essential`: 提供 GCC、Make 等编译工具。
- `gfortran`: Fortran 编译器，用于 OpenBLAS 和 HPL.
- `git` 和 `wget`: 用于下载源代码。

# Install OpenBLAS from Source

OpenBLAS 是一个高性能的 BLAS (基本线性代数子程序) 库，支持多线程计算，适用于 HPL 的矩阵运算。

1. 克隆最新版 OpenBLAS 源码:

```bash
cd ~
git clone https://github.com/OpenMathLib/OpenBLAS.git
cd OpenBLAS
```

2. 编译 OpenBLAS，支持多线程 (OpenMP) 和动态架构优化

```bash
make DYNAMIC_ARCH=1 USE_OPENMP=1 -j$(nproc)
sudo make PREFIX=/usr/local/openblas install
```

- `DYNAMIC_ARCH=1`: 支持多种 CPU 架构 (如 Skylake、Zen)，自动选择最佳指令集。
- `USE_OPENMP=1`: 启用 OpenMP 多线程支持，HPL 性能关键。
- `PREFIX=/usr/local/openblas`: 安装到自定义路径，避免覆盖系统库。
- `-j$(nproc)`: 使用所有 CPU 核心加速编译。

编译完成后，检查生成的库文件:

```bash
ls /usr/local/openblas/lib
```

应看到 `libopenblas.so` 等文件。

3. 配置 OpenBLAS 环境变量:

```bash
echo "export LD_LIBRARY_PATH=/usr/local/openblas/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
echo "export CFLAGS=-I/usr/local/openblas/include" >> ~/.bashrc
source ~/.bashrc
```

验证 OpenBLAS 安装:

```bash
pkg-config --libs --cflags openblas
```

输出类似: `-I/usr/local/openblas/include -L/usr/local/openblas/lib -lopenblas`.

# Install OpenMPI from Source

OpenMPI 是一个开源的 MPI (消息传递接口) 实现，支持分布式并行计算，HPL 需要它来运行多节点或多进程任务

1. 下载 OpenMPI 源代码:

```bash
cd ~
wget https://download.open-mpi.org/release/open-mpi/v5.0/openmpi-5.0.5.tar.gz
tar -xvzf openmpi-5.0.5.tar.gz
cd openmpi-5.0.5
```

2. 编译和安装 OpenMPI:

```bash
./configure --prefix=/usr/local/openmpi --enable-mpi-fortran --with-hwloc=internal
make -j$(nproc)
sudo make install
```

- `--prefix=/usr/local/openmpi`: 安装到自定义路径。
- `--enable-mpi-fortran`: 启用 Fortran 支持 (HPL 需要) 。
- `--with-hwloc=internal`: 使用内置 hwloc 库管理硬件拓扑。

3. 配置 OpenMPI 环境变量

```bash
echo "export PATH=/usr/local/openmpi/bin:$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc
```

验证 OpenMPI 安装:

```bash
which mpicc
which mpifort
mpirun --version
```

应显示 /usr/local/openmpi/bin/mpicc 等路径和版本信息。

# Install HPL from Source

HPL 官方源代码来自 Netlib，执行以下命令下载 HPL 2.3 版本，解压并进入目录。

```bash
cd ~
wget https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
tar -xvzf hpl-2.3.tar.gz
mv hpl-2.3 hpl
cd hpl
```

2. 配置 HPL 构建

HPL 需要自定义 Makefile 来链接 OpenBLAS 和 OpenMPI. 复制模板并编辑:

```bash
cp setup/Make.Linux_PII_CBLAS Make.Linux
```

将 `Make.Linux` 相关配置替换为

```makefile
ARCH         = Linux
TOPdir       = $(HOME)/hpl
MPdir        = /usr/local/openmpi
MPinc        = -I$(MPdir)/include
MPlib        = $(MPdir)/lib/libmpi.so
LAdir        = /usr/local/openblas/lib
LAlib        = $(LAdir)/libopenblas.so
LApaths      = -L$(LAdir)
CC           = $(MPdir)/bin/mpicc
CCFLAGS      = $(HPL_DEFS) -fopenmp -O3 -funroll-loops -fPIC
LINKER       = $(CC)
LINKFLAGS    = $(CCFLAGS)
F77          = $(MPdir)/bin/mpifort
FFLAGS       = -fopenmp -O3 -funroll-loops -fPIC
RANLIB       = ranlib
HPL_OPTS     = -DHPL_CALL_CBLAS
```

3. 构建 HPL

```bash
make arch=Linux
```

会生成可执行文件 xhpl (双精度版本) 。构建过程可能需要几分钟，输出在 bin/Linux/ 目录。

4. 测试 HPL

编译成功后，进入 bin/Linux 目录并运行测试:

```bash
cd ~/hpl/bin/Linux
export OMP_NUM_THREADS=8  # 替换为 CPU 核心数
mpirun --allow-run-as-root -np 4 ./xhpl
```

```
Finished    864 tests with the following results:
            864 tests completed and passed residual checks,
              0 tests completed and failed residual checks,
              0 tests skipped because of illegal input values.
--------------------------------------------------------------------------------

End of Tests.
```