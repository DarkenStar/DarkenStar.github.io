---
title: TVM Learning (8)-GPU and Hardware Acceleration, Part 1
date: 2024-08-24T09:28:00+08:00
lastmod: 2024-08-24T09:28:00+08:00
draft: false
author: ["WITHER"]
keywords: 
    - TVM
categories:
    - TVM Learning
tags:
    - Autotuning
description: Personal notebook 6.
summary: Personal notebook 6.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# GPU Architecture

典型的 GPU 包含一系列流多处理器 (Stream Multi-processor, SM)，每个多处理器都有许多内核 (core). GPU 具有高度并行性，可以同时执行多项任务。

![GPU Architecture](https://mlc.ai/zh/_images/gpu_arch.png "GPU Architecture")

要对 GPU 进行编程，我们需要创建一组线程块 (thread blocks)，每个 thread 映射到单个核心，而 block 映射到流式多处理器 (SM)。

![GPU Programming](https://mlc.ai/zh/_images/gpu_stream_processors.png "GPU Programming")

我们以两个长度为1024的向量加法 `C=A+B`为例，我们先把外循环 split 成两部分

```python
@tvm.script.ir_module  
class MyModuleVecAdd:
    @T.prim_func
    def main(A: T.Buffer[(1024, ), "float32"],
             B: T.Buffer[(1024, ), "float32"],
             C: T.Buffer[(1024, ), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.grid(1024):
            with T.block("C"):
                vi = T.axis.remap("S", [i])
                C[vi] = A[vi] + B[vi]

sch = tvm.tir.Schedule(MyModuleVecAdd)
block_C = sch.get_block("C")
i, = sch.get_loops(block=block_C)
i0, i1 = sch.split(i, [None, 128])
sch.mod.show()
```

得到的 TensorIR 如下

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32"), C: T.Buffer((1024,), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i_0, i_1 in T.grid(8, 128):
            with T.block("C"):
                vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                T.reads(A[vi], B[vi])
                T.writes(C[vi])
                C[vi] = A[vi] + B[vi]
```

# Build and Run the TensorIR Function on GPU

一个CUDA程序的计算被组织成三层次：网格（Grid）、线程块（Block）和线程（Thread）。网格是一个二维的数组，包含多个线程块。每个线程块也是一个二维的数组，包含多个线程。每个线程执行相同的代码，但是在执行时可以使用不同的数据。每个线程由两个索引进行表示 `threadIdx.x`和 `blockIdx.x`. 在实际应用中，有多维线程索引，但这里我们为了简化问题，将它们固定为一维表示。

![GPU Thread Block](https://mlc.ai/zh/_images/gpu_thread_blocks.png "GPU Thread Block")

* `sch.bind(i0, "blockIdx.x")` 将 `i0` 循环绑定到 GPU 的 block 索引，以便将计算分发到不同的 GPU block 上。
* `sch.bind(i1, "threadIdx.x")` 将 `i1` 循环绑定到 GPU 的 thread 索引，以便将计算分发到每个 block 内的不同的 GPU thread 上。

可以看到循环变量变成了 `T.thead_binding`

```python
sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
sch.mod.show()

#--------------------------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024,), "float32"), B: T.Buffer((1024,), "float32"), C: T.Buffer((1024,), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i_0 in T.thread_binding(8, thread="blockIdx.x"):
            for i_1 in T.thread_binding(128, thread="threadIdx.x"):
                with T.block("C"):
                    vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                    T.reads(A[vi], B[vi])
                    T.writes(C[vi])
                    C[vi] = A[vi] + B[vi]
```

然后我们可以在GPU上构建并测试程序的正确性

```python
rt_mod = tvm.build(sch.mod, target="cuda")

A_np = np.random.uniform(size=(1024,)).astype("float32")
B_np = np.random.uniform(size=(1024,)).astype("float32")
A_nd = tvm.nd.array(A_np, tvm.cuda(0))
B_nd = tvm.nd.array(B_np, tvm.cuda(0))
C_nd = tvm.nd.array(np.zeros((1024,), dtype="float32"), tvm.cuda(0))

rt_mod["main"](A_nd, B_nd, C_nd)
np.testing.assert_allclose(C_nd.numpy(), A_np + B_np)
```

# Window Sum Example

滑动窗口求和可以被视为权重为 `[1,1,1]`的卷积，对输入进行滑动并将三个相邻值相加。

![Window Sum](https://mlc.ai/zh/_images/window_sum.png "Window Sum")

跟上一节一样我们将循环split后把外循环和内循环分别bind到block和thread上

```python
@tvm.script.ir_module 
class MyModuleWindowSum:
    @T.prim_func
    def main(A: T.Buffer[(1027, ), "float32"],
             B: T.Buffer[(1024, ), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in T.grid(1024):
            with T.block("C"):
                vi = T.axis.remap("S", [i])
                B[vi] = A[vi] + A[vi + 1] + A[vi + 2]
    
sch = tvm.tir.Schedule(MyModuleWindowSum)
nthread = 128
block_C = sch.get_block("C")
i, = sch.get_loops(block=block_C)
i0, i1 = sch.split(i, [None, nthread])
sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
```

对应的TensorIR如下

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1027,), "float32"), B: T.Buffer((1024,), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i_0 in T.thread_binding(8, thread="blockIdx.x"):
            for i_1 in T.thread_binding(128, thread="threadIdx.x"):
                with T.block("C"):
                    vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                    T.reads(A[vi:vi + 3])
                    T.writes(B[vi])
                    B[vi] = A[vi] + A[vi + 1] + A[vi + 2]
```

## Cache in Shared Memory

我们可以看到在窗口滑动的过程中有一部分数据是重复的。每个 block 包含所有线程都可以在块内访问的共享内存 (shared memory)，为了避免重复从 global memory 加载，我们可以把部分数据缓存到共享内存上

1. `B[vi] = A[vi] + A[vi + 1] + A[vi + 2]` 这一行代码会重复读取 `A` 缓冲区中的数据。
2. `sch.cache_read(block_C, read_buffer_index=0, storage_scope="shared")` 创建了一个名为 `A_shared` 的共享内存缓存，用于存储 `A` 缓冲区中的一部分数据。

* `block_C` 指示缓存与 `C` block 相关联。
* `read_buffer_index=0` 指示缓存 `A` 缓冲区，因为 `A` 是 `C` block 中的第一个读取缓冲区。
* `storage_scope="shared"` 指示缓存使用共享内存。

3. `sch.compute_at(A_shared, i1)` 将 `A_shared` 的计算位置设置为 `i1` 循环，这意味着 `A_shared` 将在每个 thread 中被计算。

```python
sch = tvm.tir.Schedule(MyModuleWindowSum)
nthread = 128
block_C = sch.get_block("C")
i, = sch.get_loops(block=block_C)
i0, i1 = sch.split(i, [None, nthread])
sch.bind(i0, "blockIdx.x")
sch.bind(i1, "threadIdx.x")
A_shared = sch.cache_read(block_C, read_buffer_index=0, storage_scope="shared")
sch.compute_at(A_shared, i1)
sch.mod.show()
```

变换后的TensorIR如下，主要进行了

1. **共享内存分配：** 在每个 GPU block 的共享内存中分配了一个大小为 `(1027,)` 的缓冲区 `A_shared`。

   ```python
   A_shared = T.alloc_buffer((1027,), scope="shared")
   ```
2. 添加了一个新的 block `A_shared`，循环遍历每个 thread并将 `A` 缓冲区中的数据缓存到 `A_shared` 中：

   ```python
   for i_0 in T.thread_binding(8, thread="blockIdx.x"):
       for i_1 in T.thread_binding(128, thread="threadIdx.x"):
           for ax0 in range(130):
               with T.block("A_shared"):
                   v0 = T.axis.spatial(1027, i_0 * 128 + ax0)
                   T.reads(A[v0])
                   T.writes(A_shared[v0])
                   A_shared[v0] = A[v0]
   ```
3. 码更新了 `C` block 中的计算，使其从 `A_shared` 中读取数据：

   ```python
   with T.block("C"):
       vi = T.axis.spatial(1024, i_0 * 128 + i_1)
       T.reads(A_shared[vi:vi + 3])
       T.writes(B[vi])
       B[vi] = A_shared[vi] + A_shared[vi + 1] + A_shared[vi + 2]
   ```

---

`rane(130)` 的出现是因为需要将 `A` 缓冲区中的数据缓存到共享内存 `A_shared` 中。每个 GPU block 处理的数据范围是 `128` 个元素，对应于 `i1` 循环的范围。由于窗口求和操作需要访问 `A` 缓冲区中当前元素的三个相邻元素，因此每个 thread 需要访问 `128 + 2 = 130` 个元素。为了确保每个 thread 都能访问到所需的数据，需要将 `A` 缓冲区中 `130` 个元素缓存到 `A_shared` 中。

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1027,), "float32"), B: T.Buffer((1024,), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_shared = T.alloc_buffer((1027,), scope="shared")
        for i_0 in T.thread_binding(8, thread="blockIdx.x"):
            for i_1 in T.thread_binding(128, thread="threadIdx.x"):
                for ax0 in range(130):
                    with T.block("A_shared"):
                        v0 = T.axis.spatial(1027, i_0 * 128 + ax0)
                        T.reads(A[v0])
                        T.writes(A_shared[v0])
                        A_shared[v0] = A[v0]
                with T.block("C"):
                    vi = T.axis.spatial(1024, i_0 * 128 + i_1)
                    T.reads(A_shared[vi:vi + 3])
                    T.writes(B[vi])
                    B[vi] = A_shared[vi] + A_shared[vi + 1] + A_shared[vi + 2]
```

## Get CUDA  Source

我们可以检查相应的底层代码（CUDA ）

```python
rt_mod = tvm.build(sch.mod, target="cuda")
print(rt_mod.imported_modules[0].get_source())
```

生成的代码包含两部分：

* 在主机 (CPU) 上的调用 GPU 程序的部分；
* 相应计算的 CUDA 内核。

```cpp
#if (((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 4)) || \
     (__CUDACC_VER_MAJOR__ > 11))
#define TVM_ENABLE_L2_PREFETCH 1
#else
#define TVM_ENABLE_L2_PREFETCH 0
#endif

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ A, float* __restrict__ B);
extern "C" __global__ void __launch_bounds__(128) main_kernel(float* __restrict__ A, float* __restrict__ B) {
  __shared__ float A_shared[130];
  for (int ax0 = 0; ax0 < 130; ++ax0) {
    A_shared[ax0] = A[((((int)blockIdx.x) * 128) + ax0)];
  }
  __syncthreads();
  B[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = ((A_shared[((int)threadIdx.x)] + A_shared[(((int)threadIdx.x) + 1)]) + A_shared[(((int)threadIdx.x) + 2)]);
}
```

# Matrix Multiplication

下面我们对原始的 `1024*1024`的矩阵乘法进行优化

```python
@tvm.script.ir_module
class MyModuleMatmul:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"], 
             B: T.Buffer[(1024, 1024), "float32"], 
             C: T.Buffer[(1024, 1024), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

## Local Blocking

下面的blocking 函数使用了一种称为 局部阻塞 的优化策略，将矩阵乘法的计算分解成更小的块，并使用共享内存缓存来提高性能。

![Local Blocking](https://mlc.ai/zh/_images/gpu_local_blocking.png "Local Blocking")

1. 将三个循环 `i`、`j` 和 `k` 分别拆分成多个循环，例如将 `i` 拆分成 `i0`、`i1` 和 `i2`，分别对应于 block 索引、thread 索引和局部循环索引。
2. `k1`表示矩阵计算被拆分成多少个小块，`k0`决定了每个线程需要进行多少次累加操作。调整循环的顺序，以便在每个 thread 中计算 `k0` 循环的所有迭代，从而利用共享内存缓存。
3. 使用 `cache_write` 函数创建一个名为 `C_local` 的共享内存缓存，用于存储 `C` 矩阵的中间结果。
4. 使用 `reverse_compute_at` 函数将 `C_local` 的计算位置设置为 `j1` 循环，以便在每个 thread 中计算 `C_local` 的所有迭代，从而利用共享内存缓存。
5. 将 `i0` 和 `j0` 绑定到 GPU 的 `blockIdx.y` 和 `blockIdx.x` 线程索引，将 `i1` 和 `j1` 绑定到 GPU 的 `threadIdx.y` 和 `threadIdx.x` 线程索引。
6. 使用 `unroll` 函数展开 `k1` 循环，以便在每个 thread 中展开计算，从而提高性能。
7. 使用 `decompose_reduction` 函数分解 `k0` 循环，以便在每个 thread 中计算 `k0` 循环的所有迭代，从而利用共享内存缓存。

```python
def blocking(sch: tvm.tir.Schedule,
             tile_local_y,
             tile_local_x,
             tile_block_y,
             tile_block_x,
             tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")
  
    i, j, k = sch.get_loops(block=block_C)
  
    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])
    sch.unroll(k1)
    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)
  
    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")
  
    sch.bind(i1, "threadIdx.y")
    sch.bind(j1, "threadIdx.x")
    sch.decompose_reduction(block_C, k0)
  
    return sch  
```

进行 Local Blocking 后的TensorIR如下

```python
sch = tvm.tir.Schedule(MyModuleMatmul)
sch = blocking(sch, 8, 8, 8, 8, 4)
sch.mod.show()

#---------------------------------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        C_local = T.alloc_buffer((1024, 1024), scope="local")
        for i_0 in T.thread_binding(16, thread="blockIdx.y"):
            for j_0 in T.thread_binding(16, thread="blockIdx.x"):
                for i_1 in T.thread_binding(8, thread="threadIdx.y"):
                    for j_1 in T.thread_binding(8, thread="threadIdx.x"):
                        for i_2_init, j_2_init in T.grid(8, 8):
                            with T.block("C_init"):
                                vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2_init)
                                vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2_init)
                                T.reads()
                                T.writes(C_local[vi, vj])
                                C_local[vi, vj] = T.float32(0.0)
                        for k_0 in range(256):
                            for k_1 in T.unroll(4):
                                for i_2, j_2 in T.grid(8, 8):
                                    with T.block("C_update"):
                                        vi = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + i_2)
                                        vj = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + j_2)
                                        vk = T.axis.reduce(1024, k_0 * 4 + k_1)
                                        T.reads(C_local[vi, vj], A[vi, vk], B[vk, vj])
                                        T.writes(C_local[vi, vj])
                                        C_local[vi, vj] = C_local[vi, vj] + A[vi, vk] * B[vk, vj]
                        for ax0, ax1 in T.grid(8, 8):
                            with T.block("C_local"):
                                v0 = T.axis.spatial(1024, i_0 * 64 + i_1 * 8 + ax0)
                                v1 = T.axis.spatial(1024, j_0 * 64 + j_1 * 8 + ax1)
                                T.reads(C_local[v0, v1])
                                T.writes(C[v0, v1])
                                C[v0, v1] = C_local[v0, v1]
```

## Shared  Memory Blocking

上面的方法没有考虑相邻 thread 位于同一个 block 中，我们可以将它们需要的数据加载到 shared memory 中。

![Shared Memory Blocking](https://mlc.ai/zh/_images/gpu_shared_blocking.png "Shared Memory Blocking")

`cache_read_and_coop_fetch` 函数负责将 `A` 和 `B` 矩阵中的数据加载到共享内存中。首先使用 `cache_read` 创建一个共享内存缓存，用于存储 `A` 或 `B` 矩阵的数据。然后使用 `compute_at` 将缓存的计算位置设置为 `k0` 循环，在每个线程中计算缓存的所有迭代。最后使用 `split` 和 `vectorize` 函数对 `k0` 循环进行向量化，提高加载数据的效率。

```python
def cache_read_and_coop_fetch(sch: tvm.tir.Schedule, block, nthread, read_idx, read_loc):
    read_cache = sch.cache_read(block=block, read_buffer_index=read_idx, storage_scope="shared")
    sch.compute_at(block=read_cache, loop=read_loc)
    # vertorized cooperative fetch
    inner0, inner1 = sch.get_loops(block=read_cache)[-2:]
    inner = sch.fuse(inner0, inner1)
    _, tx, vec = sch.split(loop=inner, factors=[None, nthread, 4])
    sch.vectorize(vec)
    sch.bind(tx, "threadIdx.x")
```

其余的操作和 Local Blocking 一致

```python
def blocking_with_shared(sch: tvm.tir.Schedule,
                        tile_local_y,
                        tile_local_x,
                        tile_block_y,
                        tile_block_x,
                        tile_k):
    block_C = sch.get_block("C")
    C_local = sch.cache_write(block_C, 0, "local")
  
    i, j, k = sch.get_loops(block=block_C)
  
    i0, i1, i2 = sch.split(loop=i, factors=[None, tile_block_y, tile_local_y])
    j0, j1, j2 = sch.split(loop=j, factors=[None, tile_block_x, tile_local_x])
    k0, k1 = sch.split(loop=k, factors=[None, tile_k])

    sch.reorder(i0, j0, i1, j1, k0, k1, i2, j2)
    sch.reverse_compute_at(C_local, j1)

    sch.bind(i0, "blockIdx.y")
    sch.bind(j0, "blockIdx.x")

    tx = sch.fuse(i1, j1)
    sch.bind(tx, "threadIdx.x")
    nthread = tile_block_y * tile_block_x
    cache_read_and_coop_fetch(sch, block_C, nthread, 0, k0)
    cache_read_and_coop_fetch(sch, block_C, nthread, 1, k0)  
    sch.decompose_reduction(block_C, k0)

    return sch
```
