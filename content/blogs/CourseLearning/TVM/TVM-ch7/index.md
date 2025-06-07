---
title: TVM Learning (9)-GPU and Hardware Acceleration, Part 2
date: 2024-08-25T13:22:00+08:00
lastmod: 2024-08-25T13:22:00+08:00
draft: false
author: ["WITHER"]
keywords: 
    - TVM
categories:
    - TVM Learning
tags:
    - Autotuning
description: Personal notebook 7.
summary: Personal notebook 7.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# Key Elements of Specialized Code

下面用 low-level numpy 写的 python 代码展示了一系列在专用硬件后端可能使用到的操作。

```python
def accel_fill_zero(C):
    C[:] = 0
  
def accel_tmm_add(C, A, B):
    C[:] += A @ B.T         

def accel_dma_copy(reg, dram):
    reg[:] = dram[:]
```

我们假设基础的运算单元可以进行 `16x16`的矩阵乘法 (`accel_tmm_add`)，接收2个寄存器里的 RHS 输入和表示累加中间结果的 LHS 输入，数据拷贝使用的是专用函数 (`accel_dma_copy`).

```python
# The basis unit of computation is a 16*16*16 matrix multiplication
def lnumpy_tmm(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    # a special accumulator memory
    C_accumulator = np.empty((16, 16), dtype="float32")
    A_reg = np.empty((16, 16), dtype="float32")
    B_reg = np.empty((16, 16), dtype="float32")
  
    for i in range(64):
        for j in range(64):
            accel_fill_zero(C_accumulator)
            for k in range(64):
                accel_dma_copy(A_reg[:], A[i*16 : (i+1)*16, k*16 : (k+1)*16])
                accel_dma_copy(B_reg[:], B[j*16 : (j+1)*16, k*16 : (k+1)*16])
                accel_tmm_add(C_accumulator, A_reg, B_reg)
            accel_dma_copy(C[i*16 : (i+1)*16, j*16 : (j+1)*16], C_accumulator)
```

# A Block with Tensorized Computation

专用加速器代码的结构并非以标量计算为单位。迄今为止，我们运行的大多数 TensorIR 代码都包含一个 block，用于计算输出张量中的单个元素。许多专用加速器在张量区域内进行计算。TensorIR中的 block 可以帮助我们将这些相关计算分组。

```python
@tvm.script.ir_module   
class MatmulBlockModule:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"],
             B: T.Buffer[(1024, 1024), "float32"],
             C: T.Buffer[(1024, 1024), "float32"]) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i0, j0, k0 in T.grid(64, 64, 64):
            with T.block("tmm-16x16"):
                vi0, vj0, vk0 = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    for i1, j1 in T.grid(16, 16):
                        with T.block("tmm_init"):
                            vi1, vj1 = T.axis.remap("SS", [i1, j1])
                            C[vi0 * 16 + vi1, vj0 * 16 + vj1] = T.float32(0)
    
                for i1, j1, k1 in T.grid(16, 16, 16):
                    with T.block("tmm"):
                        vi1, vj1, vk1 = T.axis.remap("SSR", [i1, j1, k1])
                        C[vi0 * 16 + vi1, vj0 * 16 + vj1] += A[vi0 * 16 + vi1, vk0 * 16 + vk1] * B[vj0 * 16 + vj1, vk0 * 16 + vk1]
```

调用 `MatmulBlockModule.show()` 后显示的 TensorIR如下

```python
T.reads(C[vi0 * 16 + vi1, vj0 * 16 + vj1], A[vi0 * 16 + vi1, vk0 * 16 + vk1], B[vj0 * 16 + vj1, vk0 * 16 + vk1])
T.writes(C[vi0 * 16 + vi1, vj0 * 16 + vj1])
```

该代码从 `A` 和 `B` 的 `16x16` 区域读取数据，并写入 `C` 的 `16x16` 区域。在这种情况下，块的内容包含子区域计算的具体实现的进一步细节。我们称这种区块为**张量区块**，因为它们包含跨越张量子区域的计算。

```python
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, j0, k0 in T.grid(64, 64, 64):
            with T.block("tmm-16x16"):
                vi0, vj0, vk0 = T.axis.remap("SSR", [i0, j0, k0])
                T.reads(A[vi0 * 16:vi0 * 16 + 16, vk0 * 16:vk0 * 16 + 16], B[vj0 * 16:vj0 * 16 + 16, vk0 * 16:vk0 * 16 + 16])
                T.writes(C[vi0 * 16:vi0 * 16 + 16, vj0 * 16:vj0 * 16 + 16])
                with T.init():
                    for i1, j1 in T.grid(16, 16):
                        with T.block("tmm_init"):
                            vi1, vj1 = T.axis.remap("SS", [i1, j1])
                            T.reads()
                            T.writes(C[vi0 * 16 + vi1, vj0 * 16 + vj1])
                            C[vi0 * 16 + vi1, vj0 * 16 + vj1] = T.float32(0.0)
                for i1, j1, k1 in T.grid(16, 16, 16):
                    with T.block("tmm"):
                        vi1, vj1, vk1 = T.axis.remap("SSR", [i1, j1, k1])
                        T.reads(C[vi0 * 16 + vi1, vj0 * 16 + vj1], A[vi0 * 16 + vi1, vk0 * 16 + vk1], B[vj0 * 16 + vj1, vk0 * 16 + vk1])
                        T.writes(C[vi0 * 16 + vi1, vj0 * 16 + vj1])
                        C[vi0 * 16 + vi1, vj0 * 16 + vj1] = C[vi0 * 16 + vi1, vj0 * 16 + vj1] + A[vi0 * 16 + vi1, vk0 * 16 + vk1] * B[vj0 * 16 + vj1, vk0 * 16 + vk1]
```

# Transforming Loops Around Tensorized Block

我们可以对张量计算块的循环进行变换，这些循环变换可以重新组织计算该块的迭代方式，得到不同的张量程序。

```python
sch = tvm.tir.Schedule(MatmulBlockModule)

block_mm = sch.get_block("tmm-16x16")
i, j, k = sch.get_loops(block_mm)

i0, i1 = sch.split(i, [None, 4])

sch.reorder(i0, j, i1, k)
sch.mod.show()

#------------------------------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0_0, j0, i0_1, k0 in T.grid(16, 64, 4, 64):
            with T.block("tmm-16x16"):
                vi0 = T.axis.spatial(64, i0_0 * 4 + i0_1)
                vj0, vk0 = T.axis.remap("SR", [j0, k0])
                T.reads(A[vi0 * 16:vi0 * 16 + 16, vk0 * 16:vk0 * 16 + 16], B[vj0 * 16:vj0 * 16 + 16, vk0 * 16:vk0 * 16 + 16])
                T.writes(C[vi0 * 16:vi0 * 16 + 16, vj0 * 16:vj0 * 16 + 16])
                with T.init():
                    for i1, j1 in T.grid(16, 16):
                        with T.block("tmm_init"):
                            vi1, vj1 = T.axis.remap("SS", [i1, j1])
                            T.reads()
                            T.writes(C[vi0 * 16 + vi1, vj0 * 16 + vj1])
                            C[vi0 * 16 + vi1, vj0 * 16 + vj1] = T.float32(0.0)
                for i1, j1, k1 in T.grid(16, 16, 16):
                    with T.block("tmm"):
                        vi1, vj1, vk1 = T.axis.remap("SSR", [i1, j1, k1])
                        T.reads(C[vi0 * 16 + vi1, vj0 * 16 + vj1], A[vi0 * 16 + vi1, vk0 * 16 + vk1], B[vj0 * 16 + vj1, vk0 * 16 + vk1])
                        T.writes(C[vi0 * 16 + vi1, vj0 * 16 + vj1])
                        C[vi0 * 16 + vi1, vj0 * 16 + vj1] = C[vi0 * 16 + vi1, vj0 * 16 + vj1] + A[vi0 * 16 + vi1, vk0 * 16 + vk1] * B[vj0 * 16 + vj1, vk0 * 16 + vk1]
```

# Blockization -- Creating Tensorized Blocks

TensorIR 提供了一种变换原语 `blockize` 来将循环的子区域组合在一起以形成张量化的计算 block. 例如我们可以将下面2个的 `1024x1024` 矩阵乘法分解成很多个 `16x16` 的矩阵乘法。

```python
@tvm.script.ir_module 
class MatmulModule:
    @T.prim_func
    def main(
        A: T.Buffer[(1024, 1024), "float32"],
        B: T.Buffer[(1024, 1024), "float32"],
        C: T.Buffer[(1024, 1024), "float32"],
    ) -> None:
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(1024, 1024, 1024):
            with T.block("matmul"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vj, vk]

sch = tvm.tir.Schedule(MatmulModule)
i, j, k = sch.get_loops("matmul")
i, ii = sch.split(i, factors=[None, 16])
j, ji = sch.split(j, factors=[None, 16])
k, ki = sch.split(k, factors=[None, 16])
sch.reorder(i, j, k, ii, ji, ki)
sch.mod.show()

#-------------------------------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i_0, j_0, k_0, i_1, j_1, k_1 in T.grid(64, 64, 64, 16, 16, 16):
            with T.block("matmul"):
                vi = T.axis.spatial(1024, i_0 * 16 + i_1)
                vj = T.axis.spatial(1024, j_0 * 16 + j_1)
                vk = T.axis.reduce(1024, k_0 * 16 + k_1)
                T.reads(A[vi, vk], B[vj, vk])
                T.writes(C[vi, vj])
                with T.init():
                    C[vi, vj] = T.float32(0.0)
                C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vj, vk]

```

---

`blockize` 是用来将一个或多个块(block)或一个特定循环的子树合并成一个新的块。如果 `target` 是一个循环的根节点,则会将该循环下的所有块合并成一个新块，如果 `target` 是一个块的列表,则会将这些块合并成一个新块。然后将新块返回

**参数说明** :

* `target`: 需要被合并的块或循环的根节点。可以是 `LoopRV` 类型(表示一个循环)或 `List[BlockRV]` 类型(表示多个块)。
* `preserve_unit_iters`: 一个布尔值,表示是否保留块绑定中的单元迭代器。

**限制条件** :

* `blockize` 要求给定的循环下只有一个块,且该块的绑定必须能够被该循环的子空间整除。

调用 `blockize` 后的 TensorIR 如下

```python
block_mm = sch.blockize(ii)
sch.mod.show()

#-------------------------------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i_0, j_0, k_0 in T.grid(64, 64, 64):
            with T.block("matmul_o"):
                vi_o, vj_o, vk_o = T.axis.remap("SSR", [i_0, j_0, k_0])
                T.reads(A[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], B[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16])
                T.writes(C[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                with T.init():
                    for i_1, j_1 in T.grid(16, 16):
                        with T.block("matmul_init"):
                            vi_i_init, vj_i_init = T.axis.remap("SS", [i_1, j_1])
                            T.reads()
                            T.writes(C[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init])
                            C[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init] = T.float32(0.0)
                for i_1, j_1, k_1 in T.grid(16, 16, 16):
                    with T.block("matmul"):
                        vi_i, vj_i, vk_i = T.axis.remap("SSR", [i_1, j_1, k_1])
                        T.reads(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i], A[vi_o * 16 + vi_i, vk_o * 16 + vk_i], B[vj_o * 16 + vj_i, vk_o * 16 + vk_i])
                        T.writes(C[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                        C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = C[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + A[vi_o * 16 + vi_i, vk_o * 16 + vk_i] * B[vj_o * 16 + vj_i, vk_o * 16 + vk_i]
```

# Transforming TensorIR to Introduce Special Memory Scope

正如在 low-level NumPy 代码中提到的，底层 TensorIR 的一个关键要素是加速过程中使用的特殊内存范围。我们可以使用 cache_read 和 write 来创建中间内存阶段。

storage_scope 在这里指的是内存存储范围或存储层次。常见的存储范围包括:

- global: 表示数据存储在全局内存中。这是最高层次的内存范围。
- shared: 表示数据存储在GPU的共享内存中。
- local: 表示数据存储在CPU或GPU的寄存器中。这是最底层的内存范围。

`global.A_reg` 表示数据将被缓存到一个名为 A_reg 的全局内存缓存中。

![Storage Scope](https://mlc.ai/zh/_images/hardware_specialization_abc.png "Storage Scope")

```python
A_reg = sch.cache_read(block_mm, 0, storage_scope="global.A_reg")
B_reg = sch.cache_read(block_mm, 1, storage_scope="global.B_reg")
sch.compute_at(A_reg, k)
sch.compute_at(B_reg, k)

write_back_block = sch.cache_write(block_mm, 0, storage_scope="global.accumulator")
sch.reverse_compute_at(write_back_block, j)
sch.mod.show()

#-----------------------------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_global_A_reg = T.alloc_buffer((1024, 1024), scope="global.A_reg")
        B_global_B_reg = T.alloc_buffer((1024, 1024), scope="global.B_reg")
        C_global_accumulator = T.alloc_buffer((1024, 1024), scope="global.accumulator")
        for i_0, j_0 in T.grid(64, 64):
            for k_0 in range(64):
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("A_global.A_reg"):
                        v0 = T.axis.spatial(1024, i_0 * 16 + ax0)
                        v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                        T.reads(A[v0, v1])
                        T.writes(A_global_A_reg[v0, v1])
                        A_global_A_reg[v0, v1] = A[v0, v1]
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("B_global.B_reg"):
                        v0 = T.axis.spatial(1024, j_0 * 16 + ax0)
                        v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                        T.reads(B[v0, v1])
                        T.writes(B_global_B_reg[v0, v1])
                        B_global_B_reg[v0, v1] = B[v0, v1]
                with T.block("matmul_o"):
                    vi_o, vj_o, vk_o = T.axis.remap("SSR", [i_0, j_0, k_0])
                    T.reads(A_global_A_reg[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], B_global_B_reg[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16])
                    T.writes(C_global_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                    with T.init():
                        for i_1, j_1 in T.grid(16, 16):
                            with T.block("matmul_init"):
                                vi_i_init, vj_i_init = T.axis.remap("SS", [i_1, j_1])
                                T.reads()
                                T.writes(C_global_accumulator[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init])
                                C_global_accumulator[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init] = T.float32(0.0)
                    for i_1, j_1, k_1 in T.grid(16, 16, 16):
                        with T.block("matmul"):
                            vi_i, vj_i, vk_i = T.axis.remap("SSR", [i_1, j_1, k_1])
                            T.reads(C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i], A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i], B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i])    
                            T.writes(C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                            C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i] * B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i]
            for ax0, ax1 in T.grid(16, 16):
                with T.block("C_global.accumulator"):
                    v0 = T.axis.spatial(1024, i_0 * 16 + ax0)
                    v1 = T.axis.spatial(1024, j_0 * 16 + ax1)
                    T.reads(C_global_accumulator[v0, v1])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_global_accumulator[v0, v1]

```

# Tensorization

现在我们已经创建了一组映射到 TensorIR 中相应计算阶段的块。剩下的步骤是映射部分张量块，以使用映射到硬件加速指令的特定实现。这一映射过程称为**张量化**。为了实现张量化，我们首先注册一个 TensorIntrin，其中包含计算和实现的描述。

```python
@T.prim_func
def tmm16_desc(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16, scope="global.accumulator")
  
    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        for i, j, k in T.grid(16, 16, 16):
            with T.block(""):
                vii, vjj, vkk = T.axis.remap("SSR", [i, j, k])
                C[vii, vjj] = C[vii, vjj] + A[vii, vkk] * B[vjj, vkk]

@T.prim_func
def tmm16_impl(a: T.handle, b: T.handle, c: T.handle) -> None:
    A = T.match_buffer(a, (16, 16), "float32", offset_factor=16, scope="global.A_reg")
    B = T.match_buffer(b, (16, 16), "float32", offset_factor=16, scope="global.B_reg")
    C = T.match_buffer(c, (16, 16), "float32", offset_factor=16, scope="global.accumulator")
    sa = T.int32(16)#T.var("int32")
    sb = T.int32(16)#T.var("int32")
    sc = T.int32(16)#T.var("int32")
  
  
    with T.block("root"):
        T.reads(C[0:16, 0:16], A[0:16, 0:16], B[0:16, 0:16])
        T.writes(C[0:16, 0:16])
        T.evaluate(
            T.call_extern("float32", "tmm16", C.access_ptr("w"), A.access_ptr("r"), B.access_ptr("r"),
                          sa, sb, sc)
        )

tvm.tir.TensorIntrin.register("tmm16", tmm16_desc, tmm16_impl)
```

首先我们用 `decompose_reduction` 将 `C_global_accumulator` 的初始化和更新部分分开成 `T.block("matmul_init")` 和 `T.block("matmul_o_update")`

```python
sch.decompose_reduction(block_mm, k)
sch.mod.show()

#---------------------------------------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((1024, 1024), "float32"), B: T.Buffer((1024, 1024), "float32"), C: T.Buffer((1024, 1024), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        A_global_A_reg = T.alloc_buffer((1024, 1024), scope="global.A_reg")
        B_global_B_reg = T.alloc_buffer((1024, 1024), scope="global.B_reg")
        C_global_accumulator = T.alloc_buffer((1024, 1024), scope="global.accumulator")
        for i_0, j_0 in T.grid(64, 64):
            with T.block("matmul_o_init"):
                vi_o, vj_o = T.axis.remap("SS", [i_0, j_0])
                T.reads()
                T.writes(C_global_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                for i_1, j_1 in T.grid(16, 16):
                    with T.block("matmul_init"):
                        vi_i_init, vj_i_init = T.axis.remap("SS", [i_1, j_1])
                        T.reads()
                        T.writes(C_global_accumulator[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init])
                        C_global_accumulator[vi_o * 16 + vi_i_init, vj_o * 16 + vj_i_init] = T.float32(0.0)
            for k_0 in range(64):
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("A_global.A_reg"):
                        v0 = T.axis.spatial(1024, i_0 * 16 + ax0)
                        v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                        T.reads(A[v0, v1])
                        T.writes(A_global_A_reg[v0, v1])
                        A_global_A_reg[v0, v1] = A[v0, v1]
                for ax0, ax1 in T.grid(16, 16):
                    with T.block("B_global.B_reg"):
                        v0 = T.axis.spatial(1024, j_0 * 16 + ax0)
                        v1 = T.axis.spatial(1024, k_0 * 16 + ax1)
                        T.reads(B[v0, v1])
                        T.writes(B_global_B_reg[v0, v1])
                        B_global_B_reg[v0, v1] = B[v0, v1]
                with T.block("matmul_o_update"):
                    vi_o, vj_o, vk_o = T.axis.remap("SSR", [i_0, j_0, k_0])
                    T.reads(C_global_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16], A_global_A_reg[vi_o * 16:vi_o * 16 + 16, vk_o * 16:vk_o * 16 + 16], B_global_B_reg[vj_o * 16:vj_o * 16 + 16, vk_o * 16:vk_o * 16 + 16])
                    T.writes(C_global_accumulator[vi_o * 16:vi_o * 16 + 16, vj_o * 16:vj_o * 16 + 16])
                    for i_1, j_1, k_1 in T.grid(16, 16, 16):
                        with T.block("matmul"):
                            vi_i, vj_i, vk_i = T.axis.remap("SSR", [i_1, j_1, k_1])
                            T.reads(C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i], A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i], B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i])    
                            T.writes(C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i])
                            C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i] = C_global_accumulator[vi_o * 16 + vi_i, vj_o * 16 + vj_i] + A_global_A_reg[vi_o * 16 + vi_i, vk_o * 16 + vk_i] * B_global_B_reg[vj_o * 16 + vj_i, vk_o * 16 + vk_i]
            for ax0, ax1 in T.grid(16, 16):
                with T.block("C_global.accumulator"):
                    v0 = T.axis.spatial(1024, i_0 * 16 + ax0)
                    v1 = T.axis.spatial(1024, j_0 * 16 + ax1)
                    T.reads(C_global_accumulator[v0, v1])
                    T.writes(C[v0, v1])
                    C[v0, v1] = C_global_accumulator[v0, v1]

```

然后我们调用 `tensorize`，将 `block_mm`（对应于 `matmul_o_update` block ）映射到 `tmm16_impl`. 这里我们使用 `T.call_extern` 来调用环境中的外部函数。 下游编译步骤可以轻松地将实现映射到实现操作的指令。或者我们可以将 `tmm16` 映射到实现这种张量化计算的微内核。 以下代码显示了如何通过外部 C++ 代码执行此操作。

具体实现步骤如下:

1. 定义 C++ 风格的 `tmm16` 函数: 这个函数实现了一个 16x16 矩阵乘法的计算逻辑。它接受三个输入张量 `aa`、`bb` 和 `cc`，以及对应的步长 `stride_a`、`stride_b` 和 `stride_c`。函数使用三重循环执行矩阵乘法的计算,将结果累加到 `cc` 张量中。
2. 使用 TVM 的 `clang` 模块将 C++ 代码编译为 LLVM IR 代码: 首先创建一个临时目录 `temp` 用于存储生成的 LLVM IR 文件。然后调用 `clang.create_llvm()` 函数,传入 C++ 代码字符串 `cc_code`。`create_llvm()` 函数会将 C++ 代码编译为 LLVM IR 代码,并保存到 `ll_path` 指定的文件中。最后返回生成的 LLVM IR 代码。

```python
def tmm_kernel():
    cc_code = '''
        extern "C" int tmm16(float *cc, float *aa, float *bb, int stride_a, int stride_b, int stride_c) {
            for (int i = 0; i < 16; i++) {
                for (int j = 0; i < 16; j++) {
                    for (int k = 0; k < 16; k++) {
                        cc[i * stride_c + j] += aa[i * stride_a + k] * bb[j * stride_b + k];
                    }
                }
            }
            return 0;
        }
    '''
    from tvm.contrib import utils, clang
  
    temp = utils.tempdir()
    ll_path = temp.relpath("temp.ll")
    # Create LLVM ir from c source code
    ll_code = clang.create_llvm(cc_code, output=ll_path)
    return ll_code
```

调用 `sch.tensorize(block_mm, "tmm16")`报错，原因未知。

```bash
发生异常: TVMError
TVMError: invalid unordered_map<K, T> key
  File "C:\Users\17725\Desktop\Machine Learning Compilation\chapter7.py", line 186, in <module>
    sch.tensorize(block_mm, "tmm16")
tvm._ffi.base.TVMError: TVMError: invalid unordered_map<K, T> key
```
