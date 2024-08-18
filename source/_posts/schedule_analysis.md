---
title: TVM Learning (3)-Schedule Analysis
date: 2024/8/17 14:56:33
categories: TVM
tags: TVM learning
excerpt: TensorIR Schedule analysis. Storge related schedule will be in another post.
mathjax: true
katex: true
---
# LoopRV & BlockRV Object

Schedule要操作的对象主要就是LoopRV和BlockRV，对应于我们TVMScript中的循环变量和计算块部分。下面代码为在 TVM 中注册 `LoopRV` 的自定义对象类型的过程，并通过 FFI（Foreign Function Interface）机制将 C++ 中的函数暴露给 Python.

**注册过程解析：**

1. **定义类:** 首先，定义一个名为 `LoopRV` 的类，它继承自 `tvm.Object` 类。这个类表示一个与循环相关的随机变量。
2. **使用 `@_register_object` 装饰器:** `LoopRV` 类使用 `@_register_object("tir.LoopRV")` 装饰器进行注册。这个装饰器会调用 `register_object` 函数，将 `LoopRV` 类注册到 TVM 的对象系统中，并使用类型键 "tir.LoopRV" 来标识它。
3. **FFI 初始化:** `tvm._ffi._init_api("tir.schedule", __name__)` 这行代码使用 `_init_api` 函数初始化 FFI，将 C++ 中的 "tir.schedule" 模块的函数暴露给 Python。
4. **`_init_api` 和 `_init_api_prefix` 函数:** `_init_api` 函数用于初始化 FFI，它会调用 `_init_api_prefix` 函数来处理具体的函数注册过程。
5. **函数注册:** `_init_api_prefix` 函数会遍历所有 C++ 中的全局函数，找到以 "tir.schedule" 开头的函数，并将其注册到 Python 中。

```python
@_register_object("tir.LoopRV")
class LoopRV(Object):
    """A random variable that refers to a loop"""

    def __init__(self) -> None:
        """Construct a new LoopRV."""
        self.__init_handle_by_constructor__(
            _ffi_api.LoopRV  # type: ignore # pylint: disable=no-member
        )

"""FFI APIs for tvm.tir.schedule"""
import tvm._ffi

tvm._ffi._init_api("tir.schedule", __name__)  # pylint: disable=protected-access
```

{% fold info @_register_object %}

```python
def register_object(type_key=None):   
    def register(cls):
        """internal register function"""
        if hasattr(cls, "_type_index"):
            tindex = cls._type_index
        else:
            tidx = ctypes.c_uint()
            if not _RUNTIME_ONLY:
                check_call(_LIB.TVMObjectTypeKey2Index(c_str(object_name), ctypes.byref(tidx)))
            else:
                # directly skip unknown objects during runtime.
                ret = _LIB.TVMObjectTypeKey2Index(c_str(object_name), ctypes.byref(tidx))
                if ret != 0:
                    return cls
            tindex = tidx.value
        _register_object(tindex, cls)
        return cls

    if isinstance(type_key, str):
        return register

    return register(type_key)
```

**装饰器功能:**

1. **注册对象类型:** 装饰器 `register_object` 的主要作用是将一个类注册到 TVM 的对象系统中，以便 TVM 能够识别和使用该类。
2. **类型键:** 装饰器接受一个可选参数 `type_key`，用于指定该对象的类型键。类型键是一个字符串，用于唯一标识该对象类型。如果 `type_key` 未指定，则使用类的名称作为类型键。
3. **内部注册函数:** 装饰器内部定义了一个名为 `register` 的函数，该函数负责实际的注册操作。
4. **注册过程:**
   * 获取类型索引: `register` 函数首先获取该类型的索引，如果该类型已经注册，则直接获取已有的索引；否则，调用 TVM 的 C API 函数 `TVMObjectTypeKey2Index` 获取新的索引。
   * 注册对象: `register` 函数使用 `_register_object` 函数将类型索引和类对象注册到 TVM 的对象系统中。

{% endfold %}

BlockRV类的定义同理。

```python
@_register_object("tir.BlockRV")
class BlockRV(Object):
    """A random variable that refers to a block"""

    def __init__(self) -> None:
        """Construct a new BlockRV."""
        self.__init_handle_by_constructor__(
            _ffi_api.BlockRV  # type: ignore # pylint: disable=no-member
        )
```

# Schedule Primitive

`Schedule`是一组改变了计算的顺序，但保留了计算的语义的变换。它的构造函数需要一个 `IRModule`实例作为参数。我们以下面的矩阵的 element-wise乘法为例来介绍以下可鞥的变换。

```python
import tvm
from tvm import te   
import numpy as np   

# Declare some variables for use later
n = te.var("n")
m = te.var("m")

# Declare a matrix element-wise multiply
A = te.placeholder((m, n), name="A")
B = te.placeholder((m, n), name="B")
C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name="C")
print(type(A))
s = te.create_schedule([C.op])

# lower 将计算从定义转换成可以调用的IRModule
tvm.lower(s, [A, B, C], simple_mode=True).show()
```

{% fold info @tvm.lower %}

`tvm.lower` 函数是 TVM 中用于将计算图（Compute Graph）降低（lower）到更低级别的表示形式，例如 Relay IR 或 TensorIR ，该函数会返回一个IRModule.

**参数解释:**

* **`inp`:** 输入参数，可以是以下三种类型之一：`tvm.te.schedule.Schedule` 对象：表示计算图的调度信息。

  `tvm.tir.PrimFunc` 对象：表示 TensorIR 的主函数。

  `IRModule` 对象：表示一个包含多个函数的模块。
* **`args`:** 可选参数，表示输入张量的列表，仅在 `inp` 是 `tvm.te.schedule.Schedule` 对象时使用。
* **`name`:** 可选参数，表示生成的函数的名称，默认为 "main"。
* **`binds`:** 可选参数，表示一个字典，用于指定输入张量的绑定，仅在 `inp` 是 `tvm.te.schedule.Schedule` 对象时使用。
* **`simple_mode`:** 可选参数，表示是否使用简化的模式，默认为 `False`。

{% endfold %}

上述代码生成的TensorIR如下

```python
# from tvm.script import ir as I
# from tvm.script import tir as T

@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.handle, B: T.handle, C: T.handle):
        T.func_attr({"from_legacy_te_schedule": T.bool(True), "tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A_1 = T.match_buffer(A, (m, n), strides=("stride", "stride"), buffer_type="auto")
        B_1 = T.match_buffer(B, (m, n), strides=("stride", "stride"), buffer_type="auto")
        C_1 = T.match_buffer(C, (m, n), strides=("stride", "stride"), buffer_type="auto")
        for i, j in T.grid(m, n):
            C_2 = T.Buffer((C_1.strides[0] * m,), data=C_1.data, buffer_type="auto")
            A_2 = T.Buffer((A_1.strides[0] * m,), data=A_1.data, buffer_type="auto")
            B_2 = T.Buffer((B_1.strides[0] * m,), data=B_1.data, buffer_type="auto")
            C_2[i * C_1.strides[0] + j * C_1.strides[1]] = A_2[i * A_1.strides[0] + j * A_1.strides[1]] * B_2[i * B_1.strides[0] + j * B_1.strides[1]]
```

## Merge

## Fuse

 `fuse` 方法用于将一组连续的循环合并成一个循环。合并后的循环将包含所有原始循环的迭代空间。

**限制条件:**

1. 循环不能包含任何注解或线程绑定，例如 `@T.pragma` 或 `@T.thread_binding`
2. 循环必须是连续的，也就是说，每个循环的父循环必须是前一个循环。
3. 循环的起始值必须为 0
4. 每个循环的域不能依赖于其他要合并的循环。

**参数:**

* `loops`: 一个循环列表，表示要合并的循环。
* `preserve_unit_iters`: 一个布尔值，表示是否保留单位迭代的循环。默认值为 `True`，表示保留单位迭代的循环。

**返回值:**

* `fused_loop`: 一个新的循环对象，表示合并后的循环。

以 `B[i, j]=A[i, j]*2`为例，`fuse` 前对应的TensorIR

```python
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j] * 2, name="B")

func = te.create_prim_func([A, B])
func = func.with_attr("global_symbol", "main")
ir_module = IRModule({"main": func})
ir_module.show()

#----------TensorIR Before Fuse--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        B = T.match_buffer(var_B, (m, n))
        # with T.block("root"):
        for i, j in T.grid(m, n):
            with T.block("B"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i, v_j])
                T.writes(B[v_i, v_j])
                B[v_i, v_j] = A[v_i, v_j] * T.float32(2.0)
```

`fuse` 后对应的TensorIR如下

```python
sch = tvm.tir.Schedule(ir_module)
block_B = sch.get_block("B")
i, j= sch.get_loops(block_B)
sch.fuse(i, j)
sch.mod.show()

#----------TensorIR After Fuse--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        B = T.match_buffer(var_B, (m, n))
        # with T.block("root"):
        for i_j_fused in range(m * n):
            with T.block("B"):
                v_i = T.axis.spatial(m, i_j_fused % (n * m) // n)
                v_j = T.axis.spatial(n, i_j_fused % n)
                T.reads(A[v_i, v_j])
                T.writes(B[v_i, v_j])
                B[v_i, v_j] = A[v_i, v_j] * T.float32(2.0)
```

## Split

`split` 方法将一个循环拆分成多个连续的循环，每个循环的迭代次数由 `factors` 参数指定。

**限制条件:**

1. 要拆分的循环不能有任何注解 (annotation) 或线程绑定 (thread binding).
2. 要拆分的循环必须从 0 开始迭代。
3. 在 `factors` 列表中，最多只能有一个元素为 `None`，表示该元素的迭代次数将自动推断。

**参数:**

* **`loop`:** 要拆分的循环对象。
* **`factors`:** 一个列表，表示拆分后的每个循环的迭代次数。列表中的元素可以是整数、表达式或 `None`。如果列表中包含 `None`，则该元素的迭代次数将自动推断。
* **`preserve_unit_iters`:** 一个布尔值，表示是否保留单位迭代器。如果设置为 `True`，则会保留单位迭代器，否则会将单位迭代器合并到其他循环中。
* **`disable_predication`:** 一个布尔值，表示是否禁用谓词 (predicate). 如果设置为 `True`，则不会创建谓词来保护循环。

以 `B[i]=A[i]*2`为例，`split`前对应的TensorIR

```python
A = te.placeholder((m, ), name="A")
B = te.compute((m, ), lambda i: A[i] * 2, name="B")

s = te.create_schedule(B.op)
func = te.create_prim_func([A, B])
func = func.with_attr("global_symbol", "main")
ir_module = IRModule({"main": func})
ir_module.show()

#----------TensorIR Before Split--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m = T.int32()
        A = T.match_buffer(var_A, (m,))
        B = T.match_buffer(var_B, (m,))
        # with T.block("root"):
        for i in range(m):
            with T.block("B"):
                v_i = T.axis.spatial(m, i)
                T.reads(A[v_i])
                T.writes(B[v_i])
                B[v_i] = A[v_i] * T.float32(2.0)
```

`split` 后对应的TensorIR如下

```python
sch = tvm.tir.Schedule(ir_module)
block_b = sch.get_block("B")
i, = sch.get_loops(block_b)
sch.split(i, factors=[None, 32])
sch.mod.show()

#----------TensorIR After Split--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m = T.int32()
        A = T.match_buffer(var_A, (m,))
        B = T.match_buffer(var_B, (m,))
        # with T.block("root"):
        for i_0, i_1 in T.grid((m + 31) // 32, 32):
            with T.block("B"):
                v_i = T.axis.spatial(m, i_0 * 32 + i_1)
                T.where(i_0 * 32 + i_1 < m)
                T.reads(A[v_i])
                T.writes(B[v_i])
                B[v_i] = A[v_i] * T.float32(2.0)
```

## Loop Partition

`loop_partition` 方法用于将一个循环分割成多个连续的循环

**限制条件:**

* 循环不能有注解或线程绑定。
* `factors` 列表中最多只能有一个元素为 `None`
* 不支持循环的值未知的情况。

**参数:**

* `loop`: 要分割的循环。
* `factors`: 分割因子列表。
* `preserve_unit_iters`: 是否保留单位迭代的循环，默认值为 `True`。

仍以 `B[i, j]=A[i, j]*2`为例，`loop_partition`前对应的TensorIR

```python
m = 128
n = 128
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j] * 2, name="B")

func = te.create_prim_func([A, B])
func = func.with_attr("global_symbol", "main")
ir_module = IRModule({"main": func})
ir_module.show()

#----------TensorIR Before Loop Partition--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i in range(128):
            with T.block("B"):
                v_i = T.axis.spatial(128, i)
                T.reads(A[v_i])
                T.writes(B[v_i])
                B[v_i] = A[v_i] * T.float32(2.0)
```

我们指定 `factors=[2,64]`，相当于把整个循环在2和64处分成3份，`loop_partition`后对应的TensorIR如下。在使用 `loop_partition` 后，会创建多个嵌套的块，例如 `root`、`B_i_common` 以及每个分割后的循环对应的块，前两个块中会执行一个空的 `T.reads` 和 `T.writes` 操作。{% label primary @原因未知 %}

```python
sch = tvm.tir.Schedule(ir_module)
block_B = sch.get_block("B")
[i] = sch.get_loops(block_B)  # return a list of LoopRV
sch.loop_partition(i, [2, 64])
sch.mod.show()

#----------TensorIR After Loop Partition--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        with T.block("root"):
            T.reads()
            T.writes()
            with T.block("B_i_common"):
                T.reads()
                T.writes()
                with T.block("B_i0_partition"):
                    T.reads()
                    T.writes()
                    for i0 in range(2):
                        with T.block("B_i0"):
                            v_i = T.axis.spatial(2, i0)
                            T.reads(A[0:2])
                            T.writes(B[0:2])
                            B[v_i] = A[v_i] * T.float32(2.0)
                with T.block("B_i1_partition"):
                    T.reads()
                    T.writes()
                    for i1 in range(2, 66):
                        with T.block("B_i1"):
                            v_i = T.axis.spatial((2, 66), i1)
                            T.reads(A[2:66])
                            T.writes(B[2:66])
                            B[v_i] = A[v_i] * T.float32(2.0)
                with T.block("B_i2_partition"):
                    T.reads()
                    T.writes()
                    for i2 in range(66, 128):
                        with T.block("B_i2"):
                            v_i = T.axis.spatial((66, 128), i2)
                            T.reads(A[66:128])
                            T.writes(B[66:128])
                            B[v_i] = A[v_i] * T.float32(2.0)
```

## Reorder

`reorder` 方法用于重新排列循环的执行顺序。

**限制条件:**

1. 所有循环必须属于同一个循环链，这意味着它们可以按照祖先-后代关系排序，并且它们之间只有单分支循环（即没有 `if` 语句）。
2. 外层循环的范围不能依赖于内层循环。
3. 每个循环嵌套下的块绑定必须是仿射的，并且块变量必须都为数据并行或归约。
4. `ordered_loops` 中不能包含重复的循环。

**参数:**

* `ordered_loops`: 一个或多个循环列表，表示新的循环执行顺序。

`reorder_block_iter_var` 方法的功能与reorder相同，只不过它接收的参数为

* `block`: 待进行变换的BlockRV对象
* `new_order`: 整数列表，代表该block新的迭代顺序

前面章节已给出很多例子，这里不再赘述。

## Parallel

`parallel`方法将一个循环 `loopRV` 标记为并行执行，即循环的迭代可以同时在多个线程或处理器上执行，从而提高计算效率。

**限制条件:**

为了确保并行化操作的正确性和有效性，该函数需要满足以下条件：

1. 循环所在的块必须具有阶段流水线属性。这意味着该块中的计算可以被分解成多个阶段，每个阶段可以独立执行。
2. 循环下的所有块必须是完整块或归约块，并且具有仿射绑定。
3. 对于循环下的每个块，循环只能包含在数据并行块迭代的绑定中。

**参数:**

* `loop`: 要并行化的循环。

以下面的矩阵的 element-wise乘法为例，`parallel`前对应的TensorIR为

```python
A = te.placeholder((m, n), name="A")
B = te.placeholder((m, n), name="B")
C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name="C")

func = te.create_prim_func([A, B, C])
func = func.with_attr("global_symbol", "main")
ir_module = IRModule({"main": func})  
ir_module.show()

#----------TensorIR Before Parallel--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        B = T.match_buffer(var_B, (m, n))
        C = T.match_buffer(var_C, (m, n))
        # with T.block("root"):
        for i, j in T.grid(m, n):
            with T.block("C"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i, v_j], B[v_i, v_j])
                T.writes(C[v_i, v_j])
                C[v_i, v_j] = A[v_i, v_j] * B[v_i, v_j]
```

对外循环进行parallel，可以看到 `T.parallel`取代了之前的 `T.grid`，它会将所有迭代分配到多个线程或处理器上同时执行。

```python
sch = tvm.tir.Schedule(ir_module)
block_c = sch.get_block("C")
i, j = sch.get_loops(block_c)
sch.parallel(i)
sch.mod.show()

#----------TensorIR After Parallel--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        B = T.match_buffer(var_B, (m, n))
        C = T.match_buffer(var_C, (m, n))
        # with T.block("root"):
        for i, j in T.grid(m, n):
            with T.block("C"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i, v_j], B[v_i, v_j])
                T.writes(C[v_i, v_j])
                C[v_i, v_j] = A[v_i, v_j] * B[v_i, v_j]
```

## Vectorize

`vectorize`方法将一个循环 `loop` 标记为向量化执行，这意味着循环的迭代可以被分组为向量，然后在单个指令中执行，从而提高计算效率。

**限制条件:**

1. 循环所在的块必须具有阶段流水线属性，即该块中的计算可以被分解成多个阶段，每个阶段可以独立执行。
2. 循环下的所有块必须是完整块或归约块，并且具有仿射绑定。
3. 对于循环下的每个块，循环只能包含在数据并行块迭代的绑定中。

**参数:**

* `loop`: 要向量化的循环。

仍以 `B[i, j]=A[i, j]*2`为例，`loop_partition`前对应的TensorIR与 [Loop Partition](#Loop-Partition) 中的相同。

Vectorize 是一种重要的优化技术，它利用现代处理器中的 **SIMD** (Single Instruction, Multiple Data)指令，将多个数据同时进行计算，从而提升计算效率。SIMD 指令使用向量寄存器来存储和操作多个数据。向量寄存器的长度通常是 128 位或 256 位，可以存储多个数据。例如，一个 SIMD 指令可以同时对 4 个浮点数进行加法运算。将循环向量化意味着将循环的迭代分组为向量，然后使用 SIMD 指令对这些向量进行操作。`T.vectorized` 在 TVM 中用来标记一个循环已经被向量化了。

```python
sch = tvm.tir.Schedule(ir_module)
block_b = sch.get_block("B")
i, j = sch.get_loops(block_b)
sch.vectorize(j)
sch.mod.show()

#----------TensorIR After Vectorize--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        B = T.match_buffer(var_B, (m, n))
        # with T.block("root"):
        for i in range(m):
            for j in T.vectorized(n):
                with T.block("B"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(A[v_i, v_j])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = A[v_i, v_j] * T.float32(2.0)
```

## Unroll

`unroll` 函数接收一个 `LoopRV` (循环表示变量) 作为输入，作用是将一个循环展开。它本质上是将循环体复制多次，并将循环计数器替换为具体的数值。有以下几个优点

* 减少循环控制指令的执行次数，从而提高效率。
* 将循环体中的数据访问集中在一起，从而提高数据局部性，进而提高缓存命中率。
* 增加指令级并行性，从而提高程序执行速度。

```python
sch = tvm.tir.Schedule(ir_module)
block_b = sch.get_block("B")
i, j = sch.get_loops(block_b)
sch.unroll(i)
sch.mod.show()

#----------TensorIR After Vectorize--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        B = T.match_buffer(var_B, (m, n))
        # with T.block("root"):
        for i in T.unroll(m):
            for j in range(n):
                with T.block("B"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(A[v_i, v_j])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = A[v_i, v_j] * T.float32(2.0)

```

## (Reverse) Compute at

`compute_at`方法的作用是将一个生产者块（producer block）移动到一个特定循环（loop）的内部，并重新生成由该生产者块引起的循环，以确保生产者块生成的缓冲区区域能够覆盖其消费者块在该循环下所使用的区域。`reverse_compute_at`则是移动消费者块（consumer block）

{% note info %}

* **生产者块（producer block）：** 生成数据（通常是缓冲区）的代码块。
* **消费者块（consumer block）：** 使用生产者块生成的数据的代码块。

{% endnote %}

**限制条件：**

1. `block` 和 `loop` 必须在同一个作用域内。
2. 不能将 `block`移动到它自身所在的循环的祖先循环中。
3. 作用域块必须具有阶段-流水线属性。
4. 作用域块的子树必须满足紧凑数据流条件，即子树中的所有块必须是完整块或归约块。
5. 块不是作用域块的输出块，即块写入的缓冲区在作用域块下分配。
6. 块的所有消费者都在给定的循环下。

我们以 `C[i,j]=A[i,j] * 2 + 1`为例，`compute_at`前对应的TesnsorIR如下

{% note warning %}

我们在创建 `prim_func`时的输入只使用了 `A, C`，否则B就不会是作为中间变量的 `T.alloc_buffer`，调用 `compute_at`会因为违反第五条报错。

{% endnote %}

```python
A = te.placeholder((m, n), name="A")
B = te.compute((m, n), lambda i, j: A[i, j] * 2, name="B")
C = te.compute((m, n), lambda i, j: B[i, j] + 1, name="C")
func = te.create_prim_func([A, C])
fuc = func.with_attr({"global_symbol": "main"})
ir_module = IRModule({"main": func})
ir_module.show()

#----------TensorIR Before Compute_at--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_C: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        C = T.match_buffer(var_C, (m, n))
        # with T.block("root"):
        B = T.alloc_buffer((m, n))
        for i, j in T.grid(m, n):
            with T.block("B"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i, v_j])
                T.writes(B[v_i, v_j])
                B[v_i, v_j] = A[v_i, v_j] * T.float32(2.0)
        for i, j in T.grid(m, n):
            with T.block("C"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(B[v_i, v_j])
                T.writes(C[v_i, v_j])
                C[v_i, v_j] = B[v_i, v_j] + T.float32(1.0)
```

在调用 `compute_at`之后块B的计算被移动到块C的循环i之下，相当于调用 `reverse_compute_at`将块C的计算移动到块B的循环i之下，对应的TesnorIR如下

```python
sch = tvm.tir.Schedule(ir_module)
block = sch.get_block("B")
loop, _ = sch.get_loops(sch.get_block("C"))
sch.compute_at(block, loop, preserve_unit_loops=False)
''' same way
block = sch.get_block("C")
loop, _ = sch.get_loops(sch.get_block("B"))
sch.reverse_compute_at(block, loop, preserve_unit_loops=False)
'''
sch.mod.show()

#----------TensorIR After Compute_at--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_C: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        C = T.match_buffer(var_C, (m, n))
        # with T.block("root"):
        B = T.alloc_buffer((m, n))
        for i in range(m):
            for ax0 in range(n):
                with T.block("B"):
                    v_i, v_j = T.axis.remap("SS", [i, ax0])
                    T.reads(A[v_i, v_j])
                    T.writes(B[v_i, v_j])
                    B[v_i, v_j] = A[v_i, v_j] * T.float32(2.0)
            for j in range(n):
                with T.block("C"):
                    v_i, v_j = T.axis.remap("SS", [i, j])
                    T.reads(B[v_i, v_j])
                    T.writes(C[v_i, v_j])
                    C[v_i, v_j] = B[v_i, v_j] + T.float32(1.0)
```

## (Reverse) Compute Inline

`compute_inline` 方法用于将一个块（block）内联到其消费者（consumer）中。简单来说就是将一个块的计算逻辑直接嵌入到使用它结果的块中，从而消除中间块，简化计算流程。`reverse_compute_inline`则是用于将一个块（block）内联到其生产者（producer）中。

**限制条件：**

1. 要内联的块必须是一个完整的非根块（`root` 块），并且它必须只产生一个缓冲区。
2. 要内联的块不能是其作用域内的唯一叶节点。
3. 要内联的块的代码体必须是一个缓冲区存储语句，例如 `A[i, j, k, ...] = ...`。该语句的左侧索引必须是不同的原子变量，并且语句中不能包含其他变量。

以[上一节](#Reverse-Compute-at)的 `C[i,j]=A[i,j] * 2 + 1`为例，对应的TensorIR已给出。在执行Compute_inline之后块B的计算逻辑直接嵌入到块C中。

```python
sch = tvm.tir.Schedule(ir_module)
block = sch.get_block("B") # same: sch.reverse_compute_inline(sch.get_block("C"))
sch.compute_inline(block)
sch.mod.show()

#----------TensorIR After Compute_inline--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_C: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, n = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, n))
        C = T.match_buffer(var_C, (m, n))
        # with T.block("root"):
        for i, j in T.grid(m, n):
            with T.block("C"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A[v_i, v_j])
                T.writes(C[v_i, v_j])
                C[v_i, v_j] = A[v_i, v_j] * T.float32(2.0) + T.float32(1.0)

```

## Decompose Reduction

`decompose_reduction` 函数用于将一个归约块（reduction block）分解成两个独立的块初始化块（init block）和更新块（update block）

{% note info %}

* **初始化块（init block）：** 从归约块的初始化语句（init statement）转换而来。
* **更新块（update block）：** 原始的归约块，但去掉了初始化语句。

{% endnote %}

**限制条件：**

1. 要分解的块必须是一个归约块。
2. 指定的循环必须是归约块的祖先循环。
3. 指定的循环不能低于与归约块变量相关的所有循环。

以矩阵乘法 `C = A @ B`为例，`decompose_reduction`前的TensorIR为

```python
l = te.var("l")
A = te.placeholder((m, l), name="A")
B = te.placeholder((l, n), name="B")
k = te.reduce_axis((0, l), name="l")
C = te.compute((m, n), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="C")

#----------TensorIR Before Decompose Reduction--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, l = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, l))
        n = T.int32()
        B = T.match_buffer(var_B, (l, n))
        C = T.match_buffer(var_C, (m, n))
        # with T.block("root"):
        for i, j, l_1 in T.grid(m, n, l):
            with T.block("C"):
                v_i, v_j, v_l = T.axis.remap("SSR", [i, j, l_1])
                T.reads(A[v_i, v_l], B[v_l, v_j])
                T.writes(C[v_i, v_j])
                with T.init():
                    C[v_i, v_j] = T.float32(0.0)
                C[v_i, v_j] = C[v_i, v_j] + A[v_i, v_l] * B[v_l, v_j]
```

调用 `decompose_reduction` 方法后将块 `C` 分解成一个初始化块和一个更新块，并将初始化块插入到 `i` 循环之前，对应的TensorIR如下

```python
sch = tvm.tir.Schedule(ir_module)
block_c = sch.get_block("C")
i, j, k = sch.get_loops(block_c)
sch.decompose_reduction(block_c, i)
sch.mod.show()

#----------TensorIR After Decompose Reduction--------------
@I.ir_module
class Module:
    @T.prim_func
    def main(var_A: T.handle, var_B: T.handle, var_C: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        m, l = T.int32(), T.int32()
        A = T.match_buffer(var_A, (m, l))
        n = T.int32()
        B = T.match_buffer(var_B, (l, n))
        C = T.match_buffer(var_C, (m, n))
        # with T.block("root"):
        for i_init, j_init in T.grid(m, n):
            with T.block("C_init"):
                v_i, v_j = T.axis.remap("SS", [i_init, j_init])
                T.reads()
                T.writes(C[v_i, v_j])
                C[v_i, v_j] = T.float32(0.0)
        for i, j, l_1 in T.grid(m, n, l):
            with T.block("C_update"):
                v_i, v_j, v_l = T.axis.remap("SSR", [i, j, l_1])
                T.reads(C[v_i, v_j], A[v_i, v_l], B[v_l, v_j])
                T.writes(C[v_i, v_j])
```
