---
title: TVM Learning (1)-Tensor Program Abstraction in Action
date: 2024/8/15 15:07:12
categories: TVM
tags: TVM learning
excerpt: Personal notebook 1.
mathjax: true
katex: true
---
My notebook of MLC: [https://mlc.ai/summer22-zh](https://mlc.ai/summer22-zh)

# Constructing Tensor Program by TVMScript

在机器学习编译 (Machine Learning Compilation) 中，**Tensor Program** 指的是一种表示机器学习模型计算过程的程序，它以张量 (Tensor) 为基本数据单元，并使用张量操作来描述模型的计算步骤。

## Vector-Add Example

下面这段代码使用 TVM 的 `script` 模块定义了一个名为 `MyModule` 的模块，其中包含一个名为 `main` 的计算函数。

该函数实现了简单的向量加法 (vector add) 操作, 两个输入向量 `A` 和 `B` 相加，并将结果存储到输出向量 `C` 中。

```python
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np

@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer[128, "float32"], 
             B: T.Buffer[128, "float32"], 
             C: T.Buffer[128, "float32"]):
        # extra annotations for the function
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i in range(128):
            with T.block("C"):
                # declare a data parallel iterator on spatial domain
                vi = T.axis.spatial(128, i)
                C[vi] = A[vi] + B[vi]
```

**1. 模块定义:**

```python
@tvm.script.ir_module
class MyModule:
# ...
```

* `@tvm.script.ir_module`: 用于将 `MyModule` 类定义为一个 TVM 的 `IRModule` 对象。`IRModule` 是 TVM 中用于表示计算图 (Computation Graph) 的标准数据结构。
* `class MyModule:`: 定义一个名为 `MyModule` 的类，该类将包含计算函数。

{% fold info @Decorator %}
在 Python 中，装饰器 (Decorator) 是一种特殊的函数，它可以用来修改其他函数的行为，而无需直接修改被装饰的函数代码。

```python
def decorator_function(func):
    def wrapper(*args, **kwargs):
        # 在调用被装饰的函数之前执行的操作
        result = func(*args, **kwargs)
        # 在调用被装饰的函数之后执行的操作
        return result
    return wrapper

@decorator_function
def my_function(x, y):
    # 被装饰的函数
    return x + y
```

* `decorator_function`: 装饰器函数，它接收被装饰的函数作为参数，并返回一个包装函数。
* `wrapper`: 包装函数，它在调用被装饰的函数之前和之后执行一些操作。
* `@decorator_function`: 装饰器语法，将 `decorator_function` 应用到 `my_function` 上。

**装饰器的工作原理:**

1. 当 Python 遇到 `@decorator_function` 语法时，它会将 `my_function` 作为参数传递给 `decorator_function`。
2. `decorator_function` 执行，并返回一个包装函数 `wrapper`。
3. `wrapper` 函数将替换 `my_function` 的原始定义。
4. 当调用 `my_function` 时，实际上是在调用 `wrapper` 函数。

{% endfold %}

**2. 计算函数定义:**

```python
    @T.prim_func
    def main(A: T.Buffer[128, "float32"], 
            B: T.Buffer[128, "float32"], 
            C: T.Buffer[128, "float32"]):
# ...
```

* `@T.prim_func`: 这是一个装饰器，用于将 `main` 函数定义为一个 TVM 的 `prim_func` 对象。`prim_func` 是 TVM 中用于表示底层计算函数的标准数据结构。
* `def main(...)`: 定义一个名为 `main` 的函数，该函数接受三个参数：
  * `A`: 一个长度为 128 的 `float32` 类型 Buffer，表示第一个输入向量。
  * `B`: 一个长度为 128 的 `float32` 类型 Buffer，表示第二个输入向量。
  * `C`: 一个长度为 128 的 `float32` 类型 Buffer，用于存储计算结果。

**3. 函数属性:**

```python
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
```

* `T.func_attr({"global_symbol": "main", "tir.noalias": True})`： 设置函数的属性。
  * `global_symbol`: 设置函数的全局符号名称为 `main`。
  * `tir.noalias`: 设置函数的别名属性为 `True`，表示函数不会修改输入缓冲区。

**4. 计算循环:**

```python
for i inrange(128):
    with T.block("C"):
    # ...
```

`T.block` 将计算图分解成多个独立的计算块，每个块对应一个特定的计算任务，可以包含多个迭代器，这些迭代器共同定义了计算块的计算范围。

* `for i in range(128)`: 定义一个循环，迭代 128 次，用于处理每个向量元素。
* `with T.block("C")`: 定义一个名为 `C` 的计算块，该块包含循环的计算逻辑。

**5. 迭代器定义:**

```python
vi = T.axis.spatial(128, i)
```

* `vi = T.axis.spatial(128, i)`: 定义一个名为 `vi` 的空间迭代器，它遍历 128 个元素，每个元素的索引由 `i` 确定。

一般来说，空间迭代器的访问顺序对最后结果不产生影响。

**6. 计算操作:**

```python
C[vi] = A[vi] + B[vi]
```

* `C[vi] = A[vi] + B[vi]`： 将 `A` 和 `B` 中对应元素相加，并将结果存储到 `C` 中。

我们可以通过 `MyModule.show()` 来显示构建的IRModule.

```python
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[128, "float32"], B: T.Buffer[128, "float32"], C: T.Buffer[128, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i in T.serial(128):
            with T.block("C"):
                vi = T.axis.spatial(128, i)
                T.reads(A[vi], B[vi])
                T.writes(C[vi])
                C[vi] = A[vi] + B[vi]
```

## Build and Run

我们可以通过 `tvm.build`函数将一个IRModule转变成可以运行的函数，通过定义的函数名可以获取想要的函数。然后我们可以定义三个 `NDArray` 数组来调用函数。

```python
rt_mod = tvm.build(MyModule, target="llvm")
func = rt_mod["main"]
a = tvm.nd.array(np.arange(128, dtype="float32"))
b = tvm.nd.array(np.ones(128, dtype="float32")) 
c = tvm.nd.empty((128,), dtype="float32") 
func(a, b, c)
```

**`tvm.build` 函数的参数:**

* `func`: 要编译的计算图，可以是 `tvm.script.ir_module` 对象、`tvm.relay.Function` 对象或其他支持的计算图类型。
* `target`: 目标平台，例如，`llvm -mcpu=core-avx2`、`cuda`、`opencl` 等。
* `name`: 编译后的模块名称。

## Transform the Tensor Program

在 TVM 中，`tvm.tir.Schedule` 是一个用于对计算图进行手动优化的工具。它允许对计算图中的循环、块和操作进行重排序、融合、并行化等操作，以提高计算效率。

下面这段代码做了以下优化：

* **循环切分:** 将循环 `i` 切分成三个循环，可以更好地利用内存局部性，例如，将 `i_1` 和 `i_2` 的大小设置为 4，可以将数据加载到缓存中，减少内存访问次数。
* **循环重排序:** 按照 `i_0`、`i_2` 和 `i_1` 这个顺序执行。
* **并行化:** 将 `i_0` 并行化，可以利用多核 CPU 或 GPU 的计算能力，提高计算速度

```python
sch = tvm.tir.Schedule(MyModule)
# Get block by its name
block_c = sch.get_block("C")
# Get loops surronding the block
(i,) = sch.get_loops(block_c)
# Tile the loop nesting.
i_0, i_1, i_2 = sch.split(i, factors=[None, 4, 4])
# Reorder the loop.
sch.reorder(i_0, i_2, i_1)
sch.parallel(i_0)
sch.mod.show()
```

优化后的计算图如下

```python
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[128, "float32"], B: T.Buffer[128, "float32"], C: T.Buffer[128, "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i_0 in T.parallel(8):
            for i_2, i_1 in T.grid(4, 4):
                with T.block("C"):
                    vi = T.axis.spatial(128, i_0 * 16 + i_1 * 4 + i_2)
                    T.reads(A[vi], B[vi])
                    T.writes(C[vi])
                    C[vi] = A[vi] + B[vi]
```

# Constructing Tensor Program by Tensor Expression

Tensor Expression 指的是一种用于描述张量计算的数学表达式。

## Construct Vector-Add by TE

我们可以通过以下方式来创建和 [上一节](#Vector-Add-Example) 一样的IRModule.

```python
# namespace for tensor expression utility
from tvm import te

# declare the computation using the expression API
A = te.placeholder((128, ), name="A")
B = te.placeholder((128, ), name="B")
C = te.compute((128,), lambda i: A[i] + B[i], name="C")

# create a function with the specified list of arguments. 
func = te.create_prim_func([A, B, C])
# mark that the function name is main
func = func.with_attr("global_symbol", "main")
ir_mod_from_te = IRModule({"main": func})

ir_mod_from_te.show()
```

1. **定义张量:**

   ```python
   A = te.placeholder((128,), name="A")
   B = te.placeholder((128,), name="B")
   ```

   这两行代码定义了两个名为 `A` 和 `B` 的张量，它们都是一维张量，大小为 128。`te.placeholder` 函数用于创建占位符张量，它代表输入数据。
2. **定义计算:**

   ```python
   C = te.compute((128,), lambda i: A[i] + B[i], name="C")
   ```

   这行代码定义了一个名为 `C` 的张量，它表示 `A` 和 `B` 的元素相加的结果。`te.compute` 函数用于定义张量计算，它接受两个参数：

   * 第一个参数 `shape`是张量的形状，这里为 `(128,)`。
   * 第二个参 `fcompute`数是一个 lambda 函数，它定义了每个元素的计算方式，这里为 `A[i] + B[i]`，表示 `C` 的第 `i` 个元素等于 `A` 的第 `i` 个元素加上 `B` 的第 `i` 个元素。
3. **创建 PrimFunc:**

   ```python
   func = te.create_prim_func([A, B, C])
   ```

   这行代码使用 `te.create_prim_func` 函数创建了一个 PrimFunc 对象，它代表一个 TVM 的基本计算函数。`te.create_prim_func` 函数接受一个参数，即函数的输入参数列表，这里为 `[A, B, C]`
4. **设置函数名称:**

   ```python
   func = func.with_attr("global_symbol", "main")
   ```

   这行代码将函数的名称设置为 `main`，`with_attr` 函数用于设置函数的属性。
5. **创建 IRModule:**

   ```python
   ir_mod_from_te = IRModule({"main": func})
   ```

   这行代码创建了一个 IRModule 对象，它包含了 `func` 函数，并将该函数存储在 IRModule 的 `main` 字段中。

## Transforming a matrix multiplication program

下面代码展示了两个{% katex %}1024 \times 1024 {% endkatex %}矩阵相乘的IRModule创建流程。

```python
M = 1024
K = 1024
N = 1024

# The default tensor type in tvm
dtype = "float32"

target = "llvm"
dev = tvm.device(target, 0)

# Algorithm
k = te.reduce_axis((0, K), "k")
A = te.placeholder((M, K), name="A")
B = te.placeholder((K, N), name="B")
C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")

# Default schedule
func = te.create_prim_func([A, B, C])
func = func.with_attr("global_symbol", "main")
ir_module = IRModule({"main": func})
ir_module.show()


func = tvm.build(ir_module, target="llvm")  # The module for CPU backends.

a = tvm.nd.array(np.random.rand(M, K).astype(dtype), dev)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), dev)
c = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)
func(a, b, c)

# Create evaluation function
evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("Baseline: %f" % evaluator(a, b, c).mean)
```

`time_evaluator` 是 IRModule 用于评估计算图执行时间的方法。它可以帮助测量不同硬件平台上不同计算图的性能，并进行优化。

{% fold info @time_evaluator %}

```python
IRModule.time_evaluator(func, args, number=1, repeat=1, min_repeat_ms=0, f_type=0)
```

**参数解释:**

* `func`: 要评估的计算图函数。
* `args`: 计算图函数的输入参数，可以是张量或其他数据结构。
* `number`: 每次运行计算图的次数，默认值为 1。
* `repeat`: 重复运行计算图的次数，默认值为 1。
* `min_repeat_ms`: 最小运行时间，单位为毫秒。如果计算图运行时间小于 `min_repeat_ms`，则会继续运行直到达到 `min_repeat_ms`。默认值为 0。
* `f_type`: 运行模式，可以是 0（默认值）、1 或 2。
  * 0：正常运行模式。
  * 1：仅执行编译，不运行计算图。
  * 2：仅执行运行，不编译计算图。

**`func.time_evaluator` 的返回值:**

`func.time_evaluator` 返回一个函数，该函数可以用来执行评估并返回一个包含性能指标的字典。

**性能指标:**

* `mean`: 平均运行时间，单位为毫秒。
* `median`: 中位数运行时间，单位为毫秒。
* `min`: 最小运行时间，单位为毫秒。
* `max`: 最大运行时间，单位为毫秒。
* `std`: 标准差，单位为毫秒。

{% endfold %}
代码的大部分流程相同，我们来看计算部分。

1. **定义约简轴 (Reduce axis):**

   ```python
   k = te.reduce_axis((0, K), "k")
   ```

   这行代码定义了一个名为 `k` 的约简轴，表示在矩阵乘法操作中进行求和的维度，范围为 `(0, K)`
2. **定义输入矩阵 (Placeholders):**

   ```
   A = te.placeholder((M, K), name="A")
   B = te.placeholder((K, N), name="B")
   ```

   这两行代码定义了两个名为 `A` 和 `B` 的输入矩阵，它们分别代表矩阵乘法的两个输入矩阵。`A` 的形状为 `(M, K)`，`B` 的形状为 `(K, N)`
3. **定义输出矩阵 (Compute):**

   ```
   C = te.compute((M, N), lambda m, n: te.sum(A[m, k] * B[k, n], axis=k), name="C")
   ```

   这行代码定义了一个名为 `C` 的输出矩阵，它表示矩阵乘法的结果。`C` 的形状为 `(M, N)`，采用 `te.sum`计算结果。


{% fold info @te.sum %}

```python
te.sum(expr, axis=None, keepdims=False, where=None)
```

**参数解释:**

* `expr`: 要进行求和的表达式，可以是张量、标量或其他表达式。
* `axis`: 要进行求和的轴，可以是整数、元组或列表。如果 `axis` 为 `None`，则对所有轴进行求和。
* `keepdims`: 布尔值，表示是否保留求和后的维度。如果为 `True`，则保留求和后的维度，并将其大小设置为 1。如果为 `False`，则删除求和后的维度。
* `where`: 布尔值张量，表示要进行求和的元素。如果 `where` 为 `None`，则对所有元素进行求和。

{% endfold %}


创建的IRModule如下所示。

```python
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"], B: T.Buffer[(1024, 1024), "float32"], C: T.Buffer[(1024, 1024), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0, i1, i2 in T.grid(1024, 1024, 1024):
            with T.block("C"):
                m, n, k = T.axis.remap("SSR", [i0, i1, i2])
                T.reads(A[m, k], B[k, n])
                T.writes(C[m, n])
                with T.init():
                    C[m, n] = T.float32(0)
                C[m, n] = C[m, n] + A[m, k] * B[k, n]
```

我们可以将循环拆分成外层循环和内层循环可以提高数据局部性。内层循环访问的数据更接近，可以有效利用缓存。下面代码的 `block_size` 参数控制了内层循环的大小，选择合适的块大小可以最大程度地利用缓存。

```python
sch = tvm.tir.Schedule(ir_module)
block_c = sch.get_block("C")
# Get loops surronding the block
(y, x, k) = sch.get_loops(block_c)
block_size = 32
yo, yi = sch.split(y, [None, block_size])
xo, xi = sch.split(x, [None, block_size])

sch.reorder(yo, xo, k, yi, xi)
sch.mod.show()

func = tvm.build(sch.mod, target="llvm")  # The module for CPU backends.

c = tvm.nd.array(np.zeros((M, N), dtype=dtype), dev)
func(a, b, c)

evaluator = func.time_evaluator(func.entry_name, dev, number=1)
print("after transformation: %f" % evaluator(a, b, c).mean)
```

创建的IRModule如下所示。实际中我们会测试很多不同 `block_size`对应的执行时间来选择最合适的。

```python
@tvm.script.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer[(1024, 1024), "float32"], B: T.Buffer[(1024, 1024), "float32"], C: T.Buffer[(1024, 1024), "float32"]) -> None:
        # function attr dict
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        # body
        # with T.block("root")
        for i0_0, i1_0, i2, i0_1, i1_1 in T.grid(32, 32, 1024, 32, 32):
            with T.block("C"):
                m = T.axis.spatial(1024, i0_0 * 32 + i0_1)
                n = T.axis.spatial(1024, i1_0 * 32 + i1_1)
                k = T.axis.reduce(1024, i2)
                T.reads(A[m, k], B[k, n])
                T.writes(C[m, n])
                with T.init():
                    C[m, n] = T.float32(0)
                C[m, n] = C[m, n] + A[m, k] * B[k, n]
```
