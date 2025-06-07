---
title: TVM Learning (5)-Automatic Program Optimization
date: 2024-08-19T21:23:00+08:00
lastmod: 2024-08-19T21:23:00+08:00
draft: false
author: ["WITHER"]
keywords: 
    - TVM
categories:
    - TVM Learning
tags:
    - Autotuning
description: Personal notebook 4.
summary: Personal notebook 4.
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# Transform a Primitive Tensor Function

之前已经讲过如何通过 `tir.Schedule`对T.prim_func进行变换，仍以矩阵乘法为例

```python {linenos=true}
@tvm.script.ir_module
class MyModule:
    @T.prim_func
    def main(A: T.Buffer((128, 128), "float32"), # type: ignore
             B: T.Buffer((128, 128), "float32"), # type: ignore
             C: T.Buffer((128, 128), "float32")): # type: ignore
        T.func_attr({"global_symbol": "main", "tir.noalias": True})
        for i, j, k in T.grid(128, 128, 128):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = 0.0
                C[vi, vj] += A[vi, vk] * B[vk, vj]
```

对其进行 `split,` `reorder`和 `decompose_reduction`变换得到的TensorIR如下。

通过以上变换后，矩阵乘法的执行时间减少是由于：

1. **循环拆分 (`sch.split`)** ：

* 将 `j`循环拆分成了两个循环：`j_0`和 `j_1`，其中 `j_1`的因子为4（内层循环）。
* 提高数据的局部性，因为较小的数据块会在更短的时间内被频繁访问，从而更好地利用缓存。

1. **循环重排 (`sch.reorder`)** ：

* 将循环的顺序调整为 `i, j_0, k, j_1`，意味着外层循环先遍历 `i`和 `j_0`，内层循环再遍历 `k`和 `j_1`。
* 优先考虑了数据在寄存器或缓存中的重用，尤其是在内层循环操作期间 `A`矩阵中的元素。

1. **分解归约 (`sch.decompose_reduction`)** ：

* 将对 `k`的归约操作分解为初始化阶段和更新阶段，有助于将计算的两个阶段（即设置初始值和实际归约）分开。
* 提高并行化的机会，并且允许更好地利用向量化指令或其他硬件优化。

```python {linenos=true}
def schedule_mm(sch: tvm.tir.Schedule, jfactor=4):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_0, j_1 = sch.split(loop=j, factors=[None, jfactor])
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch

sch = tvm.tir.Schedule(MyModule)
sch = schedule_mm(sch)
sch.mod.show()

#-----------------------------------
@I.ir_module
class Module:
    @T.prim_func
    def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), C: T.Buffer((128, 128), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j_0 in T.grid(128, 32):
            for j_1_init in range(4):
                with T.block("C_init"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 4 + j_1_init)
                    T.reads()
                    T.writes(C[vi, vj])
                    C[vi, vj] = T.float32(0.0)
            for k, j_1 in T.grid(128, 4):
                with T.block("C_update"):
                    vi = T.axis.spatial(128, i)
                    vj = T.axis.spatial(128, j_0 * 4 + j_1)
                    vk = T.axis.reduce(128, k)
                    T.reads(C[vi, vj], A[vi, vk], B[vk, vj])
                    T.writes(C[vi, vj])
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
```

我们可以比较变换前后的计算用时

```python {linenos=true}
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
c_mm = a_np @ b_np 

a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128, 128), dtype="float32")
# Before transformation
lib = tvm.build(MyModule, target= "llvm")
f_timer_before = lib.time_evaluator("main", tvm.cpu())
print("Time cost of MyModule: %.3f ms" % (f_timer_before(a_nd, b_nd, c_nd).mean * 1000))
#Time cost of MyModule: 1.365 ms
# After transformation
lib = tvm.build(sch.mod, target="llvm")
f_timer_after = lib.time_evaluator("main", tvm.cpu())
print("Time cost of MyModule=>schedule_mm: %.3f ms" % (f_timer_after(a_nd, b_nd, c_nd).mean * 1000))
# Time cost of MyModule=>schedule_mm: 1.041 ms
```

# Transformation Trace

除了 `sch.mod`字段，`tir.Schedule`还提供了一个跟踪字段 `sch.trace`，用于显示变换IRModule的步骤。

```python {linenos=true}
print(sch.trace)
#-------------------------------------------
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  l1, l2, l3 = sch.get_loops(block=b0)
  l4, l5 = sch.split(loop=l2, factors=[None, 4], preserve_unit_iters=True, disable_predication=False)
  sch.reorder(l1, l4, l3, l5)
  b6 = sch.decompose_reduction(block=b0, loop=l3)
```

# Stochastic Schedule Transformation

在之前的变换中，我们都是指定这些函数的输入参数。实际情况下，我们需要引入随机性，根据不同变换的输入参数得出的执行时间来选择性能最好的一个。

`sample_perfect_tile`函数可以计算任务中的特定循环采样最优的切分策略。

**输入参数：**

* `loop`：要切分的循环。
* `n`：要切分成几份。
* `max_innermost_factor`：允许在最内层循环中采样的最大切分大小。此参数有助于控制平铺的粒度。
* `decision`：一个可选的整数列表，表示预先确定的切分决策。如果提供，函数将使用此决策而不是采样。

下面函数 `stochastic_schedule_mm`和 `schedule_mm`唯一的区别是指定 `j_factors`采用的是随机的策略。

```python {linenos=true}
def stochastic_schedule_mm(sch: tvm.tir.Schedule):
    block_C = sch.get_block("C", "main")
    i, j, k = sch.get_loops(block=block_C)
    j_factors = sch.sample_perfect_tile(loop=j, n=2)  # tvm.tir.expr.Var
    j_0, j_1 = sch.split(loop=j, factors=j_factors)
    sch.reorder(i, j_0, k, j_1)
    sch.decompose_reduction(block_C, k)
    return sch
```

可以发现，它是对原来的确定性变换的泛化版本，只是多了两个元素：

- 来自 `sample_perfect_tile` 的随机变量，以及我们在示例中没有涉及的其他采样操作。
- 根据随机变量采取行动的 `schedule`操作。

`j_factors` 中的元素不是整数。相它们是**符号变量**，指的是正在采样的随机变量。我们可以将这些变量传递给转换 API，以指定factors. 调用 `stochastic_schedule_mm`后的trace如下

```python {linenos=true}
sch = tvm.tir.Schedule(MyModule)
sch = stochastic_schedule_mm(sch)
print(sch.trace)

#------------------------------------------------------
def apply_trace(sch: tir.Schedule) -> None:
  b0 = sch.get_block(name="C", func_name="main")
  l1, l2, l3 = sch.get_loops(block=b0)
  v4, v5 = sch.sample_perfect_tile(loop=l2, n=2, max_innermost_factor=16, decision=[64, 2])
  l6, l7 = sch.split(loop=l2, factors=[v4, v5], preserve_unit_iters=True, disable_predication=False)
  sch.reorder(l1, l6, l3, l7)
  b8 = sch.decompose_reduction(block=b0, loop=l3)
```

# Search Over Stochastic Transformations

`stochastic_schedule_mm`实际上会根据每个采样步骤的实际决定，创建一个**程序的搜索空间**。

![Transformation Search Space](https://mlc.ai/zh/_images/auto_prog_optim_transformation_search.png "Transformation Search Space")

我们需要一种搜索算法能找到性能最好的变换。下面的函数使用最直接的搜索算法--随机搜索。它尝试重复运行 `stochastic_schedule_mm`，得到一个转换后的IR module，运行benchmark，然后将性能最好的IR module记录下来。

```python {linenos=true}
def random_search(mod: tvm.IRModule, num_trails=5):
    best_result = None
    best_sch = False
  
    for i in range(num_trails):
        sch = stochastic_schedule_mm(tvm.tir.Schedule(mod))
        lib = tvm.build(sch.mod, target="llvm")
        f_timer_after = lib.time_evaluator("main", tvm.cpu())
        result = f_timer_after(a_nd, b_nd, c_nd).mean
  
        print("=====Attempt %d, time-cost: %.3f ms====" % (i, result * 1000))
        print(sch.trace)

        # book keep the best result so far
        if best_result is None or result < best_result:
            best_result = result
            best_sch = sch  
  
    return best_sch
```

实际情况下会使用更高级的算法。还需要提供额外的工具，例如在远程设备上进行基准测试等。TVM 的 `meta_schedule` API 提供了这些功能。

`meta_schedule`是一个命名空间，用于支持在可能的变换空间中进行搜索。

- 跨多个进程的并行基准测试。
- 使用 cost model，避免每次都进行基准测试。
- 在 trace 上进行进化搜索，而不是每次都随机取样。

`tune_tir` API 仍使用随机变换来指定好程序的搜索空间并在搜索空间内找到优化的方案。

```python {linenos=true}
database = ms.tune_tir(
    mod=MyModule,
    target="llvm --num-cores=1",
    max_trials_global=64,
    num_trials_per_iter=64,
    space=ms.space_generator.ScheduleFn(stochastic_schedule_mm),
    work_dir="./tune_tmp",
    task_name="main"
)

sch_tuned = ms.tir_integration.compile_tir(database, MyModule, target="llvm --num-cores=1")
print(sch_tuned.trace)
```

{{< details title="clang error on Windows" >}}

不知道为何Windows上运行clang会出错

```bash {linenos=true}
LocalRunner: An exception occurred
Traceback (most recent call last):
  File "D:\Work\Anaconda\envs\tvm-build\lib\site-packages\tvm-0.18.dev0-py3.9-win-amd64.egg\tvm\exec\popen_worker.py", line 87, in main
    result = fn(*args, **kwargs)
  File "D:\Work\Anaconda\envs\tvm-build\lib\site-packages\tvm-0.18.dev0-py3.9-win-amd64.egg\tvm\meta_schedule\runner\local_runner.py", line 148, in _worker_func
    rt_mod = tvm.runtime.load_module(artifact_path)
  File "D:\Work\Anaconda\envs\tvm-build\lib\site-packages\tvm-0.18.dev0-py3.9-win-amd64.egg\tvm\runtime\module.py", line 696, in load_module
    _cc.create_shared(path + ".so", files)
  File "D:\Work\Anaconda\envs\tvm-build\lib\site-packages\tvm-0.18.dev0-py3.9-win-amd64.egg\tvm\contrib\cc.py", line 96, in create_shared
    _windows_compile(output, objects, options, cwd, ccache_env)
  File "D:\Work\Anaconda\envs\tvm-build\lib\site-packages\tvm-0.18.dev0-py3.9-win-amd64.egg\tvm\contrib\cc.py", line 415, in _windows_compile
    raise RuntimeError(msg)
RuntimeError: Compilation error:
clang -O2 -shared -o C:\Users\17725\AppData\Local\Temp\tmp96lbzaxg\tvm_tmp_mod.tar.so C:\Users\17725\AppData\Local\Temp\tmp96lbzaxg\tvm_tmp_mod\lib0.o
ld.lld: error: undefined symbol: _fltused

>>> referenced by C:\Users\17725\AppData\Local\Temp\tmp96lbzaxg\tvm_tmp_mod\lib0.o

clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

{{< /details >}}
