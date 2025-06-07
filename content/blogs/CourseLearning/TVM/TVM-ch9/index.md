---
title: TVM Learning (11)-Add Model Architeture in MLC LLM
date: 2024-08-08T12:00:00+08:00
lastmod: 2024-08-08T15:15:00+08:00
draft: false
author: ["WITHER"]
keywords: 
    - TVM
categories:
    - TVM Learning
tags:
    - Autotuning
description: Add Model Architeture in MLC LLM
summary: Add Model Architeture in MLC LLM
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# IRModule: The key concept in TVM Unity

IRModule 是张量函数的集合，代表我们需要在模型中执行的计算子集。例如，在 MLC-LLM 中，它可以是一个 Transformer 模块。
机器学习编译框架中的 IRModule 就像深度学习框架中的张量，是一切的基础。在整个编译流程中，模型将以 IRModule 的形式导入，然后以 IRModule 到 IRModule 的方式进行转换和优化，然后我们就可以在任何支持的平台上将 IRModule 转化为可运行的模块。IRModule 可以用 python 方式访问，例如，我们可以用 python AST 的形式显示它，以便检查、调整和调试。unity 的主要设计目标之一是实现单一抽象，将所有主要元素封装在同一模块中。这样，我们就能在此基础上进行有机的增量转换。

![TVM Unity.png](https://note.youdao.com/yws/api/personal/file/WEB3a739d336ccf45c1d34addfa952de165?method=download&shareKey=1150d6da998887fe6b987c0e7bbbc777 "TVM Unity.png")

TVMScript 是 IRModule 的 python AST 格式，用于在整套转换过程中检查 IRModules 并与之交互。与 IRModule 的交互都可以使用 TVMScript 在 python 中进行。用户将 TVMScript 解析为 IRModule 内部结构，使用 python API 操作 IRModule，并将 IRModule 打印为 TVMScript 格式。

## TVMScript Examples

用 Pytorch 框架实现矩阵乘法一般调用 `torch.matmul` 或者使用 `@` 算子。

```python
import torch 
a = torch.randn((3, 4))
b = torch.randn((4, 5))
print(torch.matmul(a, b))

'''
tensor([[ 2.5387,  2.2756, -2.2032,  2.5928, -3.6539],
        [ 2.0151,  0.0628, -0.8041, -1.6947,  0.2884],
        [-0.8118, -0.0453,  0.0742, -1.2028,  1.3722]])
'''
```

在 Relax 中可以用 IRModule 实现相同的功能。

```python
from tvm.script import ir as I
from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(A: R.Tensor((3, 4), dtype="float32"), B: R.Tensor((4, 5), dtype="float32")) -> R.Tensor((3, 5), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.matmul(A, B, out_dtype="void")
            R.output(lv)
        return lv
```

通过上述 TVMScript 创建的 IRModule 是一个完全图级别的抽象，只包含一个 R.function (Relax 函数： IRModule 中计算图的表示形式)
上述示例包含 Relax 函数中的两个重要概念：高级 Relax 算子和数据流块。

- Relax 函数包含高级 Relax 算子 `R.matmul`，它描述计算图中的节点，不包含其底层实现的信息。一个高级 Relax 算子可以映射到不同的底层实现，TVM Unity 的编译流程会生成性能良好的实现。
- `R.dataflow()` 是数据流块的一个重要作用域注解。具体来说，在数据流块内，所有操作都必须是 side-effect free. 而在数据流块之外，操作可能包含副作用。

## A more complex TVMScript example: 2-layer MLP

下面我们以一个更复杂的两层 MLP 为例，模型结构如下。

![2-layer MLP](https://note.youdao.com/yws/api/personal/file/WEBa9d8ef25e3caa5e822e9d8768efbafbe?method=download&shareKey=1142a3440c0b476839088b07a5971539 "2-layer MLP")

其对应的 Pytoch 实现如下

```python
class MLP(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MLP, self).__init__(*args, **kwargs)
        self.linear1 = torch.nn.Linear(784, 128)
        self.linear2 = torch.nn.Linear(128, 10)
      
    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x
```

对应的 IRModule 的 TVMScript 表示如下

```python
@I.ir_module
class Module:
    @R.function
    def main(inp_0: R.Tensor((1, 784), dtype="float32"), weight1: R.Tensor((128, 784), dtype="float32"), bias1: R.Tensor((1, 128), dtype="float32"), weight2: R.Tensor((10, 128), dtype="float32"), bias2: R.Tensor((1, 10), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(weight1, axes=None)
            lv_1: R.Tensor((1, 128), dtype="float32") = R.matmul(inp_0, lv, out_dtype="void")
            lv1: R.Tensor((1, 128), dtype="float32") = R.add(lv_1, bias1)
            lv2: R.Tensor((1, 128), dtype="float32") = R.nn.relu(lv1)
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(weight2, axes=None)
            lv3: R.Tensor((1, 10), dtype="float32") = R.matmul(lv2, lv4, out_dtype="void")
            lv4_1: R.Tensor((1, 10), dtype="float32") = R.add(lv3, bias2)
            R.output(lv4_1)
        return lv4_1
```

上述 Relax 函数只包含高级 Relax 算子。在 pytorch 中，`torch.nn.Linear` 计算  $y = xW^T + b$ 在 relax 中，转置由 permute_dims 实现，其次是 矩阵乘法和加法分别由 `R.matmul` 和 `R.add` 实现。

# Compilation Flow in TVM Unity

1. 将模型导入 IRModule. 对于静态模型，我们可以使用 pytorch dynamo 将 pytorch 程序跟踪为 fx 图，然后转换为 IRModule。然而，LLM 通常是动态的，因为序列长度和 kv cache 长度都是可变的。在这种情况下，我们需要直接在 IRModule 中建立模型。第一步可以抽象为 LLM -> IRModule 转换。
2. 优化模型。与传统编译器一样，我们可以在 IRModule 上应用 pass (IRModule 到 IRModule 的变换，改变计算但保留了原始 IRModule 的语义)。在这一步中，我们的目标是加速模型计算。在消费类设备上以适当速度运行 LLM 的大多数关键技术，如量化、算子融合和张量函数调度，都是在这一步实现的。
3. 在设备上部署 IRModule。对于每个 IRM 模块，我们都能将其转化为可运行模块，并在 tvm 运行时支持的任何平台上运行。IRModule 上的每个函数都将成为环境中的本地可运行函数。

以下是 2 层 MLP 模型的编译流程

```python
from tvm import relax
import tvm
from tvm.ir.module import IRModule
mod = MLPModule

def optimize_and_deploy(mod: IRModule):
    # step 2. Optimization
    
    # Use default graph optimization pipeline
    mod = relax.pipeline.get_pipeline()(mod)
    
    # Use default tensor function scheduling
    with tvm.target.Target("cuda"):
        mod  = tvm.tir.transform.DefaultGPUSchedule()(mod)
        
    # Step 3. Deploy to GPU
    ex = relax.build(mod, "cuda")
    vm = relax.VirtualMachine(ex, tvm.cuda())
    
    # test correctness
    import numpy as np
    input_np = np.random.rand(1, 784).astype("float32")
    weight1_np = np.random.rand(128, 784).astype("float32")
    bias1_np = np.random.rand(1, 128).astype("float32")
    weight2_np = np.random.rand(10, 128).astype("float32")
    bias2_np = np.random.rand(1, 10).astype("float32")
    tvm_nd_arrays = [tvm.nd.array(np_array, device=tvm.cuda()) for np_array in [input_np, weight1_np, bias1_np, weight2_np, bias2_np]]
    # call into the runnable function converted from IRModule
    nd_res = vm["main"](*tvm_nd_arrays)
    numpy_res = (input_np @ weight1_np.T + bias1_np) @ weight2_np.T + bias2_np
    np.testing.assert_allclose(numpy_res, nd_res.numpy(), rtol=1e-5)

optimize_and_deploy(mod)  
```

# Build IRModule in Pytorch Style

构建 IRModule 最直接的方法是手动编写 TVMScript。这种方法适用于小型模型，但 LLM 的 IRModule 非常庞大和复杂，手工编写并不现实。TVM Unity 提供了另一个类 nn.Module，可以像 pytorch 模块一样轻松构建 IRModule.
用 Pytorch 手动编写的一个 Linear 层如下

```python
class TorchLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.randn(out_features))
        else:
            bias = None
    
    def forward(self, x):
        return x @ self.weight.T + self.bias
```

在 Relax 中的实现如下

```python
from tvm.relax.testing import nn

class RelaxLinear(nn.Module):
    def __init__(self, in_features, out_features, dtype: str, bias=True) -> None:
        super(RelaxLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((out_features, in_features), dtype, name="linear_weight")
        if bias:
            self.bias = nn.Parameter((1, out_features), dtype, name="linear_bias")
        else:
            self.bias = None
    
    def forward(self, x: relax.Expr) -> relax.Var:
        return nn.emit(relax.op.linear(x, self.weight, self.bias))
```

与 Pytorch 的结构非常相似，只是前向函数实际上并不执行计算。它使用作为输入传递的占位符跟踪算子的计算图。
`nn.emit(relax.op.linear(input, self.weight, self.bias))` 表示在构建的 IRModule 中添加高级 linear 算子。
通过堆叠 1 个线性层、1 个 relu 层和 1 个线性层，就可以构建例子中的 MLP.

```python
class RelaxMLP(nn.Module):
    def __init__(self, in_features, hidden_dims, out_features, dtype="float32") -> None:
        super(RelaxMLP, self).__init__()
        self.linear1 = RelaxLinear(in_features, hidden_dims, dtype)
        self.lienar2 = RelaxLinear(hidden_dims, out_features, dtype)
    
    def forward(self, x: relax.Expr) -> relax.Var:
        hidden = self.linear1(x)
        hidden = nn.emit(relax.op.nn.relu(hidden))
        out = self.lienar2(hidden)
        return out
```

直接调用 nn.Module 的前向函数就可以代替原先在 `with bb.dataflow():` 下的操作，将 `nn.Module` 构建成 IRModule 的步骤如下

```python
def build_relax(mod: nn.Module):   
    # relax.BlockBuilder can construct end-to-end models step by step in an IRModule that starts empty
    bb = relax.BlockBuilder()
    # relax nn.Module
    model = mod(784, 128, 10)
    # create a function called "main" in the IRModule
    with bb.function("main"):
        # define input placeholder to the relax nn.Module
        input = nn.Placeholder((1, 784), dtype="float32", name="input")
        # build dataflow block
        with bb.dataflow():
            # call forward function 
            logits = model(input)
            # The params of the constructed IRModule
            params = [input] + model.parameters()
            # return value of the dataflow block
            gv = bb.emit_output(logits)
        # return value and params of the Relax function
        bb.emit_func_output(gv, params)
    return bb.get()

build_relax(RelaxMLP).show()

#------------------------------
@I.ir_module
class Module:
    @R.function
    def main(input: R.Tensor((1, 784), dtype="float32"), linear_weight: R.Tensor((128, 784), dtype="float32"), linear_bias: R.Tensor((1, 128), dtype="float32"), linear_weight_1: R.Tensor((10, 128), dtype="float32"), linear_bias_1: R.Tensor((1, 10), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(linear_weight, axes=None)
            lv1: R.Tensor((1, 128), dtype="float32") = R.matmul(input, lv, out_dtype="void")
            lv2: R.Tensor((1, 128), dtype="float32") = R.add(lv1, linear_bias)
            lv3: R.Tensor((1, 128), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(linear_weight_1, axes=None)
            lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(lv3, lv4, out_dtype="void")
            lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, linear_bias_1)
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv
```

# Custom Operator Support

在某些情况下，我们要表示的模型包含一些自定义运算符，而这些运算符没有被提供的 Relax 运算符覆盖（如 LLaMA 中的 Rotary Embedding），或者我们要进行底层优化以加速单个内核。下面介绍如何在 IRModule 中编写自定义算子。

##  TensorIR: Low-level tensor function

TVM Unity 在 IRModule TensorIR 中提供了底层张量函数的表示方法，用户可以在其中定义自定义操作符并执行细粒度调度。
下面对比了一个矩阵乘法生成的 TVMScript TensorIR 代码和 low-level Pytorch 代码。`@T.prim_func`装饰器表示下面的函数是一个原始的张量函数，包含运算符实现的底层细节。

{{< details title="Destination Passing" >}}
`T.prim_func` 采用 destination-passing 约定，即在函数外部明确分配输入和输出空间，并将其作为参数传入。destination-passing 约定可以对内存分配进行精细调度，例如合并两个实时间隔不相交的变量的内存分配，这是在内存有限的设备上运行大型模型的关键。
{{< /details >}}

```python
from tvm.script import tir as T
@T.prim_func
def matmul(rxplaceholder: T.Buffer((T.int64(1), T.int64(784)), "float32"), rxplaceholder_1: T.Buffer((T.int64(784), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": True})
    # with T.block("root"):
    for i0, i1, k in T.grid(T.int64(1), T.int64(128), T.int64(784)):
        with T.block("matmul"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(rxplaceholder[v_i0, v_k], rxplaceholder_1[v_k, v_i1])
            T.writes(matmul[v_i0, v_i1])
            with T.init():
                matmul[v_i0, v_i1] = T.float32(0)
            matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + rxplaceholder[v_i0, v_k] * rxplaceholder_1[v_k, v_i1]

def torch_matmul(X: torch.Tensor, W: torch.Tensor):
    Y = torch.zeros(1, 128, dtype="float32")
    for i in range(1):
        for j in range(128):
            for k in range(784):
                Y[i, j] = Y[i, j] + X[i, k] * W[k, j]
    return Y
```

## Interaction between Relax function and TensorIR

为了支持 `T.prim_func`（底层部分）和 `R.function`（高层部分）之间的交互，TVM 引入了 `call_tir`, Relax 中的一个特殊运算符，用于描述计算图中的节点及其张量函数的实现。
`torch_call_tir` 是一个参考实现，用来说明 call_tir 的含义。实际上，可以有不同的底层方法来优化执行。例如，我们可能会选择提前分配所有输出内存，然后再运行执行。

```python
def torch_call_tir(prim_func, inputs, out_sinfo):
    res = torch.zeros(*out_sinfo.shape, dtype=out_sinfo.dtype)
    prim_func(*inputs, res)
    return res
```

下面是 2 层 MLP 的 IRModule，我们使用 `call_tir` 和张量原语函数 `matmul` 来替换 Relax 运算符 `R.matmul` 

```python
@I.ir_module
class Module:
    @T.prim_func
    def tir_matmul(rxplaceholder: T.Buffer((T.int64(1), T.int64(784)), "float32"), rxplaceholder_1: T.Buffer((T.int64(784), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": True})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(128), T.int64(784)):
            with T.block("matmul"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(rxplaceholder[v_i0, v_k], rxplaceholder_1[v_k, v_i1])
                T.writes(matmul[v_i0, v_i1])
                with T.init():
                    matmul[v_i0, v_i1] = T.float32(0)
                matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + rxplaceholder[v_i0, v_k] * rxplaceholder_1[v_k, v_i1]

    @R.function
    def main(inp_0: R.Tensor((1, 784), dtype="float32"), weight1: R.Tensor((128, 784), dtype="float32"), bias1: R.Tensor((1, 128), dtype="float32"),
             weight2: R.Tensor((10, 128), dtype="float32"), bias2: R.Tensor((1, 10), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(weight1, axes=None)
            lv1 = R.call_tir(cls.tir_matmul, [inp_0, lv], out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv2: R.Tensor((1, 128), dtype="float32") = R.add(lv1, bias1)
            lv3: R.Tensor((1, 128), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(weight2, axes=None)
            lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(lv3, lv4, out_dtype="float32")
            lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, bias2)
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv
```

## Implement Custom TensorIR Function

`nn.Module` 不仅支持高级 Relax 运算符，还支持自定义 TensorIR 函数。
要构建 TensorIR 函数并在 Relax 图中调用它，我们需要使用 `nn.emit_te(f_te_expr,*args)`。
- `f_te_expr` 是一个返回张量表达式（Tensor Expression，TE）的函数，是描述张量计算的 DSL.
- `args` 是 `f_te_expr` 的参数。

创建 TE 表达式的方法如下

```python
te.compute(out_shape, f_compute)
```

它描述如下的计算模式
{{< details title="itertools.product" >}}
在 Python 的 itertools 模块中，`product` 函数用于生成可迭代对象的笛卡尔积。

`product` 函数接受一个或多个可迭代对象作为参数，并返回一个迭代器，该迭代器生成所有可能的组合，其中每个组合包含来自每个输入可迭代对象的单个元素。


```python
import itertools

letters = ['a', 'b']
numbers = [1, 2, 3]

for item in itertools.product(letters, numbers):
    print(item)

# output：
# ('a', 1)
# ('a', 2)
# ('a', 3)
# ('b', 1)
# ('b', 2)
# ('b', 3)
```

`product` 函数还支持重复元素，可以使用 repeat 参数指定每个可迭代对象需要重复的次数。


```python
letters = ['a', 'b']

for item in itertools.product(letters, repeat=3):
    print(item)

# output：
# ('a', 'a', 'a')
# ('a', 'a', 'b')
# ('a', 'b', 'a')
# ('a', 'b', 'b')
# ('b', 'a', 'a')
# ('b', 'a', 'b')
# ('b', 'b', 'a')
# ('b', 'b', 'b')
```

`product` 应用场景
- 组合生成: 生成所有可能的组合，例如密码生成、彩票号码生成等。
- 多维数组遍历: 遍历多维数组的所有元素。
- 测试用例生成: 生成测试用例，覆盖所有可能的输入组合。
{{< /details >}}


```python
from itertools import product

for indices in product(range(s) for s in out_shape):
  out_tensor[*indices] = f_compute(*indices)
```

用 `emit_te` 实现 Linear 层来构建 IRModule 的代码如下

```python
from tvm import te

class RelaxLinearWithEmitTE(nn.Module):
    def __init__(self, in_features, out_features, dtype="float32", bias=True) -> None:
        super(RelaxLinearWithEmitTE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter((out_features, in_features), dtype, name="linear_weight")
        if bias:
            self.bias = nn.Parameter((1, out_features), dtype, name="linear_bias")
        else:
            self.bias = None
    
    def forward(self, x: relax.Expr) -> relax.Var:
        def my_linear(x, w, b=None):
            out_sinfo = x.shape[:-1] + [self.out_features,]
            k = te.reduce_axis((0, self.out_features), name="k")
            out = te.compute(out_sinfo, fcompute=lambda i, j: te.sum(x[i, k] * w[j, k], axis=k), name="matmul")
            if b is not None:
                return out
            else:
                return te.compute(out_sinfo, fcompute=lambda i, j: out[i, j] + b[0, j], name="add_bias")

        return nn.emit_te(my_linear, x, self.weight, self.bias)

class RelaxMLPwithEmitTE(nn.Module):
    def __init__(self, in_features, hidden_num, out_features, dtype="float32"):
       self.linear1 = RelaxLinearWithEmitTE(in_features, hidden_num, dtype=dtype)
       self.linear2 = RelaxLinearWithEmitTE(hidden_num, out_features, dtype=dtype)

    def forward(self, input: relax.Expr) -> relax.Var:
        hidden = self.linear1(input)
        hidden = nn.emit(relax.op.nn.relu(hidden))
        out = self.linear2(hidden)
        return out
    
build_relax(RelaxMLPwithEmitTE).show()

#----------------------------------------------------
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def my_linear(input: T.Buffer((T.int64(1), T.int64(784)), "float32"), linear_weight: T.Buffer((T.int64(128), T.int64(784)), "float32"), linear_bias: T.Buffer((T.int64(1), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(1), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(input[v_i, v_k], linear_weight[v_j, v_k])
                T.writes(matmul[v_i, v_j])
                with T.init():
                    matmul[v_i, v_j] = T.float32(0.0)
                matmul[v_i, v_j] = matmul[v_i, v_j] + input[v_i, v_k] * linear_weight[v_j, v_k]

    @T.prim_func(private=True)
    def my_linear1(lv1: T.Buffer((T.int64(1), T.int64(128)), "float32"), linear_weight: T.Buffer((T.int64(10), T.int64(128)), "float32"), linear_bias: T.Buffer((T.int64(1), T.int64(10)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(1), T.int64(10), T.int64(10)):
            with T.block("matmul"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(lv1[v_i, v_k], linear_weight[v_j, v_k])
                T.writes(matmul[v_i, v_j])
                with T.init():
                    matmul[v_i, v_j] = T.float32(0.0)
                matmul[v_i, v_j] = matmul[v_i, v_j] + lv1[v_i, v_k] * linear_weight[v_j, v_k]

    @R.function
    def main(input: R.Tensor((1, 784), dtype="float32"), linear_weight: R.Tensor((128, 784), dtype="float32"), linear_bias: R.Tensor((1, 128), dtype="float32"), linear_weight_1: R.Tensor((10, 128), dtype="float32"), linear_bias_1: R.Tensor((1, 10), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.my_linear, (input, linear_weight, linear_bias), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv1: R.Tensor((1, 128), dtype="float32") = R.nn.relu(lv)
            lv2 = R.call_tir(cls.my_linear1, (lv1, linear_weight_1, linear_bias_1), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            gv: R.Tensor((1, 10), dtype="float32") = lv2
            R.output(gv)
        return gv
```