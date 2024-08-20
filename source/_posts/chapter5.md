---
title: TVM Learning (7)-Integration with Machine Learning Frameworks
date: 2024/8/20 16:07:28
categories: TVM
tags: TVM learning
excerpt: Personal notebook 5.
mathjax: true
katex: true
---
# Build an IRModule Through a Builder

下面用一个矩阵乘法回顾一下如何从张量表达式创建IRModule. 先创建 `placeholder`对象表示 `T.prim_func`函数的输入。

```python
A = te.placeholder((128, 128), name="A", dtype="float32")
B = te.placeholder((128, 128), name="B", dtype="float32")
print(type(A))  # <class 'tvm.te.tensor.Tensor'>

def te_matmul(A: te.Tensor, B: te.Tensor) -> te.Tensor:
    assert A.shape[1] == B.shape[0]
    n = A.shape[0]
    m = A.shape[1]
    k = te.reduce_axis((0, A.shape[1]), name="k")
    return te.compute((n, m), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name="matmul")

C = te_matmul(A, B)  # create the result
te.create_prim_func([A, B, C]).show()

#--------------------------------------------
@T.prim_func
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), matmul: T.Buffer((128, 128), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i, j, k in T.grid(128, 128, 128):
        with T.block("matmul"):
            v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
            T.reads(A[v_i, v_k], B[v_k, v_j])
            T.writes(matmul[v_i, v_j])
            with T.init():
                matmul[v_i, v_j] = T.float32(0.0)
            matmul[v_i, v_j] = matmul[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
```

同样我们可以使用 `*`运算符对tuple解引用来实现对不同维度大小的输入进行ReLU.

```python
def te_relu(A: te.Tensor) -> te.Tensor:  # * used to unpack list
    return te.compute(A.shape, lambda *i: te.max(A(*i), 0), name="relu")
```

我们仅仅对传入输入和输出参数来创建T.prim_func，这样可以使得中间结果仅仅被分配临时内存（在Schedule.compute_at已介绍过）。可以看到矩阵乘法的中间结果 `matmul`被 `T.alloc_buffer`分配。

```python
C = te_matmul(A, B)
D = te_relu(C)
print("----------Composed Operation-----------")
te.create_prim_func([A, B, D]).show()

#-----------------------------------------------------------
@T.prim_func
def main(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32"), relu: T.Buffer((128, 128), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    matmul = T.alloc_buffer((128, 128))
    for i, j, k in T.grid(128, 128, 128):
        with T.block("matmul"):
            v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
            T.reads(A[v_i, v_k], B[v_k, v_j])
            T.writes(matmul[v_i, v_j])
            with T.init():
                matmul[v_i, v_j] = T.float32(0.0)
            matmul[v_i, v_j] = matmul[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]
    for i0, i1 in T.grid(128, 128):
        with T.block("relu"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(matmul[v_i0, v_i1])
            T.writes(relu[v_i0, v_i1])
            relu[v_i0, v_i1] = T.max(matmul[v_i0, v_i1], T.float32(0.0))
```

# Use BlockBuilder to Create an IRModule

在chapter3_exercise中也介绍了使用 `relax.BlockBuilder`来创建IRModule.  `BlockBuilder` 自带的作用域与 relax 函数中的作用域相对应。例如，`bb.dataflow()` 会创建一个数据流代码块，其中所有 `BlockBuilder`的方法调用的作用域都属于数据流作用域。每个中间结果都是一个 `relax.Var`，对应于一个存储计算结果的变量。`DataflowVar`表示该变量是数据流块（计算图）中的一个中间步骤。在底层，`bb.emit_te` 会执行以下操作：

- 为 A 和 B 创建输入 `te.placeholder`
- 调用 `te_matmul` 函数运行它们。
- 调用 `te.create_prim_func` 创建一个 TensorIR 函数。
- 通过 `call_tir` 生成对该函数的调用。

最后，函数输出由 `bb.emit_func_output` 标记。在每个函数作用域中，我们只能调用一次 `emit_func_output`。

```python
bb = relax.BlockBuilder()
with bb.function("main"):
    with bb.dataflow():
        # Each intermediate result is a relax.Var`
        # corresponding to a variable that stores the result of the computation. 
        # DataflowVar indicates that the var is an intermediate step inside a dataflow block (computational graph).
        C = bb.emit_te(te_matmul, A, B)  # tvm.relax.expr.DataflowVar
        D = bb.emit_te(te_relu, C)
        R = bb.emit_output(D)  # marks that D is a variable that can be referred to outside of the dataflow block.
    bb.emit_func_output(R, params=[A, B])  # We can only call `emit_func_output` once in each function scope.
  
MyModule = bb.get()
MyModule.show()

#-------------------------------------------------------
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def te_matmul(A: T.Buffer((T.int64(128), T.int64(128)), "float32"), B: T.Buffer((T.int64(128), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(128), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(A[v_i, v_k], B[v_k, v_j])
                T.writes(matmul[v_i, v_j])
                with T.init():
                    matmul[v_i, v_j] = T.float32(0.0)
                matmul[v_i, v_j] = matmul[v_i, v_j] + A[v_i, v_k] * B[v_k, v_j]

    @T.prim_func(private=True)
    def te_relu(lv: T.Buffer((T.int64(128), T.int64(128)), "float32"), relu: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(128), T.int64(128)):
            with T.block("relu"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv[v_i0, v_i1])
                T.writes(relu[v_i0, v_i1])
                relu[v_i0, v_i1] = T.max(lv[v_i0, v_i1], T.float32(0.0))

    @R.function
    def main(A: R.Tensor((128, 128), dtype="float32"), B: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.te_matmul, (A, B), out_sinfo=R.Tensor((128, 128), dtype="float32"))
            lv1 = R.call_tir(cls.te_relu, (lv,), out_sinfo=R.Tensor((128, 128), dtype="float32"))
            gv: R.Tensor((128, 128), dtype="float32") = lv1
            R.output(gv)
        return gv
```

值得注意，我们可以在 `emit_func_output`指定函数的输入参数列表，这样做有助于我们随时获取参数列表。我们也可以在最开始的函数作用域里面声明。

```python
with bb.function("main"):
    ...
    # specify parameters in the end
    bb.emit_func_output(R, params=[A, B])

# specify parameters in the beginning.
with bb.function("main", params=[A, B]):
    ...
    bb.emit_func_output(R)
```

# Import Model From PyTorch

我们了解了以编程方式构建 IRModule 的工具。也可以用它们将 PyTorch 中的模型转换成 IRModule. 用Pytorch实现矩阵乘法+ReLU的网络如下

```python
class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(MyModel, self).__init__(*args, **kwargs)
        self.weight = nn.Parameter(torch.randn(128, 128))
  
    def forward(self, x):
        x = torch.matmul(x, self.weight)
        x = torch.relu(x)
        return x
```

TorchFX是用来变换 `nn.Module` 实例的工具包。FX 由三个主要组件组成：symbolic_trace、中间表示和 Python 代码生成。

## Symbolic Trace

`symbolic_trace` 函数用于对一个 PyTorch 模型进行符号追踪，它会执行模型的 forward 函数，并记录所有操作（如卷积、线性层、激活函数等）以及它们之间的依赖关系。返回一个包含了模型的计算图表示的 `GraphModule` 对象。

```python
model = MyModel()
fx_module = fx.symbolic_trace(model)
fx_module.graph.print_tabular()
'''
opcode         name    target                                                         args         kwargs
-------------  ------  -------------------------------------------------------------  -----------  --------
placeholder    x       x                                                              ()           {}
get_attr       weight  weight                                                         ()           {}
call_function  matmul  <built-in method matmul of type object at 0x00007FFC426359D0>  (x, weight)  {}
call_function  relu    <built-in method relu of type object at 0x00007FFC426359D0>    (matmul,)    {}
output         output  output                                                         (relu,)      {}
'''
```

* 在 FX 中，方法输入通过特殊的 `placeholder` 节点指定。在本例中，我们有一个 `placeholder` 节点，其 `target` 为 `x` ，这意味着我们有一个名为 x 的（非自身）参数。
* `get_attr` 、 `call_function` 、 `call_module` 和 `call_method` 节点表示方法中的操作。所有这些语义的完整处理可以在 [`Node`](https://runebook.dev/zh/docs/pytorch/fx#torch.fx.Node) 文档中找到。
* [`Graph`](https://runebook.dev/zh/docs/pytorch/fx#torch.fx.Graph) 中的返回值由特殊的 `output` 节点指定。

## Graph IR

`symbolic_traced.graph` 属性是一个 `torch.fx.Graph` 对象，代表了模型的计算图的 IR 表示。

1. `graph():` 定义了一个名为 `graph` 的函数，它代表整个计算图。
2. `%x : [num_users=1] = placeholder[target=x]`定义了一个名为 `%x` 的占位符节点，它代表模型的输入数据。
   * `[num_users=1]` 表示这个节点在计算图中被使用了一次。
   * `target=x` 表示这个占位符节点对应于模型的 `x` 输入参数。
3. `%weight : [num_users=1] = get_attr[target=weight]` 定义了一个名为 `%weight` 的节点，它代表模型的权重参数。
   * `target=weight` 表示这个节点对应于模型的 `weight` 属性。
4. `%matmul : [num_users=1] = call_function[target=torch.matmul](args = (%x, %weight), kwargs = {})` 定义了一个名为 `%matmul` 的节点，它代表对输入数据 `%x` 和权重参数 `%weight` 进行矩阵乘法操作。
   * `target=torch.matmul` 表示这个节点对应于 PyTorch 的 `torch.matmul` 函数。
   * `args = (%x, %weight)` 表示该操作的输入参数是 `%x` 和 `%weight`。
   * `kwargs = {}` 表示该操作没有额外的关键字参数。
5. `%relu : [num_users=1] = call_function[target=torch.relu](args = (%matmul,), kwargs = {})` 定义了一个名为 `%relu` 的节点，它代表对矩阵乘法的结果 `%matmul` 应用 ReLU 激活函数。
   * `target=torch.relu` 表示这个节点对应于 PyTorch 的 `torch.relu` 函数。
   * `args = (%matmul,)` 表示该操作的输入参数是 `%matmul`。
6. `return relu` 表示计算图的输出是 `%relu` 节点，即经过 ReLU 激活后的结果。

```python
print(fx_module.graph)
'''
graph():
    %x : [num_users=1] = placeholder[target=x]
    %weight : [num_users=1] = get_attr[target=weight]
    %matmul : [num_users=1] = call_function[target=torch.matmul](args = (%x, %weight), kwargs = {})
    %relu : [num_users=1] = call_function[target=torch.relu](args = (%matmul,), kwargs = {})
    return relu
'''
```

## Graph Code

`symbolic_traced.code` 属性是一个字符串，它包含了模型计算图的 Python 代码表示。对于每个计算图 IR，创建与图语义匹配的有效 Python 代码。

```python
# Graph code
print(fx_module.code)
'''
def forward(self, x):
    weight = self.weight
    matmul = torch.matmul(x, weight);  x = weight = None
    relu = torch.relu(matmul);  matmul = None
    return relu
'''
```

## Create Map Function

整个翻译逻辑的主要流程如下：

- 创建一个 `node_map` ，将 `fx.Node` 映射到相应的 `relax.Var` 以表示 IRModule 中的节点。
- 按拓扑顺序遍历 fx 图中的节点。
- 根据映射输入计算节点的映射输出。

### Map Parameter

`map_param(param: nn.Parameter)`函数将 PyTorch 的 `nn.Parameter` 对象转换为 TVM Relax 的常量节点。它首先获取参数的形状和数据类型，然后使用 `relax.const` 函数创建一个常量节点，并将参数数据转换为 NumPy 数组。

```python
def map_param(param: nn.Parameter):
    ndim = len(param.data.shape)
    return relax.const(param.data.cpu().numpy(), relax.DynTensorType(ndim, "float32"))
```

### Fetch Attribution

`fetch_attr(fx_mod, target: str)`函数用于从 `fx_mod` 对象中获取指定属性值。它将 `target` 字符串拆分为多个部分，并依次访问 `fx_mod` 对象的属性，直到找到目标属性。

```python
def fetch_attr(fx_mod, target: str):
    '''
    Helper function to fetch an attr
    '''
    target_atoms = target.split('.')
    attr_itr = fx_mod
    for i, atom in enumerate(target_atoms):
        if not hasattr(attr_itr, atom):
            raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
        attr_itr = getattr(attr_itr, atom)
    return attr_itr
```

## Translate from TorchFX

`from_fx(fx_mod, input_shapes, call_function_map, call_module_map)`函数是核心转换函数，它将 `fx_mod` 对象转换为 TVM Relax 的 `IRModule` 对象。

它首先定义了几个变量：

* `input_index`: 用于跟踪输入节点的索引。
* `node_map`: 用于存储 `fx_mod` 中每个节点对应的 Relax 节点。
* `named_modules`: 用于存储 `fx_mod` 中所有模块的名称和对象。
* `bb`: 一个 `relax.BlockBuilder` 对象，用于构建 Relax 函数。
* `fn_inputs`: 用于存储函数的输入参数。
* `fn_output`: 用于存储函数的输出节点。

然后使用 `bb.function` 创建一个名为 "main" 的 Relax 函数。在函数中，遍历 `fx_mod` 的所有节点，并根据节点类型进行不同的处理：

* **`placeholder`:** 创建一个输入占位符节点。
* **`get_attr`:** 使用 `map_param` 函数将参数转换为常量节点。
* **`call_function`:** 使用 `call_function_map` 字典中指定的函数来处理函数调用。
* **`call_module`:** 使用 `call_module_map` 字典中指定的函数来处理模块调用。
* **`output`:** 设置函数的输出节点。

最后，使用 `bb.get()` 获取生成的 `IRModule` 对象。

```python
def from_fx(fx_mod, input_shapes, call_function_map, call_module_map):
    input_index = 0
    node_map = {}
    named_modules = dict(fx_mod.named_modules())
  
    bb = relax.BlockBuilder()
  
    fn_inputs = []
    fn_oputput = None
    with bb.function("main"):
        with bb.dataflow():
            for node in fx_mod.graph.nodes:
                if node.op == "placeholder":
                    # create input placeholder
                    shape = input_shapes[input_index]
                    input_index += 1
                    input_var = relax.Var(node.target, relax.TensorStructInfo(shape, "float32"))
                    fn_inputs.append(input_var)
                    node_map[node] = input_var
                elif node.op == "get_attr":
                    node_map[node] = map_param(fetch_attr(fx_mod, node.target))
                elif node.op == "call_function":
                    node_map[node] = call_function_map[node.target](bb, node_map, node)
                elif node.op == "call_module":
                    named_module = named_modules[node.target]
                    node_map[node] = call_module_map[type(named_module)](bb, node_map, node, named_module)
                elif node.op == "output":
                    output = node_map[node.args[0]]
                    assert fn_oputput is None
                    fn_oputput = bb.emit_output(output)
        bb.emit_func_output(output, fn_inputs)
    return bb.get()

```

创建的IRModule如下

```python
# map function in the from_fx function
def map_matmul(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    B = node_map[node.args[1]]
    return bb.emit_te(te_matmul, A, B)

def map_relu(bb, node_map, node: fx.Node):
    A = node_map[node.args[0]]
    return bb.emit_te(te_relu, A)

MyModule = from_fx(
    fx_module, 
    input_shapes = [(1, 128)], 
    call_function_map = {
      torch.matmul: map_matmul,
      torch.relu: map_relu, 
    },
    call_module_map={},
)

MyModule.show()

#----------------------------------------------------
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def te_matmul(x: T.Buffer((T.int64(1), T.int64(128)), "float32"), B: T.Buffer((T.int64(128), T.int64(128)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(1), T.int64(128), T.int64(128)):
            with T.block("matmul"):
                v_i, v_j, v_k = T.axis.remap("SSR", [i, j, k])
                T.reads(x[v_i, v_k], B[v_k, v_j])
                T.writes(matmul[v_i, v_j])
                with T.init():
                    matmul[v_i, v_j] = T.float32(0.0)
                matmul[v_i, v_j] = matmul[v_i, v_j] + x[v_i, v_k] * B[v_k, v_j]

    @T.prim_func(private=True)
    def te_relu(lv: T.Buffer((T.int64(1), T.int64(128)), "float32"), relu: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("relu"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv[v_i0, v_i1])
                T.writes(relu[v_i0, v_i1])
                relu[v_i0, v_i1] = T.max(lv[v_i0, v_i1], T.float32(0.0))

    @R.function
    def main(x: R.Tensor((1, 128), dtype="float32")) -> R.Tensor((1, 128), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.te_matmul, (x, metadata["relax.expr.Constant"][0]), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_tir(cls.te_relu, (lv,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            gv: R.Tensor((1, 128), dtype="float32") = lv1
            R.output(gv)
        return lv1
```

## Translate by reusing pre-defined TE libraries

[TOPI](https://tvm.apache.org/docs/reference/api/python/topi.html) (TVM OPeration Inventory) 提供了 numpy 风格的通用操作和 schedule，其抽象程度高于 TVM. 使用它里面已有的模块可以省去自己定义张量表达式的工作。

- `topi.nn.dense(x, w)`执行转置矩阵乘法 `x @ w.T`
- `topi.add` 执行广播加法。

我们可以将下面的Pytorch MLP网络翻译成IRModule

```python
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear0 = nn.Linear(784, 128, bias=True)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(128, 10, bias=True)

    def forward(self, x):
        x = self.linear0(x)
        x = self.relu(x)
        x = self.linear1(x)
        return x
  
mlp_model = MLP()

from tvm import topi

def map_nn_linear(bb, node_map, node, nn_mod):
    x = node_map[node.args[0]]
    w = map_param(nn_mod.weight)
    if nn_mod.bias is not None:
        b = map_param(nn_mod.bias)
    y = bb.emit_te(topi.nn.dense, x, w)
    return bb.emit_te(topi.add, y, b)

def map_nn_relu(bb, node_map, node, nn_mod):
    return map_relu(bb, node_map, node)

MLPModule = from_fx(
    fx.symbolic_trace(mlp_model), 
    input_shapes = [(1, 784)], 
    call_function_map={
    },
    call_module_map={
        torch.nn.Linear: map_nn_linear,
        torch.nn.ReLU: map_nn_relu,
    },
)

MLPModule.show()
#------------------------------------------------------
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv: T.Buffer((T.int64(1), T.int64(128)), "float32"), B: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax0, v_ax1], B[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv[v_ax0, v_ax1] + B[v_ax1]

    @T.prim_func(private=True)
    def add1(lv3: T.Buffer((T.int64(1), T.int64(10)), "float32"), B: T.Buffer((T.int64(10),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv3[v_ax0, v_ax1], B[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv3[v_ax0, v_ax1] + B[v_ax1]

    @T.prim_func(private=True)
    def dense(x: T.Buffer((T.int64(1), T.int64(784)), "float32"), B: T.Buffer((T.int64(128), T.int64(784)), "float32"), T_matmul_NT: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(128), T.int64(784)):
            with T.block("T_matmul_NT"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], B[v_i1, v_k])
                T.writes(T_matmul_NT[v_i0, v_i1])
                with T.init():
                    T_matmul_NT[v_i0, v_i1] = T.float32(0.0)
                T_matmul_NT[v_i0, v_i1] = T_matmul_NT[v_i0, v_i1] + x[v_i0, v_k] * B[v_i1, v_k]

    @T.prim_func(private=True)
    def dense1(lv2: T.Buffer((T.int64(1), T.int64(128)), "float32"), B: T.Buffer((T.int64(10), T.int64(128)), "float32"), T_matmul_NT: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(10), T.int64(128)):
            with T.block("T_matmul_NT"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(lv2[v_i0, v_k], B[v_i1, v_k])
                T.writes(T_matmul_NT[v_i0, v_i1])
                with T.init():
                    T_matmul_NT[v_i0, v_i1] = T.float32(0.0)
                T_matmul_NT[v_i0, v_i1] = T_matmul_NT[v_i0, v_i1] + lv2[v_i0, v_k] * B[v_i1, v_k]

    @T.prim_func(private=True)
    def te_relu(lv1: T.Buffer((T.int64(1), T.int64(128)), "float32"), relu: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("relu"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv1[v_i0, v_i1])
                T.writes(relu[v_i0, v_i1])
                relu[v_i0, v_i1] = T.max(lv1[v_i0, v_i1], T.float32(0.0))

    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.dense, (x, metadata["relax.expr.Constant"][0]), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv1 = R.call_tir(cls.add, (lv, metadata["relax.expr.Constant"][1]), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv2 = R.call_tir(cls.te_relu, (lv1,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv3 = R.call_tir(cls.dense1, (lv2, metadata["relax.expr.Constant"][2]), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            lv4 = R.call_tir(cls.add1, (lv3, metadata["relax.expr.Constant"][3]), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            gv: R.Tensor((1, 10), dtype="float32") = lv4
            R.output(gv)
        return lv4
```

## Translating into High-level Operators
在大多数机器学习框架中，首先翻译成高级内置原语算子有时会很有帮助，因为这些算子已经被很大程度上优化过。我们通过调用内置算子将模型导入 IRModule. 这些内置运算符是比TensorIR函数高级的抽象。我们可以利用不同的方法，将这些原语算子进一步转化为库函数或TensorIR函数。
可以看见relax函数里面都是调用的原始算子而不是使用`call_tir`
```python
def map_nn_relu_op(bb, node_map, node, nn_mod):
    A = node_map[node.args[0]]
    return bb.emit(relax.op.nn.relu(A))

def map_nn_linear_op(bb, node_map, node, nn_mod):
    x = node_map[node.args[0]]
    w = map_param(nn_mod.weight)
    if nn_mod.bias is not None:
        b = map_param(nn_mod.bias)
    return bb.emit(relax.op.linear(x, w, b))

MLPModuleHighLevel = from_fx(
    fx.symbolic_trace(mlp_model), 
    input_shapes=[(1, 784)], 
    call_function_map={
    },
    call_module_map={
        torch.nn.Linear: map_nn_linear_op,
        torch.nn.ReLU: map_nn_relu_op,
    },
)

MLPModuleHighLevel.show()

#------------------------------------
@I.ir_module
class Module:
    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][0], axes=None)
            lv1: R.Tensor((1, 128), dtype="float32") = R.matmul(x, lv, out_dtype="void")
            lv2: R.Tensor((1, 128), dtype="float32") = R.add(lv1, metadata["relax.expr.Constant"][1])
            lv3: R.Tensor((1, 128), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][2], axes=None)
            lv5: R.Tensor((1, 10), dtype="float32") = R.matmul(lv3, lv4, out_dtype="void")
            lv6: R.Tensor((1, 10), dtype="float32") = R.add(lv5, metadata["relax.expr.Constant"][3])
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return lv6
```
