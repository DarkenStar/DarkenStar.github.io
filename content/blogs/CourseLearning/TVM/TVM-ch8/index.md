---
title: TVM Learning (10)-Computational Graph Optimization
date: 2024-08-25T16:08:00+08:00
lastmod: 2024-08-25T16:08:00+08:00
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
# Pattern Match and Rewriting

下面代码中 `MyModule` 包含一个带有两个高级算子 `relax.opmultiply` 和 `relax.op.add` 的 relax 函数。我们的目标是找到这两个算子，并将其替换为对 `relax.ewise_fma` 算子的调用。

```python
@tvm.script.ir_module 
class MyModule:
    @R.function
    def main(x: R.Tensor((3, 4), "float32"), y: R.Tensor((3, 4), "float32")): # type: ignore
        with R.dataflow():
            cls = MyModule
            lv0 = relax.op.multiply(x, y)
            gv0 = relax.op.add(lv0, y)
            R.output(gv0)
        return gv0
```

每个 IRModule 都包含一组函数，函数体由一组称为抽象语法树（AST）的数据结构组成。
{% fold info @Abstract Syntax Tree %}
抽象语法树（Abstract Syntax Tree，AST）是一种广泛用于编程语言处理的树状数据结构。它是一种对源代码语法结构的抽象表示，去掉了编程语言的具体语法细节，但保留了代码的结构和语义信息。
AST 是一棵树状结构，其节点表示源代码中的语法结构。例如，变量声明、操作符、函数调用、控制结构（如条件语句、循环）等。每个节点包含与相应语法结构相关的信息，如操作符的类型、变量的名称、常量的值等。

```python
a = b + 1
```

这个代码可以转换为如下形式的 AST：

```scss
Assignment
├── Identifier (a)
└── BinaryOperation
    ├── Identifier (b)
    └── Constant (1)
```

{% endfold %}
每个函数都由一个 `relax.expr.Function` 节点表示。

```python
relax_func = MyModule["main"]
type(relax_func)  # <class 'tvm.relax.expr.Function'>
```

该函数包含一系列参数

```python
print(relax_func.params)  # [x, y]
```

该函数包含一个返回值表达式，和函数中的一组 binding blocks.

```python
func_body = relax_func.body
print(type(func_body))  # <class 'tvm.relax.expr.SeqExpr'>
```

函数主体 SeqExpr 包含一系列 binding.

```python
print(relax_func.body.blocks) 
'''
[x: R.Tensor((3, 4), dtype="float32")
y: R.Tensor((3, 4), dtype="float32")
with R.dataflow():
    lv0: R.Tensor((3, 4), dtype="float32") = R.multiply(x, y)
    gv0: R.Tensor((3, 4), dtype="float32") = R.add(lv0, y)
    R.output(gv0)]
'''
```

在 DataflowBlock 中,我们可以访问各个 binding ,包括 value 和 var.

```python
dataflow_block = func_body.blocks[0]
print(type(dataflow_block))  # <class 'tvm.relax.expr.DataflowBlock'>
binding = dataflow_block.bindings[0]
print(type(binding))  # <class 'tvm.relax.expr.VarBinding'>
print(binding.var)  # LHS of binding: lv0
print(binding.value)  # # LHS of binding: R.multiply(x, y)
```

![Relax Function Data Structure](https://mlc.ai/zh/_images/relax_func_data_structure.png "Relax Function Data Structure")

改写程序可以通过递归遍历 MyModule 的 AST ，并生成转换后的 AST 来实现。但是我们可以使用额外的工具支持来简化流程。下面的代码遵循一种称为 visitor pattern 的设计模式，允许我们访问每个 AST 节点并将它们重写为转换后的版本。主要目的是将形如 `a * b + c` 的表达式转换为 `ewise_fma(a, b, c)` 的形式。

`EwiseFMARewriter` 继承自 `relax.PyExprMutator`，这是 TVM 中的一个基类，用于遍历和修改表达式树中的节点。`visit_call_` 方法被重载来处理 `relax.Call` 节点，被重载来处理 `relax.Call` 节点。

如果当前节点不是加法操作，直接返回该节点，表示对该节点不进行任何修改。如果加法的第一个操作数不是乘法操作，或者第一个操作数的绑定值不是一个 `relax.Call` 节点，直接返回该加法节点。如果匹配成功，构造一个新的 `ewise_fma` 操作节点，将乘法的两个操作数和加法的第二个操作数作为参数传入。

```python
@relax.expr_functor.mutator
class EwiseFMARewriter(relax.PyExprMutator):
    def visit_call_(self, op: relax.Call):  # Reloaded
        call = self.visit_expr_post_order(op)
        add_op = tvm.ir.Op.get("relax.add")
        multiply_op = tvm.ir.Op.get("relax.multiply")
        ewise_fma_op = tvm.ir.Op.get("relax.ewise_fma")
  
        if call.op != add_op:
            return call
  
        value = self.lookup_binding(call.args[0])
        if not isinstance(value, relax.Call) or value.op != multiply_op:
            return call 
  
        fma_call = relax.Call(
            ewise_fma_op, [value.args[0], value.args[1], call.args[1]], None, None
        )
        return fma_call
  
updated_fn = EwiseFMARewriter().visit_expr(MyModule["main"])
updated_fn.show()

#-----------------------------
@R.function
def main(x: R.Tensor((3, 4), dtype="float32"), y: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((3, 4), dtype="float32"):
    with R.dataflow():
        lv0: R.Tensor((3, 4), dtype="float32") = R.multiply(x, y)
        gv0: R.Tensor((3, 4), dtype="float32") = R.ewise_fma(x, y, y)
        R.output(gv0)
    return gv0
```

使用 `remove_all_unused` 来删除代码中没有用到的 DataflowBlocks 和 VarBindings.

```python
relax.analysis.remove_all_unused(updated_fn).show()

#-------------------------------------------
@R.function
def main(x: R.Tensor((3, 4), dtype="float32"), y: R.Tensor((3, 4), dtype="float32")) -> R.Tensor((3, 4), dtype="float32"):
    with R.dataflow():
        gv0: R.Tensor((3, 4), dtype="float32") = R.ewise_fma(x, y, y)
        R.output(gv0)
    return gv0
```

# Fuse Linear and ReLU

下面在端到端模型上进行计算图的改写。采用的还是之前使用的 FashionMNIST MLP 模型。为了简化过程，直接使用高级运算符构建模型。

```python
import pickle as pkl
mlp_params = pkl.load(open("fasionmnist_mlp_params.pkl", "rb"))

def create_model():
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo((1, 784), "float32"))
    w0 = relax.const(mlp_params["w0"], "float32")
    b0 = relax.const(mlp_params["b0"], "float32")
    w1 = relax.const(mlp_params["w1"], "float32")
    b1 = relax.const(mlp_params["b1"], "float32")
  
    with bb.function("main", [x]):
        with bb.dataflow():
            lv0 = bb.emit(relax.op.matmul(x, relax.op.permute_dims(w0)))
            lv1 = bb.emit(relax.op.add(lv0, b0))
            lv2 = bb.emit(relax.op.nn.relu(lv1))
            lv3 = bb.emit(relax.op.matmul(lv2, relax.op.permute_dims(w1)))
            lv4 = bb.emit(relax.op.add(lv3, b1))
            gv = bb.emit_output(lv4)
        bb.emit_func_output(gv)
  
    return bb.get()

MLPModel = create_model()
MLPModel.show()

#-------------------------------
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
        return gv
```

我们的目标是对 matmul 和 add 进行算子融合。具体实现步骤与 FMA 相似：

1. 识别 matmul 和 add 算子。
2. 生成另一个调用 matmul 和 add 算子的子函数。
3. 将 matmul 和 add 替换为融合后的子函数。

下面代码定义了一个名为 `DenseAddFusor` 的类，用于在 TVM 的 Relax 框架中将特定的矩阵乘法和加法操作模式融合成一个高效的原语函数。

- `transform` 方法遍历模块中的每个函数。如果函数已经被标记为 primitive（即已经被融合过），则跳过。对每个函数应用  `visit_expr` 以进行模式匹配和潜在的融合操作，然后删除未使用的变量，并更新函数。最后，返回更新后的 `IRModule`。
- `visit_call_` 方法用于访问 `relax.Call` 节点（表示操作符调用）。它首先递归处理子表达式，然后尝试匹配特定模式。`match_call` 是一个内部函数，用于检查某个节点是否是特定操作符的调用。如果当前节点不是 `add` 操作，或者 `add` 操作的第一个参数不是 `matmul`（矩阵乘法）操作，则直接返回当前节点，不进行修改。如果匹配成功，则提取 `matmul` 的两个操作数 `x` 和 `w` 以及 `add` 的第二个操作数 `b`，准备进行融合。
- 通过 `relax.BlockBuilder`定义一个名为 `fused_dense_addX`新的融合函数，其中 `X` 是一个递增的计数器。该函数接收 `x`、`w`、`b` 作为参数，首先进行矩阵乘法，然后将结果与 `b` 相加，最终输出结果。
- 给新生成的融合函数添加一个属性 Primitive，标记为已经融合的原语函数。通过 `builder_` 更新全局模块，将融合函数添加到模块中 (GlobalVar 用于指代存储在 IRModule 中的全局函数)。返回一个新的 `relax.Call` 节点，该节点调用生成的融合函数，并传递原始的输入参数 `x`、`w`、`b`。

{{< details title="VisitExpr" >}}

TVM 中的 `VisitExpr` 流程是一种递归遍历 IR 节点的机制,它是实现各种 IR 转换和优化的基础。具体流程如下:

1. 首先创建一个 `ExprVisitor` 或 `ExprMutator` 的子类实例,这个子类会实现各种具体的访问逻辑。
2. 调用 `visit_expr` 方法,传入根 IR 节点。这个方法会触发整个遍历过程的启动。
3. `visit_expr` 方法会首先调用 `visit_expr_post_order` 方法,这个方法会以深度优先的方式遍历所有子节点。
4. 对于每个子节点,`visit_expr_post_order` 会根据节点的具体类型,调用相应的 `visit_XXX_` 方法。这些 `visit_XXX_` 方法是由访问器子类实现的,包含了具体的访问逻辑。
5. 在 `visit_XXX_` 方法中,如果遇到子节点,会递归调用 `visit_expr_post_order` 方法继续遍历。
6. 当遍历完整个 IR 树后,`visit_expr` 方法会返回最终的结果,即经过转换和修改的 IR 节点。

{{< /details >}}

```python
@relax.expr_functor.mutator
class DenseAddFusor(relax.PyExprMutator):
    def __init__(self, mod: IRModule) -> None:
        super().__init__(mod)
        self.mod_ = mod
        # cache pre-defined ops
        self.add_op = tvm.ir.Op.get("relax.add")
        self.dense_op = tvm.ir.Op.get("relax.matmul")
        self.counter = 0
  
    def transform(self) -> IRModule:
        for global_var, func in self.mod_.functions_items():
            if not isinstance(func, relax.Function):
                continue
            # avoid already fused primitive function
            if "Primitive" in func.attrs.keys() and func.attrs["primitive"] != 0:
                continue
            updated_fn = self.visit_expr(func)
            updated_fn = relax.analysis.remove_all_unused(updated_fn)
            self.builder_.update_func(global_var, updated_fn)
  
        return self.builder_.get()
  
    def visit_call_(self, op: relax.Call):
        call = self.visit_expr_post_order(op)
  
        def match_call(node, op):
            if not isinstance(node, relax.Call):
                return False
            return node.op == op
  
        # pattern match dense => add
        if not match_call(call, self.add_op):
            return call
  
        value = self.lookup_binding(call.args[0])
        if value is None:
            return call
  
        if not match_call(value, self.dense_op):
            return call
  
        x = value.args[0]
        w = value.args[1]
        b = call.args[1]
  
        # construct a new fused primitive function
        param_x = relax.Var("x", relax.TensorStructInfo(x.struct_info.shape, x.struct_info.dtype))
        param_w = relax.Var("w", relax.TensorStructInfo(w.struct_info.shape, w.struct_info.dtype))
        param_b = relax.Var("b", relax.TensorStructInfo(b.struct_info.shape, b.struct_info.dtype))
  
        bb = relax.BlockBuilder()
  
        fn_name = "fused_dense_add%d" % (self.counter)
        self.counter += 1
        with bb.function(fn_name, [param_x, param_w, param_b]):
            with bb.dataflow():
                lv0 = bb.emit(relax.op.matmul(param_x, param_w))
                gv0 = bb.emit_output(relax.op.add(lv0, param_b))
            bb.emit_func_output(gv0)
  
        # add primitive attribute to the fused functions
        fused_fn = bb.get()[fn_name].with_attr("Primitive", 1)
        global_var = self.builder_.add_func(fused_fn, fn_name)
  
        # construct call into the fused function
        return relax.Call(global_var, [x, w, b], None, None)

@tvm.ir.transform.module_pass(opt_level=2, name="DenseAddFuse")
class FuseDenseAddPass:
    '''The wrapper for the LowerTensorIR pass.'''
    def transform_module(self, mod, ctx):
        return DenseAddFusor(mod).transform()

MLPFused = FuseDenseAddPass()(MLPModel)
MLPFused.show()
```

融合后的 MLPFused 对应的 TensorIR 如下

> TVM 框架中使用 module_pass 来管理各种优化操作。这种机制允许将不同的优化操作（如图优化、代码生成、算子融合等）组织成一个流水线（pipeline），按顺序对模块进行处理。将 DenseAddFusor 封装为一个 module_pass，使得它能够轻松集成到 TVM 的 Pass 流水线中，与其他 Pass 一起工作，从而保证优化过程的整体性和一致性。

```python
@I.ir_module
class Module:
    @R.function
    def fused_dense_add0(x: R.Tensor((1, 784), dtype="float32"), w: R.Tensor((784, 128), dtype="float32"), b: R.Tensor((128,), dtype="float32")) -> R.Tensor((1, 128), dtype="float32"):
        R.func_attr({"Primitive": 1})
        with R.dataflow():
            lv: R.Tensor((1, 128), dtype="float32") = R.matmul(x, w, out_dtype="void")
            gv: R.Tensor((1, 128), dtype="float32") = R.add(lv, b)
            R.output(gv)
        return gv

    @R.function
    def fused_dense_add1(x: R.Tensor((1, 128), dtype="float32"), w: R.Tensor((128, 10), dtype="float32"), b: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"Primitive": 1})
        with R.dataflow():
            lv: R.Tensor((1, 10), dtype="float32") = R.matmul(x, w, out_dtype="void")
            gv: R.Tensor((1, 10), dtype="float32") = R.add(lv, b)
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][0], axes=None)
            lv2: R.Tensor((1, 128), dtype="float32") = cls.fused_dense_add0(x, lv, metadata["relax.expr.Constant"][1])
            lv3: R.Tensor((1, 128), dtype="float32") = R.nn.relu(lv2)
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][2], axes=None)
            lv6: R.Tensor((1, 10), dtype="float32") = cls.fused_dense_add1(lv3, lv4, metadata["relax.expr.Constant"][3])
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv
```

上面的例子中，我们创建了两个前缀为 fuse_matmul_add 的子函数。 这些子函数包含有融合后算子的计算信息。 这种重写的替代方法是简单地为融合算子创建一个单独的原语算子（如ewise_fma）。 但是，当我们尝试融合更多算子时，可能存在指数级数量的组合。 将融合操作分组在一起的子函数为后续的 pass 保留了原始信息，进而便于分析，无需为每个融合 pattern 引入专用的高级算子。

# Map to TensorIR Calls

为了进一步进行底层优化和代码生成，我们需要将这些高级原语运算转换为相应的 TensorIR 函数。下面代码主要功能是将 Relax 表达式树中的高层次算子（ `matmul`、`add`、`relu`）转换为对应的 TensorIR 表示，从而使得这些算子能够映射到底层的张量操作（tensor operations）。这种转换使得编译器可以生成更接近硬件的高效代码，并为后续的代码优化和生成做好准备。

1. 调用 `transform` 方法会遍历 `mod_` 中的所有函数:
   * 对于每个函数,首先调用 `visit_expr` 方法,这会触发 `VisitExpr` 流程
   * `visit_expr` 方法会调用 `visit_expr_post_order`方法进行深度优先遍历
   * 在遍历过程中对于每个 `relax.Call` 节点,会调用 `visit_call_` 方法
   * `visit_call_` 方法会检查 `op_map` 字典,如果当前操作在字典中,则调用对应的转换函数( `map_dense`, `map_add`, `map_relu`)
   * 这些转换函数会使用 `bb.call_te` 方法,将 Relax IR 操作转换为 TensorIR 操作
2. 在 `transform` 方法的最后,会调用 `builder_.get()` 方法,返回转换后的新 IR 模块。
3. 最后 `LowerToTensorIRPass` 类将 `LowerToTensorIR` 转换器包装成一个可注册到 TVM 优化 pipeline 的 pass.

`module_pass` 的 `opt_level` 参数决定了优化 pass 在优化 pipeline 中的执行顺序。 TVM 的优化 pipeline 是由多个 `module_pass` 组成的,每个 `module_pass` 都有一个 `opt_level` 属性来指定它的优化级别。

当 TVM 进行优化时,它会按照 `opt_level` 从低到高的顺序依次应用各个 `module_pass`. `opt_level=0` 的 pass 会首先被执行。这些 pass 通常会执行一些基础的、必要的转换,为后续的优化奠定基础。 随后会执行 `opt_level=1` 的 pass,这些 pass 可能会执行一些更复杂的优化,比如循环优化、内存访问优化等。依此类推,`opt_level` 越高的 pass 会在优化 pipeline 的后期执行,它们执行的优化通常也越复杂和深入。

通过合理地设置 `opt_level`,开发者可以控制各个优化 pass 的执行顺序,从而构建出针对性强、性能优秀的优化 pipeline 。这种灵活的优化管理机制是 TVM 的一大特点。

对于 `LowerToTensorIRPass`,它的 `opt_level` 被设置为 0, 说明它是一个基础的 pass, 主要用于将高级的 Relax IR 操作转换为底层的 TensorIR 操作。

```python
@relax.expr_functor.mutator
class LowerToTensorIR(relax.PyExprMutator):
    def __init__(self, mod: IRModule, op_map: dict) -> None:
        super().__init__(mod)
        self.mod_ = mod
        self.op_map = {
            tvm.ir.Op.get(k): v for k, v in op_map.items()
        }
      
    def visit_call_(self, op: relax.Call):
        call = self.visit_expr_post_order(op)
      
        if call.op in self.op_map:
            return self.op_map[call.op](self.builder_, call)
        return call
  
    def transform(self) -> IRModule:
        for global_val, func in self.mod_.functions_items():
            if not isinstance(func, relax.Function):
                continue
            updated_fn = self.visit_expr(func)
            self.builder_.update_func(global_val, updated_fn)
          
        return self.builder_.get()
  

def map_dense(bb, call):
    x, w = call.args 
    return bb.call_te(topi.nn.matmul, x, w)

def map_add(bb, call):
    a, b = call.args           
    return bb.call_te(topi.add, a, b)

def map_relu(bb, call):
    return bb.call_te(topi.nn.relu, call.args[0])

op_map = {
  "relax.matmul": map_dense,
  "relax.add": map_add,
  "relax.nn.relu": map_relu
}

@tvm.ir.transform.module_pass(opt_level=0, name="LowerToTensorIR")
class LowerToTensorIRPass:
    '''The wrapper for the LowerTensorIR pass.'''
    def transform_module(self, mod, ctx):
        return LowerToTensorIR(mod, op_map).transform()
  
MLPModelTIR = LowerToTensorIRPass()(MLPFused)
MLPModelTIR.show()
```

融合后的 TensorIR 如下

```python
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def add(lv: T.Buffer((T.int64(1), T.int64(128)), "float32"), b: T.Buffer((T.int64(128),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax0, v_ax1], b[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv[v_ax0, v_ax1] + b[v_ax1]

    @T.prim_func(private=True)
    def add1(lv: T.Buffer((T.int64(1), T.int64(10)), "float32"), b: T.Buffer((T.int64(10),), "float32"), T_add: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv[v_ax0, v_ax1], b[v_ax1])
                T.writes(T_add[v_ax0, v_ax1])
                T_add[v_ax0, v_ax1] = lv[v_ax0, v_ax1] + b[v_ax1]

    @T.prim_func(private=True)
    def matmul(x: T.Buffer((T.int64(1), T.int64(784)), "float32"), w: T.Buffer((T.int64(784), T.int64(128)), "float32"), T_matmul_NN: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(128), T.int64(784)):
            with T.block("T_matmul_NN"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], w[v_k, v_i1])
                T.writes(T_matmul_NN[v_i0, v_i1])
                with T.init():
                    T_matmul_NN[v_i0, v_i1] = T.float32(0.0)
                T_matmul_NN[v_i0, v_i1] = T_matmul_NN[v_i0, v_i1] + x[v_i0, v_k] * w[v_k, v_i1]

    @T.prim_func(private=True)
    def matmul1(x: T.Buffer((T.int64(1), T.int64(128)), "float32"), w: T.Buffer((T.int64(128), T.int64(10)), "float32"), T_matmul_NN: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"layout_free_buffers": [1], "tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, k in T.grid(T.int64(1), T.int64(10), T.int64(128)):
            with T.block("T_matmul_NN"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], w[v_k, v_i1])
                T.writes(T_matmul_NN[v_i0, v_i1])
                with T.init():
                    T_matmul_NN[v_i0, v_i1] = T.float32(0.0)
                T_matmul_NN[v_i0, v_i1] = T_matmul_NN[v_i0, v_i1] + x[v_i0, v_k] * w[v_k, v_i1]

    @T.prim_func(private=True)
    def relu(lv2: T.Buffer((T.int64(1), T.int64(128)), "float32"), compute: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv2[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(lv2[v_i0, v_i1], T.float32(0.0))

    @R.function
    def fused_dense_add0(x: R.Tensor((1, 784), dtype="float32"), w: R.Tensor((784, 128), dtype="float32"), b: R.Tensor((128,), dtype="float32")) -> R.Tensor((1, 128), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.matmul, (x, w), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            gv = R.call_tir(cls.add, (lv, b), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def fused_dense_add1(x: R.Tensor((1, 128), dtype="float32"), w: R.Tensor((128, 10), dtype="float32"), b: R.Tensor((10,), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        R.func_attr({"Primitive": 1})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.matmul1, (x, w), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            gv = R.call_tir(cls.add1, (lv, b), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(gv)
        return gv

    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][0], axes=None)
            lv2: R.Tensor((1, 128), dtype="float32") = cls.fused_dense_add0(x, lv, metadata["relax.expr.Constant"][1])
            lv3 = R.call_tir(cls.relu, (lv2,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][2], axes=None)
            lv6: R.Tensor((1, 10), dtype="float32") = cls.fused_dense_add1(lv3, lv4, metadata["relax.expr.Constant"][3])
            gv: R.Tensor((1, 10), dtype="float32") = lv6
            R.output(gv)
        return gv
```

在上面的 IRModule 中 `fused_matmul_add0` 和 `fused_matmul_add1` 仍然是 relax 函数，它们调用相应的 TensorIR `matmul` 和 `add` 函数。 我们可以将它们变成一个单一的 TensorIR 函数。

```python
MLPModelFinal = relax.transform.FuseTIR()(MLPModelTIR)
MLPModelFinal.show()

#-----------------------
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def fused_dense_add0(x: T.Buffer((T.int64(1), T.int64(784)), "float32"), w: T.Buffer((T.int64(784), T.int64(128)), "float32"), b: T.Buffer((T.int64(128),), "float32"), T_add_intermediate: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_NN_intermediate = T.alloc_buffer((T.int64(1), T.int64(128)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(128), T.int64(784)):
            with T.block("T_matmul_NN"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], w[v_k, v_i1])
                T.writes(T_matmul_NN_intermediate[v_i0, v_i1])
                with T.init():
                    T_matmul_NN_intermediate[v_i0, v_i1] = T.float32(0.0)
                T_matmul_NN_intermediate[v_i0, v_i1] = T_matmul_NN_intermediate[v_i0, v_i1] + x[v_i0, v_k] * w[v_k, v_i1]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_matmul_NN_intermediate[v_ax0, v_ax1], b[v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = T_matmul_NN_intermediate[v_ax0, v_ax1] + b[v_ax1]

    @T.prim_func(private=True)
    def fused_dense_add1(x: T.Buffer((T.int64(1), T.int64(128)), "float32"), w: T.Buffer((T.int64(128), T.int64(10)), "float32"), b: T.Buffer((T.int64(10),), "float32"), T_add_intermediate: T.Buffer((T.int64(1), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        T_matmul_NN_intermediate = T.alloc_buffer((T.int64(1), T.int64(10)))
        for i0, i1, k in T.grid(T.int64(1), T.int64(10), T.int64(128)):
            with T.block("T_matmul_NN"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(x[v_i0, v_k], w[v_k, v_i1])
                T.writes(T_matmul_NN_intermediate[v_i0, v_i1])
                with T.init():
                    T_matmul_NN_intermediate[v_i0, v_i1] = T.float32(0.0)
                T_matmul_NN_intermediate[v_i0, v_i1] = T_matmul_NN_intermediate[v_i0, v_i1] + x[v_i0, v_k] * w[v_k, v_i1]
        for ax0, ax1 in T.grid(T.int64(1), T.int64(10)):
            with T.block("T_add"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(T_matmul_NN_intermediate[v_ax0, v_ax1], b[v_ax1])
                T.writes(T_add_intermediate[v_ax0, v_ax1])
                T_add_intermediate[v_ax0, v_ax1] = T_matmul_NN_intermediate[v_ax0, v_ax1] + b[v_ax1]

    @T.prim_func(private=True)
    def relu(lv2: T.Buffer((T.int64(1), T.int64(128)), "float32"), compute: T.Buffer((T.int64(1), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(1), T.int64(128)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv2[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(lv2[v_i0, v_i1], T.float32(0.0))

    @R.function
    def main(x: R.Tensor((1, 784), dtype="float32")) -> R.Tensor((1, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv: R.Tensor((784, 128), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][0], axes=None)
            lv2 = R.call_tir(cls.fused_dense_add0, (x, lv, metadata["relax.expr.Constant"][1]), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv3 = R.call_tir(cls.relu, (lv2,), out_sinfo=R.Tensor((1, 128), dtype="float32"))
            lv4: R.Tensor((128, 10), dtype="float32") = R.permute_dims(metadata["relax.expr.Constant"][2], axes=None)
            gv = R.call_tir(cls.fused_dense_add1, (lv3, lv4, metadata["relax.expr.Constant"][3]), out_sinfo=R.Tensor((1, 10), dtype="float32"))
            R.output(gv)
        return gv
```
