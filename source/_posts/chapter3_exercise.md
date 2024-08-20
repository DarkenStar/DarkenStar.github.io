---
title: TVM Learning (6)-Exercise of End to End Model Execution
date: 2024/8/20 12:45:42
categories: TVM
tags: TVM learning
excerpt: Personal answer of E2E Model Execution
mathjax: true
katex: true
---
# Model Preparation

我们采用Pytorch框架先定一个模型，该模型接受一批图像为输入，然后对它们依次作用卷积层，激活层，池化层和全连接层，得到分类结果。并从训练好的模型里加载权重，输入图像来自FashionMNIST数据集，shape为(1, 28, 28)，我们设置batch size=4.

```python
# Load the weight map from file.
weight_map = pkl.load(open("fasionmnist_mlp_assignment_params.pkl", "rb"))
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def pytorch_model():
    list = []
    list.append(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), bias=True))
    list.append(nn.ReLU())
    list.append(nn.MaxPool2d(kernel_size=(2, 2)))
    list.append(nn.Flatten())
    list.append(nn.Linear(in_features=5408, out_features=100, bias=True))
    list.append(nn.ReLU())
    list.append(nn.Linear(in_features=100, out_features=10, bias=True))
    list.append(nn.Softmax(dim=1))
  
    model = nn.Sequential(*list).cuda()
    name_map = {
        "0.weight": "conv2d_weight",
        "0.bias": "conv2d_bias",
        "4.weight": "linear0_weight",
        "4.bias": "linear0_bias",
        "6.weight": "linear1_weight",
        "6.bias": "linear1_bias",
    }
    for name, param in model.named_parameters():
        param.data = torch.from_numpy(weight_map[name_map[name]]).cuda()
    return model
```

# Ingest Model From Pytorch

之前我们都是手写T.prim_func来实现神经网络的每一层，这样很容易出错并且不易于调试。TVM提供了 `relax.BlockBuilder`类可以从头开始一步步构造端到端模型，其中有一个名为 `emit_te`的API，它可以将一个张量表达式的算子描述转变成一个对应TensorIR函数的 `call_tir`操作。

在下面的代码中，为了构建一个执行单个ReLU算子的Relax函数，在 `emit_te_example`中我们首先定义了一个 `BlockBuilder`实例 `bb`。同样定义了一个 `128x128`大小的张量变量 `x`，它将作为ReLU操作的输入（同时也是Relax函数的输入）。

在这之后，我们用 `with bb.function(name, [*input])` API构建一个以 `x`为输入的Relax函数 `main`。然后我们构建一个dataflow block。在这个dataflow block里，我们首先用 `emit_te`生成一个调用ReLU算子的 `call_tir`。 `emit_te`会在IRModule中生成一个名字为 `relu`的TensorIR函数，然后在dataflow block中生成 `call_tir(relu, (x,), (128, 128), dtype="float32")`操作。`call_tir`之后是函数返回。在这一构造之后，BlockBuilder实例 `bb`包含构建完的IRModule，可以通过 `bb.get()`得到。

`emit_te` 的作用是将一个 TVM 张量表达式（TE）函数转换为 Relax 中的调用节点（Call Node）。它允许你在 Relax 中使用 TE 函数来进行计算，并生成相应的 TVM Script 代码。该函数首先将 Relax 表达式的参数转换为 TE 张量。然后，它调用 TE 函数，并将转换后的 TE 张量作为参数传递给它。TE 函数执行计算并返回一个 TE 张量或 TE 张量列表。该函数将返回的 TE 张量转换为 Relax 中的 Call Node. 最后，它使用 `self.emit` 方法将调用节点添加到 Relax BlockBuilder 中，并返回一个新的 Relax 变量，该变量绑定到 Call Node.

**函数参数：**

* `func`: 一个可调用对象，它代表一个 TE 函数，该函数接受 Relax 张量作为参数，并返回一个 TE 张量或 TE 张量列表。
* `*args`: `func`输入的位置参数 (relax Tensor)。
* `**kwargs`: `func`输入的的关键字参数 (relax Tensor)。
* `name_hint`: 可选参数，用于指定生成的 PrimFunc 的名称。

```python
def relu(A):
    B = te.compute(shape=(128, 128), fcompute=lambda i, j: te.max(A[i, j], 0), name="B")
    return B

def emit_te_example():
    # relax.BlockBuilder can construct e2e models 
    # step by step in an IRModule that starts empty.
    bb =relax.BlockBuilder()  
    # relax.DynTensorType is the type assigned to tensors with a known dtype and unknown shape.
    x = relax.Var("x", relax.TensorStructInfo((128, 128), "float32"))
    with bb.function("main", [x]):  # construct a Relax function main with x as input
        with bb.dataflow():
            # Emit a call node according to the te function
            # which should return a te tensor or a list of te tensors. 
            lv0 = bb.emit_te(relu, x)
            gv = bb.emit_output(lv0)  # mark the dataflow output 
        bb.emit_func_output(gv)  # mark the function output 
    return bb.get()  # return the constructed IRModule
```

可以看到通过BlockBuilder生成的IRModule包含了ReLU的TensorIR实现和一个含有调用ReLU实现的 `call_tir`的Relax函数

```python
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def relu(x: T.Buffer((T.int64(128), T.int64(128)), "float32"), B: T.Buffer((T.int64(128), T.int64(128)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(T.int64(128), T.int64(128)):
            with T.block("B"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(x[v_i, v_j])
                T.writes(B[v_i, v_j])
                B[v_i, v_j] = T.max(x[v_i, v_j], T.float32(0.0))

    @R.function
    def main(x: R.Tensor((128, 128), dtype="float32")) -> R.Tensor((128, 128), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.relu, (x,), out_sinfo=R.Tensor((128, 128), dtype="float32"))
            gv: R.Tensor((128, 128), dtype="float32") = lv
            R.output(gv)
        return gv
```

# Construct IRModule Equals to Pytorch

我们可以用 `BlockBuilder`和 `emit_te`来创建一个和之前定义的PyTorch模型等价的IRModule。首先我们要实现这些算子的张量表达式运算函数。

{% note warning %}

在加上bias的时候要和reduction操作分开进行，即不能在一个te.compute里面进行 `te.sum+bias[...]`的操作，否则会报错

```bash
TVMError
Traceback (most recent call last):
File "D:\Work\tvm\tvm0.18\tvm\src\te\operation\compute_op.cc", line 566
InternalError: Check failed: (0 == level_) is false: Reductions are only allowed at the top level of compute. Please create another tensor for further composition.
```

{% endnote %}

```python
def my_conv2d(X, K, B):  # No padding, stride = 1
    N, CI, H, W = X.shape
    CO, _, KH, KW = K.shape
    k = te.reduce_axis((0, CI), name="k")
    r = te.reduce_axis((0, KH), name="r")
    s = te.reduce_axis((0, KW), name="s")
    OH = (H - KH) + 1
    OW = (W - KW) + 1
    conv2d_te = te.compute(shape=(N, CO, OH, OW),
                           fcompute=lambda n, co, oh, ow: te.sum(
                                    X[n, k, oh + r, ow + s] * K[co, k, r, s], axis=[k, r, s]), 
                           name="conv2d")
    out = te.compute(shape=(N, CO, OH, OW), fcompute=lambda n, co, oh, ow:
                                                    conv2d_te[n, co, oh, ow] + B[0, co, 0, 0])
    return out

def my_relu(X):
    return te.compute(shape=X.shape, fcompute=lambda *i: te.max(X(*i), 0))

def my_maxpool2d(X, S):
    N, C, H, W = X.shape
    i = te.reduce_axis((0, S), name="i")
    j = te.reduce_axis((0, S), name="j")
    maxpool2d_te = te.compute(shape=(N, C, H//2, W//2), 
                              fcompute=lambda n, co, oh, ow: te.max(
                                  X[n, co, oh*S+i, ow*S+j], axis=[i, j]), name="maxpool2d")
    return maxpool2d_te

def my_flatten(X):
    N, C, H, W = X.shape
    flatten_te = te.compute(shape=(N, C*H*W), 
                            fcompute=lambda n, i: 
                                X[n, i//(H*W), i//(W)%(H), i%(W)])
    return flatten_te

def my_linear(X, W, B=None):
    FO, FI = W.shape 
    N, _ = X.shape  
    fi = te.reduce_axis((0, FI), name="FI")
    linear_te = te.compute(shape=(N, FO), fcompute=lambda i, j: te.sum(
                                                X[i, fi] * W[j, fi], axis=fi))
    if B is not None:
        out = te.compute(shape=(N, FO), fcompute=lambda i, j: B[0, j] + linear_te[i, j])
    else:
        out = linear_te
    return out   

def my_softmax(X):
    N, C = X.shape
    c = te.reduce_axis((0, C), name="c")
    max_val = te.compute(shape=(N, ), fcompute=lambda i: te.max(X[i, c], axis=c))
    exp_te = te.compute(shape=(N, C), fcompute=lambda i, j: te.exp(X[i, j] - max_val[i]))  
    sum_exp_te = te.compute(shape=(N, ), fcompute=lambda i: te.sum(exp_te[i, c], axis=c))
    softmax_te = te.compute(shape=(N, C), fcompute=lambda i, j: exp_te[i, j] / sum_exp_te[i])
    return softmax_te
```

然后我们就可以利用`BlockBuilder`构建IRModule

```python
def create_model_via_emit_te():
    batch_size = 4
    input_shape = (batch_size, 1, 28, 28)  # BCHW
    bb = relax.BlockBuilder()
    x = relax.Var("x", relax.TensorStructInfo(input_shape, "float32"))
  
    conv2d_weight = relax.const(weight_map["conv2d_weight"], "float32")
    conv2d_bias = relax.const(weight_map["conv2d_bias"].reshape(1, 32, 1, 1), "float32")
    linear0_weight = relax.const(weight_map["linear0_weight"], "float32")
    linear0_bias = relax.const(weight_map["linear0_bias"].reshape(1, 100), "float32")
    linear1_weight = relax.const(weight_map["linear1_weight"], "float32")
    linear1_bias = relax.const(weight_map["linear1_bias"].reshape(1, 10), "float32")
  
    # Build the model using BlockBuilder
    with bb.function("main", [x]):
        with bb.dataflow():
            gv_conv = bb.emit_te(my_conv2d, x, conv2d_weight, conv2d_bias)
            gv_relu1 = bb.emit_te(my_relu, gv_conv)
            gv_pool = bb.emit_te(my_maxpool2d, gv_relu1, 2)
            gv_flatten = bb.emit_te(my_flatten, gv_pool)
            gv_dense1 = bb.emit_te(my_linear, gv_flatten, linear0_weight, linear0_bias)   
            gv_relu2 = bb.emit_te(my_relu, gv_dense1)
            gv_dense2 = bb.emit_te(my_linear, gv_relu2, linear1_weight, linear1_bias)
            gv_softmax = bb.emit_te(my_softmax, gv_dense2)
            out = bb.emit_output(gv_softmax)
        bb.emit_func_output(out)

    return bb.get()
```

得到的IRModule的TensorIR如下

```python
mod = create_model_via_emit_te()   
exec = relax.build(mod, "llvm")
dev = tvm.cpu()
vm = relax.VirtualMachine(exec, dev)
print(mod.script())
```

{% fold info @mod.script %}

```python
@I.ir_module
class Module:
    @T.prim_func(private=True)
    def my_conv2d(x: T.Buffer((T.int64(4), T.int64(1), T.int64(28), T.int64(28)), "float32"), B: T.Buffer((T.int64(32), T.int64(1), T.int64(3), T.int64(3)), "float32"), C: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(1)), "float32"), compute: T.Buffer((T.int64(4), T.int64(32), T.int64(26), T.int64(26)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        conv2d = T.alloc_buffer((T.int64(4), T.int64(32), T.int64(26), T.int64(26)))
        for n, co, oh, ow, k, r, s in T.grid(T.int64(4), T.int64(32), T.int64(26), T.int64(26), T.int64(1), T.int64(3), T.int64(3)):
            with T.block("conv2d"):
                v_n, v_co, v_oh, v_ow, v_k, v_r, v_s = T.axis.remap("SSSSRRR", [n, co, oh, ow, k, r, s])
                T.reads(x[v_n, v_k, v_oh + v_r, v_ow + v_s], B[v_co, v_k, v_r, v_s])
                T.writes(conv2d[v_n, v_co, v_oh, v_ow])
                with T.init():
                    conv2d[v_n, v_co, v_oh, v_ow] = T.float32(0.0)
                conv2d[v_n, v_co, v_oh, v_ow] = conv2d[v_n, v_co, v_oh, v_ow] + x[v_n, v_k, v_oh + v_r, v_ow + v_s] * B[v_co, v_k, v_r, v_s]
        for n, co, oh, ow in T.grid(T.int64(4), T.int64(32), T.int64(26), T.int64(26)):
            with T.block("compute"):
                v_n, v_co, v_oh, v_ow = T.axis.remap("SSSS", [n, co, oh, ow])
                T.reads(conv2d[v_n, v_co, v_oh, v_ow], C[T.int64(0), v_co, T.int64(0), T.int64(0)])
                T.writes(compute[v_n, v_co, v_oh, v_ow])
                compute[v_n, v_co, v_oh, v_ow] = conv2d[v_n, v_co, v_oh, v_ow] + C[T.int64(0), v_co, T.int64(0), T.int64(0)]

    @T.prim_func(private=True)
    def my_flatten(lv2: T.Buffer((T.int64(4), T.int64(32), T.int64(13), T.int64(13)), "float32"), compute: T.Buffer((T.int64(4), T.int64(5408)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n, i in T.grid(T.int64(4), T.int64(5408)):
            with T.block("compute"):
                v_n, v_i = T.axis.remap("SS", [n, i])
                T.reads(lv2[v_n, v_i // T.int64(169), v_i % T.int64(169) // T.int64(13), v_i % T.int64(13)])
                T.writes(compute[v_n, v_i])
                compute[v_n, v_i] = lv2[v_n, v_i // T.int64(169), v_i % T.int64(169) // T.int64(13), v_i % T.int64(13)]

    @T.prim_func(private=True)
    def my_linear(lv3: T.Buffer((T.int64(4), T.int64(5408)), "float32"), B: T.Buffer((T.int64(100), T.int64(5408)), "float32"), C: T.Buffer((T.int64(1), T.int64(100)), "float32"), compute: T.Buffer((T.int64(4), T.int64(100)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        compute_1 = T.alloc_buffer((T.int64(4), T.int64(100)))
        for i, j, FI in T.grid(T.int64(4), T.int64(100), T.int64(5408)):
            with T.block("compute"):
                v_i, v_j, v_FI = T.axis.remap("SSR", [i, j, FI])
                T.reads(lv3[v_i, v_FI], B[v_j, v_FI])
                T.writes(compute_1[v_i, v_j])
                with T.init():
                    compute_1[v_i, v_j] = T.float32(0.0)
                compute_1[v_i, v_j] = compute_1[v_i, v_j] + lv3[v_i, v_FI] * B[v_j, v_FI]
        for i, j in T.grid(T.int64(4), T.int64(100)):
            with T.block("compute_1"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(C[T.int64(0), v_j], compute_1[v_i, v_j])
                T.writes(compute[v_i, v_j])
                compute[v_i, v_j] = C[T.int64(0), v_j] + compute_1[v_i, v_j]

    @T.prim_func(private=True)
    def my_linear1(lv5: T.Buffer((T.int64(4), T.int64(100)), "float32"), B: T.Buffer((T.int64(10), T.int64(100)), "float32"), C: T.Buffer((T.int64(1), T.int64(10)), "float32"), compute: T.Buffer((T.int64(4), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        compute_1 = T.alloc_buffer((T.int64(4), T.int64(10)))
        for i, j, FI in T.grid(T.int64(4), T.int64(10), T.int64(100)):
            with T.block("compute"):
                v_i, v_j, v_FI = T.axis.remap("SSR", [i, j, FI])
                T.reads(lv5[v_i, v_FI], B[v_j, v_FI])
                T.writes(compute_1[v_i, v_j])
                with T.init():
                    compute_1[v_i, v_j] = T.float32(0.0)
                compute_1[v_i, v_j] = compute_1[v_i, v_j] + lv5[v_i, v_FI] * B[v_j, v_FI]
        for i, j in T.grid(T.int64(4), T.int64(10)):
            with T.block("compute_1"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(C[T.int64(0), v_j], compute_1[v_i, v_j])
                T.writes(compute[v_i, v_j])
                compute[v_i, v_j] = C[T.int64(0), v_j] + compute_1[v_i, v_j]

    @T.prim_func(private=True)
    def my_maxpool2d(lv1: T.Buffer((T.int64(4), T.int64(32), T.int64(26), T.int64(26)), "float32"), maxpool2d: T.Buffer((T.int64(4), T.int64(32), T.int64(13), T.int64(13)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for n, co, oh, ow, i, j in T.grid(T.int64(4), T.int64(32), T.int64(13), T.int64(13), T.int64(2), T.int64(2)):
            with T.block("maxpool2d"):
                v_n, v_co, v_oh, v_ow, v_i, v_j = T.axis.remap("SSSSRR", [n, co, oh, ow, i, j])
                T.reads(lv1[v_n, v_co, v_oh * T.int64(2) + v_i, v_ow * T.int64(2) + v_j])
                T.writes(maxpool2d[v_n, v_co, v_oh, v_ow])
                with T.init():
                    maxpool2d[v_n, v_co, v_oh, v_ow] = T.float32(-340282346638528859811704183484516925440.0)
                maxpool2d[v_n, v_co, v_oh, v_ow] = T.max(maxpool2d[v_n, v_co, v_oh, v_ow], lv1[v_n, v_co, v_oh * T.int64(2) + v_i, v_ow * T.int64(2) + v_j])

    @T.prim_func(private=True)
    def my_relu(lv: T.Buffer((T.int64(4), T.int64(32), T.int64(26), T.int64(26)), "float32"), compute: T.Buffer((T.int64(4), T.int64(32), T.int64(26), T.int64(26)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1, i2, i3 in T.grid(T.int64(4), T.int64(32), T.int64(26), T.int64(26)):
            with T.block("compute"):
                v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
                T.reads(lv[v_i0, v_i1, v_i2, v_i3])
                T.writes(compute[v_i0, v_i1, v_i2, v_i3])
                compute[v_i0, v_i1, v_i2, v_i3] = T.max(lv[v_i0, v_i1, v_i2, v_i3], T.float32(0.0))

    @T.prim_func(private=True)
    def my_relu1(lv4: T.Buffer((T.int64(4), T.int64(100)), "float32"), compute: T.Buffer((T.int64(4), T.int64(100)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i0, i1 in T.grid(T.int64(4), T.int64(100)):
            with T.block("compute"):
                v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
                T.reads(lv4[v_i0, v_i1])
                T.writes(compute[v_i0, v_i1])
                compute[v_i0, v_i1] = T.max(lv4[v_i0, v_i1], T.float32(0.0))

    @T.prim_func(private=True)
    def my_softmax(lv6: T.Buffer((T.int64(4), T.int64(10)), "float32"), compute: T.Buffer((T.int64(4), T.int64(10)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        compute_1 = T.alloc_buffer((T.int64(4),))
        compute_2 = T.alloc_buffer((T.int64(4), T.int64(10)))
        compute_3 = T.alloc_buffer((T.int64(4),))
        for i, c in T.grid(T.int64(4), T.int64(10)):
            with T.block("compute"):
                v_i, v_c = T.axis.remap("SR", [i, c])
                T.reads(lv6[v_i, v_c])
                T.writes(compute_1[v_i])
                with T.init():
                    compute_1[v_i] = T.float32(-340282346638528859811704183484516925440.0)
                compute_1[v_i] = T.max(compute_1[v_i], lv6[v_i, v_c])
        for i, j in T.grid(T.int64(4), T.int64(10)):
            with T.block("compute_1"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv6[v_i, v_j], compute_1[v_i])
                T.writes(compute_2[v_i, v_j])
                compute_2[v_i, v_j] = T.exp(lv6[v_i, v_j] - compute_1[v_i])
        for i, c in T.grid(T.int64(4), T.int64(10)):
            with T.block("compute_2"):
                v_i, v_c = T.axis.remap("SR", [i, c])
                T.reads(compute_2[v_i, v_c])
                T.writes(compute_3[v_i])
                with T.init():
                    compute_3[v_i] = T.float32(0.0)
                compute_3[v_i] = compute_3[v_i] + compute_2[v_i, v_c]
        for i, j in T.grid(T.int64(4), T.int64(10)):
            with T.block("compute_3"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(compute_2[v_i, v_j], compute_3[v_i])
                T.writes(compute[v_i, v_j])
                compute[v_i, v_j] = compute_2[v_i, v_j] / compute_3[v_i]

    @R.function
    def main(x: R.Tensor((4, 1, 28, 28), dtype="float32")) -> R.Tensor((4, 10), dtype="float32"):
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.my_conv2d, (x, metadata["relax.expr.Constant"][0], metadata["relax.expr.Constant"][1]), out_sinfo=R.Tensor((4, 32, 26, 26), dtype="float32"))
            lv1 = R.call_tir(cls.my_relu, (lv,), out_sinfo=R.Tensor((4, 32, 26, 26), dtype="float32"))
            lv2 = R.call_tir(cls.my_maxpool2d, (lv1,), out_sinfo=R.Tensor((4, 32, 13, 13), dtype="float32"))
            lv3 = R.call_tir(cls.my_flatten, (lv2,), out_sinfo=R.Tensor((4, 5408), dtype="float32"))
            lv4 = R.call_tir(cls.my_linear, (lv3, metadata["relax.expr.Constant"][2], metadata["relax.expr.Constant"][3]), out_sinfo=R.Tensor((4, 100), dtype="float32"))
            lv5 = R.call_tir(cls.my_relu1, (lv4,), out_sinfo=R.Tensor((4, 100), dtype="float32"))
            lv6 = R.call_tir(cls.my_linear1, (lv5, metadata["relax.expr.Constant"][4], metadata["relax.expr.Constant"][5]), out_sinfo=R.Tensor((4, 10), dtype="float32"))
            lv7 = R.call_tir(cls.my_softmax, (lv6,), out_sinfo=R.Tensor((4, 10), dtype="float32"))
            gv: R.Tensor((4, 10), dtype="float32") = lv7
            R.output(gv)
        return gv
```

{% endfold %}

我们可以与Pytorch模型的执行结果进行比较来验证正确性。

```python
def build_mod(mod):
    exec = relax.vm.build(mod, "llvm")
    dev = tvm.cpu()
    vm = relax.VirtualMachine(exec, dev)
    return vm


def check_equivalence(mod, torch_model, test_loader):
    torch_model.eval()
    with torch.no_grad():
        rt_mod = build_mod(mod)
        for data, label in test_loader:
            data, label = data.cpu(), label.cpu()
            output_from_pytorch = torch_model(data).numpy()
            output_from_relax = rt_mod["main"](tvm.nd.array(data, tvm.cpu())).numpy()
            tvm.testing.assert_allclose(output_from_pytorch, output_from_relax, rtol=1e-4)


test_data = torchvision.datasets.FashionMNIST(
    "./data",
    download=True,
    train=False,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

mod = create_model_via_emit_te()
torch_model = pytorch_model()

check_equivalence(mod, torch_model, test_loader)
```
