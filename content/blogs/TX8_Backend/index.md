---
title: "TX8 Backend"
date: 2025-07-23T11:49:02+08:00
lastmod: 2025-07-23T11:49:02+08:00
author: ["WITHER"]

categories:
- tx8

tags:
- tx8

keywords:


description: "TX8 backend description." # 文章描述，与搜索优化相关
summary: "TX8 backend description." # 文章简单描述，会展示在主页
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

# TX8 Hardware Overview

TX8 采用的是空间计算型结构 (Special Computing Architecture)，市面上普遍采用的共享内存结构 (Shared Memory Architecture)，它的数据通信交互主要是依赖于 DDR，一个 thread 把 DDR 的数据改变之后，另外一个 thread 再从 DDR 中才能得知到这个数据已经被改变。这么做有一个很明显的缺陷，就是它瓶颈在于内存容量以及访问内存的带宽延迟。空间计算型的结构它是由中间的NOC (Network On Chip) 来构成模块之间的互联。这样很好的避免了这个 DDR 的瓶颈，同时也有了更好的 scale out 能力。

![(a) Shared Memory Architecture (b) Spatial Computing Architecture](https://share.note.youdao.com/yws/api/personal/file/WEB04f094c9d80d3990221020a51ce93433?method=download&shareKey=f6149480beae12e20d31f124e425ef84 "(a) Shared Memory Architecture (b) Spatial Computing Architecture")

下图为 TX8 两个芯片互连的逻辑结构。每个芯片由 4x4 总计 16 Tile 以 mesh 拓扑结构进行互连。每一个 Tile 是一个计算核心，是一个图灵完备 (Turing Complete) 的系统，既具有调度控制以及计算通信以及存储的能力。片上 NoC 采用的是 stream (一种轻量级 DMA 技术). 片上 DDR 大小为 64GB，芯片之间是通过 high speed IO 进行互连的。

![Tile](https://share.note.youdao.com/yws/api/personal/file/WEB98d1d8b5916da28e40cf3c77b63dcbe6?method=download&shareKey=1ca36bb999bfe3f524fc2525e1cb0ab7 "Tile")

单芯片与单卡 A100 性能对比如下表所示

| TX8         | 单卡性能  | 最大组网性能   | A100 | 单卡性能|
|--------------|------------------|----------------------|--|--------|
|**INT8**     | 256T   | 1E       | | 624T
|**BF16**   | 128T | 0.5E      | | 312T |
|**TFP32**    | 128T  | 0.5E       | | 156T |
|**FP32**    | 21T   | 40P     | | 19.5T |
| **内存带宽** | 200GB/s   | -  | **显存带宽**| 1935GB/s
| **PCIe**     | 64GB/s    | - | | 64GB/s | 
| **内存容量** | 64GB      | 128TB    | **显存容量** | 80GB  |
| **TsingMicro-Link** | 1600Gbps | -  | **NV-Link**  | 600GB/s |



# Single Tile 

下图是单 Tile 的硬件结构，实际上每个 Tile 上会有两个 kernel core 和 special core，图中只画了一个。还有个 neural core，主要是负责计算以及数据搬运等等。

![Tile Microarchitecture](https://share.note.youdao.com/yws/api/personal/file/WEB8e08cce47b2de4d373c8dd667a988463?method=download&shareKey=ba34989999d6febf20e7abeda09a81b2 "Tile Microarchitecture")

- kernel core 主要用于下发指令。它会从 DDR 中取址，然后送到这个 neural core 的 NCC controller 里面。NCC controller 又会把根据这个指令的类型下发到 CT/NE/LSU. 他们三个是执行不同种类指令的三个小模块，后面会讲到。这三个小模块会从 SPM (Scratched Pad Memory) 上读取数据，然后再计算，或者再存回 SPM上。值得注意的是，LSU 是用来负责这个数据搬运的，所以它可以把这个 SPM 上的数据直接搬到DDR，或者是从 DDR 搬到 SPM 上。CT 和 NE 都是负责计算的模块，其中 scalar unit 位于 NCC controller，是一个负责标量计算的模块。

- special core 用来和 NOC 进行连接，它可以从 DDR 中读取数据，然后通过配置 DTE 模块和这个远程的 Tile 进行通信。DTE 模块也可以通过 special core 将本 Tile 上的 SPM 与远程 Tile 上的 SPM 进行通信。

# CGRA Tensor

CGRA Tensor 模块支持算术运算，逻辑运算，位操作，激活函数，超越函数，规约，池化，数据搬移，格式转换，辅助计算。

![CGRA](https://share.note.youdao.com/yws/api/personal/file/WEB9086e7454a92ac6077331ed5c7f4fc56?method=download&shareKey=834faad50224b52bd4c82ccf738f5293 "CGRA")

Neural Core Controller 下发指令到 CTRL_UNIT，然后 CTRL_UNIT 下发指令到 RAM_ACC_UNIT. RAM_ACC_UNIT 读入 SPM 的数据，然后送入 Pipe Unit 进行运算之后把结果存回 SPM. 

CGRA 指令格式如下。例如 CGRATensor_ArithOp_V_V_abs，指令操作指的是对向量元素求绝对值。

| 指令格式 | CGRATensor_function_format_name.type |
|---------|-------------------------------------|
| **Function** | 描述该单元的主要功能，如算数运算、关系运算、逻辑运算等； |
| **Format**   | 描述数据的存储方式，如VV、VS、Tensor、VuV 分别表示</br>向量与向量计算、向量与标量计算、Tensor计算、向量与单元向量计算； |
| **Name**     | 描述具体的操作，如加、减、乘、除等； |
| **Type**     | 表示数据类型，如 bf16/fp32 等； |

下面具体讲一下在 BN 算子开发中用到的 `CGRATensor_ArithOp_V_VuV_mul_loop (bf16 *src, bf16 *dst, bf16 *unit, int rnd, int src_elem_num, int unit_elem_num, int full_src_elem_num, int  full_unit_elem_num)`.

- src/dst/unit 分别表示 也是原数据/存数/单元向量的地址。
- src_elem_num 是做一次这个 VuV 中原数据的个数。
- unit_elem_num 是做一次这个 VuV 中单元向量数据的个数。

在讲 VuV_mul_loop 之前，先来看一下这个 VuV_mul 也就是没有循环的单次版本。分为两次进行，第一次是前四个蓝色的方块与橙色方块相乘，第二次为后四个蓝色方块与橙色方块相乘。VuV_mul_loop 即把这个过程重复很多次，所以要求 `full_src_elem_num/full_unit_elem_num == src_elem_num/unit_elem_num`，并且`unit_elem_num=64`.

![VuV_mul_loop](https://share.note.youdao.com/yws/api/personal/file/WEB367ed5ab2178c80abdeb160cb55b409d?method=download&shareKey=05e534d23de8ab1be6cef33b8c8e0e4e "VuV_mul_loop")

# Tensor Layout

layout 可以分为以下几种
- layout_str: 中端使用
    - CNN Op: 1. Feature (NCHW/NHWC) etc. 2. Weight (OIHW/HWOI) etc.
    - Non-CNN Op: 大模型中常见，Tensor/NTensor，它们的区别是第 0 维是否为 1.
- mem_layout: 后端使用，代表了在芯片上的实际排布  
    - Tensor/NTensor: 数据的紧密排布
    - Cx/NCx: 对 Tensor/NTensor 格式化后的结果，方便易硬件读取。

| dtype | channel | description |
|----------------------|-------|--------------------------------------------------|
| bf16/fp16 <br>/fp32/tf32 | c <= 32 | NHWC, C向4/8/16/32对齐，N 的起始地址向 2048bit 对齐 |
|                      | c > 32  | N[CxHW64, HWC0], C0 向 4/8/16/32 对齐，N 的起始地址向2048bit 对齐<br>在一个 batch 内将 tensor 按 C 分成 Cx*64 和 C0两部分 |
| int8                | c <= 64 | NHWC, C 向 4/8/16/32/64对齐，N的起始地址向2048bit对齐 |
|                      | c > 64  | N[CxHW128, HWC0], C0 向 4/8/16/32/64 对齐，N的起始地址向 2048bit 对齐 <br> 在一个 batch 内将 tensor 按 C 分成 Cx*128 和C0 两部分 |

对于 fp16 的 2x1x2x131 的数据，NTensor 格式存储起始地址为 0x0000 按各存储格式排列如下

![NTensor Layout](https://share.note.youdao.com/yws/api/personal/file/WEB47732820f52aa7a0bfd0d22bb2e61e00?method=download&shareKey=03c9e8f103abf576755a3e33e5d5cbc5 "NTensor Layout")

NCx: 131 = 64 x 2 + 3, 将 C 分成 2(Cx) 个 64 和 4(C0). batch0 的结束地址是 0x1080 (4224), batch1 起始地址需对齐到 2048bit，即 4224-->2048*3=6144 (0x1800).

![NCx Layout](https://share.note.youdao.com/yws/api/personal/file/WEB53406947a6dafa2bbe98df922f46a954?method=download&shareKey=3e75616e0238e7360bd6930b98941909 "NCx Layout")

# Neural Engine

Neural engine 类似于 GPU Tensor Core，主要是完成各种矩阵 (op_Gemm) 和卷积 (op_Conv) 类型的高效并行 Tensor 计算。PE Array 它的进行矩阵运算的部分，一次完成 8x16x8 大小的矩阵乘法。然后它的输入有激活 input，还有 psum，还有 weight，也就是权重。

计算之后，还饿可以进行后处理，对这个结果进行 BN/量化/激活等等，然后再到输出，然后我们要用到neural engine 的算子其实并不多，只有 op_Gemm 和 op_Conv. 

![Neural Engine](https://share.note.youdao.com/yws/api/personal/file/WEBf07a7034750a8723bf35f5cb311251e2?method=download&shareKey=d0c242fdf504441bb4284c72b48908c4 "Neural Engine")

# LSU

LSU 是负责数据搬运的 DMA 控制器。具体它有三部分: 
- RDMA: Read DDR --> SPM，对应指令有 op_loadVar，op_loadConst，op_rdmaGather.
- WDMA: Write SPM --> DDR，对应指令有 op_dma_store，op_wdmaScatter.
- TDMA: 对所属 Tile SPM 上的数据进行操作，对应指令有 op_reshape，op_gatherScatter.

![LSU](https://share.note.youdao.com/yws/api/personal/file/WEBd7f94b1349b939c2c20a37abf5e57bbc?method=download&shareKey=fb7dd9facaf7ca29424c72eaa4991000 "LSU")

一种经常使用 TDMA 的情况是进行低精度到高精度的转换。以 fp16 -> fp32 为例，首先会调用 op_gatherScatter 指令把紧密排布的低精度数据读进来然后 scatter 到 SPM 上的对应位置以保留空间存储转换后的数据；然后再调用 CGAR convert_fp16_fp32 指令进行精度转换。

![fp16 to fp32 Conversion](https://share.note.youdao.com/yws/api/personal/file/WEB1079912e78f03abc7a30d0db12ffb046?method=download&shareKey=20f9d8abd3fb47b3194540d639a2f9ee "fp16 to fp32 Conversion")

# TX8 Compiler

和一般编译器差不多，先获取前端的 Tensorflow/Pytorch 等等生成的 mhlo 计算图，经过中端的处理，然后转到后端。变成后端 IR. 同时又会调用 OPLIB 算子库中的算子来生成 main.c，就是可以直接放在不同平台上运行的主程序。平台可以选择 RISCV 即真实的硬件，或者是 Cmodel 进行模拟。

BEIR 主要是接过中端传进来的 IR，然后进行各类的图优化的 Pass，包括一些算子切分，还有内存调度等等。最终 codegen 这个可编译执行的 main.c 的文件。然后再放在平台上去编译完再运行。

![TX8 Compiler Workflow](https://share.note.youdao.com/yws/api/personal/file/WEB95d5755abd98d22b6bcdaca440c0e8c4?method=download&shareKey=e72432cabeea4cbab293db667ecb0648 "TX8 Compiler Workflow")

# TX8 BE

后端 IR 使用的是 MLIR，继承 Dialect，定义了许多 Operations, Attributes, Types.
```mlir
def Tx8be_Dialect : Dialect {
    let name = "tx8be";
    let summary = "A low-level dialect for tx8 backend specification";
    let cppNamespace = "::tx8be_mir::tx8be";
    let useDefaultAttributePrinterParser = 1;
}
```

## Attribute

下面介绍一些常用的 Attribute.

`parallel_attr` 主要是表示 tensor 每个维度上数据并行和张量并行的切分策略。

```mlir
def Tx8be_ParallelAttr : Tx8be_Attr<"Parallel", "parallel_attr"> {
    let summary = "Structure of parallel information.";
    let parameters = (ins
        "ParallelModeAttr" : $parallel,
        "bool" : $is_dp_inner,    // dp dimension is in the inner, otherwise tp
        "i32" : $dp_dim_x,    // data parallel dimension at x axis
        "i32" : $dp_dim_y,    // data parallel dimension at y axis
        "i32" : $dp_dim_z,    // data parallel dimension at z axis
        "i32" : $tp_dim_x,    // tensor parallel dimension at x axis
        "i32" : $tp_dim_y,    // tensor parallel dimension at y axis
        "i32" : $tp_dim_z,    // tensor parallel dimension at z axis
        "bool" : $sharding_is_given,    // true: is given, false: is not
        "::mlir::DenseI32ArrayAttr" : $shape_spatial_sharding    // Shape split info
    );
    let cppNamespace = "::tx8be_mir::tx8be";
    let assemblyFormat = "`<` struct($params) 1";
}
```

`dev_attr` 属性包含 
- imm_size，也就是用到的这个辅助空间的大小。
- mem_layout 也就是数据的存储数据的排布。
- multi_buf_en 指是否使用 double buffer. 
- out_shape_buf_idx 指的是输出使用第几个缓冲区。
- temporal_mem_slice 是单个 Tile 每次处理的数据大小。

```mlir
def Tx8be_DevAttr : Tx8be_Attr<"Dev", "dev_attr"> {
    let summary = "Structure of op parameters on device.";
    let parameters = (ins
        "uint64_t" : $imm_size,    // Output memory addr offset
        "LayoutModeAttr" : $mem_layout,    // Layout
        "bool" : $multi_buf_en,    // for double buffering
        "int32_t" : $multi_buf_num,    // for double buffering
        "mlir::DenseI64ArrayAttr" : $out_shape_buf_idx,    // index for dynamic shape buffer on runtime
        "mlir::DenseI64ArrayAttr" : $temporal_mem_slice,    // for compute local buffer size
        "int32_t" : $source_type,    // Software pipeline stage
        "int64_t" : $imm_addr, 
        "mlir::DenseI64ArrayAttr" : $mem_addr    // use array for multibuffer
    );
    let cppNamespace = "::tx8be_mir::tx8be";
    let assemblyFormat = "`<` struct($params) `>`";
}
```

`MemScopeMode` 用于描述数据存储在哪里。

```mlir
def Tx8be_MemScopeMode : I32EnumAttr<"MemScopeMode", "Specify the memory scope", 
    [
        I32EnumAttrCase<"DDR", 0>,
        I32EnumAttrCase<"SPM", 1>,
        I32EnumAttrCase<"3DDRAM", 2>
    ]> {
        let genSpecializedAttr = 0;
        let cppNamespace = "::tx8be_mir::tx8be";
    }

```

## Types

定义了很多类型，实际上常用的就是 AnyTensorOrNone.

```mlir
def AnyTensorOrNone: AnyTypeOf<[AnyRankedTensor, NoneType]>;
def Tx8be_Tuple : NestedTupleOf<[AnyRankedTensor]>;
def AnyTensorOrTuple : AnyTypeOf<[AnyRankedTensor, Tx8be_Tuple]>;
def Tx8be_Pred : TypeAlias<I1, "pred (AKA boolean or 1-bit integer)">;
def Tx8be_PredTensor : TensorOf<[Tx8be_Pred]>;
def Tx8be_Token : Type<CPred "{$_self->isa<TokenType>()}", "token">;
def Tx8be_TensorOrTokenOrTuple : AnyTypeOf<[AnyTensor, Tx8be_Token, Tx8be_Tuple]>;
def Tx8be_SInt : SignlessIntOfWidths<[4, 8, 16, 32, 64]>;
def Tx8be_UInt : UnsignedIntOfWidths<[4, 8, 16, 32, 64]>;
def Tx8be_Int : AnyTypeOf<[Tx8be_SInt, Tx8be_UInt]>;
```

## Operations

以开发的 BatchNorm_InferenceOp 为例讲解一下 Tx8be 中关于算子的定义。首先 batchnorm 是将通道维度视作样本，计算其他维度的平均值和方差后进行归一化的操作。

$$
\begin{aligned}
BatchNorm\colon y&=\gamma\:\frac{x-Mean(x)}{\sqrt{Var(x)+\varepsilon}}+\beta\\
Mean(x)&=\frac{1}{N}\sum_{i=1}^{N}x_{i}\\
Var(x)&=\frac{1}{N}\sum_{i=1}^{N}(x_{i}-Mean(x))^{2}\end{aligned}$$

中括号内是一些需要继承的 [Interface](https://mlir.llvm.org/docs/Interfaces/). 其允许 attributes, operations 和 types 公开方法调用接口，而不需要调用者知道特定的派生类型。

arguments 指定了算子需要的输入，包括参数以及之前介绍到的一些属性。

```mlir
def Tx8be_BatchNorm_InferenceOp : Tx8be_Op<"BatchNorm_Inference",
    [DeclareOpInterfaceMethods<oplibinterface>,
    DeclareOpInterfaceMethods<ShardingInterface>,
    DeclareOpInterfaceMethods<ComputeInterface>] {
    let summary = "BatchNorm inference";
    let description = [{
        Normalizes the operand tensor across all dimensions except for the c dimension
        and produce a result tensor.
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$scale,
        AnyTensor:$offset,
        AnyTensor:$mean,
        AnyTensor:$variance,
        DefaultValueOptionalStrAttr<StrAttr, "Unknown">:$layout_str,
        // The following are backend parameters
        OptionalAttr<Tx8be_ParallelAttr>:$chip_parallel,
        OptionalAttr<Tx8be_ParallelAttr>:$tile_parallel,
        OptionalAttr<Tx8be_DevAttr>:$dev_info
    );

    let results = (outs AnyTensor:$output);
}
```
## Interface 

Interface 定义一些通用的方法或行为，这些方法没有具体实现。要通过继承某个 Interface 来具体实现该接口的方法和行为。tx8中定义了 5 个 Interface: OpLibInterface, ComputeInterface, ShapeInferenceOpInterface, ShardingInterface, StreamConfigInterface. 

BatchNorm 算子开发中只用到了前四个，下面依次介绍一下。

`ShapeInferenceOpInterface` 定义了两个方法 `inferShapes` 和 `inferLayout`. 继承这个接口的话就需要实现这两种方法。根主要是根据输入来推断输出的形状和布局。

```mlir
def ShapeInferenceOpInterface : OpInterface<"ShapeInferenceOpInterface"> {
    let description = [{
    }];

    let cppNamespace = "::tx8be_mlir";
    let methods = [
        InterfaceMethod<
            [{ }],
            /*retType=*/"mlir::LogicalResult",
            /*methodName=*/"inferShapes",  // method name
            /*args=*/(ins "DynamicShapeParam" : $shapeParam)
        >,
        InterfaceMethod<
            [{ }],
            /*retType=*/"mlir::LogicalResult",
            /*methodName=*/"inferLayout",  // method name
            /*args=*/(ins)
        >
    ];
}
```

由于 batchnorm 不对这两者进行改变，因此输出和输入相同。如果是需要改变的算子比如 transpose 就需要进行改变。

`input_data <shape=3x4x5x6, layout=NCHW> --> transpose<permutation=(0,2,3,1)> --> output_data<shape=3x5x6x4, layout=NHWC>`

```cpp
// BatchNorm_Interface.cpp
::mlir::LogicalResult tx8be::BatchNorm_InferenceOp::inferLayout() {
    auto in_op = getValidDefiningOp(getInput());
    auto cur_op = getValidDefiningOp(getOutput());
    ASSERT(in_op->hasAttr("layout_str"));
    ASSERT(cur_op->hasAttr("layout_str"));

    auto i_layout = in_op->getAttr("layout_str").cast<mlir::StringAttr>().getValue().str();
    auto ctx = cur_op->getContext();
    cur_op->setAttr("layout_str", mlir::StringAttr::get(ctx, i_layout));

    if (in_op->hasAttr("dev_info")) {
        auto i_dev_layout = getDevInfoLayoutMode(in_op);
        setDevInfoWithLayout(cur_op->getContext(), cur_op, i_dev_layout);
    }
    return ::mlir::success();
}

::mlir::LogicalResult tx8be::BatchNorm_InferenceOp::inferShapes(DynamicShapeParam &shapeParam) {
    tx8be::BatchNorm_InferenceOp::getOutput().setType(getInput().getType());
    return ::mlir::success();
}
```

## ShardingInterface

`tileShardingSplit` 和前面的 `inferShapes` 以及 `inferLayout` 不一样。后两者是从输入信息推出输出的信息。而 `tileShardingSplit` 是由输出的的切分的因子来推断出各个输入的切分因子。

![BatchNorm ShardingInterface](https://share.note.youdao.com/yws/api/personal/file/WEB8d4fed659394922243574186cf74ef3a?method=download&shareKey=d1461507273efadf9613b1496fd1501c "BatchNorm ShardingInterface")

```mlir
def ShardingInterface : OpInterface<"ShardingInterface"> {
    let description = [{
    }];

    let cppNamespace = "::tx8be_mlir";
    let methods = [
        InterfaceMethod<
            /*desc=*/[{ 
            }], 
            // vector for diff operand's info
            /*retType=*/"std::vector<tx8be_mlr::ShardingSplitParam>", 
            /*methodName=*/"tileShardingSplit", 
            /*args=*/(ins "ShardingSplitParam" : $param)
        >,
        InterfaceMethod<
            /*desc=*[{ 
            }], 
            /*retType=*/"std::vector<tx8be_mlr::SliceParam>", 
            /*methodName=*/"temporalSliceShape", 
            /*args=*/(ins "SliceParam" : $param)
        >,
        InterfaceMethod<
            /*desc=*[{ 
            }], 
            /*retType=*/"std::vector<tx8be_mlr::WindowParam>", 
            /*methodName=*/"backWindow", 
            /*args=*/(ins "const WindowParam" : $param)
        >
    ];
}
```

- Sharding 是空间上的切分，意思是将数据分散到不同的 Tile 上。
- Split 是时间上的切分，意思是切分到 Tile 上的将数据按流水线方式轮流进行 load.

`temporalSliceShape` 返回的是 sharding + split 后一个 Tile 上单次处理的数据的实际 shape.

![BatchNorm Sharding Split](https://share.note.youdao.com/yws/api/personal/file/WEB4495c3579079b37d0c2288cc51408601?method=download&shareKey=0513d93d3b51c4782d56c38acb83d0d5 "BatchNorm Sharding Split")
根据 batchnorm 算子定义 input 只能在通道维度上 sharding.
split 有两种选择
1. 对于 input 和 mean，var，scale，shift 都在 C 维度上做相同的切分。
2. 不再 split mean，var，scale，shift，只对 input 的 NHW 进行 split.

这里采用的是后者。由于 mean, variance, scale, shift 都是 1x1x1xC 的张量，因此 split 为 (1, 1, 1, 1). 切分搜索得到的符合要求的 ShardingSplitParam (下图中为 cn3) 会继续向上传递。

![Sharding Split Search](https://share.note.youdao.com/yws/api/personal/file/WEBc472f9ed0d922130ec05d93efed54186?method=download&shareKey=8bf08e0631cd4c7880b9756969fd4bef "Sharding Split Search")

```cpp
std::vector<ShardingSplitParam> tx8be::BatchNorm_InferenceOp::tileShardingSplit(ShardingSplitParam &param) {
    auto shape = getOutput().getType().getShape();
    ASSERT(shape.size() == param.outSharding.size() && shape.size() == param.outSplit.size());
    int32_t shape_size = shape.size();

    std::vector<ShardingSplitParam> result;
    result.emplace_back(param); // input

    for (int32_t i = 0; i < shape_size - 1; ++i) {
        if (result[0].outSharding.size() > 0 && result[0].outSharding[i] != 1) {  // can only shard in dim C
            result[0].outSharding.clear();
        }
        if (result[0].outSplit.size() > 0 && result[0].outSplit[shape_size - 1] != 1) {  // can only split except dim C
            result[0].outSplit.clear();
        }
    }

    ShardingSplitParam paramMean; // scale/shift/mean/variance
    if (result[0].outSharding.size() > 0) {
        paramMean.outSharding = result[0].outSharding;
    }
    paramMean.outSplit = std::vector<int32_t>(shape_size, 1);  // shape is 1x1x1xC，split must be (1, 1, 1, 1)

    ShardingSplitParam paramVar = paramMean;
    ShardingSplitParam paramScale = paramMean;
    ShardingSplitParam paramShift = paramMean;

    result.emplace_back(paramScale);
    result.emplace_back(paramShift);
    result.emplace_back(paramMean);
    result.emplace_back(paramVar);

    return result;
}
```
## OpLibInterface

`OpLibInterface` 有四个方法，
- `genOpCode`: 生成 main.c 文件的时候所调用的一个接口。
- `getOpClockCycle`: 获取 OP 的执行时间。
- `getImmSpSize`: 获取 SPM 上临时空间所需要的大小。
- `queryOpAttr`: 查询这个 OP 的一些属性。

```cpp
def OpLibInterface : OpInterface<"OpLibInterface"> {
    let description = [{
        These are the interfaces for connecting tx8be-oplib
        and codegen.
    }];

    let cppNamespace = "::tx8be_mlir";
    let methods = [
        InterfaceMethod<
            /*desc=*/[{To generate the code of op.}],
            /*retType=*/"std::string",
            /*methodName=*/"genOpCode",
            /*args=*/(ins "OpCodeParam" : $param)
        >,
        InterfaceMethod<
            /*desc=*/[{To get clock cycle of the op.}],
            /*retType=*/"uint64_t",
            /*methodName=*/"getOpClockCycle",
            /*args=*/(ins)
        >,
        InterfaceMethod<
            /*desc=*/[{To get the immediate SPM buffer size.}],
            /*retType=*/"uint32_t",
            "getImmSpSize",
            /*args=*/(ins)
        >,
        InterfaceMethod<
            /*desc=*/[{To get the opAttr info.}],
            /*retType=*/"tx8be_mlr::opAttr",
            /*methodName=*/"queryOpAttr",
            /*args=*/(ins)
        >
    ];
}
```
其中 `queryOpAttr` 接口只需要在对应的接口里给 OpAttr 里的参数赋值。
- `alignMode`: 算子的对齐要求，有Cx对齐要求，NCx 对齐要求，或者不在意存储格式的。
- `defaultLayout`: 算子默认的排布。
- `needPresetToNPU`: OP 是否需要进行预设到和硬件匹配的 layout. 当算子用到的指令是带有 NHWC 的配置时候的需要。
- `memInplace`: 输入和输出能否使用同一片内存。
- `needLoad`: 算子是否需要 load 操作，比如 mask, embedding 就不需要，会跳过loadvar op 生成。bit0 表示 arg idx0，bit1 表示 arg idx1，一共能表示 64 个输入情况。如果是const输入，loadconst 也会跳过codegen 不生成 code.
  > 一个op可能有多个 input 都没有 load，shape 更新只用最后一个没有 load 的 operand (为 0 的最高位). 如 embedding 的 shape使用最后一个 operand，第一个是 weight 不用管 gshape. scatter有的有load，有的没有，shape 更新只看没有 load 的那个。
- `needStore`:  数据是否需要进行 store 操作，会跳过store op 生成。
- `parallel`: 是否允许并行模式。
- `alignCx`: 最低维度切分是否到 64/128 (i8).
```cpp
struct OpAttr {
    ALIGN_MODE alignMode{ALIGN_MODE::NPU_UNKNOWN};  // 算子的对齐要求，有Cx对齐要求，NCx对齐要求，或者不在意存储格式的
    std::string defaultLayout{"Tensor"};           // 算子默认的layout
    bool needPresetToNPU{false};                   // op是否需要进行预设到和硬件匹配的layout. 当算子用到的指令是带有 nhwc 的配置时需要

    ENGINE_TYPE engine{NPU_ENGINE_CT};

    bool memInplace{false};                        // op的输入和输出能否使用同一片memory，比如add的out使用in0的
    uint64_t needLoad{0xFFFFFFFFFFFFFFFF};         // 算子是否需要load操作
    uint64_t needStore{0xFFFFFFFFFFFFFFFF};        // 数据是否需要进行store操作，会跳过store op生成
    bool parallel{1};                              // 一般要使能并行模式，不过有的memory可能有问题，就不使能
    bool alignCx{1};                               // 最低维度切分是否到64/128(i8)
};
```

batchnorm 允许输入 in 的 layout 为 Cx/NCx，要在 mlir 层的 `queryOpAttr()` 里将 alignMode 设置为NPU_ALIGN, 维度为 2/3/4，数据类型为 bf16/fp16/fp32/tf32. 其他输入的格式为 fp32. 输出的维度和类型与 in 保持一致。

```cpp
OpAttr tx8be::BatchNorm_InferenceOp::queryOpAttr() {
    OpAttr attr;  // 创建一个 OpAttr 对象
    attr.alignMode = ALIGN_MODE::NPU_ALIGN;  // 设置对齐模式为 NPU_ALIGN
    attr.needPresetToNPU = true;  // 设置需要预设到 NPU

    // 获取 in 的形状，并判断其第一个维度是否为 1
    auto batch = getOperand(0).getType().cast<mlir::ShapedType>().getShape()[0];
    attr.defaultLayout = batch == 1 ? "Tensor" : "NTensor";  // 根据 batch 的值设置默认布局

    return attr;  
}
```

如下图所示，后端编译器会调用 genOpCode 生成相对应的 main.c. 然后 host.cpp 再把 main.c 放到不同的平台上面去编译完再去执行。

![OpLibInterface](https://share.note.youdao.com/yws/api/personal/file/WEBdb6a91e8bb45cbce292f6fdf1fafd0f4?method=download&shareKey=91e74089c887f70b2b508d9a31b877fd "OpLibInterface")

main.c 主要做的就是 load --> compute --> store 这三步。伪代码如下，由于进行了时间上的 split，需要循环多次才能读取完整的数据。

```cpp
while(!input_done)
{
  // load
  op_dma_load Input;
  input_done = Input.load_finish;
  op_dma_load scale;
  op_dma_load shift;
  op_dma_load mean;
  op_dma_load varience;
  
  // compute
  op_batchnorm_inference(param, input, scale, shift, mean, varience, out);

  // store
  op_store_var_ncx out;
}
```

`op_batchnorm_inference` 的定义如下，其中 imm 是辅助空间，此处申请了 2xsizeof(input) Bytes.
```cpp
uint64_t op_batchnorm_inference(BATCHNORM_INFER_PARAM *param, 
                                TSR *in, TSR *scale, TSR *shift, TSR *mean, TSR *var, 
                                TSR *imm, TSR *out);
```

其中 TSR 是一个自定义的结构体，包括数据格式，地址以及一个 L_shape (load shape). 里面记录了张量完整的大小 shape_whole，以及本 Tile 上每个维度起始下标 shape_start，每个维度加载的大小 shape_slice 和 shape 的维度大小 dim.

```cpp
typedef struct L_SHAPE {
    int32_t shape_whole[MAX_SHAPE_DIM];  // the whole shape
    int32_t shape_start[MAX_SHAPE_DIM];  // start idx of the shape slice
    int32_t shape_slice[MAX_SHAPE_DIM];  // length of the shape slice
    int32_t shape_real[MAX_SHAPE_DIM];   // real length of the shape slice
    int32_t dim;                         // dimension of the shape
} L_SHAPE;

typedef struct G_SHAPE {
    int32_t spatial_start[MAX_SHAPE_DIM];  // [start, end]
    int32_t spatial_end[MAX_SHAPE_DIM];
    int32_t dynamic_offset[MAX_SHAPE_DIM];
    int32_t shape[MAX_SHAPE_DIM];
    int32_t dim;
    int32_t done;                         // done for dma load finish
    int32_t batch_offset[MAX_SHAPE_DIM];
} G_SHAPE;

typedef struct TSR {
    Data_Format format;
    uint64_t addr;
    L_SHAPE* shape;
} TSR;
```

![BatchNorm Design](https://share.note.youdao.com/yws/api/personal/file/WEB863424dc56bf5d86b25f817e06a1c716?method=download&shareKey=de577bac9b2b5b108c4a3e8a275d07ea "BatchNorm Design")

对于非 fp32 类型数据 (以 fp16 为例) 计算过程与空间分配如下图所示。
1. 类型转换成 fp32: gatherScatter.
2. 调用 fp16->fp32 函数进行转换。
3. 循环计算 x-Mean (因为对 in 的 NHW 维度进行了 split)，结果存入 imm_a.
4. Varience 自加 epsilon(1e-6).
5. Varience 进行 rsqrt 操作。
6. Varience 与 x-Mean 进行循环乘。
7. 循环乘 scale.
8. 循环加 shift.
9. fp32 转回 f16.
10. gatherScatter 到 out 处。

![Batchnorm Computation Flow](https://share.note.youdao.com/yws/api/personal/file/WEB427f45ff2571bf94eea6a0b81f897ba1?method=download&shareKey=57a2c93fd8b252f86c47c8e71e325f2c "Batchnorm Computation Flow")


这里需要注意的是 shift(1, 1, 1, C) 和归一化后的 x(N, H, W, C) 相乘的时候，这时候就用到了之前所说的 VuV_mul 和 VuV_mul_loop 指令。

当 C <= 32 时，一个 batch 内的数据排布如下 (以 (4x112x2x30) x (1x1x1x30) 为例)，此时我们在 batch 维度上循环调用 VuV_mul 指令就可以。

![Channel <= 32](https://share.note.youdao.com/yws/api/personal/file/WEB7e47cbd16c9c6777c91a22a0c2685f91?method=download&shareKey=c305ac14d29df963a8884def061ef96f "Channel <= 32")

当 C > 32 时，需要向 64 对齐，一个 batch 内的数据排布如下 (以 (4x112x2x129) x (1x1x1x129) 为例)，每一个 Cx/C0 对应着一次 VuV_mul. 此时我们在 batch 维度上循环调用 VuV_mul_loop 指令就可以。

![Channel > 32](https://share.note.youdao.com/yws/api/personal/file/WEBfacfc3b8b6722744fa958775ab8a88f4?method=download&shareKey=968a25dfbb972d2e97c7b09f284733be "Channel > 32")

下面来说明如何调用指令，首先要明确调用的指令是属于哪一个模块的。例如第四步加 epsilon 我们需要调用 addVs 指令，其属于 CGRA 模块。

```cpp
typedef enum OP_INSTR_TYPE {
    I_CGRA,
    I_NEUR,
    I_RDMA,
    I_WDMA,
    I_TDMA,
    I_SCALAR,
    I_DTE,
    I_CSR,
} OP_INSTR_TYPE;
```

每个模块下的指令有自己的参数形式，下面列举一些。

```cpp
// I_CGRA
typedef struct CT_Param {
    uint32_t inter_type;
    Ncc_CT_GR_Ctl_Regs ctrl;
    Ncc_CT_GR_Param_Regs param;
} CT_Param;

// I_NEUR
typedef struct TsmNeInstr {
    uint32_t inter_type;
    Ncc_NE_GR_Ctl_Regs ctrl;
    Ncc_NE_GR_Param_Regs param;
} TsmNeInstr;

// I_(R/W)DMA
typedef struct DMA_Param {
    uint32_t inter_type;
    Ncc_DMA_GR_Ctl_Regs ctrl;
    Ncc_DMA_GR_Param_Regs param;
} DMA_Param;

// I_TDMA
typedef struct TD_Param {
    uint32_t inter_type;
    Ncc_TDMA_GR_Ctl_Regs ctrl;
    Ncc_TDMA_GR_Param_Regs param;
} TD_Param;
```

还是以 AddVS 指令为例，流程如下
1. 声明模块的指令参数。
2. 声明对应的指令类型指针，AddVS 属于 arith 类型的指令。getTsmOpPointer()->arith_pointer;`.
3. 根据调用指令传入参数，指令会根据传入参数配置好 ct_param 上寄存器的值。然后再进行 TsmExecute. 最后再把单词指令的执行时间进行累加。

```cpp
CT_Param ct_param = {I_CGRA, {0}, {0}};  // step 1
TsmArith *arith = (TsmArith *)getTsmOpPointer()->arith_pointer;  // step 2
// variance add epsilon
float epsilon = 1e-6;
arith->addVS(&ct_param,  // engine params
            varAddr,  // vector address
            *(uint32_t *)(&epsilon),  // scalar address
            varAddr,  // result address
            mid_tensor_info.total_num,  // vector elements num
            RND_NEAREST_EVEN,  // round method
            Fmt_FP32);  // data format
cycle_single = TsmExecute(&ct_param);
cycle_total = ADD_VALID_CYCLE(cycle_total, cycle_single);
```

## ComputeInteface

`ComputeInterface` 这个接口主要是每个 OP 通过 onednn 得到 CPU 代码。或者计算比较简单的 OP 如果在 onednn 的接口中没有找到对应的计算，也可以在 compute 接口中手写当前 OP 的 CPU 实现的 C++代码。最终生成结果会用来检验算子正确性。

```mlir
def ComputeInterface : OpInterface<"ComputeInterface"> {
  let description = [];
  let cppNamespace = "::tx8be_mlir";
  let methods = [
    InterfaceMethod<
      /*desc=*/[],
      /*retType=*/"::mlir::LogicalResult",
      /*methodName=*/"compute",
      /*args=*/(ins "ComputeParam&":$param)
    >,
  ];
}
```

# Test Case

TestCase 主要作用是写单算子或者多个 (单算子的上下文算子) 的测试，包括固定配置测试和随机配置测试,随机配置时主要对于算子支持的不同 dim, layout, dtype, shape 这四项做随机。流程主要做以下几件事。

**init_param**

通过数组来配置固定测试 case 或者随机测试范围，然后通过指定或随机的方式生成对应的输入，输出的 shape， dim 信息，除此之外参与随机的一般还包括数据对齐方式随机，数据类型随机，即在算子可支持的范围内产生随机的 FP16/FP32 不同的数据类型来保证测试的充分和全面。

除此之外还会生成 MLIR Module. 这个 module 是原来就给定的，在这里做的事情是首先新建一个空的 func. 然后在这个 func 中构造一个 block，里面去填入需要测试的这些 OP 的结构。

- Module：一个程序的容器，包含多个函数。
- Func：定义一个函数，包含多个 Block.
- Block：定义函数的基本执行单元，包含多个 Operation.
- Operation：表示具体的计算或操作，是程序中的基本指令。

![MLIR Structure](https://share.note.youdao.com/yws/api/personal/file/WEBfdb91efc229999a9b488d5131959f4b0?method=download&shareKey=a3935de13b565d4c37e06857c6c43f90 "MLIR Structure")
---

init_data

这个方法主要用来通过上面 Param 生成的 dim、输入或者输出 shape、数据类型来生成随机的数据，数据范围一定要根据算子情况配置，不然无效数值可能会在结果中出现 Nan. 还要考虑一些算子的特点，保证测试的充分性，例如创建 relu 的数据时，最好正负值都有覆盖。

---
compile

compile 方法有两个功能
1. 调用 Computelnterface 生成 onednn 或者手写 CPU 算子实现的结果。
2. 添加一些配置参数，跑出 tx8be mlir codegen 的结果。这其中会经历一些非常复杂的 pass，稍后再介绍。

---
saveInfoFile

saveInfoFile 方法主要是把创建出的 Data 数据写成.bin 文件保存。并把创建出的 module 的信息保存在 json 文件。


# Overview of Workflow

后端接收的是 MLIR 的计算图，然后经过编译器后端的处理，然后生成最后的 BE IR，其中中包含了一些 Oplib 的算子。最终这个 BEIR 会调用 OP 的算子，然后去跑在 C model 或者是实际的硬件芯片上面。后端编译器主要负责四个方面
layout 初始化和传递、const 管理、切分策略及其 SPM 分配和 DDR 分配。

# Layout Initialization and Pass.

layout 可以分为以下几种
- layout_str: 中端使用
    - CNN Op: 1. Feature (NCHW/NHWC) etc. 2. Weight (OIHW/HWOI) etc.
    - Non-CNN Op: 大模型中常见，Tensor/NTensor，它们的区别是第 0 维是否为 1.
- mem_layout: 后端使用，代表了在芯片上的实际排布  
    - Tensor/NTensor: 数据的紧密排布
    - Cx/NCx: 对 Tensor/NTensor 格式化后的结果，方便易硬件读取。


| dtype | channel | description |
|----------------------|-------|--------------------------------------------------|
| bf16/fp16 <br>/fp32/tf32 | c <= 32 | NHWC, C向4/8/16/32对齐，N 的起始地址向 2048bit 对齐 |
|                      | c > 32  | N[CxHW64, HWC0], C0 向 4/8/16/32 对齐，N 的起始地址向2048bit 对齐<br>在一个 batch 内将 tensor 按 C 分成 Cx*64 和 C0两部分 |
| int8                | c <= 64 | NHWC, C 向 4/8/16/32/64对齐，N的起始地址向2048bit对齐 |
|                      | c > 64  | N[CxHW128, HWC0], C0 向 4/8/16/32/64 对齐，N的起始地址向 2048bit 对齐 <br> 在一个 batch 内将 tensor 按 C 分成 Cx*128 和C0 两部分 |

## layoutInitPass
layoutInitPass 用于初始化计算图中 GemmOP 和 ConvOP 的 layout_str，其他的所有算子 layout_str 都设置为 UNKNOWN. 下图中的 `GemmOP layout_str = "Tensor-Tensor-Tensor"` 分别表示两个输入和输出的数据排布。

![LayoutStr](https://share.note.youdao.com/yws/api/personal/file/WEB95a6e97d5ae8432b8367189a36987f31?method=download&shareKey=8df50a04c4e2ed29be9e93d3da958e35 "LayoutStr")

## layoutTransmitPass 
layoutTransmitPass 会用已知的 GemmOP 和 ConvOP layout 信息进行扩散，得到全图的 layout_str.
1. 每个算子初始化为一个节点，有inputNodes容器和outputNodes容器分别存放自己的输入和输出节点。
2. GemmOp 和 ConvOp 作为起始节点，向前和向后推导 layout (算子的 `inferlayout()` 接口)，新推出layout 的节点作为下一批起始节点递归推导。
3. 遇到无法推导的节点 (如 Reshape，BroadCast) 则终止推导。将其余无法推导的节点 layout 直接初始化为 Tensor.

![layoutTransmitPass](https://share.note.youdao.com/yws/api/personal/file/WEB9d187e7b17a4b2ee65f01169f8c6a141?method=download&shareKey=c405975eba9c1da95539f89ac8b3de8a "layoutTransmitPass")


## layoutAlignToNpuPass
layoutAlignToNpuPass 用于在数据对齐冲突的地方插入 channelNorm，并将 layout_str 映射到 mem_layout. 在 NPU 上某些算子只支持 `COMPACT` layout，有些只支持 `ALIGN` layout，有些则都可以 `BOTH`.

1. 输入默认非对齐排布，从输入出发遍历整图，检查当前算子与其所有 user 之间的对齐要求，若冲突，记录插入点 (算子的对齐要求可以在 `OpLibInterface` 接口中的 `queryOpAttr()` 方法中查询到).
2. 根据记录的插入点，再次分析插入点前后的算子对齐要求，以确定channelnorm的方向，插入 channelnorm.
3. 赋值 `dev_info`，将 `layout_str` 映射到 `mem_layout`.

> dev_info用来描述数据在设备上的一些属性，有成员：imm_size (辅助空间大小), mem_layout, temporal_mem_slice, imm_addr, mem_addr.

![layoutAlignToNpuPass](https://share.note.youdao.com/yws/api/personal/file/WEB41f3c705e0f54e9291a5a2a7916f6045?method=download&shareKey=e9af8cbf6c2e348e4584018bcb4d4782 "layoutAlignToNpuPass")

LayoutAlignOptPass 应用几个 RewritePattern 用于删除冗余的 channelnorm.
1. **ConstChannelNormErase**: ConstantOp 维度为 1 并且只有 1 个 user 的时候可以删去并且将 devInfolayout 设置为 Cx.

{{< details title = "ConstChannelNormErase Implementation" >}}
```cpp{linenos=true}
// const can be directly considered to be aligned
// constop(dim < 2) -> channelNorm -> constop
struct ConstChannelNormErase : public mlir::OpRewritePattern<txbe::ConstantOp> {
  ConstChannelNormErase(mlir::MLIRContext *context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(txbe::ConstantOp op, mlir::PatternRewriter &rewriter) const override {
    // If const has multi user, can not erase
    if (!op->hasOneUse()) return mlir::failure();

    auto user = *op->getUsers().begin();
    if (!isa<txbe::ChannelNormOp>(user)) return mlir::failure();

    auto shape = op->getResult(0).getType().dyn_cast<mlir::ShapedType>().getShape();
    if (shape.size() > 1) return mlir::failure();

    llvm::SmallVector<Operation*> userVec;
    userVec.insert(userVec.end(), user->getUsers().begin(), user->getUsers().end());
    for (auto channelNormUser : userVec) {
      channelNormUser->replaceUsesOfWith(user->getResult(0), op->getResult(0));
    }

    // set align=true
    setDevInfoWithLayout(op->getContext(), op->getLayoutStr().str(), true);

    if (user->use_empty()) rewriter.eraseOp(user);

    return success();
  }
};
```
{{< /details >}}

{{< details title = "RedudantChannelnormErase Implementation" >}}
2. **RedudantChannelnormErase**: 如果该 channelnormOp 的输入是来自一个 constOp 并且只有一个输出，则检查是否还有其他的 channelnormOp 也使用。如果是，则让它们直接使用该 channelnormOp 的结果，以消除多余的 channelnormOp.

```cpp
// A pass to erase redundant channel normalization operations
struct RedundantChannelNormErase : public mlir::OpRewritePattern<tx8be::ChannelNormOp> {
  RedundantChannelNormErase(mlir::MLIRContext *context) : OpRewritePattern<tx8be::ChannelNormOp>(context, /*benefit=*/1) {}

  mlir::LogicalResult matchAndRewrite(tx8be::ChannelNormOp op, mlir::PatternRewriter &rewriter) const override {
    // Define the input operation and its defining operation
    // def represents the operation that generates the op input data
    auto def = op.getInput().getDefiningOp();

    // Check if the defining operation is a ConstantOp and has more than one result
    if (isa<tx8be::ConstantOp>(def) && (def->getNumResults() > 1)) {
      return mlir::failure(); // Fail if conditions are not met
    }

    // Get the size in bits of the input shape
    auto size = op.getInput().getType().cast<ShapedType>().getSizeInBits();

    Operation *sameOp = nullptr; // Pointer to a potentially redundant operation

    // Iterate over all users of the defining operation
    for (auto user : def->getUsers()) {
      if (user == op) { // Skip if the user is the current operation
        continue;
      }
      if (isa<tx8be::ChannelNormOp>(user)) { // Check if the user is another ChannelNormOp
        sameOp = user; // Store the redundant operation
        break;
      }
    }

    if (!sameOp) return mlir::failure(); // Fail if no redundant operation is found

    // Replace all uses of the redundant operation with the current operation's results
    op->replaceAllUsesWith(sameOp->getOpResults());

    if (op->use_empty()) { // Erase the current operation if it has no more uses
      rewriter.eraseOp(op);
    }

    return success(); // Return success if the rewrite is completed
  }
};
```
{{< /details >}}

# Const Management

常量统一使用 `ConstContainer` 类来进行管理。通过 map 来记录每个常量对应的 ParamInfo. 一个常量可能被分配到多个芯片上，每个芯片上数据可能相同，也可能不同。

```cpp{linenos=true}
struct ParamInfo {
    std::vector<uint8_t>* data_ptr;  // const value
    std::set<int32_t> chip_id;  // which chips has this const, -1 indicates all chip has the same param.
    uint32_t label;  // Indicates whether the data is assigned to a certain chip_id. 
};

// class ConstContainer {
class ConstContainer {
public:
    ConstContainer();
    virtual ~ConstContainer();
    // some public functions

private:
    std::map<uint32_t, std::vector<ParamInfo>> _data;
    std::map<uint32_t, std::map<int32_t, uint64_t>> oidToSize;
    std::map<uint32_t, uint32_t> oidToNid;
};
```

## MoveConstantPass

MoveConstantPass: 创建图的 `ConstContainer`，然后应用 `ConstantToLoadConst` Rewrite Pattern. 转换完成后会调用 `updateConstContainer` 更新 `ConstContainer` 各个 const 的 ID. 用一个大小为 `4*1024*tile_num` (DDR_BANK_SIZE) `thresholdSize` 将大于这个值的 const 全部放在前面，小的放在后面。

```cpp{linenos=true}
void MoveConstantPass::runOnOperation() {
  // create constant container
  createConstContainer();
  // get module op
  ModuleOp module = getOperation();
  // Set pattern
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  patterns.insert<ConstantToLoadConst>(context);
  const FrozenRewritePatternSet frozen_patterns =
      FrozenRewritePatternSet(std::move(patterns));
  // Set config
  GreedyRewriteConfig config;
  config.useTopDownTraversal = true;

  for (auto func : module.getOps<func::FuncOp>()) {
    Region &body = func.getBody();
    if (failed(applyPatternsAndFoldGreedily(body, frozen_patterns, config))) {
      llvm::errs() << "Failed when move const in main graph.\n";
      signalPassFailure();
    }
  }

  for (auto subgraph : module.getOps<tx8be::SubgraphOp>()) {
    Region &body = subgraph.getBody();
    if (failed(applyPatternsAndFoldGreedily(body, frozen_patterns, config))) {
      llvm::errs() << "Failed when move const in subgraph.\n";
      signalPassFailure();
    }
  }

  TileInfo tinfo = get_tileinfo(module);
  updateConstContainer(tinfo.tile_num);  // update id by thresholdSize
  updateLdConstop();
}
```


`ConstantToLoadConst` 首先通过分析该常量的所有 users，来判断这个常量是否需要 LoadConstOp. 如果需要加载，它会将原始常量的数据注册到一个全局容器中并获得一个 ID，然后创建一个新的 LoadConstOp ，并将此 ID 及其他硬件属性赋予它。接着，它会更新所有使用者，将它们的输入从旧的 ConstantOp 重定向到这个新的 LoadConstOp，最后再删除无用的原始常量。最后再更新所有 const 的 ID.

{{< details title = "ConstantToLoadConst Implementation" >}}
```cpp{linenos=true}
struct ConstantToLoadConst : public mlir::OpRewritePattern<tx8be::ConstantOp> {
  ConstantToLoadConst(mlir::MLIRContext *context)
      : OpRewritePattern<tx8be::ConstantOp>(context, /*benefit=*/) {}

  mlir::LogicalResult
  matchAndRewrite(tx8be::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    uint32_t id = 0;
    // Store constant data to constant container 
    // ...

    // Determine if this constant operation needs an explicit load instruction.
    bool needLoad = false;
    auto v = op.getOutput();
    // Iterate over all operations that use this output value.
    for (auto user_op : v.getUsers()) {
        // Get the argument index of the user op that corresponds to our output value.
        int32_t arg_idx = getArgumentIdx(user_op, v);
        // Assert that the user operation implements our custom OpLibInterface.
        ASSERT(llvm::dyn_cast<tx8be::OpLibInterface>(user_op));
        // Get the library attributes for this user operation.
        auto opAttr = llvm::dyn_cast<tx8be::OpLibInterface>(user_op).queryOpAttr();
        // Skip if the user is a TupleOp, which might have special handling.
        if (isa<tx8be::TupleOp>(user_op)) {
            continue;
        }

        if (opAttr.needLoad & (1 << arg_idx)) {  // Check if the 'needLoad' attribute
            needLoad = true;
        } else {
            ASSERT(needLoad == false);
        }
    }

    // Set attributes
    // ...

    // Safely iterate over the users. This is important because we are modifying the use-list inside the loop.
    for (auto &use : llvm::make_early_inc_range(op.getOutput().getUses())) {
        Operation *userOp = use.getOwner();
        // Create the new, hardware-specific LoadConst operation.
        txbe::LoadConstOp newLoadConst =
            rewriter.create<txbe::LoadConstOp>(op.getLoc(), op.getOutput().getType(), ValueRange{}, attrs);

        if (!needLoad) {  // this constant does not need an explicit load... 
            // Get a builder to set attributes.
            OpBuilder builder(newLoadConst.getContext());
            // Set a 'bypasscodegen' attribute, signaling special handling for this op in later stages.
            newLoadConst.getOperation()->setAttr("bypasscodegen", builder.getBoolAttr(true));
        }
        // Set the layout string attribute on the new LoadConst op.
        newLoadConst->setAttr("layout_str", op->getAttr("layout_str"));
        // CRITICAL STEP: Rewire the user's operand to point to the result of the new LoadConst op.
        userOp->setOperand(use.getOperandNumber(), newLoadConst);
    }

    // After all uses have been replaced, erase the original, now-dead ConstantOp.
    rewriter.eraseOp(op);
    return success();
    }
}
```
{{< /details >}}

## constNormPass
constNormPass: 遍历图中的 LoadConstOp. 它会寻找一个特定的模式：如果一个 LoadConstOp 的唯一 user 是一个 ChannelNormOp，那么会通过 `constChannelNormErase` 函数进行消除和将对其信息同步到 LoadConstOp. 最后通过 `processMultiUse` 确保所有加载同一个底层常量数据的 LoadConstOp 实例，都具有完全相同的内存布局。

{{< details title = "ConstNormPass Implementation" >}}
```cpp
void ConstNormPass::runOnOperation() {
    ModuleOp module = getOperation();
    func::FuncOp mainGraphFunc = getMainFuncOp(module);
    SmallVector<Operation *> deletedChannelnorm;

    // Walk the main function to find a specific pattern: LoadConst -> ChannelNorm.
    mainGraphFunc.walk([&](Operation* constOp) {
        if (isa<tx8be::LoadConstOp>(constOp)) {
            std::unordered_set<Operation*> users;
            users.insert(constOp->getUsers().begin(), constOp->getUsers().end());
            bool flag = false;
            // Check if any user is a ChannelNormOp.
            for (auto user : users) {
                if (isa<tx8be::ChannelNormOp>(user)) {
                flag = true;
                break;
                }
            }
            
            // If the LoadConst has exactly one user, and that user is a ChannelNormOp,
            // mark the ChannelNormOp for deletion.
            if (flag && users.size() == 1) {
                for (auto it : users) {
                // The erase logic is commented out, maybe handled by constChannelNormErase or done later.
                    deletedChannelnorm.push_back(it);
                }
            }
        }
    });

    // Erase all the marked ChannelNormOps. This is done in a separate loop
    // to avoid iterator invalidation issues.
    for (auto op : deletedChannelnorm) {
        op->erase();
    }

    // Set up and run a nested pass pipeline.
    OpPassManager thisPM(this->getOpName().value());
    // This pipeline will only apply to LoadConstOp operations inside functions.
    OpPassManager &loadConstOpPM = thisPM.nest<func::FuncOp>().nest<tx8be::LoadConstOp>();
    // Add the ConstNormDoPass to the pipeline.
    loadConstOpPM.addPass(std::make_unique<ConstNormDoPass>());

    // Run the newly constructed pipeline on the module.
    auto result = this->runPipeline(thisPM, getOperation());

    // After the pipeline, run a final cleanup/consistency check function.
    processMultiUse(module);

    // change unpack input0 qweight shape after ConstNormDoPass. (Original comment)
    // This logic is likely inside the runOnOperation() method of ConstNormDoPass.
    mainGraphFunc.walk([&](Operation* constOp) {
      if (isa<tx8be::LoadConstOp>(constOp)) {
          // Collect all users of this LoadConstOp.
          std::unordered_set<Operation*> users;
          users.insert(constOp->getUsers().begin(), constOp->getUsers().end());
          
          // Check if any user is an UnpackOp.
          bool flag = false;
          for (auto user : users) {
              if (isa<tx8be::UnpackOp>(user)) {
                  flag = true;
                  break;
              }
          }

          // If there is exactly one user, and it's an UnpackOp...
          if (flag && users.size() == 1) {
              for (auto it : users) {
                  // This check seems to ensure we are modifying the correct operand.
                  if (constOp->getResult(0) == it->getOperand(0)) {
                  // Get the original shape and type.
                  llvm::SmallVector<int64_t, 6> oShape;
                  auto type = constOp->getResult(0).getType().cast<ShapedType>();
                  auto shape = type.getShape();

                  // Apply the shape transformation: e.g., for unpacking packed data.
                  oShape.push_back((int32_t)shape[0] / 4);
                  oShape.push_back((int32_t)shape[1] * 4);

                  // Create a new tensor type with the new shape.
                  auto oType = mlir::RankedTensorType::get(oShape, type.getElementType());
                  // Update the type of the LoadConstOp's result in-place.
                  constOp->getResult(0).setType(oType);
                  }
              }
          }
      }
    });
  }
```
{{< /details >}}

`constChannelNormErase` 处理 LoadConstOp -> ChannelNormOp 这种模式。让所有原本使用 ChannelNormOp 计算结果的操作，现在改为直接使用 ChannelNormOp 的输入数据。获取 LoadConstOp 当前的设备信息和 layout，计算出一个新的经过对齐的布局 `align_dev_layout`，然后用这个新布局去更新 LoadConstOp.

{{< details title = "constChannelNormErase Implementation" >}}
```cpp{linenos=true}
// This function erases a ChannelNormOp by bypassing it and updating the source constant's layout.
void constChannelNormErase(tx8be::ChannelNormOp op) {
  // Find the defining operation of the ChannelNorm's operand, which should be a LoadConstOp.
  auto defOp = llvm::dyn_cast_or_null<tx8be::LoadConstOp>(op->getOperand(0).getDefiningOp());
  // If the source is not a LoadConstOp, do nothing.
  if (!defOp) return;

  // Collect all users of the ChannelNormOp's result.
  llvm::SmallVector<Operation*> userVec;
  userVec.insert(userVec.end(), op->getUsers().begin(), op->getUsers().end());

  for (auto user : userVec) { // Replace all uses of the ChannelNormOp's result with the result of the LoadConstOp..
    user->replaceUsesOfWith(op->getResult(0), op->getOperand(0));
  }
  
  // After bypassing, the layout of the source constant might need to be adjusted
  // to reflect the transformation that the ChannelNormOp was supposed to perform.
  // set const layout to cx mode 
  auto dev_layout = getDevInfoLayoutMode(defOp);
  auto align_dev_layout = get_aligned_layout((LAYOUT_MODE)dev_layout);
  setDevInfoWithLayout(defOp->getContext(), defOp, static_cast<tx8be::LayoutMode>(align_dev_layout));
}
```
{{< /details >}}

`processMultiUse` 保证所有对同一份常量数据的引用，其 mem_layout 都是完全一致的。流程如下
1. `processMultiUse` 遍历计算图中的所有 LoadConstOp，以 `const_map_id` 为 key，将所有指向同一个物理常量的 LoadConstOp 实例分组存放在一起。
2. 遍历这个 map，只处理那些包含多个 LoadConstOp 实例的组 (`kv.second.size() > 1`).
3. 在每个组内，确定一个正确的布局。代码逻辑是以组内的第一个 LoadConstOp 的布局为基准，但如果发现组内有 `is_cx_layout`，则会采用这个优先的布局作为标准。
4. 一旦确定了标准布局，会再次遍历该组内的所有 LoadConstOp 实例。调用 `setDevInfoWithLayout` 函数，强制将每一个实例的布局属性修改为刚才确定的那个标准布局。

{{< details title = "processMultiUse Implementation" >}}
```cpp{linenos=true}
// This function processes multi-use constants to ensure their layouts are consistent.
void ConstNormPass::processMultiUse(ModuleOp module) {
  func::FuncOp mainGraphFunc = getMainFuncOp(module);
  
  // When a const is used by multiple users, multiple loadconsts will be generated,
  // but only one loadconst will have its layout set. The others will be skipped.
  // We need to go over them uniformly. 
  // First, find all previous useless constant ops.

  // Group all LoadConstOp instances by their underlying constant data ID (const_map_id).
  std::unordered_map<int32_t, std::vector<mlir::Operation *>> allconst;
  mainGraphFunc.walk([&](Operation* constOp) {
    if (isa<tx8be::LoadConstOp>(constOp)) {
      auto cOp = llvm::dyn_cast<tx8be::LoadConstOp>(constOp);
      uint32_t t_map_id = cOp.getConstMapId();
      allconst[t_map_id].emplace_back(constOp);
    }
  });

  // Based on duplication, find if the layout needs to be changed to cx. 
  // Check if there is also a Cx with the same layout. 

  // Iterate over each group of LoadConstOps that share the same data.
  for (auto &kv : allconst) {
    if (kv.second.size() > 1) { // Process only if there are multiple users.
      // Assume the layout of the first user is the correct one.
      auto layout = (LAYOUT_MODE)getDevInfoLayoutMode(kv.second.front());
      
      // This loop is for validation, checking if layouts are inconsistent.
      for (auto op : kv.second) {
        auto layout2 = (LAYOUT_MODE)getDevInfoLayoutMode(op);
        if (is_cx_layout(layout2) != ALIGN_NOT) {
          layout = layout2;
          break;
        }
      }

      // Force all LoadConstOps in this group to have the same, correct layout.
      for (auto op : kv.second) {  
        auto ctx = op->getContext();
        ASSERT(op->hasAttr("dev_info") && "Must have dev_info!");
        setDevInfoWithLayout(ctx, op, (tx8be::LayoutMode)layout);
      }
    }
  }
}
```
{{< /details >}}

# Sharding Search and SPM Management

第一步是对算子进行 Group 划分，插入 load & store. 对每一个 subGraph 会应用如下的 3 个 Pass:

- **GroupPatternPass**：应用配置好的 group config (opt_group).
- **GroupOptimizationPass**: 如果没有配置，则会为每个 compute op 创建一个 group.
- **GroupLdStPass**: 为每个需要的 groupOp 插 入loadOp 和 storeOp，并添加 group_tag. 
    - group_tag = 0: 需要 load 或 store，意味着该 group 需要后续的切分搜索。
    - group_tag = 2: 不需要 load 或 store，意味着该 group 的op 都在 DDR 上操作，无需参与后续的切分搜索。

SPM 上一定要能放下切分后的结果。Group 是切分搜索和 SPM 分配的基本的单位。思想就是尽量把连续执行的算子组合在一起，一直在 SPM 上运行而不是存回 DDR 再读入，以此来减少访存时间。GroupOp 在 td 文件中定义所包含的输入如下:

```tablegen
let regions = (region SizedRegion<1>:$body);

let arguments = (ins
    Variadic<AnyTensorOrNone>:$operands,      // 输入参数为 操作数的数量可变的的张量
    DefaultValuedAttr<BoolAttr, "false">:$pipeline_parallel, // 是否用流水线并行
    DefaultValuedAttr<I32Attr, "1">:$sp_stage_num,
    OptionalAttr<Tx8e_RegionAttr>:$dev_region, // 设备的空间属性
    OptionalAttr<UI32Attr>:$spm_alloc_size,   // group占用的spm大小
    OptionalAttr<I32Attr>:$group_tag,         // 0: 正常切分, 1: split nht, 2: 不切分 (reshape)
    OptionalAttr<DenseI32ArrayAttr>:$stream_online_check,
    OptionalAttr<DenseI32ArrayAttr>:$stream_offline,
    DefaultValuedAttr<BoolAttr, "true">:$need_barrier,   // 是否需要tile同步
    DefaultValuedAttr<SI32Attr, "-1">:$group_id,         // group id序号
    DefaultValuedAttr<SI32Attr, "-1">:$template_id      // 复用其他group的id, 小于0为不复用
);

let results = (outs Variadic<AnyTensorOrNone>:$results);
```

还有一些常用到的结构体
`SecsInfo` 记录了单个 Op在分布式策略搜索过程中的所有状态和信息。

```cpp
struct SecsInfo {
  std::vector<int32_t> sharding;  // space 
  std::vector<int32_t> split;  // time
  std::vector<int32_t> splitry;  // 当前搜索的 sharding 的 split
  std::vector<int32_t> reduceSplit;  // 针对需要进行规约 (Reduction) 的维度的切分策略。
  int32_t reducesplit[0];  // 一个标记位，用于指示reduceSplit是否被使用

  // ******************** 以下变量为factorSpace使用部分 ********************

  bool sfinish[1];  // 标记 split/reduceSplit 相关的策略是否已确定。

  // 枚举类型，定义了当前算子所处的切分模式，特别关注需要通信的Reduce维度。
  /* SHARDING_MODE 的可能值解释：
   * SHARDING_INIT: 初始状态，尚未确定模式。
   * 0: 不切分规约 (reduce) 维度。意味着数据在每个设备上是完整的，无需通信。
   * 1: 单边切分规约维度。例如，只切分权重，不切分输入，数据在不同tile上需要通信。
   * 2: 两边都切分规约维度。
   * 3: 对权重(weight)的输出通道(output channel)维度进行切分，但不属于张量并行(TP)，可能需要fn/oc通信。
  */
  SHARDING_MODE shardingMode{SHARDING_INIT};

  bool rfinish[1];   // 标记 reduceSplit 相关的策略是否已完成处理。
  bool nfirst[0];  // 标记搜索方向。1: search from dim0 -> dim n-1
  bool finish[0];  // 表示该算子的策略搜索是否已全部完成。整个搜索流程: sharding -> shardingmode -> split -> reduceSplit
  std::vector<bool> sliceShapeMin; // 标记切分后的张量 (slice) 在每个维度上是否已达到某个最小尺寸限制。

  // ******************** 以下变量为sliceInfo使用部分 ********************

  std::vector<int64_t> TemporalShape;  // 切分后，临时的张量形状
  std::vector<int> reduce_sharding_space;  // 规约维度切分的搜索空间
  std::vector<int> reduce_sharding;  // // 最终选定的规约维度切分策略
  bool sharding2_finish[0];  // 标记第二阶段切分 (可能与规约相关) 是否完成
};
```

## GroupPatternPass
`GroupPatternPass` 其核心功能是在给定的计算图 (subgraphOp) 中，通过一种高效的模式匹配算法，识别出预定义的、可优化的子图模式 (Operator Patterns)，并将匹配到的算子 (Operations) 进行分组。这种分组通常是图优化 (如算子融合、算子调度) 的第一步。

该 Pass 首先获取配置，决定从哪里加载模式 (一个 map，其键是模式，即一个算子序列 `std::vector<TX8BE_OPS>`，值是一个整数 `int`，代表模式优先级，越大优先级越高) 。然后，它调用 `aca.insertPatterns` 将这些模式"编译"到 Automation 引擎中。接着，调用 `aca.search` 执行匹配。最后，从 manager 中获取匹配结果 (groups) ，并对这些 groups 进行后续处理，例如创建新的逻辑分组和进行拓扑排序。

```cpp
 void GroupPatternPass::runOnOperation() {
     TFUNC_SCOPE(DEBUG);
     auto subgraphOp = getOperation(); // Get the current operation (e.g., a function) the pass is running on.
 
     PatternManager manager; // A manager to hold graph rewriting information.
     Automation aca(&manager); // Custom 'Automation' class for pattern matching logic.
 
     auto minfo = getModuleConfig(getModuleByOp(getOperation())); 
     std::string path = ""; 
     auto temp = path != "" ? getPatternsFromFile(path)  // Load patterns from a file if path is specified.
                           : (patternConfigMap.at(static_cast<GroupPatternMode>(minfo.opt_group))); // Otherwise, load from a pre-defined map using a config key.
     TLOG(INFO) << "[GroupPatternPass] config id: " << minfo.opt_group;
 
     aca.insertPatterns(temp); // Insert the loaded patterns into the Automation engine. This is the starting point for building the matching structure.
     TLOG(INFO) << "[Automation]: \n" << printTree(aca.root);
 
     aca.search(subgraphOp); // Execute the search for all patterns on the given subgraph. (search function code is not provided but its role is clear).
      manager.applyAll(); 

     auto groups = manager.getGroups(); // Retrieve the groups of operations that were matched.
     manager.show();

     auto newGroups = createGroups(subgraphOp, groups); // Create new group structures from the matched results.
     for (auto group : newGroups) {
         sortTopologically(group->getBlock()); // Topologically sort the operations within each new group to maintain data dependencies.
     }
 }
```

`insertPatterns` 对于每一个模式，它首先调用 processPattern 来处理其中的 OR, WILDCARD 算子。
- 当遇到 OR 时，它会将模式拆分。例如，A B OR C D 这样的模式会被拆解成两个独立的模式 A B 和 C D 进行处理。
- 当遇到 WILDCARD 时，它会生成多个模式。根据代码 `for (int i = 0; i < 5; i++)` 和 `temp.push_back(*(it - 1))`，OP * 可能会被扩展成 OP, OP OP, OP OP OP, OP OP OP OP 等一系列重复模式。
- 它通过递归调用自身，以处理一个模式中包含多个特殊算子的情况。
最终，它返回一个由多个具体、无特殊算子的模式组成的列表。然后，它将这些扩展后的具体模式逐一传递给 `insertPattern` 函数，以构建 Trie 树。

```cpp
void Automation::insertPatterns(std::map<std::vector<TX8BE_OPS>, int> patterns) {
    std::vector<std::vector<TX8BE_OPS>> tempPatterns;
    for (auto it : patterns) { // Iterate through each pattern from the input map.
          auto temp = processPattern(it.first); // Pre-process the pattern. This can expand one pattern into many.
          for (auto p : temp) { // For each of the generated concrete patterns...
            insertPattern(p, it.second); // ...insert it into the main data structure (the Trie).
        }
    }
}
```

`insertPattern` 接收一个具体的模式，并将其插入到 Trie 树中。Trie 树是实现高效前缀匹配的关键。从root节点开始 遍历模式中的每个 op. 如果当前节点没有指向op的子节点，就创建一个然后移动到该子节点。当模式遍历完成后，在最终的节点上存储完整模式本身 (`node->pattern`) 和它的 ID (`node->output`) 。这表明一个有效的模式在此结束。

{{< details title = "insertPattern Implementation" >}}
```cpp
struct TrieNode {
    TrieNode(TX8BE_OPS id) : id(id) {} // Constructor to initialize the node with an operation ID.
    TX8BE_OPS id; // The operation (Op) type this node represents. This is the 'character' in our sequence.
    std::vector<int> output; // Stores the integer IDs of the patterns that end at this node. A non-empty vector indicates a valid pattern match.
    std::vector<TX8BE_OPS> pattern; // Stores the complete operator sequence for the pattern that ends here.
    std::unordered_map<TX8BE_OPS, NodePtr> children; // A map from an operation type to the next node in the trie. `NodePtr` is likely a shared_ptr or unique_ptr to another TrieNode.
};

void Automation::insertPattern(const std::vector<TX8BE_OPS> pattern, int index) {
    patterns_.push_back(pattern); // Store the raw pattern vector.
    auto node = root; // Start from the root of the Trie.
    for (auto op : pattern) { // Iterate through each operation in the pattern sequence.
        if (node->children.find(op) == node->children.end()) { // If a path for this operation does not exist...
            node->children[op] = std::make_shared<TrieNode>(op); // ...create a new node in the Trie.
        }
        node = node->children[op]; // Move to the next node in the Trie.
    }
    node->pattern = pattern; // At the end of the pattern, mark this node as a terminal node by storing the full pattern.
    node->output.push_back(index); // Store the original pattern index/ID at this terminal node.
}
```
{{< /details >}}

`searchOp` 函数的功能是：给定一个起始 Trie 节点 (parentNode) 和一个MLIR算子 (op)，它会尝试将 op 与parentNode 的子节点进行匹配，并在匹配成功后，递归地对其所有后继算子 (users) 进行 DFS 模式匹配，最终返回这条路径上所能找到的“最佳”匹配模式的末端Trie节点。

这里的“最佳”通常指最长的匹配模式，或者在有多个同样长度的模式时，选择优先级最高的那个 (根据节点中的 `output.front()`) 大小比较来判断。

{{< details title = "searchOp Implementation" >}}
```cpp
NodePtr Automation::searchOp(NodePtr parentNode, Operation* op) {
    auto opId = getOpId(op); // Get the enumerated ID (e.g., TX8BE_OPS::CONV) for the current MLIR operation.

    if (isRealOp(op) && parentNode->children.find(opId) == parentNode->children.end()) {
        // If the current op is a "real" operation (not a terminator, etc.) but cannot be found in the children of the parent Trie node, it's a mismatch.
        // This 'if' block seems to be an early exit for a specific case, possibly redundant with the final return.
    }

    if (parentNode->children.find(opId) != parentNode->children.end()) { // If a path exists in the Trie for the current operation `opId`. This is a potential match.

        // If the current op matches, continue downwards
        auto currentNode = parentNode->children[opId]; // Move to the matched Trie node.
        auto tempNode = currentNode; // `tempNode` will store the longest match found so far starting from this path.

        // --- Query Operation Attributes and Users ---
        auto queryInterface = llvm::dyn_cast<tx8e_mlir::OpLibInterface>(op); // Get a specific interface from the operation for querying attributes.
        auto needStore = queryInterface.queryOpAttr().needStore; // Check an attribute, e.g., if the op's result needs to be stored.
        llvm::SmallSet<Operation*, 1> users; // Find all direct users of the current operation's result.
        for (auto user : op->getUsers()) {
            users.insert(user);
        }

        auto sortedUsers = manager_->sortOps(users); // Sort the users, likely topologically or based on some priority.

        // --- Recursively Search Through Users ---
        for (auto user : sortedUsers) {
            if (!isRealOp(user)) continue; // Skip non-essential ops.

            auto interface = llvm::dyn_cast<tx8e_mlir::OpLibInterface>(user);
            auto needLoad = interface.queryOpAttr().needLoad;
            if (!needStore && needLoad) continue; // Skip paths with certain attribute mismatches (e.g., store-load dependency).

            // Recursively call searchOp for the user operation, starting from the current Trie node.
            auto terminalNode = searchOp(currentNode, user);

            // --- Update Best Match ---
            if (!terminalNode->output.empty() && !tempNode->output.empty()) { // If both the previous best match (`tempNode`) and the new match (`terminalNode`) are valid patterns...
                // Compare priority, take the one with the highest priority as the current node pattern)
                if (terminalNode->output.front() > tempNode->output.front()) { / ...update `tempNode` to the new one if it has a higher priority (assuming the int ID represents priority).
                    tempNode = terminalNode;
                }
            } else if (!terminalNode->output.empty()) { // If `tempNode` was not a valid pattern end, but `terminalNode` is, update it.
                tempNode = terminalNode;
            }
        }
        // TFOOTER(TRACE)
        return tempNode; // Return the node corresponding to the longest/best pattern found from this point.
    }

    // Indicates parent node cannot match current op, return parent node)

    return parentNode; // If no match was found for `opId` in the Trie, return the original `parentNode`.
}
```
{{< /details >}}

`search`遍历计算子图 (subgraph) 中的每一个算子，并以该算子为起点，尝试进行模式匹配。

1. 预处理阶段 (第一个 walk)
在正式开始匹配之前，函数会先遍历一次整个子图，目的是收集和注册一些元数据：
- `manager_->opOrder_`: 一个 vector 记录图中所有算子的出现顺序。
- `manager_->opIndexMap_`: 为每个算子分配一个唯一的整数索引。
这些信息对于后续的管理和可能的图变换 (如拓扑排序) 非常重要。

2. 逐点匹配阶段 (第二个 walk)它再次遍历子图中的每一个算子 op 每次都是从 Trie 树的根节点 root 开始 `searchOp(root, op)` 函数。意味着尝试从零开始匹配所有已知的模式。 searchOp 会返回从 op 开始能找到的最长/最优的匹配模式的末端节点 (terminalNode). 
- 如果其 output 列表不为空，说明 searchOp 成功地找到了一条完整的匹配路径。函数就会将这个匹配结果记录下来：在 manager 中更新 Pattern 对象，并在本地的 result map 中建立从起始算子 op到模式ID的映射。
- 反之说明从 op 开始无法匹配任何完整的模式，于是就什么也不做，继续检查下一个算子。

{{< details title = "search Implementation" >}}
```cpp
void Automation::search(tx8e::SubgraphOp subgraph) {
    // k: the starting operation of a matched pattern
    // v: the type/ID of the matched pattern
    std::map<Operation*, int> result;

    manager_->initDefsMap(subgraph); // Initialize manager with definition information from the subgraph.
    subgraph->walk([&](Operation* op) { // First pass: walk through the subgraph to gather metadata.
        manager_->opOrder_.insert(op); // Record the sequential order of all operations.
        manager_->opIndexMap_[op] = index++; // Assign a unique index to each operation.
    });

    // Second pass: walk through the subgraph again to perform the actual pattern matching.
    subgraph->walk([&](Operation* op) {
        // Skip the return operation of the subgraph as it's not part of a computational pattern.
        if (isa<tx8e::SubgraphOp, tx8e::SubgraphReturnOp>(op)) {
            return WalkResult::skip(); // In newer MLIR, this might be `return;`. Skips processing this op's children.
        }

        auto pattern = std::make_shared<Pattern>(op); // Create a Pattern object, representing a potential match starting at `op`.
        manager_->patterns_.push_back(pattern); // Add this potential pattern to the manager's list.
        manager_->patternMap_[op] = pattern; // Map the operation `op` to its corresponding Pattern object.

        // terminalNode 就是最后匹配到的一个Node (terminalNode is the final matched Node)
        // This is the main call to the recursive search function, starting from the Trie root for each `op`.
        auto terminalNode = searchOp(root, op);

        // If the Node has an output, it means a match was found. If multiple matches exist, they are replaced based on priority during the search phase
        // The final result is a match for the highest-priority pattern
        if (!terminalNode->output.empty()) { // Check if the search returned a valid pattern-terminating node.
            // If a match was found, update the Pattern object with the results from the terminal node.
            pattern->setPattern(terminalNode->output.front(), terminalNode->pattern);
            // Record the result: map the starting operation `op` to the matched pattern's ID.
            result[op] = terminalNode->output.front();
        }

        return WalkResult::advance(); // Proceed to the next operation in the walk.
    });
}
```
{{< /details >}}

## GroupOptimizationPass

会遍历一个计算 subGraph 中的所有 OP. 对于每一个通过筛选的普通计算操作，会调用 `createSingleGroup` 函数来为其创建一个专属的 GroupOp. 
`createSingleGroup` 会检查原始 OP 的所有输入。如果输入来自另一个计算操作，那么这个输入就会成为新 GroupOp 的输入。如果输入是 LoadConstOp，则被视为这个分组的内部依赖，而不是外部输入。原始 op 的所有输出会直接成为新 GroupOp 的输出。

新的 GroupOp 拥有上一步定义的输入和输出。原始的操作 op 和它的常量依赖 (dependencies) 被移动到这个新创建的 GroupOp 内部。最后，修改原始操作 OP 的连接关系，使其在分组内部能够正确地接收输入并产生输出。伪代码如下

```
for op in subGraph.ops:

  // 检查操作的类型
  if op == (GroupOp || ReturnOp || LoadConstOp || NoneOp):
    continue

  createSingleGroup(op)

------------------------------------
createSingleGroup(op):
  for pre_op in op.inputsOp:
    // 判断前置操作是否为“加载常量”或“空操作”
    if pre_op == (LoadConstOp || NoneOp):
      // 如果是，则将其添加到依赖项 (dependencies) 集合中
      dependencies.add(pre_op)
    else:
      // 如果是其他普通操作，则将其结果添加到新分组的输入 (groupInput) 中
      groupInput.add(pre_op.result)

  for result in op.results:  // 遍历当前操作的所有输出结果
    // 将这些结果添加到新分组的输出 (groupOutput) 中
    groupOutput.add(result)

  // 使用收集好的输入和输出创建一个新的 GroupOp (分组操作) 
  create GroupOp(groupInput, groupOutput)

  // 将依赖项 (如常量) 移动到新分组的末尾 (或内部) 
  move dependencies to group end

  // 将原始操作 op 本身也移动到新分组的末尾 (或内部) 
  move op to group end

  // 修改原始操作 op 的输入和输出，使其在新分组内部正确连接
  change op input and output
```

![GroupOptimizationPass](https://share.note.youdao.com/yws/api/personal/file/WEBca62e625dd418d0b51deb2e46c83f873?method=download&shareKey=3b9dddfeca5108a0665fce242dd1019d "GroupOptimizationPass")

## GroupLdStPass
`GroupLdStPass` 作用用是处理 GroupOp 的输入和输出，通过显式插入 Load 和 Store 操作，来“固化”和“隔离”GroupOp 的边界。

Load 插入流程
1. 识别 Load 需求: 函数遍历 GroupOp 的每一个输入参数v。然后，它查找所有在 GroupOp 外部使用 v 的算子 (userOp) 。通过检查这些userOp的属性 (needLoad) ，它判断哪些 userOp 需要一个显式的 Load 操作来获取 v 的值。
2. 处理特殊布局: 代码中有一段特殊的逻辑 (`if(isa<...>)`) ，用于处理 Add、Sub 等二元算子。它检查输入的layout 如果存在不匹配的情况 (例如一个NCx布局和一个Tensor布局) ，它可能会强制layout统一，以确保硬件能够正确计算。
3. 插入 LoadVarOp: 在确定了所有需要 Load 的外部用户后，如果这样的用户存在 (`usersLoad.size() != 0`)，它会在GroupOp的入口处创建一个tx8e::LoadVarOp操作。
4. 重定向数据流: 将所有外部用户对原始输入 v 的连接 (SSA use-def chain) ，全部断开，并重新连接到新创建的LoadVarOp的输出上 (replaceUsesOfWith).

Store 插入流程
1. 识别存储需求: 函数找到 GroupOp 内部的 return 操作，并遍历它的每一个操作数 (即 GroupOp 的输出值). 通过检查产生这些输出值的内部算子 (pre_op) 的needStore属性，来判断哪些输出需要被显式地Store，以便外部世界能够访问。
2. 插入 StoreVarOp: 如果一个输出值需要被存储，函数会在 GroupOp 的末尾、return 操作之前，创建一个tx8e::StoreVarOp 接收 GroupOp 的内部计算结果。
3. 更新返回结果: StoreVarOp本身也有一个输出。函数会更新 GroupOp 的 return 操作，使其返回 StoreVarOp 的输出，而不是原始的内部计算结果。


{{< details title = "GroupLdStPass Implementation" >}}
```cpp
void GroupLdStPass::runOnOperation() {
  subgraph.walk([&](tx8e::GroupOp g_op) {
    // ...
    //  For each group's input, insert a load. If used by multiple ops, multiple loads are inserted
    for (auto v : g_op.getBody().front().getArguments()) { // Iterate over each input argument of the group.
        Operation* pre_op = getValidDefiningOp(v); // Find the operation that produces this input.
        // ...
        std::map<Operation*, int32_t> usersLoad; // A map to store users that need to load this input.

        for (auto userOp : v.getUsers()) { // Find all users of this input argument.
            // ...
            // Check if the user needs a 'load' based on its attributes.
            if ((!opAttr.needLoad && (1 << arg_idx))) {
                continue;
            }
            // If a load is needed, record the user and its argument index.
            usersLoad.insert(std::make_pair(userOp, arg_idx));

          // This block handles complex layout logic for Add/Sub/Mul/Div ops.
          // It seems to ensure that if one input to 'add' is rank1 tensor, the other is also handled correctly,
          // potentially by forcing a specific layout (`LayoutMode::Cx`).
          if (isa<tx8e::AddOp, tx8e::SubOp, tx8e::DivOp, tx8e::MulOp>(userOp)) {
              // ... [复杂布局逻辑]
          }
          
          if (usersLoad.size() != 0) {  // there are users that require a load operation.
              std::vector<NamedAttribute> tmp_attrs;
              // ... [构建LoadVarOp的属性]
              // Create the Load operation.
              auto ld = builder.create<tx8e::LoadVarOp>(g_op.getLoc(), v.getType(), v, tmp_attrs);
              // ... [设置动态shape属性]
              
              // For each user that needs the load...
              for (auto userOp : usersLoad) {
                  // ...replace its use of the original input `v` with the result of the new `Load` operation `ld`.
                  userOp.first->replaceUsesOfWith(v, ld.getOutput());
              }
          }
      }

      // For each group's output, insert a store
      builder.setInsertionPointToEnd(&block); // Set the insertion point to the end of the group's body.
      Operation *g_return = g_op.getBody().front().getTerminator(); // Get the return operation of the group.

      for (int i = 0; i < g_return->getNumOperands(); ++i) { // Iterate over each output of the group.
          auto value = g_return->getOperand(i);
          auto pre_op = value.getDefiningOp(); // Find the operation inside the group that produces this output.
          // ...
          // Check if this output value needs to be stored for external users.
          if (!(llvm::dyn_cast<tx8e::OpLibInterface>(pre_op)).queryOpAttr().needStore && (1 << i)) {
              continue;
          }

          // ... [构建StoreVarOp的属性]
          // Create the Store operation.
          auto st = builder.create<tx8e::StoreVarOp>(g_op.getLoc(), value.getType(), value, st_attrs);
          // ... [设置动态shape属性]

          // Update the group's return instruction to return the result of the store op.
          g_return->setOperand(i, st.getOutput());
      }

      g_return->moveBefore(gBlock, block.end()); // Move the return instruction (not standard MLIR, might be custom logic).
      updateIR(g_op); // Update the IR of the group op.
    }
  });
}
```
{{< /details >}}

## GroupMappingPass

`GroupMappingPass` 作用是将顶层模块 (Module) 中定义的全局维度信息 (x_dim 和 y_dim) 设置到每一个 GroupOp 或 GroupOp 的调用点上。

{{< details title = "GroupMappingPass Implementation" >}}
```cpp
// Defines a function to perform a simple mapping of groups.
void simpleGroupMapping(ModuleOp module) {
  // Get x and y dimension from the module's attributes.
  // These attributes are likely defined globally for the entire compilation.
  uint32_t x_dim = module->getAttrOfType<IntegerAttr>(tx8e::ModuleAttr::TileDx).getInt();
  uint32_t y_dim = module->getAttrOfType<IntegerAttr>(tx8e::ModuleAttr::TileDy).getInt();

  // Create an OpBuilder instance, which is a helper to create/modify MLIR operations.
  OpBuilder builder(module.getContext());
  // Get the 'main' function from the module.
  func::FuncOp main = module.getMainFuncOp();
  // Get the first block (entry block) of the main function.
  auto& main_block = main.getBody().front();

  for (auto& inner : main_block.getOperations()) {  // Iterate over all operations within the main function's body
    if (isa<tx8e::CallOp>(inner)) {  // The module's main function contains CallOps. This implies an indirect call to a subgraph.
      // Find the subgraph definition ('SubraphOp') using the symbol name from the CallOp.
      tx8e::SubgraphOp sg = module.lookupSymbol<tx8e::SubgraphOp>(
          llvm::dyn_cast<tx8e::CallOp>(inner).getCallee());
      
      // Walk through the operations inside the called subgraph.
      // We are looking for the 'GroupOp' which is the actual unit of computation.
      sg.walk([&](tx8e::GroupOp gop) {
        // Set a 'dev_region' attribute on the located GroupOp.
        setDevRegionAttr(builder, module.getContext(), gop.getOperation(), x_dim, y_dim);
      });
    }

    if (isa<tx8e::GroupOp>(inner)) {  // The module's main function directly contains GroupOps.
      // Directly set the 'dev_region' attribute on the GroupOp found in the main function.
      setDevRegionAttr(builder, module.getContext(),
                       llvm::dyn_cast<tx8e::GroupOp>(inner).getOperation(), x_dim, y_dim);
    }
  }
}

void GroupMappingPass::runOnOperation() {  // It will operate on the entire ModuleOp.
  auto module = getOperation();
  simpleGroupMapping(module);
}
```
{{< /details >}}

## GroupCostPass
`GroupCostPass` 作用是为一个 GroupOp 在所有可能的切分策略中，通过 Cost Model 搜索并应用最优的一个。算法流程如下。

准备阶段 (Preparation):
  1. Bailout Condition: `if (gop->hasAttr("group_tag") && ... == 2) return;` 如果 GroupOp的 `group_tag==2`，那么这个 Pass 就无需为它搜索切分策略了，直接返回。
  2. 拷贝编译选项: `costoption_lg.dynCompile = compileOption_->dynCompile;` 从一个全局的`compileOption_` 中拷贝了一系列编译参数到局部的 costoption_lg 中. 表明 Pass 的行为可以被外部配置所影响。
  3. 创建搜索空间: `auto space = std::make_shared<SliceSpace>();` 创建了一个名为 space 的对象，这个 SliceSpace 类封装了该 GroupOp 的完整搜索空间。它包含了所有可能的张量切分方式。
  3. 模板机制: `if (useTemplate) { ... }` 检查 `compileOption_->sliceHelpMap` 的全局映射。如果之前已经为相似的 GroupOp (由 GroupKey 标识) 计算过最优策略，它就会直接从缓存中读取结果 (sliceHelp) ，从而避免昂贵的重复搜索。如果找到了模板，它会直接应用并提前返回。

搜迭代搜索循环 (The Core: Iterative Search Loop)

  1. `while (1)` 循环: 这个无限循环是搜索算法的主体。
  2. 探索策略: 在循环内部，space对象会生成一个候选的切分策略。这通过 `space->shardingLevel` 和`space->factorSpace_` 来控制，它们共同定义了当前正在尝试的切分维度和方式。
  3. 判断搜索是否完成: `if (space->shardingLevel.isSpaceFinish() && ...)`. 在每次迭代开始时，它会检查是否已经遍历了所有的切分可能性。如果搜索空间已耗尽，循环就会终止。
  4. 成本估算: 如果找到一个有效的候选策略，接下来就是估算这个策略的成本。动态构建Pass流水线: 
  - `auto pm = std::make_unique<LgPassManager>(...);` 添加一系列估算Pass:
    - `pm->add_pass(createDataSplitNewPass(space));` // 根据策略进行数据切分
    - `pm->add_pass(createTimeStepNewPass(space));` // 划分时间步
    - `pm->add_pass(createSPMAllocPass(space));`    // 模拟SPM (片上内存) 分配
    - `pm->add_pass(createEstimatePass(space));`    // 估算性能/成本
  - 运行估算流程: `pm->run(gop);`
  5. 比较和选择最优解: 估算完成后，`space->status` 会被更新 (SSTATUS_OK 表示估算成功，SSTATUS_SA_MemAlloc 表示内存分配失败) . 如果估算成功，它会获取成本 t，并与已知的 bestCost 进行比较。如果当前策略更优，就更新 bestCost 和 bestStrategy。

应用最优策略 (Applying the Best Strategy)
  1. 应用策略: `sliceHelp.strategy = space->strategy;` 和后续的 `compileOption_->IRHelp.ops[gop] = space->stageOps;` 等赋值操作，就是将搜索到的最优策略结果 (包括每个操作的切分方式、循环信息等) 保存到 compileOption_中，供后续的 Pass 使用。
  2. 具体计算: `gop->walk(...)` 它遍历 GroupOp 内部的操作 (如GemmOp) ，并根据策略 (lSharding, rSharding) 计算出具体的循环边界 (ls, rs) 和分片长度 (pLen) ，这些信息会被存入 gls (`GroupLoopSpace`) 对象中。

### DataSplitNewPass

其中也包括好几个 pass
`DS_SpaceInitPass` 作用是初始化分布式策略的搜索空间。对 groupOp 中的每一个算子，它会调用 `space_->shardinglevel.init` 这个函数会根据算子自身的特性、全局约束 (如 max_sharding) 以及用户配置 (如 opt_search) ，生成该算子所有可能的切分方式。

`init` 函数首先获取了算子的维度 dim 和目标切分路数 maxSharding，然后调用 getShardings 找出一个张量在所有维度上进行整数倍切分、且总切分路数恰好等于 maxSharding 的所有组合来填充 shardings 列表。随后，将这些组合 (并额外加上了不切分的方案) 包装成带有性能评估因子的 ShardingSpace 对象，并存入一个有序集合 `std::set<ShardingSpace> spaces` 中。ShardingSpace 重载了小于操作符用于对切分策略排序。

```cpp
struct ShardingSpace {
  std::list<ShardingInfo> shardings;
  // 预估的性能参数，即空间上能用到pow(2,x)个tile
  uint8_t factor[4];

  // 关键点：重载小于操作符，定义排序规则
  bool operator<(const ShardingSpace &other) const {
    // 性能高的在前面
    return factor > other.factor;
  }

  bool operator==(const ShardingSpace &other) const {
    return factor == other.factor;
  }
};
```

```cpp
void ShardingLevel::init(mlir::Operation* op, int32_t maxSharding, bool nFirst, int32_t opt_search) {
  // ... 清理和准备工作 ...
  
  // 1. 获取算子输出Tensor的维度数量 (Rank) 
  int32_t dim = op->getResult(0).getType().cast<ShapedType>().getRank();

  // 2. 准备容器
  std::vector<std::vector<int32_t>> shardings; // 用于接收所有合法的sharding方案
  std::vector<int32_t> tempSharding(dim, 0);   // 一个临时的、大小为dim的向量，用于递归

  // 3. 调用核心递归函数，启动搜索
  //    - curDim=0: 从第0维开始搜索
  //    - allDim=dim: 总共有dim个维度
  //    - curSharded=1: 当前已累乘的切分系数为1 (乘法单位元) 
  //    - maxSharding: 最大切分数目，即为每个 chip 的 tile 数目 (16)
  getShardings(0, dim, 1, maxSharding, shardings, tempSharding);

  // 4. 手动添加“不切分”的方案
  //    递归函数只会寻找乘积等于maxSharding的组合，但[1, 1, ..., 1] (不切分)
  //    是一个非常重要的基础方案，这里手动添加进去。
  shardings.push_back(std::vector<int32_t>(dim, 1));
  
  // ... 后续处理 ...
  for (auto sharding : shardings) {
    ShardingSpace newShardingSpace;
    if (isValid) {
      // 1. 为每个sharding方案计算性能因子
      newShardingSpace.factor = getFactor(op, sharding);
    }

    // ... (省略部分逻辑) ...
    
    // 2. 将包含factor的ShardingSpace对象插入set中
    spaces.insert(newShardingSpace);
  }
}
```

`getShardings` 函数采用的是递归算法，目标是找到所有整数向量 `s = {s_0, s_1, ..., s_{dim-1}}`，使得 `s_0 * s_1 * ... * s_{dim-1} == maxSharding`. 

```cpp
void ShardingLevel::getShardings(int32_t curDim, int32_t allDim, int32_t curSharded, int32_t maxSharding,
                                std::vector<std::vector<int32_t>>& shardings, std::vector<int32_t>& sharding) {
  // 1. 递归终止条件 (Base Case) 
  if (curDim == allDim) { // 已经处理完所有维度
    if (curSharded == maxSharding) { // 并且累乘结果正好等于目标
      // // succeeded
      shardings.emplace_back(sharding); // 找到了一个合法解，存入结果列表
    }
    return; // 回溯
  }

  // 2. 递归主体：遍历当前维度的所有可能切分系数
  for (int32_t i = 1; i <= maxSharding; ++i) {
    // 尝试将当前维度(curDim)的切分系数设为 i
    sharding[curDim] = i;
    // 更新已累乘的切分系数
    curSharded *= i;

    // 3. 剪枝优化 (Pruning) ：这是算法效率的关键！
    // 如果当前累乘的结果已经超过了目标，那么无论后续维度如何取值，
    // 最终结果必然大于 maxSharding，所以没有必要继续递归下去了。
    if (curSharded <= maxSharding) {
      // 如果还有希望，则对下一个维度进行递归搜索
      getShardings(curDim + 1, allDim, curSharded, maxSharding, shardings, sharding);
    }

    // 4. 回溯 (Backtracking) 
    // 无论上面的递归是否成功，当它返回后，我们需要“撤销”当前的选择，
    // 以便在 for 循环的下一次迭代中尝试新的值。
    curSharded /= i; 
  
  }
}
```

`getFactor` 遍历每个维度，基于内存对齐等硬件限制，计算出该维度上最大合理的切分数量 maxShardingDim.
将用户提议的切分数量 `sharding[i]` 与 maxShardingDim 取最小值，得到该维度上的有效切分数量。将所有维度上的有效切分数量相乘，得到总的有效并行度 tileNum. 对 tileNum 取以2为底的对数并向上取整后返回。

```cpp
uint8_t ShadingLevel::getFactor(mlir::Operation* op, std::vector<uint32_t> sharding) {
  int tileNum = 1;
  for (int i=0; i<rank; ++i) {
    // a. 判断是否需要对齐：这里只对最后一个维度特殊处理
    bool align = i == rank - 1;

    // b. 获取对齐基数 (alignBase)
    //    如果需要对齐，则调用 GetAlignBase 获取一个对齐值，否则为1 (相当于不对齐) 。
    //    这个 alignBase 很可能代表硬件一次最优处理的最小数据块大小。
    uint32_t alignBase = align ? GetAlignBase(shape[i], dtype) : 1;

    // c. 计算当前维度的最大合理切分路数 (maxShardingDim)
    //    一个维度能被切成多少份，不仅取决于它的总大小，还取决于对齐要求。
    //    例如，一个维度大小为100，但硬件要求必须按16对齐处理，那么最多只能切成 ceil(100/16) = 7 份。
    auto maxShardingDim = CEIL(shape[i], alignBase);

    // d. 计算“有效”的切分路数并累乘
    //    这是关键！它在“提议的切分路数(sharding[i])”和“最大合理切分路数(maxShardingDim)”之间取最小值。
    //    这意味着，即使你提议将一个维度切16份，但如果硬件限制最多只能切7份，那也只能算7份的贡献。
    //    这可以防止对一个维度进行“无效的过度切分”。
    tileNum *= MIN(maxShardingDim, sharding[i]);
    return static_cast<uint8_t>(std::ceil(log2(tileNum))); // 向上取整
  }
}
```

一个例子如下
```
storeOp outShape[3, 4, 128, 4096]
level0: [1, 1, 1, 16], [1, 1, 2, 8], [1, 1, 4, 4]...   factor=16
level1:[1, 8, 1, 2], [1, 8, 2, 1]....                  factor=8
level2:[1, 16, 1, 1]                                   factor=4 
level3:[16, 1, 1, 1]                                   factor=2
```
`DS_TileShardingPass` 按顺序遍历 groupOp 中的算子，并像一个状态机一样检查和更新各算子的分布式策略状态。其在每次执行时，仅为当前的待定算子，从其预先生成并排好序的候选策略列表中，选出下一个最优的切分方案并进行更新，然后立即终止当次运行。整个图的最终切分方案是通过反复执行此 Pass，将决策从图的入口逐步传播到出口而最终确定的。

![An Example of Sharding](https://share.note.youdao.com/yws/api/personal/file/WEBda0206ff3ade37b3cb94b73f3a564489?method=download&shareKey=bd6f962093568ee599a982e7b7b1a300 "An Example of Sharding")

`DS_TileSplitPass` 首先检查算子是否需要 `reduceSplit` (例如 GeMM 切分 k 维度). 如果 reduce 维度切分状态为 `s.srfinish = true` 才会进行后续的 split 方案。
1. 从后向前 (或根据 nFirst 标志决定方向) 检查算子的各个维度，找到第一个“还可以再切分”的维度。判断依据是该维度切分后的大小是否已达到系统设定的最小值 (s.sliceShapeMin) .
2. 一旦找到目标维度 updateDim，它会调用一个名为 `getNextSplit` 的函数。它会根据当前维度的切分值 `s.splitTry[updateDim]` 计算出下一个可能的切分值。例如，如果当前是 2，getNextSplit 可能会返回 4.
3. 更新与记录：它将这个新的切分值更新到尝试性方案 `s.splitTry` 中，并记录下这次更新`space_->splitRecord.update(...)`.
4. 在对当前算子的循环结束时，它会将探索出的 `s.splitTry` 赋值给最终方案 `s.split`.

![An Example of Split try of above Sharding](https://share.note.youdao.com/yws/api/personal/file/WEBab1aedb258273670b60e2b54295f1f6c?method=download&shareKey=afd6b38561485627a11fd118670b8431 "An Example of Split try of above Sharding")


`DS_SlicePropagatePass` 后序遍历 (即从 groupOp 的输出到输入) 的方式反向传播切分决策，其逻辑是：对于每一个算子 (消费者)，它会调用该算子实现的 `ShardingInterface` 接口中的 `tileShardingSplit` 方法，来精确计算出其上游算子 (生产者) 应该如何切分数据以满足消费者的需求。这如果自动接口推导失败，它会回退去读取算子上预设的 `tile_parallel` 属性作为人工指令。

![An Example of Propagation](https://share.note.youdao.com/yws/api/personal/file/WEBb14ede2d32d997490222f36c1f0acc21?method=download&shareKey=a244cf2ec9b5f31b5f3ef0ff2aa370a0 "An Example of Propagation")

`DS_UpdateSliceIRPass` 核心策略是通过分析图中 reduceOp 来反向推断和划分流水线阶段。通过检查每个 reduceOp 自身的并行复杂度 (例如，tpSplit > 1) 来判断其上游的计算类型，从而为不同的流水线打上诸如 STAGEIC2OC (模型并行规约段) 或 STAGEOC2NH (模式切换段) 的标签。在完成对所有算子的阶段划分后，它会最终计算每个阶段的流水线深度，并整理输出一份包含并行循环类型、算子分组和流水线阶段信息的完整执行。

1. 首先从 reduceOps 栈中取出一个关卡算子。然后，它利用 `getNEOPTPSlice` 等辅助函数，分析这个算子自身的切分策略，判断它具体采用了哪种张量并行方式。

2. 确定连接到当前这个 reduceOp 的上一段流水线是什么类型
  - `if (tpSplit > 1)`: 如果这个关卡算子本身是一个张量并行度大于 1 的算子，代码就判断出：通往这个算子的路径，是一段需要最终进行集合通信 (C) 的路径。因此，它将这段路径的类型标记为 `STAGEIC2OC`.
  - `else if (s.reduceSplit > 1)`：如果不是上面那种情况，代码会检查另一种模型并行模式。如果一个算子的规约维度被切分了，同样意味着后续需要一个 AllReduce 集合通信。因此，它把这段路径标记为 `STAGEIC2IC`.
  - 如果两个条件都不满足，意味着这可能是一个不同并行模式之间的切换，例如从模型并行切换回数据并行，此时会使用默认的 `STAGEOC2NH` 标记.
3. 通过 `updateLoopStage` 函数，将两个 reduceOp 算子之间的所有普通算子，都归类到刚刚在第 2 步中决策出的 lastRuduceLoopStage.
4. 处理完所有的 reduceOp 后遍历所有算子，根据 LoopStageMap_ 中的记录，将算子放入对应的“篮子”里。

![DS_UpdateSliceIRPass](https://share.note.youdao.com/yws/api/personal/file/WEBa90c52245265edade98cd9f5b15d59bb?method=download&shareKey=ccf9d74acc3642b1eac881133e98c6b9 "DS_UpdateSliceIRPass")

### TS_SwPipelinePass

TS_SwPipelinePass 核心是调用 getPipeline 函数。其内部通过顺序执行以下三个关键步骤，。

`getInitPipelineOps`
1. 为每个流水线阶段 (如 STAGENH2OC, STAGEOC2IC等) 创建一个独立的 pipeline 列表。
2. 按 IC -> OC -> NH 顺序来拼接这些列表。在拼接时，它会检查每个阶段的循环次数。如果循环次数大于1：它并不会简单地将操作列表复制多次，而是创建一个特殊的、类型为 PipelineOpsBase 的 **Repeat 节点**。这个节点内部包含需要重复的子流水线 (`repeatBase.repeat`) 和重复次数 (`repeatBase.repeatTimes`) . 然后，它将这个Repeat 节点作为一个单一的、原子性的元素，插入到下一个阶段的流水线中。这是一种高效表示嵌套循环的方法。
如果循环次数不为 1：它就直接使用 splice 操作，将当前阶段的算子列表完整地移动并拼接到下一个阶段的尾部。

经过层层拼接和嵌套，该函数最终返回一个名为 groupPipeline 的 std::list。这个列表就是一份完整的、线性的逻辑执行剧本，其中所有的嵌套循环都被抽象成了 Repeat 节点。

![getInitPipelineOps](https://share.note.youdao.com/yws/api/personal/file/WEB58fc5fb9b7b1fb4880eb020bce68afd5?method=download&shareKey=f4ae610fb730689a22fe38a7226ee6e5 "getInitPipelineOps")

`pipeline `

主要工作是处理上一阶段生成的 Repeat 节点，并对流水线的衔接处进行深度优化，以减少气泡 (硬件空闲周期) 。

1. 当它在流水线中遇到一个 Repeat 节点时，它会对该节点内部的子流水线再次调用pipeline函数 (`auto inner = pipeline((it).repeat, ...)`). 通过这种方式展开任意层级的嵌套循环。

2. 在处理循环的边界时，它调用 getRetract 和 doRetract 这对复杂的优化工具。
  - `getRetract`: 在连接两个循环迭代 (或不同的流水线段) 时，通过 canParallel 函数检查后一个迭代的“头部指令”是否可以和前一个迭代的“尾部指令”并行执行，从而计算出最大可以“回缩” (即提前执行) 的指令数量。
  - `doRetract`: 在 getRetract 探明了可回缩的数量后，doRetract 负责物理地修改流水线。它通过 splice 操作，将后一个迭代头部的指令，合并到前一个迭代尾部的指令列表中，从而填补了潜在的执行空隙。

`getEnginsPipeline` 将优化后的操作序列，翻译成具体的、分配到不同硬件引擎的指令。
1. 函数遍历输入的 pipelineOps 列表。列表中的每个元素 opsBase 代表一个流水线周期 (一“帧”) 内需要共同执行的一组MLIR操作。
2. 对于每个周期，它创建一个 enginsBase 对象。这个对象是一个结构体，包含了分别对应不同硬件引擎 (如 `ld` for Load, `st` for Store, `ne` for Neural Engine, `tdma` for DMA) 的成员变量。

3. 遍历当前周期的所有 op，通过查询每个 op 的 engine 属性 `queryOpAttr().engine`，得知这个操作预定由哪个硬件引擎来执行。接着，它将这个 op 的指针存放到 enginsBase 对象中对应的引擎 slot 里。例如，一个 `NPU_ENGINE_LOAD` 类型的操作会被放入 `enginsBase.ld` 列表。

函数最终返回一个 `std::list<PipelineBase>` 描述了在同一个时钟周期内，加载、存储、计算等多个硬件单元应该同时执行**哪些不同的操作。

### SPMAllocPass
SPMAllocPass 包括三个 pass，下面依次介绍，首先介绍用到的数据结构

`BufferLabel` 作为缓冲区的唯一标识符，将其链接到程序中的特定 `mlir::Value` ，并注意它是否为 Imm.
```cpp
/**
 * @struct BufferLabel
 * @brief A unique identifier for a memory buffer.
 *
 * This struct links a buffer to a specific MLIR Value and tracks whether it's
 * a special "immediate" buffer. It's used as a key in maps to associate
 * MLIR Values with their buffer metadata.
 */
struct BufferLabel {
    // The MLIR Value that this buffer represents, typically a tensor produced
    // by an operation.
    mlir::Value v;

    // A flag indicating if this buffer holds a special "immediate" value.
    // Immediate values might be treated differently during allocation (e.g.,
    // small constants or internal scratchpads for an op).
    bool isImm{false};

    /**
     * @brief Equality operator to compare two labels.
     *
     * Two labels are considered equal if they refer to the same MLIR Value
     * and have the same 'isImm' status. This is necessary for using
     * BufferLabel as a key in std::map or std::unordered_map.
     */
    bool operator==(const BufferLabel& other) const {
        return (v == other.v) && (isImm == other.isImm);
    }
};
```

`ValueBuffer` 包含单个缓冲区所需的所有元数据，包括其标识、生存期、大小和最终内存位置。

```cpp
/**
 * @struct ValueBuffer
 * @brief Represents the metadata for a single memory buffer, including its
 * lifetime, size, and allocation information.
 */
struct ValueBuffer {
    // The unique identifier for this buffer.
    BufferLabel label;

    // Represents the starting point of the buffer's lifetime (inclusive),
    // measured in pipeline cycles. After memory allocation, this field may be
    // repurposed to store the starting memory address.
    int64_t start;

    // Represents the ending point of the buffer's lifetime (inclusive),
    // measured in pipeline cycles. After memory allocation, this field may be
    // repurposed to store the ending memory address.
    int64_t end;

    // The total size of this buffer in bytes, as required by its tensor shape.
    int64_t allSize{0};

    // Size of an intermediate/temporary buffer that an operator might need
    // internally. This is often allocated contiguously with the main output
    // buffer. For example, the final output address would be 'offset + immSize'.
    int64_t immSize{0};

    // The final memory offset assigned to this buffer in the scratchpad memory.
    // This value is determined by the final memory allocation pass.
    int64_t offset{0};

    /**
     * @brief Less-than operator, used for sorting ValueBuffer objects.
     *
     * The active implementation sorts buffers primarily by their lifetime start
     * time. This is a common strategy for greedy "first fit" style memory
     * allocation algorithms. The commented-out code shows an alternative
     * strategy of sorting by buffer size.
     */
    bool operator<(const ValueBuffer& other) const {
        // return this->allSize < other.allSize; // Alternative sorting by size
        return this->start <= other.start;
    }
};
```

`SA_BufferLifePass`的核心功能是分析并确定每一个需要存放在 ScratchPad Memory 中的数据块 (Buffer，即mlir::Value对应的张量) 的生命周期。

1. 构建“定义-使用”时间表。Pass 的输入是 `TS_SwPipelinePass` 生成的最终流水线执行序列 pipelineReal. 这个序列的每一项都代表一个流水线周期，以及该周期内各个硬件引擎执行的操作。代码遍历这个流水线序列，逐个周期 (由timeStepNum计数) 地分析。它会构建两个核心的映射表：
  - `opIsTemp`: 记录在哪一个时间步 (timeStepNum) ，有哪些值 (mlir::Value) 被定义或产出。例如，ld (加载) 和 ne (计算) 操作的输出都会被记录。
  - `consumerOps`: 记录在哪一个时间步，有哪些值被作为输入消费掉了。

产出：这个步骤完成后，Pass就拥有了一份完整的、按时间步索引的“谁在何时被创建”和“谁在何时被使用”的清单。

2. 确定每个Buffer的生命周期。Pass会遍历所有算子和它们的输入 (operands) ，为每一个作为输入的 Value (即inValue) 计算其生命周期。
  - 确定生命周期终点 (end)：一个 Value 的生命周期，在其被作为输入 (被消费) 时达到一个终点。因此，当代码在时间步 curTs 处理一个消费者算子时，其输入 inValue 的 `buf.end` 就被设置为 curTs.
  - 确定生命周期起点 (start)：为了找到inValue何时被创建，代码会调用一个 `getNearestProducer` 的函数。这个函数会拿着当前的消费时间 curTs 和 inValue，去第一步生成的 opIsTemp (定义时间表) 中反向查找，找到离 curTs 最近的、inValue 被定义的那个时间步 `buf.start`.

计算出的 start 和 end，连同 Value 的标识 (BufferLabel) ，被封装在 ValueBuffer 结构体中，并存入一个全局的数据结构 `space_->vBuffer` 里。

3. 特殊情况处理
  - `In-place`: 对于输入和输出复用同一块内存的 in-place 操作，其生命周期计算必须追溯到最初提供这块内存的那个非in-place算子。代码通过 `getInplaceIndex` 递归地回溯in-place链，以确保生命周期的 start 时间是正确的、最开始的那个定义时间。
  - 中间值 (imm) 与累加值 (Psum): 代码会识别一些特殊的、可能在多个时间步中存在的中间值或累加值 (由getImmSize 或 isPsumValue 识别) . 对于这些值，它们可能会有多个离散的生存区间。Pass 中可能包含一些后处理逻辑，将这些离散的区间合并成一个从“最早的start”到“最晚的end”的连续大区间，以简化后续的内存分配。

`SA_BufferMergePass` 的任务就是清理这些冗余或复杂的生命周期记录，具体来说，就是合并那些存在时间上重叠或包含关系的生命周期区间，为后续的内存分配器提供一个最簡洁、无冗余的区间列表。

遍历由上一个 Pass 生成的 space_->vBuffer 这个map. 其中的每一项，key 是缓冲区的唯一标识 BufferLabel，value是该缓冲区所有生命周期区间的列表 `std::vector<ValueBuffer>`. 对于每一个value的生命周期列表，它都调用 `mergeOverlap` 来进行处理。最后，它用函数返回的、经过清理和合并的新的列表，来替换掉 map 中旧的列表。该函数流程如下

1. 根据 ValueBuffer 重载的 `operator<` (即按 start 时间升序) ，将所有生命周期区间进行排序。
2. 遍历已排序的列表，将 start 时间相同的连续区间收集到一个临时的 buf 向量中。遇到一个不同 start 时间的区间时，它会按照结束时间 end 排序之前收集的 buf，然后将处理后的结果 (除了最后一个元素) 重新放回 valueBuf.
3. 合并被完全包含的子区间。它维护着当前最大的生命周期区间 (`[usedTSStart, usedTSEnd]`). 遍历列表中的每一个区间 `*it`. 根据 `bool isSub = ((*it).start >= usedTSStart) && ((*it).end <= usedTSEnd);` 判断区间是否在时间上被上一个“激活”的区间完全覆盖。
  - 如果 isSub 为 true，意味着 *it 是一个冗余的子区间。因为只要为那个更大的激活区间分配了内存，这个子区间的内存需求自然也就满足了。因此，代码通过 valueBuf.erase(it); 将这个冗余的子区间直接删除。
  - 如果 isSub 为 false，说明遇到了一个新的、没有被覆盖的生命周期，于是它将成为新的“激活”区间，用于和后续的区间进行比较。

# Compile Option 1: opt_barrier

由 `groupDAGPass` 实现。通过 group 间的依赖关系来给 group 定层级，同一层级的 group 只有最后一个 group 需要 barrier. 
1. 初始化所有 group 的 need_barrier 属性为 false。

2. 从后往前遍历 group，若 group 的结果无 user 或要 return，设置 layerNum 为 0，否则设置为 userOp 的 layerNum + 1. 同时维护两个 vector: firstOpInLayers 和lastOpInLayers 来记录每一层级的第一个 op 和最后一个 op. 遍历结束把 lastOpInLayers中的 group 的 need_barrier 属性设为 true.

![opt_barrier](https://share.note.youdao.com/yws/api/personal/file/WEB8e3541f411fb9039713f3992b450f4b9?method=download&shareKey=4f5dfad0d8b5a4bd8cf49ce94d5ad7c9 "opt_barrier")

# Compile Option 1: opt_ddr

由 `ddrConstReorderPass` 和 `ddrVarReorderPass` 实现。通过改变 const 和 var 在 ddr 中的排布，使其对齐 DDR_BANK(4096Bytes)，实现加速读取。

![opt_ddr](https://share.note.youdao.com/yws/api/personal/file/WEBd3b8e4e14ab8e2c23f22a625b1d03ddd?method=download&shareKey=948a94db71e0adcddb5f6107ad94a6d0 "opt_ddr")