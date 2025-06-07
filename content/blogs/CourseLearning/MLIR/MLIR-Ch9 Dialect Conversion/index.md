---
title: MLIR-Ch9 Dialect Conversion
date: 2024-11-12T15:22:23+08:00
lastmod: 2024-11-12T15:22:23+08:00
draft: false
author: ["WITHER"]
keywords: 
    - MLIR
categories:
    - MLIR
tags:
    - jeremykun MLIR learning
description: Personal MLIR learning notes 9.  
summary: Personal MLIR learning notes 9.  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
MLIR 的主要原则之一是逐步下降，即存在许多级别的 IR 粒度，并且逐步下降 IR 的不同部分，仅在不再对优化有用时丢弃信息。在本文中，将完成其中的第一步：使用所谓的方言转换基础设施将多方言 lowering 为标准MLIR方言的组合。

# The Type Obstacle

如果不是针对类型，方言转换 (lowering) 本质上与普通 pass 相同：编写一些重写模式并将其应用于 IR. 对于每个需要 lowering 的 OP ，通常会有一个重写模式。

类型使这个问题变得更加复杂，我将通过poly的示例来演示这个问题。

poly.add 对两个多项式进行相加并返回结果多项式。我们想 lowering poly。例如，添加到 arith.addi 算术运算的矢量化循环中。但 arith 并不知道 poly.poly 类型的存在。

如果必须使扩展 arith 以了解poly，需要对 arith 进行上游更改。添加 op 的 operands 以允许实现某种接口的类型，例如 integer-like 或 containers of integer-like.

所以，除了 lowering op，还需要 lowering poly.` poly<N>` 变成张量 `<Nxi32>`. 这就是类型障碍发挥作用的地方。一旦更改了特定值的类型，例如，在 lowering 生成该值作为输出的 OP 时，那么该值的所有下游用户仍然期望使用旧类型，并且在 lowering 它们之前在技术上是无效的。在每次传递之间，MLIR运行验证器以确保IR有效，因此如果没有一些特殊处理，这意味着需要在一次传递中转换所有类型和 OP ，否则这些验证器将失败。但是用标准重写规则管理所有这些将是困难的：对于每个重写规则，您都必须不断检查参数和结果是否已经转换。

例如在 lowering 一个生成该值作为输出的 OP 时，所有依赖该值的下游用户仍然期望旧的类型，因此在技术上这些下游用户在未被 lowering 之前是无效的。MLIR 在每次转换 (pass) 之间运行验证器以确保中间表示 (IR) 是有效的，因此如果没有特殊处理，这意味着所有类型和 OP 必须在一个转换中全部转换，否则验证器会失败。但是，使用标准的重写规则来管理这一切会很困难：对于每个 OP 重写规则，你需要不断地检查参数和结果是否已经转换。

MLIR 通过一个围绕标准转换的包装器来处理这种情况，这个包装器被称为[方言转换框架(dialect conversion framework)](https://mlir.llvm.org/docs/DialectConversion/). 使用这个框架需要用户继承不同的类来实现普通的重写，设置一些额外的元数据，并以特定的方式 `将类型转换与 OP 转换分开`，我们稍后会看到具体方式。但从高层次来看，这个框架通过以某种排序顺序 lowering  OP 、同时转换类型，并让 OP 转换器能够访问每个 OP 的原始类型以及在 OP 被框架访问时的进行中的转换类型。每个基于 OP 的重写模式都期望在访问后使该 OP 的类型合法，但不需要担心下游 OP.

## Modes of Conversion

当对一组 OP 进行转换时，有几种不同的转换模式可供选择：

- Partial Conversion
  - 使尽可能多的对目标的操作合法化，但将允许未显式标记为“非法”的预先存在的操作保持未转换。这允许在存在未知操作的情况下部分降低输入。
  - 可以通过 `applyPartialConversion` 进行部分转换。
- Full Conversion
  - 使所有输入操作合法化，并且只有当所有操作都正确地合法化到给定的转换目标时才成功。这确保了在转换过程之后只存在已知的操作。
  - 可以通过 applyFullConversion 进行完整转换。
- Analysis Conversion
  - 如果要应用转换，`Analysis Conversion` 将分析哪些操作对给定的转换目标是合法的。这是通过执行 'Partial' Conversion 并记录哪些操作如果成功将被成功转换来完成的。注意，没有 rewrites 或转换实际应用于输入操作。
  - 可以通过 a `pplyAnalysisConversion` 应用分析转换。

## Conversion Target

转换目标是在转换过程中被认为是合法的内容的正式定义。转换框架生成的最终操作必须在converontarget上标记为合法，这样重写才能成功。根据转换模式的不同，现有操作不一定总是合法的。操作和方言可以标记为下列任何规定的合法性行为：

- Legal: 表明给定操作的每个实例都是合法的，即属性、操作数、类型等的任何组合都是有效的。
- Dynamic: 此操作表示给定操作的某些实例是合法的。这允许定义微调约束，例如，`arith.addi` 仅在操作32位整数时合- Illegal: 此操作表示给定操作的实例不合法。为使转换成功，必须始终转换标记为“非法”的操作。此操作还允许有选择地将特定操作标记为非法，否则将是合法的方言。

未明确标记为合法或非法的操作和方言与上述（“未知”操作）分开，并被区别对待，例如，出于上述部分转换的目的。

最后，方言转换框架会跟踪任何未解决的类型冲突。如果在转换结束时仍存在类型冲突，会发生以下两种情况之一。转换框架允许用户可选地实现一个称为类型物化器 (type materializer) 的功能，它会插入新的中间 OP 来解决类型冲突。因此，第一种可能是方言转换框架使用你的类型物化器钩子来修补 IR，转换成功结束。如果这些钩子失败，或者你没有定义任何钩子，那么转换会失败。

这种基础设施的复杂性部分还与上游 MLIR 中一个更困难的 lowering 流水线有关：缓冲区化流水线 (bufferization pipeline). 这个流水线本质上将使用 value semantics 的操作的 IR 转换为使用 pointer semantics 的中间表示。例如，张量类型 (tensor type) 及其相关操作具有 value semantics，这意味着每个操作在语义上都会生成一个全新的张量作为输出，并且所有操作都是 pure 的 (有一些例外情况) 。另一方面， memref 具有 pointer semantics，意味着它更接近于对物理硬件的建模，需要显式的内存分配，并支持对内存位置进行变动的操作。

由于缓冲区化过程复杂，它被拆分为 sub-passes，分别处理与上游 MLIR 各相关方言特定的缓冲区化问题 (参见文档，例如 arith-bufferize、func-bufferize 等) 。每个缓冲区化转换都会产生一些内部无法解决的类型冲突，这些冲突需要自定义的类型物化 (type materializations) 来解决。为了在所有相关方言中处理这些问题，MLIR 团队构建了一个专门的方言，称为缓冲区化方言 (bufferization dialect) ，用来存放中间操作。你会注意到像 to_memref 和 to_tensor 这样的操作，它们扮演了这一角色。然后还有一个最终缓冲区化转换 (finalizing-bufferize pass) ，其作用是清理任何残留的缓冲区化或物化操作。

# Lowering Poly with Type Materializations

跟之前写 Pass tablegen 的时候大同小异，主要是需要定义 dependent dialects. Lowering 必须以这种方式依赖于包含将创建的操作或类型的任何方言，以确保 MLIR 在尝试运行 pass 之前加载这些方言。

```cpp {linenos=true}
// include/Conversion/PolyToStandard/PolyToStandard.td

#ifndef LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_
#define LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_

include "mlir/Pass/PassBase.td"

def PolyToStandard : Pass<"poly-to-standard"> {
  let summary = "Lower `poly` to standard MLIR dialects.";

  let description = [{
    This pass lowers the `poly` dialect to standard MLIR, a mixture of affine,
    tensor, and arith.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tutorial::poly::PolyDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_CONVERSION_POLYTOSTANDARD_POLYTOSTANDARD_TD_
```

下一步需要定义 ConversionTarget，告诉 MLIR 哪些 OP 需要进行 lowering，可以定义整个需要下降的 dialect 为 illegal，确保在转换完成后没有该 dialect. 这里使用 `applyPartialConversion` 而不是 `applyFullConversion` 的原因是报错消息更直观。Partial Conversion 可以看到步骤以及最后无法修补的冲突类型。

```cpp {linenos=true}
// lib/Conversion/PolyToStandard/PolyToStandard.cpp

struct PolyToStandard : impl::PolyToStandardBase<PolyToStandard> {
  using PolyToStandardBase::PolyToStandardBase;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    // TODO: implement pass

    ConversionTarget target(*context);
    target.addIllegalDialect<PolyDialect>();  //  declare an entire dialect as “illegal”

    RewritePatternSet patterns(context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
```

接下来需要定义一个 [TypeConverter](https://github.com/llvm/llvm-project/blob/11b9ec5f240ebb32013c33b0c2c80cb7f05ba213/mlir/include/mlir/Transforms/DialectConversion.h#L38) 的子类将 poly dialect 下的 type 转换成其他类型. 其中类型转换和 materialization 是分别通过 `addConversion` 和 `addMaterialization` 完成的。这里我们将属于 poly.poly 类型的 degreBound 转换成 Tensor.

```cpp {linenos=true}
class PolyToStandardTypeConverter : public TypeConverter
{
public:
    PolyToStandardTypeConverter(MLIRContext* ctx)
    {
        addConversion([](Type type) { return type; });
        addConversion([ctx](PolynomialType type) -> Type {
            int degreeBound = type.getDegreeBound();
            IntegerType elementType = IntegerType::get(
                ctx, 32, IntegerType::SignednessSemantics::Signless);
            return RankedTensorType::get({degreeBound}, elementType);
        });
    }
};
```

接下来就是要转换 Poly 中的各种 op，需要继承 [OpConversionPattern](https://github.com/llvm/llvm-project/blob/11b9ec5f240ebb32013c33b0c2c80cb7f05ba213/mlir/include/mlir/Transforms/DialectConversion.h#L511)，重写里面的 `matchAndRewrtite` 方法. 以 poly.add 为例，根据父类里的定义，这里 `OpAdaptor` 即为 `AddOp:OpAdaptor`，它使用 tablegen 定义的名称作为 op 的参数和方法名称的结果，而不是之前的的getOperand. `AddOp` 参数包含原始的、未类型转换的操作数和结果。ConversionPatternRewriter类 似于PatternRewriter，但有与方言转换相关的其他方法，例如 convertRegionTypes，用于为嵌套区域的操作应用类型转换。对IR

```cpp {linenos=true}
struct ConvertAdd : public OpConversionPattern<AddOp>
{
    ConvertAdd(MLIRContext* context) : OpConversionPattern<AddOp>(context)
    {
    }

    using OpConversionPattern::OpConversionPattern;

    LogicalResult matchAndRewrite(
        AddOp op, OpAdaptor adaptor,
        ConversionPatternRewriter& rewriter) const override
    {
        auto addOp = rewriter.create<arith::AddIOp>(
            op->getLoc(), adaptor.getLhs(), adaptor.getRhs());
        rewriter.replaceOp(op.getOperation(), addOp);
        return success();
    }
};
```

下面我们需要将 ConvertAdd 添加进 `PolyToStandard::runOnOperation` 中定义的 RewriterPatternSet 中。

```cpp {linenos=true}
void runOnOperation() {
  ...
  RewritePatternSet patterns(context);
  PolyToStandardTypeConverter typeConverter(context);
  patterns.add<ConvertAdd>(typeConverter, context);
}
```