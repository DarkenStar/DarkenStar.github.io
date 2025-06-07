---
title: MLIR-Ch7 Verifiers
date: 2024-11-10T23:51:23+08:00
lastmod: 2024-11-10T23:51:23+08:00
draft: false
author: ["WITHER"]
keywords: 
    - MLIR
categories:
    - MLIR
tags:
    - jeremykun MLIR learning
description: Personal MLIR learning notes 7.  
summary: Personal MLIR learning notes 7.  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# Purposes of a Verifier

Verifiers 确保具体的 MLIR 程序中的类型和操作格式正确。验证器会在每次优化 pass 之前和之后运行，帮助确保单个 pass, folders, rewrite patterns 等都能生成正确的 IR. 这使得每个操作的约束条件（invariants）能够得到强制执行，同时简化了传递的实现，因为它们可以依赖这些约束条件，从而避免检查边界情况。多数情况下验证代码是用 Traits 来实现的。

# Trait-based Verifiers

上一章我们加入了 `SameOperandsAndResultElementType` 从而让 `poly.add` 的输入可以既是 poly 或者张量类型的 poly. 从技术上讲，这向 IR 添加了一个验证器，但是为了更清楚地演示这一点，这一章将限制该行为，我们将 Trait 改成 `SameOperandsAndResultType` 以断言输入和输出类型必须全部一致。

这样会自动生成一些新功能。首先，验证引擎会使用 `verifyTrait` 来检查类型是否一致。在这里，`verifyInvariants` 是 `Operation` 基类中的一个方法，当某些 Traits 注入验证逻辑时，生成的代码会覆盖这个方法，用于检查操作类型上的类型约束。(如果是自定义验证器，则会使用名为 `verify` 的方法，以与 `verifyInvariants` 区分开来) 由于 `SameOperandsAndResultType` 是一个通用检查，因此它不会影响生成的代码。

下面展示了 AddOp 的 inferReturnTypes 方法

```c {linenos=true}++
::llvm::LogicalResult AddOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
    inferredReturnTypes.resize(1);  // Represent AddOp's output as a single type.
    ::mlir::Builder odsBuilder(context);
    if (operands.size() <= 0)  // Check that there is at least one operand.
        return ::mlir::failure();
    ::mlir::Type odsInferredType0 = operands[0].getType();
    inferredReturnTypes[0] = odsInferredType0;  // Set the output type to the first operand's type.
    return ::mlir::success();
}
```

有了类型推导钩子，我们可以简化操作的汇编格式，类型只需要指定一次，而不是三次 (`(type, type) -> type`). 同时也需要更新所有测试的 mlir 以启用这个新的 assemblyFormat.

```tablegen {linenos=true}
let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($output))"; 
```

我们可以从 AddOp 的 build 方法中看到现在不需要指定返回值，而是通过 `inferReturnTypes` 来推导。

```c {linenos=true}++
void AddOp::build(::mlir::OpBuilder& odsBuilder,
                  ::mlir::OperationState& odsState, ::mlir::Value lhs,
                  ::mlir::Value rhs) {
    odsState.addOperands(lhs);
    odsState.addOperands(rhs);

    ::llvm::SmallVector<::mlir::Type, 2> inferredReturnTypes;
    if (::mlir::succeeded(AddOp::inferReturnTypes(
            odsBuilder.getContext(), odsState.location, odsState.operands,
            odsState.attributes.getDictionary(odsState.getContext()),
            odsState.getRawProperties(), odsState.regions,
            inferredReturnTypes)))
        odsState.addTypes(inferredReturnTypes);
    else
        ::mlir::detail::reportFatalInferReturnTypesError(odsState);
}
```

`EvalOp` 无法使用 `SameOperandsAndResultType`，因为它的操作数需要不同的类型。然而，我们可以使用 `AllTypesMatch`，它会生成类似的代码，但将验证限制在某些特定类型的子集上。

```td {linenos=true}
def Poly_EvalOp : Op<Poly_Dialect, "eval", [AllTypesMatch<["point", "output"]>]> {
  let summary = "Evaluates a Polynomial at a given input value.";
  let arguments = (ins Polynomial:$input, AnyInteger:$point);
  let results = (outs AnyInteger:$output);
}
```

可以看到相似的 `inferReturnTypes` 方法，由于 EvalOp 是返回多项式在某个整数点上的值，因此推断的返回值类型需要与第二个操作数类型一致。

```c {linenos=true}++
::llvm::LogicalResult EvalOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::std::optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes,
    ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
    inferredReturnTypes.resize(1);
    ::mlir::Builder odsBuilder(context);
    if (operands.size() <= 1)
        return ::mlir::failure();
    ::mlir::Type odsInferredType0 = operands[1].getType();
    inferredReturnTypes[0] = odsInferredType0;
    return ::mlir::success();
}
```

# A Custom Verifier

如果需要添加自定义的 verifier 我们需要在 def 的时候添加 `let hasVerifier = 1`. 我们会发现生成的类里面定义了 verify 方法。

```c {linenos=true}++
class EvalOp ... {
  ...
  ::mlir::LogicalResult verify();
};
```

因此我们需要在 PolyOps.cpp 中实现它。

```c {linenos=true}++
// lib/Dialect/Poly/PolyOps.cpp
LogicalResult EvalOp::verify() {
    return getPoint().getType().isSignlessInteger(32)
               ? success()
               : emitError("argument point must be a 32-bit integer");
}
```

# A Trait-based Custom Verifier

在 MLIR 中，每个 Trait 都有一个可选的 `verifyTrait` 钩子，这个钩子会在通过 `hasVerifier` 创建的自定义验证器之前执行。我们可以利用这个钩子定义通用的验证器，使其适用于多个操作。比如，我们可以通过扩展上一节的内容，创建一个通用的验证器，用于断言所有整数类型的操作数必须是 32 位。

因此我们先需要 def 一个新的 Trait，然后将它加入到 `EvalOp` 中.

```tablegen {linenos=true}
  let cppNamespace = "::mlir::tutorial::poly";
}
```

我们可以看到生成的代码里有一个新类需要我们实现

```c {linenos=true}++
class EvalOp : public ::mlir::Op<
    EvalOp, ::mlir::OpTrait::ZeroRegions,
    //...,
    ::mlir::tutorial::poly::Has32BitArguments,
    //...
> {
  // ...
};
```

我们需要新建一个 PolyTraits.h 文件并且让 PolyOps.h 包含它

```c {linenos=true}++
// 
// /include/mlir-learning/Dialect/Poly/PolyOps.h

#ifndef LIB_DIALECT_POLY_POLYTRAITS_H_
#define LIB_DIALECT_POLY_POLYTRAITS_H_

#include "mlir/include/mlir/IR/OpDefinition.h"

namespace mlir::tutorial::poly {

template <typename ConcreteType>
class Has32BitArguments : public OpTrait::TraitBase<ConcreteType, Has32BitArguments> {
 public:
  static LogicalResult verifyTrait(Operation *op) {
    for (auto type : op->getOperandTypes()) {
      // OK to skip non-integer operand types
      if (!type.isIntOrIndex()) continue;

      if (!type.isInteger(32)) {
        return op->emitOpError()
               << "requires each numeric operand to be a 32-bit integer";
      }
    }

    return success();
  }
};

}

#endif  // LIB_DIALECT_POLY_POLYTRAITS_H_
```

这样做的优点是具有更强的通用性，但缺点是需要进行繁琐的类型转换来支持特定的操作及其命名参数。例如，这里我们无法直接调用 `getPoint`，除非对操作进行动态转换为 `EvalOp`.