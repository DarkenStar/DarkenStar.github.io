---
title: MLIR-Ch6 Folders and Constant Propagation
date: 2024-11-09T20:51:23+08:00
lastmod: 2024-11-09T20:51:23+08:00
draft: false
author: ["WITHER"]
keywords: 
    - MLIR
categories:
    - MLIR
tags:
    - jeremykun MLIR learning
description: Personal MLIR learning notes 6.  
summary: Personal MLIR learning notes 6.  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# Constant Propagation vs Canonicalization

`-sccp` Sparse Conditional Constant Propagation 是稀疏条件常数传播，它试图推断 op 何时具有常量输出，然后用常量值替换 op 。重复这个过程，它在程序中尽可能地“传播”这些常量。

例如对于如下的函数

```mlir
func.func @test_arith_sccp() -> i32 {
  %0 = arith.constant 7 : i32
  %1 = arith.constant 8 : i32
  %2 = arith.addi %0, %0 : i32
  %3 = arith.muli %0, %0 : i32
  %4 = arith.addi %2, %3 : i32
  return %2 : i32
}
```

`-sccp` 优化后的结果如下：

```mlir
func.func @test_arith_sccp() -> i32 {
  %c63_i32 = arith.constant 63 : i32
  %c49_i32 = arith.constant 49 : i32
  %c14_i32 = arith.constant 14 : i32
  %c8_i32 = arith.constant 8 : i32
  %c7_i32 = arith.constant 7 : i32
  return %c14_i32 : i32
}
```

需要注意的是：sccp 不会删除死代码；这里没有展示的是 sccp 的主要作用，它可以通过控制流 (if 或者 loop) 传播常量。

一个相关的概念是 canonicalization，`--canonicalize` pass 隐藏了 MLIR 中的许多繁重工作。它与 sccp 有一点重叠，因为它也计算常量并在 IR 中具体化它们。例如，在上面的 IR 上使用 `——canonicalize` pass 的结果如下

```mlir
func.func @test_arith_sccp() -> i32 {
  %c14_i32 = arith.constant 14 : i32
  return %c14_i32 : i32
}
```

中间的常量都被修剪掉了，剩下的只是返回值，没有任何 op. **规范化不能通过控制流传播常量**。

这两者都是通过折叠 (folding) 来支持的，折叠是采取一系列 op 并将它们合并在一起为更简单的 op 的过程。它还要求我们的方言具有某种常量 op ，该 op 与折叠的结果一起插入。

以这种方式支持折叠所需的大致步骤是：

1. 添加一个常量 op.
2. 添加实例化钩子。
3. 为每个 op 添加 folders.

# Making a Constant Operation

我们目前只支持通过 `from_tensor`  op 从 `arith.constant` 创建常量。

```mlir
%0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
%p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
```

一个常量 op 可以将上述两个操作简化成一个 op. `from_tensor` op 还可以用于根据数据 (而不仅仅是常数) 构建一个多项函数，因此即使在我们实现了 `poly.constant` 之后，它也应该保留。

```mlir
%0 = poly.constant dense<[2, 8, 20, 24, 18]> : !poly.poly<10>
```

[fold](https://mlir.llvm.org/docs/Canonicalization/#canonicalizing-with-the-fold-method) 可以用于向 sccp 等 pass 传递信号，表明 op 的结果是常量，或者它可以用于说 op 的结果等效于由不同 op 创建的预先存在的值。对于常量的情况，还需要一个 `materializeConstant` 钩子来告诉 MLIR 如何获取常量结果并将其转化为适当的 IR  op. 常量 op 的定义如下

```
def Poly_ConstantOp: Op<Poly_Dialect, "constant", [Pure, ConstantLike]> {
  let summary = "Define a constant polynomial via an attribute.";
  let arguments = (ins AnyIntElementsAttr:$coefficients);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$coefficients attr-dict `:` type($output)";
}
```

`ConstantLike` trait 标记的 op 被视为常量值生成 op ，可以在编译时进行常量折叠等优化。`arguments` 定义 op 的输入是一个具有 `AnyIntElementsAttr` 的值，使得 op 可以处理任意包含整数的集合，而不仅仅是特定位宽的整数。

# Adding Folders

我们为定义的 op 都加上 `let hasFolder = 1;` 它在 .hpp.inc 中添加了如下形式的声明。`FoldAdaptor` 定义为 `GenericAdaptor` 类型的别名，而 `GenericAdaptor` 包含了一个 `Attribute` 数组的引用，这个数组提供了对 op 属性的访问接口。

Attribute 类的核心作用是：

- 表示常量值：Attribute 用于表示操作的静态、不可变的常量值，例如整数、浮点数、字符串、类型信息等。这些值在编译期已知且不可更改。
- 支持编译器优化：通过提供常量值的表示，Attribute 支持 MLIR 的优化流程，如折叠 (folding) 、规范化 (canonicalization), 常量传播 (constant propagation) 等。
- 跨方言的通用接口：Attribute 是一个抽象接口，允许不同方言 (dialects) 定义自己的常量表示，同时通过统一的 API 进行操作。
- 轻量级和高效：Attribute 是一个值类型 (passed by value) ，内部仅存储指向底层存储的指针，依赖 MLIRContext 的唯一化机制 (uniquing) 确保内存效率和一致性。

```c++
using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;

::mlir::OpFoldResult fold(FoldAdaptor adaptor);
```

我们需要在 `PolyOps.cpp` 中实现这个函数。如果 `fold` 方法决定 op 应被替换为一个常量，则必须返回一个表示该常量的 `Attribute`，该属性可以作为 `poly.constant` 操作的输入。`FoldAdaptor` 是一个适配器，它具有与操作的 C++ 类实例相同的方法名称，但对于那些已经被折叠的参数，会用表示其折叠结果常量的 `Attribute` 实例替换。这在折叠加法和乘法操作时尤为重要，因为折叠的实现需要立即计算结果，并且需要访问实际的数值来完成计算。

对于 `poly.constant` 我们只需要返回输入的 attribute.

```c++
OpFoldResult ConstantOp::fold(ConstantOp::FoldAdaptor adaptor) {
  return adaptor.getCoefficients();
}
```

对于 from_tensor 我们需要有一个额外的强制转换作为断言，因为张量可能是用我们不希望作为输入的奇怪类型构造的。如果 `dyn_cast` 结果是 `nullptr`， MLIR 将其强制转换为失败的 `OpFoldResult`.

```c++
OpFoldResult FromTensorOp::fold(FromTensorOp::FoldAdaptor adaptor) {
  // Returns null if the cast failed, which corresponds to a failed fold.
  return dyn_cast<DenseIntElementsAttr>(adaptor.getInput());
}
```

BinOp 稍微复杂一些，因为这些 fold 方法中的每一个 op 都接受两个 `DenseIntElementsAttr` 作为输入，并期望我们为结果返回另一个 `DenseIntElementsAttr`.

对于 elementwise op 的 add/sub，我们可以使用现有的方法 `constFoldBinaryOp`，它通过一些模板元编程技巧，允许我们只指定元素 op 本身。

```c++
OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
  return constFoldBinaryOp<IntegerAttr, APInt>(
      adaptor.getOperands(), [&](APInt a, APInt b) { return a + b; });
}
```

对于 mul，我们手动的通过循环计算每个系数。`getResult()` 方法来自于 `OneTypedResult` 类模板及其内部类 `Impl` 是一个 MLIR Trait，它主要用于那些返回单一特定类型结果的 op 。

```c++
OpFoldResult MulOp::fold(MulOp::FoldAdaptor adaptor) {
    auto lhs = llvm::dyn_cast<DenseIntElementsAttr>(adaptor.getOperands()[0]);
    auto rhs = llvm::dyn_cast<DenseIntElementsAttr>(adaptor.getOperands()[1]);

    if (!lhs || !rhs) {
        return nullptr;
    }
    auto degree =
        mlir::cast<PolynomialType>(getResult().getType()).getDegreeBound();
    auto maxIndex = lhs.size() + rhs.size() - 1;
    SmallVector<llvm::APInt, 8> results;
    results.reserve(maxIndex);
    for (int64_t i = 0; i < maxIndex; i++) {
        results.push_back(APInt((*lhs.begin()).getBitWidth(), 0));
    }

    int64_t i = 0;
    for (auto lhsIt = lhs.value_begin<APInt>(); lhsIt != lhs.value_end<APInt>();
         lhsIt++) {
        int64_t j = 0;
        for (auto rhsIt = rhs.value_begin<APInt>();
             rhsIt != rhs.value_end<APInt>(); rhsIt++) {
            results[(i + j) % degree] += (*lhsIt) * (*rhsIt);
            j++;
        }
        i++;
    }
    return DenseIntElementsAttr::get(
        RankedTensorType::get(static_cast<int64_t>(results.size()),
                              mlir::IntegerType::get(getContext(), 32)),
        results);
}
```

# Adding a Constant Materializer

最后我们添加常量实例化函数，这是一个 dialect 级别的特性，我们在 `PolyDialect.td` 中添加 `let hasConstantMaterializer = 1;` 则会在 .hpp.inc 中添加如下形式的声明。

```c++
::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
                                         ::mlir::Attribute value,
                                         ::mlir::Type type,
                                         ::mlir::Location loc) override;
```

该函数作用是将给定 Attribute (上面每个折叠步骤的结果) 的单个常量 op 实例化为所需的结果 Type.

```c++
Operation *PolyDialect::materializeConstant(
    OpBuilder &builder, Attribute value, Type type, Location loc) {
  auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
  if (!coeffs)
    return nullptr;
  return builder.create<ConstantOp>(loc, type, coeffs);
}
```