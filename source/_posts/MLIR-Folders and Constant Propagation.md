---
title: MLIR-Folders and Constant Propagation
date: 2024/11/09 20:51:55
categories: MLIR
tags: jeremykun MLIR learning
excerpt: Personal MLIR learning notes 6.  
mathjax: true
katex: true
---

# Constant Propagation vs Canonicalization

`-sccp` Sparse Conditional Constant Propagation 是稀疏条件常数传播，它试图推断操作何时具有恒定输出，然后用恒定值替换操作。重复这个过程，它在程序中尽可能地“传播”这些常量。可以把它想象成在编译期间急切地计算值，然后将它们作为常量放入编译后的程序中。

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

中间的常量都被修剪掉了，剩下的只是返回值，没有任何操作。**规范化不能通过控制流传播常量**。

这两者都是通过折叠 (folding) 来支持的，折叠是采取一系列操作并将它们合并在一起为更简单的操作的过程。它还要求我们的方言具有某种常量操作，该操作与折叠的结果一起插入。

以这种方式支持折叠所需的大致步骤是：
1. 添加一个常量操作。
2. 添加实例化钩子。
3. 为每个操作添加 folders.

# Making a Constant Operation

我们目前只支持通过 `from_tensor` 操作从个 `arith.constant` 值创建常量。

```mlir
%0 = arith.constant dense<[1, 2, 3]> : tensor<3xi32>
%p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
```

一个常量操作可以将他们简化成一个操作。`from_tensor` 操作还可以用于根据数据 (而不仅仅是常数) 构建一个多项函数，因此即使在我们实现了 `Poly.constant` 之后，它也值得存在。

```mlir
%0 = poly.constant dense<[2, 8, 20, 24, 18]> : !poly.poly<10>
```

fold 可以用于向 sccp 等 pass 传递信号，表明操作的结果是恒定的，或者它可以用于说操作的结果等效于由不同操作创建的预先存在的值。对于恒定的情况，还需要一个 `materializeConstant` 钩子来告诉 MLIR 如何获取恒定结果并将其转化为适当的 IR 操作。常量操作的定义如下
```
def Poly_ConstantOp: Op<Poly_Dialect, "constant", [Pure, ConstantLike]> {
  let summary = "Define a constant polynomial via an attribute.";
  let arguments = (ins AnyIntElementsAttr:$coefficients);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$coefficients attr-dict `:` type($output)";
}
```
`ConstantLike` trait 标记的操作被视为常量值生成操作，可以在编译时进行常量折叠等优化。`arguments` 定义操作的输入是一个具有 `AnyIntElementsAttr` 的值，使得操作可以处理任意包含整数的集合，而不仅仅是特定位宽的整数。

# Adding Folders

我们为定义的操作都加上 `let hasFolder = 1;` 它在 .hpp.inc 中添加了如下形式的声明。`FoldAdaptor` 定义为 `GenericAdaptor` 类型的别名，而 `GenericAdaptor` 包含了一个 `Attribute` 数组的引用，这个数组提供了对操作属性的访问接口。

```c++
using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
::mlir::OpFoldResult fold(FoldAdaptor adaptor);
```

我们需要在 `PolyOps.cpp` 中实现这个函数。如果 `fold` 函数决定操作可以用一个常量替代，那么它应该返回一个表示该常量的 `Attribute`，就是折叠后的常量值，之后可以用作 `poly.constant` 操作的输入。`FoldAdaptor` 具有与操作类 (如 `AddOp`, `MulOp` 等) 相同的方法和访问接口，使得 `fold` 函数能够以类似操作实例的方式来访问操作参数。如果操作的某些参数已经在之前被折叠成常量，这些参数会在 `FoldAdaptor` 中被替换为 `Attribute`，即折叠后的常量值。

对于 poly.constant 我们只需要返回输入的 attribute.

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

BinOp 操作稍微复杂一些，因为这些 fold 方法中的每一个操作都接受两个 `DenseIntElementsAttr` 作为输入，并期望我们为结果返回另一个 `DenseIntElementsAttr`.

对于 elementwise 操作的 add/sub，我们可以使用现有的方法 `constFoldBinaryOp`，它通过一些模板元编程技巧，允许我们只指定元素操作本身。

```c++
OpFoldResult AddOp::fold(AddOp::FoldAdaptor adaptor) {
  return constFoldBinaryOp<IntegerAttr, APInt>(
      adaptor.getOperands(), [&](APInt a, APInt b) { return a + b; });
}
```

对于 mul，我们手动的通过循环计算每个系数。`getResult()` 方法来自于 `OneTypedResult` 类模板及其内部类 `Impl` 是一个 MLIR Trait，它主要用于那些返回单一特定类型结果的操作。
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

该函数作用是将给定属性 (上面每个折叠步骤的结果) 的单个常量操作实例化为所需的结果类型。

```c++
Operation *PolyDialect::materializeConstant(
    OpBuilder &builder, Attribute value, Type type, Location loc) {
  auto coeffs = dyn_cast<DenseIntElementsAttr>(value);
  if (!coeffs)
    return nullptr;
  return builder.create<ConstantOp>(loc, type, coeffs);
}
```