---
title: MLIR — Using Traits
date: 2024/11/08 23:06:23
categories: MLIR
tags: jeremykun MLIR learning
excerpt: Personal MLIR learning notes 5.
mathjax: true
katex: true
---
# Traits and Loop Invariant Code Motion

为了提高代码重用性，MLIR 提供了 Traits 和 Interfaces Traits，用于增强操作 (Operation) 或类型的功能，提供结构化的约束和功能接口，方便在编译优化和生成过程中进行更强大和灵活的操作。

[Traits](https://mlir.llvm.org/docs/Traits/) 是一种机制，用于抽象许多属性、操作、类型等对象的实现细节和公共特性。通过使用 Traits，可以为对象指定其特殊属性和约束。例如，指定某个操作是否有副作用，或者它的输出类型是否与输入类型相同。Traits 将特定的行为或限制抽象出来，使这些行为可以复用在不同的对象上，而不需要在每个对象中重复实现相同的逻辑。

[Interfaces](https://mlir.llvm.org/docs/Interfaces/) 是一种通用的机制，用于与 IR 进行交互。它们的目标是使转换或分析可以基于这些接口进行，而无需了解具体的操作或 dialect 的内部实现。通过这种方法，编译器可以在实现转换和分析时不依赖于特定 dialect 或操作，从而更轻松地扩展编译器的功能。

Loop Invariant Code Motion 是一种编译器优化技术。它会检查循环体中的操作，如果发现某些操作在循环内部执行没有必要（即它们的结果在每次循环中保持不变），就会将这些操作移出循环体。这可以减少循环中的重复计算，提高效率。

要让某个自定义操作可以被这种 pass 识别并移出循环体，需要添加两个关键的 Traits 来表明该操作在循环外执行是安全的：

- `NoMemoryEffect`: 表示该操作不会产生任何与内存写入相关的副作用。
- `AlwaysSpeculatable`: 这是一个 Traits 列表 (通常包含两个 Traits)，表示一个操作是 speculatable，编译器可以在不影响程序逻辑的前提下，将其提前计算或移动到其他位置。

在 MLIR 中，Loop Invariant Code Motion (LICM) 会将具有 `NoMemoryEffect` 和 `AlwaysSpeculatable` 这两个 Traits 的操作移动到循环体外部，但前提是该操作的操作数在整个循环体中保持不变。这样可以避免循环内部的重复计算，从而优化代码执行效率。MLIR 提供了一个方便的组合 Trait，叫做 `Pure`，它包含了 `NoMemoryEffect` 和 `AlwaysSpeculatable` 这两个 Traits. 因此，直接添加 `Pure` Trait 到操作的定义中就能让编译器自动识别它为可移动到循环外部的操作。

`TypeOrContainer` 是一个用于处理操作输入和输出类型的机制，它可以匹配单个类型 (如 `f32` 或 `i32`) 以及容器类型(如 `vector<f32>` 或 `tensor<i32>`)，使得一个操作可以被设计为同时支持标量类型和集合类型。

```
include "mlir/Interfaces/SideEffectInterfaces.td"

def PolyOrContainer: TypeOrContainer<Polynomial, "poly-or-container">;

class Poly_BinOp<string mnemonic>: Op<Poly_Dialect, mnemonic, [Pure]> {
    let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
    let results = (outs PolyOrContainer:$output);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}
```

加入 `Pure` trait 后生成的 .hpp.inc 中关于操作的定义继承了新的内容

```cpp
class AddOp
    : public ::mlir::Op< AddOp,
        ::mlir::OpTrait::ZeroRegions,
        ::mlir::OpTrait::OneResult,
        ::mlir::OpTrait::OneTypedResult<::mlir::tutorial::poly::PolynomialType>::Impl,
        ::mlir::OpTrait::ZeroSuccessors,
        ::mlir::OpTrait::NOperands<2>::Impl,
        ::mlir::OpTrait::OpInvariants,
        ::mlir::ConditionallySpeculatable::Trait,            // <-- new
        ::mlir::OpTrait::AlwaysSpeculatableImplTrait,   // <-- new
        ::mlir::MemoryEffectOpInterface::Trait>          // <--- new
```

`NoMemoryEffect` interface 则在生成的 .cpp.inc 中添加了一个简单的函数

```cpp
void AddOp::getEffects(
    ::llvm::SmallVectorImpl<
        ::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>>&
        effects) {
}
```

我们可以写一个 .mlir 来测试 `%2` 的计算是否能优化到循环外：

```mlir
// RUN: /leaning/build/chapter5/tools/05-tutorial-opt %s --loop-invariant-code-motion > %t
// RUN: FileCheck %s < %t

module {
    // CHECK-LABEL: func.func @test_loop_invariant_code_motion
    func.func @test_loop_invariant_code_motion() -> !poly.poly<10> {
        %0 = arith.constant dense<[1,2,3]> : tensor<3xi32>
        %p0 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>

        %1 = arith.constant dense<[9,8,16]> : tensor<3xi32>
        %p1 = poly.from_tensor %0 : tensor<3xi32> -> !poly.poly<10>
        // CHECK: poly.mul

        // CHECK: affine.for
        %ret_val = affine.for %i = 0 to 100 iter_args(%sum_iter = %p0) -> !poly.poly<10> {
            // The polt.mul should be hoisted out of the loop.
            // CHECK-NOT: poly.mul
            %2 = poly.mul %p0, %p1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
            %sum_next = poly.add %sum_iter, %2 :  (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
            affine.yield %sum_next : !poly.poly<10>
        }

        return %ret_val: !poly.poly<10>
    }
}
```

# Passes Already Handled by Pure

给某个操作加上 `Pure` Trait 后，下列 Pass 就会自动识别并优化该操作：

* `--control-flow-sink`: 将只在条件语句的某一个分支中使用的操作移动到对应的分支中，以减少无效代码的执行。需要操作无内存副作用 (memory-effect free)，通常可以通过 `Pure` Trait 来满足。
* `--cse` (Constant Subexpression Elimination): 常量子表达式消除。当某些重复的计算结果已经存在时，消除不必要的重复计算，提高效率。需要操作没有内存副作用（memory-effect free），因此 `Pure` Trait 也可以满足这一要求。
* `--inline`: 将函数调用“内联”到调用位置，以减少函数调用的开销。在某些情况下，这可以减少调用栈的深度或优化代码执行的性能。
* `--mem2reg`: 将内存中的存储/加载操作转换为对实际值的直接使用，从而减少内存访问，提高运行效率。
* `--remove-dead-values`: 移除未使用的函数参数或返回值，以减少不必要的数据传递或内存占用。
* `--sroa`  (Scalar Replacement of Aggregates): 将聚合类型（例如数组或结构体）拆分为标量值，通常会对内存布局进行重排，以便更好地利用内存。
* `--symbol-dce`  (Symbol Dead Code Elimination): 消除不再使用的私有函数 (死代码)，减少不必要的代码量。

# Elementwise Mappings

有四种 traits 可以把标量运算扩展到张量运算或者反过来

- `Elemntwise`: 标记逐元素的操作，仅适用于向量或张量，不允许广播。
  * 如果任何结果是向量或张量，至少有一个操作数必须是向量或张量。
  * 如果任何操作数是向量或张量，至少有一个结果并且所有结果必须是向量或张量。
  * 所有操作数和结果的向量或张量类型必须具有相同的形状。形状可以是动态的，但对于不匹配的形状，行为是未定义的。
  * 该操作必须在操作数和结果上逐元素进行，即在单元素向量或张量上应用时，每个元素的结果应相同。
- `Scalarizable`: 标记可以系统化地进行标量化的逐元素操作。该操作将向量或张量的操作数和结果替换为相应的标量类型。即，该操作表示对向量或张量中每个元素的逐一操作。
  ```mlir
  %tensor_select = "arith.select"(%pred_tensor, %true_val, %false_val)
                  : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>)
                  -> tensor<?xf32>
  // Can be scalarized to
  %scalar_select = "arith.select"(%pred, %true_val_scalar, %false_val_scalar)
                  : (i1, f32, f32) -> f32

  ```
- `Vectorizable`: 提供了与 `Scalarizable` 相反的操作。所有的标量操作数和结果将被替换为相应的向量类型。即，该操作表示同时作用于多个元素。允许通过广播将标量提升为向量，以便在向量操作中保持一致的操作结构。
- `Tensorizable`: 提供了与 `Scalarizable` 相反的操作，允许在张量和标量之间进行推理。允许通过广播将标量提升为张量，以便在张量操作中保持一致的操作结构。
  ```mlir
  %scalar = "arith.addf"(%a, %b) : (f32, f32) -> f32
  // Can be tensorized to
  %tensor = "arith.addf"(%a, %b) : (tensor<?xf32>, tensor<?xf32>)
              -> tensor<?xf32>
  // Also supports broadcasting
  %scalar_pred = "arith.select"(%pred, %true_val, %false_val)
                  : (i1, tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  // Can be tensorized to
  %tensor_pred = "arith.select"(%pred, %true_val, %false_val)
                  : (tensor<?xi1>, tensor<?xf32>, tensor<?xf32>)
              -> tensor<?xf32>
  ```

`ElementwiseMappable` Trait 包含了以上所有的 Traits. 我们可以修改 `Poly_BinOp` 定义如下：

```
// PolyOps.td
def PolyOrContainer : TypeOrContainer<Polynomial, "poly-or-container">;

class Poly_BinOp<string mnemonic> : Op<Poly_Dialect, mnemonic, [Pure, ElementwiseMappable]> {
  let arguments = (ins PolyOrContainer:$lhs, PolyOrContainer:$rhs);
  let results = (outs PolyOrContainer:$output);
  ...
}
```

添加这个 Trait 后，生成的 .cpp.inc 文件定义了许多检查操作数类型的函数，下面是其中一个：

```cpp
static ::llvm::LogicalResult __mlir_ods_local_type_constraint_PolyOps1(
    ::mlir::Operation* op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
    if (!(((::llvm::isa<::mlir::tutorial::poly::PolynomialType>(type))) ||
            (((type.hasTrait<::mlir::ValueSemantics>())) &&
            ([](::mlir::Type elementType) {
                return (::llvm::isa<::mlir::tutorial::poly::PolynomialType>(
                    elementType));
            }(::llvm::cast<::mlir::ShapedType>(type).getElementType()))))) {
        return op->emitOpError(valueKind)
                << " #" << valueIndex
                << " must be poly-or-container, but got " << type;
    }
    return ::mlir::success();
}
```

该函数首先检查 `type` 是否为 `PolynomialType`；如果不是，则进一步检查它是否具有 `ValueSemantics` Trait，并且是一个 `ShapedType`（即容器类型，如 `vector` 或 `tensor`），其中包含的元素类型是 `PolynomialType`.