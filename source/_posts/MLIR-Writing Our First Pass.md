---
title: MLIR-Writing Our First Pass
date: 2024/10/30 11:42:34
categories: MLIR
tags: jeremykun MLIR learning
excerpt: Personal MLIR learning notes 2.  
mathjax: true
katex: true
---
# Tutorial-opt and Project Organization

编译器可能将 mlir-opt 作为子例程在前端 (c++ -> 某些MLIR方言) 和后端 (MLIR 的 LLVM 方言 -> LLVM -> 机器码) 之间运行。

在一个独立于 MLIR 代码库的项目中可以自由定义自己的方言和 Pass，而不必直接修改 MLIR 主项目。MLIR 提供了注册钩子，使得可以轻松构建一个包含自定义方言和 Pass 的工具 (我将它命名为 tutorial-opt).

典型的 MLIR 代码库似乎将代码分成两个具有大致相同层次结构的目录：一个 `include/` 目录用于存放头文件和tablegen 文件，一个 `lib/` 目录用于存放实现代码。在这两个目录中，将有一个 `Transform/` 子目录，用于存储在方言中转换代码的 pass，`Conversion/` 子目录用于在方言之间转换的 pass ，`Analysis/` 子目录用于分析 pass，等等。这些目录中的每一个都可能有它们所操作的特定方言的子目录。

尽管 MLIR 提供了许多定义循环和控制流的机制，最高级的是 affine dialect. 它被设计用来进行多面体循环分析 (polyhedral loop analysis).

{% fold info@Polyhedral Loop Analysis %}
多面体循环分析的核心思想是将程序中的循环和数组访问抽象为数学形式，使得可以应用几何变换来优化代码。这种数学形式通常表示为 **整数线性不等式的集合** ，这些不等式定义了循环迭代空间和数组访问的范围。

1. **迭代空间（Iteration Space）** ：程序中的循环嵌套可以被表示为一个多维的迭代空间。例如，对于一个双层嵌套循环：

```C
for (i = 0; i < N; i++) {
    for (j = 0; j < M; j++) {
        A[i][j] = A[i][j] + 1;
    }
}
```

   这里的迭代空间是二维的，由 `(i, j)` 构成。

2. **访问关系（Access Relations）** ：每个数组的访问模式（例如 `A[i][j]`）也可以被表示为几何关系。这种关系定义了哪些迭代变量访问哪些数组元素。
3. **多面体表示（Polyhedral Representation）** ：在多面体循环分析中，循环的迭代空间和数组访问模式可以用整数线性不等式来表示，从而形成一个多面体。例如，`0<=i<N` 和 `0<=j<M` 是两个简单的线性不等式，它们表示循环的边界。
   {% endfold %}

一个简单的对数组求和的函数如下

```mlir
func.func @sum_buffer(%buffer: memref<4xi32>) -> i32 {
    %sum_0 = arigh.constant 0 : i32
    %sum = affine.for %i = 0 to 4 iter_args(%sum_iter = %sum_0) -> (i32) {
        %t = affine.load %buffer[%i] : memref<4xi32>
        %sum_next = arith.addi %sum_iter, %t : i32
        affine.yield %sum_next : i32
    }
    return %sum: i32
}
```

`iter_args` 是一种自定义的语法，它定义了在循环体中操作的累积变量 (符合 SSA 形式) 以及初始值。

{% fold info@Single Static Assignment (SSA) Form %}
在 MLIR 中，SSA 形式是表示和管理程序变量的一种标准形式。在 SSA 形式 中，每个变量在其整个生命周期中只能被赋值一次。这意味着每个变量（或寄存器）在程序的每个基本块内，只会被唯一地定义一次，而后续对该变量的引用只能使用之前的那个定义。

MLIR 和 LLVM IR 中都有一个特别的 SSA 形式的操作，叫做 `phi` 函数。当控制流有分支时，`phi` 函数用于从不同的控制流路径中选择适当的值。

SSA 形式的优点

* 简化分析：由于每个值只赋值一次，简化了数据流分析和依赖性分析。编译器可以更容易地跟踪变量的定义和使用关系。
* 便于优化：编译器可以更方便地进行代码优化，如常量折叠、死代码消除等。
* 支持并行化：通过准确的依赖关系分析，编译器可以识别哪些操作可以并行执行。
  {% endfold %}

# Affine Full Unroll Pass
MLIR 提供了一个方法来完全展开循环，因此我们的 pass 将是对这个函数调用的一个包装，以便在编写更有意义的 pass 之前展示基础设施的其余部分。我们直接从 C++ API 实现，而不是使用如模式重写引擎、方言转换框架或 tablegen 等更特殊的功能。

```C++
// include/mlir-learning/Transform/Affine/AffineFullUnroll.h
class  AffineFullUnrollPass 
    : public PassWrapper<AffineFullUnrollPass, OperationPass<mlir::FuncOp>> {

private:
    void runOnOperation() override;

    StringRef getArgument() const final {return "affine-full-unroll";}

    StringRef getDescription() const final {
        return "Perform full unrolling of all affine.for loops";
    }
};

// lib/Transform/Affine/AffineFullUnroll.cpp
using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

void AffineFullUnrollPass::runOnOperation() {

    getOperation().walk(
        [&](AffineForOp op) {  
            if (failed(loopUnrollFull(op))) {
                op.emitError("unrolling failed");
                signalPassFailure();
            }
        });
}
```

该类的定义使用了奇异递归模板模式 (Curiously Recurring Template Pattern, CRTP). PassWrapper 是 MLIR 框架中的一个模板类，它将 Pass 封装成一个可以在 MLIR 编译过程中运行的操作。 `OperationPass<mlir::FuncOp>` 将此 pass 锚定到 `FuncOp`.

- `runOnOperation` 中调用了 `getOperation` 方法，它是 MLIR 中 `Pass` 类提供的一个方法，返回当前操 `Operation`.  `walk` 方法是 MLIR 提供的一个遍历方法，用来遍历操作树中的每个节点。它会递归地遍历操作树中的所有子操作，并对每个操作应用传入的回调函数 (lambda func). 当运行这个 Pass 时，它会在每一个 `AffineForOp` 类型的操作上执行 `runOnOperation` 函数。
- `getArgument` 方法返回 Pass 的命令行参数。这个返回值 `affine-full-unroll` 表示这个 Pass 的名称，可以在运行时通过命令行参数指定是否启用该 Pass.
- `getDescription` 方法会在调用像 `mlir-opt` 这样的工具时若有 `--help` 参数则返回 Pass 的描述信息。

{% fold info@Callback Function %}
回调函数 (Callback Function) 是一种通过将函数作为参数传递给另一个函数，来实现某些特定操作的机制。回调函数通常在某个事件发生或某个特定条件满足时被调用。简而言之，回调函数就是**被调用的函数**，它会在特定的时机被执行。

在这个例子中，`invokeCallback` 函数接收到 `printMessage` 函数的地址，并在 `main` 函数中调用它。

```c++
#include <iostream>

// 回调函数的定义
void printMessage() {
    std::cout << "Hello, World!" << std::endl;
}

// 接受回调函数作为参数的函数
void invokeCallback(void (*callback)()) {
    // 调用回调函数
    callback();
}

int main() {
    // 将回调函数传递给另一个函数
    invokeCallback(printMessage);
    return 0;
}
```

在现代 C++ 中，回调函数通常通过 lambda 表达式传递。下面的例子中 `invokeCallback` 函数接受一个 `std::function<void()>` 类型的回调函数参数。在 `main` 函数中，传入了一个 Lambda 表达式作为回调函数。

```c++
#include <iostream>

void invokeCallback(std::function<void()> callback) {
    callback();
}

int main() {
    // 使用 Lambda 表达式作为回调函数
    invokeCallback([](){
        std::cout << "Hello from Lambda!" << std::endl;
    });
    return 0;
}
```

{% endfold %}

# Registering the Pass

接下来我们需要在 tutorial.cpp 中注册这个 Pass。

```c++
#include "mlir-learning/Transform/Affine/AffineFullUnroll.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char** argv) {
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();

    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
```

- `mlir::registerAllDialects(registry);` 会调用 MLIR 库的函数，将所有可用的方言注册到 `registry` 中。方言是 MLIR 中用来定义各种中间表示的抽象，可以理解为不同类型的 IR.
- `mlir::PassRegistration<mlir::tutorial::AffineFullUnrollPass>();` 将自定义的 `AffineFullUnrollPass` 注册到 MLIR 的 Pass 系统中。Pass 是对 MLIR 中间表示进行操作的基本单元，通常用于优化或转
  换。
- `MlirOptMain` 是 MLIR 提供的一个函数，处理命令行参数，并执行相应的 Pass.
  - argc 和 argv：来自命令行的参数。
  - "Tutorial Pass Driver"：这是一个程序描述字符串，通常是给用户的信息。
  - registry：之前创建的 DialectRegistry，它包含了所有已注册的方言。
- `mlir::asMainReturnCode(...)` 将 `MlirOptMain` 的返回值转换为标准的退出代码 (0 表示成功，非零值表示失败).

# Test the Pass

我们写一个 .mlir 来测试我们的 Pass，这是一个对数组进行累加的函数。FileCheck 检查经过 Pass 后函数中不会存在 `affine.for` 指令。

```mlir
// RUN: /leaning/build/chapter2/tools/02-tutorial-opt %s --affine-full-unroll > %t
// RUN: FileCheck %s < %t

func.func @test_single_nested_loop(%buffer: memref<4xi32>) -> (i32) {
  %sum_0 = arith.constant 0 : i32
  // CHECK-NOT: affine.for
  %sum = affine.for %i = 0 to 4 iter_args(%sum_iter = %sum_0) -> i32 {
    %t = affine.load %buffer[%i] : memref<4xi32>
    %sum_next = arith.addi %sum_iter, %t : i32
    affine.yield %sum_next : i32
  }
  return %sum : i32
}
```

经过优化后的函数如下

```mlir
#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module {
  func.func @test_single_nested_loop(%arg0: memref<4xi32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.load %arg0[%c0] : memref<4xi32>
    %1 = arith.addi %c0_i32, %0 : i32
    %2 = affine.apply #map(%c0)
    %3 = affine.load %arg0[%2] : memref<4xi32>
    %4 = arith.addi %1, %3 : i32
    %5 = affine.apply #map1(%c0)
    %6 = affine.load %arg0[%5] : memref<4xi32>
    %7 = arith.addi %4, %6 : i32
    %8 = affine.apply #map2(%c0)
    %9 = affine.load %arg0[%8] : memref<4xi32>
    %10 = arith.addi %7, %9 : i32
    return %10 : i32
  }
}
```

# A Rewrite Pattern Version

当想要对一个给定的 IR 子结构重复应用相同的变换子集，直到该子结构被完全去除时，需要写一个重写模式引擎。重写模式是 `OpRewritePattern` 的子类，它有一个名为 `matchAndRewrite` 的方法来执行转换。

```c++
// chapter2/lib/Transform/Affine/AffineFullUnroll.cpp
struct AffineFullUnrollPattern : public mlir::OpRewritePattern<AffineForOp>
{
    AffineFullUnrollPattern(mlir::MLIRContext* context)
        : mlir::OpRewritePattern<AffineForOp>(context, 1) {
    }

    // 一般在 OpRewritePattern 中，IR 的更改要通过 PatternRewriter
    // PatternRewriter 处理 OpRewritePattern中发生的突变的原子性
    LogicalResult matchAndRewrite(AffineForOp op,
                                   PatternRewriter& rewriter) const override{
        return loopUnrollFull(op);
    }
};

```

- `AffineFullUnrollPattern` 继承自 `OpRewritePattern<AffineForOp>`，`OpRewritePattern` 是 MLIR 中用于对特定操作类型 (在这里是 `AffineForOp`) 进行模式匹配和重写的基类。模板参数 `AffineForOp` 表示我们要为 `AffineForOp` 这个操作创建一个模式。
- 构造函数初始化了基类 `OpRewritePattern<AffineForOp>`，并传递了两个参数
  - `context`：`MLIRContext` 是 MLIR 的上下文，保存着所有的操作、方言和类型等信息。在这里，`context` 用来初始化模式对象。
  - `benefit` 是一个表示模式匹配优先级的整数值，优先级越高的模式越先应用。
- `matchAndRewrite` 是在 MLIR 中进行模式重写的核心方法。它的目的是：检查某个操作是否符合当前模式的要求。如果操作匹配模式，则执行重写操作，通常会用新的 IR 替换原来的 IR。
  - `AffineForOp op` 表示要进行模式匹配的 `AffineForOp` 操作。
  - `PatternRewriter &rewriter` 是一个用于生成新的 MLIR 操作的工具，它可以修改 IR.

我们同样要像上一节一样在头文件中声明一个 `AffineFullUnrollPassAsPatternRewrite` 类，然后实现其 `runOnOperation` 方法。

```c++
// chapter2/lib/Transform/Affine/AffineFullUnroll.cpp
void AffineFullUnrollPassAsPatternRewrite::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<AffineFullUnrollPattern>(&getContext());
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
```

- `RewritePatternSet` 是 MLIR 中一个容器，用于存储多个 `Rewrite Pattern`. 每个模式都是针对某种特定操作进行的优化规则。`RewritePatternSet` 会把所有这些规则聚合在一起，方便在后续的步骤中批量应用。
- 然后通过 `patterns.add<AffineFullUnrollPattern>`，将一个 Rewrite Pattern (这里是上面定义的 AffineFullUnrollPattern) 添加到 patterns 集合中。
- `applyPatternsAndFoldGreedily`是 MLIR 提供的一个函数，用于将定义的模式应用到给定的操作 (getOperation()) 上。这个函数使用贪心策略，在一次遍历中尽可能多地应用模式，直到无法再应用为止。

{% fold info@std::move() %}

`std::move` 是 C++11 引入的一个标准库函数，它的主要作用是将一个对象转换为右值引用，以便启用**移动语义** (Move Semantics). 简单来说，`std::move` 本身并不实际移动对象，而是为对象提供一个指示，告诉编译器该对象可以被**移动**而不是**复制**。

在 C++ 中，有两种主要的值类别:

* **左值 (Lvalue)** ：表示可以取地址的对象，可以理解为拥有持久生命周期的对象。它通常是变量、数组元素、对象成员等。
* **右值 (Rvalue)** ：表示临时对象、非持久生命周期的对象，通常是返回值、字面常量等。

```c++
#include <iostream>
#include <vector>
#include <utility>  // std::move

class MyClass {
public:
    MyClass() {
        std::cout << "Constructor\n";
    }
    MyClass(const MyClass& other) {
        std::cout << "Copy Constructor\n";
    }
    MyClass(MyClass&& other) noexcept {
        std::cout << "Move Constructor\n";
    }
    MyClass& operator=(const MyClass& other) {
        std::cout << "Copy Assignment\n";
        return *this;
    }
    MyClass& operator=(MyClass&& other) noexcept {
        std::cout << "Move Assignment\n";
        return *this;
    }
};

int main() {
    MyClass obj1;  // Constructor
    MyClass obj2 = std::move(obj1);  // Move Constructor

    MyClass obj3;
    obj3 = std::move(obj2);  // Move Assignment
}

```

{% endfold %}

# A proper greedy RewritePattern

接下来写一个用重写模式定义的 `MulToAddPass`，它会将 `y=C*x` 形式的乘法转换为 `y=C/2*x+C/2*x` 形式的加法当 C 是偶数。否则转换成 `y=1+(C-1)/2*x+(C-1)/2*x` 形式的加法。

## PowerOfTwoExpand

* 获取了 `rhs` 的定义操作（`rhs.getDefiningOp<arith::ConstantIntOp>()`），以确保右操作数是一个常数。
* 如果右操作数的值是 2 的幂，即 `(value & (value - 1)) == 0`，则进行优化。
  * 将 `value` 除以 2 然后生成新的常数 `newConstant`。
  * 计算新的乘法 `lhs * newConstant`，并将其加倍（通过 `AddIOp` 来实现 `lhs * value`）。
  * 最终用新的加法替代原来的乘法。

```c++
struct PowerOfTwoExpand : public OpRewritePattern<MulIOp>
{
    PowerOfTwoExpand(MLIRContext* context)
        : OpRewritePattern<MulIOp>(context, 2) {
    }

    LogicalResult matchAndRewrite(MulIOp op,
                                  PatternRewriter& rewriter) const override {
        // Value represents an instance of an SSA value in the MLIR system
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();

        if (!rhsDefiningOp) {
            return failure();
        }

        int64_t value = rhsDefiningOp.value();
        bool is_power_of_two = (value & (value - 1)) == 0;

        if (!is_power_of_two) {
            return failure();
        }

        auto newConstant = rewriter.create<ConstantOp>(
            rhsDefiningOp->getLoc(),
            rewriter.getIntegerAttr(rhs.getType(), value / 2));
        auto newMul = rewriter.create<MulIOp>(op->getLoc(), lhs, newConstant);
        auto newAdd = rewriter.create<AddIOp>(op->getLoc(), newMul, newMul);

        rewriter.replaceOp(op, newAdd);
        rewriter.eraseOp(rhsDefiningOp);
        return success();
    }
};
```

## **PeelFromMul**

这个 Pass 的目标是将一个常数乘法转化为加法形式，适用于常数值 `rhs` 不为 2 的幂时。

* 将 `rhs` 减去 1，然后生成一个新的常数 `newConstant`（即 `value - 1`）。
* 用 `lhs * newConstant` 进行计算，并将结果加上 `lhs`（即 `lhs * value` 转化为 `(lhs * (value - 1)) + lhs`）。
* 最终用新的加法替代原来的乘法。

```c++
struct PeelFromMul : public OpRewritePattern<MulIOp>
{
    PeelFromMul(MLIRContext* context) : OpRewritePattern<MulIOp>(context, 1) {
    }

    LogicalResult matchAndRewrtite(MulIOp op, PatternRewriter& rewriter) const {
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        auto rhsDefiningOp = rhs.getDefiningOp<arith::ConstantIntOp>();
        if (!rhsDefiningOp) {
            return failure();
        }

        int64_t value = rhsDefiningOp.value();
        // Beacause PowerOfTwoExpand has higher benefit,
        // value must not be power of 2
        auto newConstant = rewriter.create<ConstantOp>(
            rhsDefiningOp->getLoc(),
            rewriter.getIntegerAttr(rhs.getType(), value - 1));
        auto newMul = rewriter.create<MulIOp>(op.getLoc(), lhs, newConstant);
        auto newAdd = rewriter.create<AddIOp>(op.getLoc(), newMul, lhs);

        rewriter.replaceOp(op, newAdd);
        rewriter.eraseOp(rhsDefiningOp);

        return success();
    }
};
```

## Add the Pass

之后我们同样在 `runOnOperation` 方法中注册 `PowerOfTwoExpand` 和 `PeelFromMul` 两个模式。

```c++
void MulToAddPass::runOnOperation() {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<PowerOfTwoExpand>(&getContext());
    patterns.add<PeelFromMul>(&getContext());
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}
```

## Test the Pass

我们同样创建一个 .mlir 文件来测试我们的 Pass. 我们希望 Pass 能够将递归地将乘法转化为加法形式，

```mlir
// RUN: /leaning/build/chapter2/tools/02-tutorial-opt %s --mul-to-add > %t
// RUN: FileCheck %s < %t

func.func @just_power_of_two(%arg0: i32) -> i32 {
    %0 = arith.constant 8: i32
    %1 = arith.muli %arg0, %0: i32
    func.return %1: i32
}

// CHECK-LABEL: func.func @just_power_of_two(
// CHECK-SAME:    %[[ARG:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK:   %[[SUM_0:.*]] = arith.addi %[[ARG]], %[[ARG]]
// CHECK:   %[[SUM_1:.*]] = arith.addi %[[SUM_0]], %[[SUM_0]]
// CHECK:   %[[SUM_2:.*]] = arith.addi %[[SUM_1]], %[[SUM_1]]
// CHECK:   return %[[SUM_2]] : i32
// CHECK: }

func.func @power_of_two_plus_one(%arg: i32) -> i32 {
  %0 = arith.constant 9 : i32
  %1 = arith.muli %arg, %0 : i32
  func.return %1 : i32
}

// CHECK-LABEL: func.func @power_of_two_plus_one(
// CHECK-SAME:    %[[ARG:.*]]: i32
// CHECK-SAME:  ) -> i32 {
// CHECK:   %[[SUM_0:.*]] = arith.addi %[[ARG]], %[[ARG]]
// CHECK:   %[[SUM_1:.*]] = arith.addi %[[SUM_0]], %[[SUM_0]]
// CHECK:   %[[SUM_2:.*]] = arith.addi %[[SUM_1]], %[[SUM_1]]
// CHECK:   %[[SUM_3:.*]] = arith.addi %[[SUM_2]], %[[ARG]]
// CHECK:   return %[[SUM_3]] : i32
// CHECK: }
```

经过优化后的代码如下：

```mlir
module {
  func.func @just_power_of_two(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    return %2 : i32
  }
  func.func @power_of_two_plus_one(%arg0: i32) -> i32 {
    %0 = arith.addi %arg0, %arg0 : i32
    %1 = arith.addi %0, %0 : i32
    %2 = arith.addi %1, %1 : i32
    %3 = arith.addi %2, %arg0 : i32
    return %3 : i32
  }
}
```

# Summary

使用模式重写引擎通常比编写遍历AST的代码更容易。不需要大型 case/switch 语句来处理 IR 中可能出现的所有内容。因此可以单独编写模式，并相信引擎会适当地组合它们。