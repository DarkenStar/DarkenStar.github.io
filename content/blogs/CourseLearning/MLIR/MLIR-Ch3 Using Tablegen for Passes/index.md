---
title: MLIR-Ch3 Using Tablegen for Passes
date: 2024-11-06T09:37:32+08:00
lastmod: 2024-11-06T09:37:32+08:00
draft: false
author: ["WITHER"]
keywords: 
    - MLIR
categories:
    - MLIR
tags:
    - jeremykun MLIR learning
description: Personal MLIR learning notes 3.  
summary: Personal MLIR learning notes 3.  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# What is Tablegen?

TableGen 是一个用于生成代码和描述结构的 DSL 和工具，最初由 LLVM 开发，后来被 MLIR 继承并扩展。它主要用于以声明式的方式定义和生成 MLIR 的各种组件，例如 Dialects、Operations、Attributes、Types 和 Passes，从而减少手动编写重复性 C++ 代码的工作量。

mlir-tablegen 并没有清楚地告诉你哪些函数没有实现，也没有解释必须编写的函数。确定缺失内容的主要方法是尝试用一些使用它的代码来构建生成的代码，然后筛选数百行 c++ 编译器错误，这反过来又需要了解生成代码中的各种模板操作。生成的代码将使用必须知道的符号，以便在正确的位置导入或提前声明，并且它要求管理生成的代码所在的名称空间。

# Tablegen Files and the mlir-tblgen Binary

TableGen 允许你定义变量，并且这些变量可以在多个定义中重复使用。

TableGen允许你在定义中嵌入C++代码片段。这些代码片段会被插入到TableGen生成的C++类中，并且这些C++代码片段可以访问前面定义的变量。这使得TableGen能够生成高度定制化的C++代码。如果需要为你的 pass 编写特殊的构造函数，就可以在 `PassBase.td`中用 TableGen 的语法写下相应的 C++ 代码。

下面给出了一个以 tablegen 语法重写上一章的 `AffineFullUnroll `pass 的例子

```cpp
// mlir-learning/Transform/Affine/Pass.td
include "mlir/Pass/PassBase.td"

def AffineFullUnroll : Pass<"affine-full-unroll"> {
  let summary = "Fully unroll all affine loops";
  let description = [{
    Fully unroll all affine loops. (could add more docs here like code examples)
  }];
  let dependentDialects = ["mlir::affine::AffineDialect"];
}
```

TableGen 拥有类似的类和继承的概念。`: Pass<...>` 表示一个类继承自 [PassBase.td](https://github.com/llvm/llvm-project/blob/1b74459df8a6d960f7387f0c8379047e42811f58/mlir/include/mlir/Pass/PassBase.td#L95) 文件中定义的 `Pass` 基类

`def` 用于定义一个具体实例，它会生成对应的 C++ 代码。 也就是说，使用 `def` 定义的类实例会被 TableGen 处理，最终转换成实际的代码，而仅仅使用 `class` 定义的类则不会直接生成代码，只作为模板或基类存在。

上面代码说明 TableGen 允许定义字符串变量和列表。 TableGen 还有一个重要功能：它允许定义变量并在多个定义中复用这些变量，还可以定义 C++ 代码片段，并将这些片段插入到生成的类中。 这些 C++ 代码片段可以使用前面定义的变量。例如 [PassBase.td](https://github.com/llvm/llvm-project/blob/1b74459df8a6d960f7387f0c8379047e42811f58/mlir/include/mlir/Pass/PassBase.td#L82) 类定义了一个代码构造函数变量。 如果需要为你的 Pass 类编写特殊的构造函数，可以在 PassBase.td 中编写相应的 C++ 代码。 这意味着 TableGen 不仅仅是简单的文本替换，它能够处理更复杂的代码生成逻辑，包括变量的跨定义使用和 C++ 代码的嵌入。

和上一章不同的是，这次我们也需要在 include 目录下写一个 CMakeLists.txt

```cmake
set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Transform-Affine-Passes-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-learning/Transform/Affine/Pass.td)
mlir_tablegen(mlir-learning/Transform/Affine/Pass.h.inc -gen-pass-decls -name=Affine)
mlir_tablegen(mlir-learning/Transform/Affine/Pass.md -gen-pass-doc)
add_public_tablegen_target(${TARGET_NAME})

set(
    ALL_TABLEGEN_TARGETS
    ${PROJECT_TARGET_PREFIX}-Transform-Affine-Passes-IncGen
    #${PROJECT_TARGET_PREFIX}-Transform-Arith-Passes-IncGen
)

# Add the generated files to a global property, so they can be used in the library
set_property(
    GLOBAL PROPERTY ${PROJECT_TARGET_PREFIX}-TABLEGEN-TARGETS
    ${ALL_TABLEGEN_TARGETS}
)
```

- `set(LLVM_TARGET_DEFINITIONS mlir-learning/Transform/Affine/Pass.td)`: 这行代码设置了 TableGen 的输入文件。
- `mlir_tablegen(mlir-learning/Transform/Affine/Pass.h.inc -gen-pass-decls -name=Affine)`: 这行调用了 `mlir_tablegen` 命令，它将 Pass.td 文件作为输入，生成一个名为 Pass.h.inc 的头文件，其中包含 Pass 的声明 (`-gen-pass-decls`)，并且命名空间为 Affine (`-name=Affine`).
- `mlir_tablegen(mlir-learning/Transform/Affine/Pass.md -gen-pass-doc)`: 这行同样调用 mlir_tablegen，生成一个名为 Pass.md 的文件，包含 Pass 的文档信息 (`-gen-pass-doc`).
- `add_public_tablegen_target(${TARGET_NAME})`: 这行代码将 TableGen 生成的目标添加到 CMake 项目中，使其成为一个公共目标，其他部分可以依赖它。
- `set(ALL_TABLEGEN_TARGETS ...)`: 这行代码定义了一个列表 `ALL_TABLEGEN_TARGETS`，包含所有 TableGen 生成的目标。
- `set_property(GLOBAL PROPERTY ...)`: 这行代码将所有 TableGen 生成的目标添加到全局属性 `${PROJECT_TARGET_PREFIX}-TABLEGEN-TARGETS}` 中。 使得构建系统能够跟踪和管理所有由 TableGen 生成的文件，确保它们被正确地包含在库或可执行文件中。

# .inc Files

我们同样创建和上一章相同的文件 (可以先不写)，需要注意的是由于 TableGen 生成的 .inc 文件位于构建目录下，在 lib 的 CMakeLists.txt 中我们需要在 `target_include_directories` 命令中加入 `${CMAKE_OUTPUT_DIR}/include`

下面我们来逐段看生成的 .inc 文件

1. 头部保护和条件编译

```c
//===----------------------------------------------------------------------===//
// AffineFullUnroll
//===----------------------------------------------------------------------===//
#ifdef GEN_PASS_DECL_AFFINEFULLUNROLL
std::unique_ptr<::mlir::Pass> createAffineFullUnroll();
#undef GEN_PASS_DECL_AFFINEFULLUNROLL
#endif // GEN_PASS_DECL_AFFINEFULLUNROLL
```

这部分代码使用了预处理宏 `GEN_PASS_DECL_AFFINEFULLUNROLL`。  如果这个宏被定义，则编译器会生成 `createAffineFullUnroll()` 函数的声明。

2. Pass 的实现

```c
#ifdef GEN_PASS_DEF_AFFINEFULLUNROLL
namespace impl {
  std::unique_ptr<::mlir::Pass> createAffineFullUnroll();
} // namespace impl
namespace impl {
  template <typename DerivedT>
  class AffineFullUnrollBase : public ::mlir::OperationPass<> {
    // ... (Pass 的方法定义) ...
  };
} // namespace impl
std::unique_ptr<::mlir::Pass> createAffineFullUnroll() {
  return impl::createAffineFullUnroll();
}
#undef GEN_PASS_DEF_AFFINEFULLUNROLL
#endif // GEN_PASS_DEF_AFFINEFULLUNROLL
```

这部分是 Pass 的主要实现。它使用了 `GEN_PASS_DEF_AFFINEFULLUNROLL` 宏来控制编译。如果该宏被定义，则编译器会编译 AffineFullUnrollBase 类以及 `createAffineFullUnroll` 函数。

- `AffineFullUnrollBase` 是一个基类模板，使用 CRTP (Curiously Recurring Template Pattern) 技术，允许派生类通过 DerivedT 获取自身的类型信息。 这是一种常见的 C++ 设计模式，用于实现静态多态。它定义了 Pass 的基本信息，例如名称、描述、命令行参数、依赖的 Dialect (这里是 `mlir::affine::AffineDialect`).
- `createAffineFullUnroll` 函数负责创建 `AffineFullUnroll` Pass 的实例。 它使用了 `impl` 命名空间，这是一种常见的 C++ 代码组织方式，用于隐藏实现细节。

3. Pass 注册

```c
#ifdef GEN_PASS_REGISTRATION

//===----------------------------------------------------------------------===//
// AffineFullUnroll Registration
//===----------------------------------------------------------------------===//

inline void registerAffineFullUnroll() {
::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
return createAffineFullUnroll();
});
}

// Old registration code, kept for temporary backwards compatibility.
inline void registerAffineFullUnrollPass() {
::mlir::registerPass([]() -> std::unique_ptr<::mlir::Pass> {
return createAffineFullUnroll();
});
}
//===----------------------------------------------------------------------===//
// Affine Registration
//===----------------------------------------------------------------------===//

inline void registerAffinePasses() {
registerAffineFullUnroll();
}
#undef GEN_PASS_REGISTRATION
#endif // GEN_PASS_REGISTRATION

```

# Complete .hpp & .cpp

TableGen根据 `.td`文件生成Pass的代码，生成的代码包含注册函数，这些注册函数最终会被调用，将Pass注册到MLIR系统中。 我们可以通过写一个 `Passes.h`文件集中管理所有Pass的注册，简化构建过程。

```c
// include/mlir-learning/Transform/Affine/Pass.h
#include "mlir-learning/Transform/Affine/AffineFullUnroll.h"

namespace mlir::tutorial {
#define GEN_PASS_REGISTRION
#include "mlir-learning/Transform/Affine/Pass.h.inc"
}
```

然后再对应的 AffineFullUnroll.hpp 中定义 `GEN_PASS_DECL_AFFINEFULLUNROLL` 宏，以实现创建 Pass 函数的声明。

```cpp
#pragma once 

#include "mlir/Pass/Pass.h"

namespace mlir::tutorial
{
#define GEN_PASS_DECL_AFFINEFULLUNROLL
#include "mlir-learning/Transform/Affine/Pass.h.inc"
}  // namespace mlir::tutorial

```

同样在 cpp 中需要定义 `GEN_PASS_DEF_AFFINEFULLUNROLL` 宏，然后写你对应的实现 (与上一章相同). 问题是仅仅查看生成的代码并不能直接看出还需要实现哪些函数，需要通过其他方法来确定。

* **编译并查看编译器错误信息:**  最直接的方法是尝试编译代码。编译器会指出哪些函数没有实现，从而告诉你需要实现哪些函数。
* **与基类进行比较:**  可以将生成的代码与基类（`OperationPass`和 `Pass`）进行比较。通过比较，可以发现唯一需要实现的函数是 `runOnOperation()`。  这需要你熟悉MLIR Pass的继承结构和各个函数的作用。
* **观察缺失的函数:**  如果之前已经从原始API手动实现过类似的Pass，可以观察生成的代码中哪些函数已经存在（例如 `getArgument`），哪些函数缺失（例如 `runOnOperation`）。 通过对比，可以确定还需要实现哪些函数。

具体的实现与上一章相同，这里我们要继承 .inc 文件中生成的类

```cpp
#include "mlir-learning/Transform/Affine/AffineFullUnroll.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Pass/Pass.h"

namespace mlir::tutorial
{

#define GEN_PASS_DEF_AFFINEFULLUNROLL
#include "mlir-learning/Transform/Affine/Pass.h.inc"

using mlir::affine::AffineForOp;
using mlir::affine::loopUnrollFull;

class AffineFullUnroll : public impl::AffineFullUnrollBase<AffineFullUnroll>
{
public:
    using AffineFullUnrollBase::AffineFullUnrollBase;

    void runOnOperation() override {
        getOperation()->walk([&](AffineForOp op) {
            if (failed(loopUnrollFull(op))) {
                op.emitError("unrolling failed");
                signalPassFailure();
            }
        });
    }
};
}  // namespace mlir::tutorial
```

最后在 `tutorial.cpp` 中使用 .inc 文件生成的 `registerAffinePasses`

```cpp
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir-learning/Transform/Affine/Pass.h"

int main(int argc, char** argv) {
    // Register all built-in MLIR dialects
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);

    mlir::tutorial::registerAffinePasses();
  
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
```