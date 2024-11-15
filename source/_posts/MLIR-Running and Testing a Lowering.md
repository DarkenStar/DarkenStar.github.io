---
title: MLIR-Running and Testing a Lowering
date: 2024/10/28 18:49:23
categories: MLIR
tags: jeremykun MLIR learning
excerpt: Personal MLIR learning notes 1.  
mathjax: true
katex: true
---
# Dialects and Lowerings

MLIR 中两个重要的概念是 dialects 和 lowerings. dialects 是是编译器代码中程序的文本或数据结构描述，lowerings 是用来将特定 dialect 的 IR 转换成另一种 dialect 的 IR 的过程。

在 MLIR 的工作流程

1. 定义高级或者高级 dialects. 它们都是由由一系列的类型，操作，元数据和语义组成。
2. 编写一组 lowering passes，将程序的不同部分从高级 dialect 逐渐转换为越来越低级的 dialect，直到变成机器码。在此过程中，将运行 optimizing passes 使代码更高效。

{% note info %}
高级 dialect 的存在使得可以轻松编写 optimizing passes. lowering passes 和 optimizing passes 之间没有特别的区别，它们在 MLIR中 都被称为 passes，并且是通用的 IR 重写模块。
{% endnote %}

# Two Example Programs

下面的一段代码用了 math dialect 中定义的 counting leading zeros (ctlz) 操作。计算输入的前导零的数量并直接返回。

```mlir
func.func @main(%arg0: i32) -> i32 {
    % 0 = math.ctlz %arg0 : i32
    func.return %0 : i32
}
```

变量名前缀为 `%` 表示它是一个 SSA 值。函数前缀为 `@`. 程序中每个变量都有对应的类型 (`i32`)，而 `func` 的类型是 `i32 -> i32`.

每个声明都围绕着 `math.ctlz` 等表达式锚定，该表达方式指定 dialect `math` 和 ctlz 操作。操作的其余语法由 dialect 定义的 parser 确定，因此许多操作将具有不同的语法。末尾的：i32 表示输出类型。

`func` 自己也是一个 dialect，`func.func` 被看作一个操作，括号和函数体是他 syntax 的一部分。在 MLIR 中，被括号包裹的一系列操作称作一个 region. 一个操作可以没有或者有很多 regions.

简单来说 operations 可能会被包括在 regions 中 (例如 for 循环的 body). 每个 region 包含一系列的 blocks (显式或者隐式). 一个 block 由一系列操作组成，并且只有一个入口和一个出口。

{% note info %}
在 MLIR 中，多个 dialects 经常共存在同一个程序中，因为它会逐渐 lowering 到某个最终的后端设备。
{% endnote %}

下面的代码包含了一个 ctlz 函数的实现并且在主函数中调用它。

```mlir
func.func @main(%arg0: i32) -> i32 {
    %0 = func.call@my_ctlz(%arg0) : (i32) -> i32
    func.return %0 : i32
}

func.func @my_ctlz(%arg0: i32) -> i32 {
    %c32_i32 = arith.constant 32 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg0, %c0_i32 : i32  // % 0 = (arg0 == 0)
    %1 = scf.if %0 -> (i32) {  // scructured control flow
        scf.yield %c32_i32 : i32
    } else {
        %c1 = arith.constant 1 : index
        %c1_i32 = arith.constant 1 : i32
        %c32 = arith.constant 32 : index
        %c0_i32_0 = arith.constant 0 : i32
        %c0_i32_0 = arith.contant 0 : i32
        %2:2 = scf.for %arg1 = %c1 to %c32 step %c1 iter_args(%arg2 = %arg0, %arg3 = %c0_i32_0) -> (i32, i32) {
            %3 = arigh.cmpi slt, %arg2, %c0_i32 : i32  // signed less than
            %4:2 = scf.if %3 -> (i32, i32) {  // means MSB of %arg2 is 1
                scf.yield %arg2, %arg3 : i32, i32
            } else {
                %5 = arith.addi %arg3, %c1_i32 : i32  // increment count
                %6 = arith.shli %arg2, %c1_i32 : i32  // shift left by 1
                scf.yield %6, %5 : i32, i32
            }
            scf.yield %4#0, %4#1 : i32, i32  // pass to iter_args
        }
        scf.yield %2#1 : i32
    }
    func.return %1 : i32
}
```

`scf` dialect 定义了 structured control flow 相关的操作: `scf.if` 是一个条件分支，`scf.for` 是一个循环。`scf.yield` 表示该控制流返回值。 `%2:2` 表示返回的变量是一个包含两个值的 tuple. `%4#0, %4#1` 分别表示返回变量的第一个和第二个值。`index` 类型专门用于表示索引、内存地址和循环计数，是与平台无关的类型，通常与平台指针大小一致。

`scf.for` 控制流中 `%arg1` 是循环变量，`%c1 to %c32` 表示循环将从 `%c1` 直迭代到 `%c32` (不包括). `step %c1` 表示循环变量每次迭代增加值。 `iter_args` 示迭代参数，在每次迭代中，`%arg2` 和 `%arg3` 会根据 `scf.yield` 的值更新，并在下一次迭代时继续传递。

# Lowerings and the math-to-funcs Pass

上一节提供了 2 个版本的 ctlz 程序。一般大部分机器支持计算前导 0 指令，这样就可以直接将 `match.ctlz` 转换成对应的指令。否则就要采用更低级的操作，正如第二个版本所做的一样。

mlir-opt 工具将会解析 .mlir 文件并进行优化，使用方式是 `mlir-opt path/to/xxx.mlir`. 我们对第一个版本直接使用该命令生成的优化后的代码如下。

```mlir
// (base) root@f4faf6ad6814:/# mlir-opt ./test/ctlz.mlir 
module {
  func.func @main(%arg0: i32) -> i32 {
    %0 = math.ctlz %arg0 : i32
    return %0 : i32
  }
}
```

我们也可以用 mlir 自带的 `convert-math-to-funcs` pass 进行 lowering. 生成的正好是我们第二个版本。CL 命令为 `mlir-opt --convert-math-to-funcs=convert-ctlz ./test/ctlz.mlir`

# Lit and FileCheck

Lit 全称为 LLVM Integrated Tester. 它通常用来运行 .mlir 文件的测试，来验证某些 Pass 的输出是否符合预期。Lit 测试通常会结合 FileCheck 工具来检查命令的输出。

FileCheck 是 LLVM 项目中的一个文本匹配工具。它通过特殊的注释标记 (CHECK, CHECK-NEXT, CHECK-LABEL .etc) 来指定预期的输出，来验证程序的行为是否符合预期。这些标记在 .mlir 文件中作为注释，用于表示某些期望的输出。

以第一个版本的 lit 测试文件举例。一个lit测试文件包含一些行，`RUN:` 作为注释的开头，后面的文本描述了要运行的shell 脚本，并用一些字符串指示 lit 进行替换。在本例中，`%s` 指当前文件路径。FILECHECK 接受传递给 stdin 的输入，扫描作为 CLI 参数传递的文件中的 CHECK 注释，它执行一些逻辑来确定断言是否通过。

```mlir
// RUN: mlir-opt %s --convert-math-to-funcs=convert-ctlz | FileCheck %s

func.func @main(%arg0: i32) -> i32 {
  // CHECK-NOT: math.ctlz
  // CHECK: call
  %0 = math.ctlz %arg0 : i32
  func.return %0 : i32
}
```

FileCheck 可以做更多的事情，比如使用正则表达式捕获变量名，然后在以后的 CHECK 断言中引用它们。

`lit.py` 通常位于 `/path/to/llvl-project/llvm/utils/lit` 目录下。我们需要在要测试的文件目录下配置一个 `lit.cfg`. 然后可以运行命令 `python /path/to/llvm-project/llvm/utils/lit/lit.py /test/path`

```cfg
# lit.cfg 文件的基本内容
import os
import lit.formats

# 配置名称
config.name = "MLIRTestSuite"

# 使用 Shell Test 作为测试格式
config.test_format = lit.formats.ShTest()

# 指定测试文件的扩展名
config.suffixes = ['.mlir']

# 环境变量设置
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.test_source_root, 'Output')
```

# Functinal Testing

在 MLIR 中，lowering 过程本身是 syntactic. CHECK 不能检查生成的代码是否有错误，或者保证生成正确的结果。我们可以用 Lit 进行 functinal testing. 我们可以编写一些 lit 测试来验证程序的行为是否符合预期。Lit 测试可以运行在不同的平台上，并且可以指定不同的参数。

解决这个问题的一种方法是继续通过 LLVM 将 MLIR 代码向下编译为机器码并运行它，断言有关输出的某些内容。虽然 `RUN` 可以运行任何东西，但需要引入更多的依赖项。实现这一目标的一个稍微轻量级的方法是使用 `mlir-cpu-runner`，它是一些最低级别的 MLIR dialect (特别是 llvm dialect) 的解释器。

{% note info %}
在 MLIR 到 LLVM IR 的编译过程中，最终会将 MLIR 中的 llvm dialect 直接映射或转换为标准的 LLVM IR 之后，LLVM IR 就可以进一步被转换为目标机器码，或者通过 JIT 编译直接执行。
{% endnote %}

下面一段程序用 lit 检验 i32 格式下的 7 前导 0 个数为 29.

```mlir
// RUN: mlir-opt %s \
// RUN:   --pass-pipeline="builtin.module( \
// RUN:      convert-math-to-funcs{convert-ctlz}, \
// RUN:      func.func(convert-scf-to-cf,convert-arith-to-llvm), \
// RUN:      convert-func-to-llvm, \
// RUN:      convert-cf-to-llvm, \
// RUN:      reconcile-unrealized-casts)" \
// RUN: | mlir-cpu-runner -e test_7i32_to_29 -entry-point-result=i32 > %t
// RUN: FileCheck %s --check-prefix=CHECK_TEST_7i32_TO_29 < %t

func.func @test_7i32_to_29() -> i32 {
  %arg = arith.constant 7 : i32
  %0 = math.ctlz %arg : i32
  func.return %0 : i32
}
// CHECK_TEST_7i32_TO_29: 29
```

这条命令具体做了以下几件事情：

1. `mlir-opt %s` ：运行 `mlir-opt` 工具，`%s` 代表当前的 `.mlir` 文件。这个工具会将 MLIR 文件中定义的内容按顺序进行优化和转换。
2. `--pass-pipeline` ：定义了一个 Pass 管道（Pipeline）。这表示一系列的 Pass 要按顺序执行。这些 Pass 包括：

   * `convert-math-to-funcs{convert-ctlz}`：将 `math.ctlz` 操作转换为一个函数调用。
   * `convert-scf-to-cf` 和 `convert-arith-to-llvm`：分别将结构化控制流（`scf`）和算术操作（`arith`）转换为控制流方言（`cf`）和 LLVM IR 方言（`llvm`）。
   * `convert-func-to-llvm`：将标准的 `func` 转换为 `llvm` 方言。
   * `convert-cf-to-llvm`：将控制流方言中的操作转换为 `llvm` 方言中的等效操作。
   * `reconcile-unrealized-casts`：处理类型转换，以确保所有操作的输入输出类型一致。
3. `mlir-cpu-runner`：接收转换后的 MLIR 代码并执行。这里使用了 `-e test_7i32_to_29` 指定入口函数 `@test_7i32_to_29`，并使用 `-entry-point-result=i32` 表示函数返回一个 `i32` 类型的结果。
4. 重定向输出到 `%t`：将运行结果重定向到一个临时文件 `%t`。
5. `FileCheck %s --check-prefix=CHECK_TEST_7i32_TO_29 < %t` ：使用 `FileCheck` 来验证运行结果。`--check-prefix=CHECK_TEST_7i32_TO_29` 表示使用 `CHECK_TEST_7i32_TO_29` 前缀的检查指令。

经过以上一系列 pass 的优化后生成的 mlir 如下

```mlir
module {
  llvm.func @test_7i32_to_29() -> i32 {
    %0 = llvm.mlir.constant(7 : i32) : i32
    %1 = llvm.mlir.constant(29 : i32) : i32
    llvm.return %1 : i32
  }
  llvm.func linkonce_odr @__mlir_math_ctlz_i32(%arg0: i32) -> i32 attributes {sym_visibility = "private"} {
    %0 = llvm.mlir.constant(32 : i32) : i32
    %1 = llvm.mlir.constant(0 : i32) : i32
    %2 = llvm.icmp "eq" %arg0, %1 : i32
    llvm.cond_br %2, ^bb1, ^bb2
  ^bb1:  // pred: ^bb0
    llvm.br ^bb10(%0 : i32)
  ^bb2:  // pred: ^bb0
    %3 = llvm.mlir.constant(1 : index) : i64
    %4 = llvm.mlir.constant(1 : i32) : i32
    %5 = llvm.mlir.constant(32 : index) : i64
    %6 = llvm.mlir.constant(0 : i32) : i32
    llvm.br ^bb3(%3, %arg0, %6 : i64, i32, i32)
  ^bb3(%7: i64, %8: i32, %9: i32):  // 2 preds: ^bb2, ^bb8
    %10 = llvm.icmp "slt" %7, %5 : i64
    llvm.cond_br %10, ^bb4, ^bb9
  ^bb4:  // pred: ^bb3
    %11 = llvm.icmp "slt" %8, %1 : i32
    llvm.cond_br %11, ^bb5, ^bb6
  ^bb5:  // pred: ^bb4
    llvm.br ^bb7(%8, %9 : i32, i32)
  ^bb6:  // pred: ^bb4
    %12 = llvm.add %9, %4 : i32
    %13 = llvm.shl %8, %4 : i32
    llvm.br ^bb7(%13, %12 : i32, i32)
  ^bb7(%14: i32, %15: i32):  // 2 preds: ^bb5, ^bb6
    llvm.br ^bb8
  ^bb8:  // pred: ^bb7
    %16 = llvm.add %7, %3 : i64
    llvm.br ^bb3(%16, %14, %15 : i64, i32, i32)
  ^bb9:  // pred: ^bb3
    llvm.br ^bb10(%9 : i32)
  ^bb10(%17: i32):  // 2 preds: ^bb1, ^bb9
    llvm.br ^bb11
  ^bb11:  // pred: ^bb10
    llvm.return %17 : i32
  }
}
```

`^bb` 是 MLIR 中用于表示基本块 (basic block) 的符号。基本块是程序中一组有序的指令，具有一个入口点，且在执行时指令是按顺序执行的，直到通过 `llvm.br` (branch) 跳转到其他基本块。 注释中的 `// pred` 表示该块的前驱块。
