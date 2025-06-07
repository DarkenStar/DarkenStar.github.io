---
title: MLIR-Ch4 Defining a New Dialect
date: 2024-11-07T18:16:23+08:00
lastmod: 2024-11-07T18:16:23+08:00
draft: false
author: ["WITHER"]
keywords: 
    - MLIR
categories:
    - MLIR
tags:
    - jeremykun MLIR learning
description: Personal MLIR learning notes 4.  
summary: Personal MLIR learning notes 4.  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# Sketching Out a Dseign

TableGen 也可以用来定义 dialect. 本文将定义一个单未知数多项式运算的 dialect，系数用 uint32_t 类型表示。，并提供通过从标准 MLIR 类型指定多项式系数来定义多项式的操作，提取关于多项式的数据以将结果存储在标准MLIR类型中，以及对多项式进行算术运算。

# An Empty Dialect

我们首先用 TableGen 定义一个空的 dialect. 它和上一章定义 Pass 没什么不同，只不过 include 的是 DialectBase.td 文件。同时也定义了命名空间为 `::mlir::tutorial::poly`.

```tablegen {linenos=true}
include "mlir/IR/DialectBase.td"

def Poly_Dialect : Dialect {
  let name = "poly";
  let summary = "A dialect for polynomial math";
  let description = [{
    The poly dialect defines types and operations for single-variable
    polynomials over integers.
  }];

  let cppNamespace = "::mlir::tutorial::poly";
}
```

我们需要在 include 目录下的 CMakeLists.txt 文件中添加

```cmake {linenos=true}
set(TARGET_NAME "${PROJECT_TARGET_PREFIX}-Dialect-PolyDialect-IncGen")
set(LLVM_TARGET_DEFINITIONS mlir-learning/Dialect/Poly/PolyDialect.td)
mlir_tablegen(mlir-learning/Dialect/Poly/PolyDialect.hpp.inc --gen-dialect-decls)
mlir_tablegen(mlir-learning/Dialect/Poly/PolyDialect.cpp.inc --gen-dialect-defs)
add_public_tablegen_target(${TARGET_NAME})
```

然后在 tutorial-opt.cpp 中注册所有 mlir 自带的所有 dialect 后进行构建，我们可以查看生成的 .hpp.inc 和.cpp.inc 文件。

```cpp {linenos=true}
namespace mlir {
namespace tutorial {

class PolyDialect : public ::mlir::Dialect {
  explicit PolyDialect(::mlir::MLIRContext *context);

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~PolyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("poly");
  }
};
} // namespace tutorial
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tutorial::PolyDialect)
```

编译器会报错，因为 inc 不会包含 Dialect 等类所在的头文件。这需要我们自己在 PolyDialect.h 文件中进行 include，这样 当重新构建的时候该文件注入变不会报错

```cpp {linenos=true}
// include/mlir-learning/Dialect/Poly/PolyDialect.h
#ifndef LIB_DIALECT_POLY_POLYDIALECT_H
#define LIB_DIALECT_POLY_POLYDIALECT_H

#include "mlir/IR/DialectImplementation.h"  // include mannually

#include "mlir-learning/Dialect/Poly/PolyDialect.hpp.inc"

#endif
```

生成的 .cpp.inc 如下，他只包含了该类基本的构造函数和析构函数。

```cpp {linenos=true}
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tutorial::poly::PolyDialect)
namespace mlir {
namespace tutorial {
namespace poly {

PolyDialect::PolyDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<PolyDialect>())
  
     {
  
  initialize();
}

PolyDialect::~PolyDialect() = default;

} // namespace poly
} // namespace tutorial
} // namespace mlir
```

然后我们可以在 tutorial-opt.cpp 中注册该 dialect.

```cpp {linenos=true}
/* other includes */
#include "mlir-learning/Dialect/Poly/PolyDialect.h"

int main(int argc, char** argv) {
    // Register all built-in MLIR dialects
    mlir::DialectRegistry registry;
    // Register our Dialect
    registry.insert<mlir::tutorial::poly::PolyDialect>();
    mlir::registerAllDialects(registry);
    return mlir::asMainReturnCode(
        mlir::MlirOptMain(argc, argv, "Tutorial Pass Driver", registry));
}
```

# Adding a Trival Type

下面我们需要定义自己的 poly.poly 类型.

```tablegen {linenos=true}
// poly_types.td
#ifndef LIB_DIALECT_POLY_POLYTYPES_TD_
#define LIB_DIALECT_POLY_POLYTYPES_TD_

include "mlir-learning/Dialect/Poly/PolyDialect.td"
include "mlir/IR/AttrTypeBase.td"

// a base class for all types in the dialect
class Poly_Type<string name, string typeMnemonic> : TypeDef<Poly_Dialect, name> {
    let mnemonic = typeMnemonic;
}

def Polynomial: Poly_Type<"Polynomial", "poly"> {
  let summary = "A polynomial with u32 coefficients";

  let description = [{
    A type for polynomials with integer coefficients in a single-variable polynomial ring.
  }];
}
#endif 
```

在 MLIR 的 TableGen 文件中，class 和 def 的用法和含义有所不同

- `class` 用于定义一个模板或基类，可以被其他类型或定义继承和重用。它本身不会创建实际的对象或具体类型，它只是一种结构，可以包含参数和默认属性。其他定义可以通过继承该类来获得其功能。
- `def` 用于创建一个具体的实例，比如一个类型、操作或属性。它会将所定义的内容应用到 TableGen 中，使其成为可用的具体类型或功能。

这里我们定义了一个名为 `Poly_Type` 的类，参数为 `name`（类型的名称）和 `typeMnemonic`（类型的简写或助记符）。这个类继承自 `TypeDef<Poly_Dialect, name>`. 然后 `def` 特定的多项式类型 `Polynomial`，继承自 `Poly_Type`.

在 MLIR 的 TableGen 中，[TypeDef](https://github.com/llvm/llvm-project/blob/630ba7d705fa1d55096dbbf88c6886d64033a780/mlir/include/mlir/IR/AttrTypeBase.td#L281) 本身也是一个类，它接受模板参数，用于指定该类型所属的 dialect 和名称字段。其作用包括将生成的C++类与该 dialect 的命名空间相关联。

生成的 .hpp.inc 文件如下。生成的类 `PolynomialType` 就是在我们的 TableGen 文件中定义的 `Polynomial` 类型后面加上了 Type.

```cpp {linenos=true}
#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES


namespace mlir {
class AsmParser;
class AsmPrinter;
} // namespace mlir
namespace mlir {
namespace tutorial {
namespace poly {
class PolynomialType;
class PolynomialType : public ::mlir::Type::TypeBase<PolynomialType, ::mlir::Type, ::mlir::TypeStorage> {
public:
  using Base::Base;
  static constexpr ::llvm::StringLiteral name = "poly.poly";
  static constexpr ::llvm::StringLiteral dialectName = "poly";
  static constexpr ::llvm::StringLiteral getMnemonic() {
    return {"poly"};
  }

};
} // namespace poly
} // namespace tutorial
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::tutorial::poly::PolynomialType)

#endif  // GET_TYPEDEF_CLASSES
```

生成的 .cpp.inc 文件如下。TableGen 试图为 dialect 中的 `PolynomialType` 自动生成一个 类型解析器 (type parser) 和类型打印器 (type printer). 不过此时这些功能还不可用，构建项目时会看到一些编译警告。

代码中使用了 头文件保护 (header guards) 来将 `cpp` 文件分隔为两个受保护的部分。这样可以分别管理类型声明和函数实现。

`GET_TYPEDEF_LIST` 只包含类名的逗号分隔列表。原因在于 `PolyDialect.cpp` 文件需要负责将类型注册到 dialect 中，而该注册过程通过在方言初始化函数中将这些 C++ 类名作为模板参数来实现。换句话说，`GET_TYPEDEF_LIST` 提供了一种简化机制，使得 `PolyDialect.cpp` 可以自动获取所有类名称列表，便于统一注册，而不需要手动添加每一个类型。

* **`generatedTypeParser`** 函数是为 `PolynomialType` 定义的解析器。当解析器遇到 `PolynomialType` 的助记符（`poly`）时，会将 `PolynomialType` 类型实例化。`KeywordSwitch` 使用 `getMnemonic()` 来匹配 `PolynomialType` 的助记符（`poly`）。如果匹配成功，则调用 `PolynomialType::get()` 来获取类型实例。`Default` 子句在助记符不匹配时执行，记录未知的助记符，并返回 `std::nullopt` 表示解析失败。
* **`generatedTypePrinter`** 函数为 `PolynomialType` 提供了打印功能。当类型为 `PolynomialType` 时，打印其助记符（`poly`），否则返回失败。`TypeSwitch` 用于检查 `def` 类型是否是 `PolynomialType`。如果是，打印助记符；否则返回失败，表示该类型不属于此方言。
* `PolyDialect::parseType` 和 `PolyDialect::printType` 作为方言接口调用这两个函数，从而实现类型的解析和打印功能。

```cpp {linenos=true}
#ifdef GET_TYPEDEF_LIST
#undef GET_TYPEDEF_LIST

::mlir::tutorial::poly::PolynomialType

#endif  // GET_TYPEDEF_LIST

#ifdef GET_TYPEDEF_CLASSES
#undef GET_TYPEDEF_CLASSES

static ::mlir::OptionalParseResult generatedTypeParser(::mlir::AsmParser &parser, ::llvm::StringRef *mnemonic, ::mlir::Type &value) {
  return ::mlir::AsmParser::KeywordSwitch<::mlir::OptionalParseResult>(parser)
    .Case(::mlir::tutorial::poly::PolynomialType::getMnemonic(), [&](llvm::StringRef, llvm::SMLoc) {
      value = ::mlir::tutorial::poly::PolynomialType::get(parser.getContext());
      return ::mlir::success(!!value);
    })
    .Default([&](llvm::StringRef keyword, llvm::SMLoc) {
      *mnemonic = keyword;
      return std::nullopt;
    });
}

static ::llvm::LogicalResult generatedTypePrinter(::mlir::Type def, ::mlir::AsmPrinter &printer) {
  return ::llvm::TypeSwitch<::mlir::Type, ::llvm::LogicalResult>(def)    .Case<::mlir::tutorial::poly::PolynomialType>([&](auto t) {
      printer << ::mlir::tutorial::poly::PolynomialType::getMnemonic();
      return ::mlir::success();
    })
    .Default([](auto) { return ::mlir::failure(); });
}

namespace mlir {
namespace tutorial {
namespace poly {
} // namespace poly
} // namespace tutorial
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tutorial::poly::PolynomialType)
namespace mlir {
namespace tutorial {
namespace poly {

/// Parse a type registered to this dialect.
::mlir::Type PolyDialect::parseType(::mlir::DialectAsmParser &parser) const {
  ::llvm::SMLoc typeLoc = parser.getCurrentLocation();
  ::llvm::StringRef mnemonic;
  ::mlir::Type genType;
  auto parseResult = generatedTypeParser(parser, &mnemonic, genType);
  if (parseResult.has_value())
    return genType;
  
  parser.emitError(typeLoc) << "unknown  type `"
      << mnemonic << "` in dialect `" << getNamespace() << "`";
  return {};
}
/// Print a type registered to this dialect.
void PolyDialect::printType(::mlir::Type type,
                    ::mlir::DialectAsmPrinter &printer) const {
  if (::mlir::succeeded(generatedTypePrinter(type, printer)))
    return;
  
}
} // namespace poly
} // namespace tutorial
} // namespace mlir

#endif  // GET_TYPEDEF_CLASSES
```

在设置 C++ 接口以使用 TableGen 文件时，通常会按照以下步骤来组织代码文件和包含关系。

- `PolyTypes.h` 是唯一被允许包含 `PolyTypes.h.inc` 的文件。
- `PolyTypes.cpp.inc` 文件包含了 TableGen 为 `PolyDialect` 中的类型生成的实现。我们需要在 `PolyDialect.cpp` 中将其包含进去，以确保所有实现都能在该方言的主文件中使用。
- `PolyTypes.cpp` 文件应该包含 `PolyTypes.h`，以便访问类型声明，并在该文件中实现所有需要的额外功能。

```plaintexxt {linenos=true}
./Ch3-DefiningANewDialect/
├── CMakeLists.txt
├── include
│   ├── CMakeLists.txt
│   └── mlir-tutorial
│       └── Dialect
│           └── Poly
│               ├── PolyDialect.hpp
│               ├── PolyDialect.td
│               ├── PolyOps.hpp
│               ├── PolyOps.td
│               ├── PolyTypes.hpp
│               └── PolyTypes.td
├── lib
│   ├── CMakeLists.txt
│   └── Dialect
│       └── Poly
│           └── PolyDialect.cpp
```

为了让类型解析器和打印器能够正确编译和运行，需要最后在方言的 TableGen 文件中添加 `let useDefaultTypePrinterParser = 1`;，这个指令告诉 TableGen 使用默认的类型解析和打印器。当这个选项启用后，TableGen 会生成相应的解析和打印代码，并将这些实现作为 `PolyDialect` 类的成员函数。

```cpp {linenos=true}
/// Parse a type registered to this dialect.
  ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;

  /// Print a type registered to this dialect.
  void printType(::mlir::Type type,
                 ::mlir::DialectAsmPrinter &os) const override;
```

我们可以写一个 .mlir 来测试属性是是否获取正确。在 MLIR 中自定义的 dialect 前都需要加上 `!`.

```mlir {linenos=true}
    // CHECK-LABEL: test_type_syntax
    func.func @test_type_syntax(%arg0: !poly.poly<10>) -> !poly.poly<10> {
        // CHECK: poly.poly
        return %arg0: !poly.poly<10>
    }
```

# Add a Poly Type Parameter

我们需要为多项式类型添加一个属性，表示它的次数上限。

```
// include/mlir-tutorial/Dialect/Poly/PolyTypes.td
let parameters = (ins "int":$degreeBound);
let assemblyFormat = "`<` $degreeBound `>`";
```

第一行定义了类型的一个参数 `degreeBound`，类型为 `int`. 表示在实例化该类型时，用户可以指定一个整数值作为类型的参数。`parameters` 中的 (`ins "int":$degreeBound`) 指定了输入参数的类型和名称，其中 int 是数据类型，`$degreeBound` 是参数的占位符。`assemblyFormat` 用于定义该类型在 MLIR 文本格式中的打印和解析格式。`"<" $degreeBound ">"` 表示该类型的参数会用尖括号包裹。第二行是必需的，因为现在一个 Poly 类型有了这个关联的数据，我们需要能够将它打印出来并从文本 IR 表示中解析它。

加上这两行代码后进行 build 会发现多了一些新的内容。

- `PolynomialType` 有一个新的 `int getDegreeBound()` 方法，以及一个静态 `get` 工厂方法。
- `parse` 和 `print` 升级为新格式。
- 有一个名为 `typestorage` 的新类，它包含 int 形参，并隐藏在内部细节名称空间中。

MLIR会自动生成简单类型的 storage 类，因为它们不需要复杂的内存管理。如果参数更复杂，就需要开发者手动编写 storage 类来定义构造、析构和其他语义。复杂的 storage 类需要实现更多细节，以确保类型能够在 MLIR 的 dialect 系统中顺利运行。

```cpp {linenos=true}
// include/mlir-learning/Dialect/Poly/PolyTypes.hpp.inc
  static ::mlir::Type parse(::mlir::AsmParser &odsParser);
  void print(::mlir::AsmPrinter &odsPrinter) const;
  int getDegreeBound() const;

// include/mlir-learning/Dialect/Poly/PolyTypes.cpp.inc

struct PolynomialTypeStorage : public ::mlir::TypeStorage {
    /* lots of code */
};

PolynomialType PolynomialType::get(::mlir::MLIRContext *context, int degreeBound) {
  return Base::get(context, std::move(degreeBound));
}

::mlir::Type PolynomialType::parse(::mlir::AsmParser &odsParser) {
    /* code to parse the type */
}

void PolynomialType::print(::mlir::AsmPrinter &odsPrinter) const {
  ::mlir::Builder odsBuilder(getContext());
  odsPrinter << "<";
  odsPrinter.printStrippedAttrOrType(getDegreeBound());
  odsPrinter << ">";
}

int PolynomialType::getDegreeBound() const {
  return getImpl()->degreeBound;
}
```

# Adding Some Simple Operations

下面我们定义一个简单的多项式加法操作

```
// include/mlir-tutorial/Dialect/Poly/PolyOps.td
include "PolyDialect.td"
include "PolyTypes.td"

def Poly_AddOp : Op<Poly_Dialect, "add"> {
  let summary = "Addition operation between polynomials.";
  let arguments = (ins Polynomial:$lhs, Polynomial:$rhs);
  let results = (outs Polynomial:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($output)";
}
```

和刚才定义 types 非常相近，但基类是 Op，arguments 对应于操作的输入，assemblyFormat 更复杂。生成的 .hpp.inc 和 .cpp.inc 非常复杂。我们可以编写一个 .mlir 来测试。

```mlir {linenos=true}
  // CHECK-LABEL: test_add_syntax
  func.func @test_add_syntax(%arg0: !poly.poly<10>, %arg1: !poly.poly<10>) -> !poly.poly<10> {
    // CHECK: poly.add
    %0 = poly.add %arg0, %arg1 : (!poly.poly<10>, !poly.poly<10>) -> !poly.poly<10>
    return %0 : !poly.poly<10>
  }
```

生成的代码定义了以下几个方面：

1. Adaptor Classes:

   - AddOpGenericAdaptorBase 和 AddOpAdaptor: 提供了便捷的方式来访问操作的操作数 (operands) 和属性 (attributes)。它们在编写转换和重写模式时特别有用。
2. Properties Handling:

   - 诸如 setPropertiesFromAttr , getPropertiesAsAttr , computePropertiesHash 等函数是 MLIR 操作属性系统的接口。虽然在这个特定的 AddOp 实现中，有些函数可能是空实现或返回默认值，但它们是操作定义结构的一部分。
3. Builder Methods:

   - 多个重载的 AddOp::build 静态方法。这些方法用于在代码中以编程方式创建 AddOp 的实例。
4. Verification:

   - AddOp::verifyInvariantsImpl() 和 AddOp::verifyInvariants() : 这些方法用于检查一个 AddOp 实例是否符合其定义。例如，它们会验证操作数的数量和类型是否正确，结果类型是否符合预期。代码中调用了像 __mlir_ods_local_type_constraint_PolyOps2 这样的辅助函数来进行类型约束检查。
5. Assembly Format Parsing and Printing:

   - AddOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result) : 这个方法定义了如何从 MLIR 的文本汇编格式中解析出 AddOp 。当 MLIR 工具读取 .mlir 文件时，会调用此方法。
   - AddOp::print(::mlir::OpAsmPrinter& _odsPrinter) : 这个方法定义了如何将 AddOp 实例打印成 MLIR 的文本汇编格式。
6. Type ID Definition:

   - MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::tutorial::poly::AddOp) : 这个宏用于 MLIR 的运行时类型信息 (RTTI) 系统，为 AddOp 类型生成一个唯一的标识符。