---
title: "Cpp Lambda Expression"
date: 2025-08-15T12:16:48+08:00
lastmod: 2025-08-15T12:16:48+08:00
author: ["WITHER"]

categories:
- Productivity
- CPP

tags:
- CPP

keywords:


description: "Lambda Expression in CPP." # 文章描述，与搜索优化相关
summary: "Lambda Expression in CPP." # 文章简单描述，会展示在主页
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

C++ Lambda 表达式（也称为 Lambda 函数）是一种在代码中定义匿名函数的便捷方式。它特别适用于需要一个简短、临时的函数对象的场景，例如作为标准库算法的参数。

一个完整的 Lambda 表达式的通用语法如下：

```cpp
[capture-list](parameters) mutable exception -> return_type {
    // 函数体
    statement;
}
```

1. 捕获列表 `[capture_list]` 用于控制 Lambda 函数如何从其所在的父作用域“捕获”变量。
    - `[]`：空捕获列表。表示不捕获任何外部变量。Lambda 函数体内不能访问父作用域中的任何变量。
    - `[=]`：以值（by value）的方式捕获所有外部变量。在 Lambda 函数体内，你只能读取这些变量的值，不能修改它们（除非使用 mutable 关键字）。这相当于创建了外部变量的一份拷贝。
   - `[&]`：以引用（by reference）的方式捕获所有外部变量。在 Lambda 函数体内，你可以修改这些外部变量，并且修改会影响到原始变量。
   - `[this]`：以值的方式捕获当前对象的 this 指针。这使得你可以在 Lambda 函数体内访问当前对象的成员变量和成员函数。
   - `[a, &b]`：指定捕获列表。这里 a 以值的方式捕获，而 b 以引用的方式捕获。你可以混合使用值捕获和引用捕获。
   - `[=, &b]`：以值的方式捕获所有变量，但变量 b 除外，它以引用的方式捕获。
   - `[&, a]`：以引用的方式捕获所有变量，但变量 a 除外，它以值的方式捕获。

    ```cpp
    int x = 10;
    int y = 20;

    // 不捕获任何变量
    auto f1 = []() { return 5; };

    // 以值的方式捕获 x 和 y
    auto f2 = [=]() { return x + y; };

    // 以引用的方式捕获 x 和 y
    auto f3 = [&]() { x = 15; y = 25; };

    // 混合捕获
    auto f4 = [x, &y]() { y = 30; return x + y; };
    ```

2. 参数列表 (parameters): 和普通函数的参数列表一样，这部分是可选的。
    ```cpp
    // 没有参数
    auto greet = []() { std::cout << "Hello, World!" << std::endl; };

    // 接收两个 int 参数
    auto add = [](int a, int b) { return a + b; };

    // C++14 以后，可以使用 auto 进行泛型参数声明
    auto generic_add = [](auto a, auto b) { return a + b; };
    ```

3. mutable 关键字 (可选): 默认情况下，通过值捕获的变量在 Lambda 函数体内是 const 的，不能修改它们。如果希望能够修改这些按值捕获的变量的拷贝（注意，这不会影响原始变量），需要使用 mutable 关键字。

    ```cpp
    int value = 100;

    auto counter = [value]() mutable {
        value++; // 如果没有 mutable，这里会编译错误
        return value;
    };

    std::cout << counter() << std::endl; // 输出 101
    std::cout << counter() << std::endl; // 输出 102
    std::cout << value << std::endl;     // 原始 value 仍然是 100
    ```
4. 异常规范 exception (可选):可以使用 noexcept 来指明该 Lambda 函数不会抛出任何异常。

    ```cpp
    auto safe_divide = [](int a, int b) noexcept {
        return (b == 0) ? 0 : a / b;
    };
    ```
5. 返回类型 `-> return_type` (可选): 
在大多数情况下，编译器可以自动推断出 Lambda 表达式的返回类型。但如果函数体包含多个 return 语句，或者希望明确指定返回类型，就可以使用这个语法。

    ```cpp
    // 编译器可以自动推断返回类型为 int
    auto add = [](int a, int b) { return a + b; };

    // 明确指定返回类型为 double
    auto divide = [](int a, int b) -> double {
        if (b == 0) {
            return 0;
        }
        return static_cast<double>(a) / b;
    };
    ```