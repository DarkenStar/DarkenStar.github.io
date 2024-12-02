---
title: pybind11-Installation
date: 2024/12/02 15:20:12
categories: Python-C++ Binding
tags: pybind
excerpt: pybind11 installation record
mathjax: true
katex: true
---
# Introduction

pybind11 是一个轻量级的纯头文件库，可以在 Python 中公开 C++ 类型，反之亦然，主要用于创建现有 C++ 代码的 Python 绑定。它的目标和语法与 David Abrahams 的 Boost.Python 库相似：通过使用编译时自省推断类型信息，最大限度地减少传统扩展模块中的模板代码。

> pybind11 is a lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code. Its goals and syntax are similar to the excellent Boost.Python library by David Abrahams: to minimize boilerplate code in traditional extension modules by inferring type information using compile-time introspection.

# Git Clone Source Code

首先在你想下载的文件夹下(我的是 `D:\`) git clone pybind11 的源码后进行构建，并运行测试用例

```bash
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake ..
cmake --build . --config Release --target check
```

如果没有报错则安装成功

# Visual Studio Project Properties Configuration

创建一个 Visual Studio 的一个空项目，并新建一个 .cpp 文件，以一个简单的加法程序作为测试

```cpp
#include"pybind11/pybind11.h"

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "TestPybind plugin";
    m.def("add", &add, "A function that adds two integers");
}
```

接着我们打开项目的 Property Pages (属性页)，修改 Configuration Type 为 Dynamic Library(.dll)

![Configuration Type](https://note.youdao.com/yws/api/personal/file/WEBc899c7ae6555c5350aa0552639832df1?method=download&shareKey=32e95705185dd0d926c832a9d7894285 "Configuration Type")

{% note info %}
这里需要注意我们要使上一行 Target Name (默认和 ProjectName 相等) 和 `PYBIND11_MODULE` 的模块名一致，否则后面从 python import 时候会报错。
{% endnote %}

然后在 Advanced 中修改后缀名称为 .pyd

![Target File Extension](https://note.youdao.com/yws/api/personal/file/WEB3157936ceabae45bba0942113730f1d7?method=download&shareKey=359620a4deb60a22105eace5a412f310 "Target File Extension")

接着我们需要在 C/C++ 的 General 选项卡中添加 python 和 pybind11 的包含目录，我是通过 miniconda 安装的 python，因此 python.h 所在的包含目录位置为 `C:\Users\$(UserName)\miniconda3\include`. pybind11 的包含目录在刚才 git clone 源码的文件夹下 `D:\pybind11\include`

![Additional Include Directories](https://note.youdao.com/yws/api/personal/file/WEBf8dbe9524a2884d432a37e74f42865eb?method=download&shareKey=99b52578a779fe100736f36709a0f225 "Additional Include Directories")

![Add python & pybind11 include Directories](https://note.youdao.com/yws/api/personal/file/WEB812be16a1deefe9a72e87ee433d243b7?method=download&shareKey=bdd01c679498e701324c4e4a8cddca37 "Add python & pybind11 include Directories")

然后在 Linker 的 General 选项卡中添加 python 的库目录 (前文已经说过 pybind11 是一个 header—only 库) `C:\Users\$(UserName)\miniconda3\libs` 

![Additional Library Directories](https://note.youdao.com/yws/api/personal/file/WEB59d1852e15cfb3c91f569b08213e22ac?method=download&shareKey=2413a1377116273ecf26de07d7dbca74 "Additional Library Directories")

![Add python Library Directories](https://note.youdao.com/yws/api/personal/file/WEB6979ba5b4c026a5d7b4e749ea2492f02?method=download&shareKey=d114804fa69efd7c11f291ac0ddcaea1 "Add python Library Directories")


右键项目进行 build，成功后会在项目目录下的 `x64\Debug` 文件夹下 生成 .pyd 文件，可以在命令行中进行测试。

![Test Module](https://note.youdao.com/yws/api/personal/file/WEBb5c3df3e13e95ccc9250a15c90233b6c?method=download&shareKey=5fc5f0db83ca30c9d0e3d705de285f0c "Test Module")