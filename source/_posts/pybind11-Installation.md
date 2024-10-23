---
title: pybind11-Installation
date: 2024/10/15 10:00:12
categories: Python-C++ Binding
tags: pybind
excerpt: pybind11 installation record
mathjax: true
katex: true
---
# Introduction

pybind11 是一个轻量级的纯头文件库，可以在 Python 中公开 C++ 类型，反之亦然，主要用于创建现有 C++ 代码的 Python 绑定。它的目标和语法与 David Abrahams 的 Boost.Python 库相似：通过使用编译时自省推断类型信息，最大限度地减少传统扩展模块中的模板代码。

> pybind11 is a lightweight header-only library that exposes C++ types in Python and vice versa, mainly to create Python bindings of existing C++ code. Its goals and syntax are similar to the excellent Boost.Python library by David Abrahams: to minimize boilerplate code in traditional extension modules by inferring type information using compile-time introspection.

# Installation

1. 选择要安装的 conda 环境并执行

```bash
pip install pybind11
```

通过 PyPI 来下载 Pybind11 的 Python 包，里面包含了源码已经CMake文件。

2. 创建一个 C++ 文件 example.cpp 用于测试

```C++
#include <pybind11/pybind11.h>
namespace py = pybind11


int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin";  // 可选的模块文档字符串
    m.def("add", &add, "A function which adds two numbers");
}
```

3. 创建 CMakeLists.txt 用于编译

```cmake
cmake_minimum_required(VERSION 3.4)
project(example)

set(CMAKE_CXX_STANDARD 11)

# 设置 Python 路径
set(Python_EXECUTABLE "D:/Work/Anaconda/envs/pytorch/python.exe")
set(Python_INCLUDE_DIRS "D:/Work/Anaconda/envs/pytorch/include")
set(Python_LIBRARIES "D:/Work/Anaconda/envs/pytorch/libs/python3.12.lib") # 替换为实际 python 版本

find_package(Python REQUIRED COMPONENTS Interpreter Development)

set(pybind11_DIR "D:/Work/Anaconda/envs/pytorch/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

pybind11_add_module(example example.cpp)
```

- `Python_xxx` 路径，在 anaconda prompt 激活对应环境后执行 `echo %CONDA_PREFIX%` 会返回环境地址，之后再加上对应的 executable, include, libraries 路径
- `pybind11_DIR` 可以使用 `python -m pybind11 --cmakedir` 获取 pybind11 的路径。

3. 进行构建

```bash
mkdir build
cd build
cmake --build . --config Release
```

成功后会在 `build/Release` 下生成 .pyd 后缀的 python 扩展模块文件

4. 导入模块进行测试

方法 1：将 `.pyd` 文件放入 Python 工作目录

1. **移动 `.pyd` 文件** ： 将 `example.cp312-win_amd64.pyd` 文件移动到你的 Python 项目目录或者 Python 的 `site-packages` 目录。在 `site-packages` 目录中，Python 会自动识别并加载这些模块。
   找到你的 `site-packages` 目录路径，可以在 Python 中运行以下命令：

```python
   import site
   print(site.getsitepackages())
```

   然后，将 `.pyd` 文件移动到输出路径中的其中一个目录。

```bash
   move E:\example\Project1\build\Release\example.cp312-win_amd64.pyd D:\Work\Anaconda\envs\pytorch\Lib\site-packages\
```

方法 2：在代码中动态添加路径

如果不想移动文件，可以在代码中动态添加模块文件的路径。

```python
import sys
import os

# 将 .pyd 文件所在的目录添加到 sys.path
sys.path.append(r"E:\example\Project1\build\Release")  # 替换为实际路径

import example

# 测试
result = example.add(3, 5)
print(result)  # 应输出 8
```

Docker 容器中的 SSL 证书验证失败

```bash
apt-get update --allow-releaseinfo-change
apt-get install --reinstall ca-certificates
update-ca-certificates
```

import jax 时候报错

当前安装的 `jax` 版本是 `0.4.29`，而 `opt-einsum` 的版本是 `3.4.0`

将 `opt-einsum` 降级到 `3.3.0`：

```bash
pip install opt-einsum==3.3.0

```
