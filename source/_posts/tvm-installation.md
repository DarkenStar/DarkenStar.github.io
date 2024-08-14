---
title: TVM Building on Windows
categories: TVM
tags: TVM learning
excerpt: A Record of TVM Building on Windows 
---
# Preparation

1. Visual Studio: [Download Link](https://visualstudio.microsoft.com/), Community version is enough.
2. CMake: [Download Link](https://cmake.org/download/), choose latest release (binary distribution is convenient).
3. Anaconda:  [Download Link](https://anaconda.org/),  used to manage python enveriment.
4. Git:  [Download Link](https://git-scm.com/download/win),  used to pull tvm repo.

# Compile LLVM

Suppose you want to build TVM in directory `E:\tvm\`

Download and compile LLVM project: [Getting Started with the LLVM System — LLVM 20.0.0git documentation](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm), following the guide of Getting the Source Code and Building LLVM. (I can't build tvm from pre-built binary LLVM.)

{% note info %}

在 CMake 中，-S 参数指定了 CMakeLists.txt 文件所在的目录，-B 参数指定了构建文件的生成目录。

```bash
cmake -S <source_dir> -B <build_dir>
```

其中：

* `<source_dir>`：包含 CMakeLists.txt 文件的源代码目录。
* `<build_dir>`：用于存放构建文件的目录。

{% endnote %}

```bash
# Check out LLVM (including subprojects like Clang):
git clone --config core.autocrlf=false https://github.com/llvm/llvm-project.git
# change directory
cd llvm-project
# build
cmake -S llvm -B build -G Visual Studio
cmake --build build
```

If build process is successful, you will find `build/release/bin/llvm-config.exe`

# Compile TVM

Following the official guidance:  [Install from Source — tvm 0.18.dev0 documentation (apache.org)](https://tvm.apache.org/docs/install/from_source.html)

## Get Source from Github

```bash
# clone 
git clone --recursive https://github.com/dmlc/tvm tvm
# check submodule
git submodule init 
git submodule update
```

## Modify config.cmake file 

Modify ` /cmake/config.cmake`

Modify `set(USE_LLVM OFF)` to  `set(USE_LLVM "/path/to/llvm-config --link-static")`,  `/path/to/llvm-config` is the path of `llvm-config.exe `in the last step.

If want to use CUDA: `set(USE_CUDA ON)`

## Build Anaconda Environment

Install all necessary build dependencies.

```bash
cd ./tvm
# Create a conda environment with the dependencies specified by the yaml
conda env create --file conda/build-environment.yaml
# Activate the created environment
conda activate tvm-build
```

## Build Shared Library

TVM support build via MSVC using cmake.

{% note info %}

`cmake --build build --config Release -- /m` 用于在 `build` 目录中构建项目，并使用 `Release` 配置，同时启用多线程构建。

* `cmake --build build`：告诉 CMake 在 `build` 目录中执行构建操作。
* `--config Release`：指定构建配置为 `Release`。这通常意味着启用优化选项，以生成更小的可执行文件并提高性能。
* `-- /m`：这是 Windows 平台上的一个参数，用于启用多线程构建。它告诉 CMake 使用多个处理器核心来加速构建过程。

{% endnote %}

```bash
mkdir build
cd build
cmake -A x64 -Thost=x64 ..
cd ..
cmake --build build --config Release -- /m
```

If build process is successful, you will find `tvm.dll, tvm_runtime.dll` .etc in `tvm/build/Release`.

## Import tvm package into python

Open anaconda prompt in  `tvm/python`.

```bash
conda activate tvm-build
python setup.py install
```

Verify whether the package is installed or not.

```python
python
>>import tvm
>>print(tvm.__version__)
```
