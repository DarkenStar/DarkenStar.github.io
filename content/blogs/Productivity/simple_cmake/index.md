---
title: A Simple Cmake Example
date: 2025-06-06T18:32:00+08:00
lastmod: 2025-06-06T18:32:00+08:00
draft: false
author: ["WITHER"]
keywords: 
    - cmake
categories:
    - Productivity
    - cmake Learning
tags:
    - cmake Learning
description: A Simple Cmake Example
summary: A Simple Cmake Example
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
CMake 入门教程：从项目结构到链接库

1. 核心理念：源码外构建 (Out-of-Source Builds)

在开始之前，最重要的一点是理解 CMake 的核心哲学：源码外构建。这意味着所有由构建过程产生的文件（例如 Makefiles、Visual Studio 项目文件、目标文件 .o、可执行文件 .exe、库文件 .a 或 .so）都应该与你的源代码完全分离开。这样做最大的好处是能保持你的源码目录永远干净整洁。我们将创建一个 build 目录来存放所有这些生成的文件。

2. 推荐的项目目录结构 📂

一个良好组织的 C++ 项目结构不仅清晰，也让 CMake 的配置工作事半功倍。这是一个推荐的、可扩展的目录结构：my_project/

```plaintext
│
├── build/                  # 构建目录 (初始为空，所有生成文件都在此)
│
├── include/                # 存放项目全局头文件
│   └── my_app/
│       └── my_lib.h
│
├── src/                    # 存放所有源文件 (.cpp)
│   │
│   ├── main.cpp            # 主程序入口
│   │
│   └── my_lib/             # 一个独立的库模块
│       ├── CMakeLists.txt  # 这个库自己的 CMake 配置文件
│       └── my_lib.cpp
│
└── CMakeLists.txt          # 整个项目的顶层 CMake 配置文件
```

- build/: 这个目录用于执行所有构建命令，源码不会被污染。include/: 存放可以被项目内其他部分（或被其他项目）引用的头文件。按模块组织可以避免头文件名冲突。src/: 存放所有 .cpp 源文件。
- src/my_lib/: 将项目按功能模块化是一种好习惯。每个模块（比如一个库）可以有自己的 CMakeLists.txt 文件，负责管理自身的编译。
- CMakeLists.txt (顶层): 这是整个项目的入口，负责设置全局配置、找到并构建所有子模块，最后生成主程序。

3. 编写各层级的 CMakeLists.txt 📝我们将采用“自下而上”的方式来编写配置文件，先从底层的库开始，再到顶层的项目。
   第 1 步: 库的 CMakeLists.txt (src/my_lib/CMakeLists.txt

)这个文件只负责一件事：将 my_lib.cpp 和相关的头文件编译成一个库。# 文件位置: src/my_lib/CMakeLists.txt

```cmake
# 使用 add_library 命令创建一个库。
# 语法: add_library(<库名称> [STATIC | SHARED] <源文件...>)
#
# <库名称>: 我们称之为 my_lib，这是其他部分链接此库时使用的名字。
# STATIC:   生成静态链接库 (.a, .lib)。
# SHARED:   生成动态/共享链接库 (.so, .dll)。
#           如果不指定，默认是 STATIC。
# <源文件>:  用于编译这个库的源文件列表。
add_library(my_lib STATIC my_lib.cpp)

# 为这个库目标指定它需要包含的头文件目录。
# 语法: target_include_directories(<目标> <PUBLIC|PRIVATE|INTERFACE> <路径...>)
#
# <目标>:    就是我们上面用 add_library 创建的 my_lib。
# PUBLIC:   表示此头文件路径不仅 my_lib 自己需要，任何链接了 my_lib 的目标也需要。
#           这是最关键的设置，它实现了依赖的自动传递。
# PRIVATE:  表示此头文件路径只有 my_lib 内部编译时需要，不会传递给链接它的目标。
# INTERFACE:表示此头文件路径只有链接它的目标需要，my_lib 自己编译时不需要。
target_include_directories(my_lib
  PUBLIC
    # ${PROJECT_SOURCE_DIR} 是一个非常有用的内置变量，指向顶层 CMakeLists.txt 所在的目录。
    # 我们将项目的全局 include 目录暴露出去。
    ${PROJECT_SOURCE_DIR}/include
)
```

- `add_library()` 定义了一个编译目标——一个库。
- `target_include_directories()` 为这个目标指定了头文件搜索路径。使用 `PUBLIC `关键字至关重要使得任何链接到 `my_lib` 的程序都能自动找到 my_lib.h，无需在链接方再次手动添加头文件路径。

第 2 步: 顶层的 CMakeLists.txt 这个文件是整个项目的总指挥，负责设置全局配置、调用子模块，并生成最终的可执行文件。

```cmake
# 文件位置: my_project/CMakeLists.txt

# 1. 指定 CMake 的最低版本要求。这是每个顶层文件都应该有的第一行。
cmake_minimum_required(VERSION 3.10)

# 2. 定义项目信息。
# 语法: project(<项目名称> VERSION <版本号> LANGUAGES <语言>)
# 这会创建一些有用的变量，比如 PROJECT_NAME, PROJECT_SOURCE_DIR。
project(MyApp VERSION 1.0 LANGUAGES CXX)

# 3. 设置 C++ 标准 (这是现代 CMake 推荐的方式)。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 4. 打印一条消息，方便调试时查看变量值 (可选)。
message(STATUS "Project source directory is: ${PROJECT_SOURCE_DIR}")

# 5. 添加子目录。
# 这个命令会告诉 CMake 去处理 src/my_lib 目录下的 CMakeLists.txt 文件。
# 当执行到这里时，上面定义的 my_lib 库目标就会被创建出来。
add_subdirectory(src/my_lib)

# 6. 添加可执行文件。
# 语法: add_executable(<可执行文件名> <源文件...>)
# 我们将主程序命名为 app，它由 src/main.cpp 编译而来。
add_executable(app src/main.cpp)

# 7. 链接库！这是将所有部分组合在一起的关键步骤。
# 语法: target_link_libraries(<目标> <PUBLIC|PRIVATE|INTERFACE> <要链接的库...>)
#
# <目标>: 我们要链接的目标，即 app。
# PRIVATE: 表示 app 的编译需要 my_lib，但这个依赖关系不会继续传递。
#          对于可执行文件，通常使用 PRIVATE。
# <要链接的库>: 我们在子目录中定义的库目标 my_lib。
target_link_libraries(app PRIVATE my_lib)
```

- add_subdirectory() 使得顶层文件保持简洁，只负责“指挥”，具体实现则交给各个子模块。
- target_link_libraries() 负责将不同的编译目标（库和可执行文件）链接在一起，形成依赖关系。

4. 如何构建项目 🚀
   现在已经写好了所有的 CMakeLists.txt 文件，可以开始构建了。整个过程都在终端中完成。

```bash
# 1. 确保你位于项目的根目录 (my_project)
cd path/to/my_project

# 2. 创建并进入我们规划好的 build 目录

mkdir build
cd build

# 3. 运行 CMake 来生成构建系统。
# '..' 指向上一级目录，也就是 my_project/ 根目录，CMake 会在那里寻找顶层的 CMakeLists.txt。
# -DCMAKE_BUILD_TYPE=Debug 指定了构建类型为 Debug，会包含调试信息。

cmake -DCMAKE_BUILD_TYPE=Debug ..

# CMake 会扫描你的系统，找到 C++ 编译器，然后根据 CMakeLists.txt 的内容
# 生成特定平台的构建文件（在 Linux/macOS 上是 Makefile，在 Windows 上是 Visual Studio sln 文件）。

# 4. 编译项目
# 这个命令会调用底层的构建工具（如 make 或 msbuild）来执行真正的编译和链接工作。
# '--build .' 是一个平台无关的命令，告诉 CMake 在当前目录执行构建。

cmake --build .

# 或者在 Linux/macOS 上，你可以直接运行:
# make
# 编译完成后，你会在 build 目录（或其子目录）下找到你的可执行文件 `app` 和库文件 `libmy_lib.a`。
```
