---
title: "astra-Sim"
date: 2025-06-09T13:34:39+08:00
lastmod: 2025-06-09T13:34:39+08:00
author: ["WITHER"]

categories:
- Source Code Reading

tags:
- astra-sim

keywords:
- astra-sim

description: "source code reading of astra-sim" # 文章描述，与搜索优化相关
summary: "source code reading of astra-sim" # 文章简单描述，会展示在主页
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
# Build Analytical Backend
`build.sh` 脚本是构建过程的高级控制器。其核心职责是解析用户意图，执行预构建步骤，并以正确的参数调用底层的 CMake 工具链。


1.  **选项解析**: 脚本通过 `getopts` 处理以下命令行标志：
    * `-t <target>`: 指定编译目标。有效值为 `all`, `congestion_unaware`, `congestion_aware`。此值将作为变量传递给 CMake。
    * `-l`: 触发清理 (`cleanup`) 流程，删除所有构建产物并终止脚本。
    * `-d`: 启用调试 (`Debug`) 模式进行编译。

2.  **环境准备 (`setup`, `compile_chakra_et`)**:
    * `setup` 函数负责创建用于存放中间文件和最终产物的 `build` 目录，确保源码树的清洁。同时，它会根据系统核心数设置一个上限为 16 的并发编译线程数，以优化编译效率。
    * `compile_chakra_et` 函数负责处理 `et_def.proto` 这一 Protobuf 依赖。它检查目标文件是否存在，若不存在，则调用 `protoc` 编译器生成相应的 C++ 和 Python 源码。

3.  **构建执行 (`compile_astrasim_analytical`, `compile_astrasim_analytical_as_debug`)**:
    * 这两个函数是脚本与 CMake 交互的核心。它们根据用户是否指定 `-d` 标志，决定是执行标准 `Release` 构建还是 `Debug` 构建。关键在于它们会将用户指定的 `build_target` 作为 `-DBUILDTARGET` 参数传递给 CMake。

4.  **后处理 (`create_symlink_*`)**:
    * 编译完成后，`create_symlink_congestion_unaware` 和 `create_symlink_congestion_aware` 等函数会为生成的二进制文件创建符号链接。此举旨在维持对旧文件路径的向后兼容性。

---

`CMakeLists.txt` 文件是项目的构建蓝图，它向 CMake 阐述了项目的结构、依赖关系以及编译规则。

1.  **编译环境设定**:
    * `cmake_minimum_required(VERSION 3.15)`: 规定了运行此配置所需的最低 CMake 版本。
    * `set(CMAKE_CXX_STANDARD 17)` 和 `set(CMAKE_CXX_STANDARD_REQUIRED ON)`: 强制项目必须在支持 C++17 标准的编译环境中构建。

2.  **编译标志 (Compiler Flags)**:
    * 此文件为不同的构建类型（`CMAKE_BUILD_TYPE`）定义了不同的编译器标志。
    * **`Release`** (默认模式): `set(CMAKE_CXX_FLAGS_RELEASE "-O3")` 指示编译器进行高等级优化，以追求最大化程序性能。
    * **`Debug`**: `set(CMAKE_CXX_FLAGS_DEBUG "...")` 包含一系列用于调试的标志：
        * `-O0`: 关闭所有优化，确保编译后的代码与源码行为一致。
        * `-g`: 在可执行文件中包含调试符号，这是 GDB 等调试器工作的前提。
        * `-fsanitize=address,undefined,leak`: 启用 AddressSanitizer、UndefinedBehaviorSanitizer 和 LeakSanitizer。这些是强大的运行时诊断工具，用于捕获内存访问错误、未定义行为及内存泄漏。

3.  **项目结构与依赖**:
    * `project(AstraSim_Analytical)`: 声明项目名称。
    * `add_subdirectory(...)`: 此指令是组织项目的关键。它将 `AstraSim` 核心库、`Analytical` 网络后端和 `AstraSim_Analytical` 前端等多个子模块纳入构建过程。

4.  **用户自定义选项**:
    * `set(BUILDTARGET "all" CACHE STRING ...)`: 此行定义了一个名为 `BUILDTARGET` 的可缓存变量。这使得用户可以通过 `cmake -D` 命令从外部注入该变量的值。此变量随后会被子目录中的 `CMakeLists.txt` 文件用来实现条件编译。

# Build ns-3 Backend 
构建命令为 `./build/astra_ns3/build.sh -c`，他会执行该脚本里的 compile 函数
```bash{linenos=true}
function compile {
cd "${NS3_DIR}"
./ns3 configure --enable-mpi
./ns3 build AstraSimNetwork -j 12
cd "${SCRIPT_DIR:?}"
}
```
## `./ns3 configure --enable-mpi`
1. 参数解析 (`parse_args`): 脚本的 `argparse` 模块会识别出 `configure` 子命令和 `--enable-mpi` 选项。`--enable-mpi` 是一个预定义的"On-Off"选项，用于控制 MPI (Message Passing Interface) 分布式仿真功能的支持。
2. 进入配置步骤 (`configuration_step`): 由于检测到 configure 命令，脚本会调用 `configuration_step` 函数。
3. 调用 CMake (`configure_cmake`): `configuration_step` 函数内部会调用 `configure_cmake`. 这个函数是会动态地构建一个 cmake 命令。
    - 它会检测到 `--enable-mpi` 选项，并通过 `on_off_condition` 函数将其转换为 CMake 变量 `-DNS3_MPI=ON`.
    - 最终组装出的命令为为 `cmake -S . -B cmake-cache -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=default -DNS3_ASSERT=ON -DNS3_LOG=ON -DNS3_WARNINGS_AS_ERRORS=OFF -DNS3_MPI=ON --warn-uninitialized`
4. 执行配置: 脚本通过 `subprocess.run()` 执行这条 cmake 命令
## `./ns3 build AstraSimNetwork -j 12`
1. 参数解析 (`parse_args`): 脚本识别出 `build` 子命令，目标 `AstraSimNetwork`，以及并行任务数 `-j 12`. 前者会被存入 `args.build` 列表，后者会被存入 `args.jobs`.
2. 进入构建步骤 (`build_step`): 脚本检测到 `build` 命令，并调用 `build_step` 函数。
3. 调用 CMake 构建 (`cmake_build`): `build_step` 函数会遍历 `args.build` 列表中的所有目标。在这里，它会为 `AstraSimNetwork` 这个目标调用 `cmake_build` 函数。
    - cmake_build 函数会组装出一条 `cmake --build` 命令。
    - 将目标 AstraSimNetwork 转换为 `--target AstraSimNetwork`.
    - 将并行任务数 12 转换为 `-j 12`.
    - 最终组装出的命令为 `cmake --build cmake-cache --target AstraSimNetwork -j 12`.
# Error When Building ns-3
## call of overloaded ‘format(...)’ is ambiguous ❌
### 问题诊断 🩺

错误信息 `call of overloaded ‘format(...)’ is ambiguous` 的意思是，编译器在你的代码中遇到了一个名为 `format` 的函数调用，但它找到了多个同名的、并且参数类型都能匹配的 `format` 函数定义，导致编译器不知道该选择哪一个，因此产生了“歧义”（ambiguous）。

**这个歧义的来源是：**

1.  **`std::format` (来自 C++20 标准库)**: 你的项目很可能正在使用支持 C++20 或更高版本的现代编译器（如 GCC 11+）。C++20 标准库引入了一个新的格式化函数 `std::format`。
2.  **`fmt::format` (来自 {fmt} 库)**: `spdlog` 这个日志库是基于一个非常流行的第三方格式化库 `{fmt}` 构建的。这个库也提供了一个功能几乎完全相同的 `fmt::format` 函数。在 `spdlog` 的上下文中，它通常可以直接以 `format` 的形式被调用。

当你的代码（这里是 `spdlog_setup` 的一部分）简单地调用 `format(...)` 时，如果 C++20 的 `<format>` 头文件被包含，编译器就会同时看到 `std::format` 和 `spdlog` 内部的 `fmt::format`。由于两者都能处理字符串字面量 (`const char[]`) 和 `std::string`，编译器无法决定用哪个，从而报错。

---

### 关于 `using fmt::format;` 为何仍然无效的解释

原因是，除了常规的命名空间查找规则，C++ 还有一个更强大的规则叫做**参数依赖查找（Argument-Dependent Lookup, ADL）**，有时也被称为 Koenig 查找。

---

我们来梳理一下编译器在看到 `format(...)` 这行代码时的“思考过程”：

1. **在当前作用域查找**

    编译器看到了你的 `using fmt::format;` 声明。很好，它在当前作用域里找到了一个叫做 `format` 的函数（也就是 `fmt::format`）。这成为了**候选者 A**。

2. **参数依赖查找 (ADL) —— 问题的根源**
    
    接下来，编译器会检查 `format(...)` 函数的所有参数类型。在你的错误日志里，我们看到了 `const std::string&` 这样的参数。
    * ADL 规则规定：如果一个函数的参数是某个命名空间 `N` 下的类型（比如 `std::string` 是 `std` 命名空间下的），那么编译器**也必须**去那个命名空间 `N` (这里是 `std`) 里面去查找同名的函数。
    * 由于 `std::string` 是 `std` 命名空间的成员，ADL 规则被触发，编译器自动地去 `std` 命名空间里寻找名为 `format` 的函数。
    * 因为你使用了 C++20 编译器，它在 `std` 命名空间里成功找到了 `std::format`。这成为了**候选者 B**。

3.  **产生歧义**

    现在编译器陷入了困境。它手头有两个同样匹配的候选函数：
    * **候选者 A**: `fmt::format` (通过 `using` 声明找到)
    * **候选者 B**: `std::format` (通过 ADL 在参数的命名空间里找到)

    `using` 声明只是将一个名字引入当前作用域，它并**没有足够的“特权”**去压制一个通过 ADL 找到的同样优秀的候选者。因为两个函数都能完美处理你传入的参数，编译器无法做出选择，所以它只能放弃并报告“调用是模糊的 (ambiguous)”。

### 结论与最终解决方案 ✅

这个 C++ 的特性意味着，只要你的函数参数中包含了 `std` 命名空间里的类型（如 `std::string`, `std::vector` 等），ADL 就有可能被触发，从而把 `std` 里的函数（如 `std::format`, `std::to_string` 等）也拉入候选列表，造成意想不到的冲突。

因此，唯一能 100% 消除歧义、让编译器别无选择的方法，就是使用**显式的命名空间限定**：

```cpp{linenos=true}
// 这样做，是在直接告诉编译器：“别去猜了，我就是要调用 fmt 命名空间里的这个 format！”
// 这会完全绕过 ADL 和其他查找规则，直达目标。
fmt::format(...);
```
# Runing Arguments
执行仿真需要传递一些参数，命令模板如下
```bash{linenos=true}
{ASTRA_SIM_BIN} \
  --workload-configuration=${WORKLOAD_CONFIG} \
  --system-configuration=${SYSTEM_CONFIG} \
  --network-configuration=${NETWORK_CONFIG} \
  --remote-memory-configuration=${REMOTE_MEMORY_CONFIG}
```
## WORKLOAD_CONFIG

astra-sim 使用的是 Chakra (Execution Trace) 作为 workload 层的输入。将 chakra 作为 python package 安装后有几个命令通过 pyproject.toml 对应到 python函数。

{{< details title="Explanation of toml file">}}
`pyproject.toml` 是一个标准化的配置文件，用于定义 Python 项目的元数据、依赖关系以及构建和开发工具的配置。

---

1. `[build-system]` 构建系统配置，这部分定义了如何构建你的 Python 包。

* `**requires**`: 列出了构建项目本身所必需的包。这些是构建环境的依赖，而不是你代码运行时的依赖。
    * `setuptools`, `setuptools-grpc`: 表明此项目使用 `setuptools` 作为其构建工具，并需要 `setuptools-grpc` 插件。
* `**build-backend**`: 指定了构建工具中实际执行构建过程的 Python 对象（入口点）。
    * `setuptools.build_meta`: 这是 `setuptools` 提供的标准构建后端。

---

2. `[project]`：这部分包含了项目的基本信息，这些信息会展示在 PyPI (Python Package Index) 上。

* `**name**`: 包的名称，即 `pip install chakra` 中的 `chakra`。
* `**requires-python**`: 运行此包所需的最低 Python 版本，这里是 `3.7` 或更高。
* `**version**`: 当前包的版本号。
* `**readme**`: 指向一个文件，该文件的内容将作为项目在 PyPI 上的详细描述。
* `**license**`: 指向包含许可证信息的文件。
* `**authors**`：项目的作者信息。
* `**dependencies**`: **项目运行时的依赖项**。当用户 `pip install chakra` 时，这些包也会被一并安装。
    * `protobuf==5.*`: 需要版本为 5.x 的 `protobuf` 库。
    * `graphviz`, `networkx`, `pydot`: 其他标准的第三方库依赖。
    * `HolisticTraceAnalysis @ git+...`: 这是一个特殊的依赖。它直接从 GitHub 仓库的一个**特定 commit** (`d731cc...`) 来安装。这确保了项目依赖于一个稳定且不会意外变动的版本。

---

3. `[project.urls]`：项目相关链接，这些链接会显示在 PyPI 页面的侧边栏，为用户提供更多信息的入口。

* `**Homepage**`, `**Documentation**`, `**Repository**`: 分别指向项目主页、文档和代码仓库的 URL。

---

4. `[tool.setuptools]`：这部分是针对构建工具 `setuptools` 的详细配置。

* `**package-dir**`: 定义了 Python 包名与实际源代码目录之间的映射关系。
    * 例如，`"chakra.src.converter" = "src/converter"` 表示当用户 `import chakra.src.converter` 时，Python 会从 `src/converter/` 目录下寻找代码。这使得项目可以使用 `src` 布局。
* `**package-data**`: 指定需要包含在最终发布包中的非 Python 文件。
    * `"chakra.schema.protobuf" = ["et_def.proto"]`: 表示需要将 `et_def.proto` 这个文件打包到 `chakra.schema.protobuf` 这个包里。

---

5. `[project.scripts]`：这部分定义了在安装包时应创建的命令行工具。

* `**chakra_converter = "chakra.src.converter.converter:main"**`: 这行配置意味着，当用户安装此包后，他们可以在终端中直接运行 `chakra_converter` 命令。执行此命令时，系统会调用 `chakra.src.converter.converter` 模块中的 `main` 函数。

---

6. `[tool.ruff]`：这部分是用于配置 `Ruff` 高性能代码检查（Linter）和格式化（Formatter）工具。

* `**target-version**`, `**line-length**`, `**exclude**`: 基本配置，如目标 Python 版本、每行最大长度和要排除检查的文件。
* `**[tool.ruff.lint]**`: Linter 的具体配置。
    * `**select**`: 启用一系列代码规则集（例如 `D` 代表文档字符串 `pydocstyle`，`I` 代表导入排序 `isort`）。
    * `**ignore**`: 全局禁用的特定规则。注释中解释了忽略它们的原因（例如，规则冲突或待办事项）。
    * `**per-file-ignores**`: 针对特定文件或目录禁用规则。例如，`"**/tests/*" = ["D"]` 表示在所有测试文件中都禁用文档字符串检查。
* `**[tool.ruff.format]**`: 格式化器的配置，如使用空格作为缩进风格。

---

7. `[tool.pyright]`：这部分配置了 `Pyright`，一个由微软开发的静态类型检查工具。

* `**typeCheckingMode**`: 类型检查的严格程度，这里是 `basic`（基础模式）。
* `**exclude**`：在进行类型检查时要忽略的文件和目录。
* `**report...**`：关闭特定的错误或警告报告。

---

8. `[tool.vulture]`：这部分配置了 `Vulture`，一个用于发现项目中未使用（"死"）代码的工具。

* `**ignore_names**`: 让 Vulture 忽略某些特定的变量名或函数名，即使它们看起来未使用。
* `**min_confidence**`: 设置报告问题的最低置信度阈值。`100` 表示只有在 Vulture 100% 确定代码是无用的时候才会报告，这可以有效减少误报。
{{< /details >}}

```toml{linenos=true}
[project.scripts]
chakra_converter = "chakra.src.converter.converter:main"
chakra_generator = "chakra.src.generator.generator:main"
chakra_jsonizer = "chakra.src.jsonizer.jsonizer:main"
chakra_timeline_visualizer = "chakra.src.timeline_visualizer.timeline_visualizer:main"
chakra_trace_link = "chakra.src.trace_link.trace_link:main"
chakra_visualizer = "chakra.src.visualizer.visualizer:main"
```
### Generate Execution Trace

ASTRA-sim 的 ET 命名格式为 `{path prefix/trace name}.{npu_id}.et`. Chakra ET 的获取流程如下图所示[^1]。
1. Collect ET from PyTorch
    - PyTorch ET 负责 CPU 算子，并明确表示它们之间的依赖关系。
    - Kineto Trace 编码 GPU 算子及其开始和结束时间。
2. Merge Trace by `chkra_trace_link`：将它们合并为一个 PyTorch ET+. 该格式本质上遵循 PyTorch ET 的模式，但同时也编码了 GPU 操作符及其依赖关系。
3. Convert to Chakra ET by `chakra_converter`
![Overview of Trace Collection](https://private-user-images.githubusercontent.com/7621438/294028976-67228699-cec5-4a4d-b03e-e76647a80ce8.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NDk1NDQxNDUsIm5iZiI6MTc0OTU0Mzg0NSwicGF0aCI6Ii83NjIxNDM4LzI5NDAyODk3Ni02NzIyODY5OS1jZWM1LTRhNGQtYjAzZS1lNzY2NDdhODBjZTgucG5nP1gtQW16LUFsZ29yaXRobT1BV1M0LUhNQUMtU0hBMjU2JlgtQW16LUNyZWRlbnRpYWw9QUtJQVZDT0RZTFNBNTNQUUs0WkElMkYyMDI1MDYxMCUyRnVzLWVhc3QtMSUyRnMzJTJGYXdzNF9yZXF1ZXN0JlgtQW16LURhdGU9MjAyNTA2MTBUMDgyNDA1WiZYLUFtei1FeHBpcmVzPTMwMCZYLUFtei1TaWduYXR1cmU9OWE4NzAyMGQ0NWQ0MDA2MzIzMmY1MmNhYWU4YWUzNTJiNjI3OTAzZDk2ZDU3NDIwMWJhZTFlMjNjZDhjN2JmMyZYLUFtei1TaWduZWRIZWFkZXJzPWhvc3QifQ.-DDH2mackHVASqoCbmyvN2xl8vZemaa73OiLmBER1o0 "Overview of Trace Collection")

具体的教程和例子可以在 [Conversion Guide](https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces#3-from-raw-traces-to-chakra-a-step-by-step-conversion-guide) 和 [Practical Example](https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces#3-from-raw-traces-to-chakra-a-step-by-step-conversion-guide) 找到。

### Using ET Converter
可以将 astra-sim 1.0 的文本输入转换成 Chakra ET.
```bash{linenos=true}
cd ./extern/graph_frontend/chakra/
pip3 install .
chakra_converter Text \
    --input ../../../examples/text_converter/text_workloads/Resnet50_DataParallel.txt \
    --output ../../../examples/text_converter/text_workloads/Resnet50_DataParallel \
    --num-npus 8 \
    --num-passes 1
```
workload 文本格式要求如下，其中通信大小单位是字节，计算时间以周期数表示。
- 第一行：(DATA/HYBRID_TRANSFORMER/HYBRID_DLRM)
  - 该行指定训练循环的并行化类型。DATA 表示纯数据并行方法，HYBRID_TRANSFORMER 表示专为 Transformer DNN 网络设计的混合并行方法，而 HYBRID_DLRM 表示专为 DLRM DNN 网络优化的混合并行方法。
- 第二行：(int)
  - 该行表示 DNN 的层数。
- 后续行：每行描述一层。层的描述格式如下：
  - {(string: 层名称)
  - (int: 保留变量)
  - (int: 前向计算时间)
  - (ALLREDUCE/ALLGATHER/ALLTOALL: 前向通信类型)
  - (int: 前向通信大小)
  - (int: 输入梯度计算时间)
  - (ALLREDUCE/ALLGATHER/ALLTOALL: 输入梯度通信类型)
  - (int: 输入梯度通信大小)
  - (int: 权重梯度计算时间)
  - (ALLREDUCE/ALLGATHER/ALLTOALL: 权重梯度通信类型)
  - (int: 权重梯度通信大小)
  - (集合通信完成后，权重/输入/输出更新的延迟)}`

{{< notice note>}}
每一层的参数写要在同一行！！！
{{< /notice >}}

### Enable Communicator Groups
astra-sim 2.0 支持[通信组](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/communicators.html)。可以通过指定 `--comm-group-configuration` JSON 文件来指定，默认只有一个通信组。

```json
{
// The first/second communicator group, with ID 0/1, includes GPU IDs from 0-3/4-7. 
//   "0": [0, 1, 2, 3],
//   "1": [4, 5, 6, 7]
  "<communicator_group_id>" : [gpu_ids]
}
```
## SYSTEM_CONFIG

# System Layer

Workload 层会遍历 Chakra ET 中的节点，并为每个节点所指代的操作发出相应的命令。System 层接收这些命令，并将其转换为适合网络、计算或内存后端的格式，从而正确模拟操作。根据操作的类型，系统层的行为会有所不同，具体如下：

- 计算操作：向计算后端发出调用，以模拟操作的持续时间。
- 内存操作：  内存
- 通信操作：将集合通信分解为点对点的发送和接收消息，并向网络后端发出“发送”或“接收”调用，以模拟消息的传输过程。

## Collective Scheduler

![Collective Scheduler](https://astra-sim.github.io/astra-sim-docs/_images/system_overview_queue.svg "Collective Scheduler")

每个队列都有许多 `StreamBaseline` 对象 (图中右上角)，代表了整个集合通信的流程，`phase_to_go` 是一个用于表示这些阶段的队列，`my_current_phase` 是指向当前执行阶段的指针。

```cpp{linenos=true}
class StreamBaseline : public BaseStream {
  public:
    StreamBaseline(Sys* owner,
                   DataSet* dataset,
                   int stream_id,
                   std::list<CollectivePhase> phases_to_go,
                   int priority);
    // my_current_phase[CollectivePhase] is defined in BaseStream
    void init();
    void call(EventType event, CallData* data);
    void consume(RecvPacketEventHandlerData* message);
};
```
对于每个 stream `proceed_to_next_vnet_baseline` (astra-sim/system/Sys.cc) 用于推进通信阶段并且负责在队列之间移动 stream 对象。以下几种情况会调用该函数：
1. stream 第一次被移动出 ready_list 并且将被插入到 `active_streams`.
2. stream 完成了一个通信阶段并且等待下一个阶段。
3. stream 完成了所有的通信阶段。

(2-1) 到 (2-5) 描述了该函数的行为
1. 查看当前持有 stream 的队列: 从队列中删除 `StreamBaseline` 对象 (流的完成顺序可能与它们开始执行的顺序不同)。
2. 修改 `StreamBaseline` 对象: 已完成的集合通信阶段从 `phases_to_go` 中弹出，`my_current_phase` 现在指向下一个待执行的阶段。
3. 使用 `insert_stream` 将 `StreamBaseline` 对象插入到下一个队列中。
4. 调用函数 `notify_stream_removed` 函数查看前一个队列的头部。 `stream_pointer` 指向队列中第一个未运行的 stream (标记为蓝色)。该函数通过调用 `StreamBaseline::init()` 来启动 stream 的下一个阶段的执行。

5. 使用 `notify_stream_added` 触发新队列头部 stream 的通信阶段执行。

在其他情况下，`proceed_to_next_vnet_baseline` 会执行上述步骤的一部分。具体如下：

1. 刚从 `ready_list` 中移除：  
   `proceed_to_next..` 会初始化 stream (1-2)，将其插入到第一个队列中 (1-3)，并触发该队列头部的流执行。

2.  stream 完成：  
   该函数会从之前的队列中删除 stream (3-1)，并触发之前队列头部的 stream 执行。此外，`StreamBaseline` 对象会被删除，并调用 `notify_stream_finished`，以通知 `Sys` 对象 stream 已经结束 (3-6)

## Collective Implementation

![Overview of Collective Implementation](https://astra-sim.github.io/astra-sim-docs/_images/coll_implementation.svg "Overview of Collective Implementation")
模拟器将集体通信分解为发送和接收消息的方式有两种。目前最常用的方法是模拟器实现一组预定义的常见算法 (例如 Ring、DoubleBinary、HalvingDoubling 等)。这种“原生”实现逻辑位于模拟器的代码库中，允许用户快速探索一组预定义的算法。

自 2024 年 8 月以来，ASTRA-sim 支持了一种新的集合通信算法表示方式。System 层通过暴露一个集体 API，可以接收任意集体算法的定义。

这两种方法都是对 `CollectivePhase::Algorithm` 对象的实现，该对象是 System 层中的调度单元. [generate_collective_phase](https://github.com/astra-sim/astra-sim/blob/15a4334ade00fe1040fd00495cd13fd1ea5177e4/astra-sim/system/Sys.cc#L1037) 会根据不同的算法在创建 [CollectivePhase](https://github.com/astra-sim/astra-sim/blob/15a4334ade00fe1040fd00495cd13fd1ea5177e4/astra-sim/system/CollectivePhase.hh#L17) 的时候传入对应的 Algorithm.

### ASTRA-Sim Native Implementation

相关的实现都位于[该文件夹](https://github.com/astra-sim/astra-sim/tree/master/astra-sim/system/collective)下, naive 实现的限制是当需要模拟一个新的集合通信算法时算法，必须实现整个集合？随着不规则集合通信 (如 TACOS(Topology Aware CollectiveS), MSCCLang(基于 DSL)) 中工作的增加，快速模拟和迭代各种算法的需求变得越来越多。

### Chakra Based Arbitrary Definition Through Collective API

因此一个新的 AP来接受任何集合通信算法的定义，而不局限于预定义的规则通信模式。对于通信表示，使用 Chakra ET 模式作为单独的图。将集合通信算法表示为Chakra ET 模式中 COMM_SEND，COMM_RECV 节点的图。也就是说，System 层不是将集合通信分解为发送和接收消息，而是简单地遵循 Chakra 图中已经表示的分解。由于已经使用 Chakra ET 来表示 workload，使用 Chakra ET 来额外定义集合通信算法提供了一种轻松简单的方式来遍历整个图。

如上图所示当 workload 层发出 AllReduce 集体操作时，System 层不会运行模拟器代码库中已有的原生实现逻辑，而是会遍历通过 API 提供的 Chakra ET，该 ET 表示集合通信算法。需要注意 workload Chakra 图和集合通信算法的 Chakra 图是解耦的，并通过不同的输入点提供。最终，asytra-sim 模拟器会将通信节点替换为集体实现。

## Input Files for Collective API

### ASTRA-sim Native

```json
// ...
  "active-chunks-per-dimension": 1,
  "all-reduce-implementation": ["ring"],
  "all-gather-implementation": ["ring", "doubleBinaryTree"],
  "all-to-all-implementation": ["ring", "doubleBinaryTree", "halvingDoubling"],
// ...
```
`all-*-implementation` 指定了模拟器将如何将给定的集合通信分解为发送和接收消息。All-Gather 操作列表中的两个条目表示模拟器将按两个维度分解 ——第一个维度使用 Ring 算法，第二个维度使用 doubleBinaryTree 算法。

{{< quote >}}
Native Implementation Requires That the Dimensions for Collective Algorithms Are Same Across All Collectives.
{{< /quote >}}

{{< notice warning >}}
**Native 实现要求所有集体操作的维度必须相同**。换句话说，如果一个集合通信算法被定义为二维的，那么其他集合通信算法也必须是二维操作。上述只是一个例子。
{{< /notice>}}

### Collective API

```json
// ...
  "active-chunks-per-dimension": 1,
    "all-reduce-implementation-chakra": ["/app/hoti2024/demo5/inputs/custom_ring"],
// ...

```
需要注意这里要使用 `all-*-implementation-chakra`，而不是 `all-*-implementation`. 另外  Chakra ET 文件与传递给 workload 层的文件是不同的，每一项的值是 Chakra ET 文件的绝对路径，不包括最后的 `{rank}.et` 字符串 (类似于 Workload 层输入)。此外，即使有许多维度，列表也只接受一个值。这是因为跨维度通信的概念已经包含在 ET 中。

{{< github name="Collective API" link="https://github.com/astra-sim/collectiveapi" description="参考该仓库实现" >}}

# Network Backend
## Analytical Network Backend
Analytical Network 模拟器通过数学方程模拟所有网络行为。因此，该后端最适合于大规模分布式平台的建模和仿真。目前支持两种分析模式
- congestion_**unaware** analytical network simulator
- congestion_**aware** analytical network simulator

---

- T**Topology**

Analytical Network 支持三种拓扑结构: Ring, FullConnected, Switch. 并且可以堆叠来表示多维网络。

![Basic Network Building Block](https://astra-sim.github.io/astra-network-analytical-docs/_images/network-building-blocks.svg "Basic Network Building Block")

```yaml
topology: [ Ring, Switch ]  # 2D topology
topology: [ Ring, Ring, Ring ]  # 3D topology
```
![Example of 2D & 3D Topologies](https://astra-sim.github.io/astra-network-analytical-docs/_images/multidim-network-example.svg "Example of 2D & 3D Topologies")

---

- **NPUs Count**

指定了每个维度上的设备数目

```yaml
npus_count: [ 5 ]  # 5 NPUs
npus_count: [ 4, 2 ]  # 4 × 2 = 8 NPUs
npus_count: [ 4, 2, 2 ]  # 4 × 2 × 2 = 16 NPUs
```
![NPUs Count Example](https://astra-sim.github.io/astra-network-analytical-docs/_images/npus-count-example.svg "NPUs Count Example")


---

- **Bandwidth** & **Latency**

`latency` 定义了每条单向链路的延迟 (ns).
`bandwidth` 定义了每条单向链路的带宽 (GB/s).

{{< notice note >}}
$1 GB = 2^{30} B$ and $1 s = 10^9 ns$
{{< /notice >}}

## ns3 backend

下面是用 ns3 后端进行方针的一个执行命令。这里使用了 `--network-backend` 和 `--logical-topology` 这两个参数。需要说明的是，Analytical Backend 中仅使用了-`-network-backend` 参数，这是因为分析型后端的逻辑拓扑与物理拓扑是相同的，而 ns3 则允许我们将逻辑拓扑与物理拓扑分离。

```bash
   # {NS3_DIR} is the directory of the ns-3 backend. That is, '{ASTRA_SIM_ROOT_DIRECTORY}/extern/network_backend/ns-3'
    cd "${NS3_DIR}/build/scratch"
    ./ns3.42-AstraSimNetwork-default \
        --workload-configuration="${SCRIPT_DIR:?}"/../../extern/graph_frontend/chakra/one_comm_coll_node_allgather  \
        --system-configuration="${SCRIPT_DIR:?}"/../../inputs/system/Switch.json  \
        --network-configuration="../../../ns-3/scratch/config/config.txt"   \
        --remote-memory-configuration="${SCRIPT_DIR:?}"/../../inputs/remote_memory/analytical/no_memory_expansion.json \
        --logical-topology-configuration="${SCRIPT_DIR:?}"/../../inputs/network/ns3/sample_8nodes_1D.json   \
        --comm-group-configuration=\"empty\"
```





---
[^1]: [Overview of Trace Collection](https://github.com/mlcommons/chakra/wiki/Chakra-Execution-Trace-Collection-%E2%80%90-A-Comprehensive-Guide-on-Merging-PyTorch-and-Kineto-Traces#2-overview-of-trace-collection-and-simulation-methodology)