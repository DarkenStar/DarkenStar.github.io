---
title: "Tx8read"
date: 2025-06-11T10:21:42+08:00
lastmod: 2025-06-11T10:21:42+08:00
author: ["WITHER"]

categories:
- Work

tags:
- tx8-script

keywords:
- bsh

description: "tx8 regression" # 文章描述，与搜索优化相关
summary: "tx8 regression" # 文章简单描述，会展示在主页
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

# TestGraphCompute
```cpp{linenos=true}
int main(int argc, char **argv) {
    // 1. 初始化与命令行参数处理
    Timer timer("TestGraphCompute");
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    // 解析命令行参数
    cl::ParseCommandLineOptions(argc, argv, "tx8be compiler\n");

    // 2. 初始化 MLIR 模块和上下文
    mlir::OwningOpRef<mlir::ModuleOp> module;
    mlir::MLIRContext context;
    // 定义一个正则表达式，用于从命令行选项中提取 codegen_path 参数
    std::regex pattern("codegen_path=([a-zA-Z0-9_]+)");  
    std::smatch matches;

    std::string cachePath = "codegen";  // 默认文件夹名字
    if (std::regex_search(optionstr, matches, pattern)) {  // 寻找命令行选项中是否指定 codegen_path
        cachePath = matches[1];

    // 3. 加载 MLIR 模块
    // 如果 cache 为 2 或 4，则从缓存路径加载模块；否则，使用默认的 gModelFil
    gModelFile = (cache == 2 || cache == 4) ? cachePath + "/cache.mlir" : gModelFile;
    if (int error = getMLIRFromFile(module, context, gModelFile)) {
        return error;
    }

    // 4. 配置模块
    auto mconfig = getModuleConfig(module.get());
    if (optionstr.size() > 0) {
        mconfig.option += optionstr;
    }
    mconfig.constCache = cache;

    updateModuleConfig(module, context, mconfig);
    mconfig = getModuleConfig(module.get());
    showModuleConfig(mconfig);

    // 5. 处理多卡信息
    json_info_multi_card_t *multi_card_jinfo = nullptr;
    multi_card_jinfo = (cache == 2 || cache == 4) ? get_multi_card_info_from_file(cachePath + "/model_info.json")
                                                : parseMultiCardModuleInfo(module, context);
    if (0) {
        dumpIR(module.get(), true);
    }

    // 6. 读取参考路径
    if (!fast_codegen) {  // NOT fast_codegen
        std::vector<std::string> in_files = parseStringArgs(gInputBin, std::string(","));
        std::vector<std::string> out_files = parseStringArgs(gOutputBin, std::string(","));
        // 定义一个 lambda 函数，用于从文件中读取参考文件路径。
        auto getRefFiles = [] (std::string gFile, std::vector<std::string> &files) {
            if (gFile.size() == 0) {
                return 0;
            }
            if (!llvm::sys::fs::exists(gFile)) {
                return 0;
            }
            std::ifstream gf(gFile);
            std::string text((std::istreambuf_iterator<char>(gf)), (std::istreambuf_iterator<char>()));
            files = parseStringArgs(text, std::string(","));
            return 0;
        };
        if ((gInputBin.size() == 0) && (gInputFile.size() != 0)) {
            getRefFiles(gInputFile, in_files);
            getRefFiles(gOutputFile, out_files);
        }
    }
    
    // 7. computeGolden
    if (cache < 2) {
    for (int32_t i = 0; i < mconfig.tile.chip_num; i++) {  // 遍历芯片数量，创建对应的目录结构
        // 构造并创建创建目 codegen/node_0_0/chip0/agent/data 
        std::string path = "codegen/node_0_0/chip";  
        path += std::to_string(i) + "/agent/data";
        createDir(path);
        }
        // 调用 computeGolden 函数，计算参考输出保存到 codegenPath
        computeGolden(module, multi_card_jinfo, in_files, out_files, mconfig.codegenPath);
    }

    // 调用 moduleCompileCodegen 函数，对 MLIR 模块进行编译和代码生成
    int32_t ret = moduleCompileCodegen(module, context);
    ASSERT(ret == true);
    
    // 9. 获取内存大小
    // 从模块中获取立即数 (Immediate) 和常量参数的 DDR 大小
    uint64_t imm_size = module.get()->hasAttr(tx8be::ModuleAttr::ImmDdrSize) ?
    module.get()->getAttrOfType<mlir::IntegerAttr>(tx8be::ModuleAttr::ImmDdrSize).getInt() : 2147483648;  
    uint64_t params_size = module.get()->hasAttr(tx8be::ModuleAttr::ConstDdrSize) ?
        module.get()->getAttrOfType<mlir::IntegerAttr>(tx8be::ModuleAttr::ConstDdrSize).getInt() : 0;
    
    // 10. 更新每个芯片的内存大小信息
    for (int32_t i = 0; i < mconfig.tile.chip_num; i++) {
        multi_card_jinfo->chip_infos[i]->imm_size = imm_size;
        multi_card_jinfo->chip_infos[i]->params_size.emplace_back(params_size);
    }

    // 11. 保存多卡模型信息
    std::vector<int32_t> chipIds;
    if (module.get()->hasAttr(tx8be::ModuleAttr::ChipIds)) {  // 获取芯片 ID 
        mlir::ArrayAttr chipIdsAttr = module.get()->getAttrOfType<mlir::ArrayAttr>(tx8be::ModuleAttr::ChipIds);
        for (int i = 0; i < chipIdsAttr.size(); i++) {
            chipIds.push_back(chipIdsAttr[i].cast<mlir::IntegerAttr>().getInt());
        }
    }
    // 多卡模型文件保存到 codegenPath 路径下
    saveMultiCardModelJson(multi_card_jinfo, mconfig.codegenPath, chipIds);
    // uint64_t ddrSize = getModelDDRSize(multi_card_jinfo);
    return 0;
    }
}
```
## computeGolden

输入参数的来源：
- `module`：在 main 文件中，通过 `getMLIRFromFile` 函数从文件中加载 MLIR 模块
- `multi_card_jinfo`：在 main 文件中，通过 `get_multi_card_info_from_file` 或 `parseMultiCardModuleInfo` 从 JSON 文件或 MLIR 模块中提取多卡信息。
- `in_files` 和 `out_files`：在 main 文件中，通过 `parseStringArgs` 或 `getRefFiles` 解析输入和输出文件路径。
- `mconfig.codegenPath`：在 main 文件中，通过命令行选项或默认值设置代码生成路径，并传递给 computeGolden。

computeGolden 函数生成的数据（输入和输出的二进制文件）将保存到指定路径 mconfig.codegenPath.

```cpp{linenos=true}
void computeGolden(mlir::OwningOpRef<mlir::ModuleOp> &module,
                   json_info_multi_card_t *multi_card_jinfo,
                   std::vector<std::string> &inFiles,
                   std::vector<std::string> &outFiles,
                   std::string file_path) {
    // 定义形状类型，用于存储多维张量的形状信息
    using ShapeType = std::vector<std::vector<int64_t>>;

    // 用于存储多芯片的输入和输出数据
    std::vector<std::vector<int8_t *>> multiInputDdata;
    std::vector<std::vector<int8_t *>> multiOutputData;

    std::vector<std::thread> threads;

    uint32_t chip_num = multi_card_jinfo->chip_num;  // 获取芯片数量
    auto tile_info = get_tileinfo(module.get());  // 从 MLIR 模块中提取 tile 信息
    std::vector<ShapeType> outShapes(chip_num);  // 存储每个芯片的输出形状信息

    for (int i = 0; i < chip_num; i++) {
        chip_info_t *chip_info = multi_card_jinfo->chip_infos[i];

        // 分配当前芯片的输入和输出数据指针数组
        std::vector<int8_t *> input_data(chip_info->input_num);
        std::vector<int8_t *> output_data(chip_info->output_num);

        // 用于 OneDNN 计算的输入和输出缓冲区
        std::vector<char *> computeInputs;
        std::vector<char *> computeOutputs;

        // 当前芯片的输入和输出文件路径
        std::vector<std::string> chipInFiles;
        std::vector<std::string> chipOutFiles;

        parseInOutfile(inFiles, chipInFiles, i, chip_num, chip_info->input_num);
        parseInOutfile(outFiles, chipOutFiles, i, chip_num, chip_info->output_num);

        // 生成当前芯片的输入输出数据
        genInputs4SingleChip(computeInputs, input_data, chip_info, chipInFiles);
        genOutputs4SingleChip(computeOutputs, output_data, chip_info, chipOutFiles);

        // 如果输入文件为空，则生成随机输入数据并校正
        if (chipInFiles.size() == 0) {
            updateSpecialInputData(computeModuleRef, computeInputs, input_data, chip_info);
        }

        multiInputDdata.push_back(input_data);
        multiOutputData.push_back(output_data);

        if (outFiles.size() == 0) {
            threads.emplace_back(moduleComputeInterface, std::ref(computeModuleRef), std::ref(outShapes[i]), computeInputs, computeOutputs, i);
        }
    }

    // 等待所有线程完成计算
    for (auto &thread : threads) {
        thread.join();
    }

    // 遍历每个芯片，保存输入和输出数据
    for (int32_t i = 0; i < chip_num; i++) {
        chip_info_t *chip_info = multi_card_jinfo->chip_infos[i];

        uint32_t node_id = get_node_id(i, tile_info);
        int32_t relative_chip_id = get_relative_chip_id(i, tile_info);

        // 构造当前芯片的数据保存路径  file_path/node_x_y/chip_z/agent/data
        std::string data_path = file_path + "/node_" + std::to_string(node_id / tile_info.node_y) + "_" + std::to_string(node_id % tile_info.node_y) + "/chip" + std::to_string(relative_chip_id) + "/agent/data";
        createDir(data_path);

        // bin格式保存当前芯片的输入数据  
        for (uint32_t j = 0; j < chip_info->input_num; j++) {
            inout_tensor_info_t *tensor = chip_info->input[j];
            saveInOutTensor(tensor->shape, tensor->dim, tensor->layout, tensor->dtype,
                           data_path + "/in" + std::to_string(j) + ".bin", multiInputDdata[i][j]);
        }

        // bin格式保存当前芯片的输出数据  out_j_ref.bin
        for (uint32_t j = 0; j < chip_info->output_num; j++) {
            inout_tensor_info_t *tensor = chip_info->output[j];
            int32_t tensorShape[MAX_SHAPE_DIM];

            // 根据 outShapes 或原始形状计算输出张量的形状
            for (int idx = 0; idx < tensor->dim; ++idx) {
                if (!outShapes[i].empty()) {
                    tensorShape[idx] = outShapes[i][j][idx];
                } else {
                    tensorShape[idx] = tensor->shape[idx];
                }
            }

            // 保存输出数据
            saveInOutTensor(tensorShape, tensor->dim, tensor->layout, tensor->dtype,
                           data_path + "/out" + std::to_string(j) + "_ref.bin", multiOutputData[i][j]);
        }

        // 释放当前芯片的输入和输出数据内存
        for (uint32_t j = 0; j < chip_info->input_num; j++) {
            free(multiInputDdata[i][j]);
        }
        for (uint32_t j = 0; j < chip_info->output_num; j++) {
            free(multiOutputData[i][j]);
        }
    }
}
```

# run_code_gen_layer

主要用于运行代码生成 (codegen) 相关的任务，以下是函数的详细功能解释：

1. 解析参数：
  - 接受至少两个参数：$1 是可执行文件的名称，$2 是种子文件 (seed file) 
  - 如果有更多参数 ($# > 2)，则将额外参数存储为配置参数 (config_params) 
  - 从 config_params 中提取 `codegen_path` (代码生成输出路径) ，如果未指定则使用默认值 "codegen"
2. 切换工作目录：
  - 切换到 `${BEMLIR_PROJECT_ROOT}/build/bin` 目录。
  - 删除旧的 `codegen_path` 目录，确保环境干净。
3. 执行可执行文件：
  - 使用 `${layer_cmd}` (即 ./$1) 运行指定的可执行文件，传入种子文件和配置参数。
  - 检查返回值，如果失败 `(ret != 0)`，则恢复目录并返回错误。
4. 处理生成的代码：
  - 根据参数中的 `chip_num` 或 `static_shape` 判断 `host_type`.
  - 调用 `get_codegen_file` 处理生成的代码文件。
5. 运行 cmodel 测试:
  - 根据参数中的 `fast_codegen` 或 `not_run` 设置 cmp_flag.
  - 调用 `run_on_cmodel` 在 cmodel 上运行生成的代码。
  - 检查返回值，失败则返回错误。


```bash
# 定义 run_codegen_layer 函数，用于运行代码生成层测试流程
function run_codegen_layer() {
    # 1. 打印开始时间，用于调试和性能追踪
    echo -n "time==>>run_codegen_layer-start   "; date;

    # 2. 函数参数说明
    # $1: 可执行文件名称
    # $2: 种子文件 (seed file）
    layer_cmd="./$1"  # 在当前目录下执行的可执行文件路径
    seed_file=$2      # 种子文件或配置文件
    config_params=""  # 配置参数，默认为空

    # 默认代码生成输出路径
    codegen_path="codegen"

    # 3. 检查是否有超过2个参数
    if [ $# -gt 2 ]; then
        # 提取除前两个参数外的所有参数作为配置参数
        config_params=$*
        config_params=${config_params#*}  # 移除第一个参数 (可执行文件）
        config_params=${config_params#*}  # 移除第二个参数 (种子文件）
    fi

    # 4. 如果配置参数中包含 --codegen_path，提取其值作为代码生成路径
    if [[ ${config_params} == *"--codegen_path="* ]]; then
        # 提取 --codegen_path= 后面的值
        codegen_path=${config_params#*codegen_path=}
        # 移除可能存在的引号或其他字符
        codegen_path=${codegen_path%%\"*}
        codegen_path=${codegen_path-*}
        codegen_path=${codegen_path*}
    fi

    # 5. 切换到 build/bin 目录执行命令
    pushd ${BEMLIR_PROJECT_ROOT}/build/bin
        # 删除旧的 codegen_path 目录，确保环境干净
        rm -rf ${codegen_path}
        # 执行层命令，传入种子文件和配置参数
        ${layer_cmd} ${seed_file} ${config_params}
        # 捕获命令的返回值
        ret=$?  
        # 如果命令执行失败 (返回码非0），恢复目录并返回错误
        if [[ ${ret} -ne 0 ]]; then
            popd
            echo ${ret}
            return ${ret}
        fi
    popd  # 恢复原始目录

    # 6. 打印代码生成完成的时间
    echo -n "time==>>run_codegen_layer-codegen=== "; date;

    # 7. 根据参数判断主机类型
    host_type=0
    # 如果参数中包含 chip_num 或 static_shape，则将 host_type 设为 1
    if [[ $* == *"chip_num"* ]] || [[ $* == *"static_shape"* ]]; then
        host_type=1
    fi

    # 8. 调用 get_codegen_file 处理生成的代码文件
    get_codegen_file ${codegen_path} ${host_type}
    # 捕获返回值
    ret=$?
    # 如果处理失败，恢复目录并返回错误
    if [[ ${ret} -ne 0 ]]; then
        popd
        echo ${ret}
        return ${ret}
    fi

    # 9. 初始化比较标志
    cmp_flag=""
    # 如果参数中包含 fast_codegen 或 not_run，则设置 cmp_flag 为 "not_run"
    if [[ $* == *"fast_codegen"* ]] || [[ $* == *"not_run"* ]]; then
        cmp_flag="not_run"
    fi

    # 10. 在 cmodel 上运行生成的代码，传入比较标志
    run_on_cmodel ${codegen_path} ${cmp_flag}
    # 捕获返回值
    ret=$?
    # 如果运行失败，恢复目录并返回错误
    if [[ ${ret} -ne 0 ]]; then
        popd
        echo ${ret}
        return ${ret}
    fi

    # 11. 打印结束时间
    echo -n "time==>>run codegen layer-end===   "; date;
}
```

## get_codegen_file

`get_codegen_file` 用于整理代码生成的结果 (位于 `${BEMLIR_PROJECT_ROOT}/build/bin/${codegen_case}`)，为每个节点生成版本信息 (version.txt)，并将生成的文件复制到测试目录 (`${BEMLIR_PROJECT_ROOT}/external/tx8be-oplib/tests/test_codegen`)，最后调用 `get_codegen_host` 完成主机相关处理。

```bash
# Function to process and organize generated codegen files
function get_codegen_file() {
    # Print all input arguments for debugging
    echo "$*"

    # Assign first argument as the codegen case name or path
    codegen_case=$1
    # Second argument: 0 for host thread mode, 1 for host stream mode
    # Note: $2 is passed to get_codegen_host

    # Change to the codegen case directory under build/bin
    pushd "${BEMLIR_PROJECT_ROOT}/build/bin/${codegen_case}"
        # Find node directories matching node_[0-9]+_[0-9] pattern (e.g., node_123_4)
        node_dirs=$(find . -maxdepth 1 -type d -regex '.*/node_[0-9]+_[0-9]' -exec basename {} \;)
        # Iterate through each node directory
        for dir in $node_dirs; do
            # Check if libTX8MLIRTransforms.a exists to determine version type
            if [ ! -e "${BEMLIR_PROJECT_ROOT}/lib/libTX8MLIRTransforms.a" ]; then
                # Write 'tx8be-mlir' to version.txt if library is absent
                echo -e "tx8be-mlir" > "${dir}/version.txt"
            else
                # Write 'tx8be-mlir-sdk' to version.txt if library is present
                echo -e "tx8be-mlir-sdk" > "${dir}/version.txt"
            fi
            # Append git status to version.txt to record repository state
            git status --porcelain >> "${dir}/version.txt"
            # Append last two git commits to version.txt for version history
            git log -2 >> "${dir}/version.txt"
        done
    popd

    # Change to the test_codegen directory
    pushd "${BEMLIR_PROJECT_ROOT}/external/tx8be-oplib/tests/test_codegen"
        # Remove existing codegen_case directory to ensure a clean state
        rm -rf "${codegen_case}"
        # Copy the codegen_case directory from build/bin
        cp -r "${BEMLIR_PROJECT_ROOT}/build/bin/${codegen_case}" .
    popd

    # Call get_codegen_host to process host-related tasks
    get_codegen_host "${codegen_case}" "$2"
}
```

## get_codegen_host

`get_codegen_host ` 用于为 host 环境准备代码生成用例的测试文件。它在指定的测试用例目录中处理 node & chip 相关的文件，复制必要的配置文件、源代码和构建脚本，并根据 host_type 选择不同的主机实现文件 host_thread.cpp 或 host_stream.cpp.

1. 函数输入参数：

- `$1 (codegen_case)`: 代码生成用例的名称或路径，通常是一个目录 (例如 codegen0 或 codegen1) ，表示测试用例的根目录。
- `$2 (host_type)`: 主机执行模式，0: host_thread.cpp，1: host_stream.cpp.

2. 切换到 `${{OPLIB_PROJECT_ROOT}}/tests/test_codegen/${codegen_case}` 目录:

- 使用 find 命令查找符合 node_[0-9]+_[0-9] 模式 (例如 node_123_4) 的子目录，表示代码生成中的节点。
- 对每个 node_dir 追加版本信息和复制相关文件。

3. 处理 chip 目录:

- 在每个节点目录下，查找符合 ` chip[0-9]+` 模式 (例如 chip0, chip1) 的子目录
- 为每个 dir 复制 Makefile_tile 到 `./${node_dir}/${dir}/Makefile`. 在 `./${node_dir}/${dir}/` 下创建 16 个子目录 (tiles0 - tiles15)，并为每个子目录复制 Makefile_main 到 t `iles$i/Makefile`

4. 根据 `host_type` 复制 host_thread.cpp 或 host_stream.cpp 到当前目录的 host.cpp.
5. 复制 CMakeLists.txt 和 Makefile 到当前目录。

```bash
# Function to prepare host-related files for a codegen test case
function get_codegen_host() {
    # Print all input arguments for debugging
    echo "$*"

    # Assign first argument as the codegen case name or path
    codegen_case=$1
    # Assign second argument as host type (0: thread mode, 1: stream mode)
    host_type=$2
    # Define relative path for test_codegen directory
    oplib_path="tests/test_codegen"

    # Change to the test_codegen directory for the codegen case
    pushd "${{OPLIB_PROJECT_ROOT}}/tests/test_codegen/${codegen_case}"
        # Find node directories matching node_[0-9]+_[0-9] pattern (e.g., node_123_4)
        node_dirs=$(find . -maxdepth 1 -type d -regex '.*/node_[0-9]+_[0-9]' -exec basename {} \;)
        for node_dir in $node_dirs; do  # # Iterate through each node directory
            # Append oplib version info to version.txt
            echo -e "\n\ntx8be-oplib:" >> "${node_dir}/version.txt"
            # Append git status to version.txt to record repository state
            git status --porcelain >> "${node_dir}/version.txt"
            # Write last two git commits to version.txt for version history
            git log -2 >> "${node_dir}/version.txt"

            # Copy all stream-related files to node directory
            cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/stream*" "./${node_dir}/"
            # Copy CMakeLists_chip.txt as CMakeLists.txt for node
            cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/CMakeLists_chip.txt" "./${node_dir}/CMakeLists.txt"
            # Copy main_kcore.c to node directory
            cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/main_kcore.c" "./${node_dir}/"
            # Copy Makefile_chip as Makefile for node
            cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/Makefile_chip" "./${node_dir}/Makefile"

            # Find chip directories matching chip[0-9]+ pattern (e.g., chip0, chip1)
            chip_dirs=$(find "./${node_dir}" -maxdepth 1 -type d -regex '.*/chip[0-9]+' -exec basename {} \;)
            # Iterate through each chip directory
            for dir in $chip_dirs; do
                # Copy Makefile_tile as Makefile for chip
                cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/Makefile_tile" "./${node_dir}/${dir}/Makefile"
                # Create Makefiles for 16 tiles (tiles0 to tiles15)
                for ((i=0; i<16; i++)); do
                    dst_file="./${node_dir}/${dir}/tiles${i}/Makefile"
                    # Copy Makefile_main to each tile's Makefile
                    cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/Makefile_main" "$dst_file"
                done
            done
        done

        # Copy host implementation based on host_type
        if [ $host_type -eq 0 ]; then
            # Use thread-based host implementation
            cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/host_thread.cpp" "host.cpp"
        else
            # Use stream-based host implementation
            # Note: Fixed typo '$t{OPLIB_PROJECT_ROOT}' to '${{OPLIB_PROJECT_ROOT}}'
            cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/host_stream.cpp" "host.cpp"
        fi

        # Copy top-level CMakeLists.txt for test case
        cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/CMakeLists.txt" .
        # Copy top-level Makefile for test case
        cp "${{OPLIB_PROJECT_ROOT}}/tools/codegen/Makefile" .
    # Restore original directory
    popd
}
```

## run_on_cmodel

`run_on_cmodel` 用于在指定的测试用例目录中运行 cmodel 仿真任务。函数的主要功能包括环境设置、构建、执行仿真脚本或程序，并处理错误。以下是详细的功能说明：

1. 函数输入参数：

- $1 (case_name): 来自 `run_codegen_layer` 的 `codegen_path`，可能附加 `host_type`.
- $2 (run_flag): 运行标志，来自 `run_codegen_layer` 的 `cmp_flag` 用于控制仿真执行的方式 (例如是否运行或运行模式) 。

2. 切换工作目录并执行:

- 切换到测试用例目录 `${{OPLIB_PROJECT_ROOT}}/tests/test_codegen/${case_name}`
- 运行 `cmake .. -DUSING_RISCV=OFF`，配置构建系统，禁用 RISCV 支持。
- 运行 `make -j` 并动态设置并行任务数 (基于 CPU 核心数，`cat /proc/stat | grep cpu[0-9] -c`)

3. 仿真执行: 根据参数运行仿真脚本 (host_sim.sh) 或 host_sim.

```bash
# Function to run a cmodel simulation for a given test case
function run_on_cmodel() {
    # Assign first argument as the test case name
    case_name=$1
    # Assign second argument as the run flag (controls execution mode)
    run_flag=$2

    # Check if case_name is empty
    if [ -z "$case_name" ]; then
        echo "Error: case_name is empty"
        return 1
    fi

    # Check if the test case directory exists
    if [ ! -d "${{OPLIB_PROJECT_ROOT}}/tests/test_codegen/${case_name}" ]; then
        echo "Can not find ${case_name}"
        return 1
    fi

    # Change to the test case directory
    pushd "${{OPLIB_PROJECT_ROOT}}/tests/test_codegen/${case_name}"  # FIXED DIR
        rm -rf build
        mkdir build
        cd build
        # Run cmake to configure the build, disabling RISCV support
        cmake .. -DUSING_RISCV=OFF
        # Run make with parallel jobs based on CPU core count
        make -j$(cat /proc/stat | grep cpu[0-9] -c)
        ret=$?  # Capture the return code
        # If make fails, restore directory, print error, and exit
        if [[ $ret -ne 0 ]]; then
            popd
            echo $ret
            return $ret
        fi

        # Check if run_flag is empty
        if [ -z "$run_flag" ]; then
            if [ -e ../host_sim.sh ]; then  # Check if host_sim.sh exists in the parent directory
                cp ../host_sim.sh .
                sh ./host_sim.sh
                ret=$?  # Capture the return code
                # If script fails, restore directory, print error, and exit
                if [[ $ret -ne 0 ]]; then  
                    popd
                    echo $ret
                    return $ret
                fi
            else
                ./host_sim ../  # Run host_sim with parent directory as argument
                ret=$?  # Capture the return code
                # If host_sim fails, restore directory, print error, and exit
                if [[ $ret -ne 0 ]]; then
                    popd
                    echo $ret
                    return $ret
                fi
            fi
        # Check if run_flag is "0" or "1"
        elif [[ $run_flag == "0" ]] || [[ $run_flag == "1" ]]; then
            # Run host_sim with parent directory and run_flag
            ./host_sim ../ "$run_flag"
            ret=$?  # Capture the return code
            # If host_sim fails, restore directory, print error, and exit
            if [[ $ret -ne 0 ]]; then
                popd
                echo $ret
                return $ret
            fi
        fi
    popd

    echo "${FUNCNAME[0]} $* passed"
}
```

# run_codegen_case_soc_rtt

run_codegen_case_soc_rtt 位于 `tx8-oplib/scripts/regression.sh`，函数用于在 SOC 环境下运行 RTT (Real-Time Transfer) 测试。其主要流程如下：

1. 初始化和参数获取：
  - 函数从命令行参数中获取 `case_name`, `copy_option`, 和 `multi_graph_enable`.
  - 检查 `case_name` 是否为空，如果为空则输出错误信息并返回 1.
2. 环境设置和目录导航：
  - 将工作目录切换到 `${OPLIB_PROJECT_ROOT}/tests/test_codegen/${case_name}`. 如果目录不存在，则输出错误信息并返回 1。
3. 构建和配置：
  - 执行 `rm -rf ${case_name}_build` 清理之前的构建文件。
  - 根据 `multi_graph_enable` 设置 `CONFIG_ARGS`，如果启用多图则设置为 "-DMULTI_GRAPH=1"，否则为空。
  - 调用 cmake 命令生成构建文件，指定构建目录为 `${case_name}_build`，并根据 `copy_option` 设置 `COPY_RTT_FLAG`.
  - 执行 make 命令进行实际构建，目标包括 all 和 chip_out.
4. 错误处理和退出：
  - 每次关键步骤执行后，检查返回状态 `$ret`，如果非 0，则弹出目录并返回错误码。
  - 构建成功后输出 `${FUNCNAME[0]} "passed"` 表示通过。
5. 清理和返回: 函数结束时弹出目录，恢复原始工作目录。

```bash
function run_codegen_case_soc_rtt() {
    echo "${FUNCNAME[0]} 'start'"  # 输出函数名和"start"表示开始

    case_name=$1                  # 获取用例名称
    copy_option=$2                 # 获取复制选项
    multi_graph_enable=$3          # 获取多图启用标志

    if [ -z "$case_name" ]; then   # 如果用例名称为空
        echo "case_name($case_name) not found "  # 输出错误信息
        return 1                   # 返回错误码 1
    fi

    case_dir=${OPLIB_PROJECT_ROOT}/tests/test_codegen/${case_name}  # 设置用例目录
    pushd ${case_dir}              # 切换到用例目录

    rm -rf ${case_name}_build      # 清理之前的构建文件
    ret=$?; if [ [ $ret -ne 0 ]]; then popd; echo $ret; return $ret; fi  # 检查清理是否成功

    if [ -z "$multi_graph_enable" ]; then  # 如果多图启用标志为空
        CONFIG_ARGS=""                 # 配置参数为空
    else                                 # 否则
        CONFIG_ARGS="-DMULTI_GRAPH=1"  # 设置多图配置参数
    fi

    cmake -B "${case_name}_build" -DUSING_RISCV=ON -TX8FW_BASE=${OPLIB_PROJECT_ROOT}/release/riscv/tx8-yoc-rt-thread-smp ${CONFIG_ARGS} ; ret=$?; if [ [ $ret -ne 0 ]]; then popd; echo $ret; return $ret; fi  # 生成构建文件

    make -j -C "${case_name}_build" --target all chip_out ; ret=$?; if [ [ $ret -ne 0 ]]; then popd; echo $ret; return $ret; fi  # 执行构建

    popd                          # 恢复到原始目录
    echo "${FUNCNAME[0]} 'passed'" # 输出函数名和"passed"表示通过
}
```
## export_tx8fw_to_env 
`export_tx8fw_to_env` 函数的主要目的是设置与 TX8FW 相关的环境变量，以便后续构建或运行时使用。以下是其流程：

1. 设置 SDK 路径：
  - 定义 TX8FW 的 SDK 路径 `soc_sdk_path` 为 `${OPLIB_PROJECT_ROOT}/3rd_party/tx8-yoc-rt-thread-smp`.
2. 检查路径是否存在:
  - 检查路径 `${soc_sdk_path}/tool/tx8fw-xuantie-sdk` 是否存在。如果不存在，打印错误信息并退出，状态码为 1.
3. 导出环境变量: 打印并设置以下环境变量
  - TX8FW_SDK_INSTALL_DIR：指向 ${soc_sdk_path}/tool/tx8fw-xuantie-sdk。
  - TX8FW_TOOLCHAIN_VARIANT：设置为 cross-compile。
4. 清理目录: 使用 popd 命令恢复到之前的目录.

```bash
function export_tx8fw_to_env() {
    soc_sdk_path=${OPLIB_PROJECT_ROOT}/3rd_party/tx8-yoc-rt-thread-smp  # 设置 TX8FW SDK 路径

    pushd ${soc_sdk_path}/tool/tx8fw-xuantie-sdk  # 切换到 TX8FW SDK 工具目录

    if [ ! -d "xuantie-900-gcc-elf-newlib-x86_64-V2.8.0" ]; then  # 检查指定 SDK 目录是否存在
        echo "${soc_sdk_path}/tool/tx8fw-xuantie-sdk didn't exist"  # 如果不存在，打印错误信息
        exit 1  # 退出并返回状态码 1
    fi

    echo "export TX8FW_SDK_INSTALL_DIR=${soc_sdk_path}/tool/tx8fw-xuantie-sdk"  # 打印并设置 TX8FW SDK 安装目录环境变量
    export TX8FW_SDK_INSTALL_DIR=${soc_sdk_path}/tool/tx8fw-xuantie-sdk  # 导出 TX8FW SDK 安装目录环境变量
    export TX8FW_TOOLCHAIN_VARIANT=cross-compile  # 导出工具链变体为 cross-compile

    popd  # 恢复到之前的目录
}
```
## build_oplib_with_soc 函数
`build_oplib_with_soc` 函数用于构建 OPLib 并结合特定 SoC 配置。以下是其流程：

打印项目根目录：
打印 OPLIB_PROJECT_ROOT 环境变量，用于调试或日志记录。
1. 切换目录和初始化：
  - 使用 pushd 切换到 `OPLIB_PROJECT_ROOT` 目录。
  - 定义变量 `rm=rf build`, `mkdir=build` 和 `cd=build`，这些变量实际上是模拟命令（rm -rf build、mkdir build 和 cd build）。
2. 设置复制标志：
  - 检查 `$1` (即 `copy_option`) 是否为 "NOT_COPY"，如果是，则设置 `COPY_RTT_FLAG` 为 `--DRTT_HOST_COPY=OFF`，否则为空。
3. 导出环境变量并构建：
  - 调用 `export_tx8fw_to_env` 函数设置 TX8FW 相关环境变量。
  - 运行 cmake 命令，生成构建文件，指定构建选项 `-DUSING_RISCV=ON` 和 `TX8FW_BASE`，并根据 `COPY_RTT_FLAG` 添加额外参数。
  - 使用 make 命令执行构建，目标包括 grep epilog 和 c
4. 清理目录: 使用 popd 命令恢复到之前的目录.

```bash
function build_oplib_with_soc() {
    echo ${OPLIB_PROJECT_ROOT}  # 打印 OPLib 项目根目录路径

    pushd ${OPLIB_PROJECT_ROOT}  # 切换到 OPLib 项目根目录

    rm=rf build  # 定义清理构建目录的命令
    mkdir=build  # 定义创建构建目录的命令
    cd=build     # 定义切换到构建目录的命令

    COPY_RTT_FLAG=""  # 初始化 RTT 复制标志
    if [ "$1" == "NOT_COPY" ]; then  # 如果传入的复制选项为 NOT_COPY
        COPY_RTT_FLAG="--DRTT_HOST_COPY=OFF"  # 设置 RTT 复制标志为关闭
    fi

    export_tx8fw_to_env  # 调用函数导出 TX8FW 相关环境变量

    cmake .. -DUSING_RISCV=ON -TX8FW_BASE=${OPLIB_PROJECT_ROOT}/release/riscv/tx8-yoc-rt-thread-smp ${COPY_RTT_FLAG}  # 生成构建文件，指定 RISCV 和 TX8FW 路径
    ret=$?; if [ $ret -ne 0 ]; then popd; return $ret; fi  # 检查 cmake 是否成功，失败则返回

    make -j cat /proc/stat | grep epilog -c  # 执行构建并检查 epilog 相关信息
    ret=$?; if [ $ret -ne 0 ]; then popd; return $ret; fi  # 检查 make 是否成功，失败则返回

    popd  # 恢复到之前的目录
    echo ${FUNCNAME[0]} "passed"  # 输出函数名和"passed"表示构建成功
}
```

## run_on_soc_rtt

`run_on_soc_rtt`，用于在特定 SoC 和 RTT 环境下运行测试用例。以下是其主要流程：

1. 初始化和参数获取:
  - 函数从命令行参数中获取 `case_name`, `rtt_option` 和 `multi_graph_enable`.
  - 检查 `case_name` 是否为空，如果为空则输出错误信息并返回 1。
2. 目录切换和清理:
  - 将工作目录切换到 `${OPLIB_PROJECT_ROOT}/tests/test_codegen/${case_name}`.
  - 执行 `rm -rf ${case_name}_build` 清理之前的构建文件。
3. 配置设置: 根据 `multi_graph_enable` 设置 CONFIG_ARGS，如果启用多图则设置为 "-DMULTI_GRAPH=1"，否则为空。
4. 构建和运行：
  - 使用 cmake 生成构建文件，指定构建目录为 `${case_name}_build`，并设置 `-DUSING_RISCV=ON` 和 `-TX8FW_BASE` 路径。
  - 使用 make 命令执行构建，目标包括 all 和 chip_out.
5. 清理和返回: 使用 popd 恢复到原始目录。

```bash
function run_on_soc_rtt() {
    echo "${FUNCNAME[0]} 'start'"  # 输出函数名和"start"表示开始

    case_name=$1                  # 获取用例名称
    rtt_option=$2                 # 获取 RTT 选项
    multi_graph_enable=$3         # 获取多图启用标志

    if [ -z "$case_name" ]; then   # 如果用例名称为空
        echo "case_name($case_name) not found "  # 输出错误信息
        return 1                   # 返回错误码 1
    fi

    case_dir=${OPLIB_PROJECT_ROOT}/tests/test_codegen/${case_name}  # 设置用例目录
    pushd ${case_dir}              # 切换到用例目录

    rm -rf ${case_name}_build      # 清理之前的构建文件
    ret=$?; if [ [ $ret -ne 0 ]]; then popd; echo $ret; return $ret; fi  # 检查清理是否成功，失败则返回

    if [ -z "$multi_graph_enable" ]; then  # 如果多图启用标志为空
        CONFIG_ARGS=""                 # 配置参数为空
    else                                 # 否则
        CONFIG_ARGS="-DMULTI_GRAPH=1"  # 设置多图配置参数
    fi

    cmake -B "${case_name}_build" -DUSING_RISCV=ON -TX8FW_BASE=${OPLIB_PROJECT_ROOT}/release/riscv/tx8-yoc-rt-thread-smp ${CONFIG_ARGS} ; ret=$?; if [ [ $ret -ne 0 ]]; then popd; echo $ret; return $ret; fi  # 生成构建文件，指定 RISCV 和 TX8FW 路径

    make -C "${case_name}_build" --target all chip_out ; ret=$?; if [ [ $ret -ne 0 ]]; then popd; echo $ret; return $ret; fi  # 执行构建，目标为 all 和 chip_out

    popd                          # 恢复到原始目录
    echo "${FUNCNAME[0]} 'passed'" # 输出函数名和"passed"表示通过
}
```