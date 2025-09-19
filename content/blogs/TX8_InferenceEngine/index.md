---
title: "TX8 Inference Engine"
date: 2025-08-07T21:47:33+08:00
lastmod: 2025-08-07T21:47:33+08:00
author: ["WITHER"]

categories:
- tx8

tags:
- tx8

keywords:


description: "TX8 inference engine description." # 文章描述，与搜索优化相关
summary: "TX8 inference engine description." # 文章简单描述，会展示在主页
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

# Overview

1. vLLM 调用 txda 初始化 context. 程序结束后释放 context. 
2. vLLM 根据 hf_dir 中的配置文件创建模型, 其中 config.json 决定使用 TxNN 中定义的哪个 model class; 也决定了这个 class init 时的参数是什么.
3. 调用 TxNN 的 model class 的 load weight 函数和 forward 函数, 具体实现都依赖于 TxDA.

# Argument

```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf_dir', type=str, help='Huggingface model path')
    parser.add_argument('--max_tokens', type=int, default=32, help='Max generated tokens')
    parser.add_argument('--tp_size', type=int, default=1, help='Model TP size')  # tensor parallel degree, must Keep consistent with model.json
    parser.add_argument(
        '--device',
        type=lambda e: [int(e) for e in str(e).split(',')],
        default=[0, 1, 2, 3],
        help="Tx device idxs; E.g., --device 0,1,2,3"
    )
    parser.add_argument("--log_file", type=str, default="", help="Log file path")
    args = parser.parse_args()

    # Get case_dir from huggingface model config
    with open(f"{args.hf_dir}/config.json", "r", encoding="utf-8") as file:
        hf_config = json.load(file)
        args.case_dir = hf_config["param_dir"]  # kcorebin path, with model_info.json and chip0/, chip1/ ... in the directory

    print(
        "Inference info:\n"
        f"| - hf_dir: {args.hf_dir}\n"
        f"| - case_dir: {args.case_dir}\n"
        f"| - max_tokens: {args.max_tokens}\n"
        f"| - tp_size: {args.tp_size}\n"
        f"| - ENV:TXDA_VISIBLE_DEVICES: "
        f"{os.environ.get('TXDA_VISIBLE_DEVICES', default='Not specified')}\n"
        f"| - device: {args.device}\n"
        f"| - log_file: {args.log_file}"
    )
    return args
```

- `hf_dir`: 指的是模型的 hugging_face 路径。
- `case_dir`: 是解析 `hf_dir` 下的 config.json 文件的字段，指的是后端生成的 kcore 文件夹路径。除此之外还需要在 json 文件中额外配置并行度相关信息。

```json
{
  // ...
  "param_dir": "...",
  "tensor_parallel": {
    "use_tp": false,
    "parallel_size": 1
  }
}
```

会根据我们传入的 `--device` 和 `tp_size` 参数创建逻辑到实际物理芯片的映射。这里 InternLM3 使用的是 4 卡机器，张量并行度为 2. 计算图中 batchsize=1.
```cpp
// setDeviceIdx
std::unordered_map<uint32_t, std::vector<uint32_t>> c4_TP2x2 = {
    {0, {0, 1}},
    {1, {2, 3}}
};

std::unordered_map<uint32_t, std::vector<uint32_t>> c4_TP2x2_hw = {
    {0, {0, 1}},
    {1, {2, 3}}
};

auto set_TP2_C4_Map = [&](std::unordered_map<uint32_t, std::vector<uint32_t>>& user_map,
                          std::unordered_map<uint32_t, std::vector<uint32_t>>& hw_map,
                          std::vector<uint32_t>& user_settings) {
    for (auto it_TP2x2 : user_map) {
        if (user_settings == it_TP2x2.second) {
            device_setting_ = hw_map[it_TP2x2.first];
        }
    }
};

if (TxdATPMode::C4TP2B1 == tp_mode_) {
    set_TP2_C4_Map(c4_TP2x2, c4_TP2x2_hw, user_settings);
}
```

TxdaContext::initDevice 的主要功能是初始化设备 (TsmDevice)，并将其封装为 TxdaDevice 对象，最终存储到设备组 (device_group_) 中。

TxdaContext::initChipProperty 的主要功能是初始化芯片相关的属性和配置，包括设备组、模型数量、环境变量解析、内存地址初始化以及设备启动参数的设置。

TxdaContext::initModel：负责初始化模型，解析案例目录，设置 TP 大小和 TP 模式，并调用 TxdaModel::init 初始化每个模型 的基本属性，包括案例目录、JSON 配置和编译选项。

# TxNN

configuration_internlm3.py 中主要负责模型结构参数的初始化 (根 hugging face 中的 config.json 保持一致) 以及张量并行的设置。

input_output_internlm3.py 负责在 python 端为要从后端加载进来的输入和输出注册张量和开辟空间。-
- InputBuffer 会创建一个 IntermediateKVCache 实例。这是 KV Cache 的实际存储空间，大小根据序列长度、头的数量预先分配好。还会注册各种输入相关的缓冲区 
    - input_ids: 当前需要处理的输入 token ID。
    - start_index, seq_len: 用于管理当前生成序列的位置信息。
    - k_gather_end_indices: 为每一层配置 对应的 KV cache 的起始索引。

OutputBuffer 和 InputBuffer 类似，它也创建了一个 IntermediateKVCache 实例。这个 Cache 用于存放计算完成后更新的 Key 和 Value，这些更新后的值将在下一步生成token时作为输入使用。
- 注册所有KV Cache层: 它立即将所有层的 KV Cache 缓冲区注册到自己的 _buffer 中。
- 注册输出缓冲区: 预先分配并注册了用于存放最终结果的内存空间：
    - lm_head_reshape_out: 用于存放语言模型最后的输出（通常是 logits），其维度为 (序列长度, 词汇表大小)。
    - lm_head_argmax_out: 用于存放最终预测出的 token ID（对 logits 取 argmax 后的结果）。


里面负责定义模型，在 python 端为数据开辟好内存，GraphBasedInternlm3ForCausalLM 在初始化时，会创建 InputBuffer 和 OutputBuffer 的实例。它为 tensor_parallel 的每一路都创建了缓冲区。

当 forward 方法接收 input_ids. 不在 Python 中执行复杂的数学运算。而是通过 load_input 把后端编译生成的 bin 文件数据填充到 InputBuffer 中.

通过 TxDA context 把加载好的数据放到硬件上然后 launch kernel将 InputBuffer 和 OutputBuffer 的内存地址传递给底层引擎。底层引擎直接从 InputBuffer 指向的内存中读取输入，执行高效的模型计算，然后将结果直接写入 OutputBuffer 指向的内存中。

# vllm

主要利用的是其推理框架，在 LLMEngine 里添加了自己的 TxNPUExecutor 里面包含自己的 TxNPUWorker，其中包含 TxNPUCacheEngine 和 TxNPUModelRunner.

TxNPUMModelRunner 定义了专门从后端编译出的 chip_out 文件夹中的 param.bin 文件里加载模型参数到 TxNN 的定义模型的 ModelLoader. 然后推理的时候调用的 forward 流程如上。