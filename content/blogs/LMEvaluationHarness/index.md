---
title: "LM Evaluation Harness"
date: 2025-11-25T09:50:09+08:00
lastmod: 2025-11-25T09:50:09+08:00
author: ["WITHER"]

categories:
- Source Code Reading
tags:
- LLM Eval

keywords:
- LLM Eval

description: "Source code reading of lm-evaluation-harness." # 文章描述，与搜索优化相关
summary: "Source code reading of lm-evaluation-harness." # 文章简单描述，会展示在主页
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

# Command-line Interface

大多数用户通过从 Github 克隆代码库，以可编辑模式安装该包，

```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ".[dev]"
```
运行 `python -m lm_eval` 脚本来使用。

同样地，也可以通过命令行中的 `lm-eval` 入口点来运行本库。

该模式支持多种命令行参数，可以通过运行 `-h` 或 `--help` 查看详细信息: 

- `--model` : 选择要评估的模型类型或提供商。必须是对应于所使用的模型类型/提供商名称的字符串。请参阅 [MAIN README](https://github.com/EleutherAI/lm-evaluation-harness/tree/main#model-apis-and-inference-servers) 获取已启用模型名称和支持的库或 API 的完整列表。

- `--model_args` : 控制传递给模型构造函数的参数。接受一个字符串，包含传递给模型类的逗号分隔的关键字参数，格式为 `"arg1=val1,arg2=val2,..."`，例如 `--model_args pretrained=EleutherAI/pythia-160m,dtype=float32`. 关于支持哪些关键字参数的完整列表，请参阅 `lm_eval.api.model.LM` 子类的初始化方法，例如 [`HFLM`](https://github.com/EleutherAI/lm-evaluation-harness/blob/365fcda9b85bbb6e0572d91976b8daf409164500/lm_eval/models/huggingface.py#L66).

- `--tasks` : 确定要评估哪些任务或任务组。接受以逗号分隔的任务名称或任务组名称列表。必须完全由有效的任务/组组成。可以通过 `--tasks list` 查看支持的任务列表。

- `--num_fewshot` : 设置放入上下文中的少样本 (few-shot) 示例的数量。必须是一个整数。

- `--gen_kwargs` : 接受与 `--model_args` 格式相同的参数字符串，并创建一个关键字参数字典。这些参数将传递给所有被调用的 `generate_until` (自由形式或贪婪生成任务) 任务的模型，用于设置采样温度 (temperature) 或 `top_p` / `top_k` 等选项。关于每种模型类型支持哪些参数，请参考相应库的文档 (例如 `transformers.AutoModelForCausalLM.generate()` 的文档) 。这些 kwargs 将应用于所有被调用的 `generate_until` 任务——我们目前不支持在单次运行中为每个任务设置唯一的 `gen_kwargs` 或 `batch_size` 值。要在任务级别控制这些参数，请在该任务的 YAML 文件中进行设置。

- `--batch_size` : 设置用于评估的批次大小. 可以是正整数，或者是 `"auto"` 以自动选择能放入内存的最大批次大小，从而加速评估。可以通过传递 `--batch_size auto:N` 在评估期间重新选择最大批次大小 `N` . 这有助于进一步加速评估，因为 `lm-eval` 会按上下文长度降序排列文档。

- `--max_batch_size` : 如果传递了 `--batch_size auto`，则设置尝试放入内存的最大批次大小。

- `--device` : 设置将模型放置在哪个设备上。必须是字符串，例如 `"cuda"`, `"cuda:0"`, `"cpu"`, `"mps"`.默认为 "cuda"，如果是运行多 GPU 或非本地模型类型，则可以忽略。

- `--output_path` : 格式为 `dir/file.jsonl` 或 `dir/` 的字符串。提供保存高级结果的路径，可以是指定的文件或指定的目录。如果同时传递了 `--log_samples`，则每个文档的输出和指标也将保存到该目录中。

- `--log_samples` : 如果传递此标志，则模型的输出以及输入模型的文本将按文档粒度进行保存。必须与 `--output_path` 一起使用。

- `--limit` : 接受一个整数，或 0.0 到 1.0 之间的浮点数。如果传递，将限制每个任务评估的文档数量为前 X 个文档 (如果是整数) 或前 X% 的文档。对于调试非常有用，尤其是在昂贵的 API 模型上。

- `--use_cache` : 应为一个可以写入 sqlite db 文件的路径。接受格式为 `/path/to/sqlite_cache_` 的字符串，以便为每个进程 (0-NUM_GPUS) 在 `/path/to/sqlite_cache_rank{i}.db` 创建缓存数据库。这允许缓存先前运行的结果，因此无需重新运行结果即可重新评分或再次运行给定的 (模型, 任务) 对。

- `--cache_requests` : 可以是 "true"、"refresh" 或 "delete". "true" 表示应使用缓存。"refresh" 表示您希望重新生成缓存，如果您更改了给定任务的数据集配置，则应运行此选项。"delete" 将删除缓存。缓存文件存储在 `lm_eval/cache/.cache` 下，除非您通过环境变量 `LM_HARNESS_CACHE_PATH` 指定不同的路径。例如 `LM_HARNESS_CACHE_PATH=~/Documents/cache_for_lm_harness`.

- `--check_integrity` : 如果使用此标志，将运行所选每个任务的库测试以确认任务完整性。

- `--write_out` : 用于诊断目的，以观察传递给模型的任务文档的格式。如果使用此标志，则打印每个任务第一个文档的提示 (prompt) 和黄金目标字符串 (gold target string).

- `--show_config` : 如果使用，将在评估完成时打印每个运行任务的完整 `lm_eval.api.task.TaskConfig` 内容 (任务 YAML 文件中的非默认设置). 当在本地修改任务的配置 YAML 以传输用于调试的确切配置或出于可复现性目的时，这非常有用。

- `--include_path` : 接受文件夹的路径。如果传递，则包含 `lm-eval` 兼容任务配置的所有 YAML 文件都将作为可用任务添加到任务注册表中。用于当用户在 `lm_eval/tasks/` 以外的文件夹中为自己的任务编写配置文件时。

- `--system_instruction`: 指定要预置到提示 (prompt) 前的系统指令字符串。

- `--apply_chat_template` : 此标志指定是否将聊天模板应用于提示。可以通过以下方式使用: 
  - `--apply_chat_template` : 不带参数使用时，将唯一的可用聊天模板应用于提示。对于 Hugging Face 模型，如果没有专用的聊天模板，将应用默认的聊天模板。
  - `--apply_chat_template template_name` : 如果模型有多个聊天模板，则将指定的模板应用于提示。

    对于 Hugging Face 模型，默认聊天模板可以在 Transformers Tokenizer 的 [`default_chat_template`](https://github.com/huggingface/transformers/blob/fc35907f95459d7a6c5281dfadd680b6f7b620e3/src/transformers/tokenization_utils_base.py#L1912) 属性中找到。

- `--fewshot_as_multiturn` : 如果开启此标志，少样本 (Fewshot) 示例将被视为多轮对话。问题作为用户内容提供，答案作为助手响应提供。需要 `--num_fewshot` 设置为大于 0，并且开启 `--apply_chat_template`.

- `--predict_only`: 生成模型输出而不计算指标。与 `--log_samples` 一起使用以检索解码后的结果。

- `--seed`: 设置 python 的 random、numpy 和 torch 的种子。接受逗号分隔的 3 个值列表，分别对应 python 的 random、numpy 和 torch 种子，或者单个整数以为所有三个设置相同的种子。值为整数或 'None' (不设置种子) 。默认为 `0,1234,1234` (为了向后兼容) 。例如 `--seed 0,None,8` 设置 `random.seed(0)` 和 `torch.manual_seed(8)`. 这里 numpy 的种子未设置，因为第二个值是 `None`. 例如，`--seed 42` 将所有三个种子设置为 42.

- `--wandb_args`: 跟踪评估运行到 Weights and Biases 的日志记录，并包含传递给 `wandb.init` 的参数，例如 `project` 和 `job_type`.完整列表[在此](https://docs.wandb.ai/ref/python/init)。例如，`--wandb_args project=test-project,name=test-run`.还允许传递记录事物的步骤 (传递给 `wandb.run.log`) ，例如 `--wandb_args step=123`.

- `--hf_hub_log_args` : 将评估结果记录到 Hugging Face Hub。接受以逗号分隔的参数字符串。可用参数: 
  - `hub_results_org` - Hugging Face Hub 上的组织名称，例如 `EleutherAI`.如果未提供，结果将推送到 Hugging Face 令牌的所有者，
  - `hub_repo_name` - Hugging Face Hub 上的存储库名称 (已弃用，应改用 `details_repo_name` 和 `results_repo_name`) ，例如 `lm-eval-results`，
  - `details_repo_name` - Hugging Face Hub 上用于存储详细信息的存储库名称，例如 `lm-eval-results`，
  - `results_repo_name` - Hugging Face Hub 上用于存储结果的存储库名称，例如 `lm-eval-results`，
  - `push_results_to_hub` - 是否将结果推送到 Hugging Face Hub，可以是 `True` 或 `False`，
  - `push_samples_to_hub` - 是否将样本结果推送到 Hugging Face Hub，可以是 `True` 或 `False`.需要设置 `--log_samples`，
  - `public_repo` - 存储库是否公开，可以是 `True` 或 `False`，
  - `leaderboard_url` - 排行榜的 URL，例如 `https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard`.
  - `point_of_contact` - 结果数据集的联系人，例如 `yourname@example.com`.
  - `gated` - 是否对详细信息数据集进行门控 (gated) ，可以是 `True` 或 `False`.

- `--metadata`: 传递给 TaskConfig 的 JSON 字符串。用于某些需要传递额外元数据进行处理的任务。例如，`--metadata '{"key": "value"}'`.

# cli_evaluate

`cli_evaluate` 函数是 `lm-evaluation-harness` 的命令行入口点，负责解析参数、初始化组件并启动评估流程。

1.  **参数解析 (Argument Parsing)**:
    *   如果未传入 `args`，则调用 `setup_parser()` 创建参数解析器，并调用 `parse_eval_args(parser)` 解析命令行参数。
    *   这包括模型名称 (`--model`)、模型参数 (`--model_args`)、任务列表 (`--tasks`)、批次大小 (`--batch_size`) 等。

2.  **延迟导入 (Lazy Imports)**:
    *   为了加快 CLI 启动速度，核心模块 (如 `evaluator`, `utils`, `TaskManager` 等) 在参数解析后才被导入。

3.  **日志与环境设置 (Logging & Environment)**:
    *   处理 `wandb` 参数并初始化 `WandbLogger` (如果启用).
    *   调用 `utils.setup_logging` 设置日志级别。
    *   设置环境变量 `TOKENIZERS_PARALLELISM="false"` 以避免 tokenizer 的并行警告。
    *   初始化 `EvaluationTracker` 用于结果跟踪和 Hugging Face Hub 集成。

4.  **参数校验与预处理 (Validation & Preprocessing)**:
    *   检查 `predict_only` 和 `log_samples` 是否与 `output_path` 冲突 (必须指定输出路径).
    *   检查 `fewshot_as_multiturn` 是否与 `apply_chat_template` 配合使用。
    *   合并 `model_args` 和 `metadata`.

5.  **任务管理器初始化 (Task Manager Initialization)**:
    *   实例化 `TaskManager`，加载任务配置。如果指定了 `include_path`，也会加载该路径下的任务。

6.  **任务选择 (Task Selection)**:
    *   根据 `--tasks` 参数确定要运行的任务: 
        *   如果是 `list`, `list_groups` 等命令，打印任务列表并退出。
        *   如果是目录，加载该目录下的所有 YAML 任务配置。
        *   如果是任务名称列表，调用 `task_manager.match_tasks` 匹配并加载任务。
    *   检查是否有未找到的任务，如果有则报错。

7.  **远程代码信任设置 (Trust Remote Code)**:
    *   如果启用了 `--trust_remote_code`，设置环境变量并更新 `model_args`，允许加载 Hugging Face Hub 上包含自定义代码的数据集或模型。

8.  **执行评估 (Execute Evaluation)**:
    *   调用核心函数 **`evaluator.simple_evaluate(...)`**. 
    *   将所有解析和处理后的参数 (模型、任务、批次大小、随机种子等) 传递给它。
    *   这个函数会负责模型加载、数据准备、推理和指标计算，并返回结果字典 `results`.

9.  **结果处理与输出 (Results Handling & Output)**:
    *   如果 `results` 不为空: 
        *   如果启用了 `log_samples`，从结果中提取样本数据。
        *   如果启用了 `show_config`，打印完整的配置 JSON。
        *   **W&B 记录**: 如果启用了 Weights & Biases，记录评估结果和样本。
        *   **保存结果**: 调用 `evaluation_tracker.save_results_aggregated` 将结果保存到本地文件或推送到 Hugging Face Hub。
        *   **打印表格**: 使用 `make_table(results)` 格式化并打印最终的评估结果表格 (包括每个任务的指标).
        *   **清理**: 结束 W&B 运行。

## TaskManager Initialization

`TaskManager.__init__` 时并不立即加载所有任务的具体内容 (不实例化任务对象) ，而是建立一个**索引**. 它知道有哪些任务、它们在哪里 (文件路径) 、是什么类型。只有在真正需要运行某个任务时，才会根据这个索引去加载具体的配置和代码。

以下是 `TaskManager.__init__` 方法的详细执行流程: 

1.  **设置日志 (Logging Setup)**:
    *   如果传入了 `verbosity` 参数，调用 `utils.setup_logging` 来配置日志级别。

2.  **存储元数据 (Store Metadata)**:
    *   保存传入的 `include_path` (额外的任务路径) 和 `metadata` (全局元数据).

3.  **初始化任务索引 (Initialize Task Index)**:
    *   调用 **`self.initialize_tasks(...)`** 方法。
    *   该方法会扫描默认的 `lm_eval/tasks/` 目录以及用户提供的 `include_path`.
    *   它遍历这些目录下的所有 `.yaml` 文件，解析配置，并构建一个字典 `self._task_index`.
    *   `_task_index` 记录了每个任务/组/标签的名称、类型 (`task`, `group`, `tag`, `python_task`) 以及对应的 YAML 文件路径。

4.  **分类整理 (Categorization)**:
    *   基于 `_task_index`，将所有条目分类并排序存储到不同的列表中，以便快速访问: 
        *   `self._all_tasks`: 所有已注册名称的列表 (包括任务、组和标签).
        *   `self._all_groups`: 所有类型为 `group` 的名称列表。
        *   `self._all_subtasks`: 所有具体任务 (类型为 `task` 或 `python_task`) 的名称列表。
        *   `self._all_tags`: 所有类型为 `tag` 的名称列表。

5.  **初始化分组映射 (Initialize Group Map)**:
    *   创建一个 `collections.defaultdict(list)` 类型的 `self.task_group_map`.
    *   这个字典用于在后续加载过程中处理任务别名或重复任务时的分组逻辑 (尽管在初始化阶段它还是空的).

# simple_evaluate

`simple_evaluate` 函数是 `lm-evaluation-harness` 库的主要入口点，它的主要职责是**解析配置、初始化模型和任务、应用覆盖设置，最后调用底层的 `evaluate` 函数执行实际评估**. 

以下是结合代码的详细流程解析: 

1. **初始化与环境设置 (Initialization & Setup)**
    *   **日志与校验**: 设置日志级别 (`setup_logging`)，并检查参数冲突 (如 `limit` 和 `samples` 不能同时存在).
    *   **聊天模板警告**: 如果检测到模型参数包含 "instruct" 或 "chat" 但未启用 `apply_chat_template`，会发出警告。
    *   **随机种子**: 设置 Python `random`、`numpy` 和 `torch` 的随机种子，以确保评估结果的可复现性。
    *   **缓存清理**: 如果设置了 `delete_requests_cache`，则清理请求缓存。

2. **模型初始化 (Model Initialization)**
代码根据传入的 `model` 参数类型进行不同处理: 
    *   **字符串 (String)**: 如果 `model` 是字符串 (如 `"hf"`) ，它会解析 `model_args`，并使用 `lm_eval.api.registry.get_model(model)` 动态加载并初始化对应的模型类 (例如 `HFLM`).
    *   **对象 (Object)**: 如果 `model` 已经是 `LM` 类的实例，则直接使用，跳过初始化。
    *   **缓存包装**: 如果指定了 `use_cache`，会将模型包装在 `CachingLM` 中，以便将模型输出缓存到 SQLite 数据库中，加快后续运行速度。

3. **任务加载与配置调整 (Task Loading & Config Adjustment)**
这是该函数最关键的逻辑之一: 
    *   **加载任务**: 使用 `TaskManager` 和 `get_task_dict` 根据任务名称加载实际的任务对象。
    *   **配置覆盖 (`_adjust_config` 内部函数)**: 这是一个递归函数，用于遍历所有任务。
        *   **`gen_kwargs`**: 将命令行传入的生成参数 (如 `temperature`) 应用到任务配置中。
        *   **`predict_only`**: 如果开启，将任务指标强制覆盖为 `"bypass"`，只生成不评分。
        *   **`num_fewshot`**: 将命令行传入的少样本数量覆盖任务的默认配置 (除非任务明确配置为 0).
        *   **`fewshot_random_seed`**: 设置少样本采样的随机种子。

4. **执行评估 (Execution)**
    *   **完整性检查**: 如果 `check_integrity` 为真，运行任务的测试套件。
    *   **调用核心函数**: 调用 **`evaluate(...)`** 函数。
        *   这是实际发生“魔法”的地方。它接收准备好的模型对象和配置好的任务字典，执行构建请求、模型推理、收集结果等核心流程。

5. **结果汇总与元数据 (Result Aggregation & Metadata)**
    *   **Rank 0 处理**: 在分布式环境中，只有主进程 (Rank 0) 负责汇总结果。
    *   **元数据附加**:
        *   收集模型名称、参数、批次大小、设备信息。
        *   记录 Git commit hash、开始时间。
        *   添加环境信息 (`add_env_info`) 和 Tokenizer 信息 (`add_tokenizer_info`)。
    *   **返回结果**: 最终返回包含所有指标、配置和 (可选的) 样本详情的 `results` 字典。

## HFLM Initialization

[`HFLM`](https://github.com/EleutherAI/lm-evaluation-harness/blob/365fcda9b85bbb6e0572d91976b8daf409164500/lm_eval/models/huggingface.py#L66) (Hugging Face Language Model) 的初始化流程是将命令行参数转化为实际模型加载和配置的关键步骤。

以下是结合命令行参数的详细初始化流程解释: 

初始化入口为上文中提到的`create_from_arg_string` / `create_from_arg_obj`. 当你在命令行使用 `--model hf` 时，`lm_eval` 会调用 `HFLM` 类的构造函数。命令行参数 `--model_args` (如 `pretrained=gpt2,dtype=float16`) 会被解析并传递给 `__init__` 方法。

`HFLM` 的初始化主要分为以下几个关键阶段: 

1. **基础配置与参数解析**
    *   **`pretrained`** (对应 `--model_args pretrained=...`):
        *   这是最核心的参数，指定模型名称 (如 `meta-llama/Llama-2-7b-hf`) 或本地路径。
        *   如果没有提供，默认会尝试使用 `backend` 参数 (通常用于旧版兼容).
    *   **`revision`**: 指定 Hugging Face Hub 上的特定版本 (分支或 commit hash).
    *   **`subfolder`**: 指定 Hub 仓库中的子文件夹。
    *   **`tokenizer`**:
        *   默认使用 `pretrained` 指定的路径。
        *   如果指定了不同的 tokenizer (例如在微调模型时使用基础模型的 tokenizer)，可以在这里覆盖。

2. **硬件与加速设置**
    *   **`device` (对应 `--device ...`)**:
        *   指定模型运行的设备 (如 `cuda`, `cpu`, `cuda:0`).
        *   如果未指定，代码会尝试自动检测可用的 GPU.
    *   **`parallelize` (对应 `--model_args parallelize=True`)**:
        *   如果设置为 `True`，会启用简单的模型并行。
        *   它通常会触发 `device_map="auto"`，利用 `accelerate` 库将模型层自动分配到多个 GPU 上。
    *   **`device_map`**:
        *   更细粒度的控制，直接传递给 `transformers.AutoModel.from_pretrained`.
        *   常用于大模型推理，例如 `device_map="auto"` 或 `device_map="balanced"`.

3. **模型加载 (Model Loading)**
这是最耗时的步骤，代码会调用 `transformers` 库加载模型: 
    *   **`dtype`** (对应 `--model_args dtype=...`):
        *   设置模型权重的数据类型 (如 `float16`, `bfloat16`, `float32`, `auto`).
        *   `auto` 会根据硬件自动选择最佳精度。
    *   **`trust_remote_code`** (对应 `--trust_remote_code` 或 `--model_args trust_remote_code=True`):
        *   允许加载 Hub 上包含自定义 Python 代码的模型 (如 ChatGLM, MPT 等).
    *   **`auth_token`**: 用于访问私有模型 (现在通常通过 `huggingface-cli login` 处理，代码中可能体现为 `token` 参数).
    *   **加载逻辑**:
        ```python
        self.model = self.AUTO_MODEL_CLASS.from_pretrained(
            pretrained,
            revision=revision,
            torch_dtype=self._get_dtype(dtype),
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            # ... 其他参数
        )
        ```
        *   `AUTO_MODEL_CLASS` 通常是 `AutoModelForCausalLM` (GPT类) 或 `AutoModelForSeq2SeqLM` (T5类).

4. **Tokenizer 加载**
    *   **`padding_side`**:
        *   通常设置为 `"left"` (左填充) ，这对于仅解码 (Decoder-only) 模型 (如 GPT, LLaMA) 至关重要，以确保生成的 token 位置正确。
    *   **`truncation_side`**: 通常也设置为 `"left"`.

5. **批次大小处理 (Batch Size)**
    *   **`batch_size` (对应 `--batch_size ...`)**:
        *   如果是整数 (如 `16`) ，直接设置。
        *   如果是 `"auto"`，初始化时不设定具体值，而是在运行时通过 `_detect_batch_size` 动态探测最大可用显存。

6. **其他高级配置**
*   **`peft` / `delta`**: 支持加载 LoRA 或其他 PEFT 微调适配器。
*   **`autogptq` / `gptqmodel`**: 支持加载 GPTQ 量化模型。


| 命令行参数 | `HFLM` 初始化参数 | 作用 |
| :--- | :--- | :--- |
| `--model hf` | (Class Selection) | 选择 `HFLM` 类 |
| `--model_args pretrained=X` | `pretrained` | 模型名称/路径 |
| `--model_args dtype=float16` | `dtype` | 权重精度 |
| `--model_args parallelize=True`| `parallelize` | 开启多卡并行 (`device_map="auto"`) |
| `--model_args trust_remote_code=True` | `trust_remote_code` | 允许执行远程代码 |
| `--device cuda:0` | `device` | 指定运行设备 |
| `--batch_size auto` | `batch_size` | 自动批次大小检测 |

## get_task_dict

`get_task_dict` 是 `simple_evaluate` 中用于解析和加载任务的核心步骤。它将用户输入的任务列表（可能是字符串名称、字典配置或 `Task` 对象）转换为一个统一的、可执行的任务字典。

以下是详细的流程解析：

1. **输入处理与分类**
`get_task_dict` 接收 `task_name_list`（即 `simple_evaluate` 中的 `tasks` 参数）和 `task_manager`.
它首先将输入的任务列表按类型分类：
    *   **字符串列表 (`string_task_name_list`)**: 如 `["gsm8k", "mmlu"]`.
    *   **其他列表 (`others_task_name_list`)**: 包含字典配置或直接传入的 `Task` 对象。

2. **加载字符串类型的任务**
对于字符串类型的任务名，调用 **`task_manager.load_task_or_group(string_task_name_list)`**.
    *   **`load_task_or_group`**:
        *   遍历任务名列表，对每个任务名调用 `_load_individual_task_or_group`.
    *   **`_load_individual_task_or_group`**:
        *   **查找**: 在 `task_manager.task_index` 中查找任务名对应的 YAML 路径或类型。
        *   **加载配置**: 调用 `_get_config` 读取 YAML 文件。
        *   **实例化**:
            *   如果是普通任务 (`task`)，创建 `ConfigurableTask` 实例。
            *   如果是 Python 任务 (`python_task`)，动态导入并实例化指定的 Python 类。
            *   如果是组 (`group`)，创建 `ConfigurableGroup`，并递归加载其包含的所有子任务。
            *   如果是标签 (`tag`)，找到所有打该标签的任务并加载。
        *   **返回**: 一个字典，键是任务名，值是任务对象（或嵌套的组字典）。

3. **加载字典配置和对象类型的任务**
    遍历 `others_task_name_list`：
    *   **字典 (`dict`)**: 调用 `task_manager.load_config(config)`，这本质上也是调用 `_load_individual_task_or_group`，但直接使用传入的字典作为配置，而不是去文件系统查找。
    *   **Task 对象 (`Task`)**: 直接将其添加到结果字典中，使用 `get_task_name_from_object` 获取键名。

4. **合并与校验**
    *   **合并**: 将上述三部分加载的结果（字符串来源、配置来源、对象来源）合并为 `final_task_dict`.
    *   **重复检查**: 调用 `_check_duplicates`.
        *   它使用 `get_subtask_list` 展开所有组，检查是否有同一个子任务被包含在多个不同的组中被同时调用。
        *   如果存在这种情况（例如 Group A 包含 Task X，Group B 也包含 Task X，且同时评估 Group A 和 B），会抛出 `ValueError`，因为这可能导致配置冲突（如不同的 `num_fewshot`）。

5. **返回值**
函数最终返回 **`final_task_dict`**. 
    *   **类型**: `dict`
    *   **结构**:
        *   **Key**: 任务名称（字符串）。
        *   **Value**:
            *   如果是单个任务：`lm_eval.api.task.Task` 的实例（通常是 `ConfigurableTask`）。
            *   如果是任务组：一个嵌套的字典，结构同上，包含该组下的子任务。


# evaluate

`evaluate` 函数是 `lm-evaluation-harness` 的核心执行引擎，负责协调模型、任务和数据，完成整个评估过程。

以下是结合代码的详细流程解析: 

1.  **初始化与校验 (Initialization & Validation)**:
    *   **参数检查**: 确保 `limit` 和 `samples` 没有冲突。
    *   **日志记录**: 记录正在评估的任务。
    *   **聊天模板警告**: 如果启用了 `apply_chat_template`，发出警告 (因为这会影响 loglikelihood 任务).
    *   **兼容性检查**:
        *   检查是否尝试在非多模态模型上运行多模态任务。
        *   检查是否尝试运行标记为“不安全代码”的任务 (如 `humaneval`) 而未确认。

2.  **构建请求 (Build Requests)**:
    *   遍历所有任务 (`eval_tasks`)。
    *   计算样本限制 (`limit`)。
    *   调用 **`task.build_all_requests(...)`**: 它将原始数据集转换为模型可理解的 `Instance` 对象 (包含 Prompt、Target 等).
    *   **请求聚合**: 将生成的 `Instance` 按请求类型 (如 `loglikelihood`, `generate_until`) 分类存储到 `requests` 字典中。
    *   **分布式填充**: 如果是多 GPU 环境 (`lm.world_size > 1`)，计算每个 rank 的请求数量差异，并记录需要填充的请求数 (`padding_requests`)，以确保所有 GPU 处理相同数量的批次。

3.  **模型推理 (Model Inference)**:
    *   遍历 `requests` 字典中的每种请求类型 (`reqtype`).
    *   **克隆请求**: 根据 `req.repeats` 复制请求 (某些评估可能需要多次采样).
    *   **填充请求**: 如果需要，添加填充请求以对齐分布式批次。
    *   **执行推理**: 调用 **`getattr(lm, reqtype)(cloned_reqs)`**. 这会动态调用模型类上的对应方法 (如 `lm.loglikelihood(reqs)` 或 `lm.generate_until(reqs)`) ，执行实际的前向传播或生成。
    *   **收集结果**: 将模型返回的结果 (`resps`) 填回对应的请求对象中。
    *   **同步**: 在多 GPU 环境下，等待所有 rank 完成。

4.  **后处理与指标计算 (Post-processing & Metrics)**:
    *   遍历每个任务。
    *   **应用过滤器**: 调用 `task.apply_filters()` 处理模型输出 (如正则提取答案).
    *   **按文档分组**: 将请求按原始文档 ID (`doc_id`) 分组并排序。
    *   **计算指标**:
        *   遍历文档迭代器。
        *   调用 **`task.process_results(...)`**: 对比模型输出和真实标签，计算该样本的指标 (如准确率、BLEU 等).
        *   **记录样本**: 如果 `log_samples` 为真，构建包含输入、输出、目标和指标的详细样本字典，用于后续分析。
        *   收集所有样本的指标值。

5.  **结果汇总 (Result Aggregation)**:
    *   **分布式收集**: 如果是多 GPU 环境，使用 `torch.distributed.gather_object` 将所有 rank 的样本日志和指标数据收集到主进程 (Rank 0)。
    *   **Rank 0 处理**:
        *   **聚合指标**: 调用 `task_output.calculate_aggregate_metric(...)` 计算最终的平均指标 (如准确率) 和置信区间 (Bootstrap CI).
        *   **合并结果**: 使用 `consolidate_results` 和 `consolidate_group_results` 将各个任务的结果整理成结构化的字典。
        *   **准备输出**: 构建最终的 `results_dict`，包含结果、配置、版本信息、样本数量等。
        *   **处理样本**: 如果需要，对图像样本进行哈希处理以减小体积。

6.  **返回**:
    *   主进程返回完整的 `results_dict`.
    *   其他进程返回 `None`.

## build_all_requests

`task.build_all_requests` 是 `lm-evaluation-harness` 中将原始数据集转换为模型可执行请求的核心方法。

1.  **缓存检查与加载**:
    *   构建一个唯一的 `cache_key`，包含任务名、少样本数、rank、world_size、聊天模板配置等。
    *   尝试调用 `load_from_cache`.如果命中且不需要重写 (`rewrite_requests_cache=False`)，直接加载并返回，跳过后续步骤。

2.  **获取文档迭代器**:
    *   调用 `self.doc_iterator(...)` 获取当前 rank 需要处理的文档列表。
    *   这个迭代器会处理数据分片 (Distributed Data Parallel) 和样本限制 (`limit`)。

3.  **遍历文档并构建请求**:
    *   遍历每个文档 `doc`: 
        *   **构建上下文 (`fewshot_context`)**:
            *   调用 `self.fewshot_context(...)`.
            *   该方法会根据 `num_fewshot` 从训练集中采样示例，拼接任务描述 (`description`)、少样本示例和当前文档的输入部分。
            *   如果启用了 `apply_chat_template`，会构建对话格式的上下文。
        *   **构建请求 (`construct_requests`)**:
            *   调用 `self.construct_requests(doc, ctx, ...)`.
            *   这是抽象方法，由具体的 Task 子类 (如 `ConfigurableTask`, `MultipleChoiceTask`) 实现。
            *   它返回一个或多个 `Instance` 对象。

4.  **扁平化与存储**:
    *   将生成的 `Instance` 列表扁平化 (因为 `construct_requests` 可能返回列表).
    *   将结果存储在 `self._instances` 中。

5.  **保存缓存**:
    *   如果启用了缓存，调用 `save_to_cache` 将构建好的 `instances` 保存到磁盘。

`Instance` 的 `request_type` (即 `reqtype`) 是在 `construct_requests` 方法中创建 `Instance` 对象时指定的。它通常直接对应于 Task 的 `OUTPUT_TYPE`，但也有例外 (如多选任务).

以下是几种常见情况的对应关系: 

1.  **`generate_until` (生成任务)**:
    *   **Task 类型**: `ConfigurableTask` (当 `output_type="generate_until"`)
    *   **`construct_requests` 逻辑**:
        ```python
        # lm_eval/api/task.py -> ConfigurableTask.construct_requests
        elif self.OUTPUT_TYPE == "generate_until":
            arguments = (ctx, self.config.generation_kwargs)
        # ...
        return Instance(request_type="generate_until", ...)
        ```

2.  **`loglikelihood` (对数似然任务)**:
    *   **Task 类型**: `ConfigurableTask` (当 `output_type="loglikelihood"`)
    *   **`construct_requests` 逻辑**:
        ```python
        elif self.OUTPUT_TYPE == "loglikelihood":
            arguments = (ctx, self.doc_to_target(doc))
        # ...
        return Instance(request_type="loglikelihood", ...)
        ```

3.  **`multiple_choice` (多选任务)**:
    *   **Task 类型**: `ConfigurableTask` (当 `output_type="multiple_choice"`) 或 `MultipleChoiceTask`.
    *   **特殊情况**: 虽然 Task 的 `OUTPUT_TYPE` 可能是 `"multiple_choice"`，但底层的 `reqtype` 通常是 `"loglikelihood"`.
    *   **`construct_requests` 逻辑 (以 `MultipleChoiceTask` 为例)**:
        ```python
        # lm_eval/api/task.py -> MultipleChoiceTask.construct_requests
        return [
            Instance(
                request_type="loglikelihood",  # 注意这里！
                doc=doc,
                arguments=(ctx, " {}".format(choice)),
                ...
            )
            for i, choice in enumerate(doc["choices"])
        ]
        ```
    *   **解释**: 多选任务通常通过计算每个选项的对数似然 (loglikelihood) ，然后选择概率最高的选项来实现。因此，它会生成多个 `loglikelihood` 类型的请求。

4.  **`loglikelihood_rolling` (困惑度任务)**:
    *   **Task 类型**: `PerplexityTask` 或 `ConfigurableTask` (当 `output_type="loglikelihood_rolling"`).


| Task `OUTPUT_TYPE` | Instance `request_type` | 说明 |
| :--- | :--- | :--- |
| `generate_until` | `generate_until` | 文本生成 |
| `loglikelihood` | `loglikelihood` | 计算 (Context, Target) 的条件概率 |
| `multiple_choice` | `loglikelihood` | **特殊**: 转化为多个 loglikelihood 请求 |
| `loglikelihood_rolling` | `loglikelihood_rolling` | 计算整个序列的困惑度 |

# HFLM Run Requests
`HFLM` 类实现了 `lm-evaluation-harness` 中 Hugging Face 模型的评估逻辑。虽然 `Task` 定义了四种 `output_type`，但在 `HFLM` 层面，它们主要映射到两个核心方法: `loglikelihood` (处理 `loglikelihood` 和 `multiple_choice`) 和 `generate_until` (处理 `generate_until`).`loglikelihood_rolling` 有其专门的实现。

以下是针对四种 `output_type` 的详细函数流程解释: 

1. `loglikelihood` (对数似然):
用于计算给定上下文 (Context) 和目标 (Target) 的条件概率: $P(\text{Target} | \text{Context})$.  `_loglikelihood_tokens` (被 `loglikelihood` 方法调用，它是 `TemplateLM` 的一部分，最终会调用 `_loglikelihood_tokens`).
    1.  **排序与分组**: 使用 `Collator` 对请求进行排序 (按长度降序，优化批处理效率) 和分组 (按上下文分组，优化多选任务).
    2.  **批处理**: 将请求分块 (`chunks`).
    3.  **编码与拼接**:
        *   将 Context 和 Target 拼接。
        *   对于 Causal LM (Decoder-only)，输入是 `[Context, Target]`.
        *   对于 Seq2Seq LM (Encoder-Decoder)，输入是 `Context`，标签是 `Target`.
    4.  **模型推理**:
        *   调用 `self._model_call` 获取 logits.
        *   计算 `log_softmax`.
    5.  **提取 Logits**:
        *   从完整的序列 logits 中切片出仅对应 `Target` 部分的 logits.
        *   对于 Causal LM，这涉及到移位 (shift) ，因为位置 $i$ 的输出预测的是位置 $i+1$ 的 token。
    6.  **计算结果**:
        *   **Sum LogProb**: 将 Target 所有 token 的对数概率求和。
        *   **Greedy Match**: 检查模型预测的 token 是否与 Target 完全一致 (用于 Exact Match 指标).
    7.  **返回**: 返回 `(log_probability, is_exact_match)` 元组。

2. `multiple_choice` (多选)
此类型本质上是多个 `loglikelihood` 请求的集合。
    *   在 `Task` 层面，一个多选问题 (例如 A, B, C, D 四个选项) 会被转化为 4 个独立的 `loglikelihood` 请求: 
        1.  (Context, "Option A")
        2.  (Context, "Option B")
        3.  ...
    *   `HFLM` 接收到这些请求后，处理流程与上述 `loglikelihood` 完全一致。
    *   **优化**: `HFLM` 中的 `_lookup_one_token_cont` 优化逻辑可以检测到共享相同上下文的请求，从而避免重复计算上下文部分的 KV Cache (如果启用).

3. `loglikelihood_rolling` (滚动对数似然 / 困惑度)
用于计算长文本的困惑度 (Perplexity) ，通常通过滑动窗口方式进行。
    1.  **滑动窗口构建**:
        *   遍历每个请求的文本。
        *   使用 `utils.get_rolling_token_windows` 将长文本切分为多个重叠或相连的窗口。
        *   每个窗口包含 `(Context, Target)`，其中 Context 是上一个窗口的末尾。
    2.  **扁平化**: 将所有请求的所有窗口展平为一个巨大的列表 `all_windows`.
    3.  **分布式填充**: 如果是多 GPU，计算并添加填充窗口以对齐批次。
    4.  **批处理推理**:
        *   按 `batch_size` 遍历 `all_windows`.
        *   调用 `self._loglikelihood_tokens` 计算每个小窗口的对数似然。
    5.  **重组结果**:
        *   移除填充。
        *   根据 `request_window_counts` 将扁平化的结果重新组合回对应的原始请求。
        *   对每个请求的所有窗口的 loglikelihood 求和，得到整段文本的总 loglikelihood.
    6.  **返回**: 返回每个请求的总 loglikelihood 列表。

4. `generate_until` 用于自由文本生成任务。
    1.  **排序与分组**:
        *   按输入长度降序排序。
        *   按 `gen_kwargs` (生成参数，如 temperature, top_p, stop sequences) 分组。因为同一个批次内的生成参数必须一致。
    2.  **批处理**: 按组分块。
    3.  **编码**:
        *   编码 Context.
        *   处理左截断 (Left Truncation): 如果 Context 太长，保留最后 `max_length - max_gen_toks` 个 token。
    4.  **模型生成**:
        *   调用 `self._model_generate`.
        *   传递 `stopping_criteria` (处理 `until` 停止词).
        *   传递 `generation_kwargs`.
    5.  **解码与后处理**:
        *   将生成的 Token ID 解码为字符串。
        *   **截断**: 去除 Context 部分 (如果是 Causal LM).
        *   **停止词处理**: 截断在 `until` 停止词之后的内容。
        *   **Thinking Token 处理**: 如果模型有“思考”过程 (如 DeepSeek R1) ，根据 `think_end_token` 移除思考部分的输出。
    6.  **返回**: 返回生成的字符串列表。

# YAML 配置参数解析

1. **基础信息 (Basic Info)**
    *   **`task`**: (必填) 任务的唯一标识符名称。例如 `gsm8k`.
    *   **`task_alias`**: (可选) 任务别名。在结果表格中显示的名称，用于简化显示 (例如将 `mmlu_abstract_algebra` 显示为 `abstract_algebra`).
    *   **`tag`**: (可选) 标签列表。用于对任务进行分类 (例如 `math`, `science`) ，方便批量运行。
    *   **`group`**: (可选) 组名。如果这是一个组配置文件，使用此字段定义组名。
    *   **`metadata`**: (可选) 元数据字典，通常包含 `version` (版本号).

2. **数据集配置 (Dataset Configuration)**
    *   **`dataset_path`**: (必填) Hugging Face Hub 上的数据集名称 (如 `gsm8k`) 或本地加载脚本路径 (如 `json`, `arrow`).
    *   **`dataset_name`**: (可选) 数据集的具体配置名称 (Subset/Config).例如 MMLU 有多个子任务，这里指定具体是哪一个。如果不需要，设为 `null`.
    *   **`dataset_kwargs`**: (可选) 传递给 `datasets.load_dataset` 的额外参数字典。
        *   `data_dir`: 指定本地数据目录。
        *   `data_files`: 指定本地文件路径。
    *   **`training_split`**: 训练集拆分名称 (如 `train`).
    *   **`validation_split`**: 验证集拆分名称 (如 `validation`).
    *   **`test_split`**: 测试集拆分名称 (如 `test`).评估通常优先使用此拆分。
    *   **`fewshot_split`**: 用于抽取少样本 (Few-shot) 示例的拆分名称。
    *   **`process_docs`**: (可选) 数据预处理函数。使用 `!function` 引用 Python 函数，用于清洗或转换原始数据。

3. **提示词与格式 (Prompt & Formatting)**
    *   **`doc_to_text`**: (必填) 定义输入给模型的 Prompt 格式。
        *   可以是字段名字符串。
        *   可以是 Jinja2 模板 (如 `"{{passage}}\nQuestion: {{question}}?\nAnswer:"`).
        *   可以是 Python 函数 (使用 `!function`).
    *   **`doc_to_target`**: (必填) 定义期望的目标输出 (Ground Truth).
        *   对于生成任务：目标文本字符串或模板。
        *   对于多选任务：正确选项的索引 (整数) 或对应的标签字段。
    *   **`doc_to_choice`**: (多选任务必填) 定义选项列表。
        *   可以是静态列表 (如 `['A', 'B', 'C', 'D']`).
        *   可以是 Jinja2 模板或字段名。
    *   **`target_delimiter`**: (可选) 输入和目标之间的分隔符，默认为空格 `" "`.

4. **评估指标 (Metrics)**
    *   **`metric_list`**: (必填) 定义要计算的指标列表。每个指标包含：
        *   **`metric`**: 指标名称 (如 `acc` (准确率), `exact_match`, `bleu`).也可以是自定义函数。
        *   **`aggregation`**: 聚合方式 (如 `mean`).
        *   **`higher_is_better`**: 布尔值，`true` 表示分数越高越好。
    *   **`output_type`**: (通常由类自动推断，也可指定) 任务输出类型，如 `generate_until` (生成) , `multiple_choice` (多选) , `loglikelihood`.

5. **生成配置 (Generation Config - 仅生成任务)**
    *   **`generation_kwargs`**: 传递给模型生成函数的参数字典。
        *   `max_gen_toks`: 最大生成 token 数。
        *   `do_sample`: 是否采样。
        *   `temperature`: 温度参数。
        *   `until`: 停止词列表。

6. **过滤器 (Filters - 可选)**
    *   **`filter_list`**: 定义对模型输出进行后处理的过滤器列表。
        *   `name`: 过滤器名称 (如 `flexible-extract`).
        *   `filter`: 过滤器类型 (如 `regex`).
        *   `regex_pattern`: 正则表达式模式。

7. **高级/其他**
    *   **`num_fewshot`**: 默认的少样本数量。
    *   **`class`**: (高级) 指定自定义 Python 任务类的路径 (使用 `!function`) ，用于无法仅通过 YAML 配置的复杂任务。