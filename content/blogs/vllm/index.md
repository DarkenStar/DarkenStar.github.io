---
title: VLLM Sourse Code Reading
date: 2025-06-07T18:15:55+08:00
lastmod: 2025-06-07T18:15:55+08:00
author: ["WITHER"]

categories:
  - Source Code Reading

tags:
  - vllm

keywords:
  - vllm

description: "vllm structure" # 文章描述，与搜索优化相关
summary: "vllm structure" # 文章简单描述，会展示在主页
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

# Basic

```python {linenos=true}
from vllm import LLM, SamplingParams

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
```

# Architecture
![VLLM Architecture Overview](https://note.youdao.com/yws/api/personal/file/WEB713ed0d773cac101706cdaa862d71dda?method=download&shareKey=09c7c358d0427427384e027f0ced662a "VLLM Architecture Overview")
- LLM: 最上层的类，构造函数中会根据传入的参数构建 EngineArgs 然后创建 LLMEngine 对象。
- LLMEngine: 包含一些组件 InputPreprocessor, ExecutorBase 负责模型推理的最上层的类
- ExecutorBase 会初始化 N 个 WorkerWrapperBase (包装实际的 worker，类比成 GPU)
  - Worker: 在 GPU 上执行 (一部分) 模型推理。每个 worker 与一个 GPU 相关联，负责维护 KV Cache 并在 GPU 上执行模型推理。在分布式推理的情况下，每个 worker 被分配模型的一部分。
    - ModelRunner:  执行模型推理并负责采样新 token.
    - CacheEngine: 负责初始化和管理 GPU 和 CPU KV Cache. 还提供了对 KV Cache 进行操作的方法。通过 `initialize_cache()` 初始化。
- Scheduler: 负责推理时候对请求的调度。组件包括一个 BlockSpaceManager (KV Cache blocks 管理的核心类) 以及三个队列 waiting, running & swapped.

# LLMEngine  Initialization

- InputPreprocessor: 主要是在 `add_request()` 方法中将输入的 prompt 放入 tokenizer 进行处理。
- InputRegistry: 根据目标模型对 InputPreprocessor 之后的数据进行处理。

## Init Executor

```python {linenos=true}
class DistributedExecutorBase(ExecutorBase):
    """Abstract superclass of distributed executor implementations."""

    def __init__(self, *args, **kwargs):
        # This is non-None when the execute model loop is running
        # in the parallel workers. It's a coroutine in the AsyncLLMEngine case.
        self.parallel_worker_tasks: Optional[Union[Any, Awaitable[Any]]] = None

        super().__init__(*args, **kwargs)
```

ExecutorBase 的构造函数中会调用 `self._init_executor()` 对应到具体子类的函数。如果采用 TP 或 PP 的话 对应到的是 RayDistributedExecutor，否则对应到的是 UniProcExecutor. 下面以后者为例。

```python {linenos=true}
class UniProcExecutor(ExecutorBase):

    uses_ray: bool = False

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """
        self.driver_worker = WorkerWrapperBase(vllm_config=self.vllm_config,
                                               rpc_rank=0)
        distributed_init_method = get_distributed_init_method(
            get_ip(), get_open_port())
        local_rank = 0
        # set local rank as the device index if specified
        device_info = self.vllm_config.device_config.device.__str__().split(
            ":")
        if len(device_info) > 1:
            local_rank = int(device_info[1])
        rank = 0
        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=(not self.parallel_config)
            or (rank % self.parallel_config.tensor_parallel_size == 0),
        )
        self.collective_rpc("init_worker", args=([kwargs], ))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")

    def collective_rpc(self,
                       method: Union[str, Callable],
                       timeout: Optional[float] = None,
                       args: Tuple = (),
                       kwargs: Optional[Dict] = None) -> List[Any]:
        if kwargs is None:
            kwargs = {}
        answer = run_method(self.driver_worker, method, args, kwargs)  # 初始化 Worker
        return [answer]
```

- Executor: 初始化具体的继承自 ExecutorBase 的对象，该对象的初始化过程中会调用 `init_worker()` 初始化 Worker (被 WorkerWrapperBase 包装)，调用 `init_device()` 初始化设备，和调用具体 Worker 对象的 model_runner 的 `load_model()` 将模型加载到设备上。
  - Worker: 构造函数中会初始化 `GPUModelRunnerBase` 对象，确定计算 attention 使用的 backend 还有 CUDAGraphRunner 用于将模型的计算过程记录为一个静态图，在后续的推理中，通过直接 replay 这个静态图来避免动态调度和重复的内核启动开销。

## initialize_kv_caches

LLMEngine 构造函数在初始化 ExecutorBase 后会调用 `initialize_kv_caches()` 来初始化 Worker 中的 KV Cache，流程如下:

1. 该函数会首先通过 [Worker.determine_num_available_blocks()](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/worker/neuron_worker.py#L69) 确定 GPU 和 CPU 可用的 block 数量。后者在 `memory_profiling` 上下文中进行 [profile_run()](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/worker/model_runner.py#L1239) 模拟模型在最大负载 (max_num_batched_tokens 和 max_num_seqs) 下执行一次推理。测量内存使用并分解为权重、激活张量和非 PyTorch 部分。留给 KV Cache 的内存大小为 `total_mem * max_utilization - weight_mem - act_mem - nontorch_mem`.  再除以每一个 block 能存储的的 KV Cache 大小 `cache_size = Cache_config.block_size * num_attention_layers * 2*num_heads*head_size` 即可得到最多能分配多少个 GPU block. 而 CPU block 数量由预设的 `swap_size // cache_size` 所确定。
2. 确定了 GPU 和 CPU 的 block 数量后会调用 [Worker.initialize_cache()](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/worker/worker.py#L285) 方法，里面首先会调用 `Worker._init_cache_engine()` 根据传入的 GPU block 个数初始化 CacheEngine (初始化 attn_backend，调用 [CacheEngine._allocate_kv_cache()](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/worker/cache_engine.py#L68) 为模型的每一层 transformer 开辟 CPU 和 GPU 的 KV Cache 内存)，然后会调用 [bind_kv_cache()](https://github.com/vllm-project/vllm/blob/main/vllm/utils.py#L2163) 将 GPU KV Cache Tensor 绑定到对应的模型的注意力层，它筛选需要 KV Cache 的注意力层，按层索引排序并去重后为每个设备绑定对应的 Tensor.
3. 预热之后进行 capture_model 记录计算图。

## Init Scheduler

构造函数中会初始化 [BlockSpaceManager](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/core/block_manager.py#L61). 首先会创建一个 `CpuGpuBlockAllocator`，为 CPU 和 GPU 块维护单独的内存池，并允许在这些内存池中分配、释放、分叉和交换块。它会为 CPU 和 GPU 中的 blocks 分别创建一个 `BlockAlloctor`. 还会初始化一个空的 `Dict[SeqId, BlockTable]`， 表示对应 seq 的 KV Cache 所使用的物理内存块。还会初始化一些调度时所需要的数据，后文再谈。

还会初始化 waiting(包含新的或 preempted prefill 请求), running & swapped(被换出的 decoding 请求), 它们是 `Deque[SequenceGroup]`，其中 [SequenceGroup(SG)](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/sequence.py#L633) 是一组由同一个 prompt 生成的 Sequences 和对应的采样参数。

- SequenceGroupOutputProcessor: 抽象基类借接口，会分为 SingleStepOutputProcessor (支持 beam seaching) 和 MultiStepOutputProcessor (支持 speculatice decoding)

# LLM Generate

## _validate_and_add_requests

里面会调用 `_add_request()` 给 prompt 分配 reqest_id 后会调用 `LLMEngine.add_request()` 将其添加到请求池中，并将在调用 `LLMEngine.step()` 时由调度器处理。确切的调度策略由调度程序确定。主要就是进行 tokenize，然后打包成 SG 后加入 waiting.

## __run_engine

调用 generate 时首先会将 prompt 包装成 SG，它是包含某个 prompt 生成的所有 Sequence，以及一些其他在调度时需要的信息的结构。Scheduler 里面包含三个 `Deque[SequenceGroup]`: waiting, running & swapped.
generate() --> _run_engine() --> step() --> Scheduler.schedule() --> Scheduler._schedule()
Scheduler 的一些操作与 BlockManager 息息相关，我们在下面先简要说明逻辑，有关其具体结构和操作流程在后文中解释。

## step

执行一次 decoding 迭代并返回新生成的结果。
![Overview of the step function](https://i.imgur.com/sv2HssD.png "Overview of the step function")
主要流程如下

1. 调度要在下一次迭代中执行的 seq 和要交换入/出/复制的令牌块。根据调度策略，Sequences 可能被抢占/重新排序。
2. 调用分布式执行器来执行模型。
3. 处理模型输出。主要包括： decoding 相关输出，使用 _beam_search 与否的模型输出更新调度 seq 组和释放已完成的 seq 组。
4. 读取上一次调度的元数据和输出
5. 如果没有剩余步骤且，调用 `Scheduler.schedule()` 执行新调度，生成 seq 组元数据、调度输出和异步标志。
6. 获取并重置已完成请求 ID，清理内存
7. 如果不允许异步且有输出队列，处理模型输出。
8. 从 Cache 获取上一次迭代的 sampled_token_ids，构造 ExecuteModelRequest 后调用 `Executor.execute_model()` (最后是由 ModelRunner) 执行模型推理，获取输出。

## _schedule_prefill()

1. 检查 budget 是否耗尽
2. 取出队列head 部的 SequenceGroup (prefill 阶段 SequenceGroup 只有一个初始 prompt Sequence)
3. 计算 uncached 和 cached 的新 token 数
4. 调用 `BlockSpaceManager.can_allocate()` 检查是否能分配足够内存。
5. 若能满足 budget，从 waiting 中移除 SequenceGroup. 调用 `_allocate_and_set_running()` 分配内存并设置为 RUNNING 状态。

## _schedule_running()

1. 取出队列head 部 SequenceGroup 并计算其包含 seq 的 #uncached_token. 这里不需要 #cached_token 因为若使用 chunked prefill，该信息已经在第一次 prefill 时使用，如果不使用那么他就是进行 decoding 的 seq ，不需要用到这个信息。
2. 从 running 移除该 SequenceGroup. 循环调用 `Scheduler._can_append_slots()` 检查是否有足够的空间存储该 SequenceGroup 的 KV Cache，若不能，进入抢占逻辑
3. 从 budget 中减去当前 SequenceGroup 的 token 和 seq 数
4. 若 running 有其他 SequenceGroup，抢占最低优先级（队列尾部）的，若该 SequenceGroup 只有一个正在运行的 Sequence 则抢占模式为 RECOMPUTE 加入到 `preempted`，否则为 SWAP 加入到 `swapped_out`.
5. 分配 slot 并更新 blocks_to_copy，根据该 Sequence 处于 decoding(生成 1 个 token 的 KV Cache ) 或者 prefill(生成 #uncached_token 的 KV Cache) 加入到 `prefill_seq_group` 或者 `decode_seq_groups`，并更新 budget.
6. 返回 decode_seq_groups：存储 decoding  SequenceGroup. prefill_seq_groups：存储分块 prefill  SequenceGroup. preempted：被抢占需重新计算的 SequenceGroup. swapped_out：被交换到 CPU 的 SequenceGroup. keys_to_swap_out 和 keys_to_copy：内存块交换和复制的映射

## _schedule_swapepd()

1. 循环遍历 swapped 队列，取出队列head 部的 SequenceGroup，调用 `BlockManager.can_swap_in()` (实际上是 SWAPPED 状态的 `can_swap`)
2. 获取 SequenceGroup 中处于 SWAPPED 的 Sequence 个数和 token 个数，是否满足预算。
3. 调用 `_swap_in`(实际上是 `BlockManager.swap_in()`) 执行交换，更新 blocks_to_swap_in，将 Sequence 状态由 SWAPPED 变为 RUNNING.
4. 调用 `_append_slots` 给被换入的 Sequence 分配 block.
5. 根据 SequenceGroup 的状态添加到不同队列。
6. 返回blocks_to_swap_in：记录需要从 CPU 交换到 GPU 的块映射。blocks_to_copy：记录需要复制的块映射（例如写时复制）。decode_seq_groups 和 prefill_seq_groups：分别存储 decoding 和 prefill  SequenceGroup. infeasible_seq_groups：存储无法调度的 SequenceGroup. swapped_queue：引用交换队列。leftover_swapped：暂存无法立即调度的 SequenceGroup.

## _schedule_chunked_prefill()

主要思想是: 1.安排尽可能多的 decoding 请求。2.调度未完成的 prefill 请求。3.调度交换请求。4.安排新的 prefill 请求。

1. 初始化 budget，限制最大批处理 token 数和 seq 数。
2. 从 running 和 waiting 生成 `PartialPrefillMetadata`

- prefills: running 和 waiting 中未完成 prefill 的 #SequenceGroup.
- long_prefills: running 中需要进行 prefill 的 token 数很多的 #SequenceGroup.
- waiting_long_prefills: waiting 中需要进行且能进行的 (未超过 ScheduleConfig 限制) prefill 的 token 数很多的 #SequenceGroup.

3. 调用 `_schedule_running`.
4. 在 running 调度返回中无无抢占或交换时(说明有足够空间) 执行 `_schedule_swapped`
5. 调用 `_schedule_prefills`.
6. 更新 waiting，添加 running 调度中返回的被抢占的 seq  `running_scheduled.preempted`.
7. 按优先级更新 running.
8. swapped_in.decode_seq_groups：交换回来的 decoding 请求。
9. swapped_in.prefill_seq_groups：交换回来的 prefill 请求。
10. running_scheduled.decode_seq_groups：运行中的 decoding 请求。
11. running_scheduled.prefill_seq_groups（按完成顺序）：未完成的分块 prefill 。使用 _order_finishing_prefills_first 确保即将完成的 prefill 优先，便于下一轮转为 decoding.
12. prefills.seq_groups：新 prefill 请求。
13. 将运行队列中交换出去的 `running_scheduled.swapped_out` 添加到 swapped.
14. 按顺序组合所有调度的 SequenceGroup: prefill 优先（满足注意力机制假设），decoding 次之。
15. 调整 lookahead_slots 数量。若所有被调度的均为 prefill 且未启用多步调度，设置 num_lookahead_slots = 0(避免推测 decoding 路径). 否则，使用 running 计算的 lookaheadh slots 数量。

## _schedule_default

尽可能多地批处理 prefill 请求，然后调度 decoding 请求. 在 GPU 内存压力下，需要 preempt 或 swap out 运行中的 decoding 请求。

1. swapped 为空则进行 `_schedule_prefills`.
2. 如果没有调度任何 prefill 请求，调用 `_schedule_running`.
3. 如果 running 调度结果中没有发生抢占或换出时 (否则说明资源不够)，执行 `_schedule_swapped`.
4. 更新 waiting, running & swapped 三个队列。

## After schedule

调度结果返回后，

1. 遍历调度结果中的 SequenceGroup
2. 遍历该 SequenceGroup 中状态为 RUNNING 的 Sequence. 获取其数据，对应的 BlockID 列表，并更新其访问时间。若使用 prefix_caching, 则调用 `BlockManager.get_common_computed_block_ids()` 获取共享的已计算的部分的 BlockID 列表。
3. 如果该 SequenceGroup 处于 prefill 阶段，则判断这次调度后是否能完成 prefill.
4. 构造返回结果，标记所有调度 SequenceGroup 的 blocks 为已计算。

# BlockSpaceManager

用于将 SequenceGroup 操作映射到其包含的对应组件的操作。

- CpuGpuBlockAlloctor: 根据是否采用 prefix caching 分别为 CPU 和 GPU 初始化一个 Alloctor
  - [PrefixCachingBlockAlloctor](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/core/block/prefix_caching_block.py#L53): 基于哈希值维护 block 的Cache)重用具有相同哈希值的 block，以避免冗余的内存分配。
    - `Dict[PrefixHash, BlockId]` 将用于 prefix caching blocks 的哈希值与其 BlockID 对应。
    - `Dict[BlockId, BlockTracker]` 为每个物理 block 初始化一个 BlockTracker.
    - [NaiveBlockAllocator](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/core/block/naive_block.py#L13) 用于分配不作为 prefix caching 的 blocks. 有一个 `RefCounter` 表示某个物理 block 被多少逻辑 block 指向。
    - `Evictor` 采用 LRU 策略驱逐已经Cache) blocks.
    - `CopyOnWriterTracker` 用于将原先的 block ID 映射到目的 block ID.
- Dict[SeqId, BlockTable]: [BlockTable](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/core/block/block_table.py#L11) 用于将单个 seq 的 KV Cache 映射到物理内存分配。会在调用 [_allocate_sequence()](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/core/block_manager.py#L148) 时被初始化。包含一个 [BlockList](https://github.com/vllm-project/vllm/blob/main/vllm/core/block/common.py#L231) (block 列表和一个表示对应 ID 的 int 列表) 和 BlockpaceManager 的 BlockAllocator.
- ComputedBlocksTracker: 维护一个 `Dict[SeqId, List[int]]` ( seq id到 seq 块哈希列表的映射)。Cache)个 seq 的完整块 (块全部被占满) 的哈希值。当一个 seq 进行 decoding 时，也相应更新 seq 的哈希值。还有一个 `Dict[int, int]` ( seq id到已计算 token 数的映射)

## can_allocate

在 `_schedule_prefills` 中被调用。

```python {linenos=true}
def can_allocate(self,
                seq_group: SequenceGroup,
                num_lookahead_slots: int = 0) -> AllocStatus:
```

1. 取出该 SequenceGroup 中处于 WAITING 状态的第一个 Sequence (i.e. prompt).
2. 调用 `BlockTable.get_num_required_blocks()` 计算存储 token 和 lookahead slots 所需的最小 block 数 (假设无 prefix caching), i.e. `cdiv(len(token_ids) + num_lookahead_slots, block_size)`.
3. 调用 `BlockAlloctor.get_num_free_blocks()` 获取 GPU 上空闲的 block 数 (非 prefix_caching 中的空闲个数 + 可以被驱逐的个数).
4. 返回分配状态

- NEVER: `#total - #required < #watermark`
- OK: `#free  - #required >= #watermark`
- LATER: `#free  - #required < #watermark`

## allocate

```python {linenos=true}
def allocate(self, seq_group: SequenceGroup) -> None:
```

在 `_schedule_prefills` 中步骤 4 中调用的 `_allocate_and_set_running` 内部被调用。

1. 取出该 SequenceGroup 中处于 WAITING 状态的第一个 Sequence (i.e. prompt).
2. 调用 `BlockManager._allocate_sequence()` 创建一个 BlockTable，在获取 token_ids 列表后调用 `BlockTable.allocate()` 为该 Sequence 分配 blocks.
3. 将 token_ids 按 _block_size 大小进行分块。最后一块可能不能占满一个 block.
4. 对于能够占满一个 block 的 token_ids 分块，调用 `BlockAlloctor.allocate_immutable_block()`. 该函数优先从Cache)查找是否已有相同内容的块，若有则直接复用该块并增加其引用计数；否则调用 `BlockAlloctor.allocate_mutable_blocks()` 分配一个新的 block，并将 token_ids 添加到该 block 中. 该函数会尝试从非 prefix caching blocks 中分配一个 block_id，若没找到则会驱逐一个。
5. 对于最后一个可能被没占满的 block 调用 `BlockAlloctor.allocate_mutable_blocks()`.

## can_append_slots

```python {linenos=true}
def can_append_slots(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> bool:
```

确定 GPU KV Cache 中是否有足够的空间来继续生成指定的 SequenceGroup. 上层接口为 `Scheduler._can_append_slots()`，在 `_schedule_running` 中步骤 2 中确定是否需要进行抢占时被调用。

1. 遍历该 Sequence Group 中处于 RUNNING 状态的 Sequence 对应的 BlockTable
2. 调用 `BlockTable.get_unseen_token_ids()` 获取该 Sequence 还未被Cache) token 部分。
3. 调用 `BlockTable.get_num_blocks_touched_by_append_slots()` 获取Cache)余部分和 lookahead 部分需要几个 block.
4. 调用 `BlockAlloctor.get_num_free_blocks()` 获取 GPU 上空闲的 block 数.
5. 需要个数小于空闲个数返回 True.

## append_slots

```python {linenos=true}
def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:
```

上层接口为 `Scheduler._append_slots()`. 在 `_schedule_running` 中检查到有空间添加，`_schedule_swapped` 中有 budget 进行换入，`_schedule_prefills` 中允许进行 chunked prefill 时被调用。

1. 调用 `BlockTable.append_token_ids()`. 该方法将 tokens 添加到 BlockTable 中的现有 block 中。会调用 `BlockTable.ensure_num_empty_slots()`， 它查看当前能够容纳多少个 token. 如果没有足够的空间，则使用 `BlockAlloctor.allocate_mutable_block()` 方法分配新 block.
2. 调用 `BlockAllocator.clear_copy_on_writes()` 返回一个映射源 block ID 到当前 COW 的目标 block ID 的元组的列表.

## _can_swap

```python {linenos=true}
def _can_swap(self,
              seq_group: SequenceGroup,
              device: Device,
              status: SequenceStatus,
              num_lookahead_slots: int = 0) -> AllocStatus:
```

根据 status 区分上层接口: RUNNING/SWAPPED 表示需要把该 SequenceGroup 处于 RUNNING/SWAPPED 状态的 Sequence 对应的 blocks 从 GPU/CPU 换到 CPU/GPU.

1. 获取 SequenceGroup 中符合指定状态的 seq  Sequence，然后根据 SeqID 获取对应的 BlockTable.
2. 调用 `BlockTable.get_num_blocks_touched_by_append_slots()` 计算添加未存储 token 加上 lookahead_slots 所需的 block 数量。
3. 调用 `BlockAlloctor.get_num_full_blocks_touched()` 获取当前有被使用的 block 数量。
4. 如果总块数小于被使用的加上需要的 block 数量 返回 Never. 如果空闲块减去 被使用的加上需要的 block 数量后仍大于等于 watermark_blocks，返回 OK. 否则为 LATER.

## swap_in

调用的是  `self.block_allocator.swap(blocks=blocks, src_device=Device.CPU, dst_device=Device.GPU)`，即 blocks 从原设备的换出，换入到目的设备。
进一步则是 `BlockAlloctor.swap_in()`，该函数遍历传入的 blocks，若已经被占满调用 `BlockAlloctor.allocate_immutable_block()`. 否则调用 `BlockAlloctor.allocate_mutable_blocks()` 分配一个新的 block 后将原 block的 token 数据追加到新 block.

## swap_out

同上，最终调用的是 `BlockAlloctor.swap_out()`. 该函数对传入的每个 block 调用 `_free_block_id`，逐个处理释放逻辑。若 block 有哈希值，refcount -1，若减去后为 0 则将 block 信息添加到 evictor 中，从跟踪系统中移除，然后设置 BlockId 为 None. 否则就直接设置为 None. 若无哈希值则释放 BlockId，减去对应的 refcount，但保留 block 对象本身.

# Attention

[XFormersImpl](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/attention/backends/xformers.py#L354) 中使用了 vllm 自己写的 PagedAttention kernel.

```python {linenos=true}
class XFormersImpl(AttentionImpl[XFormersMetadata]):
  def __init__(
      self,
      num_heads: int,
      head_size: int,
      scale: float,
      num_kv_heads: int,
      alibi_slopes: Optional[List[float]],
      sliding_window: Optional[int],
      kv_cache_dtype: str,
      blocksparse_params: Optional[Dict[str, Any]] = None,
      logits_soft_cap: Optional[float] = None,
      attn_type: str = AttentionType.DECODER,
  ) -> None:
```

其中 `attn_type` 分为四种，下面我们主要分析 DECODER 的情况。

- DECODER: 使用 decoding 器的 self-attention block table 来Cache)KV(GPT).
- ENCODER: 不进行 KV Cache)用于 Encoder-Decoder 模编码器分支。编码器通常一次性处理整个输入 seq 。
- ENCODER-ONLY: 不进行 KV Cache)BERT).
- ENCODER_DECODER: 用于编码器- decoding 器模型中的交叉注意力部分，其中 KV  seq 长度与编码器 seq 长度一致(T5).

```python {linenos=true}
def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads * head_size]
        key: Optional[torch.Tensor],  # [num_tokens, num_kv_heads * head_size]
        value: Optional[torch.Tensor],  # [num_tokens, num_kv_heads * head_size]
        kv_cache: torch.Tensor,  # [2, num_blocks, block_size * num_kv_heads * head_size]
        attn_metadata: "XFormersMetadata",
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
```

[AttentionMetadata](https://github.com/vllm-project/vllm/blob/5eeabc2a4400fde9b030f2f72746a2b03db059bd/vllm/attention/backends/.py#L104) 类定义如下

```python {linenos=true}
@dataclass
class AttentionMetadata:
    """Attention metadata for prefill and decode batched together."""
    num_prefills: int  # prefill 请求的总数
    num_prefill_tokens: int  # 所有 prefill 请求中的 token 总数。
    num_decode_tokens: int  # decodeing token 的数量，等同于 decoding 请求的数量
    slot_mapping: torch.Tensor  # (num_tokens,)，指定每个输入 token 存储到 KV cache 中的 slot 索引
    # block_idx = x // block_size, block_offset = x % block_size
    multi_modal_placeholder_index_maps: Optional[Dict[
        str, MultiModalPlaceholderMap.IndexMap]]
    enable_kv_scales_calculation: bool
```

forward 方法如下，简化了成了 DECODER 情况的逻辑。
主要流程为

1. 调用 `PagedAttention.split_kv_cache` 分离并 reshape KV Cache 张量后 调用 PagedAttention.write_to_paged_cache`
   写入当前 key 和 value 到Cache)。
2. 分离 prefill 和 decoding 的 token，初始化输出。对于 prefill 部分根据是否采用了 prefix_caching 调用 `self._run_memory_efficient_xformers_forward` 或 `PagedAttention.forward_prefix` 计算注意力。
3. 调用 `get_seq_len_block_table_args` 获取 decoding Sequence 对应的 BlockTable后调用 `PagedAttention.forward_decode` 计算注意力。

```python {linenos=true}
def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,  # [num_tokens, num_heads * head_size]
        key: torch.Tensor,    # [num_tokens, num_kv_heads * head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads * head_size]
        kv_cache: torch.Tensor,  # [2, num_blocks, block_size * num_kv_heads * head_size]
        attn_metadata: "XFormersMetadata",
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
 
    # 将 query 重塑为 [num_tokens, num_heads, head_size]
    query = query.view(-1, self.num_heads, self.head_size)
    # key 和 value 必须非空（自注意力要求），重塑为 [num_tokens, num_kv_heads, head_size]
    key = key.view(-1, self.num_kv_heads, self.head_size)
    value = value.view(-1, self.num_kv_heads, self.head_size)

    # 如果 KV Cache)空，处理Cache)辑
    if kv_cache.numel() > 0:
        # 从 kv_cache 分离出 key_cache 和 value_cache
        # key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
        # value_cache: [num_blocks, num_kv_heads, head_size, block_size]
        key_cache, value_cache = PagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size)

        # 更新自注意力的 KV Cache)        # 使用 attn_metadata.slot_mapping 指定 token 存储位置
        PagedAttention.write_to_paged_cache(
            key, value, key_cache, value_cache, attn_metadata.slot_mapping,
            self.kv_cache_dtype, layer._k_scale, layer._v_scale)

    # 获取 prefill 和 decoding 阶段的 token 数量
    (num_prefill_query_tokens, num_prefill_kv_tokens, num_decode_query_tokens) = \
        get_num_prefill_decode_query_kv_tokens(attn_metadata, AttentionType.DECODER)

    # 创建输出张量与 query 相同
    output = torch.empty_like(query)

    # 分离 prefill 和 decoding 的 QKV
    decode_query = query[num_prefill_query_tokens:]  #
    query = query[:num_prefill_query_tokens]     
    key = key[:num_prefill_kv_tokens]             
    value = value[:num_prefill_kv_tokens]         

    # 处理 prefill 阶段（如果存在）
    if prefill_meta := attn_metadata.prefill_metadata:
        if kv_cache.numel() == 0 or prefill_meta.block_tables.numel() == 0:
            # 普通注意力（无Cache)缀）
            out = self._run_memory_efficient_xformers_forward(
                query, key, value, prefill_meta, attn_type=AttentionType.DECODER)
            output[:num_prefill_query_tokens] = out
        else:
            # 前缀Cache)意力
            out = PagedAttention.forward_prefix(
                query, key, value, self.kv_cache_dtype, key_cache, value_cache,
                prefill_meta.block_tables, prefill_meta.query_start_loc,
                prefill_meta.seq_lens_tensor, prefill_meta.max_query_len,
                self.alibi_slopes, self.sliding_window, layer._k_scale, layer._v_scale)
            output[:num_prefill_query_tokens] = out

    # 处理 decoding 阶段（如果存在）
    if decode_meta := attn_metadata.decode_metadata:
        # 获取 decoding 所需的 seq 长度和 BlockTable 参数
        seq_lens_arg, max_seq_len_arg, block_tables_arg = \
            get_seq_len_block_table_args(decode_meta, False, AttentionType.DECODER)

        # 运行 decoding 注意力
        output[num_prefill_query_tokens:] = PagedAttention.forward_decode(
            decode_query, key_cache, value_cache, block_tables_arg, seq_lens_arg,
            max_seq_len_arg, self.kv_cache_dtype, self.num_kv_heads, self.scale,
            self.alibi_slopes, layer._k_scale, layer._v_scale)

    # 将输出 reshape 为 [num_tokens, num_heads * head_size]
    return output.view(-1, self.num_heads * self.head_size)
```

## write_to_paged_cache

调用的是已经注册到 torch.ops 中的 CUDA 函数。其对应的 host 函数为每个 token 分配一个 CUDA block，每个 CUDA block 的线程数被限制在最多 512 个。主要的 kernel 函数如下。

```C {linenos=true}
// scalar_t: 输入 key 和 value 的数据类型（如 float、half）
// cache_t: Cache)key_cache 和 value_cache 的数据类型（如 half、uint8_t）
// kv_dt: KV Cache) FP8 数据类型（如 kAuto 或具体 FP8 格式）
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x, block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size, block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]，指定每个 token 的Cache)置
    const int key_stride, const int value_stride,  // key 和 value 在 token 维的步幅
    const int num_heads, const int head_size,      // 注意力head 数和每个head 的维度
    const int block_size, const int x,             // Cache)大小和 key_cache 中 head_size 的拆分因子
    const float* k_scale, const float* v_scale)    // key 和 value 的缩放因子，用于数据类型转换
  const int64_t token_idx = blockIdx.x;  // host 函数定义 block 个数与 token 个数相同
  const int64_t slot_idx = slot_mapping[token_idx]; {
  
  // Cache Block
  const int64_t block_idx = slot_idx / block_size;  // 块索引
  const int64_t block_offset = slot_idx % block_size;  // 块内偏移

  const int n = num_heads * head_size;  // 每个 token 的维度数目
  // CUDA Block 级别并行，每个线程处理token 的一个维度
  for (int i = threadIdx.x; i < n; i += blockDim.x) {  
    // 计算输入 key 和 value 的源索引
    const int64_t src_key_idx = token_idx * key_stride + i;
    const int64_t src_value_idx = token_idx * value_stride + i;

    // 计算当前处理的head 索引和head 内偏移
    const int head_idx = i / head_size;      // 第几个head 
    const int head_offset = i % head_size;   // head 内的第几个元素

    // 将 head_offset 拆分为 x_idx 和 x_offset（仅用于 key_cache）
    const int x_idx = head_offset / x;       // head_size/x 维的索引
    const int x_offset = head_offset % x;    // x 维的偏移

    // 计算 key_cache 的目标索引，按维度逐步偏移
    const int64_t tgt_key_idx =
        block_idx * num_heads * (head_size / x) * block_size * x +  // 块偏移
        head_idx * (head_size / x) * block_size * x +               // head 偏移
        x_idx * block_size * x +                                    // head_size/x 偏移
        block_offset * x + x_offset;                                // 块内和 x 偏移

    // 计算 value_cache 的目标索引，按维度逐步偏移
    const int64_t tgt_value_idx =
        block_idx * num_heads * head_size * block_size +            // 块偏移
        head_idx * head_size * block_size +                         // head 偏移
        head_offset * block_size +                                  // head_size 偏移
        block_offset;                                               // 块内偏移

    // 从输入张量读取当前元素
    scalar_t tgt_key = key[src_key_idx];
    scalar_t tgt_value = value[src_value_idx];

    // 根据 kv_dt 类型决定存储方式
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      // 如果是 kAuto，直接存储，不进行类型转换
      key_cache[tgt_key_idx] = tgt_key;
      value_cache[tgt_value_idx] = tgt_value;
    } else {
      // 否则，使用 scaled_convert 进行类型转换（如 FP8 量化）
      key_cache[tgt_key_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_key, *k_scale);
      value_cache[tgt_value_idx] =
          fp8::scaled_convert<cache_t, scalar_t, kv_dt>(tgt_value, *v_scale);
    }
  }
}
```

## _run_memory_efficient_xformers_forward

也同样简化成 DECODER 的逻辑的情况

```python {linenos=true}
def _run_memory_efficient_xformers_forward(
        self,
        query: torch.Tensor,  # [num_prefill_tokens, num_heads, head_size]
        key: torch.Tensor,    # [num_prefill_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_prefill_tokens, num_kv_heads, head_size]
        attn_metadata: "XFormersMetadata",
    ) -> torch.Tensor:

    original_query = query  # 保存原始 query，用于最后 reshape 输出

    # 处理 GQA/MQA
    if self.num_kv_heads != self.num_heads:
        # reshape Q to [num_prefill_tokens, num_kv_heads, num_queries_per_kv, head_size]
        query = query.view(query.shape[0], self.num_kv_heads,
                           self.num_queries_per_kv, query.shape[-1])
        # expand K to [num_prefill_tokens, num_kv_heads, num_queries_per_kv, head_size]
        key = key[:, :, None, :].expand(key.shape[0], self.num_kv_heads,
                                        self.num_queries_per_kv, key.shape[-1])
        # expand V to  [num_prefill_tokens, num_kv_heads, num_queries_per_kv, head_size]
        value = value[:, :, None, :].expand(value.shape[0], self.num_kv_heads,
                                            self.num_queries_per_kv, value.shape[-1])

    # 获取或设置 attention bias
    attn_bias = _get_attn_bias(attn_metadata, AttentionType.DECODER)
    if attn_bias is None:
        assert attn_metadata.seq_lens is not None  # 确保 seq 长度信息存在
        if self.alibi_slopes is None:
            # 创建 causal mask
            attn_bias = BlockDiagonalCausalMask.from_seqlens(
                attn_metadata.seq_lens, device=query.device)
            if self.sliding_window is not None:
                # 如果有滑动窗口，应用局部注意力
                attn_bias = attn_bias.make_local_attention(self.sliding_window)
            attn_bias = [attn_bias]
        else:
            # 使用 ALiBi 偏置（线性偏置注意力）
            attn_bias = _make_alibi_bias(self.alibi_slopes, self.num_kv_heads,
                                        query.dtype, attn_metadata.seq_lens)
        _set_attn_bias(attn_metadata, attn_bias, AttentionType.DECODER)

    # 执行 xFormers 高效注意力计算
    if self.alibi_slopes is None:
        # 为 QKV 添加 batch
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        out = xops.memory_efficient_attention_forward(
            query, key, value, attn_bias=attn_bias[0], p=0.0, scale=self.scale)
    else:
        # ALiBi 模式直接使用 attn_bias
        assert attn_metadata.seq_lens is not None
        output = torch.empty_like(original_query)
        start = 0
        # xformers 不支持在自定义 bias 的情况下每个 seq 的长度不同
        for i, seq_len in enumerate(attn_metadata.seq_lens): 
            end = start + seq_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=attn_bias[i],
                p=0.0,
                scale=self.scale)
            output[start:end].copy_(out.view_as(original_query[start:end]))
            start += seq_len

    # 将输出 reshape 为原始 query 
    return out.view_as(original_query)
```

## forward_prefix

不考虑 ALiBi 的情况调用的是 triton 编写的 [_fwd_kernel()](https://github.com/vllm-project/vllm/blob/d1695758b2f65fd314d1aee71ba2469ceba67a5b/vllm/attention/ops/prefix_prefill.py#L22) 每个线程块独立处理一个 Q 的一部分，对 KV Cache 和 当前 KV 分别采取 flash-attention 的计算策略。

```python {linenos=true}
import triton
import triton.language as tl

@triton.jit
def _fwd_kernel(
    # --- 输入张量 ---
    Q,  #  Query 张量: [total_seq_len, num_heads, head_dim]
        # total_seq_len 是所有 batch  seq 长度的总和，当前块为 [BLOCK_M, BLOCK_DMODEL_PADDED]
    K,  # 键张量（当前输入）: [total_seq_len, num_kv_heads, head_dim]
    V,  # 值张量（当前输入）: [total_seq_len, num_kv_heads, head_dim]
    K_cache,  # 键Cache) [num_blocks, num_kv_heads, head_dim, block_size, x]
              # 用于存储上下文部分的 K
    V_cache,  # 值Cache) [num_blocks, num_kv_heads, head_dim, block_size]
              # 用于存储上下文部分的 V
    B_Loc,  # 块索引表: [batch_size, max_seq_len // block_size]
            # 记录每个 batch 中每个块的块编号
    sm_scale,  # softmax 缩放因子，通常为 1/sqrt(head_dim)
    k_scale,  # 用于 FP8 精度转换的缩放因子
    v_scale,  # 用于 FP8 精度转换的缩放因子
    B_Start_Loc,  #  batch 起始位置: [batch_size + 1]
                  # 每个 batch 的全局 seq 起始索引，最后一个元素是总长度
    B_Seqlen,  #  batch  seq 长度: [batch_size]
               # 每个 batch 的总 seq 长度（上下文 +  Query ）
    block_size,  # 每个Cache)的大小
    x,  # K_cache 的额外维度分片因子（通常为 1 或小整数）
    Out,  # 输出张量: [total_seq_len, num_heads, head_dim]
          # 存储注意力计算结果
    # --- 步幅参数 ---
    stride_b_loc_b,  # B_Loc 的 batch 步幅
    stride_b_loc_s,  # B_Loc 的 seq 块步幅
    stride_qbs,  # Q 的 batch / seq 步幅，通常为 num_heads * head_dim
    stride_qh,   # Q 的head 步幅，通常为 head_dim
    stride_qd,   # Q 的head_size步幅，通常为 1
    stride_kbs,  # K 的 batch / seq 步幅
    stride_kh,   # K 的head 步幅
    stride_kd,   # K 的head_size步幅
    stride_vbs,  # V 的 batch / seq 步幅
    stride_vh,   # V 的head 步幅
    stride_vd,   # V 的head_size步幅
    stride_obs,  # Out 的 batch / seq 步幅
    stride_oh,   # Out 的head 步幅
    stride_od,   # Out 的head_size步幅
    stride_k_cache_bs,  # K_cache 的块步幅
    stride_k_cache_h,   # K_cache 的head 步幅
    stride_k_cache_d,   # K_cache 的head_size步幅
    stride_k_cache_bl,  # K_cache 的块内偏移步幅
    stride_k_cache_x,   # K_cache 的额外维度步幅
    stride_v_cache_bs,  # V_cache 的块步幅
    stride_v_cache_h,   # V_cache 的head 步幅
    stride_v_cache_d,   # V_cache 的head_size步幅
    stride_v_cache_bl,  # V_cache 的块内偏移步幅
    # --- 超参数 ---
    num_queries_per_kv: int,  # 每个 KV head 对应的 Query head 数量
    IN_PRECISION: tl.constexpr,  # 输入精度（例如 tl.float32）
    BLOCK_M: tl.constexpr,  #  Query 块大小
    BLOCK_DMODEL: tl.constexpr,  # head 维度大小
    BLOCK_DMODEL_PADDED: tl.constexpr,  # head 维度填充到 2 的幂次
    BLOCK_N: tl.constexpr,  # KV 块大小
    SLIDING_WINDOW: tl.constexpr,  # 滑动窗口大小，0 表示无窗口
    SKIP_DECODE: tl.constexpr,  # 是否跳过解码（仅处理上下文）
):
    # --- 网格定义 ---
    # grid = (batch_size, num_heads, max_seq_len // BLOCK_M)
    cur_batch = tl.program_id(0)  # 当前 batch 索引
    cur_head = tl.program_id(1)   # 当前head 索引
    start_m = tl.program_id(2)    # 当前 Query 块索引

    # --- 计算 KV head 索引 ---
    cur_kv_head = cur_head // num_queries_per_kv  # 当前 KV head 索引

    # --- 加载 batch 信息 ---
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)  # 当前 batch 总 seq 长度
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)  # 当前 batch 全局起始索引
    cur_batch_in_all_stop_index = tl.load(B_Start_Loc + cur_batch + 1)  # 下一 batch 起始索引
    cur_batch_query_len = (cur_batch_in_all_stop_index - 
                          cur_batch_in_all_start_index)  # 当前 batch  Query 长度
    cur_batch_ctx_len = cur_batch_seq_len - cur_batch_query_len  # 上下文长度

    # --- 计算 Query 块起始位置 ---
    block_start_loc = BLOCK_M * start_m  # 当前 Query 块的起始位置

    # --- 初始化索引范围 ---
    offs_n = tl.arange(0, BLOCK_N)  # KV 块内偏移: [0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)  # head_size 偏移: [0, BLOCK_DMODEL_PADDED)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)  #  Query 块内偏移: [start_m * BLOCK_M, (start_m + 1) * BLOCK_M)

    # --- 计算 Q 的偏移量 ---
    # off_q: [BLOCK_M, BLOCK_DMODEL_PADDED]
    # 定位当前 Query 块在 Q 张量中的内存地址
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +  #  batch 和 seq 偏移
        cur_head * stride_qh +  # head 偏移
        offs_d[None, :] * stride_qd  # head_size偏移
    )
    # 示例: 假设 Q [100, 4, 64], stride_qbs=256, stride_qh=64, stride_qd=1
    # cur_batch_in_all_start_index=20, cur_head=1, start_m=1, BLOCK_M=16
    # offs_m=[16, 17, ..., 31], offs_d=[0, 1, ..., 63]
    # off_q[0, 0] = (20 + 16) * 256 + 1 * 64 + 0 * 1 = 9216 + 64 = 9280
    # off_q[0, 1] = (20 + 16) * 256 + 1 * 64 + 1 * 1 = 9281

    # --- 创建head_size维度掩码 ---
    dim_mask = tl.where(tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(tl.int1)  # [BLOCK_DMODEL_PADDED]
    # 屏蔽填充部分，例如 BLOCK_DMODEL=64, BLOCK_DMODEL_PADDED=128，则后 64 个值为 0

    # --- 加载 Q 数据 ---
    q = tl.load(Q + off_q,
                mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len),
                other=0.0)  # [BLOCK_M, BLOCK_DMODEL_PADDED]
    # 加载当前 Query 块，掩码确保不加载超出 Query 长度和填充维度的数据

    # --- 初始化online softmax 变量 ---
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # 最大值
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # 归一化因子
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)  # 注意力累加

    # --- 计算上下文注意力（Q 对 KV Cache) ---
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)  # 确保 start_n 是 BLOCK_N 的倍数
        # --- 加载 Cache 索引 ---
        bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                     ((start_n + offs_n) // block_size) * stride_b_loc_s,
                     mask=(start_n + offs_n) < cur_batch_ctx_len,
                     other=0)  # [BLOCK_N]
        # bn 是当前 KV Cache的块编号
        # 示例: B_Loc=[0, 1, 2, ...], cur_batch=0, start_n=16, block_size=16, offs_n=[0, 1, 2, 3]
        # bn = B_Loc[0, 1]（若 stride_b_loc_b=8, stride_b_loc_s=1，则地址为 0*8 + 1*1 = 1）

        # --- 计算 K_cache 偏移量 ---
        # off_k: [BLOCK_DMODEL_PADDED, BLOCK_N]
        off_k = (
            bn[None, :] * stride_k_cache_bs +  # 块偏移
            cur_kv_head * stride_k_cache_h +   # head 偏移
            (offs_d[:, None] // x) * stride_k_cache_d +  # head_size偏移（分片）
            ((start_n + offs_n[None, :]) % block_size) * stride_k_cache_bl +  # 块内偏移
            (offs_d[:, None] % x) * stride_k_cache_x  # 额外维度偏移
        )
        # 示例: bn=[1], cur_kv_head=1, stride_k_cache_bs=4096, stride_k_cache_h=1024, stride_k_cache_d=16
        # offs_d=[0, 1, ..., 63], start_n=16, offs_n=[0, 1, 2, 3], block_size=16, x=1
        # off_k[0, 0] = 1*4096 + 1*1024 + (0//1)*16 + (16+0)%16*256 + (0%1)*1 = 4096 + 1024 = 5120

        # --- 加载 K_cache 数据 ---
        k_load = tl.load(K_cache + off_k,
                         mask=dim_mask[:, None] & ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                         other=0.0)  # [BLOCK_DMODEL_PADDED, BLOCK_N]
        # 处理 FP8 精度
        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * tl.load(k_scale)).to(q.dtype)
        else:
            k = k_load

        # --- 计算 QK 注意力分数 ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)  # [BLOCK_M, BLOCK_N]
        qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk, float("-inf"))
        qk *= sm_scale
        if SLIDING_WINDOW > 0:
            qk = tl.where((cur_batch_ctx_len + offs_m[:, None]) - 
                          (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk, -10000)

        # --- online softmax 更新 ---
        m_ij = tl.max(qk, 1)  # [BLOCK_M]
        p = tl.exp(qk - m_ij[:, None])  # [BLOCK_M, BLOCK_N]
        l_ij = tl.sum(p, 1)  # [BLOCK_M]
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # --- 更新累加器 ---
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # 加载 V_cache
        off_v = (
            bn[:, None] * stride_v_cache_bs +
            cur_kv_head * stride_v_cache_h +
            offs_d[None, :] * stride_v_cache_d +
            (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl
        )
        v_load = tl.load(V_cache + off_v,
                         mask=dim_mask[None, :] & ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                         other=0.0)  # [BLOCK_N, BLOCK_DMODEL_PADDED]
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * tl.load(v_scale)).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)

        # 更新 m_i 和 l_i
        l_i = l_i_new
        m_i = m_i_new

    # --- 计算自注意力（Q 对当前 K 和 V） ---
    # 计算 K 和 V 的初始偏移
    off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
             offs_d[:, None] * stride_kd)  # [BLOCK_DMODEL_PADDED, BLOCK_N]
    off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
             offs_d[None, :] * stride_vd)  # [BLOCK_N, BLOCK_DMODEL_PADDED]
    k_ptrs = K + off_k  # 初始指针
    v_ptrs = V + off_v

    # 检查当前 Query 块是否有效
    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0)

    # 遍历当前输入的 K 和 V
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # --- 加载 K 数据 ---
        # 全局偏移: (cur_batch_in_all_start_index + start_n) * stride_kbs 定位 batch 和 seq 块
        # 示例: K [100, 4, 64], stride_kbs=256, cur_batch_in_all_start_index=20, start_n=8
        # 基地址偏移 = (20 + 8) * 256 = 7168
        # k_ptrs[0, 0] = K + 0 + 1*64 + 0*1 + 7168 = K + 7232
        k = tl.load(k_ptrs + (cur_batch_in_all_start_index + start_n) * stride_kbs,
                    mask=dim_mask[:, None] & ((start_n + offs_n[None, :]) < cur_batch_query_len),
                    other=0.0)  # [BLOCK_DMODEL_PADDED, BLOCK_N]

        # --- 计算 QK 注意力分数 ---
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        qk *= sm_scale
        # 应用因果掩码
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk, float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where(offs_m[:, None] - (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk, -10000)

        # --- online softmax 更新 ---
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij

        # --- 更新累加器 ---
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        v = tl.load(v_ptrs + (cur_batch_in_all_start_index + start_n) * stride_vbs,
                    mask=dim_mask[None, :] & ((start_n + offs_n[:, None]) < cur_batch_query_len),
                    other=0.0)  # [BLOCK_N, BLOCK_DMODEL_PADDED]
        p = p.to(v.dtype)
        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)

        # 更新 m_i 和 l_i
        l_i = l_i_new
        m_i = m_i_new

    # --- 存储输出 ---
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
        cur_head * stride_oh + offs_d[None, :] * stride_od
    )
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc,
             mask=dim_mask[None, :] & (offs_m[:, None] < cur_batch_query_len))
```

## forward_decode
调用的是 [paged_atention_kernel](https://github.com/vllm-project/vllm/blob/400d483e87b71315bbb73edb0da9fd629212ca82/csrc/attention/attention_kernels.cuh#L90)
gridDim = (num_heads, num_seqs, 1). decode 的时候每个 seq 的 Query 的 toekn 数目都是 1，
- gridDim = (num_heads, num_seqs, 1): 每个线程块负责一个 seq 的 一个 head，函数定义如下
```C {linenos=true}++
template <typename scalar_t, typename cache_t, int HEAD_SIZE, int BLOCK_SIZE,  // default 16
          int NUM_THREADS /*=128*/, vllm::Fp8KVCacheDataType KV_DTYPE, 
          bool IS_BLOCK_SPARSE,
          int PARTITION_SIZE = 0>  // Zero means no partitioning.
__device__ void paged_attention_kernel(
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads, head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads, head_size, block_size]
    const int num_kv_heads,               // [num_heads]
    const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ seq_lens,      // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    // 矩阵每一维度的 stride，便于移动指针
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    const float* k_scale, const float* v_scale, const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step) 
```

首先先计算一下当前线程对应的各种参数，这里根据模板函数定义不使用 PARTITIONING.
```C {linenos=true}++
// grid = (num_heads, num_seqs, 1) 一个 thread block 处理一个 seq 的 一个 head
  const int seq_idx = blockIdx.y;
  const int partition_idx = blockIdx.z;
  const int max_num_partitions = gridDim.z;  // 1
  const int seq_len = seq_lens[seq_idx];  // 该 seq token 数

  // 计算块范围和 token 范围
  const int num_seq_blocks = DIVIDE_ROUND_UP(seq_len, BLOCK_SIZE);  // seq 要分几块读取
  const int num_blocks_per_partition =  num_seq_blocks;  // 分了几块
  const int start_block_idx = 0;  // 起始块索引
  const int end_block_idx = num_seq_blocks;  // 结束块索引
  const int num_blocks = end_block_idx - start_block_idx;  // 当前分区块数
  const int start_token_idx = start_block_idx * BLOCK_SIZE;  // 起始 token 索引
  const int end_token_idx = MIN(start_token_idx + num_blocks * BLOCK_SIZE, seq_len);  // 结束 token 索引
  const int num_tokens = end_token_idx - start_token_idx;  // 当前分区 token 数

  // 线程组织参数
  constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);  // 几个 thread 处理一个 token 32/16=2
  constexpr int NUM_THREAD_GROUPS = NUM_THREADS / THREAD_GROUP_SIZE;  // 一个 thread block 被分成几组 128/2=64
  constexpr int NUM_TOKENS_PER_THREAD_GROUP = DIVIDE_ROUND_UP(BLOCK_SIZE, WARP_SIZE);  // 每线程处理的 token 数 16/32=1
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;  // warp 个数 128/32=4
  const int thread_idx = threadIdx.x;  // 线程索引
  const int warp_idx = thread_idx / WARP_SIZE;  // 线程位于第几个 warp
  const int lane = thread_idx % WARP_SIZE;  // 线程是该 warp 中的第几个

  const int head_idx = blockIdx.x;
  const int num_heads = gridDim.x;
  // 考虑 GQA MQA
  const int num_queries_per_kv = num_heads / num_kv_heads;
  const int kv_head_idx = head_idx / num_queries_per_kv;
  const float alibi_slope =
      alibi_slopes == nullptr ? 0.f : alibi_slopes[head_idx];
```

定义 thread group ，保证其一次访问的数据为 16 Bytes，需要计算其中每个 thread 处理几个元素。
```C {linenos=true}++
// VEC_SIZE 即为一个 thread group 中每个线程需要处理元素个数，
constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);  // 16/2/2=4 
using K_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
using Q_vec = typename Vec<scalar_t, VEC_SIZE>::Type;
using Quant_vec = typename Vec<cache_t, VEC_SIZE>::Type;

constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;  // 每个 thread 处理几个元素 64/2=32
constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;  // 这几个元素相当于几个向量  32/4=8
// thread_idx = thread_group_idx * THREAD_GROUP_SIZE + thread_group_offset
const int thread_group_idx = thread_idx / THREAD_GROUP_SIZE;  // 线程位于第几个 thread group
const int thread_group_offset = thread_idx % THREAD_GROUP_SIZE;  // 线程是该 thread group 中第几个线程
```

下面将 Q 加载进共享内存。
![loadQ](https://note.youdao.com/yws/api/personal/file/WEB7a7b85b64fbddcf13d703135a4bf6d32?method=download&shareKey=6ca032c977b9f14a0864999633e8e08f "loadQ")
```C {linenos=true}++
  const scalar_t* q_ptr = q + seq_idx * q_stride + head_idx * HEAD_SIZE;
  __shared__ Q_vec q_vecs[THREAD_GROUP_SIZE][NUM_VECS_PER_THREAD];  // HEAD_SIZE * VEC_SIZE * sizeof(scalar_t) 大小

#pragma unroll
  for (int i = thread_group_idx; i < NUM_VECS_PER_THREAD; i += NUM_THREAD_GROUPS) {  // NUM_ELEMS_PER_THREAD / VEC_SIZE
    // 使得每个 thread group 的线程访问相邻的 vec
    const int vec_idx = thread_group_offset + i * THREAD_GROUP_SIZE;
    q_vecs[thread_group_offset][i] =
        *reinterpret_cast<const Q_vec*>(q_ptr + vec_idx * VEC_SIZE);
  }
  __syncthreads(); 
```


假设块不稀疏并且把不采用量化，加载 K 并计算 Q@K.T. 核心思想是一个 thread group 访问 16 Bytes. 一个 thread 访问一个 vec，一个向量包含的元素个数 `VEC_SIZE = 16 / sizeof (scalar_t) / THREAD_GROUP_SIZE`
1. 1st for 循环确定的是每次迭代中每个 warp 处理的是哪一个 block，一共要循环 num_seq_blocks / NUM_WARPS 次
2. 2nd for 循环确定的是该 warp 中的每个 thread group 访问的是该 block 的第几个 token. 即每个线程组处理一个 token.
3. 3rd for 循环确定的是该 thread group 中的每个 thread 访问的是第几个 vec. 该循环使得该 thread group 里面的线程读取一个完整的 headsize. 一次迭代读取的大小为 16 Bytes.

首先将 block_table 指针移动到存储该 kv cache 的首个 blockID 处，取出实际的物理块 ID，用在第三个 for 循环中将指针移动到该 K cache block 起始处. 由于
k_cache 的 shape 是 `[num_blocks, num_kv_heads, head_size/x, block_size, x]`，在第三个 for 循环中 k_ptr 被移动到了该 thread_group 要读取的 block 的 token 的 head 处。`vec_idx * VEC_SIZE` 即为 thread 要读取的元素开始位置，/x 表示对应的是第几个 16Bytes 划分, offset1 移动的是 dim3，offset2 移动的 则是 dim4.

3rd loop 结束后已经读取了一个 K cache 的完整 head_size 到寄存器中，因此 qk 为一个 token 的一个 head 的 Score Matrix. 根据 token_idx 由每个 thread group 里的 第一个线程负责将累加和到 logits 中并更新 qk_max。
```C {linenos=true}++
  // Memory planning.
  extern __shared__ char shared_mem[];
  // NOTE(woosuk): We use FP32 for the softmax logits for better accuracy.
  float* logits = reinterpret_cast<float*>(shared_mem);
  // Workspace for reduction.
  __shared__ float red_smem[2 * NUM_WARPS];  // 前一半用于存储 qk_max 后一半用于存储 exp_sum

  // x == THREAD_GROUP_SIZE * VEC_SIZE
  // 每次 thread group 一次取的元素数量 保证为 16 bytes
  constexpr int x = 16 / sizeof(cache_t);
  float qk_max = -FLT_MAX;

  // 指针移动到当前 seq 对应的首个 blockID
  const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq; 

  for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx; block_idx += NUM_WARPS) {  
    // 每个 warp 处理一个 block

    const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);  // 该 warp 当前处理的 block 对应的 id

    // Load a key to registers.
    for (int i = 0; i < NUM_TOKENS_PER_THREAD_GROUP; i++) {  // BLOCK_SIZE(16) / WARP_SIZE(32) = 1
      const int physical_block_offset = (thread_group_idx + i * WARP_SIZE) % BLOCK_SIZE;  // thread group 处理的是该 block 的第几个 token
      const int token_idx = block_idx * BLOCK_SIZE + physical_block_offset;  // 该 token 是该 seq 的第几个
      K_vec k_vecs[NUM_VECS_PER_THREAD];

#pragma unroll
      for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {  // NUM_ELEMS_PER_THREAD(32) / VEC_SIZE(4) = 8
        const cache_t* k_ptr = k_cache + 
                                physical_block_number * kv_block_stride +  // 移动到该 block 起始处
                                kv_head_idx * kv_head_stride +  // 移动到对应的 head 处
                                physical_block_offset * x;  // 移动到对应的 token 处
        const int vec_idx = thread_group_offset + j * THREAD_GROUP_SIZE;  // 该 thread 要读取 head_size 划分成的第几个 vec
        const int offset1 = (vec_idx * VEC_SIZE) / x;  // 第几个 16Bytes 划分
        const int offset2 = (vec_idx * VEC_SIZE) % x;  // 划分的第几个元素

        if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
          k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + offset1 * BLOCK_SIZE * x + offset2);
        }
      }

      // Compute dot product.
      // This includes a reduction across the threads in the same thread group.
      float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs[thread_group_offset], k_vecs);
      // Add the ALiBi bias if slopes are given.
      qk += (alibi_slope != 0) ? alibi_slope * (token_idx - seq_len + 1) : 0;

      if (thread_group_offset == 0) {  // 每个线程组的第一个线程进行更新 max
        // Store the partial reductions to shared memory.
        // NOTE(woosuk): It is required to zero out the masked logits.
        const bool mask = token_idx >= seq_len;
        logits[token_idx - start_token_idx] = mask ? 0.f : qk;
        // Update the max value.
        qk_max = mask ? qk_max : fmaxf(qk_max, qk);
      }
    }
  }
```
![load k & QK Mul](https://note.youdao.com/yws/api/personal/file/WEB36d66a13612972c7f567ed8f20600664?method=download&shareKey=9a305814befc64b17e64feb1c8d76b17 "load k & QK Mul")

上面这一段结束后下面每个 warp 内 thread group 中的第一个线程已经记录了该 group 的 qk_max. 下一步则是在 warp 内进行 qk_max 归约，存储在共享内存 red_smem 中。 由于一个 warp 处理的是一个 block，相当于现在 red_smem 每个元素存储了对应 block 内的 qk_max.
```C {linenos=true}++
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  if (lane == 0) {
    red_smem[warp_idx] = qk_max;
  }
  __syncthreads();
```

下一步则是在 thread block 内对所有 warp 进行规约，得到该 seq 最后的 qk_max. 然后广播到所有线程中。之后每个线程计算 exp 存入 logits，每个 warp 内的 exp 求和结果存储在 red_smem 的后一半中。最后则是计算 softmax 存到 logits.
```C {linenos=true}++
  qk_max = lane < NUM_WARPS ? red_smem[lane] : -FLT_MAX;
#pragma unroll
  for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2) {
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
  }
  // Broadcast the max qk value to all threads.
  qk_max = VLLM_SHFL_SYNC(qk_max, 0);

// Get the sum of the exp values.
  float exp_sum = 0.f;
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;
    exp_sum += val;
  }
  exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

  // Compute softmax.
  const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
  for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    logits[i] *= inv_sum;
  }
  __syncthreads();
```

加载 v 的逻辑与 k 相同，但没有使用 thread group 概念，而是让一个 thread 一次加载 16 Bytes.