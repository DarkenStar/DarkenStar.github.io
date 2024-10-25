---
title: xDiT Principle
date: 2024/9/27 17:48:35
categories: Distributed Training
tags: Essay
excerpt: This is a brief introduction to the xDiT Principle.
mathjax: true
katex: true
---
# Parse Config Arguments

会从命令行参数中获取有关 Model, Runtime, Parallel Processing & Input 有关的信息。前三者被包含在 `engine_config` 中，而最后者则被包含在 `input_config` 中。在 `create_config()` 函数中，会初始化 `_WORLD` 全局变量，它是一个 `GroupCoordinator` 实例。很明显它只有一个包含所有的设备进程组。
{% fold info @GroupCoordinator %}
`GroupCoordinator` 类是一个 PyTorch 的进程组封装器，主要用于管理一组进程之间的通信。它可以根据不同的通信后端（如 NCCL、Gloo、MPI 等）来协调进程之间的操作。包含以下信息

* `rank`: 当前进程的全局索引（全局唯一）。
* `ranks`: 组内所有进程的全局索引列表。
* `world_size`: 组的大小，即进程的数量 `len(ranks)`
* `local_rank`: 当前进程在本地节点中的索引。
* `rank_in_group`: 当前进程在组内的索引。
* `cpu_group`: 用于 CPU 通信的进程组。
* `device_group`: 用于设备（如 GPU）通信的进程组。

```python
if we have a group of size 4 across two nodes:
Process | Node | Rank | Local Rank | Rank in Group
  0     |   0  |  0   |     0      |       0
  1     |   0  |  1   |     1      |       1
  2     |   1  |  2   |     0      |       2
  3     |   1  |  3   |     1      |       3
```

`__init__` 方法接收以下参数：

* `group_ranks`: 一个包含多个进程索引列表的列表，每个子列表表示一个进程组。
* `local_rank`: 当前进程的本地索引。
* `torch_distributed_backend`: 指定用于通信的后端类型 (如 "gloo" 或 "nccl").

初始化过程：

1. 使用 `torch.distributed.get_rank()` 获取当前进程的全局索引。
2. 遍历传入的 `group_ranks` 列表，为每个子列表创建一个新的设备组和 CPU 组。
3. 如果当前进程的索引在当前子列表中，则设置该进程的组内信息 (包括 `ranks`、`world_size` 和 `rank_in_group`).
4. 确保 CPU 组和设备组都已成功创建。
5. 根据是否可用 CUDA 设置当前设备为 GPU 或 CPU.

{% endfold %}

```python
def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    args = xFuserArgs.add_cli_args(parser).parse_args()  # Add Command Line Interface (CLI) arguments
    engine_args = xFuserArgs.from_cli_args(args)  # Extract CLI args and pass them to xFuserArgs Constructor
    engine_config, input_config = engine_args.create_config()  # Init _WORLD. engine_config: model, run_time & parallel infos, input_config: input shape, prompt & sampler infos
    local_rank = get_world_group().local_rank
```

关于可以支持的并行策略如下，包括 Data Parallel, Sequence Parallel, Pipefusion Parallel & Tensor Parallel.

```bash
Parallel Processing Options:
  --use_cfg_parallel    Use split batch in classifier_free_guidance. cfg_degree will be 2 if set
  --data_parallel_degree DATA_PARALLEL_DEGREE
                        Data parallel degree.
  --ulysses_degree ULYSSES_DEGREE
                        Ulysses sequence parallel degree. Used in attention layer.
  --ring_degree RING_DEGREE
                        Ring sequence parallel degree. Used in attention layer.
  --pipefusion_parallel_degree PIPEFUSION_PARALLEL_DEGREE
                        Pipefusion parallel degree. Indicates the number of pipeline stages.
  --num_pipeline_patch NUM_PIPELINE_PATCH
                        Number of patches the feature map should be segmented in pipefusion parallel.
  --attn_layer_num_for_pp [ATTN_LAYER_NUM_FOR_PP ...]
                        List representing the number of layers per stage of the pipeline in pipefusion parallel
  --tensor_parallel_degree TENSOR_PARALLEL_DEGREE
                        Tensor parallel degree.
  --split_scheme SPLIT_SCHEME
                        Split scheme for tensor parallel.
```

从 CLI 解析的参数后会在 `create_config()` 中组成如下的 [ParallelConfig](https://github.com/xdit-project/xDiT/blob/6f92383e76b5f8bbaf8f45e6863d1e69b0d2f955/xfuser/config/config.py#L185).

- `DataParallelConfig`: 总的并行度为 `dp_degree * cfg_degree`.
  - `dp_degree`: 相当于对 batch 维度进行切分，
  - `cfg_degree`: Class-free Guidance(cfg) 用于控制无条件的图片生成 (若使用相当于 `batchsize *= 2`).
- `SequenceParallelConfig`: 总的并行度为 `sp_degree = ulysses_degree * ring_degree`
  - `ulysses_degree`: 用于控制 [DeepSeed-Ulesses](https://arxiv.org/abs/2309.14509) 的序列并行度。
  - `ring_degree`: 用于控制计算 Ring Attention 时对 Q K V 沿着 Sequence 维度的切分块数。
- `TensorParallelConfig`: 总的并行度为 `tp_degree`.
  - `tp_degree`: 用于控制 [2D Tensor Parallel](https://arxiv.org/abs/2104.05343) 的并行度。
  - `split_scheme`: 用于控制张量切分方式.
- `PipeFusionParallelConfig`: 总的并行度为 `pp_degree=num_pipeline_patch`.
  - `pp_degree`: 用于控制 [PipeFusion](https://arxiv.org/abs/2112.11446) 中模型 Transoformer Blocks 的切分个数。
  - `num_pipeline_patch`: 用于控制对 latent feature map 的切分块数.
  - `attn_layer_num_for_pp`: 是一个 list，表示 `pp_degree` 里每个 stage 的 Transformer 层数。

{% note warning %}

关于 PipeFusion，原文说切分的 patch 数和 pipeline 大小可以不同，但这里要求 `len(attn_layer_num_for_pp)=pp_degree`

{% endnote %}

{% note primary %}
设备数必须等于 `dp_degree * cfg_degree * sp_degree * tp_degree * num_pipeline_patch`，并且 `pp_degree` 必须小于等于设备数。
`ulysses_degree` 必须要大于且能被 attention 的头数整除。
{% endnote %}

```python
parallel_config = ParallelConfig(
    dp_config=DataParallelConfig(
        dp_degree=self.data_parallel_degree,
        use_cfg_parallel=self.use_cfg_parallel,
    ),
    sp_config=SequenceParallelConfig(
        ulysses_degree=self.ulysses_degree,
        ring_degree=self.ring_degree,
    ),
    tp_config=TensorParallelConfig(
        tp_degree=self.tensor_parallel_degree,
        split_scheme=self.split_scheme,
    ),
    pp_config=PipeFusionParallelConfig(
        pp_degree=self.pipefusion_parallel_degree,
        num_pipeline_patch=self.num_pipeline_patch,
        attn_layer_num_for_pp=self.attn_layer_num_for_pp,
    ),
)
```

# Construct Pipeline

解析完配置参数并构建了 `engine_config` 后，下一步是构建模型的 pipeline.

```python
    pipe = xFuserPixArtAlphaPipeline.from_pretrained(  # First construct a PixArtAlphaPipeline, then pass it and engine_config to xFuserPipelineBaseWrapper
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")
    pipe.prepare_run(input_config)
```

xFuserPixArtAlphaPipeline 继承自 [xFuserPipelineBaseWrapper](https://github.com/xdit-project/xDiT/blob/6f92383e76b5f8bbaf8f45e6863d1e69b0d2f955/xfuser/model_executor/pipelines/base_pipeline.py#L61)，_init_runtime_state 函数经过一番调用后会使用 [initialize_model_parallel](https://github.com/xdit-project/xDiT/blob/6f92383e76b5f8bbaf8f45e6863d1e69b0d2f955/xfuser/core/distributed/parallel_state.py#L265) 初始化 `_RUNTIME` 有关模型参数的部分和模型并行的全局变量 `_DP, _CFG, _PP, _SP, _TP`，它是一个 DiTRuntimeState (继承 RuntimeState) 实例，记录了每个 Group 包含的设备索引，除此之外还包括 PipeFusionParallel 中有关 patch 索引的参数 (在稍后 pipeline 执行的时候计算).

```python
class xFuserPipelineBaseWrapper(xFuserBaseWrapper, metaclass=ABCMeta):

    def __init__(
        self,
        pipeline: DiffusionPipeline,
        engine_config: EngineConfig,
    ):
        self.module: DiffusionPipeline
        self._init_runtime_state(pipeline=pipeline, engine_config=engine_config)

        # backbone
        transformer = getattr(pipeline, "transformer", None)
        unet = getattr(pipeline, "unet", None)
        # vae
        vae = getattr(pipeline, "vae", None)
        # scheduler
        scheduler = getattr(pipeline, "scheduler", None)

        if transformer is not None:
            pipeline.transformer = self._convert_transformer_backbone(transformer)
        elif unet is not None:
            pipeline.unet = self._convert_unet_backbone(unet)

        if scheduler is not None:
            pipeline.scheduler = self._convert_scheduler(scheduler)

        super().__init__(module=pipeline)
   
   
    def _convert_transformer_backbone(
        self,
        transformer: nn.Module,
    ):
				#...

            logger.info("Transformer backbone found, paralleling transformer...")
            wrapper = **xFuserTransformerWrappersRegister.get_wrapper(transformer)**
            transformer = wrapper(transformer=transformer)
        return transformer
```

## initialize_model_parallel

该函数中会初始化一个 `RankGenerator`，它接收每个并行方法的设备组大小和并行度大小顺序。其主要的方法是通过 [generate_masked_orthogonal_rank_groups](https://github.com/xdit-project/xDiT/blob/6f92383e76b5f8bbaf8f45e6863d1e69b0d2f955/xfuser/core/distributed/utils.py#L4) 函数确定每个并行组由包含哪些设备，先把并行方法按照并行度从小到大排列成 `tp-sp-pp-cfg-dp`. 再根据要生成的并行组产生对应的 `mask`. 即如果要生成 `pp` 组对应的 rank，那么 `mask = [0, 0, 1, 0, 0]`

该函数首先会生成需要生成的并行组的大小组成的 masked_shape 和不需要生成的 unmasked_shape. 首先要用 prefix_product 计算 `global_stride`，即每个并行度的设备组包含几个设备。再根据 `mask` 取出对应的 `mask_stride` 和 `unmaskd_stride`. `group_size = mask_stride[-1]` 即为最大并行度的组包含的设备数。`num_of_group = num_of_device / mask_stride[-1]` 即为要生成几个并行度最大的组。先遍历要生成的每个设备组，并用 decompose 函数确定该设备组在不需要并行维度上的索引；再遍历该组中的每个设备的 lock rank，确定该设备在需要并行维度上的索引，最后用 inner_product 确定该设备的 global rank.

```python
def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:

    def prefix_product(a: List[int], init=1) -> List[int]:  # Exclusive
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a: List[int], b: List[int]) -> int:
        return sum([x * y for x, y in zip(a, b)])

    def decompose(index, shape, stride=None):  # index: 第几个并行组  # shape: 并行组大小的 list
        """
        This function solve the math problem below:
            There is an equation: index = sum(idx[i] * stride[i])
            And given the value of index, stride.
            Return the idx.
        This function will used to get the pp/dp/pp_rank from group_index and rank_in_group.
        """
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]  #  计算在每个并行维度上的索引
        # stride is a prefix_product result. And the value of stride[-1]
        # is not used.
        assert (
            sum([x * y for x, y in zip(idx, stride[:-1])]) == index
        ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]  # 需要采取并行的维度
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]  # 不需要的

    global_stride = prefix_product(parallel_size)  # exclusive 前缀积 表示大的并行维度包括几个设备
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]  # 最大的一个并行维度包括几个设备
    num_of_group = world_size // group_size  # 分成几个大组

    ranks = []  
    for group_index in range(num_of_group):  # 遍历每个设备组
        # get indices from unmaksed for group_index.
        decomposed_group_idx = decompose(group_index, unmasked_shape)  # 得到在不需要采取并行的维度上的索引
        rank = []
        for rank_in_group in range(group_size):  # 遍历该组中的每个设备 local rank
            # get indices from masked for rank_in_group.
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)  # 得到最大并行组的每个设备在采取并行的维度上的索引
            rank.append(  // 相加得到全局rank
                inner_product(decomposed_rank_idx, masked_stride)  
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks
```

## Hybrid Parallelsim Design

xDiT支持四种并行方式：PipeFusion、Sequence、Data 和 CFG Parallel。其中，Data 和 CFG Parallel在图像间并行相对简单，而 PipeFusion和 Sequence 在图像内部的不同 Patch 间并行则较为复杂。能

PipeFusion 利用 Input Tempor Redundancy特点，使用过时的 KV（Stale KV）进行 Attention 计算，这使得 PipeFusion 无法像大型语言模型那样轻松地实现并行策略的混合。使用标准的序列并行接口，如RingAttention、Ulysses或 USP，无法满足 SP 与PipeFusion混合并行的需求。

我们对这个问题具体说明，下图展示了pipe_degree=4，sp_degree=2的混合并行方法。设置 `num_pipeline_patch`=4，图片切分为 M=`num_pipeline_patch*sp_degree`=8 个 Patch，分别是 P0~P7.

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/hybrid_pp_scheme.png" alt="hybrid process group config"  width="60%">
</div>

Standard SP Attention 的输入Q，K，V 和输出 O 都是沿着序列维度切分，且切分方式一致。如果不同 rank 的输入 patch 没有重叠，每个 micro step 计算出 fresh KV 更新的位置在不同 rank 间也没有重叠。如下图所示，standard SP 的 KV Buffer 中黄色部分是 SP0 rank=0 拥有的 fresh KV，绿色部分是 SP1 rank=1 拥有的fresh KV，二者并不相同。在这个 diffusion step 内，device=0 无法拿到 P1,3,5,7 的 fresh KV 进行计算，但是 PipeFusion 则需要在下一个 diffusion step 中，拥有上一个diffusion step 全部的 KV. standard SP 只拥有 1/sp_degree 的 fresh kv buffer，因此无法获得混合并行推理正确的结果。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/hybrid_workflow.png" alt="hybrid parallel workflow">
</div>

xDiT专门定制了序列并行的实现方式，以适应这种混合并行的需求。xDiT使用 `xFuserLongContextAttention` 把SP的中间结果存在 KV Buffer 内。效果如下图，每个 micro-step SP 执行完毕后，SP Group 内不同 rank 设备的 fresh KV是 replicate 的。这样一个 diffusion step 后，SP Group 所有设备的 KV Buffer 都更新成最新，供下一个 Diffusion Step 使用。

<div align="center">
    <img src="https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/kvbuffer_hybrid.png" alt="kvbuffer in hybrid parallel">
</div>

{% fold info @Another Example %}
假设一共有 16 个 GPU，索引表示为 g0 ... g15，并行方法和并行度设置如下

`dp_degree (2) * cfg_degree (2) * pp_degree (2) * sp_degree (2) = 16`.

那么一共会创建 2 data parallel-groups, 8 CFG groups, 8 pipeline-parallel groups & 8 sequence-parallel groups:

- 2 data-parallel groups:
  [g0, g1, g2, g3, g4, g5, g6, g7],
  [g8, g9, g10, g11, g12, g13, g14, g15]
- 8 CFG-parallel groups:
  [g0, g4], [g1, g5], [g2, g6], [g3, g7],
  [g8, g12], [g9, g13], [g10, g14], [g11, g15]
- 8 pipeline-parallel groups:
  [g0, g2], [g4, g6], [g8, g10], [g12, g14],
  [g1, g3], [g5, g7], [g9, g11], [g13, g15]
- 8 sequence-parallel groups:
  [g0, g1], [g2, g3], [g4, g5], [g6, g7],
  [g8, g9], [g10, g11], [g12, g13], [g14, g15]
  {% endfold %}

## Convert Model

[_split_transformer_blocks](https://github.com/xdit-project/xDiT/blob/6f92383e76b5f8bbaf8f45e6863d1e69b0d2f955/xfuser/model_executor/models/transformers/base_transformer.py#L76) 会对 transformer block 进行分配，如果 parallel_config 指定了 attn_layer_num_for_pp，即存有每个 pipeFusion 的设备被分配的 transformer block 数量的列表，按其进行分配；否则平均分。

```python
def _split_transformer_blocks(self, transformer: nn.Module,):
    # omit

    # transformer layer split
    attn_layer_num_for_pp = (  # 获取每个 pipeFusion 的设备被分配的 transformer block 数量
        get_runtime_state().parallel_config.pp_config.attn_layer_num_for_pp
    )
    pp_rank = get_pipeline_parallel_rank()
    pp_world_size = get_pipeline_parallel_world_size()
    if attn_layer_num_for_pp is not None:
        if is_pipeline_first_stage():
            transformer.transformer_blocks = transformer.transformer_blocks[ : attn_layer_num_for_pp[0]]
        else:
            transformer.transformer_blocks = transformer.transformer_blocks[sum(attn_layer_num_for_pp[: pp_rank - 1]) : 
                                                                            sum(attn_layer_num_for_pp[:pp_rank])]
    else:  # 没有指定则平均分
        num_blocks_per_stage = (len(transformer.transformer_blocks) + pp_world_size - 1) // pp_world_size
        start_idx = pp_rank * num_blocks_per_stage
        end_idx = min((pp_rank + 1) * num_blocks_per_stage, len(transformer.transformer_blocks),)
        transformer.transformer_blocks = transformer.transformer_blocks[start_idx:end_idx]
    # position embedding
    if not is_pipeline_first_stage():
        transformer.pos_embed = None
    if not is_pipeline_last_stage():
        transformer.norm_out = None
        transformer.proj_out = None
    return transformer
```

同时也会 convert 原先的 transformer backbone 为 [xFuserPixArtTransformer2DWrapper](https://github.com/xdit-project/xDiT/blob/main/xfuser/model_executor/models/transformers/pixart_transformer_2d.py#L21)，具体表现为只有 pipeline 的第一阶段进行 position embedding，最后一阶段进行 unpatchify 变为原来的图像形状。

```python

@xFuserTransformerWrappersRegister.register(PixArtTransformer2DModel)
class xFuserPixArtTransformer2DWrapper(xFuserTransformerBaseWrapper):
    def __init__(
        self,
        transformer: PixArtTransformer2DModel,
    ):
        super().__init__(
            transformer=transformer,
            submodule_classes_to_wrap=[nn.Conv2d, PatchEmbed],
            submodule_name_to_wrap=["attn1"],
        )

    @xFuserBaseWrapper.forward_check_condition
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):  
	'''
	......
	'''
	height, width = self._get_patch_height_width()
        # * only pp rank 0 needs pos_embed (patchify)
        if is_pipeline_first_stage():
            hidden_states = self.pos_embed(hidden_states)
	'''
	......
	'''
	if is_pipeline_last_stage():
	'''
	......
	'''
	else:
	    output = hidden_states

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

```

# Pipeline Execution

在进行 warm up 后便会进行模型推理和采样器的去噪过程。模型推理通过调用 pipeline 的 `__call__` 方法实现。在原先 diffusers 包中的 [PixaeArtAlphaPipeline](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/pixart_alpha/pipeline_pixart_alpha.py) 基础上做了一些修改。我们直接看修改的部分。

`get_runtime_state()` 返回 `_RUNTIME` ，再调用 `set_input_parameters` 方法，设置输入参数和计算 PipeFusionParallel 中有关 patch 索引的参数。

```python
get_runtime_state().set_input_parameters(
    height=height,
    width=width,
    batch_size=batch_size,
    num_inference_steps=num_inference_steps,
)
```

该函数会计算

* pipeline parallel 中每个 patch 的高度，必须是 `patch_size * num_sp_patches` 的整数倍。
* 将每个流水线阶段的 patch 高度均匀地分配给 `num_sp_patches` 个序列并行设备，计算每个设备的 patch 高度和起始索引。

然后会对 prompt 嵌入后的正样本和负样本在 cfg parallel 组中的设备进行分割, rank 0 负样本，rank 1 正样本。

```python
if do_classifier_free_guidance:
    (prompt_embeds, 
    prompt_attention_mask,) = self._process_cfg_split_batch(negative_prompt_embeds,
                                                                            prompt_embeds,
                                                                            negative_prompt_attention_mask,
                                                                            prompt_attention_mask,)

def _process_cfg_split_batch(self,
        concat_group_0_negative: torch.Tensor,
        concat_group_0: torch.Tensor,
        concat_group_1_negative: torch.Tensor,
        concat_group_1: torch.Tensor,):
  
  if get_classifier_free_guidance_world_size() == 1:
      concat_group_0 = torch.cat([concat_group_0_negative, concat_group_0], dim=0)
      concat_group_1 = torch.cat([concat_group_1_negative, concat_group_1], dim=0)
  elif get_classifier_free_guidance_rank() == 0:
      concat_group_0 = concat_group_0_negative
      concat_group_1 = concat_group_1_negative
  elif get_classifier_free_guidance_rank() == 1:
      concat_group_0 = concat_group_0
      concat_group_1 = concat_group_1
  else:
      raise ValueError("Invalid classifier free guidance rank")
  return concat_group_0, concat_group_1
```

# Async Pipeline

## Initialize Pipeline

首先会初始化 pipeline，rank 0 会接收 warmup 阶段的 latents 然后沿着 H 维度进行分块，rank -1 也会沿着 H 维度进行分块。然后为每个 patch 创建接收的任务，注意 rank 0 第一次是从 warmup 阶段接收 latents，所以他的需要接收的 timestep 少一个。
`patch_latents` 表示当前设备正在处理的 patch 数据，它会在流水线的每一阶段进行处理和传递。`last_patch_latents` 只在流水线的最后阶段设备中使用，用来存储每个 patch 的最终计算结果。

```python
if len(timesteps) == 0:
    return latents
num_pipeline_patch = get_runtime_state().num_pipeline_patch
num_pipeline_warmup_steps = get_runtime_state().runtime_config.warmup_steps
patch_latents = self._init_async_pipeline(
    num_timesteps=len(timesteps),
    latents=latents,
    num_pipeline_warmup_steps=num_pipeline_warmup_steps,
)
last_patch_latents = (  # 每个 pipeline group 最后的设备接收所有的 patch
    [None for _ in range(num_pipeline_patch)]
    if (is_pipeline_last_stage())
    else None
)

def _init_async_pipeline(
    self,
    num_timesteps: int,
    latents: torch.Tensor,
    num_pipeline_warmup_steps: int,
):
    get_runtime_state().set_patched_mode(patch_mode=True)

    if is_pipeline_first_stage():
        # get latents computed in warmup stage
        # ignore latents after the last timestep
        latents = (get_pp_group().pipeline_recv()
                  if num_pipeline_warmup_steps > 0
                  else latents)
        patch_latents = list(latents.split(get_runtime_state().pp_patches_height, dim=2))
    elif is_pipeline_last_stage():
        patch_latents = list(latents.split(get_runtime_state().pp_patches_height, dim=2))
    else:
        patch_latents = [None for _ in range(get_runtime_state().num_pipeline_patch)]

    recv_timesteps = (num_timesteps - 1 if is_pipeline_first_stage() else num_timesteps)
    # construct receive tasks for each patch
    for _ in range(recv_timesteps):
        for patch_idx in range(get_runtime_state().num_pipeline_patch):
            get_pp_group().add_pipeline_recv_task(patch_idx)

    return patch_latents
```

# Iterate Over Timesteps

对于每个 `timestep`（即每个去噪步骤），会对每个 patch 执行：

1. 如果当前设备是流水线的最后一阶段 (`is_pipeline_last_stage()`)，将当前 patch 的数据保存到 `last_patch_latents` 中。
2. 如果不是第一阶段的第一个时间步 (`i == 0`)，调用 `recv_next()` 来异步接收来自上一设备的 patch 数据（非阻塞操作，通过 `irecv` 完成）。
3. 对每个 patch 执行模型的前向传播 `_backbone_forward`，根据当前时间步 `t` 进行推理和计算。
4. 如果当前设备是最后一阶段，调用 `_scheduler_step` 来根据噪声进行去噪，并将数据发送给下一个设备 `pipeline_isend`。
5. 对于非最后阶段的设备，继续将当前 patch 的计算结果发送到下一设备。

`get_pp_group().pipeline_isend` 用于将当前 patch 发送到下一个设备，使用的是 torch.distributed.isend，这是非阻塞发送。
`get_pp_group().recv_next` 会准备好接收来自上一个设备的数据，recv_buffer 用来存放接收到的数据。irecv 实现非阻塞接收，可以在等待数据的同时进行其他操作。

{% note warning %}
scheduler_step 只对单独的 patch 进行，原因未知。
{% endnote %}

```python
first_async_recv = True
for i, t in enumerate(timesteps):
    for patch_idx in range(num_pipeline_patch):
        if is_pipeline_last_stage():
            last_patch_latents[patch_idx] = patch_latents[patch_idx]

        if is_pipeline_first_stage() and i == 0:
            pass
        else:
            if first_async_recv:
                get_pp_group().recv_next()  
                first_async_recv = False
            patch_latents[patch_idx] = get_pp_group().get_pipeline_recv_data(
                idx=patch_idx
            )
        patch_latents[patch_idx] = self._backbone_forward(
            latents=patch_latents[patch_idx],
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            added_cond_kwargs=added_cond_kwargs,
            t=t,
            guidance_scale=guidance_scale,
        )
        if is_pipeline_last_stage():
            patch_latents[patch_idx] = self._scheduler_step(
                patch_latents[patch_idx],  # pred noise
                last_patch_latents[patch_idx],  # last timestep noise
                t,
                extra_step_kwargs,
            )
            if i != len(timesteps) - 1:
                get_pp_group().pipeline_isend(
                    patch_latents[patch_idx], segment_idx=patch_idx
                )
        else:
            get_pp_group().pipeline_isend(
                patch_latents[patch_idx], segment_idx=patch_idx
            )

        if is_pipeline_first_stage() and i == 0:
            pass
        else:
            if i == len(timesteps) - 1 and patch_idx == num_pipeline_patch - 1:
                pass
            else:
                get_pp_group().recv_next()

        get_runtime_state().next_patch()  # switch to next: (self.pipeline_patch_idx + 1) % self.num_pipeline_patch

    if i == len(timesteps) - 1 or (
        (i + num_pipeline_warmup_steps + 1) > num_warmup_steps
        and (i + num_pipeline_warmup_steps + 1) % self.scheduler.order == 0
    ):
        progress_bar.update()
        assert callback is None, "callback not supported in async " "pipeline"
        if (
            callback is not None
            and i + num_pipeline_warmup_steps % callback_steps == 0
        ):
            step_idx = (i + num_pipeline_warmup_steps) // getattr(
                self.scheduler, "order", 1
            )
            callback(step_idx, t, patch_latents[patch_idx])
```

## Construct Final Latents

timestep 遍历完成后，仍然有最后的操作要进行，这些操作的主要目的是将流水线并行中各个 patch 的结果拼接起来，形成完整的输出结果。尤其是对于最后一个设备，还需要处理 序列并行（sequence parallelism） 的合并操作。通过 all_gather 操作将每个设备上处理的 patch 结果收集起来，然后从每个设备的 `sp_latents_list` 中，提取出对应于 `pp_patch_idx` 的 patch 数据并将它们拼接起来。

```python
latents = None
if is_pipeline_last_stage():
    latents = torch.cat(patch_latents, dim=2)
    if get_sequence_parallel_world_size() > 1:
        sp_degree = get_sequence_parallel_world_size()
        sp_latents_list = get_sp_group().all_gather(
            latents, separate_tensors=True
        )
        latents_list = []
        for pp_patch_idx in range(get_runtime_state().num_pipeline_patch):
            latents_list += [
                sp_latents_list[sp_patch_idx][
                    ...,
                    get_runtime_state().pp_patches_start_idx_local[pp_patch_idx] : get_runtime_state().pp_patches_start_idx_local[pp_patch_idx + 1],
                    :,
                ]
                for sp_patch_idx in range(sp_degree)
            ]
        latents = torch.cat(latents_list, dim=-2)
return latents
```

# Decode Latents

为了避免 VAE 中的 [Decoder](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/vae.py#L185) 在对 8192px 分辨率图像进行 conv2D 的过程中出现 OOM 的问题， xDiT 使用了序列并行和 patch 并行的 [PatchConv2d](https://github.com/xdit-project/DistVAE/blob/a7e7ee7ec222f45af1214984561c8c645be8aece/distvae/models/layers/conv2d.py#L13) 和 [PatchGroupNorm](https://github.com/xdit-project/DistVAE/blob/a7e7ee7ec222f45af1214984561c8c645be8aece/distvae/models/layers/normalization.py#L59) 来替换掉原有 Decoder 中的 [UpDecoderBlock2D](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_2d_blocks.py#L2682) 对应的层。

## PatchGroupNorm

PatchGroupNorm 在 H 维度上划分为多个 patch，每个设备求自己所负责的部分和。
{% fold info@GroupNorm Principles %}
假设输入张量 x 的形状为 [N, C, H, W]，其中 N 表示批量大小（Batch Size），C 表示通道数（Channels），H 和 W 分别表示高度和宽度。在 GN 中，通道数 C 被划分为 G 组，每个组包含 C/G 个通道。计算每个组内即 [C/G, H, W] 维度上的均值和方差。特别的 G=1 时，GN 退化为 BN。G=C 时，GN 退化为 LN。
{% endfold %}

1. 获取高度信息

```python
class PatchGroupNorm(nn.Module):
  
    ''' def __init__(self, ...)'''

    def forward(self, x: Tensor) -> Tensor:
        height = torch.tensor(x.shape[-2], dtype=torch.int64, device=x.device)
        dist.all_reduce(height)  # 收集所有进程的高度并汇总。最终每个进程的 height 都将表示全局的高度和。

```

2. 计算每个组的通道数量以及每个进程内的元素数量

```python
channels_per_group = x.shape[1] // self.num_groups  # 每个组的通道数量
nelements_rank = channels_per_group * x.shape[-2] * x.shape[-1]  # 当前进程负责的每个组中的元素总
nelements = channels_per_group * height * x.shape[-1]  # 所有进程的每个组中的元素总数
```

3. 计算每个组的均值

```python
x = x.view(x.shape[0], self.num_groups, -1, x.shape[-2], x.shape[-1])  #  [batch_size, num_groups, channels_per_group, height, width]
group_sum = x.mean(dim=(2,3,4), dtype=torch.float32)  # 对每个组的所有元素 (channels_per_group, height, width) 求平均
group_sum = group_sum * nelements_rank  # 加权后的局部和 = 局部均值 * 当前进程的元素数量
dist.all_reduce(group_sum)  # 收集并汇总所有进程的局部和，得到全局和
E = (group_sum / nelements)[:, :, None, None, None].to(x.dtype)  # 计算全局的均值 E
```

4. 计算每个组的方差

```python
# 和计算均值同样的操作
group_var_sum = torch.empty((x.shape[0], self.num_groups), dtype=torch.float32, device=x.device)
torch.var(x, dim=(2,3,4), out=group_var_sum)  
group_var_sum = group_var_sum * nelements_rank
dist.all_reduce(group_var_sum)
var = (group_var_sum / nelements)[:, :, None, None, None].to(x.dtype)
```

5. 归一化并缩放 {% mathjax %} y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta {% endmathjax %}

```python
x = (x - E) / torch.sqrt(var + self.eps)
x = x * self.weight[:, :, None, None, None] + self.bias[:, :, None, None, None]
return x
```

## PatchConv2d

`PatchConv2d` 将潜在空间中的特征映射分割成多个 patch，跨不同设备进行序列并行 VAE 解码。这种技术将中间激活所需的峰值内存减少到 1/N，其中 N 是所使用的设备数量。对于 VAE 中的卷积算子，需要对如下图所示的 halo 区域数据进行通信。

![Patch VAE Conv](https://raw.githubusercontent.com/xdit-project/xdit_assets/main/methods/patchvaeconv.png "Patch VAE Conv")

```python
class PatchConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None,
        block_size: Union[int, Tuple[int, int]] = 0
    ) -> None:

        if isinstance(dilation, int):
            assert dilation == 1, "dilation is not supported in PatchConv2d"
        else:
            for i in dilation:
                assert i == 1, "dilation is not supported in PatchConv2d"
        self.block_size = block_size
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, 
            groups, bias, padding_mode, device, dtype)
```

`_conv_forward` 函数是 `PatchConv2d` 类的核心，它负责在输入张量上执行卷积操作，特别是在分布式计算场景下处理跨进程的输入切分、halo 区域的传递和计算。以下是使用的辅助函数的简要功能说明

* `_get_world_size_and_rank `：获取当前分布式环境中的进程总数 `world_size` 和当前进程的编号 `rank`
* `_calc_patch_height_index`：根据每个进程的输入高度，计算所有进程的起始和结束高度索引。
* `_calc_halo_width_in_h_dim`：计算当前进程在 h 维度上所需的上方和下方的 halo 区域宽度。
* `_calc_bottom_halo_width`：计算当前进程从下方相邻进程需要接收的 halo 区域的宽度。
* `_calc_top_halo_width`：计算当前进程从上方相邻进程需要接收的 halo 区域的宽度。
* `_adjust_padding_for_patch`：根据当前进程的 `rank` 和总进程数调整输入数据的填充方式，防止边界重复计算。

1. 获取输入信息以及通信组信息

```python
def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        bs, channels, h, w = input.shape

        world_size, rank = self._get_world_size_and_rank()

        if (world_size == 1):  # 处理非分布式情况
            if self.padding_mode != 'zeros':
                return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                weight, bias, self.stride,
                                _pair(0), self.dilation, self.groups)
            return F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
```

2. 获取输入的元数据

```python
patch_height_list = [torch.zeros(1, dtype=torch.int64, device=f"cuda:{rank}") for _ in range(dist.get_world_size())]
dist.all_gather(patch_height_list, torch.tensor([h], dtype=torch.int64, device=f"cuda:{rank}"))  # 收集所有进程的输入高度
patch_height_index = self._calc_patch_height_index(patch_height_list)  # 计算每个进程块的起始高度和结束高度的索引
halo_width = self._calc_halo_width_in_h_dim(rank, patch_height_index, self.kernel_size[0], self.padding[0], self.stride[0])  # 计算当前进程块的上下 halo 区域的宽度
```

3. 计算相邻进程的 halo 区域 (也就是自己需要接发送的部分)

通过计算前一个进程的 bottom_halo_width 和后一个进程的 top_halo_width 得出自己需要发送的部分

```python
prev_bottom_halo_width: int = 0
next_top_halo_width: int = 0
if rank != 0:
    prev_bottom_halo_width = self._calc_bottom_halo_width(rank - 1, patch_height_index, self.kernel_size[0], self.padding[0], self.stride[0])
if rank != world_size - 1:
    next_top_halo_width = self._calc_top_halo_width(rank + 1, patch_height_index, self.kernel_size[0], self.padding[0], self.stride[0])
    next_top_halo_width = max(0, next_top_halo_width)

```

4. 进行 halo 区域的发送与接收

异步发送，同步接收

```python
to_next = None
to_prev = None
top_halo_recv = None
bottom_halo_recv = None
if next_top_halo_width > 0:
    bottom_halo_send = input[:, :, -next_top_halo_width:, :].contiguous()
    to_next = dist.isend(bottom_halo_send, rank + 1)

if halo_width[0] > 0:  # not rank 0
    top_halo_recv = torch.empty([bs, channels, halo_width[0], w], dtype=input.dtype, device=f"cuda:{rank}")
    dist.recv(top_halo_recv, rank - 1)

if prev_bottom_halo_width > 0:  # not rank N-1
    top_halo_send = input[:, :, :prev_bottom_halo_width, :].contiguous()
    to_prev = dist.isend(top_halo_send, rank - 1)

if halo_width[1] > 0:
    bottom_halo_recv = torch.empty([bs, channels, halo_width[1], w], dtype=input.dtype, device=f"cuda:{rank}")
    dist.recv(bottom_halo_recv, rank + 1)

```

5. 拼接 halo 区域

```python
if halo_width[0] < 0:  # Remove redundancy at the top of the input
input = input[:, :, -halo_width[0]:, :]
 
if top_halo_recv is not None:  # concat the halo region to the input tensor 
    input = torch.cat([top_halo_recv, input], dim=-2)
if bottom_halo_recv is not None:
    input = torch.cat([input, bottom_halo_recv], dim=-2)
```

6. 等待发送完成再开始计算

```python
if to_next is not None:
    to_next.wait()
if to_prev is not None:
    to_prev.wait()
```

7. 进行卷积和后处理

为了减少 memory spike 一次计算 block_size*block_size 的区域，并将结果拼接起来

```python
padding = self._adjust_padding_for_patch(self._reversed_padding_repeated_twice, rank=rank, world_size=world_size)
if self.block_size == 0 or (h <= self.block_size and w <= self.block_size):
    if self.padding_mode != 'zeros':
        conv_res = F.conv2d(F.pad(input, padding, mode=self.padding_mode),
                            weight, bias, self.stride, _pair(0), self.dilation, self.groups)
    else:
        conv_res = F.conv2d(input, weight, bias, self.stride,
                            self.padding, self.dilation, self.groups)
    return conv_res
else:
    if self.padding_mode != "zeros":
        input = F.pad(input, padding, mode=self.padding_mode)
    elif self.padding != 0:
        input = F.pad(input, padding, mode="constant")

    _, _, h, w = input.shape
    num_chunks_in_h = (h + self.block_size - 1) // self.block_size  # h 维度的 block 数量
    num_chunks_in_w = (w + self.block_size - 1) // self.block_size  # w ...
    unit_chunk_size_h = h // num_chunks_in_h
    unit_chunk_size_w = w // num_chunks_in_w
  
outputs = []
for idx_h in range(num_chunks_in_h):
    inner_output = []
    for idx_w in range(num_chunks_in_w):
        start_w = idx_w * unit_chunk_size_w
        start_h = idx_h * unit_chunk_size_h
        end_w = (idx_w + 1) * unit_chunk_size_w
        end_h = (idx_h + 1) * unit_chunk_size_h

        # 计算每个块的开始和结束索引，调整块的边界
        # ...

        # 对当前块执行卷积操作
        inner_output.append(
            F.conv2d(
                input[:, :, start_h:end_h, start_w:end_w],
                weight,
                bias,
                self.stride,
                0,
                self.dilation,
                self.groups,
            )
        )
    outputs.append(torch.cat(inner_output, dim=-1))
return torch.cat(outputs, dim=-2)

```
