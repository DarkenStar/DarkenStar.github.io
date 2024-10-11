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

从 CLI 解析的参数后会在 `create_config()` 中组成如下的 `ParallelConfig`.

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

解析完配置参数并构建了 `engine_config` 后，下一步是构建模型的 pipeline. 里面会初始化全局变量 `_RUNTIME` 有关模型参数的部分和模型并行的全局变量 `_DP, _CFG, _PP, _SP, _TP`，它是一个 DiTRuntimeState (继承 RuntimeState) 实例，记录了每个 Group 包含的设备索引，除此之外还包括 PipeFusionParallel 中有关 patch 索引的参数 (在稍后 pipeline 执行的时候计算).

```python
    pipe = xFuserPixArtAlphaPipeline.from_pretrained(  # First construct a PixArtAlphaPipeline, then pass it and engine_config to xFuserPipelineBaseWrapper
        pretrained_model_name_or_path=engine_config.model_config.model,
        engine_config=engine_config,
        torch_dtype=torch.float16,
    ).to(f"cuda:{local_rank}")
    pipe.prepare_run(input_config)
```

## Construct Parallel Groups

`RankGenerator` 确定每个并行的 group 包含哪些设备，先把并行方法按照并行度从小到大排列成 `tp-sp-pp-cfg-dp`. 再根据要生成的并行组产生对应的 `mask`. 即如果要生成 `pp` 组对应的 rank，那么 `mask = [0, 0, 1, 0, 0]`

首先要用 exclusive_prefix_product 计算 `global_stride`，即每个并行度的一个组包含几个设备。再根据 `mask` 取出对应的 `mask_stride` 和 `unmaskd_stride`. `group_size = mask_stride[-1]` 即为最大并行度的组包含的设备数。`num_of_group = num_of_device / mask_stride[-1]` 即为要生成几个并行度最大的组。

最后循环计算得到每个 group 包含的设备索引。

```python
ranks = [] 
for group_index in range(num_of_group):  # 对于每个 group
    # 从当前的 group_index 中计算出 unmasked 并行方法的局部索引
    decomposed_group_idx = decompose(group_index, unmasked_shape)
    rank = []
    for rank_in_group in range(group_size):
        # 计算当前组内每个设备对应的 mask 并行方法的索引
        decomposed_rank_idx = decompose(rank_in_group, masked_shape)
        rank.append(
            inner_product(decomposed_rank_idx, masked_stride)  # 设备组内 rank 在 mask 并行方法的位置
            + inner_product(decomposed_group_idx, unmasked_stride)  # 当前组在全局的起始位置
        )
    ranks.append(rank)
return ranks
```

例如在 `tp=1, sp=2, pp=4, cfg=1, dp=1` 的情况下，生成的并行组如下图所示。

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

会对 transformer block 进行分配，如果 parallel_config 指定了 attn_layer_num_for_pp，即存有每个 pipeFusion 的设备被分配的 transformer block 数量的列表，按其进行分配；否则平均分。

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

首先会初始化 pipeline，rank 0 会接收 warmup 阶段的 latents 然后沿着 H 维度进行分块，rank -1 也会沿着 H 维度进行分块。然后为每个 patch 创建接收的任务，注意 rank 0 第一次是从 warmup 阶段接收 latents，所以他的接收任务少一个。
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
                patch_latents[patch_idx],
                last_patch_latents[patch_idx],
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

        get_runtime_state().next_patch()  # 切换到下一个 (self.pipeline_patch_idx + 1) % self.num_pipeline_patch

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
