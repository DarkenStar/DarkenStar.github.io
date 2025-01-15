---
title: Comparsion of Parallelsim Metods in ViT
date: 2024/11/13 16:05:23
categories: Paper Reading
tags: Distributed Training
excerpt: Paper reading of Pyramid Attention Broadcast.
mathjax: true
katex: true
---
# Basic Transformer Block

符号含义表示如下

| Symbol | Description             | Symbol | Description |
| ------ | ----------------------- | ------ | ----------- |
| a      | 注意力头数              | n      | 并行度大小  |
| b      | batchsize               | s      | 序列长度    |
| h      | 隐藏层维度              | v      | 词汇表大小  |
| L      | tranformer layer 层数数 |        |             |

基本 transformer block 结构如下，输入是形状为 (b, s, h) 的三维张量，其中 b 为 batchsize. 每个变压器层由一个具有注意头的自注意块组成，随后是一个具有两层的 MLP，第一层将隐藏维度增加到 4h，然第二层将其减少到 h. 每个变压器层的输入和输出具有相同的形状.

![Basic Transformer Architecture](https://note.youdao.com/yws/api/personal/file/WEB6446c9e0a905932db1f9e39fa91c01ba?method=download&shareKey=f26e075bcfc51b8c093388f69d39b40d "Basic Transformer Architecture")
![Self-attention Block](https://note.youdao.com/yws/api/personal/file/WEBfbdc229aca70349939d6e3306e78c434?method=download&shareKey=cff3f2903a8e16c5c46d607749a4b3c1 "Self-attention Block")

## Model Parameters

QKVO Linear 的权重形状均为 `h*h`, 偏置形状均为 `h*1`；MLP 两个 Linear 的权重形分别为 `h*4h` 和 `4h*h`，偏置形状分别为 `4h*1` 和 `h*1`. 因此每个模型的参数量为 `(12hh+13h)L`，占用大小还要 `x2`.

{% note info %}
在传统的 LLM 中最后还需要经过 logits layer，将隐藏层维度 `h` 转换成词汇表大小 `v`，参数量还要加上 `hv`.
{% endnote %}

## FLOPs Calculation

对于浮点数计算量 (FLOPs)，只考虑占主要部分的通用矩阵乘法 (GEMMs). 对于 Attention 部分，QKV Linear 的计算量为 `6bshh`，attention matrix (Q@K.T) 的计算量为 `2bssh`, attention@V 的计算量为 `2bssh`, O Linear 的计算量为 `2bshh`. MLP 的两个线性层的每一个计算量都为 `8shh`. 相加后得到正向传播中总计算量为 `(24bshh + 4bssh)L` bytes.

{% note info %}
在传统的 LLM 中最后还需要经过 logits layer，将隐藏层维度 `h` 转换成词汇表大小 `v`，其计算量为 `2bshv`.
{% endnote %}

反向传播因为要计算输入和权重的梯度，其计算量为正向传播的两倍，因此整个模型的计算量为 `72BLshh(1+s/(6h))`.

# Activation Memory

激活的定义为在前向传播中产生并且需要在反向传播中进行梯度计算的张量，即不包括模型参数和优化器状态。并且不考虑相对非常小的激活。例如 LayerNorm 层的输入还需要张量每个通道的均值和方差 (大小均为 bs)，由于 h 大小通常超过 1k，因此只考虑输入张量所占激活的大小 bsh，忽略掉 2bs. 假设数据格式为 fp16/bf16，即每个数据占用 2 bytes 的存储空间，需要特殊处理的是 dropout 层的 mak，每个元素均为 unsigned int，只占用 1 byte.

Attention 部分激活占用如下 (共计 11bsh + 5bssa)

- QKV Linear: 三个线性层需要的输入相同，占用 2bsh bytes.
- Q@K.T: 需要存储 Q 和 K，占用 4bsh bytes.
- Softmax: 需要存储大小为 2bssa bytes 的输入
- Softmax droppot: 需要存储一个大小为 bssa bytes 的 mask.
- attention@V: 需要存储 dropout 的输出和 V，分别占用 2bssa 和 2bsh bytes.
- O Linear: 需要存储注意力的输出，占用 2bsh bytes.
- O dropout 需要存储一个大小为 bsh bytes 的 mask;

MLP (共计 18bsh): 第一层和第二层的输入分别占用 2bsh 和 8bsh bytes. GeLU 层需要第二层的输入用于反向传播，占用大小为 8bsh bytes. dropout 需要一个大小为 bsh bytes 的 mask.

LayerNorm (共计 4bsh): 需要存储该层的输入，占用 2bsh bytes. 一共有两个 LayerNorm.

加起来就可以得到每个 transformer block 需要激活大小为 bsh(34+5sa/h) bytes.

# Tensor Parallelsim

[Megatron 张量并行](https://darkenstar.github.io/2024/10/02/MegatronLM/#Model-Parallel-Transformers) 的思想是将输入进行连续的两个矩阵乘法的第一个按列切分成 t 份，第二个按行切分成 t 份. 在 Transformer block 中体现为利用多头注意力本身的并行性将 Attention 计算中的 QKV 按列进行切分，O Linear 的权重按行进行切分；MLP 中第一个线性层的权重按列进行切分，第二个权重按行进行切分。

在这种并行方式下，前向传播和反向传播均需要进行 2 次 All-Reduce 通信，由于每次 All-Reduce 通信可以看作 Reduce-Scatter + All-Gather, 因此每次每个设备的通信量为 8αbsh bytes，其中 α=(n-1)/n.

对于激活，2*LayerNorm, QKV Linear 的输入, O dropout mask，MLP 第一层的输入和 MLP dropout 不会被切分，因此每个设备每个 block 要占用的激活为 bsh(10+24/n+5as/(hn))

2D Tensor Parallelsim

2D张量并行将激活第一个矩阵的列切分成 m*n 份，第二个权重 (权重形状为 he) 的行被切分成 m 份，列被切分成 n 份。以下图为例，Rank0-Rank2为通信组 x，Rank0-Rank1为 通信组 y. 第一个矩阵经过一次通信组 y 的 AllGather 后与本设备第二个矩阵进行矩阵乘积，得到的部分和经过一次通信组 x 间的ReduceScatter，计算出正确结果。第一次 AllGather 通信每个设备通信的大小为 bsh(n-1)/(mn). 第二次 ReduceScatter 通信每个设备通信的大小为 bse(m-1)/n.



# Megatron Sequence Parallelsim

Megatron 张量并行中 LayerNorm 以及 O Linear 和 MLP 之后的 dropouts 在每个设备中都有一个副本。这些模块不需要大量的计算，但需要占用 10bsh bytes 大小的激活内存。[Megatron-SP]() 沿着序列维度划分这些模块来减少激活内存，但需要配合 TP 一起使用，本质上是将 TP 中的 All-Reduce 拆成了在 TP 前进行 All-Gather 和在 TP 后进行 Reduce-Scatter. 但除去第一个 LayerNorm 外的每一个模块的激活都得到了切分。Megatron-SP 这里选择每个设备存储自己的部分并在反向传播中插入一次额外的 All-Gather 通信。因此通信量为 10bsh, 每个设备每个 block 需要占用的激活为 bsh/n*(34+5as/h)

![Transformer layer with Megatron-SP](https://note.youdao.com/yws/api/personal/file/WEB6800d68e35ee4215289de6aa75f01884?method=download&shareKey=e67ffd54e4d1fe7cf3a10e81108af366 "Transformer layer with Megatron-SP")

# Pipeline Parallelsim

流水线张量并行仅仅将 L 个 Transformer block 平均分到 p 个设备上，并没有划分激活所要占用的内存。在考虑 1F1B 策略下 batchsize 进一步被划分成 p 个 micro batch. 第一个 stage 必须存储 p 个 micro batch 的激活。每个 stage 包含 L/p 层，所以无论流水线并行大小 p 如何，第一个 stage 必须存储 p × L/p = L 层的激活值。在 Megatron-LM 中的 interleaving schedule 需要存储 L(1 + (p−1)/(pm)) 层的激活，其中 m 是 interleaving 的数量。

{% note info %}
在使用 output-tensor-deallocation 优化 (输出传到下一个 stage 后就释放) 的情况下，可以为为每个设备节省 bshr 内存，其中 r 是每个设备正在运行的 micro batch 的数量，在第一个 stage r=p 时达到峰值。
{% endnote %}

# Deepseed-Ulysses Sequence Parallel

[DS-SP](https://darkenstar.github.io/2024/10/21/Deepseed%20Ulysses/) 也是利用多头注意力的并行性，首先将输入按序列维度切分到每个设备上，每个设备占有的输入形状为 b*(s/n)*h. 在计算 Attention 之前对 QKV 进行 All-to-All 通信变成按隐藏层维度切分 ((a 要能整除 n))，通信量为 6αbsh/n bytes. 计算完 score@v 之后再进行一次 All-to-All 通信，通信量为 2αbsh/n bytes，总计通信量为 8αbsh/n bytes. 激活占用上 Attention 中 Softmax 及其 dropout mask 和 attention 没有被切分，激活占用量为 bsh(34/n+5sa/h). 因此，它不适合 GQA 和 MQA 情况, GQA 的并行度被限制在了组数，MQA 则完全没法使用。而且由于张量并行也需要在 a 维度上进行划分，SP-Ulysses 和 TP 是冲突的。

# Ring-Attention Sequence Parallel

[Ring-SP](https://darkenstar.github.io/2024/09/26/Ring_Attention/#Putting-it-Together) 实际上为环状的 FlashAttention，将输入沿着序列维度切分到每个设备上，在 Attention 计算过程中每个设备向相邻设备通信 KV 并更新自己的 Softmax 矩阵，通信量为 4bsh bytes. 激活占用和 DS-SP 一样为 bsh(34/n+5sa/h).

# Unified Sequence Parallel

[USP](https://darkenstar.github.io/2024/11/14/USP-A%20Unified%20Sequence%20Parallelism%20Approach%20for%20Long%20Context%20Generative%20AI/#Unified-Ulysses-Ring-Sequence-Parallelism) 将 SP 进程组分割成两个正交的进程组：SP-Ring 进程组和 SP-Ulysses 进程组。可以将其视为一个 2D mesh ，每一列上运行 SP-Ring，每一行上运行 SP-Ulysses. 具体方法为 QKV 的切分 和 All-to-All 和 DS-Ulysses 相同，然后采用 Ring-Attention 的方式进行计算。如果遇到使用 casual mask 的情况需要加上 balance load 策略，把序列长度分为 2*(ring_degree) 大小，按照 0->1->...->(ring_degree-1)->(ring_degree-1)->...->0 的顺序进行分配。USP 消除了 SP-ulysses的头数限制。并且 USP可以通过调整 SP-Ulysses 进程组数目来更好的适应不同带宽的网络结构，可以让 All-to-All 操作在高带宽中运行，而异步 P2P 通信在低带宽部分运行。

# Comparsion of Different Parallelsim in Training

<table border="1">
  <tr>
    <th rowspan="2"></th>
    <th colspan="4" style="text-align: center;">Communication (FWD+BWD)</th>
    <th rowspan="2">Split Dim</th>
    <th colspan="3" style="text-align: center;">Memory</th>
  </tr>
  <tr>
    <th>Param</th>
    <th>Cost</th>
    <th>Act</th>
    <th>Cost</th>
    <th>P/G</th>
    <th>OS</th>
    <th>Act</th>
  </tr>
  <tr>
    <td>DS-SP</td>
    <td>AllReduce</td>
    <td>12O(h²)</td>
    <td>8*All2All</td>
    <td>(8/N)O(bsh)</td>
    <td>a/s</td>
    <td>P+G</td>
    <td>6P</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>Ring-SP</td>
    <td>AllReduce</td>
    <td>12O(h²)</td>
    <td>P2P</td>
    <td>4O(bsh)</td>
    <td>L/L</td>
    <td>P+G</td>
    <td>6P</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>DP</td>
    <td>AllReduce</td>
    <td>12O(h²)</td>
    <td>0</td>
    <td>0</td>
    <td>b/b</td>
    <td>P+G</td>
    <td>6P</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>ZeRO1</td>
    <td>AllGather + ReduceScatter</td>
    <td>12O(h²)</td>
    <td>0</td>
    <td>0</td>
    <td>a/s</td>
    <td>P+G</td>
    <td>6P/N </td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>USP + ZeRO1</td>
    <td>AllGather + ReduceScatter</td>
    <td>12O(h²)</td>
    <td>P2P + 8*All2All</td>
    <td>≤ 4O(bsh)</td>
    <td>a/s</td>
    <td>P+G</td>
    <td>6P/N</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>USP + ZeRO2</td>
    <td>AllGather + ReduceScatter</td>
    <td>12O(h²)</td>
    <td>P2P + 8*All2All</td>
    <td>≤ 4O(bsh)</td>
    <td>a/s</td>
    <td>P+(G/N)</td>
    <td>6P/N</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>USP + ZeRO3</td>
    <td>2*AllGather + ReduceScatter</td>
    <td>18O(h²)</td>
    <td>P2P + 8*All2All</td>
    <td>≤ 4O(bsh)</td>
    <td>a/s</td>
    <td>(P+G)/N</td>
    <td>6P/N</td>
    <td>A/N</td>
  </tr>
  <tr>
    <td>TP</td>
    <td>0</td>
    <td>0</td>
    <td>4*AllReduce</td>
    <td>8O(bsh)</td>
    <td>a/h</td>
    <td>(P+G)/N</td>
    <td>6P/N</td>
    <td>αA</td>
  </tr>
  <tr>
    <td>Megatron-SP</td>
    <td>0</td>
    <td>0</td>
    <td>6*AllGather + 4*ReduceScatter</td>
    <td>10O(bsh)</td>
    <td>a/h</td>
    <td>(P+G)/N</td>
    <td>6P/N</td>
    <td>A/N</td>
  </tr>
</table>

# Analysis

1. All2All 通信使得 DS-SP 的通信开销大于 DP. 使用 Ring-SP 时，尽管异步的 P2P 通信是可以重叠的，理想的性能也是只与 DP 相同。因此只有当批 batchsize 不足以进行切分时才考虑使用 SP.
2. Megatron-SP 通信量高于 DS-SP 和 Ring-SP. SP-Ring 对于 KV 的通信可以与计算重叠。Megatron-SP 的通信量不会随着并行度的增加而减少，而 DS-SP 可以做到。 DS-SP 和 Ring-SP 具有较低的激活通信成本，但需要同步梯度和参数。不过参数通信量相对于激活通信量较小，可以通过计算进行重叠。GQA/MQA 也可以降低它俩的通信成本，而 Megatron-SP 不受影响。
3. 相同配置下使用 USP+Zero3 来代替 Megatron-SP 并不会增加可训练序列的长度。但与 Megatron-SP 相比，USP 能在通过提高并行度来增加可以训练的序列长度。
4. Megatron-SP 并行维度受限于注意力头数目。USP 可以通过提高 Ring-SP 的并行度来扩展，以在大规模配置下训练更大模型。

# Sora Inference Modeling Analysis Process

我们需要准备模型的输入：

1. 隐空间采样的噪声 z，形状与想生成的视频时常和分辨率相关。生成 1s 的视频为 25.5 frames，经过 VAE Encoder 后输出的通道数为 4，帧数会被压缩到 `num_frame*5//17`，分辨率的长宽分别被压缩到原来的 1/8. 因此 z 的形状应该为 `(B, 4, num_frame*5//17, img_size[0]//8, img_size[1]//8)`.
2. 输入的 prompt 会经过 DeepFloyd/t5-v1_1-xxl 编码，该编码器最大的 token 数为 300，编码维度为 4096，文本长度不足时会填充到 300. 因此编码后的 prompt 的形状为 `(B, 1, 300, 4096)`.
3. 当前去噪的时间步 t，形状为 `(B, )`
4. 生成视频的 fps，形状为 `(1, )`

还需要准备相关的模型配置，包括 mesh 形状，sub_mesh 的形状，并行策略以及 stage_ids. 如果需要将模型的 transformer block 切分成多段，则需要配置 sub_mesh 和 stage_ids.

- mesh_shape: (num_x, num_y)
- submesh_shape: `[(num_x, num_y, loc_x, loc_y), ]`
- stage_ids: `[(submesh0_start, submesh0_end), ]`
- strategy: 并行策略

然后初始化模型，Sora 的整体结构如下 我们初始化一个 Pipeline(包含整个流程的函数)，它会有一个或多个 Stage 用于保存模型的不同层，与 stage_ids 中对应。我们将模型分解成 Embedding_blocks(PatchEmbed3D, TimestepEmbedder, SizeEmbedder, Captionembedder, t_block), STDiT3_blocks 和 T2IFinalLayer. 将这个分解函数作为 Pipeline 的 sharding_func.

![Open-Sora](https://note.youdao.com/yws/api/personal/file/WEB9688e46ada2523df1ec522a7649be19a?method=download&shareKey=991eba9aad6eca9f41599d2ad4f75c34 "Open-Sora")

## Init Pipeline

我们需要根据配置以及 PipePatch 并行度和 SP 并行度初始化 Pipeline. 这其中会根据 stage_ids 分配每个 Stage 保存模型的哪些层以及对应的 submesh 大小。

```python
def construct_stages(self, submeshes: List[Tuple], stages_ids: List[Tuple]):
    # construct layers for each stage
    first_part, module_list, last_part = self.parse_func(self.model)
    modules = list()
    num = len(stages_ids)
    for idx in range(num):
        submesh = submeshes[idx]
        stage_id = stages_ids[idx]
        # get stage layers from user config stage ids in module list
        layers = list(module_list[stage_id[0]: stage_id[1] + 1])
        if idx == 0 and first_part is not None:
            # concat module first part(if exists) bef module list to stage_0
            layers = first_part + layers
        if idx == num - 1 and last_part is not None:
            # concat module last part(if exists) aft module list to last stage
            layers.extend(last_part)

        modules.append(layers)
        # deepcopy module for xla device tracing use
        stage_module = [copy.deepcopy(layer) for layer in layers]
        self.stages.append(
            Stage(idx, stage_module, submesh, self, ))
    return modules
```

## Write Sharding Function

要根据选择的不同的并行策略对每个 Stage 的模型权重，输入，输出进行切分。这里同样我们单独处理 Embedding_blocks, STDiT3_blocks 和 T2IFinalLayer. 让 stage0 包括对 Embedding_blocks 的处理，stage(N-1) 包括对 T2IFinalLayer 的处理。需要注意的是 DS-ulysses 我们需要对 Q@K.T 的结果 和 S@V 的结果也进行切分 SPMD 才会插入正确的 All2All，因此这部分只能放在网络的 forward 里面进行。

```python
def shard_sora_one_stage(modules, shard_strategy, mesh):
    total_len = len(modules)
    # first 5 modules are embedding layers
    for i in range(0, 5):
        shard_sora_embedding(modules[i], shard_strategy, mesh)
    for i in range(5, total_len - 2):
        shard_sora_block(modules[i][0], shard_strategy, mesh)  # shard spatial
        shard_sora_block(modules[i][1], shard_strategy, mesh)  # shard temporal
    shard_sora_final(modules[-1], shard_strategy, mesh)


def shard_sora_first_stage(modules, shard_strategy, mesh):
    for i in range(0, 5):
        shard_sora_embedding(modules[i], shard_strategy, mesh)
    for i in range(5, len(modules)):
        shard_sora_block(modules[i][0], shard_strategy, mesh)  # shard spatial
        shard_sora_block(modules[i][1], shard_strategy, mesh)  # shard temporal


def shard_sora_stage(modules, shard_strategy, mesh):
    for module in modules:
        shard_sora_block(module[0], shard_strategy, mesh)  # shard spatial
        shard_sora_block(module[1], shard_strategy, mesh)  # shard temporal


def shard_sora_last_stage(modules, shard_strategy, mesh):
    total_len = len(modules)
    for i in range(0, total_len - 2):
        shard_sora_block(modules[i][0], shard_strategy, mesh)  # shard spatial
        shard_sora_block(modules[i][1], shard_strategy, mesh)  # shard temporal
    # skip norm layer mark sharding
    shard_sora_final(modules[total_len - 1], shard_strategy, mesh)
```

## Construct Pipeline

然后为了处理多 stage 的情况，我们需要保存每个 stage 的输入和输出的形状。这一步相当于放到 cuda 上重走一遍整个模型的 forward，记录下每一层输入和输出的形状，保存为 json 一遍。实际上对于每个固定生成大小的视频进行一次就行，下次直接读取这个文件。因为现在都采用 [xformers.ops.memory_efficient_attention](https://facebookresearch.github.io/xformers/components/ops.html)，需要输入张量在 cuda 上，我们需要手动在模型的 forward 函数中写一个 navie 的 attention 计算流程好让 torch_xla 能对张量进行跟踪。

## Trace mhlo Graph

根据上一步得到的每个 Stage 的输入形状，创建输入张量，放入 xla_device 上，执行 forward. 最后导出输出的 mhlo 计算图。这里需要注意第一个 stage 包含多个非连续的模块，因此需要单独处理，最后一个 stage 最后一层的输入与其他 block 不同，因此也要单独处理。

```python
def trace_stage_mhlo_graph(self, check_res=False):
    """
    trace stage nn modules to mhlo graph
    """
    # (NOTE): construct xla mesh before trace tensors generate,
    # i.e., before any xla device call to avoid xla computation client construct
    xla_mesh = None
    if self.shard_func is not None:
        xla_mesh = self._construct_stage_xla_mesh()  # create mesh from submesh info
    # Create xla device trace tensors, move module to xla device
    if self.stage_id == 0:
        self.trace_tensors = self._generate_trace_tensors()
    else:
        z = self.parent_pipeline.stages[self.stage_id -1].outputs
        y = self.parent_pipeline.stages[0].y_embedded.to('cpu').to(xm.xla_device())
        t_mlp = self.parent_pipeline.stages[0].t_mlp.to('cpu').to(xm.xla_device())
        self.trace_tensors = [z, y, t_mlp]
    for module in self.modules:
        if isinstance(module, tuple):
            for mod in module:
                mod.to('cpu').to(xm.xla_device())  # first load to cpu
        else:
            module.to('cpu').to(xm.xla_device())
    # get pipeline exec mode
    assert self.parent_pipeline is not None
    exec_mode = self.parent_pipeline.exec_mode
    # load lora cofifg
    lora_config = self.parent_pipeline.lora_config

    print("Enter trace mhlo graph for stage: ", self.stage_id)
    # Trigger shard func to mark sharding the model
    if self.shard_func is not None:
        self.shard_func(self.modules, self.shard_strategy, xla_mesh)

    if exec_mode == EXEC_MODE.INFER:
        # set stage name & dump file path
        self._set_stage_name_dump_file(
            exec_mode, "fw")
        num_sampling_steps = 30
        num_timesteps = 1000
        timesteps = [(1.0 - i / num_sampling_steps) * num_timesteps for i in range(num_sampling_steps)]
        # FIXME: 原先是为每个stage单独生成trace_tensor, 现在要把上一个的结果传给下一个 stage
        #for i in range(30):
        start = sum(self.parent_pipeline.pipeline_patches_height_list[:self.stage_id - 1]) if self.stage_id != 0 else 0
        end = start + self.parent_pipeline.pipeline_patches_height_list[self.stage_id]
    
        if self.stage_id == 0:
            outputs = self._forward([self.trace_tensors[0][...,start:end,:]] + self.trace_tensors[1:], xla_mesh)  # outputs is a list
        else:
            outputs = self._forward(self.trace_tensors, xla_mesh)

        if check_res:
            # check xla results compared to gpu results
            check_result_error(self.outputs, outputs)
        else:
            # use torch xla _get_xla_tensors_hlo interface
            # to eliminate redundant live tensors as ret values
            os.environ["XLA_DUMP_POST_OPTIMIZATIONS"] = "true"
            torch_xla._XLAC._get_xla_tensors_hlo(outputs)
```

## Analyze mhlo Graph

接下来我们要遍历上一步得出的 mhlo 图。

### OpView

从根节点的 ir 开始遍历上一步导出的整个计算图。根据传入 ir 的类型定义调用对应的 visit 函数读取其属性进行操作。主要通过 rsqrt 的位置来划分一个 Transformer block 中第几个 dot 和 dot_general 对应的是什么操作。对于 Sora 来说划分情况如下。这里需要注意的是 mhlo 图记录的是拓扑排序的顺序，不是程序顺序执行的顺序，因此第一个 block 会掺杂着 Embedding_blocks 的一些 dot 操作。因此我们从第二个 block 的第一个 rsqrt 位置开始统计。

```python
def collect_rms_ops(self):
  rms_collector = RMSCollector()
  rms_collector.visit(self.root_op)
  self.rms_locs = rms_collector.rms_locs
  # construct attention block & ffn block ranges
  # exclude the rsqrt in T2IFinalLayer
  att_rm_locs = self.rms_locs if len(self.rms_locs) % 2 == 0 else self.rms_locs[:-1] 

  for i in range(8, len(att_rm_locs), 4):  # a block has 4 rsqrt, start from 2nd block to avoid embedding
      self.spt_qkv_ranges.append((att_rm_locs[i+0], att_rm_locs[i+1]))
      self.spt_attn_ranges.append((att_rm_locs[i+2], att_rm_locs[i+3]))
      self.cro_attn_ranges.append((att_rm_locs[i+2], att_rm_locs[i+3]))

  for i in range(8, len(att_rm_locs), 4):  # ORG: range(8, len(att_rm_locs), 4): 
      start = self.rms_locs[i+3]
      if i+4 >= len(self.rms_locs):
          end = None
      else:
          end = self.rms_locs[i+4]
      self.ffn_ranges.append((start, end))

```

| module                         | operator                      |
| ------------------------------ | ----------------------------- |
|                                | `RMSNorm(x)`                |
| **Self Attention**       | `dot(x, qkvLinear.weight)`  |
|                                | `RMSNorm(q)`                |
|                                | `RMSNorm(k)`                |
|                                | `dot_general(q, k)`         |
|                                | `dot_general(s, v)`         |
|                                | `dot(attn, oLinear.weight)` |
| **Cross Attention**      | `dot(x, qLinear.weight)`    |
|                                | `dot(y, kvLinear.weight)`   |
|                                | `dot_general(q, k)`         |
|                                | `dot_general(s, v)`         |
|                                | `dot(attn, oLinear.weight)` |
|                                | `RMSNorm(x) `               |
| **Feed Forward Network** | `dot(x, upLinear.weight)`   |
|                                | `dot(x, downLinear.weight)` |

```python
def visit_dot(self, node):
    dot_lineno = _parse_loc_lineno(node)

    if self.block_cnt < len(self.spt_attn_ranges):
        spt_att_range = self.spt_attn_ranges[self.block_cnt]
        cro_att_range = self.cro_attn_ranges[self.block_cnt]
        spt_qkv_range = self.spt_qkv_ranges[self.block_cnt]
        ffn_range = self.ffn_ranges[self.block_cnt]

        # lie in RMS ops closed attention block
        if dot_lineno > spt_att_range[0] and dot_lineno < spt_att_range[1]:
            #import pdb;pdb.set_trace()
            self.att_block_dots.append(node)
            self.spt_dot_cnt += 1
        elif dot_lineno > cro_att_range[0] and dot_lineno < cro_att_range[1]:
            self.att_block_dots.append(node)
            self.cro_att_dot_cnt += 1
        # lie ffn block
        if dot_lineno > spt_qkv_range[0] and dot_lineno < spt_qkv_range[1]:
            self.spt_qkv_cnt += 1
            self.ffn_block_dots.append(node)
            # pixart pass
        elif dot_lineno > ffn_range[0]:
            if ffn_range[1] is not None:
                if dot_lineno < ffn_range[1]:
                    self.ffn_block_dots.append(node)
                    self.ffn_dot_cnt += 1
            else:
                if self.ffn_dot_cnt < 2:                 
                    self.ffn_block_dots.append(node)
                    self.ffn_dot_cnt += 1
        # Traversal of one block
        if self.spt_qkv_cnt == 1 and self.spt_att_dot_cnt == 4 and \
            self.spt_dot_cnt == 4 and self.ffn_dot_cnt == 2:
            self.attention_blocks.append(self.att_block_dots)
            self.ffn_blocks.append(self.ffn_block_dots)
            self.block_cnt += 1
            # reset each block level counters
            self.spt_qkv_cnt = 0
            self.spt_att_dot_cnt = 0
            self.spt_dot_cnt = 0
            self.ffn_dot_cnt = 0
        
            self.att_block_dots = []
            self.ffn_block_dots = []

    self.generic_visit(node)
```

保存好一个 Transformer block 中每个 dot 或 dotgeneral 对应的是什么操作后，我们便可以访问这个 ir. 这里需要注意只要两个相乘的矩阵有一个是二维张量 (比如线性层的权重)，mhlo 都会将另一个 reshape 成二维张量。dot 算子 (`jaxlib.mlir.dialects._mhlo_ops_gen.DotOp`) 两个操作数都是二维的张量，qkvLinear 对应的是第一个 dot 操作。左操作数的 shape 为 `(BST,3C)`. 当两个相乘的矩阵都是 3 维及以上张量的时候就会生成 dot_general 该算子的两个相乘的矩阵都会被 reshape 成三维张量。Self-Attention 的第一个 dot_general 左操作数的 shape 为 `(BTN_A,S,C)`. 这样我们就可以得到 `BT=(BST)/S, N_A=(BTN_A)/(BT)`. 同样我们可以得到 OLinear, FFN 中 upLinear 和 downLinear 权重的形状. 以及 Cross-Attention 模块的对应信息。由于之前遍历是从第二个 block 开始的，因此总层数要 ＋1. 最后将得到的参数打包成一个字典返回。

### Communication View

我们以同样的方式定义各种集合通信算子的 visit 函数用于评估该算子的通信量，遍历到对应的 ir 后调用它。

AllReduce 将所有的数据通过规约操作集成到各个设备中。

![AllReduce](https://note.youdao.com/yws/api/personal/file/WEB6e4d9c026bc0632af5040321998fb3ab?method=download&shareKey=f901430ac6bfa781d0b462f0170981d3 "AllReduce")

在 Ring-AllReduce 的 ReduceScatter 步骤中，每个进程发送 M 个元素 N-1 次，总共为 M(N-1). 在 AllGather 步骤中，每个进程发送它计算的块的结果。这是额外的 M 个元素发送了 N-1 次。总的通信量加起来是 2M(N-1).

![Ring-AllReduce](https://note.youdao.com/yws/api/personal/file/WEB69d2b3957cd1863481bff0e785dc9a82?method=download&shareKey=32e60903bafe5dbf240af91c67486e1b "Ring-AllReduce")

All-Gather 示意图如下，每个设备开始拥有初始的一部分数据，通信后每个设备都有一份完整的数据。总的通信量为 M(N-1).

![AllGather](https://note.youdao.com/yws/api/personal/file/WEBe7e6e7a1230ed9ba7ba037556e489d51?method=download&shareKey=5afdf2b669a500a6844aa9e281fe1ac3 "AllGather")

All2All 示意图如下，每个设备把自己的第 i 块数据发送给第 i 个设备。

![All2All](https://note.youdao.com/yws/api/personal/file/WEBddc785dcc80dd741fc1f469a85823cd4?method=download&shareKey=085c0a5681116b4e1683d4d6ae5d080f "All2All")

基于 Bruck 算法的 All2All 流程如下

1. 局部循环移位 (Local Shift of Data-Blocks)
   每个进程将其本地的数据块重新排列，进行初始的循环移位。对于进程 p 和数据块索引 i: R[i]=S[(p+i)%P]. 其中 S[i] 是进程本地初始的数据，R[i] 是移位后的数据。
2. 全局通信 (Global Communication)
   一共进行 log(P) 次通信。
   每一步中每个进程将一部分数据发送给相邻的进程，并接收相邻进程发送的数据。若数据块索引 i 用 radix-2 表示的第 k 位为 1，则数据块会被发送到目标进程。
   对于进程 p: 发送数据到进程 ((p + 2^k) % P)，接收来自进程 ((p - 2^k) % P) 的数据。
   每次发送后，进程将接收到的数据更新到其本地数据中。
3. 局部逆向移位 (Local Inverse Shift of Data-Blocks)
   在完成所有全局通信之后，每个进程执行逆向移位，以恢复数据块的正确顺序。对于每个数据块索引 i: R[i]=R[(p−i+P)%P]

在进程是 2 次幂的情况下每个设备每次要通信 M*P/2大小的数据，总共为 MPlog(P)/2.

![Example of the Bruck Algorithm with 4 Processes](https://note.youdao.com/yws/api/personal/file/WEBa083a4c6002019e62b23c0b24b59a812?method=download&shareKey=58f1b8055307d53e43ce86b9e1762989 "Example of the Bruck Algorithm with 4 Processes")

### TFLOPS View

计算量主要分成两种，element-wise 的操作计算量为元素个数。两个形状分别为 mxk 和 kxn 的矩阵相乘计算量为 2mkn. 被计入 element-wise 操作的算子有 add, subtract, multiply, divide, rsqrt, negate, exponential. 被计入矩阵乘法的算子有 dot, dot_general.

## Performance Analysis

我们根据提取出的 Transformer block 的信息送入性能分析器进行分析. tx8 的配置如下

| Parameter         | Value  |
| ----------------- | ------ |
| TILE_NUM          | 16     |
| SRAM (MB)         | 3      |
| NOC BW (GB/s)     | 128    |
| DRAM BW (GB/s)    | 100    |
| DRAM LATENCY (us) | 0.1    |
| GEMM (TFLOPS)     | 8      |
| VECTOR (TOPS)     | 0.0625 |
| HOP LATENCY (us)  | 0.01   |

根据提取出的信息构建的 STDiT 的 spt_blk, tmp_blk, cross_blk 的参数字典如下.

```python
spatial_config = {"B": self.config["B_spt"], "S_Q": self.config["S_Q_spt"], "S_KV": self.config["S_KV_spt"], "D_QKV": self.config["D_QKV"], 
                  "H_QKV": self.config["H_QKV"], "N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_spt"], "H_O": self.config["H_O_spt"] }
temporal_config = {"B": self.config["B_tmp"], "S_Q": self.config["S_Q_tmp"], "S_KV": self.config["S_KV_tmp"], "D_QKV": self.config["D_QKV"], 
                  "H_QKV": self.config["H_QKV"], "N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_tmp"], "H_O": self.config["H_O_tmp"] }
cross_config = {"B": self.config["B_cro"], "S_Q": self.config["S_Q_cro"], "S_KV": self.config["S_KV_cro"], "D_QKV": self.config["D_QKV"], 
                "H_QKV": self.config["H_QKV"],"N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_cro"], "H_O": self.config["H_O_cro"],
                "D_FU": self.config["D_FU"], "H_FU": self.config["H_FU"], "D_FD": self.config["D_FD"], "H_FD": self.config["H_FD"]}
```

根据这些参数再构建每个层的输入输出形状，计算类型和计算量，以 `Gate_ResAdd` 为例:

```python
GB = 2**30

class Gate_ResAdd():
  '''
  Construct each op after MHSA on the config file
  '''
  def __init__(self, config: dict, name: str) -> None:
      self.config = config
      self.name = name
      # {name:{type:"", size:"", ishape:[], wshape:[]/None, oshape:[]}}
      self.ops = {}
      self.construct_model()
  
  def construct_model(self):
      GB = 2**30
      ResAdd_input_shape = [self.config['B'], self.config['S_Q'], self.config['D_O']]
      ResAdd_weight_shape = [1, self.config['D_O']]
      ResAdd_output_shape = ResAdd_input_shape
      ResAdd_compute = 2*ResAdd_input_shape[0]*ResAdd_input_shape[1]*ResAdd_input_shape[2]/GB
      self.ops[self.name+"_"+"ResAdd"] = {"name":"ResAdd", 
                                          "type": "Vector", 
                                          "ishape": ResAdd_input_shape, 
                                          "wshape": ResAdd_weight_shape, 
                                          "oshape": ResAdd_output_shape, 
                                          "compute": ResAdd_compute}
```

就像这样构建整个 Transformer block 的所有操作

```python
class STDIT2_block():
  def __init__(self, config) -> None:
      self.config = config
      # {name:{type:"", size:"", ishape:[], wshape:[]/None, oshape:[]}}
      self.ops = {}
      self.construct_model()
  
  def construct_model(self):
      spatial_config = {"B": self.config["B_spt"], "S_Q": self.config["S_Q_spt"], "S_KV": self.config["S_KV_spt"], "D_QKV": self.config["D_QKV"], 
                      "H_QKV": self.config["H_QKV"], "N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_spt"], "H_O": self.config["H_O_spt"] }
      temporal_config = {"B": self.config["B_tmp"], "S_Q": self.config["S_Q_tmp"], "S_KV": self.config["S_KV_tmp"], "D_QKV": self.config["D_QKV"], 
                      "H_QKV": self.config["H_QKV"], "N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_tmp"], "H_O": self.config["H_O_tmp"] }
      cross_config = {"B": self.config["B_cro"], "S_Q": self.config["S_Q_cro"], "S_KV": self.config["S_KV_cro"], "D_QKV": self.config["D_QKV"], 
                      "H_QKV": self.config["H_QKV"],"N_A": self.config["N_A"], "H_A": self.config["H_A"], "D_O": self.config["D_O_cro"], "H_O": self.config["H_O_cro"],
                      "D_FU": self.config["D_FU"], "H_FU": self.config["H_FU"], "D_FD": self.config["D_FD"], "H_FD": self.config["H_FD"]}

      self.spatial_modulate = Modulate(spatial_config, name="spatial")
      self.spatial_block = MHSA_block(spatial_config, name="spatial")
      self.spatial_gate_resadd = Gate_ResAdd(spatial_config, name="spatial")
      self.temporal_modulate = Modulate(temporal_config, name="temporal")
      self.temporal_block = MHSA_block(temporal_config, name="temporal")
      self.temporal_gate_resadd = Gate_ResAdd(temporal_config, name="temporal")
      self.cross_block = MHSA_block(cross_config, name="cross")
      self.cross_gate_resadd = Gate_ResAdd(cross_config, name="cross")
      self.mlp_modulate = Modulate(cross_config, name="mlp")
      self.ffn_block = FFN_block(cross_config)
      self.mlp_gate_resadd = Gate_ResAdd(cross_config, name="mlp")

      op_list = [self.spatial_modulate.ops, self.spatial_block.ops, self.spatial_gate_resadd.ops, 
                  self.temporal_modulate.ops, self.temporal_block.ops, self.temporal_gate_resadd.ops, 
                  self.cross_block.ops, self.cross_gate_resadd.ops, self.mlp_modulate.ops, self.ffn_block.ops, self.mlp_gate_resadd.ops]

      for op_dict in op_list:
          self.ops.update(op_dict)
      print(self.ops.keys())
```

然后就可以将构建好的 ops 放入 mapper 进行分析。刚才那些操作会被分成 3 类 `vector_mapper`, `gemm_auto_opt_mapper` 和 `flashatten_mapper`. 我们根据操作的类型送入对应的 mapper 进行分析，具体如下

```python
def STDIT2_mapper(model, arch, QKV_fusion=True, preset=True, details=True):
  config = model.config
  Layers = config['L']
  spatial_config = {'B': config['B_spt'], 'S_Q': config['S_Q_spt'], 'S_KV': config['S_KV_spt'], 'H_A': config['H_A'], 'N_A': config['N_A'], 'Q': config['Q']}
  temporal_config = {'B': config['B_tmp'], 'S_Q': config['S_Q_tmp'], 'S_KV': config['S_KV_tmp'], 'H_A': config['H_A'], 'N_A': config['N_A'], 'Q': config['Q']}
  cross_config = {'B': config['B_cro'], 'S_Q': config['S_Q_cro'], 'S_KV': config['S_KV_cro'], 'H_A': config['H_A'], 'N_A': config['N_A'], 'Q': config['Q']}
  ops = model.ops
  mapping_result = {}
  
  '''=========================
  == Spatial Branch Mapping ==
  ========================='''
  TmTn = [256, 32] if preset else None
  mapping_result['spatial_Modulate'] = vector_mapper(ops['spatial_Modulate'],arch,splits=None,details=details)
  mapping_result['spatial_RMSNorm']= vector_mapper(ops['spatial_RMSNorm'],arch,splits=None,details=details)
  mapping_result['spatial_Q_proj'] = gemm_auto_opt_mapper(ops['spatial_Q_proj'], arch, TmTn=TmTn, details=details)
  mapping_result['spatial_K_proj'] = gemm_auto_opt_mapper(ops['spatial_K_proj'], arch, TmTn=TmTn, details=details)
  mapping_result['spatial_V_proj'] = gemm_auto_opt_mapper(ops['spatial_V_proj'], arch, TmTn=TmTn, details=details)
  Tx_Ty = [256, 256] if preset else None
  mapping_result['spatial_Flashatten'] = flashatten_mapper(spatial_config, arch, Tx_Ty=Tx_Ty, details=details, Head_fused=True)  # FIXME
  mapping_result['spatial_ResAdd']=vector_mapper(ops['spatial_ResAdd'],arch,splits=None,details=details)
  
  '''==========================
  == Temporal Branch Mapping ==
  =========================='''
  mapping_result['temporal_Modulate'] = vector_mapper(ops['temporal_Modulate'],arch,splits=None,details=details)  # 切分 30 份也无法满足SRAM要求
  mapping_result['temporal_RMSNorm']= vector_mapper(ops['temporal_RMSNorm'],arch,splits=None,details=details)
  mapping_result['temporal_Q_proj'] = gemm_auto_opt_mapper(ops['temporal_Q_proj'], arch, TmTn=TmTn, details=details)
  mapping_result['temporal_K_proj'] = gemm_auto_opt_mapper(ops['temporal_K_proj'], arch, TmTn=TmTn, details=details)
  mapping_result['temporal_V_proj'] = gemm_auto_opt_mapper(ops['temporal_V_proj'], arch, TmTn=TmTn, details=details)
  Tx_Ty = [256, 256] if preset else None
  mapping_result['temporal_Flashatten'] = flashatten_mapper(temporal_config, arch, Tx_Ty=Tx_Ty, details=details, Head_fused=True)  # FIXME
  mapping_result['temporal_ResAdd']=vector_mapper(ops['temporal_ResAdd'],arch,splits=None,details=details)
  
  '''====================================
  == Cross Branch Mapping 2x per block ==
  ===================================='''
  #mapping_result['spatial_RMSNorm']= vector_mapper(ops['spatial_RMSNorm'],arch,splits=None,details=details)
  mapping_result['cross_Q_proj'] =  gemm_auto_opt_mapper(ops['cross_Q_proj'], arch, TmTn=TmTn, details=details)
  mapping_result['cross_Q_proj_2'] = mapping_result['cross_Q_proj']
  mapping_result['cross_K_proj'] =  gemm_auto_opt_mapper(ops['cross_K_proj'], arch, TmTn=TmTn, details=details)
  mapping_result['cross_K_proj_2'] = mapping_result['cross_K_proj']
  mapping_result['cross_V_proj'] =  gemm_auto_opt_mapper(ops['cross_V_proj'], arch, TmTn=TmTn, details=details)
  mapping_result['cross_V_proj_2'] = mapping_result['cross_V_proj']
  Tx_Ty = [256, 256] if preset else None
  mapping_result['cross_Flashatten'] =  flashatten_mapper(cross_config, arch, Tx_Ty=Tx_Ty, details=details, Head_fused=True)  # FIXME
  mapping_result['cross_Flashatten_2'] = mapping_result['cross_Flashatten']
  mapping_result['cross_ResAdd'] =  vector_mapper(ops['cross_ResAdd'],arch,splits=None,details=details)  
  # HACK: Gate_ResAdd *2 了, cross 无gate 这里无 _2
  
  
  '''====================================
  == Feed Forward Network 2x per block ==
  ===================================='''
  mapping_result['mlp_Modulate'] = vector_mapper(ops['mlp_Modulate'],arch,splits=None,details=details)
  mapping_result['mlp_Modulate_2'] = mapping_result['mlp_Modulate']
  mapping_result['FFNup&SiLU'] = gemm_auto_opt_mapper(ops['FFNup'],arch,TmTn=TmTn,fusion_op2=ops['SiLU'],details=details)
  mapping_result['FFNup&SiLU_2'] = mapping_result['FFNup&SiLU']
  # mapping_result['FFNgate'] = gemm_auto_opt_mapper(ops['FFNgate'], arch, TmTn=TmTn, details=details)
  # mapping_result['Hadamard'] = vector_mapper(ops['Hadamard'], arch, splits=None)
  TmTn = [4, 128] if preset else None
  mapping_result['FFNdown'] = gemm_auto_opt_mapper(ops['FFNdown'], arch, TmTn=TmTn, details=details)
  mapping_result['FFNdown_2'] = mapping_result['FFNdown']
  mapping_result['mlp_ResAdd'] = vector_mapper(ops['mlp_ResAdd'], arch, splits=None, details=details)
  mapping_result['mlp_ResAdd_2'] = mapping_result['mlp_ResAdd']
```

mapper 会遍历所有可能的切分策略放入 tx8 执行并选择最好的那一个。对于 vector 类型的算子只会沿着 sequence 维度切分；对于 GEMM 算子则会沿着 m, k, n 维度都进行切分；对于 flash-attention 的切分则与原算法相同，外循环遍历 K, V 的每一块，内循环遍历 Q 的每一块。这样就可以得到每个 tx8 上最优的切分方式对应的通信用时，计算用时和利用率。再用之前统计出的每个 die 上通信量除以 die2die 带宽得到通信用时，由此得到总的推理用时。
