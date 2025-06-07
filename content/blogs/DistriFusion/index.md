---
title: "DistriFusion"
date: 2024-10-23T14:28:37+08:00
lastmod: 2024-10-23T14:28:37+08:00
author: ["WITHER"]

categories:
- Paper Reading

tags:
- Diffusion Models

keywords:
- DistriFusion

description: "Paper reading about DistriFusion." # 文章描述，与搜索优化相关
summary: "Paper reading about DistriFusion." # 文章简单描述，会展示在主页
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
# Abstract

DistriFusion 将模型输入分割成多个 patch 后分配给 GPU。但是直接实现这样的算法会破坏 patch 之间的交互并失去保真度，而同步 GPU 之间的激活将产生巨大的通信开销。为了克服这一困境，根据观察到的相邻扩散步输入之间的高度相似性提出了 **displaced patch parallelism**，该方法通过重用前一个时间步骤中预先计算的 feature map 来利用扩散过程的顺序性，为当前步提供 context. 该方法支持异步通信，可以通过计算实现流水线化。

# Introduction

![Original, Navie Patch & DistriFusion](https://note.youdao.com/yws/api/personal/file/WEBdce9158a9908f3ebe7782f7bf5b29f61?method=download&shareKey=a571f8710a6ac4e8b859402edd5c069b "Original, Navie Patch & DistriFusion")

加速扩散模型推理主要集中在两种方法上：减少采样步骤和优化网络推理。随着计算资源的快速增长，利用多个 GPU 来加速推理是很有吸引力的。例如在 NLP 中， LLM 已经成功地利用了 GPU 之间的张量并行性，从而显著降低了延迟。然而，对于扩散模型，由于激活尺寸大，张量并行这样的技术不太适合扩散模型。多个 GPU 通常只用于 batch 推理，当生成单个图像时，通常只涉及一个GPU.

> Techniques like tensor parallelism are less suitable for diffusion models due to the large activation size, as communication costs outweigh savings from distributed computation.

自然而然的一种方法是将图像分成几个 patch 后分配给不同的设备进行生成。由于各个 patch 之间缺乏相互作用，它在每个 patch 的边界处都有一个清晰可见的分界线。

DistriFusion 也是基于 patch parallelism. 关键在于扩散模型中相邻去噪步骤的输入是相似的，因此，只在第一步采用同步通信。后续步骤重用前一步中预先计算的激活，为当前步骤提供全局上下文和 patch 交互。通过异步通信有效地隐藏了计算中的通信开销。并且还稀疏地在指定的区域上进行卷积和注意力计算，从而按比例减少每个设备的计算量。

# Method

## Displaced Patch Parallelism.

在预测 $\epsilon_{\theta}(\mathbf{x}_{t})$ 时 (忽略条件 c 和时间步 t 的输入) ，首先将 $\mathbf{x}_{t}$ 分割成多个 patch $\mathbf{x}_t^{(1)},\mathbf{x}_t^{(2)},\ldots,\mathbf{x}_t^{(N)}$ ，对于每一层 l 和设备 i，在获得输入激活 patch $\mathbf{A}_{t}^{l,(i)}$ 后异步处理两个操作：首先，对于设备i， 激活 $\mathbf{A}_{t}^{l,(i)}$ 首先 scatter 到上一步旧的激活 $\mathbf{A}_{t+1}^{l}$ 中。然后将此分散操作的输出送入稀疏算子 Fl (线性、卷积或注意层)，该算子专门对新区域执行计算并产生相应的输出。同时，对 $\mathbf{A}_{t}^{l,(i)}$ 执行 AllGather 操作，为下一步的全尺寸激活 $\mathbf{A}_{t}^{l}$ 做准备。

![Overview of DistriFusion](https://note.youdao.com/yws/api/personal/file/WEBfee0ed5c1a6065b8adb21371ea3cbc31?method=download&shareKey=66860ad5956c2a8afb949b3fd821015d "Overview of DistriFusion")

我们对除第一层 (采用同步通信获得其他设备上的输入) 外的每一层重复这个过程。然后将最终输出 Gather 在一起以近似 $\epsilon_{\theta}(\mathbf{x}_{t})$，用于计算 $\mathbf{x}_{t-1}$

![Timeline Visualization on Each Device](https://note.youdao.com/yws/api/personal/file/WEB41fa5a52bf206399a49358ada4f5c07b?method=download&shareKey=686b36b99eb8ced2b48594a380d17d62 "Timeline Visualization on Each Device")

## Sparse Operations

对于每一层 l，如果原始算子 Fl 是一个卷积层、线性层或交叉注意层，调整使其专门作用于新激活的区域。这可以通过从 scatter 输出中提取最新部分并将其输入到 Fl 中来实现。对于 self-attention，将其转换为 cross-attention，仅在设备上保留来自新激活的 Q，而 KV 仍然包含整个特征图。

## Corrected Asynchronous GroupNorm

仅对新 patch 进行归一化或重用旧特征都会降低图像质量。同步 AllGather 所有均值和方差将产生相当大的开销。为了解决这一困境，DistriFusion 在陈旧的统计数据中引入了一个校正项。计算公式如下

$$
\mathbb{E}[\mathbf{A}_t]\approx\underbrace{\mathbb{E}[\mathbf{A}_{t+1}]}_{\text{stale global mean}}+\underbrace{\mathbb{E}[\mathbf{A}_t^{(i)}]-\mathbb{E}[\mathbf{A}_{t+1}^{(i)}]}_{\text{correction}}
$$

同样对二阶矩 $\mathbb{E}[\mathbf{A}^2_t]$ 也采用这种计算方式，然后通过 $\mathbb{E}[\mathbf{A}^2_t] - \mathbb{E}[\mathbf{A}_t]^2$ 来计算方差。对于方差结果为负的部分，将使用新鲜 patch 的局部方差代替。

# Code Implementation

Distrifusion 中主要就是将 [UNet2DConditionModel](https://github.com/huggingface/diffusers/blob/9366c8f84bfe47099ff047272661786ebb54721d/src/diffusers/models/unets/unet_2d_condition.py#L71) 中的 Conv2d, Attention 和 GroupNorm 替换成对应的 patch 实现的网络结构 [DistriUNetPP](https://github.com/mit-han-lab/distrifuser/blob/cfb9ea624ef95020aafcda929a69ba4100f99e9d/distrifuser/models/distri_sdxl_unet_pp.py#L15). 这里继承的 BaseModel 类为集成了 PatchParallelismCommManager 类 (介绍见后文) 的网络。

![UNet2DConditionModel](https://note.youdao.com/yws/api/personal/file/WEB6bd750d9e4b5d582be9d1f41cc267bc5?method=download&shareKey=39d825554b65a9c57f59a1dd9a23fb28 "UNet2DConditionModel")

```python {linenos=true}
class DistriUNetPP(BaseModel):  # for Patch Parallelism
    def __init__(self, model: UNet2DConditionModel, distri_config: DistriConfig):
        assert isinstance(model, UNet2DConditionModel)
        if distri_config.world_size > 1 and distri_config.n_device_per_batch > 1:
            for name, module in model.named_modules():
                if isinstance(module, BaseModule):
                    continue
                ''' 
                Substitute Conv2d, Attention, GroupNorm with DistriConv2dPP, DistriSelfAttentionPP, DistriCrossAttentionPP, DistriGroupNorm 
                '''
                for subname, submodule in module.named_children():  
                    if isinstance(submodule, nn.Conv2d):
                        kernel_size = submodule.kernel_size
                        if kernel_size == (1, 1) or kernel_size == 1:
                            continue
                        wrapped_submodule = DistriConv2dPP(  
                            submodule, distri_config, is_first_layer=subname == "conv_in"
                        )
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, Attention):
                        if subname == "attn1":  # self attention
                            wrapped_submodule = DistriSelfAttentionPP(submodule, distri_config)
                        else:  # cross attention
                            assert subname == "attn2"
                            wrapped_submodule = DistriCrossAttentionPP(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)
                    elif isinstance(submodule, nn.GroupNorm):
                        wrapped_submodule = DistriGroupNorm(submodule, distri_config)
                        setattr(module, subname, wrapped_submodule)

        super(DistriUNetPP, self).__init__(model, distri_config)
```

## PatchParallelismCommManager

[PatchParallelismCommManager](https://github.com/mit-han-lab/distrifuser/blob/cfb9ea624ef95020aafcda929a69ba4100f99e9d/distrifuser/utils.py#L112) 类主要处理异步通信的部分。

```python {linenos=true}
class PatchParallelismCommManager:
    def __init__(self, distri_config: DistriConfig):
        self.distri_config = distri_config

        self.torch_dtype = None
        self.numel = 0  # 已经注册的张量的累计总元素数量
        self.numel_dict = {}  # 记录每个 layer_type 所注册的张量的累计元素数量

        self.buffer_list = None  # 在每个设备上存储所有注册张量的数据，通信所用的 buffer

        self.starts = []  # 记录每个注册张量的起始位置 (在 buffer_list 中的起始索引)
        self.ends = []    #                 结束                       结束
        self.shapes = []  # 记录每个注册张量的 shape

        self.idx_queue = []  # 需要进行通信的张量索引的队列

        self.handles = None  # 存储每个设备通信操作的句柄的 list, 用于检查通信是否完成
```

成员函数功能介绍如下

1. `register_tensor(self, shape: tuple[int, ...] or list[int], torch_dtype: torch.dtype, layer_type: str = None) -> int`: 用于注册张量的形状和数据类型，同时计算并记录张量在缓冲区中的起始位置和结束位置。

    * 如果尚未指定 `torch_dtype`，则将传入的 `torch_dtype` 设为类成员的默认数据类型。
    * 计算传入张量形状的总元素数 `numel`，并更新 `starts`、`ends` 和 `shapes` 列表。
    * 如果指定了 `layer_type`，更新 `numel_dict` 中该层类型对应的元素数目。

2. `create_buffer(self)` : 每个设备上为所有注册的张量创建一个统一的缓冲区。

    * 为每个设备创建一个形状为 `(numel,)` 的张量，并将其放入 `buffer_list` 中。
    * 输出在各设备上创建的缓冲区总参数量。

3. `get_buffer_list(self, idx: int) -> list[torch.Tensor]`: 返回每个设备上对应于指定索引 `idx` 的缓冲区张量。

    * 根据 `starts` 和 `ends` 信息，从 `buffer_list` 中提取指定索引 `idx` 的张量片段并调整其形状。

4. `communicate(self)`: 调用 `dist.all_gather` 将缓冲区中的张量在不同设备间进行广播。

    * 确定当前需要通信的张量范围 (根据 `idx_queue` 中的索引).
    * 调用 `dist.all_gather` 在设备组内进行异步广播通信，并将句柄存储在 `handles` 中。

5. `enqueue(self, idx: int, tensor: torch.Tensor)`: 将指定索引 `idx` 处的张量数据复制到 `buffer_list` 中，并将索引添加到通信队列 `idx_queue`。

    * 如果通信队列不为空且索引为 0，则先执行一次通信操作。
    * 将张量数据复制到 `buffer_list` 中的对应位置。
    * 当通信队列长度达到 `distri_config` 中设定的通信检查点值时，进行通信。

6. `clear(self)`: 执行一次所有待通信张量的通信，并等待所有异步操作完成。

    * 如果通信队列不为空，则进行通信操作。
    * 遍历所有句柄，等待所有异步操作完成后，将句柄设为 `None`.

## DistriConv2dPP

[DistriConv2dPP](https://github.com/mit-han-lab/distrifuser/blob/cfb9ea624ef95020aafcda929a69ba4100f99e9d/distrifuser/models/distri_sdxl_unet_pp.py#L10) 计算自己负责 patch 部分的卷积，需要通信其他设备需要自己负责 patch 的上下 padding 部分。

* `__init__`：构造函数，初始化成员变量，设置是否为第一层卷积。
* `naive_forward`：执行标准的前向传播，不进行任何切片操作。这是单个设备处理时的普通卷积操作。
* `sliced_forward`：处理输入张量的切片操作。根据当前设备索引 (`split_idx`) 计算输入张量在高度方向的起始和结束位置，并在必要时为切片后的张量添加 padding 后进行卷积操作。

```python {linenos=true}
class DistriConv2dPP(BaseModule):
    def __init__(self, module: nn.Conv2d, distri_config: DistriConfig, is_first_layer: bool = False):
        super(DistriConv2dPP, self).__init__(module, distri_config)
        self.is_first_layer = is_first_layer

    def naive_forward(self, x: torch.Tensor) -> torch.Tensor:
        #  x: [B, C, H, W]
        output = self.module(x)
        return output

    def sliced_forward(self, x: torch.Tensor) -> torch.Tensor:
        '''...'''

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        distri_config = self.distri_config
        # 等待上一步通信完成
        if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
            if self.comm_manager.handles[self.idx] is not None:
                self.comm_manager.handles[self.idx].wait()
                self.comm_manager.handles[self.idx] = None

        boundary_size = self.module.padding[0]
        if self.buffer_list is None:  # buffer_list 存储的是每个 devive 进行卷积所需要的其他 devive 的数据
            if self.comm_manager.buffer_list is None:
                self.idx = self.comm_manager.register_tensor(
                    shape=[2, x.shape[0], x.shape[1], boundary_size, x.shape[3]],
                    torch_dtype=x.dtype,
                    layer_type="conv2d",
                )
            else:
                self.buffer_list = self.comm_manager.get_buffer_list(self.idx)
                    
            def create_padded_x():
                '''拼接接收到的数据'''
                if distri_config.split_idx() == 0:  # rank 0
                    concat_x = torch.cat([x, self.buffer_list[distri_config.split_idx() + 1][0]], dim=2)
                    padded_x = F.pad(concat_x, [0, 0, boundary_size, 0], mode="constant")
                elif distri_config.split_idx() == distri_config.n_device_per_batch - 1:  # rank n-1
                    concat_x = torch.cat([self.buffer_list[distri_config.split_idx() - 1][1], x], dim=2)
                    padded_x = F.pad(concat_x, [0, 0, 0, boundary_size], mode="constant")
                else:  # other ranks
                    padded_x = torch.cat(
                        [
                            self.buffer_list[distri_config.split_idx() - 1][1],
                            x,
                            self.buffer_list[distri_config.split_idx() + 1][0],
                        ],
                        dim=2,
                    )
                return padded_x
            
            # 提取当前输入张量需要发送给其他设备的部分
            boundary = torch.stack([x[:, :, :boundary_size, :], x[:, :, -boundary_size:, :]], dim=0) 

            # 直接用上一步的 buffer 拼接
            padded_x = create_padded_x()  
            output = F.conv2d(
                padded_x,
                self.module.weight,
                self.module.bias,
                stride=self.module.stride[0],
                padding=(0, self.module.padding[1]),
            )
            if distri_config.mode != "no_sync":
                self.comm_manager.enqueue(self.idx, boundary)  # 插入自己要发送的数据

        self.counter += 1
        return output
```

## DistriSelfAttentionPP

[DistriSelfAttentionPP](https://github.com/mit-han-lab/distrifuser/blob/cfb9ea624ef95020aafcda929a69ba4100f99e9d/distrifuser/modules/pp/attn.py#L107) 只负责计算自己 patch 的输出，需要完整的 KV，将 self attention 运算变成 cross-attention 计算。需要通信自己的 KV.

```python {linenos=true}
class DistriSelfAttentionPP(DistriAttentionPP):
    def __init__(self, module: Attention, distri_config: DistriConfig):
        super(DistriSelfAttentionPP, self).__init__(module, distri_config)

    def _forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0):
        attn = self.module  # 获取 Attention 模块
        distri_config = self.distri_config
        residual = hidden_states  # 残差连接

        batch_size, sequence_length, _ = hidden_states.shape

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)  # Q Projection
        encoder_hidden_states = hidden_states
        kv = self.to_kv(encoder_hidden_states)  # KV Projection

        if self.buffer_list is None:  # 如果缓冲区未创建
            full_kv = torch.cat([kv for _ in range(distri_config.n_device_per_batch)], dim=1)

        new_buffer_list = [buffer for buffer in self.buffer_list]
        new_buffer_list[distri_config.split_idx()] = kv
        full_kv = torch.cat(new_buffer_list, dim=1)
        if distri_config.mode != "no_sync":
            self.comm_manager.enqueue(self.idx, kv)

        # 将 full_kv 分割为 key 和 value
        key, value = torch.split(full_kv, full_kv.shape[-1] // 2, dim=-1)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # multi-head attention
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states, *args)  # O Projection

        hidden_states = attn.to_out[1](hidden_states)  # Dropout

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
```

## DistriGroupNorm

[DistriGroupNorm](https://github.com/mit-han-lab/distrifuser/blob/cfb9ea624ef95020aafcda929a69ba4100f99e9d/distrifuser/modules/pp/groupnorm.py#L9) 根据上一步全特征图的以及当前步 patch 的均值和二阶矩近似当前步的全特征图均值和方差。需要通信 patch 均值和二阶矩。

```python {linenos=true}

class DistriGroupNorm(BaseModule):
    def __init__(self, module: nn.GroupNorm, distri_config: DistriConfig):
        assert isinstance(module, nn.GroupNorm)
        super(DistriGroupNorm, self).__init__(module, distri_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module = self.module
        distri_config = self.distri_config

        if self.comm_manager is not None and self.comm_manager.handles is not None and self.idx is not None:
            if self.comm_manager.handles[self.idx] is not None:
                self.comm_manager.handles[self.idx].wait()
                self.comm_manager.handles[self.idx] = None

        assert x.ndim == 4
        n, c, h, w = x.shape
        num_groups = module.num_groups
        group_size = c // num_groups

        if self.buffer_list is None:
            if self.comm_manager.buffer_list is None:
                n, c, h, w = x.shape
                self.idx = self.comm_manager.register_tensor(  # register for E[x], E[x^2]
                    shape=[2, n, num_groups, 1, 1, 1], torch_dtype=x.dtype, layer_type="gn"
                )
            else:
                self.buffer_list = self.comm_manager.get_buffer_list(self.idx)

        x = x.view([n, num_groups, group_size, h, w])
        # 计算 patch 均值和二阶矩
        x_mean = x.mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        x2_mean = (x**2).mean(dim=[2, 3, 4], keepdim=True)  # [1, num_groups, 1, 1, 1]
        slice_mean = torch.stack([x_mean, x2_mean], dim=0)

        if self.buffer_list is None:
            full_mean = slice_mean
        else:
            # Equation 2 in the paper E[A_t] = E[A_(t+1)] + (E[A^i_t] - E[A^i_(t+1)]), same for E[A^2_t]
            correction = slice_mean - self.buffer_list[distri_config.split_idx()]
            full_mean = sum(self.buffer_list) / distri_config.n_device_per_batch + correction
            self.comm_manager.enqueue(self.idx, slice_mean)

        full_x_mean, full_x2_mean = full_mean[0], full_mean[1]
        var = full_x2_mean - full_x_mean**2
        # 计算方差
        slice_x_mean, slice_x2_mean = slice_mean[0], slice_mean[1]
        slice_var = slice_x2_mean - slice_x_mean**2
        var = torch.where(var < 0, slice_var, var)  # Correct negative variance

        num_elements = group_size * h * w
        var = var * (num_elements / (num_elements - 1))
        std = (var + module.eps).sqrt()
        output = (x - full_x_mean) / std
        output = output.view([n, c, h, w])
        # scale and shift
        if module.affine:
            output = output * module.weight.view([1, -1, 1, 1])
            output = output + module.bias.view([1, -1, 1, 1])
            
        self.counter += 1
        return output
```