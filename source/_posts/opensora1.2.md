---
title: Open-Sora 1.2
date: 2024/9/19 13:39:23
categories: Project
tags: Open-Sora
excerpt: Open-Sora 1.2 STDIT3 Analysis
mathjax: true
katex: true
---
# Configuration

[Open-Sora 推理网址](https://github.com/DeepLink-org/AIChipBenchmark/tree/main/llm/Open-sora)

&emsp;&emsp;配置文件位于 /configs/opensora-v1-2/inference/sample.py, 可以配置的参数如下

```python
resolution = "240p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples/samples/"
seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5

model = dict(
    type="STDiT3-XL/2",
    from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
    qk_norm=True,
    enable_flash_attn=True,
    enable_layernorm_kernel=True,
)
vae = dict(
    type="OpenSoraVAE_V1_2",
    from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
    micro_frame_size=17,
    micro_batch_size=4,
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=300,
)
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
```

&emsp;&emsp;要生成的图像大小 `image_size` 由 `resolution` 和 `aspect_ratio` 计算。若 `aspect_ratio` 存在于 /dataset/aspect.py 预定义好的 `ASPECT_RATIO_{aspect_ratio}` 字典中，则直接取出对应的 `image_size`，计算公式如下

```python
num_pixels = int(resolution**2)
image_size[0] = int(num_pixels/(1+aspect_ratio)*aspect_ratio)
image_size[1] = int(num_pixels/(1+aspect_ratio))
```

&emsp;&emsp;要生成的帧数 `num_frames` 可以直接指定数字或者指定 /dataset/aspect.py 中预定义字典 `NUM_FRAMES_MAP` 里的倍数或者秒数 (fps=25.5).

```python
NUM_FRAMES_MAP = {
    "1x": 51,
    "2x": 102,
    "4x": 204,
    "8x": 408,
    "16x": 816,
    "2s": 51,
    "4s": 102,
    "8s": 204,
    "16s": 408,
    "32s": 816,
}
```

&emsp;&emsp;命令行推理 (禁用 apex 和 flash-attn 需要加上 `--layernorm-kernel False --flash-attn False \`)

```python
python scripts/inference.py configs/opensora-v1-2/inference/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --num-sampling-steps 30 --flow 5 --aes 6.5 \
  --prompt "a beautiful waterfall"
```

# Overview

![Embedding](https://note.youdao.com/yws/api/personal/file/WEB3f5e72ab5eab88d7299cb9f0deba6088?method=download&shareKey=f936795c6da0572463855691cc04d85a "Embedding")
![STDiT3Block](https://note.youdao.com/yws/api/personal/file/WEBc7afbea575846f1adeb3156544fd33c8?method=download&shareKey=56ceb35d92faee737498a9b8772bc3e0 "STDiT3Block*56")
![Final Layer](https://note.youdao.com/yws/api/personal/file/WEBc2e69a1a684ef0c4cac56bc5d8ebccef?method=download&shareKey=30c76b9abc7d94610122c808c66d62c1 "Final Layer")

# Embedder Layer

&emsp;&emsp;会对输入视频噪声 (x)，编码后的 prompt (y)，去噪时间步 (t)，fps 进行嵌入。

## PositionEmbedding2D

```python
class PositionEmbedding2D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        assert dim % 4 == 0, "dim must be divisible by 4"
        half_dim = dim // 2  # 后续会生成两个维度的 embedding (行和列)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # (dim//4, )

    def _get_sin_cos_emb(self, t: torch.Tensor):  # t: (h*w, )
        out = torch.einsum("i,d->id", t, self.inv_freq)  # 相当于外积运算, (h*w, dim//4)
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        return torch.cat((emb_sin, emb_cos), dim=-1)  # (h*w, dim//2)

    @functools.lru_cache(maxsize=512)
    def _get_cached_emb(
        self,
        device: torch.device,
        dtype: torch.dtype,
        h: int,
        w: int,
        scale: float = 1.0,
        base_size: Optional[int] = None,
    ):
        grid_h = torch.arange(h, device=device) / scale  # e.g. (0, 1)
        grid_w = torch.arange(w, device=device) / scale  # e.g. (0, 1, 2)
        if base_size is not None:
            grid_h *= base_size / h
            grid_w *= base_size / w
        # 生成网格，并交换行和列 (width-first indexing)
        grid_h, grid_w = torch.meshgrid(
            grid_w,  # 每一列对应的行坐标 一共有 2 列
            grid_h,  # 每一行对应的列坐标 一共有 3 行
            indexing="ij",
        )  # here w goes first
        grid_h = grid_h.t().reshape(-1)  # (h*w, )
        grid_w = grid_w.t().reshape(-1)  # (h*w, )
        emb_h = self._get_sin_cos_emb(grid_h)  # (h*w, dim//2)
        emb_w = self._get_sin_cos_emb(grid_w)  # (h*w, dim//2)
        return torch.concat([emb_h, emb_w], dim=-1).unsqueeze(0).to(dtype)  # (1, h*w, dim)

    def forward(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        scale: Optional[float] = 1.0,
        base_size: Optional[int] = None,
    ) -> torch.Tensor:
        return self._get_cached_emb(x.device, x.dtype, h, w, scale, base_size)
```

## TimestepEmbedder

&emsp;&emsp;为每个时间步创建唯一的表示，使用正弦和余弦函数来生成时间序列的嵌入。dim 被分为两部分，一部分用于计算正弦嵌入，另一部分用于计算余弦嵌入。`t[:, None] * freqs[None]` 时间步 t 和频率 freqs 进行外积，生成嵌入的值。然后再计算正弦和余弦嵌入，拼接起来。最后将生成的 embedding 矩阵经过两个 Linear 层，得到最终的输出。

```python
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half)  # 生成频率的倒数，使其均匀分布在 log 轴上
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]  # (N, dim//2)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # (N, dim)，前一半列是 cos，后一半列是 sin
        if dim % 2:  # 处理维度为基数的情况，补一个全 0 列使最后维度为 dim
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
```

## CaptionEmbedder (encode_text)

&emsp;&emsp;`CaptionEmbedder` 中的 Mlp 来自于 [timm.models.vision_transformer.Mlp](https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/mlp.py#L13) (Linear->GELU->Dropout->Norm(Identity)->Linear->Dropout). 推理时不进行 `token_drop`.

```python

class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate="tanh"), token_num=120,):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        self.register_buffer(
            "y_embedding",
            torch.randn(token_num, in_channels) / in_channels**0.5,
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption
```

&emsp;&emsp;`mask` 是可选的掩码张量，用来标记 y 中的 token 是有效的 (1) 还是无效的 (0). `masked_select` 会保留 `mask` 中为 1 的索引处的 token 并重新调整成 `(1, N_valid_token, self.hidden_size)` 形状。然后统计 batch 中每个样本有效的 token 数量，并将其转换成列表。如果没有提供 `mask`，则假设所有 token 都有效。

```python
def encode_text(self, y, mask=None):
    y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
    if mask is not None:
        if mask.shape[0] != y.shape[0]:
            mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
        mask = mask.squeeze(1).squeeze(1)
        y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
        y_lens = mask.sum(dim=1).tolist()
    else:  # 没有提供 mask，假设所有 token 都是有效的
        y_lens = [y.shape[2]] * y.shape[0]
        y = y.squeeze(1).view(1, -1, self.hidden_size)  # (1, N_token, self.hidden_size)
    return y, y_lens
```

## PatchEmbed3D

&emsp;&emsp;对输入的视频噪声进行 3D 卷积 (stride 和 kernel_size 设置成 `patch_size` 元组) 后 flatten 并从 `(B, C, T, H, W)` reshape 成 `(B, N, C)`，其中 `N = T * H * W`.

```python
class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(2, 4, 4), in_chans=3, embed_dim=96, norm_layer=None, flatten=True,):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x
```

# STDiT3Block (backbone)

&emsp;&emsp;一个 STDiT3Block 由 Attention，MultiHeadCrossAttention 和 MLP 组成。

## Attention

&emsp;&emsp;正常的注意力机制，使用 `q @ k.transpose(-2, -1)` 计算注意力矩阵，然后使用 softmax 归一化。
&emsp;&emsp;`is_causal` 分支用于处理 因果注意力机制（causal attention）。这是在某些模型（尤其是自回归模型，如 GPT 等生成模型）中非常常见的注意力机制，确保了模型在进行推理或生成时，当前时间步只能关注过去的时间步，不能看到未来的时间步。

```python
class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
  
        self.is_causal = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=self.is_causal,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

## MultiHeadCrossAttention

&emsp;&emsp;这里选择把 B 和 N 维度合并看作一个序列的长度，把计算交叉注意力转换成计算自注意力的过程([xformers.ops.memory_efficient_attention](https://facebookresearch.github.io/xformers/_modules/xformers/ops/fmha.html#memory_efficient_attention))。因此计算噪声和 prompt 的交叉注意力之前加入分块对角掩码 ([BlockDiagonalMask](https://facebookresearch.github.io/xformers/_modules/xformers/ops/fmha/attn_bias.html#BlockDiagonalMask))，标出 batch 中每个 q 和 kv 分别的 sequence 长度。

![Block Diagonal Mask](https://note.youdao.com/yws/api/personal/file/WEB50d575a78f8831989f0962ec53ea02ba?method=download&shareKey=570c5b3c1769bbbbef7bac31ec2331dc "Block Diagonal Mask")

```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)  # (1, BN, #heads, #dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```

## Forward

&emsp;&emsp;涉及自注意力 (self-attention)、交叉注意力（cross-attention）、MLP、时序处理等操作。在多头自注意力和 MLP 的基础上加入了输入的调制(modulation)、输出门控和残差连接 (residual connections).

### Parameters

```python
def forward(self, x, y, t, mask=None, x_mask=None, t0=None, T=None, S=None):
    # prepare modulate parameters
    B, N, C = x.shape
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.scale_shift_table[None] + t.reshape(B, 6, -1)
    ).chunk(6, dim=1)
    if x_mask is not None:
        shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
            self.scale_shift_table[None] + t0.reshape(B, 6, -1)
        ).chunk(6, dim=1)

```

**输入参数** ：

* `x`: 视频噪声 token，形状 `(B, N, C)`，其中 `B` 是 batch 大小，`N` 是 token 数量，`C` 是每个 token 的特征维度。
* `y`: 文本信息，形状 `(B, token_num, C)`。
* `t`: 时间步嵌入 `(B, 6*C)`，用于调制输入。
* `mask`: 文本掩码。
* `x_mask`: 时序掩码，用于控制某些时序数据的处理。
* `t0`: 时间步为 0 时的嵌入，用于对 `x_mask` 处理的特殊处理。
* `T`: 帧数
* `S`: 空间上每个帧的 patch 数

**调制参数** ：

* `self.scale_shift_table` 是用于生成调制参数的表。`t` 被映射为 6 个形状为 `(B, 1, C)` 调制量：`shift_msa`、`scale_msa`、`gate_msa` 用于多头自注意力，`shift_mlp`、`scale_mlp`、`gate_mlp` 用于 MLP。
* 如果 `x_mask` 存在（说明有时序信息），则同样对 `t0`（时间步为 0 的嵌入）生成对应的调制参数。

### t2i_modulate

```python
    # modulate (attention)
    x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)  # x * (1 + shift_msa) * scale_msa
    if x_mask is not None:
        x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
        x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)
```

**调制操作** ：

* `t2i_modulate` 是一个调制函数 (见注释)，通过 `shift` 和 `scale` 对输入进行变换，这里对进行归一化的 `norm1(x)` 进行调制。
* 如果存在 `x_mask`，则根据掩码使用 `t0` 进行特定的调制。

### Compute Attention Score

```python
    # attention
    if self.temporal:
        x_m = rearrange(x_m, "B (T S) C -> (B S) T C", T=T, S=S)
        x_m = self.attn(x_m)
        x_m = rearrange(x_m, "(B S) T C -> B (T S) C", T=T, S=S)
    else:
        x_m = rearrange(x_m, "B (T S) C -> (B T) S C", T=T, S=S)
        x_m = self.attn(x_m)
        x_m = rearrange(x_m, "(B T) S C -> B (T S) C", T=T, S=S)
```

`self.temporal` 为 `True`，则表示需要进行时序上的注意力计算，将 `x_m` 变为 `(B S) T C` 形状，表示 batch 中每个空间 patch 上的时序序列进行注意力操作。否则，在空间上进行注意力计算。

### Residual Connection

```python
    # modulate (attention)
    x_m_s = gate_msa * x_m
    if x_mask is not None:
        x_m_s_zero = gate_msa_zero * x_m
        x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

    # residual
    x = x + self.drop_path(x_m_s)
```

&emsp;&emsp;使用 `gate_msa` 对注意力的输出进行门控（gate），并通过 drop_path 处理后与到原始输入 x 相加 (残差连接)。

### Cross Attention and Residual Connection

```python
    # cross attention
    x = x + self.cross_attn(x, y, mask)
```

&emsp;&emsp;x 和条件 y 的信息融合，把交叉注意力输出加到 x 上，完成残差连接。

### MLP and Residual Connection

```python
    # modulate (MLP)
    x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
    if x_mask is not None:
        x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
        x_m = self.t_mask_select(x_mask, x_m, x_m_zero, T, S)

    # MLP
    x_m = self.mlp(x_m)

    # modulate (MLP)
    x_m_s = gate_mlp * x_m
    if x_mask is not None:
        x_m_s_zero = gate_mlp_zero * x_m
        x_m_s = self.t_mask_select(x_mask, x_m_s, x_m_s_zero, T, S)

    # residual
    x = x + self.drop_path(x_m_s)

    return x
```

&emsp;&emsp;和自注意力层一样，通过 `shift_mlp` 和 `scale_mlp` 进行调制，再经过 MLP 层，再通过 `gate_mlp` 进行门控，再加到原始输入 x 上，完成残差连接。

# T2IFinalLayer

&emsp;&emsp;将输入进行 LayerNorm 后进行调制，然后通过线性层将 channel 调整为 `num_patch * out_channels`，其中 `num_patch=prod(patch_size)`, `out_channels` 若预测方差则为 `2*in_channels` 否则为 `in_channels`. 最后通过 `unpatchify` 变换成原来的形状。

广播操作的条件：两个张量在每个维度上要么相等，要么有一个维度的大小为 1.

```python
class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels, d_t=None, d_s=None):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels
        self.d_t = d_t
        self.d_s = d_s

    def t_mask_select(self, x_mask, x, masked_x, T, S):
        # x: [B, (T, S), C]
        # mased_x: [B, (T, S), C]
        # x_mask: [B, T]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        masked_x = rearrange(masked_x, "B (T S) C -> B T S C", T=T, S=S)
        x = torch.where(x_mask[:, :, None, None], x, masked_x)
        x = rearrange(x, "B T S C -> B (T S) C")
        return x

    def forward(self, x, t, x_mask=None, t0=None, T=None, S=None):
        if T is None:
            T = self.d_t
        if S is None:
            S = self.d_s
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        if x_mask is not None:
            shift_zero, scale_zero = (self.scale_shift_table[None] + t0[:, None]).chunk(2, dim=1)
            x_zero = t2i_modulate(self.norm_final(x), shift_zero, scale_zero)
            x = self.t_mask_select(x_mask, x, x_zero, T, S)
        x = self.linear(x)
        return x
```

# Forward

&emsp;&emsp;输入视频噪声的形状为 `(B,in_channels,T,H,W)`，是已经经过 VAE 的，`in_channels=4`， H 和 W 都为原来的 1/8. `timestep` 是当前的去噪时间步骤. y 是经过 tokenizer 编码后的 prompt.

- `get_dynamic_size()` 函数会获取划分 patch (默认 1x2x2) 后每个patch的大小 (不能整除会补 0).
- 对划分 patch 后的 H 和 W 创建 Position Embedding 矩阵，形状为 `(1, H*W, dim)`
- `self.fps_embedder` 是一个 `SizeEmbedder`，它是 `TimestepEmbedder` 的子类，它用于将标量信息编码到时间步维度上。它会先将标量形状变成 `(B, 1)`，即在第 0 维上 repeat B 次，然后进行时间嵌入编码。
- `t` 和 `fps` 经过嵌入后相加，经过 `t_block` (SiLU->Linear) 后形状为 `(B, 6*dim)`.
- 对编码后的 prompt 进行嵌入，具体过程如上。
- 对输入的视频噪声进行 `PatchEmbed3D`，具体过程如上。然后 reshape 成 `(B, T, S, C)`, 加上 Position Embedding 矩阵，reshape 回 `(B, N, C)`。
- 前面已经保证了 H 能被卡数整除，若采用了 sequence parallelism，则将 S 维度进行划分。
- 遍历 `self.spatial_blocks` 和 `self.temporal_blocks`，对输入的 patch 进行空间和时间卷积。
- 通过 `final_layer` 得到输出，并 reshape 成 `(B, C_out, T, H, W)` 并返回。

```python
    def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):
        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)

        # adjust for sequence parallelism
        # we need to ensure H * W is divisible by sequence parallel size
        # for simplicity, we can adjust the height to make it divisible
        if self.enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())  # Get the sequence parallel size
            if H % sp_size != 0:
                h_pad_size = sp_size - H % sp_size
            else:
                h_pad_size = 0

            if h_pad_size > 0:
                hx_pad_size = h_pad_size * self.patch_size[1]

                # pad x along the H dimension
                H += h_pad_size
                x = F.pad(x, (0, 0, 0, hx_pad_size))

        S = H * W
        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === blocks ===
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x = auto_grad_checkpoint(spatial_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
            x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)

        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x
```
