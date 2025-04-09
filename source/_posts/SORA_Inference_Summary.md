---
title: OpenSora Inference Sumamry
date: 2025/3/06 16:06:20
categories: Paper Reading
tags: blog
excerpt: OpenSora Inference Sumamry.
mathjax: true
katex: true
---
# Model Architecture

OpenSora 的整体结构如下，Embedding Layer 包括 5 个
- PatchEmbedding3D: 对输入隐藏层的噪声进行 3D 卷积 `Conv3d(4, 1152, kernel_size=(1, 2, 2), stride=(1, 2, 2))`
- TimestepEmbedder: 对当前的时间步进行 Sinusoidal Embedding 后经过两个线性层
- SizeEmbedder: scalar 版本的 TimestepEmbedder
- t_block: ` Linear(in_features=1152, out_features=6912, bias=True)` 生成 MHSA 和 MLP 所需要的 shift, scale & gate 参数.
- CaptionEmbedder: 对编码后的 prompt 经过两个线性层。

主干由 28 个 STDiT3 Block 组成。每个 STDiT3 block 由一个 spatial block + temporal block 组成。spatial/temporal block 由 MHSA + MHCA + FFN 组成。不同的是 spatial block 中的 MHSA 的序列大小为空间维度 (HW)，时间维度 T 被看作 batch_size 处理，temporal block 则是反过来。

T2IFinal Layer 由一个 LayerNorm + Linear 组成，将维度缩放回 `patch_size*vae_outchannels` 再 reshape 成 `(B, C, T, H, W)`
![Open-Sora](https://note.youdao.com/yws/api/personal/file/WEB9688e46ada2523df1ec522a7649be19a?method=download&shareKey=991eba9aad6eca9f41599d2ad4f75c34 "Open-Sora")

想要生成 `t` 秒分辨率为 `(H,W)` 的视频，采用 class-free guidance，则 `B=2`. 经过 VAE 后的 shape 为 `(B, 4, t*25.5*5//17, H/8, W/8)`，之后经过卷积时如果不能被 patchsize 整除需要先 padding 后再进行，最后对应 Transform block 的输入 shape 为 `(B, C, 7.5t, H/16, W/16)`. 以下是 DeepLink 测试配置下的具体值

| shape        | S    | T   | numel=ST |
| ------------ | ---- | --- | -------- |
| 204_640_360  | 920  | 60  | 55200    |
| 408_640_360  | 920  | 120 | 110400   |
| 51_1280_720  | 3600 | 15  | 54000    |
| 102_1280_720 | 3600 | 30  | 108000   |

# Overall Computation

一个标准的 Transformer block 的 GEMM 计算量包括 QKVLinear(6bshh), Q@K.T(2bssh), Score@V(2bssh), OLinear(2bshh) 以及 FFN 的两个 Linear(16bshh)，共计 (24bshh + 4bssh). OpenSora 的一个Transformer block 由 spatial+temporal_attention + 2cross_attention + 2FFN 组成，一共有 28 个这样的 block. Spatial Attention 中 b = BT, s = S, Temporal Attention 中 b = BS, S = T, Cross Attention 中 b = B, S_q = TS, S_k = S_v = TOKEN (最大 prompt 长度, 300) 代入公式可得 OpenSora

- Spatial Attention Comp = `8BTShh+4BTSSh`
- Temporal Attention Comp = `8BSThh+4BSTTh`
- Cross Attention Comp = `2B(TS*2+TOKEN*2)hh+4BST*TOKEN*h`, `TS*2` 代表 QO Linear, `TOKEN*2` 代表 KV Linear
- Feed Forward Network Comp = 16BSThh
  则一个 OpenSora Transformer Block 总计算量为 `56BTShh+4BTSh(S+T+2*TOKEN)+8B*TOKEN*hh` FLOPs. 令 N = TS 则可化简为 `(56BN+8B*TOKEN)*hh+4BNh(S+T+2*TOKEN)`. 对于生成 shape 为 204_640_360 的视频每个 block TFLOPs = `(56*2*55200+8*2*300)*1152^2+4*2*55200*1152*(920+60+300) = 8.86T`, 整个 Backbone GEMMs TFLOP=`8.86*28=256.94T`

用 torch-xla 进行一遍推理后 trace mhlo (后文叙述) 得到的结果如下，考虑到前面的 Embedding 层和后面的 T2IFinalLayer，手算结果基本准确。

| shape        | GEMM TFLOPS | Vector TFLOPS |
| ------------ | ----------- | ------------- |
| 204_640_360  | 260.978     | 0.795         |
| 408_640_360  | 523.479     | 1.608         |
| 51_1280_720  | 292.026     | 1.160         |
| 102_1280_720 | 584.284     | 2.324         |

# Overall Communication

## Megatron

Megatron 张量并行中每个 transformer block 的通信量为 Attention 计算和 FFN 的各一次 AllReduce, 一次 AllRedce 通信量为 `2BTSh(N-1)/N*2` Bytes，一个 block 有 `3*2` 次，总共便是 `24BTSh(N-1)/N Bytes`.

## Two Dimension

2D 张量并行将输入分别将 B 和 M 沿着 X 和 Y 维度切分 b_xNm_y，将第一个矩阵的行和列分别沿着 X 和 Y 维度切分 m_xh_y，进行乘法前，输入会沿着 Y 轴进行 All-Gather b_xNM，通信量为 `BTSM/N*(N_y-1)*2` Bytes，权重沿着 X 轴进行 All-Gather Mh_y，通信量为 `MH/N*(N_x-1)*2` Bytes，这样输出的 B 和 H 沿着 X 和 Y 维度切分 b_xNh_y. 第二个矩阵的行和列分别沿着 Y 和 X 维度切分 h_ym_x，进行乘法前，权重沿着 X 轴进行 All-Gather h_yM，通信量为 `MH/N*(N_x-1)*2` Bytes. 这样相乘后 Y 轴的每个设备上都存有部分和结果 b_xNM，再沿着 Y 轴进行 Reduce-Scatter b_xNm_y，通信量为 `BTSM/N*(N_y-1)*2` Bytes.

![2D Sharding](https://note.youdao.com/yws/api/personal/file/WEBecc8be2a08e69abe6a5f208e137d696d?method=download&shareKey=38a3525ee0dc08efaaabbb5f1fcf6a13 "2D Sharding")

一次这样切分计算通信量总计为 ``(BTSM/N*(N_y-1)*2 + MH/N*(N_x-1)*2)*2`` Bytes. 一个 Transformer block 有 `3*2` 次，其中Attention 和 CrossAttention的 QKVOLinear(M=1152, H=1152), FFNUp&Down(M=1152, H=4H),考虑 cfg B 只能为 2，因此 `N_x=2,N_y=8`.

{% mathjax %} \begin{aligned}
&\left(
  \left(
    \frac{BTSM}{N} \cdot (N_y - 1) \cdot 2 + \frac{MH}{N} \cdot (N_x - 1) \cdot 4
  \right) \cdot 2
\right. \\
&+ \left.
  \left(
    \frac{BTSM}{N} \cdot (N_y - 1) \cdot 2 + \frac{4MH}{N} \cdot (N_x - 1) \cdot 2
  \right)
\right) \\
&\cdot 2 \cdot 2 \, \text{Bytes} \\
& = \left(\frac{BTSM}{N} \cdot (N_y - 1) \cdot 6+\frac{MH}{N} \cdot (N_x - 1)\cdot 8 \right)\cdot 2 \cdot 2 \, \text{Bytes}
\end{aligned}
{% endmathjax %}

```python
((BTS*1152/16*7*4+1152*1152/16*4*2)*2+(BTS*1152/16*7+1152*4608/16)*4)*2*28 
=(BTS*1152/16*7*24+1152*1152/16*32)*28 
=(BTS*1152/2*7*3+1152*1152*2)*28
```


DS-Ulysses 每个 Transformer block 的通信量 QKVO 各一次 All2All：
Spatial/Temporal Attention: {% mathjax %}4(BTSh/N^2)\cdot 2Bytes\cdot N (\sqrt{N}-1)=4BTSh\cdot 2Bytes\cdot(\sqrt{N}-1)/N{% endmathjax %}
Cross Attention: {% mathjax %}B(2TS+2TOKEN)h\cdot 2Bytes\cdot (\sqrt{N}-1)/N{% endmathjax %}
加起来便是 {% mathjax %} B(6TS+2TOKEN)h\cdot 2Bytes\cdot(\sqrt N-1)/N{% endmathjax %},一个 block 有 2 次。

Ring-Attention 通过将 QKV 沿着序列维度切分到每个设备上，计算的同时异步通信 KV. 因此 Spatial/Temporal Attention 通信量为 `2*BTSh*(N-1)/N*2` Bytes, Cross Attention 的通信量为 `2*B*TOKEN*h*(N-1)/N*2`. 一个 Transformer block 各有两次，因此一个 block 总的通信量为 `2*B(TS+TOKEN)h*(N-1)/N*2*2` Bytes. spatial 和 temporal block 之间需要一次 All2All 将切分的序列维度在空间和时间维度上面转换。

理论通信量如下 

| shape/method         | Megatron(GB) | 2Dimension(GB) | DS-Ulysses(GB) | Ring-Attention(GB) |
|-----------------------|--------------|----------------|----------------|---------------------|
| 204_640_360          | 80.124    | 34.892       | 16.0538     | 25.001             |
| 408_640_360          | 160.248   | 69.716       | 32.078      | 50.002             |
| 51_1280_720          | 78.382     | 34.135       | 15.705     | 24.468             |
| 102_1280_720         | 156.764    | 68.202       | 31.382     | 48.936             |

# Computation Disassembly
## Megatron
<table border="1" style="border-collapse: collapse;">
  <tr style="background-color: #f2f2f2;">
    <th style="background-color: #f2f2f2;">Operation</th>
    <th>IShape</th>
    <th>WShape</th>
    <th>OShape</th>
    <th>Comp</th>
    <th>204.640_360(GFLOPs)</th>
    <th>Utilization(%)</th>
    <th>Latency(ms)</th>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">RMSNorm</td>
    <td>(B,S,T,h)</td>
    <td>(h, )</td>
    <td>(B,S,T,h)</td>
    <td>4BSh</td>
    <td>0.473</td>
    <td>19.260</td>
    <td>2.466</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">t2i_modulate</td>
    <td>(B,S,T,h)</td>
    <td>(2, h)</td>
    <td>(B,S,T,h)</td>
    <td>2BSh</td>
    <td>0.237</td>
    <td>9.635</td>
    <td>2.466</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">QKV_Linear</td>
    <td>(B,S,T,h)</td>
    <td>(h,3h/N)</td>
    <td>(B,S,T,3h/N)</td>
    <td>6BShh/N</td>
    <td>51.614</td>
    <td>85.148</td>
    <td>0.4694</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">RMSNorm(Q)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>(1,HA)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>4BSh/N</td>
    <td>0.0299</td>
    <td>18.480</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">RMSNorm(K)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>(1,HA)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>4BSh/N</td>
    <td>0.0290</td>
    <td>18.480</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">RoPE(Q)+scale</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>(1,S)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>(3+1)BSh/N</td>
    <td>0.0299</td>
    <td rowspan=5; >90.602</td>
    <td rowspan=5; >0.278</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">RoPE(K)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>(1,HA)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>4BSh/N</td>
    <td>0.0199</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">Q@K.T</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>(BT, HA/N, HA, S)</td>
    <td>(BT,NA/N,S,S)</td>
    <td>2BThSS/N</td>
    <td>13.621</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">Softmax+DropOut</td>
    <td>(BT,NA/N,S,S)</td>
    <td>None</td>
    <td>(BT,NA/N,S,S)</td>
    <td>6BT(NA)SS/N</td>
    <td>0.037</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">Score@V</td>
    <td>(BT,NA/N,S,S)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>2BThSS/N</td>
    <td>13.621</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">O_Linear</td>
    <td>(B, T, S, h/N)</td>
    <td>(h/N, h)</td>
    <td>(B,T,S,h)</td>
    <td>2BThh/N</td>
    <td>17.205</td>
    <td>21.8477</td>
    <td>0.610</td>
  </tr>
  <tr>
    <td style="background-color:rgb(100, 20, 100);color: white;">Gate_ResAdd</td>
    <td>(B,T,S,h)</td>
    <td>(h,)</td>
    <td>(B,T,S,h)</td>
    <td>2BSh</td>
    <td>0.237</td>
    <td>9.635</td>
    <td>2.4668</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27);color: white;">Q_Linear</td>
    <td>(B,S,T,h)</td>
    <td>(h,h/N)</td>
    <td>(B,T,S,h/N)</td>
    <td>2BThh/N</td>
    <td>17.056</td>
    <td>85.148</td>
    <td>0.156</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27);color: white;">scale</td>
    <td>(B,NA/N,S,T,HA)</td>
    <td>None</td>
    <td>(B,NA/N,S,T,HA)</td>
    <td>BTh/N</td>
    <td>0.007</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27);color: white;">KV_Linear</td>
    <td>(B,TOKEN,h)</td>
    <td>(h,2h/N)</td>
    <td>(B,TOKEN,2h/N)</td>
    <td>4B(TOKEN)h/N</td>
    <td>0.188</td>
    <td>8.993</td>
    <td>0.016</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27);color: white;">Q@K.T</td>
    <td>(B,NA/N,TS,HA)</td>
    <td>(B,NA/N,HA,TOKEN)</td>
    <td>(B,NA/N,TS,TOKEN)</td>
    <td>2BTh(TOKEN)/N</td>
    <td>8.883</td>
    <td rowspan="3">52.696</td>
    <td rowspan="3">0.286</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27);color: white;">Softmax+DropOut</td>
    <td>(B,NA/N,TS,TOKEN)</td>
    <td>None</td>
    <td>(B,NA/N,TS,TOKEN)</td>
    <td>6BT(NA)S(TOKEN)/N</td>
    <td>0.0116</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27);color: white;">Score@V</td>
    <td>(B,NA/N,TS,TOKEN)</td>
    <td>(B, NA/N, TOKEN, HA)</td>
    <td>(B,NA/N,S,T,HA)</td>
    <td>2BTh(TOKEN)/N</td>
    <td>8.883</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27);color: white;">O_Linear</td>
    <td>(B,T,S,h/N)</td>
    <td>(h/N,h)</td>
    <td>(B,T,S,h)</td>
    <td>2BThh/N</td>
    <td>17.056</td>
    <td>21.847</td>
    <td>0.610</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27);color: white;">Gate_ResAdd</td>
    <td>(B,T,S,h)</td>
    <td>(h,)</td>
    <td>(B,T,S,h)</td>
    <td>2BSh</td>
    <td>0.237</td>
    <td>9.635</td>
    <td>2.470</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35);color: white;">RMSNorm</td>
    <td>(B,S,T,h)</td>
    <td>(h,)</td>
    <td>(B,S,T,h)</td>
    <td>4BSh</td>
    <td>0.474</td>
    <td>19.260</td>
    <td>2.466</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35);color: white;">t2i_modulate</td>
    <td>(B,S,T,h)</td>
    <td>(2,h)</td>
    <td>(B,S,T,h)</td>
    <td>2BSh</td>
    <td>0.237</td>
    <td>9.635</td>
    <td>2.466</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">FFN_Linear1</td>
    <td>(B,TS,h)</td>
    <td>(h,4h/N)</td>
    <td>(B,T,S,4h/N)</td>
    <td>8BThh/N</td>
    <td>68.225</td>
    <td>98.641</td>
    <td>0.740</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">GeLU</td>
    <td>(B,TS,4h/N)</td>
    <td>None</td>
    <td>(B,TS,4h/N)</td>
    <td>2BTh/N</td>
    
    
    
  </tr>
  <tr>
    <td style="background-color:rgb(12, 88, 35); color: white;">FFN_Linear2</td>
    <td>(B,TS,4h/N)</td>
    <td>(4h/N,h)</td>
    <td>(B,T,S,h)</td>
    <td>8BThh/N</td>
    <td>68.225</td>
    <td>81.220</td>
    <td>0.656</td>
  </tr>
</table>

<style>
  table {
    width: 100%;
    border: 1px solid black;
  }
  th, td {
    border: 1px solid black;
    padding: 8px;
    text-align: center;
  }
  th {
    background-color: #f2f2f2;
  }
</style>

## Two Dimension
<table border="1" style="border-collapse: collapse;">
  <tr style="background-color: #f2f2f2;">
    <th style="background-color: #f2f2f2; color: white;">Operation</th>
    <th>IShape</th>
    <th>WShape</th>
    <th>OShape</th>
    <th>Comp</th>
    <th>204_640_360(GFLOPs)</th>
    <th>Utilization(%)</th>
    <th>Latency(ms)</th>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RMSNorm</td>
    <td>(B/Nx,ST,h)</td>
    <td>(h,)</td>
    <td>(B/Nx,ST,h)</td>
    <td>4BSt/hNx</td>
    <td>0.237</td>
    <td>19.148</td>
    <td>1.237</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">t2i_modulate</td>
    <td>(B/Nx,ST,h)</td>
    <td>(2,h)</td>
    <td>(B/Nx,ST,h)</td>
    <td>2BSt/hNx</td>
    <td>0.118</td>
    <td>9.581</td>
    <td>1.236</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">QKV_Linear</td>
    <td>(B/Nx,ST,h)</td>
    <td>(h,3h/Ny)</td>
    <td>(B/Nx,ST,3h/Ny)</td>
    <td>6BSt/h/N</td>
    <td>51.614</td>
    <td>90.435</td>
    <td>0.442</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RMSNorm(Q)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>(1,HA)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>4BSt/hN</td>
    <td>0.0299</td>
    <td>18.480</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RMSNorm(K)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>(1,HA)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>4BSt/hN</td>
    <td>0.0299</td>
    <td>18.480</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RoPE(Q)+scale</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>(1,S)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>(3+1)BSt/hN</td>
    <td>0.0299</td>
    <td rowspan=5; >94.684</td>
    <td rowspan=5; >0.251</td>

  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RoPE(K)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>(1,HA)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>4BSt/hN</td>
    <td>0.0199</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">Q@K.T</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>(BT/Nx,NA/Ny,HA,S)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>2BThS/N</td>
    <td>13.621</td>
    
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">Softmax+DropOut</td>
    <td>(BT/Nx,NA/Ny,S,S)</td>
    <td>None</td>
    <td>(BT/Nx,NA/Ny,S,S)</td>
    <td>6BT(NA)S/N</td>
    <td>0.0370</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">Score@V</td>
    <td>(BT/Nx,NA/Ny,S,S)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>(BT/Nx,NA/Ny,S,HA)</td>
    <td>2BThS/N</td>
    <td>13.621</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">O_Linear</td>
    <td>(B/Nx,TS,h/Ny)</td>
    <td>(h/Ny,h)</td>
    <td>(B/Nx,TS,h)</td>
    <td>2BThS/hN</td>
    <td>17.056</td>
    <td>41.322</td>
    <td>0.322</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">Gate_ResAdd</td>
    <td>(B/Nx,TS,h)</td>
    <td>(h,)</td>
    <td>(B/Nx,TS,h)</td>
    <td>2BSt/hNx</td>
    <td>0.118</td>
    <td>9.582</td>
    <td>1.236</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Q_Linear</td>
    <td>(B/Nx,ST,h)</td>
    <td>(h,/Ny)</td>
    <td>(B/Nx,TS,h/Ny)</td>
    <td>2BTS/h/N</td>
    <td>17.056</td>
    <td>90.435</td>
    <td>0.281</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">scale</td>
    <td>(B/Nx,NA/Ny,ST,HA)</td>
    <td>None</td>
    <td>(B/Nx,NA/Ny,ST,HA)</td>
    <td>BTS/hN</td>
    <td>0.007</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">KV_Linear</td>
    <td>(B/Nx,TOKEN,h)</td>
    <td>(h,2h/Ny)</td>
    <td>(B,TOKEN,2h/Ny)</td>
    <td>4B(TOKEN)h/N</td>
    <td>0.188</td>
    <td>0.0184</td>
    <td>15.758</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Q@K.T</td>
    <td>(B/Nx,NA/Ny,TS,HA)</td>
    <td>(B/Nx,NA/Ny,HA,TOK)</td>
    <td>(B/Nx,NA/Ny,TS,TOK)</td>
    <td>2BThS(TOKEN)/N</td>
    <td>8.883</td>
    <td>53.470</td>
    <td>0.273</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Softmax+DropOut</td>
    <td>(B/Nx,NA/Ny,TS,TOK)</td>
    <td>None</td>
    <td>(B/Nx,NA/Ny,TS,TOK)</td>
    <td>6BT(NA)S(TOKEN)/N</td>
    <td>0.012</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Score@V</td>
    <td>(B/Nx,NA/Ny,TS,TOK)</td>
    <td>(BT/Nx,NA/Ny,TOK)</td>
    <td>(B/Nx,NA/Ny,TS,HA)</td>
    <td>2BThS(TOKEN)/N</td>
    <td>8.883</td>
    <td>53.470</td>
    <td>0.273</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">O_Linear</td>
    <td>(B/Nx,TS,h/Ny)</td>
    <td>(h/Ny,h)</td>
    <td>(B/Nx,TS,h)</td>
    <td>2BTS/h/N</td>
    <td>17.056</td>
    <td>98.745</td>
    <td>2.159</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Gate_ResAdd</td>
    <td>(B/Nx,TS,h)</td>
    <td>(h,)</td>
    <td>(B/Nx,TS,h)</td>
    <td>2BTS/hNx</td>
    <td>0.118</td>
    <td>9.635</td>
    <td>2.470</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">RMSNorm</td>
    <td>(B/Nx,ST,h)</td>
    <td>(h,)</td>
    <td>(B/ST,h)</td>
    <td>4BSt/hNx</td>
    <td>0.237</td>
    <td>19.260</td>
    <td>2.461</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">t2i_modulate</td>
    <td>(B/Nx,ST,h)</td>
    <td>(2,h)</td>
    <td>(B/Nx,ST)</td>
    <td>2BSt/hNx</td>
    <td>0.118</td>
    <td>9.635</td>
    <td>2.470</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">FFN_Linear1</td>
    <td>(B/Nx,TS,h)</td>
    <td>(h,4h/Ny)</td>
    <td>(B/Nx,TS,4h/Ny)</td>
    <td>8BTS/hN</td>
    <td>68.225</td>
    <td>98.687</td>
    <td>0.740</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">GeLU</td>
    <td>(B/Nx,TS,4h/Ny)</td>
    <td>None</td>
    <td>(B/Nx,TS,4h/Ny)</td>
    <td>2BTS/4hNx</td>
    <td>68.225</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">FFN_Linear2</td>
    <td>(B/Nx,TS,4h/Ny)</td>
    <td>(4h/Ny,h)</td>
    <td>(B/Nx,TS,h)</td>
    <td>8BTS/hN</td>
    <td>68.225</td>
    <td>97.161</td>
    <td>0.533</td>
  </tr>
</table>

<style>
  table {
    width: 100%;
    border: 1px solid black;
  }
  th, td {
    border: 1px solid black;
    padding: 8px;
    text-align: center;
  }
  th {
    background-color: #f2f2f2;
  }
</style>

## DeepSpeed-Ulysses
<table border="1" style="border-collapse: collapse;">
  <tr style="background-color: #f2f2f2;">
    <th style="background-color: #f2f2f2; color: white;">Operation</th>
    <th>IShape</th>
    <th>WShape</th>
    <th>OShape</th>
    <th>Comp</th>
    <th>204_640_360(GFLOPs)</th>
    <th>Utilization(%)</th>
    <th>Latency(ms)</th>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RMSNorm</td>
    <td>(B,ST/N,h)</td>
    <td>(h,)</td>
    <td>(B,ST/N,h)</td>
    <td>4BSt/hN</td>
    <td>0.237</td>
    <td>18.480</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">t2i_modulate</td>
    <td>(B,ST/N,h)</td>
    <td>(2,h)</td>
    <td>(B,ST/N,h)</td>
    <td>2BSt/hN</td>
    <td>0.119</td>
    <td>9.2592</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">QKV_Linear</td>
    <td>(BT,S/N,h)</td>
    <td>(h,3h)</td>
    <td>(BT,S/N,3h)</td>
    <td>6BTS/hN</td>
    <td>51.614</td>
    <td>83.026</td>
    <td>0.486</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RMSNorm(Q)</td>
    <td>(BT,NA,S/N,HA)</td>
    <td>(1,HA)</td>
    <td>(BT,NA,S/N,HA)</td>
    <td>4BSt/hN</td>
    <td>0.0299</td>
    <td>18.480</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RMSNorm(K)</td>
    <td>(BT,NA,S/N,HA)</td>
    <td>(1,HA)</td>
    <td>(BT,NA,S/N,HA)</td>
    <td>4BSt/hN</td>
    <td>0.0299</td>
    <td>18.480</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RoPE(Q)+scale</td>
    <td>(BT,NA,S/N,HA)</td>
    <td>(1,S/N)</td>
    <td>(BT,NA,S/N,HA)</td>
    <td>(3+1)BSt/hN</td>
    <td>0.0299</td>
    <td rowspan=5; >90.602</td>
    <td rowspan=5; >0.278</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">RoPE(K)</td>
    <td>(BT,NA,S/N,HA)</td>
    <td>(1,S/N)</td>
    <td>(B,NA,ST/N,HA)</td>
    <td>3BSt/hN</td>
    <td>0.0199</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">Q@K.T</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>(BT,NA/N,HA,S)</td>
    <td>(BT,NA/N,S)</td>
    <td>2BThS/N</td>
    <td>13.621</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">Softmax+DropOut</td>
    <td>(BT,NA/N,S)</td>
    <td>None</td>
    <td>(BT,NA/N,S)</td>
    <td>6BT(NA)S/N</td>
    <td>0.0370</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">Score@V</td>
    <td>(BT,NA/N,S)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>(BT,NA/N,S,HA)</td>
    <td>2BThS/N</td>
    <td>13.621</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">O_Linear</td>
    <td>(B,TS/N,h)</td>
    <td>(h,)</td>
    <td>(B,TS/N,h)</td>
    <td>2BTS/hN</td>
    <td>17.2045884375</td>
    <td>83.025511</td>
    <td>0.161891</td>
  </tr>
  <tr>
    <td style="background-color: rgb(100, 20, 100); color: white;">Gate_ResAdd</td>
    <td>(B,TS/N,h)</td>
    <td>(h,)</td>
    <td>(B,TS/N,h)</td>
    <td>2BTS/hN</td>
    <td>0.119</td>
    <td>9.259260</td>
    <td>0.163147</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Q_Linear</td>
    <td>(B,TS/N,h)</td>
    <td>(h,)</td>
    <td>(B,TS/N,h)</td>
    <td>2BTS/hN</td>
    <td>17.056</td>
    <td>83.096255</td>
    <td>0.16035</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">scale</td>
    <td>(B,NA,ST/N,HA)</td>
    <td>None</td>
    <td>(B,NA,ST/N,HA)</td>
    <td>BTS/hN</td>
    <td>0.007</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">KV_Linear</td>
    <td>(B,TOKEN/N,h)</td>
    <td>(h,2h)</td>
    <td>(B,TOKEN/N,2h)</td>
    <td>4B(TOKEN)h/N</td>
    <td>0.188</td>
    <td>6.78544</td>
    <td>0.02163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Q@K.T</td>
    <td>(B,NA/N,TS,HA)</td>
    <td>(B,NA/N,HA,TOKEN)</td>
    <td>(B,NA/N,TS,TOKEN)</td>
    <td>2BThS(TOKEN)/N</td>
    <td>8.883</td>
    <td>52.696</td>
    <td>0.286</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Softmax+DropOut</td>
    <td>(B,NA/N,TS,TOKEN)</td>
    <td>None</td>
    <td>(B,NA/N,TS,TOKEN)</td>
    <td>6BT(NA)S(TOKEN)/N</td>
    <td>0.012</td>
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Score@V</td>
    <td>(B,NA/N,TS,TOKEN)</td>
    <td>(BT,NA/N,TOKEN,HA)</td>
    <td>(B,NA/N,TS,HA)</td>
    <td>2BThS(TOKEN)/N</td>
    <td>8.883</td>
    <td>52.670</td>
    <td>0.286</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">O_Linear</td>
    <td>(B,TS/N,h)</td>
    <td>(h,)</td>
    <td>(B,TS/N,h)</td>
    <td>2BTS/hN</td>
    <td>17.056</td>
    <td>83.096</td>
    <td>0.160</td>
  </tr>
  <tr>
    <td style="background-color: rgb(206, 134, 27); color: white;">Gate_ResAdd</td>
    <td>(B,TS/N,h)</td>
    <td>(h,)</td>
    <td>(B,TS/N,h)</td>
    <td>2BTS/hN</td>
    <td>0.118</td>
    <td>9.260</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">RMSNorm</td>
    <td>(B,TS/N,h)</td>
    <td>(h,)</td>
    <td>(B,TS/N,h)</td>
    <td>4BSt/hN</td>
    <td>0.237</td>
    <td>18.480</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">t2i_modulate</td>
    <td>(B,TS/N,h)</td>
    <td>(2,h)</td>
    <td>(B,TS/N,h)</td>
    <td>2BSt/hN</td>
    <td>0.119</td>
    <td>9.260</td>
    <td>0.163</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">FFN_Linear1</td>
    <td>(B,TS/N,h)</td>
    <td>(h,4h)</td>
    <td>(B,TS/N,4h)</td>
    <td>8BTS/hN</td>
    <td>68.225</td>
    <td>98.641</td>
    <td>0.750</td>
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">GeLU</td>
    <td>(B,TS/N,4h)</td>
    <td>None</td>
    <td>(B,TS/N,4h)</td>
    <td>2BTS/4hN</td>
    
    
    
  </tr>
  <tr>
    <td style="background-color: rgb(12, 88, 35); color: white;">FFN_Linear2</td>
    <td>(B,TS/N,4h)</td>
    <td>(4h,)</td>
    <td>(B,TS/N,h)</td>
    <td>8BTS/hN</td>
    <td>68.225</td>
    <td>88.560</td>
    <td>0.602</td>
  </tr>
</table>

<style>
  table {
    width: 100%;
    border: 1px solid black;
  }
  th, td {
    border: 1px solid black;
    padding: 8px;
    text-align: center;
  }
  th {
    background-color: #f2f2f2;
  }
</style>

分层的各个计算用时占比和利用率的示意图如下所示，可以看出 Vector 用时占主要部分，由于 Megatron 无法切分导致用时最长。

![Layer Comparison](https://note.youdao.com/yws/api/personal/file/WEB20acef6202af4bcb5ad4e616542e8ca4?method=download&shareKey=5650a19fe84e2daeea136089e90d2660 "Layer Comparison")

## Ring Attention

# MLIR Visitor


IRVisitor 类是一个基于 JAX 的 MLIR 结构遍历器，设计用于递归地遍历 MLIR IR 的各种节点（如操作、区域、块等）。它是一个基类，提供了通用的访问逻辑，允许子类通过重写特定方法来处理不同的节点类型。以下是对其工作原理的详细解释，重点说明它如何遍历整个 MHLO. 核心方法 visit 是遍历的入口方法，接收一个节点（node）作为参数。根据 node 的类型，动态构造访问器方法名：
* 如果 node 是 ir.Operation（一个具体的 MLIR 操作），方法名设为 visit_operation。
* 如果 node 是 ir.OpView（操作的视图，通常是特定操作的封装），解析其名称，尝试匹配特定操作类型（如 visit_add），如果没有匹配则回退到 visit_opview。
* 如果 node 是 ir.Region（表示操作中的一个代码区域），方法名设为 visit_region。
* 如果 node 是 ir.Block（区域中的一个基本块），方法名设为 visit_block。
* 使用 getattr 获取对应的方法，如果没有定义特定方法，则调用 generic_visit 作为默认行为. `generic_visit(self, node)` 是当没有特定访问器方法时调用的通用方法。

## TFlops Visitor
TFLopsIRVisitor 重写了基类 IRVisitor 中的部分方法，针对特定 MHLO 操作计算 TFLOPS：
- `visit_dot`: 处理 dot 操作。
获取输入张量的形状：
  - lhs_shape：左操作数（node.lhs）的形状，使用 ir.RankedTensorType 解析。
  - result_shape：输出张量（node.result）的形状。
  - 确定收缩维度（contract_dim）：假设 dot 操作的收缩维度是 lhs_shape 的最后一个维度（-1），即两个张量相乘时对齐的维度。
计算 TFLOPS：`2 * contract_dim * np.prod(np.array(result_shape))`. 将元组 `("dot", dot_tflops, node.location, node)` 添加到 self.op_tflops. 递归调用 self.generic_visit(node) 继续遍历该节点的子结构（如果有）。

- `visit_dot_general`：处理 dot_general 操作。
获取输入和输出张量的形状：
  - lhs_shape：左操作数（node.lhs）的形状。
  - result_shape：输出张量（node.result）的形状。
  - 获取收缩维度信息：从 `node.attributes` 中提取 `dot_dimension_numbers` 属性，解析为 `mhlo.DotDimensionNumbers` 对象。
  - contract_dims：左操作数的收缩维度索引。
  - contract_size：通过 np.prod(np.array(lhs_shape)[contract_dims]) 计算收缩维度的总元素数。
计算 TFLOPS：`2 * contract_size * np.prod(np.array(result_shape))`. 递归调用 self.generic_visit(node) 继续遍历子结构。

为了确定一个 Transformer block 中的 dot 和 dot_general 操作分别对应于什么，我们需要一个 `RMSCollector`，当遇到 `rsqrt` 操作时，`visit_rsqrt` 方法通过 `_parse_loc_lineno` 提取行号，并添加到 `RMSCollector.rms_locs` 用于临时存储 RMS Norm 的行号。根据 rms_locs 分割为 spt_qkv_ranges、spt_attn_ranges、ffn_ranges. 遍历 IR，遇到 dot 或 dot_general 时，根据行号匹配到特定块，更新计数器。当计数器满足条件，记录块并重置状态。输出 attention_blocks 和 ffn_blocks 包含所有匹配的块。

```python
'''
== ATTN ==
--- RMSNorm ---
dot: QKV_Linear
--- RMSNormQ ---
--- RMSNormK ---
dot_general: Q@K.T
dot_general: attn@V
dot: O_Linearj
== CROSS ==
dot: Q_Linear
dot: KV_Linear
dot_general: Q@K.T
dot_general: attn@V
dot: O_Linear
== MLP ==
--- RMSNorm ---
dot: MLP_Linear1
dot: MLP_Linear2
'''
```

Vector 计算量则是通过统计 add, subtract, multiply, divide, rsqrt, negate & exponential，操作对应的元素个数即为计算量。

## Comm Visitor
`CommIRVisitor` 继承 IRVisitor 基类，专门用于遍历 MHLO 中的通信操作，计算每个通信操作（communication op）的通信量（comm volume）和通信延迟（comm latency）。支持多种集体通信操作（如 all_reduce、all_gather、reduce_scatter 和 all_to_all）`CommIRVisitor` 为每种通信操作实现了特定的 visit 方法，计算通信量和延迟。`_get_ring_comm_stastics` 用于为环形通信模式计算集体通信操作的通信量和延迟。通过节点的 `replica_groups` 属性获取本次通信的设备数 `num_device`，环形通信中每个设备与除自身外的其他设备通信的平均比例为 `(num_devices - 1) / num_devices`. 前三种通信操作都可以使用 Ring 模式，需要注意由于 AllReduce 实际上是 ReduceScatter + AllGather，因此计算通信量的时候要乘以 2. 环状通信量计算公式为 {% mathjax %} \mathrm{commvolume}=\mathrm{np.prod}(\text{tensor shape})\times2\times[\text{multiplier}]\times\frac{\text{num devices}-1}{\text{num devices}} {% endmathjax %}

  {% fold info @Sinusoidal Embedding %}

1. 计算频率(freqs)
   函数首先计算一组频率，用于生成正弦和余弦波。频率的计算基于指数衰减：
   定义half=dim/2,即嵌入维度的一半。生成一个从 0 到 half-1 的序列：`torch.arange(start=0,end-=half)`
   频率公式为：
   {% mathjax %}
   freqs[i]=\exp\left(-\frac{\log(\text{maxperiod})}{half}\cdot i\right),\quad i=0,1,\ldots,\text{half}-1
   {% endmathjax %}
   其中：

- max_period 是最大周期（默认10000）。
- i是频率的索引。
- 频率从高到低变化，随着 i 增加，freqs 随指数哀减。

2. 计算角度(args)
   将输入时间步t与频率frqs相乘，得到角度：

- t是一个形状为 `(N,)` 的张量，通过 `t[:,None]` 扩展为 `(N,1)`.
- freqs 是一个形状为 `(half,)` 的张量，通过广播与 t 相乘。
  角度公式为：{% mathjax %} args[n,i]=t[n]\cdot freqs[i],\quad n=0,1,\ldots,N-1,\quad i=0,1,\ldots,\text{half}-1 {% endmathjax %}
  结果是一个形状为 `(N,half)` 的张量。

3. 生成正弦和余弦嵌入
   对角度 args 分别应用余弦和正弦函数，然后拼接：
   {% mathjax %} mbedding[n,:D]=[\cos(args[n,0]),\cos(args[n,1]),\ldots,\cos(args[n,\text{half}-1]),\sin(args[n,0]),\sin(args[n,1]),\ldots,\sin(args[n,\text{half}-1])] {% endmathjax %}
   前 half 个维度是余弦值，后 half 个维度是正弦值。

{% endfold %}


# Result

与单个 A100 以及采用 DS-Ulysses 并行策略的单机 8 卡 A100 的推理结果做对比。由上述计算量拆解可以看出 Vector 操作占主导部分，由于 Megatron 无法切分序列维度导致了重复计算，而 TwoDimension 可以切分 batch_size 维度，DS_Ulysses 则是可以完全切分序列维度。RingAttention 由于本身 Attention 计算的序列长度不是很长，再在计算时进行进一步切分导致了 FlashAttention 的利用率极低，导致用时很长。



| 分辨率       | Backbone 理论计算量 | device          | 设备数量 | 并行方法     | latency(s) STDT3 | alloc_mem(GB) | comm_volume(MB) | 利用率 |
|--------------|---------------------|-----------------|----------|--------------|------------------|---------------|-----------------|--------|
| 204_640_360  | 260.978/0.795       | A100            | 1        | xxx        | 99.00/115.43     |               |                 | 0.253  |
| 408_640_360  | 523.479/1.608       | A100            | 1        | xxx        | 202.3/250.66     |               |                 | 0.248  |
| 51_1280_720  | 292.026/1.160       | A100            | 1        | xxx        | 104.24/104.13    |               |                 | 0.269  |
| 102_1280_720 | 584.284/2.324       | A100            | 1        | xxx        | 206.92/206.66    |               |                 | 0.271  |
| 204_640_360  | 260.978/0.795       | A100            | 8        | DS_Ulysses   | 13.76/28.82      |               |                 | 0.124  |
| 408_640_360  | 523.479/1.608       | A100            | 8        | DS_Ulysses   | 26.61/51.62      |               |                 | 0.178  |
| 51_1280_720  | 292.026/1.160       | A100            | 8        | DS_Ulysses   | 25.26/34.56      |               |                 | 0.139  |
| 102_1280_720 | 584.284/2.324       | A100            | 8        | DS_Ulysses   | 36.5/53.19       |               |                 | 0.192  |
| 204_640_360  | 326,227/1.667       | TX8 | 16x(4x4) | Megatron     | 73.38            | 7.021         | 143956.143      | 0.088  |
| 408_640_360  | 654,693/3.26        | TX8 | 16x(4x4) | Megatron     | 145.71           | 13.807        | 267921.362      | 0.089  |
| 51_1280_720  | 375,531/2.218       | TX8 | 16x(4x4) | Megatron     | 72.18            | 24.126        | 140835.66       | 0.085  |
| 102_1280_720 | 751,315/4.442       | TX8 | 16x(4x4) | Megatron     | 142.95           | 48.016        | 281662.397      | 0.086  |
| 204_640_360  | 495,591/1.5         | TX8 | 16x(4x4) | Two_Dimension | 50.43            | 3.749         | 107759.849      | 0.129  |
| 408_640_360  | 992,661/3.021       | TX8 | 16x(4x4) | Two_Dimension | 100.59           | 7.26          | 215253.543      | 0.13   |
| 51_1280_720  | 530,634/1.945       | TX8 | 16x(4x4) | Two_Dimension | 49.77            | 12.19         | 105423.029      | 0.138  |
| 102_1280_720 | 1061,151/3.9        | TX8 | 16x(4x4) | Two_Dimension | 99.03            | 24.141        | 210579.804      | 0.139  |
| 204_640_360  | 584,310/0.431       | TX8 | 16x(4x4) | DS_Ulysses   | 36.93            | 3.749         | 103923.135      | 0.1    |
| 408_640_360  | 1130,119/0.864      | TX8 | 16x(4x4) | DS_Ulysses   | 73.65            | 7.26          | 103152.307      | 0.1    |
| 51_1280_720  | 588,764/0.446       | TX8 | 16x(4x4) | DS_Ulysses   | 36.84            | 12.19         | 116269.846      | 0.11   |
| 102_1280_720 | 1177,769/0.892      | TX8 | 16x(4x4) | DS_Ulysses   | 72.42            | 24.141        | 202365.35       | 0.11   |
| 204_640_360  | 584,310/0.431       | TX8 | 16x(4x4) | Ring_Attention   | 43.70            |          |       |     |
| 408_640_360  | 1130,119/0.864      | TX8 | 16x(4x4) | Ring_Attention   | 86.37            |          |       |     |
| 51_1280_720  | 588,764/0.446       | TX8 | 16x(4x4) | Ring_Attention   | 43.69           |          |       |    |
| 102_1280_720 | 1177,769/0.892      | TX8 | 16x(4x4) | Ring_Attention   | 84.38            |        |        |   |