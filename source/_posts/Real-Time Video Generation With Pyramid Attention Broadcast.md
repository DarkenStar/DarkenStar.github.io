---
title: Real-Time Video Generation With Pyramid Attention Broadcast
date: 2024/11/13 16:05:23
categories: Paper Reading
tags: Distributed Training
excerpt: Paper reading of Pyramid Attention Broadcast.
mathjax: true
katex: true
---
# Abstract

我们观察到 Attention 输出差异在扩散过程中呈现 U 型，表明显著的冗余。因此提出了金字塔注意力广播 (Pyramid Attention Broadcast, PAB)，在三种模型中表现出卓越的效果，实现了高达 720p 视频的实时生成。

![Results and Speed Comparison of PAB and 1 GPU](https://note.youdao.com/yws/api/personal/file/WEBf950ef91053e850a0adb2514c4853940?method=download&shareKey=73ede235df7f10620306857f711995bd "Results and Speed Comparison of PAB and 1 GPU")

# Introduction

基于 DiT 的视频生成在改进的质量的同时也增加了更多的内存占用、计算量和推理时间。模型压缩方法通常需要巨大的计算资源和数据集进行额外的训练。一部分工作通过缓存和重用部分网络输出来减轻冗余，从而消除了额外的训练。然而这种基于卷积网络的方法因为模型架构和计算模式的不同不能直接适用于视频 DiT.

我们仔细研究了视频 DiT 中的注意力，并得出了下图所示的两个观察结果

1. 相邻扩散步骤之间的注意力差异呈现 U 形，在中间 70% 的步骤中保持稳定，表明注意力有相当大的冗余性。
2. 在稳定的中间区间内，不同的 Attention 模块也表现出不同程度的差异。空间，时间，交叉注意力的差异依次减小。

![Comparison of the Attention Outputs MSE between Neighbor Diffusion Steps](https://note.youdao.com/yws/api/personal/file/WEB0bb054b37ba0cec928b8260ed128675d?method=download&shareKey=9d91f907f275bb2052b40f1ff1c1f589 "Comparison of the Attention Outputs MSE between Neighbor Diffusion Steps")

根据上面的发现，我们以金字塔的方式对不同的注意力使用不同的广播范围。我们发现这种广播策略也可以很好地适用于 MLP 层和连续的扩散时间步之间。此外，为了实现高效的分布式推理，我们提出了广播序列并行，这大大减少了生成时间和通信成本。

# How to Achieve Real-Time Video Generation

下图给出了基于 DiT 的视频扩散模型的基本架构。主干由空间和时间 Transformer block 组成。交叉注意力使模型能够在每个步骤中合入来自条件输入的信息。

![Overview of Backbone of Current DiT-based Video Generation Models](https://note.youdao.com/yws/api/personal/file/WEB3e8f95a99fa87b6930a88097ae1764ee?method=download&shareKey=5e2d7773ea97a7d73c5cb28af41b9247 "Overview of Backbone of Current DiT-based Video Generation Models")

如下图 (b) 所示，视频 DiT 中注意力占总推理时间的比例明显大于 CNN 方法。图 (a) 描绘了不同阶段注意力输出的可视化差异。对于中间部分差异很小，模式相似。图 (c) 展示了所有扩散步骤中注意力输出的量化差异。注意力输出的差异在中间部分的扩散步骤中表现出大约 70% 的相似。空间注意力输出有着最大的差异，其次是时间注意力，然后是交叉注意力。

![Visualization of Attention Differences in Open-Sora](https://note.youdao.com/yws/api/personal/file/WEBdf6bd70172b57b8bbfb32fe9c5be4f92?method=download&shareKey=0d40acd0a959af6a2310cd7d5b1cb23b "Visualization of Attention Differences in Open-Sora")

基于上述发现，我们提出的 PAB 方法与之前重复使用 attention score 的方法不同，我们选择将**注意力输出**从一些扩散步骤广播到中间部分。这样可以在后续步骤中完全绕过冗余的注意力计算，从而显著降低计算成本。公式可以表示如下

{% mathjax %}
O_{\mathrm{attn.}}=\{F(X_t),\underbrace{Y_t^*,\cdots,Y_t^*}_{\text{broadcast range}},F(X_{t-n}),\underbrace{Y_{t-n}^*,\cdots,Y_{t-n}^*,\cdots}_{\text{broadcast range}}\}.
{% endmathjax %}

`<br>`{% mathjax %}O_{\mathrm{attn.}}{% endmathjax %} 表示注意模块在所有时间步长的输出，{% mathjax %}F(X_{t-n}){% endmathjax %} 表示时间步 t 进行的注意力计算，{% mathjax %}Y_t^*{% endmathjax %}表示注意结果从时间步 t 开始广播。

为了在保持质量的同时提高效率，为每种注意力类型定制广播范围，如下图所示。广播范围的确定基于两个关键因素：每种注意类型的变化率和稳定性。

![Overview of Pyramid Attention Broadcast](https://note.youdao.com/yws/api/personal/file/WEB118af684d50e0e2c4261a8c74c1a57f6?method=download&shareKey=e0cf3e9df05a852d867d81955fe8d34c "Overview of Pyramid Attention Broadcast")

我们在动态序列并行 (DSP) 的基础上引入广播序列并行来提高视频生成速度。如下图所示，通过广播时间注意力，可以在不损失质量的情况下消除序列并行方法在模型内的通信开销。

![Comparison between Original Sequence Parallelism and Ours](https://note.youdao.com/yws/api/personal/file/WEBfb182db5e8ce47dc003bb9138e4b72d6?method=download&shareKey=dfab593e0fcf534c8a9f7501f7df2af0 "Comparison between Original Sequence Parallelism and Ours")

# Experiments

## Experiment Setup

- Models: 选择 Open-Sora, Open-Sora- plan & Latte 作为实验模型。
- Metrics: VBench, PSNR, LPIPS, SSIM.
- Baselines: Δ-DiT, T-GATE，它们都是基于缓存的方法。
- Implementation details: 采用 PyTorch 框架，8x NVIDIA H100 80GB GPUs，默认使用 FlashAttention.

## Qualitative Results

下表展示了我们的方法和在三个模型上与两个 baseline 之间的四个指标比较。结论如下

1. 我们的方法与两个 baseline 质量相当，同时在单个 GPU 上实现了高达 1.58 倍的加速。
2. 我们的方法在所有三个模型中都表现良好，它们使用了不同的训练策略和去噪 scheduler，证明了它的泛化性。

<br>
<table>
  <tr>
    <th>model</th>
    <th>method</th>
    <th>VBench (%) ↑</th>
    <th>PSNR ↑</th>
    <th>LPIPS ↓</th>
    <th>SSIM ↑</th>
    <th>FLOPs (T)</th>
    <th>latency (s)</th>
    <th>speedup</th>
  </tr>
  <tr>
    <td rowspan="6">Open-Sora</td>
    <td>original</td>
    <td>79.22</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>3230.24</td>
    <td>26.54</td>
    <td>-</td>
  </tr>
  <tr>
    <td>∆-DiT</td>
    <td>78.21</td>
    <td>11.91</td>
    <td>0.5692</td>
    <td>0.4811</td>
    <td>3166.47</td>
    <td>25.87</td>
    <td>1.03×</td>
  </tr>
  <tr>
    <td>T-GATE</td>
    <td>77.61</td>
    <td>15.50</td>
    <td>0.3495</td>
    <td>0.6760</td>
    <td>2818.40</td>
    <td>22.22</td>
    <td>1.19×</td>
  </tr>
  <tr style="background-color: #d3d3d3;">
    <td><b>PAB<sub>246</sub></b></td>
    <td><b>78.51</b></td>
    <td><b>27.04</b></td>
    <td><b>0.0925</b></td>
    <td><b>0.8847</b></td>
    <td><b>2657.70</b></td>
    <td><b>19.87</b></td>
    <td><b>1.34×</b></td>
  </tr>
  <tr>
    <td>PAB<sub>357</sub></td>
    <td>77.64</td>
    <td>24.50</td>
    <td>0.1471</td>
    <td>0.8405</td>
    <td>2615.15</td>
    <td>19.35</td>
    <td>1.37×</td>
  </tr>
  <tr style="background-color: #d3d3d3;">
    <td>PAB<sub>579</sub></td>
    <td>76.95</td>
    <td>23.58</td>
    <td>0.1743</td>
    <td>0.8220</td>
    <td>2558.25</td>
    <td><b>18.52</b></td>
    <td>1.43×</td>
  </tr>

下图展示了在利用具有广播序列并行性的多个 GPU 进行推理时，我们的方法在三种不同的模型中随着 GPU 数量的增加，加速几乎是线性的。当使用 8 个 GPU 时，实现了 10.60 倍加速。

![Speedups](https://note.youdao.com/yws/api/personal/file/WEB29579ca0a8a65785909de5d69d3e78a7?method=download&shareKey=75fd5b821906f255b6475d73bd148ae1 "Speedups")

## Ablation Study

消融实验的设置为 PAB246，Open-Sora 模型，使用单个 NVIDIA H100 GPU 生成 2s 480p 视频。

下表展示了不同组件的影响，我们单独禁用每个组件的广播策略，并测量 VBench 分数和延迟的增加。结果显示虽然对 VBench 分数的影响可以忽略不计，但每个组件都有助于整体加速。

| broadcast strategy | latency (s) | ∆    | VBench (%) ↑ |
| ------------------ | ----------- | ----- | ------------- |
| w/o spatial attn.  | 21.74       | +1.87 | 78.45         |
| w/o temporal attn. | 23.95       | +4.08 | 78.98         |
| w/o cross attn.    | 20.98       | +1.11 | 78.58         |
| w/o mlp            | 20.27       | +0.40 | 78.59         |
| all components     | 19.87       | –    | 78.51         |

下图展示了广播范围的影响，结果表示广播范围和视频质量之间明显的反比关系。此外，可以观察到不同的广播范围对不同注意力的影响是不同的。

![Evaluation of Attention Broadcast Ranges](https://note.youdao.com/yws/api/personal/file/WEB31674d82742e94dc74297a93caa11674?method=download&shareKey=6d5abc5385260295c615a20e2e60f8e5 "Evaluation of Attention Broadcast Ranges")

以前的研究通常重复利用 attention score，但我们发现广播注意力输出效果更好。下表比较了广播 attention score 与注意力输出所获得的加速和视频质量。结果表明，广播注意力输出保持了相似的质量，同时提供了更好的效率，主要有两个原因：

1. 注意力输出变化小，因为尽管像素发生了变化，但注意力计算后的结果仍然相似。这进一步表明了注意力计算中的显著冗余。
2. 广播 attention score 降低了使用优化的内核带来的效率提升。

<table>
  <tr>
    <th>broadcast object</th>
    <th>VBench (%)</th>
    <th>latency (s)</th>
  </tr>
  <tr>
    <td>original</td>
    <td>79.22</td>
    <td>26.54</td>
  </tr>
  <tr>
    <td><b>attention scores</b></td>
    <td><b>78.53</b></td>
    <td>29.12</td>
  </tr>
  <tr style="background-color: #d3d3d3;">
    <td><b>attention outputs</b></td>
    <td><b>78.51</b></td>
    <td><b>19.87</b></td>
  </tr>
</table>

## Scaling Ability

为了评估方法的可扩展性，在每个实验中，我们都将 PAB246 作为 Open-Sora 的基准配置，仅更改视频大小，并行方法和 GPU 数量。

我们使用 8x NVIDIA H100 GPUs 生成 2s 480p 视频，对 3 种序列并行方法比较了使用和不使用我们的方法的扩展效率。下表结果表明：

1. PAB 显着减少了所有序列并行方法的通信量。此外，与其他技术相比，我们的方法实现了最低的通信成本，并在 8 个 GPU 上实现了近线性扩展。
2. 单独实现序列并行不足以获得最佳性能，因为跨多个设备的通信开销很大。

<table>
  <tr>
    <th rowspan="2">method</th>
    <th colspan="2">w/o PAB</th>
    <th colspan="2">w/ PAB</th>
  </tr>
  <tr>
    <th>comm. (G)</th>
    <th>latency (s)</th>
    <th>comm. (G)</th>
    <th>latency (s)</th>
  </tr>
  <tr>
    <td>original</td>
    <td>–</td>
    <td>97.51</td>
    <td>–</td>
    <td>71.25</td>
  </tr>
  <tr>
    <td>Megatron-SP</td>
    <td>184.63</td>
    <td>17.17</td>
    <td>104.62</td>
    <td>14.78</td>
  </tr>
  <tr>
    <td>DS-Ulysses</td>
    <td>46.16</td>
    <td>12.34</td>
    <td>26.16</td>
    <td>9.85</td>
  </tr>
  <tr>
    <td>DSP</td>
    <td>23.08</td>
    <td>12.01</td>
    <td>–</td>
    <td>–</td>
  </tr>
  <tr style="background-color: #d3d3d3;">
    <td><b>ours</b></td>
    <td><b>–</b></td>
    <td><b>–</b></td>
    <td><b>13.08</b></td>
    <td><b>9.29</b></td>
  </tr>
</table>

为了评估我们的方法在处理更大视频尺寸的能力，我们在各种视频长度和分辨率上进行了测试，如下图所示。结果表明，随着视频大小的增加，我们可以在单个 GPU 上提供稳定的加速，并在扩展到多个 GPU 时提供更好的扩展能力。

![Scaling Viideo Size](https://note.youdao.com/yws/api/personal/file/WEB5fca4576217161dd60ae27fab51f6c67?method=download&shareKey=aeda29896dd87d12ad066432bc68534e "Scaling Viideo Size")

我们通过在 8 个和 16 个设备上推理的 FPS 来评估 PAB 的速度。首先拆分批处理，并对每个批处理应用序列并行；通过这种方式，PAB 可以几乎线性扩展到 16 个设备。如下图所示，我们可以在  8台设备上实现 480p 视频的实时高 FPS 视频生成，甚至在 16 台设备上实现 720p 视频生成.

![Real-Time Video Generation Performance](https://note.youdao.com/yws/api/personal/file/WEBe127b73dd18c1ebbbcb1bdc75a1c4353?method=download&shareKey=46e5fff12e7ddf764b421c03a2f73dd3 "Real-Time Video Generation Performance")

下图展示了各种组件的时间消耗的分解。结果表面注意力计算本身并不会消耗大量的时间，因为对每个维度分别进行注意计算，序列的长度会短得多。然而，与注意力相关的操作，如归一化和线性层，比注意力机制本身要费时得多。

![Runtime Breakdown for Generating a 2s 480p Video](https://note.youdao.com/yws/api/personal/file/WEB364bd203f6dfea7fef873d0956e6b0ca?method=download&shareKey=b8f48d04f34ff8ded09c0be1cb244b38 "Runtime Breakdown for Generating a 2s 480p Video")

# Discussion and Conclusion

PAB 利用注意力差异呈现 U 型，通过金字塔式传播减少冗余。此外，广播序列并行化方法显著提高了分布式推理效率。然而 PAB 的性能可能会因输入数据的复杂性而有所不同，特别是在动态场景下。固定广播策略可能不适用于所有视频类型和任务。
