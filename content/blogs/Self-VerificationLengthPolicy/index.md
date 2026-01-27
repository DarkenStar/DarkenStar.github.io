---
title: "Self Verification Length Policy"
date: 2025-11-12T16:56:09+08:00
lastmod: 2025-11-12T16:56:09+08:00
author: ["WITHER"]

categories:
- Paper Reading

tags:
- Speculative Decoding

keywords:
- Speculative Decoding

description: "Paper Reading of SVIP." # 文章描述，与搜索优化相关
summary: "Paper Reading of SVIP." # 文章简单描述，会展示在主页
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

# 1 Introduction

![Overview of SVIP](https://share.note.youdao.com/yws/api/personal/file/WEBd463d92bd6e41b0547ab46587d0a3509?method=download&shareKey=35e54ee5c546fbcecd83a915a1a88a3d "Overview of SVIP")

文中敏锐地指出，现有的推测解码方法大多采用一种固定长度策略 (fixed-length policy)  ，即无论上下文如何，草稿模型每次都尝试生成一个固定数量 (例如5个) 的词元. 论文的核心论点是: 这种固定长度的策略是次优的，甚至可能是低效的。  

作者首先引入了 神谕草稿长度 (*Oracle Draft Length*) 的概念，即在给定上下文中，目标模型实际会接受的草稿词元的最大数量 。论文通过实验 (如图1所示) 有力地证明，这个“神谕长度”绝非一成不变，而是剧烈波动的。特别是在长上下文 和复杂推理任务 (如AIME基准测试) 中 ，可接受的长度方差极大 (例如，在AIME上，可接受长度从6到17不等) .

一个固定的长度 (比如5) 在这种情况下显然存在问题: 
- 当模型非常有把握、神谕长度为10时，它只生成了5个，浪费了潜在的加速空间。
- 当模型非常不确定、神谕长度只有2时，它却生成了5个，后3个词元几乎注定被拒绝，这浪费了草稿模型的计算资源，并可能拖慢整个流程。

既然神谕长度是动态变化的，我们该如何预测它呢？作者提出了一个极其巧妙的洞察: 我们不需要目标模型来告诉我们何时停止，草稿模型自己“知道”它何时开始变得不确定 。

这个“不确定性”的量化指标就是信息论中的经典概念——预测熵 (Prediction Entropy).

![Agreement scores and sorted vocabulary log  probability at the rejection phenomenon ](https://share.note.youdao.com/yws/api/personal/file/WEBe2bbd093ab110af2d7d74a1ad5d67a84?method=download&shareKey=d487dc2119587f91d7dc11ce814e5f29 "Agreement scores and sorted vocabulary log  probability at the rejection phenomenon ")

作者通过实验 (图3a和表1) 证明了这一点: 
- 当草稿模型生成的词元即将被目标模型拒绝时，草稿模型在该位置的KL散度 (衡量与目标模型分布的差异) 会突然飙升 。
- 更重要的是，在被拒绝的词元位置，草稿模型自身的预测熵也显著更高 。例如，在AIME基准上，被接受词元的平均熵仅为0.25，而被拒绝词元的平均熵高达1.30.

# 2 Draft Model Knows When to Stop

词元拒绝现象发生得很突然，并且在拒绝发生的位置，草稿模型的熵 (entropy) 会非常高。因此，SVIP 的核心目标就是利用草稿模型自身的熵来检测这种不确定性，从而动态地决定何时停止起草。

**Lower Bound of Acceptance** 

SVIP 的推导始于对推测解码系统 (SD systems) 理论接受率的深入研究。给定目标模型 $p$ 和草稿模型 $q$，一个草稿词元 $x_t$ 的接受概率为 $min(1, \frac{p(x_t)}{q(x_t)})$.

在整个分布上，期望接受概率 $\beta$ 为: 

$$
\begin{aligned}\beta&=\sum_xq(x)\cdot\min\left(1,\frac{p(x)}{q(x)}\right)\\&=\sum_x\min\left(p(x),q(x)\right),\end{aligned}\tag{1}
$$

$\beta$ 指在当前上下文中，草稿模型 $q$ 生成的任何一个词元被目标模型 $p$ 接受的平均概率或期望值。

论文指出，$\beta$ 与 $p$ 和 $q$ 之间的总变差距离 (TVD) 相关。随后，论文利用 Pinsker 不等式推导出了 $\beta$ 的一个下界，称之为“神谕下界” (oracle bound) : 

$$
\begin{aligned}\beta&=1-\mathrm{TVD}(p,q)&\\&\geqslant1-\sqrt{\frac12\mathbb{K}\mathbb{L}(q||p)}&\\&=1-\sqrt{\frac12H_{q,p}-\frac12H_q}&\end{aligned}\tag{2}
$$

其中 $H_{q,p}$ 是 $q$ 和 $p$ 之间的交叉熵， $H_q$ 是草稿模型 $q$ 自身的熵。

{{< details title="公式 2 推理过程">}}

首先 两个概率分布 $p$ 和 $q$ 之间的 TVD 定义为: 

$$
TVD(p,q) = \frac{1}{2} \sum_{x} |p(x) - q(x)|
$$
对于任意两个非负数 $a$ 和 $b$ (在这里即 $p(x)$ 和 $q(x)$) ，存在一个恒等式: 

$$
a + b = |a - b| + 2 \cdot min(a, b)
$$

可以验证一下这个恒等式: 如果 $a \ge b$，则 $a+b = (a-b) + 2b$，成立。如果 $b > a$，则 $a+b = (b-a) + 2a$，也成立。

将恒等式应用于概率我们将这个恒等式应用于 $p(x)$ 和 $q(x)$: 

$$
p(x) + q(x) = |p(x) - q(x)| + 2 \cdot min(p(x), q(x))
$$. 
将上式在词表中的所有 $x$ 上进行求和: 

$$
\sum_{x} (p(x) + q(x)) = \sum_{x} |p(x) - q(x)| + \sum_{x} (2 \cdot min(p(x), q(x)))
$$. 
利用概率性质简化由于 $p$ 和 $q$ 都是概率分布，它们在各自词表上所有 $x$ 的概率之和都等于 1. 代入等号左侧。现在，我们的等式变成了: 

$$
2 = \sum_{x} |p(x) - q(x)| + 2 \cdot \sum_{x} min(p(x), q(x))
$$. 
求解 $min$ 的和我们将等式变形，

$$
\sum_{x} min(p(x), q(x)) = 1 - \frac{1}{2} \sum_{x} |p(x) - q(x)|
$$. 
替换回 $\beta$ 和 TVD 即可得到: 

$$\beta = 1 - TVD(p,q)$$

{{< /details >}}

这个“神谕下界”虽然理论上成立，但在实际的草稿生成 (drafting) 阶段无法使用。因为计算它需要 $H_{q,p}$，而 $H_{q,p}$ 依赖于目标模型 $p$ 的瞬时访问，这在起草阶段是不可行的。

**Approximating the Oracle Bound with Draft  Distribution** 为了让这个下界变得实用，论文提出了一个关键的近似方法: 仅使用草稿模型自身的熵 $H_q$ 来近似交叉熵 $H_{q,p}$. 首先，假设 $H_{q,p} = \gamma H_q$，其中 $\gamma$ 是一个随机变量，代表 $H_{q,p}$ 和 $H_q$ 之间的比率。将此代入神谕下界，得到: 

$$
\beta \ge 1 - \sqrt{\frac{1}{2}(\gamma-1)H_q}
$$

为了让这个界限变得实用，论文进一步**用一个常数 $c$ 来近似 $\gamma$**. 最终的估计下界: 

$$
\beta\geq1-\sqrt{\frac12(\gamma-1)H_q}\approx1-\sqrt{cH_q}.
$$

这个估计下界 (approximation bound) 仅依赖于草稿模型的熵 $H_q$，因此可以在起草阶段被高效计算。

**Detecting Rejection with Draft Entropy** SVIP 利用这个下界来动态调整草稿长度。在草稿模型生成每一个词元后，计算这个估计下界 ($1 - \sqrt{cH_q}$) . 如果这个值低于某个预设的接受率门槛 $\hat{h}$，就停止起草，即 $1 - \sqrt{cH_q} < \hat{h}$. 由于 $c$ 和 $\hat{h}$ 都是常数超参数，论文将它们合并为一个的阈值 $h$，其中 $h = (1 - \hat{h}) / \sqrt{c}$. 最终的SVIP停止准则: 

$$\sqrt{H_q(x_{<t})} > h$$ 

在给定的前缀 $x_{<t}$ 下，如果草稿模型分布的熵的平方根 $H_q$ 超过了阈值 $h$，就意味着估计的接受概率太低，此时应立即停止起草。

**Justifying the Approximation Bound** 为了确保这个估计下界是安全 ，它必须始终低于神谕下界。通过推导，这要求随机变量 $\gamma$ 必须满足 $\gamma \le 2c + 1$.  n$\gamma$ 是一个 $\ge 1$ 的右偏随机变量。如果将其建模为 $\gamma = 1 + X$ (其中 $X \sim Gamma(\alpha, \beta)$) ，那么这个近似有效的概率 $P(\gamma \le 2c+1)$ 就可以被计算出来。

选择 $c$ (即选择 $h$) 是一个权衡: 
- $c$ 太小: $\gamma \le 2c+1$ 的概率变低，导致下界估计可能失效。
- $c$ 太大: 下界 $1 - \sqrt{cH_q}$ 会变得过松，导致估计的接受率过低，过早停止起草，影响加速效率。

论文通过实验 (图4) 证明，只要 $h$ (即 $c$) 被恰当选择 (在图中设置为 0.18) ，这个估计下界在大多数情况下都能很好地工作。

![Comparison between the actual acceptance  probability, the acceptance probability  lower bound, and the estimated lower  bound. Each position on the x-axis  corresponds to a token, which has been sorted according  to the actual acceptance probability.](https://share.note.youdao.com/yws/api/personal/file/WEBad828bea4e494d842b1d57c854c61493?method=download&shareKey=ce782f68496a1fb3fc488ad7e6d411b8 "Comparison between the actual acceptance  probability, the acceptance probability  lower bound, and the estimated lower  bound. Each position on the x-axis  corresponds to a token, which has been sorted according  to the actual acceptance probability.")

# 3 Experiments

论文主要对比了三种不同的草稿长度 (draft length) 策略：

1.  **Constant (固定值)**：使用一个固定的草稿长度，实验中设为 5.
2.  **Heuristic (启发式)**：Hugging Face `transformers` 库中实现的策略。其逻辑是：如果当前迭代中所有草稿词元都被接受，则下一轮草稿长度+2；否则-1。
3.  **SVIP (本文方法)**：使用草稿模型的熵来动态决定何时停止起草.

评估指标

* **主要指标**：**加速比 (Speedup)**。计算方式为：仅使用目标模型进行自回归解码所需的时间 / 使用推测解码 (SD) 所需的时间。
* **辅助指标**：平均接受的草稿长度 (Accepted draft lengths) 和草稿词元接受率 (Draft token accept rate) 。

---

### 实验 1：长文本生成 (Long-form Generation)

这部分实验在常规的长文本生成任务上验证 SVIP。

**实验设置：**
* **基准 (Benchmark)**：MT-Bench。
* **模型**：
    * **目标模型 (Target)**：Qwen2.5-14B 和 Qwen2.5-32B.
    * **草稿模型 (Draft)**：不同大小的 Qwen2.5 模型 (0.5B, 1.5B, 3B).
* **上下文长度**：最高测试到 8K (8192) 上下文。
* **解码策略**：使用**采样 (Sampling)** 并设置 `temperature=1`.  (作者发现，若使用贪婪解码 (greedy decoding)，模型在长文本生成中易陷入“重复幻觉”，导致熵极低和加速比被夸大，因此主实验采用采样) 。
* **SVIP 设置**：熵阈值 $h$ 设为 0.3.

**实验结果：**

1.  **SVIP 显著优于基线 (图 5)**：
    * 在所有的模型组合 (如 14B+0.5B, 32B+1.5B) 和所有上下文长度 (从 128 到 8192) 上，SVIP (红线) 的加速比都**持续且显著地**优于 "Constant 5" (蓝线) 和 "Heuristic" (绿线) 策略。

2.  **SVIP 为什么更快？ (图 6)**：
    * **"Heuristic" 策略 (绿线) **：倾向于生成**非常长**的草稿 (在 8K 上下文时平均长度接近 8) ，但其**接受率很低** (约 60%)，导致效率低下。
    * **"SVIP" 策略 (红线) **：当检测到高熵 (不确定性) 时会**提前停止**。因此，它生成的草稿**更短** (平均约 2.5-3.0)，但换来了**极高的接受率** (在长上下文时接近 90%). 这证明了 SVIP 的加速来自于更高的效率，而非盲目生成长草稿。

3.  **SVIP 完美拟合“神谕长度” (图 7b)**：
    * 论文分析了 (实际草稿长度 - 神谕草稿长度) 的差值 (Delta Draft Length) 。
    * 结果显示，"Constant" 和 "Heuristic" 策略都倾向于“**过度生成**” (over-generate) ，即生成的草稿远超目标模型能接受的长度。
    * 而 SVIP 的差值**几乎完美地保持在 0 附近** (平均 delta < 0.5) ，说明它能精确地在目标模型即将拒绝时停止起草。

4.  **SVIP 可增强 SOTA 方法 (表 2)**：
    * 为了证明 SVIP 的灵活性和正交性，作者将其应用在当时的 SOTA 推测解码系统 **EAGLE-2** 之上。
    * 在 Vicuna-7B @ 4K 上下文的测试中，EAGLE-2 本身的加速比为 1.24x，而 **EAGLE-2 + SVIP** 达到了 1.41x (相对提升 13.7%).
    * 证明 SVIP 是一个即插即用的模块，能为其他已优化的 SD 系统带来额外收益。

---

### 实验 2：长文本推理 (Long-form Reasoning)

这部分实验在更困难、输出更长的推理任务上验证 SVIP.

**实验设置：**
* **基准 (Benchmark)**：MATH, AIME, 和 GPQA。这些都是推理密集型任务，平均生成长度从 1.3K 到 6.2K 不等。
* **模型**：
    * **目标模型**：QwQ-32B-Preview (一个推理专用模型) 。
    * **草稿模型**：由于 QwQ 没有小模型，作者专门为其蒸馏训练了一个 1.5B 的草稿模型。

**实验结果 (表 3)：**

* **SVIP 在推理任务上优势巨大**：SVIP 在所有推理基准上均大幅超越基线。
* **平均加速比 (Avg)**：
    * SVIP: **1.71x**
    * Constant: 1.49x
    * Heuristics: 1.28x
* **在最长的 AIME 任务上 (平均 6.2K 长度)**：
    * SVIP: **1.77x**
    * Constant: 1.58x
    * Heuristics: 1.34x
* **分析 (表 4 和 图 8)**：
    * 分析表明，SVIP 的行为与实验 1 一致：它实现了与 "Constant" 策略相似的平均草稿长度，但是**接受率高得多**。
    * 图 8 的个例分析显示，在推理过程中，SVIP 提议的草稿长度**剧烈振荡** (从接近 0 到 35 以上) ，这再次凸显了固定长度策略的局限性，以及 SVIP 动态性的必要性。
    * 表 4 验证了 SVIP 的动机：在 QwQ 模型中，熵很高的词元 (如 "Alright") 接受率确实很低 (0.22) ，而熵很低的词元 (如 "ponents") 接受率则为 1.00.