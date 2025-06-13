---
title: "Fast-dLLM"
date: 2025-06-12T23:01:49+08:00
lastmod: 2025-06-12T23:01:49+08:00
author: ["WITHER"]

categories:
- Paper Reading

tags:
- DiffusionLLM

keywords:
- DiffusionLLM

description: "Paper Reading of Fast-dLLM" # 文章描述，与搜索优化相关
summary: "Paper Reading of Fast-dLLM" # 文章简单描述，会展示在主页
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

# Introduction

Diffusion LLMs 被视为下一代文本生成技术的有力竞争者，其核心优势在于理论上可以并行生成多个 token，从而有望实现比自回归模型快几个数量级的推理速度。谷歌的 Gemini Diffusion 和 Inception Labs 的Mercury等模型已经展示了其惊人的潜力，宣称能达到每秒上千 token 的生成速度。

当前开源的扩散LLM (LLaDA、Dream) 在实际应用中的速度远远达不到预期，甚至比优化良好的自回归模型还要慢。这篇论文的工作，就是要拆掉阻碍扩散 LLM 起飞的两座大山。

1. 无法使用 KV Cache

扩散LLM的注意力机制是双向的，即一个 token 的生成不仅依赖于它前面的内容，也依赖于它后面的内容（尽管后面可能是未知的 MASK token ）。这种特性使得过去的信息和未来的信息相互纠缠，无法像自回归模型那样简单地缓存和复用过去的信息。导致扩散LLM在每一步推理中都需要进行大量的重复计算，严重拖慢了速度。

Fast-dLLM 的第一个核心贡献，就是提出了一种分块近似 (block-wise approximate) KV Cache 机制。

{{< quote >}}
While the bidirectional nature of attention in Diffusion LLMs precludes a fully equivalent KV Cache, our approximation closely resembles an ideal cache in practice. 
{{< /quote >}}

它将待生成的文本序列分成若干个块. 在生成某一个块 (比如Block 1) 时，它会提前计算并缓存其他所有块 (比如 Prompt 和 Block 0) 的 KV. 在这个块的内部生成过程中，这些缓存被反复利用。当这个块生成完毕后，再整体更新一次所有块的KV缓存 。

这个方法的近似在于，在一个块的生成过程中，缓存是固定的，而实际上随着块内 token 的不断去噪和清晰化，这些缓存理论上也应该随之微调。但论文通过可视化实验（图3）有力地证明，在相邻的推理步骤中，KV 激活值的 余弦相似度非常高，几乎接近于1. 这说明使用固定的近似缓存带来的误差微乎其微，完全可以用极小的精度损失换取巨大的速度提升。

论文还进一步提出了双缓存 (DualCache) 版本，不仅缓存了前面的“前缀”（prefix），还缓存了后面的“后缀”（suffix，通常是 MASK  token ） ，从而进一步压榨了计算优化的空间，实现了更快的速度。

2. 并行解码带来的质量下降

扩散LLM的另一大理论优势是 并行解码 (Parallel Decoding)，即一次性预测和生成多个 token  。然而，实践再次证明，当并行解码的 token 数量增多时，生成文本的质量会急剧下降 。

论文深刻地剖析了其根源：条件独立性假设 (conditional independence assumption) 的破坏 。在并行解码时，模型是独立地为每个待生成的 MASK 位置预测一个概率分布，然后从中采样。但实际上，一句话中的 token 之间存在着强烈的依赖关系。论文举了一个例子:

{{< quote >}}
Consider an example from [30]: The list of poker hands that consist of two English words are: The subsequent two words could be, for instance, "high card," "two pair," "full house," or "straight flush." [...] However, the multi-token prediction procedure in MDMs first generates a probability distribution for each token and then samples from these distributions independently. This independent sampling can lead to undesirable combinations, such as "high house."
{{< /quote >}}

模型可能会独立地预测出 "high" 和 "house"这两个词，但把它们组合在一起就成了毫无意义的 high house. 这是因为模型在并行预测时忽略了 token 间的联合概率，而错误地直接使用了边缘概率的乘积。

为了解决这个问题，Fast-dLLM提出了第二个核心贡献：置信度感知并行解码 (Confidence-Aware Parallel Decoding) 策略 。这个想法非常直观且有效：我们只对那些模型非常有把握的 token 进行并行解码。

具体来说，在每一步解码时，模型会为每个待生成的 MASK 位置计算一个 置信度分数 (比如softmax概率的最大值). 然后，设定一个全局的置信度阈值 τ，只有那些置信度超过这个阈值的 token 才会被揭开，而置信度不足的 token 则继续保持 MASK 状态，留到下一步再做决策。为了避免无限循环，如果没有任何 token 的置信度达标，模型会强制解码置信度最高的那一个。

这个策略的精妙之处在于，它在理论上是站得住脚的。论文通过定理一从数学上证明了：当模型对一组 token 的预测置信度足够高时 (即 p>1−ϵ，且 ϵ 足够小)，基于独立边缘概率的“贪心并行解码”与基于真实联合概率的“贪心串行解码”会得到完全相同的结果。

![Effectiveness of Components of Fast-dLLM across Different Approaches](https://share.note.youdao.com/yws/api/personal/file/WEBccefa918e999469a4faa3badff3c32b9?method=download&shareKey=c9e48ddb1e1f0600394ce8baa1d84426 "Effectiveness of Components of Fast-dLLM across Different Approaches")

Fast-dLLM 的创新性体现在它是一种 training-free 的加速框架。它没有修改模型结构，也不需要重新训练，而是通过两项即插即用的推理策略——“分块近似KV缓存”和“置信度感知并行解码”，分别从减少重复计算和提升并行效率两个维度，精准地解决了当前开源扩散 LLM 面临的核心瓶颈。 实验结果在 LLaDA 和 Dream 等模型上，结合两种策略，实现了高达 27.6 倍的端到端吞吐量提升，同时在多个基准测试上几乎没有精度损失。

# 2. Preliminary

### 2.1. Masked Diffusion Model

针对离散数据的扩散模型最早在 Argmax Flows and Multinomial Diffusion 和 Deep Unsupervised Learning using
Nonequilibrium Thermodynamics 中被探提出。随后 D3PM 提出了一个更通用的框架，通过特定的转移矩阵 $Q_{t}$ 定义了前向加噪过程的离散状态马尔可夫链，并通过最大化 ELBO 来学习反向过程的参数化模型 $p_{\theta}(x_{0}|x_{t})$. CTMC 进一步将 D3PM 扩展到连续时间，将其形式化为一个连续时间马尔可夫链 (CTMC) 框架。在另一种不同的方法中，SEDD 通过参数化似然比 $\frac{p_{t}(y)}{p_{t}(x)}$ 来学习反向过程，并采用去噪分数熵来训练该比率。

在各种离散扩散的噪声处理方式中，**Masked Diffusion Models, MDMs**，也被称为吸收状态离散扩散模型，获得了相当大的关注。MDMs 采用一种前向加噪过程，其中 token 被逐步替换为一个特殊的 MASK  token  。这个过程由以下转移概率定义：

$$
q_{t|0}(x_{t}|x_{0})=\prod_{i=1}^{n}q_{t|0}(x_{t}^{i}|x_{0}^{i})=\prod_{i=1}^{n}Cat(x_{t}^{i};(1-t)\delta_{x_{0}^{i}}+t\delta_{[MASK]}) \quad(1)
$$ 

* $q_{t|0}(x_t|x_0)$: 表示给定原始序列 $x_0$，得到噪声序列 $x_t$ 的概率 。
* $\prod_{i=1}^{n}$: 连乘符号，表示整个序列的噪声过程是序列中每个 token （token）独立进行噪声过程的概率乘积 。
* $Cat(\cdot)$: 代表**类别分布 (Categorical Distribution)** 。
* $t \in [0,1]$: 表示**扩散时间**或**掩码级别**。当 $t=0$ 时，序列完全是原始的；当 $t=1$ 时，序列被完全替换为 `[MASK]`  token 。
* $(1-t)\delta_{x_{0}^{i}}+t\delta_{[MASK]}$: 在时间 `t`，第 `i` 个 token 有 $1-t$ 的概率保持其原始身份 $x_0^i$，有 $t$ 的概率变成 `[MASK]`  token 。`$\delta$` 是克罗内克函数，用于指定概率。

最近，MDLM 和 RADD 的工作表明，对于 MDMs 不同的参数化是等价的。此外，他们证明了 MDMs 的训练目标可以被简化或直接从数据似然中推导出来 。这导出了以下目标函数，即 $log~p_{\theta}(x)$ 的一个 ELBO:

$$
-log~p_{\theta}(x)\le\int_{0}^{1}\frac{1}{t}\mathbb{E}_{q_{t,0}(x_{t}|x_{0})}[\sum_{i:x_{t}^{i}=[MASK]}-log~p_{\theta}(x_{0}^{i}|x_{t})]dt:=\mathcal{L}_{MDM}. \quad(2)
$$ 

* $-log~p_{\theta}(x)$: 模型的目标是最大化生成真实数据 $x$ 的对数似然，这等价于最小化它的负对数似然。这个公式给出了负对数似然的一个* ELBO.
* $\int_{0}^{1}...dt$: 对所有可能的噪声级别 `t` (从0到1) 进行积分，意味着模型需要学会在任何噪声水平下都能很好地复原数据 。
* $\mathbb{E}_{q_{t,0}(x_{t}|x_{0})}[...]$: 表示对所有可能的噪声样本求期望。在训练时，我们根据公式(1)随机生成一个带 `[MASK]` 的噪声序列 $x_t$.
* $\sum_{i:x_{t}^{i}=[MASK]}-log~p_{\theta}(x_{0}^{i}|x_{t})$: 
    * $\sum_{i:x_{t}^{i}=[MASK]}$: 对所有被 `[MASK]` 的位置 `i` 进行求和 。
    * $-log~p_{\theta}(x_{0}^{i}|x_{t})$: 这是交叉熵损失。它的意思是，给定带有 `[MASK]` 的序列 $x_t$，模型 $p_{\theta}$ 需要预测在位置 i 上的原始 token  $x_0^i$ 应该是什么。模型预测得越准，这个损失值就越小。

### 2.2. MDMs 的生成过程

对于公式1中定义的前向过程，其解析上的逆过程在生成时计算效率低下，因为它通常每步只修改一个 token 。一个常见的加速策略是采用 $\tau$-leaping 近似法来处理反向过程。在 MDMs 的背景下，这允许一个迭代式的生成过程，其中多个被掩码的 token 可以从一个噪声水平 t 近似地单步恢复到一个更早的水平 s < t.

$$
q_{s|t}(x_s|x_t)=\prod_{i=0}^{n-1}q_{s|t}(x_{s}^{i}|x_{t})
$$

其中

$$
q_{s|t}(x_{s}^{i}|x_{t})=\begin{cases}1, & \text{if } x_{t}^{i}\ne[MASK], x_{s}^{i}=x_{t}^{i} \\ \frac{s}{t}, & \text{if } x_{t}^{i}=[MASK], x_{s}^{i}=[MASK] \\ \frac{t-s}{t}q_{0|t}(x_{s}^{i}|x_{t}), & \text{if } x_{t}^{i}=[MASK], x_{s}^{i}\ne[MASK]\end{cases} \quad(3)
$$

* $q_{s|t}(x_{s}^{i}|x_{t})$: 表示从 `t` 时刻的 token  $x_t^i$ 变为 `s` 时刻的 token  $x_s^i$ 的概率 。
* **Case 1**: 如果一个 token 在 `t` 时刻就不是 `[MASK]`，那么它在更早的 `s` 时刻也保持不变 。
* **Case 2**: 一个在 t 时刻是 `[MASK]` 的 token ，在更早的 s 时刻仍然是 `[MASK]`. 
* **Case 3**: 这是关键的去噪步骤。如果一个 token 在 `t` 时刻是 `[MASK]`，模型会尝试在 s 时刻预测出一个具体的 token.
    * $\frac{t-s}{t}$: 代表一个在 `t` 时刻被掩码的 token，在 `s` 时刻被“揭示”出来的概率 。
    * $q_{0|t}(x_{s}^{i}|x_{t})$: 这是由神经网络模型给出的预测分布。模型会观察整个带有 `[MASK]` 的上下文 $x_t$，然后为当前位置预测一个最有可能的原始 token ，并给出一个在整个词汇表上的概率分布 。

在涉及条件数据的场景中，例如根据一个 propmt p 生成一个回应 $x_{0}$，MDM 的反向过程 (公式3所定义) 需要进行调整。具体来说，模型用于揭示一个 token  $x_{s}^{i}$ 的预测分布 $q_{0|t}(x_{s}^{i}|x_{t})$ 现在也需要以 prompt p 为条件，即 $q_{0|t}(x_{s}^{i}|x_{t},p)$ 。

**并行解码的诅咒**
直接逆转公式1的前向过程来进行生成是缓慢的，通常每步只改变一个 token. 一个常见的加速策略是采用 $\tau$-leaping 近似法来处理反向过程。对于 MDMs，这意味着多个被掩码的 token 将在一个步骤中并行生成。然而，由于条件独立性假设，多 token 预测中出现了一个重大挑战。考虑一个例子：由两个英文单词组成的扑克手牌列表是：随后的两个词可能是，例如，high card，two pair，full house，或 straight flush. 值得注意的是，这两个词之间存在着关联。然而，MDMs 中的多 token 预测过程首先为每个 token 生成一个概率分布，然后独立地从这些分布中进行采样。这种独立采样可能导致不希望的组合，例如 high house.

为了将其形式化，考虑揭示两个 token 位置 i 和 j. 由于条件独立性假设，MDMs 从 $p(x_{s}^{i}|x_{t})\cdot p(x_{s}^{j}|x_{t})$ 中采样这些 token. 然而，真实的联合概率需要考虑它们之间的依赖关系：

$$
p(x_{s}^{i},x_{s}^{j}|x_{t})=p(x_{s}^{i}|x_{t})\cdot p(x_{s}^{j}|x_{t},x_{s}^{i})
$$

或者对称地，通过将 i 依赖于条件 j. 这种假设的独立生成与真实的依赖性数据分布之间的差异，会降低生成序列的质量和连贯性。当在单一步骤中同时揭示大量 token 时，这个问题会变得更加严重。

# 3. Methodology

## 3.1. Pipeline Overview

**Fast-dLLM**，建立在 MDM 架构之上，以实现高效和高质量的序列生成。为了加速推理，整体流水线融合了两大关键策略：通过 KV Cache 实现的高效注意力计算，以及一个由预测置信度引导的 并行解码方案。具体来说，我们采用了分块解码设计的 KV Cache，它允许在不同步骤间复用注意力激活值，并显著减少了冗余计算。在每个块内部，进一步提出了置信度感知的并行解码，它能根据置信度分数选择性地更新 token ，从而在保持输出质量的同时提高效率。通过结合这些策略，Fast-dLLM 在对生成性能影响最小的情况下，显著加快了 MDM 的推理速度。整体流程在算法 1 中进行了总结。 

## 3.2. Key-Value Cache for Block-Wise Decoding

![Illustration of our Key-Value Cache for Block-Wise Decoding](https://share.note.youdao.com/yws/api/personal/file/WEBe66f192a665248e7559ffa12a0bf10c1?method=download&shareKey=8952caa17d664bd8bcc33b9ebcec321e "Illustration of our Key-Value Cache for Block-Wise Decoding")

如上图所示，我们采用了一种分块解码的策略来支持 KV Cache 的使用。一开始计算并存储 prompt 的 KV 缓存，这个缓存将在整个块 0的解码过程中被复用。在每个块的内部，相同的缓存会被多个解码步骤复用。在完成一个块的解码之后，更新**所有** token  (不仅仅是新生成的 token ) 的缓存。这个缓存更新可以与解码步骤联合执行，因此与不使用缓存相比，没有额外的计算开销。由于掩码扩散模型中使用的是完全注意力机制，这种方法导致了一个近似的解码过程。 

我们的近似 KV 缓存方法的有效性，源于我们观察到 KV 激活值在相邻的推理步骤中表现出高度的相似性，如下图所示。图 a 中红色方框区域突显了块内的相似性分数，这些分数始终接近于1。这表明在分块解码期间，前缀 (prefix) 的键和值的差异可以忽略不计，使我们能够安全地复用缓存而不会有显著的准确率损失。 此外，我们实现了一个我们 KV 缓存机制的双向版本，名为 **DualCache**，它不仅缓存前缀 token ，还缓存后缀 (suffix)  token ，在我们的分块解码方案中，后缀完全由掩码 token 组成。如表3所示，DualCache 带来了进一步的加速。图 b 中的红色方框区域进一步证明，在分块解码期间，后缀的键和值的差异也可以忽略不计。 

![Heatmaps of Key-Value Activation Cosine Similarity Across Inference Steps in LLaDA](https://share.note.youdao.com/yws/api/personal/file/WEB2030e80c11d3d306e335a2dc5931b101?method=download&shareKey=6a5005c556aaa11edb4006a48b755b4a "Heatmaps of Key-Value Activation Cosine Similarity Across Inference Steps in LLaDA")

## 3.3. Confidence-Aware Parallel Decoding


尽管存在一些方法，例如使用辅助模型来显式地捕捉不同位置 token 之间的依赖关系，但它们通常会增加整个流水线的复杂性。与这些方法相反，我们提出了一个简单而有效的**置信度感知解码算法**，旨在缓解这种条件独立性问题。

在每次迭代中，我们不是冒然地使用它们独立的边缘概率来揭示所有被掩码的 token ，而是为每个 token 计算一个置信度分数 (例如最大的 softmax 概率). 只有那些置信度超过一个阈值的 token 才会在当前步骤被揭示；其余的则保持掩码状态，并在未来的步骤中重新考虑。如果没有 token 的置信度超过阈值，就揭示置信度最高的那一个，以确保过程能够进行并防止无限循环。这个策略在加速生成的同时，减少了由不确定或模糊预测引起的错误。 

一个关键问题是

{{< quote >}}
*When is it theoretically justifiable to decode tokens in parallel using independent marginals, despite the true joint distribution potentially containing dependencies?*
{{< /quote >}}

以下结果来回答了在高置信度情况下，greedy parallel 解码等同于 greedy sequential 解码的条件，并量化了两种分布之间的差异。在给出定理之前，我们将定义其表述中使用的数学符号。

设 $p_{\theta}(\cdot|E)$ 表示一个 MDM 在给定 E (包括 prompt $p_{0}$ 和先前生成的 token) 的条件下给出的 PMF. 假设模型要为不在 E 中的位置 $i_{1},...,i_{n}$ 预测 n 个 token.

令 $X=(X_{i_{1}},...,X_{i_{n}})$ 是 n 个 token 的向量，其中每个 $X_{i_{j}}$ 在词汇表 V 中取值。设 $p(X|E)\equiv p_{\theta}(X_{i_{1}},...,X_{i_{n}}|E)$ 是模型给出的联合条件 PMF。设 $p_{j}(X_{i_{j}}|E)\equiv p_{\theta}(X_{i_{j}}|E)$ 是位置 $i_{j}$ 的边缘条件 PMF。并行解码使用边缘概率的乘积来生成 token ：$q(X|E)=\tilde{\prod}_{j=1}^{n}p_{j}(X_{i_{j}}|E)$。定理1的证明及相关讨论见附录A。 

**定理 1 (高置信度下的并行解码).** 假设存在一个特定的 token 序列 $x^{*}=(x_{i_{1}},...,x_{i_{n}})$，使得对于每个 $j\in\{1,...,n\}$，模型对 $x_{i_{j}}$ 都有很高的置信度：$p_{j}(X_{i_{j}}=x_{i_{j}}|E)>1-\epsilon$，对于某个很小的 $\epsilon>0$. 那么，以下结论成立：

1. **贪婪解码的等价性**：如果 $(n+1)\epsilon\le1$（即 $\epsilon\le\frac{1}{n+1}$），那么
$$
\text{argmax}_{z} p(z|E) = \text{argmax}_{z} q(z|E) = x^{*}. \quad (4)
$$ 

这意味着 greedy parallel 解码 (选择 argmax q) 与贪婪序贯解码 (选择 argmax p) 产生相同的结果。  这个界是紧的：如果 $\epsilon > \frac{1}{n+1}$，则存在满足高置信度边缘假设的分布 $p(X|E)$，使得 argmax $p(z|E)$ ≠ argmax $q(z|E)$。 

2. *Distance and Divergence Bounds*：为简洁起见，将 $p(\cdot|E)$ 和 $q(\cdot|E)$ 表示为 p 和 q。 

**$L_p$ Distance ($p \ge 1$)**: 对于 $n>1$，$D_{p}(p,q)<((n-1)^{p}+2n)^{1/p}\epsilon$。特别地，对于总变差距离 ($D_{TV}(p,q)=\frac{1}{2}D_{1}(p,q)$)，$D_{TV}(p,q)<\frac{3n-1}{2}\epsilon$。 

**Forward KL Divergence**: 对于 $n > 1$，$D_{KL}(p||q)<(n-1)(H_{b}(\epsilon)+\epsilon~ln(|\mathcal{V}|-1))$，其中 $H_{b}(\epsilon)=-\epsilon~ln~\epsilon-(1-\epsilon)ln(1-\epsilon)$ 是二元熵函数，而 $|\mathcal{V}|$ 是词汇表的大小。 

# 4. Experiments

---
## 4.1 Experimental Setup

* **硬件与环境** 🖥️: 所有实验均在单张 **NVIDIA A100 80GB GPU** 上进行，batch size=1.
* **评测模型** 🧠: **LLaDA**  和 **Dream**.
* **评测基准** 📊: 采用了四个广泛使用的基准数据集：**GSM8K**、**MATH**、**HumanEval** 和 **MBPP**.
* **核心指标** ⏱️:
    * **准确率 (Accuracy)**: 衡量模型在具体任务上的表现。
    * **吞吐量 (Throughput)**: 以 tokens/sec 为单位，反映端到端的真实解码速度。
* **超参数** ⚙️:
    * **缓存块大小**: 在 4 到 32 之间进行探索。
    * **置信度阈值**: 在 0.5 到 1.0 之间进行探索。
    * 实验默认使用 **PrefixCache**，块大小为 **32**，置信度阈值为 **0.9**.

---
## 4.2 Main Results: Performance and Speed

实验结果表明，Fast-dLLM 在各种任务和设置上都取得了显著的速度提升，同时对模型准确率的影响微乎其微 。

* 加速效果:
    * 单独引入 KV Cache 机制，通常能带来 **2x-3.6x** 的速度提升。
    * 当 KV Cache 和并行解码两种策略结合使用时，性能提升更为显著。在 LLaDA 模型上，最 高可达 **11.0x** 的吞吐量提升；在 Dream 模型上，最高可达 **7.8x** 的提升 。
* 极小的精度损失: 在所有基准测试中，加速后模型的准确率与原始基线模型的差距基本保持在 **1-2个百分点** 以内，有时甚至略有提高。
* 对长序列更友好: 实验还发现，在处理更长的文本序列时 (例如 few-shot 场景或长代码生成)，Fast-dLLM 的加速效果更为明显。


下表以 GSM8K (5-shot) 任务为例，直观展示了 Fast-dLLM (即 +Cache+Parallel) 相较于基线模型的性能提升。

| 模型 | 生成长度 | 配置 | 准确率 (%) | 吞吐量 (tok/s) | 相对加速 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LLaDA** | 256 | Baseline | 79.3 | 6.7 | 1x |
| | | **Fast-dLLM** | **78.5** | **54.4** | **8.1x** |
| | 512 | Baseline | 77.5 | 3.2 | 1x |
| | | **Fast-dLLM** | **77.2** | **35.3** | **11.0x** |
| **Dream** | 256 | Baseline | 75.0 | 9.1 | 1x |
| | | **Fast-dLLM** | **74.8** | **48.2** | **5.3x** |
| | 512 | Baseline | 76.0 | 7.7 | 1x |
| | | **Fast-dLLM** | **74.0** | **42.9** | **5.6x** |

---
## 4.3 Ablations and Analysis

为了深入理解各个组件的贡献，论文进行了一系列详细的消融实验。

* **输入与生成长度的影响**:
    * 实验证明，更长的上下文 (prefill，如从 5-shot 增加到 8-shot) 和更长的生成长度，都能显著放大加速效果。
    * 在 8-shot 和 1024 生成长度的设置下，**DualCache** 实现了 **27.6x** 端到端加速。

* **PrefixCache vs. DualCache**:
    * **DualCache** 通常比只缓存前缀的 **PrefixCache** 实现更高的加速比，尤其是在长序列生成任务中 。

* **缓存块大小的影响**:
    * **small block size**：准确率最高，但因频繁更新缓存导致开销较大，速度提升有限 。
    * **small block size**：速度快，但可能因上下文不匹配导致准确率下降 。
    * 实验发现，块大小为 **32** 时在速度和精度之间取得了最佳平衡。

![Impact of Cache Block Size on Accuracy and Throughput](https://share.note.youdao.com/yws/api/personal/file/WEB9772b6d4b4341a7ccb12bee9eef34910?method=download&shareKey=1e3a007e630de1a9cbf8b3d9f318f307 "Impact of Cache Block Size on Accuracy and Throughput")

* **动态阈值 vs. 固定步数策略**:
    * 论文提出的 **置信度感知并行解码** 策略，在性能上持续优于每步固定解码 K 个 token 的 baseline 方法。
    * 在达到相似甚至更高准确率的同时，该动态策略能实现更高的平均每步解码 token 数，从而获得更高的吞吐量。

![Threshold VS Fxied Step](https://share.note.youdao.com/yws/api/personal/file/WEBd7916aff1aba60846ae1e971b2800e0a?method=download&shareKey=88d29eb3e40615a74c4846d278413e5b "Threshold VS Fxied Step")

# 5. Related Work

本章节回顾了与 Fast-dLLM 相关的两个核心领域：扩散语言模型的发展，以及大语言模型的通用加速技术。

---

## 5.1. Diffusion LLM

扩散模型作为一种强大的生成范式，最初在图像和音频等连续数据领域取得了巨大成功，随后其影响力扩展到了 NLP. 特别是离散扩散模型的最新进展为大语言模型提供了一种替代自回归 (AR) 范式的可行方案 。

* **理论基础的发展**:
    * 离散数据的扩散模型最早由 [29, 11] 探索 。
    * **D3PM** 提出了一个更通用的框架，将前向加噪过程建模为离散状态马尔可夫链，并通过最大 ELBO 来学习反向过程。
    * **CTMC** 将 D3PM 扩展到连续时间设定 。
    * **SEDD** 采用了不同的方法，通过参数化边际似然比来学习反向过程 。
    * **MDMs** 近期受到了广泛关注，其中 **MDLM** 和 **RADD** 的研究表明，MDMs 的不同参数化方法是等价的，并且其训练目标可以被简化 。

* **与预训练语言模型的结合**: 一个关键的突破是将离散扩散与现有的大语言模型架构相结合 。
    * **Diffusion-NAT** [40] 将离散扩散的去噪过程与 BART 的非自回归解码相结合，通过迭代式地优化被掩码的 token ，实现了比同类自回归 Transformer 快20倍的生成速度 。
    * **LLaDA** [21]、**DiffuLLaMA** [7] 和 **Dream** [36] 等框架将扩散模型扩展到了 7B 参数的规模，通过在扩散时间步上进行递归式的 token 预测，展现了与 LLaMA3 等主流自回归模型相匹敌的性能 。


## 5.2. LLM Acceleration

- KV Cache

由于 LLaDA 等扩散语言模型采用的是 **full attention**，将 KV 缓存直接应用于这类模型并非易事。 一篇相关的研究 **Block diffusion**  通过**分块生成 (block-by-block)** 的方式，克服了先前扩散语言模型的局限，使得缓存和复用先前已解码块的键和值成为可能 。
 
- Non-Autoregressive Generation

非自回归 (NAR) 生成标志着一种根本性的转变，它通过同时生成多个 token 来显著加速推理过程。NAR 方法最初被用于神经机器翻译，现已扩展到语法纠错、文本摘要和对话系统等多种任务 
。
尽管 NAR 在速度上优势巨大，但它通常以牺牲一定的生成质量为代价。扩散语言模型是 NAR 领域一个新兴的范式；然而，先前的工作（如 LLaDA）在实践中难以实现预期的加速，因为并行生成会导致输出质量显著下降。

# Weakness

近似缓存的误差累积效应：论文证明了在相邻步骤中，KV激活值的差异很小 。但随着生成块的增多，这种“近似”带来的微小误差是否会累积，并在生成非常长的文本（如数万 token 的小说）时导致语义漂移或一致性下降？论文的最长测试序列为1024 ，对于更长的序列，其鲁棒性有待进一步验证。

对模型能力的依赖：“置信度感知解码”策略的有效性，隐式地依赖于模型本身具有良好的“校准度”（calibration），即模型的置信度能够真实反映其预测的正确性。如果模型本身“过于自信”或“不够自信”，可能会导致该策略效果不佳。论文没有对所用模型的校准度进行分析。
定理一的理论与实践差距：论文坦诚地指出了定理一的局限性

> In practice, while MDM may not strictly satisfy this property, its behavior typically offers a close approximation.  

理论证明假设了一个“理想的”联合概率分布，而真实模型是否以及在多大程度上符合这个理想假设，是一个需要进一步探究的问题。理论和实践之间的差距可能在某些刁钻的（adversarial）或分布外（Out-of-Distribution）的场景下被放大。
超参数的敏感性与调优成本：尽管论文分析了块大小和阈值的影响，但并未提供一套系统性的方法来为新模型或新任务选择最佳超参数。在实际应用中，这可能意味着需要为每个特定用例进行成本不菲的网格搜索（grid search），增加了方法的应用门槛。
评估维度的局限性：论文主要使用了基于准确率的基准测试。但在开放式生成、对话等任务中，评估指标（如流畅度、一致性、多样性）更为复杂。Fast-dLLM是否会在这些“软”指标上引入不易察觉的负面影响，需要更全面的评估。