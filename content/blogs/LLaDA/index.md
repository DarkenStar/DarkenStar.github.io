---
title: "LLaDA"
date: 2025-06-12T13:43:16+08:00
lastmod: 2025-06-12T13:43:16+08:00
author: ["WITHER"]

categories:
- Paper Reading

tags:
- DiffusionLLM

keywords:
- DiffusionLLM

description: "Paper Reading of LLaDA" # 文章描述，与搜索优化相关
summary: "Paper Reading of LLaDA" # 文章简单描述，会展示在主页
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


LLM 主要的思想是 *generative modeling* 的思想是通过最大似然估计来优化模型的分布 $\log p_\theta(\cdot)$ 来逼近数据的分布 $\log p_{\text{data}}(\cdot)$
$$
\underbrace{\max_\theta\mathbb{E}_{p_{\text{data}}(x)}\log p_\theta(x)\Leftrightarrow\min_\theta\operatorname{KL}(p_{\text{data}}(x)||p_\theta(x)).}_{\text{Generative modeling principles}} \quad(1)
$$

当前，几乎所有我们熟知的大语言模型，从GPT系列到LLaMA系列，都基于*autoregressice modeling* 来实现。这种范式的核心是 **next-token prediction** ，即根据已经生成的文本序列，逐 toekn 地预测下一个最有可能出现的 token. 

$$
\underbrace{p_\theta(x)=p_\theta(x^1)\prod_{i=2}^Lp_\theta(x^i\mid x^1,\ldots,x^{i-1})}_{\text{Autoregressive formulation}} \quad(2)
$$

这种单向、顺序的生成方式在处理需要双向推理的任务时表现不佳，一个典型的例子就是 **Reversal Curse** ——模型知道 A is B，却往往无法推断出 B is A.

LLM 能力的核心基石是生成式建模原理本身，即通过最大似然估计让模型学习真实世界的数据分布 ，而非自回归这一具体的实现形式。

{{< quote >}}
**It is the generative modeling principles (i.e., Eq. (1)), rather than the autoregressive formulation (i.e., Eq. (2)) itself, that fundamentally underpin the essential properties of LLMs.**
{{< /quote >}}

1. 大语言模型的可扩展性 (scalability) ——即模型越大、数据越多、效果越好的特性——并非自回归模型所独有 。相反，这种可扩展性来源于更底层的生成式建模原理，而这些原理恰好保证了*fisher consistency*[^1] 

2. *instruction-following* 和 *in-context learning*[^2] 并非自回归模型所独有，而是所有设计得当的条件生成模型 (conditional generative models) 在处理结构化语言任务时都应具备的内在属性 。

因此作者提出了**LLaDA** (**L**arge **L**anguage **D**iffusion with m**A**sking)，一个从零开始训练的、参数量达到 8B 的扩散语言模型。

![Zero&Few-Shot Benchmarks](https://share.note.youdao.com/yws/api/personal/file/WEB0c215954f8c354f24d2d478a8eb89fab?method=download&shareKey=94170299ede39d5102cf1cf6e397c5c7 "Zero&Few-Shot Benchmarks")

LLaDA 使用了 Masked Diffusion Model (MDM)，该方法结合了离散随机掩蔽过程，并训练了一个掩码预测器来近似其反向过程。

# 2 Approach

![A Conceptual Overview of LLaDA](https://share.note.youdao.com/yws/api/personal/file/WEBe77426aa5b23c3364ad557f96d735ff7?method=download&shareKey=0293b80db53bfd7b8a9ba03f15a6f802 "A Conceptual Overview of LLaDA")

## 2.1 Probabilistic Formulation
与公式(2)中的自回归模型不同，LLaDA通过**前向过程 (forward process)** 和 **反向过程 (reverse process)** 来定义模型分布 $p_{\theta}(x_{0})$。

### Forward Process

逐步地、独立地 mask $x_{0}$ 中的 token，直到在 $t=1$ 时序列被完全 mask. 

给定 $x_{0}$ 时 $x_{t}$ 的条件分布可以被分解为：

$$
q_{t|0}(x_{t}|x_{0}) = \prod_{i=1}^{L} q_{t|0}(x_{t}^{i}|x_{0}^{i})
$$ 


对于 $t \in (0,1)$，序列 $x_{t}$ 是部分被掩码的，其中每个 token 有 $t$ 的概率被mask，或有 $1-t$ 的概率保持不变。

$$
q_{t|0}(x_{t}^{i}|x_{0}^{i}) = \begin{cases} 1-t, & x_{t}^{i} = x_{0}^{i} \\ t, & x_{t}^{i} = M \end{cases}
$$


其中 M 表示掩码 token. 直观上，每个 token 要么保持不变，要么被掩码，而被掩码的概率随着 t 从 0 到 1 线性增加。在 $t=1$ 时，所有 token 都被 mask.

## Reverse Process
反向过程则通过在 $t=1\rightarrow 0$ 从完全被掩码的序列中生成新数据。

对于 $0 \le s < t \le 1$，反向过程的条件分布分解为：

$$
q_{s|t}(x_{s}|x_{t}) = \prod_{i=1}^{L} q_{s|t}(x_{s}^{i}|x_{t})
$$

其中每个 token 的条件分布为：

$$
q_{s|t}(x_{s}^{i}|x_{t}^{i}) = \begin{cases} 1, & x_{t}^{i} \ne M, x_{s}^{i} = x_{t}^{i} \\ \frac{s}{t}, & x_{t}^{i} = M, x_{s}^{i} = M \\ \frac{t-s}{t}q_{0|t}(x_{s}^{i}|x_{t}), & x_{t}^{i} = M, x_{s}^{i} \ne M \\ 0, & \text{otherwise} \end{cases}
$$

需要估计的关键函数是条件分布 $q_{0|t}(x_{s}^{i}|x_{t})$，它在输入 $x_{t}$ 中对应位置被掩码的情况下，预测出原始的 token. 类似于连续扩散模型中的数据预测形式。如 (Ou et al., 2024) 所证明，可以推导出一个等价但无时间依赖的参数化形式

$$
q_{0|t}(x_s^i|x_t)=p_{\text{data}}(x_0^i|x_t^\text{UM}),\quad\forall i\text{ such that }x_t^i=\mathbf{M}
$$

其中 $x_{t}^{\text{UM}}$ 表示 $x_{t}$ 中未被掩码 token 的集合，它与原始数据 $x_{0}$ 中对应的 token 相同，因为未掩码的 token 仅由 $x_{0}$ 决定且与时间 t 无关 。直观上，这意味着估计数据预测函数等同于估计在干净数据上的条件分布，而后者是时不变的。因此，时间 t 不需要作为输入提供给参数化模型 。

尽管 MDM 的推导过程不简单，但其实现是直接的。我们首先引入**掩码预测器**，一个参数化模型 $p_{\theta}(\cdot|x_{t})$ (例如一个没有因果掩码的 Transformer)，它将任意 t 时刻的 $x_{t}$ 作为输入，并同时预测所有被 mask 的 token. 然后，我们如下定义模型分布 $p_{\theta}(x_{0})$：从一个被完全 mask 序列的 $x_{1}$ 开始，从 $t=1$ 到 0 模拟一个由 $p_{\theta}(\cdot|x_{t})$ 参数化的近似反向过程。在 $t=0$ 时刻推导出的边缘分布即代表了模型分布 $p_{\theta}(x_{0})$ 。

掩码预测器将 $x_{t}$ 作为输入并同时预测所有被掩码的 token (表示为 M). 它通过一个仅在被掩码 token 上计算的交叉熵损失进行训练：

$$
\mathcal{L}(\theta)\triangleq-\mathbb{E}_{t,x_{0},x_{t}}[\frac{1}{t}\sum_{i=1}^{L}I[x_{t}^{i}=M]log~p_{\theta}(x_{0}^{i}|x_{t})], \quad(3)
$$ 

其中，$x_{0}$ 从训练数据中采样，$t$ 从`[0, 1]`中均匀采样{{< sidenote >}} Notably, LLaDA employs a masking ratio that *varies randomly* between 0 and 1 while masked language models (Devlin, 2018) use a fixed ratio. {{< /sidenote >}}，$x_{t}$ 从前向过程中采样。指示函数 $I[\cdot]$ 确保损失仅针对被掩码的 token 计算。一旦训练完成，便可以模拟一个由该掩码预测器参数化的反向过程（详见2.4节），并将模型分布 $p_{\theta}(x_{0})$ 定义为该过程的边缘分布。

总的来说该方法通过在正向过程中逐步屏蔽令牌并在反向过程中学习恢复数据分布来训练生成模型，所有这些都在（近似）最大似然估计框架下。

## Pretraining

- LLaDA 8B 模型在一个包含 2.3T tokens 的高质量、多源数据集上从零开始进行预训练。该数据集覆盖了通用文本、代码、数学和多语言内容 。
- 训练总共消耗了 0.13M H800 GPU小 hours. 训练序列长度固定为4096. 其核心训练步骤是：对每个序列随机采样一个掩码率 t，并独立地以该概率掩码每个 token，然后让模型去预测被掩码的部分 。

- **架构调整** 相较于LLaMA3 8B，LLaDA 8B在架构上做了一些必要调整，如使用标准的 MHA 而非 GQA，并相应地调整了 FFN 的维度以保持模型总参数量相当 。
- **优化器与学习率** 训练使用了 AdamW 优化器和一个特殊的 Warmup-Stable-Decay 学习率调度策略。整个8B模型的训练实验只执行了一次，没有进行任何超参数调优。

| | Our ARM Baseline 1B | LLaDA IB | Our ARM Baseline 7B | LLaDA 8B | LLaMA3 8B |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Layers** | 22 | 22 | 28 | 32 | 32 |
| **Model dimension** | 2048 | 2048 | 4096 | 4096 | 4096 |
| **Attention heads** | 32 | 32 | 32 | 32 | 32 |
| **Vocabulary size** | 126,464 | 126,464 | 126,464 | 126.464 | 128,000 |
| **FFN dimension** | 5634 | 5634 | 13.440 | 12,288 | 14,336 |
| **Key/Value heads** | 4 | 4 | 8 | 32 | 8 |
| **Total parameters** | 1.49 B | 1.49 B | 6.83 B | 8.02 B | 8.03 B |
| **Non-embedding parameters** | 0.97 B | 0.97 B | 5.80 B | 6.98 B | 6.98 B |

## Supervised Fine-Tuning
 
我们通过使用配对数据 $(p_{0}, r_{0})$ 进行监督微调 (SFT)来增强LLaDA遵循指令的能力，其中 $p_{0}$ 是 prompt，$r_{0}$ 表示响应(response). 这是针对LLM最简单、最基础的 post-training 方法。从技术上讲，这要求模型对条件分布 $p_{\theta}(r_{0}|p_{0})$ 进行建模，而非预训练中的 $p_{\theta}(x_{0})$。

其实现方式与预训练类似。如图2(b)所示，保持 prompt 部分不变，并像处理 $x_{0}$ 一样，独立地 mask response 中的 token. 然后，将提示和被掩码的响应 $r_{t}$ 一同送入预训练好的掩码预测器，以计算用于 SFT 的损失

$$
-\mathbb{E}_{t,p_{0},r_{0},r_{t}}[\frac{1}{t}\sum_{i=1}^{L^{\prime}}I[r_{t}^{i}=M]log~p_{\theta}(r_{0}^{i}|p_{0},r_{t})] \quad (5)
$$

其中，$L^{\prime}$ 表示稍后指定的动态长度。这种方法与预训练是完全兼容的。本质上，将 $p_{0}$ 和 $r_{0}$ 拼接起来可以被视为干净的预训练数据 $x_{0} $，而将 $p_{0}$ 和 $r_{t}$ 拼接起来则可作为其被掩码后的版本 $x_{t}$. 这个过程与预训练完全相同，唯一的区别在于所有被掩码的 token 恰好都出现在 $r_{0}$ 部分。

LLaDA 8B 模型在一个包含 4.5M 对样本的数据集上进行了 SFT. 与预训练过程一致，数据准备和训练都遵循了现有LLM (Chu et al., 2024; Yang et al., 2024) 中使用的 SFT 协议，没有引入任何额外的技术来优化 LLaDA 的性能。该数据集涵盖了多个领域，包括代码、数学、指令遵循和结构化数据理解。我们在每个 mini-batch 中的短样本对末尾附加 EOS token，以确保所有数据长度相等。在训练期间将 EOS视为一个普通 token ，并在采样时将其移除，使得LLaDA能够自动控制响应的长度。

我们在SFT数据上训练了 3 个 epoch，其调度策略与预训练阶段相似。学习率在最初 50 次迭代中从 0 线性增加到 $2.5 \times 10^{-5}$，然后保持不变。在最后 10\% 的迭代中，学习率性降低到 $2.5 \times 10^{-6}$. 此外，我们将权重衰减设置为 0.1，全局 batch size 设置为 256，每个 GPU 的本地 batch size 设置为 2. SFT实验只执行了一次，没有进行任何超参数调优。

## Inference 

作为一个生成式模型，LLaDA既能 **采样 (sampling)** 新文本，也能 **评估 (evaluating)** 候选文本的似然。

我们先从采样说起。如图 2(c) 所示，给定一个 prompt $p_{0}$，我们通过离散化反向过程来从模型分布 $p_{\theta}(r_{0}|p_{0})$ 中进行采样，这个过程从一个被完全掩码的 response 开始。总的采样步数是一个超参数，为 LLaDA 提供了一个在效率和样本质量之间的权衡（详见3.3节分析）。我们默认使用均匀分布的时间步。此外，生成长度也被视为一个超参数，它指定了采样过程开始时完全被掩码句子的长度。如附录B.4所述，由于预训练和SFT都是在可变长度的数据集上进行的，最终结果对这个长度超参数不敏感。

在一个从时间 $t \in (0, 1]$ 到 $s \in [0, t)$的中间步骤中，我们将 $p_{0}$ 和 $r_{t}$ 同时送入掩码预测器，并一次性预测所有被掩码的 token. 随后 *remask* $\frac{s}{t}$ 比例的已预测 token. 得到$r_{s}$，从而确保反向过程的转换与前向过程保持一致，以实现准确采样。

受 LLM 采样中退火技巧的启发，我们探索了两种确定性但有效的重掩码策略。
- **low-confidence remasking**: remask 那些基于预测置信度最低的 $\frac{s}{t}$ 比例的 token. 
- **semi-autoregressive remasking**: 对于经过 SFT 的 LLaDA 模型，将序列分成几个块，并从左到右地生成. 在每个块内部，采用反向过程进行采样。

![A Conceptual Overview of the Semi-autoregressive Sampling](https://share.note.youdao.com/yws/api/personal/file/WEB13df3bff501e46425bb65c2defedecde?method=download&shareKey=838350c5b31c7e78112324263cdf5621 "A Conceptual Overview of the Semi-autoregressive Sampling")

对于条件似然评估，我们自然可以利用公式(5)中的上界。然而，我们发现下面这个等价形式（公式6）表现出更低的方差，在评估时更为稳定：

$$
-\mathbb{E}_{l,r_{0},r_{l}}[\frac{L}{l}\sum_{i=1}^{L}I[r_{l}^{i}=M]log~p_{\theta}(r_{0}^{i}|p_{0},r_{l})] \quad(6)
$$

其中，$l$ 从 ${1, 2, ..., L}$ 中均匀采样，$r_{l}$ 是通过从 $r_{0}$ 中不放回地均匀采样 $l$ 个没被 mask 的 token 得到的。此外，我们还采用了 unsupervised classifier-free guidance.

虽然这两个形式的期望值相同，但它们的方差不同。直观上，在公式 (5) 中，我们期望 $x_{t}=[p_0,r_t]$ 有 $t$ 比例的 token 被掩码。然而，前向过程的随机性常常会导致偏差，尤其当 $x_{t}$ 包含的 token 很少时。相比之下，在公式 (6) 中，$r_{l}$ 中被掩码 token 的比例 $\frac{l}{L}$ 是确定的。

虽然理论分析取决于数据分布，但经验结果表明，公式 (5) 需要超过 1000 次蒙特卡洛估计才能得到稳定结果，而公式 (6) 仅需 128 次估计即可达到稳定。

Any-order autoregressive models (AO-ARM)  通过对 L 个变量所有可能的排列顺序进行自回归来描述联合分布。为了学习这样的分布，AO-ARM 利用一个权重共享的神经网络来为所有单变量条件概率建模，并使用掩码 token 来表示缺失的变量。在训练期间，模型会最小化在所有顺序的均匀分布 $U_{\pi}$ 上的期望负对数似然：

$$
-\mathbb{E}_{x_{0},\pi \sim U_{\pi}}[\sum_{i=1}^{L}log~p_{\theta}(x_{0}^{\pi(i)}|x_{0}^{\pi(<i)}; \pi)]
$$ (15)

直观上，$x_{0}^{\pi(<i)}$ 可以被理解为一个被掩码的 token 序列 $x_{t}$，其中索引在 $\pi(\ge i)$ 的 token 被掩码 。可以进一步证明，公式 (15) 等价于公式 (12) 。这种联系解释了 LLaDA 的双向推理能力，即使它在推理过程中从未被显式使用 。

Nie et al. (2024) 引入了无监督的无分类器指导，这是一种即插即用的技术，可以平衡与提示的对齐度和文本多样性 。具体来说，无监督的无分类器指导在推理时采用以下修改过的掩码预测器 ：

$$
\tilde{p}_{\theta}(r_{0}|p_{0},r_{t}) \propto \frac{p_{\theta}(r_{0}|p_{0},r_{t})^{1+w}}{p_{\theta}(r_{0}|m,r_{t})^{w}}
$$ (16)

其中，$m$ 是一个与 $p_{0}$ 长度相同的掩码序列，$w$ 是一个控制 $p_{0}$ 强度的超参数 。我们在下游任务中采用了无监督的无分类器指导，详见附录 B.5 。


# 3 Experiment

实验主要围绕以下三个核心方面展开：

1. 可扩展性 (Scalability)：研究 LLaDA 的性能是否随着计算资源和模型规模的增加而稳定提升。通过与自建的自回归模型 (ARM) 基线在相同数据上进行对比，结果显示 LLaDA 表现出强大的可扩展性，其性能增长趋势与 ARM 相当，甚至在 MMLU 和 GSM8K 等任务上更具优势。


2. 基准测试结果 (Benchmark Results)：将 8B 规模的 LLaDA 与 LLaMA3 8B、LLaMA2 7B 等主流模型在涵盖通用，数学，代码和中文四大类的 15 个标准基准上进行对比。
    - 预训练模型：LLaDA 8B Base 模型的性能全面超越 LLaMA2 7B，并与 LLaMA3 8B 整体上具有竞争力，尤其在数学和中文任务上表现突出。

    - 微调模型：仅经过 SFT 的 LLaDA 8B Instruct 模型，在未进行强化学习对齐的情况下，其性能在多数任务上得到提升 ，并展现出令人印象深刻的 Instruction Follow 能力。


3. 反向推理 (Reversal Reasoning)：为了量化模型克服“反转诅咒”的能力，实验在一个中文古诗补全任务上进行了测试。结果表明，LLaDA 在正向和反向任务上表现均衡，一致性强，而 GPT-4o 等模型则在反向任务上表现出显著的性能下降。

# Reference 

[^1]: 简单来说就是拥有无限数据、一个足够大的网络和最优训练的理想条件下，模型有能力恢复出真实的数据分布。


[^2]: 在不更新其自身参数的情况下，仅通过在 Prompt 中提供少量示例 (few-shot) 或任务描述 (zero-shot)，就能当场学会并执行一个新任务的能力。