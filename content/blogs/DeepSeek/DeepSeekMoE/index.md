---
title: "DeepSeekMoE"
date: 2025-06-19T17:04:18+08:00
lastmod: 2025-06-19T17:04:18+08:00
author: ["WITHER"]

categories:
- Paper Reading

tags:
- DeepSeek

keywords:
- MLA

description: "Paper Reading of DeepSeekMoE" # 文章描述，与搜索优化相关
summary: "Paper Reading of DeepSeekMoE" # 文章简单描述，会展示在主页
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

# Preliminary: Mixture-of-Experts for Transformers

一个标准的 Transformer backbone LLM 由堆叠层标准 Transformer 块构成，每个块可以表示如下:

$$
\mathbf{h}_t^l = \sum_{i=1}^{N} \left( g_{i,t} \text{FFN}_i(\mathbf{u}_t^l) \right) + \mathbf{u}_t^l \tag{3}
$$

$$
g_{i,t} = \begin{cases} s_{i,t} & \text{if } s_{i,t} \in \text{Topk}(\{s_{j,t} | 1 \le j \le N\}, K) \\ 0 & \text{otherwise} \end{cases} \tag{4}
$$

$$
s_{i,t} = \text{Softmax}_i(\mathbf{u}_t^T \mathbf{e}_i^l) \tag{5}
$$

- $N$: 专家总数
- $\text{FFN}_i(\cdot)$: 第 $i$ 个专家的 FFN.
- $g_{i,t}$: 第 $i$ 个专家的门控值。
- $s_{i,t}$: token 对专家的亲和度。
- $\text{Topk}(\cdot, K)$: 在为第 $t$ 个 token 和所有 $N$ 个专家计算出的亲和度分数中，包含 $K$ 个最高分数的集合，
- $\mathbf{e}_i^l$: 第 $l$ 层中第 $i$ 个专家的中心。

注意到 $g_{i,t}$ 是稀疏的，说明在 $N$ 个门控值中只有 $K$ 个非零。这种稀疏性确保了 MoE 层内的计算效率，即每个 token 只会被分配给 $K$ 个专家并由它们计算。此外，在上述公式中，为了简洁起见，我们省略了层归一化操作。

# DeepSeekMoE Architecture

![Illustration of DeepSeek MoE](https://arxiv.org/html/2401.06066v1/x2.png "Illustration of DeepSeek MoE")


## Fine-Grained Expert Segmentation

在上图 (a) 的情况下将每个专家 FFN 的中间隐藏层维度缩小到原先的 1/m，专家数增加 m 倍。这样可以在参数量和计算量保持不变的情况下使得每个 token 被路由到更多的专家。通过细粒度的专家划分，MoE 层的输出可以表示为

$$
\begin{aligned}
\mathbf{h}_t^l &= \sum_{i=1}^{mN} \left( g_{i,t} \text{FFN}_i(\mathbf{u}_t^l) \right) + \mathbf{u}_t^l \quad&(6)\\
g_{i,t} &= \begin{cases} s_{i,t} & \text{if } s_{i,t} \in \text{Topk}(\{s_{j,t} | 1 \le j \le mN\}, mK) \\ 0 & \text{otherwise} \end{cases} \quad&(7)\\
s_{i,t} &= \text{Softmax}_i(\mathbf{u}_t^T \mathbf{e}_i^l) \quad&(8)
\end{aligned}
$$

- $mN$: 细粒度专家的总数。
- $mK$: 非零门控值的数量也将增加到。

专家参数总数等于 $N$ 乘以标准 FFN 中的参数数量。从组合可能性的角度来看，细粒度专家分割策略显著增强了激活专家的组合灵活性。

## Shared Expert Isolation

如上图 (c) 所示 将 $K_s$ 个专家隔离出来作为共享专家。无论路由模块如何，每个 token 都会被确定性地分配给这些共享专家。为了保持计算量不变，其他路由专家中激活的专家数量将减少 $K_s$.

$$
\begin{aligned}
\mathbf{h}_t^l &= \sum_{i=1}^{K_S} \text{FFN}_i(\mathbf{u}_t^l) + \sum_{i=K_S+1}^{mN} \left( g_{i,t} \text{FFN}_i(\mathbf{u}_t^l) \right) + \mathbf{u}_t^l \quad&(9)\\
g_{i,t} &= \begin{cases} s_{i,t} & \text{if } s_{i,t} \in \text{Topk}(\{s_{j,t} | K_S+1 \le j \le mN\}, mK - K_S) \\ 0 & \text{otherwise} \end{cases} \quad&(10)\\
s_{i,t} &= \text{Softmax}_i(\mathbf{u}_t^T \mathbf{e}_i^l) \quad&(11)
\end{aligned}
$$


于是在 DeepSeekMoE 中，共享专家的数量为 $K_S$，路由专家的总数为 $mN - K_S$，非零门控值的数量是 $mK - K_S$

## Load Balance Consideration

自动学习的路由策略可能会遇到负载不平衡的问题
1. 存在路由崩溃的风险，即模型总是只选择少数几个专家，导致其他专家无法得到充分训练。
2. 如果专家分布在多个设备上，负载不平衡会加剧计算瓶颈。
---
*Expert-Level Balance Loss.*
$$ 
\begin{aligned}
\mathcal{L}_{\text{ExpBal}} &= \alpha_1 \sum_{i=1}^{N'} f_i P_i \quad&(12)\\
f_i &= \frac{N'}{K'T} \sum_{t=1}^{T} \mathbf{1}(\text{Token } t \text{ selects Expert } i) \quad&(13)\\
P_i &= \frac{1}{T} \sum_{t=1}^{T} s_{i,t} \quad&(14)
\end{aligned}
$$

$\mathcal{L}_{\text{ExpBal}}$ 的目的是**促进专家之间的负载均衡**，避免出现某些专家过载 (被选中太多次) 而其他专家闲置 (很少被选中) 的情况。

* $N'$: 表示可路由的专家总数，即 $mN - K_S$。
* $K'$: 表示每个 token 选择的路由专家数量，即 $mK - K_S$。
* $T$: 表示总的 token 数量。

该损失函数的解释如下
* $f_i$ (Expert Load/Utilization):
    * 公式 (13) 计算的是专家 $i$ 在一个批次/序列中被选中的频率。
    * $\mathbf{1}(\text{Token } t \text{ selects Expert } i)$ 是一个指示函数，如果 token $t$ 选中了专家 $i$，则为 1，否则为 0。
    * $\frac{1}{T} \sum_{t=1}^{T} \mathbf{1}(\text{Token } t \text{ selects Expert } i)$ 得到了专家 $i$ 在 $T$ 个 token 中被选中的平均次数 (频率) 。
    * 前面的 $\frac{N'}{K'}$ 是一个归一化因子。当所有专家被均匀选中时，每个专家被选择的平均次数为$TK'/N'$，此时 $f_i$ 的期望值为 1. 如果专家 $i$ 被选中次数多于平均，则 $f_i > 1$；反之 $f_i < 1$.

$f_i$ 可以理解为**专家 $i$ 的归一化负载或利用率**。
* $P_i$ (Expert Routing Probability):
    * 公式 (14) 计算的是专家 $i$ 在所有 token 中平均的门控亲和度分数。
    * $s_{i,t}$ 是 token $t$ 对专家 $i$ 的原始亲和度分数 (Softmax 之前的输出) 。

$P_i$ 可以理解为**专家 $i$ 被门控网络选择的平均倾向性**。
* $\mathcal{L}_{\text{ExpBal}} = \alpha_1 \sum_{i=1}^{N'} f_i P_i$:
    * 这个损失项是 $f_i$ 和 $P_i$ 乘积的和。它的目标是**最小化这个和**。
    * 如果某个专家 $i$ 的 $f_i$ (负载高) 和 $P_i$ (被倾向性选择的概率高) 都很大，那么 $f_i P_i$ 就会很大，导致损失增大。
    * 通过最小化这个损失，模型会被**激励将 token 分配给那些负载较低或被选择倾向性较低的专家**。这有助于分散负载，使得所有专家都能得到训练和利用，从而提高模型的整体效率和性能。
    * $\alpha_1$ 是一个超参数，用于控制这个平衡损失在总损失中的权重。

---

*Device-Level Balance Loss.*
$$
\begin{aligned}
\mathcal{L}_{\text{DevBal}} &= \alpha_2 \sum_{i=1}^{D} \hat{f}_i \hat{P}_i \quad&(15)\\
\hat{f}_i &= \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i} f_j \quad&(16)\\
\hat{P}_i &= \sum_{j \in \mathcal{E}_i} P_j \quad&(17)
\end{aligned}
$$


 $\mathcal{L}_{\text{DevBal}}$ 的目的是**促进专家在不同计算设备之间的负载均衡**。在分布式训练中，通常会将专家分散到不同的设备上，如果某些设备上的专家过于繁忙，而另一些设备上的专家闲置，就会导致计算瓶颈和效率低下。

* $D$: 表示计算设备的数量。
* $\mathcal{E}_i$: 表示分配给第 $i$ 个设备的所有专家的集合。$|\mathcal{E}_i|$ 是这个集合中专家的数量。

该损失的解释如下

* $\hat{f}_i$ (Device-level Expert Load): 公式 (16) 计算的是第 $i$ 个设备上所有专家的平均负载 ($f_j$). 代表了**设备 $i$ 的总体计算负载**。
* $\hat{P}_i$ (Device-level Routing Probability): 公式 (17) 计算的是第 $i$ 个设备上所有专家的平均路由倾向性 ($P_j$) 的总和。代表了**设备 $i$ 的专家集合被门控网络选择的总体倾向性**。

* $\mathcal{L}_{\text{DevBal}} = \alpha_2 \sum_{i=1}^{D} \hat{f}_i \hat{P}_i$:
    * 这个损失项是 $\hat{f}_i$ 和 $\hat{P}_i$ 乘积的和，目标也是**最小化这个和**。
    * 如果某个设备 $i$ 的 $\hat{f}_i$ (负载高) 和 $\hat{P}_i$ (被选择倾向性高) 都很大，那么 $\hat{f}_i \hat{P}_i$ 就会很大，导致损失增大。
    * 通过最小化这个损失，模型会被**激励将 token 路由到那些整体负载较低的设备上的专家**。这确保了分布式训练或推理时，所有设备都能得到更均匀的利用，避免了单个设备成为瓶颈。
    * $\alpha_2$ 是一个超参数，用于控制这个损失在总损失中的权重。
---
超参数 $\alpha_1$ 和 $\alpha_2$ 的设置策略:
* **小的 $\alpha_1$ (专家级平衡因子) : 用于“减轻路由崩溃的风险”。路由崩溃指的是少数专家被过度使用，导致它们“饱和”而无法有效学习，同时其他专家则完全未被使用。较小的 $\alpha_1$ 意味着我们允许一定程度的专家专业化，但仍会进行微调以避免极端的不平衡。
* **大的 $\alpha_2$ (设备级平衡因子) : 用于“促进跨设备的平衡计算”。这意味着我们更强烈地要求模型将计算负载均匀地分散到所有可用的计算设备上，以最大限度地提高并行效率。设备级的负载均衡对于分布式系统而言至关重要。
 