---
title: Broadcasting on Meshes with Wormhole Routing
date: 2024/12/17 13:21:03
categories: Paper Reading
tags: blog
excerpt: Broadcasting on Meshes with Wormhole Routing
mathjax: true
katex: true
---
# Minimum Spanning Tree Broadcasting

最流行的广播算法是形成基于从发出广播的节点 (root) 的最小生成树。

## Naive Hypercube-Like Mesh Algorithm

假设 mesh 的行数目 (r) 和列数目 (c) 都是二的整数幂 (因此节点数 p 也是). 将节点的索引用二进制表示。最初，只有 root 拥有消息。对于 i=1,...,log(p)，在算法的第 i 步 ，所有已经拥有消息的节点 Pj 将消息发送给节点 Pk，其中 k 与 j 仅在节点索引的第 i 位上不同。记通信的启动时间为 α s，带宽为 1/β B/s，则总共需要 log(p) 步。如下图所示该算法的传播路径会造成链路冲突，因此总时间为

{% mathjax %}
\begin{aligned}
T_{\mathrm{naive}}&=\sum_{i=1}^{\log(r)}\left[\alpha+2^{(i-1)}n\beta\right]+\sum_{i=1}^{\log(c)}\left[\alpha+2^{(i-1)}n\beta\right]\\
                  &=\log(p)\alpha+(r+c-2)n\beta.\end{aligned}
{% endmathjax %}

![Hypercube MST Broadcast Viewing Nodes As a Linear Array](https://note.youdao.com/yws/api/personal/file/WEB1de53731c8226999768d1230138b798c?method=download&shareKey=7266f4c11fe6c1edee9b4d7f837f8e2a "Hypercube MST Broadcast Viewing Nodes As a Linear Array")

## Conflict-Free Hypercube-Like Mesh Algorithm

将上述算法的传播顺序换一下，改成 k 与 j 仅在第 (log(p)-i+1) 位上不同。如下图所示，在 mesh 网络中该算法相当于首先在根所在列内进行广播操作，然后是在每一行内单独广播。这些单独广播中的每一个节点都在一条线上，因此避免了链路冲突。这样总共只需要

{% mathjax %}
T_{\mathrm{conflict-free}}=\log(p)(\alpha+n\beta)
{% endmathjax %}

![Hypercube MST Broadcast Using Alternative Order of Steps](https://note.youdao.com/yws/api/personal/file/WEB79ff03165e0980859cc7bb3dbf3edc98?method=download&shareKey=2b90410cbe1aa34d2f876df796cf3dc4 "Hypercube MST Broadcast Using Alternative Order of Steps")

## MST Broadcast on Non-Power-of-Two Meshes

对于节点个数不是 2 的整数幂的 mesh，可以将节点视为线性数组。如下图所示，在每一步中，数组被尽可能平均分成两半，根节点向数组中不包含自己的那一半的对应位置节点发送消息。接下来，这两个节点都成为网络中两个递归广播的根节点。但因为数据广播在物理上会通过共享的通信链路， 在 mesh 上使用该算法可能会导致网络冲突。

### Separating Dimension

为了不产生链路冲突可以先在根节点所在的行内广播，然后在列内独立广播。然而，这样需要的时间会变为

{% mathjax %}
T_{\mathrm{separate-dim}}=\lceil\log(r)\rceil(\alpha+n\beta)+\lceil\log(c)\rceil(\alpha+n\beta)
{% endmathjax %}

<br>下面将研究可能产生网络冲突的通信类型，来设计更好的 MST 广播算法。

### Avoiding Network Conflicts on General Meshes

MST 广播中所有通信都位于 (逻辑线性数组的) 不相交的区域中，因此没有重叠。每一步中对于任意两对通信的节点 i,j 和 k,l，假设它们的索引关系为 {% mathjax %} i<j<k<l {% endmathjax %}，则仅有以下 4 种通信类型

{% mathjax %}
\begin{aligned}
&\langle i,j\rangle\mathrm{~and}\langle k,l\rangle&&\text{(Both to the right)}&&\text{(2)}
\\&\langle j,i\rangle\operatorname{and}\langle l,k\rangle&&\text{(Both the left)}&&\text{(3)}
\\&\langle i,j\rangle\operatorname{and}\langle l,k\rangle&&\text{(Towards each other)}&&\text{(4)}
\\&\langle j,i\rangle\operatorname{and}\langle k,l\rangle&&\text{(Away from each other)}&&\text{(5)}
\end{aligned}
{% endmathjax %}

<br>假设网络的路由算法为 x 方向优先的 XY-routing. 穷举所有情况可知逻辑路径 (2)-(4) 所使用的物理路径不冲突, (5) 则会发生冲突，一个例子如下图所示。

![Example of the Creation of Conflict Dependent on the Routing Algorithm](https://note.youdao.com/yws/api/personal/file/WEBea1d041ab8fce680334d27d4a3c1c5de?method=download&shareKey=df753b610d859b95f6a4af141cfee612 "Example of the Creation of Conflict Dependent on the Routing Algorithm")

### Recursive Splitting Broadcast

根据上面的结论可以修改对于非 2 整数幂节点数的 **Recrusive Splitting Broadcast (RSbcast)** 算法为：将节点视为线性数组。每一步中，数组尽可能平均分成两半，根节点向数组中不包含根的那一半的**某个**节点发送消息。接下来，这两个节点都成为两个单独部分内递归广播的根节点。节点的选择满足离根节点尽可能远 (在逻辑线性数组排布中) 的另一半节点中，并且该节点下一步生成方向朝着原始根节点的消息。
- 由 RSbcast 生成的所有消息都在逻辑阵列的不相交的分区内发送，或者在时间上不相交。因为同一分区内的节点只有在前一步向另一分区发送消息之后才会向分区内的节点发送消息。
- 任何来自原始根节点左边节点的消息 (L-node) 将向右发送，而来自原始根节点右边节点的消息 (R-node) 将向左移动。并且，从 L-node 或 R-node 发出的消息会分别被发送到 L-node 或 R-node. 根据之前的分析，并不会产生链路冲突。

| Case                                           | Satisfies Equation |
|------------------------------------------------|--------------------|
| both originate at L-nodes                      | (2)                |
| both originate at R-nodes                      | (3)                |
| one originates at L-node, one at R-node        | (4)                |
| one originates at L-node, one at root          | (2) or (4)         |
| one originates at R-node, one at root          | (3) or (4)         |

因此如果路由算法为 x 方向优先的 XY-routing，则 RSbcast 算法不会引起冲突，广播完成需要 {% mathjax %} \lceil\log(p)\rceil {% endmathjax %} 步，总用时为

{% mathjax %}
T_{\mathrm{RSbcast}}=\lceil\log(p)\rceil (\alpha+n\beta)
{% endmathjax %}

![Recusive Splitting  Broadcast (RSbcast) Viewing Nodes As a Linear Array](https://note.youdao.com/yws/api/personal/file/WEB8ab2561e152d6e589369136d76a20628?method=download&shareKey=d5ff810941063539d46c7ada7fb808ce "Recusive Splitting  Broadcast (RSbcast) Viewing Nodes As a Linear Array")

并且对于节点个数为 2 的整数幂的 MST 广播和 RSbcast，都不需要知道物理 mesh 的行数和列数。

# Pipelined Broadcast

对于通信量较少的情况延迟是主要因素，这意味着具有最少步骤的算法是最优的。然而，通信量较大时 RSbcast 的性能很差，因为整个消息要重传 {% mathjax %} \lceil\log(p)\rceil {% endmathjax %} 次。

对于 mesh 网络的流水线算法，为了避免过多的链路冲突，需要尽可能将通信限制在物理拓扑中最近的邻居。Disjoint-Edge Fence (DEF) 广播算法流程为：假设广播的根节点为 P0，节点以棋盘形式编码 (第一列黑，第二列白，以此类推). 黑色节点在偶数步向东发送，奇数步向南发送，而白色节点以相反的顺序发送。因此，数据沿着下图所示的两个边缘不相交的栅栏发送。根节点交替向东和向南发送消息，填充长度为 r+c 的两条流水线。假设消息被平均分为 k 份，总执行时间为

{% mathjax %}
T_{\mathrm{cdf}}=(k+r+c)\left(\alpha+\frac{n}{k}\beta\right)
{% endmathjax %}

<br>对 k 求导可得消息最佳分割的数目

{% mathjax %}
k_{\mathrm{opt}}=
\begin{cases}
1&\mathrm{if}\sqrt{(r+c)n\beta/\alpha}<1\\
n&\mathrm{if}\sqrt{(r+c)n\beta/\alpha}>n\\
\sqrt{(r+c)n\beta/\alpha}&\text{otherwise.}
\end{cases}
{% endmathjax %}

尽管流水线长度为 r+c-2，由于同一数据需要 2 步向两个不同方向发送，因此最终需要 r+c+k 步而不是 r+c+k-2 步。

# Scatter-Collect Broadcast

通过将要发送的消息先 scatter 到所有节点，然后将整个 gather 到每个节点，可以在 mesh 网络上进行有效的广播。

## 1-D Scatter–Collect

RSbcast 算法可以修改为首先每一步将发送的消息向量分成两半 (scatter)，然后所有节点在逻辑上形成一个环，每一份消息循环发送直到所有节点都拥有所有完整的消息 (collect). 算法如下图所示，以节点 0 作为根节点，需要经过 {% mathjax %} \lceil\log(p)\rceil {% endmathjax %} 步的 scatter ，以及 p-1 步的 collect. 在 collect 阶段节点会接收它们已经拥有的的片段使得通信存在冗余。假设节点数目 p 是 2 的 d 次幂，通信量 n 能被 p 整除，则广播花费时间为

{% mathjax %}
\begin{aligned}
T_{\mathrm{sb}1}&=\sum_{i=0}^{d-1}\left[\alpha+\frac n{2^{i+1}}\beta\right]+\sum_{i=1}^{p-1}\left[\alpha+\frac n{2^d}\beta\right]\\
&=(d+p-1)\alpha+2\frac{p-1}{p}n\beta\\
&=(p+\log_2(p)-1)\alpha+2\frac{p-1}pn\beta.
\end{aligned}
{% endmathjax %}

![Scatter–collect for p=4 with Source 0](https://note.youdao.com/yws/api/personal/file/WEB67826645a725de15b8ec7d2957be0f24?method=download&shareKey=96adf23eceb0f07bc82ab9070fe3b28e "Scatter–collect for p=4 with Source 0")

与 RSbcast 相比，需要额外的 p-1 次启动开销 ，但传输时间的系数从 {% mathjax %} \lceil\log(p)\rceil {% endmathjax %} 变成 (p-1)/p.

## 2-D Scatter–Collect

对于 2D mesh 网络可以使用 1D 版本的 scatter-collect，但是在 collect 阶段可以通过在每个维度单独执行来使得环的长度为 p-1. 算法流程如下
1. 在根节点所在列中 scatter. 原始消息被分割成 r 个份存储在该列的节点中。
2. 每一行独立执行一次 scatter. 根节点所在列中的每个节点都是其行的根节点。在这一阶段结束时，每一行拥有原始消息的被分成 c 份，存储在这一行的节点之间。这样原始消息在 mesh 中被分割成 rc=p 份。
3. 每一行独立地形成一个逻辑环进行 collect，直到每个节点拥有属于该行的原始消息部分。
4. 每一列独立地形成一个逻辑环进行 collect，直到每个节点都拥有整个原始消息。

这样 2-D Scatter–Collect 广播算法所花费时间为 

{% mathjax %}
\begin{aligned}
T_{\mathbf{sb}2}&=\sum_{i=0}^{d_{1}-1}\left[\alpha+\frac{n}{2^{i+1}}\beta\right]+\sum_{i=0}^{d_{2}-1}\left[\alpha+\frac{n}{2^{d_{1}}}\frac{1}{\beta}\right]
+\sum_{i=1}^{c-1}\left[\alpha+\frac{n}{2^{d_{1}}2^{d_{2}}}\beta\right]+\sum_{i=1}^{r-1}\left[\alpha+\frac{n}{2^{d_{1}}}\beta\right]\\
&=(d_{1}+d_{2}+r+c-2)\alpha+2\frac{p-1}{p}n\beta\\
&=(\log_{2}(p)+r+c-2)\alpha+2\frac{p-1}{p}n\beta
\end{aligned}
{% endmathjax %}
