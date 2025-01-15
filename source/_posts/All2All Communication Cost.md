---
title: All2All Communication Cost
date: 2025/1/12 16:05:23
categories: Paper Reading
tags: blog
excerpt: Communication cost of All2All communication in different topologies.
mathjax: true
katex: true
---
在 All2All 通信中，每个设备给其他设备发送大小为 m 的不同的消息。此操作相当于使用一维数组分区对分布在 p 个进程中的二维数据数组进行转置，因此也被称作全交换 (**total exchange**)

## Ring / Bidirectional Linear Array

线性数组拓扑结构的 All2All 通信中，每个设备需要发送 p-1 份大小为 m 的数据。用 {i,j} 表示消息需要从设备 i 发送到设备 j. 首先，每个节点将所有要发送的数据作为一个大小为 m(p-1) 的合并消息发送给它邻居 (假设所有设备通信方向相同)。当邻居收到这个消息后提取他所需要的那一部分，发送剩下的大小为 m(p-2). 每个设备一共发送 p-1 次，每次要发送的消息大小减少 m.

由此可以得出在 p 个设备组成的线性数组拓扑上进行 All2All 每个设备需要向相邻设备通信 p-1 次，第 i 次通信的消息大小为 m(p-i). 如果向两个方向都进行发送，那么每个方向都只用发送原先一半的数据。

{% mathjax %}
\begin{aligned}T_{ring}&=\quad\sum_{i=1}^{p-1}(t_{s}+t_{w}m(p-i))\\&=\quad t_{s}(p-1)+\sum_{i=1}^{p-1}it_{w}m\\&=\quad(t_{s}+t_{w}mp/2)(p-1).\end{aligned}
{% endmathjax %}

## Mesh

若 p 个设备组成大小为 {% mathjax %}\sqrt{p} \times \sqrt{p}{% endmathjax %}  的 mesh 进行 All2All 通信，每个设备首先将其 p 个数据按照目的设备的列进行分组，即分成  {% mathjax %}\sqrt{p} {% endmathjax %} 组，每组包含大小为 {% mathjax %}m\sqrt{p} {% endmathjax %} 的消息。假设 3x3 的 mesh，则第一组消息的目的节点为 {0,3,6}，第二组消息的目的节点为 {1,4,7}，第三组消息的目的节点为 {2,5,8}

首先同时分别在每一行中进行 All2All 通信，每一份数据大小为 {% mathjax %}m\sqrt{p} {% endmathjax %}. 通信结束后每个设备拥有的是该行目的设备为所在列的所有数据。然后将数据按照目的设备所在的行进行分组。即设备 {0,3,6} 第一组消息的目的节点为 0，第二组消息的目的节点为 3，第三组消息的目的节点为 6. 然后同时分别在每一列中进行 All2All 通信。

我们只需要将 Linear Array 拓扑结构中的公式的 p 换成 {% mathjax %}\sqrt{p} {% endmathjax %}，m 换成 {% mathjax %}m\sqrt{p} {% endmathjax %}，再乘以 2 就得到在 mesh 上进行 All2All 的时间

{% mathjax %}
T_{mesh}=(2t_{s}+t_{w}mp)(\sqrt{p}-1).
{% endmathjax %}

## Hypercube

超立方体拓扑在每个维度上都有两个节点，一共有 {% mathjax %}\log{p} {% endmathjax %} 个维度。在一共有 p 个节点超立方体中，在某个维度 $d$ 上，超立方体可以被划分为两个 (n−1) 维的子立方体，这两个子立方体通过维度 d 上的 p/2 条链路相连。

在 All2All 通信的任何阶段，每个节点都持有 $p$ 个大小为 $m$ 的数据包。当在特定维度上通信时，每个节点发送 p/2 个数据包 (合并为一条消息)。这些数据包的目的地是由当前维度的链路连接的另一个子立方体包含的节点。在上述过程中，节点必须在每个 {% mathjax %}\log{p} {% endmathjax %} 通信步骤之前在本地重新排列消息。

在 {% mathjax %}\log{p} {% endmathjax %} 步中的每一步，每个设备沿当前维度的双向链路交换大小为 mp/2 的数据。因此在 hypercube 上进行 All2All 的时间为

{% mathjax %}
T_{hcube}=(t_{s}+t_{w}mp/2)\log p.
{% endmathjax %}

值得注意的是与 ring 和 mesh 算法不同，超立方体算法不是最优的。每个设备发送和接收大小为 m(p- 1) 的数据，超立方体上任意两个节点之间的平均距离为 {% mathjax %}\log{p}/2 {% endmathjax %}. 因此，网络上的总数据流量为 {% mathjax %}p\times m(p - 1)\times(\log{p})/2 {% endmathjax %}. 每个超立方体一共有 {% mathjax %}p\log{p}/2 {% endmathjax %} 条双向链路，如果流量能够被平分，则通信用时下界应该为

{% mathjax %}
\begin{aligned}T_{min}&=\frac{t_{w}pm(p-1)(\log p)/2}{(p\log p)/2}\\&=t_{w}m(p-1).\end{aligned}
{% endmathjax %}

## Optimal Algorithm in Hypercube

在超立方体上，执行 All2All 的最佳方法是让每一对节点彼此直接通信。因此，每个节点只需执行 p-1 次通信，每次与不同设备交换大小为 m 的数据。设备必须在每次通信中选择不会出现拥塞的通信对象。在第 j 次通信中，节点 i 与节点 {% mathjax %}i \oplus j{% endmathjax %} 交换数据。在超立方体上，从节点 i 到节点 j 的消息必须经过至少 l 条链路，其中 l 是 i 和 j 之间的汉明距离 (即 {% mathjax %}i \oplus j{% endmathjax %} 的二进制表示中的非零比特数). 我们通过 E-cube 路由来选择路径：

1. 将当前节点地址 C 与目标节点地址 D 进行 XOR 操作，得到 {% mathjax %}R=C\oplus D{% endmathjax %}.
2. 找到 R 的最低有效非零位，决定下一步跳转的维度。
3. 沿选定维度跳转到下一个节点，更新当前节点地址。
4. 重复上述步骤，直到 R=0， 即到达目标节点。
   对于节点i和节点j之间的消息传输，该算法保证每一步的通信时间为 t_s + t_wm，因为在节点 i 和节点 j 之间的链路上沿着同一方向传播的任何其他消息都不存在竞争，切每一步只切换一个维度，通信距离为 1. 整个 All2All 的总通信时间为

{% mathjax %}
T_{xor}=(t_{s}+t_{w}m)(p-1).
{% endmathjax %}
