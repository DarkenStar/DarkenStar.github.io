---
title: "UnderstandingMetricsForLLMs"
date: 2025-10-23T12:47:03+08:00
lastmod: 2025-10-23T12:47:03+08:00
author: ["WITHER"]

categories:
- category 1
- category 2

tags:
- tag 1
- tag 2

keywords:
- word 1
- word 2

description: "" # 文章描述，与搜索优化相关
summary: "" # 文章简单描述，会展示在主页
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

传统上，语言模型的性能是通过困惑度 (Perplexity)、交叉熵 (Cross Entropy) 和每字符比特数（Bit Per Character）来衡量的。随着语言模型越来越多地被用作其他NLP任务的预训练模型，它们也经常根据它们在下游任务中的表现进行评估。GLUE基准分数是对语言模型进行更广泛的多任务评估的一个例子。

与直觉相反的是，更多的指标实际上会使比较语言模型变得更加困难，特别是当语言模型在特定下游任务上的表现通常不可靠时。

令人困惑的一点是，语言模型通常旨在将困惑都最小化，但由于我们无法获得零困惑，我们可以得到的困惑的下限是什么？如果我们不知道最优值，我们怎么知道我们的语言模型有多好？

此外，与准确度等指标不同，无论两个模型是如何训练的，在同一测试集上，90%的准确度肯定优于60%的准确度，除非我们知道文本是如何预处理的，词汇量大小，上下文长度等，否则认为模型的困惑程度小于另一个模型的困惑程度并不意味着什么。例如，虽然字符级语言模型的困惑度可能比另一个模型在单词级的困惑度小得多，但这并不意味着字符级语言模型比单词级语言模型好。因此，我们如何比较使用不同符号集的不同语言模型的性能呢？或者我们应该这样做吗？

# Understanding perplexity, bits-per-character, and cross entropy

大多数语言模型下一个符号概率估计为每个符号在给定其前面符号的概率的乘积：

$$
\begin{aligned}P_{(w_{1},w_{2},\ldots,w_{n})}&=p(w_{1})p(w_{2}|w_{1})p(w_{3}|w_{1},w_{2})...p(w_{n}|w_{1},w_{2},..,w_{n-1})\\&=\prod_{i=1}^np(w_i|w_1,...,w_{i-1})\end{aligned} \tag{1}
$$

另外，一些语言模型在给定相邻符号的情况下估计每个符号出现的概率，这也被称为完形填空任务 (cloze task).

## Entropy

任何语言的目的都是传达信息。为了测量一条信息中传递的平均信息量，我们使用了一个被称为“熵”的度量，这个度量是由克劳德·香农（Claude Shannon）提出的。应当注意的是，语言语境中的熵与热力学语境中的熵是相关的，但又不相同。

香农在其里程碑式的论文《Prediction and Entropy of Printed English》中对语言熵进行了直观的解释：

“熵是一个统计参数，从某种意义上说，它衡量了语言文本中每个字母平均产生多少信息。如果以最有效的方式将语言翻译成二进制数（0或1），熵就是原始语言中每个字母所需的二进制数的平均值。”

> "The entropy is a statistical parameter which measures, in a certain sense, how much information is produced on the average for each letter of a text in the language. If the language is translated into binary digits (0 or 1) in the most efficient way, the entropy is the average number of binary digits required per letter of the original language."

我们知道，对于 8 位 ASCII 码，每个字符由 8 位组成。然而，这并不是英语语言中表示字母的最有效的方法，因为所有字母都使用相同的位数来表示，而不管它们有多常见（更理想的方案是使用更少的位数来表示更常见的字母，即非均匀编码）. 因此，我们英语的字符级熵小于 8.

香农通过一个测量信息量的函数 $F_N$ 来近似任何语言的熵 $H$，或者换句话说，熵，扩展到文本的相邻 N 个字母。令 $b_n$ 表示一个连续的字母块 $(w_1, w_2, \cdots, w_n)$

$$
\begin{aligned}F_{N}&=-\sum_{b_n}p(b_n)\mathrm{log}_2p(w_n|b_{n-1})\\&=-\sum_{b_n}p(b_n)\mathrm{log}_2p(b_n)+\sum_{b_{n-1}} \sum_{w_n} p(b_{n-1}, w_n) \log_2 p(b_{n-1})
\\&=-\sum_{b_n}p(b_n)\mathrm{log}_2p(b_n)+\sum_{b_{n-1}}p(b_{n-1})\mathrm{log}_2p(b_{n-1}).\end{aligned} \tag{2}
$$

定义 $K_{N}=-\sum_{b_{n}}p(b_{n})\mathrm{log}_{2}p(b_{n})$, 我们有

$$
F_N=K_N-K_{N-1} \tag{3}
$$

香农定义语言熵为:

$$
H=\lim_{N\to\infty}F_N \tag{4}
$$

注意，根据这个定义，熵是使用无限数量的符号来计算的。在实践中，我们只能从有限的文本样本近似经验熵。

## Cross Entropy

由于语言 L 中缺乏无限量的文本，因此语言的真实分布是未知的。语言模型旨在从样本文本中学习一个分布 Q，用于接近于语言经验分布 P. 为了测量两个分布的“接近度”，经常使用交叉熵。在数学上。Q 相对于 P 的交叉熵定义如下：

$$
\mathrm{H(P,~Q)=E_P[-log}Q]
$$

当 P 和 Q 是离散分布时有:

$$
\begin{aligned}\mathrm{H(P,Q)}&=-\sum_xP(x)\mathrm{log}Q(x)\\&=-\sum_xP(x)[\mathrm{log}P(x)+\mathrm{log}Q(x)-\mathrm{log}P(x)]\\&=-\sum_xP(x)[\mathrm{log}P(x)+\mathrm{log}\frac{Q(x)}{P(x)}]\\&=-\sum_xP(x)\mathrm{log}P(x)-\sum_xP(x)\mathrm{log}\frac{Q(x)}{P(x)}\\&=H(P)+D_{KL}(P||Q)\end{aligned}
$$

其中 $H(P)+D_{KL}(P||Q)$ 是 Q 和 P 的 Kullback–Leibler (KL) 散度，也被称为 Q 关于 P 的相对熵。因此，Q 相对于 P 的交叉熵是以下两个值的和：

1. 对P的任何可能结果进行编码所需的平均比特数 [即 H(P) - P的熵].
2. 使用针对Q优化的代码编码P的任何可能结果所需的额外比特数。

需要注意的是，由于经验熵 H(P) 是不可优化的，当我们以最小化交叉熵损失为目标训练语言模型时，真正的目标是最小化分布的 KL 散度，这是我们的语言模型从语言的经验分布中学到的。

## Perplexity

维基百科将困惑定义为：

衡量概率分布或概率模型对样本的预测程度。
> "a measurement of how well a probability distribution or probability model predicts a sample."

直觉上，困惑可以被理解为一种不确定性的度量。语言模型的困惑程度可以看作是预测下一个符号时的困惑程度。考虑一个熵为 3 比特的语言模型，其中每个比特编码两种概率相等的可能结果。这意味着在预测下一个符号时，该语言模型必须在 8 个可能的选项中做出选择。因此，我们可以认为该语言模型的困惑度为 8.

数学上，语言模型的困惑度定义为：

$$
\mathrm{PPL}(P,Q)=2^{{\mathrm{H}(P,Q)}}
$$

## Bits-per-character and bits-per-word

每字符位数（BPC）是另一个经常被报道用于最新语言模型的度量。它精确地测量了其命名的数量：编码一个字符所需的平均位数。根据定义，熵就是BPC的平均数。一些语言模型报告交叉熵损失和BPC的原因纯粹是技术上的。

虽然熵和交叉熵是使用log以2为基数（以 bit 为单位）来定义的，但流行的机器学习框架，包括TensorFlow和PyTorch，使用自然对数（单位是 nat）来实现交叉熵损失。这是因为计算自然对数比计算以 2 为底的对数要快。

请记住，BPC 特定于字符级语言模型。当我们有单词级别的语言模型时，这个数量被称为每单词比特数（BPW）——编码一个单词所需的平均比特数。后续将进一步讨论BPC和BPW之间的关系。

# Reasoning about entropy as a metric

因为我们可以从困惑度计算出交叉熵，反之亦然，从本节开始，我们将只研究交叉熵。

对于用于机器学习模型的许多指标，我们通常知道它们的界限。例如，准确度的最佳值是100%，而单词错误率和均方误差的最佳值是0。如果我们的模型达到99.9999%的准确率，我们就可以肯定地知道，我们的模型非常接近于它可能做到的最好。

当一个语言模型的交叉熵损失为7时，如果我们不知道最好的结果应该是什么，我们就不知道它离最好的结果有多远。不知道我们的目标是什么会使我们在决定投入多少资源来改进模型方面面临挑战。

## Mathematical bounds

首先一个分布的熵最小值为 0 (即语言只有一个字符). 其次，我们知道概率分布的熵在均匀分布时达到最大值。这暗示了这样一个事实，即对于所有共享同一组符号（词汇表）的语言，具有最大熵的语言是所有符号以相同概率出现的那个。

令 $|V|$ 为某种语言分布 P 的词汇表大小。

$$
\mathrm{H}(\mathrm{P})\leq-|\mathrm{V}|*\frac{1}{|\mathrm{V}|}*\log(\frac{1}{|\mathrm{V}|})=-\mathrm{log}(\frac{1}{|\mathrm{V}|})=\log(|\mathrm{V}|) \tag{6}
$$

如果我们认为英语是一种有27个符号（英文字母加上空格）的语言，那么它的字符级熵最多为 $\log(27)=4.7549$.

一个20岁的美国人平均认识42000个单词，所以他们的单词级熵最多为 $\log(42,000)=15.3581$.

第三，因为语言模型的交叉熵损失将至少是语言模型所训练的文本的经验熵。如果底层语言的经验熵为 7，则交叉熵损失将至少为 7.

最后，根据香农的定义，熵是当 N 趋于无穷时的 $F_N$. 下面将证明，随着 N 的增加，$F_N$ 减小。直观上的解释为前一个序列越长，模型在预测下一个符号时就越不容易混淆。

我们将通过证明对于所有对所有 $N \ge 1$ 都有 $F_N \ge F_{N+1}$ 来证实这一点。假设每个字符来自一个包含m个字母的词汇表 $x_1, x_2, \cdots, x_m$. $F_N$ 表示的是在 N 个相邻的文本字母的统计数据而产生的信息量或熵。

$$
\begin{aligned}F_N-F_{N+1}&=-\sum_{b_n}p(b_n)\log p(w_n|b_{n-1})+\sum_{b_{n+1}}p(b_{n+1})\log p(w_{n+1}|b_n)\\&=\sum_{b_{n-1}}[\sum_{w_n,w_{n+1}}p(b_{n+1})\mathrm{log}p(w_{n+1}|b_n)-\sum_{w_n}p(b_n)\mathrm{log}p(w_n|b_{n-1})]\\&\geq\sum_{b_{n-1}}[\sum_{w_n,w_{n+1}}p(b_{n-1})\mathrm{log}p(w_n|b_{n-1})-\sum_{w_n}p(b_{n-1},w_n)\mathrm{log}p(w_n|b_{n-1})]\\&=\sum_{b_{n-1}}[\sum_{w_{n+1}}\log p(w_{n+1}|b_{n-1})\log p(w_{n-1}|b_{n-1})-\sum_{w_n}p(b_{n-1},w_n,w_{n+1})-\sum_{w_n}p(b_{n-1},w_n)\mathrm{log}p(w_n|b_{n-1})]\\&=\sum_{b_{n-1}}[\sum_{w_{n-1}}\log p(w_{n+1}|b_{n-1})p(b_{n-1},w_n)\mathrm{log}p(w_n|b_{n-1})]\\&=0\end{aligned}
$$

第二个等号解释: 由于 $b_n = b_{n-1} \cdot w_n$，求和 $\sum_{b_n}$ 等价于 $\sum_{b_{n-1}} \sum_{w_n}$（对所有可能的块前缀 $b_{n-1}$ 和接在其后的符号 $w_n$ 求和）


这意味着对于无限数量的文本，使用较长上下文长度的语言模型通常应该比使用较短上下文长度的语言模型具有更低的交叉熵值。例如，使用上下文长度为32的语言模型应该比使用上下文长度为24的语言模型具有更低的交叉熵。对于有限数量的文本，这可能会很复杂，因为语言模型可能无法看到足够长的序列来进行有意义的预测。

## Comparing perplexities across language models

到目前为止，我们只在字符层面探索了熵。然而，也有 word-level 和 subword-level 语言模型，这导致我们思考周围的问题。是否有可能比较不同符号类型的语言模型的熵？换句话说，我们能否将字符级熵转换为词级熵，反之亦然？

格雷夫斯使用了这个简单的公式：如果一个单词平均需要 m 个比特来编码，而一个单词包含 l 个字符，那么编码一个字符平均需要 m/l 比特。在他的论文《Generating Sequences with Recurrent Neural Networks》中，由于数据集中一个单词平均有5.6个字符，因此使用以下方法计算单词级困惑度：$2^{5.6BPC}$

香农使用了类似的推理。通过Zipf定律，即“任何单词的频率与其在频率表中的排名成反比”，Shannon近似得出了英语中单词的频率，并估计单词水平 $F_1$ 为 11.82. 估计英语单词的平均长度为4.5，因此倾向于将 $F_4$ 和 $F_5$ 设置为 11.82/4.5=2.62.

不直接使用 q-LM 的原始输出 Logits 进行验证，而是增加一个轻量级的“修正网络”或“修正参数”，在验证前对 q-LM 的输出进行校准，使其更接近 fp-LM 的行为。