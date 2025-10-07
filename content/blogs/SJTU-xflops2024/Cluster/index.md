---
title: "Cluster"
date: 2025-09-19T14:23:09+08:00
lastmod: 2025-09-19T14:23:09+08:00
author: ["WITHER"]

categories:
- HPC

tags:
- HPC

keywords:
- 

description: "Solution of SJTU-xflops2024 Cluster." # 文章描述，与搜索优化相关
summary: "Solution of SJTU-xflops2024 Cluster." # 文章简单描述，会展示在主页
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

# Preliminary: NOIP2012 国王的游戏

[题目网址](https://www.luogu.com.cn/problem/P1080). 我们需要安排 $n$ 位大臣的顺序，使得获得金币数最多的那个大臣的金币数尽可能小。每个大臣有左手数 $ a_i $ 和右手数 $ b_i $，国王有左手数 $ a_0 $ 和右手数 $ b_0 $。大臣 $ i $ 获得的奖赏是:

$$
r_i=\lfloor\frac{\prod_{j=0}^{k-1}a_j}{b_k}\rfloor 
$$

其中 
- $ k $ 是大臣 $ i $ 在队列中的位置 (国王固定在位置 0)，
- $\prod_{j=0}^{k-1} a_j$ 是从国王到大臣 $ i $ 前一位的所有人的左手数的乘积，
- $ b_k $ 是大臣 $ i $ 的右手数。

目标是找到一种排列，使得所有大臣的奖赏中的最大值最小，即:

$$\min_{\text{permutation}} \left( \max_{k=1}^n \left( \lfloor \frac{a_0 \cdot a_1 \cdot \ldots \cdot a_{k-1}}{b_k} \rfloor \right) \right)$$

我们需要找到排序的关键:

假设我们有两个大臣 $ i $ 和 $ j $，在队列中相邻，位置分别是 $ k $ 和 $ k+1 $. 他们的奖赏分别为: 
$$
\begin{aligned}
r_i &= \lfloor \frac{a_0 \cdot a_1 \cdot \ldots \cdot a_{k-1}}{b_i} \rfloor \\
r_j &= \lfloor \frac{a_0 \cdot a_1 \cdot \ldots \cdot a_{k-1} \cdot a_i}{b_j} \rfloor 
\end{aligned}
$$

果交换 $ i $ 和 $ j $，则奖赏变为: 
$$
\begin{aligned}
r_j' &= \lfloor \frac{a_0 \cdot a_1 \cdot \ldots \cdot a_{k-1}}{b_j} \rfloor \\
r_i' &= \lfloor \frac{a_0 \cdot a_1 \cdot \ldots \cdot a_{k-1} \cdot a_j}{b_i} \rfloor
\end{aligned}
$$

我们希望最大奖赏尽可能小，因此需要比较交换前后最大奖赏的变化。令前 $ k-1 $ 人的左手数乘积 $ p = a_0 \cdot a_1 \cdot \ldots \cdot a_{k-1} $，则:
- 原最大奖赏: $ \max\left( \lfloor \frac{p}{b_i} \rfloor, \lfloor \frac{p \cdot a_i}{b_j} \rfloor \right) $
- 交换后最大奖赏: $ \max\left( \lfloor \frac{p}{b_j} \rfloor, \lfloor \frac{p \cdot a_j}{b_i} \rfloor \right) $

若交换后最大值更小，则: 
$$\max\left( \lfloor \frac{p}{b_j} \rfloor, \lfloor \frac{p \cdot a_j}{b_i} \rfloor \right) < \max\left( \lfloor \frac{p}{b_i} \rfloor, \lfloor \frac{p \cdot a_i}{b_j} \rfloor \right)$$

先不考虑向下取整，显然 $\frac{p \cdot a_i}{b_j} > \frac{p}{b_j}, \frac{p \cdot a_j}{b_i} > \frac{p}{b_i}$. 由于我们希望交换后最大值较小，则: $\frac{a_j}{b_i} < \frac{a_i}{b_j} \rightarrow a_j \cdot b_j < a_i \cdot b_i$. 

因此，如果 $ a_i \cdot b_i < a_j \cdot b_j $，则将大臣 $ i $ 放在 $ j $ 前面可能减小最大奖赏。到此为止贪心排序的算法已经确定，由于数字较大需要使用高精度计算。

# Problem Analysis

题目要求我们安排 $ n $ 台服务器的叠放顺序，使得超算集群的坍塌风险最小。每个服务器有重量 $ w_i $ 和承重能力 $ c_i $。某台服务器的风险值定义为其上方所有服务器的重量之和减去其承重能力，即: 

$$r_i = \left( \sum_{j \text{ above } i } w_j \right) - c_i$$

设服务器按顺序从下到上为 $ s_1, s_2, \ldots, s_n $，则第 $ i $ 台服务器（位置  i \），从下到上）的风险值为: $ \text{r}_i = \left( \sum_{j=i+1}^n w_j \right) - c_i $. 其中 $\sum_{j=i+1}^n w_j$  是位于 $ i $ 上方的服务器的重量之和。目标是最小化所有服务器风险值的最大值，即:

$$\min_{\text{permutation}} \left( \max_{i=1}^n \left( \sum_{j=i+1}^n w_j - c_i \right) \right)$$

类似于国王的游戏问题，我们可以通过相邻交换分析来推导最优排序规则。假设有两台相邻服务器 $ s_i $ 和 $ s_{i+1} $，分别在位置 $ k $ 和 $ k+1 $. 设:
- $ s_i $ 的重量为 $ w_i $，承重能力为 $ c_i $. 
- 上方服务器 (位置 $ k+2 $ 到 $ n $) 的重量和为 $ S $. (若 $ k+1 = n $，则 $ S = 0 $).

交换前的风险值:
$$
\begin{aligned}
r_k &= (S + w_{i+1}) - c_i \\
r_{k+1} &= S - c_{i+1}
\end{aligned}
$$

交换后的风险值:
$$
\begin{aligned}
r_k' &= (S + w_{i}) - c_{i+1} \\
r_{k+1}' &= S - c_{i}
\end{aligned}
$$

显然有 $r_k > r_{k+1}', r_k' > r_{k+1}$ 若交换后最大风险不大于原最大风险, 则需要 $r_k' <= r_k \rightarrow w_i + c_i < w_{i+1} + c_{i+1}$. 即我们需要把自重+称重较小的集群放在上方。

1. 按 $ c_i + w_i $ 的升序从上到下排列服务器。
2. 初始化上方重量和 $ S = 0 $，最大风险值为负无穷.
3. 遍历服务器: 
    - 对于位置 $ i $  计算风险值 $r_i = S - c_i$.
    - 更新最大风险值。
    - 更新 $ S = S + w_i $

# High Precision

由于数据位数可能超过 `long long` 所能表示的范围，我们需要自定义一个 `BigInt` 结构体
处理高精度运算，它基于分段存储 (每段存储 4 位十进制数，基数为 10000) 来表示大整数，支持加法、减法、比较和符号处理。

## BigInt

成员变量: 
- `vector<int> digits`: 低位优先存储数字的“块”（从低位到高位）。每个 int 代表一个 4 位十进制数（0-9999），这样可以减少存储空间和运算次数（相当于每块处理 10^4 的范围）。
- `bool negative`: 符号标志，true 表示负数。零的符号始终为 false。

构造函数: 输入 `long long x`，取绝对值后反复 `% 10000` 和 `/ 10000` 分解成块，并且从低位开始填充，确保高位在前。如果 x`=0，digits=[0]`.

```cpp{linenos=true}
struct BigInt{
    std::vector<int> digits;  // 每位存储 4 位十进制数；
    bool negative;

    BigInt(long long x = 0) : negative(x < 0) {
        x = std::abs(x);
        if (x == 0) {
            digits.push_back(0);
        }
        while (x) {
            digits.push_back(x % 10000);
            x /= 10;
        }
    }
    // ... other func
};
```

## operator+

逻辑类似于小学手算竖式加法，但分块处理。

符号处理:
- 相同，直接相加绝对值，结果符号相同。
- 不同，转为减法: `a + b = a - (-b)`(异号相加等价于大数减小数).

绝对值相加: 
1. 初始化进位 `carry=0`.
2. 遍历最大长度: `sum = carry + (a.digits[i] if i < len(a) else 0) + (b.digits[i] if i < len(b) else 0)`。
3. 计算结果和进位: `res.digits[i] = sum % 10000，carry = sum / 10000`.
4. 如果 `carry` 仍有余，继续追加高位块。

```cpp{linenos=true}
BigInt operator+(const BigInt &other) const {
    if (negative != other.negative) {  // convert to minus
        return negative ? (other - (-*this)) : (*this - (-other));
    }

    BigInt res;
    res.negative = negative;
    res.digits.clear();
    long long carry = 0;
    size_t max_size = std::max(digits.size(), other.digits.size());
    for (size_t i = 0; i < max_size || carry; ++i) {
        long long sum = carry;
        if (i < digits.size()) {
            sum += digits[i];
        }
        if (i < other.digits.size()) {
            sum += other.digits.size();
        }
        res.digits.push_back(sum % 10000);
        carry = sum / 10000;
    }
    return res;
}
```

## operator-

减法更复杂，需要处理借位和符号。代码假设 `|this| >= |other|`，但通过符号递归处理一般情况。

1. 符号处理: 
- 不同，转为加法: `a - b = a + (-b)`.
- 相同，比较绝对值大小决定结果符号 (代码中简化，递归到加法或直接减).`

2. 绝对值相减逻辑 (假设 |a| >= |b|): 
    1. 初始化借位 `borrow=0`.
    2. 遍历 a 的长度: `diff = a.digits[i] - borrow - (b.digits[i] if i < len(b) else 0)`.
    3. 如果 `diff < 0` 说明需要借位: `diff += 10000，borrow=1`；否则 `borrow=0`.
    res.digits[i] = diff.

3. 清理前导零: `digits` 长度大于 1 并且最高位是 0 时持续弹出。完成后如果结果为 0 置 `negative = false`.

{{< notice note >}}
代码中减法假设 `this >= other` 的绝对值，如果不是，会通过反向减法再添加负号。
{{< /notice >}}

```cpp{linenos=true}
BigInt operator-(const BigInt &other) const {  // assume |this| > |other|
    if (negative != other.negative) {  // convert to add
        return negative ? -((-*this) + other) : *this + (-other);
    }

    if (abs_less(other)) {
        BigInt res = other - *this; // Compute -(other - this)
        return -res;
    }

    BigInt res;
    res.negative = negative;
    res.digits.clear();
    long long borrow = 0;
    int max_size = digits.size();
    for (size_t i = 0; i < max_size; ++i) {
        long long diff = digits[i] - borrow;
        if (i < other.digits.size()) {
            diff -= other.digits[i];
        }
        if (diff < 0) {
            diff += 10000;
            borrow = 1;
        } else {
            borrow = 0;
        }
        res.digits.push_back(diff);
    }
    while (res.digits.size() > 1 && res.digits.back() == 0) {  // remove leading zero
        res.digits.pop_back();
    }

    if (res.digits.size() == 1 && res.digits[0] == 0) {
        res.negative = false;
    }
    return res;
};
```

前缀 `operator-` 情况只需要复制 `digits` 并且翻转 `negative` (0 除外).

```cpp{linenos=true}
BigInt operator-() const {
    BigInt res = *this;
    if (res.digits.size() == 1 && res.digits[0] == 0) {
        res.negative = false;
    } else {
        res.negative = !negative;
    }
    return res;
}
```

## abs_less and operator<

`abs_less()` 比较两个数的绝对值先比较位数 (digits.size())，如果相同则从高位到低位逐块比大小。

```cpp{linenos=true}
bool abs_less(const BigInt &other) const {
    if (digits.size() != other.digits.size()) {
        return digits.size() < other.digits.size();
    }
    for (int i = digits.size() - 1; i >= 0; --i) {
        if (digits[i] != other.digits[i]) {
            return digits[i] < other.digits[i];
        }
    }
    return false;
} 
```

符号比较 (<): 
- 不同: 负 < 正。
- 同正: 用 `abs_less` 比绝对值。
- 同负: 负数中绝对值小的更大，所以反向用 `abs_less`.

```cpp{linenos=true}
bool operator<(const BigInt other)  const {
    if (negative != other.negative) {
        return negative && !other.negative;
    }
    if (!negative) {
        return abs_less(other);
    }
    return other.abs_less(*this);
}
```

## to_string

如果是负数，添加 `"-"`. 输出高位块 (digits.back() 无补零)，然后从次高到低位用 `%04d` 补零来确保每块 4 位。例如 `digits=[1110, 111, 1] → "1" + "0111" + "1110" = "11111110"`.

```cpp{linenos=true}
std::string to_string() const {
    std::string res = negative ? "-" : "";
    res += std::to_string(digits.back());
    for (int i = digits.size() - 2; i >= 0; --i) {
        char buf[5];
        snprintf(buf, 5, "%04d", digits[i]);
        res += buf;
    }
    return res;
}
```

# solve 

题目要求返回 `long long`，但由于使用了高精度，我们将结果转换为字符串后用 `std::stoll` 转回 `long long`.

```cpp{linenos=true}
long long solve(Server *servers, long long n){
    std::sort(servers, servers + n, [](const Server &a, const Server &b) {
        return (a.weight + a.capacity) < (b.weight + b.capacity);
    });

    BigInt sum(0);  // above weight sum
    BigInt max_risk(LLONG_MIN);  

    for (long long i = n - 1; i >= 0; i --) {  // top to down
        BigInt risk = sum - BigInt(servers[i].capacity);
        if (max_risk < risk) {
            max_risk = risk;
        }
        sum = sum + BigInt(servers[i].weight);
    }

    std::string risk_str = max_risk.to_string();
    return std::stoll(risk_str);
}
```