---
date: '2025-07-23T09:55:34+08:00'
title: '02 Sliding Window'
author: ["WITHER"]

categories:
- Leetcode

tags:
- sliding window

keywords:
- sliding window

description: "Algorithm questions about sliding window." # 文章描述，与搜索优化相关
summary: "Algorithm questions about sliding window." # 文章简单描述，会展示在主页
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

# 209. Minimum Size Subarray Sum

[题目网址](https://leetcode.cn/problems/minimum-size-subarray-sum/description/). 这题是滑动窗口的典型例题，我们维护一个 `left` 和 `right` 指针构成的一个窗口，对子数组的右端点不断右移进行枚举。每进行一次右移 (窗口扩大之后)，我们判断将左端点进行右移之后的子数组之和是否 `>= target`，如果是则更新答案，反之则继续扩大窗口直到满足条件为止。

这里的 `while` 循环中不需要判断 `left <= right` 的条件，当两个指针指向同一位置时，把 `sum - nums[left]` 结果为 0，而 `target` 是一个正整数，条件自动不满足，这也是枚举右端点的一个好处。虽然此处是两个循环，但是直到外层 `for` 循环结束，内循环的 `left` 最多移动了 n 次，因此时间复杂度为 $O(n)$. 由于只用到了几个变量，空间复杂度为 $O(1)$.

使用滑动窗口需要问题具有单调性，本题中窗口扩大肯定子数组之和越来越大，反之和越来越小。滑动窗口的核心要点为
1. 维护一个有条件的滑动窗口；
2. 右端点右移，导致窗口扩大，是不满足条件的罪魁祸首；
3. 左端点右移目的是为了缩小窗口，重新满足条件。

```cpp{linenos=true}
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int ans = nums.size() + 1;  // subarray length can't exceed len+1
        int left = 0, sum = 0;
        for (int right = 0; right < nums.size(); right++) { // right pointer move right to unsatisfy the condition
            sum += nums[right];
            while (sum >= target) {  // inner loop max interation is n during the outloop, so the time complexity is O(n)
                ans = min(ans, right - left + 1);
                sum -= nums[left];
                left++;
            }
        }
        return ans < nums.size() + 1 ? ans : 0;
    }
};
```

# Subarray Product Less Than k

[题目网址](https://leetcode.cn/problems/subarray-product-less-than-k/description/). 这题思路与上一题一样，但需要注意的是需要返回的是元素的乘积严格小于 k 的连续子数组的数目。当窗口 `[left, right]` 对应的子数组满足要求时，`[left + 1, right], [left + 2, right], ..., [right, right]` 全都满足要求，因此答案个数加上 `right - left + 1`.

```cpp{linenos=true}
class Solution {
public:
    int numSubarrayProductLessThanK(vector<int>& nums, int k) {
        int ans = 0;
        int left = 0, product = 1;
        if (k <= 1)
            return 0;
        for (int right = 0; right < nums.size(); right++) {
            product *= nums[right];
            while (product >= k) {
                product /= nums[left];
                left++;
            }
            ans += right - left + 1;
        }
        return ans;
    }
};
```