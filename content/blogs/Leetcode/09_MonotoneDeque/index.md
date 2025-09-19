---
title: "09 Monotone Deque"
date: 2025-08-21T09:09:14+08:00
lastmod: 2025-08-21T09:09:14+08:00
author: ["WITHER"]

categories:
- Leetcode

tags:
- Monotone Deque

keywords:

description: "Algorithm questions about monotone deque." # 文章描述，与搜索优化相关
summary: "Algorithm questions about monotone deque." # 文章简单描述，会展示在主页
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

# 239. 

通过模拟滑动窗口可以发现如果 `nums[i] <= nums[j], i < j`，那么只要 nums[j] 还在窗口内，nums[i] 就永远不可能成为这个窗口的最大值。因为 nums[j] 不仅比 nums[i] 大（或相等），而且比它更晚离开窗口。因此，在这种情况下，nums[i] 可以被安全地丢弃。

因此使用一个双端队列来存储数组元素的索引，并始终保持队列中索引对应的元素值是严格单调递减的。这样一来，队列的队首元素所对应的数组值，就永远是当前窗口的最大值。

为了维护这个单调递减的队列，我们需要遵循以下两个规则：

1. 移除出界元素：在窗口向右滑动时，首先要检查队首的索引是否已经超出了当前窗口的左边界。如果是，则说明队首元素已经过期，需要从队列中移除（pop_front）。因此**队列中要记录的是元素在数组中的下标。** 
2. 维持队列单调递减：当一个新的元素准备入队时，为了维持队列的单调性，我们需要从队尾（back） 开始，向前比较。如果队尾的元素小于或等于当前要入队的元素，那么队尾的元素就不可能成为未来任何窗口的最大值（因为当前元素更“新”也更“大”），所以应该将队尾元素出队（pop_back）.重复此过程，直到队列为空或者队尾元素大于当前元素，然后才将当前元素的索引入队（push_back）.

同时当窗口形成后 (即遍历到的元素数量达到 k) 才开始记录结果。

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ans;
        deque<int> dq;
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            while (!dq.empty() && nums[dq.back()] <= nums[i]) {
                dq.pop_back();
            }
            dq.push_back(i);
            // can use if because we judge with each i
            if (i - dq.front() + 1 > k) {  // excess the window
                dq.pop_front();
            }
            if (i >= k - 1) {
                ans.push_back(nums[dq.front()]);
            }
        }
        return ans;
    }
};
```