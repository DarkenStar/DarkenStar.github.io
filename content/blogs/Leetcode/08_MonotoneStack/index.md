---
title: "08 Monotone Stack"
date: 2025-08-19T08:33:21+08:00
lastmod: 2025-08-19T08:33:21+08:00
author: ["WITHER"]

categories:
- Leetcode

tags:
- Monotone Stack

keywords:

description: "Algorithm questions about monotone stack." # 文章描述，与搜索优化相关
summary: "Algorithm questions about monotone stack." # 文章简单描述，会展示在主页
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

## 739.

这道题的本质是寻找每个元素右侧第一个比它大的元素。这类 "下一个更大元素" (Next Greater Element) 的问题，正是单调栈的经典应用场景。单调栈是一种特殊的栈，它在任何时候，栈内的元素都保持着单调递增或单调递减的顺序。当一个新元素需要入栈时，我们会通过弹出栈顶元素的方式来维护这个单调性。

对于这道题，我们需要找到的是每个温度值右边的第一个更高温度。我们可以维护一个单调递减的栈，栈中存储的是数组的下标。

从右向左遍历:
- 如果栈不为空，且栈顶元素天数对应的温度 <= 当前天数温度，并且由于我们是倒序遍历，它也不可能是当前天数右边的第一个更高温。所以，这个栈顶元素对于当前来说没有用了，直接弹出。重复这个过程，直到栈为空，或者栈顶元素对应的温度大于当前温度。

- 经过上一步的维护，如果此时栈不为空，那么栈顶元素就是右侧第一个比当前更高的温度的下标。我们就可以计算它们之间的天数差。如果栈为空，说明 右侧没有比它更热的天了，答案为 0.

- 处理完之后，将当前下标压入栈中。因为我们已经把所有比它矮（温度低）的元素都弹出了，所以 i入栈后，栈仍然保持单调递减。

```cpp
class Solution {
public:
    std::vector<int> dailyTemperatures(std::vector<int>& temperatures) {
        int n = temperatures.size();
        std::vector<int> answer(n, 0);
        std::stack<int> st; 

        for (int i = n - 1; i >= 0; --i) {
            while (!st.empty() && temperatures[i] >= temperatures[st.top()]) {
                st.pop();
            }

            if (!st.empty()) {
                answer[i] = st.top() - i;
            }

            st.push(i);
        }
        return answer;
    }
};
```

从左向右遍历:
- 如果栈不为空，且当前温度大于栈顶元素对应天数的温度，这说明我们找到了栈顶下标对应的答案。相减计算天数差后将栈顶元素弹出。重复这个过程，直到栈为空，或者当前温度不再大于栈顶元素对应的温度。
- 经过上一步处理后，我们将当前下标压入栈中。因为只有当当前天数温 <= 栈顶温度时，循环才会停止，所以入栈后，栈仍然能保持从栈底到栈顶的单调递减性。当前元素将会等待它右侧的更高温度出现。
- 遍历结束后，那些仍然留在栈中的下标，意味着它们右侧没有更高的温度，而结果数组初始化时就是 0，所以无需额外处理。

```cpp
class Solution {
public:
    std::vector<int> dailyTemperatures(std::vector<int>& temperatures) {
        int n = temperatures.size();
        std::vector<int> answer(n, 0); 
        std::stack<int> st; 

        for (int i = 0; i < n; ++i) {
            while (!st.empty() && temperatures[i] > temperatures[st.top()]) {
                int prev_index = st.top(); 
                answer[prev_index] = i - prev_index; 
                st.pop();
            }
            
            st.push(i);
        }
        return answer;
    }
};
```

- 时间复杂度: $O(n)$. 虽然有嵌套的 while 循环，但每个下标最多被压入栈一次，弹出栈一次。
- 空间复杂度: $O(n)$. 在最坏的情况下 (温度单调递减)，所有柱子的下标都会被压入栈中。

## 42. 

这道题的本质是对于每个位置，找到其左右两边比它高的柱子，然后计算这个位置能接的雨水。维护一个单调递减的栈，栈中存储的是柱子的下标，其对应的高度是严格递减的。

- 从左到右遍历每个柱子。当栈不为空，且当前柱子的高度大于栈顶下标对应的柱子的高度时，说明一个凹槽的右边界已经找到。
- 将栈顶元素 弹出，其下标记为 mid，height[mid] 就是凹槽的底部高度。此时，如果栈仍然不为空，新的栈顶元素就是凹槽的左边界，其下标记为 left. 凹槽的宽度 `w = i - left - 1`.
凹槽的高度取决于左右两个边界中较矮的那个。接水的高度是这个较矮边界的高度减去底部的高度，即 `h = min(height[i], height[left]) - height[mid]`. 将这部分雨水量 `w * h` 累加。
- 只要当前柱子仍然比新的栈顶元素高，就说明当前柱子可以作为更多凹槽的右边界，因此重复上述计算过程。
- 当 while 循环结束后 (即栈为空，或当前柱子不再高于栈顶柱子)，我们将当前柱子的下标压入栈中。

```cpp
class Solution {
public:
    int trap(std::vector<int>& height) {
        if (height.empty()) {
            return 0;
        }

        std::stack<int> st;
        int total_water = 0;

        for (int i = 0; i < height.size(); ++i) {
            while (!st.empty() && height[i] > height[st.top()]) {
                int mid_index = st.top();
                st.pop();

                if (st.empty()) {
                    break;
                }

                int left_index = st.top(); // 凹槽的左边界

                int width = i - left_index - 1;
                
                int bound_height = std::min(height[i], height[left_index]);
                int effective_height = bound_height - height[mid_index];

                total_water += width * effective_height;
            }

            st.push(i);
        }

        return total_water;
    }
};
```

## 496. 
从左往右遍历，只要遍历到比栈顶元素更大的数，就意味着栈顶元素找到了答案，记录答案，并弹出栈顶。我们只需把在 nums1 中的元素入栈这样空间复杂度就为 $O(m)$.

```cpp
class Solution {
public:
    vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
        int m = nums1.size();
        int n = nums2.size();
        vector<int> ans(m, -1);
        stack<int> st;
        unordered_map<int, int> map;
        for (int i = 0; i < nums1.size(); i++) {
            map[nums1[i]] = i;
        }

        for (int i = 0; i < n; i ++) {
            while(!st.empty() && nums2[i] > st.top()) {
                ans[map[st.top()]] = nums2[i];
                st.pop();
            }
            if (map.contains(nums2[i])) {
                st.push(nums2[i]);
            }
        }
        return ans;
    }
};
```

## 503

循环两遍数组，取模遍历，注意数组中元素只用压栈一次。

```cpp
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        int n = nums.size();
        vector<int> ans(n, -1);
        stack<int> st;
        for (int i = 0; i < 2*n; i++) {
            while (!st.empty() && nums[i%n] > nums[st.top()]) {
                ans[st.top()] = nums[i%n];
                st.pop();
            }
            if (i < n)
                st.push(i);
        }
        return ans;
    }
};
```

## 901.

维护一个价格单调递减的栈。栈中不仅要存价格，还要存该价格对应的跨度。查看栈顶的元素 {top_price, top_span}。

当前价格创建一个初始跨度 curSpan = 1. 如果栈顶价格 <= 当前价格，它的跨度可以被吸收合并到当前价格的跨度中。初始跨度加上加栈顶元素跨度后弹出栈顶元素。持续这个过程，直到栈为空，或者栈顶的价格比当前价格更高。

经过上一步的合并，curSpan 已经计算出了最终的结果。我们将当前的价格和它计算出的总跨度 {price, curSpan} 作为一个新的元素压入栈中，以供后续的价格使用。然后返回 curSpan.

```cpp
class StockSpanner {
public:
    stack<pair<int, int>> st;
    StockSpanner() {
        
    }
    
    int next(int price) {
        int curSpan = 1;
        while (!st.empty() && price >= st.top().first) {
            curSpan += st.top().second;
            st.pop();
        }
        st.push({price, curSpan});
        return curSpan;
    }
};
 ```