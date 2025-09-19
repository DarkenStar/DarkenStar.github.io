---
title: "03 BinarySearch"
date: 2025-07-24T09:59:53+08:00
lastmod: 2025-07-24T09:59:53+08:00
author: ["WITHER"]

categories:
- Leetcode

tags:
- Binary Search

keywords:
- 
- 

description: "Algorithm questions about binary search." # 文章描述，与搜索优化相关
summary: "Algorithm questions about binary search." # 文章简单描述，会展示在主页
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
# Preliminary: Algorithms for Binary Search in Different Intervals

对于在一个有序数组里返回第一个 `>= target` 元素的位置的问题，如果用暴力搜索法遍历数组，则时间复杂度为 $O(n)$，这是因为没有利用数组有序的性质。

- 对于闭区间写法我们初始化一个左指针 left 指向数组的第一个元素，右指针 right 指向数组中最后一个元素。要保证的是 left 左侧的元素全部都小于 target, right 右侧的元素全部都大于等于 target，这也被称为**循环不变量**，因此更新的时候我们要令 `left = mid + 1` 或者 `right = mid - 1`. 当 `left <= right` 时，区间内都有元素。循环结束后，有 `L = R + 1`. *为了防止溢出，计算中点的时候要写成* `mid = left + (right - left) / 2`.

- 左闭右开区间的写法，需要把 right 初始化为数组长度，`right = mid`，因为此时区间不包括 right，区间不为空的情况为 `left < right`. 循环结束后 left 和 right 指向的时同一个位置，返回谁都可以。对于左开右闭区间也是同理。

- 是左开右开区间的写法，需要额外把 left 初始化为 -1. 区间不为空的条件为 `left + 1 < right`，更新的是时候需要也使得 `left = mid`. 循环结束后返回 right.

再考虑额外的三种情况，如果数组元素都是整数:
- `> target` 可以看作 `>= target + 1`
- `< target` 可以看作 `>= target` 左边的那个位置。
- `<= target` 可以看作 `>= target + 1` 左边的那个位置。

# Question Type 1: Maintain Properties Outside the Interval

## 34. Find First and Last Position of Element in Sorted Array

[题目网址](https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/). 我们先求出第一个 `>= target` 元素的位置，记作 start. 如果 `start == nums.size()` 则说明数组所有元素都 `< target`. 如果 `nums[start] != target` 就说明数组中不存在这个元素。

对于 target 的末尾位置则可以看作 `>= target + 1` 起始位置的左边一个元素，由于 start 存在那么 end 肯定存在 (最差 `= start`).

```cpp{linenos=true}
class Solution {
    // return the index of a sorted array where its ele >= target
    int lowerBoundIndex(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {  // loop until the interval is empty, i.e. right + 1 = left
            int mid = left + (right - left) / 2;  // avoid overflow
            if (nums[mid] < target) {  // shows left ele of the interval all < target
                left = mid + 1;  // [mid + 1, right]
            } else {  // shows right ele of the interval all >= target
                right = mid - 1;  // [left, mid - 1]
            }
        }
        return left;
    }
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int start = lowerBoundIndex(nums, target);
        // all ele < target || all ele > target
        if (start == nums.size() || nums[start] != target)
            return {-1, -1};
        int end = lowerBoundIndex(nums, target + 1) - 1;
        return {start, end};
    }
};
```
    
## 2529. Maximum Count of Positive Integer and Negative Integer

[题目网址](https://leetcode.cn/problems/maximum-count-of-positive-integer-and-negative-integer/). 思路和上题一样，负数个数即为数组第一个 `>= 0` 元素的索引。正数个数即为数组长度减去第一个 `>=1` 的元素的索引。`lowerBoundIndex` 函数一样，不再赘述。

```cpp{linenos=true}
class Solution {
public:
    int maximumCount(vector<int>& nums) {
        // Find first index where nums[i] >= 0 (count of negatives)
        int negCount = lowerBoundIndex(nums, 0);
        // Find first index where nums[i] >= 1 (start of positives)
        int startOfPos = lowerBoundIndex(nums, 1);
        // Positive count is total length minus index of first >= 1
        int posCount = nums.size() - startOfPos;
        return max(negCount, posCount);
    }
};
```

## 2300. Successful Pairs of Spells and Potions

[题目网址](https://leetcode.cn/problems/successful-pairs-of-spells-and-potions/). 对 potions 进行排序，遍历 spells，对于每个咒语找到满足 sucess 要求的最小药水的能量强度对应的索引，`potions.size()` 减去该索引则为能成功配对的数目。

```{linenos=true}
class Solution {
public:
    vector<int> successfulPairs(vector<int>& spells, vector<int>& potions, long long success) {
        sort(potions.begin(), potions.end());
        vector<int> ans;
        for (int spell : spells) {
            long long target = ceil((double)success / spell);
            ans.push_back((int) potions.size() - lowerBoundIndex(potions, target));
        }
        return ans;
    }
};
```

## 2563. Count the Number of Fair Pairs

[题目网址](https://leetcode.cn/problems/count-the-number-of-fair-pairs/). 对 nums 进行排序后从左往右遍历，对于每个元素在其右侧数组中查找第一个 `>= upper + 1` 的索引和第一个 `>= lower` 的索引，两者相减则是该元素的公平数对数目。

```cpp{linenos=true}
class Solution {
    int lowerBoundIndex(vector<int>& nums, int target, int start, int end) {
        if (start < 0 || end > nums.size() || start > end) {
            return -1;
        }
        int left = start, right = end - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }
public:
    long long countFairPairs(vector<int>& nums, int lower, int upper) {
        if (nums.size() == 1)
            return 0;
        long long ans = 0, n = nums.size();
        sort(nums.begin(), nums.end());大于
        for (int i = 0; i < nums.size(); i++) {
            int left = lowerBoundIndex(nums, lower - nums[i], i+1, n);
            int right = lowerBoundIndex(nums, upper + 1 - nums[i], i+1, n);
            ans += right - left;
        }
        return ans;
    }   
}; 
```

## 275. H Index II
 
[题目网址](https://leetcode.cn/problems/h-index-ii/). 我们需要找到一个分界点 mid，使得从 mid 到末尾的论文数量 `n - mid` 至少为 `citations[mid]`. 同时，我们希望 `n - mid` 尽可能小。换句话说，我们需要找到最小的 mid 使得 `citations[mid] >= n - mid`，这样 h = n - mid 就是满足条件的最大值。

根据闭区间的写法，left 左侧的引用次数都满足 H index 的定义，即我们需要判断是否有 `n - mid` 论文严格大于 `citation[mid]`. 当循环结束时，left 会指向第一个满足 `citations[mid] >= n - mid` 的位置。

```cpp{linenos=true}
class Solution {
public:
    int hIndex(vector<int>& citations) {
        int left = 0, right = citations.size() - 1, n = citations.size();
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (n - mid > citations[mid]) {  // try bigger h index, which means smaller mid
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return n - left;
    }
};
```

## 875. Koko Eating Bananas

[题目网址](https://leetcode.cn/problems/koko-eating-bananas/). 这里题目意思有点难以理解，总之来说就是吃掉香蕉花费时间是向上取整的。当吃香蕉的时间小于等于警卫回来的速度的时候我们记录下答案并尝试更慢的速度。

```cpp{linenos=true}
class Solution {
public:
    int minEatingSpeed(vector<int>& piles, int h) {
        int left = 1, right = *max_element(piles.begin(), piles.end()), ans = 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;  // eating speed = piles[mid]
            long long time = 0;
            for (int pile : piles) {
                time += ceil((double) pile / mid);  // convert to double to avoid truncation
            }
            if (time <= h) {  // try slower speed
                ans = mid;
                right = mid - 1;
            } else {  // too slow to eat all piles
                left = mid + 1;
            }
        }
        return ans;
    }
};
```

## 2187. Minimum Time to Complete Trips

[题目网址](https://leetcode.cn/problems/minimum-time-to-complete-trips/). 一样的思路，判断 `mid` 时间内能完成的趟数是否满足要求，若满足则尝试更短的时间。

```cpp{linenos=true}
class Solution {
public:
    long long minimumTime(vector<int>& time, int totalTrips) {
        long long left = 1; 
        long long right = ranges::min(time) * 1LL * totalTrips;
        long long ans = 1;
        while (left <= right) {
            long long mid = left + (right - left) / 2;
            long long curTrips = 0;
            for (int t : time) {
                curTrips += mid / t;
            }
            if (curTrips < totalTrips) {  // not enough
                left = mid + 1;
            } else {
                ans = mid;
                right = mid - 1;
            }
        }
        return ans;
    }
};
```

## 2861. Maximum Number of Alloys

[题目网址](https://leetcode.cn/problems/maximum-number-of-alloys/). 
1. 遍历每一种合金配方，计算一次它最多能造多少件。
2. 对每种合金用二分查找来寻找能制造的最大数量，不断更新最大值。
    - 每一种合金最大能制造的数量可以初始化为库存最少的金属数量加上 budget (即假设购买金属花费为 1). 制造 `mid` 份需要的总花费我们先需要遍历每种金属需要的数量和对应库存的大小，以判断需不需要额外购买。

```cpp{linenos=true}
class Solution {
public:
    int maxNumberOfAlloys(int n, int k, int budget, vector<vector<int>>& composition, vector<int>& stock, vector<int>& cost) {
        long long ans = 0;
        // Perform binary search on each alloy
        for (auto comp : composition) {
            long long left = 0, right = ranges::min(stock) + budget;
            long long maxNum = 0;
            while (left <= right) {
                long long mid = left + (right - left) / 2;
                long long totalCost = 0;
                for (int i = 0; i < n; i++) {  // compute the cost to produce mid alloys
                    totalCost += max(0LL, mid * comp[i] - stock[i]) * cost[i];
                }
                if (totalCost <= budget) {
                    maxNum = mid;
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
            ans = max(ans, maxNum);
        }
        return ans;
    }
};
```

## 2439. Minimize Maximum of Array

[题目网址](https://leetcode.cn/problems/minimize-maximum-of-array/). 这类最小化最大值或最大化最小值的问题，通常是二分查找答案的经典应用场景。

我们需要找到一个最小的 x，使得我们可以通过操作将数组中所有元素都变得不大于 x. 假设这个最小答案是 ans.
- 如果我们尝试一个目标值 x >= ans，那么我们一定能做到 (因为如果能把所有数都控制在 ans 以下，那控制在更大的 x 以下当然也可以).
- 如果我们尝试一个目标值 x < ans，那么我们一定做不到 (因为 ans 已经是能达到的最小值了).

解题的关键是如何判断能否将所有元素变得 `<= x` ? 对于任意前缀` nums[0...k]`，无论如何对这个子数组进行操作，其总和 `sum(nums[0...k])` 是固定的，因为只能向左传递值。因此，检查的逻辑就是从左到右遍历数组，检查每一个前缀子数组是否满足条件。

计算前缀和 `prefixSum = nums[0] + ... + nums[i]`. 如果 `prefixSum > (i+1) * x`，这意味着即便我们把这个前缀和 prefixSum 完美地平均分配给这 i+1 个数，它们的平均值也已经超过 x 了。这表明这 i+1 个数中至少会有一个数大于 x. 由于值不能向右传递，这个前缀子数组无法从别处获得帮助来减小自己的总和，所以这种情况是不可能实现的。直接返回 false. 如果遍历完所有前缀都没有违反这个条件，说明目标 x 是可行的，返回 true.

```cpp{linenos=true}
class Solution {
    bool check(vector<int>& nums, long long target) {
        long long prefixSum = 0;
        for (int i = 0; i < nums.size(); i++) {
            prefixSum += nums[i];
            if (prefixSum > target * (i + 1))
                return false;
        }
        return true;
    }

public:
    int minimizeArrayValue(vector<int>& nums) {

        int left = nums[0];  // nums[0] cannot decrease during process
        int right = ranges::max(nums);
        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (!check(nums, mid)) {  // cannot find min max < mid
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return left;
    }
};
```

## 2517. Maximum Tastiness of Candy Basket

[题目网址](https://leetcode.cn/problems/maximum-tastiness-of-candy-basket/). 思路与上题相同，关键是给定一个目标甜蜜度 x，如何判断能否选出 k 个糖果，使得它们两两价格差至少为 x ?
这里可以使用贪心策略。为了让糖果之间的间隔尽可能大，我们应该让它们在价格轴上拉得更开。

1. 对 price 数组进行升序排序。以便在一个有序的序列上进行贪心选择。
2. 进行贪心算法选择糖果。
    - 选择价格最低的糖果 `price[0]` 作为我们礼盒中的第一颗糖果。
    -  向后遍历排序后的 price 数组，寻找下一颗可以放入礼盒的糖果。要满足甜蜜度为 x 的条件，下一颗糖果的价格必须与我们上一颗选定的糖果的价格差至少为 x. 
    - 假设上一颗选的糖果价格是 last_price，在数组中继续往后找，直到找到第一个 `price[i]` 满足 `price[i] >= last_price + x`. 把这颗糖果选入礼盒，更新 `last_price = price[i]`，然后继续用同样的方法寻找下一颗。

3. 统计一共选出了多少颗糖果。如果最终选出的糖果数量 `count >= k`，那么说明甜蜜度 x 是可以实现的，返回 true. 否则，返回 false.

```cpp{linenos=true}
class Solution {
    bool check(vector<int>& price, int k, int target) {
        // greedy: always choose first
        int count = 1;
        int last_price = price[0];

        for (int i = 1; i < price.size(); i++) {
            if (price[i] - last_price >= target) {
                count++;          
                last_price = price[i]; 
            }
        }
        return count >= k;
    }

public:
    int maximumTastiness(vector<int>& price, int k) {
        sort(price.begin(), price.end());
        int left = 0, right = ranges::max(price) - price[0];
        int ans = 0;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (check(price, k, mid)) {  // can find k-1 diff >= mid Tastiness
                ans = mid;
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return ans;
    }
};
```

# Question Type 2: Select the Half where the Answer is Located

## 162. Find Peak Element

[题目网址](https://leetcode.cn/problems/find-peak-element/). 传统的二分查找依赖于数组的 单调性 (完全有序). 比如，我们要查找一个数 target，我们会比较 `nums[mid]` 和 target：
- 如果 `nums[mid] < target`，我们就知道 target 不可能在左半部分。
- 如果 `nums[mid] > target`，我们就知道 target 不可能在右半部分。

然而，在本题中，数组 nums 不是有序的。那凭什么可以舍弃一半的区间呢？这里的关键在于，我们不是在找一个特定的值，而是在找一个满足 **特定性质** (即为峰值) 的元素。二分查找的本质是通过每一步判断，将搜索空间缩小一半。我们来看看在这个问题里如何做到这一点。

我们取中间元素 `nums[mid]`，并观察它和它右边邻居 `nums[mid+1]` 的关系：

1. 如果 `nums[mid] < nums[mid+1]`:

这意味着 `nums[mid]` 本身肯定不是峰值 (因为它比右边的小)。更重要的是，这表明 从 mid 到 mid+1 处，数组是上坡的。既然是上坡，那么只要我们一直往右走就必然 会遇到一个峰值。为什么？因为数组的右边界是负无穷，你不可能无限地上坡，总会有一个点开始下坡，那个转折点就是一个峰值。

因此，我们可以断定在 mid 的右侧区域 (即 `[mid + 1, right]`) 必定存在一个峰值。于是，我们可以安全地舍弃左半部分，令 `left = mid + 1`.

2. 如果 `nums[mid] > nums[mid+1]`:

这意味着 `nums[mid]` 本身可能是一个峰值 (如果它也大于左边的 `nums[mid-1]`). 更重要的是，这表明 从 mid 到 mid+1 处，数组是下坡的。既然是下坡，那么往左看，mid 的左侧区域必然存在一个峰值。为什么？因为数组的左边界是负无穷，不可能无限地下坡。在 mid 的左边，要么 `nums[mid]` 自己就是一个峰值，要么在它左边的某个元素是峰值。

因此，我们可以 断定 在 mid 及 mid 的左侧区域 (即 `[left, mid]`) 必定存在一个峰值。于是，我们可以安全地舍弃右半部分，令 `right = mid - 1`.

```cpp{linenos=true}
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        if (nums.size() == 1) return 0;
        int left = 0, right = nums.size() - 2;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[mid + 1]) {  // summit is at left of mid
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }
};
```

## 153. Find Minimum in Rotated Sorted Array

[题目网址](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/description/). 利用旋转的特性，假设数组 `[0, 1, 2, 4, 5, 6, 7]` 旋转后变成 `[4, 5, 6, 7, 0, 1, 2]`. 旋转后的数组有以下特点:

- 它被分成了两个各自有序的部分  `[4, 5, 6, 7]` 和 `[0, 1, 2]`.
- 第一部分的所有元素都大于第二部分的所有元素。

最小值就是第二部分的第一个元素，它也是整个数组中唯一一个比它前一个元素小的数 (把数组看作环形). 二分查找的核心是每次都舍弃一半不可能是答案的区间。那么我们如何判断该舍弃哪一半呢？关键在于 `nums[mid]` 和数组端点 `nums[right]` 的比较。

1. `nums[mid] > nums[right]`: 说明 mid 肯定在第一段较大的有序数组里。

例如一开始在 `[4, 5, 6, 7, 0, 1, 2]` 中，right 指向 2, mid 指向 7，那么说明 mid 在第一段，那么最小值 (也就是第二段的开头) 一定在 mid 的右边。所以，可以安全地舍弃 `[left, mid]` 这个区间, `left = mid + 1`.

2. `nums[mid] <= nums[right]`: 说明 mid 肯定在第二段较小的有序数组里，或者是数组没有旋转的情况。

例如第二次循环在 `[4, 5, 6, 7, 0, 1, 2]` 中， mid 指向 1，right 指向 2，说明 mid 在第二段，那么最小值可能是 `nums[mid]` 本身，或者在 mid 的左边。为了严格应用闭区间的写法，我们必须做一些条件判断处理。

```cpp{linenos=true}
class Solution {
public:
    int findMin(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) 
            return nums[0];
        if (nums[0] < nums[n - 1])  // rotate n times
            return nums[0];
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mid + 1 < n && nums[mid] > nums[mid + 1])  // find maximum
                return nums[mid + 1];
            if (mid - 1 >= 0 && nums[mid] < nums[mid - 1])  // find minimum
                return nums[mid];
            if (nums[mid] > nums[right]) {  // minimum at right slope
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return nums[left];
    }
};
```


## 33. Search in Rotated Sorted Array

[题目网址](https://leetcode.cn/problems/search-in-rotated-sorted-array/description/). 和第 153 题一样，数组的整体单调性被破坏了。一个简单的 `if (target < nums[mid])` 判断无法告诉我们应该去左边还是右边。例如，在 `[4,5,6,7,0,1,2]` 中搜索 `target = 0`，第一次循环 `mid = 7`，虽然 `0 < 7`，但 0 却在 7 的右边。

但无论数组如何旋转，当我们取中点 mid 时，`[left, mid]` 和 `[mid, right]` 这两个区间中，至少有一个是完全有序的。首先判断 mid 所指向的位置是否为 target. 然后我们可以判断 `target` 是否存在有序部分的子数组中来移动指针，如果循环结束了，说明找不到。

1. nums[left] <= nums[mid]: 说明从 left 到 mid 的左半部分是完全有序的。

既然左半边有序，我们就可以准确判断 target 是否落在这个区间内：如果 `target >= nums[left]` 并且 `target < nums[mid]`，那么 target 就在这个有序的左半部分。我们就去左边找，更新 `right = mid - 1`.否则，target 就不在有序的左半部分，那它只可能在无序的右半部分。我们就去右边找，更新 `left = mid + 1`.

2. `nums[left] > nums[mid]`: 说明旋转点在 `[left, mid]` 之间，因此从 mid 到 right 的右半部分是完全有序的。

既然右半边有序，我们就可以准确判断 target 是否落在这个区间内：果 `target > nums[mid]` 并且` target <= nums[right]`，那么 target 就在这个有序的右半部分。我们就去右边找，更新 `left = mid + 1`. 否则，target 就不在有序的右半部分，那它只可能在无序的左半部分。我们就去左边找，更新 `right = mid - 1`.

```cpp{linenos=true}
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target)
                return mid;
            if (nums[mid] >= nums[left]) {  // [left, mid] is sorted
                if (nums[mid] > target && nums[left] <= target)
                    right = mid - 1;
                else
                    left = mid + 1;
            } else  {  // [mid, right] is sorted
                if (nums[mid] < target && nums[right] >= target)
                    left = mid + 1;
                else
                    right = mid - 1;
            }
        }
        return -1;
    }
};
```

## 154. Find Minimum in Rotated Sorted Array II

[题目网址](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array-ii/description/). 这道题和 153 最大的区别，也是唯一的难点，就在于重复元素的存在，使得我们无法在所有情况下通过比较 `nums[mid]` 和端点来确定范围。当严格小于和严格大于端点的情况下和之前相同。`nums[mid] == nums[right]` 是我们无法判断的情况。但是，我们可以确定的是 `nums[right]` 肯定不是唯一的最小值候选者 (因为 mid 位置还有一个和它一样大的数). 因此，我们可以安全地将 right 指针向左移动一位，把 `nums[right]` 这个元素去掉，然后在缩小后的区间里继续寻找。

最好的和平均情况下，大部分时间还是在进行二分查找，复杂度是 $O(\log n)$. 但在数组所有元素都相等的最坏的情况下，例如数组是 `[1, 1, 1, 1, 1]`，mid 和 right 指向的元素会一直相等，算法一直会进行 right--，线性地从右向左扫描，时间复杂度变为 $O(n)$. 这是为了处理重复元素所必须付出的代价。

```cpp{linenos=true}
class Solution {
public:
    int findMin(vector<int>& nums) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (mid + 1 < nums.size() && nums[mid] > nums[mid + 1])
                return nums[mid + 1];
            if (mid - 1 >= 0 && nums[mid] < nums[mid - 1])
                return nums[mid];
            if (nums[mid] > nums[right]) {
                left = mid + 1;
            } else if (nums[mid] < nums[right]) {
                right = mid - 1;
            } else {   // if nums[right] is minmimum, at least nums[mid] is also.
                right--;
            }
        }
        return nums[left];
    }
};
```
