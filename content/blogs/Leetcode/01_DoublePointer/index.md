---
date: '2025-07-20T22:22:34+08:00'
title: '01 Double-pointer'
author: ["WITHER"]

categories:
- Leetcode

tags:
- double-pointer

keywords:
- double-pointer

description: "Algorithm questions about double-pointer." # 文章描述，与搜索优化相关
summary: "Algorithm questions about double-pointer." # 文章简单描述，会展示在主页
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

# 167. Two Sum II

[题目网址](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)。如果采用暴力枚举法，对于每一个数都枚举后面的数与其相加的结果是否等于 `target`，则需要两重 for 循环，时间复杂度为 $O(n^2)$. 暴力枚举并没有使用到题目中**数组已经排好序**的条件，如果选取俩两个数加起来 `> target`，那么说明他们中间和右边的数加起来肯定 `> target`，就可以把右边的数去掉，反之可以把左边的数去掉，直到两个指针相遇。这种双指针的作法时间复杂度为 $O(n)$. 根据 0x3f 的解释为花费 $O(1)$ 的时间 (通过左右指针的两个数) 得到了 $O(n)$ 的信息 (左右指针中间的数相加后的结果与 `target` 的关系). 只用到了两个额外变量，空间复杂度为 $O(1)$.

```cpp{linenos=true}
class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        // vector is ascend order
        int left = 0, right = numbers.size() - 1;  // two pointers
        vector<int> ans(2);
        for (int i = 0; i < numbers.size(); i++) {
            if (numbers[left] + numbers[right] == target) {
                // the question requires index start from 1.
                ans[0] = left + 1;
                ans[1] = right + 1;
                break;
            } else if (numbers[left] + numbers[right] < target) {
                left++;
            } else {
                right--;
            }
        }
        return ans;
    }
};
```

# 15. 3Sum

有了上一题的经验，为了使用双指针，我们首先对数组进行排序。由于题目说输出的三元组的顺序并不重要，假定三元组中满足 `i < j < k`，可以通过枚举 `nums[i]` 将问题转换成上一题的形式。由于不能出现相同的三元组，因此如果当前枚举的数和前一个数相同的话就跳过这个数。排序的时间复杂度为 $O(n\log n)$, 二重循环的时间复杂度为 $O(n^2)$，所以算法总体时间复杂度为 $O(n^2)$. 没有用到额外变量，空间复杂度为 $O(1)$.
- 优化 1: 对于当前枚举的数，如果与后的两个数相加结果都 > 0，说明**整个数组**的后续数中不可能出现三数之和 = 0 的情况，直接退出循环。
- 优化 2: 对于当前枚举的数，如果与数组末尾的两个数相加结果都 < 0，说明**当前枚举的数后续数**中不可能出现三数之和 = 0 的情况，直接进行下一次循环。

```cpp{linenos=true}
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        // the question doesn't require the order of answer.
        sort(nums.begin(), nums.end());  // can only apply double-pointer in sorted array

        for (int i = 0; i < nums.size() - 2; i++) {
            if (i > 0 && nums[i] == nums[i - 1]) continue;  // avoid repeat
            int j = i + 1;
            int k = nums.size() - 1;
            if (nums[i] + nums[k] + nums[k - 1] < 0)  // can't find other 2 ele at idx i
                continue;  // iter next i
            if ((nums[i] + nums[j] + nums[j + 1] > 0))  // can't find other 2 ele from idx i
                break;  // return
            while (j < k) {
                if (nums[i] + nums[j] + nums[k] < 0) {
                    j++;
                } else if (nums[i] + nums[j] + nums[k] > 0) {
                    k--;
                } else {
                    ans.push_back({nums[i], nums[j], nums[k]});
                    // find next
                    j++;
                    k--;
                    // avoid repeat
                    while ((j < k) && nums[j] == nums[j - 1]) j++;
                    while ((j < k) && nums[j] == nums[k + 1]) k--;
                }
            }
        }
        return ans;
    }
};
```

# 2824. Count Pairs whose Sum is Less than Target

[题目网址](https://leetcode.cn/problems/count-pairs-whose-sum-is-less-than-target/description/). 数组的顺序对和没有影响，因此先进行排序后再套双指针模板。一旦找到和小于 `target` 的左右指针位置，则说明他们中间所有的下标对之和都小于 `target`.

```cpp{linenos=true}
class Solution {
public:
    int countPairs(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int ans = 0, left = 0, right = nums.size() - 1;
        while (left < right) {
            if (nums[left] + nums[right] >= target) {
                right--;
            } else { // numbers between left and right all little than target
                ans += right - left;  
                left++;
            }
        }
        return ans;
    }
};
```

# 16. 3sum-cloest

[题目网址](https://leetcode.cn/problems/3sum-closest/description/). 依然是先排序再使用双指针，但这里要注意的是根据是从负方向接近还是从正方向接近来移动指针。

```cpp{linenos=true}
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(), nums.end());
        int ans = 0, diff = INT_MAX;
        for (int i = 0; i < nums.size() - 2; i++) {
            int j = i + 1;
            int k = nums.size() - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum - target > diff) {
                    k--;
                } else if (sum - target < -diff) {
                    j++;
                } else {
                    diff = abs(sum - target);
                    ans = sum;
                    if (sum - target < 0) 
                        j++;
                    else
                        k--;
                }
            } 
        }
        return ans;
    }
};
```

# 18. 4sum

[题目网址](https://leetcode.cn/problems/4sum/). 与三数之和一样，先对数组进行排序，要使用双指针的话我们需要用两重 for 循环枚举两个数，问题则变成找到另外两个数使得四数之和等于 `target`，就可以使用双指针解题。其中优化跳过情况的道理和 3sum 相同。*相加结果可能超过 int(32bit) 范围，需要用 long long(64 bit) 存储*。

循环的时候需要注意
1. 在 C++ 标准库中，`std::vector::size()` 返回的类型是 `size_t`，一个无符号整数类型，意味着它不能表示负数。
2. 当 nums 的元素少于 4 个时 nums.size() - 3 会变成负数。但因为 `nums.size()` 是无符号的 `size_t`，所以这个计算是在无符号整数的上下文中进行的。在无符号整数的世界里会被 wrap around 变成一个非常大的正数（。
3. 于是，循环条件 `i < nums.size() - 3` 就变成了 i < 一个巨大的正数。循环会启动，在第一次迭代时，当 i = 0，代码尝试访问 `nums[0]`. 如果 `nums` 为空，这会立即导致段错误。即使 nums 有 0-3 个元素，循环也会继续执行，试图访问 `nums[0], nums[1], nums[2], nums[3]`... 很快就会超出向量的边界，从而引发 heap-buffer-overflow.

```cpp{linenos=true}
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
ranges::sort(nums);
        vector<vector<int>> ans;
        int n = nums.size();
        for (int i = 0; i < n - 3; i++) {
            long long x = nums[i];
            if (i > 0 && nums[i] == nums[i - 1]) continue;
            if (x + nums[i + 1] + nums[i + 2] + nums[i + 3] > target) break;
            if (x + nums[n - 1] + nums[n - 2] + nums[n - 3] < target) continue;
            for (int j = i + 1; j < n - 2; j++) {
                long long y = nums[j];
                if (j > i + 1 && nums[j] == nums[j - 1]) continue;
                if (x + y + nums[j + 1] + nums[j + 2] > target) break;
                if (x + y + nums[n - 1] + nums[n - 2] < target) continue;
                int k = j + 1;
                int l = n - 1;
                while (k < l) {
                    long long sum = x + y + nums[k] + nums[l];
                    if (sum > target) {
                        l--;
                    } else if (sum < target) {
                        k++;
                    } else {
                        ans.push_back({nums[i], nums[j], nums[k], nums[l]});
                        k++;
                        l--;
                        while (k < l && nums[k] == nums[k - 1]) k++;
                        while (k < l && nums[l] == nums[l + 1]) l--;
                    }
                }
            }
        }
        return ans;
    }
};
```

# 611. Valid Triangle Number

[题目网址](https://leetcode.cn/problems/valid-triangle-number/description/). 这个题比较巧的一点是我们对排序后的数组从最大边开始往前遍历，这样可以保证找到双指针对应的边长加起来大于枚举的边长时其他两条边加起来也大于第三边。

```cpp{linenos=true}
class Solution {
public:
    int triangleNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int ans = 0;
        // reverse iteration can ensure if left + right > i, then a triangle exists.
        for (int i = nums.size() - 1; i > 1; i--) {
            int left = 0, right = i - 1;
            while (left < right) {
                if (nums[left] + nums[right] > nums[i]) {
                    ans += right - left;
                    right--;
                } else {
                    left++;
                }
            }
        }
        return ans;
    }
};
```

# 11. Container with Most Water

[题目网址](https://leetcode.cn/problems/container-with-most-water/description/). 假设左右指针的柱子已经圈定了一个范围，如果想找到比它面积更大的范围，只能移动指针找一个更高的柱子，这样虽然宽度变小了但是高度增加了，面积有可能变大。因此从两端开始我们不断移动高度较小的指针直至相遇。到相遇时遍历了数组一遍，因此时间复杂度为 $O(n)$. 只用到了常数个额外变量，因此空间复杂度为 $O(1)$.

```cpp{linenos=true}
class Solution {
public:
    int maxArea(vector<int>& height) {
        // area = (right - left) * min(height[left], height[right])
        int left = 0, right = height.size() - 1;
        int ans = 0;
        while (left < right) {
            int area = (right - left) * min(height[left], height[right]);
            ans = max(ans, area);
            // if (height[left] < height[right]) {
            //     left++;  // width decrease but height may increase
            // } else {
            //     right--;
            // }
            height[left] < height[right] ? left++ : right--;
        }
        return ans;
    }
};
```

# 42. Trapping Rain Water

[题目网址](https://leetcode.cn/problems/trapping-rain-water/description/). 假设每个位置是一个宽度为 1 的桶，该位置能接多少水取决于该位置左侧木板最大高度和右侧木板最大高度中较小者。初始化左右指针相向移动，并记录该侧的当前最大柱子高度，对于左右指针所处的位置我们可以确定的是较矮者能装多少水，计算之后就可以移动该指针。到相遇时遍历了数组一遍，因此时间复杂度为 $O(n)$. 只用到了常数个额外变量，因此空间复杂度为 $O(1)$.

```cpp{linenos=true}
class Solution {
public:
    int trap(vector<int>& height) {
        // each pos can hold water min(lMax - height[pos], rMax - height[pos])
        int ans = 0, lMax = 0, rMax = 0;
        int left = 0, right = height.size() - 1;
        while (left < right) {
            // update prefix & suffix max
            lMax = max(lMax, height[left]);
            rMax = max(rMax, height[right]);
            if (lMax < rMax) {
                ans += lMax - height[left];
                left++;
            } else {
                ans += rMax - height[right];
                right--;
            }
        }
        return ans;
    }
};
```

# 125. Valid Palindrome

[题目网址](https://leetcode.cn/problems/valid-palindrome/description/). 这题的思路比较直观，初始化左右指针后若其中一方碰见非数字或字母就直接移动。否则进行判断，发现有一处不相等时即可跳出循环返回 `false`.

```cpp{linenos=true}
class Solution {
public:
    bool isPalindrome(string s) {
        int left = 0, right = s.size() - 1;
        bool ans = true;
        while (left < right) {
            if (!(isalpha(s[left]) || isalnum(s[left]))) {
                left++;
                continue;
            }
            if (!(isalpha(s[right]) || isalnum(s[right]))) {
                right--;
                continue;
            } 
            if (tolower(s[left]) == tolower(s[right])) {
                left++;
                right--;
            } else {
                ans = false;
                break;
            }
        }
        return ans;
    }
};
```

# 2105. Watering Plants II

[题目网址](https://leetcode.cn/problems/watering-plants-ii/description/). 这题思路也比较直观，初始化左右指针后相向移动，某一方的水不够时就需要补充并更新容量为浇水完之后的容量，增加计数器。若最后两者相遇则还需要判断当前剩余水量多的一方是否足够。

```cpp{linenos=true}
class Solution {
public:
    int minimumRefill(vector<int>& plants, int capacityA, int capacityB) {
        int left = 0, right = plants.size() - 1;
        int volumeA = capacityA, volumeB = capacityB;
        int ans = 0;
        while (left < right) {
            if (volumeA >= plants[left]) {
                volumeA -= plants[left];
                left++;
            } else {
                volumeA = capacityA - plants[left];  // fill then watering 
                ans++;
                left++;
            }
            if (volumeB >= plants[right]) {
                volumeB -= plants[right];
                right--;
            } else {
                volumeB = capacityB - plants[right];  // fill then watering 
                ans++;
                right--;
            }
        }
        if (left == right && max(volumeA, volumeB) < plants[left])
            ans++; 
        
        return ans;
    }
};
```