---
title: "Hot100"
date: 2025-08-23T13:46:45+08:00
lastmod: 2025-08-23T13:46:45+08:00
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

# Hash

## 1

由于题目规定不能用重复元素，因此要先判断哈希表中是否已经有 `target - nums[i]` 再将 `nums[i]` 加入到哈希表 (否则 `2*nums[i] = target` 情况就会加入重复元素)

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> ans;
        unordered_map<int, int> map;
        for (int i = 0; i < nums.size(); i++) {
            if (map.find(target - nums[i]) != map.end()) {
                ans = {i, map[target - nums[i]]};
                break;
            }
            map[nums[i]] = i;
        }
        return ans;
    }
};
```

## 49

> 字母异位词是通过重新排列不同单词或短语的字母而形成的单词或短语，并使用所有原字母一次。

这提示我们如果两个单词按字母表顺序排序后相等，那么就是字母异位词。可以用一个哈希表来存储排序后相等的字符串。

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> ans;
        unordered_map<string, vector<string>> map;
        for (int i = 0; i < strs.size(); i++) {
            string s = strs[i];
            ranges::sort(s);
            map[s].push_back(strs[i]);
        }
        for (auto [key, value] : map) {
            ans.push_back(value);
        }
        return ans;
    }
};
```

## 128

用一个哈希集合来存储 `nums` 中的不同元素，这样可以实现平均 $O(1)$ 时间复杂度的查找。**遍历集合中的数字** x，如果发现 x-1 也在集合中，则跳过该数字。因为从 x-1 开始查找的连续长度肯定比从 x 开始的长。从每一个可能的起点 (不存在比他小 1 的数字) 查找连续长度并不断更新。

```cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        int ans = 0;
        unordered_set<int> s(nums.begin(), nums.end());
        for (int num : s) {
            if (s.contains(num - 1)) {
                continue;
            }
            int x = 1;
            while (s.contains(num + 1)) {
                x++;
                num++;
            }
            ans = max(ans, x);
            if (ans * 2 >= nums.size()) {
                break;
            }
        }
        return ans;
    }
};
```

# Double-Pointer
## 283

把 0 视作空位
- 慢指针 slow 指向下一个非零元素应该被放置的位置。换句话说，slow 左边的所有元素（不包括slow指向的位置）都是处理好的、不为零的元素。
- 快指针 fast 开始向右遍历数组，若遇到非零元素则将其和 slow 位置的元素交换，然后，将慢指针 slow 向右移动一位。遇到零则什么都不做继续向前遍历。

这样 `[slow, fast - 1]` 所形成的区间内均为 0.
```cpp
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int slow = 0;
        for (int& num : nums) {
            if (num) {
                swap(num, nums[slow]);
                slow++;
            }
        }
    }
};
```

## 12

给定左右模板的位置 left 和 right。容器能接水的高度取决于较矮的那个。当相向移动指针的时候，宽度变短，想要盛水更多只能寄希望于接水高度增加。因此 `left < right` 的时候我们移动指向较矮木板的指针。

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int ans = 0;
        int left = 0, right = height.size() - 1;
        while (left < right) {
            int area = min(height[left],height[right]) * (right - left);
            ans = max(area, ans);
            height[left] < height[right] ? left++ : right--;
        }
        return ans;
    }
};
```

## 15

想找到 `a + b + c = 0`，如果能确定一个数 a，问题就变成了在数组剩下的部分寻找两个数 b 和 c，使得 `b + c = -a`. 这就从三数之和问题降维成了我们熟悉的两数之和问题。对整个数组进行排序，然后遍历排序后的数组，对于每个元素 `nums[i]`，我们将其视为 a，然后在它后面的区间 `[i+1, n-1]` 内使用双指针法寻找 b 和 c.

去重注意点
1. 枚举的端点 nums[i] 和上一个 nums[i-1] 相等时需要调过。
2. 双指针遍历找到一个可行解时，移动 j 和 k 直到他们指向位置的元素和加入答案中的值不相等。

剪枝优化
1. `nums[i] + nums[i+1] + nums[i + 2] > 0`: 说明以 i 及之后为端点的所有三元组之和全都 > 0. 直接退出循环。
2. `nums[i] + nums[n - 2] + nums[n - 1] < 0`: 说明 以 i 为端点的所有三元组之和和全都 < 0. 枚举下一个端点。

```cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ans;
        int n = nums.size();
        ranges::sort(nums);
        for (int i = 0; i < n - 2; i++) {
            if (i > 0 &&  nums[i] == nums[i - 1])
                continue;
            if (nums[i] + nums[i+1] + nums[i + 2] > 0)
                break;
            if (nums[i] + nums[n - 2] + nums[n - 1] < 0)
                continue;
            
            int j = i + 1, k = n - 1;
            while (j < k) {
                int sum = nums[i] + nums[j] + nums[k];
                if (sum < 0) {
                    j++;
                } else if (sum > 0) {
                    k--;
                } else {
                    ans.push_back({nums[i], nums[j], nums[k]});
                    j++;
                    k--;
                    while(j < k && nums[j] == nums [j - 1]) j++;
                    while(j < k && nums[k] == nums [k + 1]) k--;
                }
            }
        }
        return ans;
    }
};
```

## 42

同样是接雨水问题，每个柱子能接水的量为左右两侧柱子较矮者减去自己的高度。因此初始化两个指针指向左右端点，从左往右遍历过程中看哪边柱子矮就移动哪边，不断更新左右侧柱子的最大高度。最后左右指针一定会在高度最高的柱子相遇，而这个位置是无法接水的。

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int ans = 0;
        int left = 0, right = height.size() - 1;
        int lMax = 0, rMax = 0;
        for (int i = 0; i < height.size(); i++) {
            lMax = max(lMax, height[left]);
            rMax = max(rMax, height[right]);
            ans += min(lMax, rMax) - height[i];
            lMax < rMax ? left++ : right--;
        }
        return ans;
    }
};
```

# Sliding Window

## 3

我们滑动窗口维护的是一段没有重复字符的子串，需要用一个哈希表来记录子串中字符对应的下标。

通过从左向右遍历来尝试扩大窗口
- 若发现字符已存在，则 `left` 到 `map[s[right]]` 的所有字符都需要被删除。窗口左端点变为 `map[s[right]] + 1`，更新当前的无重复子串长度，以及这个重复字符对应的下标。
- 否则当前无重复子串长度 + 1，更新答案，将字符及对应下标记录在哈希表中。

```cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> map;
        int ans = 0;
        int left = 0;
        int len = 0;
        for (int right = 0; right < s.size(); right++) {
            if (map.find(s[right]) != map.end()) {
                while (left <= map[s[right]]) {
                    map.erase(s[left]);
                    left++;
                }
                len = right - left + 1;
            } else {
                len++;
                ans = max(ans, len);
            }
            map[s[right]] = right;
        }
        return ans;
    }
};
```

## 438

维持一个和 p 字符串长度相等的窗口，在 s 字符串上滑动。我们只需要判断窗口内的字符串是不是 p 的一个异位词。不需要每次都对窗口内的子串进行排序，而是通过字符频率来判断。由于题目说了字符串只包含小写字母，因此可以用长度为 26 的数组来存储频率。

首先构造第一个窗口，判断是否相同后向后滑动，在每一步循环中更新窗口内字符的频率，然后再次进行比较判断。

```cpp
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> ans;
        int sLen = s.length(), pLen = p.length();
        if (sLen < pLen) {
            return ans;
        }
        vector<int> pFreq(26, 0), wFreq(26, 0);
        // init first window
        for (int i = 0; i < pLen; i++) {
            pFreq[p[i] - 'a']++;
            wFreq[s[i] - 'a']++;
        }
        if (pFreq == wFreq) {
            ans.push_back(0);
        }
        for (int right = pLen; right < sLen; right++) {
            int left = right - pLen + 1;
            // insert and remove
            wFreq[s[right] - 'a']++;
            wFreq[s[left - 1] - 'a']--;
            if (pFreq == wFreq) {
                ans.push_back(left);
            } 
        }
        return ans;
    }
};
```

# Substr

## 560

定义 `pre[i]` 为从 `nums[0]` 到 `nums[i]` 的前缀和。那么，从索引 j 到 i (`j <= i`) 的子数组的和就可以表示为 `pre[i] - pre[j-1]`. 题目要求我们找到和为 k 的子数组，也就是说，我们需要找到满足 `pre[i] - pre[j-1] == k` 的 `(i, j)` 组合的数量。

将上面的等式变换一下，就得到 `pre[j-1] == pre[i] - k`. 对于当前的索引 i，我们不再需要向前遍历 j 来检查每一个子数组的和。我们只需要知道，在 0 到 i-1 的范围内，有多少个 j-1 使得 `pre[j-1]` 的值恰好等于 `pre[i] - k`.

我们可以用一个哈希表来存储出现过的前缀和及出现的次数。初始化的时候为 `{0, 1}` 表示前缀和为 0 的情况出现过 1 次。初始化前缀和 `preSum = 0`. 遍历数组的时候一边累加前缀和一边查找 `preSum - k` 出现过的次数。**注意要先查找后添加** (`2 * preSum = k` 情况下就多找了).

```cpp
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> map;
        map[0] = 1;
        int ans = 0, preSum = 0;
        for (int i = 0; i < nums.size(); i++) {
            preSum += nums[i];
            if (map.count(preSum - k)) {
                ans += map[preSum - k];
            }
            map[preSum]++;
        }
        return ans;
    }·
};
```

## 239 

维护一个从头到尾单调递减的双端队列。移动窗口 (遍历数组) 的过程中，如果数组元素 >= 队尾元素就一直将队尾元素弹出，直到条件不满足或者队列为空，然后将元素插入队尾。

同时为了不超出窗口大小，队列中需要记录的是元素的下标，并在每次循环的过程中判断当前元素下标减去队头元素下标是否超出窗口大小。当遍历到的下标 `i >= k - 1` 时说明窗口形成，将队头元素对应数组中的值加入答案。

```cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> ans;
        deque<int> dq;
        for (int i = 0; i < nums.size(); i++) {
            while (!dq.empty() && nums[i] >= nums[dq.back()]) {
                dq.pop_back();
            }
            dq.push_back(i);
            if (i - dq.front() + 1 > k) {
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

## 76

还是滑动窗口的思想，不断枚举子串的右端点，如果当前窗口包含 t，我们就不断移动左端点来缩小窗口，不断更新长度最小的子串直至当前窗口不再包含 t.

对于如何判断窗口是否包含 t，由于 s 和 t 都只由大小写字母组成，我们可以用一个数组 cnt 来记录**窗口中字母小于 t 中出现的次数**，并用一个变量 less 来记录当前窗口中有多少字母的次数低于 t 中对应字母的次数。

遍历过程中将 cnt 对应字母频率 -1 (出现负数也不影响判断)，`cnt[s[right]] == 0` 时说明 t 中对应字母已经被完全覆盖，`less--`. 当 `less == 0` 说明窗口包含 t. 在缩小窗口的过程中还原 cnt 数组，如果 cnt[s[left]] == 0，那么还原后窗口该字母出现的次数又会小于 t 的，`less++`.

```cpp
class Solution {
public:
    string minWindow(string s, string t) {
        if (s.length() < t.length()) {
            return "";
        }
        int cnt[128]{}, less = 0;
        for (char c : t) {
            if (cnt[c] == 0) {
                less++;
            }
            cnt[c]++;
        }
        int left = 0;
        int ans_left = -1, ans_right = s.length();
        for (int right = 0; right < s.length(); right++) {
            char c = s[right];
            cnt[c]--;
            if (cnt[c] == 0) {
                less--;
            }
            
            while (less == 0) {
                if (right - left < ans_right - ans_left) {
                    ans_right = right;
                    ans_left = left;
                }
                if (cnt[s[left]] == 0) {
                    less++;
                }
                cnt[s[left]]++;
                left++;
            }
        }
        return ans_left < 0 ? "" : s.substr(ans_left, ans_right - ans_left + 1);
    }
};
```

# Array 
## 53

定义 `f[i]` 为以 `nums[i]` 为结尾的最大子数组和。`f[i]` 可以选择和之前的拼在一起 `f[i-1] + nums[i]` 或者自成一个子数组 `nums[i]`. 如果之前的最大子数组和 < 0 则拼在一起只会更小，所以我们有

$$
f[i]=\begin{cases}nums[i],&i=0\\\max(f[i-1],0)+nums[i],&i\geq1\end{cases}
$$

答案为 f 数组中最大的那一个。

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        vector<int> f(nums.size());
        f[0] = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            f[i] = max(f[i - 1], 0) + nums[i];
        }
        return ranges::max(f);
    }
};
```

观察到我们更新的等式只用到了两个状态，因此可以降低空间复杂度

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int ans = nums[0];
        int f = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            f = max(f, 0) + nums[i];
            ans = max(ans, f);
        }
        return ans;
    }
};
```

## 56

两个区间 `[a, b], [c, d]` 重合的充要条件为 `a <= d && c <= b`. 先按照区间开始时间排序就保证了 `a <= c <= d`. 遍历数组的时候若 `c <= b` 就说明两个区间可以重合，然后更新结束时间为两个区间的较大者。

```cpp
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        ranges::sort(intervals);
        vector<vector<int>> ans;
        int begin = intervals[0][0], end = intervals[0][1];
        for (auto interval : intervals) {
            if (interval[0] <= end) {
                end = max(end, interval[1]);
            } else{
                ans.push_back({begin, end});
                begin = interval[0];
                end = interval[1];
            }           
        }
        ans.push_back({begin, end});
        return ans;
    }
};
```

## 189

设数组大小为 n.
1. 反转前 n - k 个元素。
2. 反转后 k 个元素。
3. 反转整个数组。

```cpp
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k %= n;
        // 1. 反转前 n - k 个元素
        std::reverse(nums.begin(), nums.begin() + n - k);
        // 2. 反转后 k 个元素
        std::reverse(nums.begin() + n - k, nums.end());
        // 3. 反转整个数组
        std::reverse(nums.begin(), nums.end());
    }
};
```

## 238

对于数组中的任意一个位置 i，`answer[i]` 的值是**它左边所有元素的乘积 乘以 右边所有元素的乘积**。我们可以分两步来计算：

1. 计算前缀乘积 (Prefix Products): 创建一个数组（或者直接利用结果数组 answer），answer[i] 存储 nums[0] 到 nums[i-1] 的所有元素的乘积。

2. 计算后缀乘积 (Suffix Products) 并得出最终结果: 从后向前遍历数组。引入一个变量 `suffix_product` 来记录右侧所有元素的累积乘积。在遍历到位置 i 时，先将 `answer[i]`（此时存储的是前缀乘积）乘以 `suffix_product`，然后更新 `suffix_product` 为 `suffix_product * nums[i]`，为下一个位置的计算做准备。

```cpp
class Solution {
public:
    vector<int> productExceptSelf(vector<int>& nums) {
        int n = nums.size();
        vector<int> suf(n, 1);
        for (int i = n - 2; i >= 0; i--) {
            suf[i] = suf[i + 1] * nums[i + 1];
        }
        int pre = nums[0];
        for (int i = 1; i < n; i++) {
            suf[i] *= pre;
            pre *= nums[i];
        }
        return suf;
    }
};
```

## 41

$O(1)$ 的空间复杂度限制意味着我们不能使用哈希表等额外的数据结构来记录数字的出现情况。必须在输入数组 nums 本身上进行修改和标记，以达到记录信息的目的。

我们的目标是找到缺失的第一个正整数。假设数组的长度为 n，那么这个缺失的数一定在 `[1, n+1]` 这个范围内。
1. 如果 1 到 n 都在数组 nums 中，那么缺失的第一个正整数就是 n+1.
2. 如果 1 到 n 中有任何一个数不在 nums 中，那么缺失的第一个正整数就在 `[1, n]` 这个区间内。

因此，我们的问题转化为了：检查 1 到 n 这些数字是否在 nums 数组中。

我们可以利用数组的索引来充当哈希表的键，数组中的元素来充当值，从而建立一种映射关系。具体来说，我们希望数字 k 能够被放到索引为 k-1 的位置上。例如，数字 1 应该被放到索引 0，数字 2 应该被放到索引 1，以此类推。

第一次遍历数组时，只要 `nums[i]` 是一个在 `[1, n]` 范围内的正数，并且它没有被放到正确的位置上 (即 `nums[i] != nums[nums[i] - 1]`)，我们就继续交换。
> `nums[i] != nums[nums[i] - 1]` 是为了防止当两个相同数字需要交换时陷入死循环。例如` nums = [1, 1], i = 0, nums[0] = 1`, 遍历到 nums[1] 时 `nums[nums[1]-1] = nums[0] = 1`. 说明**要进行交换的位置上的值已经是正确的**。

经过上一步的整理，数组 nums 已经尽可能地把数字 k 放在了索引 k-1 的位置 (在答案范围内且没有重复的)。现在我们再遍历一次数组：检查 nums[i] 是否等于 i+1。第一个不满足条件的索引 i，就意味着 i+1 是缺失的第一个正整数。遍历完成都满足说明缺失的第一个正整数为 n+1.

```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            while (nums[i] >= 1 && nums[i] <= n && nums[i] != nums[nums[i] - 1]) {
                swap(nums[i], nums[nums[i] - 1]);
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return n + 1;
    }
};
```

# Matrix

## 73

可以利用矩阵的第一行和第一列来存储哪些行和列需要被置零。

首先，我们需要两个布尔变量 `isFirstRowZero` 和 `isFirstColZero` 来单独记录第一行和第一列是否本身就包含 0. 因为第一行第一列 `matrix[0][0]` 的状态是共享的，所以需要分开记录。

用第一行/列做标记：遍历除第一行和第一列之外的矩阵部分，如果 `matrix[i][j] == 0`，则将对应的第一行 `matrix[i][0]` 和第一列 `matrix[0][j]` 的元素置零。

再次遍历除第一行和第一列之外的矩阵部分。如果 `matrix[i][0] == 0` 或 `matrix[0][j] == 0`，说明第 i 行或第 j 列需要被清零，因此将 `matrix[i][j]` 置为 0.

最后，根据步骤 1 中记录的 `isFirstRowZero` 和 `isFirstColZero` 的值来决定是否将第一行和第一列整体置零。

```cpp
class Solution {
public:
    void setZeroes(std::vector<std::vector<int>>& matrix) {
        int m = matrix.size();
        if (m == 0) return;
        int n = matrix[0].size();
        
        bool isFirstColZero = false;
        bool isFirstRowZero = false;
        
        // 1. 检查第一列是否需要置零
        for (int i = 0; i < m; ++i) {
            if (matrix[i][0] == 0) {
                isFirstColZero = true;
                break;
            }
        }
        
        // 2. 检查第一行是否需要置零
        for (int j = 0; j < n; ++j) {
            if (matrix[0][j] == 0) {
                isFirstRowZero = true;
                break;
            }
        }
        
        // 3. 用第一行和第一列记录其他行列的零状态
        // 从 (1, 1) 开始遍历
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0;
                    matrix[0][j] = 0;
                }
            }
        }
        
        // 4. 根据第一行和第一列的标记，更新矩阵（不包括第一行第一列）
        for (int i = 1; i < m; ++i) {
            for (int j = 1; j < n; ++j) {
                if (matrix[i][0] == 0 || matrix[0][j] == 0) {
                    matrix[i][j] = 0;
                }
            }
        }
        
        // 5. 最后处理第一行和第一列
        if (isFirstRowZero) {
            for (int j = 0; j < n; ++j) {
                matrix[0][j] = 0;
            }
        }
        
        if (isFirstColZero) {
            for (int i = 0; i < m; ++i) {
                matrix[i][0] = 0;
            }
        }
    }
};
```

## 54

维护四个变量，分别代表当前待遍历矩阵的上、下、左、右四个边界。在每一轮循环中，我们沿着这四个边界走一圈  (👉👇👈👆)，然后向内收缩边界，直到边界相遇或交错。

当螺旋收缩到只剩一行或一列时，上面第 1、2 步执行完后，边界条件可能就不满足了（例如，top > bottom）。因此，在执行第 3、4 步之前，需要再次检查边界条件，防止重复添加元素。

```cpp
class Solution {
public:
    std::vector<int> spiralOrder(std::vector<std::vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        std::vector<int> ans;
        ans.reserve(m * n); 

        int top = 0, bottom = m - 1, left = 0, right = n - 1;

        while (left <= right && top <= bottom) {
            // 1. 从左到右遍历上边界
            for (int j = left; j <= right; ++j) {
                ans.push_back(matrix[top][j]);
            }
            top++; // 上边界下移

            // 2. 从上到下遍历右边界
            for (int i = top; i <= bottom; ++i) {
                ans.push_back(matrix[i][right]);
            }
            right--; // 右边界左移

            // 检查边界，防止在只剩一行或一列时重复遍历
            if (top <= bottom) {
                // 3. 从右到左遍历下边界
                for (int j = right; j >= left; --j) {
                    ans.push_back(matrix[bottom][j]);
                }
                bottom--; // 下边界上移
            }

            if (left <= right) {
                // 4. 从下到上遍历左边界
                for (int i = bottom; i >= top; --i) {
                    ans.push_back(matrix[i][left]);
                }
                left++; // 左边界右移
            }
        }
        return ans;
    }
};
```

## 48

位于 i 行 j 列的元素，去到 j 行 `n−1−i` 列，即 `(i,j) -> (j,n−1−i)`.
 因此可以通过先转置再纵向对称翻转实现。

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int m = matrix.size();
        int n = matrix[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = i + 1; j < n; j++) {
                swap(matrix[i][j], matrix[j][i]);
            }
        }
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n / 2; j++) {
                swap(matrix[i][j], matrix[i][n - 1 - j]);
            }
        }
    }
};
```

## 240

右上角的元素，它是当前行的最大值，同时也是当前列的最小值。每一步都可以排除掉一行或者一列，从而不断缩小搜索范围。从右上角开始搜索:
- `matrix[row][col] > target`: 当前元素是其所在列的最小值，如果它都比 target 大，那么这一整列下方的所有元素必定也比 target 大。因此，可以完全排除当前列。向左移动，`col--`.

- `matrix[row][col] < target`: 因为当前元素是其所在行的最大值，如果它都比 target 小，那么这一整行左边的所有元素必定也比 target 小。因此，可以完全排除当前行。向下移动，`row++`.

当超出下边界或超出左边界时说明找不到目标值。

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size();
        int n = matrix[0].size();
        int row = 0, col = n - 1;
        while (row < m && col >= 0) {
            if (matrix[row][col] > target) {
                col--;
            } else if (matrix[row][col] < target) {
                row++;
            } else {
                return true;
            }
        }
        return false;
    }
};
```

# Linked List

## 160

使用两个指针 pa 和 pb 分别指向 headA 和 headB. 同时遍历两个链表:

如果 pa 或 pb 到达链表末尾（nullptr），将其切换到另一个链表的头部继续遍历。这样，两个指针最终会“走过相同的距离”，要么在相交节点相遇，要么都到达 nullptr.

> 设链表 A 的长度为 a + c，链表 B 的长度为 b + c，其中 c 是相交部分的长度。指针 pa 遍历 a + c + b 后，指针 pb 遍历 b + c + a 后，它们会在相交节点相遇 (无交点则都到达 nulll ptr).

```cpp
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        ListNode*pa = headA, *pb = headB;
        while (pa != pb) {
            pa = pa ? pa->next : headB;
            pb = pb ? pb->next : headA;
        }
        
        return pa;
    }
};
```

## 206

从 `pre = nullptr` 开始，这样可以自然地将原头节点的 `next` 设置为 `nullptr`.

```cpp
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr, *cur = head;
        while (cur) {
            ListNode* n = cur->next;
            cur->next = pre;
            pre = cur;
            cur = n;
        }
        return pre;
    }
};
```  

## 141

慢指针 slow 一次移动一步，快指针 fast 一次移动两步。如果能相遇说明有环，否则快指针会先走到链表末尾 nullptr.

```cpp
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                return true;
            }
        }
        return false;
    }
};
```

## 142

假设进环前的路程为 a，环长为 b。设慢指针走了 x 步时，快慢指针相遇，此时快指针走了 2x 步。显然 2x-x=nb（快指针比慢指针多走了 n 圈），即 x=nb. 也就是说慢指针总共走过的路程是 nb，但这 nb 当中，实际上包含了进环前的一个小 a，因此慢指针在环中只走了 nb-a 步，它还得再往前走 a 步，才是完整的 n 圈。所以，我们让头节点和慢指针同时往前走，当他俩相遇时，就走过了最后这 a 步。

```cpp
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                while (head != slow) {
                    slow = slow->next;
                    head = head->next;
                }
                return slow;
            }
        }
        return nullptr;
    }
};
```

## 21

创建一个 dummy 节点，作为合并后的新链表头节点的前一个节点。

比较 list 1 和 list2的节点值，如果 list1 的节点值小，则把 list1 加到新链表的末尾，然后把 list1 替换成它的下一个节点。反之同样。

直到一个链表为空就把另一个链表直接加到新链表末尾，返回 `dummy.next`.

```cpp
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;
        while (list1 && list2) {
            if (list1->val < list2->val) {
                tail->next = list1;
                list1 = list1->next;
            } else {
                tail->next = list2;
                list2 = list2->next;
            }
            tail = tail->next;
        }
        tail->next = list1 ? list1 : list2;
        return dummy.next;
    }
};
```

## 2

注意两个链表遍历完后可能还有进位。

```cpp
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode dummy;
        ListNode* p = &dummy;
        int carry = 0;
        while (l1 && l2) {
            int num = l1->val + l2->val + carry;
            carry = num >= 10 ? 1 : 0;
            num %= 10;
            p->next = new ListNode(num);
            p = p->next;
            l1 = l1->next;
            l2 = l2->next;
        }
        while (l1) {
            int num = l1->val + carry;
            carry = num >= 10 ? 1 : 0;
            num %= 10;
            p->next = new ListNode(num);
            p = p->next;
            l1 = l1->next;
        }
        while (l2) {
            int num = l2->val + carry;
            carry = num >= 10 ? 1 : 0;
            num %= 10;
            p->next = new ListNode(num);
            p = p->next;
            l2 = l2->next;
        }
        if (carry) {
            p->next = new ListNode(carry);
        }
        return dummy.next;
    }
};
```

## 19

为了简化需要删除头节点的逻辑，我们需要添加一个哨兵节点 dummy. 慢指针 slow 在链表 dummy，先移动快指针 slow 到正数第 n 个节点。然后同时移动快慢指针，fast 到达 nullptr 时，左端点就在倒数第 n 个节点。

```cpp
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        ListNode dummy(0, head);
        ListNode* slow = &dummy, *fast = head;
        for (int i = 0; i < n; i++) {
            fast = fast->next;
        }
        while (fast) {
            slow = slow->next;
            fast= fast->next;
        }
        slow->next = slow->next->next;
        return dummy.next;
    }
};
```

## 24

用一个指针 pre 指向已经翻转部分的最后一个节点，cur 指向下一个要翻转的节点。当要翻转的一对节点 `cur && cur->next` 都存在时:
1. `pre->next = cur->next`
2. `cur->next = cur->next->next`
3. `cur->next->next = cur`

此时 cur 成为已经翻转部分的最后一个节点，让 pre 指向它，cur 再指向 `cur->next`. 由于需要对头节点进行翻转，所以我们初始化哨兵节点来作为一开始已经翻转部分的最后一个节点。

```cpp
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (!head || !head->next) {
            return head;
        }
        ListNode dummy(0, head);
        ListNode* pre = &dummy, *cur = head;
        while (cur && cur->next) {
            ListNode* next = cur->next;
            pre->next = next;
            cur->next = next->next;
            next->next = cur;

            pre = cur;
            cur = cur->next;
        }
        return dummy.next;
    }
};
```

## 25

1. 通过一次遍历计算出链表总长度，从而确定总共需要反转多少个分组。
2. pre 指向已经翻转的部分的最后一个节点， 内循环进行每一组的链表反转。反转结束后 cur 指向的是下一组的开始节点。重新链接反转后的子链表。

```cpp
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        int len = 0;
        ListNode* cur = head;
        while (cur) {
            len++;
            cur = cur->next;
        }
        int ng = len / k;
        ListNode dummy(0, head);
        ListNode* pre = &dummy;
        cur = head;
        for (int i = 0; i < ng; i++) {
            ListNode* p = nullptr;
            for (int j = 0; j < k; j++) {
                ListNode* next = cur->next;
                cur->next = p;
                p = cur;
                cur = next;
            }
            pre->next->next = cur;
            ListNode* next_pre = pre->next;
            pre->next = p;
            pre = next_pre;
        }
        return dummy.next;
    }
};
```

## 234

1. 用快慢指针找到链表中间位置 (len / 2) 的节点。
2. 翻转后一半链表。
3. 同时从头尾开始遍历判断值是否相等。

```cpp
class Solution {
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr, *cur = head;
        while (cur) {
            ListNode* n = cur->next;
            cur->next = pre;
            pre = cur;
            cur = n;
        }
        return pre;
    }

    ListNode* middleList(ListNode* head) {
        ListNode* slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }

public:
    bool isPalindrome(ListNode* head) {
        ListNode* mid = middleList(head);
        ListNode* tail = reverseList(mid);
        while (head && tail) {
            if (head->val != tail->val) {
                return false;
            }
            head = head->next;
            tail = tail->next;
        }
        return true;
    }
};
```

## 138

1. 创建交织链表: 遍历原链表，对于每个节点，创建一个新节点（副本），并将其插入到原节点和原节点的下一个节点之间。例如，原链表 `A -> B -> C` 变成 `A -> A' -> B -> B' -> C -> C'`.

2. 设置 random 指针: 对于原链表的每个节点 N，其副本节点 N' 紧随其后。
如果 `N->random` 指向某个节点 M，则 `N'->random` 应指向 M' (M 的副本). 由于 M' 是 `M->next`，我们可以直接设置 N`->next->random = N->random->next`.

3. 分离新旧链表: 遍历交织链表，将新节点和旧节点分开，恢复原链表并提取新链表。确保正确设置 next 指针，断开新旧节点之间的连接。

```cpp
class Solution {
public:
    Node* copyRandomList(Node* head) {
        if (!head) return nullptr;
       
        Node * cur = head;
        while (cur) {
            cur->next = new Node(cur->val, cur->next, nullptr);
            cur = cur->next->next;
        }
        cur = head;
        while (cur) {
            if (cur->random) {
                cur->next->random = cur->random->next;
            }
            cur = cur->next->next;
        }
        Node *newHead = head->next;
        cur = head;
        while (cur->next->next) {
            Node* copy = cur->next;
            cur->next = copy->next;
            copy->next = cur->next->next;

            cur = cur->next;
        }
        cur->next = nullptr;
        return newHead;
    }
};
```

## 148

1. 遍历链表，获取链表长度。
2. 自底向上归并排序: 将链表中的每个节点都看作是一个长度为 1 的、已经排好序的子链表。在内部循环每一轮中找到每一对要合并的子链表 head1 和 head2，然后将它们合并，并链接到上一段合并好的链表的末尾。
    - 第一轮：将相邻的、长度为 1 的子链表两两合并，形成多个长度为 2 的有序子链表。
    - 第二轮：将相邻的、长度为 2 的子链表两两合并，形成多个长度为 4 的有序子链表。
    - 第三轮：将相邻的、长度为 4 的子链表两两合并，形成多个长度为 8 的有序子链表。
3. 每次从 dummy 节点开始重复这个过程，归并后将子链表的长度 subLen 翻倍 (1, 2, 4, 8, ...)，直到 subLen >= 整个链表的长度。

```cpp
class Solution {
    ListNode* merge(ListNode* head1, ListNode* head2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;

        while (head1 && head2) {
            if (head1->val < head2->val) {
                tail->next = head1;
                head1 = head1->next;
            } else {
                tail->next = head2;
                head2 = head2->next;
            }
            tail = tail->next;
        }
        tail->next = head1 ? head1 : head2;
        return dummy.next;
    }
public:
    ListNode* sortList(ListNode* head) {
        int len = 0;
        ListNode* p = head;
        while (p) {
            len++;
            p = p->next;
        }
        ListNode dummy(0, head);
        for (int subLen = 1; subLen < len; subLen*= 2) {
            ListNode* pre = &dummy;  // pre records the tail of last merged 2 segments
            ListNode* cur = dummy.next;  // cur records the start node to merge

            while (cur) {
                // find first segment chainList with subLen
                ListNode* head1 = cur;
                for (int i = 1; i < subLen && cur && cur->next; i++) {
                    cur = cur->next;
                }
                // find second segment chianList and cut the connection with first segement
                ListNode* head2 = nullptr;
                if (cur) {
                    head2 = cur->next;
                    cur->next = nullptr;
                }

                // find the tail of second segment
                cur = head2;
                for (int i = 1; i < subLen && cur && cur->next; i++) {
                    cur = cur->next;
                }

                // record the next round start node to merge and cut the connection with second segement
                ListNode* nextSub = nullptr;
                if (cur) {
                    nextSub = cur->next;
                    cur->next = nullptr;
                }

                ListNode* merged = merge(head1, head2);
                pre->next = merged;

                // move pre to the end of merged segment
                while (pre->next) {
                    pre = pre->next;
                }

                cur = nextSub;
            }
        }
        return dummy.next;
    }
};
```


## 23

直接自底向上合并链表：

1. 两两合并：把 lists[0] 和 lists[1] 合并，合并后的链表保存在 lists[0] 中；把 lists[2] 和 lists[3] 合并，合并后的链表保存在 lists[2] 中；依此类推。
2. 四四合并：把 lists[0] 和 lists[2] 合并（相当于合并前四条链表），合并后的链表保存在 lists[0] 中；把 lists[4] 和 lists[6] 合并，合并后的链表保存在 lists[4] 中；依此类推。
3. 八八合并：把 lists[0] 和 lists[4] 合并（相当于合并前八条链表），合并后的链表保存在 lists[0] 中；把 lists[8] 和 lists[12] 合并，合并后的链表保存在 lists[8] 中；依此类推。
4. 依此类推，直到所有链表都合并到 lists[0] 中。最后返回 lists[0].

```cpp
class Solution {
    ListNode* merge(ListNode* head1, ListNode* head2) {
        ListNode dummy(0);
        ListNode* tail = &dummy;
        while (head1 && head2) {
            if (head1->val < head2->val) {
                tail->next = head1;
                head1 = head1->next;
            } else {
                tail->next = head2;
                head2 = head2->next;
            }
            tail = tail->next;
        }
        tail->next = head1 ? head1 : head2;
        return dummy.next;
    }
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        if (lists.empty()) return nullptr;
        if (lists.size() == 1) return lists[0];
        
        for (int step = 1; step < lists.size(); step *= 2) {
            for (int i = 0; i < lists.size() - step; i += step * 2) {
                lists[i] = merge(lists[i], lists[i + step]);
            }
        }
        
        return lists[0];
    }
};
```

## 146

- 双向链表 (`std::list`): 维护数据的使用顺序。
    - 链表中存储 (key, value) 对。
    - 链表头部 (front)：存放最近访问过的数据。
    - 链表尾部 (back)：存放最久未被访问的数据。
- 哈希表 (`std::unordered_map`): 实现 O(1) 的快速查找。通过 key，我们能立刻定位到它在链表中的位置。
    - key：存储缓存项的键。
    - value：存储一个指向双向链表中对应节点的指针或迭代器。

- `get(key)`:

    通过哈希表查找 key. 如果未找到直接返回 -1.如果找到了:

    1. 从哈希表中获取到链表节点的指针/迭代器。
    2. 通过指针/迭代器获取节点中的 value.
    3. 将这个节点从它当前的位置移动到链表的头部（表示它刚刚被访问过）。
    4. 返回 value.

- put(key, value) 操作: 通过哈希表查找 key.
    
    如果找到了 (key 已存在)：

    1. 从哈希表中获取到链表节点的指针/迭代器。
    2. 更新该节点中的 value.
    3. 将这个节点移动到链表的头部。

    如果未找到 (key 是新的)：
    1. 检查缓存是否已满，如果已满:
        - 获取链表尾部的节点。
        - 从哈希表中删除尾部节点的 key。
        - 从链表中删除该尾部节点。
    3. 在链表头部创建一个新节点，存储 (key, value).
    4. 在哈希表中插入新的 key，并让其 value 指向刚创建的链表头节点。