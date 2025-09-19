---
title: "06 Backtracking"
date: 2025-08-04T09:47:31+08:00
lastmod: 2025-08-04T09:47:31+08:00
author: ["WITHER"]

categories:
- Leetcode

tags:
- Backtracking

keywords:


description: "Algorithm questions about backtracking." # 文章描述，与搜索优化相关
summary: "Algorithm questions about backtracking." # 文章简单描述，会展示在主页
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

# Preliminary

- **原问题**: 构造长度为 n 的字符串。
-枚举一个字母后
- **子问题**: 构造长度为 n-1 的字符串。

回溯有一个增量构造答案的过程，这个过程通常用递归来实现。跟之前递归一样，只需要考虑边界条件和非边界条件的逻辑。总结成回溯三问:
1. 当前操作? 用一个 path 数组记录路径上的字母。当前操作为枚举 `path[i]` 要填入的字母。
    > 这里需要注意递归参数中的 i 不是指的第 i 个，而是下标 `>= i` 的这部分。
2. 子问题? 构造字符串 `>= i` 的这部分。
3. 下一个子问题? 构造字符串 `>= i` 的这部分。
整个过程和在一棵树上进行 dfs 是类似的 (dfs(i) --> dfs(i+1)).

## 17. Letter Combinations of a Phone Number

[题目网址](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/). 
- 边界条件: 枚举完当前路径后把 path 添加进 ans.
- 递归条件: 遍历当前要枚举的数字所对应的字母，添加进 path 之后进行下一层次递归。结束后恢复现场。

```cpp
class Solution {
public:
    unordered_map<char, string> phoneMap{
            {'2', "abc"},
            {'3', "def"},
            {'4', "ghi"},
            {'5', "jkl"},
            {'6', "mno"},
            {'7', "pqrs"},
            {'8', "tuv"},
            {'9', "wxyz"}
        };
    string path;
    vector<string> ans;
    void dfs(int i, string digits) {
        if (i == digits.size()) {
            ans.push_back(path);
            return;
        }
        
        for (auto& c : phoneMap[digits[i]]) {
            path += c;
            dfs(i + 1, digits);
            path.pop_back();
        }

    }
    vector<string> letterCombinations(string digits) {
        if (0 == digits.size())
            return vector<string>{};
        dfs(0, digits);
        return ans;
    }
};
```

- 时间复杂度：$O(n4^n)$，其中 n 为 digits 的长度。最坏情况下每次需要枚举 4 个字母，递归次数为一个满四叉树的节点个数，那么一共会递归 $O(4^n)$ (等比数列和) ，再算上加入答案时复制 path 需要 $O(n)$ 的时间，所以时间复杂度为 $O(n4^n)$.
- 空间复杂度：$O(n)$. 返回值的空间不计。

# Subset Backtracking

子集型回溯本质上是看对于每个元素是**选/不选**。

## 78. Subsets

[题目网址](https://leetcode.cn/problems/subsets/). 对于生成子集有两种思路，关键是在于当前操作定义的区别.
- 输入角度: 对数组中的每个元素 nums[i]，进行二元决策：要么选择该元素加入当前子集 (path) ，要么不选择。每次递归处理一个元素，逐步构建所有可能的子集。这样子问题和下一个子问题与上一题的定义一样。
```cpp
void dfs(int i, vector<int>& nums) {
    if (i == nums.size()) {
        ans.push_back(path);
        return;
    }

    // don't choose  nums[i], continue dfs
    dfs(i + 1, nums);

    // choose nuns[i], add to path then dfs, pop when return
    path.push_back(nums[i]);
    dfs(i + 1, nums);
    path.pop_back();
}

=== Testing dfs (Decision-Based) ===
Not choosing nums[0] = 1, before dfs: path: []
  Not choosing nums[1] = 2, before dfs: path: []
    Not choosing nums[2] = 3, before dfs: path: []
      Reached end, adding path: []
    Choosing nums[2] = 3, before dfs: path: [3]
      Reached end, adding path: [3]
  Choosing nums[1] = 2, before dfs: path: [2]
    Not choosing nums[2] = 3, before dfs: path: [2]
      Reached end, adding path: [2]
    Choosing nums[2] = 3, before dfs: path: [2, 3]
      Reached end, adding path: [2, 3]
Choosing nums[0] = 1, before dfs: path: [1]
  Not choosing nums[1] = 2, before dfs: path: [1]
    Not choosing nums[2] = 3, before dfs: path: [1]
      Reached end, adding path: [1]
    Choosing nums[2] = 3, before dfs: path: [1, 3]
      Reached end, adding path: [1, 3]
  Choosing nums[1] = 2, before dfs: path: [1, 2]
    Not choosing nums[2] = 3, before dfs: path: [1, 2]
      Reached end, adding path: [1, 2]
    Choosing nums[2] = 3, before dfs: path: [1, 2, 3]
      Reached end, adding path: [1, 2, 3]
Result: [[], [3], [2], [2, 3], [1], [1, 3], [1, 2], [1, 2, 3]]
```
- 答案视角: 从索引 i 开始，枚举后续所有可能的元素加入子集。每次递归时，将当前路径 path 作为子集加入结果集，然后尝试添加后续元素。为了避免重复子集的出现，需要枚举的下标 `j >= i` 的数。子问题变为下标 `>= i` 的数中构造子集。下一个子问题为从下标 `>= j+1` 的数中构造子集。由于子集的长度没有限制，因此递归到的每个节点都是答案。从 dsf(i) 枚举 dfs(i+1), dfs(i+2), ... , dfs(n).

```cpp
void dfs2(int i, vector<int>& nums) {
    ans.push_back(path);
    
    for (int j = i; j < nums.size(); j++) {
        path.push_back(nums[j]);
        dfs2(j + 1, nums);
        path.pop_back();
    }
    return;
}

=== Testing dfs2 (Enumeration-Based) ===
Adding path: []
Choosing nums[0] = 1, before dfs2: path: [1]
  Adding path: [1]
  Choosing nums[1] = 2, before dfs2: path: [1, 2]
    Adding path: [1, 2]
    Choosing nums[2] = 3, before dfs2: path: [1, 2, 3]
      Adding path: [1, 2, 3]
  Choosing nums[2] = 3, before dfs2: path: [1, 3]
    Adding path: [1, 3]
Choosing nums[1] = 2, before dfs2: path: [2]
  Adding path: [2]
  Choosing nums[2] = 3, before dfs2: path: [2, 3]
    Adding path: [2, 3]
Choosing nums[2] = 3, before dfs2: path: [3]
  Adding path: [3]
Result: [[], [1], [1, 2], [1, 2, 3], [1, 3], [2], [2, 3], [3]]
```

## 131. Palindrome Partitioning

[题目网址](https://leetcode.cn/problems/palindrome-partitioning/). 
- 输入视角: 对于每个索引 i，决定是否在 i 和 i+1 之间插入分割点 (即是否将 `s[last:i+1)` 作为一个回文子串).

```cpp
void dfs(int i, int last, string s) {  // whether choose ',' between i and i+1
    if (i == s.size()) {
        ans.push_back(path);
        return;
    }
    
    if (i < s.size() - 1)  // must choose ',' when i==n-1
        dfs(i + 1, last, s);  //  not ,
    
    if (isPalindrome(s.substr(last, i - last + 1))) {
        path.push_back(s.substr(last, i - last + 1));
        dfs(i+1, i+1, s);
        path.pop_back();
    }
    return;
}
```

若没有 `if (i < s.size() - 1)` 代码仍会执行“不分割”分支 `dfs(i + 1, last, s)`，导致触发终止条件，记录当前 path。
但此时，path 不包含最后一个字符。

- 答案视角: 枚举从 i 到字符串末尾的所有可能子串，检查是否为回文。如果 `s[i:j+1)` 是回文串，加入 path，递归处理剩余部分 (dfs2(j + 1, s)) 。
回溯时移除当前子串，继续尝试其他分割点。

```cpp
void dfs2(int i, string s) {
    if (i == s.size()) {
        ans.push_back(path);
        return;
    }
    for (int j = i; j < s.size(); j++) {  // enumerate all possible ','
        if (isPalindrome(s.substr(i, j - i + 1))) {
            path.push_back(s.substr(i, j - i + 1));
            dfs2(j + 1, s);
            path.pop_back();
        }
    }
    return;
}
```

Q: 为什么触发终止条件时 path 中的子串都是回文？

A: 当 `i == s.size()` 时，说明已经遍历了字符串 s 的所有字符。由于递归过程中只有回文子串才会被加入 path，并且每次分割后 last 会被更新为新的起点，path 中的子串序列覆盖了从字符串开头到结尾的所有字符。

## 784. Letter Case Permutation

[题目网址](https://leetcode.cn/problems/maximum-rows-covered-by-columns/). 直接将当前字符加入 path，递归到下一位。只对处理字母的情况下进行恢复。将 path 中对应位置的字符切换大小写 (大写变小写或小写变大写) ，递归到下一位。
移除当前字符恢复 path 进行回溯，以尝试其他组合。

```cpp
class Solution {
public:
    string path;
    vector<string> ans;

    void dfs(int i, string s) {
        if (i == s.size()) {
            ans.push_back(path);
            return;
        }
        path += s[i];
        dfs(i + 1, s);

        if (isalpha(s[i])) {
            path[i] = islower(s[i]) ? toupper(s[i]) : tolower(s[i]);
            dfs(i + 1, s);
        }
        path.pop_back();

        return;
    }
    vector<string> letterCasePermutation(string s) {
        dfs(0, s);
        return ans;
    }
};
```

## 2397. Maximum Rows Covered by Columns

[题目网址](https://leetcode.cn/problems/maximum-rows-covered-by-columns/). 思路很简单，选/不选每一列，判断能覆盖多少行，这里用位运算进行优化。
- `rowMasks[i]`: 第 i 行的 1 位置表示为位掩码，第 j 位为 1 表示 `matrix[i][j] == 1`.
- 枚举列组合：使用 mask (0 到 $2^n - 1$) 表示所有可能的列选择组合。`__builtin_popcount(mask)` 检查 mask 中 1 的个数从而判断是否选中了 numSelect 列。
- 检查覆盖: 对于每行，若 `rowMasks[i] & mask == rowMasks[i]`，说明该行的所有 1 都在选中的列中 (或该行无 1) .

```cpp
class Solution {
public:
    int maximumRows(vector<vector<int>>& matrix, int numSelect) {
        int m = matrix.size(), n = matrix[0].size();
        vector<int> rowMask(m, 0);  // which col is 1 in row[i]
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (matrix[i][j]) {
                    rowMask[i] |= (1 << j); 
                }
            }
        }

        int maxRows = 0;
        for (int mask = 0; mask < (1 << n); mask++) {  // col th bit 1/0 indicates select col or not 
            if(__builtin_popcount(mask) != numSelect) continue;
            int covered = 0;
            for (int i = 0; i < m; i++) {
                if ((rowMask[i] & mask) == rowMask[i])
                    covered++;
            }
            maxRows = max(maxRows, covered);
        }
        return maxRows;
    }
};
```

## 2212. Maximum Points in an Archery Competition

[题目网址](https://leetcode.cn/problems/maximum-points-in-an-archery-competition/). 递归枚举每个区域是否得分：
- 选择不得分: 一箭不射，跳到下一区域。
- 选择得分：射出 `aliceArrows[i] + 1` 箭，得 i 分，剩余箭数减少。
注意递归到最后一个区域的时候需要将所有箭射出。

{{< details title="Member Variable Initilization in C++" >}}
不能直接在类定义中使用构造函数初始化语法，如 `vector<int> ans(12,0)`. 它被视为函数声明 (由于c++中最令人烦恼的解析) ，而不是变量初始化。编译器期望参数声明器 (例如，函数参数列表) ，导致期望参数声明器出错。

在 c++11 之前，成员变量只能在构造函数的初始化列表中或在构造函数体中初始化。
在 c++11 及更高版本中，可以使用大括号初始化 ({}) 或默认成员初始化来使用类内初始化，但不能使用圆括号 () 。例如:
```cpp
vector<int> ans = vector<int>(12, 0); // Valid, but verbose
vector<int> ans{vector<int>(12, 0)}; // Valid
vector<int> ans = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // Valid
```
{{< /details >}}

```cpp
class Solution {
public:
    int maxScores = 0;
    vector<int> ans;

    void dfs(int i, int numArrows, vector<int>& aliceArrows, int score, vector<int>& bobArrows) {
        if (i == aliceArrows.size()) {
            if (score >= maxScores) {
                maxScores = score;
                ans = bobArrows; // Update ans with current bobArrows
                if (numArrows > 0) {
                    ans[11] += numArrows; // Allocate remaining arrows to region 11
                }
            }
            return;
        }

        // Pruning: Check if remaining score potential can beat maxScores
        int maxPossibleScore = score;
        for (int j = i; j < aliceArrows.size(); ++j) {
            maxPossibleScore += j; // Assume all remaining regions are scored
        }
        if (maxPossibleScore < maxScores || numArrows < 0) return;

        // Option 1: Don't score in region i (use 0 arrows)
        bobArrows[i] = 0;
        dfs(i + 1, numArrows, aliceArrows, score, bobArrows);

        // Option 2: Score in region i (use aliceArrows[i] + 1 arrows)
        if (numArrows >= aliceArrows[i] + 1) {
            bobArrows[i] = aliceArrows[i] + 1;
            dfs(i + 1, numArrows - aliceArrows[i] - 1, aliceArrows, score + i, bobArrows);
            bobArrows[i] = 0; // Backtrack
        }
    }

    vector<int> maximumBobPoints(int numArrows, vector<int>& aliceArrows) {
        vector<int> bobArrows(12, 0); // Initialize bobArrows with size 12
        ans.resize(12, 0); // Initialize ans with size 12
        dfs(0, numArrows, aliceArrows, 0, bobArrows);
        return ans;
    }
};
```

## 2698. Find the Punishment Number of an Integer

[题目网址](https://leetcode.cn/problems/find-the-punishment-number-of-an-integer/). 

- 终止条件: index 到达了字符串 s 的末尾，说明整个字符串已经被成功分割。此时，检查 sum 是否等于目标 i.
- 选/不选回溯: 对于每个位置我们可以选择分割或者不分割。要注意递归到最后一个数字的时候必须进行分割。

```cpp
class Solution {
public:
    int ans = 0;
    bool dfs(int i, string s, int start, int sum, int target) {
        if (i == s.size()) {
           return sum == target;
        }

        if (i < s.size() - 1)  // must select last 
            if (dfs(i + 1, s, start, sum, target))
                return true;

        string subs = s.substr(start, i - start + 1);
        if (dfs(i + 1, s, i + 1, sum + stoll(subs), target))
            return true;

        return false;
    }
    int punishmentNumber(int n) {
        for (int i = 0; i <= n; i++)
            if (dfs(0, to_string(i*i), 0, 0, i))
                ans += i * i;
        return ans;
    }
};
```

# Combination Backtracking

回顾子集问题的搜索树，我们可以发现每一层节点的数字个数是相同的，它们恰好可以表示从 n 个数中选择 1, 2, ..., n 个数的情况。从 n 个数中选 k 个数的**组合**可以看作是**长度固定的子集**。因此如果是找 k 个数的组合，我们可以提前返回而不用继续递归。

## 77. Combinations

[题目网址](https://leetcode.cn/problems/combinations/). 从大到小进行枚举，假设当前路径长度为 m，那么还需要选 d = k - m 个数。如果当前从 i 开始枚举，如果 i < d，最后必然无法选出 k 个数，不需要继续递归，这是一种**剪枝**技巧。

> 这里如果从小到大枚举，判断条件的不等式会稍微复杂一些。

这里仍然可以采用选或不选 以及 枚举两种思路。

```cpp
class Solution {
    vector<int> path;
    vector<vector<int>> ans;

    void dfs(int i, int k) {
        int m = path.size();
        if (m == k) {
            ans.push_back(path);
            return;
        }
            
        if (i < k - m)  // not enough num to add
            return;

        dfs(i - 1, k);  // don't choose i
        
        path.push_back(i);
        dfs(i - 1, k);
        path.pop_back();
        
        return;
    }

    void dfs2(int i, int k) {
        int m = path.size();
        if (m == k) {
            ans.push_back(path);
            return;
        }
            
        if (i < k - m)  // not enough num to add
            return;

        for (int j = i; j > 0; j--) {
            path.push_back(j);
            dfs2(j - 1, k);
            path.pop_back();
        }
        return;
    }
public:
    vector<vector<int>> combine(int n, int k) {
        dfs(n, k);
        // dfs2(n, k);
        return ans;
    }
};
```
回溯的时间复杂度有一个公式: **叶子的个数 x 从根到叶子的路径长度**。因此本题的时间复杂度为 $O(C_{n}^{k}k)$ 空间复杂度为 $O(k)$ 需要一个数组来存储组合数路径。

## 216. Combination Sum III

[题目网址](https://leetcode.cn/problems/combination-sum-iii/). 这题多了一个选出的数目和为 n 的限制，和上题一样设还需要选 d = k - m 个数，使得它们的和为 t (初始为 n，每选一个数就减少). 这里可以剪枝的情况有
1. 剩余数字不够，即 i < d.
2. 当前所选数字之和已经大于 n，即 t < 0.
3. 即使剩余数字全部选最大前几个，它们的和也小于 t，即 $i+\cdots+(i-d+1)=\frac{(i+i-d+1)\cdot d}2 < t$.

```cpp
class Solution {
    vector<int> path;
    vector<vector<int>> ans;

    void dfs(int i, int k, int t) {
        int m = path.size();
        if (m == k && t == 0) {
            ans.push_back(path);
            return;
        }
        int d = k - m;  // need to choose d nums
        if (i < d || t < 0 || (i + i - d + 1) * d / 2 < t)   // cur sum > n or sum of first d th big nums < t
            return;
        dfs(i - 1, k, t);  // don't choose i

        path.push_back(i);
        dfs(i - 1, k, t - i);
        path.pop_back();
        return;
    }

    void dfs2(int i, int k, int t) {
        int m = path.size();
        if (m == k && t == 0) {
            ans.push_back(path);
            return;
        }
        int d = k - m;  // need to choose d nums
        if (i < d || t < 0 || (i + i - d + 1) * d / 2 < t)   // cur sum > n or sum of first d th big nums < t
            return;
        
        for (int j = i; j > 0; j--) {
            path.push_back(j);
            dfs2(j - 1, k, t - j);
            path.pop_back();
        }
        return;
    }
public:
    vector<vector<int>> combinationSum3(int k, int n) {
        dfs(9, k, n);
        // dfs2(9, k, n);
        return ans;
    }
};
```

## 22. Generate Parentheses

[题目网址](https://leetcode.cn/problems/generate-parentheses/). 对于字符串的每个前缀，左括号的个数都需要大于右括号的个数。这题可以看出从 2n 个位置中选 n 个位置放左括号。对于一个位置要么选择放左括号或者选择放右括号。 

```cpp
class Solution {
public:
    string path;
    vector<string> ans;

    void dfs(int left, int diff, int n) {  // choose n pos '(' in 2n
        if (path.size() == 2 * n) {
            ans.push_back(path);
            return;
        }

        if (diff > 0) {  // if num of prefix '(' > num of ')' 
            path += ')';
            dfs(left, diff - 1, n);  // then can choose ')'
            path.pop_back();
        }
        
        if (left < n) {
            path += '(';
            dfs(left + 1, diff + 1, n);
            path.pop_back();
        }
        return;
    }

    vector<string> generateParenthesis(int n) {
        dfs(0, 0, n);
        return ans;
    }
};
```

## 39. Combination Sum

[题目网址](https://leetcode.cn/problems/combination-sum/). 这题因为数字可以重复选择，因此进行递归的时候下标要注意。我们先对 candidates 数组进行排序以方便剪枝。
- 终止条件: 如果 target == 0，说明当前路径的数字和达到目标值，将 path 加入答案 ans。
- 剪枝: 如果 `i == candidates.size()` 或当前递归的最小数字 `candidates[i] > target`，说明无法继续选择，直接返回。


选/不选方法
- 不选当前数字:直接递归调用 `dfs(i + 1, candidates, target)`，跳到下一个数字。
- 选择当前数字: 将 `candidates[i]` 加入 path 后递归调用 `dfs(i, candidates, target - candidates[i])`，注意这里是 i (而非 i + 1)，因为允许重复选择当前数字。回溯时弹出 path 中最后一个数字，恢复状态。

```cpp
void dfs(int i, vector<int>& candidates, int target) {
    if (target == 0) {
        ans.push_back(path);
        return;
    }

    if (i == candidates.size() || candidates[i] > target)
        return;
    
    dfs(i + 1, candidates, target);

    path.push_back(candidates[i]);
    dfs(i, candidates, target - candidates[i]);  // can choose many times
    path.pop_back();
    return;
}
```

枚举下一个方法: 循环从下标 `j = i` 到 `candidates.size() - 1` 枚举从当前下标 i 开始的所有可能数字，逐步构建满足 target 的组合。每次循环尝试选择 candidates[j]，并递归到允许重复选择的状态 `dfs2(j, ...)`.

```cpp
    void dfs2(int i, vector<int>& candidates, int target) {
        if (target == 0) {
            ans.push_back(path);
            return;
        }

        if (i == candidates.size() || candidates[i] > target)
            return;
        
        for (int j = i; j < candidates.size(); j++) {
            path.push_back(candidates[j]);
            dfs2(j, candidates, target - candidates[j]);
            path.pop_back();
        }
        return;
    }
```

## 93. Restore IP Addresses

[题目网址](https://leetcode.cn/problems/restore-ip-addresses/). 用一个 `vector<string>` 来存储当前路径上已经分割好的段。如果 s 本身长度 < 4 或 >12 则说明无法分割直接返回。

首先明确分割的子串是否是一段非法的地址的条件:
1. 子串长度大于 1 且以 '0' 开头。
2. 将子串转为整数的结果大于 255。
3. 子串长度大于 3.

并且可以发现就算延长子串的长度也不能使其合法。

- 终止条件: 当 path 中已经有 4 段时，分割就结束了。此时，还需要检查是否已经用完了整个字符串 (` startIndex == s.length()`). 如果同时满足这两个条件，说明找到了一个合法的 IP 地址。我们将 path 中的 4 段用 . 连接起来，存入最终的结果数组 ans 中。无论是否合法都需要返回。

选/不选思路:
- 剪枝: 如果当前没分割成四段但是已经用完了数组，或者当前对应分割的子串已经非法就直接返回。
- 不分割: dfs(i + 1, ...).
- 分割: 当前子串加入 path 后继续递归 `dfs(i + 1, i + 1, s)` 然后回溯。

```cpp
bool valid(string s) {
    if (s.size() > 3) return false;
    if (s[0] == '0' && s.size() > 1) return false;
    if(stoi(s) > 255) return false;
        
    return true;
}

void dfs(int i, int start, string s) {
    if (path.size() == 4) {
        if (start == s.size()) {
            string ip = path[0] + '.' + path[1] + '.' + path[2] + '.' + path[3];
            ans.push_back(ip);
        }
        return;
    }

    if (i > s.size() - 1)
        return;

    string segment = s.substr(start, i - start + 1);
    if (!valid(segment)) return;

    dfs(i + 1, start, s);

    path.push_back(segment);
        dfs(i + 1, i + 1, s);
        path.pop_back();

    return;
}
```

枚举所有分割点思路:
- 从 start 开始，向后遍历，尝试截取长度为 1/2/3 的子串作为下一个 IP 段。
- 在循环中，需要不断检查当前截取的子串是否合法，不合法就没必要继续深入了。
- 如果当前子串 segment 合法，就把它加入到 path 中继续递归 `dfs2(i + 1, s)` 后回溯。

```cpp
void dfs2(int start, string s) {
    if (path.size() == 4) {
        if (start == s.size()) {
            string ip = path[0] + '.' + path[1] + '.' + path[2] + '.' + path[3];
            ans.push_back(ip);
        }
        return;
    }
    for (int i = start; i < s.size(); i++) {
        string segment = s.substr(start, i - start + 1);
        if (!valid(segment))  // no need to insert .
            break;
        path.push_back(segment);
        dfs2(i + 1, s);
        path.pop_back();
    }
    return;
}
```

# Permutation Backtracking

## 46. Permutations

[题目网址](https://leetcode.cn/problems/permutations/). 相比于组合来说，排列对顺序是有要求的。当选了一个数之后，用一个集合 s 记录剩余未选数字来告诉下面节点还可以选哪些数字。
- 当前操作: 从 s 中枚举 `path[i]` 中要填入的数字 x. 这样就明确了递归的参数为 i 和 s.
- 子问题: 构造排列 >= i 的部分，剩余未选数字集合 s.
- 下一个子问题: 构造排列 >= i + 1 的部分，剩余未选数字集合 s - {x}.

由于 c++ set 删除元素的开销为 $O(\log n)$，这里更高效的操作方法是用一个数组来标数字是否被选中。

```cpp
class Solution {
    vector<int> path;
    vector<vector<int>> ans;

void dfs(vector<int>& nums, vector<bool>& used) {
        if (path.size() == nums.size()) {
            ans.push_back(path);
            return;
        }

        for (int i = 0; i < nums.size(); ++i) {
            if (!used[i]) {  // select unused
                used[i] = true;  // mark as used
                path.push_back(nums[i]);
                dfs(nums, used);
                path.pop_back();
                used[i] = false;  // recovery
            }
        }
    }

public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<bool> used(nums.size(), false);
        dfs(nums, used);
        return ans;
    }
};
```

这里的重点是时间复杂度分析。之前说过可以用叶子节点个数 x 根到叶子节点路径长度来进行估算，按照这种方式得出的时间复杂度为 $O(n*n!)$. 更精确的方式是计算节点的个数，这样也就知道了递归的次数。

最后一层节点个数为 $A_n^n$, 倒数第二层节点个数为 $A_n^{n-1}$...以此类推。根据公式 $A_{n}^{m}=\frac{n!}{(n-m)!}$，我们有

$$
\sum_{k=0}^nA_n^k=\sum_{k=0}^n\frac{n!}{(n-k)!}=n!\sum_{k=0}^n\frac1{(n-k)!}=n!\sum_{m=0}^n\frac1{m!}
$$

其中 $m = n - k$. 这个和可以表示为

$$
\sum_{k=0}^nA_n^k=n!\left(\frac1{0!}+\frac1{1!}+\frac1{2!}+\cdots+\frac1{n!}\right)
$$

求和项为 $e^x$ 在 $x=1$ 处的泰勒展开，因此这个和接近于 $n!\cdot e$，由于节点个数为整数，因此这棵树的节点个数为 $\lfloor n!\cdot e \rfloor$. 再算上把 path 添加进 ans 的时间，总的时间复杂度为 $O(n*n!)$. 空间复杂度为 $O(n)$，使用了额外的数组来记录路径和标记。

## 51. N Queens

[题目网址](https://leetcode.cn/problems/n-queens/). 一个 nxn 的棋盘上要放 n 个皇后并且要求不同行，不同列，不在同一斜线其实就是要求每行每列有且仅有一个皇后。因为如果有一行/列不放皇后，剩下 n-1 行/列就要放 n 个皇后，必然不满足要求。

用一个 col 数组来记录皇后的位置，`col[i]` 表示第 i 行的皇后被放置在第几列。那么 col 本身就是一个 0~n-1 的排列。对于右上方的斜线，行号 r + 列号 c 是一个定值；对于左上方的斜线，行号 r - 列号 c 是一个定值。我们可以从第 0 行开始向下枚举，用两个数组分别标记 r+c 和 r-c 是否有出现过。

```cpp
class Solution {
public:
    vector<int> col;
    vector<vector<string>> ans;
    
    vector<string> buildBoard(const vector<int>& queens, int n) {
        vector<string> board(n, string(n, '.'));
        for (int r = 0; r < n; r++) {
            board[r][queens[r]] = 'Q'; // Place queen in row r, column queens[r]
        }
        return board;
    }

    void dfs(int r, vector<bool>& used, vector<bool> diag1, vector<bool> diag2) {
        int n = used.size();
        if (r == n) {
            ans.push_back(buildBoard(col, n));
            return;
        }

        for (int c = 0; c < n; c++) {
            if (!used[c] && !diag1[r + c] && !diag2[r - c + n - 1]) {
                used[c] = true;
                diag1[r + c] = true;
                diag2[r - c + n - 1] = true;  // avoid negative
                col.push_back(c);
                dfs(r + 1, used, diag1, diag2);
                col.pop_back();
                diag2[r - c + n - 1] = false;
                diag1[r + c] = false;
                used[c] = false;
            }
        }
    }
    vector<vector<string>> solveNQueens(int n) {
        vector<bool> used(n, false);
        vector<bool> diag1(2*n - 1, false);  // right_up
        vector<bool> diag2(2*n - 1, false);  // left_up

        dfs(0, used, diag1, diag2);
        return ans;
    }
};
```

## 357. Count Numbers With Unique Digits

[题目网址](https://leetcode.cn/problems/count-numbers-with-unique-digits). 使用数学组合和动态规划的思路解决。

定义 f(k) 为 k 位数字中，各位数都不同的数的个数。
- f(1) = 9  (1-9)
- f(2) = 9 * 9 (第一位不能是0，第二位不能和第一位相同)
- f(3) = 9 * 9 * 8
- ...
- f(k) = 9 * 9 * 8 * ... * (10 - k + 1)

可以看出 k >= 2 时 f(k) $ = 9A_9^{k-1}$. 由于题目要求的是小于 $10^n$ 的所有个数，因此答案为是 1 (数字0) + f(1) + f(2) + ... + f(n). 

```cpp
class Solution {
public:
    int countNumbersWithUniqueDigits(int n) {
        if (n == 0) return 1;
        int ans = 9, cur = 9;
        for (int i = 0; i < n - 1; i ++) {
            cur *= 9 - i;
            ans += cur;
        }
        return ans + 1;  // add 0
    }
};
```

## 2850. Minimum Moves to Spread Stones over Grid

[题目网址](https://leetcode.cn/problems/minimum-moves-to-spread-stones-over-grid). 这个问题的本质不是模拟每一步移动，而是找到一个最佳的分配方案。
- 源头 (Surplus): 某些单元格的石头数量大于 1。这些是“多余”的石头，可以被移走。一个有 k 个石头的单元格可以提供 k - 1 个石头。
- 目的地 (Deficit): 某些单元格的石头数量为 0。这些是“空缺”的单元格，需要石头。每个这样的单元格需要 1 个石头。
- 石头数为 1 的单元格是平衡的，我们不需要对它们进行任何操作。

由于石头总数是 9，并且目标是每个格子都有 1 个石头，那么**多余石头的总数必然等于空缺格子的总数**。
我们可以创建两个列表：
- `from_list`: 存储所有“多余”石头的起始坐标。如果一个单元格 `(r, c)` 有 k > 1 个石头，我们就把这个坐标重复 k - 1 次加入列表。
- `to_list`: 存储所有“空缺”单元格的坐标。如果 `grid[i][j] == 0`，我们就把 `(i, j)` 加入列表。

经过这个操作后，`from_list` 和 `to_list` 的大小一定是相等的。将一个石头从 (r1, c1) 移动到 (r2, c2)，最少的移动次数等于它们之间的曼哈顿距离 |r1 - r2| + |c1 - c2|. 现在问题就变成了：
如何将 from_list 中的每一个石头与 to_list 中的每一个空格进行一一配对，使得所有配对的曼哈顿距离之和最小？

对于 k 个源头和 k 个目的地，寻找最优匹配的组合总数是 k! 极端情况下一个格子有 9 个石头，其他 8 个格子都是 0. 这时 k = 8，总共组合数有 40320 种情况。

> `std::next_permutation` 是一个定义在 `<algorithm>` 头文件中的函数。它的核心功能是将一个序列 (如 vector、string 或数组) 就地转换为它的下一个字典序排列。

```cpp
class Solution {
public:
    int minimumMoves(vector<vector<int>>& grid) {
        vector<pair<int, int>> from_coords;
        vector<pair<int, int>> to_coords;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (grid[i][j] > 1) {
                    for (int k = 0; k < grid[i][j] - 1; k++) {  // can supply num - 1 stones.
                        from_coords.push_back({i, j});
                    }
                } else if (grid[i][j] == 0) {
                    to_coords.push_back({i, j});
                }
            }
        }

        if (to_coords.empty())
            return 0;
        
        sort(from_coords.begin(), from_coords.end());
        int minMove = INT_MAX;
        do {
            int curMove = 0;
            for (int i = 0; i < from_coords.size(); i++) {
                curMove += abs(from_coords[i].first - to_coords[i].first) + 
                        abs(from_coords[i].second - to_coords[i].second);
            }
            minMove = min(curMove, minMove);
        } while(next_permutation(from_coords.begin(), from_coords.end()));

        return minMove;
    }
};
```