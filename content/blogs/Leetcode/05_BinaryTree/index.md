---
title: "05 Binary Tree"
date: 2025-07-28T10:25:56+08:00
lastmod: 2025-07-28T10:25:56+08:00
author: ["WITHER"]

categories:
- Leetcode


tags:
- Binary Tree

keywords:
- 

description: "Algorithm questions about binary tree." # 文章描述，与搜索优化相关
summary: "Algorithm questions about binary tree." # 文章简单描述，会展示在主页
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

# Preliminary: How to Think about the Recursion of Binary Tree

1. 如何思考二叉树相关问题？
- 不要一开始就陷入细节，而是思考整棵树与其左右子树的关系。

2. 为什么需要使用递归？
- 子问题和原问题是相似的，他们执行的代码也是相同的 (类比循环) ，但是子问题需要把计算结果返回给上一级，这更适合用递归实现。

3. 为什么这样写就一定能算出正确答案？
- 由于子问题的规模比原问题小，不断“递”下去，总会有个尽头，即递归的边界条件 (base case)，直接返回它的答案“归”。
- 类似于数学归纳法 (多米诺骨牌) ，n=1时 类似边界条件；n=m 时类似往后任意一个节点

4. 计算机是怎么执行递归的？
- 当程序执行“递”动作时，计算机使用栈保存这个发出“递”动作的对象，程序不断“递”，计算机不断压栈，直到边界时，程序发生“归”动作，正好将执行的答案“归”给栈顶元素，随后程序不断“归”，计算机不断出栈，直到返回原问题的答案，栈空。

5. 另一种递归思路
- 维护全局变量，使用二叉树遍历函数，不断更新全局变量最大值

## 104. Maximum Depth of Binary Tree

[题目网址](https://leetcode.cn/problems/maximum-depth-of-binary-tree/). 
- 递归模型：节点的深度等于其左子树和右子树深度的最大值加上 1 (当前节点本身).
- 递归终止条件：如果节点为空，返回 0.
对于每个节点，递归计算左子树和右子树的深度，然后返回其中最大值 + 1.

```cpp
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (root == nullptr)  // base case
            return 0;
        // treeHight = max(leftTreeHeight, rightTreeHeight) + 1
        int leftTreeHeight = maxDepth(root->left);
        int rightTreeHeight = maxDepth(root->right);

        return max(leftTreeHeight, rightTreeHeight) + 1;
    }
};
```

二叉树的节点数为 N，递归过程中每个节点恰好被访问一次，因此时间复杂度为 $O(N)$. 空间复杂度取决于递归调用栈的深度。下最坏情况二叉树为一个单链表形式的树栈的深度为 N，因此空间复杂度为 $O(N)$.

## 111. Minimum Depth of Binary Tree

[题目网址](https://leetcode.cn/problems/minimum-depth-of-binary-tree/). 需要特别注意：
- 叶子节点是没有子节点的节点 (即 `!node->left && !node->right`).
- 如果一个节点只有一棵子树 (左子树或右子树为空) ，不能直接取另一棵子树的深度，因为非叶子节点的深度不取决于非叶子 子节点。

```cpp
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root) return 0;

        if (!root->left) return minDepth(root->right) + 1;  // only has right
        if (!root->right) return minDepth(root->left) + 1;  // only has left
        // left and right both exist
        return min(minDepth(root->left), minDepth(root->right)) + 1;
    }
};
```

## 112. Path Sum

[题目网址](https://leetcode.cn/problems/path-sum/). 
- 递归模型：左子树或者有一条和为 `targetSum - node->val` 的路径
- 递归终止条件：如果是叶子节点，则判断其值是否等于 targetSum.

对于每个节点，递归计算左子树和右子树的深度，然后返回其中最大值 + 1.

```cpp
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (!root) return false;
        if (!root->left && !root->right)
            return root->val == targetSum;

        return hasPathSum(root->left, targetSum - root->val) || hasPathSum(root->right, targetSum - root->val);
    }
};
```

## 129. Sum Root to leaf Numbers

[题目网址](https://leetcode.cn/problems/sum-root-to-leaf-numbers/). 这题的思路有点像秦九韶算法，递归函数 `dfs` 功能是计算以当前节点为根的所有从根到叶节点的路径数字之和。对于当前节点 root，新的路径数字为 `curr * 10 + root->val`.

```cpp
class Solution {
    int dfs(TreeNode* root, int curr) {
        if (!root) return 0; 
        curr = curr * 10 + root->val;
        if (!root->left && !root->right) return curr;
        return dfs(root->left, curr) + dfs(root->right, curr);
    }
public:
    int sumNumbers(TreeNode* root) {
        return dfs(root, 0);
    }
};
```

## 1448. Count Good Nodes in Binary Tree

[题目网址](https://leetcode.cn/problems/count-good-nodes-in-binary-tree/). 
- 递归模型：根节点的左子树好节点数目加上右子树好节点数目加上当前节点本身是不是。函数传入一个变量不断更新当前经过路径中的最大值。
- 递归终止条件：如果节点为空，返回 0.

```cpp
class Solution {
    int dfs(TreeNode* node, int curMax) {  // left num + right num + if self
        if (!node) {
            return 0;
        }
        int isSelf = 0;
        if (curMax <= node->val) {
            isSelf = 1;
            curMax = node->val;
        } 
        return dfs(node->left, curMax) + dfs(node->right, curMax) + isSelf;
    }

public:
    int goodNodes(TreeNode* root) {
        return dfs(root, root->val);
    }
};
```

## 987. Vertical Order Traversal of a Binary Tree

[题目网址](https://leetcode.cn/problems/vertical-order-traversal-of-a-binary-tree/). 题目对于返回值的要求为按列索引 col 从小到大。同一列内，节点按行号 row 升序排序；若行号相同，则按节点值 val 升序排序。我们可以用递归方式遍历每个节点，构建三元组 (col, row, val). 然后对这个三元组进行排序后分组构建答案。

```cpp
class Solution {
    void dfs(TreeNode* root, vector<tuple<int, int, int>>& nodes, int row, int col) {
        if (!root) return;
        nodes.emplace_back(col, row, root->val); // record (col, row, val)
        dfs(root->left, nodes, row + 1, col - 1); 
        dfs(root->right, nodes, row + 1, col + 1); 
    }
public:
    vector<vector<int>> verticalTraversal(TreeNode* root) {
        vector<tuple<int, int, int>> nodes; //  (col, row, val)
        dfs(root, nodes, 0, 0);

        sort(nodes.begin(), nodes.end());

        vector<vector<int>> ans;
        if (nodes.empty()) return ans;

        ans.push_back({}); // first col
        int lastCol = get<0>(nodes[0]); // determin whether node is in another col 
        ans.back().push_back(get<2>(nodes[0]));

        for (int i = 1; i < nodes.size(); ++i) {
            int col = get<0>(nodes[i]);
            int val = get<2>(nodes[i]);
            if (col != lastCol) { // another col
                ans.push_back({});
                lastCol = col;
            }
            ans.back().push_back(val); 
        }

        return ans;
    }
};
```

## 100. Same Tree
[题目网址](https://leetcode.cn/problems/same-tree/). 
- 递归模型: 两棵树相同的条件为，它们的值相同并且左右子树也都相同。
- 边界条件: 如果当前两个根节点有一个为空，若相同则要求另一个也为空。

```cpp
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (!p || !q) {
            return p == q;
        }
        // 2 trees are same if their
        // val are same and left trees and right trees are all same
        return p->val == q->val && isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};
```

## 101. Symmetric Tree
[题目网址](https://leetcode.cn/problems/symmetric-tree/). 
- 递归模型: 一棵树对称的条件为，其左子树的左子树和右子树的右子树相同并且左子树的右子树和右子树的左子树相同。我们对上一题传入的参数稍加修改，变成比较一棵树的左子树和另一棵树的右子树是否相等，并且其右子树和另一棵树的左子树相等。
- 边界条件: 如果当前有一颗子树为空，若对称则要求另一个也为空。

```cpp
class Solution {
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (!p || !q)
            return p == q;
        return p->val == q->val && isSameTree(p->left, q->right) && isSameTree(p->right, q->left);
    }
public:
    bool isSymmetric(TreeNode* root) {
        return isSameTree(root->left, root->right);
    }
};
```

## 110. Balanced Binary Tree
[题目网址][题目网址](https://leetcode.cn/problems/balanced-binary-tree/). 
- 递归模型: 平衡二叉树的定义天然符合递归使用条件，对于每个节点，左右子树的高度差绝对值 ≤ 1 并且左右子树也必须是平衡二叉树。 
    - 可以通过递归计算每个节点的高度，同时检查是否平衡。如果子树不平衡，返回 -1 表示不平衡。
    - 不平衡的条件为左右子树有一颗不平衡或者当前根节点的左右子树高度相差 > 1.
- 边界条件: 如果当前节点为空，返回其高度 0.

```cpp
class Solution {
    int getHeight(TreeNode* root) {  // if not balanced return -1
        if (!root)
            return 0;
        int leftHeight = getHeight(root->left);
        int rightHeight = getHeight(root->right);
        if (leftHeight == -1 || rightHeight == -1 || abs(leftHeight - rightHeight) > 1)
            return -1;
        return max(leftHeight, rightHeight) + 1;
    }
public:
    bool isBalanced(TreeNode* root) {
        return getHeight(root) != -1;
    }
};
```

## 199. Binary Tree Right Side View
[题目网址](https://leetcode.cn/problems/binary-tree-right-side-view/). 
使用前序遍历，通过递归深度跟踪当前节点的层级。
如果当前层级深度等于结果数组的大小，说明这是该层第一次访问的节点，将其值加入。
优先递归右子树，然后左子树，确保每层最右节点优先被记录。

```cpp
class Solution {
    void dfs(TreeNode* root, int d) {
        if (!root)
            return;
        if (d == ans.size())
            ans.push_back(root->val);
        dfs(root->right, d + 1);
        dfs(root->left, d + 1);
    }
public:
    vector<int> ans;
    vector<int> rightSideView(TreeNode* root) {
        dfs(root, 0);
        return ans;      
    }
};
```

## 965. 

- 递归模型: 单值二叉树的左右子树一定也是单值二叉树。将根节点的值层层下传进行比对。
- 边界条件: 如果当前节点为空，返回 true. 否则判断当前节点值是否等于 target. 若相等再递归判断左右子树。

```cpp
class Solution {
    bool dfs(TreeNode* root, int target) {
        if (!root)
            return true;
        if (root->val != target)
            return false;
        return dfs(root->left, target) && dfs(root->right, target);
    }
public:
    bool isUnivalTree(TreeNode* root) {
        return dfs(root, root->val);
    }
};
```

## 226.

- 递归模型: 对于一棵已经翻转的二叉树，它的左右子树和原先相比都是翻转的。我们翻转当前根节点的左右子树后就进行递归的翻转。
- 边界条件: 如果当前节点为空，不用进行翻转，返回 nullptr.

```cpp
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return nullptr;
        // reverse current root
        TreeNode * tmp = root->left;
        root->left = root->right;
        root->right = tmp;
        // recursive
        invertTree(root->right);
        invertTree(root->left);
        return root;
    }
};
```

## 617. 
- 递归模型: 合并后的二叉树的是对其左右子树进行合并后的结果。两节点均非空时，将两个节点的值相加。
然后递归合并左右子树，这里选择直接合并到 root1 上，将返回值赋值给 root1->left 和 root1->right.
- 边界条件: 当一棵树为空时，无需继续合并直接返回另一棵树。

```cpp
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        if (!root1) return root2;
        if (!root2) return root1;
        
        // merge current node
        root1->val += root2->val;
        
        // merge to root1
        root1->left = mergeTrees(root1->left, root2->left);
        root1->right = mergeTrees(root1->right, root2->right);
        
        return root1;
    }
};
```

## 1080.
- 递归模型: 递归函数采用后序遍历，传入从根节点到当前节点为止的和。递归处理左右子树，获取子树是否有效。如果左右子树都被删除说明当前节点无有效路径，删除当前节点。否则，保留节点。
- 边界条件: 如果是空节点，返回 nullptr. 如果是叶子节点，检查路径和 `sum + node->val < limit`. 如果是，返回 nullptr.否则返回该节点。

```cpp
class Solution {
    TreeNode* dfs(TreeNode* root, int sum, int limit) {
        if (!root)
            return 0;
        if (!root->left && !root->right) {
            if (root->val + sum < limit)
                return nullptr;
            else
                return root;
        }
        root->left = dfs(root->left, sum + root->val, limit);
        root->right = dfs(root->right, sum + root->val, limit);
        if (!root->left && !root->right)
            return nullptr;
        return root;
    }
public:
    TreeNode* sufficientSubset(TreeNode* root, int limit) {
        return dfs(root, 0 , limit);
    }
};
```

# Binary Search Tree

二叉搜索树定义如下：
- 节点的左子树只包含 严格小于 当前节点的数。
- 节点的右子树只包含 严格大于 当前节点的数。
- 所有左子树和右子树自身必须也是二叉搜索树

## 98.

- 前序遍历：先判断再递归，递归函数除了要传入根节点之外还要在传入两个值表示开区间的范围，先判断当前节点是否在范围内。如果是则递归左子树，将开局间右边界替换为当前节点的值。然后再递归右子树，将开局间左边界替换为当前节点的值。
- 中序遍历：先递归遍历左子树，再访问根节点，再递归遍历右子树得到的是一个升序的数组。因此我们记录访问到的上一个节点的值然后进行比较。
- 后序遍历：先递归再判断。从子树开始验证，向上汇总信息。每个节点返回其子树的有效性及子树值的范围 `[min_val, max_val]`. 遍历左右子树获取他们的最小值和最大值范围，然后检查当前节点是否满足左子树最大值 < 当前节点值 < 右子树最小值。若不满足条件，返回 `{LLONG_MIN, LLONG_MAX}` 用于进行判断，否则更新当前根节点的最小值和最大值范围后返回。

```cpp
class Solution {
    bool preOrder(TreeNode* root, long long left, long long right) {
        if (!root)
            return true;
        if (root->val <= left || root->val >= right)
            return false;
        return preOrder(root->left, left, root->val) && preOrder(root->right, root->val, right);
    }

    bool midOrder(TreeNode* root, long long& pre) {  // reference
        if (!root)
            return true;
        if (!midOrder(root->left, pre))
            return false;
        if (root->val <= pre)
            return false;
        pre = root->val;  // record the previous node val.
        return midOrder(root->right, pre);
    }

    pair<long long, long long>  postOrder(TreeNode* root) {  // return cur tree min, max
        if (!root) 
            return {LLONG_MAX, LLONG_MIN};
        auto[lMin, lMax] = postOrder(root->left);
        if (root->val <= lMax)   
            return {LLONG_MIN, LLONG_MAX};
        auto[rMin, rMax] = postOrder(root->right);
        if (root->val >= rMin)   
            return {LLONG_MIN, LLONG_MAX};
        return {min(lMin, (long long)root->val), max(rMax, (long long)root->val)};
    }

public:
    bool isValidBST(TreeNode* root) {
        //return preOrder(root, LLONG_MIN, LLONG_MAX);

        // long long x = LLONG_MIN;
        // return midOrder(root, x);

        return postOrder(root).second != LLONG_MAX;
    }
};
```

## 938. Range Sum of BST

[题目网址](https://leetcode.cn/problems/range-sum-of-bst/). 从根节点开始，判断当前节点值是否在范围内。根据 BST 性质：
- 如果 val < low，跳过左子树 (左子树所有值都 < low).
- 如果 val > high，跳过右子树 (右子树所有值都 > high).
- 否则，递归处理左右子树，并累加当前节点值。

```cpp
class Solution {
public:
    int rangeSumBST(TreeNode* root, int low, int high) {
        if (!root)
            return 0;
        if (root->val < low)
            return rangeSumBST(root->right, low, high);
        if (root->val > high) 
            return rangeSumBST(root->left, low, high);
        return root->val + rangeSumBST(root->left, low, high) + rangeSumBST(root->right, low, high);
    }
};
```

## 105. Construct Binary Tree From Preorder and Inorder Traversal

[题目网址](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/). 对于任意一颗树而言，前序遍历的形式总是 `[ 根节点, [左子树的前序遍历结果], [右子树的前序遍历结果] ]`. 根节点总是前序遍历中的第一个节点。而中序遍历的形式总是 `[ [左子树的中序遍历结果], 根节点, [右子树的中序遍历结果] ]`. 根节点将中序遍历分为左子树和右子树两部分。

核心思想是通过前序遍历可以确定根节点，通过中序遍历可以确定左子树和右子树的范围。

- 递归模型: 从前序遍历的第一个元素得到根节点。在中序遍历中找到根节点的位置，根节点左侧是左子树，右侧是右子树。根据中序遍历中左子树和右子树的节点数量，分割前序遍历数组，分别递归构建左子树和右子树。为了快速找到中序遍历中根节点的位置，可以使用哈希表存储中序遍历的元素及其索引，降低时间复杂度。
- 边界条件: 是当子树范围为空时返回 nullptr.

时间复杂度：$O(N)$，其中 N 是节点数。哈希表查找和递归处理每个节点一次。空间复杂度：$O(N)$，用于哈希表和递归调用栈。

```cpp
class Solution {
    TreeNode* buildTreeHelper(vector<int>& preorder, int preStart, int preEnd,
                             vector<int>& inorder, int inStart, int inEnd,
                             unordered_map<int, int>& inorderMap) {
        if (preStart > preEnd || inStart > inEnd)
            return nullptr;
        // the first ele in preorder is root
        TreeNode* root = new TreeNode(preorder[preStart]);

        int rootIndex = inorderMap[root->val];
        int leftTreeNum = rootIndex - inStart;

        root->left = buildTreeHelper(preorder, preStart + 1, preStart + leftTreeNum, inorder, inStart, rootIndex - 1, inorderMap);
        root->right = buildTreeHelper(preorder, preStart + leftTreeNum + 1, preEnd, inorder, rootIndex + 1, inEnd, inorderMap);
        return root;
    }
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        // record the index in the inorder
        unordered_map<int, int> inorderMap;
        for (int i = 0; i < inorder.size(); i++)
            inorderMap[inorder[i]] = i;
        return buildTreeHelper(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1, inorderMap);
    }
};
```

## 106. Construct Binary Tree From Inorder and Postorder Traversal

[题目网址](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/). 思路和上一题相同，后序遍历的形式总是 `[ [左子树的前序遍历结果], [右子树的前序遍历结果], 根节点 ]`. 取出后序遍历最后一个元素作为根节点之后通过中序遍历确定其左右子树的数目。

```cpp
class Solution {
    unordered_map<int, int> inorderMap;
    TreeNode* buildTreeHelper(vector<int>& inorder, int inStart, int inEnd,
                             vector<int>& postorder, int postStart, int postEnd
                            ) {
        if (inStart > inEnd)
            return nullptr;
        TreeNode* root = new TreeNode(postorder[postEnd]);
        int rootIndex = inorderMap[root->val];
        int leftTreeNum = rootIndex - inStart;
        root->left = buildTreeHelper(inorder, inStart, rootIndex - 1, 
                                    postorder, postStart, postStart + leftTreeNum - 1);
        root->right = buildTreeHelper(inorder, rootIndex + 1, inEnd, 
                                    postorder, postStart + leftTreeNum, postEnd - 1);
        return root;
    }
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        //unordered_map<int, int> inorderMap;
        for (int i = 0; i < inorder.size(); i++)
            inorderMap[inorder[i]] = i;
        return buildTreeHelper(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
    }
};
```

## 889. Construct Binary Tree From Preorder and Postorder Traversal

[题目网址](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/). 如果只知道前序遍历和后序遍历，这棵二叉树不一定是唯一的。因为我们无法确定左右子树的边界在哪。这里假设 `preorder[preStart + 1]` (即前序遍历的第二个元素) 是左子树的根节点，并找到其在后序遍历中的位置来分割子树。这样问题就转换成之前两题的形式，

```cpp
class Solution {
    unordered_map<int, int> postorderMap;
    TreeNode* helper(vector<int>& preorder, int preStart, int preEnd, vector<int>& postorder, int postStart, int postEnd) {
        if (preStart > preEnd)
            return nullptr;
        
        TreeNode* root = new TreeNode(preorder[preStart]);
        if (preStart == preEnd)
            return root;
            
        int leftVal = preorder[preStart + 1];  // assume preorder[1] is root->left
        int leftIndex = postorderMap[leftVal];
        int leftTreeNum = leftIndex - postStart + 1;
        root->left = helper(preorder, preStart + 1, preStart + leftTreeNum,
                            postorder, postStart, postStart + leftTreeNum - 1);
        root->right = helper(preorder, preStart+ leftTreeNum + 1, preEnd,
                            postorder, postStart + leftTreeNum, postEnd - 1);
        return root;
    }
public:
    TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder) {
       for (int i = 0; i < postorder.size(); i++) {
            postorderMap[postorder[i]] = i;
       }
       return helper(preorder, 0, preorder.size() - 1, postorder, 0, postorder.size() - 1);
    }
};
```

## 1110. Delete Nodes and Return Forest

[题目网址](https://leetcode.cn/problems/delete-nodes-and-return-forest/). 这道题很自然的要想到使用后序遍历，因为如果先删除了根节点，我们无法追踪它的左右子树是否需要被删除。所以我们先删除其左右子树，再判断当前节点是否需要删除。若需要删除将返回为非空的左右子树加入答案，返回 nullptr. 否则返回根节点.

注意递归结束后我们需要判断题目本身的跟根节点是否需要被删除，如果不会被删除也要将其加入答案。

```cpp
class Solution {
    vector<TreeNode*> ans;
    std::unordered_set<int> hashSet;
    TreeNode* dfs(TreeNode* root) {
        if (!root)
            return nullptr;
        root->left = dfs(root->left);
        root->right = dfs(root->right);

        // if current root need to be deleted,
        // append its no empty left and right to ans
        if (hashSet.find(root->val) != hashSet.end()) {
            if (root->left)
                ans.push_back(root->left);
            if (root->right)
                ans.push_back(root->right);
            return nullptr;
        }
         return root;   
    }
public:
    vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete) {
        hashSet = std::unordered_set<int>(to_delete.begin(), to_delete.end());

        if (dfs(root))
            ans.push_back(root);
        return ans;
    }
};
```

## 236. Lowest Common Ancestor of a Binary Tree

[题目地址](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/). 题目要求找最近的公共祖先，所以递归的时候我们要自底向上。

对于当前节点 root，递归检查左右子树：
- 如果当前节点为空，直接返回 nullptr. 
- 如果 p 或 q 等于当前节点，则当前节点是候选节点，直接返回 root.

> 如果下面有 q 或 p，那么当前节点就是最近公共祖先，直接返回当前节点。如果下面没有 q 和 p，那既然都没有要找的节点了，也不需要递归，直接返回当前节点。

- 递归查询左子树和右子树，分别返回左子树和右子树中是否找到 p 或 q.
    - 如果左右子树都找找到 p 或 q，则当前节点 root 是 LCA.
    - 如果仅在左子树或右子树找到 p 或 q，则返回找到的那个节点。
    - 如果左右子树都没有找到，返回 nullptr.

> 返回值的准确含义是「最近公共祖先的候选项」。对于最外层的递归调用者来说，返回值是最近公共祖先的意思。但是，在递归过程中，返回值可能是最近公共祖先，也可能是空节点 (表示子树内没找到任何有用信息)、节点 p 或者节点 q (可能成为最近公共祖先，或者用来辅助判断上面的某个节点是否为最近公共祖先).

```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        // if we can find p or q in its subtree, then root is a candidate
        // otherwise no need to continue recursiving.
        if (!root || root == p || root == q)
            return root;

        TreeNode* left = lowestCommonAncestor(root->left, p, q);
        TreeNode* right = lowestCommonAncestor(root->right, p, q);

        if (left && right)  // current node is a candidate
            return root;
        return left ? left : right;
    }
};
```

## 235. Lowest Common Ancestor of a Binary Search Tree

[题目地址](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/). 跟上题思路一样，只不过我们可以利用 BST 的性质，如果当前节点的值大于 (小于) p 和 q 的值就可以只用在其左 (右) 子树中进行递归；

```cpp
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q)
            return root;
        TreeNode *left = nullptr, *right = nullptr;
        if (root->val > p->val && root->val > q->val) {  // only need find left subtree
            left = lowestCommonAncestor(root->left, p, q);
        } else if (root->val < p->val && root->val < q->val) {  // only need find right subtree
           right = lowestCommonAncestor(root->right, p, q);
        } else {
            left = lowestCommonAncestor(root->left, p, q);
            right = lowestCommonAncestor(root->left, p, q);
        }
        if (left && right)  // current node is a candidate
            return root;
        return left ? left : right;
    }
};
```

## 1123. Lowest Common Ancestor of Deepest Leaves

[题目地址](https://leetcode.cn/problems/lowest-common-ancestor-of-deepest-leaves/). 题目中所说最深叶子节点的最近公共祖先意思是包含所有最深叶子节点的最小子树的根节点。
我们一样使用后序遍历自底向上访问，对于任意一个节点 node:
1. 如果 node 的左子树包含所有最深的叶子节点，那么我们应该在左子树里继续寻找答案。
2. 如果 node 的右子树包含所有最深的叶子节点，那么我们应该在右子树里继续寻找答案。
3. 如果 node 的左、右子树都包含了最深的叶子节点 (也就是说，最深的叶子节点分布在左右两侧)，那么这个 node 本身就是我们要找的答案——最近公共祖先。

我们可以通过子树的深度来判断属于以上哪种情况。对于一个节点 node，我们计算它左子树的最大深度 left_depth 和右子树的最大深度 right_depth.
1. 如果 `left_depth > right_depth`，说明这个子树里最深的叶子只存在于左边。
2. 如果 `right_depth > left_depth`，说明这个子树里最深的叶子只存在于右边。
3. 如果 `left_depth == right_depth`，说明最深的叶子分布在左右两边，那么当前节点 node 就是其子树范围内最深叶子的 LCA.

因此设计的递归函数需要返回两个信息:
1. 当前 node 为根的子树中，最深叶子节点的 LCA 是谁。
2. 当前 node 为根的子树的最大深度是多少。

```cpp
class Solution {
    pair<TreeNode*, int> dfs(TreeNode* root) {
        if (!root)
            return pair<TreeNode*, int>{nullptr, 0};
        auto[leftLca, leftDepth] = dfs(root->left);
        auto[rightLca, rightDepth] = dfs(root->right);
        if (leftDepth > rightDepth) {
            return pair<TreeNode*, int>{leftLca, leftDepth + 1};
        } else if (leftDepth < rightDepth) {
            return pair<TreeNode*, int>{rightLca, rightDepth + 1};
        } else {
            return pair<TreeNode*, int>{root, rightDepth + 1};
        }
       
    }
public:
    pair<TreeNode*, int> dfs(TreeNode* root) {
        if (!root)
            return pair<TreeNode*, int>{nullptr, 0};
        auto[leftLca, leftDepth] = dfs(root->left);
        auto[rightLca, rightDepth] = dfs(root->right);
        if (leftDepth > rightDepth) {
            return pair<TreeNode*, int>{leftLca, leftDepth + 1};
        } else if (leftDepth < rightDepth) {
            return pair<TreeNode*, int>{rightLca, rightDepth + 1};
        } else {
            return pair<TreeNode*, int>{root, rightDepth + 1};
        }
       
    }
    TreeNode* lcaDeepestLeaves(TreeNode* root) {
        return dfs(root).first;
    }
};
```

## 2096. Step-By-Step Directions From a Binary Tree Node to Another

[题目地址](https://leetcode.cn/problems/step-by-step-directions-from-a-binary-tree-node-to-another/). 在二叉树中，两个节点之间的最短路径必然经过它们的 LCA. 两点之间的路径可以分为两部分：
- 从 startValue 向上到 LCA (全是 'U')。
- 从 LCA 向下到 destValue (全是 'L' 或 'R')。

整体流程为
1. 使用 DFS 分别找到从根到 startValue 和从根到 destValue 的路径 (记录 'L' 和 'R').
2. 比较两条路径，找到第一个不同的位置 (对应 LCA 的子树分叉点)。
3. 对于 startValue 到 LCA 的路径，转换为 'U' (因为需要向上走).
4. 对于 LCA 到 destValue 的路径，直接使用 'L' 和 'R'.

```cpp
class Solution {
    string ans;
    bool findPath(TreeNode* root, int val, string& path) {
        if (!root)
            return false;
        if (root->val == val)
            return true;
        // try left tree
        path.push_back('L');
        if (findPath(root->left, val, path))
            return true;
        path.pop_back();
        // try right tree
        path.push_back('R');
        if (findPath(root->right, val, path))
            return true;
        path.pop_back();
        
        return false;
    }
public:
    string getDirections(TreeNode* root, int startValue, int destValue) {
        string startPath, destPath, ans;
        findPath(root, startValue, startPath);
        findPath(root, destValue, destPath);
        int cnt = 0;
        int len = min(startPath.size(), destPath.size());
        for (int i = 0; i < len; i++) {
            if (startPath[i] != destPath[i])
                break;
            cnt++;
        }
        // substitute path from start to LCA with U
        ans = string(startPath.size() - cnt, 'U');
        // add remaining L and R from LCA to dest
        ans += destPath.substr(cnt);
        return ans;
    }
};
```