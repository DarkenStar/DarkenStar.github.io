---
title: "04 Chain List"
date: 2025-07-25T13:28:50+08:00
lastmod: 2025-07-25T13:28:50+08:00
author: ["WITHER"]

categories:
- Leetcode

tags:
- Chain List

keywords:


description: "Algorithm questions about chain list." # 文章描述，与搜索优化相关
summary: "Algorithm questions about chain list." # 文章简单描述，会展示在主页
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

# List Reverse Questions
## 206. Reverse Linked List
[题目网址](https://leetcode.cn/problems/reverse-linked-list/).

思路很简单，利用三个指针进行反转，注意移动的顺序，先移动 pre 再移动 cur. 链表的定义如下，后续题目不再重复。

```cpp{linenos=true}{linenos=true}
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur) {  // not reach the end
            ListNode* next = cur->next;  // record next node to reverse
            // be care about order
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
};
```

## 92. Reverse Linked List II
[题目网址](https://leetcode.cn/problems/reverse-linked-list-ii/). 
1. 定位反转区间的前驱节点:

    使用一个哑节点 (dummy node) 指向头节点，以便处理反转从头节点开始的情况，此时反转后原头节点会变成反转区间的尾节点。遍历链表，找到第 left-1 个节点 (即反转区间的前一个节点，记为 p0).

2. 反转指定区间:

    从第 left 个节点到第 right 个节点进行反转，类似于反转整个链表的逻辑，但只操作指定区间。使用三个指针 (pre、cur、next) 进行局部反转：
    - cur 指向当前处理的节点。
    - next 保存 cur 的下一个节点。
    - 将 cur 的 next 指向 pre，实现反转。
    - 移动指针：pre 移到 cur，cur 移到 next，重复直到反转 right - left + 1 次。

3. 连接反转后的区间:

    反转完成后，`p0->next` 指向反转后的区间头节点 (即原来的第 right 节点) 。反转区间的最后一个节点 (原来的第 left 节点) 连接到剩余的链表部分。
4. 返回 `dummy.next`.

- 时间复杂度: $O(n)$，其中 n 是链表长度，遍历一次定位区间，再反转指定区间。
- 空间复杂度: $O(1)$，只使用了常数额外空间。  

```cpp{linenos=true}{linenos=true}
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int left, int right) {
        // initialize a dummy node to handle from head
        ListNode dummy(0, head);
        ListNode* pre = &dummy;
        int cnt = 0;
        while (cnt < left - 1) {
            pre = pre->next;
            cnt++;
        }
        ListNode* p0 = pre;  // record the previous node of the node to be reversed
        pre = nullptr;
        ListNode* cur = p0->next;
        ListNode* next = cur->next;
        while (cnt < right) {
            next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
            cnt++;
        }
        p0->next->next = cur;
        p0->next = pre;
        return dummy.next;  // handle reverse from head
    }
};
```


## 25. Reverse Nodes in k Group
[题目网址](https://leetcode.cn/problems/reverse-nodes-in-k-group/description/). 
1. 遍历一遍链表，得到总长度，这样后续可以很方便地判断剩余节点是否足够形成一个 k 大小的分组。
2. 和上一题一样使用一个 dummy 节点作为虚拟头节点，这极大地简化了对链表头部的操作，无需为第一个 k 分组编写特殊的处理逻辑。
3. 进入主循环，只要剩余节点数 len 大于等于 k，就处理一个分组。在循环内部，通过一个 for 循环来翻转 k 个节点，并在 for 循环结束后，立即将这个翻转好的分组与链表的前后部分连接起来。上一个区间的开头反转后在末尾，即为下一个区间的 p0. 我们要提前记录下它再将反转好的子链表接入。

```cpp{linenos=true}{linenos=true}
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        int len = 0;
        ListNode* cur = head;
        while (cur) {  // record the list length
            len++;
            cur = cur->next;
        }

        ListNode dummy(0, head);
        ListNode* p0 = &dummy, *pre = &dummy;
        cur = pre->next;

        while (len >= k) {
            len -= k;
            for (int i = 0; i < k; i++) {
                ListNode* next = cur->next;
                cur->next = pre;
                pre = cur;
                cur = next;
            }
            // 上一个区间的开头反转后在末尾，即为下一个区间的 p0
            ListNode* next_p0 = p0->next;
            p0->next->next = cur;
            p0->next = pre;
            p0 = next_p0;
        }
        return dummy.next;
    }
};
```

## 24. Swap Nodes in Pairs
[题目网址](https://leetcode.cn/problems/swap-nodes-in-pairs/description/). 思路很简单，用三个指针就可以完成一对的交换，注意判断条件为 `cur && cur->next`，否则不用交换。
```cpp{linenos=true}{linenos=true}
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode dummy(0, head);
        ListNode* pre = &dummy;
        ListNode* cur = head;
        ListNode* next = cur->next;
        while (cur && cur->next) {
            ListNode* next = cur->next;
 
            cur->next = next->next;
            next->next = cur;
            pre->next = next;
            // move to next pair
            pre = cur;
            cur = cur->next;
        }
        return dummy.next;
    }
};
```

## 445. Add Two Numbers II

[题目网址](https://leetcode.cn/problems/add-two-numbers-ii/description/).
1. 调用 reverseList 两次，将 l1 和 l2 都反转。这样，链表的头节点就变成了个位数，问题就转化为了两数相加。
2. 遍历两个反转后的链表，模拟加法过程，生成一个新的结果链表。这个结果链表也是反的 (个位在前).
3. 将这个新生成的结果链表再次反转，得到最终答案。

```cpp{linenos=true}{linenos=true}
class Solution {
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur) {
            // pre->cur->next
            ListNode* next = cur->next;
            cur->next = pre;  // pre<->cur next
            pre = cur;  // cur->pre->next
            cur = next;
        }
        return pre;
    }
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        ListNode* reverseL1 = reverseList(l1);
        ListNode* reverseL2 = reverseList(l2);
        ListNode dummy(0);
        ListNode* pre = &dummy;
        int carry = 0;
        //ListNode* p1 = reverseL1, *p2 = reverseL2;
        while (reverseL1 || reverseL2) {
            int sum = 0;
            if (reverseL1) {
                sum += reverseL1->val;
                reverseL1 = reverseL1->next;
            }
                
            if (reverseL2) {
                sum += reverseL2->val;
                reverseL2 = reverseL2->next;
            }
            sum += carry;
            carry = sum >= 10 ? 1 : 0;
            int value = carry ? sum - 10 : sum;
            pre->next = new ListNode(value);
            pre = pre->next;
        }
        // carry==1, another node
        pre->next = carry ? new ListNode(carry) : nullptr;
        return reverseList(dummy.next);

    }
};
```

## 2816. Double a Number Represented as a Linked List
        
[题目网址](https://leetcode.cn/problems/double-a-number-represented-as-a-linked-list/). 只需一次遍历，利用一个 pre 指针来记录当前节点的前一个节点，从而将在当前节点产生的进位 (最多为1) 加到前一个节上。

1. 创建一个 dummy 节点，并让它指向原始链表的头节点 head。这样做的好处是统一了所有节点的处理逻辑，特别是当原头节点 head 自身需要进位时，进位可以被记录在 dummy 节点上。

2. 使用 cur 指针从 head 开始遍历整个链表。在循环中，先把当前节点 cur 的值乘以 2. 然后检查 cur->val 是否大于等于 10。如果大于 10，说明产生了进位，将 `cur->val` 减去 10，然后将前一个节点 pre 的值加 1 把进位送给了前一位。

3. 遍历结束后，检查 dummy 节点的值。如果 dummy.val 变成了 1，说明整个链表的最高位产生了进位。这时，需要创建一个值为 1 的新头节点，并将其连接到原链表上。否则说明没有最高位进位，直接返回 `dummy.next` (也就是修改后的原链表头节点) 即可。

> 这里要注意 dummy 是一个临时变量无法直接取地址返回需要创建一个新的指向 head 的节点。

```cpp{linenos=true}{linenos=true}
class Solution {
public:
    ListNode* doubleIt(ListNode* head) {
        ListNode* cur = head;
        ListNode dummy(0, head);
        ListNode* pre = &dummy;
        while (cur) {
            ListNode* next = cur->next;
            cur->val *=  2;
            if (cur->val >= 10) {
                cur->val -= 10;
                pre->val++;
            }
            pre = cur;
            cur = next;
        }
        // return dummy.val == 1 ? &dummy : dummy.next;return new ListNode(1, dummy.next);
        // dummy is a tempory variable 
        return dummy.val == 1 ? new ListNode(1, dummy.next) : dummy.next;
    }
};
```

# Fast & Slow Pointers Questions

## 876. Middle of the Linked List

[题目网址](https://leetcode.cn/problems/middle-of-the-linked-list/description/). 核心思想是使用两个指针，slow 和 fast，它们都从链表的头结点开始移动。slow 指针每次向前移动一步，而 fast 指针每次向前移动两步。当 fast 指针到达链表的末尾时，slow 指針正好指向链表的中间结点。

- 链表长度为奇数时: fast 指针最终会停在最后一个节点上 (`fast->next == nullptr`). 此时，slow 指针正好位于链表的正中间。
- 链表长度为偶数时: fast 指针最终会越过链表末尾，变为 nullptr. 根据题目要求，我们需要返回第二个中间结点，而 slow 指针此时正好指向这个结点。

```cpp{linenos=true}
class Solution {
public:
    ListNode* middleNode(ListNode* head) {
        ListNode* slow = head, *fast = head;
        while (fast && fast->next) {  // ensure fast not access nullptr->next
            slow = slow->next;  // move one step per loop
            fast = fast->next->next;
        }
        // if len is odd, fast is the last ele at the end of loop 2k = (n-1) --> k = (n-1)/2 is the mid of list
        //           even         nullptr               2k = n --> k = n/2 is the right mid of list
        return slow;
    }
};
```

## 141. Linked List Cycle

[题目网址](https://leetcode.cn/problems/linked-list-cycle/description/). 同样使用快慢指针
- 如果链表中没有环：fast 指针将首先到达链表的末尾 (即 fast 或 fast->next 变为 nullptr).
- 如果链表中存在环：fast 指针会先于 slow 指针进入环。由于 fast 比 slow 移动得快，它会在环内追赶 slow 指针。因为它们的速度差是固定的，所以 fast 指针最终必然会在环中的某个节点追上并与 slow 指针相遇。

在遍历过程中，不断检查 slow 和 fast 是否指向同一个节点。如果相遇，则说明存在环；如果 fast 指针到达了链表末尾，则说明没有环。

```cpp{linenos=true}
class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode* slow = head, *fast=head;
        while (fast && fast->next) {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow)
                return true;
        }
        return false;
    }
};
```

## 142. Linked List Cycle II

[题目网址](https://leetcode.cn/problems/linked-list-cycle-ii/description/). 解法分为两个阶段:

1. 使用快慢指针判断是否有环并找到相遇点。
2. 从相遇点找到环的入口。当 slow 和 fast 相遇后，我们将任意一个指针重新指向链表头 head，另一个指针保持在相遇点不动。然后，两个指针每次向前移动一步，当它们再次相遇时，相遇的节点就是环的入口。证明如下。

- a: 链表头到环入口的距离为=。
- b: 环的入口到快慢指针相遇点的距离。
- c: 从相遇点到环入口的剩余距离。
那么环的周长为 L = b + c.

当 slow 和 fast 第一次相遇时：
- slow 走过的距离 a + b
- fast 走过的距离 = 2 * (a + b) (因为 fast 的速度是 slow 的两倍). 也可以表示为它比 slow 多走了 n 圈环的长度 (n 是整数，n >= 1):

因此我们有 

$$2\times(a+b)=(a+b)+n\times(b+c) \implies a = (n-1)(b+c) + c$$

意味着从链表头到环入口的距离 a
等于从相遇点绕环 n-1 圈，再走 c 步到达环入口的距离。我们设置两个指针：
- ptr1 从 head 出发，要走 a 步才能到环入口。
- ptr2 从相遇点出发，也要走 a 步 (即 (n-1)L + c 步) 才能到环入口。

它们必然会在环的入口点相遇。

```cpp{linenos=true}
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        ListNode* slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (fast == slow) {
                ListNode* ptr1 = head;
                ListNode* ptr2 = slow; 
                while (ptr1 != ptr2) {
                    ptr1=ptr1->next;
                    ptr2=ptr2->next;
                }
                return ptr1;
            }
        }
        return nullptr;
    }
};
```

## 143. Reorder List
[题目网址](https://leetcode.cn/problems/reorder-list/description/).
1. 使用快慢指针法找到链表的中间节点。
2. 从中间节点的下一个节点开始，将链表的后半部分完全反转。
3. 将反转后的后半部分链表与前半部分链表进行“拉链式”交错合并，直到反转后的后半部分链表为 nullptr.

`1->2->3->4` --> `1->2->3 & 4`  --> `1->(4->2)->3`

```cpp{linenos=true}
class Solution {
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur) {
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    ListNode* middleList(ListNode* head) {
        ListNode* slow = head, * fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;
    }

public:
    void reorderList(ListNode* head) {
        ListNode* mid = middleList(head);
        ListNode* midNext = mid->next;
        mid->next = nullptr;
        ListNode* reverseSecondHead = reverseList(midNext);
        ListNode* p = head;
        while (reverseSecondHead != nullptr) {
            ListNode* firstNext = p->next;  // record first half node
            p->next = reverseSecondHead;
            reverseSecondHead = reverseSecondHead->next;
            p = p->next;
            p->next = firstNext;
            p = p->next;
        }
       
    }
};
```

## 234. Palindrome Linked List

[题目网址](https://leetcode.cn/problems/palindrome-linked-list/description/). 通过反转后半部分链表来实现 $O(1)$ 空间复杂度。

1. 使用快慢指针法找到链表前半部分的尾部。当 fast 到达或越过链表末尾时，slow 正好停在前半部分的尾节点上。
2. 反转后半部分链表：从 slow 的下一个节点开始，即后半部分的头节点，将整个后半部分链表反转。
3. 从两个子链表的头部开始，逐一比较节点的值。如果出现任何不匹配，链表就不是回文。如果两个指针都顺利走完，则说明是回文。

```cpp{linenos=true}
class Solution {
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur) {
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    ListNode* middleList(ListNode* head) {  // return the left node of middle if len is even
        ListNode* slow = head, * fast = head;
        while (fast->next && fast->next->next) { 
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;  
    }   
public:
    bool isPalindrome(ListNode* head) {
        if (!head->next) return true;
        ListNode* firstHalfEnd = middleList(head);
        ListNode* secondHalfStart = reverseList(firstHalfEnd->next); 
        while (head && secondHalfStart) {
            if (head->val !=secondHalfStart->val)
                return false;
            head = head->next;
            secondHalfStart = secondHalfStart->next;
        }
        return true;
    }
};
```

## 2130. Maximum Twin Sum of a Linked List

[题目网址](https://leetcode.cn/problems/maximum-twin-sum-of-a-linked-list/description/). 和上题目一样思路，不再赘述.

```cpp{linenos=true}
class Solution {
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur) {
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }

    ListNode* middleList(ListNode* head) {  // return the left node of middle if len is even
        ListNode* slow = head, * fast = head;
        while (fast->next && fast->next->next) { 
            slow = slow->next;
            fast = fast->next->next;
        }
        return slow;  
    }   
public:
    int pairSum(ListNode* head) {
        ListNode* firstHalfEnd = middleList(head);
        ListNode* secondHalfStart = reverseList(firstHalfEnd->next); 
        int ans = 0;
        while (head && secondHalfStart) {
            int sum = 0;
            sum += head->val + secondHalfStart->val;
            head = head->next;
            secondHalfStart = secondHalfStart->next;
            ans = max(sum, ans);
        }
        return ans;
    }  
};
```

# Node Deletion Questions 

## 237. Delete Node in a Linked list

[题目网址](https://leetcode.cn/problems/delete-node-in-a-linked-list/). `*node = *node->next; `可以称之为狸猫换太子将 node 变成下一个节点，然后把真正的下一个节点从链表中删除。这行代码的本质是将下一个节点对象的内容完整地复制到当前节点对象中。

> 这个方法无法删除链表中的 最后一个节点。因为如果 node 是最后一个节点，node->next 将是 nullptr ，执行 `*node->next` 会导致程序崩溃。不过，题目保证了要删除的节点不会是尾节点。严格意义上，该方法并没有释放原始节点的内存，而是修改了它的内容。

```cpp{linenos=true}
class Solution {
public:
    void deleteNode(ListNode* node) {
        // Copy the content of the next node object completely to the current node object.
        *node = *node->next;
    }
};
```

## 19. Remove nth Node from End of List
[题目网址](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/).
1. 为了能删除 right 指向的节点，我们实际上需要找到它前面的那个节点.创建一个 dummy 节点，让它指向 head. right 和 left 都从 dummy 开始。
2. 先行让 right 指针先向前移动 n 步。然后同时移动 right 和 slow 指针。
3. 当 right 指针到达链表的最后一个元素时，slow 指针所在的位置正好就是要被删除的前一个节点。然后执行删除操作。

## 83. Remove Duplicates from Sorted List

[题目网址](https://leetcode.cn/problems/remove-duplicates-from-sorted-list/).  用一个指针对链表进行遍历，当发现重复元素的时候就不停进行删除，然后移动到下一个节点继续进行判断。这里虽然有两个 while 循环，但最外层循环退出后只遍历了链表一遍，因此时间复杂度为 $O(N)$.

```cpp{linenos=true}
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (!head) return nullptr;
        ListNode* cur = head;
        while (cur && cur->next) {
            while (cur->next && cur->val == cur->next->val)  // handle duplicates
                cur->next = cur->next->next;
            cur = cur->next;
        }
        return head;
    }
};
```

## 82. Remove Duplicates from Sorted List II

[题目网址](https://leetcode.cn/problems/remove-duplicates-from-sorted-list-ii/). 思路和上一题大致相同，区别是因为要删除所有重复元素的节点 (包括自己)，因此需要一个指针记录第一个重复元素的前一个元素的位置。而且可能删除 head，因此要初始化一个 dummy 指向 head.

```cpp{linenos=true}
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        ListNode dummy(101, head);
        ListNode* pre = &dummy;
        ListNode* cur = head;
        while (cur && cur->next) {
            if (cur->val == cur->next->val) {
                while (cur->next && cur->val == cur->next->val) {  // delete duplicates
                    cur->next = cur->next->next;
                }
                pre->next = cur->next;  // delete itself
                cur = pre->next;
            } else {
                pre = cur; 
                cur = cur->next;
            }
        }
        return dummy.next;
    }
};
```

## 203. Remove Linked List Elements

[题目网址](https://leetcode.cn/problems/remove-linked-list-elements/). 同样的思路，如果当前节点的值等于 val，就删除这个节点，否则继续向前移动。

```cpp{linenos=true}
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        if (!head) return nullptr;
        ListNode dummy(-1, head);
        ListNode* pre = &dummy;
        ListNode* cur = head;
        while (cur) {
            if (cur && cur->val == val) {  // delete the target
                pre->next = cur->next;
                cur = cur->next;
            } else {  // move forward
                pre = cur;
                cur = cur->next;
            }
        }
        return dummy.next;
    }
};
```

## 3217. Delete Nodes from Linked List Present in Array

[题目网址](https://leetcode.cn/problems/delete-nodes-from-linked-list-present-in-array/). 一样的思路，先初始化一个哈希表 unordered_set 对数组元素进行去重，然后遍历的时候在其中进行查找判断。

```cpp{linenos=true}
class Solution {
public:
    ListNode* modifiedList(vector<int>& nums, ListNode* head) {
        unordered_set<int> unique_set(nums.begin(), nums.end());
        ListNode dummy(-1, head);
        ListNode* pre = &dummy;
        ListNode* cur = head;
        while (cur) {
            if (cur && unique_set.find(cur->val) != unique_set.end()) {  // delete the target
                pre->next = cur->next;
                cur = cur->next;
            } else {  // move forward
                pre = cur;
                cur = cur->next;
            }
        }
        return dummy.next;
    }
};
```

## 2487. Remove Nodes from Linked List

[题目网址](https://leetcode.cn/problems/remove-nodes-from-linked-list/). 由于正向遍历无法得知当前节点是否需要被删除，因此我们先将链表反转，然后再遍历。若当前节点的值比下一个节点大，则进行删除；否则继续向右移动。最后将删除后的链表再次反转即为答案。

```cpp{linenos=true}
class Solution {
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = nullptr;
        ListNode* cur = head;
        while (cur) {
            ListNode* next = cur->next;
            cur->next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
public:
    ListNode* removeNodes(ListNode* head) {
        ListNode* revHead = reverseList(head);
        ListNode* cur = revHead;
        while (cur->next) {
            if (cur->val > cur->next->val) {
                cur->next = cur->next->next;
            } else {
                cur = cur->next;
            }
        }
        return reverseList(revHead);
    }
};
```

## 1669. Merge in Between Linked Lists

[题目网址](https://leetcode.cn/problems/merge-in-between-linked-lists/). 初始化 left 指向第一个需要被删除的节点的左边的节点，right 指向最后一个需要被删除的节点。然后将 `left->next` 接入 list2，遍历到 list2 的最后一个节点的时候接入 `right->next`.

```cpp{linenos=true}
class Solution {
public:
    ListNode* mergeInBetween(ListNode* list1, int a, int b, ListNode* list2) {
        ListNode dummy(0, list1);

        ListNode* right = &dummy;
        for (int i = 0; i < b + 1; i ++) {  // move to the last node needs to be deleted
            right = right->next;  
        }
        
        ListNode* left = &dummy;
        for (int i = 0; i < a; i++) {  // move to the one step before first node need sto be deleted
            left = left->next;
            right = right->next;
        }
        left->next = list2;

        while (left->next)
            left = left->next;
        
        left->next = right->next;

        return dummy.next;
    }
};
```