---
title: How to Use git rebase
date: 2025-06-06T17:38:00+08:00
lastmod: 2025-06-06T17:38:00+08:00
draft: false
author: ["WITHER"]
keywords: 
    - Git
categories:
    - Git Learning
tags:
    - Git Learning
description: Use of git rebase
summary: Use of git rebase
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---
# What Can git rebase Do

`rebase` 的字面意思是“变基”——也就是改变一个分支的“基础”提交点。它的主要目标是：将一系列的提交以更整洁、线性的方式应用到另一个分支上，从而创造一个干净、没有多余合并记录的项目历史。

假设你的项目历史是这样的：你在 main 分支上切出了一个 feature 分支，之后 main 分支和你自己的 feature 分支都有了新的 commits.

```plaintext {linenos=true}
      A---B---C   <-- feature
     /
D---E---F---G   <-- main
```
如果你在 feature 分支上运行 git rebase main，Git 会做一件非常神奇的事：

1. Git 会暂时“收起” feature 分支上的所有提交 (A, B, C).
2. 将 feature 分支的起点移动到 main 分支的最新提交 G 上。
3. 把刚才收起的提交 (A, B, C) 依次重新应用到新的起点上，形成新的提交 A', B', C'

```plaintext {linenos=true}
              A'--B'--C'  <-- feature
             /
D---E---F---G   <-- main
```
**A' 和 A 的内容虽然一样，但它们的 Commit ID 是不同的，因为它们的父提交变了。rebase 相当于重写了历史。**

现在，再切换回 main 分支，执行 `git merge feature`，由于 main 分支的所有历史现在是 feature 分支历史的子集，Git 只会进行一次 Fast-forward 合并，不会产生新的合并提交。最终结果如下

```plaintext {linenos=true}
D---E---F---G---A'--B'--C'  <-- main, feature
```

最终的项目历史是一条完美的直线，非常清晰，就像所有开发都是按顺序发生的一样。rebase 重写了历史，抹去了分支开发的“并行”痕迹。

# Compared to merge

要理解 rebase，最好的方法就是和 merge 对比。如果在 main 分支上运行 `git merge feature`，结果会是这样

```plaintext {linenos=true}
      A---B---C
     /         \
D---E---F---G---H   <-- main (H 是一个合并提交)
```

`merge` 做的事情是：
1. 找到两个分支的共同祖先 E.
2. 将两个分支的修改整合起来，创建一个全新的 Merge Commit，也就是 H. 该提交有两个父提交点 C 和 G.

merge 完全全保留了历史的真实性。它清楚地记录了“在某个时间点，我们把一个分支合并了进来”。但如果项目频繁合并，历史记录会充满大量的合并提交，形成一个复杂的“菱形”或“意大利面条”式的网状结构，难以阅读。

# How to use rebase 

假设你正在 feature-login 分支上开发，同时主分支 main 也有了新的更新。

1. 确保 main 分支处于最新的状态 

```bash {linenos=true}
git checkout main
git pull origin main
```

2. 切换到你正在开发的分支 `git checkout feature-login`
3. 把 main 分支上的最新修改 rebase 到你当前的 feature-login 分支上 `git rebase main`
4. 解决冲突 (如果有的话). 因为 rebase 是逐个应用提交，所以可能会在某个提交应用时发生冲突。此时，rebase 会暂停。
    - 打开冲突文件，手动解决冲突（和 merge 冲突一样）。
    - 解决后，使用 `git add <filename>` 将文件标记为已解决。
    - 然后，继续 rebase 过程 `git rebase --continue`
    - 如果中途想放弃，可以回到 rebase 开始前的状态 `git rebase --abort`
5. 合并到主分支
rebase 成功后，你的 feature-login 分支就已经包含了 main 的所有更新，并且**你的提交都在最前面**。现在可以进行一次干净的快进合并。
```bash {linenos=true}
git checkout main
git merge feature-login
```

# When NOT to Use rebase

**永远不要对一个已经推送到 remote，并且可能被团队其他人使用的公共分支 (如 main, develop)进行 rebase！**因为 rebase 会重写历史。如果你 rebase 了一个公共分支并强制推送 (`git push --force`)，那么所有团队成员的本地历史记录都将与远程的“新历史”产生严重分歧。

正确用法是只在你自己的、还未与他人分享的本地分支上使用 rebase，用来整理你自己的提交记录，以便在合并到公共分支前有一个干净的历史。

# Advanced Use git rebase -i

`git rebase -i` 允许你在 rebase 的过程中，对你的提交进行编辑、合并、拆分或删除。这常用于在合并到 main 分支前，将自己本地凌乱的提交（如 "修复拼写错误", "临时提交", "又改了一点"）整理成几个有意义的提交。

假设你的 feature-login 分支有 3 个凌乱的提交，你想把它们合并成一个。
1. 启动交互式 rebase `git rebase -i HEAD~3`. 其中 `HEAD~3` 表示从当前提交 (HEAD) 往前数 3 个提交。
2. 编辑 Rebase 脚本
Git 会打开一个文本编辑器，列出这 3 个提交：

```plaintext {linenos=true}
pick a31ab34 complete login UI
pick 58c34bb fix a button bug
pick 948f2cb add backend verify logic
```

在文件下方会有指令说明。你可以修改每一行前面的 pick 命令。比如，我们想把后两个提交合并到第一个里面：

```plaintext {linenos=true}
pick a31ab34 complete login UI
squash 58c34bb fix a button bug
squash 948f2cb add backend verify logic
```
3. 保存并退出编辑器
Git 会开始合并提交，并弹出另一个编辑器，让你为这个合并后的新提交编写一个新的 commit message. 整理好后保存退出。现在再用 `git log` 查看，你会发现原来 3 个凌乱的提交已经变成了一个干净、完整的提交。