---
title: PDB commands collections
date: 2024/11/29 00:07:42
categories: Useful Tricks
tags: PDB
excerpt: Common commands of PDB.
mathjax: true
katex: true
---

1. `where(w)` 找出当前代码运行位置

2. `list(l)` 显示当前代码的部分上下文

3. `list <line number>` 显示指定行的上下文

4. `list <line number1, line number2>` 显示指定开始行到结束行的代码

5. `up(u)` 返回上个调用点

6. `down(d) `返回下个调用点

7. `args(a)` 显示当前所有变量

8. `print(p) `打印表达式结果

9. `!` 运行python命令，比如 `!test='hello'` 将会把 `test` 变量的值改变为 `hello`

10. `pp` 打印美化过的表达式结果

11. `step` 步进运行至下行代码（如果是调用函数，则运行至所调用函数的第一行）

12. `next` 运行至下行代码（如果是调用函数，会直接运行完此函数）

13. `until` 运行至当前代码端底部

14. `return` 运行至return代码处

15. `break` <line number> 运行时设置断点

16. `continue` 运行程序直至遇到下一个断点

17. `break` <file name:line number> 运行时设置另一个文件的断点

18. `break `显示断点情况

19. `disable` <break number> 将指定的断点失效（但存在）

20. `enable` <break number> 将指定的断点生效

21. `clear` <break number> 删除断点

22. `tbreak` <line number> 运行时设置临时断点（运行一次后自动删除）

23. `break <line number> <condition>` 运行时设置断点，当满足condition条件时触发断点，ex: break 11 i > 10 表示在第11行代码处，当变量i大于10时，触发断点

24. `condition <break number> <condition>` 设置指定断点的触发条件

25. `ignore <break number> <n>` 忽略指定断点n次

26. `commands <break number> ... end` 对指定断点编写脚本，当运行到该断点时自动执行
