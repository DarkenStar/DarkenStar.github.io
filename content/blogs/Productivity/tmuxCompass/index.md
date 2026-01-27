---
title: "TmuxCompass"
date: 2025-10-21T19:33:33+08:00
lastmod: 2025-10-21T19:33:33+08:00
author: ["WITHER"]

categories:
- Productivity

tags:
- tmux

keywords:
- tmux

description: "Direction of tmux." # 文章描述，与搜索优化相关
summary: "Direction of tmux." # 文章简单描述，会展示在主页
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
tmux 是一个终端复用器（Terminal Multiplexer）. 它允许你在一个终端窗口中创建和管理多个独立的终端会话。你可以随时断开（detach）与这些会话的连接，而它们会在后台继续运行，之后你可以随时重新连接（attach）回去。

# 核心概念

tmux 的结构分为三层：

* **会话 (Session)**：一个会话是一个独立的工作区，包含了多个窗口。你可以为不同的项目创建不同的会话。
* **窗口 (Window)**：一个窗口占据整个终端屏幕，就像浏览器中的标签页。一个会话可以有多个窗口。
* **窗格 (Pane)**：一个窗口可以被分割成多个窗格，每个窗格都是一个独立的伪终端。

# 基本操作

tmux 的所有命令都需要通过一个前缀键 (Prefix) 来激活。默认的前缀键是 Ctrl+b.

操作流程是：先按下 Ctrl+b，松开，然后再按下一个指令键。

## 会话 (Session) 管理

在你的常规终端中执行以下命令：

- `tmux` 或 `tmux new`:	创建一个默认名称的新会话。
- `tmux new -s <session-name>`:	创建一个指定名称的新会话。
- `tmux ls`: 列出所有正在运行的 tmux 会话。
- `tmux attach -t <session-name>`: 连接到指定名称的会话。
- `tmux kill-session -t <session-name>`: 销毁指定名称的会话。
- `tmux kill-server`: 销毁所有会话。

从会话中分离 (Detach) 当你处于 tmux 会话中时，可以通过按下 **Ctrl+b，然后按 d** 安全地退出，让会话在后台继续运行。

## 窗口 (Window) 管理

在 tmux 窗口中，使用以下快捷键管理窗格

| 快捷键 (前缀: Ctrl+b) | 描述                                   |
| :-------------------: | :------------------------------------- |
|           c           | 创建一个新窗口 (create)                |
|           w           | 列出所有窗口，并可通过方向键选择切换。 |
|           p           | 切换到上一个窗口 (previous)            |
|           n           | 切换到下一个窗口 (next)                |
|          0-9          | 直接通过数字编号切换到指定窗口。       |
|           ,           | 重命名当前窗口。                       |
|           &           | 关闭当前窗口 (会提示确认)              |


## 窗格 (Pane) 管理

| 快捷键 (前缀: Ctrl+b) | 描述                                   |
| :-------------------: | :------------------------------------- |
|           %           | 垂直分割当前窗格（左右布局）                |
|           "           | 水平分割当前窗格（上下布局） |
|           ↑ ↓ ← →           | 在窗格之间进行导航。            |
|           z           | 将当前窗格最大化/恢复 (zoom)                |
|          x          | 关闭当前窗格 (会提示确认)
|           space           | 在预设的布局之间切换。                       |
|           {           | 关闭当前窗口 (会提示确认)              |
|           }	       |将当前窗格与下一个窗格交换位置。 |

**调整窗格大小**: 按住 Ctrl+b 不松开，然后连续按方向键 ↑ ↓ ← → 来调整窗格大小。

# 复制模式

tmux 有自己的一套复制粘贴机制，对于在不同窗格间复制文本非常有用。

1. 进入复制模式: 按下 Ctrl+b，然后按 [.
2. 移动光标: 用方向键 ↑ ↓ ← → 移动光标。默认是 Emacs 风格的快捷键，可以配置为 vi 风格（见下方配置）.
3. 开始选择:
    - Emacs 模式: 按下 Ctrl+space。
    - vi 模式: 按下 space。
4. 选择文本: 移动光标来选中文本。
5. 复制文本:
    - Emacs 模式: 按下 Alt+w.
    - vi 模式: 按下 Enter.
6. 粘贴文本: 按下 Ctrl+b，然后按 ].

# 自定义配置 (~/.tmux.conf)

你可以通过创建并编辑 `~/.tmux.conf` 文件来高度自定义 tmux.

基础配置示例: 创建一个文件 ~/.tmux.conf 并填入以下内容：

```bash
# 将前缀键从 C-b 改为 C-a，更方便按
unbind C-b
set-option -g prefix C-a
bind-key C-a send-prefix

# 开启鼠标支持 (选择窗格、调整大小、滚动)
set -g mouse on

# 使用 `|` 和 `-` 进行窗格分割，更直观
bind | split-window -h
bind - split-window -v
unbind '"'
unbind %

# 使用 hjkl 进行窗格导航 (vi 风格)
bind h select-pane -L
bind j select-pane -D
bind k select-pane -U
bind l select-pane -R

# 开启 vi 风格的复制模式
set-window-option -g mode-keys vi

# 复制快捷键 (vi 风格)
# 按下 v 开始选择，y 复制，p 粘贴
bind-key -T copy-mode-vi 'v' send -X begin-selection
bind-key -T copy-mode-vi 'y' send -X copy-selection-and-cancel

# 粘贴快捷键
bind p paste-buffer

# 状态栏配置
set -g status-bg black
set -g status-fg white
set -g status-left '#[fg=green]#H' # 主机名
set -g status-right '#[fg=yellow]%Y-%m-%d %H:%M' # 日期和时间
set -g window-status-current-style bg=red

# 设置窗口和窗格索引从 1 开始
set -g base-index 1
setw -g pane-base-index 1
```

修改配置文件后，有两种方式让它生效：
1. 重启 tmux: 关闭所有 tmux 会话并重新打开。
2. 重新加载配置: 在 tmux 会话中，按下 Ctrl+b，然后输入 `:` 进入命令模式，输入 `source-file ~/.tmux.conf` 并回车。

可以为这个操作绑定一个快捷键，在 `~/.tmux.conf` 中加入: 

```bash
# 使用 R 快速重载配置
bind r source-file ~/.tmux.conf \; display "Reloaded!"
```
之后只需按下 Ctrl+b 然后 r 即可。