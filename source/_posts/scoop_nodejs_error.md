---
title: Error When Install nodejs by Scoop
date: 2024/11/24 22:27:42
categories: Bugs
tags: scoop
excerpt: Bugs caused when installing nodejs by scoop.
mathjax: true
katex: true
---

```bash
PS D:\MyHexoBlog> $env:Path += ";C:\Users\17725\scoop\persist\nodejs\bin"
PS D:\MyHexoBlog> hexo
&: C:\Users\17725\scoop\persist\nodejs\bin\hexo.ps1:24:7
Line |
  24 |      & "node$exe"  "$basedir/node_modules/hexo-cli/bin/hexo" $args
     |        ~~~~~~~~~~
     | The term 'node.exe' is not recognized as a name of a cmdlet, function, script file, or executable program. Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
PS D:\MyHexoBlog> node -v
node: The term 'node' is not recognized as a name of a cmdlet, function, script file, or executable program.
Check the spelling of the name, or if a path was included, verify that the path is correct and try again.
PS D:\MyHexoBlog> $PROFILE
C:\Users\17725\Documents\PowerShell\Microsoft.PowerShell_profile.ps1
```

这个文件是 PowerShell 的用户配置文件，可以在其中添加全局的环境变量或者初始化脚本，以便在每次启动 PowerShell 时加载。使用任意文本编辑器（例如 VS Code 或记事本）打开此文件。如果文件不存在，可以先创建一个新文件：

```bash
New-Item -ItemType File -Path $PROFILE -Force
```

在文件末尾，添加以下内容（确保路径与实际安装的 Node.js 路径一致）

```bash
# 添加 Node.js 路径到系统环境变量
$env:Path += ";C:\Users\17725\scoop\apps\nodejs\23.3.0"
```

保存文件后，重新启动 PowerShell 或运行以下命令使更改生效：

```bash
. $PROFILE
```

执行以下命令，检查是否已正确加载 Node.js 路径：

```bash
node -v
# should print v23.3.0
```
