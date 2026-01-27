---
title: "HuggingfaceCompass"
date: 2025-10-18T22:19:07+08:00
lastmod: 2025-10-18T22:19:07+08:00
author: ["WITHER"]

categories:
- Productivity

tags:
- huggingface

keywords:
- huggingface

description: "huggingface compass in CN." # 文章描述，与搜索优化相关
summary: "huggingface compass in CN." # 文章简单描述，会展示在主页
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

# Solution of ConnectionError

在中国大陆网络环境下，HF Hub 被墙或延迟高，因此要考虑用 Hugging Face 镜像。

- 在当前 bash 命令行中输入:

    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    ```

- Linux 系统下可选将配置写入 `~/.bashrc` 永久生效: 
    ```bash
    echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc && source ~/.bashrc
    ```

- 在 Python 脚本中**开头**设置:

    ```python
    import os
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    ```

# Access Llama2 Series Model

Llama 2 模型自 2023 年起就是 gated 的，主要是请求访问和设置 token. 批准通常自动或几小时内（Meta 现在基本自动）。

1. 创建/登录 Hugging Face 账户：
    - 去 https://huggingface.co/join 创建免费账户（用邮箱验证）。
    - 登录后，去模型页面：https://huggingface.co/meta-llama/Llama-2-70b-hf
    - 点击 "Request access" 或 "Agree to access" 按钮。
        - 填写表单：用户名、邮箱、用途（e.g., "Research on LLM quantization"），同意 Meta 的 Llama 2 许可（Community License Agreement）.
    - 提交请求。Meta 通常自动批准（几分钟到几小时）；检查邮箱或 HF 通知。
    - 如果已请求过，检查状态：模型页面 > Settings > Access requests（或你的 HF 设置）.

2. 生成并设置访问 Token：
    - 登录 HF > Settings > Access Tokens > New token.
        - 类型用 fine-grained 访问，选择 Public gated repositories（默认可能没开）. 生成后，复制 token（e.g., hf_xxxxxxxx）.

    - 在服务器上使用以下命令登录 HF CLI（安装 huggingface_hub 如果无：`pip install huggingface_hub`）：
        ```bash
        huggingface-cli login
        ```
    - 粘贴 token，按 Enter. 成功后，token 会存到 `~/.cache/huggingface/token`.
    -  成功后使用 `huggingface-cli whoami` 命令应显示你的用户名。

如果 `huggingface-cli login` 超时（connection timeout）通常是网络连接问题，尤其是访问 HF 的 API 服务器（https://huggingface.co 或 api.huggingface.co）时。

这时使用环境变量设置 token，在终端输入

```bash
export HF_TOKEN=hf_your_token_here  # 替换成你的 token
```

或者写入 `~/.bashrc` 永久生效：`echo 'export HF_TOKEN=hf_your_token_here' >> ~/.bashrc && source ~/.bashrc`

用一下命令测试登录，输出出你的用户名就成功。

```bash
python -c "
import os
from huggingface_hub import login, whoami
login(token=os.environ['HF_TOKEN'])
print(whoami())
"
```