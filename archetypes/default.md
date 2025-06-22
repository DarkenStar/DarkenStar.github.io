---
date: '{{ .Date }}'
draft: true
mermaid: true #是否开启mermaid
title: '{{ replace .File.ContentBaseName "-" " " | title }}'
---
