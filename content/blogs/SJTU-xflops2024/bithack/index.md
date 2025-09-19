---
title: "Bithack"
date: 2025-09-18T11:48:24+08:00
lastmod: 2025-09-18T11:48:24+08:00
author: ["WITHER"]

categories:
- HPC

tags:
- HPC

keywords:
- 

description: "Solution of SJTU-xflops2024 Bithack." # 文章描述，与搜索优化相关
summary: "Solution of SJTU-xflops2024 Bithack." # 文章简单描述，会展示在主页
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

# Problem Description

详情请见[题目链接](https://github.com/HPC-SJTU/Xflops2024_1st_exam/tree/main/Bithack)，这里简要说明下:题目要求是优化 `rotate_the_bit_vector_left()` 函数，该函数将 `bit_vector` 的一段从左往右数第 `bit_offset` 位为起始点, 长度为 `bit_length` 的子数组的前 `bit_amount` 位循环左移。

`bit_vector.h` 和 `bit_vector.c` 中有一些有用的函数声明定义。
- `modulo()`: 返回 `r = n (mod m)` 的结果，其中 `0 <= r < m`.
- `bitmask()`: 返回和一个 byte 相与时保留从右往左数的第 `bit_index` 位的掩码。
- `bit_vector_get_bit_sz()`: 获取一个 `bit_vector` 对象的位数。
- `bit_vector_get()`: 获取一个 `bit_vector` 第 `bit_index` 位的值。
- `bit_vector_set()`: 将一个 `bit_vector` 第 `bit_index` 位的值设置为 `value`.

# Vanilla Method Analysis

perf 是 Linux 内核内置的性能分析工具 (Performance Counters for Linux)，它可以帮助采样程序的 CPU 使用情况、函数调用栈、热点代码等，从而识别性能瓶颈。

{{< details title="How to install perf tools in WSL2" >}}
```bash
# windows
wsl --update 
# wsl 2
sudo apt update
sudo apt install flex bison  
sudo apt install pkg-config   # operator '&&' has no right operand
sudo apt install libdwarf-dev libelf-dev libnuma-dev libunwind-dev \
libnewt-dev libdwarf++0 libelf++0 libdw-dev libbfb0-dev \
systemtap-sdt-dev libssl-dev libperl-dev python-dev-is-python3 \
binutils-dev libiberty-dev libzstd-dev libcap-dev libbabeltrace-dev libtraceevent-dev
git clone https://github.com/microsoft/WSL2-Linux-Kernel --depth 1
cd WSL2-Linux-Kernel/tools/perf
make -j8 # parallel build
sudo cp perf /usr/local/bin
```
{{< /details >}}

我们来看一下原始 `rotate_the_bit_vector_left()` 的执行流程:
1. 它有一个循环，需要旋转 `bit_left_amount` 次。
2. 在循环内部，它调用 `rotate_the_bit_vector_left_one()` 函数。
3. 该函数的作用是将子数组左旋一位。它内部又有一个循环，需要遍历 `bit_length - 1` 次，每次都调用 `bit_vector_get` 和 `bit_vector_set` 来移动一个比特位。

所以，总的操作次数大约是 `bit_left_amount * bit_length` 次比特读写。

编译好后使用 `perf record ./everybit -s` 命令生成 `perf.data` 文件后使用 `perf report` 命令显示采样结果。

```
Samples: 105  of event 'cpu-clock:ppp', Event count (approx.): 26250000
Overhead  Command   Shared Object         Symbol
  52.38%  everybit  everybit              [.] bit_vector_set
  33.33%  everybit  everybit              [.] bit_vector_get
  10.48%  everybit  everybit              [.] rotate_the_bit_vector
   0.95%  everybit  [kernel.kallsyms]     [k] _raw_spin_unlock_irqrestore
   0.95%  everybit  [kernel.kallsyms]     [k] put_cpu_partial
   0.95%  everybit  [kernel.kallsyms]     [k] queue_work_on
   0.95%  everybit  ld-linux-x86-64.so.2  [.] 0x00000000000131d2
```

根据 perf 的性能分析报告，超过 85% 的时间都消耗在了 `bit_vector_set` (52.38%) 和 `bit_vector_get` (33.33%) 这两个函数上，说明当前的算法对单个比特位的读写操作过于频繁。

# Optimization 1: 3-step Rotation

我们可以借鉴数组旋转的经典“三步反转法”思想，但在这里更直观的方法是使用一个临时缓冲区。想象一下把字符串 "ABCDEFG" 左旋 3 位:

1. 保存: 先把要被移到末尾的前3个字符 "ABC" 保存起来。
2. 移动: 把后面的 "DEFG" 移动到开头，字符串变成 "DEFG___"。
3. 放回: 把保存的 "ABC" 放到末尾的空白处，得到最终结果 "DEFGABC"。

这个过程只涉及三次批量操作，而不是像原算法那样执行 3 次（旋转位数）* 7 次（长度）的单字符移动。具体步骤和对应代码如下

1. 分配缓冲区: 根据要旋转的位数 `bit_left_amount`，申请一块足够大的内存作为临时缓冲区。
2. 保存前缀: 将子数组最前面的 `bit_left_amount` 个比特位复制到缓冲区中。
3. 移动主体: 将子数组中剩下的 `bit_length - bit_left_amount` 个比特位整体向前移动 `bit_left_amount` 位。
4. 写回前缀: 将缓冲区里保存的比特位写回到子数组的末尾。
5. 释放缓冲区: 释放第一步申请的内存。

```C{linenos=true}
static void rotate_the_bit_vector_left(bit_vector_t* const bit_vector,
                                const size_t bit_offset,  // 开始旋转的起点
                                const size_t bit_length,  // 旋转的长度
                                const size_t bit_left_amount) { // 旋转的位数 
  if (bit_length == 0 || bit_left_amount == 0) {
    return;
  }

  const size_t effective_amount = bit_left_amount % bit_length;
  if (effective_amount == 0) {
    return;
  }

  // 1. 分配临时缓冲区来存储被旋转到末尾的 bits
  const size_t prefix_bytes = (effective_amount + 7) / 8;
  char* prefix_buffer = (char*) calloc(prefix_bytes, sizeof(char));

  if (prefix_buffer == NULL) {
    return;
  }

  // 2. 将子数组前 effective_amount 个 bits 复制到临时缓冲区
  for (size_t i = 0; i < effective_amount; i++) {
    if (bit_vector_get(bit_vector, bit_offset + i)) {  // 子数组第 i 位是否为 1
      prefix_buffer[i / 8] |= bitmask(i);
    }
  }

  // 3. 将子数组剩下的部分向前移动
  const size_t bits_to_move = bit_length - effective_amount;
  for (size_t i = 0; i < bits_to_move; i++) {
    bool bit = bit_vector_get(bit_vector, bit_offset + effective_amount + i);  // 要移动的 bit
    bit_vector_set(bit_vector, bit_offset + i, bit);
  }

  // 4. 将缓冲区保存的 bits 写回子数组末尾
  const size_t paste_offset = bit_offset + bits_to_move;
  for (size_t i = 0; i < effective_amount; i++) {
    bool bit = (prefix_buffer[i / 8] & bitmask(i)) != 0;  // 从缓冲区读入 bit
    bit_vector_set(bit_vector, paste_offset + i, bit);
}

  free(prefix_buffer);
}
```

该方法给出的评分如下
```
check result: PASSED
performance of -s: 28
performance of -m: 33
performance of -l: 38
------score--------
-s : 70.00 /100
-m : 77.27 /100
-l : 80.00 /100
total score: 77.18 /100
```

# Optimization 2: From bit-by-bit To byte-level

进一步优化的点是将按位拷贝操作改成一次操作 8 个字节 (需要考虑对齐问题). 拷贝任务分为三个阶段处理，以优化性能并处理非对齐的位偏移: 

1. 头部处理: 处理目标地址非字节对齐的位。
    - 计算目标偏移的非对齐部分。如果目标偏移不是字节的开始点，需要先拷贝少量位，直到目标地址对齐到字节边界。拷贝的位数是剩余到下一个字节边界的位数，但不能超过 `num_bits`.
2. 中间块处理: 以 64 位为单位高效拷贝对齐的块。根据源是否字节对齐，分两种情况: 
    - 源和目标都按字节对齐，直接使用 memmove 拷贝所有字节。
    - 源非字节对齐: 使用 memcpy 读取 8 字节到 `uint64_t word`，避免未对齐访问问题。
读取下一个字节 (`next_byte = src_ptr[8]`)，用于位移拼接。将当前 64 位右移并拼接下一字节的位 `(word >> src_bit_shift) | (next_byte << (64 - src_bit_shift))` 后将结果写入目标地址。
3. 尾部处理: 处理剩余不足 64 位的部分。逐 bit 拷贝 (`|=`  设置为 1，`&= ~` 清零).

```C{linenos=true}
static void bit_block_move_ultimate(char* dest_data, size_t dest_bit_offset,
                                    const char* src_data, size_t src_bit_offset,
                                    size_t num_bits) {
  if (num_bits == 0) {
    return;
  }

  // 同样，重叠拷贝的处理对于一个健壮的函数是必需的。
  // 这里的 assert 仅用于指出这个简化。
  if (dest_data == src_data && dest_bit_offset > src_bit_offset) {
      assert(dest_bit_offset >= src_bit_offset + num_bits);
      // 实际代码中需要实现反向拷贝
  }

  // 1. 处理头部的非对齐 bit，使得目标地址按字节对齐
  size_t dest_align_offset = dest_bit_offset % 8;
  if (dest_align_offset != 0) {
      size_t bits_in_head = 8 - dest_align_offset;
      if (bits_in_head > num_bits) {
          bits_in_head = num_bits;
      }

      // 逐 bit 拷贝头部
      for (size_t i = 0; i < bits_in_head; ++i) {
          if ((src_data[(src_bit_offset + i) / 8] >> ((src_bit_offset + i) % 8)) & 1) {
              dest_data[(dest_bit_offset + i) / 8] |= (1 << ((dest_bit_offset + i) % 8));
          } else {
              dest_data[(dest_bit_offset + i) / 8] &= ~(1 << ((dest_bit_offset + i) % 8));
          }
      }

      num_bits -= bits_in_head;
      src_bit_offset += bits_in_head;
      dest_bit_offset += bits_in_head;
  }

  // 2. 处理中间的 64 位对齐块
  size_t num_blocks = num_bits / 64;
  if (num_blocks > 0) {
      char* dest_ptr = dest_data + dest_bit_offset / 8;
      const char* src_ptr = src_data + src_bit_offset / 8;
      size_t src_bit_shift = src_bit_offset % 8;

      if (src_bit_shift == 0) { // 源和目标都已对齐，可以直接 memmove
          memmove(dest_ptr, src_ptr, num_blocks * 8);
      } else {
        for (size_t i = 0; i < num_blocks; ++i) {
            uint64_t word;
            // 【修正 #3 & #4】安全地读取源数据
            // 使用 memcpy 来避免不对齐读取，同时它也比逐字节拼接高效
            memcpy(&word, src_ptr, sizeof(uint64_t));
            
            // 为了拼接，我们需要下一个字节的数据
            uint64_t next_byte = 0;
            // 计算需要拷贝的总字节数
            size_t total_bits_processed = (i + 1) * 64;
            size_t required_src_bytes = (src_bit_offset + total_bits_processed - dest_bit_offset + 7) / 8;

            next_byte = (uint64_t)src_ptr[8];
            
            uint64_t result = (word >> src_bit_shift) | (next_byte << (64 - src_bit_shift));

            memcpy(dest_ptr, &result, sizeof(uint64_t));

            dest_ptr += 8;
            src_ptr += 8;
        }
      }

      size_t bits_in_middle = num_blocks * 64;
      num_bits -= bits_in_middle;
      src_bit_offset += bits_in_middle;
      dest_bit_offset += bits_in_middle;
  }

  // 3. 处理尾部剩余的不足 64 位的 bit
  if (num_bits > 0) {
      for (size_t i = 0; i < num_bits; ++i) {
          if ((src_data[(src_bit_offset + i) / 8] >> ((src_bit_offset + i) % 8)) & 1) {
              dest_data[(dest_bit_offset + i) / 8] |= (1 << ((dest_bit_offset + i) % 8));
          } else {
              dest_data[(dest_bit_offset + i) / 8] &= ~(1 << ((dest_bit_offset + i) % 8));
          }
      }
  }
}
```

完整的函数如下

```C{linenos=true}
void rotate_the_bit_vector(bit_vector_t* const bit_vector,
                     const size_t bit_offset,
                     const size_t bit_length,
                     const ssize_t bit_right_amount) {
  assert(bit_offset + bit_length <= bit_vector->bit_sz);

  if (bit_length == 0) {
    return;
  }
  const size_t left_shift = modulo(-bit_right_amount, bit_length);
  if (left_shift == 0) {
      return;
  }

  // 1. 分配一个足以容纳整个旋转区域的大临时缓冲区。
  // 这是解决内存崩溃问题的关键，确保了后续 bithack_memcpy 的所有读取都是安全的。
  const size_t temp_buf_bytes = (bit_length + 7) / 8;
  char* temp_buffer = (char*)malloc(temp_buf_bytes);
  if (temp_buffer == NULL) {
      // 在实际项目中，应有更完善的错误处理
      exit(1); 
  }

  // 将旋转区域分为两部分：
  // part_A (前部): 原本在前面，需要被旋转到末尾的部分。
  // part_B (后部): 原本在后面，需要被旋转到前面的部分。
  const size_t part_A_len = left_shift;
  const size_t part_B_len = bit_length - left_shift;

  // 2. 将 part_B 拷贝到临时缓冲区的开头
  bit_block_move_ultimate(temp_buffer, 0, 
      bit_vector->buf, bit_offset + part_A_len,
      part_B_len);

  // 3. 将 part_A 拷贝到临时缓冲区的末尾，紧随 part_B 之后
  bit_block_move_ultimate(temp_buffer, part_B_len,
      bit_vector->buf, bit_offset,
      part_A_len);

  // 4. 至此，temp_buffer 中已是旋转后的正确序列 [B|A].
  // 将整个临时缓冲区一次性拷贝回原位。
  bit_block_move_ultimate(bit_vector->buf, bit_offset,
      temp_buffer, 0,
      bit_length);

  // 5. 释放临时缓冲区
  free(temp_buffer);

}
```

最后也是达到了满分
```
check result: PASSED
performance of -s: 36
performance of -m: 40
performance of -l: 45
------score--------
-s : 100.00 /100
-m : 100.00 /100
-l : 100.00 /100
total score: 100.00 /100
```