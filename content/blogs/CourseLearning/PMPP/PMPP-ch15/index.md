---
title: PMPP Learning-Chapter 15 Graph traversal
date: 2024-09-18T16:05:12+08:00
lastmod: 2024-09-18T16:05:12+08:00
draft: false
author: ["WITHER"]
keywords: 
    - CUDA
categories:
    - CUDA
tags:
    - PMPP learning
description: Personal notebook 15 of Programming Massively Parallel 
summary: Personal notebook 15 of Programming Massively Parallel  
comments: true
images: 
cover:
    image: ""
    caption: ""
    alt: ""
    relative: true
    hidden: true
---

# 15 Graph Traversal

图是一种表示实体之间关系的数据结构。所涉及的实体表示为顶点，关系表示为边。图的遍历是指从一个顶点出发，依次访问图中所有与之相邻的顶点，直到所有顶点都被访问过为止。

## 15.1 Background

下图展示了一个有向的简单图的例子。我们为每个顶点分配一个唯一的数字，称为顶点编号 (*vertex id*).

![A Simple Graph Example with 9 Vertices and 15 Directional Edges](https://note.youdao.com/yws/api/personal/file/WEBe6e0bafe63910bccfb3577f88479dffb?method=download&shareKey=84d69b480eb0abc7389b157e34df8def "A Simple Graph Example with 9 Vertices and 15 Directional Edges")

图的直观表示是邻接矩阵 (*adjacency matrix*). 如果存在一条从源顶点 i 到目的顶点 j 的边，则邻接矩阵元素 `a[i][j]` 的值为 1，否则为 0. 下图展示了对应的邻接矩阵。

![Adjacent Matrix Representation of the Example Graph](https://note.youdao.com/yws/api/personal/file/WEB50516a130f544bbffa7beab784efb84a?method=download&shareKey=f6419b90e7b997b86ba1213964c6672d "Adjacent Matrix Representation of the Example Graph")

稀疏连接的图可以用稀疏矩阵表示，下图展示了用三种不同存储格式的邻接矩阵: CSR, CSC 和 COO. 我们将行下标和指针数组分别称为 `src` 和 `srcPtrs` 数组，列下标和指针数组分别称为 `dst` 和 `dstPtrs` 数组。在图的 CSR 表示中，每个源顶点指针(`srcPtrs`) 给出顶点出边的起始位置。在图的 CSC 表示中，每个目的顶点指针 (`dstPtrs`) 给出顶点入边的起始位置。在图的 COO 表示中，`src` 和 `dst` 数组分别存储源顶点和目的顶点的编号。

![Three Sparse Matrix Representations of the Adjacency Matrix](https://note.youdao.com/yws/api/personal/file/WEB2d0fabda0f28ddd99a4173aeba461b7d?method=download&shareKey=8204d08f33f6447a23ca5a1c595ea505 "Three Sparse Matrix Representations of the Adjacency Matrix")

## 15.2 Breadth-first Search (BFS)

BFS 通常用于找到从图的一个顶点到另一个顶点所需遍历的最短边数。一种方法是，给定一个被称为根的顶点，用从根到某个顶点所需要遍历的最小边数来标记每个顶点。

下图(A)展示示了以顶点 0 为根的 BFS 结果。如果另一个顶点作为根，BFS 的结果将完全不同。下图(B)是为以顶点 2 为根的 BFS 的结果。可以将 BFS 的标记操作看作是构建一个搜索根节点的 BFS 树。树由所有标记的顶点和在搜索过程中从一个顶点到下一个顶点的遍历的边组成。

![(A and B) Two Examples of BFS Results for Two Different Root Vertices](https://note.youdao.com/yws/api/personal/file/WEB86988e0b8a20689df2c463bc1a37c06a?method=download&shareKey=1439c86c169aaab86a80e4bef46363e3 "(A and B) Two Examples of BFS Results for Two Different Root Vertices")

下图展示了 BFS 在计算机辅助设计 (Computer-Aided Design, CAD) 中的一个重要应用。迷宫路由 (maze routing) 将芯片表示为图。路由块是顶点。从顶点 i 到顶点 j 的边表示可以将一条线从块 i 延伸到块 j.

![Maze Routing in Integrated Circuits](https://note.youdao.com/yws/api/personal/file/WEB95b0ba416ed35e18d378b0a2cab1841b?method=download&shareKey=83ec350ed8c87f47c217d4b32ffc5d0d "Maze Routing in Integrated Circuits")

## 15.3 Vertex-centric Parallelization of BFS

以顶点为中心的并行实现将线程分配给顶点，并让每个线程对其顶点执行操作，这通常涉及迭代该顶点的邻居。当处理不同层级的迭代时，并行实现遵循相同的策略。为每一层调用一个单独的内核的原因是，我们需要等待前一层的所有顶点都被标记，然后再继续标记下一层的顶点。下面实现了一个 BFS 内核，根据前一个层级的顶点标签来标记属于该层级的所有顶点。该内核将每个线程分配给一个顶点，检查其顶点是否属于前一层。如果是，线程将遍历出边，将所有未访问的邻居标记为属于当前级别。这种以顶点为中心的实现通常被称为自顶向下或 push 实现，因为其需要访问给定源顶点的出边。多个线程可以将该标志赋值为 1，代码仍然可以正确执行。这个性质称为幂等性 (*idempotence*).

```cpp {linenos=true}
struct CSRGRAPH {
    int numVertices;
    int* scrPtrs;  // Strating outgoing edge index of each vertex
    int* dstList;  // Destination vertex index of each edge
};
__global__ 
void bfs_kernel_csr(CSRGRAPH graph, unsigned int* level, unsigned int* visited, unsigned int currLevel) {
    unsigned vertexId = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexId < graph.numVertices) {
        if (level[vertexId] == currLevel - 1) {
            for (int i = graph.scrPtrs[vertexId]; i < graph.scrPtrs[vertexId + 1]; i++) {
                unsigned int neighbor = graph.dstList[i];
                if (level[neighbor] == 0xFFFFFFFF) {  // unvisited neighbor
                    level[neighbor] = currLevel;
                    visited[neighbor] = 1;
                    *visited = 1;  // flag to indicate whether reached the end of the graph
                }
            }
        }
    }
}
```

下图展示了该内核如何执行从第 1 层 (`currLevel-1`) 到第 2 层 (`currLevel`) 的遍历。

![Example of a Vertex-centric Push BFS Traversal from Level 1 to Level 2](https://note.youdao.com/yws/api/personal/file/WEB631a2ec2144a24f5b99cda39d3f0da53?method=download&shareKey=30e9a4215568e79524f1e082a1d4c65b "Example of a Vertex-centric Push BFS Traversal from Level 1 to Level 2")

第二个以顶点为中心的并行实现将每个线程分配给一个顶点，迭代顶点的入边。每个线程首先检查其顶点是否已被访问。如果没被访问，线程将遍历入边，如果线程找到一个属于前一层的邻居，线程将把它的顶点标记为属于当前层。这种以顶点为中心的实现通常被称为自底向上或 pull 实现。实现要求能访问给定目标顶点的入边，因此要采用 CSC 表示。
以顶点为中心的 pull 实现的内核代码如下，对于一个线程来说，要确定它的顶点处于当前层，只需要该顶点有一个邻居s属于前一层中就足够了。

```cpp {linenos=true}
struct CSCGRAPH {
    int numVertices;
    int* dstPtrs;  // Starting incoming edge index of each vertex
    int* scrList;  // Source vertex index of each edge
};
__global__ 
void bfs_kernel_csc(CSCGRAPH graph, unsigned int* level, unsigned int* visited, unsigned int currLevel) {
    unsigned vertexId = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertexId < graph.numVertices) {
        if (level[vertexId] == 0xFFFFFFF) {  // loop through its incoming edges if not visited
            for (int i = graph.dstPtrs[vertexId]; i < graph.dstPtrs[vertexId + 1]; i++) {
                unsigned int neighbor = graph.scrList[i];
                if (level[neighbor] == currLevel - 1) {
                    level[vertexId] = currLevel;
                    *visited = 1;  // flag to indicate whether reached the end of the graph
                    break;  // Only need 1 neighbor in previous level to identify the vetex is currLevel
                }
            }
        }
    }
}
```

下图展示了这个内核如何执行从第 1 层到第 2 层的遍历。

![Example of a Vertex-centric Pull (bottom-up) Traversal from Level 1 to Level 2](https://note.youdao.com/yws/api/personal/file/WEB316aa0df65a0a6249ead5e5ecc6290e9?method=download&shareKey=d2f16ad21467d1bedd052c11f45ee5b3 "Example of a Vertex-centric Pull (bottom-up) Traversal from Level 1 to Level 2")

在比较推和拉以顶点为中心的并行实现时，需要考虑两个对性能有重要影响的关键差异。
1. 在 push 实现中，线程在其顶点的循环遍历所有邻居；而在 pull 实现中，线程可能会提前跳出循环。
2. 在 push 实现中，只有被标记为前一层的顶点的线程在遍历其邻居列表；而在 pull 实现中，任何被标记为未访问顶点的线程会遍历其邻居列表。
基于两种实现的差异，常见的优化方法是对低层级使用 push 实现，然后对较高层级使用 pull 实现。这种方法通常被称为方向优化 (*directional optimization*) 实现。选择何时切换通常取决于图的类型。低度图通常有很多层；高度图中，从任何顶点到任何其他顶点只需要很少的层。因此对于高度图来说从 push 实现切换到 pull 实现通常比低度图要早得多。
如果要使用方向优化的实现，则图的 CSR 和 CSC 表示都需要储存。但对于无向图来说，其邻接矩阵是对称的，因此 CSR 和 CSC 表示是相同的的，只需要存储其中一个，就可以被两个实现使用。

## 15.4 Edge-centric Parallelization of BFS

在这个实现中，每个线程被分配到一条边。它检查边的源顶点是否属于前一层以及边的目标顶点是否未被访问。
以边为中心的并行实现的内核代码如下。每个线程使用 COO `src` 数组找到其边缘的源顶点，并检查顶点是否属于前一级。通过此检查的线程将使用 COO `dst` 数组确定边的目的顶点，并检查其是否未被访问过。

```cpp {linenos=true}
struct COOGRAPH {
    int numVertices;
    int numEdges;
    int* srcList;  // Source vertex index of each edge
    int* dstList;  // Destination vertex index of each edge
};
__global__ 
void bfs_kernel_coo(COOGRAPH graph, unsigned int* level, unsigned int* visited, unsigned int currLevel) {
    unsigned edgeId = blockIdx.x * blockDim.x + threadIdx.x;
    if (edgeId < graph.numEdges) {
        unsigned int src = graph.srcList[edgeId];
        if (level[src] == currLevel - 1) {
            unsigned int neighbor = graph.dstList[edgeId];
            if (level[neighbor] == 0xFFFFFFFF) {  // unvisited neighbor
                level[neighbor] = currLevel;
                visited[neighbor] = 1;
                *visited = 1;  // flag to indicate whether reached the end of the graph
            }
        }
    }
}
```

下图展示了该内核如何执行从从第 1 层到第 2 层的遍历。

![Example of an Edge-centric Traversal from Level 1 to Level 2](https://note.youdao.com/yws/api/personal/file/WEB1a12c34603381779af3db3e8ff96ca11?method=download&shareKey=835bfaff2b7020013da3dcc13912a00c "Example of an Edge-centric Traversal from Level 1 to Level 2")

与以顶点为中心的并行实现相比，以边为中心的并行实现的优点如下
1. 有更多的并行性。在以顶点为中心的实现中，如果顶点的数量很少，可能不会启动足够的线程来完全占用设备。因为一个图通常有比顶点更多的边，以边为中心的实现可以启动更多的线程。
2. 具有较小的负载不平衡和控制发散。在以顶点为中心的实现中，每个线程迭代不同数量的边。相反，在以边为中心的实现中，每个线程只遍历一个边。
以边为中心的实现的缺点如下
1. 需要检查图中的每条边。相反，以顶点为中心的实现中，如果确定顶点与当前层级无关，则会跳过整个边列表。
2. 使用 COO 格式存储图，与以顶点为中心的实现使用的 CSR 和 CSC 相比，它需要更多的存储空间来存储边。

## 15.5 Improving efficiency with frontiers

在前两节中的方法中，我们会检查每个顶点或每条边是否属和当前层有关。这种策略的优点是内核是高度并行的，并且不需要跨线程进行任何同步。缺点是启动了许多不必要的线程，并执行了大量无用的工作。我们可以让处理前一层顶点的线程将它们访问的顶点作为 frontier. 因此，对于当前层级，只需要为该 frontier 中的顶点启动线程。

![Example of a Vertex-centric Push (top-down) BFS Traversal from Level 1 to Level 2 with Frontiers](https://note.youdao.com/yws/api/personal/file/WEB8c15c5b4a331ade6411685dd9c0d1f1a?method=download&shareKey=bd7f5e6781a9073f31fdb3005f541380 "Example of a Vertex-centric Push (top-down) BFS Traversal from Level 1 to Level 2 with Frontiers")

对应的内核代码如下。首先为 frontier 的每个元素分配一个线程，使用 CSR `srcPtrs` 数组来定位顶点的出边并进行迭代。对于每个出边，线程使用 CSR `dst` 数组确定其目的顶点，若未被访问过，并将其标记为属于当前层级。为了避免多个线程将邻居视为未访问，应该以原子方式执行邻居标签的检查和更新。`atomicCAS` 内置函数提供 compare-and-swap 的原子操作。如果比较成功,与其他原子操作一样，`atomicCAS` 返回存储的旧值。因此，我们可以通过比较返回值与被比较的值来检查该顶点是否被访问过。

```cpp {linenos=true}
__global__
void frontier_bfs_kernel(CSRGRAPH graph, unsigned int* level,
    unsigned int* prevFroniter, unsigned int* currFroniter,
    unsigned int numPrevFroniter, unsigned int* numCurrFroniter,
    unsigned int* currLevel) {
    // Each thread processes a node in prevFroniter.
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFroniter) {
        unsigned int vertexId = prevFroniter[i]; 
        // All its neighbouring nodes are traversed.
        for (unsigned int edge = graph.scrPtrs[vertexId]; edge < graph.scrPtrs[vertexId + 1]; edge++) {
            unsigned int neighbor = graph.dstList[edge];
            if (atomicCAS(level + neighbor, 0xFFFFFFFF, currLevel) == 0xFFFFFFFF) {  // check if neighbor is unvisited
                unsigned int currFroniterIndex = atomicAdd(numCurrFroniter, 1);
                currFroniter[currFroniterIndex] = neighbor;
            }
        }
    }
}
```

这种基于 frontier 的方法的优势在于，它通过只启动处理相关顶点的线程减少了冗余工作。缺点是长延迟原子操作的开销，特别是当这些操作竞争访问相同的地址时。对于 atomicAdd 操作争用会很高，因为所有线程都增加同一个计数器。

## 15.6 Reducing Contention with Privatization

私有化可以应用于对 numCurrFrontier 的增加，以减少插入 frontier 时的争用。我们可以让每个线程块在整个计算过程中维护自己的本地 frontier，并在完成后更新全局 frontier. 本地 frontier 及其计数器可以存储在共享内存中，从而支持对计数器和存储到本地边界的低延迟原子操作。此外，当将共享内存中的 frontier 存储到全局内存中的公共 frontier 时，访问可以合并。

下图说明了 frontier 私有化的执行情况。

![Privatization of Frontiers Example](https://note.youdao.com/yws/api/personal/file/WEB34a0e090d131236cb469365789ba6a21?method=download&shareKey=cd8a9dcd79e858b456d2bcd1d99c16c3 "Privatization of Frontiers Example")

对应的内核代码如下。注意到公共 frontiner 的索引 `currFrontierIdx` 是用 `currFrontierIdx_s` 表示的，而 `currFrontierIdx_s` 是用 `threadIdx.x` 表示的。因此，相邻线程存储到连续的全局内存位置，这意味着内存访问是合并的。
```cpp {linenos=true}
#define LOCAL_FRONTIER_SIZE 4
__global__
void private_frontier_bfs_kernel(CSRGRAPH graph, unsigned int* level,
    unsigned int* prevFroniter, unsigned int* currFroniter,
    unsigned int numPrevFroniter, unsigned int* numCurrFroniter,
    unsigned int* currLevel) {

    // Initialize privative frontier
    __shared__ unsigned int currFrontier_s[LOCAL_FRONTIER_SIZE];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    // Perform BFS on private frontier
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFroniter) {
        unsigned int vertexId = prevFroniter[i];
        for (unsigned int edge = graph.scrPtrs[vertexId]; edge < graph.scrPtrs[vertexId + 1]; edge++) {
            unsigned int neighbor = graph.dstList[edge];
            if (atomicCAS(level + neighbor, 0xFFFFFFFF, currLevel) == 0xFFFFFFFF) {  // Once a new frontier node is found,
                unsigned currFroniterIndex = atomicAdd(&numCurrFrontier_s, 1);
                if (currFroniterIndex < LOCAL_FRONTIER_SIZE) {  // Try to add it to the private frontier (currFrontier_s)
                    currFrontier_s[currFroniterIndex] = neighbor;
                } else {
                    numCurrFrontier_s = LOCAL_FRONTIER_SIZE;  // frontier is full, stop adding new elements
                    unsigned int currFrontierIdx = atomicAdd(numCurrFroniter, 1);
                    currFroniter[currFrontierIdx] = neighbor;
                }
            }
        }
    }

    // Copy private frontier to global frontier
    __syncthreads();
    __shared__ unsigned int currFrontierStartIdx;  // Start index of private frontier in global frontier
    if (threadIdx.x == 0) {
        currFrontierStartIdx = atomicAdd(numCurrFroniter, numCurrFrontier_s);
    }
    __syncthreads();

    // Commit private frontier to global frontier
    for (unsigned int j = threadIdx.x; j < numCurrFrontier_s; j += blockDim.x) {
        unsigned int currFroniterIdx = currFrontierStartIdx + j;
        currFroniter[currFroniterIdx] = currFrontier_s[j];
    }
}
```
