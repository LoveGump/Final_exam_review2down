# Ch3 图

## 3.1基本定义与应用

- 无向图

  <img src="/Users/gump/大二资料（更新版）/alg/笔记/ch3/assert/image-20250528212757470.png" alt="image-2025052821275740" style="zoom:80%;" />

图的两种表示：

- 邻接矩阵
- 邻接表

* **Def.** 无向图 G = (V, E) 的一条<font color="red">路径P</font>定义为如下的节点序列 $v_1, v_2, \dots, v_{k-1}, v_k$，其中 $v_i, v_{i+1}$ 是 E 中的一条边。P被称为从 $v_1$ 到 $v_k$ 的路径。
* **Def.** 如果一条路径所有的节点都是相互不同的，就称为<font color="red">简单的</font>。
* **Def.** 如果对于无向图中任意两个顶点u,v, 都存在一条路径，那么无向图被称为是<font color="red">连通的</font>。
* **Def.** 路径 $v_1,v_2,\dots,v_k$ 被称为一个**圈 (Cycle)**，如果$ v_1=v_k$,$k > 2$，而且前 k−1 个顶点两两不同。

## BFS(广度优先)

```
BFS(s):
  // Initialization
  Set Discovered[s] = true
  For all other vertices v, set Discovered[v] = false
  Initialize L[0] with the single element s
  Set layer counter i = 0
  Set BFS tree T = ∅ (empty set)

  // Main loop
  While L[i] is not empty:
    Initialize an empty list L[i+1]
    For each node u in L[i]:
      For each edge (u,v) incident to u:
        If Discovered[v] == false:
          Set Discovered[v] = true
          Add edge (u,v) to tree T
          Add vertex v to list L[i+1]
        Endif
      Endfor
    Increment layer counter i (i = i + 1)
  Endwhile
```

定理：如果图是邻接表给出，BFS算法的上述实现将以O(m+n)时间运行。

Pf：考虑结点 u, 存在 deg(u) 条与u相连的边 (u, v)，所以处理所有边的时间是 $\sum_{u \in V}  deg(u) = 2m$；此外对于顶点, 还要O(n)的额外时间来管理数组Discovered.

定理：BFS算法产生的层 $L_i$​ 就是到源点s距离为i的顶点集合. 存在一条s到t的路径当且仅当 t 出现在某一层中.

定理：设T是图G = (V, E)的一棵宽度优先搜索树，(x, y)是G中的一条边.那么 x ， y 所属的层数至多相差 1.

## DFS（深度优先）

```
DFS(s)
  初始化S为具有一个元素s的栈
  While S 非空
    从S中取出一个节点u
    If Explored[u]=false then
      置Explored[u]=true
      For每条与u关联的边(u,v)
        把v加到栈S
      Endfor 
    EndIf
  EndWhile
```

BFS算法发现的是从始点s可达的结点。把这个集合R看作G的包含s的连通分支。

定理：算法结束产生的集合R恰好是G的包含s的连通分支。

定理：对**无向图**中任两个结点s与t,它们的连通分支或者相等，或者不相交 。

定理：对**有向图**中的任何两个结点s与t,它们的**强连通**分支或者相等，或者不相交 。

定义：u和v是相互可达的，如果彼此之间存在到达对方的路径。那么此图是强连通的。

引理：设s是图G中的任意一个结点。G是强连通的，当且仅当图中的每个结点能够与s相互可达。

1. **任选一个节点 s (Pick any node s)。**
2. 从节点 s 在图 G 中运行广度优先搜索 (Run BFS from s in G)。
   - 这一步检查从节点 s 出发是否能到达图中的所有其他节点。
3. 从节点 s 在图 $G_{rev}$ (G 的反向图) 中运行广度优先搜索 (Run BFS from s in  $G_{rev}$)。
   -  $G_{rev}$ 是将原图 G 中所有边的方向都反转后得到的图。在  $G_{rev}$ 中从 s 出发进行 BFS，等价于检查在原图 G 中是否所有节点都能到达节点 s。
4. 当且仅当两次 BFS 都遍历到了图中所有节点时，返回真 (Return true iff all nodes reached in both BFS executions)。
   - 如果节点 s 能到达所有其他节点 (步骤2)，并且所有其他节点都能到达 s (步骤3)，那么图中任意两个节点 u, v 之间都可以通过路径 u → s → v 实现互相到达。因此，整个图是强连通的。

幻灯片的最后一行指出： “存在 O(m+n) 的有效算法判别图G是否强连通。” 这里的 m 是图的边数，n 是图的顶点数。上面描述的两次 BFS 的算法本身就是 O(m+n) 的，因为单次 BFS 的时间复杂度是 O(m+n)。经典的强连通分量算法（如 Tarjan 算法或 Kosaraju 算法）也是 O(m+n) 的，并且可以找出图中所有的强连通分量，而不仅仅是判断整个图是否强连通。

## 二分性测试

- 二部图

定理：如果一个图是二部图，那么它不可能包含一个奇圈。

### 图的连通性与二部图判定定理

**定理**: 设 **G** 是一个连通图，$L_0,\dots ,L_k $是从顶点 **s** 由 BFS 算法生成的层。那么下面两件事一定恰好成立其一：

1. **G 中没有边与同一层的两个结点相交。**

   在这种情况下，**G 是二部图**。其中偶数层的结点可以着一种颜色（例如，红色），奇数层的结点可以着另一种颜色（例如，蓝色）。这直接利用 BFS 的分层特性来划分图的两个部分。

2. **G 中有一条边与同一层的两个结点相交。**

   此种情形下，图中存在一个奇数长度的圈 (奇圈)，因此该图不可能是二部图。

   **解释**: 如果同一层 $L_j $中的两个节点 u 和 v 之间存在一条边，那么从源点 s 到 u 的路径长度为 j，从源点 s 到 v 的路径长度也为 j。这两个路径加上 u 和 v之间的边 (u,v) 会形成一个环。这个环的长度是 $j+j+1=2j+1$，这是一个奇数。图论中的一个基本结论是：一个图是二部图当且仅当它不包含奇数长度的圈。

引理：图G是二部图**当且仅当**图中没有奇圈。

## 有向无圈图与拓扑排序

DAG(有向无环图)

定理：如果G有一个拓扑排序，那么G是一个DAG.

命题：在每一个DAG中，存在一个没有输入边的结点。✅（反证）

定理 如果G是一个DAG, 那么G有一个拓扑排序。（归纳）

- 证明思路 (Proof Idea)：

1. **归纳基础 (Base Case):**
   - 单个顶点的图显然存在拓扑排序。 (A graph with a single vertex obviously has a topological sort.)
2. **归纳假设 (Inductive Hypothesis):**
   - 假设所有节点数小于 n 的DAG均存在拓扑排序。 (Assume all DAGs with fewer than n nodes have a topological sort.)
3. **关键引理 (Key Lemma):**
   - DAG中必存在入度为0的顶点。 (A DAG must contain at least one vertex with an in-degree of 0.)
     - **解释 (Explanation):** 若所有顶点入度 ≥1，则可以从任意节点开始，通过反复回溯其前驱顶点 (a vertex that has an edge pointing to the current vertex)。由于图是有限的，这个回溯过程最终必然会重复访问某个顶点，从而形成一个环。但这与图是DAG（无环图）的前提相矛盾。因此，DAG中必存在入度为0的顶点。
4. **归纳步骤 (Inductive Step):**
   - **选取 (Select):** 选取图中一个入度为0的顶点 v。根据关键引理，这样的顶点一定存在。将其作为拓扑排序的第一个元素。
   - **移除 (Remove):** 从图中移除顶点 v 以及所有从 v 出发的边。得到的剩余子图仍然是一个DAG（因为移除顶点和边不会产生环），并且其节点数变为 n−1。
   - **应用假设 (Apply Hypothesis):** 根据归纳假设，这个节点数为 n−1 的子图存在一个拓扑排序。
   - **构造 (Construct):** 将顶点 v 放在这个子图拓扑排序的前面，就得到了原图 G 的一个完整拓扑排序。

定理：下面算法在O(m+n)时间内找到一个拓扑排序.

Pf.考虑边逐次递减的代价O(m);追踪被删除的结点代价O(n).

<img src="/Users/gump/大二资料（更新版）/alg/笔记/ch3/assert/image-20250528222650905.png" alt="image-20250528222650905" style="zoom:80%;" />