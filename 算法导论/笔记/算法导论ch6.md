## 动态规划

### 1.带权区间调度

#### 1.1问题描述

存在单一资源`R`，有`n`个需求`{1, 2, ..., n}`，每个需求指定一个开始时间`bi`与一个结束时间`ei`，在时间区间`[bi, ei]`内该需求想要占用资源`R`，资源`R`一旦被占用则无法被其他需求利用。每个需求`i`带有一个权值`vi`，代表该需求被满足之后能够带来的价值或者贡献。如果两个需求的时间区间不重叠，那么它们是相容的。带权值的区间调度问题即，对上面所描述的情境，求出一组相容子集`S`使得`S`中的区间权值之和最大。

#### 1.2解决思路

假设将活动按照它们的结束时间`ei`从早到晚进行排序。`e1 <= e2 <= ... <= en`。排序之后我们会得到如下所示的活动顺序图。

![](/Users/gump/大二资料（更新版）/alg/笔记/assert/带权区间调度_活动顺序图.jpg)

两种情况：

- 第一，参加活动`n`。要参加活动`n`的话必须去掉`{1, 2, ..., n-1}`中所有跟`n`冲突的活动。也就是说，我们只能从活动`{1, 2, ..., p(n)}`中选取要参加的活动组合，而活动`{p(n)+1, p(n)+2, ..., n-1}`则全部不能选择，因为它们都跟活动`n`冲突了（回顾前面`p(j)`的定义）。那么这个时候选择参加活动`n`之后，我们会得到新的活动组合，这个组合的奖励之和会变成`OPT(p(n)) + vn`，即从活动`{1, 2, ..., p(n)}`中能够获取的最大奖励再加上活动`n`的奖励。
- 第二，放弃活动`n`，这种情况下计算活动奖励就很简单了，选择参加的活动组合没有发生变化，故奖励仍是从`{1, 2, ..., n-1}`中能够获取的最大奖励`OPT(n-1)`。

递推公式如下：

```
 OPT[j] = max(v(j) + OPT(p(j)), Opt(j − 1))
```

算法内容如下：

```
OPT(j):
    If j == 0 then
        return 0
    Else
        return max(OPT(p(j)) + vj, OPT(j-1))
    Endif
```

上面的递归算法是尾递归，可以使用迭代对其进行优化。使用一个数组(动态规划的专业名词叫做备忘录)来记录求解过的值，避免重复求解。改进之后可以得到下面这个迭代算法，时间复杂度为`O(n)`。

```
OPT[n] = {0}
For j = 1 to n:
    OPT[j] = max(OPT(p(j)) + vj, OPT(j-1))
Endfor
```

该算法已经可以帮助我们找到最大奖励了。但仅知道一个最大奖励并没有太大意义，我们更需要知道通过参加哪些活动来取得最大奖励。因此，我们在计算最大活动奖励的过程中，还需要记录一下选取了哪些活动。

我们定义一个记录数组`S`，继续回到我们前面讨论过的选择活动`n`的时候面临的两种选择。如果采取第一选择，即参加活动`n`，我们便记录`S[n][0] = true`，`S[n][1] = p(n)`，代表我们在考虑活动`n`的时候选择了参加活动`n`，搭配上前面`{1, 2, ..., p(n)}`中的最优组合。如果采取第二选择，即放弃活动`n`，那么我们记录`S[n][0] = false`，`S[n][1] = n-1`，代表我们放弃活动`n`，此时活动的选择情况还是与之前考虑活动`n-1`时候的情况一致。于是我们可以得到如下算法：

```
OPT[n] = {0}
初始化 S[n][2]
For j = 1 to n:
    If OPT(p(j)) + vj >= OPT(j-1) then
        OPT[j] = OPT(p(j)) + vj
        S[j][0] = true
        S[j][1] = p[j]
    Else
        OPT(j) = OPT(j-1)
        S[j][0] = false
        S[j][1] = n-1
    Endif
Endfor
```



通过上面算法便可以得到想要的活动组合的记录表了。然后反向搜索一下记录表便可得到最优的活动组合。

```
j = n
while j != 0:
    If S[j][0] then
        print j
    Endif
    j = S[j][1]
Endwhile
```







### 2.分段最小二乘问题

给出一组n个点，需要确定一条能够最好的拟合（fit）这些点的直线。有error = 点到直线的距离的平方，而这条直线具有最小的error。

使用多条直线能够做到更好的拟合，但是分成越多的段对于我们的数据分析没有意义。因为有指数中不同的划分（相当于求子集），使用暴力搜索是不现实的。

解决方法：
我们引入一个可变的变量C，作为算法的惩罚值，拟合划分的线越多，C值越大。因此我们的目标就是找到最小的：C + 各条线的error值。
定义e(i,j) 是拟合pi到pj这些点时的error；OPT(i)为pi到pj的最优解（OPT(0) = 0）。此时可知：

```cpp
OPT(j)=e(i,j)+C+OPT(i−1)OPT(j)
  		=e(i,j)+C+OPT(i−1)
```

我们无法确定i的值，但是可以选择能够给出最小值的i的值。

```cpp
OPT(j)=min1≤i≤j（e(i,j)+C+OPT(i−1)）OPT(j)=min1≤i≤j（e(i,j)+C+OPT(i−1)）
```

由此我们可以得到伪代码：
计算每个点的分段最小二乘法的error值，然后储存在数组中，之后遍历数组来获得最好的拟合线。

```
SegmentedLeastSquares(n) 
    Array   M[0...n]
    M[0] ← 0
    for all pairs (i, j)
        compute least squares error[i, j] for segment pi . . . pj
    for j ← 1 to n
        M [j ] ← min1≤i≤j  (error[i, j ] + C + M [i − 1] )

Find Segments(j)
    find i that minimizes error[i,j]+C+M[i−1] 
    if i > 1 then Find Segments(i − 1)
    output the best fit line for segment pi...pj
12345678910111213
```

### 分段最小二乘

### 背包问题

### RNA二级结构

### 序列比对

#### 问题描述

**目标：** 给定两个字符串 $X = x_1 x_2 \ldots x_m$ 和 $Y = y_1 y_2 \ldots y_n$，寻找最小罚分的比对方式。

**定义：** 配对 $x_i - y_j$ 和 $x_{i'} - y_{j'}$ 称为**交叉**，如果 $i < i'$，但是 $j > j'$。

**定义：** 一个**比对** $M$ 是一些有序配对 $x_i - y_j$ 的集合，每集合中，每一项至多出现在一个配对中，而且没有配对交叉。

**定义：** $OPT(i, j)$ = 字符串 $x_1 x_2 \ldots x_i$ 与 $y_1 y_2 \ldots y_j$ 比对的最小罚分

- **Case 1: $x_i - y_j$ 在 $OPT$ 中** ： 罚分 = $x_i$ 与 $y_j$ 不匹配的罚分 + $OPT(i-1, j-1)$ 

- **Case 2a: $OPT$ 中 $x_i$ 没有匹配** ： 罚分 = $x_i$ 处空位罚分 + $OPT(i-1, j)$
- **Case 2b: $OPT$ 中 $y_j$ 没有匹配** *：罚分 = $y_j$ 处空位罚分 + $OPT(i, j-1)$

#### 序列比对 - 动态规划递推公式

**递推公式：**

$$
OPT(i, j) =
\begin{cases}
    j\delta & \text{if } i = 0 \\
    \min \begin{cases}
        \alpha_{x_i y_j} + OPT(i-1, j-1) \\
        \delta + OPT(i-1, j) \\
        \delta + OPT(i, j-1)
    \end{cases} & \text{otherwise} \\
    i\delta & \text{if } j = 0
\end{cases}
$$

其中，$\alpha_{x_i y_j}$ 是 $x_i$ 和 $y_j$ 比对的成本（相同则为 0，不同则为不匹配罚分），$\delta$ 是空位罚分。

**定理 6.16：** 对于 $i, j \ge 1$，
$$
OPT(i, j) = \min \begin{cases}
    \alpha_{x_i y_j} + OPT(i-1, j-1) \\
    \delta + OPT(i-1, j) \\
    \delta + OPT(i, j-1)
\end{cases}
$$
$(i, j)$ 在最优比对中，当且仅当达到上面最小值时。

```
Sequence-Alignment(m, n, X=x₁x₂...x<0xE2><0x82><0x98>, Y=y₁y₂...y<0xE2><0x82><0x99>, δ, α) {
  for i = 0 to m
  	M[i, 0] = iδ
  for j = 0 to n
  	M[0, j] = jδ
  for i = 1 to m
  	for j = 1 to n
  		M[i, j] = min(α[xᵢ, yⱼ] + M[i-1, j-1],
  							δ + M[i-1, j],
  							δ + M[i, j-1])
  return M[m, n]
}
```

时间空间复杂度$O(mn)$

#### 线性空间的序列比对：



```
Space-Efficient-Alignment(X, Y)
  数组 B[0...m, 0...1]
  初始化对每个 i 令 B[i, 0] = iδ
  For j = 1, ..., n
    B[0, 1] = jδ
    For i = 1, ..., m
      B[i, 1] = min(α[xᵢ, yⱼ] + B[i-1, 0], δ + B[i-1, 0], δ + B[i, 0])
    Endfor
    将 B 的第二列移到第一列，为下一次迭代留出空间
    对每个 i 修改 B[i, 0] = B[i, 1]
  Endfor
  return B[m, 0]
```

- **空间复杂度：** $\Theta(m)$ （或 $\Theta(\min(m, n))$ 如果考虑交换序列）。

-  **时间复杂度：** $\Theta(mn)$。

不能恢复比对本身。只能找数值，不能恢复过程。

#### Hirschberg's algorithm 最小编辑距离优化算法

在上面算法的基础上，Hirschberg提出了新的算法，将**动态规划** 和**分治法** 相结合，使得时间复杂度不变，空间复杂度优化到$O(m+n)$，并且可以输出对应的字符串。

![在这里插入图片描述](https://ad.itadn.com/c/weblog/00-blog-img/images/2024-12-15/AsTxm3ub2SJMH1YW5pZI7hv6wjc8.png)





