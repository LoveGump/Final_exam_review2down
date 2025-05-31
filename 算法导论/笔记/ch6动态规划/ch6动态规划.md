# ch6动态规划

## 6.1带权区间调度

带权区间调度问题 (Weighted Interval Scheduling problem) 是一个经典的算法问题。简单来说，你有一系列任务，每个任务都有一个**开始时间**、一个**结束时间**以及一个**权重** (或者说，完成这个任务能获得的“收益”)。你的目标是选择一部分任务来执行，要求是这些被选中的任务在时间上**不能重叠**，并且使得这些被选中任务的**总权重最大化**。

![image-20250530125121525](/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530125121525.png)

------

### 核心思想

解决这个问题通常采用**动态规划** (Dynamic Programming) 的方法。基本思路是：

1. **排序**：首先，按照任务的**结束时间**对所有任务进行升序排序。这一点非常关键。
2. **定义子问题**：设 `dp[i]` 表示考虑前 `i` 个任务 (按照结束时间排序后) 时，可以获得的最大权重。
3. 递推关系：对于第 $i$个任务，我们有两种选择：
   - **不选择任务 `i`**：那么 `dp[i]` 的值就等于 `dp[i-1]` (即不包含任务 `i` 时，前 `i-1` 个任务能获得的最大权重)。
   - **选择任务 `i`**：如果选择任务 `i`，那么我们就不能选择任何与任务 `i` 时间上重叠的任务。我们需要找到在任务 `i` 开始之前结束的最后一个任务 `j` (即 `finish_time[j] <= start_time[i]`)。这种情况下，`dp[i]` 的值就等于 `weight[i] + dp[j]` (任务 `i` 的权重加上不与任务 `i` 冲突的前 `j` 个任务能获得的最大权重)。如果不存在这样的任务 `j`，则 `dp[i]` 就等于 `weight[i]`。 因此，`dp[i]` 的最终值为以上两种选择中较大的那个： `dp[i] = max(dp[i-1], weight[i] + dp[p[i]])` 其中 `p[i]` 是在任务 `i` 开始之前结束的最后一个任务的索引 (如果不存在则为 0，且 `dp[0]` 设为 0)。
4. **寻找 `p[i]`**：为了有效地找到 `p[i]`，可以在排序后的任务列表中使用二分查找。

------

### 算法步骤概览

1. 将所有区间（任务）按照**结束时间**从小到大排序。
2. 创建一个 DP 数组 `dp`，`dp[i]` 表示处理到第 `i` 个区间时的最大权重。
3. 初始化 `dp[0]` 为 0 (或者根据实际情况，如果区间从 1 开始索引，则 `dp[0]` 为第 1 个区间的权重)。
4. 遍历排序后的区间：
   - 对于当前区间 `i`，计算其权重 `w_i`。
   - 找到在区间 `i` 开始之前结束的最后一个区间 `j` (即 `p[i]`)。这可以通过遍历或者二分查找实现。
   - $dp[i] = max(dp[i-1], w_i + dp[j])$ (如果 `j` 不存在，则 `dp[j]` 为 0)。
5. DP 数组中的最大值（通常是最后一个元素 `dp[n-1]`，如果 `n` 是区间数量）就是最终答案。



### 代码：

```c++
#include <iostream>
using namespace std;

const int MAX_REQ_NUM = 30;

struct Request {
  int beginTime;
  int endTime;
  int value;
};

bool operator<(const Request& r1, const Request& r2) {
  return r1.endTime < r2.endTime;
}

class DP {
public:
  void setRequestNum(const int& requestNum) {
    this->requestNum = requestNum;
  }
  void init() {
    for (int i = 1; i <= requestNum; ++i) {
      cin >> reqs[i].beginTime >> reqs[i].endTime >> reqs[i].value;
    }
  }
  // 预备，根据结束时间对所有请求排序，初始化数组p
  void prepare() {
    // 按照结束时间排序
    sort(reqs + 1, reqs + requestNum + 1);
    // 初始化p数组
    memset(p, 0, sizeof(p));
    // 计算每个请求的前一个请求
    for (int i = 1; i <= requestNum; ++i) {
      for (int j = i-1; j > 0; --j) {
        if (reqs[j].endTime <= reqs[i].beginTime) {
          p[i] = j;
          break;
        }
      }
    }
    memset(record, 0, sizeof(record));
  }
  // 动态规划算法
  void solve() {
    optimal[0] = 0;
    for (int i = 1; i <= requestNum; ++i) {
      // opt = max(opt[i-1], opt[p[i]] + reqs[i].value)
      if (optimal[p[i]] + reqs[i].value >= optimal[i-1]) {
        optimal[i] = optimal[p[i]] + reqs[i].value;
        record[i][0] = 1;
        record[i][1] = p[i];
      } else {
        optimal[i] = optimal[i-1];
        record[i][0] = 0;
        record[i][1] = i-1;
      }
    }
  }
  // 输出结果
  void output() {
    cout << "[MAX VALUE]: " << optimal[requestNum]
         << "\n[Activities]:\n";
    for (int i = requestNum; i > 0; i = record[i][1]) {
      if (record[i][0] == 1) {
        cout << "activity-" << i << endl;
      }
    }
  }
private:
  Request reqs[MAX_REQ_NUM + 1]; // 请求数组
  int requestNum; // 请求数量
  int p[MAX_REQ_NUM + 1]; // 记录每个请求的前一个请求
  int optimal[MAX_REQ_NUM + 1]; //  最优值数组
  int record[MAX_REQ_NUM + 1][2]; // 记录选择的请求
};

int main() {
  int requestNum;
  DP dp;
  cin >> requestNum;
  dp.setRequestNum(requestNum);
  dp.init();
  dp.prepare();
  dp.solve();
  dp.output();
  return 0;
}

```



------

### 复杂度

- 时间复杂度：
  - 排序：`O(n log n)`，其中 `n` 是任务数量。
  - 动态规划：对于每个任务，如果使用二分查找寻找 `p[i]`，则为 `O(n log n)`；如果线性扫描，则为 `O(n^2)`。通常采用二分查找优化。
  - 因此，总的时间复杂度通常是 `O(n log n)`。
- **空间复杂度**：`O(n)`，用于存储 DP 数组和排序后的任务。

这个问题在资源分配、任务调度等领域有广泛应用。

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530125410555.png" alt="image-2025053015410555" style="zoom:67%;" />

这样会造成很多资源的浪费，于是有采用递归备忘录的算法

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530125553613.png" alt="image-20250530125553613" style="zoom:67%;" />

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530125730786.png" alt="image-20250530125730786" style="zoom:67%;" />

这样只是得到了最优解的数值，如果我们需要计算最优的区间集合：

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530125924449.png" alt="image-20250530125924449" style="zoom:67%;" />

算法的时间复杂度也是$O(n)$

## 6.2分段的最小二乘

### 核心思想

动态规划的核心在于将原问题分解为一系列重叠的子问题，并存储子问题的解，以避免重复计算。对于分段最小二乘问题，这意味着我们逐步构建最优的分段方案。

------

### 动态规划解法步骤

假设我们有 $n$ 个数据点 $(x_1,y_1),(x_2,y_2),…,(x_n,y_n)$。

1. **定义子问题和状态**:
   - 我们定义 $DP[i]$ 为拟合前 $i$ 个数据点 $(x_1,y_1),…,(x_i,y_i)$ 所能得到的最小总误差（通常是残差平方和）。
   - 我们的目标是计算出 $DP[n]$。
2. **定义误差函数**:
   - 令 $error(j,i)$ 表示用一个单独的线段（或选定的模型，如线性模型）拟合数据点 $(x_j,y_j),…,(x_i,y_i)$ 所产生的最小二乘误差。这个误差可以通过标准的最小二乘法计算得到。
3. **建立递推关系**:
   - 对于 $DP[i]$，我们可以考虑最后一个线段覆盖了从 $j+1$ 到 $i $的数据点 (其中 $0≤j<i$)。
   - 那么，$DP[i]$ 的值可以通过尝试所有可能的最后一个分段点$ j$ 来获得：$ DP[i]=min_{0≤j<i}{DP[j]+error(j+1,i)}$
   - 这里的 $DP[0]$ 通常定义为$ 0$（没有点，没有误差）。
   - $DP[j]$ 是拟合前$ j$ 个点的最小误差，而 $error(j+1,i)$ 是当前最后一个线段（从点 $j+1$ 到点 $i$）的拟合误差。
4. **惩罚项 (可选)**:
   - 有时，我们不希望有过多的分段。可以在递推关系中加入一个**惩罚项** C (常数)，代表每增加一个分段的成本。 $DP[i]=min_{0≤j < i}{DP[j]+error(j+1,i)+C}$
   - 这样，算法会在拟合误差和分段数量之间进行权衡。选择合适的 $C$ 很重要。
5. **计算顺序**:
   - $DP$ 表可以从 $DP[0]$ 开始，依次计算 $DP[1],DP[2],…,DP[n]$。
   - 为了计算 $DP[i]$，你需要所有 $DP[j]$ (其中$ j<i$ ) 的值。
6. **回溯找到分段点**:
   - 在计算$ DP$ 表的过程中，通常会额外存储一个数组（例如 `P[i]`），记录使得 DP[i] 达到最小值的那个 $j$。
   - 当 $DP[n]$ 计算完毕后，可以通过 `P` 数组从后向前回溯，找出所有的最优分段点。例如，最后一个分段是从 $P[n]+1$ 到$ n$，倒数第二个分段是从 $P[P[n]]+1$ 到 $P[n]$，依此类推。

------

### 算法复杂度

- **预计算误差**: 计算所有可能的 $error(j,i)$。如果每个 $error(j,i)$ 的计算需要 $O(i−j)$ 时间（例如线性拟合），那么预计算所有 $O(n^2)$ 个区间的误差可能需要$ O(n^3)$ 的时间。在某些情况下，可以优化到 $O(n^2)$。
- **DP 计算**: DP 表的计算本身有 $n$ 个状态，每个状态需要$ O(n)$ 的转移，所以是 $O(n^2)$。

因此，总的时间复杂度通常是 $O(n^3)$ 或在优化误差计算后达到 $O(n^2)$。空间复杂度是 $O(n^2)$（用于存储 $error(j,i)$）或 $O(n)$（如果 $error(j,i) $即时计算）。

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530135353146.png" alt="image-20250530135353146" style="zoom:67%;" />

## 6.3子集和与背包

### 子集和问题 (Subset Sum Problem)

子集和问题

**问题定义：**

给定 **n 个项**，通常表示为集合 {1, 2, ..., n}。 每个项 **i** 都有一个给定的**非负权重** $w_i$。 同时，给定一个**上界（或容量）W**。

**目标：**

我们需要从这 n 个项中选择一个**子集 S**，需要满足以下两个条件：

1. **和不超过上界**：子集 S 中所有项的权重之和必须小于或等于 W。 数学表示为：$\sum_{i \in S}w_i \leq W$
2. **最大化子集和**：在满足第一个条件的前提下，我们希望选出的子集 S 的权重之和尽可能大。 数学表示为：最大化 $\sum_{i \in S}w_i$

简单来说，就是从一堆带重量的物品中，挑选一部分物品，要求它们的总重量不超过一个给定的限制，并且希望这个总重量尽可能地接近（或等于）这个限制，以达到最大化。

#### **算法：动态规划**

我们将使用一个一维数组 `dp` 来存储中间结果。

**1. 状态定义：**

`dp[j]`：表示当背包（或子集和的上限）容量为 `j` 时，能够从中选择的项的子集所能达到的最大权重和。我们的目标是求 `dp[W]`。

**2. 状态转移方程：**

对于集合中的每一个项（设其权重为 `w`，这里我们按顺序处理每个项 $w_i$）： 我们要决定是否将当前这个项 `w` 加入到子集中。

遍历所有可能的容量 `j`（从 `W` 向下到 `w`，这是为了保证每个项只被考虑一次，即0/1特性）：

- **不选择当前项 `w`**：那么 `dp[j]` 的值保持不变，等于在考虑当前项之前的 `dp[j]`。
- **选择当前项 `w`** (前提是 $j \geq w$，即当前容量 `j` 能够容纳下项 `w`)：那么 `dp[j]` 的值可以更新为 `dp[j-w] + w`。这表示，我们使用了当前项 `w`，然后用剩余的容量 `j-w` 去填充，所能得到的最大权重和是 `dp[j-w]`。

因此，状态转移方程为： `dp[j] = max(dp[j], dp[j-w] + w)`  (对于当前项的权重 `w`， `j` 从 `W` 递减到 `w`)

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530200627780.png" alt="image-20250530200627780" style="zoom:80%;" />

**3. 初始化：**

- `dp` 数组的大小为 `W+1`。
- `dp[0] = 0`：当容量为0时，能得到的最大权重和为0（不选任何项）。
- `dp[j] = 0` 对于所有 `j > 0`：在开始考虑任何项之前，对于任何正容量，最大和都是0。或者，可以理解为如果没有任何物品，那么对于任何容量j，最大和都是0。实际上，在迭代过程中，如果一个容量j不能由任何物品组合而成，它的值也会保持为0，或者是由比它小的容量加上某个物品的重量转移而来。更严谨的初始化是 `dp[0]=0`，其他 `dp[j]` 可以初始化为0（表示目前能达到的最大和）或者负无穷（如果要求必须恰好等于某个值，但这里是小于等于W并最大化，所以0是合适的）。

**4. 迭代过程：**

- 外层循环遍历每个项 $w_i$ (从 i=1 到 n)。
- 内层循环遍历容量 $j$ (从 $W$递减到 $w_i$)。
  - 在内层循环中，应用状态转移方程：$dp[j] = max(dp[j], dp[j-w_i] + w_i)$

**5. 最终结果：**

`dp[W]` 就是我们要求的最终答案，即在不超过总权重 `W` 的前提下，能得到的最大子集和。如果 `dp[W]` 的值仍然是初始值（比如0，并且没有任何物品的重量也是0），需要根据实际情况判断是否有解。但由于题目是最大化，所以 `dp[W]` 会给出在 ≤W 条件下的最大和。

**伪代码示例：**

```c++
function solve_subset_sum_maximization(weights, W):
  n = length(weights)
  dp = array of size (W + 1) initialized to 0

  for i from 0 to n-1: // 遍历每一个项
    current_weight = weights[i]
    for j from W down to current_weight: // 逆序遍历容量
      dp[j] = max(dp[j], dp[j - current_weight] + current_weight)

  return dp[W]
```

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530201032969.png" alt="image-20250530201032969" style="zoom:80%;" />

时间复杂度：

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530201154690.png" alt="image-20250530201154690" style="zoom:80%;" />

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530201300155.png" alt="image-20250530201300155" style="zoom:67%;" />

## 6.5RNA二级结构

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530202447841.png" alt="image-20250530202447841" style="zoom:67%;" />

对一个单螺旋RNA分子 $B = b1_,b_2,\dots ,b_n$ , 确定具有最大碱基配对个数的二级结构$S$.

### 动态规划

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530202702841.png" alt="image-20250530202702841" style="zoom:67%;" />



<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530202843929.png" alt="image-20250530202843929" style="zoom:67%;" />



<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530202945250.png" alt="image-20250530202945250" style="zoom:67%;" />

伪代码：

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530203349134.png" alt="image-20250530203349134" style="zoom:80%;" />

### 时间复杂度分析

1. **状态数量**: 动态规划表 `OPT(i, j)` 存储了所有子序列 `bᵢ...bⱼ` 的最优解。如果RNA序列的长度为 `n`，那么 `i` 和 `j` 的范围都是从 0 到 `n-1` (或1到 `n`)。因此，大约有 `n * n = n²` 个状态需要计算。所以状态数量为 $O(n^2)$。

2. **每个状态的计算时间**: 对于每个状态 `OPT(i, j)`，计算它需要：

   - `OPT(i, j-1)`: 这是一个查表操作，时间复杂度为 O(1)。
   - $1 + max_t { OPT(i, t-1) + OPT(t+1, j-1) }$:
     - 内部的 `max_t` 循环遍历 `t` 从 `i` 到 `j-1`。在最坏情况下，`t` 的取值范围可以达到 `j-i` 个，也就是 $O(n)$。
     - 在 `max_t`循环的每一次迭代中：
       - `can_pair(bₜ, bⱼ)`: 判断是否可以配对，通常是 $O(1)$。
       - `OPT(i, t-1)` 和 `OPT(t+1, j-1)`: 查表操作，各为 $O(1)$。
       - 加法和比较操作： $O(1)$。
     - 因此，计算 `max_t { ... }` 部分的时间复杂度为 $O(n)$。

   所以，计算一个状态 `OPT(i, j)` 的总时间是$ O(1)+O(n)=O(n)$。

3. **总时间复杂度**: 总时间复杂度 = (状态数量) × (每个状态的计算时间) 总时间复杂度 = $O(n^ 2)×O(n)=O(n^3)$。

**空间复杂度**: 存储动态规划表 `OPT(i, j)` 需要 $O(n^2)$ 的空间。

------

### C++ 代码示例

```c++
#include <iostream>
#include <vector>
#include <string>
#include <algorithm> // For std::max

/**
 * @brief 判断两个碱基是否可以配对
 * @param base1 第一个碱基
 * @param base2 第二个碱基
 * @return 如果可以配对返回 true，否则返回 false
 */
bool can_pair(char base1, char base2) {
    if ((base1 == 'A' && base2 == 'U') || (base1 == 'U' && base2 == 'A')) {
        return true;
    }
    if ((base1 == 'G' && base2 == 'C') || (base1 == 'C' && base2 == 'G')) {
        return true;
    }
    // Add G-U wobble pair if needed
    if ((base1 == 'G' && base2 == 'U') || (base1 == 'U' && base2 == 'G')) {
         return true;
    }
    return false;
}
/**
 * @brief 计算最大碱基对数，使用动态规划方法
 * @param sequence RNA 序列
 * @return 最大碱基对数
 * @attention 每个对的两端被至少四个插入的碱基所分割
 */
int solve_max_base_pairs_cpp(const std::string& sequence) {
    int n = sequence.length();
    if (n < 6) return 0;

    std::vector<std::vector<int>> opt(n, std::vector<int>(n, 0));

    // 按区间长度递增填表
    for (int len = 1; len <= n; ++len) {
        for (int i = 0; i + len - 1 < n; ++i) {
            int j = i + len - 1;
            // 情形1：区间太短，不能配对
            if (i >= j - 4) {
                opt[i][j] = 0;
                continue;
            }
            // 情形2：bj不配对
            int res = opt[i][j-1];

            // 情形3：bj与bt配对
            for (int t = i; t <= j - 4; ++t) {
                if (can_pair(sequence[t], sequence[j])) {
                    int left = (t > i) ? opt[i][t-1] : 0;
                    int right = (t+1 <= j-1) ? opt[t+1][j-1] : 0;
                    res = std::max(res, 1 + left + right);
                }
            }
            opt[i][j] = res;
        }
    }
    return opt[0][n-1];
}
```

## 6.6序列比对

### 全局序列比对问题 (Global Alignment Problem)

**全局序列比对问题**旨在找出两条给定序列（例如长度为 m 的序列 S1 和长度为 n 的序列 S2）在它们**整个长度**上的最优比对。“最优性”由一个打分系统确定，该系统通常包括：

- **匹配得分 (match score)**：当相同字符对齐时获得的分数。
- **错配罚分 (mismatch penalty)** (或得分)：当不同字符对齐时应用的罚分（或较低的得分）。
- **空位罚分 (gap penalty)**：当一个字符与一个空位（代表插入或删除事件）对齐时应用的罚分。为简单起见，我们这里将考虑一个**线性空位罚分 (linear gap penalty)**，即每个空位导致一个恒定的罚分，记为 g (通常为一个负值或从匹配/错配得分中扣除的值)。

目标是找到一个具有最大可能总得分（或最小总罚分）的比对。

------

### Needleman-Wunsch 算法步骤

Needleman-Wunsch 算法使用**动态规划 (dynamic programming)** 来解决此问题。它主要包括三个阶段：

1. **初始化 (Initialization)**： 创建一个大小为$ (m+1)×(n+1)$ 的二维矩阵（称之为 $F$）。矩阵的行对应序列 $S_1$ 的字符（加一个初始空行），列对应序列$ S_2$ 的字符（加一个初始空列）。

   - $F[0][0]=0$。
   - 初始化第一行：对于 $j=1…n$，$F[0][j]=F[0][j−1]+g$ (或者直接 $j×g $如果$ g$ 是每单位空位的罚分)。这代表序列 $S_2$ 的前 $j$ 个字符与序列$ S_1$ 中的 $j$个空位对齐。
   - 初始化第一列：对于 $i=1…m$，$F[i][0]=F[i−1][0]+g $(或者直接 $i×g$)。这代表序列 S1 的前 i 个字符与序列 S2 中的 i 个空位对齐。

2. **矩阵填充 (Scoring)**： 迭代填充矩阵 $F$ 的其余部分。对于每个单元格$ F[i][j]$（其中 $1≤i≤m$ 且 $1≤j≤n$），其得分是基于以下三种可能性中的最大值来计算的：

   - **S1[i] 与 S2[j] 对齐 (对角线方向)**： 得分 = $F[i−1][j−1]+score(S_1[i],S_2[j])$。 其中，$score(S_1[i],S_2[j])$ 是 $S_1$ 的第 $i$ 个字符 ($S_1[i]$) 与 $S_2$ 的第 $j$ 个字符 ($S_2[j]$) 对齐的得分（匹配得分或错配罚分）。
   - **S1[i] 与一个空位对齐 (来自上方)**： 得分 = $F[i−1][j]+g$。
   - **S2[j] 与一个空位对齐 (来自左方)**： 得分 = $F[i][j−1]+g$。

   因此，F[i][j] 的计算公式为：
   $$
   F[i][j] = \max
   \begin{cases}
   F[i-1][j-1] + \text{score}(S_1[i], S_2[j]) \\
   F[i-1][j] + g \\
   F[i][j-1] + g
   \end{cases}
   $$
   

   在此步骤中，通常会额外存储一个指针矩阵，记录下每个单元格$ F[i][j] $的值是由哪个（或哪些，如果得分相同)前驱单元格计算得到的，这将在回溯阶段使用。

3. **回溯 (Traceback / Alignment Reconstruction)**： 一旦整个矩阵填充完毕，最优全局比对的总得分就位于矩阵的右上角单元格$ F[m][n]$。要重建比对本身：

   - 从单元格$ F[m][n] $开始。
   - 根据计算 $F[i][j]$ 时选择的路径（即指针矩阵的指示，或重新比较三个来源的得分）反向移动到前一个单元格：
     - 如果 $F[i][j] $的值来源于 $F[i−1][j−1]+score(S1[i],S2[j])$（对角线移动），则意味着 $S_1[i]$ 与 $S_2[j]$ 对齐。在比对结果中记录 $S_1[i] $和 $S_2[j]$。然后移动到 $F[i−1][j−1]$。
     - 如果$ F[i][j] $的值来源于$ F[i−1][j]+g$（向右移动的结果），则意味着$ S_1[i]$ 与一个空位对齐。在比对结果中记录 $S_1[i]$ 和 `-` (空位符)。然后移动到$ F[i−1][j]$。
     - 如果$ F[i][j] $的值来源于$ F[i][j−1]+g$（向上移动的结果），则意味着$ S_2[j] $与一个空位对齐。在比对结果中记录 `-` 和 $S2[j]$。然后移动到$ F[i][j−1]$。
   - 重复此过程，直到到达 $F[0][0]$。
   - 由于比对是从后向前构建的，所以最后需要将得到的比对序列反转，即可得到最终的全局比对结果。

------

### 时间复杂度证明

设序列 $S_1$ 的长度为 $m$，序列 $S_2$ 的长度为 $n$。

1. **初始化阶段**：

   - 设置$ F[0][0] $需要常数时间，O(1)。
   - 初始化第一行（n 个单元格）和第一列（m 个单元格）。每个单元格的计算需要常数时间。
   - 总初始化时间：$O(m+n)$。

2. **矩阵填充阶段**：

   - 需要填充的矩阵部分有 $m×n $个单元格（即$ F[i][j] $其中$ 1≤i≤m $且$ 1≤j≤n$）。

   - 对于每个单元格 $F[i][j]$ 的计算：

     - 需要访问$ F[i−1][j−1]$、$F[i−1][j] $和 $F[i][j−1] $这三个先前计算好的值（常数时间）。
  - 进行一次字符比较 $S_1[i]$ vs$ S_2[j]$ 以获取$ score(S_1[i],S_2[j]) $（常数时间）。
     - 进行三次加法运算（常数时间）。
  - 进行两次比较运算以找出三个值的最大值（常数时间）。
     - 进行一次赋值操作（常数时间）。
     - 如果需要存储指针，也需要常数时间。
     
   - 因此，计算每个单元格的得分需要常数时间，记为 O(1)。
   
   - 由于有 $m×n $个单元格需要填充，此阶段的总时间为 $O(m×n)$。

3. **回溯阶段**：

   - 回溯路径从$ F[m][n]$ 开始，到$ F[0][0] $结束。
   - 在每一步回溯中，我们从当前单元格$ (i,j) $移动到 $(i−1,j−1)$、$(i−1,j)$ 或 $(i,j−1)$ 中的一个。这意味着$ i$ 或 $j$ (或两者) 至少减 1。
   - 路径的最大长度为$ m+n$（例如，如果路径只包含向上和向左的移动，或者只包含对角移动）。
   - 在回溯的每一步中，决定移动方向和构建比对字符的操作都需要常数时间。
   - 因此，回溯阶段的总时间为$ O(m+n)$。

**总体时间复杂度**： 总时间复杂度是以上三个阶段的复杂度之和：$ T=O(m+n) (初始化)+O(m×n) (矩阵填充)+O(m+n) (回溯)$

在这些项中，O(m×n) 是主导项（假设 m 和 n 不是非常小）。 因此，**Needleman-Wunsch 算法的时间复杂度为 O(m×n)**。

------

<img src="/Users/gump/大二资料（更新版）/alg/all_files/笔记/ch6/insert/image-20250530212207858.png" alt="image-2020530212207858" style="zoom:80%;" />

```c++
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
using namespace std;

/**
 * @brief 判断字符是否为元音字母
 * @param ch 字符
 * @return 如果是元音字母返回 true，否则返回 false
 */
bool is_vowel(char ch) {
    ch = tolower(ch);
    return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u';
}

/**
 * @brief 计算两个字符的匹配分数
 * @param a 第一个字符
 * @param b 第二个字符
 * @return 匹配分数
 */
int match_score(char a, char b) {
    // 匹配分数规则：
    // 1. 相同字符得分0
    if (a == b) return 0;
    // 2. 元音与元音或辅音与辅音得分1
    if (is_vowel(a) && is_vowel(b)) return 1;
    if (!is_vowel(a) && !is_vowel(b)) return 1;
    // 3. 元音与辅音得分3
    return 3;
}

/**
 * @brief 使用动态规划进行序列比对
 * @param seq1 第一个序列
 * @param seq2 第二个序列
 * @param gap_penalty 缺口罚分
 * @return 最小罚分
 */
int sequence_alignment(const string& seq1, const string& seq2, int gap_penalty = 2) {
    int m = seq1.size(), n = seq2.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

    // 初始化
    for (int i = 0; i <= m; ++i) dp[i][0] = i * gap_penalty;
    for (int j = 0; j <= n; ++j) dp[0][j] = j * gap_penalty;

    // 动态规划填表
    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int cost = match_score(seq1[i - 1], seq2[j - 1]);
            dp[i][j] = min({
                dp[i - 1][j - 1] + cost, // 匹配/替换
                dp[i - 1][j] + gap_penalty, // seq1 匹配一个空
                dp[i][j - 1] + gap_penalty  // seq2 匹配一个空
            });
        }
    }

    // 回溯输出比对路径
    string align1, align2;
    int i = m, j = n;
    while (i > 0 && j > 0) {
        int score = dp[i][j];
        if (score == dp[i - 1][j - 1] + match_score(seq1[i - 1], seq2[j - 1])) {
            align1 = seq1[i - 1] + align1;
            align2 = seq2[j - 1] + align2;
            --i; --j;
        } else if (score == dp[i - 1][j] + gap_penalty) {
            align1 = seq1[i - 1] + align1;
            align2 = '-' + align2;
            --i;
        } else {
            align1 = '-' + align1;
            align2 = seq2[j - 1] + align2;
            --j;
        }
    }
    while (i > 0) {
        align1 = seq1[i - 1] + align1;
        align2 = '-' + align2;
        --i;
    }
    while (j > 0) {
        align1 = '-' + align1;
        align2 = seq2[j - 1] + align2;
        --j;
    }

    // 输出结果
    cout << "最小罚分: " << dp[m][n] << endl;
    cout << "比对结果:" << endl;
    cout << align1 << endl;
    cout << align2 << endl;

    return dp[m][n];
}
```

## 6.7通过分治策略在**线性空间**的序列比对

### 背景：标准动态规划的空间问题

标准的全局比对算法（如 Needleman-Wunsch）需要一个 O(mn) 大小的动态规划矩阵来存储所有子问题的得分。当序列非常长时（例如在基因组比对中），这样大的空间需求是不可接受的。Hirschberg 算法巧妙地解决了这个问题。

------

### `Hirschberg` 算法核心思想

`Hirschberg` 算法的核心思想是：

1. **线性空间计算中间行/列的最优划分点**：它利用一个事实——计算动态规划矩阵中任何一行的值，只需要前一行（或当前行和前一行）的值。因此，可以用线性空间（例如 $O(n)$，如果$ n $是较短序列的长度）计算出整个动态规划矩阵的最后一行（或列）的得分，或者任何中间行的得分。
2. 分治：
   - 首先，算法找到序列 $S_1$ (长度 m) 的中间点（比如第 $\frac {m} {2}$ 行）。
   - 然后，它在线性空间内计算出最优比对路径穿过这一中间行的哪一列（设为 $j_{split}$）。
   - 一旦找到这个划分点 $( \frac m2,j_{split})$，原问题就被分解为两个较小的子问题：
     1. 比对 $S_1[1\dots \frac m2]$ 和$ S_2[1\dots j_{split}]$。
     2. 比对$ S_1[ \frac m2 + 1 \dots m]$ 和 $S_2[j_{split}+1\dots n]$。
   - 这两个子问题再通过递归调用 `Hirschberg` 算法本身来解决。
3. **递归基例**：当序列足够短时（例如，其中一个序列长度为0或1），可以直接用标准的 `Needleman-Wunsch` 算法（此时空间消耗很小）或直接给出比对结果。

------

###  如何在线性空间找到中间划分点

假设我们要比对序列 $S_1$ (长度 $m$) 和$ S_2$ (长度 $n$)。我们选择 $S_1$ 的中间行 $i_{mid}= \frac m2$。

1. **计算前向得分 (Forward Scores)**：
   - 使用 Needleman-Wunsch 算法的递推关系，但只保留两行动态规划表（当前行和上一行），从左上角 $(0,0)$ 开始计算到第 $i_{mid}$ 行。
   - 这样可以得到$S_1[1\dots i_{mid}]$ 和$ S_2[1\dots j]$对齐的对于所有 $j \in [0,n]$ 的最优得分。我们称这一行的得分为 $ScoreL[j]$。此步骤空间复杂度为 O(n)。
2. **计算后向得分 (Backward Scores)**：
   - 同样地，我们可以通过比对 $S_1$ 的后半部分（从 $S_1[m]$ 到 $S_1[i_{mid}+1]$，即$ S_1 $的反向序列的后半部分）与 $S_2$ 的相应部分（从$ S_2[n] $到 $S_2[j+1]$，即$ S_2$ 的反向序列的后半部分）来计算得分。
   - 这等价于计算反向序列 $S_1rev$ 和 $S_2rev$ 的比对。具体来说，是从右下角 $(m,n)$ 开始，向上计算到第 $i_{mid}$ 行。我们只保留两行动态规划表，计算出从 $(m,n)$ 到 $(i_{mid},j) $的最优路径得分。
   - 这样可以得到 $S_1[i_{mid}+1…m] $与 $S_2[j+1…n]$ 对齐的对于所有$ j∈[0,n]$ 的最优得分。我们称这一行的得分为 $ScoreR[j] $(注意索引对应关系，通常是$ S_1rev[1…m−i_{mid}]$ 与 $S_2rev[1…n−j]$ 的得分)。此步骤空间复杂度也为 $O(n)$。
3. **找到划分列 $j_{split}$**：
   - 最优比对路径一定会穿过第 $i_{mid}$ 行的某个单元格 $(i_{mid},j)$。
   - 对于第 $i_{mid}$ 行的每一个可能的列 j（从 0 到 n），通过 $(i_{mid},j)$ 的最优路径的总得分是 $ScoreL[j]+ScoreR[j]$ (这里 $ScoreR[j]$ 指的是从 $(m,n) $到 $(i_{mid},j)$ 的路径得分，或者更准确地说是 $S_1[i_{mid}…m]$ 与 $S_2[j…n]$ 比对时，强制 $S_1[i_{mid}]$ 与 $S_2[j]$ 对齐或其中一个与空位对齐的后续路径得分；或者更简单地， $S_1[1…i_{mid}] $与 $S_2[1…j]$ 的得分加上 $S_1[i_{mid}+1…m]$ 与 $S_2[j+1…n]$ 的得分)。
   - 我们需要找到一个列 $j_{split}$，使得 $ScoreL[j_{split}]+ScoreR[j_{split}] $最大。这个$ j_{split}$ 就是最优路径在第 $i_{mid}$ 行穿过的列。



## 6.8图中的最短路径--没有负圈`Dijkstra`算法

Dijkstra 算法**单源最短路径算法**。它用于解决带权重的有向图或无向图中，从一个指定的起始节点（源点）到图中所有其他节点的最短路径问题。**标准的 Dijkstra 算法要求图中所有边的权重都是非负的**。

### 核心原理 (Core Principle)

Dijkstra 算法的核心思想是**贪心策略 (Greedy Strategy)**，并结合了**广度优先搜索 (BFS)** 的一些特点。它逐步构建最短路径树，具体来说：

1.  **初始化**:
    * 创建一个距离数组 `dist`，`dist[s]` (源点 s 到自身的距离) 初始化为 0，所有其他节点 `v` 的 `dist[v]` 初始化为无穷大（表示目前不可达）。
    * 创建一个集合 `S` (或者标记数组)，用于存放已经找到最短路径的节点。初始时，`S` 为空。
    * 创建一个优先队列 `Q` (通常用最小堆实现)，并将所有节点加入队列，节点的优先级由其在 `dist` 数组中的值决定。

2.  **迭代过程**:
    * 当 `Q` 不为空时，执行以下操作：
        * 从 `Q` 中提取出当前 `dist` 值最小的节点 `u` (贪心选择)。
        * 将节点 `u` 加入到集合 `S` 中 (表示 `u` 的最短路径已确定)。
        * 对于节点 `u` 的每一个未被加入 `S` 的邻居节点 `v`：
            * 通过 `u` 到达 `v` 的路径长度为 `dist[u] + weight(u, v)` (其中 `weight(u, v)` 是边 `(u, v)` 的权重)。
            * 如果这个路径长度小于 `dist[v]` (即发现了更短的路径)，则更新 `dist[v]` 的值为 `dist[u] + weight(u, v)`。这个过程称为**松弛 (Relaxation)**。同时，更新 `v` 在优先队列 `Q` 中的优先级（如果使用优先队列）。

3.  **结束**:
    * 当所有节点都被加入到 `S` 中 (或者优先队列 `Q` 为空) 时，算法结束。此时，`dist` 数组中存储的就是从源点 `s` 到各个节点的最短路径长度。

---

### 算法步骤 (Algorithm Steps)

以下是 Dijkstra 算法更具体的步骤：

1.  **准备工作**:
    * 给定一个图 $G=(V, E)$，其中 $V$ 是顶点集合，$E$ 是边集合，以及一个源顶点 $s \in V$。
    * 为每个顶点 $v \in V$ 设置一个距离属性 $d[v]$，并初始化 $d[s] = 0$，$d[v] = \infty$ (对于所有 $v \neq s$）。
    * 创建一个集合 $S$ (已访问顶点集)，初始为空。
    * 创建一个优先队列 $Q$，并将所有顶点加入其中。

2.  **主循环**:
    * **While** $Q$ 不为空 **do**:
        * $u \leftarrow$ $Q$ 中具有最小 $d$ 值的顶点 (Extract-Min 操作)。
        * $S \leftarrow S \cup \{u\}$ (将 $u$ 加入已访问集合)。
        * **For** 每一条从 $u$ 出发的边 $(u, v)$ **do**:
            * **If** $d[v] > d[u] + w(u, v)$ **then** (其中 $w(u, v)$ 是边 $(u, v)$ 的权重):
                * $d[v] \leftarrow d[u] + w(u, v)$ (松弛操作)。
                * 如果使用优先队列，需要更新 $v$ 在 $Q$ 中的优先级 (Decrease-Key 操作)。
                * 通常还需要记录路径，可以设置一个前驱数组 `prev[v] = u`，表示 $v$ 的最短路径上的前一个节点是 $u$。

3.  **结果**:
    * 算法结束后，$d[v]$ 就是从源点 $s$ 到顶点 $v$ 的最短路径长度。可以通过前驱数组 `prev` 回溯得到具体的最短路径。

---

### 图解示例 (Illustrative Example)

![Dijkstra Algorithm Animation](https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif)
*(图片来源: Wikimedia Commons, 作者: Redjar)*

上图展示了 Dijkstra 算法在一个简单图上的执行过程。算法从源点开始，逐步扩展到邻近节点，不断更新到达各节点的最短距离，直到所有可达节点的最短路径都被找到。红色节点表示已确定最短路径的节点，边上的数字代表权重。

---

### 优缺点 (Advantages and Disadvantages)

**优点 (Advantages):**

* **简单直观**: 算法逻辑相对容易理解和实现。
* **高效性**: 对于稀疏图，使用优先队列（如斐波那契堆）优化后，时间复杂度可以达到 $O(E + V \log V)$，其中 $E$ 是边数，$V$ 是顶点数。对于稠密图，使用邻接矩阵和简单扫描，时间复杂度为 $O(V^2)$。
* **保证最优解**: 只要边的权重为非负，Dijkstra 算法总能找到从源点到其他所有节点的最短路径。
* **应用广泛**: 是许多路由协议和路径规划问题的基础算法。

**缺点 (Disadvantages):**

* **不能处理负权边**: 如果图中存在负权重的边，Dijkstra 算法可能无法给出正确的最短路径。在这种情况下，需要使用 Bellman-Ford 算法或 SPFA 算法。
* **单源限制**: 算法计算的是从一个源点到所有其他节点的最短路径。如果需要计算所有节点对之间的最短路径，可以对每个节点运行一次 Dijkstra 算法，或者使用 Floyd-Warshall 算法。
* **全局搜索**: 对于某些特定场景，如果只需要找到到单个目标点的最短路径，且图中存在启发式信息（例如地理位置），A\* 算法可能会更高效。

### 代码

```c++
#include <algorithm>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>
using namespace std;

class DijkstraHeap {
   private:
    int vertices;
    vector<vector<pair<int, int>>> graph;  // 邻接表存储图，pair<目标顶点, 权重>

   public:
    DijkstraHeap(int v) : vertices(v), graph(v) {}

    /**
     * @brief 添加一条边到图中
     * @param u 起点
     * @param v 终点
     * @param weight 边的权重
     */
    void addEdge(int u, int v, int weight) { graph[u].push_back({v, weight}); }

    /**
     * @brief 使用Dijkstra算法计算从起点到所有顶点的最短路径
     * @param start 起点
     * @return 返回一个pair，包含距离数组和前驱节点数组
     */
    pair<vector<int>, vector<int>> dijkstra(int start) {
        // 初始化距离数组和前驱节点数组
        vector<int> distances(vertices, numeric_limits<int>::max());
        vector<int> predecessors(vertices, -1);
        distances[start] = 0;

        // 优先队列，存储{距离, 顶点}对
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
        pq.push({0, start});

        while (!pq.empty()) {
            int currentDistance = pq.top().first;
            int currentVertex = pq.top().second;
            pq.pop();

            // 如果当前距离大于已知距离，跳过
            if (currentDistance > distances[currentVertex]) {
                continue;
            }

            // 遍历当前顶点的所有邻居
            for (const auto& neighbor : graph[currentVertex]) {
                int nextVertex = neighbor.first;
                int weight = neighbor.second;
                int distance = currentDistance + weight;

                // 如果找到更短的路径，更新距离和前驱节点
                if (distance < distances[nextVertex]) {
                    distances[nextVertex] = distance;
                    predecessors[nextVertex] = currentVertex;
                    pq.push({distance, nextVertex});
                }
            }
        }

        return {distances, predecessors};
    }

    /**
     * @brief 获取从起点到终点的最短路径
     * @param start 起点
     * @param end 终点
     * @return 返回一个包含路径顶点的向量
     */
    vector<int> getShortestPath(int start, int end) {
        auto [distances, predecessors] = dijkstra(start);
        vector<int> path;

        if (distances[end] == numeric_limits<int>::max()) {
            return path;  // 没有路径
        }

        // 重建路径
        for (int current = end; current != -1;
             current = predecessors[current]) {
            path.push_back(current);
        }
        reverse(path.begin(), path.end());
        return path;
    }
};
```



# Bellman-Ford （可以处理带负权的边）

Bellman-Ford 算法（贝尔曼-福特算法）是一种用于解决**带权有向图中单源最短路径问题**的算法。与 Dijkstra 算法不同，**Bellman-Ford 算法可以处理边权为负数的情况**。此外，它还能**检测图中是否存在从源点可达的负权环路（Negative Cycle）**。如果存在负权环路，那么最短路径理论上可以无限小（通过不断在环路中循环）。

### 核心原理 (Core Principle)

Bellman-Ford 算法的核心思想是**迭代松弛 (Iterative Relaxation)**。它通过对图中的所有边进行多次松弛操作，逐步逼近最短路径。

1. **初始化**:

   - 创建一个距离数组 `dist`，`dist[s]` (源点 `s` 到自身的距离) 初始化为 0，所有其他节点 `v` 的 `dist[v]` 初始化为无穷大（表示目前从源点不可达）。
   - （可选）创建一个前驱数组 `predecessor`，用于记录最短路径上的前一个节点，方便路径回溯。`predecessor[v]` 初始化为 `null`。

2. **迭代松弛**:

   - 对图中的所有边进行 `|V| - 1` 次迭代（其中 `|V|` 是图中顶点的数量）。
   - 在每一次迭代中，遍历图中的每一条边 `(u, v)`，如果通过顶点 `u` 到达顶点 `v` 的路径比当前已知的到 `v` 的路径更短（即 `dist[u] + weight(u, v) < dist[v]`），则更新 `dist[v]` 的值为 `dist[u] + weight(u, v)`，并记录 `predecessor[v] = u`。

   这个过程基于一个重要的性质：从源点 `s` 到任何其他顶点 `v` 的最短路径，如果不存在负权环路，其包含的边数最多为 `|V| - 1`。因此，经过 `|V| - 1` 次迭代后，所有不涉及负权环路的最短路径都应该已经被找到。

3. **负权环路检测**:

   - 在完成 `|V| - 1` 次迭代后，再进行**一次额外**的迭代。
   - 遍历图中的每一条边 `(u, v)`，如果仍然可以进行松弛操作（即 `dist[u] + weight(u, v) < dist[v]`），则说明图中存在从源点可达的负权环路。这意味着没有定义明确的最短路径（因为可以无限次地遍历这个负权环来减少路径长度）。

------

### 算法步骤 (Algorithm Steps)

以下是 Bellman-Ford 算法更具体的步骤：

1. **初始化**:

   - 对于图中的每个顶点 `v`：
     - `dist[v] = ∞`
     - `predecessor[v] = null`
   - `dist[source] = 0`

2. **迭代松弛**:

   - For `i` from `1` to `|V| - 1` do:

     - For 每一条边 `(u, v)`及其权重 `w(u, v)` 在图 `G` 中 ` do`:
       
       - If `dist[u] + w(u, v) < dist[v]` then :
       
           - `dist[v] = dist[u] + w(u, v)`
       
           - `predecessor[v] = u`
       
   
3. **检测负权环路**:

   - For 每一条边 `(u, v)`及其权重 `w(u, v)`在图 `G`中 do:

      - If `dist[u] + w(u, v) < dist[v]`   then :

       - **Return** "图中存在负权环路" （或者标记受影响的节点，表明它们的最短路径无法确定）
   
4. **返回结果**:

   - 如果未检测到负权环路，则 `dist` 数组包含从源点到所有其他顶点的最短路径长度，`predecessor` 数组可以用来重构最短路径。

------

### 优缺点 (Advantages and Disadvantages)

**优点 (Advantages):**

- **能处理负权边**: 这是 Bellman-Ford 算法相对于 Dijkstra 算法最主要的优势。
- **能检测负权环路**: 可以报告图中是否存在使得最短路径无限小的负权环路。
- **原理相对简单**: 算法的迭代松弛思想比较直观。

**缺点 (Disadvantages):**

- **时间复杂度较高**: Bellman-Ford 算法的时间复杂度为 O(V⋅E)，其中 V 是顶点数，E 是边数。在稠密图中，这可能高达 O(V3)。相比之下，Dijkstra 算法使用优先队列优化后可以达到 O(E+VlogV)，在很多情况下更快。
- **对于没有负权边的图，Dijkstra 更优**: 如果图中所有边的权重都是非负的，Dijkstra 算法通常是更好的选择，因为它更快。

------



```c++
#include <iostream>
#include <vector>
#include <limits>
using namespace std;

class BellmanFord {
private:
    int vertices;
    struct Edge {
        int from, to, weight;
        Edge(int f, int t, int w) : from(f), to(t), weight(w) {}
    };
    vector<Edge> edges;

public:
    BellmanFord(int v) : vertices(v) {}

    /**
     * @brief 添加一条边到图中
     * @param from 起点
     * @param to 终点
     * @param weight 边的权重
     */
    void addEdge(int from, int to, int weight) {
        edges.emplace_back(from, to, weight);
    }

    /**
     * @brief 使用Bellman-Ford算法计算从起点到所有顶点的最短路径
     * @param start 起点
     * @return 返回一个pair，包含距离数组和前驱节点数组，如果存在负环则返回空数组
     */
    pair<vector<int>, vector<int>> bellmanFord(int start) {
        // 初始化距离数组和前驱节点数组
        vector<int> distances(vertices, numeric_limits<int>::max());
        vector<int> predecessors(vertices, -1);
        distances[start] = 0;

        // 进行V-1次松弛操作
        for (int i = 1; i < vertices; i++) {
            for (const Edge& edge : edges) {
                if (distances[edge.from] != numeric_limits<int>::max() &&
                    distances[edge.from] + edge.weight < distances[edge.to]) {
                    distances[edge.to] = distances[edge.from] + edge.weight;
                    predecessors[edge.to] = edge.from;
                }
            }
        }

        // 检查是否存在负环
        for (const Edge& edge : edges) {
            if (distances[edge.from] != numeric_limits<int>::max() &&
                distances[edge.from] + edge.weight < distances[edge.to]) {
                // 存在负环，返回空数组
                return {{}, {}};
            }
        }

        return {distances, predecessors};
    }

    /**
     * @brief 获取从起点到终点的最短路径
     * @param start 起点
     * @param end 终点
     * @return 返回路径列表，如果不存在路径或存在负环则返回空列表
     */
    vector<int> getShortestPath(int start, int end) {
        auto [distances, predecessors] = bellmanFord(start);

        // 检查是否存在负环或无法到达终点
        if (distances.empty() || distances[end] == numeric_limits<int>::max()) {
            return {};
        }

        // 重建路径
        vector<int> path;
        for (int current = end; current != -1; current = predecessors[current]) {
            path.push_back(current);
        }
        reverse(path.begin(), path.end());
        return path;
    }
};

```









------

## 背包问题 (Knapsack Problem)

背包问题有多种变体，这里我们讨论最常见的 **0/1 背包问题**。

**问题描述:** 给定 `n` 个物品，每个物品有一个重量 `w[i]` 和一个价值 `v[i]`。有一个背包，其最大承重为 `W`。如何选择物品放入背包，使得在不超过背包承重的前提下，包内物品的总价值最大。每个物品要么不选，要么选一次（0/1选择）。

**动态规划解法:**

我们可以创建一个二维整数数组 `dp`，其中 `dp[i][j]` 表示在前 `i` 个物品中选择，放入容量为 `j` 的背包中所能获得的最大价值。

**状态定义:** `dp[i][j]`：整数，表示从前 `i` 个物品中进行选择，当背包容量为 `j` 时，能获得的最大总价值。

**状态转移方程:** 对于第 `i` 个物品 (对应 `weights[i-1]` 和 `values[i-1]`)，我们有两种选择：

1. **不放入第 `i` 个物品**: 如果不放入当前物品，那么最大价值与只考虑前 `i-1` 个物品在容量为 `j` 的背包中的最大价值相同。即 `dp[i][j] = dp[i-1][j]`。
2. **放入第 `i` 个物品**: 如果放入当前物品 (前提是背包剩余容量 `j >= weights[i-1]`)，那么当前物品的价值 `values[i-1]` 会被计入，剩余的容量 `j - weights[i-1]` 用来从前 `i-1` 个物品中获取最大价值。即 `dp[i][j] = values[i-1] + dp[i-1][j - weights[i-1]]`。

我们需要在这两种选择中取价值较大者： `dp[i][j] = max(dp[i-1][j], values[i-1] + dp[i-1][j - weights[i-1]])`  (当 `j >= weights[i-1]` 时) 如果 `j < weights[i-1]` (当前物品放不下)，则只能不放： `dp[i][j] = dp[i-1][j]`

**基本情况 (Base Cases):**

- `dp[0][j] = 0` 对于所有 `j >= 0`：不选择任何物品时，总价值为 0。
- `dp[i][0] = 0` 对于所有 `i >= 0`：背包容量为 0 时，无法装入任何物品，总价值为 0。

**最终结果:** `dp[n][W]`，其中 `n` 是物品的数量，`W` 是背包的总容量。

**空间优化:** 注意到在计算 `dp[i][j]` 时，我们只依赖于 `dp[i-1]` 行的值。因此，我们可以将二维数组优化为一维数组 `dp[j]`，其中 `dp[j]` 表示容量为 `j` 的背包能获得的最大价值。为了确保在计算 `dp[j]` 时使用的是第 `i-1` 轮（即 `dp[i-1]`）的值，我们需要**逆序**遍历容量 `j` (从 `W` 到 `weights[i-1]`)。

优化后的状态转移方程： `dp[j] = max(dp[j], values[i-1] + dp[j - weights[i-1]])`  (对于每个物品 `i`，`j` 从 `W` 倒序到 `weights[i-1]`)