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