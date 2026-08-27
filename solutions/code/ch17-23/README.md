# 第 17--23 章可执行示例

本目录保存第 17、18、19、20、21、23 章题解对应的自检程序。每个 CUDA 源文件都能
独立编译并含有 `main`，会验证输入和边界、使用 `common/cuda_check.hpp` 中的
`device_buffer` 管理设备内存、检查内核启动与同步错误，并与纯 CPU 参考结果作精确或容差
比较。所有 CUDA 程序都支持 `--cpu-only`，无 GPU 时也能验证算法。

两个第 23 章程序和 `ch17_23_cpu_references.cpp` 会由上级
`solutions/code/CMakeLists.txt` 自动发现并注册为 CTest；MPI 可用时还会注册两秩测试。

## 程序目录

| 源文件 | 可执行内容 |
|---|---|
| `ch17_23_cpu_references.cpp` | 集中执行稀疏格式、BFS、卷积、注意力、cenergy 和第 23 章计数的确定性 CPU 断言 |
| `ch17_ex03_coo_to_csr.cu` | 坐标验证、COO 行直方图、CUB 排他扫描、原子散布到 CSR |
| `ch17_ex04_hybrid_spmv.cu` | 主机构造 HYB、设备执行 ELL SpMV、主机累加 COO 溢出项 |
| `ch17_ex05_jds_spmv.cu` | 稳定构造 JDS 并在设备执行 JDS SpMV |
| `ch18_ex02_direction_bfs.cu` | 同时保存 CSR/CSC 的方向优化推送/拉取 BFS |
| `ch18_ex03_cooperative_bfs.cu` | 每层一次全网格屏障的协作式 BFS |
| `ch19_ex01_subsample_forward.cu` | 批处理 NCHW 平均下采样、通道偏置、sigmoid 和尾部边界 |
| `ch19_ex03_convolution.cu` | NCHW valid 卷积前向以及 `dX`、`dF`、`db` |
| `ch20_ex01_online_attention.cu` | 512 线程因果分块注意力、共享 K/V/S、在线状态及真实 CUB warp 归约 |
| `ch21_ex01_04_cenergy.cu` | 严格 `q/r` 常量内存分块 cenergy 与对齐的 `float4` 四点粗化 |
| `ch23_ex01_partition.cpp` | 五点模板分区数量计算 |
| `ch23_ex01_halo_exchange_mpi.cpp` | 两个及以上 MPI 秩的非阻塞双向周期 halo 交换和逐元素核验 |

## 代码清单映射（15 项）

下表按题号和清单语义定位，不使用会因完整题面或排版修改而漂移的源码行号。

| 题解代码清单 | 对应程序 | 实际验证 |
|---|---|---|
| 第 17 章第 3 题，COO 转 CSR 内核和启动序列 | `ch17_ex03_coo_to_csr.cu` | CUB `DeviceScan::ExclusiveSum`、零行哨兵和任意顺序 7 非零项；原子散布结果规范化后比较 |
| 第 17 章第 4 题，HYB 构造与 ELL 内核 | `ch17_ex04_hybrid_spmv.cu` | 测试 `K=1` 的混合格式和 `K=0` 的纯 COO 端点 |
| 第 17 章第 5 题，JDS SpMV | `ch17_ex05_jds_spmv.cu` | 检查稳定行置换 `[0,2,3,1]` 和段指针 `[0,4,7]` |
| 第 18 章第 2 题，方向优化 BFS | `ch18_ex02_direction_bfs.cu` | 推送和拉取两种模式都会执行，并与队列 BFS 层号比较 |
| 第 18 章第 3 题，协作式 BFS | `ch18_ex03_cooperative_bfs.cu` | 启动前检查协作能力和可驻留块数 |
| 第 19 章第 1 题，NCHW 下采样 | `ch19_ex01_subsample_forward.cu` | 用不能整除的 `5x6` 输入验证尾部不会越界 |
| 第 19 章第 3 题，卷积反向内核 | `ch19_ex03_convolution.cu` | 执行 `dX`、`dF`、`db` 及其匹配的前向定义 |
| 第 20 章第 1 题，因果 FlashAttention 主机分配与启动 | `ch20_ex01_online_attention.cu` | 512 线程、`N=96,d=128`、RAII、资源检查及三 tile 因果 CPU 黄金结果 |
| 第 20 章第 2 题，CUB `WarpReduce` | `ch20_ex01_online_attention.cu` | 16 个并发 warp 各有独立临时区；另以双 warp 微测试得到 528 和 1552 |
| 第 20 章第 3 题，在线状态 `initialize()` | `ch20_ex01_online_attention.cu` | 每 warp 两个查询行的输出和分母置 0，最大值置负无穷 |
| 第 21 章第 1 题，cenergy 边界保护 | `ch21_ex01_04_cenergy.cu` | 标量和向量内核都检查 x、y、切片及原子数，CPU 另验证 `q/r` 奇点 |
| 第 21 章第 1 题，cenergy 主机启动函数 | `ch21_ex01_04_cenergy.cu` | 显式给出四项启动配置，并在同一 stream 顺序覆盖常量块 |
| 第 21 章第 4 题，对齐向量写回 | `ch21_ex01_04_cenergy.cu` | 执行 `float4` 读改写，并为未对齐行和尾部提供标量回退 |
| 第 21 章第 4 题，原书重叠索引反例 | `ch21_ex01_04_cenergy.cu` | 反例不执行；程序使用互不重叠的所有权映射 |
| 第 21 章第 4 题，修正后的粗化索引 | `ch21_ex01_04_cenergy.cu` | 实际代码为 `first_x = linear_x * coarsening_factor` |

## 章节题目映射（27 项）

六个题解文件的每道题都列在下表。概念题映射到其分析所依赖的数据布局或算法程序；这些
程序验证正确性，不冒充硬件性能基准。

| 题号 | 源文件 | 可执行关联 |
|---|---|---|
| 17.1 稀疏表示 | `ch17_23_cpu_references.cpp`、`ch17_ex05_jds_spmv.cu` | 检查 7 个非零项矩阵的 CSR/JDS 数组 |
| 17.2 存储量 | `ch17_23_cpu_references.cpp` | 按题设元数据模型断言 COO 21、CSR 19、ELL 20、JDS 21 |
| 17.3 COO 转 CSR | `ch17_ex03_coo_to_csr.cu` | 完整 CPU/GPU 转换 |
| 17.4 ELL--COO HYB | `ch17_ex04_hybrid_spmv.cu` | 完整 CPU/GPU SpMV |
| 17.5 JDS SpMV | `ch17_ex05_jds_spmv.cu` | 完整 CPU/GPU SpMV |
| 17.6 分段 ELL 对齐权衡 | `ch17_ex05_jds_spmv.cu` | 给出紧凑无填充 JDS 基线并验证段长不变量 |
| 17.7 COO 非确定性与串行行 CSR | `ch17_ex03_coo_to_csr.cu`、`ch17_23_cpu_references.cpp` | GPU 原子散布规范化比较；CPU 行累加次序固定 |
| 17.8 选择 HYB 的 `K` | `ch17_ex04_hybrid_spmv.cu`、`ch17_23_cpu_references.cpp` | 测试 `K=1` 溢出及 `K=0` 纯 COO |
| 18.1 图数组与 BFS 活动 | `ch17_23_cpu_references.cpp` | 断言 15 条边的 CSR、层号和活动总数 |
| 18.2 方向优化 BFS | `ch18_ex02_direction_bfs.cu` | 推送 CAS 与拉取单写者实现 |
| 18.3 单全网格屏障 BFS | `ch18_ex03_cooperative_bfs.cu` | 协作内核与有界共享前沿 |
| 19.1 下采样前向 | `ch19_ex01_subsample_forward.cu` | 完整 CPU/GPU 前向操作 |
| 19.2 NCHW/NHWC/CHWN 布局 | `ch19_ex01_subsample_forward.cu`、`ch19_ex03_convolution.cu` | 给出池化和卷积的具体 NCHW 线性化 |
| 19.3 卷积反向 | `ch19_ex03_convolution.cu` | 前向、`dX`、`dF`、`db`；CPU CTest 另作有限差分检查 |
| 19.4 隐式 GEMM 访存 | `ch19_ex03_convolution.cu` | 提供数学等价的直接卷积基线和尾部边界检查 |
| 20.1 注意力主机代码 | `ch20_ex01_online_attention.cu` | 完整分配、资源检查、512 线程启动、同步和因果比较 |
| 20.2 CUB `WarpReduce` 语义 | `ch20_ex01_online_attention.cu` | 两个并发逻辑 warp 验证独立临时存储与 lane 0 结果 |
| 20.3 在线状态初始化 | `ch20_ex01_online_attention.cu` | 每 warp 两行的寄存器状态实际调用设备 `initialize()` |
| 20.4 K/V 合并加载 | `ch20_ex01_online_attention.cu` | 512 线程协作加载连续 K/V，并把 K 转置加 padding 后放入共享内存 |
| 21.1 cenergy 主机启动 | `ch21_ex01_04_cenergy.cu` | 显式 stream 顺序的标量分块实现 |
| 21.2 粗化运算计数 | `ch21_ex01_04_cenergy.cu` | 标量和四点实现直接呈现两种循环结构 |
| 21.3 粗化缺点 | `ch21_ex01_04_cenergy.cu` | 固定因子 4 使线程数下降和单线程状态增加可见 |
| 21.4 向量化 cenergy | `ch21_ex01_04_cenergy.cu` | 标量/向量 CPU/GPU 交叉比较 |
| 21.5 邻域分歧 | `ch21_ex01_04_cenergy.cu` | 给出讨论中数据相关谓词所依赖的逐线程空间距离计算 |
| 23.1 分区和 halo 数量 | `ch23_ex01_partition.cpp`、`ch23_ex01_halo_exchange_mpi.cpp`、`ch17_23_cpu_references.cpp` | 检查 1984/128/124/1860/512，并交换完整 64-float 行 |
| 23.2 MPI 元素大小 | `ch23_ex01_halo_exchange_mpi.cpp`、`ch17_23_cpu_references.cpp` | 使用 `MPI_FLOAT` 并断言题设 1000×4 字节模型 |
| 23.3 阻塞/非阻塞 MPI 与零计数消息 | `ch23_ex01_halo_exchange_mpi.cpp` | 实际执行 `Irecv`/`Isend`/`Waitall` 并逐项核验 128 个 halo 值；零计数合法性另由标准语义说明 |

`ch21_ex01_04_cenergy.cu` 的原子块使用进程级唯一常量符号；单次调用依赖同一 stream 的
有序语义，不能让两个主机线程或不同 stream 并发覆盖该符号。需要可重入接口时，应改用
每次调用独立的只读设备缓冲区。

## 构建与运行

必须把全部产物放在仓库外：

```powershell
& D:\translation\pmpp5-zh\solutions\code\build.ps1 `
  -BuildDirectory D:\translation\tmp\pmpp5-solutions-code-vs2022
```

驱动脚本会配置并编译 C++/MPI CTest 目标，运行 CTest（Microsoft MPI 可用时含 2 秩
测试），再以 `nvcc` 编译链接每个 `ch*.cu`，最后逐个运行 `--cpu-only`。CUDA 13.3 的
CCCL/CUB 在 MSVC 下要求标准预处理器，因此 `nvcc` 主机参数必须包含
`-Xcompiler=/Zc:preprocessor`。
