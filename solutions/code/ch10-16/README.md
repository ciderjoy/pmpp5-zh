# 第 10--16 章可执行示例

本目录把第 10--16 章题解中的 24 个算法与 CUDA 核心 `lstlisting` 扩展为 9 个
独立可构建程序。每个 `.cu` 文件均含 `main`、输入与边界检查、设备内存生命周期、
Runtime API 与内核启动错误检查、显式同步、CPU 参考实现和精确或容差比较，并复用
`../common/cuda_check.hpp`。每个程序都接受 `--cpu-only`；未检测到 CUDA 设备时也会
自动执行同一组 CPU 参考断言并成功退出。

`reference_algorithms.hpp` 汇集不依赖 CUDA 的参考算法；
`ch10_16_cpu_references.cpp` 将第 10--16 章的 CPU 语义检查注册为普通 C++ CTest。
示例没有依赖 CUB：扫描阶段使用本目录内核实现，使其兼容本仓库当前的 Windows
主机编译链，同时保留题解要求的算法阶段和同步边界。

## Listing 映射

以下 24 行与第 10--16 章的 24 个 `lstlisting` 一一对应。

| 章--题 | 题解 listing | 源码文件 | 可执行覆盖 |
|---|---|---|---|
| 第 10 章第 3 题 | `right_owned_sum` | `ch10_ex03-05_reduction.cu` | 右侧线程持有中间和、块内同步与精确小规模参考 |
| 第 10 章第 4 题 | `coarsened_max` | `ch10_ex03-05_reduction.cu` | 粗化最大值归约、非整除输入和 `-inf` 填充值 |
| 第 10 章第 5 题 | `coarsened_sum_any` | `ch10_ex03-05_reduction.cu` | 任意块宽归约、尾块与多轮设备归约 |
| 第 11 章第 4 题 | `scan_vec_padded` | `ch11_ex04-07_scan.cu` | 向量化加载、共享内存填充和尾部块内扫描 |
| 第 11 章第 5 题 | `decoupledLookback` | `ch11_ex04-07_scan.cu` | 动态块编号、发布状态和解耦回看 |
| 第 11 章第 7 题 | `brent_kung_scan` | `ch11_ex04-07_scan.cu` | Brent--Kung 上扫/下扫、尾块和全局块偏移 |
| 第 12 章第 1 题 | `eval` / `scan` / `scatter` | `ch12_ex01-04_filter.cu` | 严格分离的求值、排他扫描、散射三内核管线 |
| 第 12 章第 2 题 | `publishLookback` / `stable_filter_onepass` | `ch12_ex01-04_filter.cu` | 单遍稳定过滤的发布/获取协议与输出长度 |
| 第 12 章第 3 题 | `stable_filter_private` | `ch12_ex01-04_filter.cu` | 块私有过滤、块次序稳定性与部分块 |
| 第 12 章第 4 题 | `stable_filter_coarse` | `ch12_ex01-04_filter.cu` | 每线程粗化、稳定局部排名和任意输入长度 |
| 第 13 章第 3 题 | 精确 tile 加载与线程比例分区 | `ch13_ex01-03_merge.cu` | 64 位比例边界、精确分段长度和稳定合并 |
| 第 14 章第 1 题 | `pack` / `scatter_one_bit` | `ch14_ex01-03_radix_sort.cu` | 一位打包、排他扫描、稳定散射及 32 轮排序 |
| 第 14 章第 2 题 | `pack` / `scatter_multibit` | `ch14_ex01-03_radix_sort.cu` | `RBITS=4` 多位直方图、静态位宽限制及稳定散射 |
| 第 14 章第 3 题 | `pack` / `scatter_one_bit_x4` | `ch14_ex01-03_radix_sort.cu` | 每线程四元素、尾部保护和稳定散射 |
| 第 14 章第 4 题 | `co_rank` / `serial_merge` / `merge_pass` | `ch14_ex04_merge_sort.cu` | `size_t` 分段合并、尾段和最终缓冲区选择 |
| 第 15 章第 1 题 | `loadTileVec4` | `ch15_ex01-03_gemm.cu` | 先判合法行、再构造向量指针，并处理标量尾部 |
| 第 15 章第 2 题 | `writeTileVec4` | `ch15_ex01-03_gemm.cu` | 对齐向量写回、非整除列边界和往返校验 |
| 第 15 章第 3 题 | `gemm_rearranged` | `ch15_ex01-03_gemm.cu` | 重排的 `128x128x8` GEMM、K 尾块及浮点容差比较 |
| 第 16 章第 1 题 | `fw_square` | `ch16_ex01_floyd_warshall.cu` | 64 位 Floyd--Warshall、每个 k 阶段同步及精确参考 |
| 第 16 章第 2 题 | `swTile` / `sw_rect_wave` | `ch16_ex02-06_smith_waterman.cu` | 非方形 Smith--Waterman 波前、边界 tile 和精确 DP 表 |
| 第 16 章第 3 题 | `sw_cooperative` | `ch16_ex02-06_smith_waterman.cu` | cooperative launch、设备支持/占用率/驻留资源检查 |
| 第 16 章第 4 题 | `sw_unidirectional` | `ch16_ex02-06_smith_waterman.cu` | 单向 tile 领取、发布次序和轮询依赖 |
| 第 16 章第 5 题 | `sw_hyper_oneway` | `ch16_ex02-06_smith_waterman.cu` | hypertile 单向调度、生命周期和边界 tile |
| 第 16 章第 6 题 | `sw_max` DPX / fallback | `ch16_ex02-06_smith_waterman.cu` | `sm_90` DPX 分支与旧架构等价回退 |

## 构建与验证

从仓库根目录运行；所有构建输出必须位于仓库外，且不得把 `NUL` 用作 Windows 下的
编译输出文件名：

```powershell
solutions/code/build.ps1 `
  -BuildDirectory D:\translation\tmp\pmpp5-solutions-code-build-10-16
```

脚本先由 CMake 构建普通 C++ 并运行 CTest，再用 `nvcc` 编译、链接所有 `ch*.cu`，
最后逐个执行 CUDA 程序的 `--cpu-only` 路径。默认 `sm_75` 构建验证第 16-6 题的回退
路径；另以 `sm_90` 编译 `ch16_ex02-06_smith_waterman.cu` 可验证 DPX 分支能够编译。
有兼容 GPU 时，省略 `--cpu-only` 即可执行设备结果与 CPU 参考的对照。构建产物不得
放入或提交到本目录。
