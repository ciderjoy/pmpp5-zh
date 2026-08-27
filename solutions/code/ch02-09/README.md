# 第 2--9 章可执行示例

本目录把第 2--9 章题解中的算法与 CUDA 核心片段扩展为独立程序。每个 `.cu` 文件均含
`main`、输入与边界检查、设备内存生命周期、Runtime API 错误检查、内核启动后同步、
CPU 参考实现和容差比较，并复用 `../common/cuda_check.hpp`。所有程序接受
`--cpu-only`；未检测到 CUDA 设备时也自动运行 CPU 参考路径并成功退出。

受 Windows Smart App Control 或企业代码完整性策略管理的环境不应执行新生成、未签名的
CUDA `.exe`。`ch02_09_cpu_references.cpp` 将本范围的 CPU 语义检查集中为普通 C++ CTest；
默认构建验收会运行该 CTest，并逐个以 `--cpu-only` 运行由 `nvcc` 生成的 CUDA 程序，
从而验证各程序的主机参考路径而不启动设备内核。传入 `-SkipRun` 时才只编译、链接而不运行。

## Listing 映射

以下 14 行与第 2--9 章的 14 个 `lstlisting` 一一对应。第 4、8、9 章没有
`lstlisting`。

| 章--题 | 题解 listing | 源码文件 | 可执行覆盖 |
|---|---|---|---|
| 第 2 章第 6 题 | `cudaMalloc((void **)&A_d, ...)` | `ch02_ex06-10_runtime_affine.cu` | `device_buffer` 对分配与释放作 RAII 封装 |
| 第 2 章第 7 题 | `cudaMemcpy(A_d, A_h, ..., cudaMemcpyHostToDevice)` | `ch02_ex06-10_runtime_affine.cu` | 主机到设备及设备到主机复制 |
| 第 2 章第 8 题 | `cudaError_t err` | `ch02_ex06-10_runtime_affine.cu` | 显式保存并检查启动错误 |
| 第 2 章第 10 题 | `__host__ __device__ float affine(...)` | `ch02_ex06-10_runtime_affine.cu` | 主机参考与设备内核共同调用 `affine` |
| 第 2 章第 10 题 | `#if defined(__CUDA_ARCH__)` 设备分支 | `ch02_ex06-10_runtime_affine.cu` | `execution_lane` 分别编译主机和设备路径 |
| 第 3 章第 1 题 | 每线程一行/每线程一列矩阵乘 | `ch03_ex01_matmul.cu` | 两个内核分别与 CPU 矩阵乘比较 |
| 第 3 章第 2 题 | 方阵--向量乘内核与主机存根 | `ch03_ex02_matvec.cu` | 非整除尺寸、完整主机生命周期及 CPU 参考 |
| 第 5 章第 10 题 | 共享瓦片块内转置 | `ch05_ex10_block_transpose.cu` | 非整除矩阵、块内屏障及边界方形子区 |
| 第 6 章第 2 题 | 转角矩阵乘 | `ch06_ex02_corner_matmul.cu` | 行主序左矩阵、列主序右矩阵及带填充共享瓦片 |
| 第 6 章第 4 题 | `float4` 向量加法 | `ch06_ex04_float4_vector_add.cu` | 16 B 对齐设备分配和 3 元素标量尾部 |
| 第 7 章第 8 题 | 基本三维卷积 | `ch07_ex08-11_convolution.cu` | 运行期滤波器、三维边界与 CPU 相关运算 |
| 第 7 章第 9 题 | 常量内存三维卷积 | `ch07_ex08-11_convolution.cu` | `cudaMemcpyToSymbol` 与常量缓存系数 |
| 第 7 章第 10 题 | 完整输入共享瓦片三维卷积 | `ch07_ex08-11_convolution.cu` | `8^3` 加载线程、`6^3` 输出区和一致屏障 |
| 第 7 章第 11 题 | 输出线程二维卷积 | `ch07_ex08-11_convolution.cu` | `16^2` 线程循环加载 `20^2` 输入瓦片 |

## 补充索引覆盖

第 2 章第 1--3 题没有 `lstlisting`，但其索引公式是 CUDA 核心代码，因此另行提供：

| 章--题 | 核心内容 | 源码文件 |
|---|---|---|
| 第 2 章第 1 题 | 每线程一个元素的线性索引 | `ch02_ex01-03_indexing.cu` |
| 第 2 章第 2 题 | 每线程两个相邻元素及独立尾部检查 | `ch02_ex01-03_indexing.cu` |
| 第 2 章第 3 题 | 每块两个连续区段的分段式索引 | `ch02_ex01-03_indexing.cu` |
| 第 2 章第 2 题补充基线 | 现有一元素向量加法示例 | `ch02_ex02_vector_add.cu` |

## 构建与验证

从仓库根目录运行，构建输出必须位于仓库外：

```powershell
solutions/code/build.ps1 `
  -BuildDirectory D:\translation\tmp\pmpp5-solutions-code-build-02-09
```

默认模式会运行 CTest，并用 `--cpu-only` 逐个执行所有 `nvcc` 编译、链接的 `ch*.cu`
程序；它会检查每个程序的 CPU 参考路径，但不会启动 CUDA 设备内核。使用 `-SkipRun`
可改为只编译和链接。需要做 GPU/CPU 对照时，应在本机策略允许且存在 CUDA 设备的环境中
不带 `--cpu-only` 单独运行相应程序。构建产物不得放入本目录。
